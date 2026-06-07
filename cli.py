#!/usr/bin/env python3
import sys
import click
from brain import db, decay, extract, graph, visualize, config, portability


@click.group()
def cli():
    pass


# ── setup ─────────────────────────────────────────────────────────────────────

@cli.command()
def setup():
    """Set your name so the brain knows who it belongs to."""
    current = config.get_user()
    if current:
        name = click.prompt(f"Your name (currently '{current}')", default=current)
    else:
        name = click.prompt("Your name")
    config.set_user(name.strip())
    conn = db.connect()
    db.ensure_identity_anchor(conn, name.strip())
    click.echo(f"Brain configured for {name}.")


# ── add ───────────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("text", required=False)
@click.option("--file", "-f", "file_path", type=click.Path(exists=True))
@click.option("--url", "-u")
@click.option("--source", "-s", default="")
def add(text, file_path, url, source):
    """Ingest text, a file, or a URL into the brain."""
    if file_path:
        try:
            raw = open(file_path, encoding="utf-8", errors="replace").read()
        except OSError as e:
            click.echo(f"Could not read file: {e}", err=True)
            sys.exit(1)
        source = source or file_path
    elif url:
        try:
            import httpx
            from bs4 import BeautifulSoup
            r = httpx.get(url, timeout=15, follow_redirects=True)
            soup = BeautifulSoup(r.text, "html.parser")
            raw = soup.get_text(separator=" ", strip=True)
            source = source or url
        except Exception as e:
            click.echo(f"Failed to fetch URL: {e}", err=True)
            sys.exit(1)
    elif text:
        raw = text
    else:
        click.echo("Provide text, --file, or --url.", err=True)
        sys.exit(1)

    if not raw or not raw.strip():
        click.echo("Nothing to ingest — the input is empty.", err=True)
        sys.exit(1)

    conn = db.connect()
    _run_decay(conn)

    user = config.get_user()
    if user:
        db.ensure_identity_anchor(conn, user)

    click.echo("Extracting knowledge...")
    try:
        node_ids, edge_ids = extract.ingest(conn, raw, source=source, user=user)
    except Exception as e:
        click.echo(f"Extraction failed: {e}", err=True)
        sys.exit(1)

    click.echo(f"Added {len(node_ids)} node(s), {len(edge_ids)} edge(s).")
    for nid in node_ids[:6]:
        n = db.get_node(conn, nid)
        if n:
            click.echo(f"  [{n['type']}] {n['name']}")
    if len(node_ids) > 6:
        click.echo(f"  ... and {len(node_ids) - 6} more")


# ── show ──────────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--min-weight", default=0.0, show_default=True)
@click.option("--type", "type_filter", default=None)
@click.option("--color-by", default="type", type=click.Choice(["type", "cluster"]), show_default=True)
@click.option("--3d", "threed", is_flag=True, help="render an interactive 3D (WebGL) graph")
def show(min_weight, type_filter, color_by, threed):
    """Open interactive graph in browser (2D, or 3D with --3d)."""
    conn = db.connect()
    _run_decay(conn)
    path = visualize.show(conn, min_weight=min_weight, type_filter=type_filter,
                          color_by=color_by, threed=threed)
    click.echo(f"Graph opened: {path}")


# ── serve ─────────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--port", default=8000, show_default=True)
@click.option("--interval", default=3.0, show_default=True, help="seconds between change checks")
@click.option("--no-open", is_flag=True, help="don't auto-open the browser")
def serve(port, interval, no_open):
    """Serve a live graph that reloads as you talk to the brain."""
    from brain import server
    server.serve(port=port, interval=interval, open_browser=not no_open)


# ── query ─────────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("query")
@click.option("--min-weight", default=0.0, show_default=True)
@click.option("--limit", default=10, show_default=True)
@click.option("--semantic", is_flag=True, help="Rank by meaning (embeddings) instead of keywords.")
def query(query, min_weight, limit, semantic):
    """Search for nodes matching a query (keyword by default; --semantic for meaning)."""
    conn = db.connect()
    _run_decay(conn)

    if semantic:
        from brain import llm
        scored = graph.semantic_search(conn, llm.embed(query), min_weight=min_weight, limit=limit)
        if not scored:
            click.echo("No embedded nodes yet — run `brain reindex` first.")
            return
        for score, r in scored:
            db.touch_node(conn, r["id"])
            click.echo(f"[{r['type']:8s}] {r['name']:30s}  sim={score:.3f}  {r['content'][:55]}")
        conn.commit()
        return

    results = graph.query_nodes(conn, query, min_weight=min_weight)[:limit]
    if not results:
        click.echo("No results.")
        return
    for r in results:
        click.echo(
            f"[{r['type']:8s}] {r['name']:30s}  w={r['weight']:.2f}  {r['content'][:60]}"
        )


# ── context ───────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("topic", required=False, default="")
@click.option("--depth", default=3, show_default=True)
@click.option("--min-weight", default=0.2, show_default=True)
def context(topic, depth, min_weight):
    """Generate an AI context document about a topic (or the full brain)."""
    conn = db.connect()
    _run_decay(conn)

    all_nodes, used_fallback = graph.collect_context_nodes(
        conn, topic=topic, depth=depth, min_weight=min_weight
    )
    if not all_nodes:
        click.echo("No nodes found — the brain is empty. Run `brain add` first.", err=True)
        sys.exit(1)
    if used_fallback:
        click.echo(
            f"Nothing matched '{topic}'; synthesising from the whole brain instead.",
            err=True,
        )

    click.echo(
        f"Synthesising context from {len(all_nodes)} node(s)...", err=True
    )
    doc = graph.synthesize_context(all_nodes, topic=topic)
    click.echo(doc)


# ── reorganize ────────────────────────────────────────────────────────────────

@cli.command()
def reorganize():
    """Retrofit existing nodes into the person-rooted hierarchy + re-score importance."""
    conn = db.connect()
    _run_decay(conn)
    user = config.get_user()
    if not user:
        click.echo("Run `brain setup` first so the tree has a root.", err=True)
        sys.exit(1)
    click.echo("Reorganizing via the LLM...")
    edges, rescored = extract.reorganize(conn, user)
    if not edges and not rescored:
        click.echo("Nothing to reorganize.")
        return
    click.echo(f"Reorganized: {edges} hierarchy edge(s), {rescored} importance update(s). "
               f"Run `brain tree` to see it.")


# ── tree ──────────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--min-weight", default=0.0, show_default=True)
def tree(min_weight):
    """Print the brain as a hierarchy rooted at you."""
    conn = db.connect()
    _run_decay(conn)
    kids = graph.children_map(conn)
    seen = set()

    def render(nid, depth):
        if nid in seen:
            return
        seen.add(nid)
        n = db.get_node(conn, nid)
        if not n or n["archived"] or n["weight"] < min_weight:
            return
        click.echo("  " * depth + f"- {n['name']} [{n['type']}] w={n['weight']:.2f} imp={n['importance']:.2f}")
        for child in sorted(kids.get(nid, []),
                            key=lambda c: -((db.get_node(conn, c) or {"weight": 0})["weight"])):
            render(child, depth + 1)

    # roots = identity node first, then anything with no parent
    has_parent = {e["source_id"] for e in db.all_edges(conn) if e["relation"] == "part_of"}
    user = config.get_user()
    roots = []
    if user:
        ident = db.get_node_by_name(conn, user)
        if ident:
            roots.append(ident["id"])
    roots += [n["id"] for n in db.all_nodes(conn)
              if n["id"] not in has_parent and n["id"] not in roots]
    for r in roots:
        render(r, 0)


# ── status ────────────────────────────────────────────────────────────────────

@cli.command()
def status():
    """Show graph stats."""
    conn = db.connect()
    result = _run_decay(conn)
    s = db.stats(conn)
    click.echo(f"Nodes:    {s['active']} active, {s['archived']} archived ({s['total']} total)")
    click.echo(f"Edges:    {s['edges']}")
    click.echo(f"Avg weight: {s['avg_weight']}")
    click.echo(f"By type:  {s['by_type']}")
    if any(result.values()):
        click.echo(
            f"Decay:    nodes updated={result['updated']} archived={result['archived']} "
            f"deleted={result['deleted']} | edges pruned={result['edges_pruned']}"
        )

    risky = decay.at_risk_nodes(conn)
    if risky:
        click.echo("Fading soon:")
        for r in risky:
            left = "soon" if r["days_left"] < 1 else f"~{r['days_left']:.0f}d"
            click.echo(f"  [{r['type']:8s}] {r['name'][:34]:34s} w={r['weight']:.2f} ({left} to archive)")


# ── decay ─────────────────────────────────────────────────────────────────────

@cli.command("decay")
def decay_cmd():
    """Run the decay pass manually."""
    conn = db.connect()
    result = decay.run_decay(conn)
    click.echo(
        f"updated={result['updated']} archived={result['archived']} deleted={result['deleted']}"
    )


# ── prune ─────────────────────────────────────────────────────────────────────

@cli.command()
def prune():
    """Delete all archived nodes immediately."""
    conn = db.connect()
    rows = conn.execute("SELECT id FROM nodes WHERE archived=1").fetchall()
    for r in rows:
        db.delete_node(conn, r["id"])
    conn.commit()
    click.echo(f"Pruned {len(rows)} archived node(s).")


# ── clear ─────────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--yes", "-y", is_flag=True, help="Skip the confirmation prompt.")
def clear(yes):
    """Erase the ENTIRE brain — all nodes, edges, and history (irreversible).

    Tip: run `brain export backup.json` first if you might want it back.
    """
    conn = db.connect()
    s = db.stats(conn)
    if s["total"] == 0:
        click.echo("Brain is already empty.")
        return
    if not yes:
        click.confirm(
            f"This permanently deletes {s['total']} node(s) and {s['edges']} edge(s). Continue?",
            abort=True,
        )
    counts = db.clear(conn)
    click.echo(
        f"Cleared {counts['nodes']} node(s), {counts['edges']} edge(s), "
        f"{counts['log']} log entr(ies)."
    )


# ── reindex ───────────────────────────────────────────────────────────────────

@cli.command()
def reindex():
    """Compute embeddings for all active nodes (enables `query --semantic`)."""
    from brain import llm
    conn = db.connect()
    nodes = db.all_nodes(conn)
    done = 0
    with click.progressbar(nodes, label="Embedding nodes") as bar:
        for node in bar:
            try:
                vec = llm.embed(f"{node['name']}. {node['content'] or ''}")
                db.set_embedding(conn, node["id"], vec)
                done += 1
            except Exception as e:
                click.echo(f"  skipped {node['name']}: {e}", err=True)
    conn.commit()
    click.echo(f"Reindexed {done}/{len(nodes)} node(s).")


# ── synthesize ────────────────────────────────────────────────────────────────

@cli.command()
def synthesize():
    """Run the synthesis job (find connections, surface insights)."""
    conn = db.connect()
    _run_decay(conn)
    _synthesize(conn)


# ── merge ─────────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("id1")
@click.argument("id2")
def merge(id1, id2):
    """Merge two nodes (id2 into id1)."""
    conn = db.connect()
    n1 = db.get_node(conn, id1)
    n2 = db.get_node(conn, id2)
    if not n1 or not n2:
        click.echo("One or both nodes not found.", err=True)
        sys.exit(1)
    db.merge_nodes(conn, id1, id2)
    click.echo(f"Merged {n2['name']} → {n1['name']}")


# ── forget ────────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("node_id")
def forget(node_id):
    """Immediately archive a node."""
    conn = db.connect()
    db.archive_node(conn, node_id)
    conn.commit()
    click.echo(f"Archived {node_id}.")


# ── reinforce ─────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("node_id")
def reinforce(node_id):
    """Manually boost a node's weight to 1.0."""
    conn = db.connect()
    db.touch_node(conn, node_id)
    conn.commit()
    click.echo(f"Reinforced {node_id}.")


# ── export / import ─────────────────────────────────────────────────────────────

@cli.command("export")
@click.argument("path", required=False, default="brain-export.json")
def export_cmd(path):
    """Export the whole brain (nodes + edges) to a JSON file."""
    conn = db.connect()
    data = portability.export_to_file(conn, path)
    click.echo(f"Exported {len(data['nodes'])} node(s), {len(data['edges'])} edge(s) to {path}")


@cli.command("import")
@click.argument("path", type=click.Path(exists=True))
def import_cmd(path):
    """Import nodes/edges from a JSON export (merges; skips duplicates)."""
    conn = db.connect()
    n, e = portability.import_from_file(conn, path)
    click.echo(f"Imported {n} new node(s), {e} new edge(s).")


# ── helpers ───────────────────────────────────────────────────────────────────

def _run_decay(conn):
    return decay.run_decay(conn)


def _synthesize(conn):
    """Find isolated nodes and connect them to the rest of the graph."""
    nodes = db.all_nodes(conn, min_weight=0.3)
    if len(nodes) < 2:
        click.echo("Not enough nodes to synthesize.")
        return

    connected = set()
    for e in db.all_edges(conn):
        connected.add(e["source_id"])
        connected.add(e["target_id"])
    isolated = sum(1 for n in nodes if n["id"] not in connected)
    click.echo(f"Found {isolated} isolated node(s), trying to connect...")

    made = graph.connect_isolated_nodes(conn)
    for m in made:
        click.echo(f"  Connected: {m['source']} --{m['relation']}--> {m['target']}")
    click.echo(f"Synthesis complete. {len(made)} new connection(s).")


if __name__ == "__main__":
    cli()
