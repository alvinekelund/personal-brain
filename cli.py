#!/usr/bin/env python3
import sys
import click
from brain import db, decay, extract, graph, visualize, config


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

    existing_nodes = db.all_nodes(conn)
    existing_names = [n["name"] for n in existing_nodes]

    click.echo("Extracting knowledge...")
    try:
        extracted = extract.extract(raw, source=source, existing_names=existing_names, user=user)
    except Exception as e:
        click.echo(f"Extraction failed: {e}", err=True)
        sys.exit(1)

    new_nodes = extracted.get("nodes", [])
    entity_links = extract.link_entities(new_nodes, existing_nodes)
    if entity_links:
        click.echo(f"  Linked: {entity_links}")

    node_ids, edge_ids = extract.merge_into_db(conn, extracted, source, raw, entity_links=entity_links)
    click.echo(f"Added {len(node_ids)} node(s), {len(edge_ids)} edge(s).")

    for n in new_nodes[:5]:
        display = entity_links.get(n["name"], n["name"])
        click.echo(f"  [{n.get('type', '?')}] {display}")
    if len(new_nodes) > 5:
        click.echo(f"  ... and {len(new_nodes) - 5} more")


# ── show ──────────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--min-weight", default=0.0, show_default=True)
@click.option("--type", "type_filter", default=None)
@click.option("--color-by", default="type", type=click.Choice(["type", "cluster"]), show_default=True)
def show(min_weight, type_filter, color_by):
    """Open interactive graph in browser."""
    conn = db.connect()
    _run_decay(conn)
    path = visualize.show(conn, min_weight=min_weight, type_filter=type_filter, color_by=color_by)
    click.echo(f"Graph opened: {path}")


# ── query ─────────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("query")
@click.option("--min-weight", default=0.0, show_default=True)
@click.option("--limit", default=10, show_default=True)
def query(query, min_weight, limit):
    """Search for nodes matching a query."""
    conn = db.connect()
    _run_decay(conn)
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

    # re-point all edges from n2 to n1
    for edge in db.edges_for_node(conn, id2):
        src = id1 if edge["source_id"] == id2 else edge["source_id"]
        tgt = id1 if edge["target_id"] == id2 else edge["target_id"]
        if src != tgt:
            db.add_edge(conn, src, tgt, edge["relation"], edge["weight"])
    db.delete_node(conn, id2)
    db.touch_node(conn, id1)
    conn.commit()
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
