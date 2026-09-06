#!/usr/bin/env python3
import sys
import click
from brain import db, decay, extract, graph, visualize, config, portability, vault
from brain import loops, decisions, doctor as doctor_mod, now as now_mod
from datetime import datetime as _dt


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
    # stage lines go to stderr so a scheduled task's log shows where a slow
    # Gemini call is stuck instead of a silent 10-minute wait (L-061)
    extract.ON_STAGE = lambda msg: click.echo(f"  · {msg}", err=True)
    inbox_before = len(loops.inbox_list(vault.vault_dir()))
    try:
        node_ids, edge_ids = extract.ingest(conn, raw, source=source, user=user)
    except Exception as e:
        click.echo(f"Extraction failed: {e}", err=True)
        sys.exit(1)
    finally:
        extract.ON_STAGE = None
    routed = len(loops.inbox_list(vault.vault_dir())) - inbox_before
    if routed:
        click.echo(f"{routed} action item(s) routed to LOOPS-INBOX.md (not the graph) — triage with `brain loop inbox`")

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


# ── mcp ───────────────────────────────────────────────────────────────────────

@cli.command()
def mcp():
    """Run the MCP server (stdio) — plug the brain into Claude Code & other agents.

    Register it once with: claude mcp add brain -- brain mcp
    """
    from brain import mcp as mcp_server
    mcp_server.serve()


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
    for r in results:
        path = r["path"] if "path" in r.keys() and r["path"] else ""
        click.echo(
            f"[{r['type']:8s}] {r['name']:30s}  w={r['weight']:.2f}  {r['content'][:60]}"
            + (f"  → {path}" if path else "")
        )
    from brain import index as vindex
    files = vindex.search(conn, query, k=6, seed_node_ids=[r["id"] for r in results])
    if not results and not files:
        click.echo("No results.")
        return
    if files:
        click.echo("\nfiles (the vault is the source of truth):")
        for f in files:
            click.echo(f"  {f['path']:42s} {f['title'][:38]:38s} {', '.join(f['why'])}")


# ── ask ───────────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("question")
def ask(question):
    """Ask your brain a question; it answers from what it knows."""
    conn = db.connect()
    _run_decay(conn)
    res = graph.answer_question(conn, question)
    click.echo(res["answer"])
    if res.get("files"):
        click.echo("\nfiles: " + ", ".join(res["files"]))
    if res["sources"]:
        click.echo("sources: " + ", ".join(res["sources"]))


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


# ── digest ────────────────────────────────────────────────────────────────────

@cli.command()
def digest():
    """A quick 'state of your brain' — what's important, fading, and open."""
    conn = db.connect()
    _run_decay(conn)
    d = graph.digest(conn, config.get_user())
    if d["top"]:
        click.echo("Top of mind:")
        for t in d["top"]:
            click.echo(f"  [{t['type']:8s}] {t['name']}  (imp {t['importance']})")
    if d["tasks"]:
        click.echo("Open loops (LOOPS.md):")
        for t in d["tasks"]:
            click.echo(f"  - {t}")
    if d["fading"]:
        click.echo("Fading soon:")
        for f in d["fading"]:
            left = "soon" if f["days_left"] < 1 else f"~{f['days_left']:.0f}d"
            click.echo(f"  - {f['name']} ({left})")
    if d["areas"]:
        click.echo("By area: " + ", ".join(f"{n} ({c})" for n, c in d["areas"]))


# ── vault ─────────────────────────────────────────────────────────────────────

@cli.command("vault")
@click.option("--dir", "dest", type=click.Path(), default=None,
              help="Render here instead of the configured vault directory.")
@click.option("--set-dir", "set_dir", type=click.Path(), default=None,
              help="Persist a vault directory in config, then render there.")
def vault_cmd(dest, set_dir):
    """Render the graph into the markdown vault (the brain's file layer).

    Also runs automatically after every `brain add` / web add (disable with
    config vault_auto=false). Curated files in the vault are never touched.
    Generated views are committed (the CLI commits its own writes).
    """
    if set_dir:
        cfg = config.load()
        cfg["vault_dir"] = set_dir
        config.save(cfg)
        dest = set_dir
    conn = db.connect()
    _run_decay(conn)
    paths = vault.render(conn, config.get_user(), dest=dest)
    root = dest or vault.vault_dir()
    loops.git_commit_paths(root, ["DIGEST.md", "graph"], "vault: render generated views")
    click.echo(f"Vault rendered: {len(paths)} generated file(s) in {root}")
    for p in paths:
        click.echo(f"  {p}")


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
    breakdown = graph.category_breakdown(conn, config.get_user())
    if breakdown:
        click.echo("By area:  " + ", ".join(f"{name} ({n})" for name, n in breakdown))
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

@cli.command("index")
@click.option("--no-embed", is_flag=True, help="Skip embeddings (keyword + graph links only).")
@click.option("--status", "show_status", is_flag=True, help="Report how current the index is; change nothing.")
def index_cmd(no_embed, show_status):
    """Index the vault directory so `brain ask` / brain_search route to the right files (D-014).

    Incremental: unchanged files (by content hash) are skipped; deleted files drop out;
    graph nodes whose name matches a file's title or alias get their `path` stamped.
    """
    from brain import index as vindex
    conn = db.connect()
    root = vault.vault_dir()
    if show_status:
        s = vindex.status(conn, root)
        changed = len(s["new"]) + len(s["stale"]) + len(s["removed"])
        click.echo(f"{s['indexed']} indexed / {s['on_disk']} on disk · {s['node_links']} node link(s) · "
                   f"{s['embedded']} embedded · {changed} changed since last index · "
                   f"ledger {s['ledger_embedded']}/{s['ledger_total']} embedded, {len(s['ledger_stale'])} stale")
        for label, items in (("new", s["new"]), ("changed", s["stale"]), ("removed", s["removed"]),
                             ("ledger stale", s["ledger_stale"])):
            for p in items[:20]:
                click.echo(f"  {label}: {p}")
        return
    s = vindex.build(conn, root, embed=not no_embed)
    click.echo(f"Indexed {s['files']} vault file(s): {s['added']} added, {s['updated']} updated, "
               f"{s['removed']} removed, {s['unchanged']} unchanged · {s['links']} file link(s), "
               f"{s['node_links']} node link(s), {s['embedded']} embedded · "
               f"{s['ledger_embedded']} ledger line(s) embedded")
    if s["no_frontmatter"]:
        click.echo("  no front-matter: " + ", ".join(s["no_frontmatter"][:10]))


@cli.command()
@click.option("--all", "everything", is_flag=True,
              help="re-embed every active node, not only the ones without an embedding")
def reindex(everything):
    """Embed the active nodes that lack an embedding (parallel; enables semantic
    search and dedup). --all re-embeds everything, e.g. after a model change."""
    from brain import llm
    if not llm.have_key():
        click.echo("No GEMINI_API_KEY — embeddings need the model.", err=True)
        sys.exit(1)
    conn = db.connect()
    nodes = db.all_nodes(conn)
    if everything:
        conn.execute("UPDATE nodes SET embedding = NULL WHERE archived = 0")
        conn.commit()
    todo = [n["id"] for n in nodes if everything or not n["embedding"]]
    if not todo:
        click.echo(f"All {len(nodes)} active node(s) already have embeddings (--all to redo).")
        return
    done = extract.embed_nodes(conn, todo)
    click.echo(f"Embedded {done}/{len(todo)} node(s)" + ("" if done == len(todo) else " — the rest failed; rerun later")
               + f" ({len(nodes)} active).")


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
    """Merge two nodes (the second into the first) — ids from `brain tree`, or exact names.

    The survivor keeps one part_of parent (the other becomes a cross-link) and
    the higher importance; the doctor's "A ~ B (brain merge)" hint pastes straight in."""
    conn = db.connect()
    n1 = db.get_node(conn, id1) or db.get_node_by_name(conn, id1)
    n2 = db.get_node(conn, id2) or db.get_node_by_name(conn, id2)
    if not n1 or not n2:
        click.echo("One or both nodes not found — use ids from `brain tree` or the exact names.", err=True)
        sys.exit(1)
    if n1["id"] == n2["id"]:
        click.echo("That is the same node twice.", err=True)
        sys.exit(1)
    db.merge_nodes(conn, n1["id"], n2["id"])
    vault.auto_render(conn, config.get_user())
    click.echo(f"Merged {n2['name']} → {n1['name']}")


@cli.command()
@click.argument("node")
@click.argument("parent")
def move(node, parent):
    """Re-home NODE under PARENT (ids from `brain tree`, or exact names).

    The deterministic alternative to `reorganize`: one node, one new parent,
    tree rules enforced (no cycles, categories only under you, nothing else
    directly under you)."""
    conn = db.connect()

    def resolve(ref):
        return db.get_node(conn, ref) or db.get_node_by_name(conn, ref)

    n, p = resolve(node), resolve(parent)
    if not n or not p:
        click.echo("Node or parent not found — use an id from `brain tree` or the exact name.", err=True)
        sys.exit(1)
    user = config.get_user()
    ident = db.get_node_by_name(conn, user) if user else None
    err = db.move_node(conn, n["id"], p["id"], ident_id=ident["id"] if ident else None)
    if err:
        click.echo(f"Not moved: {err}", err=True)
        sys.exit(1)
    vault.auto_render(conn, user)
    click.echo(f"Moved {n['name']} → under {p['name']}")


@cli.command()
@click.argument("node")
@click.argument("new_name")
def rename(node, new_name):
    """Rename NODE (id or exact name) to NEW_NAME — id, edges and its place in the tree stay."""
    from brain import llm
    conn = db.connect()
    n = db.get_node(conn, node) or db.get_node_by_name(conn, node)
    if not n:
        click.echo("Node not found — use an id from `brain tree` or the exact name.", err=True)
        sys.exit(1)
    old = n["name"]
    err = db.rename_node(conn, n["id"], new_name)
    if err:
        click.echo(f"Not renamed: {err}", err=True)
        sys.exit(1)
    new_name = new_name.strip()
    if llm.have_key():  # the embedding covers the name — refresh it, best-effort
        try:
            db.set_embedding(conn, n["id"], llm.embed(f"{new_name}. {n['content'] or ''}"))
            conn.commit()
        except Exception as e:
            click.echo(f"  (embedding not refreshed: {e} — `brain reindex` later)", err=True)
    vault.auto_render(conn, config.get_user())
    click.echo(f"Renamed {old!r} → {new_name!r}  (run `brain index` to re-link it to a vault file)")


@cli.command()
@click.argument("node")
@click.argument("new_type")
def retype(node, new_type):
    """Change NODE's type (event, fact, artifact, concept, insight, skill, project,
    person, organization); its decay half-life follows the new type."""
    conn = db.connect()
    n = db.get_node(conn, node) or db.get_node_by_name(conn, node)
    if not n:
        click.echo("Node not found — use an id from `brain tree` or the exact name.", err=True)
        sys.exit(1)
    err = db.retype_node(conn, n["id"], new_type)
    if err:
        click.echo(f"Not retyped: {err}", err=True)
        sys.exit(1)
    vault.auto_render(conn, config.get_user())
    click.echo(f"Retyped {n['name']!r}: {n['type']} → {db.get_node(conn, n['id'])['type']}")


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


# ── loops / decisions / today / doctor — the vault's action layer ─────────────

def _vault_root():
    return vault.vault_dir()


def _parse_day(s):
    return _dt.strptime(s, "%Y-%m-%d").date() if s else None


def _die(e):
    click.echo(f"error: {e}", err=True)
    sys.exit(1)


@cli.group()
def loop():
    """Manage LOOPS.md, the open-loop ledger (the brain's task system)."""


@loop.command("add")
@click.argument("title")
@click.option("--due", required=True, help="YYYY-MM-DD: hard deadline or act/review-by date.")
@click.option("--next", "next_", required=True, help="The single concrete next action.")
@click.option("--owner", default="alvin", show_default=True, help="alvin | claude | waiting:<who>")
@click.option("--area", default="other", show_default=True, help=" | ".join(loops.AREAS))
@click.option("--prio", default=2, show_default=True, type=click.IntRange(1, 3))
@click.option("--since", default=None, help="Backdate when the loop was really opened.")
@click.option("--date", "today", default=None, help="Pretend today is YYYY-MM-DD (tests/migration).")
@click.option("--from-inbox", "from_inbox", type=int, default=None,
              help="Also remove inbox item N (this loop is its triage).")
@click.option("--no-commit", is_flag=True)
def loop_add(title, due, next_, owner, area, prio, since, today, from_inbox, no_commit):
    """Open a loop; re-renders NOW.md and commits the vault."""
    root = _vault_root()
    try:
        l = loops.add(root, title, due, owner, area, next_, prio=prio, since=since,
                      today=_parse_day(today), commit=not no_commit)
        if from_inbox is not None:
            gone = loops.inbox_drop(root, from_inbox, action="triaged", commit=not no_commit)
            click.echo(f"triaged inbox item {from_inbox}: {gone['text']}")
    except loops.LoopError as e:
        _die(e)
    click.echo(l.to_line())


@loop.command("inbox")
@click.option("--drop", type=int, default=None, help="Discard inbox item N (remembered — never re-added).")
@click.option("--clear", is_flag=True, help="Discard every inbox item (remembered — never re-added).")
@click.option("--no-commit", is_flag=True)
def loop_inbox(drop, clear, no_commit):
    """Show action items the extractor found (triage with `loop add --from-inbox N`)."""
    root = _vault_root()
    try:
        if drop is not None:
            gone = loops.inbox_drop(root, drop, commit=not no_commit)
            click.echo(f"dropped: {gone['text']}")
        if clear:
            click.echo(f"cleared {loops.inbox_clear(root, commit=not no_commit)} item(s)")
    except loops.LoopError as e:
        _die(e)
    items = loops.inbox_list(root)
    if not items:
        click.echo("inbox empty")
    for i, it in enumerate(items, 1):
        src = f"  (from {it['source']})" if it["source"] else ""
        click.echo(f"{i:>3}. {it['date']}  {it['text']}{src}")


@loop.command("done")
@click.argument("lid")
@click.option("--note", default="", help="How it closed (kept on the closed line).")
@click.option("--date", "today", default=None)
@click.option("--no-commit", is_flag=True)
def loop_done(lid, note, today, no_commit):
    """Close a loop (moves it to the Closed list)."""
    try:
        l = loops.done(_vault_root(), lid, note=note, today=_parse_day(today), commit=not no_commit)
    except loops.LoopError as e:
        _die(e)
    click.echo(l.to_line())


@loop.command("edit")
@click.argument("lid")
@click.option("--title")
@click.option("--due")
@click.option("--next", "next_")
@click.option("--owner")
@click.option("--area")
@click.option("--prio", type=click.IntRange(1, 3))
@click.option("--note")
@click.option("--date", "today", default=None)
@click.option("--no-commit", is_flag=True)
def loop_edit(lid, title, due, next_, owner, area, prio, note, today, no_commit):
    """Change fields on an open loop (also bumps `touched`)."""
    try:
        l = loops.edit(_vault_root(), lid, today=_parse_day(today), commit=not no_commit,
                       title=title, due=due, next_=next_, owner=owner, area=area, prio=prio, note=note)
    except loops.LoopError as e:
        _die(e)
    click.echo(l.to_line())


@loop.command("touch")
@click.argument("lid")
@click.option("--date", "today", default=None)
@click.option("--no-commit", is_flag=True)
def loop_touch(lid, today, no_commit):
    """Mark a loop as reviewed today without changing it."""
    try:
        l = loops.touch(_vault_root(), lid, today=_parse_day(today), commit=not no_commit)
    except loops.LoopError as e:
        _die(e)
    click.echo(l.to_line())


@loop.command("list")
@click.option("--all", "show_all", is_flag=True, help="Include closed loops.")
@click.option("--area", default=None)
def loop_list(show_all, area):
    """Print open loops (prio, then due)."""
    ledger = loops.load(_vault_root())
    rows = sorted(ledger.open, key=lambda l: (l.prio, l.due, l.id))
    if show_all:
        rows += sorted(ledger.closed, key=lambda l: (l.done, l.id), reverse=True)
    if area:
        rows = [l for l in rows if l.area == area]
    for l in rows:
        click.echo(l.to_line())
    if ledger.errors:
        click.echo(f"\n{len(ledger.errors)} parse error(s) — run `brain loop lint`", err=True)


@loop.command("lint")
@click.option("--date", "today", default=None)
def loop_lint(today):
    """Validate LOOPS.md and NOW.md's rendered block. Exit 1 on errors."""
    errors, warnings = loops.lint(_vault_root(), _parse_day(today))
    for w in warnings:
        click.echo(f"⚠ {w}")
    for e in errors:
        click.echo(f"✗ {e}")
    if not errors:
        click.echo(f"✓ LOOPS.md ok ({len(loops.load(_vault_root()).open)} open, {len(warnings)} warning(s))")
    sys.exit(1 if errors else 0)


@loop.command("render")
@click.option("--no-commit", is_flag=True)
def loop_render(no_commit):
    """Regenerate NOW.md's hot section from LOOPS.md."""
    root = _vault_root()
    ledger = loops.load(root)
    if ledger.errors:
        _die("LOOPS.md has parse errors — run `brain loop lint`")
    changed = loops.render_now(root, ledger)
    if changed and not no_commit:
        loops.git_commit(root, "loops: render NOW.md")
    click.echo("NOW.md updated" if changed else "NOW.md already current")


@cli.command()
@click.argument("title")
@click.option("--what", required=True, help="The decision, one sentence.")
@click.option("--why", required=True)
@click.option("--rejected", default="—", help="Alternatives considered and dropped.")
@click.option("--revisit", default="—", help="What would reopen this decision.")
@click.option("--source", default="—", help="Session / log / artifact that settled it.")
@click.option("--date", "when", default=None)
@click.option("--no-commit", is_flag=True)
def decide(title, what, why, rejected, revisit, source, when, no_commit):
    """Append an entry to DECISIONS.md (append-only ledger)."""
    try:
        d = decisions.append(_vault_root(), title, what, why, rejected, revisit, source,
                             when=_parse_day(when), commit=not no_commit)
    except decisions.DecisionError as e:
        _die(e)
    click.echo(d.to_md())


@cli.command("decisions")
@click.option("--lint", "do_lint", is_flag=True)
@click.option("--last", default=0, type=int, help="Show only the last N entries.")
def decisions_cmd(do_lint, last):
    """List decisions, or --lint the ledger."""
    root = _vault_root()
    if do_lint:
        errs = decisions.lint(root)
        for e in errs:
            click.echo(f"✗ {e}")
        if not errs:
            click.echo(f"✓ DECISIONS.md ok ({len(decisions.load(root)[0])} entries)")
        sys.exit(1 if errs else 0)
    ds, errs = decisions.load(root)
    for d in (ds[-last:] if last else ds):
        click.echo(f"{d.id} · {d.date} · {d.title}\n    {d.decision}")
    for e in errs:
        click.echo(f"✗ {e}", err=True)


@cli.command()
@click.option("--date", "today", default=None, help="Pretend today is YYYY-MM-DD.")
@click.option("--days", default=7, show_default=True, help="Due-soon horizon.")
@click.option("--brief", is_flag=True, help="One plain-text line (<200 chars) for a phone push.")
@click.option("--no-doctor", is_flag=True, help="Skip the health line.")
def today(today, days, brief, no_doctor):
    """Deterministic action card: countdowns, waits, Claude-owned loops, top actions."""
    root = _vault_root()
    day = _parse_day(today)
    if brief:
        click.echo(loops.brief(root, day))
        return
    line = "" if no_doctor else doctor_mod.brief(doctor_mod.run(root, day))
    click.echo(loops.today_report(root, day, horizon=days, doctor_line=line,
                                  decisions=decisions.recent(root, day)))


@cli.command("doctor")
@click.option("--brief", is_flag=True, help="One summary line.")
@click.option("--install-hooks", is_flag=True, help="Install the vault's append-only pre-commit hook.")
@click.option("--repair", is_flag=True, help="Fix structural graph problems (orphans, multi-parent, cycles) first.")
@click.option("--date", "today", default=None)
def doctor_cmd(brief, install_hooks, repair, today):
    """Health check: binary, graph + tree integrity, key, API, capture, vault, ledgers, hooks, MCP, tasks. Exit 1 on failure."""
    root = _vault_root()
    if install_hooks:
        click.echo("pre-commit hook: " + ("installed" if decisions.install_pre_commit(root) else "vault is not a git repo"))
    if repair:
        _repair()
    checks = doctor_mod.run(root, _parse_day(today))
    click.echo(doctor_mod.brief(checks) if brief else doctor_mod.report(checks))
    sys.exit(1 if doctor_mod.worst(checks) == "fail" else 0)


def _repair():
    from brain import integrity
    conn = db.connect()
    before = integrity.check(conn, config.get_user())
    fixed = integrity.repair(conn, config.get_user())
    after = integrity.check(conn, config.get_user())
    vault.auto_render(conn, config.get_user())
    click.echo("repair: " + ", ".join(f"{k}={v}" for k, v in fixed.items() if v) if any(fixed.values()) else "repair: nothing structural to fix")
    click.echo(f"before: {before.summary()}\nafter:  {after.summary()}")
    return after


@cli.command("repair")
def repair_cmd():
    """Make the hierarchy a tree again: one parent per node, categories under you, no orphans or cycles."""
    _repair()


@cli.command()
@click.option("--threshold", default=extract.SUBGROUP_THRESHOLD, show_default=True,
              help="a category with more direct (non-category) children than this is split")
def subgroup(threshold):
    """Split oversized categories into LLM-clustered sub-categories (a category
    under a category), so a big area gets real MECE sub-structure instead of a
    flat list. Only the oversized ones are touched; `brain doctor` stays green."""
    from brain import llm
    if not llm.have_key():
        click.echo("No GEMINI_API_KEY — sub-grouping needs the model.", err=True)
        sys.exit(1)
    conn = db.connect()
    before = {n["name"] for n in db.all_nodes(conn) if n["type"] == "category"}
    moved = extract.subgroup_categories(conn, threshold=threshold)
    after = {n["name"] for n in db.all_nodes(conn) if n["type"] == "category"}
    vault.auto_render(conn, config.get_user())
    new = sorted(after - before)
    click.echo(f"Re-parented {moved} node(s) into {len(new)} new sub-categor{'y' if len(new) == 1 else 'ies'}"
               + (": " + ", ".join(new) if new else "") + ". `brain tree` to review, `brain move` to adjust.")


@cli.group()
def now():
    """NOW.md — the generated 'what is going on' view (IDENTITY + loops + areas' ## Now + people + apps)."""


@now.command("render")
@click.option("--no-commit", is_flag=True)
def now_render(no_commit):
    """(Re)generate NOW.md from its sources. Replaces a hand-written NOW.md."""
    root = _vault_root()
    changed = now_mod.write(root)
    if changed and not no_commit:
        loops.git_commit(root, "now: render NOW.md")
    click.echo("NOW.md rendered" if changed else "NOW.md already current")


@now.command("show")
def now_show():
    """Print what NOW.md would contain right now."""
    click.echo(now_mod.render_text(_vault_root()), nl=False)


@now.command("lint")
@click.option("--date", "today", default=None)
def now_lint(today):
    """Check NOW.md is current and every area/app/person file is well-formed and fresh. Exit 1 on errors."""
    errors, warnings = now_mod.lint(_vault_root(), _parse_day(today))
    for w in warnings:
        click.echo(f"⚠ {w}")
    for e in errors:
        click.echo(f"✗ {e}")
    if not errors:
        click.echo(f"✓ NOW.md current ({len(warnings)} warning(s))")
    sys.exit(1 if errors else 0)


@cli.group()
def area():
    """areas/<area>.md — one file per life area with a `## Now` block."""


@area.command("touch")
@click.argument("key")
@click.option("--date", "when", default=None, help="Set `updated:` to this date instead of today.")
@click.option("--no-commit", is_flag=True)
def area_touch(key, when, no_commit):
    """Stamp `updated:` on an area (after editing its `## Now` block) and re-render NOW.md."""
    root = _vault_root()
    try:
        p = now_mod.touch_area(root, key, _parse_day(when))
    except now_mod.NowError as e:
        _die(e)
    if not no_commit:
        loops.git_commit(root, f"area {key}: touched")
    click.echo(f"{p.relative_to(root)} updated; NOW.md re-rendered")


if __name__ == "__main__":
    # keep at end of file: every command above must be registered before dispatch
    # (`python cli.py loop ...` used to miss the vault-layer commands added below
    # the old mid-file guard; the installed `brain` entry point hid that).
    cli()
