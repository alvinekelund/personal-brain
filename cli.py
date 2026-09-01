#!/usr/bin/env python3
import sys
import click
from brain import db, decay, extract, graph, visualize, config, portability, vault
from brain import loops, decisions, doctor as doctor_mod
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
    if not results:
        click.echo("No results.")
        return
    for r in results:
        click.echo(
            f"[{r['type']:8s}] {r['name']:30s}  w={r['weight']:.2f}  {r['content'][:60]}"
        )


# ── ask ───────────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("question")
def ask(question):
    """Ask your brain a question; it answers from what it knows."""
    conn = db.connect()
    _run_decay(conn)
    res = graph.answer_question(conn, question)
    click.echo(res["answer"])
    if res["sources"]:
        click.echo("\nsources: " + ", ".join(res["sources"]))


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
        click.echo("Open tasks:")
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
@click.option("--no-commit", is_flag=True)
def loop_add(title, due, next_, owner, area, prio, since, today, no_commit):
    """Open a loop; re-renders NOW.md and commits the vault."""
    try:
        l = loops.add(_vault_root(), title, due, owner, area, next_, prio=prio, since=since,
                      today=_parse_day(today), commit=not no_commit)
    except loops.LoopError as e:
        _die(e)
    click.echo(l.to_line())


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
@click.option("--date", "today", default=None)
def doctor_cmd(brief, install_hooks, today):
    """Health check: binary, graph, key, vault freshness, ledgers, hooks, MCP, scheduled tasks. Exit 1 on failure."""
    root = _vault_root()
    if install_hooks:
        click.echo("pre-commit hook: " + ("installed" if decisions.install_pre_commit(root) else "vault is not a git repo"))
    checks = doctor_mod.run(root, _parse_day(today))
    click.echo(doctor_mod.brief(checks) if brief else doctor_mod.report(checks))
    sys.exit(1 if doctor_mod.worst(checks) == "fail" else 0)
