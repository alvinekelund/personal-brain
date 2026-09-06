"""MCP server over stdio — pure stdlib, no SDK.

Exposes the brain to any MCP client (Claude Code, Claude Desktop, Cursor, ...)
as tools, so agents can read from and write to the graph mid-conversation:

    claude mcp add brain -- brain mcp

The MCP stdio transport is newline-delimited JSON-RPC 2.0 on stdin/stdout.
stdout carries ONLY protocol messages; anything else (diagnostics) goes to
stderr. Decay runs on every tool call, same as the CLI.
"""
import json
import sys

from brain import config, db, decay, extract, graph, llm

PROTOCOL_VERSION = "2025-06-18"
SERVER_INFO = {"name": "personal-brain", "version": "0.1.0"}

INSTRUCTIONS = (
    "Persistent memory for the user, stored as a local typed knowledge graph "
    "with human-like forgetting. Call brain_context or brain_digest at the "
    "start of a session to load who the user is and what they're working on. "
    "Call brain_remember whenever the user shares something durable — facts, "
    "decisions, preferences, project updates, people. The graph is CONTEXT, not "
    "a to-do list: any open action item in remembered text is routed to the "
    "vault's LOOPS-INBOX.md for triage (`brain loop inbox`), and brain_digest "
    "lists open loops from LOOPS.md. Call brain_ask or brain_search to recall "
    "specifics; both return vault file paths — the files are the source of truth, read them. Reading a memory reinforces it; unaccessed memories fade on a "
    "forgetting curve."
)

TOOLS = [
    {
        "name": "brain_remember",
        "description": (
            "Save knowledge to the user's personal knowledge graph. Pass any text "
            "worth remembering — a fact learned, a decision made, a preference "
            "expressed, a project update, a person mentioned. Typed entities and "
            "relationships are extracted automatically and merged with what the "
            "graph already knows (duplicates reinforce rather than duplicate). "
            "Open action items are NOT stored as nodes — they go to the loop inbox."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The knowledge to remember, in plain prose. Complete sentences with names and context extract best.",
                },
                "source": {
                    "type": "string",
                    "description": "Where this came from, e.g. 'claude-code session', a URL, or a filename.",
                },
            },
            "required": ["text"],
        },
    },
    {
        "name": "brain_search",
        "description": (
            "Search the user's brain: the vault files (the source of truth) and the "
            "knowledge graph that indexes them. Ranks by meaning (embeddings) when "
            "available, falling back to stem-aware keyword match. Returns matching "
            "nodes (with the vault file each points to) and the matching files by "
            "path — read the files for the full picture. Accessing a node "
            "reinforces it against forgetting."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to look for."},
                "limit": {
                    "type": "integer",
                    "description": "Maximum results (default 8).",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "brain_ask",
        "description": (
            "Ask a natural-language question about the user; answered strictly "
            "from their vault files (routed via the graph index) plus ledgers and "
            "graph nodes, with the file paths and source ids listed — open the "
            "files when you need more than the answer. Use for recall, not for saving."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "The question to answer from the graph."},
            },
            "required": ["question"],
        },
    },
    {
        "name": "brain_context",
        "description": (
            "Generate a structured briefing about a topic — or about the whole "
            "person if no topic is given — with sections: Background, Active "
            "Skills, Current Focus, Projects, Open Questions. Use at session "
            "start to load context, or before a task that needs to know the "
            "user's situation."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "Topic to focus the briefing on. Omit for a full-person briefing.",
                },
            },
        },
    },
    {
        "name": "brain_digest",
        "description": (
            "What's top of mind for the user right now: highest-importance items, "
            "open tasks, memories about to fade, and life-area balance. Fast and "
            "deterministic — no LLM call. Good cheap default at session start."
        ),
        "inputSchema": {"type": "object", "properties": {}},
    },
]


# ── tool handlers ─────────────────────────────────────────────────────────────
# Each takes (conn, args) and returns the text content for the tool result.
# Raising is fine — the dispatcher converts exceptions into isError results.

def _fmt_node(n) -> str:
    content = (n["content"] or "").strip()
    path = n["path"] if "path" in n.keys() and n["path"] else ""
    return f"[{n['type']}] {n['name']} (weight {n['weight']:.2f}): {content}" + (f"  → {path}" if path else "")


def _fmt_files(files: list) -> str:
    return "files (the vault is the source of truth — read these):\n" + "\n".join(
        f"  {f['path']} — {f['title']} ({', '.join(f['why'])})" for f in files)


def _remember(conn, args):
    text = (args.get("text") or "").strip()
    if not text:
        raise ValueError("'text' is required and must be non-empty.")
    user = config.get_user()
    node_ids, edge_ids = extract.ingest(
        conn, text, source=args.get("source") or "mcp", user=user
    )
    names = []
    for nid in node_ids:
        n = db.get_node(conn, nid)
        if n:
            names.append(f"[{n['type']}] {n['name']}")
    summary = f"Remembered: {len(node_ids)} node(s), {len(edge_ids)} edge(s)."
    if names:
        summary += "\n" + "\n".join(f"  {x}" for x in names[:10])
    if len(names) > 10:
        summary += f"\n  ... and {len(names) - 10} more"
    return summary


def _search(conn, args):
    query = (args.get("query") or "").strip()
    if not query:
        raise ValueError("'query' is required and must be non-empty.")
    limit = int(args.get("limit") or 8)

    results = []
    if llm.have_key():  # meaning-based first; any failure falls through to keyword
        try:
            scored = graph.semantic_search(conn, llm.embed(query), limit=limit)
            results = [r for score, r in scored if score >= graph.SEMANTIC_SEED_MIN_SIM]
        except Exception:
            results = []
    if not results:
        results = graph.query_nodes(conn, query)[:limit]
    else:
        for r in results:  # semantic path bypasses query_nodes' touch — reinforce here
            db.touch_node(conn, r["id"])
        conn.commit()

    from brain import index
    files = index.search(conn, query, k=min(limit, 6), seed_node_ids=[r["id"] for r in results])
    if not results and not files:
        return f"No memories match '{query}'."
    out = "\n".join(_fmt_node(r) for r in results)
    if files:
        out = (out + "\n\n" if out else "") + _fmt_files(files)
    return out


def _ask(conn, args):
    question = (args.get("question") or "").strip()
    if not question:
        raise ValueError("'question' is required and must be non-empty.")
    res = graph.answer_question(conn, question)
    out = res["answer"]
    if res.get("files"):
        out += "\n\nfiles: " + ", ".join(res["files"])
    if res["sources"]:
        out += "\nsources: " + ", ".join(res["sources"])
    return out


def _context(conn, args):
    topic = (args.get("topic") or "").strip()
    nodes, used_fallback = graph.collect_context_nodes(conn, topic=topic)
    if not nodes:
        return "The brain is empty — nothing to brief on yet."
    file_lines, ledger_lines = graph.context_material(conn, topic, nodes)
    doc = graph.synthesize_context(nodes, topic=topic, file_lines=file_lines, ledger_lines=ledger_lines)
    if topic and used_fallback:
        doc = f"(Nothing matched '{topic}' directly; this briefs the whole brain.)\n\n" + doc
    return doc


def _digest(conn, args):
    d = graph.digest(conn, config.get_user())
    lines = []
    if d["top"]:
        lines.append("Top of mind:")
        lines += [f"  [{t['type']}] {t['name']} (importance {t['importance']}, weight {t.get('weight', 1.0)})"
                  for t in d["top"]]
    if d["tasks"]:
        lines.append("Open loops (LOOPS.md):")
        lines += [f"  - {t}" for t in d["tasks"]]
    if d["fading"]:
        lines.append("Fading soon:")
        lines += [f"  - {f['name']} (~{max(f['days_left'], 0):.0f}d left)" for f in d["fading"]]
    if d["areas"]:
        lines.append("By area: " + ", ".join(f"{n} ({c})" for n, c in d["areas"]))
    return "\n".join(lines) if lines else "The brain is empty."


HANDLERS = {
    "brain_remember": _remember,
    "brain_search": _search,
    "brain_ask": _ask,
    "brain_context": _context,
    "brain_digest": _digest,
}


# ── JSON-RPC dispatch ─────────────────────────────────────────────────────────

def _result(msg_id, result) -> dict:
    return {"jsonrpc": "2.0", "id": msg_id, "result": result}


def _error(msg_id, code, message) -> dict:
    return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}


def _call_tool(params: dict) -> dict:
    name = params.get("name", "")
    handler = HANDLERS.get(name)
    if handler is None:
        raise KeyError(name)
    conn = db.connect()
    try:
        decay.run_decay(conn)  # forgetting advances on every access, like the CLI
        text = handler(conn, params.get("arguments") or {})
        return {"content": [{"type": "text", "text": text}], "isError": False}
    except Exception as e:
        # tool-level failures (bad args, no API key, network) go back as results
        # so the calling model can read them and adapt — not as protocol errors
        return {"content": [{"type": "text", "text": f"{type(e).__name__}: {e}"}], "isError": True}
    finally:
        conn.close()


def handle_message(msg: dict):
    """Handle one decoded JSON-RPC message. Returns the response dict, or None
    when no response is due (notifications)."""
    method = msg.get("method", "")
    msg_id = msg.get("id")
    params = msg.get("params") or {}

    if method.startswith("notifications/"):
        return None
    if method == "initialize":
        # echo the client's requested version — this server only uses tool
        # primitives, which are stable across every published protocol revision
        version = params.get("protocolVersion") or PROTOCOL_VERSION
        return _result(msg_id, {
            "protocolVersion": version if isinstance(version, str) else PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": SERVER_INFO,
            "instructions": INSTRUCTIONS,
        })
    if method == "ping":
        return _result(msg_id, {})
    if method == "tools/list":
        return _result(msg_id, {"tools": TOOLS})
    if method == "tools/call":
        try:
            return _result(msg_id, _call_tool(params))
        except KeyError as e:
            return _error(msg_id, -32602, f"Unknown tool: {e.args[0]}")
    if msg_id is None:  # unknown notification — stay silent
        return None
    return _error(msg_id, -32601, f"Method not found: {method}")


def handle_line(line: str):
    """Decode and handle one wire line; never raises (protocol errors are replies)."""
    line = line.strip()
    if not line:
        return None
    try:
        msg = json.loads(line)
    except json.JSONDecodeError as e:
        return _error(None, -32700, f"Parse error: {e}")
    if not isinstance(msg, dict):
        return _error(None, -32600, "Invalid request: expected an object")
    return handle_message(msg)


def serve(stdin=None, stdout=None):
    """Run the stdio server loop until EOF."""
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    print("personal-brain MCP server on stdio", file=sys.stderr)
    for line in stdin:
        response = handle_line(line)
        if response is not None:
            stdout.write(json.dumps(response) + "\n")
            stdout.flush()
