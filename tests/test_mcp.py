"""MCP server tests — stdlib unittest only, no network (Gemini is mocked).

Drives brain/mcp.py at the JSON-RPC layer via handle_line/handle_message,
the same entry points the stdio loop uses, so the wire behaviour is what's
tested without spawning a subprocess.
"""
import io
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import brain.config as config
import brain.db as db
import brain.llm as llm
import brain.mcp as mcp

DAY = 86400.0


def rpc(method, params=None, msg_id=1):
    msg = {"jsonrpc": "2.0", "method": method, "id": msg_id}
    if params is not None:
        msg["params"] = params
    return mcp.handle_message(msg)


def call_tool(name, arguments=None):
    resp = rpc("tools/call", {"name": name, "arguments": arguments or {}})
    return resp["result"]


def tool_text(result):
    return result["content"][0]["text"]


class MCPTestCase(unittest.TestCase):
    """Each test gets a fresh temp database; the LLM boundary is offline."""

    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self._orig_db_path = db.DB_PATH
        db.DB_PATH = os.path.join(self._tmp, "brain.db")
        self._orig_config_path = config.CONFIG_PATH
        config.CONFIG_PATH = Path(self._tmp) / "config.json"
        config.save({"vault_dir": str(Path(self._tmp) / "vault")})   # never the real vault
        (Path(self._tmp) / "vault").mkdir()
        self._orig_have_key = llm.have_key
        self._orig_generate = llm.generate
        llm.have_key = lambda: False

    def tearDown(self):
        db.DB_PATH = self._orig_db_path
        config.CONFIG_PATH = self._orig_config_path
        llm.have_key = self._orig_have_key
        llm.generate = self._orig_generate

    def seed(self, name, type_="concept", content="", **kw):
        conn = db.connect()
        nid = db.add_node(conn, name, type_=type_, content=content, **kw)
        conn.commit()
        conn.close()
        return nid


class HandshakeTests(MCPTestCase):
    def test_initialize(self):
        resp = rpc("initialize", {"protocolVersion": "2025-06-18",
                                  "capabilities": {}, "clientInfo": {"name": "t"}})
        self.assertEqual(resp["id"], 1)
        r = resp["result"]
        self.assertEqual(r["protocolVersion"], "2025-06-18")
        self.assertIn("tools", r["capabilities"])
        self.assertEqual(r["serverInfo"]["name"], "personal-brain")
        self.assertTrue(r["instructions"])

    def test_initialize_echoes_older_protocol_version(self):
        resp = rpc("initialize", {"protocolVersion": "2024-11-05"})
        self.assertEqual(resp["result"]["protocolVersion"], "2024-11-05")

    def test_initialize_with_missing_version_uses_default(self):
        resp = rpc("initialize", {})
        self.assertEqual(resp["result"]["protocolVersion"], mcp.PROTOCOL_VERSION)

    def test_initialized_notification_gets_no_response(self):
        resp = mcp.handle_message({"jsonrpc": "2.0", "method": "notifications/initialized"})
        self.assertIsNone(resp)

    def test_ping(self):
        self.assertEqual(rpc("ping")["result"], {})

    def test_unknown_method_errors(self):
        resp = rpc("resources/list")
        self.assertEqual(resp["error"]["code"], -32601)

    def test_unknown_notification_stays_silent(self):
        resp = mcp.handle_message({"jsonrpc": "2.0", "method": "some/unknown"})
        self.assertIsNone(resp)


class WireTests(MCPTestCase):
    def test_parse_error(self):
        resp = mcp.handle_line("this is not json")
        self.assertEqual(resp["error"]["code"], -32700)

    def test_non_object_message(self):
        resp = mcp.handle_line("[1, 2, 3]")
        self.assertEqual(resp["error"]["code"], -32600)

    def test_blank_line_ignored(self):
        self.assertIsNone(mcp.handle_line("   \n"))

    def test_serve_loop_speaks_ndjson(self):
        stdin = io.StringIO(
            json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                        "params": {"protocolVersion": "2025-06-18"}}) + "\n"
            + json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n"
            + json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}) + "\n"
        )
        stdout = io.StringIO()
        mcp.serve(stdin=stdin, stdout=stdout)
        lines = [l for l in stdout.getvalue().splitlines() if l]
        self.assertEqual(len(lines), 2)  # notification produced no output
        first, second = (json.loads(l) for l in lines)
        self.assertEqual(first["id"], 1)
        self.assertEqual(second["id"], 2)
        self.assertTrue(second["result"]["tools"])


class ToolsListTests(MCPTestCase):
    def test_all_tools_listed_with_valid_schemas(self):
        tools = rpc("tools/list")["result"]["tools"]
        names = {t["name"] for t in tools}
        self.assertEqual(names, {"brain_remember", "brain_search", "brain_ask",
                                 "brain_context", "brain_digest"})
        for t in tools:
            self.assertTrue(t["description"])
            self.assertEqual(t["inputSchema"]["type"], "object")
            for req in t["inputSchema"].get("required", []):
                self.assertIn(req, t["inputSchema"]["properties"])

    def test_every_listed_tool_has_a_handler(self):
        listed = {t["name"] for t in mcp.TOOLS}
        self.assertEqual(listed, set(mcp.HANDLERS))


class ToolCallTests(MCPTestCase):
    def test_unknown_tool_is_protocol_error(self):
        resp = rpc("tools/call", {"name": "brain_explode", "arguments": {}})
        self.assertEqual(resp["error"]["code"], -32602)
        self.assertIn("brain_explode", resp["error"]["message"])

    def test_missing_required_arg_is_tool_error_not_crash(self):
        result = call_tool("brain_remember", {})
        self.assertTrue(result["isError"])
        self.assertIn("text", tool_text(result))

    def test_handler_exception_becomes_is_error_result(self):
        self.seed("Football", content="plays football")  # so retrieval reaches the LLM
        llm.generate = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("api down"))
        result = call_tool("brain_ask", {"question": "football"})
        self.assertTrue(result["isError"])
        self.assertIn("api down", tool_text(result))

    def test_remember_ingests_nodes(self):
        llm.generate = lambda *a, **k: json.dumps({
            "nodes": [{"name": "Rust", "type": "skill",
                       "content": "Learning Rust.", "confidence": 0.9, "importance": 0.7}],
            "edges": [],
        })
        result = call_tool("brain_remember", {"text": "I started learning Rust",
                                              "source": "test"})
        self.assertFalse(result["isError"])
        self.assertIn("Rust", tool_text(result))
        conn = db.connect()
        node = db.get_node_by_name(conn, "Rust")
        self.assertIsNotNone(node)
        self.assertEqual(node["type"], "skill")
        conn.close()

    def test_search_keyword_fallback_finds_and_reinforces(self):
        nid = self.seed("Transformers", content="attention is all you need")
        conn = db.connect()
        conn.execute("UPDATE nodes SET weight=0.5 WHERE id=?", (nid,))
        conn.commit()
        conn.close()
        result = call_tool("brain_search", {"query": "transformers"})
        self.assertFalse(result["isError"])
        self.assertIn("Transformers", tool_text(result))
        conn = db.connect()
        self.assertEqual(db.get_node(conn, nid)["weight"], 1.0)  # access reinforces
        conn.close()

    def test_search_semantic_path_reinforces(self):
        nid = self.seed("Neural nets", content="deep learning")
        conn = db.connect()
        db.set_embedding(conn, nid, [1.0, 0.0])
        conn.execute("UPDATE nodes SET weight=0.5 WHERE id=?", (nid,))
        conn.commit()
        conn.close()
        llm.have_key = lambda: True
        orig_embed = llm.embed
        llm.embed = lambda *a, **k: [1.0, 0.0]
        try:
            result = call_tool("brain_search", {"query": "machine learning"})
        finally:
            llm.embed = orig_embed
        self.assertIn("Neural nets", tool_text(result))
        conn = db.connect()
        self.assertEqual(db.get_node(conn, nid)["weight"], 1.0)
        conn.close()

    def test_search_no_results(self):
        result = call_tool("brain_search", {"query": "nothing here"})
        self.assertFalse(result["isError"])
        self.assertIn("No memories", tool_text(result))

    def test_ask_answers_with_sources(self):
        self.seed("Espoo", type_="fact", content="Alvin lives in Espoo")
        llm.generate = lambda *a, **k: "In Espoo."
        result = call_tool("brain_ask", {"question": "where does Alvin live"})
        text = tool_text(result)
        self.assertIn("In Espoo.", text)
        self.assertIn("sources: Espoo", text)

    def test_context_synthesizes_document(self):
        self.seed("ML", content="machine learning studies")
        llm.generate = lambda *a, **k: "## Background\nA doc."
        result = call_tool("brain_context", {"topic": "ML"})
        self.assertIn("## Background", tool_text(result))

    def test_context_empty_brain(self):
        result = call_tool("brain_context", {})
        self.assertFalse(result["isError"])
        self.assertIn("empty", tool_text(result))

    def test_digest_is_deterministic_and_offline(self):
        self.seed("Thesis", type_="project", content="MS thesis", importance=0.9)
        self.seed("Email Heli", type_="task", importance=0.3)
        llm.generate = lambda *a, **k: self.fail("digest must not call the LLM")
        result = call_tool("brain_digest")
        text = tool_text(result)
        self.assertIn("Thesis", text)
        self.assertIn("Email Heli", text)

    def test_digest_empty_brain(self):
        self.assertIn("empty", tool_text(call_tool("brain_digest")))

    def test_decay_runs_on_tool_calls(self):
        nid = self.seed("Old meeting", type_="event", importance=0.1)
        conn = db.connect()
        conn.execute("UPDATE nodes SET last_accessed=? WHERE id=?",
                     (time.time() - 60 * DAY, nid))
        conn.commit()
        conn.close()
        call_tool("brain_digest")
        conn = db.connect()
        self.assertEqual(db.get_node(conn, nid)["archived"], 1)
        conn.close()


if __name__ == "__main__":
    unittest.main()
