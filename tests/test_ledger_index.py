"""The ledgers as retrieval targets: `brain index` embeds every loop and decision
(incrementally, by content hash) and `graph.ledger_context` adds cosine hits to
the keyword hits when a query vector is available. Motivated by a real miss on
Sep 6 2026: "what money is Alvin still owed?" never reached L-064 "collect the
remaining 9 repayments" because the two share no keyword."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import brain.graph as graph
import brain.index as index
import brain.llm as llm
from brain import decisions, loops

from test_index import IndexTestCase


class LedgerEmbeddingTests(IndexTestCase):
    def setUp(self):
        super().setUp()
        # the base fixture's LOOPS.md is a deliberately malformed stub; start clean
        for name in ("LOOPS.md", "DECISIONS.md"):
            (self.root / name).unlink(missing_ok=True)
        loops.add(self.root, "Red Sox block: collect the remaining 9 repayments", "2026-09-21",
                  "alvin", "life", "chase the nine who have not paid", commit=False)
        decisions.append(self.root, "Vault git stays local-only", "No remote for the vault.",
                         "It holds private data.", commit=False)
        self.embed_calls: list[str] = []

        def fake_embed(text, **kw):
            self.embed_calls.append(text)
            t = text.lower()
            return [1.0, 0.0] if ("repayment" in t or "owed" in t or "paid" in t) else [0.0, 1.0]

        llm.have_key = lambda: True
        llm.embed = fake_embed

    def test_index_embeds_ledger_lines_incrementally(self):
        s = index.build(self.conn, self.root, embed=True)
        self.assertEqual(s["ledger_embedded"], 2)                        # one loop + one decision
        rows = {r[0]: r[1] for r in self.conn.execute("SELECT key, embedding FROM ledger_embeddings")}
        self.assertEqual(set(rows), {"L-001", "D-001"})
        self.assertTrue(all(rows.values()))
        n = len(self.embed_calls)
        s = index.build(self.conn, self.root, embed=True)                # nothing changed: nothing re-embedded
        self.assertEqual(s["ledger_embedded"], 0)
        self.assertEqual(len(self.embed_calls), n)
        loops.edit(self.root, "L-001", next_="chase the nine, then close", commit=False)
        s = index.build(self.conn, self.root, embed=True)                # the changed line is re-embedded
        self.assertEqual(s["ledger_embedded"], 1)

    def test_a_dropped_ledger_line_leaves_the_table(self):
        index.build(self.conn, self.root, embed=True)
        (self.root / "DECISIONS.md").unlink()
        index.build(self.conn, self.root, embed=True)
        keys = {r[0] for r in self.conn.execute("SELECT key FROM ledger_embeddings")}
        self.assertEqual(keys, {"L-001"})

    def test_semantic_hit_reaches_ledger_context(self):
        index.build(self.conn, self.root, embed=True)
        self.assertEqual(index.ledger_semantic(self.conn, [1.0, 0.0])[0][0], "L-001")
        q = "what money is still owed to me"
        lines, sources = graph.ledger_context(q, self.root, conn=self.conn, query_vector=[1.0, 0.0])
        self.assertIn("L-001", sources)                                  # no keyword overlap: cosine found it
        self.assertNotIn("D-001", sources)                               # the orthogonal decision stays out
        self.assertTrue(any("remaining 9 repayments" in ln for ln in lines))
        lines, sources = graph.ledger_context(q, self.root)              # keyword-only path unchanged
        self.assertNotIn("L-001", sources)

    def test_without_a_key_nothing_is_embedded_and_nothing_breaks(self):
        llm.have_key = lambda: False
        s = index.build(self.conn, self.root, embed=True)
        self.assertEqual(s["ledger_embedded"], 0)
        self.assertEqual(index.ledger_semantic(self.conn, [1.0, 0.0]), [])
        lines, sources = graph.ledger_context("repayments", self.root, conn=self.conn, query_vector=[1.0, 0.0])
        self.assertIn("L-001", sources)                                  # keyword hit still works

    def test_status_reports_stale_ledger_lines(self):
        s = index.status(self.conn, self.root)
        self.assertEqual((s["ledger_total"], s["ledger_embedded"]), (2, 0))
        self.assertEqual(s["ledger_stale"], ["D-001", "L-001"])          # never indexed: everything is stale
        index.build(self.conn, self.root, embed=True)
        s = index.status(self.conn, self.root)
        self.assertEqual((s["ledger_stale"], s["ledger_embedded"]), ([], 2))
        loops.edit(self.root, "L-001", next_="chase the nine, then close", commit=False)
        self.assertEqual(index.status(self.conn, self.root)["ledger_stale"], ["L-001"])
        (self.root / "DECISIONS.md").unlink()
        self.assertEqual(index.status(self.conn, self.root)["ledger_stale"], ["D-001", "L-001"])  # vanished counts too
