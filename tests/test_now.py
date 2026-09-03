"""Tests for the generated NOW.md (brain/now.py). Temp vault, fixed dates, no git."""
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import brain.loops as loops
import brain.now as now

TODAY = date(2026, 9, 2)


def w(p: Path, text: str):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


class NowTestCase(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        w(self.root / "IDENTITY.md", "**Alvin Ekelund** — Harvard SM Data Science, classes began Sep 2 2026.\n")
        w(self.root / "areas/harvard.md",
          "---\narea: harvard\nupdated: 2026-09-02\naliases: [mit, cross-reg, petition]\n---\n"
          "# Harvard SM in Data Science\n\n## Now\n- 6.4212 approved; 9.522 pending\n- AM 207 backstop\n\n## Later\n- spring plan\n")
        w(self.root / "areas/jobs.md",
          "---\narea: jobs\nupdated: 2026-08-31\naliases: [bain, bcg]\n---\n# Job search\n\n## Now\n- Bain applied\n")
        w(self.root / "apps/crimson-track.md",
          "---\nname: Crimson Track\npurpose: course planner, plan of record\nurl: https://claude.ai/code/artifact/abc\nupdated: 2026-09-01\n---\n# Crimson Track\n")
        w(self.root / "apps/wakequest.md",
          "---\nname: WakeQuest\npurpose: alarm you have to earn\ntype: native\nupdated: 2026-09-01\n---\n# WakeQuest\n")
        w(self.root / "apps/README.md", "# apps\n")
        w(self.root / "people/anna-houstecka.md", "---\nperson: anna-houstecka\nname: Anna Houstecka\nrole: Alvin's girlfriend\n---\n# Anna\n")
        w(self.root / "people/boaz-barak.md", "---\nperson: boaz-barak\nname: Boaz Barak\nrole: Harvard ML\n---\n# Boaz\n")
        w(self.root / "log/2026-09-01.md", "- Bain applied; Kaelbling approved the MIT petition.\n")
        loops.add(self.root, "Lock the fourth seat", "2026-09-09", "alvin", "harvard", "enroll", prio=1, today=TODAY, commit=False)
        loops.add(self.root, "Draft email", "2026-09-02", "claude", "harvard", "draft", prio=1, today=TODAY, commit=False)
        loops.add(self.root, "Boaz follow-up", "2026-09-08", "waiting:boaz-barak", "harvard", "nudge", prio=3, today=TODAY, commit=False)
        loops.add(self.root, "Bainworks", "2026-09-08", "alvin", "jobs", "register", prio=1, today=TODAY, commit=False)


class FrontmatterTests(unittest.TestCase):
    def test_parse_and_dump(self):
        fm, body = now.parse_frontmatter("---\narea: harvard\nupdated: 2026-09-02\naliases: [mit, cross-reg]\n---\n# T\nbody\n")
        self.assertEqual(fm, {"area": "harvard", "updated": "2026-09-02", "aliases": ["mit", "cross-reg"]})
        self.assertEqual(body, "# T\nbody\n")
        self.assertEqual(now.parse_frontmatter("# no fm\n"), ({}, "# no fm\n"))
        self.assertEqual(now.dump_frontmatter(fm), "---\narea: harvard\nupdated: 2026-09-02\naliases: [mit, cross-reg]\n---\n")
        with self.assertRaises(now.NowError):
            now.parse_frontmatter("---\nnot a pair\n---\n")

    def test_now_block(self):
        body = "# T\n## Now\n- a\n\n- b\n## Later\n- c\n"
        self.assertEqual(now.now_block(body), ["- a", "- b"])
        self.assertEqual(now.now_block("# T\n## Later\n- c\n"), [])


class RenderTests(NowTestCase):
    def test_render_sections(self):
        text = now.render_text(self.root)
        self.assertTrue(text.startswith("# NOW"))
        self.assertIn(now.GEN_MARK, text)
        self.assertIn("**Alvin Ekelund**", text)
        self.assertIn("- **harvard:** L-002 Draft email — Sep 2 P1 · claude; L-001 Lock the fourth seat — Sep 9 P1", text)
        self.assertIn("- **jobs:** L-004 Bainworks — Sep 8 P1", text)
        self.assertIn("⏳ **waiting on:** boaz-barak — Boaz follow-up (by Sep 8, L-003)", text)
        self.assertNotIn("enroll", text.split("## 📍")[0])          # next actions belong to `brain today`
        self.assertIn("### Harvard SM in Data Science · updated Sep 2\n- 6.4212 approved; 9.522 pending\n- AM 207 backstop", text)
        self.assertIn("### Job search · updated Aug 31", text)
        self.assertNotIn("spring plan", text)                           # only the ## Now block renders
        self.assertIn("**Anna Houstecka** — Alvin's girlfriend · **Boaz Barak** — Harvard ML (⏳ L-003)", text)
        self.assertIn("- **Crimson Track** — course planner, plan of record — https://claude.ai/code/artifact/abc · updated Sep 1", text)
        self.assertIn("- **WakeQuest** — alarm you have to earn — native (no url) · updated Sep 1", text)
        self.assertTrue(text.rstrip().endswith("`brain doctor`."))
        self.assertNotIn(TODAY.isoformat(), text)                       # deterministic: no clock

    def test_index_only_people_stay_out_of_now(self):
        """`now: false` keeps a person retrievable in people/ without lengthening NOW.md."""
        w(self.root / "people/kosuke-imai.md",
          "---\nperson: kosuke-imai\nname: Kosuke Imai\nrole: teaches STAT 286\nnow: false\n---\n# Kosuke\n")
        text = now.render_text(self.root)
        self.assertNotIn("Kosuke Imai", text)
        self.assertIn("**Anna Houstecka** — Alvin's girlfriend", text)
        errors, _ = now.lint(self.root, TODAY)
        self.assertFalse([e for e in errors if "kosuke" in e])

    def test_write_is_idempotent_and_loop_ops_rerender(self):
        self.assertTrue(now.write(self.root))
        self.assertFalse(now.write(self.root))
        self.assertTrue(now.is_generated(self.root))
        loops.done(self.root, "L-002", today=TODAY, commit=False)
        text = (self.root / "NOW.md").read_text()
        self.assertNotIn("Draft email", text)                           # loops.done re-rendered the whole file
        self.assertEqual(now.stale(self.root), "")

    def test_stale_detection(self):
        self.assertIn("missing", now.stale(self.root))
        w(self.root / "NOW.md", "# NOW\nhand written\n")
        self.assertIn("hand-written", now.stale(self.root))
        now.write(self.root)
        p = self.root / "NOW.md"
        p.write_text(p.read_text().replace("AM 207 backstop", "AM 207 fallback"))
        self.assertIn("out of date", now.stale(self.root))
        errors, _ = loops.lint(self.root, TODAY)                        # loop lint delegates to now.stale
        self.assertTrue(any("out of date" in e for e in errors))

    def test_more_than_four_now_lines_are_truncated(self):
        w(self.root / "areas/jobs.md", "---\narea: jobs\nupdated: 2026-09-02\n---\n# Jobs\n## Now\n" + "".join(f"- l{i}\n" for i in range(6)))
        text = now.render_text(self.root)
        self.assertIn("- l3", text)
        self.assertNotIn("- l4", text)


class LintTests(NowTestCase):
    def test_clean_after_render(self):
        now.write(self.root)
        errors, warnings = now.lint(self.root, TODAY)
        self.assertEqual(errors, [])
        self.assertEqual(len(warnings), 1)          # only the deliberately stale jobs.md (see next test)

    def test_log_mention_after_update_warns(self):
        # jobs.md updated 2026-08-31; log/2026-09-01.md says "Bain applied" (alias bain) → stale
        now.write(self.root)
        _, warnings = now.lint(self.root, TODAY)
        self.assertTrue(any("areas/jobs.md: updated 2026-08-31 but 2026-09-01.md" in w_ for w_ in warnings), warnings)
        # harvard.md updated 2026-09-02, the log is dated 09-01 → not stale even though it mentions the petition
        self.assertFalse(any("harvard" in w_ for w_ in warnings))
        now.touch_area(self.root, "jobs", when=TODAY)
        self.assertEqual(now.lint(self.root, TODAY)[1], [])
        self.assertIn("updated: 2026-09-02", (self.root / "areas/jobs.md").read_text())
        self.assertIn("### Job search · updated Sep 2", (self.root / "NOW.md").read_text())

    def test_errors(self):
        now.write(self.root)
        w(self.root / "areas/training.md", "# Training\n## Now\n- x\n")                          # no front-matter
        w(self.root / "areas/life.md", "---\narea: life\nupdated: soon\n---\n# Life\n")             # bad date
        w(self.root / "areas/other.md", "---\narea: garden\nupdated: 2026-09-01\n---\n# Garden\n")  # unknown key
        w(self.root / "areas/harvard2.md", "---\narea: harvard\nupdated: 2026-09-01\n---\n# Dup\n")  # duplicate
        errors, warnings = now.lint(self.root, TODAY)
        joined = "\n".join(errors)
        self.assertIn("areas/training.md: front-matter needs `area:", joined)
        self.assertIn("areas/life.md: bad or missing date", joined)
        self.assertIn("unknown area 'garden'", joined)
        self.assertIn("duplicate area 'harvard'", joined)
        self.assertTrue(any("out of date" in e for e in errors))       # sources changed, NOW.md not re-rendered
        self.assertTrue(any("no `## Now` block" in w_ for w_ in warnings))

    def test_app_and_people_warnings(self):
        now.write(self.root)
        w(self.root / "apps/deck.md", "# Deck\n**URL:** x\n")
        w(self.root / "apps/radar.md", "---\nname: Radar\npurpose: p\n---\n# Radar\n")
        w(self.root / "people/heli.md", "---\nperson: heli\nname: Heli\n---\n# Heli\n")
        now.write(self.root)
        _, warnings = now.lint(self.root, TODAY)
        joined = "\n".join(warnings)
        self.assertIn("apps/deck.md: no front-matter", joined)
        self.assertIn("apps/radar.md: artifact without `url:`", joined)
        self.assertIn("people/heli.md: no `role:`", joined)

    def test_touch_unknown_area(self):
        with self.assertRaises(now.NowError):
            now.touch_area(self.root, "garden", when=TODAY)


if __name__ == "__main__":
    unittest.main()
