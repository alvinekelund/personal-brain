# Personal Brain

[![CI](https://github.com/alvinekelund/personal-brain/actions/workflows/ci.yml/badge.svg)](https://github.com/alvinekelund/personal-brain/actions/workflows/ci.yml)

A local-first personal knowledge graph that doubles as a memory layer for AI agents. Paste text, a URL, or a file — Gemini extracts entities and relationships automatically. Every node decays via an Ebbinghaus forgetting curve; accessing it resets the clock. The graph is explorable as an interactive Pyvis visualization, any slice of it can be synthesized into a structured context document, and a built-in MCP server lets Claude Code, Claude Desktop, or any MCP client read and write the brain mid-conversation.

No cloud. No accounts. Data lives in `~/.personal-brain/brain.db`.

---

## What it does

**Ingestion** — point it at anything: a paragraph, an article, a book chapter, a meeting transcript. Gemini Flash extracts typed nodes (concepts, skills, projects, people, facts, insights, events) and semantic edges (builds_on, requires, contradicts, part_of, etc.). Long input is chunked so nothing is dropped, and the chunks are extracted in parallel. Every model call has a wall-clock budget and an ingest has a deadline past which the best-effort stages (entity linking, embeddings) are skipped rather than started, so a slow model can never stall a scheduled run; `brain add` prints each stage with elapsed time, retry and rate-limit waits included. The extractor is shown the existing nodes *relevant to the text* (keyword, then semantic, then the most important — not the oldest sixty), sub-categories as "Area > Sub-category", and refuses names that mean nothing out of context ("New Project (Harvard)", "the meeting", a bare "September 2026"); a re-mentioned node absorbs genuinely new content instead of dropping it, and a parentless fact goes under the node its own edge names. The graph is *context*, not a to-do list: open action items in the text are returned separately and routed to the vault's `LOOPS-INBOX.md` for triage into real loops — they never become nodes.

**Person-rooted hierarchy** — it's a graph, but with a backbone: every node is placed under a `category` (Career, Hobbies, Relationships, …) that hangs off *you*, forming a `You → Category → Topic → Detail` tree via `part_of` — while cross-links between branches keep it a graph. `brain tree` prints the hierarchy; `brain reorganize` retrofits an existing flat brain into it; `brain subgroup` splits an oversized area into sub-categories (a category under a category), and `brain move` / `rename` / `retype` / `merge` curate it deterministically.

**Importance-weighted forgetting** — each node has an `importance` (0–1, scored at ingest) and a type half-life. Important nodes decay much slower and never auto-archive; one-off details still fade. People and organisations never decay — but only while they matter: below importance 0.4 a one-off sponsor or a passing acquaintance is remembered like a concept, not forever. Accessing a node resets it to 1.0 and propagates a freshness boost up its branch. Decay runs automatically on every CLI call but is purely time-based: nodes and edges carry a `last_decayed` clock and only the interval since then is applied, so call frequency never compounds it. Hierarchy (`part_of`) edges may fade but are never pruned — the spine survives long silences.

**Search** — `brain query "x"` (stem-aware keyword) or `brain query "x" --semantic` (embedding cosine — finds by *meaning*, e.g. "machine learning" surfaces your neural-net nodes). `brain reindex` (re)computes embeddings; new nodes are embedded automatically on `add`.

**Visualization** — `brain show` opens an interactive force-directed graph: categories are large hubs, the `part_of` backbone is solid arrows, cross-links are dashed, node size reflects importance. Hover for content; filter by weight or type.

**Context injection** — `brain context "ML internships"` seeds from keyword → semantic → whole-brain (in that order), BFS-traverses (hub-aware), adds the topic's vault file excerpts and ledger lines (the file wins over the graph; dates and numbers come from the files), then calls Gemini to synthesise a structured document: Background, Active Skills, Current Focus, Projects, Open Questions. Pipe it straight into any AI conversation.

**Synthesis** — `brain synthesize` finds isolated nodes and connects them to the graph, surfacing relationships Gemini notices across your knowledge.

**Generated NOW.md** — the vault's "what is going on" view is never hand-written. `brain now render` composes it from files that each have one writer: `IDENTITY.md` (a curated paragraph), `LOOPS.md` (open loops grouped by area), every `areas/<area>.md` front-matter (`area:`, `updated:`, `aliases:`) plus its `## Now` block (≤4 lines), `people/*.md` and `apps/*.md` front-matter. `brain now lint` proves NOW.md is current and flags an area whose `updated:` is older than a log entry that mentions one of its aliases — the fact moved, the area did not. Together with `brain today` (what to do, time-sorted) and the agent's rules file (how to behave), a session reads three things with three distinct jobs.

**Action layer** — `LOOPS.md` (one loop per line in a strict grammar: id, title, due, owner alvin/claude/waiting:<who>, area, prio, next action) and `DECISIONS.md` (append-only: decision, why, rejected, revisit-if). Both are written only by `brain loop` / `brain decide`; `lint` fails on hand edits and a git pre-commit hook rejects removals from the decision ledger. NOW.md's "hot" section is rendered from the loops. `brain today` turns them into a deterministic action card (countdowns, aging waits, Claude-owned loops, top 3 next actions) and `brain doctor` checks the whole wiring — binary, DB, key, vault freshness, hooks, MCP registration, scheduled tasks — so a broken brain announces itself at session start instead of failing silently.

**Tree integrity** — the person-rooted hierarchy is checked on every `brain doctor`: orphans, nodes with more than one `part_of` parent, categories not hanging off the person or a parent category, cycles, edges left behind by a deleted node, legacy task nodes, near-duplicate names (including a bare first name vs a full name, and two same-type nodes that `brain index` maps to one vault file), nodes without embeddings, and categories with more direct children than the sub-grouping threshold (the cure is named: `brain subgroup`). `brain repair` fixes the structural ones deterministically and never deletes a node. Ingest and reorganize keep the invariant at write time: a planned parent replaces the old one; `brain merge` keeps one parent (the other becomes a cross-link) and `brain move` re-homes a node under the same rules.

**Ambient capture that resists replay** — the SessionEnd hook ignores turns older than 36 hours (a resumed old session is not today's truth), skips automation sessions (scheduled-task prompts), tells the distiller to keep only the latest state when a plan is superseded, and hashes every distilled fact into `~/.personal-brain/capture-seen.jsonl` so a fact re-stated in a later session is never ingested twice. Every run is logged; `brain doctor` reports the last one.

**Ledger-aware answers** — `brain ask` / `brain_ask` put matching decisions (with their revisit triggers) and loops in front of the graph nodes, and cite their ids as sources, so "what did I decide about X" is answered from the ledger rather than from whatever the graph happened to extract. `brain index` embeds every loop and decision too, so a question worded nothing like the ledger ("what am I owed?") still reaches the right line. A who-question leads with people — person files and person nodes seed first — and every graph line in the prompt says where the node is filed (`(organization, under Companies & Organizations)`; a category lists its members), so structure questions are answered from the spine, not guessed. File excerpts cut on sentence boundaries with a visible marker, keep a long matching line by windowing it, and for log files lead with the newest matching entries.

**The vault is the brain; the graph is its index (D-014)** — `brain index` walks the vault directory (`profile/`, `courses/`, `applications/`, `orgs/`, `projects/`, `people/`, `apps/`, `topics/`, `docs/`, `areas/`, `log/`, with `ALVIN.md` as the hub), records every file's path, kind, title, aliases, search tokens, content hash and (with a key) embedding, links files to each other by the paths they mention and to graph nodes whose name matches a title or alias, and stamps each such node with its `path`. Incremental by content hash; generated views and the CLI-owned ledgers are skipped. `brain ask` / `brain_ask` then route a question to files first — keyword, graph-hop and embedding signals — read the top files from disk, and answer with the paths cited; `brain query` / `brain_search` list the matching files under the nodes. `brain doctor` warns when the index is stale. The index never holds a fact of its own: change a file, run `brain index`, and every answer follows.

**Agent memory (MCP)** — `brain mcp` runs an MCP server over stdio (pure stdlib, no SDK). Any MCP client gets five tools — `brain_remember`, `brain_search`, `brain_ask`, `brain_context`, `brain_digest` — so an agent can load who you are at session start, recall specifics mid-task, and deposit new knowledge back into the graph as you work. Memories an agent reads are reinforced; ones nothing touches fade. Where built-in assistant memory is a flat list of disconnected facts, this is a typed graph with forgetting.

**Ambient capture (Claude Code hook)** — `integrations/claude_code_capture.py` wires into Claude Code as a SessionEnd hook: when a session ends, it distills durable facts from what *you* typed (never tool output, never assistant text) and ingests them — so memory accumulates without ever saying "remember this". Capture is summary-level and auditable (`~/.personal-brain/capture.log`), trivial sessions are skipped, and secrets are excluded by construction and by prompt. Each session carries an ingest watermark (`capture-state.json`), so a long-lived session that ends repeatedly only ever mines the turns typed since its last capture — never the whole transcript again.

**The vault stays committed** — every write the CLI or extractor makes to the vault commits itself: loop/decision edits commit as before, and an ingest commits its own generated output (`DIGEST.md`, `graph/`, `LOOPS-INBOX.md`) in a scoped commit that never sweeps up curated files you're mid-editing. `brain doctor`'s vault-git check can therefore stay strict: dirt means a human left something uncommitted, not that the extractor ran.

---

## Half-lives by node type

| Type | Half-life | Rationale |
|------|-----------|-----------|
| task | 5 days | Actionable items fade once stale |
| event | 7 days | Ephemeral |
| fact | 21 days | Specific facts fade fast |
| artifact | 30 days | Documents/files are temporary references |
| concept | 60 days | Ideas need reinforcement |
| insight | 90 days | Synthesis products |
| skill | 180 days | Skills are durable |
| project | 365 days | Projects are long-lived |
| person / organization | never while importance ≥ 0.4; below that, 60 days like a concept | People and institutions that matter don't expire; a one-off sponsor or a passing acquaintance does |
| category | never | Structure doesn't expire — the spine is never pruned |

These are *base* half-lives. A node's `importance` (0–1) stretches its effective
half-life by up to 5× and sets a weight floor, so important nodes persist far
longer (and never auto-archive) while trivia decays on the base schedule.

---

## Install

```bash
pip install -e .                    # installs dependencies from pyproject.toml
export GEMINI_API_KEY=...           # or put it in ~/.personal-brain/.env
```

---

## Testing

The core logic layer (decay, search, dedup/merge, graph traversal, context
seeding, `.env` loading) is covered by a stdlib-only test suite — no API key or
network required (the Gemini boundary is mocked):

```bash
python -m unittest discover -s tests
```

CI runs these on every push across Python 3.10–3.12.

For an end-to-end check against live Gemini (ingests varied content into an
isolated temp brain, prints the hierarchy/search/context, and asserts structural
invariants), run:

```bash
python scripts/scenario.py
```

---

## Usage

```bash
# Ingest
brain add "I've been reading about transformers and attention mechanisms"
brain add --file notes.txt
brain add --url https://arxiv.org/abs/1706.03762

# Explore
brain serve                      # live web app (add/search/context/synthesize) with a 2D/3D toggle
brain show                       # one-shot interactive 2D graph
brain show --3d                  # interactive 3D (WebGL) graph
brain show --min-weight 0.3      # hide faded nodes
brain show --type concept        # filter by type

# Search
brain query "attention"               # keyword / stem search
brain reindex                         # embed the nodes still lacking embeddings (--all to redo every node)
brain query "machine learning" --semantic   # rank by meaning, not keywords
brain index [--no-embed] [--status]   # index the vault directory: file paths, aliases, links, node paths (D-014)
brain ask "where do I want to study?"        # Q&A — files first (paths cited), then ledgers and graph nodes

# Context document (pipe into Claude, ChatGPT, etc.)
brain context "machine learning"
brain context > context.md

# Agent memory — register the MCP server once, then Claude remembers you
claude mcp add brain -- brain mcp     # Claude Code
brain mcp                             # or point any MCP client at this (stdio)

# Ambient capture — add as a Claude Code SessionEnd hook (~/.claude/settings.json)
# and durable facts from each session flow into the graph automatically:
#   {"hooks": {"SessionEnd": [{"hooks": [{"type": "command", "async": true,
#     "command": "python /path/to/integrations/claude_code_capture.py"}]}]}}

# Hierarchy
brain tree                       # print the person-rooted hierarchy
brain reorganize                 # retrofit existing flat nodes into the hierarchy

# Markdown vault — the brain's file layer
brain vault                      # render the graph into ~/.personal-brain/vault/
                                 #   (DIGEST.md + graph/<life-area>.md; also auto-runs
                                 #   after every add — curated notes alongside are never touched)
brain vault --set-dir ~/notes    # persist a different vault location

# Action layer — the vault's task system (no LLM, deterministic, session-safe)
brain loop add "Lock the fourth seat" --due 2026-09-09 --owner alvin --area harvard --prio 1 \
               --next "click Enroll Selected"     # opens L-nnn, re-renders NOW.md, commits the vault
brain loop done L-003 --note "enrolled 9.522"    # close; ids are never reused
brain loop edit L-004 --due 2026-09-12 --owner waiting:protopapas --next "nudge if silent"
brain loop list [--all] [--area jobs]            # open loops by prio, then due
brain loop lint                                  # grammar, duplicate ids, NOW.md drift; exit 1 on errors
brain loop inbox [--drop N | --clear]            # action items the extractor found; triage with `loop add --from-inbox N`
                                                 #   (drops/triages are remembered in .loops-inbox-seen.jsonl —
                                                 #    a re-extraction of the same item is never re-added)
brain decide "Fourth-seat plan of record" --what "..." --why "..." --rejected "..." --revisit "..."
brain decisions [--last 5] [--lint]              # DECISIONS.md is append-only (git pre-commit enforced)
brain today [--brief] [--date YYYY-MM-DD]        # action card: health line, countdowns, waits, Claude-owned loops, top 3
brain doctor [--brief] [--install-hooks] [--repair]  # binary, graph + tree integrity, key+TLS, capture, vault, ledgers, NOW.md, hooks, MCP, tasks
brain repair                                     # one parent per node, categories under you, no orphans/cycles (never deletes)
brain now render | show | lint                   # NOW.md is GENERATED: IDENTITY.md + loops by area + areas/*.md `## Now` + people + apps
brain area touch harvard                         # stamp `updated:` on areas/harvard.md after editing its `## Now` block; re-renders NOW.md

# Maintenance
brain digest                     # at-a-glance: top of mind, open loops (from LOOPS.md), fading, by area
brain status                     # stats + decay report + what's fading soon
brain synthesize                 # find and surface new connections
brain reindex [--all]            # embed nodes lacking embeddings (parallel); --all re-embeds everything
brain decay                      # run decay manually
brain prune                      # delete archived nodes
brain clear                      # erase the entire brain (asks to confirm; -y to skip)

# Manual ops
brain reinforce <node-id>        # boost weight to 1.0
brain forget <node-id>           # immediately archive
brain merge <id1> <id2>          # merge id2 into id1 (keeps one parent, takes the higher importance)
brain move <node> <parent>       # re-home a node (ids or exact names); no cycles, categories only under you
brain rename <node> <new-name>   # rename in place (refuses a name another node carries — merge instead)
brain retype <node> <type>       # change the type; the decay half-life follows (never into/out of category)
brain describe <node> "<text>"   # replace the content (a correction; add appends on re-mention)
brain subgroup [--threshold N]   # split oversized categories into LLM-clustered sub-categories

# Backup / portability
brain export backup.json         # dump the whole graph to JSON, embeddings included (--lean to leave them out)
brain import backup.json         # merge a JSON export back in (skips duplicates)
```

---

## Web app

```bash
brain serve          # opens http://127.0.0.1:8000 in your browser
```

A live, single-page app over the same brain — keep it open and watch the graph
evolve as you think out loud:

- **Talk to your brain** — type a thought in the box; it's ingested and the graph
  reloads with the new nodes (no second terminal).
- **Everything inline** — search (keyword or semantic), **ask a question** (Q&A
  with sources), build a context document, run synthesize / reorganize, and view
  status & the hierarchy, all from a control bar with results in a side panel.
- **2D or 3D** — toggle between the Pyvis force-graph and a WebGL 3D force-graph;
  categories are hubs, the `part_of` backbone shows directional arrows.
- **Explore** — a min-weight slider fades out trivia; click any node to see its
  content, importance, weight, and connections.
- **Live & incremental** — when the brain changes, the 2D graph updates *in place*
  (new nodes animate in, existing node positions are preserved) instead of a full
  reload, so it stays smooth and keeps its layout even on large graphs.

It's stdlib-only (`http.server`) and binds to localhost — your data never leaves
your machine.

---

## Architecture

```
Input (text / file / URL)
  ↓
Gemini Flash extraction
  → typed nodes + semantic edges + confidence scores
  ↓
Deduplication
  → merge into existing node (touch) OR create new node
  ↓
SQLite (WAL mode)
  → nodes, edges, ingestion_log tables
  ↓
Decay (on every CLI call)
  → weight(t) = last_weight × exp(-t / half_life)
  → weight < 0.1 → archived → deleted after 7 days
  ↓
Query / Output
  → brain query   → keyword search + weight sort
  → brain context → BFS traversal → Gemini synthesis
  → brain show    → Pyvis HTML graph
  → brain mcp     → MCP server (stdio) → remember/search/ask/context/digest for agents
```

---

## Scale

Honest ceilings, measured against the design rather than benchmarks:

| Nodes | What happens |
|---|---|
| **up to ~10k** | Everything as is. SQLite + WAL, JSON-text embeddings scanned in Python, vis-network for the graph, one Gemini call per stage. |
| **~10k–100k** | Semantic search and dedup scan every embedding (`json.loads` + cosine per node) and the web view ships every node: store embeddings as float32 BLOBs behind an HNSW index, add an FTS5 table for keyword search, render the viewport around the focused node instead of the whole graph. |
| **beyond** | A different product: Postgres + pgvector, batched embedding calls, an async ingest queue, `reorganize` per subtree instead of one catalogue prompt. |

A personal brain of daily notes reaches the first ceiling after years, so the code optimises for correctness and
inspectability (a tree you can read with `brain tree`, a doctor that names the cure) over throughput.

## Files

| File | Description |
|------|-------------|
| `brain/db.py` | SQLite schema, node/edge CRUD, hierarchy, relation vocab |
| `brain/decay.py` | Importance-weighted half-life decay, at-risk reporting |
| `brain/extract.py` | Gemini extraction, dedup/merge, hierarchy spine, embeddings |
| `brain/graph.py` | BFS (hub-aware), keyword + semantic search, context synthesis |
| `brain/llm.py` | Gemini REST client (generate + embed) over stdlib urllib, with retries |
| `brain/visualize.py` | Pyvis interactive graph |
| `brain/server.py` | Live auto-reloading web view (`brain serve`) |
| `brain/mcp.py` | MCP server over stdio (stdlib JSON-RPC, no SDK) — agent memory tools |
| `integrations/claude_code_capture.py` | Claude Code SessionEnd hook — ambient memory capture |
| `brain/vault.py` | Markdown vault renderer — deterministic file views of the graph |
| `brain/portability.py` | JSON export / import |
| `cli.py` | Click CLI entry point |

---

## Why this doesn't exist yet

Every tool in this space has a meaningful gap. Obsidian is fully manual. Rewind captures but doesn't structure. Mem.ai is proprietary and flat. ChatGPT memory is a list of disconnected facts with no relationships, no decay, no synthesis. EpisTwin (arxiv 2603.06290) describes the right architecture but doesn't ship code.

The combination here — automatic extraction, typed graph, forgetting, synthesis, visual exploration, local-first — doesn't exist as a single working tool.
