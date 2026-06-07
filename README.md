# Personal Brain

[![CI](https://github.com/alvinekelund/personal-brain/actions/workflows/ci.yml/badge.svg)](https://github.com/alvinekelund/personal-brain/actions/workflows/ci.yml)

A local-first personal knowledge graph. Paste text, a URL, or a file — Gemini extracts entities and relationships automatically. Every node decays via an Ebbinghaus forgetting curve; accessing it resets the clock. The graph is explorable as an interactive Pyvis visualization, and any slice of it can be synthesized into a structured context document for pasting into an AI conversation.

No cloud. No accounts. Data lives in `~/.personal-brain/brain.db`.

---

## What it does

**Ingestion** — point it at anything: a paragraph, an article, a book chapter, a meeting transcript. Gemini Flash extracts typed nodes (concepts, skills, projects, people, facts, insights, events) and semantic edges (builds_on, requires, contradicts, part_of, etc.). Long input is chunked so nothing is dropped.

**Person-rooted hierarchy** — it's a graph, but with a backbone: every node is placed under a `category` (Career, Hobbies, Relationships, …) that hangs off *you*, forming a `You → Category → Topic → Detail` tree via `part_of` — while cross-links between branches keep it a graph. `brain tree` prints the hierarchy; `brain reorganize` retrofits an existing flat brain into it.

**Importance-weighted forgetting** — each node has an `importance` (0–1, scored at ingest) and a type half-life. Important nodes decay much slower and never auto-archive; one-off details still fade. Accessing a node resets it to 1.0 and propagates a freshness boost up its branch. Decay runs automatically on every CLI call.

**Search** — `brain query "x"` (stem-aware keyword) or `brain query "x" --semantic` (embedding cosine — finds by *meaning*, e.g. "machine learning" surfaces your neural-net nodes). `brain reindex` (re)computes embeddings; new nodes are embedded automatically on `add`.

**Visualization** — `brain show` opens an interactive force-directed graph: categories are large hubs, the `part_of` backbone is solid arrows, cross-links are dashed, node size reflects importance. Hover for content; filter by weight or type.

**Context injection** — `brain context "ML internships"` seeds from keyword → semantic → whole-brain (in that order), BFS-traverses (hub-aware), then calls Gemini to synthesise a structured document: Background, Active Skills, Current Focus, Projects, Open Questions. Pipe it straight into any AI conversation.

**Synthesis** — `brain synthesize` finds isolated nodes and connects them to the graph, surfacing relationships Gemini notices across your knowledge.

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
| person / organization / category | never | Identity & structure don't expire |

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
brain show                       # open interactive graph
brain show --min-weight 0.3      # hide faded nodes
brain show --type concept        # filter by type

# Search
brain query "attention"               # keyword / stem search
brain reindex                         # embed all nodes (needed once for semantic)
brain query "machine learning" --semantic   # rank by meaning, not keywords

# Context document (pipe into Claude, ChatGPT, etc.)
brain context "machine learning"
brain context > context.md

# Hierarchy
brain tree                       # print the person-rooted hierarchy
brain reorganize                 # retrofit existing flat nodes into the hierarchy

# Maintenance
brain status                     # stats + decay report + what's fading soon
brain synthesize                 # find and surface new connections
brain reindex                    # (re)compute embeddings for semantic search
brain decay                      # run decay manually
brain prune                      # delete archived nodes
brain clear                      # erase the entire brain (asks to confirm; -y to skip)

# Manual ops
brain reinforce <node-id>        # boost weight to 1.0
brain forget <node-id>           # immediately archive
brain merge <id1> <id2>          # merge id2 into id1

# Backup / portability
brain export backup.json         # dump the whole graph to JSON
brain import backup.json         # merge a JSON export back in (skips duplicates)
```

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
```

---

## Files

| File | Description |
|------|-------------|
| `brain/db.py` | SQLite schema, node/edge CRUD, hierarchy, relation vocab |
| `brain/decay.py` | Importance-weighted half-life decay, at-risk reporting |
| `brain/extract.py` | Gemini extraction, dedup/merge, hierarchy spine, embeddings |
| `brain/graph.py` | BFS (hub-aware), keyword + semantic search, context synthesis |
| `brain/llm.py` | Gemini REST client (generate + embed) over stdlib urllib, with retries |
| `brain/visualize.py` | Pyvis interactive graph |
| `brain/portability.py` | JSON export / import |
| `cli.py` | Click CLI entry point |

---

## Why this doesn't exist yet

Every tool in this space has a meaningful gap. Obsidian is fully manual. Rewind captures but doesn't structure. Mem.ai is proprietary and flat. ChatGPT memory is a list of disconnected facts with no relationships, no decay, no synthesis. EpisTwin (arxiv 2603.06290) describes the right architecture but doesn't ship code.

The combination here — automatic extraction, typed graph, forgetting, synthesis, visual exploration, local-first — doesn't exist as a single working tool.
