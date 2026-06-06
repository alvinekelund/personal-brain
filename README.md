# Personal Brain

[![CI](https://github.com/alvinekelund/personal-brain/actions/workflows/ci.yml/badge.svg)](https://github.com/alvinekelund/personal-brain/actions/workflows/ci.yml)

A local-first personal knowledge graph. Paste text, a URL, or a file — Gemini extracts entities and relationships automatically. Every node decays via an Ebbinghaus forgetting curve; accessing it resets the clock. The graph is explorable as an interactive Pyvis visualization, and any slice of it can be synthesized into a structured context document for pasting into an AI conversation.

No cloud. No accounts. Data lives in `~/.personal-brain/brain.db`.

---

## What it does

**Ingestion** — point it at anything: a paragraph, an article, a book chapter, a meeting transcript. Gemini Flash extracts typed nodes (concepts, skills, projects, people, facts, insights, events) and semantic edges (builds_on, requires, contradicts, part_of, etc.).

**Forgetting** — each node has a weight that decays according to its type. An event has a half-life of 7 days; a skill, 180 days; a person never expires. Weight drops below 0.1 → archived. Archived for 7 days → deleted. Accessing a node resets weight to 1.0. The decay runs automatically on every CLI call — no cron job needed.

**Visualization** — `brain show` opens an interactive force-directed graph. Nodes are sized by weight, coloured by type. Hover for content, filter by weight threshold or type.

**Context injection** — `brain context "ML internships"` does a BFS traversal from relevant nodes, then calls Gemini to synthesise a structured document: Background, Active Skills, Current Focus, Projects, Open Questions. Pipe it straight into any AI conversation.

**Synthesis** — `brain synthesize` finds isolated nodes and tries to connect them to the existing graph, surfacing relationships Gemini notices across your knowledge.

---

## Half-lives by node type

| Type | Half-life | Rationale |
|------|-----------|-----------|
| event | 7 days | Ephemeral |
| fact | 21 days | Specific facts fade fast |
| concept | 60 days | Ideas need reinforcement |
| insight | 90 days | Synthesis products |
| skill | 180 days | Skills are durable |
| project | 365 days | Projects are long-lived |
| person | never | People don't expire |

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
brain query "attention"

# Context document (pipe into Claude, ChatGPT, etc.)
brain context "machine learning"
brain context > context.md

# Maintenance
brain status                     # stats + decay report
brain synthesize                 # find and surface new connections
brain decay                      # run decay manually
brain prune                      # delete archived nodes

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
| `brain/db.py` | SQLite schema, all node/edge CRUD |
| `brain/decay.py` | Ebbinghaus curve, decay runner |
| `brain/extract.py` | Gemini Flash extraction, DB merge |
| `brain/graph.py` | BFS traversal, context synthesis |
| `brain/visualize.py` | Pyvis interactive graph |
| `cli.py` | Click CLI entry point |

---

## Why this doesn't exist yet

Every tool in this space has a meaningful gap. Obsidian is fully manual. Rewind captures but doesn't structure. Mem.ai is proprietary and flat. ChatGPT memory is a list of disconnected facts with no relationships, no decay, no synthesis. EpisTwin (arxiv 2603.06290) describes the right architecture but doesn't ship code.

The combination here — automatic extraction, typed graph, forgetting, synthesis, visual exploration, local-first — doesn't exist as a single working tool.
