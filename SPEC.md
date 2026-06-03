# Personal Brain — Spec

**Status:** In progress  
**Started:** June 2026

---

## Problem

Every tool in this space has a meaningful flaw:

- **Obsidian / Roam / Logseq** — beautiful graphs but fully manual. You have to write everything yourself. Most people's graphs die after a month.
- **Rewind / Screen AI** — captures everything chronologically but doesn't structure, synthesize, or connect anything. A log, not a brain.
- **Mem.ai / Reflect** — chat-based, proprietary, no graph structure, no forgetting.
- **ChatGPT / Claude memory** — flat list of disconnected facts. No relationships. No decay. No synthesis.
- **raold/second-brain** — local RAG + KG, but no forgetting, no synthesis job, no visual exploration.
- **EpisTwin (arxiv 2603.06290)** — describes exactly the right architecture but doesn't ship code.

What's missing: a tool that (a) ingests automatically with minimal friction, (b) structures knowledge as a typed, relational graph, (c) synthesises connections, (d) forgets intelligently, and (e) lets you explore the graph visually and inject it as context into AI conversations.

---

## What this is

A local-first personal knowledge graph with:

1. **LLM-powered ingestion** — paste text, a URL, or a file. Claude extracts entities, facts, and relationships automatically.
2. **Typed graph** — nodes have types (concept, skill, project, person, insight, fact) and edges have semantic labels (builds_on, requires, contradicts, studied_by, etc.)
3. **Forgetting** — every node has a weight that decays via an Ebbinghaus-inspired curve. The half-life depends on node type. Accessing a node resets its weight.
4. **Synthesis** — periodic job that finds clusters, surfaces novel connections, detects contradictions, and generates summary insights.
5. **Visual exploration** — interactive HTML graph. Nodes sized by weight, coloured by type. Click to read content. Filter by recency, type, confidence.
6. **Context injection** — `brain context "topic"` traverses the graph from relevant nodes and synthesises a structured context document, suitable for pasting into any AI conversation. Also packaged as a Claude Code skill.

Everything runs locally. Data lives in `~/.personal-brain/`. No cloud, no accounts.

---

## Architecture

```
Ingestion
  └─ raw text / file / URL
       ↓
  Extraction (Claude Haiku)
  └─ entities, types, relationships, confidence scores
       ↓
  Deduplication (name similarity + LLM merge check)
  └─ merge into existing node OR create new node
       ↓
  Storage (SQLite)
  └─ nodes table, edges table, ingestion_log

Background
  └─ Decay job (runs on every CLI call)
       └─ weight(t) = exp(−t / half_life)
       └─ weight < 0.1 → archived
       └─ archived > 7 days → pruned

  └─ Synthesis job (explicit: `brain synthesize`)
       └─ find nodes with no connections → try to connect
       └─ find clusters → generate summary insight node
       └─ find contradictions → flag with `contradicts` edge

Query / Output
  └─ `brain query` → matching nodes + neighbours
  └─ `brain context` → BFS traversal → LLM synthesis → structured doc
  └─ `brain show` → Pyvis HTML graph → open in browser
```

---

## Data model

### Nodes

| Field | Type | Description |
|-------|------|-------------|
| id | TEXT (UUID) | Primary key |
| name | TEXT | Short label |
| type | TEXT | concept / skill / project / person / fact / insight / event |
| content | TEXT | Full description |
| source | TEXT | Where this came from |
| created_at | REAL | Unix timestamp |
| last_accessed | REAL | Unix timestamp |
| access_count | INTEGER | How many times accessed/confirmed |
| weight | REAL | Current forgetting-curve value (0–1) |
| confidence | REAL | Extraction confidence (0–1) |
| half_life_days | REAL | Type-specific decay rate |
| archived | INTEGER | 0 or 1 |

### Edges

| Field | Type | Description |
|-------|------|-------------|
| id | TEXT (UUID) | Primary key |
| source_id | TEXT | FK → nodes |
| target_id | TEXT | FK → nodes |
| relation | TEXT | Semantic label |
| weight | REAL | Edge confidence / recency |
| created_at | REAL | Unix timestamp |

### Ingestion log

Keeps raw text + list of nodes/edges created, so you can audit what came from where.

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

## Forgetting algorithm

```
weight(t) = last_weight * exp(-days_since_access / half_life_days)
```

On access: `weight = 1.0`, `last_accessed = now()`  
On `weight < 0.10`: `archived = 1`  
On `archived = 1` for 7+ days: node deleted

The decay runs automatically on every CLI invocation — no cron job needed.

---

## Synthesis

Not auto-running (too expensive). Triggered by `brain synthesize`.

Steps:
1. Load all nodes with weight > 0.3 into the graph
2. Find isolated nodes (no edges) → prompt Claude: "how does X relate to existing knowledge?"
3. Find dense clusters (>5 nodes, high internal edge weight) → generate a summary insight node
4. Find pairs with no path between them but related content → try to bridge them
5. Surface the top 5 "new connections" found

---

## Context generation

`brain context "ML internships"` or `brain context` (full profile):

1. Find relevant nodes (keyword match + high weight)
2. BFS from those nodes, depth 3, min weight 0.2
3. Group collected nodes by type
4. LLM: synthesise into a structured document (not a list of facts)

The output document has sections: Background, Active Skills, Current Focus, Projects, Open Questions.

---

## CLI

```bash
brain add "text"              # ingest text
brain add --file path         # ingest file
brain add --url https://...   # ingest URL (scrapes text)

brain show                    # open interactive graph in browser
brain show --min-weight 0.3   # filter faded nodes
brain show --type concept     # filter by type

brain query "topic"           # find relevant nodes
brain context "topic"         # generate AI context document
brain context > context.md    # pipe to file for pasting into Claude

brain status                  # graph stats, health
brain synthesize              # run synthesis job
brain decay                   # run decay manually
brain prune                   # delete archived nodes

brain merge <id1> <id2>       # manually merge two nodes
brain forget <id>             # immediately archive a node
brain reinforce <id>          # manually boost a node's weight
```

---

## Phases

### Phase 1 — Core (built now)
- SQLite schema + CRUD
- Claude-powered extraction
- Ebbinghaus decay
- CLI (add, show, query, context, status, decay, prune)
- Pyvis visualization

### Phase 2 — Synthesis + Search
- Embedding-based deduplication (sentence-transformers)
- Synthesis job
- Semantic search (vs. keyword search)
- URL scraping ingestion

### Phase 3 — Ambient ingestion
- Clipboard monitor daemon (runs in background, watches clipboard)
- Browser extension (logs visited pages + time spent)
- Import from Obsidian vault
- MCP server (expose graph to any AI via MCP protocol)

---

## What makes this different from EpisTwin

EpisTwin is closer to a research system — multimodal, complex agentic coordinator, benchmark-driven. This is a daily-use personal tool. The design priorities are:
- Minimal friction to ingest
- Trustworthy forgetting (you can inspect what's fading)
- Honest confidence scores (not everything is equally certain)
- Local-first (data never leaves your machine)
- Hackable (plain SQLite, standard Python)
