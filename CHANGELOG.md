# Changelog

Newest first. One section per day of work; each bullet names what changed and why, with the commits that carry it (`git log 5af0c6c..7642567` lists all 110 of Sep 6).

## 2026-09-06 — 110 commits, 380 tests, CI green

**Ingest and extraction**
- Gemini calls carry a wall-clock budget; `brain add` shows its stages and every retry wait; long inputs are extracted chunk-parallel (`5af0c6c`, `c36812c`, `a8ce5d7`, `e440256`).
- The extractor sees the *relevant* existing nodes (keyword + semantic + importance), re-mentioned nodes absorb new content, attributes of known entities become facts under them, vague names and bare time periods are refused, a course named by its code is a concept (`a6ece9b`, `2aeb567`, `51e960c`, `7a8990d`, `8e2ab78`).
- One thing is one node: a list of third parties is one fact, the features of one thing are its content, a site/CV edit wish is not a node; the entity-linker judges by descriptions, not names (`46c1441`, `bf7e3d9`, `bcba1f6`; decision D-020 in the vault).
- Nothing ingests before `brain setup` names the owner; `add` and the MCP tool report where each node was filed and stamp a default source (`c0a07f9`, `dedcbb1`, `0a5fa4d`).
- A forgotten node comes back when new knowledge names it (`a620810`); loop lint flags next actions that are narrative (`a439f4f`); passive relations are oriented agent-at-target and a `relates_to` never doubles a `part_of` (`e39dc3b`, `ef0be37`).

**Structure and curation**
- Deterministic curation commands: `merge` (by name too), `move`, `rename`, `retype`, `describe`, `forget`, `reinforce`, `importance`, `unlink`, `subgroup`; each re-renders and commits the vault views (`41e89b3` … `ec534d2`, `645f530`).
- Sub-categories are legitimate structure; ingest and `reorganize` see them as `Area > Sub-category` and file under the most specific one (`8f6393d`, `9f4ca51`, `5eea70b`).
- `merge_nodes` keeps the tree and loses nothing; deletes cascade; merges no longer leave cross-links to categories (`8e488f4`, `8b240b5`).

**Doctor**
- New checks, each naming its cure: dangling edges, duplicate pairs by name and by meaning (stored embeddings, 0.89 cosine), oversized categories, flat lists under a non-category, thin top-level areas, facts used as parents, category cross-links, backwards edges, redundant links; stale claims (`brain stale`: plan-tense after 30 days, dated future claims the day after, content citing a closed loop); coverage (important nodes need a vault file); backups; the recorded phone brief; a 7-day capture tally (`ba21418`, `f5a50c9`, `ee76b1a`, `7126fb6`, `620fe02`, `8d4887f`, `79b4cda`, `83ee93f`, `6b94b80`, `e1948b0`, `692208d`).
- `graph-tree` warns whenever the report is not clean; the loops warning names ids and reasons, not titles (`78be7d3`, `8985ef6`).

**Retrieval**
- Who-questions reach people; ledgers (loops, decisions) are embedded and retrieved; context briefings read the topic's files, ledgers and NOW.md; excerpts cut on boundaries and lead with the newest log entries (`5a6111d`, `2e7c767`, `94e7183`, `9c5ccef`, `6d8c90a`, `1abe436`).
- Node→file linking runs three passes (exact incl. year/qualifier-stripped names, multi-word containment, inherited from the parent entity); categories never link to files (`fb3052e`, `6b94b80`, `8d066c7`).
- Keyword search ranks the thing itself above things that mention it and knowledge above structure; semantic search skips categories; `brain ask` reports what it cited on every surface (`9f5e25f`, `4779ae0`, `3bb7ddb`, `f59e940`).

**Storage, decay, operations**
- Embeddings are packed float32 (brain.db 28 MB → 9.6 MB), migrated on connect; exports still carry plain lists (`04b250f`).
- `brain backup` takes consistent snapshots with rotation; the daily card takes one when the newest is stale and says so (`e1948b0`, `5964d1c`, `1dfb373`).
- Decay: people and organisations are immortal only while they matter; an event does not fade before its date; top of mind = importance × recency (`3566621`, `1f54928`, `e2be6c2`).
- Vault commits wait out another process's `index.lock` (`7642567`); `prune` and decay refresh the views when they forget a node (`2b0c1bd`, `722bb81`); the incremental index refreshes derived kinds and index pages rank below the file that answers (`b6ee1f2`, `56b0d65`); export/import restore importance, the decay clock, embeddings and the ingestion log (`e68cc48`, `7241bb4`).

## Before 2026-09-06

83 commits from the first version to the vault-as-brain design (D-014): `git log --before=2026-09-06`.
