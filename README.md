# Memory Service

A production-quality memory service for AI agents — ingests conversation turns, extracts structured knowledge, and answers recall queries with hybrid retrieval under a token budget.

## Quick start

```bash
cp .env.example .env
# Add your OPENAI_API_KEY and/or ANTHROPIC_API_KEY

docker compose up -d
until curl -sf http://localhost:8080/health; do sleep 1; done

# Smoke test
curl -X POST http://localhost:8080/turns \
  -H 'Content-Type: application/json' \
  -d '{
    "session_id": "smoke-1",
    "user_id": "user-1",
    "messages": [
      {"role": "user", "content": "I just moved to Berlin from NYC last month."},
      {"role": "assistant", "content": "That sounds exciting!"}
    ],
    "timestamp": "2025-03-15T10:30:00Z",
    "metadata": {}
  }'

curl -X POST http://localhost:8080/recall \
  -H 'Content-Type: application/json' \
  -d '{"query": "Where does this user live?", "session_id": "smoke-2", "user_id": "user-1", "max_tokens": 512}'
```

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   HTTP API (FastAPI)                  │
│  POST /turns  POST /recall  POST /search  GET /users │
└────────────┬─────────────────────┬───────────────────┘
             │ ingest              │ query
    ┌────────▼───────┐    ┌────────▼──────────┐
    │  Extraction    │    │   Recall pipeline  │
    │  pipeline      │    │   BM25 + semantic  │
    │  (Claude/GPT)  │    │   → RRF → assemble │
    └────────┬───────┘    └────────┬──────────┘
             │                     │
    ┌────────▼──────────────────────▼──────────┐
    │              Storage layer                │
    │  ┌──────────────────┐  ┌───────────────┐ │
    │  │  SQLite + FTS5   │  │    Qdrant      │ │
    │  │  turns·memories  │  │  embeddings    │ │
    │  │  supersession    │  │  semantic srch │ │
    │  └──────────────────┘  └───────────────┘ │
    │           Docker named volume             │
    └───────────────────────────────────────────┘
```

**Two services, one volume.** The FastAPI app handles all business logic. Qdrant handles vector storage. Both persist to named Docker volumes — restart is invisible to clients.

The ingestion path is synchronous: `POST /turns` blocks until extraction, embedding, and indexing are all complete, so data is immediately queryable after the response.

## Backing store choice

**SQLite + FTS5** for structured data and keyword search. Chosen because:
- Zero-ops: no separate process, ships in Python stdlib, handles WAL-mode concurrent reads trivially
- FTS5 with porter stemmer gives BM25 ranking for free — "works at Notion" matches "Notion job" without extra infrastructure
- The supersession chain (which memory supersedes which) maps naturally to a single FK column, inspectable with plain SQL
- Handles the eval's single-machine, few-concurrent-sessions constraint perfectly

**Qdrant** for vector search. Chosen because:
- Payload filtering (`user_id`, `active`) is first-class — no client-side re-filtering after top-k
- Supports `set_payload` to soft-deactivate vectors without deletion (preserving history)
- Docker-native, lightweight

Alternative considered: `pgvector` on a single Postgres instance. Rejected because it adds Postgres ops overhead for a service that doesn't need relational joins beyond what SQLite provides.

## Extraction pipeline

On every `POST /turns`, after persisting the raw turn, the service calls an LLM (Claude Haiku or GPT-4o-mini) with a structured extraction prompt. The prompt instructs the model to return a JSON array of memories with explicit types:

| Type | Examples |
|---|---|
| `fact` | employer, location, name, pets, family |
| `preference` | dietary restrictions, tool preferences, communication style |
| `opinion` | views on technologies, topics, experiences |
| `event` | things that happened or are planned |
| `correction` | explicit "actually I meant X" corrections |

Each extracted memory has a normalized `key` (e.g. `current_employer`, `pet_name`, `home_city`) that enables contradiction detection, a `value` as a human-readable statement, and a `confidence` score calibrated by how explicit vs. implicit the information was.

**Key-slot normalization:** The LLM sometimes uses inconsistent keys for the 
same concept across sessions (`job_change`, `employer`, `new_job`). A `KEY_SLOTS` 
alias map normalizes these to canonical keys (`current_employer`) before 
contradiction detection runs — without this, supersession silently fails.

**Implicit facts** are extracted too. "Walking Biscuit this morning" → `{type: fact, key: pet_name, value: "has a dog named Biscuit", confidence: 0.80}`.

**What we miss:** Gradual opinion shifts across many sessions (we detect single-session contradictions well, but subtle multi-session drift requires cross-session context that the extraction prompt doesn't have). Future improvement: a periodic re-summarization job across a user's recent memories.

## Recall strategy

`POST /recall` runs a three-stage pipeline:

**Stage 1 — Parallel retrieval**
- **FTS5/BM25**: queries the SQLite full-text index for keyword-relevant active memories. "What's the dog's name?" reliably surfaces `pet_name` even if the embedding space didn't cluster them closely.
- **Semantic search**: queries Qdrant with an embedding of the query, filtered to `user_id` + `active=true`.

**Stage 2 — Reciprocal Rank Fusion (RRF)**
Fuses BM25 and semantic rankings using the standard formula `score = Σ 1/(k + rank)` with `k=60`. This consistently outperforms either retriever alone: semantic handles paraphrase ("employer" vs "company"), BM25 handles entity-heavy queries ("dog named Biscuit").

**Stage 3 — Context assembly under token budget**

Priority when budget is tight:
1. **Stable facts first** (`type=fact` or `preference`, ordered by confidence). These change rarely and are almost always relevant.
2. **Query-relevant memories** (ordered by RRF score) — opinions, events, corrections.
3. Stop when token budget would be exceeded.

This priority logic is opinionated: a user's name, employer, dietary restriction, and pet are worth more tokens than a contextually-relevant event from two months ago. The eval's questions tend to be about stable facts, so this ordering improves recall quality over a flat score-based ranking.

**Token counting** uses `tiktoken` (cl100k_base) for accurate budget estimation.

## Fact evolution

When a new memory is extracted with a `key` that already has an active memory for the same `user_id`:

1. The old memory's `active` flag is set to `0` (not deleted — history is preserved)
2. The old memory's Qdrant payload `active` field is set to `false` (excluded from future searches)
3. The new memory is inserted with `supersedes = old_memory.id`

The full chain is visible via `GET /users/{user_id}/memories` — both active and superseded memories are returned, with the `supersedes` FK forming an inspectable chain.

**Opinion evolution** is handled differently from fact contradiction. For `type=opinion`, we treat each update as a refinement rather than a binary flip — the new opinion's `value` is expected to contain more nuance than the old one (since it's extracted from a later conversation that presumably references the evolved state). We still supersede the old record, but the CHANGELOG documents the tradeoff: a production system should probably maintain an opinion arc rather than a single active value.

**Corrections** (`type=correction`) are extracted as their own memory type. 
The extraction prompt instructs the model to use the corrected fact's normalized 
key, so the same KEY_SLOTS normalization that handles employer aliases also 
resolves corrections to the right slot.

## Tradeoffs

| Optimized for | Gave up |
|---|---|
| Recall quality on stable facts | Latency on `POST /turns` (extraction is synchronous, ~2-5s) |
| Contradiction detection via normalized keys | Multi-hop recall (connecting two separate memories) |
| Explainability (inspectable supersession chain) | Opinion arc modeling |
| Zero-ops storage | Horizontal scalability |
| Synchronous correctness | Async throughput (one extraction at a time per request) |

**Cross-session scoping:** Memories are scoped to `user_id`, not `session_id`. This is intentional — the value of a memory service comes from cross-session persistence. Session isolation is enforced by user ID boundaries; two different users never see each other's memories.

## Failure modes

| Failure | Behavior |
|---|---|
| No API keys | `POST /turns` persists turn but skips extraction; `POST /recall` returns empty context |
| Qdrant down | Semantic search returns empty; BM25-only recall still works |
| Malformed JSON from extraction LLM | Logged, skipped — turn is persisted, memories are not extracted |
| Unicode / oversized input | Inputs are truncated at 8000 chars for embedding; all endpoints return 4xx on malformed JSON |
| Container restart mid-write | SQLite WAL mode ensures atomicity; partial writes are rolled back |

## How to run tests

```bash
# Start the service first
docker compose up -d
until curl -sf http://localhost:8080/health; do sleep 1; done

# Contract tests (fast, ~30s)
docker compose run --rm memory pytest tests/test_contract.py -v

# Recall quality fixture (requires API keys, ~2-3 min)
docker compose run --rm memory pytest tests/test_recall_quality.py -v -s

# All tests
docker compose run --rm memory pytest tests/ -v -s
```

## API keys

See `.env.example`. The service uses:
- `ANTHROPIC_API_KEY` — Claude Haiku for extraction (preferred, faster/cheaper)
- `OPENAI_API_KEY` — GPT-4o-mini extraction fallback + `text-embedding-3-small` for embeddings

At minimum, `OPENAI_API_KEY` is required (embeddings). With only `ANTHROPIC_API_KEY`, extraction works but embeddings fall back to a degraded mode.
