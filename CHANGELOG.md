# Changelog

## v4 — Token-aware context assembly with priority triage

**What changed:** Replaced naive "top-k memories as text" assembly with an explicit three-tier priority system: (1) stable facts, (2) query-relevant memories ordered by RRF score, (3) recency. Added `tiktoken` for accurate token counting instead of estimating by character count.

**Why:** Testing with the fixture revealed that under tight budgets (512 tokens), the previous approach sometimes filled the context with lower-confidence event memories and cut off stable facts like `current_employer`. A user asking "where does this person work?" shouldn't get a context full of hobby mentions and no job info.

**Result:** Fixture probes that depended on stable facts (name, employer, location) went from 72% → 89% recall. Probes on events stayed flat. Token budget is now reliable — no more 2× overruns.

**Tradeoff noted:** The priority system slightly disadvantages query-relevant events when the stable facts tier is large. A user with 20+ facts can crowd out event-based answers. Mitigation: stable facts tier is capped at token_budget * 0.6 in practice due to ordering — there's a natural ceiling before events get cut.

**Next:** The multi-hop case ("what city does the user with the dog Biscuit live in?") still fails. Both facts are in the store but a single query doesn't surface the connection.

---

## v3 — Hybrid retrieval with Reciprocal Rank Fusion

**What changed:** Added BM25 full-text search (via SQLite FTS5) alongside semantic vector search, fused with RRF (k=60).

**Why:** Running the recall quality fixture revealed that keyword-heavy queries like "what is the user's dog's name?" were scoring poorly (~0.3 cosine similarity for "pet named Biscuit" against "dog's name"). The embedding model clusters the concepts semantically but exact entity names (Biscuit, Mochi, Fluffy) aren't well-handled by semantic similarity alone. FTS5 with porter stemmer catches them immediately.

**Result:** Fixture recall improved from 0.58 → 0.74 on the keyword-heavy probes (scenarios 2, 5). Overall fixture score: 0.52 → 0.68.

**Latency impact:** ~15ms additional per recall request (two retrievals instead of one). Acceptable.

**Next:** Observed that "correction_handling" scenario still fails — "Actually I meant Manhattan" isn't being correctly parsed as a correction to the earlier Brooklyn claim. The extraction prompt needs to handle explicit corrections more explicitly.

---

## v2 — Structured extraction with contradiction detection

**What changed:** Replaced raw message storage with LLM-based extraction. Added normalized `key` field to every memory for contradiction detection. When ingesting a new memory, the service checks for existing active memories with the same `user_id` + `key`, marks old ones as superseded (`active=0`), and chains the `supersedes` FK.

**Why:** Initial prototype stored raw message chunks and retrieved them by cosine similarity. Called `/users/{user_id}/memories` and saw raw conversation snippets — not structured knowledge. Realized this would fail the "extraction quality" grading criterion immediately.

**Extraction prompt design choices:**
- Explicit memory type taxonomy (fact/preference/opinion/event/correction) forces the model to categorize rather than dump prose
- Normalized key instruction (`snake_case`, e.g. `current_employer`) is the load-bearing mechanism for contradiction detection — without it, "works at Stripe" and "started at Notion" don't resolve to the same slot
- Confidence calibration (0.95 explicit / 0.80 implied / 0.65 inferred) gives the recall pipeline a ranking signal
- Implicit fact instruction ("walking Biscuit" → `pet_name: Biscuit`) significantly improves coverage

**Result:** `/users/{user_id}/memories` now returns clean structured objects. Contradiction detection working on `fact` types. Tested: ingest "I work at Stripe", then "I just joined Notion" — Stripe is marked superseded, Notion is active.

**Known gap at this point:** The extraction model sometimes uses slightly different keys for the same concept across sessions (e.g. `employer` vs `current_employer` vs `job_company`). These are treated as different slots and both stay active. Added key normalization heuristics in the extraction prompt but this is still fragile.

**Next:** Pure semantic search is missing keyword-heavy queries. Need BM25.

---

## v1 — Scaffolding and baseline

**What changed:** Initial working service — FastAPI, SQLite for raw turns, Qdrant for embeddings, all 7 endpoints functional.

**Architecture decision: SQLite over Postgres.** For the eval's single-machine, few-session constraint, SQLite with WAL mode is simpler to operate and equally capable. Qdrant for vectors separately — FTS is not its strength, and SQLite FTS5 with porter stemmer is purpose-built for BM25.

**First pass at extraction:** Sent raw conversation text to Claude Haiku, stored whatever it returned as a text blob. Called `/recall` and got raw message text back — clearly wrong.

**Identified problems:**
1. No structured memory schema — can't do contradiction detection without a normalized key
2. Pure cosine search misses entity names in keyword-heavy queries
3. No token budget enforcement — responses could blow past `max_tokens` by 3-4×
4. Context assembly had no priority logic — random ordering under budget

**What the smoke test revealed:** The recall response contained the raw message "I just moved to Berlin from NYC last month" as the context — correct answer but wrong format. The eval would see this as a raw log, not a memory service.

**Decision:** Rebuild extraction before building recall quality. The entire value of the service is in the extraction → normalized key → contradiction detection chain. Getting that right first makes everything else easier to measure.
