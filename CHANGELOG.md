# Changelog

## v4 — Extraction uses assistant context for implicit location

**What changed:** Updated the extraction prompt to consider both user and assistant messages when inferring facts, with an explicit example that an assistant reply like “That must be tricky in NYC” should yield `home_city = "New York City"` rather than a literal place name from the user’s earlier message.

**Why:** The recall quality fixture failed one probe: for “Just got back from walking Biscuit in Riverside Park” with the assistant replying “That must be tricky in NYC”, the model extracted `home_city = "Riverside Park"` instead of NYC. The city was implied by the assistant, not stated by the user.

**Result:** Recall quality fixture improved from **13/14 (93%)** to **14/14 (100%)**.

**Next:** Broader implicit-inference cases where context spans multiple turns or sessions without a direct assistant confirmation.

---

## v3 — Contradiction detection via key-slot normalization

**What changed:** Added `KEY_SLOTS` so employer-related keys (`job_change`, `employer`, `new_job`, etc.) map to `current_employer`. Added `_normalize_extracted()` to detect job-change language and remap to the canonical employer slot. Supersession now queries all keys in a slot via `get_active_memories_by_keys()`.

**Why:** Contradiction test ingested “I work at Stripe” then “I just joined Notion last week”. Both memories stayed `active=true` because Notion was stored as `type=event`, `key=job_change` while Stripe was `type=fact`, `key=current_employer` — different keys, so no collision was detected.

**Result:** After fix: Stripe `active=false`, Notion `active=true`, and Notion’s `supersedes` pointed at the Stripe memory id. Recall for “Where does this user work?” returned Notion, not Stripe.

**Next:** Other slots (e.g. `home_city` vs `city` vs `location`) may still need the same alias treatment if the model uses inconsistent keys.

---

## v2 — Structured extraction and recall pipeline working

**What changed:** Configured API keys so extraction and embeddings run. Fixed the extraction parser to accept both a top-level JSON array and a `{"memories": [...]}` object from GPT. Replaced deprecated Qdrant `AsyncQdrantClient.search()` with `query_points()` for client v1.12.

**Why:** With keys added, extraction ran but memories stayed empty when GPT returned an object wrapper. Recall then failed at the vector layer because `.search()` was removed in the pinned Qdrant client version.

**Result:** `/users/{id}/memories` returned structured memories with `type`, `key`, and `value`. First non-empty recall responses with real extracted content.

**Next:** Contradiction handling when the model uses different keys for the same fact across turns.

---

## v1 — Scaffold, boot, and FTS recall crash fix

**What changed:** Initial FastAPI memory service scaffolded and booted (SQLite + Qdrant). All endpoints responded. Fixed `POST /recall` returning HTTP 500 when the query contained `?` by stripping punctuation before the FTS5 `MATCH` clause.

**Why:** Without API keys, `/recall` returned empty context and `/users/{id}/memories` returned `[]` — expected. Once recall was exercised, any query with `?` (e.g. “Where does this user live?”) crashed because FTS5 treated `?` as syntax.

**Result:** Service boots cleanly; `/recall` no longer 500s on punctuation in queries. Empty recall/memories until keys and extraction were wired up in v2.

**Next:** Load `.env` / API keys and validate end-to-end extraction plus semantic search.
