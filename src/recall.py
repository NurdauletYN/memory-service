from __future__ import annotations
import logging

import tiktoken

from src.db import Database
from src.vector_store import VectorStore
from src.models import (
    RecallRequest, RecallResponse, SearchRequest, SearchResponse,
    SearchResult, Citation, Memory,
)

logger = logging.getLogger(__name__)

_tokenizer = None


def count_tokens(text: str) -> int:
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = tiktoken.get_encoding("cl100k_base")
    return len(_tokenizer.encode(text))


class RecallPipeline:
    def __init__(self, db: Database, vector_store: VectorStore):
        self.db = db
        self.vector_store = vector_store

    async def recall(self, req: RecallRequest) -> RecallResponse:
        """Hybrid recall: stable facts + BM25 + semantic, fused with RRF."""
        if not req.user_id:
            return RecallResponse(context="", citations=[])

        stable = await self.db.get_active_stable_memories(req.user_id)
        bm25_hits = await self.db.fts_search(req.user_id, req.query, limit=20)
        semantic_hits = await self.vector_store.search(
            query=req.query, user_id=req.user_id, limit=20, active_only=True
        )

        fused = self._rrf(bm25_hits, semantic_hits)
        fused_ids = [item["id"] for item in fused[:15] if not str(item["id"]).startswith("turn:")]
        fused_memories = await self.db.get_memories_by_ids(fused_ids)

        context, citations = self._assemble(
            stable_facts=stable,
            query_relevant=fused_memories,
            fused_scores={item["id"]: item["score"] for item in fused},
            max_tokens=req.max_tokens,
            query=req.query,
        )

        return RecallResponse(context=context, citations=citations)

    async def search(self, req: SearchRequest) -> SearchResponse:
        """Structured search for agent tool calls."""
        limit = min(req.limit, 50)

        semantic = await self.vector_store.search(
            query=req.query,
            user_id=req.user_id,
            session_id=req.session_id,
            limit=limit,
            active_only=True,
        )

        results = []
        for hit in semantic:
            payload = hit.get("payload", {})
            results.append(SearchResult(
                content=payload.get("text", ""),
                score=float(hit.get("score", 0)),
                session_id=payload.get("session_id", ""),
                timestamp="",
                metadata={"type": payload.get("type", "")},
            ))

        return SearchResponse(results=results)

    def _rrf(self, bm25_hits: list[dict], semantic_hits: list[dict],
             k: int = 60) -> list[dict]:
        """Reciprocal Rank Fusion of BM25 and semantic results."""
        scores: dict = {}
        id_to_data: dict = {}

        for rank, hit in enumerate(bm25_hits):
            mid = hit["id"]
            scores[mid] = scores.get(mid, 0) + 1.0 / (k + rank + 1)
            id_to_data[mid] = {"id": mid, "score": scores[mid]}

        for rank, hit in enumerate(semantic_hits):
            payload = hit.get("payload", {})
            mid = str(hit["id"])
            scores[mid] = scores.get(mid, 0) + 1.0 / (k + rank + 1)
            id_to_data[mid] = {
                "id": mid,
                "score": scores[mid],
                "payload": payload,
            }

        ranked = sorted(id_to_data.values(), key=lambda x: x["score"], reverse=True)
        return ranked

    def _assemble(
        self,
        stable_facts: list[Memory],
        query_relevant: list[Memory],
        fused_scores: dict,
        max_tokens: int,
        query: str,
    ) -> tuple[str, list[Citation]]:
        seen_ids: set[str] = set()
        sections: list[str] = []
        citations: list[Citation] = []
        token_budget = max_tokens - 50

        fact_lines: list[str] = []
        for mem in stable_facts:
            if mem.id in seen_ids:
                continue
            line = f"- {mem.key.replace('_', ' ').capitalize()}: {mem.value}"
            fact_lines.append(line)
            seen_ids.add(mem.id)

        if fact_lines:
            block = "## Known facts about this user\n" + "\n".join(fact_lines)
            cost = count_tokens(block)
            if cost <= token_budget:
                sections.append(block)
                token_budget -= cost
                for mem in stable_facts:
                    citations.append(Citation(
                        turn_id=mem.source_turn,
                        score=mem.confidence,
                        snippet=f"{mem.key}: {mem.value[:80]}",
                    ))

        relevant_lines: list[str] = []
        for mem in sorted(query_relevant, key=lambda m: fused_scores.get(m.id, 0), reverse=True):
            if mem.id in seen_ids:
                continue
            if mem.type in ("fact", "preference"):
                continue
            date_str = mem.updated_at[:10] if mem.updated_at else "unknown"
            line = f"- [{date_str}] {mem.value}"
            tokens_needed = count_tokens(line)
            if tokens_needed > token_budget:
                break
            relevant_lines.append(line)
            token_budget -= tokens_needed
            seen_ids.add(mem.id)
            citations.append(Citation(
                turn_id=mem.source_turn,
                score=float(fused_scores.get(mem.id, 0)),
                snippet=mem.value[:80],
            ))

        if relevant_lines:
            sections.append("## Relevant from recent conversations\n" + "\n".join(relevant_lines))

        context = "\n\n".join(sections)
        return context, citations
