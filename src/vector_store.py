from __future__ import annotations

import logging
import os
import uuid
from typing import Any, Optional

from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PointStruct,
    VectorParams,
)

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536


class VectorStore:
    def __init__(
        self,
        url: str,
        collection_name: str = "memories",
        embedding_dim: int = EMBEDDING_DIM,
    ):
        self.url = url
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        self._client: Optional[AsyncQdrantClient] = None
        self._openai: Optional[AsyncOpenAI] = None

    async def initialize(self) -> None:
        self._client = AsyncQdrantClient(url=self.url)
        if os.getenv("OPENAI_API_KEY"):
            self._openai = AsyncOpenAI()

        response = await self._client.get_collections()
        names = {c.name for c in response.collections}
        if self.collection_name not in names:
            await self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE,
                ),
            )
            logger.info("Created Qdrant collection %s", self.collection_name)
        else:
            logger.info("Qdrant collection %s already exists", self.collection_name)

    async def close(self) -> None:
        if self._client:
            await self._client.close()

    def _point_id(self, memory_id: str) -> str:
        """Map arbitrary memory ids (including turn: prefixes) to valid point ids."""
        try:
            uuid.UUID(memory_id)
            return memory_id
        except ValueError:
            return str(uuid.uuid5(uuid.NAMESPACE_DNS, memory_id))

    async def _embed(self, text: str) -> Optional[list[float]]:
        if not self._openai:
            logger.warning("OPENAI_API_KEY not set — skipping embedding")
            return None
        truncated = text[:8000]
        response = await self._openai.embeddings.create(
            model=EMBEDDING_MODEL,
            input=truncated,
        )
        return response.data[0].embedding

    async def upsert_memory(
        self,
        memory_id: str,
        text: str,
        user_id: Optional[str],
        session_id: str,
        turn_id: str,
        memory_type: str,
        active: bool,
    ) -> None:
        if not self._client:
            return
        vector = await self._embed(text)
        if vector is None:
            return

        point_id = self._point_id(memory_id)
        payload = {
            "text": text,
            "user_id": user_id or "",
            "session_id": session_id,
            "turn_id": turn_id,
            "type": memory_type,
            "active": active,
            "memory_id": memory_id,
        }
        await self._client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(id=point_id, vector=vector, payload=payload),
            ],
        )

    async def deactivate_memory(self, memory_id: str) -> None:
        if not self._client:
            return
        point_id = self._point_id(memory_id)
        await self._client.set_payload(
            collection_name=self.collection_name,
            payload={"active": False},
            points=[point_id],
        )

    async def delete_by_user(self, user_id: str) -> None:
        if not self._client:
            return
        await self._client.delete(
            collection_name=self.collection_name,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[
                        FieldCondition(key="user_id", match=MatchValue(value=user_id)),
                    ]
                )
            ),
        )

    async def search(
        self,
        query: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: int = 20,
        active_only: bool = True,
    ) -> list[dict[str, Any]]:
        if not self._client:
            return []

        vector = await self._embed(query)
        if vector is None:
            return []

        must: list[FieldCondition] = []
        if user_id:
            must.append(FieldCondition(key="user_id", match=MatchValue(value=user_id)))
        if session_id:
            must.append(FieldCondition(key="session_id", match=MatchValue(value=session_id)))
        if active_only:
            must.append(FieldCondition(key="active", match=MatchValue(value=True)))

        query_filter = Filter(must=must) if must else None

        response = await self._client.query_points(
            collection_name=self.collection_name,
            query=vector,
            query_filter=query_filter,
            limit=limit,
        )

        hits = []
        for point in response.points:
            payload = point.payload or {}
            hits.append({
                "id": payload.get("memory_id", str(point.id)),
                "score": point.score,
                "payload": payload,
            })
        return hits
