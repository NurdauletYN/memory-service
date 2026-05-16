from __future__ import annotations
import os
import uuid
import logging
import aiosqlite
from typing import Optional
from src.models import Memory

logger = logging.getLogger(__name__)


class Database:
    def __init__(self, path: str):
        self.path = path
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._db: Optional[aiosqlite.Connection] = None

    async def initialize(self):
        self._db = await aiosqlite.connect(self.path)
        self._db.row_factory = aiosqlite.Row
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA foreign_keys=ON")
        await self._create_schema()
        await self._db.commit()
        logger.info("Database initialized at %s", self.path)

    async def _create_schema(self):
        await self._db.executescript("""
        CREATE TABLE IF NOT EXISTS turns (
            id          TEXT PRIMARY KEY,
            session_id  TEXT NOT NULL,
            user_id     TEXT,
            messages    TEXT NOT NULL,
            timestamp   TEXT NOT NULL,
            metadata    TEXT NOT NULL DEFAULT '{}'
        );

        CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id);
        CREATE INDEX IF NOT EXISTS idx_turns_user    ON turns(user_id);

        CREATE TABLE IF NOT EXISTS memories (
            id              TEXT PRIMARY KEY,
            user_id         TEXT,
            type            TEXT NOT NULL,
            key             TEXT NOT NULL,
            value           TEXT NOT NULL,
            confidence      REAL NOT NULL DEFAULT 0.8,
            source_session  TEXT NOT NULL,
            source_turn     TEXT NOT NULL,
            created_at      TEXT NOT NULL,
            updated_at      TEXT NOT NULL,
            supersedes      TEXT,
            active          INTEGER NOT NULL DEFAULT 1,
            FOREIGN KEY(supersedes) REFERENCES memories(id)
        );

        CREATE INDEX IF NOT EXISTS idx_memories_user   ON memories(user_id);
        CREATE INDEX IF NOT EXISTS idx_memories_active ON memories(user_id, active);
        CREATE INDEX IF NOT EXISTS idx_memories_key    ON memories(user_id, key);

        CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
            id UNINDEXED,
            user_id UNINDEXED,
            content,
            tokenize='porter unicode61'
        );
        """)

    async def close(self):
        if self._db:
            await self._db.close()

    # ── turns ────────────────────────────────────────────────────────────────

    async def save_turn(self, session_id: str, user_id: Optional[str],
                        messages_json: str, timestamp: str, metadata_json: str) -> str:
        turn_id = str(uuid.uuid4())
        await self._db.execute(
            "INSERT INTO turns(id, session_id, user_id, messages, timestamp, metadata) VALUES(?,?,?,?,?,?)",
            (turn_id, session_id, user_id, messages_json, timestamp, metadata_json),
        )
        await self._db.commit()
        return turn_id

    async def get_turn(self, turn_id: str) -> Optional[dict]:
        async with self._db.execute("SELECT * FROM turns WHERE id=?", (turn_id,)) as cur:
            row = await cur.fetchone()
            return dict(row) if row else None

    async def get_recent_turns(self, user_id: str, limit: int = 10) -> list[dict]:
        async with self._db.execute(
            "SELECT * FROM turns WHERE user_id=? ORDER BY timestamp DESC LIMIT ?",
            (user_id, limit),
        ) as cur:
            return [dict(r) for r in await cur.fetchall()]

    # ── memories ─────────────────────────────────────────────────────────────

    async def save_memory(self, memory: Memory):
        await self._db.execute(
            """INSERT INTO memories
               (id, user_id, type, key, value, confidence, source_session,
                source_turn, created_at, updated_at, supersedes, active)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
            (memory.id, memory.user_id, memory.type, memory.key, memory.value,
             memory.confidence, memory.source_session, memory.source_turn,
             memory.created_at, memory.updated_at, memory.supersedes,
             1 if memory.active else 0),
        )
        fts_content = f"{memory.key} {memory.value}"
        await self._db.execute(
            "INSERT INTO memories_fts(id, user_id, content) VALUES(?,?,?)",
            (memory.id, memory.user_id or "", fts_content),
        )
        await self._db.commit()

    async def supersede_memory(self, old_id: str, updated_at: str):
        """Mark an existing memory as inactive (superseded)."""
        await self._db.execute(
            "UPDATE memories SET active=0, updated_at=? WHERE id=?",
            (updated_at, old_id),
        )
        await self._db.commit()

    async def get_active_memories_by_key(self, user_id: str, key: str) -> list[dict]:
        """Return active memories for a user matching a given key (exact or fuzzy)."""
        return await self.get_active_memories_by_keys(user_id, {key})

    async def get_active_memories_by_keys(self, user_id: str, keys: set[str]) -> list[dict]:
        """Return active memories matching any key in the set (same contradiction slot)."""
        if not keys:
            return []
        placeholders = ",".join("?" * len(keys))
        params = (user_id, *keys)
        async with self._db.execute(
            f"SELECT * FROM memories WHERE user_id=? AND key IN ({placeholders}) AND active=1",
            params,
        ) as cur:
            return [dict(r) for r in await cur.fetchall()]

    async def get_user_memories(self, user_id: str) -> list[Memory]:
        async with self._db.execute(
            "SELECT * FROM memories WHERE user_id=? ORDER BY updated_at DESC",
            (user_id,),
        ) as cur:
            rows = await cur.fetchall()
        return [_row_to_memory(dict(r)) for r in rows]

    async def get_active_stable_memories(self, user_id: str) -> list[Memory]:
        """Stable facts: facts and preferences that are active."""
        async with self._db.execute(
            """SELECT * FROM memories
               WHERE user_id=? AND active=1 AND type IN ('fact','preference')
               ORDER BY confidence DESC, updated_at DESC""",
            (user_id,),
        ) as cur:
            rows = await cur.fetchall()
        return [_row_to_memory(dict(r)) for r in rows]

    async def fts_search(self, user_id: str, query: str, limit: int = 20) -> list[dict]:
        """BM25-ranked FTS search over memory content."""
        import re
        safe = re.sub(r'[^\w\s]', ' ', query.replace('"', ' '))
        safe = " ".join(safe.split())
        if not safe:
            return []
        try:
            async with self._db.execute(
            """SELECT m.*, bm25(memories_fts) AS bm25_score
               FROM memories_fts
               JOIN memories m ON m.id = memories_fts.id
               WHERE memories_fts MATCH ? AND memories_fts.user_id=? AND m.active=1
               ORDER BY bm25_score
               LIMIT ?""",
            (safe, user_id, limit),
            ) as cur:
                return [dict(r) for r in await cur.fetchall()]
        except aiosqlite.Error as e:
            logger.warning("FTS search failed for query %r: %s", query, e)
            return []

    async def get_memories_by_ids(self, ids: list[str]) -> list[Memory]:
        if not ids:
            return []
        placeholders = ",".join("?" * len(ids))
        async with self._db.execute(
            f"SELECT * FROM memories WHERE id IN ({placeholders})", ids,
        ) as cur:
            return [_row_to_memory(dict(r)) for r in await cur.fetchall()]

    async def get_turns_by_ids(self, ids: list[str]) -> list[dict]:
        if not ids:
            return []
        placeholders = ",".join("?" * len(ids))
        async with self._db.execute(
            f"SELECT * FROM turns WHERE id IN ({placeholders})", ids,
        ) as cur:
            return [dict(r) for r in await cur.fetchall()]

    # ── cleanup ───────────────────────────────────────────────────────────────

    async def delete_session(self, session_id: str):
        await self._db.execute("DELETE FROM turns WHERE session_id=?", (session_id,))
        await self._db.execute(
            "DELETE FROM memories WHERE source_session=?", (session_id,)
        )
        await self._db.execute(
            "DELETE FROM memories_fts WHERE id IN (SELECT id FROM memories WHERE source_session=?)",
            (session_id,),
        )
        await self._db.commit()

    async def delete_user(self, user_id: str):
        await self._db.execute(
            "DELETE FROM memories_fts WHERE user_id=?", (user_id,)
        )
        await self._db.execute("DELETE FROM memories WHERE user_id=?", (user_id,))
        await self._db.execute("DELETE FROM turns WHERE user_id=?", (user_id,))
        await self._db.commit()


def _row_to_memory(row: dict) -> Memory:
    return Memory(
        id=row["id"],
        user_id=row.get("user_id"),
        type=row["type"],
        key=row["key"],
        value=row["value"],
        confidence=row["confidence"],
        source_session=row["source_session"],
        source_turn=row["source_turn"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        supersedes=row.get("supersedes"),
        active=bool(row["active"]),
    )
