from __future__ import annotations
import json
import re
import uuid
import logging
import os
from datetime import datetime, timezone
from typing import Optional

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from src.models import TurnRequest, Memory
from src.db import Database
from src.vector_store import VectorStore

logger = logging.getLogger(__name__)

EXTRACTION_PROMPT = """You are a memory extraction engine. Given a conversation turn, extract all structured memories about the USER.

Consider BOTH user and assistant messages when inferring facts. The assistant's reply often confirms or reveals information about the user (e.g. location, context) — use it alongside what the user said.

Extract memories in these categories:
- "fact": stable personal facts (name, job, location, family, pets, physical attributes)
- "preference": likes/dislikes, dietary restrictions, communication style preferences
- "opinion": views, beliefs, attitudes toward topics (may evolve over time)
- "event": specific things that happened or are planned
- "correction": explicit corrections ("actually I meant...", "sorry not X — Y")

Rules:
1. Extract IMPLICIT facts too. "Walking Biscuit this morning" → pet named Biscuit.
2. Example — assistant confirms location: user says "I skipped the bagel cart" and assistant replies "That must be tricky in NYC" → extract home_city = "New York City" (not the park name from an earlier message).
3. Use a normalized KEY for each memory (snake_case, e.g. "current_employer", "pet_name", "home_city").
4. Employer/job changes MUST use key "current_employer" (not "job_change" or "employer").
5. Corrections should reference the thing being corrected in the key.
6. If nothing extractable, return empty array.
7. Return ONLY valid JSON — no preamble, no markdown.

Output format (JSON object with a "memories" array):
{
  "memories": [
    {
      "type": "fact|preference|opinion|event|correction",
      "key": "normalized_key",
      "value": "the fact as a clear statement",
      "confidence": 0.0-1.0,
      "implicit": true|false
    }
  ]
}

Confidence guide:
- 0.95: Explicitly stated, unambiguous ("I work at Google")
- 0.80: Clearly implied ("just got back from my Berlin apartment" → lives in Berlin)
- 0.65: Inferred from context, could be wrong
- 0.50: Uncertain / mentioned in passing"""

# Canonical key -> alias keys that occupy the same slot (supersede together)
KEY_SLOTS: dict[str, frozenset[str]] = {
    "current_employer": frozenset({
        "current_employer", "employer", "job_company", "company", "workplace", "job_change",
    }),
    "home_city": frozenset({"home_city", "city", "location", "residence", "home_location"}),
}

_EMPLOYER_JOIN_RE = re.compile(
    r"(?:joined|started at|work(?:s|ing)? at|left)\s+([A-Z][A-Za-z0-9&.\-]*)",
    re.IGNORECASE,
)


class IngestionPipeline:
    def __init__(self, db: Database, vector_store: VectorStore):
        self.db = db
        self.vector_store = vector_store
        self._anthropic: Optional[AsyncAnthropic] = None
        self._openai: Optional[AsyncOpenAI] = None
        self._init_clients()

    def _init_clients(self):
        if os.getenv("ANTHROPIC_API_KEY"):
            self._anthropic = AsyncAnthropic()
        if os.getenv("OPENAI_API_KEY"):
            self._openai = AsyncOpenAI()

    async def ingest(self, turn: TurnRequest) -> str:
        now = datetime.now(timezone.utc).isoformat()

        turn_id = await self.db.save_turn(
            session_id=turn.session_id,
            user_id=turn.user_id,
            messages_json=json.dumps([m.model_dump() for m in turn.messages]),
            timestamp=turn.timestamp,
            metadata_json=json.dumps(turn.metadata),
        )

        if turn.user_id:
            extracted = await self._extract(turn)
            for ext in extracted:
                await self._store_memory(ext, turn, turn_id, now)

        turn_text = self._turn_to_text(turn)
        fake_mem_id = f"turn:{turn_id}"
        await self.vector_store.upsert_memory(
            memory_id=fake_mem_id,
            text=turn_text,
            user_id=turn.user_id,
            session_id=turn.session_id,
            turn_id=turn_id,
            memory_type="turn",
            active=True,
        )

        return turn_id

    async def _extract(self, turn: TurnRequest) -> list[dict]:
        conversation = self._turn_to_text(turn)
        raw = await self._call_extraction_llm(conversation)
        if not raw:
            return []
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, dict):
                if isinstance(parsed.get("memories"), list):
                    return parsed["memories"]
                for value in parsed.values():
                    if isinstance(value, list) and value and isinstance(value[0], dict):
                        return value
            return []
        except json.JSONDecodeError:
            try:
                start = raw.find("[")
                end = raw.rfind("]") + 1
                if start >= 0 and end > start:
                    return json.loads(raw[start:end])
            except Exception:
                pass
            logger.warning("Failed to parse extraction output: %s", raw[:200])
            return []

    async def _call_extraction_llm(self, conversation: str) -> Optional[str]:
        model = os.getenv("EXTRACTION_MODEL", "claude-haiku-4-5-20251001")

        if self._anthropic and "claude" in model:
            try:
                response = await self._anthropic.messages.create(
                    model=model,
                    max_tokens=1024,
                    system=EXTRACTION_PROMPT,
                    messages=[{"role": "user", "content": f"Extract memories from this conversation:\n\n{conversation}"}],
                )
                return response.content[0].text
            except Exception as e:
                logger.warning("Anthropic extraction failed: %s", e)

        if self._openai:
            try:
                response = await self._openai.chat.completions.create(
                    model="gpt-4o-mini",
                    max_tokens=1024,
                    messages=[
                        {"role": "system", "content": EXTRACTION_PROMPT},
                        {"role": "user", "content": f"Extract memories from this conversation:\n\n{conversation}"},
                    ],
                    response_format={"type": "json_object"},
                )
                text = response.choices[0].message.content
                parsed = json.loads(text)
                if isinstance(parsed, dict) and "memories" in parsed:
                    return json.dumps(parsed["memories"])
                if isinstance(parsed, dict) and len(parsed) == 1:
                    return json.dumps(list(parsed.values())[0])
                return text
            except Exception as e:
                logger.warning("OpenAI extraction failed: %s", e)

        logger.error("No extraction LLM available")
        return None

    def _normalize_extracted(self, extracted: dict) -> dict:
        """Map aliases and job-change events onto canonical keys for contradiction detection."""
        key = (extracted.get("key") or "unknown").strip().lower()
        value = (extracted.get("value") or "").strip()
        mem_type = extracted.get("type", "fact")

        for canonical, aliases in KEY_SLOTS.items():
            if key in aliases:
                if canonical == "current_employer":
                    match = _EMPLOYER_JOIN_RE.search(value)
                    if match:
                        value = match.group(1)
                    mem_type = "fact"
                return {**extracted, "key": canonical, "value": value, "type": mem_type}

        if mem_type == "event" and any(
            phrase in value.lower() for phrase in ("joined", "started at", "left ", "work at")
        ):
            match = _EMPLOYER_JOIN_RE.search(value)
            if match:
                return {
                    **extracted,
                    "key": "current_employer",
                    "value": match.group(1),
                    "type": "fact",
                }

        return {**extracted, "key": key, "value": value, "type": mem_type}

    def _slot_keys(self, canonical_key: str) -> frozenset[str]:
        return KEY_SLOTS.get(canonical_key, frozenset({canonical_key}))

    async def _store_memory(self, extracted: dict, turn: TurnRequest, turn_id: str, now: str):
        extracted = self._normalize_extracted(extracted)
        key = extracted.get("key", "unknown")
        value = extracted.get("value", "")
        mem_type = extracted.get("type", "fact")
        confidence = float(extracted.get("confidence", 0.8))

        if not value or not key:
            return

        existing = await self.db.get_active_memories_by_keys(
            turn.user_id, self._slot_keys(key),
        )

        supersedes_id = None
        if existing:
            most_recent = max(existing, key=lambda m: m["updated_at"])
            supersedes_id = most_recent["id"]
            await self.db.supersede_memory(most_recent["id"], now)
            await self.vector_store.deactivate_memory(most_recent["id"])
            logger.info("Superseded memory %s (key=%s) for user %s", supersedes_id, key, turn.user_id)

        new_id = str(uuid.uuid4())
        memory = Memory(
            id=new_id,
            user_id=turn.user_id,
            type=mem_type,
            key=key,
            value=value,
            confidence=confidence,
            source_session=turn.session_id,
            source_turn=turn_id,
            created_at=now,
            updated_at=now,
            supersedes=supersedes_id,
            active=True,
        )
        await self.db.save_memory(memory)
        await self.vector_store.upsert_memory(
            memory_id=new_id,
            text=f"{key}: {value}",
            user_id=turn.user_id,
            session_id=turn.session_id,
            turn_id=turn_id,
            memory_type=mem_type,
            active=True,
        )

    def _turn_to_text(self, turn: TurnRequest) -> str:
        lines = []
        for msg in turn.messages:
            role = msg.role.upper()
            if msg.name:
                role = f"TOOL({msg.name})"
            lines.append(f"{role}: {msg.content}")
        return "\n".join(lines)
