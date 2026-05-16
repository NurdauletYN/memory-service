from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: Literal["user", "assistant", "system", "tool"]
    content: str
    name: Optional[str] = None


class TurnRequest(BaseModel):
    session_id: str
    user_id: Optional[str] = None
    messages: list[Message]
    timestamp: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class Memory(BaseModel):
    id: str
    user_id: Optional[str] = None
    type: str
    key: str
    value: str
    confidence: float = 0.8
    source_session: str
    source_turn: str
    created_at: str
    updated_at: str
    supersedes: Optional[str] = None
    active: bool = True


class RecallRequest(BaseModel):
    query: str
    session_id: str
    user_id: Optional[str] = None
    max_tokens: int = 512


class Citation(BaseModel):
    turn_id: str
    score: float
    snippet: str


class RecallResponse(BaseModel):
    context: str
    citations: list[Citation] = Field(default_factory=list)


class SearchRequest(BaseModel):
    query: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    limit: int = 10


class SearchResult(BaseModel):
    content: str
    score: float
    session_id: str
    timestamp: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    results: list[SearchResult] = Field(default_factory=list)
