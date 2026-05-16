from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException, Request, status

from src.db import Database
from src.ingestion import IngestionPipeline
from src.models import RecallRequest, RecallResponse, SearchRequest, SearchResponse, TurnRequest
from src.recall import RecallPipeline
from src.vector_store import VectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    db_path = os.getenv("DATABASE_PATH", "./data/memory.db")
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")

    db = Database(db_path)
    await db.initialize()

    vector_store = VectorStore(url=qdrant_url)
    await vector_store.initialize()

    app.state.db = db
    app.state.vector_store = vector_store
    app.state.ingestion = IngestionPipeline(db, vector_store)
    app.state.recall = RecallPipeline(db, vector_store)

    logger.info("Memory service started")
    yield

    await vector_store.close()
    await db.close()
    logger.info("Memory service stopped")


app = FastAPI(title="Memory Service", version="1.0.0", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/turns", status_code=status.HTTP_201_CREATED)
async def ingest_turn(turn: TurnRequest, request: Request):
    turn_id = await request.app.state.ingestion.ingest(turn)
    return {"turn_id": turn_id}


@app.post("/recall", response_model=RecallResponse)
async def recall(req: RecallRequest, request: Request):
    return await request.app.state.recall.recall(req)


@app.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest, request: Request):
    return await request.app.state.recall.search(req)


@app.get("/users/{user_id}/memories")
async def get_user_memories(user_id: str, request: Request):
    memories = await request.app.state.db.get_user_memories(user_id)
    return {"memories": [m.model_dump() for m in memories]}


@app.delete("/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(user_id: str, request: Request):
    await request.app.state.db.delete_user(user_id)
    await request.app.state.vector_store.delete_by_user(user_id)


@app.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(session_id: str, request: Request):
    await request.app.state.db.delete_session(session_id)
