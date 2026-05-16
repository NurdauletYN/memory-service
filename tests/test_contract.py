"""
Contract tests — verify API shape and basic flows.
Run against a live server: pytest tests/test_contract.py -v
"""
import os

import httpx
import pytest

BASE = os.getenv("SERVICE_URL", "http://localhost:8080")
TIMEOUT = 30


@pytest.fixture
def client():
    return httpx.Client(base_url=BASE, timeout=TIMEOUT)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_turns_recall_and_memories(client):
    user_id = "contract-user-001"
    session_id = "contract-session-001"

    client.delete(f"/users/{user_id}")

    r = client.post("/turns", json={
        "session_id": session_id,
        "user_id": user_id,
        "messages": [
            {"role": "user", "content": "I live in Austin and work at Dell."},
            {"role": "assistant", "content": "Nice!"},
        ],
        "timestamp": "2025-04-01T12:00:00Z",
        "metadata": {},
    })
    assert r.status_code == 201
    assert "turn_id" in r.json()

    r = client.get(f"/users/{user_id}/memories")
    assert r.status_code == 200
    assert "memories" in r.json()

    r = client.post("/recall", json={
        "query": "Where does the user live?",
        "session_id": "contract-probe",
        "user_id": user_id,
        "max_tokens": 256,
    })
    assert r.status_code == 200
    body = r.json()
    assert "context" in body
    assert "citations" in body

    r = client.post("/search", json={
        "query": "Austin",
        "user_id": user_id,
        "limit": 5,
    })
    assert r.status_code == 200
    assert "results" in r.json()

    client.delete(f"/users/{user_id}")
    assert client.delete(f"/users/{user_id}").status_code == 204
