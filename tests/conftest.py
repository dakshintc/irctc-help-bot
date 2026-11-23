"""Pytest fixtures.

Tests run in offline mode so they never need an API key or network access.
If the FAISS index is missing, it is built once from data/faqs/.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("RAG_OFFLINE", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

import ingest  # noqa: E402
import vectorstore  # noqa: E402
from app import create_app  # noqa: E402
from rag import RAGEngine  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def ensure_index():
    if not vectorstore.index_exists():
        rc = ingest.main([])
        assert rc == 0, "ingest.py failed to build the index"


@pytest.fixture(scope="session")
def engine() -> RAGEngine:
    return RAGEngine(offline=True)


@pytest.fixture(scope="session")
def client():
    app = create_app(RAGEngine(offline=True))
    app.config.update(TESTING=True, RATE_LIMIT_PER_MINUTE=1000)
    with app.test_client() as test_client:
        yield test_client
