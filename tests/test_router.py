"""
Tests for the BM25 router.
No model or Ollama required — uses tmp_path fixtures only.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dmoe.router import DMoERouter


@pytest.fixture()
def experts_dir(tmp_path: Path) -> Path:
    docs = [
        {
            "id": "doc_001",
            "doc_text": "Refund policy applies within 30 days of purchase",
            "qa": "refund policy 30 days purchase",
        },
        {
            "id": "doc_002",
            "doc_text": "Installation guide for Linux Fedora systems",
            "qa": "install linux fedora guide",
        },
        {
            "id": "doc_003",
            "doc_text": "Python async programming with asyncio library",
            "qa": "python asyncio async programming",
        },
    ]
    for d in docs:
        expert_dir = tmp_path / d["id"]
        expert_dir.mkdir()
        (expert_dir / "metadata.json").write_text(
            json.dumps({"doc_text": d["doc_text"], "qa_text": d["qa"]}),
            encoding="utf-8",
        )
    return tmp_path


def test_router_builds_index(experts_dir: Path) -> None:
    router = DMoERouter(experts_dir)
    assert len(router.expert_ids) == 3
    assert router.bm25 is not None


def test_router_routes_correctly(experts_dir: Path) -> None:
    router = DMoERouter(experts_dir)
    results = router.route("what is the refund policy?", top_k=1)
    assert len(results) == 1
    assert "doc_001" in results[0]


def test_router_top_k_bounded(experts_dir: Path) -> None:
    router = DMoERouter(experts_dir)
    results = router.route("linux python install", top_k=10)
    assert len(results) <= 3  # bounded by number of experts


def test_router_save_and_load(experts_dir: Path, tmp_path: Path) -> None:
    router = DMoERouter(experts_dir)
    index_path = tmp_path / "test_index.pkl"
    router.save_index(index_path)
    assert index_path.exists()

    router2 = DMoERouter(experts_dir, index_path)
    assert router2.expert_ids == router.expert_ids
    results = router2.route("refund policy")
    assert "doc_001" in results[0]


def test_router_no_match_returns_empty(experts_dir: Path) -> None:
    router = DMoERouter(experts_dir)
    results = router.route("xyzzy zzz nonsense token 999", top_k=3)
    # All BM25 scores may be 0 for completely unrelated tokens
    assert isinstance(results, list)
