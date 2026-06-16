"""
Tests for the augmentation pipeline.
Ollama is NOT required — all external calls are mocked.
"""

from __future__ import annotations

from pathlib import Path

from dmoe.augment import (
    build_training_records,
    parse_qa_pairs,
    write_split,
)


def test_parse_qa_pairs_valid_json() -> None:
    raw = '[{"question": "What?", "answer": "This."}]'
    pairs = parse_qa_pairs(raw)
    assert len(pairs) == 1
    assert pairs[0]["question"] == "What?"


def test_parse_qa_pairs_with_surrounding_text() -> None:
    raw = 'Here are the pairs:\n[{"question": "Q?", "answer": "A."}]\nDone.'
    pairs = parse_qa_pairs(raw)
    assert len(pairs) == 1


def test_parse_qa_pairs_malformed_returns_empty() -> None:
    pairs = parse_qa_pairs("not json at all")
    assert pairs == []


def test_build_training_records_structure() -> None:
    qa_pairs = [
        {"question": "What is X?", "answer": "X is Y."},
        {"question": "How does Z work?", "answer": "Z works by..."},
    ]
    records = build_training_records("raw doc", "rewrite of doc", qa_pairs)
    assert len(records) == 2
    for r in records:
        assert "messages" in r
        roles = [m["role"] for m in r["messages"]]
        assert roles == ["system", "user", "assistant"]
        assert "rewrite of doc" in r["messages"][0]["content"]


def test_write_split_creates_files(tmp_path: Path) -> None:
    records = [
        {
            "messages": [
                {"role": "user", "content": f"Q{i}"},
                {"role": "assistant", "content": f"A{i}"},
            ]
        }
        for i in range(5)
    ]
    write_split(records, tmp_path)
    assert (tmp_path / "train.jsonl").exists()
    assert (tmp_path / "valid.jsonl").exists()
    train_lines = (tmp_path / "train.jsonl").read_text().strip().splitlines()
    assert len(train_lines) == 4  # 80% of 5


def test_write_split_minimum_valid_record(tmp_path: Path) -> None:
    records = [
        {"messages": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]}
    ]
    write_split(records, tmp_path)
    valid_lines = (tmp_path / "valid.jsonl").read_text().strip().splitlines()
    assert len(valid_lines) >= 1
