"""
router.py — BM25-based expert router for DMoE.

Builds and persists an inverted index over all expert metadata files.
At inference time, maps a query string to the top-k most relevant
LoRA adapter paths for parametric injection.

Reference: Section 3.2.2 of arXiv:2606.14243
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

from rank_bm25 import BM25Okapi
from rich.console import Console

console = Console()


class DMoERouter:
    """BM25 router that maps a query to the top-k most relevant LoRA expert paths."""

    def __init__(self, experts_dir: str | Path, index_path: str | Path | None = None):
        self.experts_dir = Path(experts_dir)
        self.expert_ids: list[str] = []
        self.corpus: list[str] = []
        self.bm25: BM25Okapi | None = None

        if index_path and Path(index_path).exists():
            self._load_index(Path(index_path))
        else:
            self._build_index()

    def _build_index(self) -> None:
        """Scan experts directory and build BM25 index from metadata.json files."""
        for expert_path in sorted(self.experts_dir.iterdir()):
            meta_file = expert_path / "metadata.json"
            if not meta_file.exists():
                continue
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            self.expert_ids.append(expert_path.name)
            # Combine doc_text + qa_text for richer BM25 representation
            self.corpus.append(meta.get("doc_text", "") + " " + meta.get("qa_text", ""))

        tokenized = [doc.lower().split() for doc in self.corpus]
        self.bm25 = BM25Okapi(tokenized)

    def _load_index(self, index_path: Path) -> None:
        """Load a previously persisted BM25 index from disk."""
        with open(index_path, "rb") as f:
            data = pickle.load(f)
        self.expert_ids = data["expert_ids"]
        self.corpus = data["corpus"]
        self.bm25 = data["bm25"]

    def save_index(self, index_path: str | Path) -> None:
        """Persist the BM25 index to disk."""
        with open(index_path, "wb") as f:
            pickle.dump(
                {"expert_ids": self.expert_ids, "corpus": self.corpus, "bm25": self.bm25}, f
            )

    def route(self, query: str, top_k: int = 3) -> list[str]:
        """
        Return top-k adapter directory paths sorted by BM25 relevance.

        Returns only experts with a positive BM25 score (i.e., at least
        one query token matched). Returns empty list if no match.
        """
        if self.bm25 is None or not self.expert_ids:
            return []
        scores = self.bm25.get_scores(query.lower().split())
        k = min(top_k, len(self.expert_ids))
        top_indices = scores.argsort()[-k:][::-1]
        return [
            str(self.experts_dir / self.expert_ids[i] / "adapter")
            for i in top_indices
            if scores[i] > 0
        ]


def build_index_cli() -> None:
    parser = argparse.ArgumentParser(description="Build BM25 index from experts directory")
    parser.add_argument("--experts-dir", required=True, help="Root experts directory")
    parser.add_argument("--out",         required=True, help="Output .pkl path")
    args = parser.parse_args()

    router = DMoERouter(args.experts_dir)
    router.save_index(args.out)
    console.print(
        f"[green]✓[/green] BM25 index: {len(router.expert_ids)} experts → {args.out}"
    )


if __name__ == "__main__":
    build_index_cli()
