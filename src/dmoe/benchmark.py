"""
benchmark.py — Evaluate DMoE against RAG baseline on HotpotQA.

Metrics: Exact Match (EM) and Token-level F1, as used in arXiv:2606.14243.
"""

from __future__ import annotations

import argparse
import re
import string
from collections import Counter

from rich.console import Console
from rich.table import Table

console = Console()


def normalize_answer(text: str) -> str:
    """Lowercase, strip punctuation and articles (standard QA normalization)."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, ground_truth: str) -> float:
    """Return 1.0 if normalized prediction equals normalized ground truth."""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def token_f1(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 between prediction and ground truth."""
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens   = normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall    = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark DMoE on HotpotQA")
    parser.add_argument("--base-model",  required=True)
    parser.add_argument("--experts-dir", required=True)
    parser.add_argument("--dataset",     default="hotpotqa")
    parser.add_argument("--split",       default="validation[:100]")
    parser.add_argument("--tau",         type=float, default=2.0)
    parser.add_argument("--top-k",       type=int,   default=3)
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        console.print("[red]Install benchmark group: uv sync --group benchmark[/red]")
        raise SystemExit(1)

    from dmoe.inference import DMoEEngine

    console.print(f"Loading dataset: {args.dataset} ({args.split})")
    ds = load_dataset(args.dataset, "distractor", split=args.split, trust_remote_code=True)

    engine = DMoEEngine(
        base_model_path=args.base_model,
        experts_dir=args.experts_dir,
        index_path=str(args.experts_dir + "/bm25_index.pkl"),
        tau=args.tau,
        top_k=args.top_k,
    )

    em_scores, f1_scores = [], []
    for sample in ds:
        question = sample["question"]
        answer   = sample["answer"]
        pred = engine.generate(question, max_tokens=64)
        em_scores.append(exact_match(pred, answer))
        f1_scores.append(token_f1(pred, answer))

    table = Table(title="DMoE Benchmark Results")
    table.add_column("Metric", style="bold")
    table.add_column("Score",  style="cyan")
    table.add_row("Exact Match (EM)", f"{sum(em_scores)/len(em_scores)*100:.1f}%")
    table.add_row("Token F1",         f"{sum(f1_scores)/len(f1_scores)*100:.1f}%")
    table.add_row("Samples evaluated", str(len(em_scores)))
    console.print(table)


if __name__ == "__main__":
    main()
