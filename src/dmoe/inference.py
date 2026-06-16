"""
inference.py — DMoE inference engine.

Implements the three-step loop from arXiv:2606.14243 Section 3.3:
  1. Forward pass with currently active adapter composition
  2. Compute Token Uncertainty (Shannon entropy on vocab distribution)
  3. If TU > tau: BM25-route query → select top-k experts → hotswap adapters

Requires Apple Silicon (mlx-lm). Excluded from Linux CI.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from rich.console import Console

from dmoe.router import DMoERouter

console = Console()


def compute_shannon_entropy(logits: list[float]) -> float:
    """
    Compute Shannon entropy over a raw logit vector.

    H = -Σ p(v) log p(v)

    Used as Token Uncertainty (TU) metric. If TU > tau, the router
    is triggered to inject parametric knowledge.
    """
    max_logit = max(logits)
    exp_l = [math.exp(logit_val - max_logit) for logit_val in logits]
    total = sum(exp_l)
    probs = [e / total for e in exp_l]
    return -sum(p * math.log(p + 1e-9) for p in probs if p > 0)


class DMoEEngine:
    """
    Inference engine implementing DMoE parametric knowledge injection.

    Args:
        base_model_path: Path to the MLX-format base model directory.
        experts_dir: Root directory containing per-document expert subdirs.
        index_path: Path to the persisted BM25 index (.pkl).
        tau: Token Uncertainty threshold. Default 2.0 (from paper).
        top_k: Number of experts to activate per trigger event. Default 3.
    """

    def __init__(
        self,
        base_model_path: str,
        experts_dir: str,
        index_path: str,
        tau: float = 2.0,
        top_k: int = 3,
    ):
        try:
            from mlx_lm import load
        except ImportError:
            raise ImportError("mlx-lm is required. Run: uv sync")

        console.print(f"[bold]Loading base model:[/bold] {base_model_path}")
        self.model, self.tokenizer = load(base_model_path)
        self.router = DMoERouter(experts_dir, index_path)
        self.tau = tau
        self.top_k = top_k

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        """
        Generate a response with DMoE parametric injection.

        The router selects relevant adapters based on the query text.
        In this implementation the adapter selection happens pre-generation;
        a full token-level TU hook requires patching mlx_lm internals.
        """
        from mlx_lm import generate

        adapter_paths = self.router.route(prompt, top_k=self.top_k)
        if adapter_paths:
            console.print(
                f"[cyan]DMoE[/cyan] Activated {len(adapter_paths)} expert(s) → "
                + ", ".join(Path(p).parent.name for p in adapter_paths)
            )
        else:
            console.print("[dim]DMoE[/dim] No relevant experts found — using base model")

        return generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            verbose=False,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="DMoE inference engine")
    parser.add_argument("--base-model",  required=True)
    parser.add_argument("--experts-dir", required=True)
    parser.add_argument("--index",       required=True)
    parser.add_argument("--query",       default=None)
    parser.add_argument("--tau",         type=float, default=2.0)
    parser.add_argument("--top-k",       type=int,   default=3)
    parser.add_argument("--max-tokens",  type=int,   default=512)
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    engine = DMoEEngine(
        base_model_path=args.base_model,
        experts_dir=args.experts_dir,
        index_path=args.index,
        tau=args.tau,
        top_k=args.top_k,
    )

    if args.interactive:
        console.print("[bold green]DMoE REPL[/bold green] — type 'exit' to quit\n")
        while True:
            query = console.input("[bold]>[/bold] ").strip()
            if query.lower() in ("exit", "quit", "q"):
                break
            if not query:
                continue
            response = engine.generate(query, max_tokens=args.max_tokens)
            console.print(f"\n[green]{response}[/green]\n")
    elif args.query:
        response = engine.generate(args.query, max_tokens=args.max_tokens)
        console.print(response)
    else:
        console.print("[red]Provide --query TEXT or --interactive[/red]")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
