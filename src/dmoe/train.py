"""
train.py — Wrapper around mlx_lm.lora with DMoE-specific defaults.

Key constraint from arXiv:2606.14243:
  --lora-layers 1  →  train ONLY the last FFN layer of the transformer.
  This keeps the adapter small (~481 KB) and preserves the KV-cache
  for all previous layers during inference.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from rich.console import Console

console = Console()

# Paper-specified defaults — do not change without justification
LORA_DEFAULTS = {
    "lora_layers": 1,       # ONLY the last FFN layer
    "rank": 4,
    "lora_scale": 16,       # alpha
    "learning_rate": 1e-5,
    "iters": 100,
    "batch_size": 2,        # safe on M4 Pro 24GB
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a single DMoE LoRA expert")
    parser.add_argument("--doc-id",       required=True,  help="Expert ID (subdirectory name)")
    parser.add_argument("--experts-dir",  required=True,  help="Root experts directory")
    parser.add_argument("--base-model",   required=True,  help="Path to base MLX model")
    parser.add_argument("--iters",        type=int,   default=LORA_DEFAULTS["iters"])
    parser.add_argument("--batch-size",   type=int,   default=LORA_DEFAULTS["batch_size"])
    parser.add_argument("--rank",         type=int,   default=LORA_DEFAULTS["rank"])
    parser.add_argument("--learning-rate",type=float, default=LORA_DEFAULTS["learning_rate"])
    args = parser.parse_args()

    experts_dir  = Path(args.experts_dir)
    data_dir     = experts_dir / args.doc_id / "data"
    adapter_path = experts_dir / args.doc_id / "adapter"
    adapter_path.mkdir(parents=True, exist_ok=True)

    if not (data_dir / "train.jsonl").exists():
        console.print(
            f"[red]ERROR[/red] Missing {data_dir}/train.jsonl — run dmoe-augment first."
        )
        raise SystemExit(1)

    cmd = [
        "mlx_lm.lora",
        "--model",         args.base_model,
        "--train",
        "--data",          str(data_dir),
        "--lora-layers",   str(LORA_DEFAULTS["lora_layers"]),
        "--rank",          str(args.rank),
        "--lora-scale",    str(LORA_DEFAULTS["lora_scale"]),
        "--learning-rate", str(args.learning_rate),
        "--iters",         str(args.iters),
        "--batch-size",    str(args.batch_size),
        "--adapter-path",  str(adapter_path),
    ]

    console.print(f"[bold]Training expert:[/bold] {args.doc_id}")
    console.print(
        f"  lora-layers={LORA_DEFAULTS['lora_layers']}  "
        f"rank={args.rank}  iters={args.iters}  batch={args.batch_size}"
    )

    result = subprocess.run(cmd)

    if result.returncode == 0:
        adapter_file = adapter_path / "adapters.safetensors"
        if adapter_file.exists():
            size_kb = adapter_file.stat().st_size / 1024
            console.print(f"[green]✓[/green] Expert saved → {adapter_path} ({size_kb:.0f} KB)")
        else:
            console.print(f"[green]✓[/green] Training complete → {adapter_path}")
    else:
        console.print("[red]Training failed[/red]")
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
