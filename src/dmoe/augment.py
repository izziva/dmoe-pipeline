"""
augment.py — Document -> QA dataset for DMoE expert training.

Each document is:
  1. Paraphrased (rewrite) to reduce verbatim memorization
  2. Converted to 3 QA pairs via local LLM (Ollama)
  3. Serialized as chat-format JSONL for mlx_lm.lora

The raw document is NOT passed to the LoRA — only the augmented pairs are.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from rich.console import Console

console = Console()

REWRITE_PROMPT = """\
Paraphrase the following document preserving ALL factual content exactly. \
Do not add, remove or interpret any information.

Document:
{document}

Paraphrased version (return only the paraphrase, no commentary):"""

QA_PROMPT = """\
Given the document below, generate exactly {n} question-answer pairs \
answerable ONLY from the document. Each pair must cover a distinct fact.
Output ONLY a valid JSON array, no extra text.

Document:
{document}

Output:
[{{"question": "...", "answer": "..."}}]"""


def call_ollama(prompt: str, model: str, timeout: int = 120) -> str:
    """Call a local Ollama model and return its text output."""
    result = subprocess.run(
        ["ollama", "run", model, prompt],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Ollama error: {result.stderr.strip()}")
    return result.stdout.strip()


def parse_qa_pairs(raw: str) -> list[dict]:
    """
    Robustly extract a JSON array from LLM output.
    Falls back to regex extraction if the output has surrounding text.
    """
    try:
        start = raw.index("[")
        end = raw.rindex("]") + 1
        return json.loads(raw[start:end])
    except (ValueError, json.JSONDecodeError):
        console.print(f"[yellow]WARN[/yellow] Could not parse QA pairs:\n{raw[:200]}")
        return []


def augment_document(doc_text: str, model: str, num_qa: int = 3) -> dict:
    """
    Run rewrite + QA generation for a single document.

    Returns:
        dict with keys 'rewrite' (str) and 'qa_pairs' (list[dict])
    """
    rewrite = call_ollama(REWRITE_PROMPT.format(document=doc_text), model)
    raw_qa = call_ollama(QA_PROMPT.format(document=doc_text, n=num_qa), model)
    qa_pairs = parse_qa_pairs(raw_qa)
    return {"rewrite": rewrite, "qa_pairs": qa_pairs}


def build_training_records(doc_text: str, rewrite: str, qa_pairs: list[dict]) -> list[dict]:
    """
    Convert augmented data into chat-format JSONL records for mlx_lm.lora.

    The rewrite is injected as system prompt context so the LoRA learns
    to answer from the document's parametric knowledge.
    """
    system = (
        "You are a knowledgeable assistant. "
        f"Use the following document to answer accurately:\n\n{rewrite}"
    )
    return [
        {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": qa["question"]},
                {"role": "assistant", "content": qa["answer"]},
            ]
        }
        for qa in qa_pairs
    ]


def write_split(records: list[dict], out_dir: Path, train_ratio: float = 0.8) -> None:
    """Write train.jsonl and valid.jsonl with an 80/20 split."""
    out_dir.mkdir(parents=True, exist_ok=True)
    split = max(1, int(len(records) * train_ratio))
    train_records = records[:split]
    valid_records = records[split:] or records[:1]  # at least 1 validation example

    (out_dir / "train.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in train_records),
        encoding="utf-8",
    )
    (out_dir / "valid.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in valid_records),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Augment a document into DMoE training data")
    parser.add_argument("--doc", required=True, help="Path to source .txt document")
    parser.add_argument("--out", required=True, help="Output directory for train/valid JSONL")
    parser.add_argument("--model", default="qwen2.5:1.5b", help="Ollama model for augmentation")
    parser.add_argument("--num-qa", type=int, default=3, help="Number of QA pairs to generate")
    args = parser.parse_args()

    doc_path = Path(args.doc)
    out_dir = Path(args.out)
    doc_text = doc_path.read_text(encoding="utf-8")

    console.print(f"[bold]Augmenting:[/bold] {doc_path.name}")
    augmented = augment_document(doc_text, args.model, args.num_qa)

    if not augmented["qa_pairs"]:
        console.print("[red]ERROR[/red] No QA pairs generated — aborting.")
        raise SystemExit(1)

    records = build_training_records(doc_text, augmented["rewrite"], augmented["qa_pairs"])
    write_split(records, out_dir)

    meta = {
        "doc_text": doc_text,
        "qa_text": " ".join(
            qa["question"] + " " + qa["answer"] for qa in augmented["qa_pairs"]
        ),
        "rewrite": augmented["rewrite"],
    }
    (out_dir.parent / "metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    n_train = max(1, int(len(records) * 0.8))
    console.print(
        f"[green]✓[/green] {len(records)} records → {out_dir} "
        f"(train: {n_train}, valid: {len(records) - n_train})"
    )


if __name__ == "__main__":
    main()
