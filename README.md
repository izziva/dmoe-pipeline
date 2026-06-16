# DMoE Pipeline

**Parametric Knowledge Injection with Decoupled Mixture-of-Experts on Apple Silicon**

Implementation of [arXiv:2606.14243](https://arxiv.org/abs/2606.14243) — *Decoupled Mixture-of-Experts for Parametric Knowledge Injection*, Tsinghua University (2026).

Instead of RAG (which bloats the context window), DMoE trains lightweight LoRA adapters (~481 KB each) on the **last FFN layer only**, storing them on disk. At inference, a BM25 router selects the top-3 relevant experts, triggered by Shannon entropy on the current token.

## Requirements

- Apple Silicon Mac (M1–M4), 16 GB+ unified memory
- macOS Sequoia 15+
- [uv](https://docs.astral.sh/uv/) package manager
- [Ollama](https://ollama.ai) for document augmentation

## Quickstart

```bash
# 1. Clone and setup
git clone https://github.com/izziva/dmoe-pipeline
cd dmoe-pipeline
uv sync --all-groups

# 2. Download base model
uv run huggingface-cli download \
    mlx-community/Qwen2.5-1.5B-Instruct-4bit \
    --local-dir ./models/qwen2.5-1.5b-4bit

# 3. Pull augmentation model
ollama pull qwen2.5:1.5b

# 4. Build an expert from a document
uv run dmoe-augment  --doc ./docs/my_doc.txt --out ./experts/doc_001/data
uv run dmoe-train    --doc-id doc_001 --experts-dir ./experts --base-model ./models/qwen2.5-1.5b-4bit
uv run dmoe-build-idx --experts-dir ./experts --out ./experts/bm25_index.pkl

# 5. Query
uv run dmoe-infer \
    --base-model ./models/qwen2.5-1.5b-4bit \
    --experts-dir ./experts \
    --index ./experts/bm25_index.pkl \
    --interactive
```

## Key Numbers (from paper)

| Model | VRAM | Expert size | Latency |
|-------|------|-------------|---------|
| Llama-3.2-1B | 7.2 GB | ~481 KB | ~2.7s |
| Qwen2.5-1.5B | 8.3 GB | ~481 KB | ~2.7s |

## Documentation

- [`CLAUDE.md`](CLAUDE.md) — full developer guide, conventions, troubleshooting
- [`PRD.md`](PRD.md) — product requirements, architecture, success metrics

## License

MIT
