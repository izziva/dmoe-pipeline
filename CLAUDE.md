# CLAUDE.md — DMoE Pipeline

Questo file descrive il progetto, le convenzioni e le istruzioni operative.
Leggilo per intero prima di modificare qualsiasi file.

---

## Cos'è questo progetto

Implementazione locale di **Decoupled Mixture-of-Experts (DMoE)** per l'iniezione
parametrica di conoscenza nei modelli linguistici, basata su:

> *Decoupled Mixture-of-Experts for Parametric Knowledge Injection*  
> Baoqing Yue et al., Tsinghua University — arXiv:2606.14243 (2026)

**Idea chiave**: invece di passare documenti nel contesto (RAG), ogni documento genera
un LoRA adapter leggero (~481 KB) addestrato **solo sull'ultimo FFN layer** del modello.
A runtime, un router BM25 seleziona i top-k=3 adapter più rilevanti, attivati dall'entropia
di Shannon (Token Uncertainty > τ=2.0) sul token corrente.

---

## Stack tecnologico

| Layer | Tool |
|-------|------|
| Packaging & env | `uv` — NON usare pip, conda o poetry |
| Compute / training | `mlx-lm`, `mlx` (Apple Silicon GPU unificata) |
| Hotswap adapter | `peft` (HuggingFace) |
| BM25 routing | `rank-bm25` |
| Augmentazione LLM | Ollama locale (qwen2.5:1.5b) |
| Linting | `ruff` |
| Testing | `pytest` + `pytest-cov` |

---

## Setup iniziale (obbligatorio)

```bash
# 1. Installa uv se non presente
brew install uv

# 2. Sincronizza l'environment (usa uv.lock — riproducibile)
uv sync --all-groups

# 3. Verifica GPU Apple Silicon
uv run python -c "import mlx.core as mx; print(mx.default_device())"
# atteso: Device(gpu, 0)

# 4. Scarica il modello base
uv run huggingface-cli download \
    mlx-community/Qwen2.5-1.5B-Instruct-4bit \
    --local-dir ./models/qwen2.5-1.5b-4bit

# 5. Avvia Ollama per l'augmentazione
ollama pull qwen2.5:1.5b
ollama serve  # terminale separato
```

---

## Workflow di sviluppo

### Regola fondamentale: usa sempre `uv run`

```bash
# CORRETTO
uv run dmoe-augment --help
uv run pytest tests/
uv run ruff check src/

# SBAGLIATO — non attiva l'environment corretto
python src/dmoe/augment.py
```

### Gestione dipendenze

```bash
uv add <package>                   # runtime
uv add --group dev <package>       # solo sviluppo
uv add --group benchmark <package>
```

Commit sempre `uv.lock` insieme alle modifiche a `pyproject.toml`.

---

## Pipeline completa — step by step

### Step 1: Augmentazione documento

```bash
uv run dmoe-augment \
    --doc ./docs/mio_documento.txt \
    --out ./experts/doc_001/data \
    --model qwen2.5:1.5b
```

Genera in `experts/doc_001/`:
- `data/train.jsonl` — istanze chat JSONL (80%)
- `data/valid.jsonl` — validation set (20%)
- `metadata.json` — testo + QA per BM25 router

### Step 2: Training expert LoRA

```bash
uv run dmoe-train \
    --doc-id doc_001 \
    --experts-dir ./experts \
    --base-model ./models/qwen2.5-1.5b-4bit
```

Output: `experts/doc_001/adapter/adapters.safetensors` (~481 KB)  
Internamente usa `--lora-layers 1` (solo ultimo FFN layer — core del paper).

### Step 3: Build BM25 index

```bash
uv run dmoe-build-idx \
    --experts-dir ./experts \
    --out ./experts/bm25_index.pkl
```

### Step 4: Inferenza

```bash
# Query singola
uv run dmoe-infer \
    --base-model ./models/qwen2.5-1.5b-4bit \
    --experts-dir ./experts \
    --index ./experts/bm25_index.pkl \
    --query "La tua domanda"

# REPL interattiva
uv run dmoe-infer \
    --base-model ./models/qwen2.5-1.5b-4bit \
    --experts-dir ./experts \
    --index ./experts/bm25_index.pkl \
    --interactive
```

### Step 5: Benchmark (opzionale)

```bash
uv sync --group benchmark
uv run dmoe-bench \
    --base-model ./models/qwen2.5-1.5b-4bit \
    --experts-dir ./experts \
    --dataset hotpotqa \
    --split "validation[:200]"
```

---

## Parametri chiave del paper (non modificare senza motivo)

| Parametro | Valore | Dove |
|-----------|--------|------|
| `--lora-layers` | `1` | Training: solo ultimo FFN layer |
| `--rank` | `4` | LoRA rank |
| `--lora-scale` | `16` | LoRA alpha |
| `--iters` | `100` | Iterazioni per expert |
| `--batch-size` | `2` | Sicuro su M4 Pro 24GB |
| `--learning-rate` | `1e-5` | LR training |
| `tau` | `2.0` | Soglia Token Uncertainty |
| `top_k` | `3` | Expert attivati per query |

---

## Struttura del repository

```
dmoe-pipeline/
├── CLAUDE.md               <- sei qui
├── PRD.md                  <- requisiti e specifiche
├── README.md               <- quickstart pubblico
├── pyproject.toml          <- dipendenze e script entry-points
├── uv.lock                 <- lock file (NON modificare a mano)
├── .python-version         <- "3.11"
├── .gitignore
├── .github/
│   └── workflows/
│       └── ci.yml          <- lint + test su push
├── models/                 <- modelli scaricati (gitignored)
├── docs/                   <- documenti sorgente .txt
├── experts/                <- adapter + indice BM25 (gitignored)
└── src/
    └── dmoe/
        ├── __init__.py
        ├── augment.py      <- documento → QA dataset
        ├── train.py        <- wrapper mlx_lm.lora
        ├── router.py       <- BM25 build + query
        ├── inference.py    <- DMoE engine
        └── benchmark.py   <- valutazione EM/F1
```

---

## Convenzioni di codice

- **Type hints** su tutte le funzioni pubbliche
- **Docstring** Google-style per ogni modulo e funzione pubblica
- **`pathlib.Path`** per i path — mai `os.path`
- **`rich`** per output CLI (Console, progress bar)
- **Niente `print()` nudi** nei moduli — usa `rich.console.Console`
- Nessun secret o path assoluto hardcodato — usa argparse / env vars
- File di modello e adapter in `.gitignore`

---

## Testing

```bash
uv run pytest                                     # tutti i test
uv run pytest tests/test_router.py -v            # specifico
uv run pytest --cov=dmoe --cov-report=html       # con coverage
```

I test usano mock per Ollama — non richiedono il modello scaricato.

---

## Linting

```bash
uv run ruff check src/ tests/      # check
uv run ruff check --fix src/       # autofix
uv run ruff format src/ tests/     # formatting
```

---

## Variabili d'ambiente

```bash
export DMOE_OLLAMA_MODEL="llama3.2:1b"              # modello augmentazione alternativo
export DMOE_BASE_MODEL="./models/llama3.2-1b-4bit" # modello base alternativo
export DMOE_FORCE_EXPERTS=1                         # disabilita TU trigger (debug)
```

---

## Troubleshooting

**`mx.default_device()` non è `Device(gpu, 0)`**  
Assicurati di usare `uv run` e non Python di sistema.

**Ollama non risponde durante augmentazione**
```bash
ollama serve   # terminale separato
ollama ps      # verifica modello caricato
```

**JSON parse error durante augmentazione**  
Lo script estrae automaticamente il primo `[...]` dall'output. Se fallisce, prova `--model llama3.2:1b`.

**OOM durante training**  
Modifica `--batch-size 1` in `dmoe-train` o direttamente in `train.py`.
