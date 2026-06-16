# Product Requirements Document — DMoE Pipeline

**Version:** 0.1.0  
**Date:** 2026-06-16  
**Paper:** Decoupled Mixture-of-Experts for Parametric Knowledge Injection (arXiv:2606.14243, Tsinghua University)  
**Status:** MVP

---

## 1. Problema

I sistemi RAG tradizionali presentano tre limitazioni strutturali:

- **Context Penalty**: ogni documento recuperato occupa finestra di contesto, aumentando la latenza linearmente
- **KV-Cache Invalidation**: aggiungere testo al prompt invalida la cache KV, forzando il ricalcolo dell'intera sequenza
- **GPU VRAM scaling**: sistemi RAG richiedono 26+ GB di VRAM per mantenere indici vettoriali + modello in memoria

L'obiettivo è implementare un'alternativa parametrica che mantenga la conoscenza su disco (SSD) invece che in contesto.

---

## 2. Soluzione — DMoE

**Decoupled Mixture-of-Experts (DMoE)** inietta conoscenza direttamente nei parametri del modello tramite
adattatori LoRA leggeri (~481 KB/documento), applicati **esclusivamente all'ultimo FFN layer** del transformer.

### Flusso a runtime

```
Query utente
    │
    ▼
[Forward pass base model]
    │
    ├─► Token Uncertainty (Shannon entropy) > τ = 2.0?
    │        │ NO → genera token normalmente
    │        │ YES ↓
    │        ▼
    │   [BM25 Router] → seleziona top-k=3 expert dal disco
    │        │
    │        ▼
    │   [Parameter Composition] → Σ ΔΘᵢ applicato solo al layer finale
    │
    ▼
Output token con conoscenza iniettata parametricamente
```

### Proprietà chiave

| Proprietà | Valore |
|-----------|--------|
| Dimensione expert | ~481 KB per documento |
| Layer target | Solo ultimo FFN layer |
| LoRA rank | 4 |
| LoRA alpha | 16 |
| Top-k expert attivi | 3 |
| Soglia entropia (τ) | 2.0 |
| VRAM (Llama-3.2-1B) | 7.2 GB |
| VRAM (Qwen2.5-1.5B) | 8.3 GB |

---

## 3. Scope MVP

### In scope

- [x] Pipeline augmentazione documento → QA dataset via Ollama locale
- [x] Training LoRA expert per singolo documento (`--lora-layers 1`)
- [x] Script batch per intera cartella documenti
- [x] BM25 router con indice persistente (pickle)
- [x] Inference engine con Token Uncertainty trigger
- [x] Hotswap adapter a runtime (mlx-lm)
- [x] CLI entry-points via `pyproject.toml` scripts
- [x] Test unitari per router e augmentazione
- [x] Benchmark su HotpotQA (EM + F1)

### Out of scope (v0.1)

- GUI o API HTTP
- Integrazione con vector database (Qdrant / ChromaDB)
- Multi-GPU / distributed training
- PDF parsing (solo .txt per ora)
- Autenticazione e multi-utente

---

## 4. Architettura dei Componenti

```
src/dmoe/
├── augment.py      # Documento → QA dataset (via Ollama)
├── train.py        # Wrapper mlx_lm.lora con configurazione DMoE
├── router.py       # BM25 index: build, persist, query
├── inference.py    # Engine DMoE: TU trigger + hotswap
└── benchmark.py    # Valutazione EM/F1 su HotpotQA
```

### Flusso dati

```
docs/*.txt
    │
    ▼ dmoe-augment
experts/<doc_id>/
    ├── data/train.jsonl
    ├── data/valid.jsonl
    ├── adapter/adapters.safetensors   ← ~481KB
    └── metadata.json                 ← doc_text + qa_text per BM25
    │
    ▼ dmoe-build-idx
experts/bm25_index.pkl
    │
    ▼ dmoe-infer
risposta con conoscenza parametrica
```

---

## 5. Requisiti Non Funzionali

### Hardware target

- Apple Silicon M4 Pro, 24 GB RAM unificata
- macOS Sequoia 15+
- Storage: ~500 MB per 1000 documenti (solo adapter .safetensors)

### Performance target

| Metrica | Target |
|---------|--------|
| Training per expert | < 5 min (100 iter, batch=2) |
| Latenza query (top-k=3) | < 3s |
| VRAM peak inference | < 10 GB |
| BM25 index build (1000 doc) | < 10s |

---

## 6. Dataset di Augmentazione — Specifiche

Ogni documento viene trasformato in istanze instruction-following **prima** del training LoRA.
Il documento grezzo **non** viene passato direttamente al modello.

### Struttura JSONL (formato chat mlx-lm)

```json
{
  "messages": [
    {"role": "system",    "content": "You are a knowledgeable assistant. Use the following document...\n\n<rewrite>"},
    {"role": "user",      "content": "<domanda estratta>"},
    {"role": "assistant", "content": "<risposta estratta>"}
  ]
}
```

### Split train/valid

- Train: 80% delle coppie QA
- Valid: 20% (minimo 1 record)

---

## 7. Metriche di Successo

| Metrica | Baseline (RAG) | Target (DMoE) |
|---------|---------------|---------------|
| HotpotQA EM | ~38% | ≥ 42% |
| HotpotQA F1 | ~51% | ≥ 55% |
| Latenza media query | ~4.2s | ≤ 2.7s |
| VRAM durante inference | ~26 GB | ≤ 10 GB |

Valori di riferimento dal paper (arXiv:2606.14243).

---

## 8. Dipendenze Esterne

| Dipendenza | Versione | Scopo |
|------------|----------|-------|
| mlx-lm | ≥ 0.22.0 | Training LoRA su Apple Silicon |
| mlx | ≥ 0.26.0 | Compute framework GPU unificata |
| peft | ≥ 0.14.0 | Hotswap adapter a runtime |
| rank-bm25 | ≥ 0.2.2 | BM25 routing degli expert |
| Ollama | ≥ 0.4.0 | LLM locale per augmentazione |
| Qwen2.5-1.5B-Instruct-4bit | — | Modello base (MLX Community) |

---

## 9. Rischi e Mitigazioni

| Rischio | Probabilità | Impatto | Mitigazione |
|---------|-------------|---------|-------------|
| QA pairs malformate (JSON parse error) | Alta | Medio | Regex fallback su output Ollama |
| OOM durante training batch | Media | Alto | batch-size=2, --batch-size 1 fallback |
| Adapter incompatibili tra versioni mlx-lm | Bassa | Alto | Pin versioni in uv.lock |
| BM25 router non trova expert pertinente | Media | Medio | Fallback a modello base senza adapter |
