# CoRe Memory (Graph-Structured Memory)

This repository contains the **CoRe external memory** research codebase: a non-LLM extraction pipeline that writes **structured memory nodes** into a **typed graph** (semantic, temporal, co-usage, and optional entity edges), then retrieves evidence with hybrid dense/BM25 seeding, optional graph expansion, and fused reranking.

## Prerequisites

- **Python 3.10** (see `pyproject.toml`)
- **[uv](https://docs.astral.sh/uv/)** recommended for installs and `uv run …`
- An **LLM API key** for evaluation scripts (OpenAI-compatible endpoint)
- Optional: **spaCy English model** `en_core_web_sm` for entity edges on LoCoMo-style runs  
  `uv run python -m spacy download en_core_web_sm`

## Quick start

From the repository root (`CoRe_Mem/`):

```bash
# Install dependencies (editable package + dev tools)
uv sync --extra dev

# Configure the LLM client (copy and edit)
cp .env.example .env
# Set LLM_API_KEY and, if needed, LLM_BASE_URL / LLM_MODEL in .env
```

Run the smallest automated checks:

```bash
# Unit / smoke test for Add + Search (no remote LLM required for the smoke test path)
uv run pytest tests/graph_mem/test_smoke_pipeline.py -q
```

## Quick experiments

All commands assume you run them from **`CoRe_Mem/`**.

### 1) Minimal graph pipeline smoke (`experiments/perma/smoke_graph_mvp.py`)

Lightweight end-to-end write + search using a toy embedder unless you force BGE-M3:

```bash
uv run python experiments/perma/smoke_graph_mvp.py
# Optional real embeddings:
#   USE_BGE_M3=1 uv run python experiments/perma/smoke_graph_mvp.py
```

### 2) LoCoMo evaluation (`data/locomo/locomo10.json` included)

Compares **graph-full**, **semantic-only**, and **full-context** baselines; writes results under `outputs/locomo_eval/<timestamp>/`.

```bash
# Fast sanity run (2 conversations, 20 QA pairs each)
uv run python experiments/locomo/eval_locomo_graph.py --conv-limit 2 --qa-limit 20

# Skip expensive full-context baseline
uv run python experiments/locomo/eval_locomo_graph.py --conv-limit 2 --qa-limit 20 --no-full-context

# After a run, optional analysis helper
uv run python experiments/locomo/analyze_results.py --mode graph_full
```

### 3) PERMA graph vs semantic (`experiments/perma/eval_perma_graph_vs_semantic.py`)

PERMA task/eval JSON is **downloaded from Hugging Face** (`ustclsc/PERMA`) on first use; you need network access and a configured LLM in `.env`.

```bash
uv run python experiments/perma/eval_perma_graph_vs_semantic.py --user-id user108 --variant c --limit 5
```

### 4) Optional: λ_core sweep or Phase-3 ablations

```bash
uv run python experiments/perma/tune_lambda_core.py --help
uv run python experiments/perma/run_phase3_ablations.py --help
```

### 5) Optional: mini E2E test with a real LLM

Requires `LLM_API_KEY` set; may hit live APIs:

```bash
uv run pytest tests/graph_mem/test_e2e_perma_mini.py -q
```

## Repository layout

```
CoRe_Mem/
├── src/graph_mem/          # Core library: schemas, extractor, embedder, graph store,
│                           # add/search pipelines, ranking, LoCoMo/PERMA loaders & eval helpers
├── experiments/
│   ├── locomo/             # LoCoMo scripts (eval, analysis utilities)
│   └── perma/              # PERMA scripts (graph vs semantic, ablations, smoke MVP)
├── tests/graph_mem/        # Pytest: smoke pipeline + optional mini e2e
├── data/locomo/            # Bundled LoCoMo JSON (e.g. locomo10.json)
├── outputs/                # Timestamped run outputs (gitignored placeholders may exist)
├── docs/                   # Design notes, requirements, workflow, status
├── .agent-os/              # Agent-oriented project metadata (optional reading)
├── pyproject.toml          # Package metadata and dependencies
├── requirements.txt        # Pinned-style deps for non-uv workflows
└── .env.example            # Example environment variables for LLM access
```

## Documentation

Higher-level architecture and workflow live under **`docs/`** (see `docs/structure.md` and `docs/workflow.md`).
