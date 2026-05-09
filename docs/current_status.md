# Current Status

## Decisions locked in
- Direction: external memory upgrades from “flat retrieval” to **graph-structured memory**.  
- Graph: node = memory; edges include semantic / temporal / co-usage (plus entity in implementation).  
- Layering: keep core/residual but differentiate via graph centrality and activity.  
- Retrieval weighting: **`PageRank / centrality prior + semantic score`** fusion (chosen).  
- Extractor: no LLM; prefer **rule backbone + lightweight model fallback**.  
- Benchmark order (**updated**): **`LoCoMo` as primary** (long multi-session dialogue, matches “memory grows with dialogue then answer”); **`PERMA`** kept for baseline/regression (task-context often aligns tightly with questions—not equivalent to full long-context streams); **`PersonaMem`** migration validation after things stabilize.

## Design summary
- Add: extract structured events → near-duplicate detection → high similarity merge, low similarity new node.  
- Connect: maintain semantic and temporal edges on new/updated nodes.  
- Search: query retrieves seeds → graph expansion → fused rerank.  
- Post-query: same-query co-hits with no prior edge update co-usage.

## Main risks
- Temporal conflict handling is hybrid MVP; `context_split` / `hard_conflict` still need refinement.  
- Co-usage edges can propagate noise—needs suppression.  
- Excessive core prior → “hub nodes dominate”.  
- Rule extractor MVP needs iteration on PERMA for coverage vs false positives.  
- Fusion weights lack a fixed tuning protocol—reproducibility risk.

## Next steps (short term)
1. **LoCoMo integrated**: pin data files and HF dataset names; loading + incremental dialogue ingest + QA eval (reuse `graph_mem` pipeline).  
2. Keep PERMA / Phase 3 scripts for regression; primary conclusions follow LoCoMo long-dialogue setup.  
3. Small samples are directional only; failure attribution waits for larger N.

## Completed this round (Graph MVP skeleton)
- New `src/graph_mem/` module:  
  - `schemas.py`: Node/Edge/StructuredRecord schema  
  - `extractor.py`: RuleExtractor (non-LLM)  
  - `formatter.py`: structured text normalization  
  - `embedder.py`: BGE-M3 embedder wrapper  
  - `graph_store.py`: NetworkX-backed graph store  
  - `add_pipeline.py`: add path (including hybrid revision)  
  - `ranking.py`: fusion scoring  
  - `search_pipeline.py`: seed + expand + rerank + co-usage updates  
- `train/perma/smoke_graph_mvp.py` runs end-to-end.  
- `tests/graph_mem/test_smoke_pipeline.py` passes (1 passed).  
- Real PERMA comparison script `train/perma/eval_perma_graph_vs_semantic.py`:  
  - Build context memory from `tasks/{user}/input_data_c.json`  
  - Load questions/options/gold from `evaluation/{user}/meta/overall/*.json`  
  - Compare `semantic-only` vs `graph-full`, unified LLM multiple-choice judging  
- `src/graph_mem/perma_data.py` (PERMA loading) and `src/graph_mem/eval_utils.py` (LLM eval helpers).  
- `src/graph_mem/perma_eval.py` (shared eval core for scripts and tests).  
- `tests/graph_mem/test_e2e_perma_mini.py`:  
  - Real PERMA mini e2e gate (`user108`, `limit=2`)  
  - Auto-skip without `LLM_API_KEY` to avoid blocking local runs  
  - With key: verifies graph vs semantic paths execute  
- Dependencies updated and installed in project `.venv` (requirements / pyproject / uv.lock synced).

## PERMA initial runs (real data)
- Commands:  
  - `python train/perma/eval_perma_graph_vs_semantic.py --user-id user108 --variant c --limit 2`  
  - `python train/perma/eval_perma_graph_vs_semantic.py --user-id user108 --variant c --limit 5`  
- Results (`user108`, clean variant):  
  - `limit=2`: graph `0.50` vs semantic `0.00`, delta `+0.50`  
  - `limit=5`: graph `0.60` vs semantic `0.20`, delta `+0.40`  
- Output dirs:  
  - `outputs/perma_eval/20260426_220502`  
  - `outputs/perma_eval/20260426_220741`  

## PERMA larger sample (new)
- `user108`, `variant=c`, `limit=20`  
  - graph: `0.65`  
  - semantic-only: `0.60`  
  - delta: `+0.05`  
  - output: `outputs/perma_eval/20260426_221531`  
- `user109`, `variant=c`, `limit=10`  
  - graph: `0.60`  
  - semantic-only: `0.90`  
  - delta: `-0.30`  
  - output: `outputs/perma_eval/20260426_223408`  

## Current takeaway (interim)
- Results moved from “stable positive gain” to **high variance across users**:  
  - Small positive gain on `user108`;  
  - Clear negative gain on `user109`.  
- Cannot claim graph-full is globally better than semantic-only yet.  
- Next: failure analysis and retrieval diagnostics before scaling eval.

## How to run (current entry points)
- Small real comparison:  
  - `python train/perma/eval_perma_graph_vs_semantic.py --user-id user108 --variant c --limit 5`  
- Mini e2e test:  
  - `python -m pytest tests/graph_mem/test_e2e_perma_mini.py -q`  

## Debug export (new)
- PERMA eval supports per-sample debug: `debug_samples.jsonl`  
- Each record may include:  
  - `seeds` (initial semantic recall)  
  - `expanded` (candidates after graph expansion)  
  - `score_breakdown` (semantic/centrality/edge/temporal/final)  
  - `graph_top_evidence` and `semantic_top_evidence`  
  - `graph_prompt_preview` and `semantic_prompt_preview`  
  - `pred_graph` / `pred_semantic` / `gold`  
- Example path:  
  - `outputs/perma_eval/20260426_225611/debug_samples.jsonl`  

## Phase 2 progress (new)
- Core/residual upgraded from “PageRank threshold only” to joint `core_score`:  
  - `core_score = 0.50 * pagerank + 0.30 * in_degree + 0.20 * retrieve_count` (min-max normalized then fused)  
- Rerank uses `core_score` as centrality prior; `lambda_core` multiplicatively boosts core nodes.  
- Each retrieval updates `retrieve_count` on hit nodes for dynamic core differentiation.  
- Debug export includes `core_score` and `retrieve_count` for prior bias diagnosis.

## Verification this round (new)
- Tests:  
  - `python -m pytest tests/graph_mem -q` → `1 passed, 1 skipped`  
- Quick PERMA:  
  - `python train/perma/eval_perma_graph_vs_semantic.py --user-id user108 --variant c --limit 5`  
  - graph: `0.80` vs semantic-only: `0.20`, delta: `+0.60`  
  - output: `outputs/perma_eval/20260426_230254`  

## PERMA multi-user quick check (new)
- Config: `variant=c`, `limit=5`, all 10 users  
- Summary: `outputs/perma_eval/multi_user_limit5_summary.json`  
- Distribution (delta = graph − semantic):  
  - Positive: `3/10` (`user108 +0.4`, `user123 +0.4`, `user914 +0.2`)  
  - Tie: `4/10` (`user112`, `user354`, `user419`, `user507`)  
  - Negative: `3/10` (`user109 -0.4`, `user1377 -0.4`, `user334 -0.8`)  
- Aggregate:  
  - `mean_delta = -0.06`  
  - `median_delta = 0.0`  
- Interim read: not a uniform global regression, but a **clear subset of users with negative graph gains**—needs diagnosis, not necessarily a full rewrite yet.

## Phase 3 implementation (this round)
- **Edge expansion ablation**: `SearchConfig` adds `expand_use_semantic` / `expand_use_temporal` / `expand_use_co_usage`; `PermaEvalConfig.graph_edge_mode` injected via `ablation_presets.search_config_for_edge_mode`.  
- **Merge ablation**: `AddConfig.merge_strategy`: `hybrid` (original) vs `latest_wins` (high sim overwrites latest node).  
- **Co-usage control**: `MemoryGraphStore.register_co_usage_between` solidifies edges by joint hit count; `decay_co_usage_edges` for decay/prune; wired to `PermaEvalConfig` / CLI.  
- **Temporal probing**: `temporal_probe_thirds=True` writes `temporal_probe_thirds` to `summary.json` (early/mid/late by eval sample order, thirds).  
- **Batch entry**: `train/perma/run_phase3_ablations.py` → `outputs/perma_eval/phase3_ablation_<ts>/phase3_ablation_summary.json` (per-user `edge_<mode>/` when multi-user; top-level `aggregate_by_graph_edge_mode` is cross-user mean).  
- **Suggested pilot**: 6 users × 15 questions × 4 edge modes, `merge=hybrid`: `python train/perma/run_phase3_ablations.py --limit 15 --use-default-six-users`  
- **Dependencies**: no new Python packages this round; `requirements.txt` / `pyproject.toml` / `uv.lock` unchanged.

## References
- Design follows GraphRAG’s “structure + graph relations” idea; goal is clear research code and decoupled modules—not copying its repo layout.  
- Reference: [microsoft/graphrag](https://github.com/microsoft/graphrag/tree/main)

## LoCoMo integration (this round, Phase 5 kickoff)

### Data
- File: `data/locomo/locomo10.json` (10 conversations, ~419 turns / ~19 sessions each)  
- QA: 1540 pairs (cat1=282, cat2=321, cat3=96, cat4=841; cat5 adversarial excluded by default)

### vs PERMA — differences and handling
- **Two speakers**: both sides ingested; text prefix `[Speaker]` for role  
- **Session order**: `time_index = (session_idx-1) * 10_000 + turn_idx` for session-aware ordering  
- **Free-form answers**: LLM generates text; default `soft_match` scoring (no extra calls); `--use-llm-judge` for LLM judge  
- **Extractor / AddPipeline / SearchPipeline**: reused as-is; interfaces compatible  

### New files
- `src/graph_mem/locomo_data.py`: LoCoMoTurn / LoCoMoQA / LoCoMoConversation / load_locomo / turn_time_index  
- `experiments/locomo/eval_locomo_graph.py`: main eval (`build_graph_from_conversation` / `run_locomo_eval` / CLI)

### Commands

```bash
# Quick check (2 conversations, 20 QA each)
uv run python experiments/locomo/eval_locomo_graph.py --conv-limit 2 --qa-limit 20

# Only cat1+2 (single-hop + temporal)
uv run python experiments/locomo/eval_locomo_graph.py --categories 1,2

# Full (10 conversations × 1540 QA)
uv run python experiments/locomo/eval_locomo_graph.py
```

Output: `outputs/locomo_eval/<timestamp>/`

## LoCoMo full-context baseline (new)
- `eval_locomo_graph.py` adds a third baseline: **full-context** (all turns formatted by session in one LLM context).  
- Largest locomo10 dialogue ~22k tokens—fits in gpt-4o-mini 128k window without truncation.  
- If over `--context-budget-chars` (default 400k chars ≈ 100k tokens), truncate from the start and keep the most recent turns.  
- Compare: graph-full vs semantic-only vs full-context.  
- Quick run (2 conv × 20 QA, all three baselines):  
  `uv run python experiments/locomo/eval_locomo_graph.py --conv-limit 2 --qa-limit 20`  
- Skip full-context to save API cost:  
  `uv run python experiments/locomo/eval_locomo_graph.py --no-full-context`

## LoCoMo debug and fixes (this round)

### Bugs found (fixed)

**Bug 1: embedding used `structured_text` (metadata prefix shrinks similarity space)**  
- Symptom: all node sims clustered 0.64–0.68; gold not separable in top-k.  
- Fix: `add_pipeline.py` embeds `record.text` (raw turn text).

**Bug 2: turn text lacked session date (relative time unresolvable)**  
- Symptom: cat2 temporal questions all wrong (“yesterday” without date).  
- Fix: `eval_locomo_graph.py` writes `[Speaker, Session N, date] {turn}` when building the graph.

**Bug 3: rerank core prior systematically down-ranked early-session nodes**  
- Symptom: semantic rank=1 node (D2:1) overtaken in expanded set by high-centrality nodes from sessions 6–7, dropped from top-8.  
- Cause: later nodes get more semantic edges → higher `core_score` → multiplicative `lambda_core` pushes them above seed facts.  
- Fix: LoCoMo eval uses semantic-dominant `RankingConfig` (alpha=0.90, lambda_core=0.0, seed_topk=20).

### After fixes (2 conv × 20 QA)
| Baseline | Before | After |
|---|---|---|
| graph-full | 2.5% | **32.5%** |
| semantic-only | 7.5% | 35.0% |
| full-context | 42.5% | 37.5% |

cat2 (temporal): graph/semantic reach 47.8%, above full-context 43.5%

---

## BM25 hybrid retrieval (this round)

### What shipped
- **Dependency**: `rank-bm25>=0.2.2`, synced to `requirements.txt` / `pyproject.toml` / `uv.lock`  
- **`src/graph_mem/graph_store.py`**:  
  - `_tokenize(text)` helper (lower + strip punctuation + whitespace tokenization)  
  - Slots: `_bm25_index`, `_bm25_node_ids`, `_bm25_dirty`  
  - `build_bm25_index()` lazy rebuild on first use or when dirty  
  - `bm25_search(query, top_k)` → `list[(node_id, score)]`  
  - `add_node()` sets `_bm25_dirty=True`  
- **`src/graph_mem/search_pipeline.py`**:  
  - `SearchConfig`: `use_bm25`, `bm25_topk`, `rrf_k`  
  - `_seed_retrieve(qvec, query_str)` dual-path + RRF  
  - `search()` / `search_with_debug()` pass `query_str`  
  - `use_bm25` **default off** (PERMA unchanged); LoCoMo eval enables explicitly  

### LoCoMo eval
`experiments/locomo/eval_locomo_graph.py`: `SearchConfig(seed_topk=20, use_bm25=True, bm25_topk=20)`

### Check (2 conv × 20 QA = 40 samples)
| Baseline | This run (BM25 on) | Previous (BM25 off) |
|---|---|---|
| graph-full | 30.0% | 32.5% |
| semantic-only | 37.5% | 35.0% |
| full-context | 40.0% | 37.5% |

> N=40: differences within noise (±5pp); no strong BM25 up/down conclusion yet.  
> Next: full 10-conv (1540 QA) run. cat1 (single-hop factual) is the main target for BM25.

### Next
- Full scale (`--conv-limit 10`); cat1 single-hop matters most for BM25  
- Sensitivity on `rrf_k` and `bm25_topk`

---

## P0 adaptive expansion + P1 entity edges (this round)

### Context
graph-full (30%) below semantic-only (37.5%):  
1. **Expansion noise on cat1**: semantic already hits gold; neighbors dilute evidence.  
2. **Coarse nodes (turn-level)**: only temporal/semantic edges—missing cross-entity links.

### P0: adaptive expansion
- `SearchConfig.adaptive_expand_threshold: float = 0.0` (default off)  
- If top-1 seed sim ≥ threshold, skip expansion (new `_maybe_expand()`)  
- `search_with_debug` adds `skipped_expand`  
- LoCoMo eval: `adaptive_expand_threshold=0.80`

### P1: Entity edges
**Deps**: `spacy>=3.7` + `en_core_web_sm` (~12MB)  
**Install**: `uv pip install spacy>=3.7 && uv run python -m spacy download en_core_web_sm`

**Schema**:  
- `StructuredRecord.entity_mentions: list[str]`  
- `EdgeType` adds `"entity"`

**extractor.py**:  
- `_get_nlp()` lazy spaCy (empty list if spaCy unavailable)  
- `extract_entities(text)` (PERSON/GPE/LOC/ORG/EVENT, length ≥3)  
- `RuleExtractor.extract_from_turn()` fills `entity_mentions`

**add_pipeline.py**:  
- `AddConfig.entity_edge_max_freq=0.35` (filter entities in >35% of nodes); `entity_edge_max_per_node=20`  
- `AddPipeline.build_entity_edges(now_ts)`: group by entity, bidirectional `entity` edges; `math.ceil` for max_count on small corpora; returns directed edge count created  

**search_pipeline.py**:  
- `SearchConfig.expand_use_entity=True` includes entity edges in `_expand()`

**eval_locomo_graph.py**:  
- `build_graph_from_conversation()` returns `(store, build_stats)` with n_nodes / n_edges / n_entity_edges  
- After build: `add.build_entity_edges(now_ts=0)`  
- Fixed scoping: `rank_cfg` / `search_cfg` / `add_cfg` no longer trapped inside `if conv_limit > 0:`  
- SearchConfig: `adaptive_expand_threshold=0.80`, `expand_use_entity=True`

### Check (2 conv × 20 QA)
| Baseline | Before P0+P1 | **After P0+P1** |
|---|---|---|
| graph-full | 30.0% | **35.0%** ↑5pp |
| semantic-only | 37.5% | 40.0% |
| full-context | 40.0% | 35.0% |
| delta graph−sem | -7.5pp | **-5.0pp** |

cat2: graph 47.8% vs semantic 52.2%—gap narrows.  
Still N=40; full 1540 QA needed.

### Next
- Full run `--conv-limit 10`  
- Entity edges may help cat3 cross-session reasoning (few cat3 samples today)

### LoCoMo full primary run + richer debug (this round)
- `search_with_debug`: `top_seed_semantic_sim`, `n_expanded`, `n_candidates_ranked`  
- Per-QA `retrieval_diagnostics`: map `evidence_refs` (e.g. D1:3) to nodes; report global semantic rank; in seed/expanded/final top-k; whether gold turn text appears in graph evidence  
- `summary.json`: `retrieval_diagnostics` aggregates + for **graph wrong but full-context right**, counts gold inside/outside top-k (retrieval vs LLM failure)  
- Post-run: `uv run python experiments/locomo/analyze_results.py --mode graph_full`

---

## Full-run attribution + recall/prompt tuning (this round)

### Full primary result (20260427_141126, 1540 QA)
| Method | Accuracy |
|------|--------|
| full-context | 42.86% |
| semantic-only | 32.86% |
| graph-full | 28.51% |

### Pipeline dropout (gold node trace, `_analyze_dropout.py`)
| Stage | Count | Share |
|------|------|------|
| Node match failure | 9 | 0.6% |
| **dense/BM25 miss (not in seed top-20)** | **407** | **26.4%** |
| P0 adaptive expand cutoff | 0 | 0% |
| **In expanded set but cut by rerank/top-8** | **520** | **33.8%** |
| Reaches final top-k | 835 | 54.2% |
| In top-k but graph still wrong | 424 | 50.8% |

- cat1 wrong despite top-k: 79% (LLM reasoning stands out)  
- cat4 wrong despite top-k: 36% (mostly recall)  
- P0 adaptive expand: no negative cutoff observed  

### Changes from attribution

**1. Wider recall** (`eval_locomo_graph.py`)  
- `seed_topk`: 20 → **40** (gold semantic rank median ~22)  
- `bm25_topk`: 20 → **40**  
- `final_topn_evidence`: 8 → **12** (33.8% lost at top-8)

**2. Prompt upgrade** (`_answer_prompt` + `_full_context_prompt`)  
- Allow cross-snippet reasoning (multi-hop)  
- Temporal inference using `[Speaker, Session N, date]`  
- Commonsense bridging for cat3  
- Answer with conclusion only  
- Numbered evidence `[1]`, `[2]` …

**3. Full validation run2**: done; see `20260427_170019/summary.json`

### Run2 (1540 QA)
| Method | run1 | run2 (+recall+prompt) | Δ |
|------|------|------|------|
| graph-full | 28.51% | **29.74%** | +1.23pp |
| semantic-only | 32.86% | **31.82%** | -1.04pp |
| full-context | 42.86% | **44.03%** | +1.17pp |
| delta graph−sem | -4.35pp | **-2.08pp** | gap halved |

Gold in final top-k: 835/1540 (54.2%) → 905/1540 (58.8%) (+70 questions)

### Run3: stopped (rerank degradation is the deeper issue)

**Gold rank from run2 debug:**  
- Semantic rank: mean 26.3, **median 4** (bimodal)  
- After rerank: mean 18.6, median 7  
- Rerank drops gold from semantic k≤12 in **73%** of cases to final top-12 in **63%** (−10pp)  
- Fusion centrality/edge_evidence hurts fringe factual nodes (early sessions, single mention, few edges)

**Root cause:** seeds and expand candidates compete under one formula; expand nodes are weak semantically but carry multi-hop bridges.

### Run4 (split-slot rerank)

**Changes:**  
- `seed_evidence_slots=10`: seeds by semantic sim  
- `expand_evidence_slots=6`: expand-only by edge proximity to seeds  
- No direct competition between the two groups  
- `final_topn_evidence=16` (10+6)

**Smoke (2 conv × 20 QA):**
| Method | Accuracy |
|------|------|
| graph-full | **42.5%** |
| semantic-only | 40.0% |
| full-context | 45.0% |

First small-sample graph > semantic (+2.5pp); full run pending.

### Run4 full (1540 QA, 20260427_194910)

| Method | run2 | run4 (split-slot) | Δ |
|------|------|------|------|
| graph-full | 29.74% | **34.81%** | **+5.07pp** |
| semantic-only | 31.82% | **35.91%** | +4.09pp |
| full-context | 44.03% | **43.96%** | ≈flat |
| delta graph−sem | -2.08pp | **-1.10pp** | gap shrinks |
| gold in top-k | 58.8% | **67.47%** | **+8.67pp** |

Split-slot: gold recall 58.8% → 67.47%; graph overall +5pp+.

**By category (official ID mapping fixed: cat1=multi-hop, cat4=single-hop):**

| Cat | Type | N | graph | semantic | full-ctx | Notes |
|---|---|---|---|---|---|---|
| 1 | Multi-hop | 282 | 18.44% | 19.15% | 29.79% | Expansion not decisive yet |
| 2 | Temporal | 321 | **31.15%** | 30.84% | 19.63% | graph > semantic ✓ |
| 3 | Open-domain | 96 | 10.42% | 12.50% | 18.75% | Needs external knowledge |
| 4 | Single-hop | 841 | 44.47% | 46.14% | 60.88% | Largest bucket; retrieval miss drives gap |

**Single-hop vs full-context (~−16pp) drivers:**  
1. Retrieval miss ~33% (full-context ~0%)  
2. Full-context: minimal reasoning for isolated facts  
3. Turn-level nodes dilute embeddings vs narrow facts  

**LoCoMo category ID erratum:**  
Official `locomo10.json` category order **does not** match paper prose (confirmed in GitHub issue #6):  
- `1` = multi-hop (not single-hop)  
- `4` = single-hop (not multi-hop)  
Comments updated in `src/graph_mem/locomo_data.py`.
