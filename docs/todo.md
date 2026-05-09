# Graph Memory TODO

## Phase 0: Documentation and baselines
- [x] Freeze benchmark order (current): LoCoMo (primary · long context) → PERMA (baseline / regression) → PersonaMem (final migration)
- [x] Document baselines: semantic-only, legacy core/residual, mem0-like
- [x] Define unified metrics dashboard (accuracy, latency, memory coverage, graph stats)

## Phase 1: Graph MVP (PERMA)
- [x] Define node/edge schema and storage API
- [x] Implement non-LLM extractor (rule backbone)
- [x] Implement Add path (dedup + create + merge)
- [x] Implement three edge types (semantic / temporal / co-usage)
- [x] Implement Search path (seed + expand + rerank)
- [x] Implement prompt builder (interpretable evidence output)
- [x] Add PERMA-style smoke runner (minimal runnable path)
- [x] Add real PERMA mini eval (graph-full vs semantic-only)
- [x] Add mini e2e regression test (real PERMA samples, limit=2)

## Phase 2: Core / residual self-differentiation
- [x] Introduce PageRank / in-degree / usage frequency for `core_score`
- [x] Implement core prior in fused scoring
- [x] Complete `lambda_core` grid search (`PermaEvalConfig.lambda_core` + `train/perma/tune_lambda_core.py`)
- [x] Compare “no prior vs prior” (grid includes `λ=0` and default / multiple `λ`)

## Phase 3: PERMA ablations and robustness
- [x] Edge ablation (`graph_edge_mode`: `full` / `semantic_only` / `semantic_temporal` / `semantic_co_usage`)
- [x] Merge strategy ablation (`merge_strategy`: `hybrid` vs `latest_wins`)
- [x] Co-usage noise control (`co_usage_min_count` solidification + `co_usage_decay` / `co_usage_prune_threshold`)
- [x] Temporal probing by thirds (`--temporal-probe`: early/mid/late by sample order)
- [ ] Failure-case analysis (defer until larger N or full runs; skip deep dive on tiny samples)
- [x] Per-sample debug export (seed / expanded / top evidence / score breakdown)

## Phase 4: PersonaMem final migration validation
- [ ] Adapt PersonaMem data ingress
- [ ] Reuse the same memory graph pipeline for evaluation
- [ ] Compare old vs graph scheme on gains and regressions
- [ ] Produce final comparison report

## Phase 5: LoCoMo primary evaluation (formerly “optional stress test”)
- [x] Pin data snapshot: `data/locomo/locomo10.json` (10 conversations, cat5 adversarial excluded by default, 1540 QA pairs)
- [x] Implement `src/graph_mem/locomo_data.py`: LoCoMoTurn / LoCoMoQA / LoCoMoConversation / load_locomo / turn_time_index
- [x] Implement `experiments/locomo/eval_locomo_graph.py` (reuse graph store / SearchPipeline; two-speaker ingest; soft_match + LLM judge scoring)
- [x] Quick validation `--conv-limit 2 --qa-limit 20` (passed; see current_status)
- [x] BM25 hybrid retrieval (`rank-bm25` + RRF, `SearchConfig.use_bm25=True`, enabled in LoCoMo eval)
- [x] P0 adaptive expansion (`adaptive_expand_threshold=0.80`, skip expansion when confident to reduce noise)
- [x] P1 Entity edges (spaCy NER → `entity_mentions` → `build_entity_edges()` → graph traversal)
- [x] Full primary run1 (20260427_141126): graph 28.5% / sem 32.9% / full 42.9%
- [x] Attribution: seed miss 26.4%, rerank cut 33.8%, topk+LLM_wrong 27.5%
- [x] Widen recall: seed/bm25 topk 20→40, final_topn 8→12
- [x] Prompt upgrade: cross-snippet reasoning, temporal inference, commonsense bridging
- [x] **Full run2** (recall + prompt): graph 29.74% (+1.23pp), gold in top-k 54.2%→58.8%
- [x] **Full run3** (final_topn 12→16): stopped; rerank degradation identified as deeper issue
- [x] Attribution: rerank drops gold from semantic rank≤12 in 73% of cases to final top-12 in 63% (-10pp)
- [x] Implement split-slot rerank (10 seed by semantics, 6 expand by graph proximity)
- [x] **Full run4** (split-slot rerank, 20260427_194910): graph 34.81% / sem 35.91% / full 43.96%
- [x] Fix LoCoMo category ID mapping (official bug: cat1=multi-hop, cat4=single-hop, opposite of paper prose)
- [ ] Narrow single-hop vs full-context gap (currently ~-16pp; main drivers ~33% retrieval miss + coarse node granularity)
- [ ] Report: accuracy, latency, graph stats (same level as PERMA scripts)
- [ ] Latency and memory under very long context (record alongside LoCoMo)

## Deliverable checklist
- [ ] Docs complete (requirements / implementation_plan / todo / current_status)
- [ ] Experiments reproducible (parameters, seeds, run naming)
- [ ] Conclusions include failure analysis, not only best numbers
