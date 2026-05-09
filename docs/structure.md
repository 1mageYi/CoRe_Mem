# Project Structure (Living Doc)

## Purpose
This is the project’s **high-level architecture overview**.  
Goals: easy to scan, minimal detail, updated over time.  
When the implementation changes, update this file first so code layout and system flow stay aligned.

## 1) End-to-end main pipeline
1. `Raw Conversation` input  
2. `Extractor` extracts structured memory snippets  
3. `Embedding` produces vectors for structured snippets  
4. `Node Builder` decides merge vs new, writes memory nodes  
5. `Edge Builder` builds semantic / temporal / co-usage edges  
6. `Retriever` runs query recall + graph expansion + fused reranking  
7. `Prompt Builder` assembles the final memory context  
8. `LLM Answer` produces the reply  
9. `Post-Query Update` updates co-usage edges and statistics  

## 2) Module layout (high level)
- `Extractor`: turns dialogue into structured event/preference records (non-LLM)  
- `Embedder`: embeddings for structured records  
- `Graph Store`: stores nodes / edges / state / versions  
- `Add Pipeline`: ingestion (dedup, merge, new, edge maintenance)  
- `Search Pipeline`: retrieval (seed, expand, rerank)  
- `Core/Residual Layering`: dynamic layering from centrality  
- `Evaluation`: LoCoMo primary eval (long dialogue), PERMA regression, PersonaMem final validation  
  - `LoCoMo Loader`: TBD (fill in after integration)  
  - `PERMA Loader`: real data loading aligned with task/eval (baseline)  
  - `Eval Runner`: graph-full vs semantic-only (`PermaEvalConfig` includes Phase 3 ablation fields)  
  - `Ablation presets`: `ablation_presets.py` (edge expansion modes, merge presets)  
  - `Mini E2E Test`: small real PERMA regression gate  

## 2.1 Code entry points (current)
- **LoCoMo primary evaluation**  
  - `experiments/locomo/eval_locomo_graph.py` (graph-full / semantic-only / full-context)  
  - `experiments/locomo/analyze_results.py` (post-debug analysis: overview / failures / gap / recall)  
  - `src/graph_mem/locomo_data.py` (data loading: LoCoMoTurn / LoCoMoQA / load_locomo)  
- PERMA evaluation (baseline / regression)  
  - `train/perma/eval_perma_graph_vs_semantic.py`  
  - `train/perma/tune_lambda_core.py` (`lambda_core` grid)  
  - `train/perma/run_phase3_ablations.py` (edge ablations)  
  - `train/perma/smoke_graph_mvp.py`  
- Shared eval core  
  - `src/graph_mem/perma_eval.py`  
- Test entry points  
  - `tests/graph_mem/test_smoke_pipeline.py`  
  - `tests/graph_mem/test_e2e_perma_mini.py`  

## 3) Node / Edge (abstract definitions)
### Node (memory node)
- Structured text content  
- Embedding  
- Time and provenance  
- Status (active / inactive / superseded / conflicted)  

### Edge (four types)
- `semantic`: semantic similarity, undirected  
- `temporal`: temporal order, directed (old → new)  
- `co-usage`: co-retrieved in one query with no prior edge, undirected  
- `entity`: shared named entities (spaCy NER), undirected  

## 4) Retrieval overview
- Step A: **Hybrid seed retrieval** (dense semantic + BM25 sparse, RRF fusion; BM25 off by default, on for LoCoMo eval)  
- Step B: graph expansion (edge-type controlled scope; skip when P0 high confidence)  
- Step C: **Split-slot scoring** (seed slots by semantic order; expand slots by graph proximity; slots do not compete)  
- Step D: pick top evidence, build prompt  

### 4.1 BM25 hybrid retrieval
- `MemoryGraphStore.bm25_search(query, top_k)`: BM25Okapi on raw turn text, lazy build + dirty flag  
- `SearchConfig.use_bm25 / bm25_topk / rrf_k`: enable/disable and RRF hyperparameters  
- RRF: `score(d) = Σ 1/(k + rank(d))`, k=60 (standard)  

### 4.2 P0: adaptive expansion
- `SearchConfig.adaptive_expand_threshold: float = 0.0` (0 = off)  
- If top-1 seed sim ≥ threshold → skip graph expansion to avoid neighbor noise on single-hop factual questions  
- Implemented in `SearchPipeline._maybe_expand()`  

### 4.3 P1: Entity edges
- `RuleExtractor` calls `extract_entities(text)` (spaCy `en_core_web_sm` NER) to fill `StructuredRecord.entity_mentions`  
- After the graph is built, `AddPipeline.build_entity_edges(now_ts)` adds bidirectional `entity` edges  
- High-frequency entities (appearing in >35% of nodes) are filtered to avoid speaker names becoming hubs  
- If `SearchConfig.expand_use_entity=True`, entity edges are traversed during expansion  

### 4.4 Split-slot rerank
- Replaces the original unified `fuse_score` to fix “hub center nodes crowding out factual fringe nodes”  
- `SearchConfig.seed_evidence_slots`: seed nodes fill the first N slots by descending semantic sim  
- `SearchConfig.expand_evidence_slots`: expand-only nodes fill the next M slots by max edge weight to seeds; no direct edge → fallback to 0.5×semantic  
- `final_topn_evidence = seed_evidence_slots + expand_evidence_slots`  
- Default off (both 0 → fallback to `fuse_score`); LoCoMo eval uses seed=10 / expand=6  

## 5) Placeholders (fill when implemented)
- Extractor implementation: `RuleExtractor (MVP)`, later optional lightweight model fallback  
- Embedding model: `BAAI/bge-m3` (chosen)  
- Conflict resolution v1: `Hybrid` (compatible merge, revision new version)  
- Stable config after search: `TBD`  

## 6) Update rules
- Must update this file when architecture paths change.  
- Must update module layout and main pipeline when modules are added or merged.  
- Keep it **high-level readable**—no implementation minutiae or experiment logs.  
