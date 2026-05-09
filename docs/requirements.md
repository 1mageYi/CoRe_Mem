# Graph Memory System Requirements

## 1. Background and goals
- The current system uses embedding retrieval plus core/residual layered memory; ablations showed limited gains from adding the second layer.
- New direction: a directed graph memory that uses relational structure to improve retrieval across time, topics, and clues.
- Goal: a research system that is interpretable, ablatable, and reproducible **without** an LLM-based extractor.

## 2. Benchmark decisions (updated)
- **Primary benchmark: `LoCoMo` (long-context conversational memory)**  
  - Rationale: multi-session, very long dialogues (typically far beyond a single task-aligned window); evaluation matches “external memory grows with dialogue, then answer”; stronger stress test for this system.  
  - Data: official `snap-research/LoCoMo` (e.g. `locomo10.json`) or HF derivatives (e.g. `Percena/locomo-mc10`); pin version and split protocol on first integration.
- **Baseline / regression: `PERMA`**  
  - Characteristics: questions often align with provided context at task granularity—more like “preference/event QA on a given snippet” than “full long-dialogue stream”; keep for ablations and regression, not as the sole long-context conclusion.
- **Final migration validation: `PersonaMem`**  
  - Rationale: compare against the original Mem-style pipeline for gains and regressions after migration.

## 3. System scope
- Input: raw multi-turn conversation (user/assistant turns).  
- Intermediate: structured memory records from a **non-LLM** extractor.  
- Storage: graph-structured memory (nodes + typed edges).  
- Retrieval: query semantic match + graph expansion + centrality prior fusion.  
- Output: memory prompt for the answering LLM (aggregated evidence text).

## 4. Functional requirements
### 4.1 Add (ingestion)
- Extract structured records from new conversation segments.  
- Near-duplicate / high-similarity check against existing nodes.  
- High similarity → merge (keep version trail and conflict flags).  
- Low similarity → create new node.  
- After write, update edges:  
  - semantic (undirected)  
  - temporal (forward in time)  
  - co-usage (undirected, updated after online retrieval)

### 4.2 Search (retrieval)
- Query embedding to retrieve seed nodes.  
- 1–2 hop graph expansion to complete candidates.  
- Fused scoring to rank evidence.  
- Final memory evidence for the prompt builder.

### 4.3 Core / residual mechanism
- No static hand-defined buckets; layering is driven by graph statistics.  
- Core vs residual from centrality (PageRank / in-degree) and activity.  
- Higher prior weight for core at retrieval time.

## 5. Non-functional requirements
- Interpretability: each hit traces back to nodes and edge evidence.  
- Ablatability: any edge type or centrality prior can be turned off for evaluation.  
- Efficiency: no LLM in the extractor—predictable latency and cost.  
- Clear structure: modular design; do not copy GraphRAG’s engineering layout wholesale.

## 6. Extractor constraints and preferences
- Explicitly **no LLM** for extraction.  
- Initial recommendation: `rule backbone + lightweight model fallback` (hybrid).  
- Structured schema should support graph building and conflict handling; prefer an event frame:  
  - `event_type` (emergence / supplement / task)  
  - `topic` / `domain`  
  - `preference_slot`  
  - `update_type` (new / update / conflict)  
  - `evidence_span`  
  - `temporal_index`

## 7. Evaluation metrics
- Task: benchmark accuracy / success rate (per benchmark protocol).  
- Memory: retrieval hit rate, evidence coverage, temporal consistency, conflict rate.  
- Graph: average degree, connectivity, core/residual ratio, stability of centrality distribution.  
- Efficiency: add/search latency, index size, tokens per sample cost.

## 8. Acceptance criteria (phased)
- Phase A (PERMA MVP): graph retrieval beats semantic-only baseline.  
- Phase B (PERMA full): three edge types + core prior yield stable gains.  
- Phase C (PersonaMem migration): no significant regression or mild positive gain on PersonaMem.
