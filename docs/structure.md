# Project Structure (Living Doc)

## 目的
这是项目的“整体框架总览文档”。  
要求：一目了然、少细节、可持续更新。  
当实现方案变化时，优先更新本文件，让代码结构与系统流程保持一致。

## 1) 端到端主链路
1. `Raw Conversation` 输入
2. `Extractor` 抽取结构化记忆片段
3. `Embedding` 生成结构化片段向量
4. `Node Builder` 判定 merge/new，写入 memory node
5. `Edge Builder` 构建 semantic / temporal / co-usage 边
6. `Retriever` 用 query 召回 + 图扩展 + 融合重排
7. `Prompt Builder` 组装最终 memory context
8. `LLM Answer` 输出回答
9. `Post-Query Update` 更新 co-usage 边与统计信息

## 2) 模块框架（高层）
- `Extractor`：把对话转成结构化事件/偏好记录（非 LLM）
- `Embedder`：对结构化记录做 embedding
- `Graph Store`：存 node/edge/state/version
- `Add Pipeline`：写入流程（dedup、merge、new、edge维护）
- `Search Pipeline`：检索流程（seed、expand、rerank）
- `Core/Residual Layering`：基于中心性动态分层
- `Evaluation`：LoCoMo 主测（长对话），PERMA 回归，PersonaMem 最终验证
  - `LoCoMo Loader`：TBD（接入后写入）
  - `PERMA Loader`：真实数据读取与 task/eval 对齐（对照）
  - `Eval Runner`：graph-full vs semantic-only 对照评测（`PermaEvalConfig` 含 Phase 3 消融字段）
  - `Ablation presets`：`ablation_presets.py`（边扩展模式、merge 预设）
  - `Mini E2E Test`：真实 PERMA 小样本回归守门

## 2.1 代码入口（当前）
- **LoCoMo 主评测**（主测）：
  - `experiments/locomo/eval_locomo_graph.py`（graph-full / semantic-only / full-context 三路对比）
  - `experiments/locomo/analyze_results.py`（debug 后分析：overview / failures / gap / recall）
  - `src/graph_mem/locomo_data.py`（数据加载：LoCoMoTurn / LoCoMoQA / load_locomo）
- PERMA 评测（对照/回归）：
  - `train/perma/eval_perma_graph_vs_semantic.py`
  - `train/perma/tune_lambda_core.py`（`lambda_core` 网格）
  - `train/perma/run_phase3_ablations.py`（edge 消融）
  - `train/perma/smoke_graph_mvp.py`
- 可复用评测核心：
  - `src/graph_mem/perma_eval.py`
- 测试入口：
  - `tests/graph_mem/test_smoke_pipeline.py`
  - `tests/graph_mem/test_e2e_perma_mini.py`

## 3) Node / Edge（抽象定义）
### Node（记忆节点）
- 结构化文本内容
- embedding
- 时间与来源信息
- 状态信息（active/inactive/superseded/conflicted）

### Edge（四类）
- `semantic`：语义相似，双向
- `temporal`：时间前后，单向（old -> new）
- `co-usage`：同次 query 共用且原无边，双向
- `entity`：共享命名实体（spaCy NER），双向

## 4) Retrieval 总览
- Step A：**Hybrid seed 召回**（semantic dense + BM25 sparse，RRF 融合；BM25 默认关，LoCoMo eval 开启）
- Step B：图扩展补全（按 edge 类型控制范围；P0 高置信度时跳过）
- Step C：**Split-slot 打分**（seed 槽按语义排序；expand 槽按图边邻近度排序；两类不竞争）
- Step D：选 top evidence，构建 prompt

### 4.1 BM25 混合检索
- `MemoryGraphStore.bm25_search(query, top_k)`：BM25Okapi on raw turn text，lazy build + dirty flag
- `SearchConfig.use_bm25 / bm25_topk / rrf_k`：控制是否启用及 RRF 超参
- RRF 公式：`score(d) = Σ 1/(k + rank(d))`，k=60（标准值）

### 4.2 P0：自适应扩展
- `SearchConfig.adaptive_expand_threshold: float = 0.0`（0=关闭）
- top-1 seed sim ≥ threshold → 直接跳过图展开，避免单跳事实题的邻居噪声污染
- 实现在 `SearchPipeline._maybe_expand()`

### 4.3 P1：Entity 边
- `RuleExtractor` 调用 `extract_entities(text)`（spaCy en_core_web_sm NER）填充 `StructuredRecord.entity_mentions`
- 全图建完后调用 `AddPipeline.build_entity_edges(now_ts)` 批量添加 `entity` 类型双向边
- 高频实体（出现在 >35% 节点）被过滤，避免说话人名成为 hub
- 图展开时 `SearchConfig.expand_use_entity=True` 则 entity 边也被遍历

### 4.4 Split-slot Rerank
- 取代原始统一 `fuse_score`，解决"图中心节点挤出事实边缘节点"问题
- `SearchConfig.seed_evidence_slots`：seed 节点按 semantic sim 降序填满前 N 个槽
- `SearchConfig.expand_evidence_slots`：expand-only 节点按与 seed 的最大边权重填满后 M 个槽；无直接边时 fallback 到 0.5×semantic
- `final_topn_evidence = seed_evidence_slots + expand_evidence_slots`
- 默认关闭（两者为 0 时退回 fuse_score），LoCoMo eval 设置 seed=10 / expand=6

## 5) 当前留空位（待实现后填充）
- Extractor 具体实现方案：`RuleExtractor (MVP)`，后续补轻量模型兜底
- Embedding 模型最终选型与版本：`BAAI/bge-m3`（已定）
- 冲突解算的最终规则（v1）：`Hybrid`（compatible merge, revision new-version）
- 参数搜索后的稳定配置：`TBD`

## 6) 更新规则
- 架构路径变化：必须更新本文件。
- 模块新增/合并：必须更新模块框架与主链路。
- 保持“高层可读”，不要塞实现细节和实验日志。
