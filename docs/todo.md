# Graph Memory TODO

## Phase 0: 文档与基线准备
- [x] 固化 benchmark 顺序（当前）：LoCoMo（主测·长上下文）→ PERMA（对照/回归）→ PersonaMem（最终迁移）
- [x] 写清 baseline 对照：semantic-only、旧 core/residual、mem0-like
- [x] 定义统一指标看板（准确率、延迟、记忆覆盖、图统计）

## Phase 1: Graph MVP（PERMA）
- [x] 定义 node/edge schema 与存储接口
- [x] 实现 non-LLM extractor（规则主干）
- [x] 实现 Add 流程（dedup + create + merge）
- [x] 实现三类 edge 构建（semantic/temporal/co-usage）
- [x] 实现 Search 流程（seed + expand + rerank）
- [x] 实现 prompt builder（可解释 evidence 输出）
- [x] 增加 PERMA 风格 smoke runner（最小可运行）
- [x] 增加真实 PERMA mini eval（graph-full vs semantic-only）
- [x] 增加 mini e2e 回归测试文件（真实 PERMA 样本，limit=2）

## Phase 2: Core/Residual 自分化
- [x] 引入 PageRank/入度/使用频次计算 core_score
- [x] 实现 core prior 融合打分
- [x] 完成 lambda_core 网格搜索（`PermaEvalConfig.lambda_core` + `train/perma/tune_lambda_core.py`）
- [x] 对比“无 prior vs 有 prior”（网格含 `λ=0` 与默认/多档 `λ`）

## Phase 3: PERMA 消融与稳健性
- [x] edge 消融（`graph_edge_mode`：`full` / `semantic_only` / `semantic_temporal` / `semantic_co_usage`）
- [x] merge 策略消融（`merge_strategy`：`hybrid` vs `latest_wins`）
- [x] co-usage 污染控制（`co_usage_min_count` 固化门槛 + `co_usage_decay` / `co_usage_prune_threshold`）
- [x] temporal probing 分段评测（`--temporal-probe`：按样本顺序三等分 early/mid/late）
- [ ] 失败样本剖析（待更大 N 或全量跑完后再做，小样本不展开）
- [x] 加入 per-sample debug 导出（seed/expanded/top evidence/score 分解）

## Phase 4: PersonaMem 最终迁移验证
- [ ] 适配 PersonaMem 数据入口
- [ ] 复用同一 memory graph 流程执行评测
- [ ] 对比旧方案与图方案增益/退化点
- [ ] 输出最终对照报告

## Phase 5: LoCoMo 主评测（原「可选压测」升格）
- [x] 选定数据快照：`data/locomo/locomo10.json`（10 对话，cat5 adversarial 默认排除，1540 QA）
- [x] 实现 `src/graph_mem/locomo_data.py`：LoCoMoTurn / LoCoMoQA / LoCoMoConversation / load_locomo / turn_time_index
- [x] 实现 `experiments/locomo/eval_locomo_graph.py`（复用 graph store / SearchPipeline；双说话人写入；soft_match + LLM judge 评分）
- [x] 快速验证 `--conv-limit 2 --qa-limit 20`（已通，见 current_status）
- [x] 实现 BM25 混合检索（`rank-bm25` + RRF，`SearchConfig.use_bm25=True`，已启用于 LoCoMo eval）
- [x] P0 自适应扩展（`adaptive_expand_threshold=0.80`，高置信度时跳过图展开防噪声）
- [x] P1 Entity 边（spaCy NER → `entity_mentions` → `build_entity_edges()` → 图遍历）
- [x] 全量主测 run1（20260427_141126）：graph 28.5% / sem 32.9% / full 42.9%
- [x] 归因分析：seed miss 26.4%，rerank cut 33.8%，topk+LLM_wrong 27.5%
- [x] 扩召回：seed/bm25 topk 20→40，final_topn 8→12
- [x] Prompt 升级：跨 snippet 推理、时序推算、常识桥接
- [x] **全量 run2**（召回 + prompt）：graph 29.74%(+1.23pp)，gold in top-k 54.2%→58.8%
- [x] **全量 run3**（final_topn 12→16）：中途停止，发现 rerank 降级问题更根本
- [x] 归因：rerank 把 gold 从语义 rank≤12 的 73% 降到 final top-12 的 63%（-10pp）
- [x] 实现 split-slot rerank（seed 10 位按语义，expand 6 位按图邻近度）
- [x] **全量 run4**（split-slot rerank，20260427_194910）：graph 34.81% / sem 35.91% / full 43.96%
- [x] 修正 LoCoMo category ID 映射（官方 bug，cat1=multi-hop，cat4=single-hop，与 paper 文字描述相反）
- [ ] 探索降低 single-hop 与 full-context 差距（当前 -16pp，主因是 ~33% 检索 miss + 节点粒度粗）
- [ ] 报告：准确率、延迟、图统计（与 PERMA 脚本同级别产物）
- [ ] 超长上下文下时延与内存占用评估（随 LoCoMo 一并记录）

## 交付物检查
- [ ] docs 完整（requirements / implementation_plan / todo / current_status）
- [ ] 实验配置可复现（参数、随机种子、run 命名）
- [ ] 结论含失败分析，不只报最好结果
