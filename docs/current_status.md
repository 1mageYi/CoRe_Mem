# Current Status

## 已确认决策
- 项目方向：external memory 从“扁平检索”升级为“图结构记忆”。
- 图结构：node = memory，edge 包含 semantic / temporal / co-usage 三类。
- 分层机制：保留 core/residual，但由图中心性与活跃度自分化。
- 检索加权：采用 `PageRank/中心性先验 + 语义分数` 融合（已选定）。
- extractor：不使用 LLM；优先走“规则主干 + 轻量模型兜底”。
- benchmark 顺序（**更新**）：**`LoCoMo` 作为主测**（长多会话对话，贴合「记忆随对话增长再问」）；`PERMA` 保留作对照与回归（其 task-context 常与题目强对齐，不等价长上下文流水）；全部稳定后再做 `PersonaMem` 迁移验证。

## 设计摘要
- 写入：先提取结构化事件，再做近重复检测；高相似 merge，低相似新建。
- 连接：新/更新节点完成 semantic、temporal 边维护。
- 检索：query 召回 seed 后图扩展，再做融合重排。
- 回写：同一次 query 命中且原无边的节点，更新 co-usage 边。

## 当前主要风险
- merge 时的时序冲突策略已落地为 hybrid MVP，但 context_split/hard_conflict 细分仍需增强。
- co-usage 边可能引入噪声传播，需要抑制机制。
- core prior 权重过大可能导致“中心节点垄断”。
- extractor 已有规则主干 MVP，仍需在 PERMA 上迭代规则覆盖与误报控制。
- 融合打分参数缺少固定调参协议，存在复现波动风险。

## 下一步（短期）
1. **接入 LoCoMo**：定版数据文件与 HF 数据集名；实现加载 + 增量写入对话 + QA 评测脚本（复用 `graph_mem` 管线）。
2. PERMA / Phase 3 脚本保留为回归：主结论以 LoCoMo 长对话设定为准。
3. 小样本仅作方向参考；失败归因等大 N 后再展开。

## 本轮已完成（Graph MVP Skeleton）
- 新建 `src/graph_mem/` 模块：
  - `schemas.py`：Node/Edge/StructuredRecord schema
  - `extractor.py`：RuleExtractor（非 LLM）
  - `formatter.py`：结构化文本规范化序列化
  - `embedder.py`：BGE-M3 embedder 封装
  - `graph_store.py`：基于 NetworkX 的图存储
  - `add_pipeline.py`：写入流程（含 hybrid revision 处理）
  - `ranking.py`：融合打分函数与参数
  - `search_pipeline.py`：seed + expand + rerank + co-usage 更新
- 新建 `train/perma/smoke_graph_mvp.py` 并跑通。
- 新建 `tests/graph_mem/test_smoke_pipeline.py`，测试通过（1 passed）。
- 新建真实 PERMA 对照评测脚本 `train/perma/eval_perma_graph_vs_semantic.py`：
  - 基于 `tasks/{user}/input_data_c.json` 构建 context memory
  - 对应 `evaluation/{user}/meta/overall/*.json` 读取 question/options/gold
  - 对照 `semantic-only` 与 `graph-full` 两条检索链路，统一使用 LLM 判选项
- 新建 `src/graph_mem/perma_data.py`（PERMA 样本加载）与 `src/graph_mem/eval_utils.py`（LLM评测工具）。
- 新建 `src/graph_mem/perma_eval.py`（可复用评测核心，脚本与测试共用）。
- 新建 `tests/graph_mem/test_e2e_perma_mini.py`：
  - 真实 PERMA mini e2e 回归守门（`user108`, `limit=2`）
  - 在无 `LLM_API_KEY` 环境下自动 skip，避免本地阻塞
  - 在有 key 环境下验证 graph/semantic 对照链路可执行
- 依赖已更新并安装到项目 `.venv`（requirements/pyproject/uv.lock 同步）。

## PERMA 初步实测（真实数据）
- 运行命令：
  - `python train/perma/eval_perma_graph_vs_semantic.py --user-id user108 --variant c --limit 2`
  - `python train/perma/eval_perma_graph_vs_semantic.py --user-id user108 --variant c --limit 5`
- 结果（`user108`, clean variant）：
  - `limit=2`: graph `0.50` vs semantic `0.00`，delta `+0.50`
  - `limit=5`: graph `0.60` vs semantic `0.20`，delta `+0.40`
- 输出目录：
  - `outputs/perma_eval/20260426_220502`
  - `outputs/perma_eval/20260426_220741`

## PERMA 扩样本结果（新增）
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

## 当前结论（阶段性）
- 结果已从“稳定正增益”更新为“用户间差异明显，尚不稳定”：
  - 在 `user108` 上有小幅正增益；
  - 在 `user109` 上出现明显负增益。
- 当前 graph-full 实现尚不能宣称整体优于 semantic-only。
- 下一步应优先做失败剖析与检索质量诊断，再扩大评测规模。

## 运行说明（当前入口）
- 小样本真实对照：
  - `python train/perma/eval_perma_graph_vs_semantic.py --user-id user108 --variant c --limit 5`
- mini e2e 测试：
  - `python -m pytest tests/graph_mem/test_e2e_perma_mini.py -q`

## 调试导出（新增）
- PERMA 评测现已支持逐样本调试导出：`debug_samples.jsonl`
- 每条样本记录包含：
  - `seeds`（初始语义召回）
  - `expanded`（图扩展后的候选集合）
  - `score_breakdown`（semantic/centrality/edge/temporal/final）
  - `graph_top_evidence` 与 `semantic_top_evidence`
  - `graph_prompt_preview` 与 `semantic_prompt_preview`
  - `pred_graph/pred_semantic/gold`
- 导出位置示例：
  - `outputs/perma_eval/20260426_225611/debug_samples.jsonl`

## Phase 2 进展（新增）
- core/residual 已从“仅 PageRank 阈值”升级为联合 `core_score`：
  - `core_score = 0.50 * pagerank + 0.30 * in_degree + 0.20 * retrieve_count`（min-max 归一化后融合）
- 检索重排已使用 `core_score` 作为中心性先验输入，`lambda_core` 对 core 节点做乘性加权。
- 每次检索后会回写命中节点的 `retrieve_count`，用于后续动态 core 分化。
- debug 导出已增加 `core_score` 与 `retrieve_count` 字段，便于定位 prior 偏置问题。

## 本轮验证结果（新增）
- 测试：
  - `python -m pytest tests/graph_mem -q` -> `1 passed, 1 skipped`
- 快速 PERMA 实测：
  - `python train/perma/eval_perma_graph_vs_semantic.py --user-id user108 --variant c --limit 5`
  - graph: `0.80` vs semantic-only: `0.20`，delta: `+0.60`
  - output: `outputs/perma_eval/20260426_230254`

## PERMA 多用户通用性快检（新增）
- 配置：`variant=c`, `limit=5`, 覆盖全部 10 个用户
- 汇总文件：`outputs/perma_eval/multi_user_limit5_summary.json`
- 分布结果（delta = graph - semantic）：
  - 正增益：`3/10`（`user108 +0.4`, `user123 +0.4`, `user914 +0.2`）
  - 持平：`4/10`（`user112`, `user354`, `user419`, `user507`）
  - 负增益：`3/10`（`user109 -0.4`, `user1377 -0.4`, `user334 -0.8`）
- 聚合统计：
  - `mean_delta = -0.06`
  - `median_delta = 0.0`
- 阶段性判断：
  - 当前不是“全局一致退化”，但存在明确的**多用户负增益子集**，属于需要继续定位但不必立即全局重构的状态。

## Phase 3 实现（本轮）
- **边扩展消融**：`SearchConfig` 增加 `expand_use_semantic` / `expand_use_temporal` / `expand_use_co_usage`；`PermaEvalConfig.graph_edge_mode` 通过 `ablation_presets.search_config_for_edge_mode` 注入。
- **Merge 消融**：`AddConfig.merge_strategy`：`hybrid`（原逻辑）vs `latest_wins`（高相似则覆盖最近节点）。
- **Co-usage 控制**：`MemoryGraphStore.register_co_usage_between` 按联合命中次数固化边；`decay_co_usage_edges` 实现衰减与剪枝；对应 `PermaEvalConfig` / CLI 参数。
- **Temporal probing**：`temporal_probe_thirds=True` 时在 `summary.json` 写入 `temporal_probe_thirds`（early/mid/late，按评测样本顺序三等分）。
- **批量入口**：`train/perma/run_phase3_ablations.py` → `outputs/perma_eval/phase3_ablation_<ts>/phase3_ablation_summary.json`（多用户时每 user 子目录 `edge_<mode>/`；顶层 `aggregate_by_graph_edge_mode` 为跨用户均值）。
- **推荐 edge 先导实验（示例）**：6 user × 15 题 × 4 种 edge，`merge=hybrid`：`python train/perma/run_phase3_ablations.py --limit 15 --use-default-six-users`
- **依赖**：本轮未新增 Python 包；`requirements.txt` / `pyproject.toml` / `uv.lock` 无需变更。

## 参考说明
- 设计参考 GraphRAG 的“结构化 + 图关系”思想，但实现目标是研究代码清晰、模块解耦，不照抄其工程结构。
- 参考仓库：[microsoft/graphrag](https://github.com/microsoft/graphrag/tree/main)

## LoCoMo 接入（本轮，Phase 5 启动）

### 数据确认
- 文件：`data/locomo/locomo10.json`（10 对话，每对话 ~419 turns / ~19 sessions）
- QA：共 1540 对（cat1=282, cat2=321, cat3=96, cat4=841；cat5 adversarial 默认排除）

### 关键差异 vs PERMA，以及处理方案
- **双说话人**：两方 turn 均写入图，text 前缀 `[Speaker]` 标注身份
- **Session 时序**：`time_index = (session_idx-1) * 10_000 + turn_idx`，session 感知时序保序
- **自由文本答案**：LLM 生成自由文本，默认 soft_match 评分（无额外调用）；`--use-llm-judge` 升级为 LLM judge
- **Extractor / AddPipeline / SearchPipeline**：完全复用，接口兼容无需修改

### 新增文件
- `src/graph_mem/locomo_data.py`：数据加载层（LoCoMoTurn / LoCoMoQA / LoCoMoConversation / load_locomo / turn_time_index）
- `experiments/locomo/eval_locomo_graph.py`：主评测脚本（build_graph_from_conversation / run_locomo_eval / CLI）

### 运行命令
`
# 快速验证（2 对话，每对话 20 题）
uv run python experiments/locomo/eval_locomo_graph.py --conv-limit 2 --qa-limit 20

# 仅 cat1+2 单跳 + 时序题
uv run python experiments/locomo/eval_locomo_graph.py --categories 1,2

# 全量（10 对话 × 1540 QA）
uv run python experiments/locomo/eval_locomo_graph.py
`
输出：`outputs/locomo_eval/<timestamp>/`

## LoCoMo 全上下文 baseline 新增（本轮）
- `eval_locomo_graph.py` 新增第三条 baseline：**full-context**（将全部对话 turn 按 session 格式化后整体放入 LLM 上下文）
- locomo10 最大对话约 22k tokens，gpt-4o-mini 128k 窗口完全容纳，不会触发截断
- 若超出 `--context-budget-chars`（默认 400k chars ≈ 100k tokens）则从开头截断保留最近 turn
- 对比维度：graph-full vs semantic-only vs full-context
- 快速跑（2 对话 × 20 题，3 baselines 均启用）：
  `uv run python experiments/locomo/eval_locomo_graph.py --conv-limit 2 --qa-limit 20`
- 跳过 full-context 节省 API 费用：
  `uv run python experiments/locomo/eval_locomo_graph.py --no-full-context`

## LoCoMo Debug 与修复记录（本轮）

### 发现的 Bug（均已修复）

**Bug 1: embedding 用 structured_text（metadata 前缀压缩相似度空间）**
- 现象: 所有节点 sim 集中在 0.64-0.68，正确节点无法从 top-k 中区分
- 修复: dd_pipeline.py 改为 embed `record.text`（原始 turn 文本）

**Bug 2: turn 文本缺 session 日期（相对时间无法解析）**
- 现象: cat2 时序题全部答错（"yesterday" 不知道日期）
- 修复: `eval_locomo_graph.py` 写图时 text 改为 `[Speaker, Session N, date] {turn}`

**Bug 3: re-ranking 公式 core prior 系统性压低早期 session 节点**
- 现象: 语义 rank=1 的节点（D2:1）在 expanded set 中被 session 6-7 的高 centrality 节点超越，挤出 top-8
- 原因: 后加入的节点因 semantic edge 多而 core_score 高，乘性 lambda_core boost 后分数反超
- 修复: LoCoMo eval 用 semantic-dominant RankingConfig（alpha=0.90, lambda_core=0.0, seed_topk=20）

### 修复后结果（2 conv × 20 题）
| Baseline | 修前 | 修后 |
|---|---|---|
| graph-full | 2.5% | **32.5%** |
| semantic-only | 7.5% | 35.0% |
| full-context | 42.5% | 37.5% |

cat2（时序题）：graph/semantic 达到 47.8%，超过 full-context 的 43.5%
