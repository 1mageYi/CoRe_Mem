# Current Status

## 已确认决策
- 项目方向：external memory 从“扁平检索”升级为“图结构记忆”。
- 图结构：node = memory，edge 包含 semantic / temporal / co-usage 三类。
- 分层机制：保留 core/residual，但由图中心性与活跃度自分化。
- 检索加权：采用 `PageRank/中心性先验 + 语义分数` 融合（已选定）。
- extractor：不使用 LLM；优先走“规则主干 + 轻量模型兜底”。
- benchmark 顺序：`PERMA` 主测，全部完成后再测 `PersonaMem`。

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
1. 接入 PERMA 样本读取，替换 smoke 脚本中的手工 turns。
2. 强化冲突分类规则：补齐 context_split 与 hard_conflict 判定。
3. 增加 baseline 评测脚本（semantic-only / graph-full）。
4. 固定参数搜索 protocol，并产出首轮 PERMA dev 结果。
5. MVP 稳定后进入 edge/core 消融，再准备 PersonaMem 迁移。

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

## 参考说明
- 设计参考 GraphRAG 的“结构化 + 图关系”思想，但实现目标是研究代码清晰、模块解耦，不照抄其工程结构。
- 参考仓库：[microsoft/graphrag](https://github.com/microsoft/graphrag/tree/main)
