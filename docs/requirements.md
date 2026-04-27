# Graph Memory System Requirements

## 1. 背景与目标
- 当前系统采用 embedding 检索 + core/residual 双层记忆，消融显示单层与双层差距不明显。
- 新方向：构建有向图记忆系统，利用节点关系提升跨时间、跨主题、跨线索的检索质量。
- 目标是在不使用 LLM extractor 的前提下，实现可解释、可消融、可复现实验的 research 系统。

## 2. Benchmark 决策（已更新）
- **主 benchmark：`LoCoMo`（长上下文对话记忆）**
  - 原因：多会话、极长对话（典型远长于单 task 对齐窗口），评测目标与「外部记忆在长对话中不断写入、再问」一致；更接近本系统的 stress test。
  - 数据入口：官方仓库 `snap-research/LoCoMo`（如 `locomo10.json`）或 HF 衍生集（例如 `Percena/locomo-mc10` 等多选/结构化版本）；首次接入时需固定版本与拆分协议。
- **对照 / 回归：`PERMA`**
  - 特点：题目与所提供的 context 往往在 task 粒度上对齐，更像「给定片段上的偏好/事件 QA」，与「整条长对话流水」不一致；保留作消融与回归，不作为长上下文主结论依据。
- **最终迁移验证：`PersonaMem`**
  - 原因：与原 Mem 类流水线对照，检验迁移到新图记忆后的增益与退化。

## 3. 系统范围
- 输入：原始多轮 conversation（user/assistant turns）。
- 中间表示：非 LLM extractor 产生的结构化记忆记录。
- 存储：图结构 memory（node + typed edges）。
- 检索：query 语义匹配 + 图扩展 + 中心性先验融合。
- 输出：供回答 LLM 的 memory prompt（证据聚合后的文本）。

## 4. 功能需求
### 4.1 写入（Add）
- 将新 conversation 片段提取为结构化记录。
- 对新记录与现有 node 做近重复/高相似检测。
- 高相似时执行 merge（保留版本轨迹与冲突标记）。
- 低相似时创建新 node。
- 写入后更新边关系：
  - semantic edge（双向）
  - temporal edge（前 -> 后）
  - co-usage edge（双向，在线检索后更新）

### 4.2 检索（Search）
- 用 query embedding 召回种子 nodes。
- 基于图关系进行 1~2 hop 扩展补全。
- 采用融合打分排序候选证据。
- 输出最终 memory evidence 供 prompt builder 组装。

### 4.3 Core/Residual 机制
- 不再手工分桶静态定义，改为图统计驱动分层。
- 以中心性（PageRank / 入度）和活跃度为 core 判据。
- 检索时 core 给予更高先验权重。

## 5. 非功能需求
- 可解释性：每个检索结果可回溯到 node 和 edge 证据链。
- 可消融性：可以关闭任一 edge 类型或中心性先验单独评测。
- 效率：extractor 不依赖 LLM，保证可控时延与成本。
- 结构清晰：模块解耦，不照抄 GraphRAG 的工程组织。

## 6. Extractor 约束与偏好
- 明确不使用 LLM 做提取。
- 初版推荐：`规则主干 + 轻量模型兜底`（hybrid）。
- 结构化 schema 应支持图构建和冲突处理，优先 event-frame：
  - event_type（emergence/supplement/task）
  - topic/domain
  - preference_slot
  - update_type（new/update/conflict）
  - evidence_span
  - temporal_index

## 7. 评估指标
- 任务指标：benchmark accuracy / success rate（按 benchmark 协议）。
- 记忆指标：检索命中率、证据覆盖率、时序一致性、冲突率。
- 图指标：平均度、连通性、core/residual 比例、中心性分布稳定性。
- 效率指标：add/search 延迟、索引大小、单位样本 token 成本。

## 8. 验收标准（阶段性）
- Phase A（PERMA MVP）：图检索方案优于 semantic-only baseline。
- Phase B（PERMA 完整）：三类 edge + core prior 融合取得稳定增益。
- Phase C（PersonaMem 迁移）：在 PersonaMem 上保持正向或不显著退化。
