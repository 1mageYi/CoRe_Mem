# V2.7 32k Teacher-First Long-Run Plan

## 目标

这轮 `v2.7` 的目标，不再先追更高 benchmark 分数，而是先把 **数据量、数据质量、内部泛化能力** 立住，再决定是否扩到 full-data。

具体来说：

- 以 **32k source-level dataset** 作为中间锚点，而不是继续使用当前 `64-per-dataset` 的极小切片
- 先在 `32k` 上建立真正的 `train / val / test` 泛化评测，再决定是否扩到 full-data
- 用 `MiniMax-M2.7` 作为 **LLM teacher**，优先提升：
  - `observation / parser` 标签质量
  - `slot assignment / lifecycle` 标签质量
  - `belief semantic fields` 标签质量
- 训练优先使用 `gpu2`
- 显式记录训练耗时、吞吐与资源占用，判断 `32k` 是否过重
- benchmark 继续只作 holdout evaluation，不回流成训练 supervision

## 冻结边界

以下部分在 `v2.7` 中**明确冻结，不作为主改动对象**：

- `core / residual` 双银行结构
- `observation -> slot -> core/residual -> retrieval -> belief -> answer` 的总体主链
- `semantic-first`

本轮允许重点改动的部分：

- `observation / parser`
- `slot assignment / lifecycle supervision`
- `belief semantic fields`
- `32k` data pipeline / split / teacher labeling / training runtime

## 硬约束

- 不做任何 `fallback`
- 不做任何 `shortcut`
- 不做任何 `benchmark-specific heuristic`
- 不使用 benchmark label / answer / protocol 输出作为训练 supervision
- 不把 full benchmark 结果回流成训练 supervision
- 不把 raw JSON exactness 当成主优化目标
- 不继续靠 rule patch 作为 retained 主收益
- benchmark 只作为 evaluation source，不作为 primary training source
- full benchmark 保持为 `holdout evaluation`
- `32k` split 必须在 **source-record level** 完成，不能先 task expansion 再 split

## 32k 锚点策略

`v2.7` 只先做 `32k`。

推荐 split：

- `24k train`
- `4k val`
- `4k test`

切分原则：

- 在 `source-record` 层切分
- 尽量按以下维度做分层：
  - `source_dataset`
  - `relation`
  - 是否存在 `lifecycle` / update-sensitive 行为
  - `parser` 来源或质量桶（如 `fallback`、`rule_first_v1`、structured-derived）

**只有在 32k internal test work well 后，才允许进入 full-data。**

## Teacher 优先级

本轮 teacher 只优先做三类标签升级：

### 1. Observation Teacher

目标字段：

- `relation`
- `value`
- `time_scope`
- `status_hint`
- `polarity`

Teacher：`MiniMax-M2.7`

### 2. Slot-Assignment Teacher

目标字段：

- `target_action`
- `matched_slot_id`
- `promote`
- `stale_old`

Teacher：`MiniMax-M2.7`

### 3. Belief-Field Teacher

目标字段：

- `belief_items`
- `support_slot_ids`
- `time_scope`
- `status`

Teacher：`MiniMax-M2.7`

## 训练原则

- 训练设备默认使用 `gpu2`
- 训练时必须记录：
  - wall-clock time
  - examples / sec
  - GPU memory
  - effective num_examples
- 先用 `32k` 建立稳定 train/val/test 与 teacher-labeled subset
- 暂不进入 full-data 训练

## 成功标准

至少同时满足：

- current-head `v2.7` 的 `32k split` artifact 落地
- current-head `v2.7` 的 data-quality audit artifact 落地
- current-head `v2.7` 的 teacher observation / slot-assignment / belief artifacts 落地
- current-head `v2.7` 的 `32k` train / eval / timing artifacts 落地
- internal `32k test` 成为真正的泛化门槛
- benchmark holdout 继续不退化
- 整个过程不引入任何 fallback / shortcut / benchmark leakage

## 主指标

- `stage2_v27_longrun_score`

## 守卫

- `v2.6` retained line 仍然可复验
- stage-2 tests 持续通过
- 不允许重新引入 `fallback / shortcut / benchmark-specific heuristic`
- full benchmark 仍然只作 holdout measurement
- `PersonaMem 128` 与 `LongMemEval-S 128` 不允许因为 teacher-first 32k 试验而明显崩坏
