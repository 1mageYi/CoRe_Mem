# V2.3 Long-Run Plan

## 目标

这轮长跑的目标是把 `CoRe_Mem` 从“`v2.2` 证据链完整”继续推进到“更强的在线系统 / 更强的 learned model / 更强的 latent memory”。

相比短版 `v2.3`，这轮主线更远，要求的不只是：

- `learned slot assignment` 接上
- `LongMemEval-S` 有一点点提升

而是：

- `LongMemEval-S` 真正成为质量主 benchmark
- `learned slot assignment` 进入在线主链，并在 current-head 上形成对照增益
- `retrieval -> composed latent -> belief -> answer` 的 retained 收益更多来自 learned latent，而不是 rule-heavy patching
- `PersonaMem 128` 在 LongMemEval-S 提升过程中不能明显退化
- 允许多卡并行实验，但单实验默认最多占 `1 GPU`，live benchmark 继续串行

## 硬约束

- 不做任何 `fallback`
- 不做任何 `benchmark-specific shortcut`
- 不把 provider prompt 小技巧当成 retained 收益
- retained 收益必须落在 `online memory -> belief -> answer` 主链
- 继续保留 `semantic-first`
- 允许格式修复/约束，但不能把外部修复伪装成模型本体收益
- 训练默认使用完整 public-data prepared tasks
- 允许多卡并行实验；默认策略是多卡并行多个单卡实验，而不是默认上多卡数据并行训练

## 成功标准

至少同时满足这些条件：

- `LongMemEval-S 64/128` 上出现 current-head 的 learned slot-assignment retained artifact
- 存在 explicit 的 `learned slot assignment vs symbolic lifecycle` gain artifact
- `LongMemEval-S` 相比 `v2.2` 基线有明确、可复验的 retained 增益
- `PersonaMem 128` 保持不明显退化
- full-data `semantic-first` 训练质量不明显退化
- 整体过程不引入任何 fallback / shortcut

## Milestone A：Baseline And Taxonomy

目标：把 `v2.2` 的 closeout 变成 `v2.3` 的明确对照基线。

- 固化 `LongMemEval-S 128` 当前基线
- 固化 `PersonaMem 128` 当前 guard 基线
- 把 `LongMemEval-S` 错误拆到：
  - `slot_assignment`
  - `retrieval`
  - `belief`
  - `projection`
  - `provider`

里程碑：

- `latest_longmemeval_stage2_semantic_analysis.json` 刷新为带 top error clusters 的版本
- `latest_stage2_v23_baseline_summary.json`

## Milestone B：Learned Slot Assignment Module

目标：让 `observation -> slot` 的关键决策不再完全依赖 rule-heavy lifecycle。

- 保留 parser 作为 schema / safety 入口
- 新增 `learned slot assignment` 模块：
  - `action`: `merge / overwrite / new / ignore`
  - `target_slot`
  - `stale_old`
- 仅保留最小 hard constraints：
  - 不跨 entity 合并
  - 不跨明显 type/confidence 边界乱写
  - 不让低置信 observation 污染 core bank

里程碑：

- slot-assignment 配置开关
- online system 中的 slot-assignment mode
- 至少 1 个 current-head slot-assignment artifact

## Milestone C：Online-Aligned Slot-Assignment Training

目标：让训练直接服务 `slot assignment / retrieval / belief` 主链。

- 新增 slot-assignment supervision
  - action classification
  - candidate-slot ranking
- `retrieval_alignment` 逐步向 ranking / support selection 靠拢
- full-data `online_aligned` 训练继续保持 semantic-first

里程碑：

- `latest_stage2_slot_assignment_train.json`
- `latest_stage2_slot_assignment_eval.json`
- full-data semantic-side 质量不明显退化

## Milestone D：LongMemEval-S 64 Validation

目标：先在较快验证环上确认 learned slot assignment 有效。

- current-head `LongMemEval-S 64`
- learned-vs-symbolic 对照
- retained gain artifact

里程碑：

- `latest_longmemeval_stage2_slot_assignment_canary.json`
- `latest_stage2_slot_assignment_gain.json`

## Milestone E：LongMemEval-S 128 Validation

目标：把 learned slot assignment 的收益推进到更大切片，避免 64-sample 偶然性。

- current-head `LongMemEval-S 128`
- gain artifact 刷新到 `128`
- 错误分析刷新

里程碑：

- current-head `LongMemEval-S 128` slot-assignment artifact
- current-head `LongMemEval-S` retained gain 为正且可解释

## Milestone F：PersonaMem Guard

目标：避免为了抬 `LongMemEval-S` 把另一个 benchmark 弄坏。

- current-head `PersonaMem 128`
- learned slot-assignment mode
- guard against major regression

里程碑：

- current-head `PersonaMem 128` slot-assignment artifact
- `provider/local` 不明显低于 `v2.2` guard baseline

## Milestone G：Recipe And Verifier Closeout

目标：让 `v2.3` 长跑从实验堆叠收口成默认 recipe。

- 默认训练命令
- 默认 benchmark 命令
- 默认 verifier
- 文档和 `.agent-os` 对齐

里程碑：

- 一套默认 slot-assignment train recipe
- 一套默认 LongMemEval-S / PersonaMem recipe
- 一套 `v2.3 long-run` verifier

## 主指标

- `stage2_v23_longrun_score`

## 守卫

- `v2.2` closeout 证据链必须仍然存在
- stage-2 tests 持续通过
- 不允许重新引入 `fallback / shortcut`
- `PersonaMem 128` 不允许因为 `LongMemEval-S` 优化而明显崩坏
