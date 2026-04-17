# V2.6 Gain-First Long-Run Plan

## 目标

这轮 `v2.6` 的目标，不再是补齐 artifact/package/verifier，而是要求 **current-head 在保持 `core / residual` 双银行结构不变的前提下，出现真实的 learned gain**。

具体来说：

- 把 `LongMemEval-S` 作为第一质量 benchmark，要求 current-head `128` canary **明确超过** `v2.5` retained baseline
- 把 `write -> retrieve -> belief` 三段 learned 主链从“有接口、有 artifact”推进到“至少一段出现真实正增益”
- 保持 `semantic-first`
- 保持 `full benchmark` 只作 holdout evaluation
- 严格禁止任何 fallback / shortcut / benchmark-specific heuristic / benchmark leakage

## 冻结边界

以下部分在 `v2.6` 中**明确冻结，不作为主改动对象**：

- `core / residual` 双银行结构
- `observation -> slot -> core/residual -> retrieval -> belief -> answer` 的总体主链
- `semantic-first`

本轮只允许重点改动三段：

- `write`
- `retrieve`
- `belief`

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
- 收益必须来自 current-head 的 `write / retrieve / belief` learned 行为提升，而不是 artifact completeness

## 成功标准

至少同时满足：

- current-head `v2.6` full-data non-tiny train/eval artifact 落地
- current-head `v2.6` 的 `write / retrieve / belief` gain artifact 均落地
- 三段中至少 **一段** 出现真实正增益
- current-head `LongMemEval-S 128` **明确高于** `v2.5` retained baseline `10/128`
- current-head `PersonaMem 128` 不明显退化
- current-head full-benchmark holdout summary 落地
- failure analysis 仍可拆到 `slot_assignment / retrieval / belief / projection / provider`
- 整个过程不引入任何 fallback / shortcut / benchmark leakage

## Milestone A：Freeze V2.5 Truth

目标：把 `v2.5` 的 package closeout 作为显式 retained baseline，而不是继续误读成 gain closeout。

里程碑：

- `latest_stage2_v25_train.json`
- `latest_stage2_v25_eval.json`
- `latest_longmemeval_stage2_v25_canary.json`
- `latest_personamem_stage2_v25_canary.json`
- `latest_stage2_v25_write_gain.json`
- `latest_stage2_v25_retrieve_gain.json`
- `latest_stage2_v25_belief_gain.json`

## Milestone B：Gain-First Write

目标：`write` 必须从 baseline decomposition 推进到真实 online gain。

里程碑：

- current-head `v2.6` write train/eval artifact
- current-head `v2.6` write gain artifact
- `write` gain artifact 显式记录 `positive_gain = true`

## Milestone C：Gain-First Retrieve

目标：`retrieve` 不再只是结构存在，而是对 `LongMemEval-S` 的 retained 结果产生真实提升。

里程碑：

- current-head `v2.6` retrieve gain artifact
- retrieval miss / failure bucket 相比 `v2.5` 有可解释改善

## Milestone D：Gain-First Belief

目标：`belief` 不只是 semantic-first 可修复，而是 current-head 在线收益的一部分。

里程碑：

- current-head `v2.6` belief gain artifact
- belief artifact 显示相对 `v2.5` retained baseline 的正增益或更低 failure concentration

## Milestone E：LongMemEval-S Quality Breakout

目标：current-head `LongMemEval-S 128` **明确突破** `v2.5` retained baseline。

里程碑：

- current-head `LongMemEval-S 128`
- `provider_exact_match > 10`
- `local_exact_match > 10`
- refreshed `LongMemEval-S` analysis artifact

## Milestone F：Cross-Benchmark Guard

目标：不允许为了拉 `LongMemEval-S` 而明显破坏 `PersonaMem`。

里程碑：

- current-head `PersonaMem 128`
- 与 `v2.5` retained baseline 相比不明显退化

## Milestone G：Holdout Full-Benchmark Evaluation

目标：用 full benchmark / larger slice 做 holdout 验证，而不是把它变成训练反馈回路。

里程碑：

- current-head `v2.6` full benchmark run summary
- trend table: current-head vs `v2.5` retained baseline
- `holdout_only = true`

## 主指标

- `stage2_v26_longrun_score`

## 守卫

- `v2.5` retained line 仍然可复验
- stage-2 tests 持续通过
- 不允许重新引入 `fallback / shortcut / benchmark-specific heuristic`
- full benchmark 只作 holdout measurement，不得回流到训练
- `PersonaMem 128` 不允许因为 `LongMemEval-S` 优化而明显崩坏
