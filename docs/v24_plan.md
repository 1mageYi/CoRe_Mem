# V2.4 Long-Run Plan

## 目标

这轮长跑的目标是把 `CoRe_Mem` 从“`v2.3` 证据链与 slot-assignment 主链已经接通”继续推进到：

- `LongMemEval-S` 真正成为质量主 benchmark
- `learned slot assignment` 从接线状态推进到可训练、可增益、可稳定复验
- `retrieval -> composed latent -> belief -> answer` 主链的 retained 收益更多来自更强的 learned latent / stronger latent
- 在提升 `LongMemEval-S` 的同时，`PersonaMem 128` 不能明显退化

这轮的重点不再是“有没有 artifact”，而是：

- 更好的 online 质量
- 更强的 full-data learned model
- 更强的 latent memory
- 更稳的 current-head online gain

## 硬约束

- 不做任何 `fallback`
- 不做任何 `benchmark-specific shortcut`
- 不把 provider prompt 小技巧当成 retained 收益
- 不把外部修复伪装成模型本体收益
- 继续保留 `semantic-first`
- 继续允许格式约束/修复，但收益必须能回流到语义正确和 online 主链
- 训练与测试默认使用完整 public-data prepared tasks
- 也就是说，本轮默认按“完整 public-data”训练和测试，而不是只靠小切片 smoke
- 允许多卡并行实验；默认策略仍是多卡并行多个单卡实验，而不是默认多卡数据并行训练

## 成功标准

至少同时满足这些条件：

- current-head full-data `v2.4` train/eval artifact 已落地
- current-head full-data semantic-first local eval 维持高质量
- `slot assignment` 不再只是 plumbing，而是出现正的训练/在线信号
- `LongMemEval-S 64/128` 在 current-head 上形成 fresh `v2.4` artifact
- `LongMemEval-S 128` 明显高于 `v2.3` 基线
- `PersonaMem 128` 不明显退化
- 整体过程不引入任何 fallback / shortcut

## Milestone A：Freeze V2.3 Baseline

目标：把 `v2.3` closeout 变成 `v2.4` 的显式基线。

- 固化 `LongMemEval-S 128` 的 `v2.3` 基线
- 固化 `PersonaMem 128` guard 基线
- 固化 slot-assignment gain baseline

里程碑：

- `latest_longmemeval_stage2_slot_assignment_canary.json`
- `latest_personamem_stage2_slot_assignment_canary.json`
- `latest_stage2_slot_assignment_gain.json`

## Milestone B：Full-Data Learned Slot Assignment

目标：让 `observation -> slot` 的 learned 路线从 current-head plumbing 推进到 full-data 训练与评测。

- full-data slot-assignment supervision
- action classification / candidate ranking
- hard constraints 保持最小化 safety boundary

里程碑：

- `latest_stage2_v24_train.json`
- `latest_stage2_v24_eval.json`
- slot-assignment semantic metrics 非零

## Milestone C：Stronger Latent Main Path

目标：让更强的 latent 真正提升 online `memory -> belief -> answer`。

- 强化 learned query / slot / retrieval / belief 主链
- 减少 rule-heavy patching 的 retained 收益占比
- 保持 semantic-first

里程碑：

- current-head local eval 维持高质量
- `token_f1` / `field_f1` 不明显低于当前 full-data semantic-first baseline

## Milestone D：LongMemEval-S 64 Validation

目标：先在较快验证环上确认 `v2.4` 路线真的提升了 online 质量。

- current-head `LongMemEval-S 64`
- `v2.4` fresh canary
- 对比 `v2.3` retained baseline

里程碑：

- `latest_longmemeval_stage2_v24_canary.json`
- `sample_count >= 64`

## Milestone E：LongMemEval-S 128 Validation

目标：把收益推进到更大切片，避免 `64`-sample 偶然性。

- current-head `LongMemEval-S 128`
- `provider/local` 明显高于 `v2.3`
- refreshed analysis

里程碑：

- current-head `LongMemEval-S 128` artifact
- `provider_exact >= 10`
- `local_exact >= 10`
- `latest_longmemeval_stage2_v24_analysis.json`

## Milestone F：Cross-Benchmark Guard

目标：避免为了拉高 `LongMemEval-S` 把 `PersonaMem` 弄坏。

- current-head `PersonaMem 128`
- 保持 learned/semantic 主链
- 守住 guard

里程碑：

- `latest_personamem_stage2_v24_canary.json`
- `provider/local` 不明显低于 `v2.3` baseline

## Milestone G：Online Gain Closeout

目标：证明这轮 retained 收益确实来自更强的 online 主链，而不是文档或 artifact 刷新。

- current-head `v2.4` online gain artifact
- 相对 `v2.3` baseline 的显式 delta
- 增益必须为正

里程碑：

- `latest_stage2_v24_online_gain.json`
- `delta_provider_exact_match >= 4` 或 `delta_local_exact_match >= 4`

## 主指标

- `stage2_v24_longrun_score`

## 守卫

- `v2.3` closeout 证据链必须仍然存在
- stage-2 tests 持续通过
- 不允许重新引入 `fallback / shortcut`
- `PersonaMem 128` 不允许因为 `LongMemEval-S` 优化而明显崩坏
