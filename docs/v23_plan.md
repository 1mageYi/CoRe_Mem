# V2.3 Plan

## 目标

`v2.3` 的目标不再是补齐证据链，而是把 `CoRe_Mem` 从“full-data semantic-first 原型已成立”推进到“更强、更稳、更依赖 learned latent 的系统”。

当前主攻方向有三条：

- 明显提升 `LongMemEval-S` 质量，而不是只保留 canary 级可运行性
- 把 `observation -> slot` 从 rule-heavy lifecycle 推进到 `learned slot assignment + hard constraints`
- 继续强化 `retrieval -> composed latent -> belief -> answer` 主链，让 latent memory 真正成为主收益来源

## 硬约束

- 不做任何兜底 `fallback`
- 不做任何 `benchmark-specific shortcut`
- 不把 provider prompt 小技巧当成 retained 收益
- retained 收益必须落在 `online memory -> belief -> answer` 主链
- 允许 semantic-first 的结构修复与格式约束，但不能把外部修复伪装成模型本体收益
- 训练与评测默认使用完整 public-data prepared tasks；benchmark 只做验证与诊断

## 成功标准

至少同时满足这些条件：

- `LongMemEval-S` 在 current-head extended canary 上明显高于当前 `v2.2` 基线
- `learned slot assignment` 已进入在线主链，并且不再完全依赖纯规则 `merge / overwrite / new`
- `learned slot assignment` 相比当前 symbolic lifecycle 有正的 online 或 local gain artifact
- `retrieval / belief / answer` 的 semantic-first 质量在 full-data 下继续保持，不因 slot-assignment learned 化而显著退化
- failure analysis 能把 `LongMemEval-S` 的错误稳定拆到 `slot assignment / retrieval / belief / projection / provider`

## Milestone A：LongMemEval-S 质量基线重建

目标：把 `LongMemEval-S` 从“机械完成”重新定义为“质量主 benchmark”。

- 固化 current-head `LongMemEval-S 128` semantic baseline
- 生成按错误层拆分的新分析
- 明确当前 top error clusters

里程碑：

- `latest_longmemeval_stage2_semantic_analysis.json` 刷新为分层版本
- 形成一份 top error clusters 摘要
- 固化 `v2.2` 的 `LongMemEval-S` 对照基线

## Milestone B：Learned Slot Assignment

目标：让 `observation -> slot` 的关键决策从 pure rule 迈向 learned。

- 保留 parser 作为 schema / safety 入口
- 新增 `slot assignment` learned scorer：
  - `action`: `merge / overwrite / new / ignore`
  - `target_slot`
  - `stale_old`
- 仅保留最小 hard constraints：
  - 不跨 entity 合并
  - 不跨明显 type/confidence 边界乱写
  - 不让低置信 observation 污染 core bank

里程碑：

- 新增 learned slot-assignment 模块与配置开关
- current-head 能跑 `learned_slot_assignment` online mode
- 至少 1 份 slot-assignment 对照 artifact

## Milestone C：在线对齐训练

目标：让训练直接服务 `slot assignment / retrieval / belief` 主链。

- `retrieval_alignment` 继续保持 semantic-first，但逐步向 ranking / support selection 靠拢
- 新增 slot-assignment supervision：
  - action classification
  - candidate-slot ranking
- 明确区分：
  - rule parser
  - learned slot assignment
  - learned belief recovery

里程碑：

- 至少 1 条 `online_aligned + slot_assignment` 训练证据链
- 至少 1 个 learned submodule 对 online 指标产生正增益
- full-data semantic eval 不明显退化

## Milestone D：LongMemEval-S 提升验证

目标：验证 learned slot assignment / stronger latent 是否真的改善 long-memory benchmark。

- current-head `LongMemEval-S 64`
- current-head `LongMemEval-S 128`
- learned-vs-symbolic 对照
- 错误分析刷新

里程碑：

- `LongMemEval-S 64/128` current-head artifacts
- `latest_stage2_longmemeval_slot_assignment_gain.json`
- retained gain 为正，且可解释

## Milestone E：V2.3 收口

目标：把新的 learned slot-assignment / stronger-latent 路线固化为默认实验主线。

- 默认训练命令
- 默认 `LongMemEval-S` 评测命令
- 默认 verifier
- 文档对齐到 `v2.3`

里程碑：

- 一套默认训练 recipe
- 一套默认 benchmark recipe
- 一套 `v2.3` verifier

## 主指标

- `stage2_v23_completion_score`

## 守卫

- `verify_stage2_v22_completion >= 19`
- stage-2 tests 持续通过
- 不允许重新引入 `fallback / shortcut`
- `PersonaMem 128` 不允许因为 LongMemEval-S 优化而明显崩坏
