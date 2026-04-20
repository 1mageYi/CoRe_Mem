# V3.3 / V33 Long-Run Plan

## 目标

这轮 `v33` 不再满足于 “learned 模块有辅助增益、但 full benchmark authoritative path 仍是 symbolic”。

这轮要验证的是：

- 保持 `core / residual` 双银行结构不变
- 保持整体主链不变：`observation -> slot -> core/residual -> retrieval -> belief -> answer`
- 但把 full benchmark authoritative path 从：
  - `memory_mode = symbolic`
  - `slot_assignment_mode = symbolic`
  进一步推进成：
  - `memory_mode = learned_memory`
  - `slot_assignment_mode = learned`
- 让 **learned write + temporal-semantic latent reader + belief graph head + answer option head**
  成为 full holdout 的正式主路径

一句话说：`v33` 要证明的不是 learned 模块“有帮助”，而是 **CoRe Mem 能否升级成以 learned latent system 为 authoritative runtime 的顶层系统**。

## 当前起点

当前 retained truth（`v32`）已经确认：

- modular backbone 为正
- write / latent / belief / answer 五条 internal 证据链为正
- ablation 已记录：
  - `latent_is_primary_driver = true`
  - `belief_contributes = true`
  - `answer_head_contributes = true`
- authoritative full holdout 已完成：
  - `LongMemEval-S 500 = provider/local 21/14`
  - `PersonaMem 512 = provider/local 183/175`

但当前 `v32` 仍然存在一个明确边界：

- authoritative full benchmark path 仍是：
  - `memory_mode = symbolic`
  - `slot_assignment_mode = symbolic`

因此，`v33` 的出发点不是继续扩大 symbolic baseline，而是：

- 把 learned runtime 真正扶正
- 让 temporal-semantic slot memory 与 learned belief graph 成为 authoritative full-holdout 路径

## 冻结边界

- 不改 `core / residual`
- 不改整体 memory framework identity
- 不做任何 `fallback`
- 不做任何 `shortcut`
- 不做任何 `benchmark-specific heuristic`
- 不做任何 `benchmark leakage`
- benchmark 只作 holdout / regression / acceptance，不回流成训练 supervision
- 不把 raw JSON exactness 当主目标
- 不把 prompt / serialization 偶然涨分写成“模型变强”

## 训练与评测锚点

### 32k 训练锚点

继续保持：

- `24k train`
- `4k val`
- `4k test`

source-level split 不变。

### full benchmark holdout

这轮的正式 external gate 是：

- `LongMemEval-S 500`
- `PersonaMem 512`

它们仍然是 holdout，不进入训练。

## 主线 A：Learned Authoritative Runtime

目标：

- current-head full benchmark official run 必须用 learned path 跑通

要求：

- `memory_mode = learned_memory`
- `slot_assignment_mode = learned`
- full holdout artifact 中明确记录上述两项
- 不能再用 symbolic authoritative path 收口

## 主线 B：Temporal-Semantic Latent Reader

目标：

- latent 不只是 retrieval scorer，而是支持 temporal-semantic slot reading 的主读者

重点：

- stronger trainable query encoder
- stronger trainable slot encoder
- stronger trainable resampler / composer
- temporal-semantic slot features
- support selection head
- latent-to-belief consistency
- composed-memory quality objective

要求：

- retained 收益不能只是 lexical tweak
- 必须能在 internal + holdout 上体现 latent 增益
- 必须形成 `temporal-slot` 独立 artifact

## 主线 C：Competition-Based Learned Write

目标：

- write 不再只是规则 lifecycle 的薄封装

方向：

- target-slot pointer / scorer
- operation head：`merge / overwrite / append / new / ignore`
- temporal reconciliation
- harder overwrite / denser multi-fact writes
- cross-session consistency

要求：

- write 提升必须能传导到 holdout
- 不能只停留在 local lifecycle 指标

## 主线 D：Belief Graph Head

目标：

- belief 不再主要依赖 heuristic decoder

方向：

- belief item selection
- relation node prediction
- support attribution
- temporal state resolution
- value extraction / copy
- belief graph consistency

要求：

- belief 提升必须能解释 benchmark 提升
- 不能只是 projection 小修补

## 主线 E：Answer / Option Head

目标：

- 对有限候选答案任务，建立通用的 answer / option head

方向：

- candidate scoring
- option ranking
- constrained best-option selection
- belief / support-aware answer evidence scoring

要求：

- 是通用能力，不是 PersonaMem-specific trick
- 必须形成独立 artifact 与 compare

## 次线

### parser exploratory track

- parser 继续保留探索，但不抢本轮主资源
- raw-text parser / teacher 只作为 exploratory track
- 不让 parser 决定 stop condition

### teacher exploratory track

- teacher 仍不重新进入 retained 主线
- 只有在 matched compare 上为正，才允许重新进入

## 里程碑

### M1. v32 Retained Truth Frozen

- retained `v32` full holdout baseline 固定
- `LongMemEval-S 500`
- `PersonaMem 512`
- authoritative path truth 明确记录为 symbolic

### M2. Learned Runtime Wiring Positive

- current-head learned authoritative runtime artifact 落地
- learned path 能跑 full holdout，不再只是 small probe

### M3. Learned Write Positive

- learned write eval 为正
- temporal / overwrite / cross-session harder cases 为正

### M4. Temporal-Semantic Latent Positive

- latent reader train artifact 落地
- temporal-slot eval artifact 为正
- latent objective eval 为正

### M5. Belief Graph Positive

- belief graph eval artifact 为正
- holdout compare 为正

### M6. Answer / Option Head Positive

- answer / option head eval artifact 为正
- finite-option holdout compare 为正

### M7. Learned Authoritative Holdout Wins

- current-head `LongMemEval-S 500` 明确高于 retained `v32`
- current-head `PersonaMem 512` 明确高于 retained `v32`
- authoritative holdout compare 必须来自 learned path

### M8. Ablation Truth

- 去掉 learned write 的 ablation
- 去掉 latent reader 的 ablation
- 去掉 belief graph head 的 ablation
- 去掉 answer head 的 ablation

必须能回答：

- latent 是否仍然是 primary driver
- belief graph 是否有独立贡献
- answer head 是否对 finite-option holdout 有独立贡献

## 成功标准

至少同时满足：

- learned authoritative runtime 成立
- learned write 为正
- temporal-semantic latent reader 为正
- direct latent objective 为正
- belief graph 为正
- answer / option head 为正
- `LongMemEval-S 500` 明确高于 retained `v32`
- `PersonaMem 512` 明确高于 retained `v32`
- authoritative full-holdout artifact 必须记录：
  - `memory_mode = learned_memory`
  - `slot_assignment_mode = learned`
- ablation 明确表明增益主要来自 latent 主链，同时 belief graph 与 answer head 有真实独立贡献
- 全过程无 `fallback / shortcut / benchmark leakage`

## 当前总原则

这轮不是继续在 symbolic runtime 上收集更多正结果，也不是继续做 provider-facing 局部修补。

这轮要验证的是：

**CoRe Mem 是否能从“symbolic-authoritative hybrid system”，升级成“learned-authoritative latent system”。**
