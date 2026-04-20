# V3.2 / V32 Long-Run Plan

## 目标

这轮 `v32` 不再继续沿着 `v31` 的 Persona provider micro-tune 路线挖，而是直接转向一个更激进但仍然诚实的主线：

- 保持 `core / residual` 双银行结构不变
- 保持整体主链不变：`observation -> slot -> core/residual -> retrieval -> belief -> answer`
- 把当前“shared backbone + task-specific adapters + partially trainable latent”的系统，推进成 **latent-first modular learner**
- 把 `write / latent / belief / answer` 四段明确拆成更专业的可训练组件
- 让 `latent` 成为真正被直接训练和直接归因的 memory substrate
- 把 `PersonaMem` 的有限候选答案问题，从脆弱的 free-form label generation 升级成 **通用的 option scoring / answer head**
- 用 full benchmark holdout 继续做唯一外部裁判：
  - `LongMemEval-S = 500`
  - `PersonaMem = 512`

一句话说：`v32` 要验证的不是“还能不能再补一点 prompt/serialization trick”，而是 **CoRe Mem 是否能升级成以 trainable latent reader 为核心、由 write/belief/answer 模块协同训练的更强 memory system**。

## 当前起点

当前 retained truth（`v30`）已经确认：

- shared backbone + task-specific adapters 已 landed
- trainable encoder / resampler 已 landed
- direct latent objective 已给出正增益
- learned belief decoder 已给出正增益
- full benchmark holdout 已真实完成：
  - `LongMemEval-S = 500`
  - `PersonaMem = 589`

同时也已经确认：

- `v31` 在 internal `32k` compare 上，latent / belief / write 三段都能为正
- 但 `v31` 在 external holdout 上最终停在 soft blocker：
  - `LongMemEval-S 500` 只能追平 retained `v30`
  - `PersonaMem 512` 的 provider exact 仍低于 retained overlap guard
  - 连续三类 Persona micro-tune / support serialization / structured MCQ pivots 都未产生 keep

因此，`v32` 的出发点不是“继续 Persona 局部修补”，而是：

- 重新设计主训练架构
- 让 option scoring 变成通用 answer capability
- 让 latent 与 belief 直接承担更多解释力

## 冻结边界

- 不改 `core / residual`
- 不改整体 memory framework identity
- 不做任何 `fallback`
- 不做任何 `shortcut`
- 不做任何 `benchmark-specific heuristic`
- 不做任何 `benchmark leakage`
- benchmark 只作 holdout / regression / acceptance，不回流成训练 supervision
- 不把 raw JSON exactness 当主目标
- 不把 free-form label generation 的偶然涨分写成“模型变强”

## 训练与评测锚点

### 32k 训练锚点

继续保持：

- `24k train`
- `4k val`
- `4k test`

source-level split 不变。

### full benchmark holdout

这轮起正式 external gate 作用的是：

- `LongMemEval-S 500`
- `PersonaMem 512`

它们不是训练集，而是这轮最重要的 external holdout。

## 主线 A：Latent-First Modular Redesign

目标：

- 不再让一个共享生成头间接扛住全部任务
- 改成 **shared backbone + modular heads/modules**

最小模块化要求：

- `write head`
- `latent retrieval / support head`
- `belief head`
- `answer / option-scoring head`

要求：

- 不能只是“多几个 adapter 名字”
- 必须体现为更清晰的模块职责与 artifact

## 主线 B：Trainable Latent Reader

目标：

- 让 latent 不只是“可训练”，而是系统真正依赖的主读者

重点：

- stronger trainable query encoder
- stronger trainable slot encoder
- stronger resampler / composer
- support selection head
- latent-to-belief consistency
- composed-memory quality objective

要求：

- retained 收益不能只是 lexical tweak
- 必须能在 internal + holdout 上体现 latent 增益

## 主线 C：Structured Belief Head

目标：

- belief 不再主要依赖 heuristic decoder

方向：

- learned belief item selection
- learned support attribution
- learned temporal conflict resolution
- latent-conditioned belief recovery

要求：

- belief 提升必须能解释 benchmark 提升
- 不能只是 answer projection 小修补

## 主线 D：Answer / Option-Scoring Head

目标：

- 对有限候选答案任务，不再依赖脆弱的 free-form 单标签生成
- 升级成通用的 option scoring / answer head

方向：

- candidate scoring
- option ranking
- constrained best-option selection
- 与 belief/support facts 联动的 answer evidence scoring

要求：

- 这是通用能力，不是 PersonaMem-specific trick
- 必须以“候选集合上的可训练判别能力”来定义
- 必须形成独立 artifact 与 compare

## 主线 E：Write Head Strengthening

目标：

- write 不只是 local lifecycle 指标变高
- 而是能帮助 latent / belief / holdout

方向：

- stronger target-slot scoring
- harder overwrite / denser multi-fact writes
- cross-session consistency
- 与 latent / belief 联动的 write ablation

## 次线

### parser exploratory track

- parser 继续保留探索，但不抢本轮主资源
- raw-text parser / teacher 只作为 exploratory track
- 不让 parser 决定本轮 stop condition

### teacher exploratory track

- 当前 teacher 仍不重新进入 retained 主线
- 只有在 matched compare 上为正，才允许重新进入

## 里程碑

### M1. Formal Full-Benchmark Baseline

- retained `v30` full holdout baseline 固定
- `LongMemEval-S 500`
- `PersonaMem 512`

### M2. Modular Architecture Positive

- current-head modular architecture / shared-backbone train artifact 落地
- write / latent / belief / answer heads 结构已 landed
- architecture compare 为正

### M3. Trainable Latent Reader Positive

- current-head latent reader train artifact 落地
- latent module 已进入 current-head 主链

### M4. Direct Latent Objective Positive

- latent objective eval artifact 为正
- holdout compare 至少一条为正

### M5. Structured Belief Head Positive

- belief head eval artifact 为正
- belief holdout compare 为正

### M6. Answer / Option-Scoring Head Positive

- current-head answer/option-scoring eval artifact 为正
- finite-option holdout compare 为正

### M7. Full Holdout Improvement

- `LongMemEval-S 500` 至少一个关键 exact 指标高于 retained `v30`
- `PersonaMem 512` 明确高于 retained overlap guard，而不是只做到 non-regression

### M8. Ablation Truth

- 去掉 latent reader 的 ablation
- 去掉 structured belief head 的 ablation
- 去掉 option-scoring head 的 ablation

必须能回答：

- 哪一段是主要增益来源
- 哪一段只是辅助
- 哪一段仍然是当前 ceiling

## 成功标准

至少同时满足：

- modular architecture 相对 `v30` 为正
- write head 为正
- latent reader 为正
- direct latent objective 为正
- structured belief head 为正
- answer / option-scoring head 为正
- `LongMemEval-S 500` 明确高于 retained `v30`
- `PersonaMem 512` 明确高于 retained overlap guard
- ablation 明确表明增益主要来自 latent 主链，同时 answer head 对 finite-option holdout 有真实贡献
- 全过程无 `fallback / shortcut / benchmark leakage`

## 当前总原则

这轮不是继续微调 Persona 提示词，也不是继续靠 provider 幸运窗口。

这轮要验证的是：

**CoRe Mem 是否能从“带一些 trainable 组件的 memory system”，升级成“以 trainable latent reader 为核心、由 write / belief / answer 模块协同训练的更强 latent-first memory system”。**
