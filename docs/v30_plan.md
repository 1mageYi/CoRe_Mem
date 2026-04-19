# V3.0 / V30 Long-Run Plan

## 目标

这轮 `v30` 不再把主线放在 teacher 或 artifact completeness，而是直接围绕 **可训练主链本身**：

- 保持 `core / residual` 双银行结构不变
- 保持整体主链不变：`observation -> slot -> core/residual -> retrieval -> belief -> answer`
- 把当前 “一个共享 `flan-t5-base + LoRA` Seq2Seq 处理多个 task” 的训练形态，推进到 **shared backbone + task-specific adapters**
- 把 `encoder / resampler` 从固定投影推进到 **真正 trainable 的 latent module**
- 给 latent 加上 **直接训练目标**
- 把 belief 从 heuristic decoder 推向 **learned belief decoder**
- 用 **full benchmark holdout** 作为正式外部基准：
  - `LongMemEval-S = 500`
  - `PersonaMem = 589`

一句话说：`v30` 要验证的不是“系统还能不能继续补证据”，而是 **CoRe_Mem 是否能在不依赖 fallback/shortcut 的前提下，把可训练部分真正升级成更强的模型与 latent 主链**。

## 当前起点

当前 retained truth（`v2.9`）已经确认：

- `write / latent / belief` 三段 gain 都是正的
- `LongMemEval-S 500` 与 `PersonaMem 512` expanded holdout 已真实完成
- 训练主线当前仍是：
  - backbone: `google/flan-t5-base`
  - training style: `LoRA`
  - single shared `Seq2Seq` model for:
    - `retrieval_alignment`
    - `lifecycle_prediction`
    - `composition_to_belief`

同时也已经确认：

- `parser` 仍是 rule-first
- `lifecycle` 仍是 rule-heavy write policy
- `encoder / resampler` 当前是 parameterized but fixed projection
- `belief decoder` 仍主要是 heuristic/semantic-first，而不是强 learned decoder

## 冻结边界

- 不改 `core / residual`
- 不改整体 memory framework identity
- 不做任何 `fallback`
- 不做任何 `shortcut`
- 不做任何 `benchmark-specific heuristic`
- 不做任何 `benchmark leakage`
- benchmark 只作 holdout / regression / acceptance，不回流成训练 supervision
- 不再把 raw JSON exactness 当主目标

## 训练与评测锚点

### 32k 训练锚点

继续保持：

- `24k train`
- `4k val`
- `4k test`

source-level split 不变。

### full benchmark holdout

这轮起正式 external baseline 作用的是：

- `LongMemEval-S 500`
- `PersonaMem 589`

它们不是训练集，而是这轮最重要的 external holdout。

## 主线 A：Shared Backbone + Task-Specific Adapters

目标：

- 结束“一切 task 都塞进一个共享 Seq2Seq + 一个共享 LoRA”的形态
- 改成 **shared backbone + task-specific adapters/heads**

最小要求：

- 至少区分：
  - `write`
  - `latent/retrieval`
  - `belief`
- 可以共享 backbone
- 但不能再让三类任务只靠同一套 adapter 竞争容量

成功标准：

- internal compare 中，task-specific adapters 相对 current shared-only baseline 为正
- 至少一段主链指标有清晰增益，而不是只有 artifact 差异

## 主线 B：Trainable Latent Module

目标：

- 把 `encoder.py` / `resampler.py` 从 fixed projection 推向真正可训练的 latent module

重点：

- trainable query encoder
- trainable slot encoder
- trainable latent token generation
- trainable resampler / composer

要求：

- retained 收益不能只是 lexical tweak
- 必须能在 internal + holdout 上体现 latent 增益

## 主线 C：Direct Latent Objectives

目标：

- latent 不能只靠下游 Seq2Seq 间接学
- 必须有直接训练目标

可接受路线：

- retrieval ranking / contrastive objective
- support-slot selection objective
- latent-to-belief consistency
- composed-memory quality objective

要求：

- 必须形成单独 artifact，能和当前 shared Seq2Seq baseline 区分
- 必须有机械可验证的正增益

## 主线 D：Learned Belief Decoder

目标：

- belief 不再主要依赖 heuristic decoder

方向：

- learned belief item selection
- learned support attribution
- learned temporal conflict resolution
- semantic-first，不追 raw JSON exactness

要求：

- belief decoder 的 retained 收益必须来自 learned path
- holdout 上不能只体现在格式修复

## 次线

### parser exploratory track

- parser 可以继续探索，但不抢本轮主资源
- observation teacher / learned parser 只作为 exploratory track
- 不让 parser 决定本轮 stop condition

### teacher exploratory track

- `v2.8` 已证明当前 teacher label 定义存在真实风险
- teacher 只作为可选 supervision source
- 只有在 matched compare 上为正，才允许重新进入 retained 主线

## 里程碑

### M1. Formal Full-Benchmark Baseline

- current-head full holdout baseline 落地
- `LongMemEval-S 500`
- `PersonaMem 589`
- baseline summary 明确记录 benchmark 仅用于 holdout

### M2. Shared Backbone + Task Adapters

- current-head modular training config / artifact 落地
- task-specific adapters compare artifact 为正

### M3. Trainable Latent Landed

- current-head encoder / resampler trainable path 落地
- latent module train artifact 存在

### M4. Direct Latent Objective Positive

- latent objective eval artifact 为正
- current-head latent gain 为正

### M5. Learned Belief Decoder Positive

- belief decoder eval artifact 为正
- belief gain artifact 为正

### M6. Full Holdout Validation

- full benchmark holdout summary 通过
- `LongMemEval-S` 不低于 `v2.9`
- `PersonaMem` 不明显退化

## 成功标准

至少同时满足：

- task-specific adapters 相对 shared-only baseline 有真实正增益
- trainable latent module 已进入 current-head 主链
- direct latent objectives 带来真实正增益
- learned belief decoder 带来真实正增益
- full benchmark holdout baseline / refresh 已真实完成
- benchmark 仍只作为 holdout，不回流训练
- 全过程无 `fallback / shortcut / benchmark leakage`

## 当前总原则

这轮不是“再做更多 trick 来减少错误”，而是：

**直接增强可训练部分本身的能力。**

允许：

- 换更合适的模型结构
- 更复杂的训练方案
- 更强的 latent 目标
- 更模块化的训练架构

不允许：

- benchmark-facing trick
- provider fallback
- heuristic 伪装成模型能力提升
