# V3.1 / V31 Long-Run Plan

## 目标

这轮 `v31` 不再以“把 architecture-first 路线立住”为目标，而是以：

**让 learned `write -> latent -> belief` 成为 full benchmark holdout 上的主要增益来源**

为唯一核心目标。

换句话说：

- `v30` 已经证明 shared backbone + task-specific adapters、trainable latent、direct latent objective、learned belief decoder 这条路线能够成立
- `v31` 要证明的是：这些提升不只是 internal / 32k 上的正增益，而是真正推动 external holdout 质量的主因

## 当前起点

当前 retained truth（`v30`）已经确认：

- `scripts/verify_stage2_v30_longrun.py --score-only = 41`
- `shared backbone + task-specific adapters` 已 landed
- `trainable encoder / resampler` 已 landed
- `direct latent objective` 已给出正增益
- `learned belief decoder` 已给出正增益
- full benchmark holdout 已真实完成：
  - `LongMemEval-S = 500`
  - `PersonaMem = 589`

同时也已经确认：

- full benchmark 当前仍主要承担 non-regression / external baseline 角色
- `LongMemEval-S` 与 `PersonaMem` 的 full holdout 提升还没有足够强到可以诚实地说“主要来自强 learned latent”

## 冻结边界

- 不改 `core / residual`
- 不改整体 memory framework identity
- 不做任何 `fallback`
- 不做任何 `shortcut`
- 不做任何 `benchmark-specific heuristic`
- 不做任何 `benchmark leakage`
- benchmark 只作 holdout / regression / acceptance，不回流成训练 supervision
- 不把 raw JSON exactness 当主目标

## 训练与评测锚点

### 32k 训练锚点

继续保持：

- `24k train`
- `4k val`
- `4k test`

source-level split 不变。

### full benchmark holdout

这轮继续保持 external holdout，但把 **稳定 keep gate** 明确成：

- `LongMemEval-S 500`
- `PersonaMem 512`

其中：

- `LongMemEval-S 500` 直接使用 full subset
- `PersonaMem 512` 使用固定 paired subset 作为稳定 gate
- `PersonaMem 589` 仍可继续生成 full report artifact，但不再作为 keep/discard 的首要判定口径

这些 benchmark 都是 external holdout，不是训练集。

## 主线优先级

### 主线 A：Latent Strengthening

这是 `v31` 的第一优先级。

目标：

- 让 latent 不只是 trainable，而是更强的 mainline module
- 让 latent 的提升在 full holdout 上可见

重点：

- stronger trainable query encoder
- stronger trainable slot encoder
- stronger resampler / composer
- latent-to-belief consistency objective
- composed-memory quality objective

要求：

- 不能只是 internal 指标变好
- 必须至少在一个 full holdout benchmark 上体现真实提升

### 主线 B：Belief Strengthening

这是第二优先级。

目标：

- belief 不只是“可以恢复”，而是更强、更稳、更能解释 benchmark 提升

重点：

- learned belief item selection
- learned support attribution
- learned temporal conflict resolution
- latent-conditioned belief recovery

要求：

- belief 提升必须能解释 benchmark 提升
- 不能只是 projection/格式层的小修补

### 主线 C：Write Strengthening

这是第三优先级。

目标：

- write 不只是 local lifecycle 指标变高
- 而是能帮助 full holdout 的系统行为

重点：

- stronger target-slot scoring
- harder overwrite / denser multi-fact writes
- cross-session consistency
- 与 latent / belief 联动的 write ablation

## 里程碑

### M1. Freeze V30 Baseline

- retained `v30` baseline 固定
- compare 只对 `v30`
- current holdout baseline 固定：
  - `LongMemEval-S 500`
  - `PersonaMem 512`

### M2. Latent Mainline Positive

- current-head latent strengthening artifact 落地
- latent internal 指标高于 `v30`
- latent holdout compare 为正

### M3. Belief Mainline Positive

- current-head belief strengthening artifact 落地
- belief internal 指标高于 `v30`
- belief holdout compare 为正

### M4. Write Mainline Positive

- current-head write strengthening artifact 落地
- write internal 指标高于 `v30`
- write holdout compare 为正或至少能够解释 non-regression

### M5. Full Holdout Improvement

- `LongMemEval-S 500` 至少一项关键质量指标高于 `v30`
- `PersonaMem 512` 不明显退化

### M6. Ablation Truth

- 去掉 latent strengthening 的 ablation
- 去掉 belief strengthening 的 ablation
- 去掉 task adapters / shared-only 的 ablation

必须能回答：

- 哪一段是主要增益来源
- 哪一段只是辅助
- 哪一段仍然是当前 ceiling

## 成功标准

至少同时满足：

- latent strengthening 为正
- belief strengthening 为正
- write strengthening 为正
- `LongMemEval-S 500` 明确高于 retained `v30`
- `PersonaMem 512` 不明显退化
- ablation 明确表明增益主要来自 learned 主链，而不是 benchmark-facing 改动
- 全过程无 `fallback / shortcut / benchmark leakage`

## 当前总原则

这轮不是继续补基础设施，也不是继续扩 teacher。

这轮要验证的是：

**CoRe_Mem 的真正增益，能不能开始主要来自更强的 learned latent 主链。**
