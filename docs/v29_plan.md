# V2.9 Long-Run Plan

## 目标

这轮 `v2.9` 不再把重点放在 teacher debugging 本身，而是把主线收紧为：

- 保持 `core / residual` 双银行结构不变
- 继续以 `32k` source-level split 作为训练锚点
- 依次推进 `write -> latent composition -> belief` 三段 learned 主链
- 在更大 holdout benchmark 上验证这些增益是否成立

一句话说：`v2.9` 要验证的不是“artifact 是否更全”，而是 **learned write、learned latent、learned belief 是否真的把 CoRe Mem 变强**。

## 冻结边界

- 不改 `core / residual`
- 不改整体主链：`observation -> slot -> core/residual -> retrieval -> belief -> answer`
- benchmark 继续只作 holdout，不回流成训练 supervision
- 不做任何 `fallback`
- 不做任何 `shortcut`
- 不做任何 `benchmark-specific heuristic`
- 不把 raw JSON exactness 当成主优化目标

## 数据与评测锚点

### 32k 训练锚点

继续使用：

- `24k train`
- `4k val`
- `4k test`

并保留 source-record level split，不允许 task expansion 后再切分。

### expanded holdout

这轮 benchmark evaluation 从历史的 `128` 级 canary 往上扩：

- `LongMemEval-S 128 -> 512`
- `PersonaMem 128 -> 512`

必要时再进入更大切片或 full holdout，但不在本轮起点就强制 full benchmark。

## 主线 A：Learned Write

目标：

- 把 `merge / new / overwrite / ignore` 的核心决策继续从 rule-heavy lifecycle 推向 learned policy
- 让 current-head 的 write gain 在 internal 与 holdout 上都能被机械确认

要求：

- 保留 hard constraints 作为 safety boundary
- 不靠 rule patch 当 retained gain
- 对照 current retained symbolic/silver baseline

## 主线 B：Stronger Latent

目标：

- 把当前偏固定投影的 `query encoder / slot encoder / composer` 推向真正可训练的 latent 主模块

要求：

- `encoder.py` 与 `resampler.py` 的 retained 收益必须来自 learned module，而不是仅靠 hand-crafted lexical tweaks
- latent 增益既要在 internal metric 上成立，也要通过 holdout guard

## 主线 C：Learned Belief

目标：

- 把 heuristic belief recovery 推向更强的 semantic-first learned belief module

要求：

- 继续保持 semantic-first
- belief gain 不能只体现在格式，而要体现在 belief semantic quality 与 online holdout

## 次线

### parser exploratory track

- `raw observation teacher / learned parser` 保留为并行探索线
- 但不再让 parser teacher 决定本轮主 stop condition

### teacher debugging

- `v2.8` 已证明当前 teacher label 定义存在负增益
- 这轮 teacher 只作为可选 supervision 来源，不再作为默认主线
- 只有在 matched compare 上出现正增益，teacher 才允许重新进入 retained 主线

## 里程碑

### M1. 32k baseline stable

- current-head `32k` 训练 / eval / timing artifacts 可复验
- `gpu2` 训练耗时、吞吐、显存记录稳定

### M2. write positive gain

- current-head `write` gain artifact 为正
- `LongMemEval-S` 不退化，最好转正

### M3. latent positive gain

- current-head `latent` gain artifact 为正
- `encoder / composer` 的 retained gain 能被机械确认

### M4. belief positive gain

- current-head `belief` gain artifact 为正
- belief semantic quality 与 holdout 不退化

### M5. expanded holdout

- current-head `LongMemEval-S 512` holdout summary 落地
- current-head `PersonaMem 512` holdout summary 落地
- `LongMemEval-S` 明确高于当前 retained `11/128` 基线趋势
- `PersonaMem` 不明显退化

## 成功标准

至少同时满足：

- `write` 有真实正增益
- `latent` 有真实正增益
- `belief` 有真实正增益
- `LongMemEval-S` 在 expanded holdout 上明显优于当前 retained baseline
- `PersonaMem` guard 不明显退化
- 全过程无 `fallback / shortcut / benchmark leakage`

## 主指标

- `stage2_v29_longrun_score`

## 守卫

- `v2.6` retained gain line 仍可复验
- `v2.7` 的 `32k split + gpu2 pilot` 仍可复验
- `v2.8` 的 blocked truth 仍可复验：
  - current teacher suite 已存在
  - current matched compare 为负
- expanded holdout 继续保持 holdout-only
