# V2.8 Teacher-Quality 32k Generalization Plan

## 目标

这轮 `v2.8` 的目标，不再先追新的 benchmark 包装分，而是先验证：

- 更高质量的 teacher labels 是否真的能提升 `32k` internal generalization
- `teacher-enhanced` 训练是否比当前 `silver/rule-derived` supervision 更强
- 在不改 `core / residual`、不引入任何 fallback / shortcut / benchmark leakage 的前提下，teacher-supervision 是否能真实提高 `observation / slot assignment / belief` 三段的 learned quality

## 冻结边界

- 不改 `core / residual`
- 不改 `core / residual` 双银行结构
- `observation -> slot -> core/residual -> retrieval -> belief -> answer` 的总体主链不改成别的框架
- benchmark 保持 holdout-only，不回流成训练 supervision

## 硬约束

- 不做任何 `fallback`
- 不做任何 `shortcut`
- 不做任何 `benchmark-specific heuristic`
- 不把 full benchmark label / answer / protocol 回流成训练 supervision
- 不把 raw JSON exactness 当成主优化目标
- 不靠 rule patch 作为 retained 主收益
- `32k` split 继续保持 source-record level，不允许 task expansion 后再切分

## 核心策略

### 1. 继续以 32k 作为锚点

当前锚点仍然是：

- `24k train`
- `4k val`
- `4k test`

### 2. 扩大 teacher 覆盖

先把 teacher 从 pilot 扩到中等规模：

- train `512`
- val `128`
- test `128`

三类目标仍然是：

- `observation`
- `slot assignment`
- `belief fields`

### 3. 修 observation teacher failure

当前 observation teacher 仍然是 `completed_with_failures`，`v2.8` 必须优先解决：

- malformed output
- field-missing output
- schema coercion failure

目标不是靠 fallback 混过去，而是提升 teacher 输出本身的 schema obey rate。

### 4. 建立 teacher-vs-silver 对照训练

在完全相同的 `32k` split 上做对照：

- `silver baseline`
- `teacher-enhanced`

对照重点：

- parser / observation semantic quality
- slot-assignment quality
- belief-field quality

### 5. internal test 优先于 full-data

只有当下面两件事同时成立，才考虑扩大 teacher coverage 或进入 full-data：

- `32k internal test` 上 teacher-enhanced 明显优于 silver baseline
- benchmark holdout 不退化，且最好继续提升

## 训练要求

- 训练优先使用 `gpu2`
- 必须记录：
  - wall-clock time
  - examples / sec
  - steps / sec
  - peak GPU memory
  - effective example count
- 训练配置要清楚区分：
  - `silver baseline`
  - `teacher-enhanced`

## 成功标准

至少同时满足：

- current-head `v2.8` teacher coverage artifacts 落地
- current-head `v2.8` teacher quality audit artifact 落地
- current-head `v2.8` teacher-vs-silver training artifacts 落地
- current-head `v2.8` teacher-vs-silver internal comparison artifact 落地
- observation teacher 成功率明显高于 `v2.7` pilot
- internal `32k test` 上 teacher-enhanced 至少一条主链指标真实优于 silver baseline
- benchmark holdout 不退化

## 主指标

- `stage2_v28_longrun_score`

## 守卫

- `v2.6` retained gain line 仍可复验
- `v2.7` retained 32k split / gpu2 pilot / teacher pilot artifact 仍可复验
- stage-2 tests 持续通过
- full benchmark 继续保持 holdout-only
- `PersonaMem 128` / `LongMemEval-S 128` 不允许因为 teacher-quality 试验而明显崩坏
