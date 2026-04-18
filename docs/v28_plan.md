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

## 最新运行时更新

- current-head 已经完成 `raw observation teacher + matched teacher-vs-silver subset` 的真实 rerun
- 当前 matched manifests 会按 changed `raw-observation / slot-assignment / belief` sample_ids 取 source-record 对齐子集，确保训练/评测真正消费 teacher 改动
- 这轮 matched compare 没有转正，反而在 internal test 上记录：
  - `delta_internal_token_f1 = -0.05357191517996174`
  - `delta_internal_field_f1 = -0.08173076923076938`
  - `delta_internal_exact_match = -0.1826923076923077`
- 当前主退化项是 `lifecycle_prediction`，说明旧的“teacher 改动未被消费”假设已经被排除
- current-head 又已完成一轮 selective integration：保持 all-changed matched subset 不变、让 `lifecycle_prediction` 保持 silver、只应用 `observation + belief` teacher；结果仍为负，`delta_internal_token_f1 = -0.03262529332240871`、`delta_internal_field_f1 = -0.02564102564102566`、`delta_internal_exact_match = -0.10576923076923073`
- 这说明当前 launch 的真实 blocker 已进一步收敛为：`lifecycle` teacher 负增益、去掉 `lifecycle` 后 `belief` teacher 仍负增益，而 observation-only raw-teacher 改动只覆盖 matched subset 的 `train=1 / val=0 / test=3`
- 因此本轮 managed run 不应继续重复当前 label suite 的 refresh；若未来继续 `v2.8`，必须先更换 teacher 生成策略或标签定义

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
