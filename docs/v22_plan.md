# V2.2 Plan

## 目标

`v2.2` 的目标不再只是证明 semantic-first 方向成立，而是把它推进到更接近“完善的 CoRe Mem 框架/系统/模型 + 更强 latent”的状态：

- learned memory 主链在 **full-data** 训练下仍然成立
- semantic-first 的提升能够稳定转化到 **online benchmark**
- `LongMemEval-S` 不再只是弱 canary，而成为真正能检验 long-memory 能力的主 benchmark
- learned path 要在 current-head 上持续优于旧的 symbolic / rule-heavy 路线
- 整个系统继续保持 **no fallback / no shortcut**

## 硬约束

- 不做任何兜底/fallback/benchmark-specific shortcut
- benchmark 只作为验证和诊断信号，不作为投机优化对象
- retained 收益必须落在 `online memory -> belief -> answer` 主链
- 允许格式修复/约束，但不允许把外部修复伪装成模型本体收益
- 使用完整 public-data prepared tasks 进行训练与本地评测
- 优先使用 current-head artifacts；历史 best 只能作为参考，不能替代 fresh evidence

## 成功标准

至少同时满足这些条件：

- current-head 上有一条 **full-data semantic-first non-tiny** 训练证据链
- full-data trained eval 在 semantic-side 指标上继续保持高位
- current-head 上有 fresh `PersonaMem 128` semantic canary
- current-head 上有 fresh `LongMemEval-S 64` 和 `128` semantic canary
- current-head 上有 LongMemEval-S 的 fresh layered analysis
- 至少一份 explicit 的 learned-vs-symbolic online gain artifact 为正
- 以上全部在 **无 fallback / 无 shortcut** 的条件下成立

## Milestone A：Full-Data Semantic Training

目标：把 semantic-first 从小切片/保留 artifact 推进到 current-head full-data 训练。

- 使用完整 public-data prepared task rows
- 保持 `online_aligned`
- 训练输出需要显式记录 semantic-side metrics

里程碑：

- `latest_stage2_semantic_full_train.json`
- `latest_stage2_semantic_full_local_eval.json`
- `trained_eval.token_f1 >= 0.85`
- `retrieval_alignment.token_f1 >= 0.90`

## Milestone B：Current-Head Semantic Benchmarks

目标：把 semantic-first 收益扩展到 fresh current-head live canaries。

- `PersonaMem 128`
- `LongMemEval-S 64`
- `LongMemEval-S 128`

里程碑：

- `latest_personamem_stage2_semantic_canary.json`
- `latest_longmemeval_stage2_semantic_canary.json`
- two benchmark artifacts must match current HEAD

## Milestone C：LongMemEval-S Quality

目标：优先把第二 benchmark 拉起来，而不是只在 PersonaMem 上显得能用。

- fresh layered analysis
- retrieval / belief / provider 分层归因
- 优先修最关键的 1-2 个 error clusters

里程碑：

- `latest_longmemeval_stage2_semantic_analysis.json`
- `LongMemEval-S` semantic canary 在 `64 -> 128` 扩样后趋势不明显退化

## Milestone D：Learned-vs-Symbolic Evidence

目标：明确证明 learned path 的价值，不让收益只停留在 local intrinsic。

- 在相同 benchmark slice 上对比 learned vs symbolic
- 记录 `provider` / `local` delta
- 保持 current-head 可复验

里程碑：

- `latest_stage2_semantic_online_gain.json`
- positive delta on at least one retained online metric

## Milestone E：V2.2 Closeout

目标：把 `v2.2` 收口成一套更成熟的默认配方。

- 固化默认 full-data semantic-first train recipe
- 固化默认 semantic canary commands
- 更新文档为 `v2.2`

里程碑：

- one default full-data training command
- one default semantic benchmark command set
- one `v2.2` verifier

## 主指标

- `stage2_v22_completion_score`

## 守卫

- `run_experiment.py --verify-only` 持续通过
- stage-2 tests 持续通过
- `verify_stage2_latent_status >= 9`
- `verify_stage2_latent_core_quality >= 10`
- `verify_stage2_v21_semantic_model >= 17`
- 不允许因为新方法而重新引入 fallback / shortcut

