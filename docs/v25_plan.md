# V2.5 Generalization-First Long-Run Plan

## 目标

这轮 `v2.5` 的目标，不再是继续补“证据链是否完整”，而是把 `CoRe_Mem` 往更成熟的系统推进：

- 让 `LongMemEval-S` 从“能提升”推进到“更稳定、更有说服力的质量主 benchmark”
- 把 `learned slot assignment` 从“可用”推进到“更泛化、更鲁棒”
- 继续强化 `retrieval -> composed latent -> belief -> answer` 主链，让 retained 收益更多来自 stronger latent
- 在不牺牲跨 benchmark 泛化的前提下，扩展到更大切片乃至全量 benchmark 验证

## 硬约束

- 不做任何 `fallback`
- 不做任何 `benchmark-specific shortcut`
- 不做任何只对单一 benchmark 生效的 prompt hack，并把它记成 retained 收益
- 不使用 benchmark label / answer / protocol 输出作为训练 supervision
- benchmark 只作为 evaluation source，不作为 primary training source
- 保持 `semantic-first`
- 允许外部格式约束、schema repair、structured coercion，但收益必须回流到语义正确与 online 主链，而不是伪装成模型进步
- 训练默认继续使用完整 public-data prepared tasks
- 全量 benchmark 仅作为 holdout acceptance / generalization measurement，不进入训练闭环

## 成功标准

至少同时满足：

- current-head full-data non-tiny semantic-first train/eval 继续维持强质量
- `learned slot assignment` 在 current-head 上形成更强的 train/eval/gain 证据，而不只是 plumbing
- `LongMemEval-S 128` 明显高于 `v2.4` 当前 `10/128` 基线
- `PersonaMem 128` 不明显退化
- 至少完成一轮 current-head 的 **full benchmark** measurement，用于验证 larger-slice/generalization
- failure analysis 能稳定拆到 `slot_assignment / retrieval / belief / projection / provider`
- 整个过程不引入任何 fallback / shortcut / benchmark leakage

## Milestone A：Freeze V2.4 Quality Baseline

目标：把 `v2.4` closeout 变成 `v2.5` 的显式质量基线。

- 固化 current-head `LongMemEval-S 128` baseline
- 固化 current-head `PersonaMem 128` baseline
- 固化 current-head `v2.4` online gain baseline

里程碑：

- `latest_longmemeval_stage2_v24_canary.json`
- `latest_personamem_stage2_v24_canary.json`
- `latest_stage2_v24_online_gain.json`

## Milestone B：Generalized Learned Slot Assignment

目标：把 `observation -> slot` 的 learned 路线从“能工作”推进到“更泛化、更鲁棒”。

- 强化 action head：`merge / new / overwrite / ignore`
- 强化 candidate slot scoring / ranking
- 保留最小 hard constraints，仅做 safety boundary
- 训练目标优先服务 online slot assignment，不回退到 rule-heavy lifecycle patching

里程碑：

- current-head `slot_assignment` train artifact
- current-head `slot_assignment` eval artifact
- current-head `slot_assignment` gain artifact
- `slot_assignment` 本地指标稳定高于 `v2.4` retained line

## Milestone C：Stronger Latent Main Path

目标：继续加强 learned query / slot / retrieval / belief 主链，让 latent 真正成为核心能力。

- 提升 learned retrieval / rerank
- 提升 belief selection / composition
- 保持 semantic-first
- 优先减少 `LongMemEval-S` 上的 belief / projection failure

里程碑：

- current-head full-data local eval 继续维持高质量
- online gain 对 `LongMemEval-S` 为正
- gain 可以被 failure taxonomy 解释，而不是随机波动

## Milestone D：LongMemEval-S Quality Ramp

目标：把 `LongMemEval-S` 作为第一质量 benchmark 持续拉升。

- 先做 `64/128` current-head canary
- 再做更大切片验证
- 形成 refreshed layered analysis

里程碑：

- current-head `LongMemEval-S 64`
- current-head `LongMemEval-S 128`
- `LongMemEval-S` refreshed analysis artifact
- `LongMemEval-S 128` 高于 `v2.4` retained `10/128`

## Milestone E：Cross-Benchmark Guard

目标：避免只为了拉 `LongMemEval-S` 而破坏 `PersonaMem`。

- 持续刷新 current-head `PersonaMem 128`
- 保持 learned / semantic 主链
- 守住 `PersonaMem` 质量 guard

里程碑：

- current-head `PersonaMem 128`
- 不明显低于 `v2.4` retained baseline

## Milestone F：Full-Benchmark Holdout Evaluation

目标：用更大范围甚至全量 benchmark 作为 holdout acceptance，而不是局部 canary 自嗨。

- 在 current-head 达到 canary thresholds 后，跑更大切片 / full benchmark
- 优先 `LongMemEval-S full`
- 再看 `PersonaMem full` 或更大切片
- 只做 evaluation，不回流成训练 supervision

里程碑：

- current-head full benchmark run summary
- 对比 current-head vs `v2.4` baseline 的表格
- 不同 slice 下趋势稳定，不出现大样本崩塌

## Milestone G：V2.5 Packaging

目标：把这轮长跑收口成可复现的 `v2.5` recipe，而不是一串实验残片。

- 固化默认训练 recipe
- 固化默认 canary recipe
- 固化默认 full-benchmark recipe
- 固化 verifier / guard / no-shortcut contract

里程碑：

- 一个默认 full-data train recipe
- 一个默认 canary recipe
- 一个默认 full-benchmark evaluation recipe
- 一套 `v2.5` verifier / guard 套件

## 主指标

- `stage2_v25_longrun_score`

## 守卫

- `v2.4` retained line 仍然可复验
- stage-2 tests 持续通过
- 不允许重新引入 `fallback / shortcut / benchmark-specific heuristic`
- full benchmark 只作 holdout measurement，不得回流到训练
- `PersonaMem 128` 不允许因为 `LongMemEval-S` 优化而明显崩坏
