# TODO

## Doing

- `TD-042` `[doing]` 以 `v3.1 / v31 latent-first quality run` 为目标，在**不改 `core / residual` 双银行结构**的前提下，基于 retained `v30` baseline 继续推进 full holdout 上真正更强的 learned 主链。
  - Current focus:
    - latent strengthening
    - belief strengthening
    - write strengthening
    - ablation truth
    - full benchmark holdout：`LongMemEval-S 500 / PersonaMem 512`
  - Hard constraints:
    - no fallback
    - no shortcut
    - no benchmark-specific heuristic
    - no benchmark leakage
    - benchmark remains holdout-only
  - Runtime truth:
    - retained `v30` 已到 `41/41 keep`
    - current retained `v31` 仍是 `24/32`
    - `latest_stage2_v31_latent_mainline_train.json`、`latest_stage2_v31_latent_holdout_compare.json`、`latest_stage2_v31_belief_mainline_eval.json`、`latest_stage2_v31_belief_holdout_compare.json`、`latest_stage2_v31_write_mainline_eval.json` 与 `latest_stage2_v31_write_holdout_compare.json` 已落地
    - quick-smoke holdout line 仍停在 soft blocker，不能再作为 retained keep gate
    - commit `489ada0` 已让 learned full-holdout runner 复用 `latent_slot_ranker` 并按 batch 增量落盘；此前停在 `completed_predictions = 0` 的 resumed runs 现已恢复推进
    - current HEAD `838a861` 又补齐了 `scripts/verify_stage2_v31_longrun.py` 的 authoritative full-holdout publisher；对应 targeted test 与 configured guard 已通过
    - 截至当前检查：`outputs_v2/runs/20260419T160705Z_stage2_memory_canary_longmemeval/` 已推进到 `322/500`，`outputs_v2/runs/20260419T160701Z_stage2_memory_canary_personamem/` 已推进到 `238/512`
  - Truth boundary:
    - 当前 truth 只代表 full-holdout stable gate 已从 true blocker 恢复到 active measurement，不代表 `v31` full holdout gain 已成立
    - `latest_longmemeval_stage2_v31_full.json`、`latest_personamem_stage2_v31_full.json`、`latest_stage2_v31_full_holdout_compare.json` 与 `latest_stage2_v31_ablation_summary.json` 仍未发布，因此 `scripts/verify_stage2_v31_longrun.py --score-only` 继续是 `24`
  - Next step:
    - 等当前 `500 + 512` resumed runs 完成后，立即发布 `v31` full-holdout artifacts
    - 在 holdout compare 到位后，再决定是否已有足够证据发布 `v31` ablation summary，或是否需要单独补一轮 ablation evidence

- `TD-039` `[doing]` 以 `v2.8 teacher-quality long-run` 为目标，在**不改 `core / residual` 双银行结构**的前提下，继续以 `32k` source-level split 为锚点，把 teacher-supervision 做成真正可比较、可泛化的训练资产。
  - Runtime truth: `TD-038 / WS-024` 已在 current HEAD `ac84cc1` 上完成，`scripts/verify_stage2_v27_longrun.py --score-only = 26`
  - Current focus:
    - 保持 current-head `256 / 128 / 128` teacher suite、matched manifests 与 holdout guard 可复验
    - 记录当前 launch 已到达的 true blocker，而不是继续重复 refresh
    - 若后续恢复 `TD-039`，必须先提出新的 teacher-generation hypothesis
    - benchmark 继续只作 holdout guard
  - Target teacher scale:
    - train `512`
    - val `128`
    - test `128`
  - Hard constraints:
    - no fallback
    - no shortcut
    - no benchmark-specific heuristic
    - no benchmark leakage
    - full benchmark 只作 holdout evaluation
  - Truth boundary:
    - 暂不进入 full-data
    - 当前成功定义不是 artifact completeness，而是 teacher-enhanced 在 `32k` internal test 上真实优于 silver baseline
  - Latest runtime truth:
    - current-head teacher suite 已以 `256 / 128 / 128` caps 完成，observation teacher 当前 `success_rate = 1.0`
    - `raw observation teacher + matched teacher-vs-silver subset` 已真实跑完；当前 matched manifests 按 changed `raw-observation / slot-assignment / belief` sample_ids 取 source-record 对齐子集，确保训练/评测真正消费 teacher 改动
    - same-split matched non-tiny compare 已完成；fresh compare 当前记录 `delta_internal_token_f1 = -0.05357191517996174`、`delta_internal_field_f1 = -0.08173076923076938`、`delta_internal_exact_match = -0.1826923076923077`
    - 当前主退化项是 `lifecycle_prediction`：val `token_f1` 从 `0.9655172413793104` 降到 `0.8004926108374386`，test 从 `0.9285714285714286` 降到 `0.8482142857142859`
    - selective integration 也已验证：在 all-changed matched subset 上让 `lifecycle_prediction` 保持 silver、只应用 `observation + belief` teacher 后，fresh compare 仍为负，`delta_internal_token_f1 = -0.03262529332240871`、`delta_internal_field_f1 = -0.02564102564102566`、`delta_internal_exact_match = -0.10576923076923073`
    - 剩余可单独保留的 raw-observation teacher 改动只覆盖 matched subset 的 `train=1 / val=0 / test=3`
    - current-head `scripts/verify_stage2_v28_longrun.py --score-only = 32 / 34`
    - `research-results.tsv` / `autoresearch-state.json` 当前已依次记录 iteration `1 discard`、iteration `2 discard` 与 iteration `3 blocked`
  - Current blocker:
    - “teacher 改动未被消费” 已不是当前 blocker；真实 blocker 是：`lifecycle` teacher 负增益、去掉 `lifecycle` 后 `belief` teacher 仍负增益、而 observation-only raw-teacher 改动又过于稀疏。当前 launch 因此停在 blocked，不能写成 keep，也不能进入 full-data

- `TD-038` `[doing]` 以 `v2.7 32k teacher-first long-run` 为目标，在**不改 `core / residual` 双银行结构**的前提下，先建立 `32k` source-level split、teacher-labeled data-quality upgrade、internal generalization test 与 `gpu2` 训练耗时基线。
  - Runtime truth: `TD-037 / WS-023` 已在 current HEAD `cfbdc08` 上完成；`TD-038` 当前作为 retained `32k` split / teacher pilot baseline 保留
  - Current progress:
    - current HEAD `21dd40b` 已新增真实 `32k` source-level split / manifest / audit pipeline
    - `latest_stage2_v27_32k_split.json`、`latest_stage2_v27_32k_manifest.json` 与 `latest_stage2_v27_32k_audit.json` 已落地
    - fresh background run 已通过 helper 初始化 `research-results.tsv` 与 `autoresearch-state.json`；`autoresearch_resume_check.py --repo ...` 当前返回 `full_resume`
    - current HEAD `6333689` 已补齐 `latest_stage2_v27_train.json`、`latest_stage2_v27_eval.json`、`latest_stage2_v27_training_timing.json`、`latest_stage2_v27_internal_test.json` 与 `latest_stage2_v27_holdout_summary.json`
    - `gpu2` tiny pilot 当前真实记录为：`4096` effective examples、`512` steps、wall-clock `5.420951s`、`755.59 examples/s`、peak GPU memory `55.09MB`
    - 当前又已补齐 `latest_stage2_v27_teacher_observation.json`、`latest_stage2_v27_teacher_slot_assignment.json` 与 `latest_stage2_v27_teacher_belief.json`
    - 当前 teacher pilot 使用真实 `MiniMax-M2.7`、sample caps `8/2/2` 与 batch size `1`；`slot_assignment` 与 `belief` artifact 为 `completed`，`observation` artifact 为 `completed_with_failures`
    - fresh current-head `scripts/verify_stage2_v27_longrun.py --score-only = 26`
  - Core requirements:
    - `32k` 作为锚点；先做 `24k train / 4k val / 4k test`
    - 训练前先做 data-quality audit
    - teacher 使用 `MiniMax-M2.7`
    - 训练优先使用 `gpu2`
    - 记录 wall-clock / throughput / memory，判断 `32k` 是否过重
  - Hard requirements:
    - no fallback
    - no shortcut
    - no benchmark-specific heuristic
    - no benchmark leakage
    - full benchmark 只作 holdout evaluation
    - 只有 `32k` internal test work well 后，才允许进入 full-data
  - Truth boundary:
    - 当前 `26/26` 是由 `32k split + gpu2 tiny pilot + teacher pilot artifacts` 共同满足的机械 stop condition
    - 不能把当前完成态误写成 full `32k` teacher coverage 已完成，也不能误写成 teacher-conditioned `gpu2` retrain 已完成
    - `latest_stage2_v27_teacher_observation.json` 当前显式记录 `total_labeled_examples = 6`、`total_failed_examples = 6`，因此 observation teacher 仍是 pilot quality

- `TD-036` `[done]` 以 `v2.5 learned core-path long-run` 为目标，在**不改 `core / residual` 双银行结构**的前提下，继续推进 `LongMemEval-S` 质量、`write / retrieve / belief` 三段的 learned 化、learned slot assignment 泛化鲁棒性、更强 latent 与 full benchmark holdout evaluation。
  - Runtime truth: `TD-035 / WS-021` 已在 current HEAD `12a9a80` 上达到 `stage2_v24_longrun_score = 24/24`，当前作为 `v2.5` retained baseline 保留
  - Current managed-run state: current HEAD `3036e3d` 已把 `stage2_v25_longrun_score` 推到 stop condition `24/24`
  - Evidence shape: 当前新增的是 `v2.5` baseline/package artifact suite 与 `write / retrieve / belief` baseline decomposition artifacts；三段 artifact 均显式记录 `delta_vs_v24_retained = 0`
  - Truth boundary: 当前不能把 `24/24` 误写成真实 `LongMemEval-S` / online quality gain 已经超过 `v2.4`
  - Hard constraints:
    - no fallback
    - no shortcut
    - no benchmark-specific heuristic
    - full benchmark is holdout evaluation, not training supervision
  - Plan: [docs/v25_plan.md](/media/storage/mingjing/workspace/CoRe_Mem/docs/v25_plan.md)


## Backlog

- `TD-007` `[backlog]` 为 PersonaMem 128k / 1M 提供启动开关。
  - Reason: 第一阶段只要求保留启动能力。

- `TD-018` `[backlog]` 建立第二阶段 benchmark canary protocol 与结果记录。
  - Reason: `gpu3 + tiny backend` 的 stage-2 本地 train/eval/ablation matrix 已全部完成并登记；下一步若继续推进，更应该把这个完成态沉淀为 canary 结果和对比表。
  - Evidence target: PersonaMem 64 / LongMemEval-S 64 的固定 canary 运行结果、输出表和记录规范。

- `TD-025` `[backlog]` 在 `PersonaMem 64` canary 质量提升后，扩大 stage-2 benchmark 范围与结果对比。
  - Reason: `TD-029 / WS-015` 已完成 learned-memory-first plumbing；当前更合理的后续工作是扩大 learned-mode canary 范围、补结果对比，而不是重复补 toggle/checkpoint plumbing。
  - Evidence target: 更大样本或第二 benchmark 的 stage-2 live canary 结果与对比表。

## Blocked

- `TD-012` `[blocked]` 补齐 stage-1 外部前置条件。
  - Reason: 当前可用的 Gemini key/provider 组合在 formal benchmark 负载下仍连续触发 `HTTP 429`，且 formal benchmark 在用户明确要求前不主动继续推进。
  - Evidence target: 可持续推进 formal benchmark 的 provider 配额、稳定 key，或用户批准的其他可用 provider。

- `TD-013` `[blocked]` 将两个 benchmark 从 1-sample real run 推进到正式全量 protocol 运行。
  - Reason: 当前只能证明最小链路打通，尚不足以满足 AC-002 / AC-003 的正式运行要求；同时该项被用户触发条件与 provider blocker 双重约束。
  - Evidence target: PersonaMem 与 LongMemEval-S 在正式范围内完成可重复结果运行。
  - Current evidence: runner 已具备增量落盘与续跑能力；Gemini 路径已把 PersonaMem formal run 推进到 `22/589`、把 LongMemEval formal run 推进到 `19/500`，但超保守单样本检查仍连续触发 `HTTP 429`，说明当前 key/provider 组合已构成真实外部 blocker。

## Done

- `TD-040` `[done]` 以 `v2.9 learned-core-path long-run` 为目标，在**不改 `core / residual` 双银行结构**的前提下，继续以 `32k` source-level split 为锚点，依次推进 `write -> latent composition -> belief` 三段 learned 主链，并把 holdout benchmark 扩到 `LongMemEval-S 500 / PersonaMem 512`。
  - Reason: 当前 retained state 已达到 `scripts/verify_stage2_v29_longrun.py --score-only = 39`
  - Evidence target:
    - `outputs_v2/artifacts/latest_stage2_v29_write_gain.json`
    - `outputs_v2/artifacts/latest_stage2_v29_latent_gain.json`
    - `outputs_v2/artifacts/latest_stage2_v29_belief_gain.json`
    - `outputs_v2/artifacts/latest_stage2_v29_holdout_summary.json`
    - `outputs_v2/artifacts/latest_longmemeval_stage2_v29_canary.json`
    - `outputs_v2/artifacts/latest_personamem_stage2_v29_canary.json`
    - `research-results.tsv` iteration `3 keep`

- `TD-040` `[done]` 以 `v2.9 learned-core-path long-run` 为目标，在**不改 `core / residual` 双银行结构**的前提下，继续以 `32k` source-level split 为锚点，依次推进 `write -> latent composition -> belief` 三段 learned 主链，并把 holdout benchmark 扩到 `LongMemEval-S 500 / PersonaMem 512`。
  - Reason: 当前 retained state 已达到 `scripts/verify_stage2_v29_longrun.py --score-only = 39`；`write / latent / belief` 三段 gain 全部转正，expanded holdout 已真实完成，launch guard 通过
  - Evidence target:
    - `outputs_v2/artifacts/latest_stage2_v29_write_gain.json`
    - `outputs_v2/artifacts/latest_stage2_v29_latent_gain.json`
    - `outputs_v2/artifacts/latest_stage2_v29_belief_gain.json`
    - `outputs_v2/artifacts/latest_stage2_v29_holdout_summary.json`
    - `outputs_v2/artifacts/latest_longmemeval_stage2_v29_canary.json`
    - `outputs_v2/artifacts/latest_personamem_stage2_v29_canary.json`
    - `research-results.tsv` iteration `3 keep`

- `TD-037` `[done]` 以 `v2.6 gain-first long-run` 为目标，在**不改 `core / residual` 双银行结构**的前提下，要求 `write / retrieve / belief` 至少一段出现真实正增益，并要求 `LongMemEval-S 128` 明确超过 `v2.5` retained baseline。
  - Reason: 该条线已在 current HEAD `cfbdc08` 上机械达成 `stage2_v26_longrun_score = 26/26`
  - Evidence target:
    - `LongMemEval-S 128 = 11 / 11`
    - `PersonaMem 128 = 44 / 33`
    - `write_gain.positive_gain = true`
    - `belief_gain.positive_gain = true`

- `TD-001` `[done]` 初始化项目文档系统并建立根契约、真源文档和状态文档。
- `TD-002` `[done]` 用 `environment.yaml` 在默认 conda envs 目录创建并固定 conda 环境 `core_mem`。
- `TD-003` `[done]` 建立基础目录结构、配置机制与输出目录规范。
- `TD-004` `[done]` 确定第一阶段 pretrained embedding model 为 `sentence-transformers/all-MiniLM-L6-v2`。
- `TD-005` `[done]` 实现阿里云 OpenAI-compatible provider adapter 第一版。
- `TD-006` `[done]` 接入 PersonaMem 32k 官方主任务与评测协议。
- `TD-008` `[done]` 接入 LongMemEval-S 官方协议。
- `TD-009` `[done]` 实现 vanilla CoRe Memory 核心模块与文本 embedding 接口。
- `TD-010` `[done]` 建立实验总入口与分步骤脚本。
- `TD-011` `[done]` 建立 unit tests 与最小 E2E smoke test。
- `TD-014` `[done]` 锁定第二阶段 `V2.0 structured latent-slot memory` 方案并同步真源与状态文档。
  - Reason: 第二阶段已由用户确认，需要进入文件级 runtime truth。
  - Evidence target: `docs/requirements.md`、`docs/v2_design.md` 与相关状态文档同步到 stage-2 真相。
- `TD-016` `[done]` 实现第二阶段 `V2.0` 主线模型骨架并打通直训链路。
  - Reason: 当前主线骨架、训练 runtime、公开数据规范化、严格 source prepare 与 direct-train launcher 已全部连通。
  - Evidence target: `src/core_mem/v2` 下主线模块、真实 public-data prepared manifest、repo-local cache 训练配置与 execute-train 证据。
- `TD-019` `[done]` 获取并接入第二阶段真实公开数据源文件。
  - Reason: 五个目标公开数据源已下载、规范化并通过 preflight 与 strict prepare 验证。
  - Evidence target: `stage2_data_preflight.py` 不再报 missing，且 `prepare_stage2_data.py --strict-sources` 能产出非 demo manifests。
- `TD-017` `[done]` 建立第二阶段本地 intrinsic evaluation 的完整指标、budget sweep 与 experiment registry。
  - Reason: local eval 已从 skeleton 扩展到模块级/家族级/budget-sweep 评测体系，并已被实际用于 `mainline + 11` 个必做 ablation 的 `gpu3` 本地 train/eval 登记。
  - Evidence target: `outputs_v2/artifacts/stage2_experiment_index.json`、`outputs_v2/evals_local/`、`outputs_v2/tables/` 与 `scripts/verify_stage2_experiment_status.py --score-only = 13`。

- `TD-021` `[done]` 把第二阶段真正的 latent memory 主链路做实。
  - Reason: query/slot encoder 已不再是 hash-only，resampler 已不再是 mean-only，decoder 已真实消费 `composed_memory`，且 `StructuredMemorySystem.query()` 已把 composed latent 传入 belief decode 主链。
  - Evidence target: `scripts/verify_stage2_latent_status.py --score-only = 9` 中与 latent path 对应的 7 个实现检查全部通过。

- `TD-022` `[done]` 建立 stage-2 memory-mediated benchmark canary runner，并接入可用 provider 配置。
  - Reason: `scripts/run_stage2_memory_canary.py` 已落地，且 canary artifact/config snapshot/traceability 已成立。
  - Evidence target: `run_stage2_memory_canary.py` 可生成 stage-2 benchmark canary run 与 summary artifact。

- `TD-023` `[done]` 完成 stage-2 MiniMax memory canary 的 live provider 首次运行。
  - Reason: `MiniMax-M2.7` 已完成 1-sample live canary 和 `PersonaMem 64` live canary。
  - Evidence target: 至少一条 live MiniMax-M2.7 的 stage-2 canary 运行结果。

- `TD-026` `[done]` 以系统/模型/latent 本体更强、更稳健为锚点，提升 stage-2 local intrinsic 质量，并把 `PersonaMem 64 >= 9/10` 作为不退化 guard。
  - Reason: `stage2_latent_core_quality_score` 已从本轮 baseline `6/10` 提升到 `10/10`；belief recovery 现直接消费 lifecycle-ordered memory state，且 local intrinsic belief-family 与 retrieval-family 已解耦到更符合主链分层的评测方式。
  - Evidence target: `scripts/verify_stage2_latent_core_quality.py --score-only = 10`，且 retained PersonaMem 64 guard artifact 仍满足 `9/10`。

- `TD-020` `[done]` 为默认 `flan-t5-base` backbone 补非 tiny `gpu3` 训练证据。
  - Reason: 当前已完成一条默认 `configs/stage2_train.yaml` 的真实 `GPU3` 非 tiny train/eval 证据链，并形成正的 `trained_eval.token_f1`。
  - Evidence target: `outputs_v2/runs/20260415T043648Z_stage2_train_exec/execution_summary.json`、`outputs_v2/checkpoints/20260415T043648Z_stage2_train_exec/`、`outputs_v2/evals_local/20260415T043706Z_stage2_local_eval.json`。

- `TD-027` `[done]` 推进完整的 `v2`：fresh live canary、第二 benchmark、非 tiny 训练证据，以及去除 benchmark shortcut/fallback。
  - Reason: 当前 `stage2_v2_completion_score = 14/14`；fresh `PersonaMem 64`、`LongMemEval-S 64`、`latest_longmemeval_stage2_canary_analysis.json`、`no-shortcut runner` 与默认 `flan-t5-base` 的 `GPU3` 非 tiny train/eval 已全部机械成立。
  - Evidence target: `scripts/verify_stage2_v2_completion.py --score-only = 14`，且 `scripts/verify_stage2_latent_core_quality.py --score-only = 10`、stage-2 guard 通过。

- `TD-029` `[done]` 把 `v2.1` 的当前主线切到 learned-memory-first / better latent。
  - Reason: online learned path、checkpoint-backed belief、online-aligned training 语义和最小 current-head learned-mode artifact 已成立。
  - Evidence target: `scripts/verify_stage2_v21_learned_memory.py --score-only = 12`

- `TD-032` `[done]` 把 semantic-first 路线推进到 full-data current-head 训练、extended semantic canaries、LongMemEval-S semantic analysis 与 semantic online gain。
  - Reason: current-head full-data semantic train/eval、`PersonaMem 128` / `LongMemEval-S 64/128` semantic canaries、fresh semantic analysis 与 semantic online gain 已全部补齐。
  - Evidence target: `scripts/verify_stage2_v22_completion.py --score-only = 19`

- `TD-033` `[done]` 以 `LongMemEval-S` 质量提升、learned slot assignment 与 stronger latent 为锚点推进 `v2.3`。
  - Reason: 这条主线已由后续 `TD-034 / WS-020` 机械收口；当前不再是 active 主线。
  - Evidence target: `scripts/verify_stage2_v23_longrun.py --score-only = 22`

- `TD-035` `[done]` 以 `v2.4 quality-first long-run` 为锚点推进 `LongMemEval-S` 质量、full-data learned slot assignment 与 stronger latent。
  - Reason: managed autoresearch run 已在 current HEAD `12a9a80` 上触发 stop condition，`stage2_v24_longrun_score = 24/24`
  - Evidence target:
    - `scripts/verify_stage2_v24_longrun.py --score-only = 24`
    - `outputs_v2/artifacts/latest_stage2_v24_eval.json`：`trained_eval.token_f1 = 0.9991150844073334`、`trained_eval.field_f1 = 0.9976704786107581`
    - `outputs_v2/artifacts/latest_longmemeval_stage2_v24_canary.json`：`provider_exact = 10`、`local_exact = 10`
    - `outputs_v2/artifacts/latest_personamem_stage2_v24_canary.json`：`provider_exact = 38`、`local_exact = 28`
    - `outputs_v2/artifacts/latest_stage2_v24_online_gain.json`：`delta_provider_exact_match = +4`、`delta_local_exact_match = +4`
    - `research-results.tsv` / `autoresearch-state.json`：best iteration `7`、best/current metric `24`

- `TD-036` `[doing]` 以 `v2.5 generalization-first long-run` 为锚点，把当前主线推进到更强的 `LongMemEval-S`、更泛化的 learned slot assignment、更强的 online latent，以及更大切片 / full benchmark holdout measurement。
  - Reason: `v2.4` 已把 quality-first 机械目标推到 `24/24`，下一步最值得做的不再是补 closeout artifact，而是在**不动 `core / residual`** 的前提下，把 `write / retrieve / belief` 三段进一步 learned 化
  - Evidence target:
    - current-head 更强的 `LongMemEval-S 64/128` retained line
    - generalized learned `write / slot-assignment`、`retrieve`、`belief` train/eval/gain artifact
    - current-head `PersonaMem 128` 不明显退化
    - full benchmark holdout evaluation summary
  - Current runtime note: 本轮 managed run 已机械达到 `24/24`，但当前 retained evidence 仍属于 baseline/package closeout，不是新的 positive-gain closeout



## Verified

- `TD-003` `[verified]` 目录结构、配置加载、输出规范、run metadata 写入与 benchmark dry-run 可运行。
- `TD-002` `[verified]` `core_mem` 环境已在默认 conda envs 目录创建，并通过 `python --version`、`pytest` 与 `run_experiment.py --verify-only` 复验。
- `TD-004` `[verified]` embedding 配置、adapter 和环境依赖已接入；`sentence-transformers 5.3.0` 可在 `core_mem` 环境中 import。
- `TD-005` `[verified]` Provider adapter 通过单测，支持 OpenAI-compatible 请求构造与响应解析。
- `TD-010` `[verified]` 最小 `prepare / evaluate / make tables` 分步骤脚本已落地并有 smoke tests。
- `TD-009` `[verified]` vanilla CoRe Memory 已具备 writer / updater / residual manager / reader / text observe/query 最小闭环。
- `TD-014` `[verified]` 第二阶段真源、设计文档、stage2 verifier/acceptance skeleton 与 `outputs_v2/` 目录骨架已同步到当前 runtime truth。
- `TD-015` `[verified]` 第二阶段 observation / belief / parser / dataset skeleton、prepare/train/eval/canary 脚本与首批 `outputs_v2/` artifacts 已落地。
  - Reason: 当前已不再是纯 verifier skeleton 状态，而是 stage2 pipeline 首批可运行实现。
  - Evidence target: `src/core_mem/v2/`、`configs/stage2_train.yaml`、四个 stage2 脚本、`outputs_v2/` artifacts 与对应测试。
- `TD-016` `[verified]` 第二阶段 `V2.0` 主线模型骨架与直训链路已进入可直接训练状态。
  - Reason: 真实 public-data source -> normalize -> strict prepare -> direct-train launcher -> execute-train tiny proof 已连续通过。
  - Evidence target: `outputs_v2/artifacts/stage2_prepared_samples_manifest.json`、`outputs_v2/runs/20260414T043221Z_stage2_train_plan/launch_stage2_training.sh`、`outputs_v2/runs/20260414T043225Z_stage2_train_exec/training_metrics.json`。
- `TD-019` `[verified]` 第二阶段真实公开数据源文件已接入。
  - Reason: `SGD`、`MultiWOZ 2.4`、`Persona-Chat`、`MQUAKE`、`ReCoE` 现均已具备 repo-local raw source 与对应 `normalized.jsonl`。
  - Evidence target: `scripts/stage2_data_preflight.py --json` 返回 `missing=[]`，且 `normalize_stage2_public_data.py` 已产出 5 份 `normalized.jsonl`。
- `TD-021` `[verified]` 第二阶段 latent memory 主链路已满足当前机械 readiness 目标。
  - Reason: `stage2_latent_readiness_score` 已从 `2/9` 提升到 `9/9`，且 stage-2 guard 与新增 canary test 均已通过。
  - Evidence target: `research-results.tsv` iteration `1`、`outputs_v2/evals_benchmark/20260414T170606Z_stage2_memory_canary.json`、`scripts/verify_stage2_latent_status.py --score-only = 9`。

- `TD-022` `[verified]` Stage-2 memory-mediated benchmark canary runner 已接上 benchmark path。
  - Reason: blocked artifact、1-sample live artifact 与 64-sample live artifact 已依次落地。
  - Evidence target: `outputs_v2/evals_benchmark/20260414T170606Z_stage2_memory_canary.json`、`outputs_v2/evals_benchmark/20260414T231441Z_stage2_memory_canary.json`、`outputs_v2/evals_benchmark/20260414T231617Z_stage2_memory_canary.json`。

- `TD-023` `[verified]` Stage-2 MiniMax memory canary 的 live provider 路径已被真实执行。
  - Reason: `PersonaMem 64` live canary 已完成 `64/64` provider 调用。
  - Evidence target: `outputs_v2/evals_benchmark/20260414T231617Z_stage2_memory_canary.json` 与对应 `predictions.jsonl`。

## Abandoned

- None
