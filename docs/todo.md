# TODO

## Doing

- `TD-038` 以 `v2.7 32k teacher-first long-run` 为目标，在**不改 `core / residual` 双银行结构**的前提下，先建立 `32k` source-level split、teacher-labeled data-quality upgrade、internal generalization test 与 `gpu2` 训练耗时基线。
  - 当前锚点：`24k train / 4k val / 4k test`
  - 当前 teacher：`MiniMax-M2.7`
  - 当前约束：不做任何 `fallback / shortcut / benchmark-specific heuristic / benchmark leakage`
  - 当前训练要求：优先使用 `gpu2`，并记录 wall-clock / throughput / memory
  - Truth boundary：先只做 `32k`；只有 internal test work well 后，才允许进入 full-data

- `TD-036` 以 `v2.5 learned core-path long-run` 为目标，在**不改 `core / residual` 双银行结构**的前提下，继续推进 `LongMemEval-S` 质量、`write / retrieve / belief` 三段的 learned 化、learned slot assignment 泛化鲁棒性、更强 latent 与 full benchmark holdout evaluation。
  - 当前起点：`TD-035 / WS-021` 已在 current HEAD `12a9a80` 上达到 `24/24`
  - 当前约束：不做任何 `fallback / shortcut / benchmark-specific heuristic`
  - 当前 managed run 状态：current HEAD `3036e3d` 已把 `stage2_v25_longrun_score` 推到 `24/24`
  - 当前边界：这轮 `24/24` 属于 baseline/package closeout；`write / retrieve / belief` 三段 artifact 当前仍显式记录 `delta_vs_v24_retained = 0`
  - 当前计划：见 [v25_plan.md](/media/storage/mingjing/workspace/CoRe_Mem/docs/v25_plan.md)

## Blocked

- `TD-012` 补齐 stage-1 外部前置条件：当前 formal benchmark 继续受 provider `HTTP 429` 阻断，且在用户明确要求前不主动继续推进。
- `TD-013` 将 PersonaMem / LongMemEval-S 从 1-sample real run 推进到正式全量 protocol 运行：当前作为 pending baseline / acceptance 项保留，直到用户要求 AI 去跑。

## Backlog

- `TD-037` 以 `v2.6 gain-first long-run` 为目标，在**不改 `core / residual` 双银行结构**的前提下，要求 `write / retrieve / belief` 至少一段出现真实正增益，并要求 `LongMemEval-S 128` 明确超过 `v2.5` retained baseline。
  - 说明：当前已在 current HEAD `cfbdc08` 上完成，`scripts/verify_stage2_v26_longrun.py --score-only = 26`

- `TD-007` 为 PersonaMem 128k / 1M 预留启动开关。
- `TD-018` 建立第二阶段 benchmark canary protocol 与结果记录。
- `TD-025` 在 `PersonaMem 64` canary 质量提升后，再扩大 stage-2 benchmark 范围与对比表。
  - 说明：`TD-025` 现视为 `TD-028 / v2.1` 的子目标之一，而不再是独立的“完整 v2 之后随手做做”项。

- `TD-030` 以 learned model / better latent 为锚点启动 `v2.1` 长跑。
  - 说明：当前机械 stop condition 已在 HEAD `d6bc4f7` 上达到 `16/16`；此项历史里程碑已完成，但不再是当前主线。

- `TD-031` 在不引入 fallback/shortcut 的前提下，把 stage-2 learned training 切到“语义优先、格式外部约束处理”主线。
  - 说明：当前机械 stop condition 已达到 `17/17`；此项历史里程碑已完成，但不再是当前主线。

- `TD-032` 以 full-data semantic-first learned memory 为目标推进 `v2.2`。
  - 说明：当前机械 stop condition 已达到 `19/19`；此项 closeout 已完成，但不再是当前主线。

## Done


- `TD-001` 初始化项目文档系统与根契约文件。
- `TD-002` 在默认 conda envs 目录创建并固定 conda 环境 `core_mem`（Python 3.10）。
- `TD-003` 建立基础目录结构、配置机制与输出目录规范。
- `TD-004` 确定第一阶段 pretrained embedding model 为 `sentence-transformers/all-MiniLM-L6-v2`。
- `TD-005` 实现阿里云 OpenAI-compatible provider adapter 的第一版。
- `TD-006` 接入 PersonaMem 32k 官方主任务与评测协议。
- `TD-008` 接入 LongMemEval-S 官方评测协议。
- `TD-009` 实现 vanilla CoRe Memory 核心模块与文本 embedding 接口。
- `TD-010` 建立实验总入口与分步骤脚本。
- `TD-011` 建立并扩展 unit tests 与最小 E2E smoke test。
- `TD-014` 锁定第二阶段 `V2.0 structured latent-slot memory` 方案，并将真源、设计文档和状态文档同步到 stage-2 真相。
- `TD-015` 搭建第二阶段 `V2.0` 的 observation normalization、belief JSON 目标生成、parser skeleton 与公开数据集接入骨架。
- `TD-016` 实现第二阶段 `V2.0` 主线模型骨架，并打通 direct-train 链路。
- `TD-019` 获取并接入第二阶段真实公开数据源文件。
- `TD-017` 建立第二阶段本地 intrinsic evaluation 管线、budget sweep 与 experiment registry，并在 `gpu3` 上完成 `mainline + 11` 个必做 ablation 的 tiny-backend 本地 train/eval 登记。
- `TD-021` 把第二阶段真正的 latent memory 主链路做实。
- `TD-022` 建立 stage-2 memory-mediated benchmark canary runner，并接入可用 provider 配置。
- `TD-023` 完成 stage-2 MiniMax memory canary 的 live provider 首次运行。
- `TD-026` 以系统/模型/latent 本体更强、更稳健为锚点，提升 stage-2 local intrinsic 质量，并把 `PersonaMem 64 >= 9/10` 作为不退化 guard。
- `TD-020` 为默认 `flan-t5-base` backbone 补非 tiny `gpu3` 训练证据。
- `TD-027` 推进完整的 `v2`：fresh live canary、第二 benchmark、非 tiny 训练证据，以及去除 benchmark shortcut/fallback。
- `TD-028` 把 `v2.1` 作为当前主线：提升真实质量、learned path 实际贡献、跨 benchmark 鲁棒性与系统化收口。
  - 当前结果：`scripts/verify_stage2_v21_robustness.py --score-only = 14/15`
  - 关键证据：fresh current-head `LongMemEval-S 64` 已提升到 `provider 5 / local 3`，current-head `PersonaMem 128` 已重新落地并保住 `local_exact_rate >= 0.20`，`latest_stage2_learned_online_gain.json` 已转正，且 `outputs_v2/evals_local/20260415T132109Z_stage2_local_eval.json` 对应的 `GPU3 non-tiny flan-t5-base train/eval` 仍成立
- `TD-029` 把 `v2.1` 的当前主线切到 learned-memory-first / better latent。
  - 当前结果：`scripts/verify_stage2_v21_learned_memory.py --score-only = 12/12`
  - 关键证据：online learned path、checkpoint-backed belief、online-aligned training 语义和最小 current-head learned-mode artifact 已成立
- `TD-032` 把 semantic-first 路线推进到 full-data current-head 训练、extended semantic canaries、LongMemEval-S semantic analysis 与 semantic online gain。
  - 当前结果：`scripts/verify_stage2_v22_completion.py --score-only = 19/19`
  - 关键证据：current-head full-data train/eval、`PersonaMem 128` / `LongMemEval-S 64/128` semantic canaries、`latest_longmemeval_stage2_semantic_analysis.json`、`latest_stage2_semantic_online_gain.json`

- `TD-033` 以 `LongMemEval-S` 质量提升、learned slot assignment 与 stronger latent 为目标推进 `v2.3`。
  - 当前结果：后续 `TD-034 / WS-020` 已将这条主线机械收口到 `scripts/verify_stage2_v23_longrun.py --score-only = 22/22`
  - 关键证据：`LongMemEval-S 128` current-head `provider/local = 6/128`、`PersonaMem 128` current-head `provider_exact = 38`、`local_exact = 16`、`latest_stage2_slot_assignment_gain.json`

- `TD-035` 以更远的 `v2.4 quality-first long-run` 为目标推进 `LongMemEval-S` 质量、full-data learned slot assignment 与 stronger latent。
  - 当前结果：managed autoresearch run 已在 current HEAD `12a9a80` 上达到 `scripts/verify_stage2_v24_longrun.py --score-only = 24/24`
  - 关键证据：`latest_stage2_v24_eval.json` 的 `trained_eval.token_f1 = 0.9991150844073334`、`field_f1 = 0.9976704786107581`；`latest_longmemeval_stage2_v24_canary.json` 的 `provider/local = 10/128`；`latest_personamem_stage2_v24_canary.json` 的 `provider_exact = 38`、`local_exact = 28`；`latest_stage2_v24_online_gain.json` 的 `delta_provider_exact_match = +4`、`delta_local_exact_match = +4`

- `TD-036` 以 `v2.5 learned core-path long-run` 为锚点，把当前主线推进到更强的 `LongMemEval-S`、更泛化的 learned slot assignment，以及 `write / retrieve / belief` 三段更真实的 learned 主模块。
  - 说明：这条线的重点不再是补 current-head closeout artifact，而是在不动 `core / residual` 的前提下，把 `v2.4` retained 线当作 baseline，继续做质量和泛化提升。
  - 关键边界：full benchmark 只做 holdout evaluation，不回流成训练 supervision；继续严格禁止任何 fallback / shortcut / benchmark-specific heuristic；不再把 raw JSON exactness 当作主优化目标。


## Notes

- 真实 runtime 状态以 `.agent-os/todo.md` 为准。
- 本文件用于面向人类的同步视图。
- 第一阶段剩余未过项仍主要是正式 benchmark 两项，但当前不作为默认主动执行主线。
- 第二阶段当前已从“方法与治理层锁定”推进到“完整 v2 milestone 已机械达成”的状态。
- 第二阶段 observation / belief / parser / dataset skeleton、`prepare/train/eval/canary` 脚本、主线 memory system、公开数据规范化、strict-source prepared manifest、direct-train launcher、完整 local eval 与评测文档都已落地，`stage2_readiness_score` 当前为 `50`，`stage2_acceptance` 当前为 `7/7`，`scripts/verify_stage2_latent_status.py --score-only` 当前为 `9/9`。
- 当前最大的未完成点已不再是 `v2.2` 的在线证据缺口；`TD-032` 当前已经把 `stage2_v22_completion_score` 推到 `19/19` 并达到 stop condition。
- 当前 `TD-035 / WS-021` 已机械完成；当前主线已继续前推到 `TD-036 / v2.5 long-run`。
- 第二阶段主线采用：
  - `Light Cross-Attention Resampler`
  - `Flan-T5 belief JSON decoder`
  - `public-datasets-first, synthetic-minimal`
  - `benchmark is evaluation source, not primary training source`
