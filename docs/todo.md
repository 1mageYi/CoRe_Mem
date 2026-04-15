# TODO

## Doing

- `TD-029` 把 `v2.1` 的当前主线切到 learned-memory-first / better latent。
  - 当前机械状态：`scripts/verify_stage2_v21_learned_memory.py --score-only = 12/12`
  - 当前 retained state：online learned path、checkpoint-backed belief、online-aligned training 语义和 learned-mode current-head artifacts 已落地；在下一轮更大 learned-mode canary 覆盖正式接管前，runtime truth 继续把该主线维持为 doing

## Blocked

- `TD-012` 补齐 stage-1 外部前置条件：当前 formal benchmark 继续受 provider `HTTP 429` 阻断，且在用户明确要求前不主动继续推进。
- `TD-013` 将 PersonaMem / LongMemEval-S 从 1-sample real run 推进到正式全量 protocol 运行：当前作为 pending baseline / acceptance 项保留，直到用户要求 AI 去跑。

## Backlog

- `TD-007` 为 PersonaMem 128k / 1M 预留启动开关。
- `TD-018` 建立第二阶段 benchmark canary protocol 与结果记录。
- `TD-025` 在 `PersonaMem 64` canary 质量提升后，再扩大 stage-2 benchmark 范围与对比表。
  - 说明：`TD-025` 现视为 `TD-028 / v2.1` 的子目标之一，而不再是独立的“完整 v2 之后随手做做”项。

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

## Notes

- 真实 runtime 状态以 `.agent-os/todo.md` 为准。
- 本文件用于面向人类的同步视图。
- 第一阶段剩余未过项仍主要是正式 benchmark 两项，但当前不作为默认主动执行主线。
- 第二阶段当前已从“方法与治理层锁定”推进到“完整 v2 milestone 已机械达成”的状态。
- 第二阶段 observation / belief / parser / dataset skeleton、`prepare/train/eval/canary` 脚本、主线 memory system、公开数据规范化、strict-source prepared manifest、direct-train launcher、完整 local eval 与评测文档都已落地，`stage2_readiness_score` 当前为 `50`，`stage2_acceptance` 当前为 `7/7`，`scripts/verify_stage2_latent_status.py --score-only` 当前为 `9/9`。
- 当前最大的未完成点已经从“latent path 还不是真实主链”和“provider key 缺失”切换为“live canary 质量不足，尚不能直接扩大 benchmark”。
- `TD-027`、`TD-028` 与 `TD-029` 已机械完成；当前更合理的后续工作是扩大 learned-mode canary 规模，而不是继续补 learned-memory plumbing。
- 第二阶段主线采用：
  - `Light Cross-Attention Resampler`
  - `Flan-T5 belief JSON decoder`
  - `public-datasets-first, synthetic-minimal`
  - `benchmark is evaluation source, not primary training source`
