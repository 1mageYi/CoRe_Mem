# Project Index

## Current Truth

- Objective: `OBJ-002`, `OBJ-003`, `OBJ-004`
- Top next action: `TD-015`
- Active workstreams: `WS-002`, `WS-003`, `WS-005`, `WS-006`, `WS-007`
- Active blockers: `BL-004`

## Objective Summary

- `OBJ-002`: 实现一个 bounded cross-session personalized memory framework。
- `OBJ-003`: 第一阶段具体实例为 `vanilla CoRe Memory`，目标是完成 PersonaMem 与 LongMemEval-S 的可复现 benchmark 运行。
- `OBJ-004`: 第二阶段具体实例为 `V2.0 structured latent-slot memory`，目标是在保留第一阶段 v1 baseline 参考价值的前提下，建立更强的 latent memory 研究主线。

## Active Workstreams

- `WS-002` `[doing]`: 基础工程与环境维护
- `WS-003` `[doing]`: Stage-1 benchmark adapters 与 protocol 对齐
- `WS-005` `[doing]`: Stage-1 评测、输出与运行元数据
- `WS-006` `[doing]`: Stage-1 测试体系
- `WS-007` `[doing]`: Stage-2 `V2.0` 设计锁定与实现准备

## Top Next Action

- `TD-015` `[ready]`: 搭建第二阶段 `V2.0` 的 observation normalization、belief JSON 目标生成与公开数据集接入骨架。
  - Needed: 在 `src/`、`scripts/`、`configs/`、`tests/` 与 `outputs_v2/` 下建立第二阶段最小可运行骨架

## Active Blockers

- `BL-004`: 当前 Gemini key/provider 组合在 formal benchmark 负载下已构成真实外部 blocker。LongMemEval-S formal run 仅推进到 `19/500`，PersonaMem formal run 仅推进到 `22/589`；即使加入 pacing、bounded retry、outer supervisor、chunked relaunch 和 ultra-slow single-sample 检查，仍连续返回 `HTTP 429`，无法把 PersonaMem 从 `22` 推进到 `23`。该 blocker 当前主要影响 stage-1 正式 benchmark，不阻断 stage-2 本地实现与训练路线。

## Recent Important Changes

- 2026-04-06: 用户确认第二阶段为 `V2.0 structured latent-slot memory`，并明确其属于 stage-2。
- 2026-04-06: 真源 `docs/requirements.md` 已扩展为 stage-1 + stage-2 双阶段规范，并写入第二阶段方法、训练、数据和评测规则。
- 2026-04-06: 第二阶段主线已锁定为 `Light Cross-Attention Resampler + Flan-T5 belief JSON decoder + LoRA/adapter + public-datasets-first, synthetic-minimal`。
- 2026-04-06: 已新增 `docs/v2_design.md`，用于汇总第二阶段方法设计、训练路线、数据策略与评测协议。
- 2026-04-06: stage-1 真实 runtime truth 保持不变：正式 benchmark 仍被外部 provider `HTTP 429` 阻断。
- 2026-04-06: 新增 `docs/current_model_implementation.md`，对白盒实现、benchmark 主链路、VanillaCoreMemory 原型和真实推理用法做了详细审阅说明。

## Read Next

- `docs/requirements.md`
- `docs/v2_design.md`
- `.agent-os/architecture-milestones.md`
- `.agent-os/todo.md`
- `.agent-os/run-log.md`
