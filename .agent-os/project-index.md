# Project Index

## Current Truth

- Objective: `OBJ-002`, `OBJ-003`, `OBJ-004`
- Top next action: `TD-017`
- Active workstreams: `WS-009`
- Active blockers: `BL-004`

## Objective Summary

- `OBJ-002`: 实现一个 bounded cross-session personalized memory framework。
- `OBJ-003`: 第一阶段具体实例为 `vanilla CoRe Memory`，目标是完成 PersonaMem 与 LongMemEval-S 的可复现 benchmark 运行。
- `OBJ-004`: 第二阶段具体实例为 `V2.0 structured latent-slot memory`，目标是在保留第一阶段 v1 baseline 参考价值的前提下，建立更强的 latent memory 研究主线。

## Active Workstreams

- `WS-007` `[done]`: Stage-2 `V2.0` 主线模型骨架与直训链路已推进到可直接训练
- `WS-008` `[done]`: Stage-2 数据、parser 与训练管线首批骨架
- `WS-009` `[doing]`: Stage-2 本地 intrinsic evaluation 与 canary 协议细化

## Top Next Action

- `TD-017` `[doing]`: 建立第二阶段本地 intrinsic evaluation 的完整指标与 budget sweep。
  - Needed: 当前 local eval 已从 skeleton 升级为模块级 + 家族级 + budget-sweep 评测体系，并已产出 JSON + summary CSV + budget CSV；下一步优先把这套体系用于更系统的训练后模型比较与 ablation 结果沉淀

## Active Blockers

- `BL-004`: 当前 Gemini key/provider 组合在 formal benchmark 负载下已构成真实外部 blocker。LongMemEval-S formal run 仅推进到 `19/500`，PersonaMem formal run 仅推进到 `22/589`；即使加入 pacing、bounded retry、outer supervisor、chunked relaunch 和 ultra-slow single-sample 检查，仍连续返回 `HTTP 429`，无法把 PersonaMem 从 `22` 推进到 `23`。该 blocker 当前只影响 stage-1 formal benchmark；stage-1 formal benchmark 同时处于“待用户显式触发”状态，不阻断 stage-2 主线。

## Recent Important Changes

- 2026-04-06: 用户确认第二阶段为 `V2.0 structured latent-slot memory`，并明确其属于 stage-2。
- 2026-04-06: 真源 `docs/requirements.md` 已重建并细化为 stage-1 + stage-2 双阶段规范，并明确 stage-2 为当前主线、stage-1 formal benchmark 为待用户触发项。
- 2026-04-06: 第二阶段主线已锁定为 `Light Cross-Attention Resampler + Flan-T5 belief JSON decoder + LoRA/adapter + public-datasets-first, synthetic-minimal`。
- 2026-04-06: `docs/v2_design.md` 已细化到 Observation/Slot/Belief schema、dataset-to-task mapping 与 metric-to-module mapping。
- 2026-04-06: stage-1 真实 runtime truth 保持不变：正式 benchmark 仍被外部 provider `HTTP 429` 阻断。
- 2026-04-06: 新增 `docs/current_model_implementation.md`，对白盒实现、benchmark 主链路、VanillaCoreMemory 原型和真实推理用法做了详细审阅说明。
- 2026-04-07: 新增 `scripts/verify_stage2_status.py`、`scripts/verify_stage2_acceptance.py`、对应测试以及 `outputs_v2/` 目录骨架；`stage2_readiness_score` 从 `0` 提升到 `20`。
- 2026-04-07: 新增 `src/core_mem/v2/`、`scripts/prepare_stage2_data.py`、`scripts/train_stage2.py`、`scripts/eval_stage2_local.py`、`scripts/run_stage2_canary.py` 与 `configs/stage2_train.yaml`；当前 `stage2_readiness_score=30`、`stage2_acceptance=7/7`、`pytest=40 tests`，且 `outputs_v2/` 已产出 prepared manifest、smoke train plan、local eval 结果与 `PersonaMem/LongMemEval-S` 的 `64` canary manifests。
- 2026-04-07: 新增 `src/core_mem/v2/encoder.py`、`lifecycle.py`、`resampler.py`、`decoder.py`、`system.py` 与端到端模型 smoke test；当前 `stage2_readiness_score=36`、`stage2_acceptance=7/7`、`pytest=42 tests`，stage-2 已从“schema/pipeline ready”推进到“memory system skeleton runnable”。
- 2026-04-07: 新增 `src/core_mem/v2/consolidation.py` 与 `projection.py`，将 stage-2 主链路推进到 `memory -> belief -> answer` 全链可跑；当前 `stage2_readiness_score=38`、`stage2_acceptance=7/7`、`pytest=42 tests`。
- 2026-04-13: 新增 `src/core_mem/v2/training.py`，将 `scripts/train_stage2.py` 从 planner 升级为可执行 trainer，并新增 `configs/stage2_train_tiny.yaml` 作为离线 tiny backend 验证配置；`core_mem` 环境已实际安装 `datasets`、`peft`、`accelerate`，且 `execute-train` tiny smoke 已在 `outputs_v2/runs/20260414T020438Z_stage2_train_exec` 跑通，当前 `stage2_readiness_score=41`、`stage2_acceptance=7/7`、`pytest=44 tests`。
- 2026-04-13: 新增 `configs/stage2_data_sources.json`、`scripts/stage2_data_preflight.py`，并将 `prepare_stage2_data.py` 扩展为支持 source-config 驱动的非 demo prepared-manifest 构建；当前 `stage2_readiness_score=44`、`stage2_acceptance=7/7`、`pytest=46 tests`。当前最直接的缺口已经从“没有数据入口”变成“`data/stage2_public/` 下还没有真实公开数据源文件”。
- 2026-04-13: 新增 `src/core_mem/v2/public_data.py` 与 `scripts/normalize_stage2_public_data.py`，从官方或作者源下载并规范化 `SGD / MultiWOZ 2.4 / Persona-Chat / MQUAKE / ReCoE`，生成 5 份真实 `normalized.jsonl`；`scripts/stage2_data_preflight.py --json` 已显示 `missing=[]`。
- 2026-04-13: `prepare_stage2_data.py` 新增 `--max-rows-per-dataset`，可在严格 source 模式下基于真实公开数据构建受控切片；`outputs_v2/artifacts/stage2_prepared_samples_manifest.json` 当前记录 `5 x 64` public-data slice，且 `outputs_v2/runs/20260414T043225Z_stage2_train_exec` 已在真实 public-data prepared manifest 上跑通 tiny `execute-train`。
- 2026-04-13: `scripts/train_stage2.py` 的 planner launcher 现已输出真正的 `--execute-train` 启动脚本；`configs/stage2_train.yaml` 与 `configs/stage2_train_tiny.yaml` 现均锁定 repo-local Hugging Face cache root，避免首次权重下载写到 repo 外目录。当前 `stage2_readiness_score=47`、`stage2_acceptance=7/7`、`pytest=53 tests`。
- 2026-04-14: 新增 `src/core_mem/v2/eval_local.py` 与 [stage2_local_evaluation.md](/home/image/workspace/CoRe_Mem/docs/stage2_local_evaluation.md)，将 local eval 从 placeholder 提升为模块级 / 指标家族级 / budget-sweep 评测体系；`scripts/eval_stage2_local.py` 现支持 `--top-k`、`--budget`、`--dataset`，并输出 JSON + summary CSV + budget CSV。
- 2026-04-14: 当前真实 public-data slice 的 local eval 结果已落地在 `outputs_v2/evals_local/20260414T055852Z_stage2_local_eval.json`，并产出对应 `outputs_v2/tables/20260414T055852Z_stage2_local_eval_summary.csv` 与 `outputs_v2/tables/20260414T055852Z_stage2_local_eval_budget_sweep.csv`；`stage2_readiness_score=50`、`stage2_acceptance=7/7`、`pytest=56 tests`。

## Read Next

- `docs/requirements.md`
- `docs/v2_design.md`
- `.agent-os/architecture-milestones.md`
- `.agent-os/todo.md`
- `.agent-os/run-log.md`
