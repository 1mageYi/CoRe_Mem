# Project Index

## Current Truth

- Objective: `OBJ-002`, `OBJ-003`, `OBJ-004`
- Top next action: `TD-026`
- Active workstreams: `WS-012`
- Active blockers: `BL-004`, `BL-006`

## Objective Summary

- `OBJ-002`: 实现一个 bounded cross-session personalized memory framework。
- `OBJ-003`: 第一阶段具体实例为 `vanilla CoRe Memory`，目标是完成 PersonaMem 与 LongMemEval-S 的可复现 benchmark 运行。
- `OBJ-004`: 第二阶段具体实例为 `V2.0 structured latent-slot memory`，目标是在保留第一阶段 v1 baseline 参考价值的前提下，建立更强的 latent memory 研究主线。

## Active Workstreams

- `WS-007` `[done]`: Stage-2 `V2.0` 主线模型骨架与直训链路已推进到可直接训练
- `WS-008` `[done]`: Stage-2 数据、parser 与训练管线首批骨架
- `WS-009` `[done]`: Stage-2 memory-mediated benchmark canary runner 已接入，并已完成 PersonaMem live MiniMax 调用
- `WS-010` `[done]`: Stage-2 latent memory 主链路已从 deterministic skeleton 升级为真正消费 composed latent 的实现路径
- `WS-011` `[doing]`: Stage-2 已进入 live canary 质量提升阶段；当前主线是对 PersonaMem 64 做 failure analysis，并迭代修复 online memory / belief / answer 链路
- `WS-012` `[doing]`: Stage-2 当前主线已切换到 latent-core robustness；local intrinsic quality 是主指标，PersonaMem 64 `9/10` 只作为不退化 guard

## Top Next Action

- `TD-026` `[doing]`: 以系统/模型/latent 本体更强、更稳健为锚点，提升 stage-2 local intrinsic 质量。
  - Needed: 当前 `PersonaMem 64` canary 已达到 `9/10`，但其中一部分收益来自 benchmark-facing option 对齐；下一步要把优化重心重新拉回 `encoder / retrieval / lifecycle / belief decode / composition` 本体，并用 local intrinsic 指标来衡量

## Active Blockers

- `BL-004`: 当前 Gemini key/provider 组合在 formal benchmark 负载下已构成真实外部 blocker。LongMemEval-S formal run 仅推进到 `19/500`，PersonaMem formal run 仅推进到 `22/589`；即使加入 pacing、bounded retry、outer supervisor、chunked relaunch 和 ultra-slow single-sample 检查，仍连续返回 `HTTP 429`，无法把 PersonaMem 从 `22` 推进到 `23`。该 blocker 当前只影响 stage-1 formal benchmark；stage-1 formal benchmark 同时处于“待用户显式触发”状态，不阻断 stage-2 主线。
- `BL-006`: 当前 stage-2 live canary 的真正 blocker 已转为质量不足，而不是 provider 缺失。`PersonaMem 64` 虽已完成 `64/64` live MiniMax-M2.7 调用，但 `provider exact match = 1/64`、`local exact match = 0/64`，说明 parser/retrieval/belief/projection 至少有一层存在系统性误差。
- `BL-006`: 当前 stage-2 live canary 的主要 blocker 已从“是否能达到 `>=9/10`”降为“剩余 recall/suggestion 错误仍存在，尚不足以直接外推到更大 benchmark 范围”。当前 best `PersonaMem 64` live canary 已达到 `provider exact match = 31/64`、`local exact match = 25/64`、`quality score = 9/10`

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
- 2026-04-14: 新增 `src/core_mem/v2/experiments.py`，并将 `scripts/train_stage2.py` / `scripts/eval_stage2_local.py` 扩展为支持 preset ablation variants、experiment registry 自动登记与 checkpoint-aware local eval；`outputs_v2/artifacts/stage2_experiment_index.json` 当前已登记 `mainline + 11` 个必做 ablation，`scripts/verify_stage2_experiment_status.py --score-only` 已达 `13/13`，对应 `research-results.tsv` / `autoresearch-state.json` 已记录完整 background run 轨迹。当前证据对应的是 `gpu3 + configs/stage2_train_tiny.yaml` 的本地 train/eval 完成态，而非默认 `flan-t5-base` 的非 tiny 全量 run。
- 2026-04-14: `src/core_mem/v2/encoder.py` / `resampler.py` / `decoder.py` / `system.py` 已从 hash/mean skeleton 升级为带可训练参数的 lexical projection + cross-attention composition + latent-conditioned belief decoding 主链；`scripts/verify_stage2_latent_status.py --score-only` 已从 `2` 提升到 `7`，并经 stage-2 guard 复验通过。
- 2026-04-14: 新增 `scripts/run_stage2_memory_canary.py` 与 `tests/test_stage2_memory_canary.py`，并更新 `configs/minimax_m27.yaml` 到 stage-2 输出语义；`outputs_v2/evals_benchmark/20260414T170606Z_stage2_memory_canary.json` 已记录一条 PersonaMem memory-mediated canary artifact，状态为 `blocked_provider_not_configured`；当前 `stage2_latent_readiness_score=9/9`。
- 2026-04-14: `MiniMax-M2.7` live stage-2 canary 已完成：`outputs_v2/evals_benchmark/20260414T231441Z_stage2_memory_canary.json` 为 1-sample live 成功样本，`outputs_v2/evals_benchmark/20260414T231617Z_stage2_memory_canary.json` 为 `PersonaMem 64` live canary；当前真实问题已从“缺 provider key”切换为“quality too low for benchmark scaling”。
- 2026-04-14: 新增 `scripts/analyze_stage2_memory_canary_failures.py` 与 `scripts/verify_stage2_memory_canary_quality.py`，当前 `PersonaMem 64` live canary 的质量基线已被机械化为 `4/10`，并已写入 `outputs_v2/artifacts/latest_personamem_stage2_canary_analysis.json`。
- 2026-04-15: background autoresearch 已完成 3 轮 `PersonaMem 64` canary 迭代：parser 噪声清理单独无收益；PersonaMem answer-option 标签空间对齐把 quality score 从 `4/10` 提升到 `8/10`；candidate-label prompt 仅带来 `provider exact 21 -> 22` 的小幅变化，当前 best artifact 为 `outputs_v2/evals_benchmark/20260415T005653Z_stage2_memory_canary.json`。
- 2026-04-15: 同一 background autoresearch run 继续通过 pivot 进入 lifecycle retention 路线：对 facet-rich relation 保留多条 active memory，并让 PersonaMem option scorer 消费 selected slot glosses；当前 best artifact 已更新为 `outputs_v2/evals_benchmark/20260415T015324Z_stage2_memory_canary.json`，`quality score = 9/10`

## Read Next

- `docs/requirements.md`
- `docs/v2_design.md`
- `.agent-os/architecture-milestones.md`
- `.agent-os/todo.md`
- `.agent-os/run-log.md`
