# TODO

## Doing

- None

## Backlog

- `TD-007` `[backlog]` 为 PersonaMem 128k / 1M 提供启动开关。
  - Reason: 第一阶段只要求保留启动能力。

- `TD-018` `[backlog]` 建立第二阶段 benchmark canary protocol 与结果记录。
  - Reason: `gpu3 + tiny backend` 的 stage-2 本地 train/eval/ablation matrix 已全部完成并登记；下一步若继续推进，更应该把这个完成态沉淀为 canary 结果和对比表。
  - Evidence target: PersonaMem 64 / LongMemEval-S 64 的固定 canary 运行结果、输出表和记录规范。

- `TD-020` `[backlog]` 为默认 `flan-t5-base` backbone 补非 tiny `gpu3` 训练证据。
  - Reason: 当前 `stage2_experiment_completion_score=13/13` 的证据是 `configs/stage2_train_tiny.yaml` 本地运行矩阵；默认 backbone 的非 tiny 全量 run 仍未验证。
  - Evidence target: 至少一条默认 `configs/stage2_train.yaml` 的可追溯 train/eval artifact。

## Blocked

- `TD-012` `[blocked]` 补齐 stage-1 外部前置条件。
  - Reason: 当前可用的 Gemini key/provider 组合在 formal benchmark 负载下仍连续触发 `HTTP 429`，且 formal benchmark 在用户明确要求前不主动继续推进。
  - Evidence target: 可持续推进 formal benchmark 的 provider 配额、稳定 key，或用户批准的其他可用 provider。

- `TD-013` `[blocked]` 将两个 benchmark 从 1-sample real run 推进到正式全量 protocol 运行。
  - Reason: 当前只能证明最小链路打通，尚不足以满足 AC-002 / AC-003 的正式运行要求；同时该项被用户触发条件与 provider blocker 双重约束。
  - Evidence target: PersonaMem 与 LongMemEval-S 在正式范围内完成可重复结果运行。
  - Current evidence: runner 已具备增量落盘与续跑能力；Gemini 路径已把 PersonaMem formal run 推进到 `22/589`、把 LongMemEval formal run 推进到 `19/500`，但超保守单样本检查仍连续触发 `HTTP 429`，说明当前 key/provider 组合已构成真实外部 blocker。

## Done

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

## Abandoned

- None
