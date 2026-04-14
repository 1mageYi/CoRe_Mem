# Acceptance Report

## Stage 1 Acceptance Tracking

- `EV-001` -> `AC-001` vanilla CoRe Memory 完整实现
  - Status: verified
  - Evidence:
    - `src/core_mem/memory/core.py` 已实现 writer / updater / residual manager / reader / workflow 最小闭环
    - `tests/test_core_memory_module.py` 已覆盖核心 memory module
    - 已接入 `sentence-transformers/all-MiniLM-L6-v2` embedding 配置与 `observe_text/query_text`
    - `scripts/verify_stage1_acceptance.py` 当前将 `AC-001` 判为 passed

- `EV-002` -> `AC-002` PersonaMem 官方协议跑通并产出结果
  - Status: partial
  - Evidence:
    - 官方 `32k` 数据已下载到 `data/personamem/`
    - `outputs/runs/20260405T024515Z_personamem/predictions.jsonl` 已产出真实预测
    - `outputs/runs/20260405T024515Z_personamem/evaluation.json` 显示最小 1-sample `accuracy=1.0`
    - `outputs/runs/20260405T042146Z_personamem/evaluation.json` 显示 Gemini 2.5 Flash 路径下的最小 1-sample `accuracy=1.0`
    - `outputs/runs/20260405T050000Z_personamem/predictions.jsonl` 当前仅推进到 `22/589`
    - 超保守单样本检查在目标 `22 -> 23` 上连续 4 次 child launch 均返回 `HTTP 429`，说明当前正式结果受真实外部 blocker 限制

- `EV-003` -> `AC-003` LongMemEval-S 官方协议跑通并产出结果
  - Status: partial
  - Evidence:
    - 官方 `S` 数据已下载到 `data/longmemeval/`
    - `outputs/runs/20260405T024833Z_longmemeval/predictions.jsonl` 已产出真实预测
    - `outputs/runs/20260405T024833Z_longmemeval/evaluation.json` 显示最小 1-sample `accuracy=1.0`
    - `outputs/runs/20260405T042150Z_longmemeval/evaluation.json` 显示 Gemini 2.5 Flash 路径下的最小 1-sample `accuracy=1.0`
    - `outputs/runs/20260405T042236Z_longmemeval/predictions.jsonl` 已在 formal run 下推进到 `19/500`，但尚未完成全量结果

- `EV-004` -> `AC-004` 结果数据、表格、prediction outputs 与实验元数据
  - Status: verified
  - Evidence:
    - `scripts/run_experiment.py` 在 dry-run 下会写出 `outputs/runs/.../run_metadata.json`
    - `run_metadata.json` 已包含 benchmark、record_count、model、provider、sample_ids、prompt_preview
    - 当前 `run_metadata.json` 已补充 `config_snapshot_path`、`run_timestamp`、`commit_hash`、`prompt_version`
    - 真实 run 已产生 `predictions.jsonl` 与 `evaluation.json`
    - `outputs/tables/stage1_summary.csv` 与 `outputs/tables/stage1_summary.json` 已生成

- `EV-005` -> `AC-005` 工程与复现闭环
  - Status: verified
  - Evidence:
    - `environment.yaml` 已成为唯一环境规范文件
    - `configs/defaults.yaml` 与 `src/core_mem/config.py` 已建立统一配置加载路径
    - `configs/gemini_flash.yaml` 已提供一个用户批准的替代 OpenAI-compatible provider 配置
    - `configs/gemini_flash_ultraslow.yaml` 已提供外部 blocker 判别用的超保守 provider 配置
    - `scripts/run_experiment.py` 支持 benchmark 子命令和 dry-run 复现入口
    - `scripts/prepare_data.py`、`scripts/evaluate_run.py`、`scripts/make_tables.py` 已提供最小分步骤路径
    - `scripts/stage1_preflight.py` 已提供外部前置条件检查
    - `scripts/verify_stage1_acceptance.py` 已提供机械 acceptance check
    - `scripts/run_experiment.py` 已支持 `--run-dir` / `--resume` 与增量 prediction 落盘
    - `scripts/run_until_complete.py` 已提供 crash 后自动冷却重启的外层 supervisor
    - `outputs/` 路径已被实际写入
    - 环境复现策略已更新为 `environment.yaml` + 默认 conda named env；repo-local conda 目录已清理
    - 默认 conda named env `core_mem` 已创建并复验

- `EV-006` -> `AC-006` 单元测试与 E2E smoke test
  - Status: verified
  - Evidence:
    - 当前 `pytest` 通过，共 56 个测试
    - 已包含 provider、config、benchmark adapter、verifier、E2E dry-run smoke test、step scripts smoke test、protocol alignment tests、traceability tests、core memory tests
    - 已新增 acceptance verifier test、resumable runner test、Gemini 配置 / rate-limit retry tests 与 benchmark supervisor tests

- `EV-007` -> `AC-007` 结论可信、结果可追溯、允许 negative result
  - Status: verified
  - Evidence:
    - dry-run 输出已保存结构化 run metadata
    - benchmark prompt preview 与 sample ids 已可追溯
    - `scripts/verify_stage1_acceptance.py` 当前返回 `5/7`

## Stage 2 Planning And Design Tracking

- `EV-008` -> `AC-008` 第二阶段 `V2.0` 设计文档与真源条款
  - Status: verified
  - Evidence:
    - `docs/requirements.md` 已细化第二阶段目标、范围、训练、数据、评测、输出与 pending benchmark 规则
    - `docs/v2_design.md` 已细化到 schema、dataset-to-task mapping 与 metric-to-module mapping
    - `docs/implementation_plan.md` 已将 stage-2 调整为当前执行主线
    - `scripts/verify_stage2_acceptance.py` 当前将 `AC-008` 判为 passed

- `EV-009` -> `AC-009` 第二阶段 observation / slot / belief 规范
  - Status: verified
  - Evidence:
    - `docs/requirements.md` 已锁定 `Observation JSON`、`Slot Record` 与 `Belief JSON` 的必要字段
    - `docs/v2_design.md` 已写入 observation、slot、belief 三者关系、schema、字段语义与训练样本单位
    - `scripts/verify_stage2_acceptance.py` 当前将 `AC-009` 判为 passed

- `EV-010` -> `AC-010` 第二阶段主线实现骨架
  - Status: verified
  - Evidence:
    - `src/core_mem/v2/__init__.py`、`src/core_mem/v2/schemas.py`、`src/core_mem/v2/parser.py`、`src/core_mem/v2/datasets.py` 已落地
    - `src/core_mem/v2/encoder.py`、`src/core_mem/v2/lifecycle.py`、`src/core_mem/v2/consolidation.py`、`src/core_mem/v2/resampler.py`、`src/core_mem/v2/decoder.py`、`src/core_mem/v2/projection.py`、`src/core_mem/v2/system.py` 已落地
    - `tests/test_stage2_schemas.py`、`tests/test_stage2_parser.py` 已覆盖 Observation / Slot / Belief schema 与 rule-first parser skeleton
    - `tests/test_stage2_model_skeleton.py` 已覆盖主线 memory system 的最小 observe/query smoke path，以及 belief evidence / answer projection
    - `scripts/verify_stage2_acceptance.py` 当前将 `AC-010` 判为 passed

- `EV-011` -> `AC-011` 第二阶段训练与本地 intrinsic evaluation 管线
  - Status: verified
  - Evidence:
    - `scripts/prepare_stage2_data.py` 已生成 `outputs_v2/artifacts/stage2_prepared_samples_manifest.json`
    - `scripts/prepare_stage2_data.py` 现在已支持 `configs/stage2_data_sources.json` 驱动的 source-config 模式，能够在非 demo 输入下生成 prepared manifests
    - `scripts/stage2_data_preflight.py` 已能机械报告 `data/stage2_public/` 中公开数据源的 available/missing 状态，当前已返回 `missing=[]`
    - `scripts/normalize_stage2_public_data.py` 已从真实 `SGD / MultiWOZ 2.4 / Persona-Chat / MQUAKE / ReCoE` raw sources 生成 5 份 `normalized.jsonl`
    - `prepare_stage2_data.py` 当前已支持 `--max-rows-per-dataset`，可在严格 source 模式下构建受控 public-data manifests
    - `scripts/train_stage2.py` 已生成 `outputs_v2/runs/20260407T042125Z_stage2_train_plan/training_plan.json`
    - `scripts/train_stage2.py` 现在已支持 `--execute-train`，并通过 `configs/stage2_train_tiny.yaml` 在 `outputs_v2/runs/20260414T020438Z_stage2_train_exec/training_metrics.json` 和 `outputs_v2/runs/20260414T043225Z_stage2_train_exec/training_metrics.json` 跑通最小训练执行链；后者直接消费真实 public-data prepared manifest
    - `configs/stage2_train.yaml` 已锁定 backbone / resampler / LoRA / batching 配置，并新增 repo-local Hugging Face cache root
    - `outputs_v2/runs/20260414T043221Z_stage2_train_plan/launch_stage2_training.sh` 现已是直接可执行的 `--execute-train` 启动脚本
    - `environment.yaml` 已补充 `datasets`、`peft`、`accelerate` 依赖，并已在 `core_mem` 环境实际安装验证
    - `outputs_v2/checkpoints/stage2_smoke_checkpoint.json` 已作为 smoke execution artifact 写出
    - `scripts/verify_stage2_acceptance.py` 当前将 `AC-011` 判为 passed

- `EV-012` -> `AC-012` 第二阶段本地评估、budget sweep 与核心 ablation 记录机制
  - Status: verified
  - Evidence:
    - `scripts/eval_stage2_local.py` 现已支持 `--top-k`、`--budget`、`--dataset`
    - `src/core_mem/v2/eval_local.py` 已实现模块级、家族级和 budget-sweep 评测逻辑
    - `docs/stage2_local_evaluation.md` 已完整说明接口、指标定义、模块映射与使用方式
    - `outputs_v2/evals_local/20260414T055852Z_stage2_local_eval.json` 已生成最新真实 local eval 结果
    - `outputs_v2/tables/20260414T055852Z_stage2_local_eval_summary.csv` 与 `outputs_v2/tables/20260414T055852Z_stage2_local_eval_budget_sweep.csv` 已落地
    - `docs/requirements.md` 与 `docs/v2_design.md` 已锁定 budget sweep 与核心 ablation 集；当前 repo 已具备记录这些结果的机械化路径

- `EV-013` -> `AC-013` 第二阶段 benchmark canary 与正式评测隔离协议
  - Status: verified
  - Evidence:
    - `docs/requirements.md` 已写入 `local-first` 与 canary-first 协议
    - `docs/v2_design.md` 已写入 canary 设计与 benchmark 作为 evaluation source 的原则
    - `scripts/run_stage2_canary.py` 已生成 `outputs_v2/evals_benchmark/20260407T042120Z_stage2_canary_plan.json`
    - `outputs_v2/evals_benchmark/20260407T042120Z_personamem_canary.json` 与 `outputs_v2/evals_benchmark/20260407T042120Z_longmemeval_canary.json` 已落地，且二者均固定为 `64` 条
