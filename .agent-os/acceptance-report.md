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
    - 当前 `pytest` 通过，共 31 个测试
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
  - Status: partial
  - Evidence:
    - `docs/requirements.md` 已新增第二阶段目标、范围、训练、数据与评测条款
    - `docs/v2_design.md` 已新增并对齐第二阶段设计真相
    - `docs/implementation_plan.md` 已新增第二阶段阶段拆分

- `EV-009` -> `AC-009` 第二阶段 observation / slot / belief 规范
  - Status: partial
  - Evidence:
    - `docs/requirements.md` 已锁定 `Observation JSON` 与 `Belief JSON` 的必要字段
    - `docs/v2_design.md` 已写入 observation、slot、belief 三者关系与训练样本单位

- `EV-010` -> `AC-010` 第二阶段主线实现骨架
  - Status: unverified
  - Evidence:
    - 当前只有设计与文档真相；代码实现尚未开始

- `EV-011` -> `AC-011` 第二阶段训练与本地 intrinsic evaluation 管线
  - Status: unverified
  - Evidence:
    - 当前只有设计与 protocol；脚本、数据接入和本地结果尚未产生

- `EV-012` -> `AC-012` 第二阶段本地评估、budget sweep 与核心 ablation 记录机制
  - Status: unverified
  - Evidence:
    - `docs/requirements.md` 与 `docs/v2_design.md` 已锁定本地指标家族、budget sweep 与 ablation 集，但实际运行证据尚未产生

- `EV-013` -> `AC-013` 第二阶段 benchmark canary 与正式评测隔离协议
  - Status: partial
  - Evidence:
    - `docs/requirements.md` 已写入 `local-first` 与 canary-first 协议
    - `docs/v2_design.md` 已写入 canary 设计与 benchmark 作为 evaluation source 的原则
