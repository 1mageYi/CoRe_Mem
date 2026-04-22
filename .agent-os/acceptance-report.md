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
    - `scripts/eval_stage2_local.py` 现已支持 `--top-k`、`--budget`、`--dataset`、`--experiment-id`
    - `src/core_mem/v2/eval_local.py` 已实现模块级、家族级、budget-sweep 和 variant-aware 评测逻辑
    - `docs/stage2_local_evaluation.md` 已完整说明接口、指标定义、模块映射与使用方式
    - `src/core_mem/v2/experiments.py` 已锁定 `mainline + 11` 个必做 ablation 的 preset variant
    - `outputs_v2/artifacts/stage2_experiment_index.json` 当前已登记 12 个 completed experiments
    - `scripts/verify_stage2_experiment_status.py --score-only` 当前返回 `13`
    - `outputs_v2/evals_local/20260414T074458Z_stage2_local_eval.json` 与同批 `outputs_v2/tables/*` 已作为最新 ablation registry 产物落地
    - `docs/requirements.md` 与 `docs/v2_design.md` 已锁定 budget sweep 与核心 ablation 集；当前 repo 已具备记录并机械验证这些结果的路径

- `EV-013` -> `AC-013` 第二阶段 benchmark canary 与正式评测隔离协议
  - Status: verified
  - Evidence:
    - `docs/requirements.md` 已写入 `local-first` 与 canary-first 协议
    - `docs/v2_design.md` 已写入 canary 设计与 benchmark 作为 evaluation source 的原则
    - `scripts/run_stage2_canary.py` 已生成 `outputs_v2/evals_benchmark/20260407T042120Z_stage2_canary_plan.json`
    - `outputs_v2/evals_benchmark/20260407T042120Z_personamem_canary.json` 与 `outputs_v2/evals_benchmark/20260407T042120Z_longmemeval_canary.json` 已落地，且二者均固定为 `64` 条

- `EV-014` -> `WS-021 / TD-035` `v2.4` 长跑阶段性证据
  - Status: verified
  - Evidence:
    - managed run 已完成 `research-results.tsv` 与 `autoresearch-state.json` 的全程记账；当前记录 best iteration `7`、best/current metric `24`
    - `scripts/verify_stage2_v24_longrun.py --score-only` 已在 current HEAD `12a9a80` 上达到 stop condition `24/24`
    - `outputs_v2/artifacts/latest_stage2_v24_train.json` 已记录 current-head full-data train，并与 retained HEAD `12a9a80` 对齐
    - `outputs_v2/artifacts/latest_stage2_v24_eval.json` 已记录 current-head full-data eval：`trained_eval.token_f1 = 0.9991150844073334`、`trained_eval.field_f1 = 0.9976704786107581`、`slot_assignment_metrics.token_f1 = 0.9961127308066084`
    - `outputs_v2/artifacts/latest_longmemeval_stage2_v24_canary.json` 与 `latest_longmemeval_stage2_v24_analysis.json` 已记录 current-head `LongMemEval-S 128`：`provider_exact_match = 10`、`provider_label_prefix_match = 10`、`local_exact_match = 10`
    - `outputs_v2/artifacts/latest_personamem_stage2_v24_canary.json` 已记录 current-head `PersonaMem 128`：`provider_exact_match = 38`、`provider_label_prefix_match = 38`、`local_exact_match = 28`
    - `outputs_v2/artifacts/latest_stage2_v24_online_gain.json` 已记录相对 `v2.3` retained baseline 的 `delta_provider_exact_match = +4`、`delta_local_exact_match = +4`
    - launch-manifest guard 已在 retained HEAD `12a9a80` 上通过：`tests/test_stage2_v24_longrun.py tests/test_stage2_model_skeleton.py tests/test_stage2_local_eval.py tests/test_stage2_memory_canary.py tests/test_stage2_memory_canary_quality.py tests/test_stage2_parser.py`
  - Boundary:
    - 当前能诚实声明的是 `v2.4 long-run` 的机械 stop condition 已达到；不能夸写成所有 benchmark 质量风险都已消失，或下一轮 stage-2 方向已经自动确定

- `EV-015` -> `WS-022 / TD-036` `v2.5` baseline/package closeout 证据
  - Status: partial
  - Evidence:
    - current HEAD `3036e3d` 上 `scripts/verify_stage2_v25_longrun.py --score-only` 已达到 stop condition `24/24`
    - `research-results.tsv` / `autoresearch-state.json` 已记录 iteration `5 keep`、best/current metric `24`
    - `outputs_v2/artifacts/latest_stage2_v25_train.json`、`latest_stage2_v25_eval.json`、`latest_longmemeval_stage2_v25_canary.json`、`latest_personamem_stage2_v25_canary.json`、`latest_longmemeval_stage2_v25_analysis.json`、`latest_stage2_v25_full_benchmark.json` 已在 current HEAD 上落地
    - `outputs_v2/artifacts/latest_stage2_v25_write_gain.json`、`latest_stage2_v25_retrieve_gain.json`、`latest_stage2_v25_belief_gain.json` 已作为 current-head baseline decomposition artifacts 落地
    - current-head `v24` baseline alias 已刷新到同一 HEAD，使 `latest_stage2_v24_eval.json`、`latest_longmemeval_stage2_v24_canary.json`、`latest_personamem_stage2_v24_canary.json` 可继续作为 `v2.5` retained baseline 复验
  - Boundary:
    - 当前 `24/24` 只说明 `v2.5` 的 artifact/verifier package 已机械齐套
    - `write / retrieve / belief` 三段 artifact 当前都显式记录 `delta_vs_v24_retained = 0`
    - 因此不能把本轮 closeout 误写成真实 `LongMemEval-S` / online quality gain 已超过 `v2.4`

- `EV-016` -> `WS-023 / TD-037` `v2.6` gain-first 长跑基线
  - Status: verified
  - Evidence:
    - `docs/v26_plan.md` 已把目标明确收紧为 gain-first：要求 current-head 的 `write / retrieve / belief` 至少一段出现真实正增益，并要求 `LongMemEval-S 128` 明确超过 `v2.5` retained baseline `10/128`
    - `scripts/verify_stage2_v26_longrun.py` 已把 mechanical target 改为 `stage2_v26_longrun_score`
    - `tests/test_stage2_v26_longrun.py` 已覆盖“只有在 positive gain 与超 baseline canary 同时出现时才可满分”的 verifier 合同
    - 当前 `v2.5` retained baseline 继续保留为对照线：`latest_stage2_v25_train.json`、`latest_stage2_v25_eval.json`、`latest_longmemeval_stage2_v25_canary.json`、`latest_personamem_stage2_v25_canary.json`、`latest_stage2_v25_write_gain.json`、`latest_stage2_v25_retrieve_gain.json`、`latest_stage2_v25_belief_gain.json`
    - fresh baseline 已通过 helper 固化：`research-results.tsv` iteration `0` 与 `autoresearch-state.json` 当前都记录 `stage2_v26_longrun_score = 9`
    - trial HEAD `077cbf3` 已加入 query-intent-aware temporal retrieval / belief scoring，并通过 stage-2 guard：`tests/test_stage2_v26_longrun.py tests/test_stage2_model_skeleton.py tests/test_stage2_local_eval.py tests/test_stage2_memory_canary.py tests/test_stage2_parser.py`
    - current session 已确认 `GPT_AGENT_API_KEY=SET`，因此历史 provider-env blocker 已被清除；新的 refine commit `a04effe` 已加入 slot-assignment prompt compaction 与 weak `other_fact` overwrite fast-path，并通过同一组 stage-2 guard
    - profiling 证据已显示 sample `51a45a95` 的 learned arbitration 次数从 `11` 压到 `1`
    - 同一 partial `LongMemEval-S 128` resumed run 已从 `2/128` 前进到 `3/128`，且 `51a45a95` 当前在 current-head 上记录 `memory_answer_local = target`、`provider_prediction = target`
    - refine commit `3051b0f` 已让 `scripts/run_stage2_memory_canary.py` 在单次 canary run 内复用 learned belief / slot-assignment predictors；fresh current-head `LongMemEval-S 128` run `outputs_v2/runs/20260417T055905Z_stage2_memory_canary_longmemeval/` 最终已完整结束在 `provider_exact_match = 11`、`local_exact_match = 10`
    - current-head `PersonaMem 128` run `outputs_v2/runs/20260417T072623Z_stage2_memory_canary_personamem/` 已完整结束在 `provider_exact_match = 44`、`local_exact_match = 33`，相对 retained `v2.5` `38 / 28` 保持正向 guard
    - refine commit `b4c997d` 已把 `v2.6` artifact 发布链并回 `scripts/verify_stage2_v26_longrun.py`，并新增 `tests/test_stage2_v26_publish.py` 覆盖 current-head publish 入口；当前 publish 入口已把 current-head `v2.5` / `v2.6` aliases、component gain artifacts、LongMemEval analysis 与 holdout full-benchmark summary 全部刷新到 current HEAD `c8c8e19`
    - trial commit `160e29f` 已在 `src/core_mem/v2/projection.py` 上加入 historical `other_fact` clause projection；对应 fresh current-head `LongMemEval-S 128` run `outputs_v2/runs/20260417T083058Z_stage2_memory_canary_longmemeval/` 在一次 `HTTP 502` 后原地 `--resume` 收口到 `provider_exact_match = 11`、`local_exact_match = 11`
    - current-head `PersonaMem 128` guard refresh `outputs_v2/runs/20260417T094824Z_stage2_memory_canary_personamem/` 在一次 read timeout 后原地 `--resume` 收口到 `provider_exact_match = 44`、`local_exact_match = 33`
    - `outputs_v2/artifacts/latest_stage2_v26_write_gain.json` 当前已记录 `positive_gain = true`、`delta_provider_exact_match = +1`、`delta_local_exact_match = +1`
    - `outputs_v2/artifacts/latest_stage2_v26_belief_gain.json` 当前已记录 `positive_gain = true`、`delta_vs_v25_retained = +1`
    - `outputs_v2/artifacts/latest_stage2_v26_full_benchmark.json` 当前已记录 `holdout_only = true`
    - `conda run -n core_mem python scripts/verify_stage2_v26_longrun.py --score-only` 当前为 `26`
  - Boundary:
    - 当前能诚实声明的是 `v2.6` 的机械 stop condition 已达成
    - 当前仍不能误写成 LongMemEval failure mass 已全面解决；remaining问题仍主要集中在 `other_fact / projection` family

- `EV-017` -> `WS-030 / TD-044` `v33` learned-authoritative full-holdout runtime 证据
  - Status: partial
  - Evidence:
    - `research-results.tsv` row `102` 已把 current-line authoritative `PersonaMem 512` full measurement 正式记账为 completed `search`
    - `autoresearch-state.json` 当前 iteration 已同步到 `102`，并记录 row-102 labels `personamem512-full / authoritative-full / provider-minus1-local-plus21 / compare-still-incomplete / verifier-flat / v33`
    - `outputs_v2/v33_full_personamem_512_query_overlap/evals_benchmark/20260421T001000Z_stage2_memory_canary.json` 已完整记录 current-line `PersonaMem 512` full holdout：`provider_exact_match = 182`、`local_exact_match = 196`
    - retained baseline `outputs_v2/artifacts/latest_personamem_stage2_v32_full.json` 当前固定为 `provider/local = 183/175`，因此 current-line Persona full measurement 已形成 `provider -1 / local +21`
    - 当前 session 已确认同一 full Persona run 的 provider failure shape 固定为 `blank = 108`、`nonlabel = 12`
    - `conda run -n core_mem python scripts/verify_stage2_v33_longrun.py --score-only` 当前仍为 `36`
  - Boundary:
    - 当前能诚实声明的是：current-line `PersonaMem 512` authoritative full holdout 已完成，且 local full-measurement gain 已成立
    - 当前不能声称 `v33` full holdout compare 已完成，因为 current-line `LongMemEval-S 500` full summary 仍缺
    - 因此 `latest_longmemeval_stage2_v33_full.json`、`latest_personamem_stage2_v33_full.json` 与 `latest_stage2_v33_full_holdout_compare.json` 仍不能发布，official verifier 也仍停在 `36`
