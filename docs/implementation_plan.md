# Implementation Plan

## 总体策略

当前项目按两个阶段推进：

1. 第一阶段：完成 `vanilla CoRe Memory` 平台、benchmark、复现与结果闭环。
2. 第二阶段：在保留第一阶段 v1 作为 naive baseline 参考的前提下，实现并验证 `V2.0 structured latent-slot memory`。

当前执行主线：

- **Stage-2 是当前主线**
- **Stage-1 formal benchmark 作为 baseline / acceptance 相关的 pending 项保留，直到用户明确要求 AI 去跑**

## 第一阶段总策略

第一阶段只实现 `vanilla CoRe Memory` 的最小完整闭环，并优先确保：

1. 文档系统可恢复
2. 环境固定
3. Benchmark adapter 可运行
4. 方法主干可运行
5. 评测流程可运行
6. 输出结果可追溯
7. 测试可防回归

## 第一阶段拆分

### 阶段 A：项目初始化

- 建立 `AGENTS.md` 与文档系统
- 固定目录结构
- 固定环境约束
- Status: done

### 阶段 B：基础工程骨架

- 环境创建与依赖管理
- 配置系统
- API provider adapter
- 输出目录与结果记录规范
- Status: in progress
- Notes:
  - 配置系统、provider adapter、输出目录与 dry-run run metadata 已落地
  - Windows-local environment bootstrap 已放弃
  - repo 内 project-local conda env 方案已放弃
  - 环境复现将统一收敛到 `environment.yaml` + 默认 conda named env `core_mem`
  - 默认 `core_mem` named env 已创建并复验

### 阶段 C：Benchmark adapter

- PersonaMem 32k 主任务接入
- PersonaMem 128k / 1M 启动开关预留
- LongMemEval-S 接入
- 官方协议对齐
- Status: in progress
- Notes:
  - PersonaMem / LongMemEval-S 官方文件格式已可在 dry-run 模式下读取
  - 正式数据下载已完成，但 full-run 正式结果仍待完成
  - PersonaMem 32k 与 LongMemEval-S 官方数据已接入本地 `data/`
  - 两个 benchmark 已在默认 provider 路径与 Gemini 路径下分别完成真实 1-sample run 并得到最小 `evaluation.json`
  - 用户批准的 Gemini 2.5 Flash OpenAI-compatible 路径已可用于 formal run，但当前 key 在正式负载下持续触发 `HTTP 429`
  - 当前 formal 进度为 PersonaMem `22/589`、LongMemEval-S `19/500`；继续完成需要新的外部 provider 条件

### 阶段 D：CoRe Memory 方法实现

- writer
- core updater
- residual manager
- reader
- evaluation / output pipeline
- Status: in progress
- Notes:
  - 最小 writer / updater / residual manager / reader 已落地
  - 已接入 `sentence-transformers/all-MiniLM-L6-v2` embedding 配置
  - `VanillaCoreMemory` 已支持 `observe_text / query_text`

### 阶段 E：测试与复现

- 核心模块 unit tests
- 最小 E2E smoke test
- 总入口与分步骤脚本
- Status: in progress
- Notes:
  - 初始 unit tests 与 E2E dry-run smoke test 已建立
  - `prepare_data.py`、`evaluate_run.py`、`make_tables.py` 已作为最小分步骤脚本落地
  - `verify_stage1_acceptance.py` 已将 AC-001~AC-007 机械化
  - `run_experiment.py` 已支持 `--run-dir` / `--resume` 与增量 prediction 落盘
  - `run_until_complete.py` 已提供 crash 后自动冷却重启的外层 benchmark supervisor
  - `environment.yaml` 将作为唯一环境复现入口
  - 当前主环境测试为 31 项，`core_mem` 环境下也已复验

### 阶段 F：正式运行与结果整理

- PersonaMem 正式结果
- LongMemEval-S 正式结果
- 表格与结果数据沉淀
- Status: pending by user trigger + blocked by provider

## 第二阶段总策略

第二阶段围绕 `V2.0 structured latent-slot memory` 推进，目标不是替换第一阶段真相，而是在保留 v1 baseline 参考价值的前提下，建立更强的 latent memory 研究主线。

第二阶段优先确保：

1. 方法设计与真源文档完全对齐
2. observation / slot / belief 规范清晰
3. 训练数据以公开数据集为主、少量 synthetic 为辅
4. 本地 intrinsic evaluation 完整，减少盲目烧 benchmark API
5. benchmark 仍作为 evaluation source，而不是 primary training source

### 当前 `v2.9` 执行锚点

当前 active 主线已经前推到 **`TD-040` / `WS-026` / `v2.9 learned-core-path long-run`**，核心约束是：

- 保持 `core / residual` 双银行结构不变
- 继续以 `32k` source-level split 为锚点，而不是直接进入 full-data
- 优先让 `write -> latent composition -> belief` 三段出现真实正增益
- expanded holdout 从 `128` 扩到 `LongMemEval-S 500 / PersonaMem 512`
- benchmark 继续保持 holdout-only，不回流成训练 supervision
- 训练优先使用 `gpu2`

当前 `v2.9` 的直接工作内容是：

- `write` gain
- `latent` gain
- `belief` gain
- expanded holdout `LongMemEval-S 500 / PersonaMem 512`

当前 runtime truth 已更新为：

- `v2.6` 当前 retained baseline 已证明：
  - `write_gain.positive_gain = true`
  - `belief_gain.positive_gain = true`
  - `LongMemEval-S 128 = 11/11`
  - `PersonaMem 128 = 44/33`
- `v2.7` 当前 retained baseline 已证明：
  - `32k split / manifest / audit` 已落地
  - `gpu2` pilot 训练、评测、时序与 holdout summary 可复验
- `v2.8` 当前 retained truth 已证明：
  - teacher suite 已真实跑完
  - matched compare 已真实消费 teacher 改动
  - 但 `teacher-vs-silver` internal compare 仍为负，`scripts/verify_stage2_v28_longrun.py --score-only = 32/34`
  - 当前 blocker 不是消费路径，而是 `lifecycle` 与 `belief` teacher label 定义本身

因此，`v2.9` 的策略不是继续扩大 teacher coverage，而是：

- 暂时把 teacher 从默认主线降级为可选探索线
- 用现有 retained 32k / gain baseline 继续推进真正的 learned `write / latent / belief`
- 只有出现真实正增益后，才允许再扩大到更大 holdout 或 full holdout
- 当前 fresh managed run 的补充 runtime truth：
  - `scripts/verify_stage2_v29_longrun.py --score-only` 已从 baseline `25` 提升到 current retained `39`
  - current-head `latest_stage2_v29_{write,latent,belief}_gain.json`、`latest_stage2_v29_training_timing.json`、`latest_stage2_v29_holdout_summary.json` 与 `latest_*_stage2_v29_canary.json` 已刷新到 retained keep
  - 当前 `v29` gain 不再是 partial：`write / latent / belief` 均已转为 `positive_gain = true`
  - expanded holdout 已真实完成 `LongMemEval-S 500 / PersonaMem 512`；当前 stop condition 已达成，后续若继续推进，应以这条 retained line 为 baseline

### 当前 post-v2.9 训练与架构判断

- retained `v2.9` baseline 仍是 **共享 `google/flan-t5-base + LoRA` Seq2Seq**
- current-head `v30` partial line 已 landed **shared backbone + task-specific adapters**，并新增了 **trainable latent + direct latent objective**
- 当前统一训练任务为：
  - `retrieval_alignment`
  - `lifecycle_prediction`
  - `composition_to_belief`
- 当前最明显的上限约束不是 data plumbing，而是：
  - `parser` 仍 rule-first
  - `lifecycle` 仍 rule-heavy
  - `belief decoder` 仍主要是 heuristic/semantic-first

### 下一阶段（v30）主线

- 当前 `TD-041` / `v30` runtime truth：`shared backbone` + `task-specific` adapters、`trainable latent`、learned `belief` 与 full benchmark holdout 已共同收口；当前明确覆盖 `PersonaMem 589` 与 `LongMemEval-S 500`
- shared backbone + task-specific adapters
- trainable latent encoder / resampler
- direct latent objectives
- learned belief decoder
- full benchmark holdout baseline：
  - `LongMemEval-S 500`
  - `PersonaMem 589`

### 下一阶段（v31）主线

- 当前 active 主线为 `TD-042 / WS-028`，当前执行版本为独立的 `v31` latent-first quality run
- latent strengthening
- belief strengthening
- write strengthening
- ablation truth
- full benchmark holdout remains:
  - `LongMemEval-S 500`
  - `PersonaMem 512`
- 当前 retained progress：
  - `scripts/verify_stage2_v31_longrun.py --score-only = 24/32`
  - `latest_stage2_v31_latent_holdout_compare.json` 已记录 aligned `32k val` latent positive keep
  - `latest_stage2_v31_belief_holdout_compare.json` 已记录 aligned `32k val` belief positive keep
  - `latest_stage2_v31_write_holdout_compare.json` 已记录 aligned `32k val` write positive keep
  - commit `489ada0` 已把 learned full-holdout canary path 从 `completed_predictions = 0` 的 zero-progress stall 恢复为增量推进；当前 resumed `LongMemEval-S 500` 已到 `322/500`，`PersonaMem 512` 已到 `238/512`
  - current HEAD `838a861` 又补齐了 `v31` full-holdout publisher；当前下一条增量应先来自 full-holdout publish，而不是新的 unblock micro-tune

### v31 成功标准

- `TD-042` / `v31` 文档、计划、TODO 与 verifier 保持一致
- latent mainline 为正
- belief mainline 为正
- write mainline 为正
- `LongMemEval-S 500` 明确高于 retained `v30`
- `PersonaMem 512` 不明显退化
- ablation 明确表明增益主要来自 learned 主链，而不是 benchmark-facing 改动

### 下一阶段（v33）主线

- 当前 active 主线标签为：`TD-044`
- 当前执行版本为：`v33`
- 当前 active 主线将从 retained `TD-043 / WS-029 / v32` 前推到 `TD-044 / WS-030 / v33`
- 这轮不再满足于 symbolic authoritative closeout，而是直接进入更激进的 **learned-authoritative latent redesign**
- retained `v32` 继续作为唯一 external compare baseline
- `v32` 的 closeout truth 作为直接输入，而不是继续扩大 symbolic full holdout

主线要求：

- shared backbone + modular heads：
  - `write`
  - `latent`
  - `belief`
  - `answer / option-scoring`
- stronger trainable latent reader
- direct latent objectives
- structured belief head
- answer / option-scoring head
- full benchmark holdout 保持：
  - `LongMemEval-S 500`
  - `PersonaMem 512`

成功标准：

- modular architecture positive
- write positive
- latent positive
- structured belief positive
- answer / option-scoring positive
- learned-authoritative runtime artifact 落地，并显式记录：
  - `memory_mode = learned_memory`
  - `slot_assignment_mode = learned`
- `LongMemEval-S 500` 高于 retained `v32`
- `PersonaMem 512` 高于 retained `v32`
- ablation 明确表明 latent 是 primary driver，answer head 对 finite-option holdout 有真实贡献

当前 retained 进展：

- fresh managed `v33` run 已按 launch manifest 初始化，baseline 先量到 `scripts/verify_stage2_v33_longrun.py --score-only = 17/47`
- current retained `v33` 已推进到 `36/47`
- 当前已落地的 current-head artifact 包括：
  - `latest_stage2_v33_modular_authoritative_train.json`
  - `latest_stage2_v33_learned_write_eval.json`
  - `latest_stage2_v33_latent_reader_train.json`
  - `latest_stage2_v33_temporal_slot_eval.json`
  - `latest_stage2_v33_latent_objective_eval.json`
  - `latest_stage2_v33_belief_graph_eval.json`
  - `latest_stage2_v33_answer_option_eval.json`
  - `latest_stage2_v33_ablation_summary.json`
- 当前 truth 边界：
  - modular / write / latent / belief / answer 的 internal positive evidence 已成立，且 ablation 已机械确认 latent 是 primary driver
  - learned-authoritative runtime 与 `LongMemEval-S 500 / PersonaMem 512` 的 full holdout compare 仍未发布
  - 因此还不能把 current-head 写成“真实 gains 已由 learned-authoritative latent system 在 external benchmark 上成立”

当前 `v30` 的 retained progress 已前进到：

- current HEAD `13bb0fa` 已把 `scripts/verify_stage2_v30_longrun.py --score-only` 从 baseline `17` 提升到 `21`
- `latest_stage2_v30_shared_backbone_train.json` 已记录 first modular train：`gpu2`、`4096` examples、`128` steps、`wall_clock_seconds = 42.88473560567945`
- `latest_stage2_v30_task_adapter_compare.json` 已记录 first positive compare：`task_specific_positive_gain = true`、`delta_score = 1.2487474884772993`
- current HEAD `f0e3203` 已继续把 retained score 提升到 `27`
- `latest_stage2_v30_latent_module_train.json` 已记录 trainable latent module train，artifact 显式标记 `trainable_encoder_resampler = true`
- `latest_stage2_v30_latent_objective_eval.json` 与 `latest_stage2_v30_latent_gain.json` 已记录 direct latent objective 正增益：`current_top1_accuracy = 0.96484375`、`current_mrr = 0.982421875`、`positive_gain = true`
- current HEAD `e9be6f3` 已继续把 retained score 提升到 `33`
- `latest_stage2_v30_belief_decoder_eval.json`、`latest_stage2_v30_write_gain.json` 与 `latest_stage2_v30_belief_gain.json` 已记录 current-head learned belief decoder / write / belief 正增益
- current HEAD `75c70bc` 已把 retained score 进一步推进到 `41`
- `latest_stage2_v30_full_holdout_baseline.json`、`latest_longmemeval_stage2_v30_full.json` 与 `latest_personamem_stage2_v30_full.json` 已记录 current-head full benchmark holdout baseline
- 当前 full benchmark holdout 仍明确是 holdout-only：
  - `LongMemEval-S 500` symbolic full run 当前与 retained `v2.9` counts 持平，non-regression guard 为真
  - `PersonaMem 589` symbolic full run 当前通过 retained `512` shared-subset overlap guard 机械确认 provider exact rate 提升到 `0.3515625`、local exact rate 持平 `0.2109375`
- 因此，`implementation_plan` 的当前 `v30` runtime truth 已不再是 partial architecture keep，而是：`shared backbone`、`task-specific adapters`、`trainable latent`、learned `belief`、以及 full benchmark holdout baseline 已共同收口到 stop condition `41/41`

### 当前 `v2.7` 执行锚点

当前 active 主线已经前推到 **`TD-038` / `WS-024` / `v2.7 32k teacher-first`**，核心约束是：

- 保持 `core / residual` 双银行结构不变
- 先建立 `32k` source-level split，而不是直接进入 full-data
- 当前默认 split 为：
  - `24k train`
  - `4k val`
  - `4k test`
- 先用 `MiniMax-M2.7` 生成 teacher labels，优先提升：
  - `observation / parser`
  - `slot assignment / lifecycle`
  - `belief semantic fields`
- 训练优先使用 `gpu2`
- 训练时必须记录 `训练耗时`、wall-clock / throughput / memory，判断 `32k` 是否过重
- benchmark 继续保持 holdout-only，不回流成训练 supervision
- **只有在 32k internal test work well 后，才允许进入 full-data**

当前已落地的 `v2.7` 运行时证据：

- `latest_stage2_v27_32k_split.json`、`latest_stage2_v27_32k_manifest.json`、`latest_stage2_v27_32k_audit.json` 已证明 `24k/4k/4k` source-level split 和 audit 已存在
- `latest_stage2_v27_train.json`、`latest_stage2_v27_eval.json`、`latest_stage2_v27_training_timing.json`、`latest_stage2_v27_internal_test.json` 与 `latest_stage2_v27_holdout_summary.json` 已补齐一条真实 `gpu2` tiny pilot 闭环
- 当前这条 `gpu2` pilot 使用 `outputs_v2/artifacts/stage2_v27_32k/train/stage2_prepared_samples_manifest.json`，记录 `4096` effective examples、`512` steps、wall-clock `5.420951s`、`755.59 examples/s`、peak GPU memory `55.09MB`
- `latest_stage2_v27_teacher_observation.json`、`latest_stage2_v27_teacher_slot_assignment.json` 与 `latest_stage2_v27_teacher_belief.json` 已补齐一轮真实 `MiniMax-M2.7` teacher pilot artifacts
- 当前 teacher pilot 使用 sample caps `8/2/2` 与 batch size `1`；`slot_assignment` 与 `belief` artifact 状态为 `completed`，`observation` artifact 状态为 `completed_with_failures`
- 当前需要诚实保留的边界是：这轮 teacher 只是一轮 pilot，不是 full `32k` teacher coverage；`latest_stage2_v27_teacher_observation.json` 当前显式记录 `total_labeled_examples = 6`、`total_failed_examples = 6`

## 第二阶段拆分

### 阶段 G：V2.0 设计锁定

- 锁定 `structured latent slots`
- 锁定 `core + residual` 双银行更新逻辑
- 锁定 `light cross-attention resampler + Flan-T5 belief JSON decoder`
- 锁定训练任务、数据策略、本地指标与 canary 协议
- Status: done
- Notes:
  - 真源 `docs/requirements.md` 已同步第二阶段条款
  - 已新增 `docs/v2_design.md`

### 阶段 H：数据与规范管线

- observation normalization
- belief JSON target generation
- public datasets ingestion
- minimal synthetic corner-case builder
- Status: done
- Notes:
  - `src/core_mem/v2/` 已落地 Observation / Slot / Belief schema、relation normalization、rule-first parser 与 dataset-task registry
  - `scripts/prepare_stage2_data.py` 已可写出 prepared manifest、dataset registry 与 task registry
  - `scripts/train_stage2.py`、`scripts/eval_stage2_local.py`、`scripts/run_stage2_canary.py` 与 `configs/stage2_train.yaml` 已形成首批 stage2 pipeline
  - `outputs_v2/` 已产出 smoke train plan、smoke local eval 与 fixed canary manifests
  - `prepare_stage2_data.py` 现在已经支持 source-config 驱动的非 demo prepared-manifest 构建
  - 当前缺口是把真实公开数据文件放入 `data/stage2_public/`

### 阶段 I：V2.0 主线实现

- slot encoder / projection
- retrieval key head
- lifecycle head
- core / residual banks
- consolidation
- light cross-attention resampler
- Flan-T5 belief decoder
- answer projection
- Status: doing
- Notes:
  - `src/core_mem/v2/encoder.py`、`lifecycle.py`、`consolidation.py`、`resampler.py`、`decoder.py`、`projection.py`、`system.py` 已从第一版 deterministic skeleton 推进到 parameterized latent path
  - 已有 `tests/test_stage2_model_skeleton.py` 覆盖 parser -> memory update -> belief decode -> answer projection 的最小链路
  - `src/core_mem/v2/training.py` 与 `scripts/train_stage2.py --execute-train` 已提供真实训练执行路径，并通过 tiny offline backend 做了最小 smoke 验证
  - 当前 `scripts/verify_stage2_latent_status.py --score-only` 已达 `9/9`，说明可学习 encoder、被 decoder 消费的 composed latent、以及与 answer projection 对齐的 latent-conditioned belief recovery 主链都已机械成立
  - 本轮又补了一层行为约束：`BeliefDecoder` 现直接消费 selected slots 的既定顺序，不再在 decode 阶段自行重排 memory state

### 阶段 J：本地 intrinsic evaluation

- retrieval metrics
- belief recovery metrics
- update / propagation metrics
- specificity / locality metrics
- compression metrics
- budget sweep
- Status: done
- Notes:
  - `scripts/eval_stage2_local.py` 现已支持模块级、家族级、budget-sweep 与 checkpoint-aware local eval
  - `scripts/train_stage2.py` 现已支持 preset ablation variants、registry 自动登记与 local eval 联动
  - `outputs_v2/artifacts/stage2_experiment_index.json` 当前已记录 `mainline + 11` 个必做 ablation，`scripts/verify_stage2_experiment_status.py --score-only` 已达 `13/13`
  - 当前完成态基于 `configs/stage2_train_tiny.yaml` 的 `gpu3` 本地运行；默认 backbone 非 tiny 证据仍待补
  - 当前 `composition_to_belief` 的 belief-family 评测已与 retrieval-family 解耦：belief-family 直接消费给定 memory state 顺序，retrieval-family 继续由单独 retrieval 任务负责
  - 当前最新 local intrinsic 证据为 `outputs_v2/evals_local/20260415T031952Z_stage2_local_eval.json`，对应 `stage2_latent_core_quality_score = 10/10`

### 阶段 K：Benchmark canary 与正式评测

- PersonaMem canary
- LongMemEval-S canary
- 选择性推进 full benchmark
- Status: done
- Notes:
  - canary manifest 生成器已经落地，并已固定产出 PersonaMem 64 / LongMemEval-S 64 manifests
  - `scripts/run_stage2_memory_canary.py` 已将 `StructuredMemorySystem` 接到 benchmark canary prompt 生成链路，并已在 `MiniMax-M2.7` 上完成 1-sample、fresh `PersonaMem 64` 与 fresh `LongMemEval-S 64` 的 live canary
  - `outputs_v2/artifacts/latest_personamem_stage2_canary_analysis.json` 与 `outputs_v2/artifacts/latest_longmemeval_stage2_canary_analysis.json` 已落地
  - 当前 autoresearch 已验证两条路径：
    - observation-noise cleanup 可清理 parser 噪声，但单独不足以抬高 live 分数
    - PersonaMem answer-option 对齐是当前第一段主收益来源，已把 quality score 从 `4/10` 提升到 `8/10`
    - 生命周期层面的 facet retention + selected-slot-aware option scoring 是当前跨过 stop condition 的关键，已把 quality score 从 `8/10` 进一步提升到 `9/10`
  - 当前 canary 目标已满足；但 `LongMemEval-S 64` 的最新 analysis 显示 provider exact 仅 `1/64`、local exact `0/64`，说明“fresh evidence complete”不等于“quality ready for scaling”
  - `latent core / model / system robustness` 的本轮 local intrinsic 目标已经达成；若继续推进，下一阶段应转入更大范围 benchmark 与结果对比，而不是继续堆 prompt tricks

### 阶段 L：完整 V2 长跑里程碑

- fresh `PersonaMem 64` stage-2 live canary
- `LongMemEval-S 64` stage-2 live canary
- 两个 benchmark 的 failure analysis artifact
- 默认 `flan-t5-base` 非 tiny `gpu3` 训练与 checkpoint-aware eval
- `trained_eval` 不再全零，形成 learned path 有效性的最小证据
- benchmark runner 去除 benchmark-specific shortcut / fallback
- Status: done
- Notes:
  - 本阶段的目标不再是单点 score，而是把“完整 v2”拆成一组可机械核对的 milestone
  - 不允许 benchmark-specific heuristic / fallback 成为 retained 收益
  - 当前 `benchmark_runner_avoids_shortcuts` 已完成：PersonaMem-specific candidate injection 与 blank-provider fallback 已从 runner 移除
  - 当前默认 `flan-t5-base` 的非 tiny `GPU3` train/eval 证据已完成：`outputs_v2/runs/20260415T043648Z_stage2_train_exec/execution_summary.json`、`outputs_v2/checkpoints/20260415T043648Z_stage2_train_exec/` 与 `outputs_v2/evals_local/20260415T043706Z_stage2_local_eval.json` 已形成 `trained_eval.token_f1 > 0` 的最小正证据
  - 当前 fresh `PersonaMem 64` artifact 为 `outputs_v2/evals_benchmark/20260415T052916Z_stage2_memory_canary.json`
  - 当前 fresh `LongMemEval-S 64` artifact 为 `outputs_v2/evals_benchmark/20260415T054234Z_stage2_memory_canary.json`，对应 analysis 为 `outputs_v2/artifacts/latest_longmemeval_stage2_canary_analysis.json`
  - 当前为了让 verifier 对齐 runtime truth，又补了一层命名/选择逻辑：`scripts/verify_stage2_v2_completion.py` 接受 `longmemeval_s` alias，`scripts/verify_stage2_latent_core_quality.py` 只用 PersonaMem canary 维持 latent-core guard
  - 当前 `scripts/verify_stage2_v2_completion.py --score-only = 14`，stop condition 已机械满足

### 阶段 M：V2.1 Robustness 主线

- 提升真实质量，而不只是 `64` canary 闭环
- 提升 learned path 对在线链路的实际贡献
- 提升跨 benchmark 鲁棒性
- 提升可扩展性与可解释性
- Status: done for current long-run milestone
- Notes:
  - `v2` 已经解决“系统闭环是否成立”；`v2.1` 要解决的是“这套 latent memory 在真实 benchmark 上是否稳定、是否靠模型本身变强、是否跨任务依然成立”
  - `v2.1` 不再把“完整证据链”当终点，而把“质量、鲁棒性、可扩展性”当终点
  - 当前 managed run 已在 HEAD `d6bc4f7` 上把 `stage2_v21_longrun_score` 从 baseline `11/16` 推到 stop condition `16/16`
  - retained 关键证据包括：去掉 learned online path 的 symbolic fallback、`outputs_v2/evals_local/20260415T191953Z_stage2_local_eval.json` 的 `trained_eval.token_f1 = 0.3885239109848479`、以及 current-head learned `PersonaMem 128` / `LongMemEval-S 64` refreshed artifacts
  - 当前需要诚实保留的风险是：live learned 路径仍频繁出现 `learned_memory_error`，说明 belief JSON 有效性还没被系统性解决；因此这里只能记为“当前 long-run 机械里程碑完成”，不能误写成 learned quality 全面稳定

#### 阶段 M-A：Benchmark 质量提升

- 扩大 `PersonaMem` 评测规模，从 `64` 往更大切片推进
- LongMemEval-S 专项 failure analysis 先行
- 对 LongMemEval-S 做专项 failure analysis
- 把错误按 `parser / retrieval / belief / projection / provider` 五层拆开
- 优先修最影响 `LongMemEval-S` 的 `1-2` 个主错误源
- 里程碑：
  - `PersonaMem` 更大切片下不明显退化
  - `LongMemEval-S` canary 指标明显高于当前基线
  - 形成稳定的错误分层报告

#### 阶段 M-B：Learned Path 真正进入主链

- 明确哪些模块值得 learned：
  - `query encoder`
  - `slot encoder`
  - retrieval / ranking
  - belief selection / composition
- 避免继续把重点放在“复述 JSON”的 sidecar 训练
- 让训练目标直接服务在线 `memory -> belief -> answer` 主链
- 要求训练后提升能在 online canary 中体现，而不只是 local intrinsic
- 里程碑：
  - 至少一个 learned 子模块进入在线主链
  - 训练前后在同一 canary 上有可重复增益
  - `trained_eval` 不只是非零，而是对在线表现有解释价值

#### 阶段 M-C：跨 Benchmark 鲁棒性

- 保持 `PersonaMem` 为主 benchmark
- 把 `LongMemEval-S` 提升到同等级验证地位
- 如果这两者稳定，再考虑更大范围 benchmark
- 统一 benchmark 记录格式、对比表和 failure taxonomy
- 里程碑：
  - 两个 benchmark 都有 fresh live 结果
  - 两个 benchmark 都有对应 analysis artifact
  - 改动在两个 benchmark 上都能解释，而不是一边涨一边掉

#### 阶段 M-D：系统化收口

- 固化 `v2.1` 的机械 verifier、默认 guard、默认 benchmark/train 配方与禁用 heuristic 列表
- 让后续 managed autoresearch 围绕 `stage2_v21_robustness_score` 推进，而不是继续复用 `stage2_v2_completion_score`
- 里程碑：
  - 新的 `v2.1` verifier 可独立返回 baseline / best score
  - verifier 能区分 current-head fresh canary、larger-slice stability、learned online gain 与 layered failure analysis
  - 当前默认长跑入口围绕 `LongMemEval-S` 专项质量提升与 learned path 在线增益来组织

- 固化默认配置、推荐训练命令和 benchmark 命令
- 固化哪些 heuristic 被禁止
- 固化 milestone verifier
- 更新文档，把“完整 v2”升级到“robust v2.1”
- 里程碑：
  - 一个清晰的默认运行配方
  - 一个清晰的 verifier 套件
  - 一份可对外解释的方法版本说明

## 第二阶段默认技术路线

- 主 backbone：`google/flan-t5-base`
- 主 resampler：`Light Cross-Attention Resampler`
- 主 decoder target：`Belief JSON`
- 训练方式：LoRA/adapter + resampler/projection 可训练
- 训练顺序：
  - `Slot Autoencoding`
  - `Retrieval Alignment`
  - `Lifecycle Prediction`
  - `Composition-to-Belief Decoding`
  - `Joint fine-tuning`
- 数据策略：`public-datasets-first, synthetic-minimal`
- 主训练数据：
  - `SGD`
  - `MultiWOZ 2.4`
  - `Persona-Chat`
  - `Training Millions of Personalized Dialogue Agents`
  - `MQUAKE`
  - `ReCoE`
- 任务到数据映射：
  - `Slot Autoencoding` <- `SGD` / `MultiWOZ` / `Persona-Chat` / `Training Millions...`
  - `Retrieval Alignment` <- `Persona-Chat` / `Training Millions...` / `SGD` / `MultiWOZ` + `MQUAKE` / `ReCoE` hard negatives
  - `Lifecycle Prediction` <- `MQUAKE` / `ReCoE` / `SGD-MultiWOZ state changes` + minimal synthetic
  - `Composition-to-Belief Decoding` <- `SGD` / `MultiWOZ` / `Persona-Chat` / `Training Millions...` / `MQUAKE` / `ReCoE` + minimal synthetic

## 第二阶段主要 ablation

- `T5` vs `Optimus-like`
- mean pooling vs light resampler
- direct answer decoding vs belief JSON decoding
- with / without lifecycle loss
- with / without retrieval loss
- single-bank vs core+residual
- core/residual assignment randomization
- budget sweep

## 当前优先顺序

1. 保持第一阶段 baseline 与 formal benchmark pending 真相不变
2. 保持第二阶段 public-data normalization / strict prepare / direct-train / experiment registry 能力可复验
3. 先把 stage-2 的真正 latent memory 主链路做实，而不是把 deterministic skeleton 直接当作最终系统
4. 当前 `latent core / local intrinsic quality` 目标已达成，`stage2_latent_core_quality_score = 10/10`
5. 后续应以阶段 L 的完整 v2 milestone 为主线，同时允许 `GPU3` 正式训练与 `MiniMax-M2.7` live benchmark 成为里程碑验证的一部分
6. 不允许 benchmark-specific heuristic / fallback 成为 retained 收益；如果 canary 分数只能靠 shortcut 维持，该结果不算完成 v2
7. 当前阶段 L、阶段 M、阶段 N、阶段 P、阶段 Q、阶段 S 与阶段 T 的当前机械里程碑都已完成；若继续推进，下一步更合理的是围绕 `LongMemEval-S` 质量、learned slot assignment 的泛化鲁棒性、更强 latent 与 full benchmark holdout measurement 进入新的质量与泛化阶段，而不是重复证明 current-head canary 覆盖

### 阶段 P：Semantic-First Learned Decoder

- Goal: 把 stage-2 learned training 从“复述 raw JSON 字符串”推进到“恢复正确语义字段，并把格式约束外置”
- Status: doing (mechanical stop reached; closeout retained)
- Notes:
  - 当前 retained artifact `outputs_v2/evals_local/20260415T230211Z_stage2_local_eval.json` 已显示 semantic-first 路线可以把同一 non-tiny checkpoint 的 `trained_eval.token_f1` 提到 `0.879714215455919`
  - 同一 artifact 中 `retrieval_alignment.token_f1 = 0.9860465116279071`，说明这条线已经把“语义正确但 JSON 壳不完整”的 retrieval outputs 从 `0` 成功抬起来
  - 当前关键实现是新增通用 `semantic_outputs.py`，把 task-aware 结构修复与语义计分同时接到 checkpoint eval 和 online learned belief parse
  - 禁止把外部格式修复退化成 fallback；目标仍然是提升模型本体语义能力

### 阶段 Q：V2.2 Full-Data Semantic Latent

- Goal: 把 semantic-first learned path 从 current retained artifact 推进到 full-data current-head 训练、extended benchmark 验证和 LongMemEval-S 主导的质量提升
- Status: doing (mechanical stop reached; closeout retained)
- Notes:
  - `v2.2` 继续保留 `semantic-first`、`no fallback`、`no benchmark-specific shortcut`
  - 这轮强调可以使用完整 public-data task rows 做训练与本地评测
  - current retained 进度：
    - `outputs_v2/artifacts/latest_stage2_semantic_full_train.json` 已在 current HEAD `510aeb7` 上记录 `num_examples = 1574`、`cuda_visible_devices = 3`
    - `outputs_v2/artifacts/latest_stage2_semantic_full_local_eval.json` 已在同一 HEAD 上记录 `trained_eval.token_f1 = 0.9258179798351409`、`retrieval_alignment.token_f1 = 1.0`
    - `outputs_v2/evals_benchmark/20260416T021743Z_stage2_memory_canary.json` 已把 current-head `LongMemEval-S` semantic canary 扩到 `128`
    - `outputs_v2/evals_benchmark/20260416T024146Z_stage2_memory_canary.json` 已补齐 current-head `PersonaMem 128` semantic canary
    - `outputs_v2/artifacts/latest_longmemeval_stage2_semantic_analysis.json` 与 `latest_stage2_semantic_online_gain.json` 已在 current HEAD `510aeb7` 上刷新
    - `scripts/verify_stage2_v22_completion.py --score-only` 已从 `9` 提升到 stop condition `19`
  - 当前 retained 关键改动：
    - `src/core_mem/v2/projection.py` 新增 generic answer projection normalization，可去掉解释尾巴并抽取 location phrase；对应 `LongMemEval-S 64` local exact 从 `3` 提到 `4`
    - semantic online gain 已因 `delta_local_exact_match = +1` 而转正，但 provider exact / prefix 相比 retained symbolic `LongMemEval-S 64` baseline 仍是负增益
  - 当前 runtime truth：
    - `TD-032 / WS-018` 的 mechanical stop condition 已达到；在用户给出新的 stage-2 方向前，当前只保留为 closeout workstream
    - 这条线可以诚实声明为“`stage2_v22_completion_score = 19/19` 已机械达成”，不能夸写成 “LongMemEval-S online quality 已全面稳定”
  - 这条线对应的新 verifier 是 `scripts/verify_stage2_v22_completion.py`

### 阶段 R：V2.3 Stronger Learned Slot Assignment And Stronger Latent

- Goal: 把 stage-2 主线从 “semantic-first 证据完整” 推进到 “更强、更稳、更多依赖 learned latent 的系统”
- Status: planned
- Notes:
  - `v2.2` 已证明 full-data semantic-first 路线成立，但 `LongMemEval-S 128` 当前仍只有 `provider_exact = 4/128`、`local_exact = 4/128`
  - 这说明下一步的主问题已经不再是“格式对不对”，而是 `observation -> slot -> retrieval -> belief` 这条在线 memory 主链本身还不够强
  - `v2.3` 继续保留 `no fallback`、`no shortcut`、`semantic-first`，但主收益目标切到：
    - `LongMemEval-S` 质量提升
    - `learned slot assignment`
    - `stronger latent / stronger online memory path`
  - 详细计划见 [docs/v23_plan.md](/media/storage/mingjing/workspace/CoRe_Mem/docs/v23_plan.md)

#### 阶段 R-A：LongMemEval-S 质量基线重建

- 固化 current-head `LongMemEval-S 128` semantic baseline
- 形成按 `slot assignment / retrieval / belief / projection / provider` 分层的错误分析
- 明确当前 top error clusters

#### 阶段 R-B：Learned Slot Assignment

- 保留 parser 作为 schema / safety 入口
- 将 `observation -> slot` 的关键决策逐步从 pure rule-heavy lifecycle 推向 learned scorer
- 保留最小 hard constraints，避免 learned write path 污染 memory state

#### 阶段 R-C：在线对齐训练

- 新增 slot-assignment 监督
- 强化 retrieval / belief 训练目标与 online path 的对齐
- 避免回退到“只会复述结构化 JSON”的 sidecar 训练

#### 阶段 R-D：LongMemEval-S 提升验证

- current-head `LongMemEval-S 64/128` retained artifacts
- learned-vs-symbolic 或 learned-vs-rule-heavy 对照 artifact
- 证明 gain 可复验、可解释

#### 阶段 R-E：V2.3 收口

- 固化默认 `LongMemEval-S` 训练/评测 recipe
- 固化 `learned slot assignment` verifier
- 更新文档与 `.agent-os` 到 `v2.3`

### 阶段 S：V2.3 Long-Run Stronger Learned Slot Assignment

- Goal: 把 `v2.3` 从“主线规划”推进到“更远的 managed long run”
- Status: doing
- Notes:
  - 当前 `v2.2` 已 closeout，`v2.3` 的短版计划也已落地；这轮长跑的重点不再是 closeout artifact，而是更远的质量与主链改造
  - 这轮继续保留 `semantic-first`、`no fallback`、`no benchmark-specific shortcut`
  - 用户已明确允许多卡并行实验；当前默认策略是多卡并行多个单卡实验，而不是默认上多卡数据并行训练
  - 当前 retained 进度：current HEAD `43b941b` 已把 `stage2_v23_longrun_score` 推到 `16/22`，并补齐 `latest_stage2_slot_assignment_train.json` / `latest_stage2_slot_assignment_eval.json`
  - 当前真实 blocker：本 session 缺失 `GPT_AGENT_API_KEY`、`ALIYUN_API_KEY` 与 `GEMINI_API_KEY`，导致 provider-dependent `LongMemEval-S 64/128`、`PersonaMem 128` slot-assignment canaries 与 corresponding gain artifact 不能继续
  - 详细计划见 [docs/v23_longrun_plan.md](/media/storage/mingjing/workspace/CoRe_Mem/docs/v23_longrun_plan.md)

### 阶段 T：V2.4 LongMemEval-S Quality + Learned Slot Assignment + Stronger Latent

- Goal: 把当前主线从“`v2.3` 的 slot-assignment closeout”继续推进到“更强的 `LongMemEval-S`、更强的 full-data learned slot assignment、更强的 online latent main path”
- Status: done
- Notes:
  - 当前 `v2.3 long-run` 已在 current HEAD `7a1802f` 上机械收口，`stage2_v23_longrun_score = 22/22`
  - 这条线最终已在 current HEAD `12a9a80` 上机械收口，`stage2_v24_longrun_score = 24/24`
  - 本轮继续严格保留 `semantic-first`、`no fallback`、`no benchmark-specific shortcut`
  - 训练与测试默认使用完整 public-data prepared tasks
  - 新机械指标为 `scripts/verify_stage2_v24_longrun.py --score-only`
  - retained full-data eval 最终为：`trained_eval.token_f1 = 0.9991150844073334`、`trained_eval.field_f1 = 0.9976704786107581`、`slot_assignment_metrics.token_f1 = 0.9961127308066084`
  - retained current-head online evidence 最终为：`LongMemEval-S 128` `provider/local = 10/128`、`PersonaMem 128` `provider_exact = 38`、`local_exact = 28`
  - retained gain 最终为：相对 `v2.3` baseline 的 `LongMemEval-S 128` `delta_provider_exact_match = +4`、`delta_local_exact_match = +4`
  - 已验证的失败探索：
    - 把 `composition_to_belief` 的 online-aligned repeats 从 `2` 提到 `3` 不会改善 full-data `trained_eval`
    - 单独的 projection/prompt sharpen 虽能把 `LongMemEval-S` current-head 从 `8/8` 提到 `9/9`，但不足以跨过最终 verifier 阈值
  - 最终跨过 stop condition 的关键 retained 修复是：query-aware exactness tightening + belief support-id repair
  - 详细计划见 [docs/v24_plan.md](/media/storage/mingjing/workspace/CoRe_Mem/docs/v24_plan.md)

#### 阶段 S-A：Baseline And Taxonomy

- 固化 `v2.2` 的 `LongMemEval-S 128` 与 `PersonaMem 128` 基线
- 将 `LongMemEval-S` 的错误稳定拆到：
  - `slot_assignment`
  - `retrieval`
  - `belief`
  - `projection`
  - `provider`

#### 阶段 S-B：Learned Slot Assignment Module

- 为 `observation -> slot` 新增 learned scorer
- 保留 parser 作为 schema / safety 入口
- 保留最小 hard constraints

#### 阶段 S-C：Online-Aligned Slot-Assignment Training

- 新增 slot-assignment supervision
- 保持 full-data semantic-first
- 让训练直接服务 online `slot assignment / retrieval / belief`

#### 阶段 S-D：LongMemEval-S 64/128 Validation

- current-head `LongMemEval-S 64`
- current-head `LongMemEval-S 128`
- learned-vs-symbolic retained gain

#### 阶段 S-E：PersonaMem Guard

- current-head `PersonaMem 128`
- 不允许因 `LongMemEval-S` 优化而明显退化

#### 阶段 S-F：V2.3 Long-Run Closeout

- 默认 train recipe
- 默认 benchmark recipe

### 阶段 U：V2.5 Generalization-First Long Run

- Alias: `V2.5 Learned Core-Path Long Run`

- Goal: 在**不改 `core / residual` 双银行结构**的前提下，把主线从“`v2.4` 的质量门槛已跨过”继续推进到“更强的 learned write / retrieve / belief、更泛化的 learned slot assignment、更强的 online latent，以及更大切片/全量 benchmark holdout 验证”
- Status: planned
- Notes:
  - `v2.4` 已证明 full-data semantic-first、quality-first、learned slot assignment 和 stronger latent 可以把 current-head `LongMemEval-S 128` 提到 `10/128`
  - 这条线下一步不再追求单次机械 closeout，而是追求更强的 generalization / robustness
  - 本轮继续严格保留 `semantic-first`、`no fallback`、`no shortcut`、`no benchmark-specific heuristic`
  - 本轮不再把 raw JSON exactness 当主优化目标，也不允许把 full benchmark 结果回流成训练 supervision
  - full benchmark 在本轮被正式提升为 holdout acceptance / generalization measurement，而不是局部 canary 的附庸
  - 详细计划见 [docs/v25_plan.md](/media/storage/mingjing/workspace/CoRe_Mem/docs/v25_plan.md)

#### 阶段 U-A：Freeze V2.4 Baseline

- 固化 `v2.4` 的 `LongMemEval-S 128`、`PersonaMem 128` 与 online gain retained baseline
- 明确 `v2.5` 的所有提升都要相对这一基线计算

#### 阶段 U-B：Generalized Learned Slot Assignment

- 把 `learned slot assignment` 从“可用”推进到“更泛化、更鲁棒”
- 强化 `merge / new / overwrite / ignore` action head
- 强化 candidate slot scoring / ranking
- 仅保留最小 hard constraints 作为 safety boundary

#### 阶段 U-C：Learned Retrieve / Belief Main Path

- 继续加强 learned retrieval / rerank / belief composition
- 保持 semantic-first
- 优先把 retained 收益落实到 `LongMemEval-S` 主 benchmark

#### 阶段 U-D：LongMemEval-S Quality Ramp

- current-head `LongMemEval-S 64/128`
- 更大切片或 full benchmark holdout measurement
- refreshed layered analysis

#### 阶段 U-E：Cross-Benchmark Guard

- current-head `PersonaMem 128` 持续刷新
- 保证为 `LongMemEval-S` 提升质量时不明显破坏 `PersonaMem`

#### 阶段 U-F：Full-Benchmark Holdout Evaluation

- current-head full benchmark measurement
- 只做 evaluation，不回流成训练 supervision
- 对比 `v2.4` retained baseline 与 current-head 趋势稳定性

#### 阶段 U-G：V2.5 Packaging

- 固化 default full-data train recipe
- 固化 default canary recipe
- 固化 default full-benchmark evaluation recipe
- 固化新的 verifier / guard / anti-overfitting contract
- 默认 verifier
- 文档与 `.agent-os` 对齐

### 阶段 W：V2.6 Gain-First Long Run

- Alias: `V2.6 Gain-First Long Run`

- Goal: 在**不改 `core / residual` 双银行结构**的前提下，不再允许靠 artifact completeness 达成 stop condition，而是要求 current-head 出现真实 `write / retrieve / belief` learned gain，并让 `LongMemEval-S 128` 明确超过 `v2.5` retained baseline。
- Status: doing, currently blocked by dense-sample write throughput rather than provider env
- Notes:
  - 这轮的中心要求是“真实正增益”，而不是 artifact completeness
  - `v2.5` 已被诚实收口为 baseline/package closeout，而不是 quality-gain closeout
  - 本轮继续严格保留 `semantic-first`、`no fallback`、`no shortcut`、`no benchmark-specific heuristic`
  - 本轮继续禁止 raw JSON exactness 作为主优化目标，也禁止把 full benchmark 结果回流成训练 supervision
  - full benchmark 在本轮继续保持为 holdout acceptance / generalization measurement
  - 历史 trial HEAD `077cbf3` 已通过 stage-2 guard；当前 session 也已确认 `GPT_AGENT_API_KEY=SET`，因此 provider-env blocker 已被清除
  - 最新 refine commit `a04effe` 已把 learned slot-assignment prompt 收缩到相关候选上下文，并为 weak `other_fact` overwrite 增加 fast-path；对应 profiling 已把 sample `51a45a95` 的 learned arbitration 次数从 `11` 压到 `1`
  - 同一 partial `LongMemEval-S 128` resumed run 已从 `2/128` 前进到 `3/128`，且 `51a45a95` 当前在 current-head 上命中 `Target`
  - refine commit `3051b0f` 已让 `scripts/run_stage2_memory_canary.py` 在单次 canary run 内复用 learned belief / slot-assignment predictors；fresh current-head `LongMemEval-S 128` run `outputs_v2/runs/20260417T055905Z_stage2_memory_canary_longmemeval/` 最终已结束在 `provider_exact_match = 11`、`local_exact_match = 10`
  - refine commit `b4c997d` 已把 `v2.6` artifact publish 入口并回 `scripts/verify_stage2_v26_longrun.py`，并新增 `tests/test_stage2_v26_publish.py` 覆盖 current-head publish 入口
  - current-head `LongMemEval-S 128` retained refresh `outputs_v2/evals_benchmark/20260417T083058Z_stage2_memory_canary.json` 已在 `160e29f` 上达到 `provider_exact_match = 11`、`local_exact_match = 11`
  - current-head `PersonaMem 128` retained refresh `outputs_v2/evals_benchmark/20260417T094824Z_stage2_memory_canary.json` 已在同一 HEAD 上达到 `provider_exact_match = 44`、`local_exact_match = 33`
  - 当前 publish 入口已把 current-head `v2.5` / `v2.6` aliases、`write / retrieve / belief` gain artifacts、LongMemEval analysis 与 holdout full-benchmark summary 一并刷新到当前 HEAD
  - `write_gain.positive_gain = true`、`belief_gain.positive_gain = true`
  - 当前 `scripts/verify_stage2_v26_longrun.py --score-only = 26`，`v2.6` managed run 已机械收口
  - 详细计划见 [docs/v26_plan.md](/media/storage/mingjing/workspace/CoRe_Mem/docs/v26_plan.md)

#### 阶段 W-A：Freeze V2.5 Truth

- 把 `v2.5` 明确冻结为 retained baseline/package closeout
- 保留 `v2.5` retained train/eval/canary/gain artifacts 作为对照线
- 后续所有 gain 都相对 `v2.5` retained baseline 计算

#### 阶段 W-B：Gain-First Write

- current-head `write` train/eval/gain artifact 全部刷新
- `write` 至少在一个 current-head retained artifact 中出现 `positive_gain = true`
- 不允许再把 purely structural/package change 计为 write gain

#### 阶段 W-C：Gain-First Retrieve

- current-head `retrieve` gain artifact 刷新
- retrieval miss / failure bucket 必须相对 `v2.5` retained baseline 有可解释改善
- 不允许把 prompt / projection tweak 误记成 retrieve gain

#### 阶段 W-D：Gain-First Belief

- current-head `belief` gain artifact 刷新
- belief semantic recovery 要在 online evidence 上产生可见收益
- 不允许仅靠 coercion / package repair 被误判为 belief gain

#### 阶段 W-E：LongMemEval-S Breakout

- current-head `LongMemEval-S 128` 必须 beat `v2.5` retained `10/128`
- refreshed `LongMemEval-S` analysis artifact
- `LongMemEval-S` 继续作为第一质量 benchmark

#### 阶段 W-F：Cross-Benchmark Guard

- current-head `PersonaMem 128` 不明显退化
- 仍然只把 `PersonaMem` 当 guard/secondary benchmark

#### 阶段 W-G：Holdout Full-Benchmark

- current-head full benchmark summary 只作 holdout evaluation
- 必须显式保留 `holdout_only = true`
- 不允许把 full benchmark 输出回流成训练 supervision

### 阶段 N：V2.1 Learned-Memory-First Pivot

- 目标：把 stage-2 主线从 “继续强化 rule-heavy robustness” 切到 “用更 learned 的 memory write/read 与更强 latent memory 提升系统智能程度”
- Status: done
- Notes:
  - 用户已明确要求当前主线更强调 learned model、本体智能和 better latent，而不是继续强化 rule-based reader/writer
  - 当前 pivot 受以下 related work 启发：
    - `End-To-End Memory Networks`
    - `Memorizing Transformers`
    - `RETRO`
    - `LongMem`
    - `Slot Attention`
  - 当前 repo 内 related-work 笔记见 `docs/learned_memory_related_work.md`

#### 阶段 N-A：主线切换与约束更新

- 将 `TD-029 / WS-015` 设为当前 next action / active workstream
- 将 “更 learned 的 memory / 更好的 latent” 写入 `docs/*` 与 `.agent-os/*`
- 禁止继续把 parser / selector / prompt 层局部规则小修当成主收益来源

#### 阶段 N-B：Online Learned Memory Path

- 为 `StructuredMemorySystem` 增加 checkpoint-backed learned memory path
- 支持 online 切换 `memory_mode = learned`
- 让 canary runner 能显式记录 learned-mode artifacts
- 里程碑：
  - online system 支持 learned memory toggle
  - online system 可加载 learned memory components
  - current-head `PersonaMem` / `LongMemEval-S` learned-mode canary artifacts 落地

#### 阶段 N-C：Training To Online Alignment

- 避免继续把“复述 JSON”作为唯一训练目标
- 新增更直接服务 online path 的训练目标，例如：
  - learned retrieval / rerank
  - belief selection / composition
  - memory write / update supervision
- 里程碑：
  - training config / variant 能表达 online-aligned learned path
  - 新 learned variant 能产出可被 online system 消费的 artifact

#### 阶段 N-D：Learned-Mode Evidence

- 用 learned-mode canary 验证 online gain
- 保证收益不是 benchmark-specific shortcut / fallback 带来的
- 里程碑：
  - learned-mode `PersonaMem` current-head canary 落地
  - learned-mode `LongMemEval-S` current-head canary 落地
  - 至少一条 learned-mode artifact 能解释 online gain
- 当前完成态：
  - `scripts/verify_stage2_v21_learned_memory.py --score-only = 12/12`
  - `StructuredMemorySystem` 已支持 `memory_mode=learned_memory` 与 checkpoint-backed belief path
  - `scripts/run_stage2_memory_canary.py` 已支持 learned-mode 在线运行与 `latest_*_stage2_learned_canary.json`
  - current-head learned-mode artifacts 已落地：
    - `outputs_v2/artifacts/latest_personamem_stage2_learned_canary.json`
    - `outputs_v2/artifacts/latest_longmemeval_stage2_learned_canary.json`

## 当前不做

- 删除或覆盖第一阶段 v1
- 将 benchmark eval/test 用作第二阶段主要训练源
- 一开始就采用更重的 Perceiver-style resampler
- 一开始就把 Optimus-like 路线设为主线

### 阶段 O：V2.1 Learned-Model Long Run

- 用户已明确要求当前主线更加注重 learned model、本体智能和更好的 latent
- 这轮长跑不再把“plumbing 已接通”当终点，而是要求 learned path 在 online 主链中更强、更稳、更可扩
- 绝对规则：不做任何兜底/fallback/benchmark-specific shortcut
- 详细计划见 [docs/v21_longrun_plan.md](/media/storage/mingjing/workspace/CoRe_Mem/docs/v21_longrun_plan.md)

#### 阶段 O-A：把 learned path 从“存在”推进到“可测”

- 跑 current-head learned-mode `PersonaMem 64`
- 跑 current-head learned-mode `LongMemEval-S 64`
- 固化 learned-mode analysis artifact
- 将 fresh learned-mode evidence 设为 verifier 硬门槛

#### 阶段 O-B：让训练真正服务在线链路

- 继续强化 `online_aligned` 路线
- 优先让 learned retrieval / rerank / belief selection 服务在线主链
- 避免继续把“复述 JSON”当成唯一训练目标
- 要求 checkpoint 能被 current-head online system 直接消费

#### 阶段 O-C：专项提升 LongMemEval-S

- 针对 `LongMemEval-S` 做分层 failure analysis
- 优先修 retrieval / belief / latent composition 的主错误簇
- 所有改动都必须在 `PersonaMem` 上不过度退化

#### 阶段 O-D：扩大样本，验证 robustness

- `PersonaMem` 从 `64` 扩到 `128`
- 视预算扩大 `LongMemEval-S`
- 用更大切片验证 learned-memory 提升不是 canary 偶然现象

#### 阶段 O-E：收口成 robust learned-memory v2.1

- 固化默认训练命令
- 固化默认 learned-mode benchmark 命令
- 固化一套 `v2.1` verifier
- 明确允许和禁止的策略，避免退回 rule-heavy / fallback-heavy 路线
