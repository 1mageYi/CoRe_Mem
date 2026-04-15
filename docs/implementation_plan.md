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

### 阶段 K：Benchmark canary 与正式评测

- PersonaMem canary
- LongMemEval-S canary
- 选择性推进 full benchmark
- Status: doing
- Notes:
  - canary manifest 生成器已经落地，并已固定产出 PersonaMem 64 / LongMemEval-S 64 manifests
  - `scripts/run_stage2_memory_canary.py` 已将 `StructuredMemorySystem` 接到 benchmark canary prompt 生成链路，并已在 `MiniMax-M2.7` 上完成 1-sample 与 PersonaMem 64 的 live canary
  - 当前真正未完成点已经从“provider key / runner 是否可用”切换为“live canary 质量是否足够支撑 benchmark scaling”
  - 当前 autoresearch 已验证两条路径：
    - observation-noise cleanup 可清理 parser 噪声，但单独不足以抬高 live 分数
    - PersonaMem answer-option 对齐是当前第一段主收益来源，已把 quality score 从 `4/10` 提升到 `8/10`
    - 生命周期层面的 facet retention + selected-slot-aware option scoring 是当前跨过 stop condition 的关键，已把 quality score 从 `8/10` 进一步提升到 `9/10`
  - 当前 canary 目标已满足；但后续若继续推进，不应再把 benchmark-facing option heuristics 当作主优化方向
  - 下一阶段的锚点应切换为：以 local intrinsic 指标提升 `latent core / model / system robustness`，同时把当前 `PersonaMem 9/10` 作为不退化 guard

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
4. 在 latent path 已成立且 live canary 已跑通的前提下，把主优化目标切换为 `latent core / local intrinsic quality`
5. 以 `PersonaMem 64 >= 9/10` 为不退化 guard，优先提升 `encoder / retrieval / lifecycle / belief decode / composition` 本体质量
6. 在本体质量明显改善后，再视用户要求补默认 `flan-t5-base` 非 tiny `gpu3` 训练证据，并扩大 benchmark

## 当前不做

- 删除或覆盖第一阶段 v1
- 将 benchmark eval/test 用作第二阶段主要训练源
- 一开始就采用更重的 Perceiver-style resampler
- 一开始就把 Optimus-like 路线设为主线
