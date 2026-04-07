# Implementation Plan

## 总体策略

当前项目按两个阶段推进：

1. 第一阶段：完成 `vanilla CoRe Memory` 平台、benchmark、复现与结果闭环。
2. 第二阶段：在保留第一阶段 v1 作为 naive baseline 参考的前提下，实现并验证 `V2.0 structured latent-slot memory`。

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
- Status: blocked by provider

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
- Status: ready

### 阶段 I：V2.0 主线实现

- slot encoder / projection
- retrieval key head
- lifecycle head
- core / residual banks
- consolidation
- light cross-attention resampler
- Flan-T5 belief decoder
- answer projection
- Status: backlog

### 阶段 J：本地 intrinsic evaluation

- retrieval metrics
- belief recovery metrics
- update / propagation metrics
- specificity / locality metrics
- compression metrics
- budget sweep
- Status: backlog

### 阶段 K：Benchmark canary 与正式评测

- PersonaMem canary
- LongMemEval-S canary
- 选择性推进 full benchmark
- Status: backlog

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

1. 保持第一阶段正式 benchmark 阻塞真相不变
2. 推进第二阶段 observation / belief / data pipeline 实现
3. 再进入 V2.0 主线模型与本地 intrinsic evaluation

## 当前不做

- 删除或覆盖第一阶段 v1
- 将 benchmark eval/test 用作第二阶段主要训练源
- 一开始就采用更重的 Perceiver-style resampler
- 一开始就把 Optimus-like 路线设为主线
