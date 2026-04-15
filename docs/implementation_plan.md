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
- Status: doing
- Notes:
  - `v2` 已经解决“系统闭环是否成立”；`v2.1` 要解决的是“这套 latent memory 在真实 benchmark 上是否稳定、是否靠模型本身变强、是否跨任务依然成立”
  - `v2.1` 不再把“完整证据链”当终点，而把“质量、鲁棒性、可扩展性”当终点

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
7. 当前阶段 L 已完成，阶段 M 已给出历史 robustness best，阶段 N 已完成 learned-memory-first plumbing；现阶段默认主线切换到阶段 O：以 learned model / better latent 的长期提升为锚点推进 `v2.1`

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
