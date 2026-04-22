# Architecture And Milestones

## Workstreams

- `WS-001`: 项目系统与恢复能力
- `WS-002`: 基础工程与环境
- `WS-003`: Stage-1 benchmark adapters
- `WS-004`: Stage-1 vanilla CoRe Memory 方法实现
- `WS-005`: Stage-1 评测、输出与复现
- `WS-006`: Stage-1 测试体系
- `WS-007`: Stage-2 `V2.0 structured latent-slot memory` 主线模型实现
- `WS-008`: Stage-2 数据、parser 与训练管线
- `WS-009`: Stage-2 本地 intrinsic evaluation 与 canary 协议
- `WS-032`: Stage-2 `v5 Core-Residual Latent Substrate` 顶会级 latent memory 路线

## Current Architecture Route

### Stage-1 路线

第一阶段采用 training-light 路线：

- pretrained embedding model 作为基础表示层
- 阿里云兼容 OpenAI API 作为主要 LLM 调用
- 在用户明确批准时，允许通过 OpenAI-compatible 配置临时切换到替代 provider（当前已验证 Google AI Studio Gemini 2.5 Flash）
- vanilla CoRe Memory 作为主方法
- `environment.yaml` 作为唯一环境复现规格，`core_mem` 使用默认 conda named env
- benchmark adapters 直接对齐官方协议
- 对长跑 benchmark，允许使用外层 supervisor 通过 `--resume` 做 crash 后冷却重启
- 结果与元数据统一沉淀到 `outputs/`

### Stage-2 路线

第二阶段采用 `V2.0 structured latent-slot memory` 路线：

- memory 基本形式为 `structured latent slots`
- `core + residual` 双银行 memory
- 新 observation 先经 residual 吸收，再由 consolidation 决定是否强化或替换 core
- answer-time 主链路为 `latent composition -> belief decoding -> answer projection`
- 主 resampler 为 `Light Cross-Attention Resampler`
- 主 decoder 为 `Flan-T5 belief JSON decoder`
- 训练方式为 LoRA/adapter + resampler/projection 可训练
- 训练数据策略为 `public-datasets-first, synthetic-minimal`
- benchmark 保持 evaluation source 地位，不作为 primary training source
- 第二阶段输出统一沉淀到 `outputs_v2/`

### Stage-2 V5 路线

`v5` 将第二阶段从 text-centered latent-slot system 推进到 core-residual latent substrate：

- pretrained encoder backbone 首批比较 `BGE / E5 / Contriever`
- learned write controller 决定 `new / merge / overwrite / stale / promote_to_core / keep_residual`
- stable persona traits 写入 core latent memory
- episodic / recent / conflict / update-sensitive facts 写入 residual latent slots
- query-conditioned latent reader 直接读取 core + residual latent state
- belief decoder 作为 interpretable bottleneck，而不是唯一 memory substrate
- PersonaMem gold 只允许在 persona/context 隔离 split 上校准薄 answer head
- 必须报告 latent-only、text-only、shuffled-latent、core-only、residual-only、no-controller 等 anti-shortcut ablations

## Milestones

- `MS-001` `[done]` 项目系统初始化完成
  - Acceptance:
    - `AGENTS.md` 与 `CLAUDE.md` 就位
    - `docs/` 与 `.agent-os/` 基础文档齐全
    - 文档恢复路径可用

- `MS-002` `[done]` Stage-1 环境与基础工程骨架完成
  - Acceptance:
    - conda 环境 `core_mem` 固定，且不放在 repo 工作区内
    - 依赖管理明确
    - 配置系统与 provider adapter 就位

- `MS-003` `[done]` Stage-1 PersonaMem 32k 接入并跑通官方协议
  - Acceptance:
    - 数据可获取
    - 主任务可运行
    - 官方评测协议可运行

- `MS-004` `[done]` Stage-1 LongMemEval-S 接入并跑通官方协议
  - Acceptance:
    - 数据可获取
    - 官方协议可运行

- `MS-005` `[done]` Stage-1 vanilla CoRe Memory 主干实现完成
  - Acceptance:
    - writer / core updater / residual manager / reader 可运行

- `MS-006` `[done]` Stage-1 测试与复现闭环完成
  - Acceptance:
    - unit tests 完整
    - E2E smoke test 可运行
    - 总入口与分步骤脚本可重跑结果

- `MS-007` `[blocked]` Stage-1 正式结果产出
  - Acceptance:
    - PersonaMem 正式结果
    - LongMemEval-S 正式结果
    - 结果表与结果数据齐全
  - Notes:
    - 当前作为 baseline / acceptance 相关的 pending 项保留
    - 只有在用户明确要求 AI 去跑时才继续执行
    - 同时仍受 provider `HTTP 429` blocker 影响

- `MS-008` `[done]` Stage-2 `V2.0` 研究设计锁定
  - Acceptance:
    - 真源 `docs/requirements.md` 已包含第二阶段条款
    - `docs/v2_design.md` 已新增并对齐真源
    - stage-2 相关状态文档已同步

- `MS-009` `[done]` Stage-2 数据与规范管线落地
  - Acceptance:
    - observation normalization schema 落地
    - belief JSON schema 落地
    - public datasets ingestion skeleton 落地
    - `outputs_v2/` 目录结构落地
  - Notes:
    - `src/core_mem/v2/` 已落地 Observation / Slot / Belief schema、rule-first parser 与 dataset registry
    - `scripts/prepare_stage2_data.py` 已产出 prepared manifest 和 task registry
    - `scripts/train_stage2.py`、`scripts/eval_stage2_local.py`、`scripts/run_stage2_canary.py` 已形成首批 stage2 pipeline
    - `outputs_v2/` 已产出 smoke train plan、local eval 结果与 canary manifests
    - `prepare_stage2_data.py` 已支持 source-config 驱动的非 demo prepared-manifest 构建
    - `scripts/normalize_stage2_public_data.py` 已将真实 `SGD / MultiWOZ 2.4 / Persona-Chat / MQUAKE / ReCoE` raw sources 规范化为 `normalized.jsonl`
    - `scripts/stage2_data_preflight.py --json` 当前已返回 `missing=[]`

- `MS-010` `[done]` Stage-2 主线模型骨架实现
  - Acceptance:
    - slot encoder / retrieval key head / lifecycle head / core-residual banks / consolidation / light resampler / Flan-T5 belief decoder 可运行
  - Notes:
    - `encoder.py`、`lifecycle.py`、`consolidation.py`、`resampler.py`、`decoder.py`、`projection.py`、`system.py` 已落地第一版 deterministic skeleton
    - 当前已能从 parser 输出一路走到 `belief decode -> answer projection` smoke path
    - `training.py` 与 `scripts/train_stage2.py --execute-train` 已新增真实训练执行路径，并有 tiny offline smoke 证据
    - 真实公开数据处理链已接上：`prepare_stage2_data.py --strict-sources` 当前可基于真实 public-data slice 生成 non-demo manifest，并在 `outputs_v2/runs/20260414T043225Z_stage2_train_exec` 上跑通 execute-train
    - `configs/stage2_train.yaml` 当前通过 repo-local Hugging Face cache root 避免首次权重下载写到 repo 外目录
    - 下一步切换到 fuller local eval / budget sweep，而不是继续把“缺少真实公开数据”当作主 blocker

- `MS-011` `[done]` Stage-2 本地 intrinsic evaluation 完成
  - Acceptance:
    - retrieval、belief、update、specificity、compression 指标可运行
    - budget sweep 与核心 ablation 可记录
  - Notes:
    - `scripts/eval_stage2_local.py` 当前已支持 `--top-k`、`--budget`、`--dataset`
    - `src/core_mem/v2/eval_local.py` 已覆盖 parser、slot、retrieval、belief、update、locality、compression 评测
    - 评测输出当前固定沉淀为 JSON + summary CSV + budget CSV
    - `docs/stage2_local_evaluation.md` 已完成接口与指标文档

- `MS-012` `[backlog]` Stage-2 benchmark canary 与后续正式评测路径建立
  - Acceptance:
    - PersonaMem 64 canary 固定
    - LongMemEval-S 64 canary 固定
    - canary 输出、表格与日志可追溯
  - Notes:
    - canary manifest 生成器与固定 `64` 子集已落地；后续只需在主线模型版本稳定后补齐运行记录与结果表

- `MS-013` `[planned]` Stage-2 `v5 Core-Residual Latent Substrate` 计划与执行契约
  - Acceptance:
    - `docs/v5_plan.md` 固定目标架构、数据隔离、训练目标、反 shortcut 评估协议
    - `.agent-os` index / todo / decisions / acceptance / run-log 同步 `TD-046 / WS-032`
    - 后续实现必须先建立 pretrained encoder comparison、PersonaMem gold-isolation checker、latent-only / shuffled-latent ablation scaffold
  - Boundary:
    - 当前只是 plan/contract milestone，不代表已有新的训练或 benchmark gain
    - `v5` 不允许把规则修补、provider prompt trick、PersonaMem option geometry 当作主贡献
