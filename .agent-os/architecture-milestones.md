# Architecture And Milestones

## Workstreams

- `WS-001`: 项目系统与恢复能力
- `WS-002`: 基础工程与环境
- `WS-003`: Stage-1 benchmark adapters
- `WS-004`: Stage-1 vanilla CoRe Memory 方法实现
- `WS-005`: Stage-1 评测、输出与复现
- `WS-006`: Stage-1 测试体系
- `WS-007`: Stage-2 `V2.0 structured latent-slot memory` 设计与实现准备
- `WS-008`: Stage-2 数据、parser 与训练管线
- `WS-009`: Stage-2 本地 intrinsic evaluation 与 canary 协议

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

## Milestones

- `MS-001` `[done]` 项目系统初始化完成
  - Acceptance:
    - `AGENTS.md` 与 `CLAUDE.md` 就位
    - `docs/` 与 `.agent-os/` 基础文档齐全
    - 文档恢复路径可用

- `MS-002` `[doing]` Stage-1 环境与基础工程骨架完成
  - Acceptance:
    - conda 环境 `core_mem` 固定，且不放在 repo 工作区内
    - 依赖管理明确
    - 配置系统与 provider adapter 就位

- `MS-003` `[doing]` Stage-1 PersonaMem 32k 接入并跑通官方协议
  - Acceptance:
    - 数据可获取
    - 主任务可运行
    - 官方评测协议可运行

- `MS-004` `[doing]` Stage-1 LongMemEval-S 接入并跑通官方协议
  - Acceptance:
    - 数据可获取
    - 官方协议可运行

- `MS-005` `[backlog]` Stage-1 vanilla CoRe Memory 主干实现完成
  - Acceptance:
    - writer / core updater / residual manager / reader 可运行

- `MS-006` `[doing]` Stage-1 测试与复现闭环完成
  - Acceptance:
    - unit tests 完整
    - E2E smoke test 可运行
    - 总入口与分步骤脚本可重跑结果

- `MS-007` `[backlog]` Stage-1 正式结果产出
  - Acceptance:
    - PersonaMem 正式结果
    - LongMemEval-S 正式结果
    - 结果表与结果数据齐全

- `MS-008` `[done]` Stage-2 `V2.0` 研究设计锁定
  - Acceptance:
    - 真源 `docs/requirements.md` 已包含第二阶段条款
    - `docs/v2_design.md` 已新增并对齐真源
    - stage-2 相关状态文档已同步

- `MS-009` `[ready]` Stage-2 数据与规范管线落地
  - Acceptance:
    - observation normalization schema 落地
    - belief JSON schema 落地
    - public datasets ingestion skeleton 落地
    - `outputs_v2/` 目录结构落地

- `MS-010` `[backlog]` Stage-2 主线模型骨架实现
  - Acceptance:
    - slot encoder / retrieval key head / lifecycle head / core-residual banks / consolidation / light resampler / Flan-T5 belief decoder 可运行

- `MS-011` `[backlog]` Stage-2 本地 intrinsic evaluation 完成
  - Acceptance:
    - retrieval、belief、update、specificity、compression 指标可运行
    - budget sweep 与核心 ablation 可记录

- `MS-012` `[backlog]` Stage-2 benchmark canary 与后续正式评测路径建立
  - Acceptance:
    - PersonaMem 64 canary 固定
    - LongMemEval-S 64 canary 固定
    - canary 输出、表格与日志可追溯
