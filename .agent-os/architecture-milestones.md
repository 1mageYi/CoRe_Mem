# Architecture And Milestones

## Workstreams

- `WS-001`: 项目系统与恢复能力
- `WS-002`: 基础工程与环境
- `WS-003`: Benchmark adapters
- `WS-004`: Vanilla CoRe Memory 方法实现
- `WS-005`: 评测、输出与复现
- `WS-006`: 测试体系

## Current Architecture Route

第一阶段采用 training-light 路线：

- pretrained embedding model 作为基础表示层
- 阿里云兼容 OpenAI API 作为主要 LLM 调用
- vanilla CoRe Memory 作为主方法
- benchmark adapters 直接对齐官方协议
- 结果与元数据统一沉淀到 `outputs/`

## Milestones

- `MS-001` `[done]` 项目系统初始化完成
  - Acceptance:
    - `AGENTS.md` 与 `CLAUDE.md` 就位
    - `docs/` 与 `.agent-os/` 基础文档齐全
    - 文档恢复路径可用

- `MS-002` `[backlog]` 环境与基础工程骨架完成
  - Acceptance:
    - conda 环境 `core_mem` 固定
    - 依赖管理明确
    - 配置系统与 provider adapter 就位

- `MS-003` `[backlog]` PersonaMem 32k 接入并跑通官方协议
  - Acceptance:
    - 数据可获取
    - 主任务可运行
    - 官方评测协议可运行

- `MS-004` `[backlog]` LongMemEval-S 接入并跑通官方协议
  - Acceptance:
    - 数据可获取
    - 官方协议可运行

- `MS-005` `[backlog]` Vanilla CoRe Memory 主干实现完成
  - Acceptance:
    - writer / core updater / residual manager / reader 可运行

- `MS-006` `[backlog]` 测试与复现闭环完成
  - Acceptance:
    - unit tests 完整
    - E2E smoke test 可运行
    - 总入口与分步骤脚本可重跑结果

- `MS-007` `[backlog]` 第一阶段正式结果产出
  - Acceptance:
    - PersonaMem 正式结果
    - LongMemEval-S 正式结果
    - 结果表与结果数据齐全
