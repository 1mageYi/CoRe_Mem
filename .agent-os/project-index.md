# Project Index

## Current Truth

- Objective: `OBJ-002`
- Top next action: `TD-002`
- Active workstreams: `WS-002`, `WS-003`, `WS-005`, `WS-006`
- Active blockers: `none`

## Objective Summary

- `OBJ-002`: 实现一个 bounded cross-session personalized memory framework，其第一阶段具体实例为 vanilla CoRe Memory，并在 PersonaMem 与 LongMemEval-S 上完成可复现、可追溯的实验运行。

## Active Workstreams

- `WS-002` `[doing]`: WSL 环境重建与平台迁移
- `WS-003` `[doing]`: Benchmark adapters 与 dry-run 接入
- `WS-005` `[doing]`: 实验入口、输出与运行元数据
- `WS-006` `[doing]`: 初始测试体系

## Top Next Action

- `TD-002` `[doing]`: 在 WSL 中创建新的 `core_mem` conda 环境，并验证 verifier / pytest / benchmark dry-run。

## Active Blockers

- None

## Recent Important Changes

- 2026-04-04: 锁定第一阶段目标、benchmark、范围、硬约束和验收标准。
- 2026-04-04: 确认文档系统采用 `docs/` 与 `.agent-os/` 并存模式。
- 2026-04-05: 完成 stage-1 平台从骨架到可执行 dry-run 的升级，支持官方 PersonaMem / LongMemEval-S 文件格式读取。
- 2026-04-05: 增加 OpenAI-compatible provider 实际调用逻辑、配置加载、结构化输出目录与更完整的测试集。
- 2026-04-05: 决定放弃 Windows 本地环境，迁移到 WSL；仓库将只保留平台中立代码、文档和 agent 状态。

## Read Next

- `docs/requirements.md`
- `.agent-os/architecture-milestones.md`
- `.agent-os/todo.md`
- `.agent-os/run-log.md`
