# Project Index

## Current Truth

- Objective: `OBJ-002`
- Top next action: `TD-012`
- Active workstreams: `WS-004`
- Active blockers: `none`

## Objective Summary

- `OBJ-002`: 实现一个 bounded cross-session personalized memory framework，其第一阶段具体实例为 vanilla CoRe Memory，并在 PersonaMem 与 LongMemEval-S 上完成可复现、可追溯的实验运行。

## Active Workstreams

- `WS-001` `[done]`: 项目系统初始化与恢复路径建设
- `WS-002` `[done]`: 第一阶段方法与 benchmark 实现准备
- `WS-003` `[backlog]`: Benchmark adapters
- `WS-004` `[doing]`: Vanilla CoRe Memory 方法实现（write path done，reader 待做）
- `WS-005` `[backlog]`: 评测、输出与复现
- `WS-006` `[partial]`: 测试体系（write path tests done）

## Top Next Action

- `TD-012` `[doing]`: Writer fact 提取逻辑（LLM prompt-based extraction）——当前 writer 接收已有 text delta，后续需接入 LLM 提取。

## Active Blockers

- None

## Recent Important Changes

- 2026-04-04: 锁定第一阶段目标、benchmark、范围、硬约束和验收标准。
- 2026-04-04: 确认文档系统采用 `docs/` 与 `.agent-os/` 并存模式。
- 2026-04-06: 环境管理从 conda 切换为 venv（`CD-005`）。
- 2026-04-06: 完成目录结构、依赖文件、模块骨架创建（`TD-001`/`TD-002`/`TD-003` done）。
- 2026-04-06: 确定 embedding model 为 `all-MiniLM-L6-v2`（`TD-004` done）。
- 2026-04-06: 实现 write path 全部核心模块（`TD-009` done）：slot、embedding、residual_manager、core_updater、writer。合并机制为在线质心，驱逐策略为 merge_count 最低优先。28 个 unit tests 全部通过。

## Read Next

- `docs/requirements.md`
- `docs/write_path_plan.md`
- `.agent-os/architecture-milestones.md`
- `.agent-os/todo.md`
- `.agent-os/run-log.md`
