# TODO

## Doing

- `TD-004` `[doing]` 调研并确定第一阶段使用的 pretrained embedding model。

## Ready

## Backlog

- `TD-005` `[backlog]` 实现阿里云 OpenAI-compatible provider adapter。
- `TD-006` `[backlog]` 接入 PersonaMem 32k 官方主任务与评测协议。
- `TD-007` `[backlog]` 为 PersonaMem 128k / 1M 提供启动开关。
- `TD-008` `[backlog]` 接入 LongMemEval-S 官方协议。
- `TD-009` `[backlog]` 实现 vanilla CoRe Memory 核心模块。
- `TD-010` `[backlog]` 建立实验总入口与分步骤脚本。
- `TD-011` `[backlog]` 建立 unit tests 与最小 E2E smoke test。

## Blocked

- None

## Done

- `TD-001` `[done]` 初始化项目文档系统并建立根契约、真源文档和状态文档。
  - Evidence: `AGENTS.md`、`CLAUDE.md`、`docs/`、`.agent-os/` 全部就位，恢复路径可用。
- `TD-002` `[done]` 创建环境管理方案（venv 替代 conda，`CD-005`），Python 3.10。
  - Evidence: `pyproject.toml` 锁定 `requires-python = "==3.10.*"`，`requirements.txt` 就位。
- `TD-003` `[done]` 建立基础目录结构、配置机制与输出目录规范。
  - Evidence: `src/`、`scripts/`、`configs/`、`outputs/`、`tests/` 已创建；`configs/default.toml`、`.gitignore`、`.env.example` 就位。

## Verified

- None

## Abandoned

- None
