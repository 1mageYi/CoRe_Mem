# TODO

## Doing

- `TD-012` `[doing]` Writer fact 提取逻辑（LLM prompt-based extraction）——当前 writer 接收已有 text delta，后续需接入 LLM 提取。

## Ready

- `TD-005` `[ready]` 实现阿里云 OpenAI-compatible provider adapter。
- `TD-006` `[ready]` 接入 PersonaMem 32k 官方主任务与评测协议。

## Backlog

- `TD-007` `[backlog]` 为 PersonaMem 128k / 1M 提供启动开关。
- `TD-008` `[backlog]` 接入 LongMemEval-S 官方协议。
- `TD-010` `[backlog]` 建立实验总入口与分步骤脚本。
- `TD-013` `[backlog]` 实现 Reader：query → top-k core + top-j residual 检索。
- `TD-014` `[backlog]` 训练 vec2text 模型解码 latent 信息。

## Blocked

- None

## Done

- `TD-001` `[done]` 初始化项目文档系统并建立根契约、真源文档和状态文档。
  - Evidence: `AGENTS.md`、`CLAUDE.md`、`docs/`、`.agent-os/` 全部就位，恢复路径可用。
- `TD-002` `[done]` 创建环境管理方案（venv 替代 conda，`CD-005`），Python 3.10。
  - Evidence: `pyproject.toml` 锁定 `requires-python = "==3.10.*"`，`requirements.txt` 就位。
- `TD-003` `[done]` 建立基础目录结构、配置机制与输出目录规范。
  - Evidence: `src/`、`scripts/`、`configs/`、`outputs/`、`tests/` 已创建；`configs/default.toml`、`.gitignore`、`.env.example` 就位。
- `TD-004` `[done]` 调研并确定第一阶段使用的 pretrained embedding model。
  - Evidence: 确定使用 `sentence-transformers/all-MiniLM-L6-v2`（384 维），理由：轻量、广泛验证、与 vec2text 兼容。
- `TD-009` `[done]` 实现 vanilla CoRe Memory 核心模块（write path）。
  - Evidence: `slot.py`、`embedding.py`、`residual_manager.py`、`core_updater.py`、`writer.py` 已实现。
  - 合并机制：在线质心（Online Centroid），recency_weight=1.5。
  - 驱逐策略：residual 按 merge_count 最低驱逐；core 强制合并进最近邻。
  - 28 个 unit tests 全部通过。
- `TD-011` `[done]` 建立 unit tests 与最小 E2E smoke test。
  - Evidence: `test_slot.py`(13)、`test_residual_manager.py`(7)、`test_core_updater.py`(5)、`test_writer.py`(4)、`test_smoke.py`(1)，共 28 个测试全部通过。

## Verified

- None

## Abandoned

- None
