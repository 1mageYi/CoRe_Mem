# Change Decisions

## `CD-001` 2026-04-04

- 决定真源文档位于 `docs/requirements.md`，而非 `.agent-os/requirements.md`。
- 决定 `docs/` 与 `.agent-os/` 并存。
- 决定 `.agent-os/todo.md` 为 agent runtime truth，`docs/todo.md` 为人类可读视图。

## `CD-002` 2026-04-04

- 决定第一阶段方法锁定为 `vanilla CoRe Memory`。
- `spectral / multi-timescale extension`、`assignment randomization ablation` 均不属于第一阶段。

## `CD-003` 2026-04-04

- 决定第一阶段 benchmark：
  - `PersonaMem` 为主 benchmark
  - `LongMemEval-S` 为第二优先级 planned benchmark
  - `LoCoMo` 为未来扩展

## `CD-004` 2026-04-04

- 决定第一阶段接受 negative result。
- 第一阶段成功标准为“系统完整、实验可运行、结论可信”，而不是“优于 baseline”。

## `CD-005` 2026-04-05

- 决定 PersonaMem 第一阶段先对齐官方 `32k` multiple-choice 主任务。
- 决定 LongMemEval 第一阶段先对齐官方 `S cleaned` 数据与官方 protocol。
- 决定 `128k / 1M` 只保留启动开关，不要求第一阶段正式结果。

## `CD-006` 2026-04-05

- 决定停止在 Windows 本地继续折腾环境，并迁移到 WSL 作为后续主执行平台。
- 决定仓库只保留平台中立代码、文档与 agent 状态，Windows-specific bootstrap 不进入 handoff commit。
