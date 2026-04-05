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
