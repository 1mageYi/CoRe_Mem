# Change Decisions

## `CD-001` 2026-04-04

- Source of truth lives in `docs/requirements.md`, not `.agent-os/requirements.md`.
- `docs/` and `.agent-os/` coexist.
- `.agent-os/todo.md` is agent runtime truth; `docs/todo.md` is the human-readable view.

## `CD-002` 2026-04-04

- Phase-1 method is locked to `vanilla CoRe Memory`.
- `spectral / multi-timescale extension` and `assignment randomization ablation` are out of scope for phase 1.

## `CD-003` 2026-04-04

- Phase-1 benchmarks:
  - `PersonaMem` primary
  - `LongMemEval-S` second priority (planned)
  - `LoCoMo` future extension

## `CD-004` 2026-04-04

- Phase 1 accepts negative results.
- Success means “system complete, experiments runnable, conclusions credible”—not necessarily beating baseline.
