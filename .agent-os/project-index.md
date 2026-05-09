# Project Index

## Current Truth

- Objective: `OBJ-002`
- Top next action: `TD-012`
- Active workstreams: `WS-004`
- Active blockers: `none`

## Objective Summary

- `OBJ-002`: Build a bounded cross-session personalized memory framework whose phase-1 instance is vanilla CoRe Memory, with reproducible, traceable runs on PersonaMem and LongMemEval-S.

## Active Workstreams

- `WS-001` `[done]`: Project system bootstrap and recovery path
- `WS-002` `[done]`: Phase-1 method and benchmark prep
- `WS-003` `[backlog]`: Benchmark adapters
- `WS-004` `[doing]`: Vanilla CoRe Memory implementation (write path done; reader pending)
- `WS-005` `[backlog]`: Evaluation, outputs, reproduction
- `WS-006` `[partial]`: Test harness (write-path tests done)

## Top Next Action

- `TD-012` `[doing]`: Writer fact extraction (LLM prompt-based) — writer currently receives text deltas; wire LLM extraction next.

## Active Blockers

- None

## Recent Important Changes

- 2026-04-04: Locked phase-1 goals, benchmarks, scope, hard constraints, acceptance.
- 2026-04-04: Documentation uses coexisting `docs/` and `.agent-os/`.
- 2026-04-06: Environment conda → venv (`CD-005`).
- 2026-04-06: Directory layout, dependency files, module skeletons (`TD-001`/`TD-002`/`TD-003` done).
- 2026-04-06: Embedding model `all-MiniLM-L6-v2` (`TD-004` done).
- 2026-04-06: Write-path modules complete (`TD-009` done): slot, embedding, residual_manager, core_updater, writer. Online centroid merge; residual eviction by lowest merge_count. 28 unit tests pass.

## Read Next

- `docs/requirements.md`
- `docs/write_path_plan.md`
- `.agent-os/architecture-milestones.md`
- `.agent-os/todo.md`
- `.agent-os/run-log.md`
