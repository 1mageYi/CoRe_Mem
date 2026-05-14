# Architecture And Milestones

## Workstreams

- `WS-001`: Project system and recovery
- `WS-002`: Base engineering and environment
- `WS-003`: Benchmark adapters
- `WS-004`: Vanilla CoRe Memory implementation
- `WS-005`: Evaluation, outputs, reproduction
- `WS-006`: Testing

## Current Architecture Route

Phase 1 uses a training-light stack:

- `all-MiniLM-L6-v2` as the embedding layer (384-d)
- Alibaba Cloud OpenAI-compatible API as the default LLM client
- Vanilla CoRe Memory as the main method
  - Merge: online centroid, recency_weight=1.5
  - Core: 32 slots; Residual: 64 slots
  - Residual eviction: lowest merge_count first
  - Core eviction: forced merge into nearest core slot by cos_sim
  - Persistence: safetensors + JSON
- Benchmark adapters follow official protocols
- Results and metadata land under `outputs/`

## Milestones

- `MS-001` `[done]` Project system bootstrap
  - Acceptance:
    - `AGENTS.md` and `CLAUDE.md` present
    - `docs/` and `.agent-os/` baseline docs present
    - Documentation recovery path works

- `MS-002` `[done]` Environment and engineering skeleton
  - Acceptance:
    - venv pinned to Python 3.10 (`CD-005`)
    - Dependencies declared (`pyproject.toml` + `requirements.txt`)
    - Config system (`configs/default.toml` + `config.py`)

- `MS-003` `[backlog]` PersonaMem 32k integrated with official protocol
  - Acceptance:
    - Data obtainable
    - Main task runs
    - Official eval protocol runs

- `MS-004` `[backlog]` LongMemEval-S integrated with official protocol
  - Acceptance:
    - Data obtainable
    - Official protocol runs

- `MS-005` `[partial]` Vanilla CoRe Memory core complete
  - Write path done: writer / core updater / residual manager runnable
  - Read path pending: reader not implemented
  - Acceptance:
    - writer / core updater / residual manager / reader runnable

- `MS-006` `[partial]` Testing and reproduction loop
  - Write-path tests done: 28 unit tests pass
  - Acceptance:
    - Unit tests complete
    - E2E smoke test runs
    - Central entrypoint and staged scripts reproduce results

- `MS-007` `[backlog]` Phase-1 formal results
  - Acceptance:
    - PersonaMem formal results
    - LongMemEval-S formal results
    - Tables and prediction outputs complete
