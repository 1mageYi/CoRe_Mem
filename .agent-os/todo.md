# TODO

## Doing

- `TD-012` `[doing]` Writer fact extraction logic (LLM prompt-based extraction) — writer currently receives text deltas; wire LLM extraction next.

## Ready

- `TD-005` `[ready]` Implement Alibaba Cloud OpenAI-compatible provider adapter.
- `TD-006` `[ready]` Integrate PersonaMem 32k official main task and evaluation protocol.

## Backlog

- `TD-007` `[backlog]` Add launch switches for PersonaMem 128k / 1M.
- `TD-008` `[backlog]` Integrate LongMemEval-S official protocol.
- `TD-010` `[backlog]` Central experiment entrypoint and staged scripts.
- `TD-013` `[backlog]` Implement Reader: query → top-k core + top-j residual retrieval.
- `TD-014` `[backlog]` Train vec2text model to decode latent information.

## Blocked

- None

## Done

- `TD-001` `[done]` Initialize project documentation system with root contract, source-of-truth docs, and status docs.
  - Evidence: `AGENTS.md`, `CLAUDE.md`, `docs/`, `.agent-os/` in place; recovery path works.
- `TD-002` `[done]` Environment setup (venv instead of conda, `CD-005`), Python 3.10.
  - Evidence: `pyproject.toml` pins `requires-python = "==3.10.*"`; `requirements.txt` present.
- `TD-003` `[done]` Base directory layout, configuration, and output conventions.
  - Evidence: `src/`, `scripts/`, `configs/`, `outputs/`, `tests/` created; `configs/default.toml`, `.gitignore`, `.env.example` present.
- `TD-004` `[done]` Survey and pick phase-1 pretrained embedding model.
  - Evidence: `sentence-transformers/all-MiniLM-L6-v2` (384-d), lightweight, widely validated, vec2text-compatible.
- `TD-009` `[done]` Implement vanilla CoRe Memory core (write path).
  - Evidence: `slot.py`, `embedding.py`, `residual_manager.py`, `core_updater.py`, `writer.py`.
  - Merge: online centroid, recency_weight=1.5.
  - Eviction: residual evicts lowest merge_count; core forced merge into nearest neighbor.
  - 28 unit tests pass.
- `TD-011` `[done]` Unit tests and minimal E2E smoke test.
  - Evidence: `test_slot.py`(13), `test_residual_manager.py`(7), `test_core_updater.py`(5), `test_writer.py`(4), `test_smoke.py`(1) — 28 tests pass.

## Verified

- None

## Abandoned

- None
