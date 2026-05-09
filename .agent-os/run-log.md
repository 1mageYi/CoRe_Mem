# Run Log

## 2026-04-04 Session 001

- Worked on: Human decisions and documentation scaffolding before project bootstrap
- State changed:
  - Froze phase-1 goals, scope, hard constraints, and acceptance criteria
  - Chose `docs/requirements.md` as highest-priority source of truth
  - Chose `docs/` and `.agent-os/` to coexist
- Evidence / artifacts:
  - Confirmed requirements draft and AGENTS draft in this session
- Next likely action:
  - Add `AGENTS.md`, source-of-truth docs, and dependency status in-repo for review

## 2026-04-06 Session 002

- Worked on: Write-path core module design and implementation
- State changed:
  - Embedding model: `sentence-transformers/all-MiniLM-L6-v2` (384 dim)
  - Merge: online centroid, recency_weight=1.5
  - Residual eviction: lowest merge_count first
  - Core eviction: forced merge into nearest core slot by cos_sim
  - Persistence: safetensors + JSON
  - No text shadow (vec2text later)
  - Fixed pyproject.toml build-backend (`setuptools.backends._legacy` → `setuptools.build_meta`)
  - Fixed venv (Python 3.11 → 3.10)
  - Implemented write path: slot.py, embedding.py, config.py, residual_manager.py, core_updater.py, writer.py
  - 28 unit tests pass
- Evidence / artifacts:
  - `docs/write_path_plan.md` — full design
  - `tests/` — 28 tests passed
  - `configs/default.toml` — updated parameters
- TODO completed: TD-004, TD-009, TD-011
- Next likely action:
  - Writer fact extraction (TD-012)
  - Reader (TD-013)
  - PersonaMem benchmark adapter (TD-006)
