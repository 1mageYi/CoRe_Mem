# Project Workflow (Coding SOP)

## Purpose
This document defines the fixed workflow for coding in this project.  
Follow it whenever you implement, refactor, add experiments, or change parameters.

## 1) Read these documents first (order)
1. `docs/requirements.md` (goals, constraints, acceptance)  
2. `docs/todo.md` (current tasks and priority)  
3. `docs/current_status.md` (decisions, risks, recent progress)  
4. `docs/structure.md` (overall architecture)  
5. `docs/implementation_plan.md` (implementation strategy and parameters)  

## 2) Development rules
- Pick one smallest shippable task from `todo`; avoid changing too many modules at once.  
- Prefer loose coupling:  
  - Add vs Search separated  
  - Extractor vs Graph Store separated  
  - Rerank vs Prompt Builder separated  
- New logic must be ablatable (toggle / comparable runs).  
- New settings must be parameterized—avoid hardcoding in the main path.  

## 3) Documentation updates after changes (mandatory)
After code changes, update:  
- `docs/current_status.md`  
  - What changed, why, results, next steps  
- `docs/todo.md`  
  - Check off completed items; add new tasks; remove obsolete ones  
- `docs/structure.md` (only if architecture changes)  
  - Module / pipeline changes only  
- `docs/implementation_plan.md` (only if the plan changes)  
  - New rules, parameters, or flow changes  

## 4) Dependency management (mandatory)
When adding or upgrading dependencies, update all three:  
1. `requirements.txt`  
2. `pyproject.toml`  
3. `uv.lock`  

And record in `current_status`:  
- Why the dependency changed  
- Affected modules  
- Whether reinstall / migration is needed  

## 5) Experiments and evaluation
- Primary benchmark: `LoCoMo` (long-context dialogue memory)  
- Baseline / regression: `PERMA`  
- Final migration check: `PersonaMem`  

Each experiment should record at least:  
- Configuration (key parameters)  
- Metrics  
- Comparison to baseline  
- Failure cases and likely causes  

Primary conclusions default to **LoCoMo (long dialogue memory)**; PERMA is a comparative baseline.

## 6) Pre-submit checklist
- [ ] Code runs (minimal path)  
- [ ] Docs updated for new logic  
- [ ] Parameters have defaults and brief notes  
- [ ] Dependency triple-sync if deps changed  
- [ ] `todo` / `current_status` updated  

## 7) Do not
- Do not land changes that “only touch code” without doc updates  
- Do not merge one-off experiment scripts into the main pipeline  
- Do not encode benchmark-specific hacks as generic logic without labeling them  
