# Acceptance Report

## Stage 1 Acceptance Tracking

- `EV-001` -> `AC-001` vanilla CoRe Memory 完整实现
  - Status: pending
  - Evidence: none yet

- `EV-002` -> `AC-002` PersonaMem 官方协议跑通并产出结果
  - Status: pending
  - Evidence: none yet

- `EV-003` -> `AC-003` LongMemEval-S 官方协议跑通并产出结果
  - Status: pending
  - Evidence: none yet

- `EV-004` -> `AC-004` 结果数据、表格、prediction outputs 与实验元数据
  - Status: partial
  - Evidence:
    - `scripts/run_experiment.py` 在 dry-run 下会写出 `outputs/runs/.../run_metadata.json`
    - `run_metadata.json` 已包含 benchmark、record_count、model、provider、sample_ids、prompt_preview

- `EV-005` -> `AC-005` 工程与复现闭环
  - Status: partial
  - Evidence:
    - `configs/defaults.yaml` 与 `src/core_mem/config.py` 已建立统一配置加载路径
    - `scripts/run_experiment.py` 支持 benchmark 子命令和 dry-run 复现入口
    - `outputs/` 路径已被实际写入

- `EV-006` -> `AC-006` 单元测试与 E2E smoke test
  - Status: partial
  - Evidence:
    - 当前 `pytest` 通过，共 9 个测试
    - 已包含 provider、config、benchmark adapter、verifier、E2E dry-run smoke test

- `EV-007` -> `AC-007` 结论可信、结果可追溯、允许 negative result
  - Status: partial
  - Evidence:
    - dry-run 输出已保存结构化 run metadata
    - benchmark prompt preview 与 sample ids 已可追溯
