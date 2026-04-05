# Run Log

## 2026-04-04 Session 001

- Worked on: 项目初始化前的人类决策锁定与文档系统搭建准备
- State changed:
  - 冻结了第一阶段目标、范围、硬约束和验收标准
  - 决定 `docs/requirements.md` 为最高优先级真源文档
  - 决定 `docs/` 与 `.agent-os/` 并存
- Evidence / artifacts:
  - 本次会话中已确认 requirements 草案与 AGENTS 草案
- Next likely action:
  - 在 repo 中创建 `AGENTS.md`、真源文档和依赖状态文档，并交由用户审阅

## 2026-04-04 Session 002

- Worked on: 项目系统初始化与研究平台骨架搭建
- State changed:
  - 创建了 `AGENTS.md` / `CLAUDE.md`
  - 创建了 `docs/` 与 `.agent-os/` 的基础文档体系
  - 建立了 `src/`、`scripts/`、`configs/`、`tests/`、`outputs/` 目录骨架
  - 添加了第一版 `environment.yml`、配置文件、provider adapter skeleton、benchmark adapter skeleton、verifier、runner 和最小测试
- Evidence / artifacts:
  - `docs/requirements.md`
  - `.agent-os/project-index.md`
  - `scripts/verify_stage1_status.py`
  - `scripts/run_experiment.py`
- Next likely action:
  - 将骨架升级为可执行的 benchmark dry-run 平台，并补齐更有意义的验证与测试

## 2026-04-05 Session 003

- Worked on: 平台从空骨架升级到 benchmark dry-run 与真实 provider 调用层
- State changed:
  - 实现了 `src/core_mem/config.py` 配置加载
  - 将 OpenAI-compatible provider 从 payload builder 升级为真实 HTTP 调用 adapter
  - 将 PersonaMem / LongMemEval-S adapter 升级为可读取官方文件格式并构造 prompt-ready records
  - 将 `scripts/run_experiment.py` 升级为支持 `personamem` / `longmemeval` 子命令与 dry-run 输出
  - 扩展 verifier，使 readiness score 从旧骨架分数提升为更贴近第一阶段要求的检查集
  - 增加 benchmark fixtures、provider tests、adapter tests、config test 和 dry-run smoke test
- Evidence / artifacts:
  - `src/core_mem/config.py`
  - `src/core_mem/providers/openai_compatible.py`
  - `src/core_mem/benchmarks/personamem.py`
  - `src/core_mem/benchmarks/longmemeval.py`
  - `tests/test_provider_adapter.py`
  - `tests/test_benchmark_adapters.py`
  - `pytest` 通过（9 tests）
  - `scripts/verify_stage1_status.py --score-only` 输出 `16`
- Next likely action:
  - 创建并固定 conda 环境 `core_mem`
  - 下载并接入 PersonaMem 32k 与 LongMemEval-S 官方数据
  - 推进 official protocol 的正式运行与结果产出

## 2026-04-05 Session 004

- Worked on: 项目内 conda 环境的真实创建与环境复现固化
- State changed:
  - 创建了项目内 prefix conda 环境 `.conda_envs/core_mem`
  - 固化了 `configs/condarc.project.yaml`
  - 添加了 `scripts/bootstrap_conda_env.ps1` 与 `scripts/run_in_core_mem.ps1`
  - 扩展 verifier，使其检查 project-local condarc、bootstrap scripts 和环境内 `python.exe`
  - 修复 `pytest.ini`，避免新环境下递归进入 `.conda_envs` 与 `.conda_pkgs`
- Evidence / artifacts:
  - `.conda_envs/core_mem/python.exe`
  - `scripts/bootstrap_conda_env.ps1`
  - `scripts/run_in_core_mem.ps1`
  - 项目内 conda 环境下 `python --version` -> `3.10.20`
  - 项目内 conda 环境下 `pytest` 通过（9 tests）
  - 项目内 conda 环境下 `scripts/run_experiment.py --verify-only` 输出 `20`
- Next likely action:
  - 下载 PersonaMem 32k 官方数据
  - 开始主 benchmark 的正式 protocol 运行

## 2026-04-05 Session 005

- Worked on: 从 Windows 本地执行迁移到 WSL 的 handoff 清理
- State changed:
  - 删除了 Windows-specific environment bootstrap / condarc 文件
  - 将仓库重新收敛到平台中立状态
  - 将 agent state 文档改为“下一步在 WSL 中重建 conda 环境”
- Evidence / artifacts:
  - 删除 `configs/condarc.project.yaml`
  - 删除 `scripts/bootstrap_conda_env.ps1`
  - 删除 `scripts/run_in_core_mem.ps1`
  - `docs/current_status.md`、`.agent-os/project-index.md`、`.agent-os/todo.md` 已切换到 WSL handoff 状态
- Next likely action:
  - 在 WSL 中 `git clone` 当前分支
  - 创建 Linux 下的 `core_mem` conda 环境
  - 用新环境复验 `verify_stage1_status.py`、`pytest` 和 benchmark dry-run
