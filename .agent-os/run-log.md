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

## 2026-04-04 Session 006

- Worked on: 将环境复现策略从 repo-local conda prefix 回退到 named env + 单一环境文件
- State changed:
  - 用户明确要求 `core_mem` 使用默认 conda envs 目录，不再允许 repo 工作区内保留 `.conda_envs/` 或 `.conda_pkgs/`
  - 环境复现真相改为 `environment.yaml`
  - 已删除 repo-local conda 产物、WSL bootstrap / runner 脚本与相关 verifier 检查
  - 仓库内 `pytest` 与 `scripts/run_experiment.py --verify-only` 在新策略下继续通过
- Evidence / artifacts:
  - `docs/requirements.md`、`docs/current_status.md`、`docs/todo.md`、`docs/implementation_plan.md`
  - `.agent-os/project-index.md`、`.agent-os/todo.md`、`.agent-os/architecture-milestones.md`
  - `pytest` 通过（9 tests）
  - `scripts/verify_stage1_status.py --score-only` 输出 `17`
- Next likely action:
  - 在默认 conda envs 目录重新创建命名环境 `core_mem`
  - 在新约束下重新启动 autoresearch
  - 补齐最小分步骤脚本并继续推进 official benchmark readiness

## 2026-04-04 Session 007

- Worked on: 恢复前台 autoresearch，继续推进复现闭环并机械识别 stage-1 外部 blocker
- State changed:
  - fresh-start autoresearch 成功恢复并延续
  - 新增 `scripts/stage1_preflight.py`
  - 新增 preflight smoke test，使 `pytest` 总数提升到 12
  - `scripts/verify_stage1_status.py --score-only` 从 `20` 提升到 `22`
  - 当前 run 被机械 blocker 停住：default named env、`ALIYUN_API_KEY`、PersonaMem / LongMemEval-S 官方数据缺失
- Evidence / artifacts:
  - `scripts/stage1_preflight.py --config configs/defaults.yaml` 返回 blocker 列表
  - `pytest` 通过（12 tests）
  - `scripts/run_experiment.py --verify-only` 输出 `22`
  - `research-results.tsv` / `autoresearch-state.json` 已记录 keep 与 blocked 行
- Next likely action:
  - 在默认 conda envs 目录创建 `core_mem`
  - 配置 `ALIYUN_API_KEY`
  - 放入 PersonaMem 32k 与 LongMemEval-S 官方数据后恢复正式 benchmark 推进

## 2026-04-04 Session 008

- Worked on: 持续执行前台 autoresearch，并将两个 benchmark 推进到真实 1-sample 运行
- State changed:
  - 用户提供 session 内 `ALIYUN_API_KEY` 后，preflight blocker 从 4 项缩减到 named env 1 项
  - 下载 PersonaMem 32k 与 LongMemEval-S 官方数据到 `data/`
  - 新增最小 vanilla CoRe Memory workflow 实现与测试
  - 补齐 run metadata traceability 字段与 config snapshot
  - 修复 PersonaMem 官方 `shared_contexts_32k.jsonl` 单键映射格式
  - 对齐 PersonaMem multiple-choice label-space 输出协议
  - 对齐 LongMemEval answer-phrase 输出协议
  - `scripts/verify_stage1_status.py --score-only` 提升到 `26`
  - `pytest` 提升到 `18` 个测试并全部通过
  - PersonaMem / LongMemEval-S 均完成真实 1-sample run，最小 `evaluation.json` 均为 `accuracy=1.0`
- Evidence / artifacts:
  - `data/personamem/questions_32k.csv`
  - `data/personamem/shared_contexts_32k.jsonl`
  - `data/longmemeval/longmemeval_s_cleaned.json`
  - `outputs/runs/20260405T024515Z_personamem/evaluation.json`
  - `outputs/runs/20260405T024833Z_longmemeval/evaluation.json`
  - `pytest` 通过（18 tests）
  - `scripts/run_experiment.py --verify-only` 输出 `26`
- Next likely action:
  - 解决 default named env `core_mem` blocker
  - 将两个 benchmark 从 1-sample real run 扩展到正式全量 protocol 运行

## 2026-04-05 Session 009

- Worked on: 继续执行前台 autoresearch，将 readiness/acceptance 机械化并为正式 benchmark 补齐可续跑执行路径
- State changed:
  - 新增 `scripts/verify_stage1_acceptance.py`，将 AC-001~AC-007 机械化
  - `scripts/verify_stage1_status.py --score-only` 从 `26` 提升到 `31`
  - `scripts/run_experiment.py` 新增 `--run-dir` / `--resume` 与增量 prediction 落盘
  - 新增 resumable runner test 与 acceptance verifier test，`pytest` 提升到 `21` 个测试并继续通过
  - 尝试正式 LongMemEval-S 全量运行时，provider 直接返回 `HTTP 403 AllocationQuota.FreeTierOnly`
- Evidence / artifacts:
  - `scripts/verify_stage1_acceptance.py --skip-pytest` 返回 `3/7`
  - `scripts/verify_stage1_status.py --score-only` 输出 `31`
  - `scripts/run_experiment.py --verify-only` 输出 `31`
  - `pytest` 通过（21 tests）
  - `research-results.tsv` 已记录 iteration `7` keep、`8` refine、`9` keep、`10` crash
- Next likely action:
  - 恢复阿里云 provider 可用额度或替换为用户批准的可用 key
  - 创建默认 conda named env `core_mem`
  - 在外部 blocker 解除后恢复 PersonaMem / LongMemEval-S 正式全量运行

## 2026-04-05 Session 010

- Worked on: 在新权限下补齐默认 conda 环境、embedding 层和非 benchmark 第一阶段项
- State changed:
  - 安装了 autoresearch hooks，供后续 session 使用
  - 将“除环境外，外部目录修改必须先问用户”写入 `AGENTS.md` 与 `change-decisions.md`
  - 创建默认 conda named env `core_mem`，并用 Python / pytest / verify-only 复验
  - 补齐 `outputs/logs` 与 `outputs/figures` 目录结构
  - 接入 `sentence-transformers/all-MiniLM-L6-v2` embedding 配置与 adapter
  - 为 `VanillaCoreMemory` 增加 `observe_text/query_text`
  - `scripts/verify_stage1_status.py --score-only` 从 `33` 提升到 `36`
  - `scripts/verify_stage1_acceptance.py --score-only` 提升到 `5`
- Evidence / artifacts:
  - `/home/image/miniconda3/envs/core_mem`
  - `CONDA_NO_PLUGINS=true conda run -n core_mem python --version` -> `Python 3.10.20`
  - `CONDA_NO_PLUGINS=true conda run -n core_mem pytest -q` 通过
  - `CONDA_NO_PLUGINS=true conda run -n core_mem python scripts/run_experiment.py --verify-only` 输出 `36`
  - `CONDA_NO_PLUGINS=true conda run -n core_mem python -c "import sentence_transformers; print(sentence_transformers.__version__)"` -> `5.3.0`
  - `pytest` 通过（23 tests）
- Next likely action:
  - 等待 provider 配额恢复后恢复 PersonaMem / LongMemEval-S 正式全量运行

## 2026-04-05 Session 011

- Worked on: 切换到用户批准的 Gemini 路径并恢复正式 benchmark 推进
- State changed:
  - 新增 `configs/gemini_flash.yaml` 作为 Google AI Studio OpenAI-compatible 试运行配置
  - `scripts/stage1_preflight.py` 改为 provider-agnostic API key 检查
  - 为 provider 增加 `max_retries` / `retry_backoff_seconds`
  - 为 `scripts/run_experiment.py` 增加对旧式 `llm_config` 的向后兼容默认值
  - 增加 Gemini 配置测试与 rate-limit retry 测试，`pytest` 提升到 `25` 个测试并继续通过
  - Gemini 下 PersonaMem / LongMemEval-S 的真实 1-sample run 成功
  - Gemini 下 LongMemEval-S formal run 从此前的 `2` 条 prediction 续跑到更高进度，不再因首个 `HTTP 429` 立即崩溃
- Evidence / artifacts:
  - `configs/gemini_flash.yaml`
  - `outputs/runs/20260405T042146Z_personamem/evaluation.json`
  - `outputs/runs/20260405T042150Z_longmemeval/evaluation.json`
  - `outputs/runs/20260405T042236Z_longmemeval/predictions.jsonl`
  - `pytest` 通过（25 tests）
  - `scripts/run_experiment.py --verify-only` 输出 `36`
  - `scripts/verify_stage1_acceptance.py` 返回 `5/7`
- Next likely action:
  - 继续推进 Gemini 下的 formal benchmark
  - 如果 `429` 压力下吞吐仍偏低，则增加显式 pacing 控制

## 2026-04-05 Session 012

- Worked on: 用前台 autoresearch 持续尝试完成 Gemini formal benchmark，并将失败路径机械化收束为真实外部 blocker
- State changed:
  - 为 provider 增加显式 pacing、retry delay cap，并补齐相应测试
  - 新增 `scripts/run_until_complete.py` 作为外层 benchmark supervisor
  - 新增 `configs/gemini_flash_ultraslow.yaml` 用于超保守单样本推进检查
  - `pytest` 提升到 `31` 个测试并继续通过
  - PersonaMem formal run 在 `outputs/runs/20260405T050000Z_personamem/` 下推进到 `22/589`
  - LongMemEval-S formal run 在 `outputs/runs/20260405T042236Z_longmemeval/` 下推进到 `19/500`
  - 即使在 pacing、bounded retry、delay cap、outer supervisor、chunked relaunch 和 ultra-slow single-sample 检查下，当前 Gemini key 仍连续返回 `HTTP 429`，无法把 PersonaMem 从 `22` 推到 `23`
  - 当前第一阶段仍停在 acceptance `5/7`，formal benchmark 两项受真实外部 blocker 限制
- Evidence / artifacts:
  - `scripts/run_until_complete.py`
  - `configs/gemini_flash_ultraslow.yaml`
  - `outputs/runs/20260405T050000Z_personamem/predictions.jsonl` -> `22`
  - `outputs/runs/20260405T042236Z_longmemeval/predictions.jsonl` -> `19`
  - `pytest` 通过（31 tests）
  - `scripts/run_experiment.py --verify-only` 输出 `36`
  - `research-results.tsv` 已记录 iteration `18`~`29` 的 pacing / supervisor / blocker 证据
- Next likely action:
  - 等待更高 quota / 更稳定的 Gemini key，或用户批准的其他可用 provider
  - provider 条件恢复后，从现有 run dir 继续 resume formal benchmark

## 2026-04-06 Session 013

- Worked on: 审阅当前 stage-1 模型、benchmark 主链路、memory 原型和已有实验，并沉淀详细白盒实现文档
- State changed:
  - 新增 `docs/current_model_implementation.md`
  - 在 `docs/current_status.md` 中明确写入：当前 `VanillaCoreMemory` 最小原型已实现，但正式 benchmark runner 仍是 full-context prompt 主链路，尚未接入 memory write/read 的正式推理闭环
  - 在 `.agent-os/project-index.md` 中补充本次审阅结论，便于后续 session 恢复
- Evidence / artifacts:
  - `docs/current_model_implementation.md`
  - `docs/current_status.md`
  - `.agent-os/project-index.md`
- Next likely action:
  - 基于该白盒审阅文档，讨论并确定下一版 memory-mediated benchmark inference 的集成方案

## 2026-04-06 Session 014

- Worked on: 将用户确认过的 stage-2 / `V2.0 structured latent-slot memory` 方案写入真源文档，并同步整个项目文档系统
- State changed:
  - `docs/requirements.md` 已扩展为 stage-1 + stage-2 双阶段真源文档
  - 根契约 `AGENTS.md` / `CLAUDE.md` 已同步第二阶段边界与恢复规则，且 `CLAUDE.md` 已重新建立为指向 `AGENTS.md` 的硬链接
  - 新增 `docs/v2_design.md`，汇总第二阶段方法、训练、数据、评测和输出规范
  - `docs/current_status.md`、`docs/implementation_plan.md`、`docs/todo.md` 已同步 stage-2 真相
  - `.agent-os/project-index.md`、`.agent-os/todo.md`、`.agent-os/architecture-milestones.md`、`.agent-os/change-decisions.md`、`.agent-os/acceptance-report.md`、`.agent-os/lessons-learned.md` 已同步 stage-2 真相
  - 全局 top next action 从受外部 provider 阻断的 stage-1 formal benchmark，切换为可独立推进的 stage-2 数据与规范管线骨架 `TD-015`
- Evidence / artifacts:
  - `docs/requirements.md`
  - `docs/v2_design.md`
  - `docs/current_status.md`
  - `docs/implementation_plan.md`
  - `docs/todo.md`
  - `.agent-os/project-index.md`
  - `.agent-os/todo.md`
  - `.agent-os/architecture-milestones.md`
- Next likely action:
  - 开始实现第二阶段 `TD-015`：observation normalization、belief JSON 目标生成与公开数据集接入骨架
