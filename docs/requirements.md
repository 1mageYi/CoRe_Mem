# Requirements

## 文档地位

本文件是本项目的最高优先级真源文档，定义项目的最终目标、范围、约束、第一阶段验收标准与明确非目标。

除非用户明确要求，否则任何人或 agent 不得私自修改本文件中的以下内容：

- 最终目标
- 第一阶段目标
- 第一阶段验收标准
- 非目标
- 硬约束

如果其他文档与本文件冲突，以本文件为准。

## 项目名称

- `OBJ-001`: `CoRe_Mem`

## 项目总目标

- `OBJ-002`: 实现一个 bounded cross-session personalized memory framework，其第一阶段具体实例为 vanilla CoRe Memory，并在指定 benchmark 上完成可复现的实验运行，产出可信结果、表格和相关实验数据。

## 研究核心假设

- `REQ-001`: 在固定 memory budget 下，vanilla CoRe Memory 作为一种 bounded cross-session personalized memory framework，相比仅依赖上下文窗口或简单文本记忆，更适合处理跨 session 的 personalized memory 问题。
- `REQ-002`: 第一阶段重点是建立完整、可复现、可验证的研究平台与方法实现，跑通指定 benchmark，并产出可信实验结果。
- `REQ-003`: 第一阶段不要求性能优于 baseline。

## 第一阶段最终目标

- `REQ-004`: 在 repo 中实现 vanilla CoRe Memory 的完整可运行框架。
- `REQ-005`: 在 PersonaMem 上按官方 protocol 跑通主任务，并产出正式结果。
- `REQ-006`: 在 LongMemEval-S 上按官方 protocol 跑通评测流程，并产出正式结果。
- `REQ-007`: 提供可复现实验总入口脚本与分步骤脚本。
- `REQ-008`: 产出结果数据、至少一张主结果表，并保存可追溯实验产物。
- `REQ-009`: 建立完整项目文档系统，使新 agent 能在无聊天上下文的情况下从文件恢复工作。
- `REQ-010`: 建立测试体系，确保后续改动不会 silently 破坏整体逻辑。

## 第一阶段范围

### 方法

- `REQ-011`: 第一阶段方法实现为 vanilla CoRe Memory。
- `REQ-012`: 方法包含 bounded core memory、bounded residual slots 与 training-light implementation。
- `REQ-013`: 必须使用 pretrained embedding model 作为基础表示层。
- `REQ-014`: `memory latent` 建立在基础表示层之上。
- `REQ-015`: 允许训练轻量 scorer / projector / gating module。
- `REQ-016`: 允许 heuristic 或 prompt-based 组件。

### Benchmark / Dataset

- `REQ-017`: PersonaMem 为第一阶段硬要求 benchmark。
- `REQ-018`: PersonaMem 使用官方主任务、官方 multiple-choice setting 和官方 evaluation protocol。
- `REQ-019`: 第一阶段正式运行 PersonaMem `32k`；`128k` 与 `1M` 只要求保留开关和启动能力。
- `REQ-020`: LongMemEval-S 为第一阶段硬要求 benchmark，但优先级次于 PersonaMem。
- `REQ-021`: LongMemEval-S 使用官方 `S` 版本和官方 evaluation protocol。
- `REQ-022`: LoCoMo 不属于第一阶段硬要求，仅作为 planned stress test / future extension。

### 工程

- `REQ-023`: 项目固定使用 conda 环境 `core_mem`。
- `REQ-024`: Python 版本固定为 `3.10`。
- `REQ-025`: repo 必须包含完整目录结构：`docs/`、`.agent-os/`、`src/`、`scripts/`、`configs/`、`outputs/`、`tests/`。
- `REQ-026`: 所有实验结果保存在 `outputs/`。
- `REQ-027`: 必须提供总入口脚本和分步骤脚本。
- `REQ-028`: 必须具备单元测试与最小 E2E smoke test。

## 第一阶段非目标

- `REQ-029`: 第一阶段不要求 outperform baseline。
- `REQ-030`: 第一阶段不在本 repo 内运行或整合 baseline。
- `REQ-031`: 第一阶段不做 assignment randomization ablation。
- `REQ-032`: 第一阶段不做 spectral / multi-timescale CoRe extension。
- `REQ-033`: 第一阶段不做复杂大规模训练。
- `REQ-034`: 第一阶段不训练独立 embedding model。
- `REQ-035`: 第一阶段不扩展到自由生成任务或 profiling-only task。
- `REQ-036`: 第一阶段不要求 PersonaMem `128k / 1M` 正式结果。
- `REQ-037`: 第一阶段不要求 LongMemEval-M 或 LoCoMo 正式结果。
- `REQ-038`: 第一阶段不把报告写作作为硬验收。
- `REQ-039`: 第一阶段不追求 SOTA 或 leaderboard 竞争性结果。

## 硬约束

- `REQ-040`: 项目 repo 根目录固定为 `C:\Users\73257\OneDrive\Columbia\26SP\6998 CLMM\final_project\CoRe_Mem`。
- `REQ-041`: 默认外部 LLM provider 为阿里云兼容 OpenAI API。
- `REQ-042`: 默认 `base_url` 为 `https://dashscope.aliyuncs.com/compatible-mode/v1`。
- `REQ-043`: 默认模型为 `qwen3-235b-a22b-instruct-2507`。
- `REQ-044`: API key 不得写入 repo 文档或源码仓库，只能通过环境变量或本地 `.env` 注入。
- `REQ-045`: 回答和文档默认使用中文，风格直接、准确、可追溯。
- `REQ-046`: 所有代码注释和 `print` / 日志输出默认使用英文。
- `REQ-047`: 若任务与本文件未对齐，应先指出缺口，不得自行扩展范围。

## 数据集与基准

### 第一阶段承诺数据集

- `REQ-048`: PersonaMem
- `REQ-049`: LongMemEval-S

### 未来参考数据集

- `REQ-050`: PersonaMem-128k
- `REQ-051`: PersonaMem-1M
- `REQ-052`: LoCoMo
- `REQ-053`: 后续轻量训练或 embedding ablation 可能涉及的辅助数据

## 方法边界

- `REQ-054`: 第一阶段方法必须包含以下核心模块：writer、core updater、residual manager、reader、benchmark adapters、evaluation / output pipeline。
- `REQ-055`: 不得进行重型大模型 finetuning。

## 可复现性要求

- `REQ-056`: 必须提供一个总入口实验脚本。
- `REQ-057`: 必须提供分步骤脚本，至少覆盖 prepare、run、evaluate、make tables；`make figures` 为建议项。
- `REQ-058`: 相同配置下可以重新跑出一份结果。
- `REQ-059`: 必须保存以下实验元数据：config snapshot、prompt template / prompt version、model name、run timestamp、commit hash。
- `REQ-060`: 必须保存原始 prediction outputs。
- `REQ-061`: 必须保存最终结果数据和表格。
- `REQ-062`: 图是建议项，不是第一阶段硬验收；若成本合理，应尽量生成至少一张主结果图。

## 测试要求

- `REQ-063`: 每个核心模块至少有一个 unit test。
- `REQ-064`: 至少有一条最小 E2E smoke test。
- `REQ-065`: 后续每一次改动都必须通过测试。
- `REQ-066`: 未通过测试的改动不得视为完成。

## 第一阶段验收标准

- `AC-001`: repo 中存在 vanilla CoRe Memory 的完整实现，且核心模块齐全并可运行。
- `AC-002`: PersonaMem 能跑完整 benchmark split，使用官方主任务和官方 evaluation protocol，并且至少在一个正式子集上产出最终结果，结果可重复生成。
- `AC-003`: LongMemEval-S 能跑官方 `S` 版本，使用官方 evaluation protocol，并且产出正式结果。
- `AC-004`: 输出正式结果数据与至少一张主结果表，并保存 prediction outputs、实验配置和运行元数据。
- `AC-005`: conda 环境固定，总入口脚本与分步骤脚本齐全，`outputs/` 目录结构清晰，新 session / 新 agent 可通过文件恢复项目状态。
- `AC-006`: 核心模块 unit tests 完整，至少一条 E2E smoke test 可运行，当前主分支改动通过测试。
- `AC-007`: benchmark 配置、模型配置、提示词、运行记录、结果产物可追溯；允许 negative result；第一阶段成功以“系统完整、实验可运行、结论可信”为准，而不是以“优于 baseline”为准。

## 明确待定项

- `RSK-001`: 具体 pretrained embedding model 名称待定。
- `RSK-002`: 第一阶段是否顺带生成主结果图待定，但不影响第一阶段验收。
- `RSK-003`: LoCoMo 是否进入第二阶段待定。
- `RSK-004`: 轻量 scorer / projector 的具体实现方式待定。

## 文件恢复优先级

- `REQ-067`: 新 session / 新 agent 恢复项目时，必须按以下优先级读取：`AGENTS.md` -> `docs/requirements.md` -> `.agent-os/project-index.md` -> 活跃项相关文档 -> 最新 run log。
- `REQ-068`: 若 `docs/requirements.md` 与其他文档冲突，以 `docs/requirements.md` 为准。

## Source Note

本文件保存用户定义的项目意图。允许为了清晰度进行忠实整理，但不得改变其含义。
