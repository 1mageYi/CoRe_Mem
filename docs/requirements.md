# Requirements

## 文档地位

本文件是本项目的最高优先级真源文档，定义项目的总目标、阶段目标、范围、约束、验收标准与明确非目标。

除非用户明确要求，否则任何人或 agent 不得私自修改本文件中的以下内容：

- 最终目标
- 第一阶段目标
- 第一阶段验收标准
- 第二阶段目标
- 第二阶段验收标准
- 非目标
- 硬约束

如果其他文档与本文件冲突，以本文件为准。

## 项目名称

- `OBJ-001`: `CoRe_Mem`

## 项目总目标

- `OBJ-002`: 实现一个 bounded cross-session personalized memory framework。
- `OBJ-003`: 第一阶段具体实例为 `vanilla CoRe Memory`，目标是建立完整、可复现、可验证的平台与 benchmark 运行闭环。
- `OBJ-004`: 第二阶段具体实例为 `V2.0 structured latent-slot memory`，目标是在保留第一阶段平台和 v1 naive baseline 参考价值的前提下，研究并实现一个更强的 latent memory 方法。

## 研究核心假设

- `REQ-001`: 在固定 memory budget 下，vanilla CoRe Memory 作为一种 bounded cross-session personalized memory framework，相比仅依赖上下文窗口或简单文本记忆，更适合处理跨 session 的 personalized memory 问题。
- `REQ-002`: 第一阶段重点是建立完整、可复现、可验证的研究平台与方法实现，跑通指定 benchmark，并产出可信实验结果。
- `REQ-003`: 第一阶段不要求性能优于 baseline。
- `REQ-004`: 第二阶段的核心研究方向是 `structured latent slots`，而不是原始文本 chunk RAG，也不是单一全局向量记忆。
- `REQ-005`: 第二阶段的 memory 应以 `observation -> slot -> belief` 为主链路，answer-time 主体应是 query-conditioned belief recovery，而不是直接回放原始文本。
- `REQ-006`: 第二阶段假设，在固定 slot/token budget 下，`core + residual` 双银行 latent memory、显式 lifecycle、query-conditioned composition 和 belief decoding 可以比 naive stage-1 方法更好地处理长期 personalized memory 与 update-sensitive memory。
- `REQ-007`: 第二阶段允许训练，但不以重型大模型全量 finetuning 为前提；优先采用 `public-datasets-first, synthetic-minimal` 的训练策略。

## 第一阶段最终目标

- `REQ-008`: 在 repo 中实现 vanilla CoRe Memory 的完整可运行框架。
- `REQ-009`: 在 PersonaMem 上按官方 protocol 跑通主任务，并产出正式结果。
- `REQ-010`: 在 LongMemEval-S 上按官方 protocol 跑通评测流程，并产出正式结果。
- `REQ-011`: 提供可复现实验总入口脚本与分步骤脚本。
- `REQ-012`: 产出结果数据、至少一张主结果表，并保存可追溯实验产物。
- `REQ-013`: 建立完整项目文档系统，使新 agent 能在无聊天上下文的情况下从文件恢复工作。
- `REQ-014`: 建立测试体系，确保后续改动不会 silently 破坏整体逻辑。

## 第一阶段范围

### 方法

- `REQ-015`: 第一阶段方法实现为 vanilla CoRe Memory。
- `REQ-016`: 方法包含 bounded core memory、bounded residual slots 与 training-light implementation。
- `REQ-017`: 必须使用 pretrained embedding model 作为基础表示层。
- `REQ-018`: `memory latent` 建立在基础表示层之上。
- `REQ-019`: 允许训练轻量 scorer / projector / gating module。
- `REQ-020`: 允许 heuristic 或 prompt-based 组件。

### Benchmark / Dataset

- `REQ-021`: PersonaMem 为第一阶段硬要求 benchmark。
- `REQ-022`: PersonaMem 使用官方主任务、官方 multiple-choice setting 和官方 evaluation protocol。
- `REQ-023`: 第一阶段正式运行 PersonaMem `32k`；`128k` 与 `1M` 只要求保留开关和启动能力。
- `REQ-024`: LongMemEval-S 为第一阶段硬要求 benchmark，但优先级次于 PersonaMem。
- `REQ-025`: LongMemEval-S 使用官方 `S` 版本和官方 evaluation protocol。
- `REQ-026`: LoCoMo 不属于第一阶段硬要求，仅作为 planned stress test / future extension。

### 工程

- `REQ-027`: 项目固定使用 conda 环境 `core_mem`，并使用 conda 默认 envs 目录中的 named environment；不得在 repo 工作区内放置 conda prefix 环境。
- `REQ-028`: repo 只保留一个环境规范文件 `environment.yaml` 用于复现依赖，不保留 project-local conda bootstrap / runner 脚本作为硬要求。
- `REQ-029`: Python 版本固定为 `3.10`。
- `REQ-030`: repo 必须包含完整目录结构：`docs/`、`.agent-os/`、`src/`、`scripts/`、`configs/`、`outputs/`、`tests/`。
- `REQ-031`: 所有第一阶段实验结果保存在 `outputs/`。
- `REQ-032`: 必须提供总入口脚本和分步骤脚本。
- `REQ-033`: 必须具备单元测试与最小 E2E smoke test。

## 第一阶段非目标

- `REQ-034`: 第一阶段不要求 outperform baseline。
- `REQ-035`: 第一阶段不在本 repo 内运行或整合 baseline。
- `REQ-036`: 第一阶段不做 assignment randomization ablation。
- `REQ-037`: 第一阶段不做 spectral / multi-timescale CoRe extension。
- `REQ-038`: 第一阶段不做复杂大规模训练。
- `REQ-039`: 第一阶段不训练独立 embedding model。
- `REQ-040`: 第一阶段不扩展到自由生成任务或 profiling-only task。
- `REQ-041`: 第一阶段不要求 PersonaMem `128k / 1M` 正式结果。
- `REQ-042`: 第一阶段不要求 LongMemEval-M 或 LoCoMo 正式结果。
- `REQ-043`: 第一阶段不把报告写作作为硬验收。
- `REQ-044`: 第一阶段不追求 SOTA 或 leaderboard 竞争性结果。

## 第一阶段验收标准

- `AC-001`: repo 中存在 vanilla CoRe Memory 的完整实现，且核心模块齐全并可运行。
- `AC-002`: PersonaMem 能跑完整 benchmark split，使用官方主任务和官方 evaluation protocol，并且至少在一个正式子集上产出最终结果，结果可重复生成。
- `AC-003`: LongMemEval-S 能跑官方 `S` 版本，使用官方 evaluation protocol，并且产出正式结果。
- `AC-004`: 输出正式结果数据与至少一张主结果表，并保存 prediction outputs、实验配置和运行元数据。
- `AC-005`: conda 环境固定，`environment.yaml` 可用于在默认 conda envs 目录复现 `core_mem`，总入口脚本与分步骤脚本齐全，`outputs/` 目录结构清晰，新 session / 新 agent 可通过文件恢复项目状态。
- `AC-006`: 核心模块 unit tests 完整，至少一条 E2E smoke test 可运行，当前主分支改动通过测试。
- `AC-007`: benchmark 配置、模型配置、提示词、运行记录、结果产物可追溯；允许 negative result；第一阶段成功以“系统完整、实验可运行、结论可信”为准，而不是以“优于 baseline”为准。

## 第二阶段最终目标

- `REQ-045`: 第二阶段目标是实现 `V2.0 structured latent-slot memory`，作为 stage-1/v1 naive CoRe Memory 的研究性迭代。
- `REQ-046`: 第二阶段优先保留第一阶段 v1 作为 naive baseline 与对照实现；允许 V2.0 成为后续默认主线，但不得在无说明的情况下抹去 v1 的可复现参考价值。
- `REQ-047`: 第二阶段方法主线必须使用 `structured latent slots` 作为记忆基本形式。
- `REQ-048`: 第二阶段 memory 不得退化为原始文本 chunk RAG。
- `REQ-049`: 第二阶段 memory 不得退化为单一全局向量记忆。
- `REQ-050`: 第二阶段 answer-time 记忆恢复主链路必须为 `latent composition -> belief decoding -> answer projection`。
- `REQ-051`: 第二阶段 belief decoder 的主输出目标为结构化 `Belief JSON`，而不是直接复原原文。
- `REQ-052`: 第二阶段主实现路线锁定为：
  - `Resampler`: Light Cross-Attention Resampler
  - `Decoder`: Flan-T5-based belief JSON decoder
  - `Trainable modules`: LoRA/adapter + resampler/projection
  - `Primary output`: Belief JSON
- `REQ-053`: 第二阶段允许将其他路线作为 ablation 或后续升级，包括：
  - `Optimus-like latent decoder`
  - 更重的 Perceiver-style resampler
  - direct-answer decoder target
- `REQ-054`: 第二阶段训练目标拆分为四个任务：
  - `Slot Autoencoding`
  - `Retrieval Alignment`
  - `Lifecycle Prediction`
  - `Composition-to-Belief Decoding`

## 第二阶段范围

### 方法

- `REQ-055`: 第二阶段 memory 表示采用 `structured latent slots`，slot 内部默认使用连续 latent tokens，而非离散 codebook token。
- `REQ-056`: 第二阶段 slot 默认不是单一向量，而是固定数量的 latent tokens；默认超参数由配置控制，可随实验调整。
- `REQ-057`: 第二阶段 memory 采用双银行结构：`core` 与 `residual`。
- `REQ-058`: `core` 负责长期稳定状态；`residual` 负责吸收新信息、近期事件、候选更新与冲突。
- `REQ-059`: 第二阶段 slot lifecycle 锁定为：
  - 状态：`active`、`stale`、`inactive`
  - 动作：`new`、`merge`、`overwrite`、`evict`
- `REQ-060`: 第二阶段 core/residual 更新机制锁定为：新 observation 先经 residual 视角吸收，再通过 consolidation 决定是否强化、晋升或替换 core。
- `REQ-061`: 第二阶段必须显式包含 consolidation 过程，用于：
  - residual -> core promotion
  - stale core handling
  - duplicate merge
  - bank 内 eviction
- `REQ-062`: 第二阶段 observation parser 采用混合策略：`rule-first, model-second`。
- `REQ-063`: 第二阶段 memory 内容是开放域的；不使用封闭 ontology 限制“能记什么”。
- `REQ-064`: 第二阶段控制层使用 `bank + soft role scores + open semantic payload`。
- `REQ-065`: 第二阶段 relation normalization 允许使用小而稳定的标准化集合做训练与评估，但该集合只服务于 normalization，不构成 memory 内容世界的硬边界。
- `REQ-066`: 第二阶段 `canonical_gloss` 仅用于解释、审计和错误分析；不作为 answer-time 主 memory 载体。
- `REQ-067`: 第二阶段核心链路固定为：`observation -> slot -> belief`。

### Observation / Belief 规范

- `REQ-068`: 第二阶段必须定义标准化 `Observation JSON`，至少包含：
  - `obs_id`
  - `source_dataset`
  - `source_dialogue_id`
  - `source_turn_id`
  - `session_id`
  - `speaker`
  - `entity`
  - `relation`
  - `value`
  - `value_type`
  - `time_scope`
  - `status_hint`
  - `polarity`
  - `confidence`
  - `evidence_text`
  - `canonical_gloss`
  - `metadata`
- `REQ-069`: 第二阶段必须定义标准化 `Belief JSON`，至少包含：
  - `query_id`
  - `entity`
  - `query_type`
  - `belief_items`
  - `global_consistency`
- `REQ-070`: `belief_items` 至少包含：
  - `relation`
  - `value`
  - `status`
  - `time_scope`
  - `confidence`
  - `support_slot_ids`
- `REQ-071`: 第二阶段 belief decoder 默认输出结构化 JSON；必要时允许在 answer projection 阶段将 belief 渲染为短 evidence block。

### 数据与训练

- `REQ-072`: 第二阶段训练数据策略锁定为 `public-datasets-first, synthetic-minimal`。
- `REQ-073`: 第二阶段 benchmark 官方评测集不是 primary training source。
- `REQ-074`: benchmark-derived proxy supervision 不作为主要训练信号；只用于本地 intrinsic evaluation、ablation 和 failure analysis。
- `REQ-075`: 第二阶段主训练数据池锁定为：
  - `State / Belief Pool`
  - `Persona / Preference Pool`
  - `Update / Propagation Pool`
  - `Minimal Synthetic Pool`
- `REQ-076`: 第二阶段主训练数据优先采用下列公开数据集：
  - `Schema-Guided Dialogue (SGD)`
  - `MultiWOZ 2.4`
  - `Persona-Chat`
  - `Training Millions of Personalized Dialogue Agents`
  - `MQUAKE`
  - `ReCoE`
- `REQ-077`: `DuLeMon`、`REALTALK`、`MS-TOD` 可作为第二优先级候选数据源，只有在清洗与接入成本可控时才进入第二阶段主训练。
- `REQ-078`: `synthetic` 数据只用于补足公开数据中难以覆盖的 lifecycle、budget stress、conflict bundle 与 multi-fact composition corner cases。
- `REQ-079`: 第二阶段训练顺序默认采用 staged training，再做 joint fine-tuning，顺序为：
  - `Slot Autoencoding`
  - `Retrieval Alignment`
  - `Lifecycle Prediction`
  - `Composition-to-Belief Decoding`
  - `Joint fine-tuning`

### Backbone 与默认实现

- `REQ-080`: 第二阶段默认 decoder backbone 为 `google/flan-t5-base`。
- `REQ-081`: 第二阶段默认 encoder 先与 decoder backbone 同源初始化。
- `REQ-082`: 第二阶段 retrieval encoder 默认先与主 encoder 共享权重；若后续实验需要，可在不破坏主线对照的前提下拆分。
- `REQ-083`: 第二阶段默认 resampler 为轻量 cross-attention 结构，作为主线实现；mean pooling 与更重 resampler 仅作为 ablation 或升级路线。
- `REQ-084`: 第二阶段默认 parser fallback 可使用同一 backbone 承担结构化抽取，不强制单独引入另一套 parser backbone。

### 本地评估与 benchmark 评估

- `REQ-085`: 第二阶段本地 intrinsic evaluation 采用 `4 个现成指标家族 + 2 个少量自定义指标`。
- `REQ-086`: 第二阶段本地评估至少记录：
  - `Recall@k`
  - `NDCG@k`
  - `Joint Belief Accuracy`
  - slot/value F1
  - `Update Success`
  - `Propagation Accuracy`
  - `Specificity / Locality`
  - `Compression Fidelity`
  - `Edit Locality at Slot Level`
- `REQ-087`: 第二阶段本地实验默认使用 `3` 个随机种子并汇报 `mean ± std`。
- `REQ-088`: 第二阶段必须支持 `slot budget` 作为配置项，并进行 budget sweep。
- `REQ-089`: 第二阶段 benchmark 评测采用 `local-first` 协议：
  - 先跑本地 intrinsic metrics
  - 再跑 canary benchmark
  - 最后才跑正式 benchmark
- `REQ-090`: 第二阶段默认 benchmark canary 子集大小为：
  - PersonaMem `64`
  - LongMemEval-S `64`
  子集采用分层固定抽样，后续所有方法共享同一子集。

### 工程与输出

- `REQ-091`: 第二阶段当前训练环境默认按单卡方案设计。
- `REQ-092`: 第二阶段允许下载额外公开 checkpoint。
- `REQ-093`: 第二阶段所有训练、评估和分析产物统一放在 `outputs_v2/`。
- `REQ-094`: `outputs_v2/` 默认子目录为：
  - `checkpoints/`
  - `runs/`
  - `evals_local/`
  - `evals_benchmark/`
  - `tables/`
  - `figures/`
  - `logs/`
  - `cache_manifests/`
  - `artifacts/`
- `REQ-095`: `outputs_v2/` 下必须保存：
  - 配置快照
  - 训练与评估元数据
  - 本地指标结果
  - benchmark canary / formal 结果
  - belief dumps
  - slot dumps
  - 关键分析产物

## 第二阶段非目标

- `REQ-096`: 第二阶段不要求抹去或删除第一阶段 v1。
- `REQ-097`: 第二阶段不要求完全不使用 synthetic data。
- `REQ-098`: 第二阶段不将 benchmark eval/test 作为主要训练源。
- `REQ-099`: 第二阶段不要求一开始就采用更重的 Perceiver-style resampler。
- `REQ-100`: 第二阶段不要求一开始就采用更纯但更难训练的 Optimus-like 路线作为主线。
- `REQ-101`: 第二阶段不要求一开始就做多卡分布式训练。
- `REQ-102`: 第二阶段不要求直接追求 SOTA 或 leaderboard 竞争性结果。

## 第二阶段验收标准

- `AC-008`: repo 中存在一份详细、可执行的第二阶段 `V2.0` 设计文档，并与真源需求保持一致。
- `AC-009`: repo 中存在第二阶段 observation / slot / belief 的标准化数据规范和训练任务定义。
- `AC-010`: repo 中存在第二阶段主线实现骨架：Light Cross-Attention Resampler、Flan-T5 belief decoder、LoRA/adapter 训练路径。
- `AC-011`: repo 中存在第二阶段训练与本地 intrinsic evaluation 管线，且可在公开数据集上运行。
- `AC-012`: repo 中存在第二阶段本地评估结果、budget sweep 和核心 ablation 的记录机制。
- `AC-013`: 第二阶段 benchmark 评测遵守 `local-first` 与 canary-first 协议，且 benchmark 正式结果与训练数据严格隔离。

## 硬约束

- `REQ-103`: 项目 repo 根目录固定为 `C:\Users\73257\OneDrive\Columbia\26SP\6998 CLMM\final_project\CoRe_Mem`。
- `REQ-104`: 默认外部 LLM provider 为阿里云兼容 OpenAI API。
- `REQ-105`: 默认 `base_url` 为 `https://dashscope.aliyuncs.com/compatible-mode/v1`。
- `REQ-106`: 默认模型为 `qwen3-235b-a22b-instruct-2507`。
- `REQ-107`: API key 不得写入 repo 文档或源码仓库，只能通过环境变量或本地 `.env` 注入。
- `REQ-108`: 回答和文档默认使用中文，风格直接、准确、可追溯。
- `REQ-109`: 所有代码注释和 `print` / 日志输出默认使用英文。
- `REQ-110`: 若任务与本文件未对齐，应先指出缺口，不得自行扩展范围。

## 数据集与基准

### 第一阶段承诺数据集

- `REQ-111`: PersonaMem
- `REQ-112`: LongMemEval-S

### 第二阶段评测数据集

- `REQ-113`: 第二阶段 benchmark 评测主集仍优先使用 PersonaMem 与 LongMemEval-S。
- `REQ-114`: 第二阶段 benchmark 评测中，stage-1/v1 应尽量保留为 naive baseline 参考。

### 第二阶段训练优先数据集

- `REQ-115`: `Schema-Guided Dialogue (SGD)`
- `REQ-116`: `MultiWOZ 2.4`
- `REQ-117`: `Persona-Chat`
- `REQ-118`: `Training Millions of Personalized Dialogue Agents`
- `REQ-119`: `MQUAKE`
- `REQ-120`: `ReCoE`

### 未来参考数据集

- `REQ-121`: PersonaMem-128k
- `REQ-122`: PersonaMem-1M
- `REQ-123`: LoCoMo
- `REQ-124`: `DuLeMon`
- `REQ-125`: `REALTALK`
- `REQ-126`: `MS-TOD`

## 方法边界

- `REQ-127`: 第一阶段方法必须包含以下核心模块：writer、core updater、residual manager、reader、benchmark adapters、evaluation / output pipeline。
- `REQ-128`: 第二阶段方法必须包含以下核心模块：observation parser、slot encoder、retrieval key head、core/residual memory banks、lifecycle module、consolidation、resampler、belief decoder、answer projection、local evaluation pipeline。
- `REQ-129`: 第二阶段训练允许使用 LoRA / adapter / resampler / projector 等轻中量训练，但不要求大模型全量 finetuning。

## 可复现性要求

- `REQ-130`: 第一阶段必须提供一个总入口实验脚本。
- `REQ-131`: 第一阶段必须提供分步骤脚本，至少覆盖 prepare、run、evaluate、make tables；`make figures` 为建议项。
- `REQ-132`: 相同配置下可以重新跑出一份第一阶段结果。
- `REQ-133`: 第一阶段必须保存以下实验元数据：config snapshot、prompt template / prompt version、model name、run timestamp、commit hash。
- `REQ-134`: 第一阶段必须保存原始 prediction outputs。
- `REQ-135`: 第一阶段必须保存最终结果数据和表格。
- `REQ-136`: 图是第一阶段建议项，不是硬验收；若成本合理，应尽量生成至少一张主结果图。
- `REQ-137`: 第二阶段必须保存训练配置、数据版本、checkpoint、local intrinsic evaluation 结果、benchmark canary 结果和关键分析 artifact。
- `REQ-138`: 第二阶段必须保存 belief dumps、slot dumps 与配置快照，以支持错误分析和可追溯恢复。

## 测试要求

- `REQ-139`: 每个第一阶段核心模块至少有一个 unit test。
- `REQ-140`: 第一阶段至少有一条最小 E2E smoke test。
- `REQ-141`: 后续每一次改动都必须通过相关测试。
- `REQ-142`: 未通过测试的改动不得视为完成。
- `REQ-143`: 第二阶段核心模块应逐步补齐单测，至少覆盖 observation normalization、slot lifecycle、resampler I/O、belief decoder I/O 与本地评估脚本。

## 明确待定项

- `RSK-001`: 第一阶段是否顺带生成主结果图待定，但不影响第一阶段验收。
- `RSK-002`: LoCoMo 是否进入后续阶段待定。
- `RSK-003`: 第二阶段是否最终拆分独立 retrieval encoder 待定。
- `RSK-004`: 第二阶段是否引入第二优先级数据集 `DuLeMon / REALTALK / MS-TOD` 待定。
- `RSK-005`: 第二阶段 `Optimus-like` 路线是否能在控制成本下形成有竞争力的 ablation 待定。

## 文件恢复优先级

- `REQ-144`: 新 session / 新 agent 恢复项目时，必须按以下优先级读取：`AGENTS.md` -> `docs/requirements.md` -> `.agent-os/project-index.md` -> 活跃项相关文档 -> 最新 run log。
- `REQ-145`: 若 `docs/requirements.md` 与其他文档冲突，以 `docs/requirements.md` 为准。

## Source Note

本文件保存用户定义的项目意图。允许为了清晰度进行忠实整理，但不得改变其含义。
