# Change Decisions

## `CD-001` 2026-04-04

- 决定真源文档位于 `docs/requirements.md`，而非 `.agent-os/requirements.md`。
- 决定 `docs/` 与 `.agent-os/` 并存。
- 决定 `.agent-os/todo.md` 为 agent runtime truth，`docs/todo.md` 为人类可读视图。

## `CD-002` 2026-04-04

- 决定第一阶段方法锁定为 `vanilla CoRe Memory`。
- `spectral / multi-timescale extension`、`assignment randomization ablation` 均不属于第一阶段。

## `CD-003` 2026-04-04

- 决定第一阶段 benchmark：
  - `PersonaMem` 为主 benchmark
  - `LongMemEval-S` 为第二优先级 planned benchmark
  - `LoCoMo` 为未来扩展

## `CD-004` 2026-04-04

- 决定第一阶段接受 negative result。
- 第一阶段成功标准为“系统完整、实验可运行、结论可信”，而不是“优于 baseline”。

## `CD-005` 2026-04-05

- 决定 PersonaMem 第一阶段先对齐官方 `32k` multiple-choice 主任务。
- 决定 LongMemEval 第一阶段先对齐官方 `S cleaned` 数据与官方 protocol。
- 决定 `128k / 1M` 只保留启动开关，不要求第一阶段正式结果。

## `CD-006` 2026-04-05

- 决定停止在 Windows 本地继续折腾环境，并迁移到 WSL 作为后续主执行平台。
- 决定仓库只保留平台中立代码、文档与 agent 状态，Windows-specific bootstrap 不进入 handoff commit。

## `CD-007` 2026-04-05

- 用户允许为了创建和管理默认 conda named env `core_mem` 修改 repo 外的 conda 环境路径。
- 除环境相关路径外，任何 repo 工作区之外目录的修改都必须先向用户确认，不能因为 sandbox 放开而自动执行。

## `CD-008` 2026-04-05

- 用户已明确批准在保持默认 provider 真相不变的前提下，临时使用 Google AI Studio 的 OpenAI-compatible 端点推进正式 benchmark。
- 当前批准的临时模型为 `gemini-2.5-flash`，配置文件为 `configs/gemini_flash.yaml`。

## `CD-009` 2026-04-06

- 用户明确确认第二阶段为 `V2.0 structured latent-slot memory`，并要求将其写入真源 `docs/requirements.md`。
- 第二阶段主线锁定为：
  - `Light Cross-Attention Resampler`
  - `Flan-T5-based belief JSON decoder`
  - LoRA/adapter + resampler/projection 可训练
  - `Belief JSON` 作为主输出目标
- 第二阶段数据策略锁定为 `public-datasets-first, synthetic-minimal`。
- benchmark 继续作为 evaluation source，而不是第二阶段 primary training source。
- 第一阶段 v1 可保留为 naive baseline；第二阶段作为 stage-2 研究主线推进。
- 第二阶段 observation parser 采用混合策略，默认 `rule-first, model-second`。
- 第二阶段当前默认 backbone 为 `Flan-T5-base`；更纯的 `Optimus-like` 路线仅作为 ablation 或后续升级。

## `CD-010` 2026-04-06

- 用户进一步确认：stage-2 是当前执行主线。
- stage-1 formal benchmark 保留为 baseline / acceptance 相关的 pending 项，直到用户明确要求 AI 去跑。
- 第二阶段真源文档必须写清：
  - `Observation JSON`
  - `Slot Record`
  - `Belief JSON`
  - 数据集到训练任务的精确映射
  - 指标到模块的精确映射
- 若 stage-2 文档与此前模糊表述冲突，以本轮确认后的细化版本为准。

## `CD-011` 2026-04-13

- 用户批准由 agent 自行在网上查找并下载第二阶段公开训练数据源，但要求若累计体量超过 `100GB` 必须先暂停并提醒。
- 当前第二阶段公开数据默认下载到 repo 内 `data/stage2_public/`，不写 repo 外缓存目录。
- 为遵守外部目录修改规则，stage-2 训练配置中的 Hugging Face cache 也默认锁到 repo 内路径，而不是用户家目录默认缓存。

## `CD-012` 2026-04-14

- 第二阶段本地评测的默认输出契约锁定为：
  - detailed JSON report
  - summary CSV
  - budget sweep CSV
- 第二阶段本地评测文档固定放在 `docs/stage2_local_evaluation.md`。
- local eval 默认必须同时暴露：
  - 指标家族视图
  - 模块视图
  - budget sweep 视图

## `CD-013` 2026-04-22

- 用户确认下一阶段进入 `v5 Core-Residual Latent Substrate` 路线，目标是顶会级 learned latent memory system，而不是继续围绕 benchmark 做规则修补或 provider prompt trick。
- `v5` 允许下载和使用 HuggingFace pretrained backbones，首批由 agent 比较 `BGE / E5 / Contriever`。
- `32k` stage2 数据只能称为 domain warm-up / adaptation，不能声称为充分的大规模预训练。
- PersonaMem gold 允许但必须严格隔离：
  - 只允许训练薄 answer/readout head
  - split 必须按 `shared_context_id` / persona 分组
  - memory writer / reader / controller / latent substrate 不得使用 gold answer
  - 最终必须分别报告 no-calibration 与 calibrated 结果
- MiniMax-M2.7 teacher 允许继续用于 raw-dialogue memory/belief/update supervision，但 teacher 不能以 silver label 为条件，也不能看到 PersonaMem gold answer。
- `v5` 的验收必须包含 anti-shortcut ablations：latent-only、text-only、shuffled-latent、shuffled-memory/persona、core-only、residual-only、no-controller、option-only baseline、answer-label permutation、option paraphrase。
