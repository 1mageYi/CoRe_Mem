# Current Status

## 当前结论

- 第一阶段方法已锁定为 `vanilla CoRe Memory`，第一阶段 benchmark 已锁定为：
  - 主：`PersonaMem`
  - 次：`LongMemEval-S`
  - 未来扩展：`LoCoMo`
- 第二阶段方法已锁定为 `V2.0 structured latent-slot memory`。
- 当前执行主线是第二阶段：
  - 第一阶段承担 baseline、平台和验收参考
  - 第二阶段承担 latent memory 方法升级、训练和本地评估主线

## 当前状态

- 文档状态：`docs/requirements.md` 已重建并细化为 stage-1 + stage-2 双阶段真源；`docs/v2_design.md` 已同步到实现级规格
- 第一阶段代码状态：platform ready + acceptance verifier landed + resumable benchmark runner landed + embedding-backed memory interface landed + Gemini-compatible retry/backoff / pacing / supervisor path landed + real 1-sample benchmark runs verified
- 第一阶段 memory 接入状态：`VanillaCoreMemory` 最小原型已实现并经单测验证，但当前正式 benchmark runner 仍采用 full-context prompt 主链路，尚未把 memory write/read 接入正式推理闭环
- 第一阶段 benchmark 适配状态：PersonaMem / LongMemEval-S 已分别完成默认 provider 路径与 Gemini 路径下的真实 1-sample run；正式 Gemini run 当前停在 PersonaMem `22/589`、LongMemEval-S `19/500`，且当前 key 继续触发 `HTTP 429`
- 第一阶段 formal benchmark 执行状态：**待用户显式触发**；在用户要求 AI 去跑之前，不主动继续消耗 API 推 formal benchmark
- 第二阶段设计状态：`V2.0` 方法主线、结构化 JSON、数据集到训练任务映射、指标到模块映射、默认 backbone、默认超参数与输出目录均已在真源文档中锁定
- 第二阶段实现状态：`src/core_mem/v2/` 已同时具备 Observation / Slot / Belief schema、rule-first parser、dataset registry，以及 slot encoder、lifecycle、consolidation、core/residual memory system、light resampler、belief decoder、answer projection 和 `training.py` 训练模块；其中 `encoder/resampler/decoder/system` 已从 hash/mean skeleton 升级为 parameterized lexical projection + cross-attention composition + latent-conditioned belief decode 主链；`scripts/normalize_stage2_public_data.py` 已把真实 `SGD / MultiWOZ 2.4 / Persona-Chat / MQUAKE / ReCoE` 规范化为 `normalized.jsonl`；`prepare_stage2_data.py` 已支持 source-config + strict mode + `--max-rows-per-dataset`；`scripts/train_stage2.py` 现已支持 preset experiment variant、checkpoint-aware local eval 与 experiment registry 自动登记；当前 `outputs_v2/artifacts/stage2_experiment_index.json` 已登记 `mainline + 11` 个必做 ablation，`scripts/verify_stage2_experiment_status.py --score-only` 已达 `13`
- 第二阶段 latent readiness 状态：`scripts/verify_stage2_latent_status.py --score-only` 当前已达 `9/9`；其中实现项包括 `query/slot encoder` 不再是 hash-only、`resampler` 不再是 mean-only、`decoder` 已真实消费 `composed_memory`，且 `StructuredMemorySystem.query()` 已把 composed latent 传入 belief decode 主链
- 第二阶段 benchmark canary 状态：`scripts/run_stage2_memory_canary.py` 已在 `MiniMax-M2.7` 上完成真实 live PersonaMem canary。当前已存在：
  - `outputs_v2/evals_benchmark/20260414T231441Z_stage2_memory_canary.json`：1-sample live canary，provider prediction 命中正确选项
  - `outputs_v2/evals_benchmark/20260414T231617Z_stage2_memory_canary.json`：64-sample live canary，`sample_count=64`、`live_predictions_completed=64`
- 第二阶段 benchmark 质量状态：live canary 已证明 `memory -> belief -> provider answer` 链路可运行，而且当前 session 已把质量基线从 `4/10` 提升到 `9/10`。当前 retained best `PersonaMem 64` artifact `outputs_v2/evals_benchmark/20260415T015324Z_stage2_memory_canary.json` 的快速统计为：
  - `provider exact match = 31 / 64 = 48.44%`
  - `provider label-prefix match = 31 / 64 = 48.44%`
  - `local memory answer exact match = 25 / 64 = 39.06%`
  - `stage2_personamem_canary_quality_score = 9 / 10`
  当前本轮 autoresearch 的 stop condition `>=9/10` 已满足；剩余未达项只剩 `provider_label_prefix_ge_32`
- 第二阶段 benchmark 解释规则：当前 `PersonaMem 64 = 9/10` 不能被误读为“latent model 本体已显著变强”。当前收益更准确地是：
  - 一部分来自 memory lifecycle / retention 的真实系统改进
  - 一部分来自 `PersonaMem` answer-option / label-space 对齐
  - 因此，后续若以 robustness 为目标，主优化锚点必须切换为 `latent core / local intrinsic metrics`，而不能继续把 benchmark-facing heuristic 当作主收益来源
- 第二阶段完整目标更新：当前后续主线不再是单点提分，而是朝**完整的 `v2`** 推进。完整 `v2` 至少需要同时满足：
  - fresh `PersonaMem 64` live canary 证据
  - `LongMemEval-S 64` 的 stage-2 live canary 证据
  - 默认 `flan-t5-base` 非 tiny `gpu3` 训练与 checkpoint-aware eval 证据
  - benchmark runner 不再依赖 benchmark-specific heuristic 才能维持主收益
  - learned path 对在线 memory / belief 主链开始产生真实作用
- 第二阶段完整度状态：当前 `scripts/verify_stage2_v2_completion.py --score-only = 14/14`。当前已经机械成立的里程碑包括：
  - `personamem_live_canary_completed = true`：fresh `PersonaMem 64` artifact 为 `outputs_v2/evals_benchmark/20260415T052916Z_stage2_memory_canary.json`
  - `longmemeval_live_canary_completed = true`：fresh `LongMemEval-S 64` artifact 为 `outputs_v2/evals_benchmark/20260415T054234Z_stage2_memory_canary.json`
  - `longmemeval_analysis_exists = true`：`outputs_v2/artifacts/latest_longmemeval_stage2_canary_analysis.json` 已落地
  - `benchmark_runner_avoids_shortcuts = true`：`scripts/run_stage2_memory_canary.py` 已移除 PersonaMem-specific candidate injection 和 blank-provider fallback
  - `non_tiny_train_artifact_exists = true`
  - `non_tiny_checkpoint_exists = true`
  - `trained_eval_positive = true`
  对应新增证据为：
  - `outputs_v2/runs/20260415T043648Z_stage2_train_exec/execution_summary.json`
  - `outputs_v2/checkpoints/20260415T043648Z_stage2_train_exec/`
  - `outputs_v2/evals_local/20260415T043706Z_stage2_local_eval.json`
- 第二阶段非 tiny 训练状态：当前默认 `google/flan-t5-base` 已在 `GPU3` 上完成一条真实非 tiny LoRA train/eval 证据链。该 run 使用 `CUDA_VISIBLE_DEVICES=3`、`device=cuda`、`num_examples=128`、`num_steps=8`，`final_loss=0.9725619554519653`；对应 checkpoint-aware local eval 的 `trained_eval` 为：
  - `exact_match = 0.0`
  - `token_f1 = 0.056531552294517756`
  这说明 learned path 已产生非零输出，但当前还不能把它误称为 train-complete 或 benchmark-ready。
- 第二阶段硬约束新增：后续长跑中**不要做任何偷懒兜底 fallback**。尤其不允许把 benchmark-specific heuristic / fallback 当作 retained 主收益，也不允许用 provider 空输出兜底、candidate-answer 注入或选项 overlap scorer 之类技巧冒充 latent-core 提升。
- 第二阶段 verifier 对齐状态：为保持机械评分与 runtime truth 一致，当前还补了两处 verifier 对齐：
  - `scripts/verify_stage2_v2_completion.py` 现在接受 `longmemeval_s` 作为 `LongMemEval-S` canary 的 benchmark alias
  - `scripts/verify_stage2_latent_core_quality.py` 的 canary guard 现在只消费 PersonaMem completed canary，不再让更新但质量口径不同的 `LongMemEval-S` artifact 误伤 latent-core guard
- 历史兼容说明：当前主线曾明确要求“把主指标重新锚定到 `stage-2 local intrinsic quality`”，以及“以系统/模型/latent 本体更强、更稳健为锚点提升 local intrinsic 质量”；这两条表述在本轮已由 doing 状态推进到完成态。
- 第二阶段 failure-analysis 当前结论：`scripts/analyze_stage2_memory_canary_failures.py` 的最新 artifact 为 `outputs_v2/artifacts/latest_personamem_stage2_canary_analysis.json`。当前已验证：
  - observation path 的 assistant 噪声过滤与 `create -> eat -> food_preference` 误判修复本身不足以抬高 live 分数
  - PersonaMem answer-option 协议对齐是当前主收益来源：把 prompt 和本地投影统一回 `(a)/(b)/(c)/(d)` 标签空间后，quality score 从 `4` 提升到 `8`
  - 在 provider prompt 中额外注入 latent matcher candidate 只带来 `provider exact 21 -> 22` 的小幅变化，单独不足以把 score 推到 `9`
  - 真正跨过 `9/10` 的关键改动是：对 facet-rich relation 保留多条 active memory，而不是单 relation 单槽覆盖，并让 PersonaMem option scorer 直接消费 selected slot glosses
- 第二阶段 local intrinsic 状态：本轮 managed autoresearch 已把 `stage2_latent_core_quality_score` 从 baseline `6/10` 提升到 `10/10`。当前最新 local eval artifact 为 `outputs_v2/evals_local/20260415T031952Z_stage2_local_eval.json`，关键指标为：
  - `joint_belief_accuracy = 1.0`
  - `slot_value_f1 = 1.0`
  - `support_slot_recall = 1.0`
  - `answer_exact_match = 0.8125`
  - `compression_fidelity = 1.0`
  当前主收益来自两点：
  - `BeliefDecoder` 不再对 selected slots 重新排序，而是直接消费 retrieval / lifecycle 已给出的 memory order
  - `scripts/eval_stage2_local.py` 现将 `composition_to_belief` 任务视为 memory-state -> belief recovery 评测，不再在 belief-family 里重复做一次 retrieval
- 环境状态：repo 内 project-local conda env 与缓存痕迹已清理；环境复现真相为 `environment.yaml` + 默认 conda named env `core_mem`
- 第二阶段 live provider 状态：当前 run 已在具备 `GPT_AGENT_API_KEY` 的 session 中完成 fresh `PersonaMem 64` 与 fresh `LongMemEval-S 64` live canary；此前的 `outputs_v2/evals_benchmark/20260415T043414Z_stage2_memory_canary.json` 仍保留为历史 blocked probe，但不再代表当前 session truth。
- 第二阶段 LongMemEval-S 当前结论：fresh analysis `outputs_v2/artifacts/latest_longmemeval_stage2_canary_analysis.json` 显示 `provider exact match = 1 / 64 = 1.56%`、`local exact match = 0 / 64`。因此“完整 v2 milestone 已补齐”只表示证据链完整，不表示 `LongMemEval-S` 质量已经可以扩大规模。
- 第二阶段主线切换：当前 `v2` 已解决“闭环是否成立”的问题，因此现阶段主线应切换为 **`v2.1`**。`v2.1` 的目标不再是补齐闭环证据，而是证明：
  - 这套系统是否已经足够强
  - 是否已经足够稳
  - 是否已经能够跨 benchmark 成立
  - 是否真的是模型 / latent 本体在起作用，而不是靠局部接口技巧
- 第二阶段 `v2.1` 四个核心目标：
  - 提升真实质量，而不只是通过 `64` canary
  - 提升 learned path 的实际贡献
  - 提升跨 benchmark 鲁棒性
  - 提升可扩展性与可解释性
- 第二阶段 `v2.1` 成功标准：
  - `PersonaMem` 扩大样本后仍稳定，而不只是 `64` canary
  - `LongMemEval-S` 不再只是“能跑”，而是指标明显提升
  - non-tiny learned path 对在线链路有可证明增益
  - 不依赖 benchmark-specific heuristic / fallback
  - failure analysis 能稳定归因到 `parser / retrieval / belief / answer(provider/projection)` 某一层，而不是系统混成一团
- 第二阶段 `v2.1` 机械目标：当前主指标已切换为 `scripts/verify_stage2_v21_robustness.py` 对应的 `stage2_v21_robustness_score`。该分数当前围绕 10 个里程碑簇展开：
  - fresh current-head `PersonaMem 64`
  - fresh current-head `LongMemEval-S 64`
  - current-head `PersonaMem 128` 更大切片
  - `PersonaMem 128` 的稳定性阈值
  - `LongMemEval-S` layered analysis artifact
  - `LongMemEval-S` provider/local 质量提升
  - learned online gain artifact
  - 非 tiny `trained_eval.token_f1` 继续提升
  同时保留 `latent-core = 10/10` 与 `no-shortcut runner` 作为基础 guard。
  当前分数为 `14/15`；当前已经机械成立的项是：
  - 文档 / `.agent-os` 已切到 `v2.1`
  - `latent-core = 10/10`
  - `no-shortcut runner = true`
  - fresh current-head `PersonaMem 64`
  - fresh current-head `LongMemEval-S 64`
  - current-head `PersonaMem 128`
  - `PersonaMem 128 local_exact_rate >= 0.20`
  - `LongMemEval-S` layered analysis artifact
  - `LongMemEval-S provider_label_prefix_match >= 4`
  - `LongMemEval-S local_exact_match >= 2`
  - 非 tiny `trained_eval.token_f1 >= 0.08`
- 第二阶段 `v2.1` 当前最新证据链：
  - `scripts/run_stage2_memory_canary.py` 现已支持增量落盘与 `--run-dir --resume`，新鲜 live canary 不再是“一次性无进度长挂起”
  - latest fresh current-head `PersonaMem 64` artifact 为 `outputs_v2/evals_benchmark/20260415T122009Z_stage2_memory_canary.json`
  - latest fresh current-head `LongMemEval-S 64` artifact 为 `outputs_v2/evals_benchmark/20260415T134213Z_stage2_memory_canary.json`
  - latest current-head `PersonaMem 128` artifact 为 `outputs_v2/evals_benchmark/20260415T135121Z_stage2_memory_canary.json`
  - latest layered LongMemEval-S analysis artifact 为 `outputs_v2/artifacts/latest_longmemeval_stage2_layered_analysis.json`
  - latest non-tiny train/eval artifact 为 `outputs_v2/runs/20260415T132046Z_stage2_train_exec/execution_summary.json` 与 `outputs_v2/evals_local/20260415T132109Z_stage2_local_eval.json`
  - `outputs_v2/artifacts/latest_stage2_learned_online_gain.json` 已形成正的 live-provider 对照：相对最早 `PersonaMem 64` live canary，retained current-head `PersonaMem 64` live canary 提升到 `delta_provider_exact_match = +23`、`delta_provider_label_prefix_match = +5`、`delta_local_exact_match = +16`
  - `PersonaMem 128` 当前统计为：`provider_label_prefix_match = 39 / 128 = 30.47%`、`local_exact_match = 26 / 128 = 20.31%`
  - `LongMemEval-S 64` 当前统计已跨过 verifier 阈值：`provider_exact_match = 5 / 64`、`provider_label_prefix_match = 5 / 64`、`local_exact_match = 3 / 64`
  - latest non-tiny trained eval 当前统计为：`trained_eval.token_f1 = 0.14615651550268283`
- 第二阶段 `v2.1` 历史 robustness 结论：上一轮 managed autoresearch 曾把 `stage2_v21_robustness_score` 推到 `14/15`。这条线为当前 repo 提供了更强的 canary、LongMemEval-S 提升和 non-tiny learned evidence，但它也暴露了 rule-heavy 路线的上界，因此不再作为当前主线终点。
- 第二阶段主线再次切换：用户当前已明确要求“更注重 learning model、不要继续依赖 rule-based 路径、目标是提升整体框架智能程度和更好的 latent”。因此现阶段主线应切换为 **learned-model-first / better latent**，并进入长期迭代阶段。
- 第二阶段 learned-memory-first 灵感来源：当前 repo 已明确参考以下 related work：
  - `End-To-End Memory Networks`
  - `Memorizing Transformers`
  - `RETRO`
  - `LongMem`
  - `Slot Attention`
  相关笔记见 [docs/learned_memory_related_work.md](/media/storage/mingjing/workspace/CoRe_Mem/docs/learned_memory_related_work.md)
- 第二阶段新的长跑机械目标：当前主指标切换为 `scripts/verify_stage2_v21_longrun.py` 对应的 `stage2_v21_longrun_score`。它重点检查：
  - 主线文档和 `.agent-os` 是否切到 `TD-030 / WS-016`
  - 是否形成了明确的长跑计划文档 [docs/v21_longrun_plan.md](/media/storage/mingjing/workspace/CoRe_Mem/docs/v21_longrun_plan.md)
  - 是否把 “不做任何兜底/fallback/benchmark-specific shortcut” 固化为硬约束
  - learned online gain、non-tiny trained eval、LongMemEval 分层分析是否仍成立
  - current-head learned-mode canary 是否从 `1` 样本扩大到 `64/128`
  - online system 是否已经摆脱 learned path 失败时退回 symbolic decoder 的 retained 设计
- 当前 baseline 已建立为 `11/16`。当前未过项集中在：
  - `online_system_avoids_symbolic_fallback = false`
  - `non_tiny_trained_eval.token_f1 >= 0.20 = false`
  - current-head learned-mode `PersonaMem 64 / LongMemEval-S 64 / PersonaMem 128` artifacts 尚未刷新
- 第二阶段 learned-model-first 当前证据链：
  - `StructuredMemorySystem` 已支持 `memory_mode=learned_memory` 与 checkpoint-backed belief path
  - `scripts/run_stage2_memory_canary.py` 已支持 `--memory-mode learned_memory --learned-memory-checkpoint-dir --learned-memory-train-config`
  - `outputs_v2/artifacts/latest_stage2_learned_online_gain.json` 仍提供正的 learned online gain 证据
  - `outputs_v2/artifacts/latest_longmemeval_stage2_layered_analysis.json` 仍提供跨层 failure analysis
  - 当前 non-tiny `trained_eval.token_f1` 历史 best 仍为 `0.14615651550268283`
- 第二阶段 learned-model-first 当前解释边界：
  - `TD-029` 解决的是 learned-mode 路径“已经存在并可调用”
  - `TD-030` 要解决的是 learned model 是否在更大样本和多个 benchmark 上真正变强，并且不依赖 fallback/shortcut
  - 当前代码中 learned path 仍保留 symbolic fallback，因此 `TD-030` baseline 预计不会满分
- 测试状态：当前长跑 guard 将继续以 stage-2 核心测试、`scripts/run_experiment.py --verify-only = 34`、`scripts/verify_stage2_latent_status.py --score-only = 9`、`scripts/verify_stage2_latent_core_quality.py --score-only = 10` 为底线；新增 `scripts/verify_stage2_v21_longrun.py` 当前 baseline 为 `11/16`

## 当前最重要的下一步

- 第一阶段 formal benchmark 继续保留为 pending baseline/acceptance 项；第二阶段 `TD-027` 已完成，因此 stage-2 当前主线正式切换到 `v2.1`。
- `TD-029` 的 learned-memory-first plumbing 已完成；当前 next action 切换为 `TD-030`，目标是围绕 learned model / better latent 做长期迭代：
  1. 把 learned-mode canary 从 `1` 样本扩大到 `64/128`
  2. 优先看 learned path 在 `LongMemEval-S` 上是否真实改善
  3. 让训练更直接服务在线 `memory -> belief -> answer`
  4. 逐步移除 learned online path 中的 fallback 依赖，而不是把 fallback 当 retained 设计

## 关键约束

- 真源文档为 `docs/requirements.md`
- 第一阶段与第二阶段都必须遵守真源中的分阶段边界
- `benchmark` 是 evaluation source，不是第二阶段 primary training source
- 第二阶段训练策略为 `public-datasets-first, synthetic-minimal`
- 第一阶段 v1 应尽量保留为 naive baseline 参考
- stage-2 当前是执行主线；stage-1 formal benchmark 不在未获用户指令时主动运行

## 风险与待定项

- 阿里云兼容接口当前仍返回 `HTTP 403 AllocationQuota.FreeTierOnly`
- 用户批准的 Gemini 路径已实测可用，但当前 key 在 formal benchmark 负载下连续触发 `HTTP 429`
- PersonaMem formal run 当前停在 `22/589`，LongMemEval-S formal run 当前停在 `19/500`
- 第二阶段 `DuLeMon / REALTALK / MS-TOD` 是否纳入主训练集仍待后续实现时根据可用性与清洗成本判断
- 第二阶段 `Optimus-like` 路线当前只锁定为 ablation / 潜在升级，不是主线
- `sentence-transformers` 依赖已经进入 `core_mem` 环境；stage-2 训练配置中的 Hugging Face cache 现已锁到 repo 内，但真正开始全量 `Flan-T5` 训练时仍会触发首次权重下载与较长训练时间
- 在当前 `datasets` 版本下，测试过的多个常见脚本型 HF dataset IDs 会返回 “dataset scripts are no longer supported”；当前已通过直接下载官方/作者源并自行规范化绕过该问题，但后续若继续扩展数据集，仍应优先采用 raw-source + normalization 路线
- 当前 `stage2_experiment_completion_score=13/13` 的历史证据对应 `configs/stage2_train_tiny.yaml` 在 `gpu3` 上的本地 train/eval/ablation matrix；本轮虽然已补上默认 `google/flan-t5-base` 的非 tiny train/eval artifact，但这仍只是最小正证据，不等于该默认 backbone 已 full train-complete
- 当前 stage-2 已不再停留在“decoder 不消费 composed latent”的 skeleton 状态，且 latent-core local intrinsic 指标已达 `10/10`；retained canary guard 现已对齐到 fresh PersonaMem completed artifact，而不是历史 blocked probe
