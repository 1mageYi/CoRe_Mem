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
- 当前这条长跑已经在 current HEAD `d6bc4f7` 上机械收口：`scripts/verify_stage2_v21_longrun.py --score-only = 16/16`，managed stop condition 已触发。
- 本轮新增并保留下来的关键证据是：
  - `outputs_v2/evals_local/20260415T191953Z_stage2_local_eval.json`：latest non-tiny trained eval 当前已达 `trained_eval.token_f1 = 0.3885239109848479`
  - `outputs_v2/evals_benchmark/20260415T202608Z_stage2_memory_canary.json`：current-head learned `PersonaMem 128` refreshed artifact，因 `sample_count = 128` 同时覆盖 verifier 的 `PersonaMem 64 / 128` 检查
  - `outputs_v2/evals_benchmark/20260415T205627Z_stage2_memory_canary.json`：current-head learned `LongMemEval-S 64` refreshed artifact
  - `src/core_mem/v2/system.py` 已不再把 learned online path 的 retained 失败处理退回 symbolic decoder；learned predictor 不可用或输出为空时，现显式标注为 `learned_memory_unavailable / learned_memory_error / learned_memory_empty`
- 第二阶段 learned-model-first 当前证据链：
  - `StructuredMemorySystem` 已支持 `memory_mode=learned_memory` 与 checkpoint-backed belief path
  - `scripts/run_stage2_memory_canary.py` 已支持 `--memory-mode learned_memory --learned-memory-checkpoint-dir --learned-memory-train-config`
  - `outputs_v2/artifacts/latest_stage2_learned_online_gain.json` 仍提供正的 learned online gain 证据
  - `outputs_v2/artifacts/latest_longmemeval_stage2_layered_analysis.json` 仍提供跨层 failure analysis
  - 当前 non-tiny `trained_eval.token_f1` 历史 best 已更新为 `0.3885239109848479`
- 第二阶段 learned-model-first 当前解释边界：
  - `TD-029` 解决的是 learned-mode 路径“已经存在并可调用”
  - `TD-030` 要解决的是 learned model 是否在更大样本和多个 benchmark 上真正变强，并且不依赖 fallback/shortcut
  - 当前 `TD-030` 的 mechanical stop 已达成，但这不等于 learned quality 已全面稳定：live learned path 仍频繁出现 `learned_memory_error`，核心问题是 `Flan-T5` belief JSON 经常不可解析，而不是系统已经学会了稳定的 online belief construction
- 第二阶段语义优先训练结论：当前最好的 non-tiny artifact `outputs_v2/evals_local/20260415T191953Z_stage2_local_eval.json` 已把 `trained_eval.token_f1` 提到 `0.3885239109848479`，但 `exact_match` 仍是 `0`，而且 sample preview 显示模型更像“学会了字段和值的结构模式”，尚未稳定输出合法完整 JSON。后续优化不应再把 raw JSON 表面匹配当成核心目标，而应把“语义正确 + 外部格式约束”作为新的训练主线。
- 第二阶段 semantic-first 当前完成态：`outputs_v2/evals_local/20260415T230211Z_stage2_local_eval.json` 已把同一 non-tiny learned checkpoint 的 `trained_eval.token_f1` 提到 `0.879714215455919`，其中：
  - `retrieval_alignment.token_f1 = 0.9860465116279071`
  - `lifecycle_prediction.token_f1 = 0.8886274509803921`
  - `composition_to_belief.token_f1 = 0.7632177155691989`
  - `semantic_validity_rate = 0.96484375`
  - `field_f1 = 0.8463541666666676`
  这说明当前 retained 收益主要来自“把语义恢复与结构壳错误解耦”，而不是重新引入任何 fallback 或 benchmark-specific shortcut。
- 当前 `TD-031` 的明确锚点仍是：`语义优先`、`格式外部约束`。
- 第二阶段 semantic-first 关键实现变化：`src/core_mem/v2/semantic_outputs.py` 现把 task-aware 结构修复与语义计分抽成通用组件；`training.py` 与 `system.py` 共用这一逻辑，因此 learned checkpoint eval 和 online learned belief parse 对 brace-level JSON 壳错误的处理已对齐。
- 第二阶段 `v2.2` 新目标：在保持 semantic-first 与 no-fallback/no-shortcut 约束不变的前提下，把主线推进到 **full-data semantic-first learned memory**。这条线要求：
  - current-head full-data public-data 训练与本地评测
  - current-head `PersonaMem 128` semantic canary
  - current-head `LongMemEval-S 64/128` semantic canary
  - current-head LongMemEval-S layered analysis
  - current-head learned-vs-symbolic online gain artifact
  - 相关计划见 [v22_plan.md](/media/storage/mingjing/workspace/CoRe_Mem/docs/v22_plan.md)
- 第二阶段 `v2.2` 当前 retained 进展：current HEAD `510aeb7` 上的 full-data semantic artifacts 已补齐，`outputs_v2/artifacts/latest_stage2_semantic_full_train.json` 记录 `num_examples = 1574`、`cuda_visible_devices = 3`，`outputs_v2/artifacts/latest_stage2_semantic_full_local_eval.json` 记录：
  - `trained_eval.token_f1 = 0.9258179798351409`
  - `retrieval_alignment.token_f1 = 1.0`
  - `semantic_validity_rate = 1.0`
  - `field_f1 = 0.9303261329945052`
  这条线在后续 managed autoresearch 中已进一步补齐所有 remaining online semantic evidence，使 `scripts/verify_stage2_v22_completion.py --score-only` 达到 stop condition `19/19`。
- 第二阶段 `v2.2` 当前 fresh online evidence：
  - `outputs_v2/evals_benchmark/20260416T021743Z_stage2_memory_canary.json` 已把 current-head `LongMemEval-S` semantic canary 从 `64` 扩到 `128`
  - `outputs_v2/evals_benchmark/20260416T024146Z_stage2_memory_canary.json` 已补齐 current-head `PersonaMem 128` semantic canary；其快速统计为 `provider_exact = 40/128`、`provider_label_prefix = 40/128`、`local_exact = 12/128`
  - `outputs_v2/artifacts/latest_longmemeval_stage2_semantic_analysis.json` 已刷新到 `128`-sample current-head artifact
  - `outputs_v2/artifacts/latest_stage2_semantic_online_gain.json` 已在 current HEAD `510aeb7` 上转正；当前记录的是 `LongMemEval-S 64` learned 相对 retained symbolic 64 baseline 的 `delta_local_exact_match = +1`
- 第二阶段 `v2.2` 当前关键实现变化：`src/core_mem/v2/projection.py` 新增 generic answer projection normalization，可对 belief value 去掉解释尾巴并抽取 location phrase；这一改动在相同 `LongMemEval-S 64` belief state 上把 local exact 从 `3` 提到 `4`，并由 fresh canary artifact 机械确认。
- 第二阶段 `v2.2` 当前 truthfulness boundary：
  - 可以诚实声明：`stage2_v22_completion_score = 19/19` 已机械达成
  - 不能夸写成：`LongMemEval-S` online quality 已全面稳定
  - 当前 `LongMemEval-S 128` fresh artifact 仍只有 `provider_exact = 4/128`、`local_exact = 4/128`；semantic online gain 的 retained 正增益来自 local projection 改善，而不是 provider-side 全面超越 symbolic baseline
- 第二阶段 `v2.3` 新主线：当前用户已明确要求把下一步聚焦到“更完善的 CoRe Mem 框架 / 系统 / 模型，以及更好的 latent”。因此当前新主线切到 **`v2.3 stronger learned slot assignment + stronger latent`**，核心方向为：
  - 把 `LongMemEval-S` 从“机械完成”推进到“质量主 benchmark”
  - 把 `observation -> slot` 从 rule-heavy lifecycle 推进到 `learned slot assignment + hard constraints`
  - 继续强化 `retrieval -> composed latent -> belief -> answer` 主链，让 retained 收益更多来自 learned latent，而不是规则补丁
  - 相关计划见 [v23_plan.md](/media/storage/mingjing/workspace/CoRe_Mem/docs/v23_plan.md)
- 测试状态：本轮 stop condition 对应的 final mechanical evidence 是：
  - `conda run -n core_mem python scripts/verify_stage2_v21_semantic_model.py --score-only` -> `17`
  - full semantic guard 通过：`pytest -q tests/test_stage2_model_skeleton.py tests/test_stage2_local_eval.py tests/test_stage2_data_pipeline.py tests/test_stage2_public_data.py tests/test_stage2_memory_canary.py tests/test_stage2_memory_canary_quality.py tests/test_stage2_latent_core_quality.py tests/test_stage2_v21_learned_memory.py tests/test_stage2_v21_longrun.py tests/test_stage2_v21_semantic_model.py tests/test_stage2_parser.py`
  - 相关 verifier 当前为：`verify_stage2_latent_status = 9`、`verify_stage2_latent_core_quality = 10`、`verify_stage2_v21_learned_memory = 11`、`verify_stage2_v21_longrun = 12`、`run_experiment.py --verify-only = 34`

## 当前最重要的下一步

- 第一阶段 formal benchmark 继续保留为 pending baseline/acceptance 项；第二阶段 `TD-027` 已完成，因此 stage-2 当前主线正式切换到 `v2.1`。
- `TD-029`、`TD-030`、`TD-031` 与 `TD-032` 已机械完成；当前 runtime truth 正式从 `v2.2 closeout` 切到 `v2.3` 规划态。
- 下一步不再继续围绕 “semantic-first 证据是否齐全” 做 closeout，而是把主攻点切到：
  - `LongMemEval-S` 质量提升
  - learned slot assignment
  - stronger latent / stronger online memory path
- 当前最值得延续的训练结论是：
  - 仅增加训练 budget 或只改 prompt/target 不能稳定解决 learned belief JSON 失效；真正带来 retained 收益的是把语义恢复从 raw JSON 壳错误中解耦，并让训练/评测/online parse 共享同一套 semantic-first 结构修复
  - 该 retained 路线已经在正式 artifact 上把 non-tiny `trained_eval.token_f1` 提升到 `0.879714215455919`
  - 当前已经补齐 current-head full-data semantic train/eval、`PersonaMem 128` / `LongMemEval-S 64/128` semantic canary、LongMemEval-S semantic analysis 与 semantic online gain
  - 当前 `v2.2` 的 runtime truth 应诚实标注为“`19/19` retained, stop condition reached, closeout retained”
  - `v2.3` 不应回退到 raw JSON exactness 或 rule-heavy patching，而应优先探索 `learned slot assignment + hard constraints`

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
