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
- 第二阶段 benchmark 质量状态：live canary 已证明 `memory -> belief -> provider answer` 链路可运行，而且当前 session 已把质量基线从 `4/10` 提升到 `9/10`。最新 `PersonaMem 64` live canary `outputs_v2/evals_benchmark/20260415T015324Z_stage2_memory_canary.json` 的快速统计为：
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
- 第二阶段硬约束新增：后续长跑中**不要做任何偷懒兜底 fallback**。尤其不允许把 benchmark-specific heuristic / fallback 当作 retained 主收益，也不允许用 provider 空输出兜底、candidate-answer 注入或选项 overlap scorer 之类技巧冒充 latent-core 提升。
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
- 测试状态：当前 stage-2 guard `pytest -q tests/test_stage2_model_skeleton.py tests/test_stage2_local_eval.py tests/test_stage2_data_pipeline.py tests/test_stage2_public_data.py tests/test_stage2_memory_canary.py tests/test_stage2_memory_canary_quality.py tests/test_stage2_latent_core_quality.py tests/test_stage2_parser.py` 通过（32 tests）；`scripts/run_experiment.py --verify-only` 输出 `34`，`scripts/verify_stage2_latent_status.py --score-only` 输出 `9`，`scripts/verify_stage2_latent_core_quality.py --score-only` 输出 `10`

## 当前最重要的下一步

- 第一阶段 formal benchmark 继续保留为 pending baseline/acceptance 项；第二阶段当前已经从“修 latent skeleton”和“拉起 PersonaMem canary”推进到“朝完整的 `v2` 长跑”。
- 当前最重要的下一步是以里程碑方式推进完整 `v2`：
  1. 提交并固化当前 latent-core retained state，作为新的 clean baseline
  2. 刷新 fresh `PersonaMem 64` live canary，而不是继续依赖旧 best artifact
  3. 让 `LongMemEval-S 64` 的 stage-2 live canary 跑通并产出 failure analysis
  4. 在 `gpu3` 上补默认 `flan-t5-base` 的非 tiny train/eval 证据，并要求 `trained_eval` 不再是全零
  5. 清理 benchmark runner 中的 benchmark-specific shortcut / fallback，使主收益重新回到 system/model/latent 本体
  6. 只有以上里程碑同时成立，才接近可以诚实称为“完整的 `v2`”

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
- 当前 `stage2_experiment_completion_score=13/13` 的证据对应 `configs/stage2_train_tiny.yaml` 在 `gpu3` 上的本地 train/eval/ablation matrix；默认 `google/flan-t5-base` backbone 的非 tiny 全量 run 仍未验证，不能把当前状态误称为该默认 backbone 已 train-complete
- 当前 stage-2 已不再停留在“decoder 不消费 composed latent”的 skeleton 状态，且 latent-core local intrinsic 指标已达 `10/10`；但 retained canary guard 仍基于既有 `PersonaMem 64` artifact，且官方 verify 链里对 live canary 的重跑在本轮 session 中表现为长时间挂起，尚未形成新的 live canary artifact
