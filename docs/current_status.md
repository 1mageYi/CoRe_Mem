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
- 第二阶段 benchmark canary 状态：`scripts/run_stage2_memory_canary.py` 已落地，并已在 `outputs_v2/evals_benchmark/20260414T170606Z_stage2_memory_canary.json` 产出一条 `PersonaMem` memory-mediated canary artifact；由于当前 session 缺少 `GPT_AGENT_API_KEY`，该 artifact 状态为 `blocked_provider_not_configured`，而不是 live MiniMax-M2.7 result
- 环境状态：repo 内 project-local conda env 与缓存痕迹已清理；环境复现真相为 `environment.yaml` + 默认 conda named env `core_mem`
- 测试状态：当前 stage-2 guard `pytest -q tests/test_stage2_model_skeleton.py tests/test_stage2_training_runtime.py tests/test_stage2_local_eval.py tests/test_stage2_data_pipeline.py tests/test_stage2_public_data.py` 通过（21 tests）；新增 `tests/test_stage2_memory_canary.py` 也已通过；`scripts/run_experiment.py --verify-only` 输出 `34`，`scripts/verify_stage1_acceptance.py` 返回 `5/7`，`scripts/verify_stage2_status.py --score-only` 输出 `50`，`scripts/verify_stage2_acceptance.py` 返回 `7/7`，`scripts/verify_stage2_experiment_status.py --score-only` 输出 `13`，`scripts/verify_stage2_latent_status.py --score-only` 输出 `9`

## 当前最重要的下一步

- 第一阶段 formal benchmark 继续保留为 pending baseline/acceptance 项；第二阶段当前最重要的下一步已经从“继续补 local eval 接口 / canary 记录”切换为“先把真正的 latent memory 主链路做实”。
- 具体来说，下一步优先级应为：
  1. 在 provider key 可用时，把 `scripts/run_stage2_memory_canary.py` 从当前 blocked artifact 推进到 live MiniMax-M2.7 canary result
  2. 视用户优先级决定是否继续补默认 `flan-t5-base` 非 tiny `gpu3` 训练证据
  3. 在上述条件满足后，再决定是否扩大 stage-2 benchmark canary / scaling

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
- 当前 stage-2 已不再停留在“decoder 不消费 composed latent”的 skeleton 状态，但 `MiniMax-M2.7` live canary 仍受 `GPT_AGENT_API_KEY` 缺失约束；因此当前 benchmark 侧证据仍是“runner + blocked artifact 已成立”，不是 live provider benchmark passed
