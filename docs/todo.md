# TODO

## Doing

- `TD-044` 以 `v3.3 / v33 learned-authoritative latent run` 为目标，在**不改 `core / residual` 双银行结构**的前提下，基于 retained `v32` baseline，继续推进真正更强的 learned latent system。
  - 当前锚点：`32k` source-level split（`24k train / 4k val / 4k test`）
  - 当前约束：不做任何 `fallback / shortcut / benchmark-specific heuristic / benchmark leakage`
  - 当前重点：
    - learned authoritative runtime
    - competition-based learned write
    - temporal-semantic latent reader
    - belief graph head
    - answer / option head
    - full benchmark holdout：`LongMemEval-S 500 / PersonaMem 512`
  - 最新真相：
    - retained `v32` 已到 `44/44 keep`
    - current retained `v33` 已到 `36/47 keep`
    - `latest_stage2_v33_modular_authoritative_train.json`、`latest_stage2_v33_learned_write_eval.json`、`latest_stage2_v33_latent_reader_train.json`、`latest_stage2_v33_temporal_slot_eval.json`、`latest_stage2_v33_latent_objective_eval.json`、`latest_stage2_v33_belief_graph_eval.json`、`latest_stage2_v33_answer_option_eval.json` 与 `latest_stage2_v33_ablation_summary.json` 已落地
    - 当前 verifier 剩余缺口只在 learned-authoritative runtime 与 full holdout compare
    - `v32` 已证明 modular latent / belief / answer 设计可以带来 full-holdout gain
    - 但 `v32` authoritative full benchmark path 仍是 `memory_mode = symbolic`、`slot_assignment_mode = symbolic`
    - `v33` 的目标不是继续扩大 symbolic baseline，而是让 learned path 接管 authoritative runtime
    - semantic-full checkpoint + `v31` latent ranker 当前是最强 learned runtime 候选：`LongMemEval-S 64` 为 `10/10`，`PersonaMem 64` 为 `21/24`
    - 但 `PersonaMem 64` provider rate 仍低于 retained `v32`；而且当前 `https://gpt-agent.cc/v1` 代理不兑现 MiniMax 官方 `reasoning_split=True` 行为，真实 failure prompt 仍直接返回 `<think>` 污染内容
    - current HEAD `2d3e59c` 已把 provider raw-output repair 接进 runner / resume path；现有 partial Persona learned-authoritative full-holdout run `outputs_v2/v33_semantic_full_persona/runs/20260420T203544Z_stage2_memory_canary_personamem/` 已从旧 summary 的 `provider exact = 5/66` 回收到当前 `15/77`
    - 同一 run 继续扩到 `99/512` 后，provider/local 为 `18/99`、`42/99`；对应 layered analysis 已确认当前剩余 Persona failures 中 `projection = 54`、`provider = 27`，并且 `other_fact` 主导 `70` 个失败 relation
    - current HEAD 本轮又试探了 grounded same-relation belief backfill，但 isolated Persona partial probe `outputs_v2/v33_grounded_projection_local/` 前 `4` 条样本对 retained `v33_semantic64_persona` 仍是 `changed = 0`
    - 本轮随后又试探了 generic query-stopword 扩展与 relation-consistent support-slot reassignment：前者对代表性 failures replay 完全无效，后者虽能把 `acd742...` / `5370...` 的 belief/support 清成 relation-consistent `music_preference`，但 targeted Persona `4`-sample gate `outputs_v2/v33_support_consistency_targeted/` 仍是 `local_exact = 0/4`
    - current HEAD `3234a4b` 已把 option scorer 收紧成 generic “support-density / unsupported-detail” ranking；clean targeted Persona `4`-sample gate `outputs_v2/v33_option_penalty_targeted_clean/` 已从 `0/4` 提到 `1/4`
    - 本轮新证伪的 write-path 假设是：即使对 `*_preference` 这类多 facet relation 在 symbolic=`new` 时直接 short-circuit learned slot-assignment，保住多条 active `music_preference` 槽位，targeted Persona `4`-sample gate 也仍是 `1/4`，没有超过 retained option-scoring 分支
    - current HEAD `65c2612` 又把 software-centric music clauses 规范成稳定 `music_preference` facet，并保留 symbolic=`new` 的 preference-facet fast-path；对应 clean targeted Persona `4`-sample gate `outputs_v2/v33_music_software_facet_local/artifacts/20260420T231900Z_compare.json` 已从 retained `1/4` 提到 `2/4`，其中 `5370...` 从错答翻正，但 `acd742...` / `a40d5...` 仍未解决
    - current HEAD `2b54785` 又把 parser 里过宽的 `from`-location 规则收紧；单样本 probe `outputs_v2/v33_music_software_facet_local/artifacts/20260420T233200Z_5370_relation_probe.json` 显示 `5370...` 在保持正确 `(b)` 的同时，belief relation 已从错误的 `location` 修回 `music_preference`
    - current HEAD `883d48d` 又新增了 generic `feedback reason` parser coverage；对应 clean targeted Persona `4`-sample gate `outputs_v2/v33_music_software_facet_local/artifacts/20260420T234300Z_compare.json` 已进一步从 retained `1/4` 提到 `3/4`，其中 `a40d5...` 从 `(c)` 翻正到 `(d)`，当前只剩 `acd742...` 未解决
    - current HEAD `eade2c3` 又新增了 restricted same-relation latent rerank 与 invalid-relation belief-coercion 修复；单样本 probe `outputs_v2/v33_latent_facet_rerank_local/artifacts/20260421T004426Z_acd742_compare.json` 已把 `acd742...` 从 `(b)` 翻正到 `(c)`，belief 也稳定落成 `music_preference=producing music with software`
    - latest non-regression probe `outputs_v2/v33_latent_facet_rerank_local/artifacts/20260421T005228Z_hard3_compare.json` 又机械确认剩余 `5370... / a40d5... / c8a763...` 三条样本都未回退，因此 current HEAD 已把 actual hard-4 gate 提回 `4/4`
    - corrected local-only Persona `smoke8` `outputs_v2/v33_latent_facet_rerank_smoke8_localonly/evals_benchmark/20260421T010549Z_stage2_memory_canary.json` 已给出 `local_exact = 5/8`、`baseline = 1/8`，说明 `eade2c3` 的 gain 已经开始从 hard-4 向更大的 Persona slice 外扩
    - partial local-only `smoke16` 在主动截断前也已给出 `local_exact = 6/6`、`baseline = 1/6`，说明这条 gain 没有在更大 slice 上立刻塌掉；current HEAD `49e1841` 又把其中 `b358...` 这类 `suggest_new_ideas` 样本暴露出的 `belief=true` boolean-like noise 修成 support-slot canonical value，说明这条 projection / belief noise 线确实值得保留
    - fresh partial `suggest_new_ideas` local-only slice `outputs_v2/v33_boolean_belief_suggest8_localonly/artifacts/20260421T021500Z_partial_summary.json` 又给出 `local = 4/4`、`baseline = 0/4`、`boolean-like belief rows = 0/4`；当前新问题已不再是 “boolean-backfill 有没有价值”，而是它是否只对前几个样本有效
    - second focused `suggest_new_ideas` slice `outputs_v2/v33_boolean_belief_suggest4b_localonly/evals_benchmark/20260421T021706Z_stage2_memory_canary.json` 已完整给出 `local = 4/4`、`baseline = 0/4`；对应 artifact `outputs_v2/v33_boolean_belief_suggest4b_localonly/artifacts/20260421T022100Z_boolean_check.json` 也确认 `boolean-like belief rows = 0/4`
    - fresh mixed Persona local-only gate `outputs_v2/v33_boolean_backfill_smoke8_localonly/artifacts/20260421T023400Z_partial_summary.json` 在与旧 `eade2c3` smoke8 相同 manifest 上，主动截断前已给出 `local = 4/4`、`baseline = 1/4`、`boolean-like belief rows = 0/4`
    - current HEAD `18f1f53` 又试探了 generic withdrawal parser coverage，synthetic parser tests 与 targeted local projection test 都能通过，但 fresh real `d71...` local-only probe `outputs_v2/v33_withdrawal_probe_d71/evals_benchmark/20260421T040035Z_stage2_memory_canary.json` 仍给出 `local_exact = 0/1`，belief 继续是 `other_fact=assisting with literacy programs`
    - fresh discard row `35` 又试探了 advice-query option-head reweighting：`0d225...` fresh local-only probe `outputs_v2/v33_advice_probe_0d/evals_benchmark/20260421T042820Z_stage2_memory_canary.json` 已翻到 `local_exact = 1/1`，`b449...` 仍保持 `1/1`，但 `d71...` probe `outputs_v2/v33_advice_probe_d71/evals_benchmark/20260421T042841Z_stage2_memory_canary.json` 仍是 `0/1`，而 official verifier 继续停在 `36`，所以整条 answer-head patch 已回滚
    - current HEAD `cd1205f` 又把 `configs/minimax_m27.yaml` 的 stage-2 runtime defaults 接进 canary runner；现在即使 CLI 不显式传 checkpoint args，learned-memory / latent / learned slot-assignment 也会默认加载 `v33` 当前 intended checkpoints。对应 full guard 已通过，fresh learned probe `outputs_v2/v33_advice_probe_0d/evals_benchmark/20260421T043908Z_stage2_memory_canary.json` 也已机械记录 non-null checkpoint dirs
    - current HEAD `3fa4d06` 又把 activity-withdrawal parser coverage 保留到 corrected runtime；fresh real `d71...` probe `outputs_v2/v33_withdrawal_probe_d71/evals_benchmark/20260421T045624Z_stage2_memory_canary.json` 继续是 `local_exact = 0/1`，但 active memory 已新增 `other_fact=step back from structured book club settings`，说明当前 `d71` 的主瓶颈已经从 parser/write 缺口收紧到 selected-slot ranking / belief selection
    - current HEAD `2cd6c9b` 又把 withdrawal clause 编成 `recent_change` advice evidence，并把 advice-query zero-overlap negatives 改成 `score-first` 排序；fresh `d71...` probe `outputs_v2/v33_withdrawal_probe_d71/evals_benchmark/20260421T051757Z_stage2_memory_canary.json` 已让 `cdc8...` 进入 selected top-k，但 belief 仍未消费它
    - current HEAD `c4e71d1` 又把 weak `other_fact` latent facet promotion 收紧到 score-proximity gate；fresh `d71...` probe `outputs_v2/v33_withdrawal_probe_d71/evals_benchmark/20260421T052743Z_stage2_memory_canary.json` 已让 selected[0] 与 learned belief 一起落成 `step back from structured book club settings`，但 local answer 仍是 `(a)`，说明当前主剩余问题已收敛到 answer-option projection
    - current HEAD `12e676a` 又把 `back / other / think / thoughts` 一起压成 option-head low-info overlap tokens；fresh probes `outputs_v2/v33_withdrawal_probe_d71/evals_benchmark/20260421T054201Z_stage2_memory_canary.json` 与 `outputs_v2/v33_advice_probe_b449/evals_benchmark/20260421T054201Z_stage2_memory_canary.json` 已分别给出 `1/1` 与 `1/1`，而 `outputs_v2/v33_advice_probe_0d/evals_benchmark/20260421T054201Z_stage2_memory_canary.json` 仍是 `0/1`
    - 当前 worktree 又撤回了 `3234a4b` 的 `support-density / unsupported-detail / option_length` 惩罚，让 `OptionScoringHead` 回到 v32 overlap-style ranking；新增 targeted regressions 已直接覆盖 `0d...` 与 `2cef...` 两类 v33 answer-head 回退
    - 对应 full guard 已重新通过，但 `scripts/verify_stage2_v33_longrun.py --score-only` 仍是 `36`；与此同时，current commit `f9ebc8b` 的 fresh `smoke8` local-only replay `outputs_v2/v33_answer_head_overlap_smoke8_localonly/evals_benchmark/20260421T062832Z_stage2_memory_canary.json` 已完整收口到 `local_exact = 8/8`、`baseline = 1/8`，并且样本级确认 `2cef...` 与 `0d...` 都已翻正。随后 current worktree 又为 `OptionScoringHead` 加入 generic morphology normalization；fresh broader gate `outputs_v2/v33_answer_head_stem_smoke16_localonly/evals_benchmark/20260421T071739Z_stage2_memory_canary.json` 已把 `local_exact` 从旧的 `12/16` 提到 `14/16`、`baseline = 2/16`，并翻正 `5370...` 与 `32b...`
  - Truth boundary：
    - 当前主矛盾已不再是 v32 能否涨分，而是 learned path 能否接管 authoritative full-holdout runtime
    - 这轮不允许在 symbolic authoritative path 上收口
    - 当前已批准的新路线是：保留同一 provider/interface，只做协议层最小 `<think>` 清洗；`8`-sample 不再决定方向，retain 直接看 `LongMemEval-S 500 / PersonaMem 512`
    - 当前真实状态是 active measurement / refine，而不是“必须先切 provider/interface 才能继续”的硬 blocker；但修复后的 Persona provider exact 仍明显低于 retained `v32`。最新 runtime truth 是：withdrawal line 已经从 retrieval / belief miss 推到 option projection，而当前 broader local-only gate 已机械证明 answer-head 仍有可保留的 generic gain：`smoke16` 从 `12/16` 提到 `14/16`。因此下一步不再是“先验证 broader gain 是否存在”，而是直接分析剩余 `344ea... / 0adf...` 两条 residual projection miss，到底应继续做 answer-head 的 generic lexical/semantic refine，还是回到 belief 粒度本身仍过粗这一层

- `TD-042` 以 `v3.1 / v31 latent-first quality run` 为目标，在**不改 `core / residual` 双银行结构**的前提下，基于 retained `v30` baseline 继续推进 full holdout 上真正更强的 learned 主链。
  - 当前锚点：`32k` source-level split（`24k train / 4k val / 4k test`）
  - 当前约束：不做任何 `fallback / shortcut / benchmark-specific heuristic / benchmark leakage`
  - 当前重点：
    - latent strengthening
    - belief strengthening
    - write strengthening
    - ablation truth
    - full benchmark holdout：`LongMemEval-S 500 / PersonaMem 512`
  - 最新真相：
    - retained `v30` 已到 `41/41 keep`
    - current retained `v31` 仍是 `24/32`
    - aligned `32k val` latent compare、belief compare 与 write compare 当前均为正，但仍只代表 apples-to-apples internal compare 为正
    - commit `489ada0` 已让 learned full-holdout runner 复用 `latent_slot_ranker` 并按 batch 增量落盘；此前停在 `completed_predictions = 0` 的 resumed runs 现已恢复推进
    - current HEAD `838a861` 又补齐了 `v31` full-holdout publisher；对应 targeted test 与 configured guard 已通过
    - 截至当前检查：`LongMemEval-S 500` resumed run 已推进到 `322/500`，`PersonaMem 512` resumed run 已推进到 `238/512`
  - Truth boundary：
    - 当前这条线已经 soft-blocked；不能再把它写成“等待一下就会 closeout”
    - 当前最诚实的结论是：`LongMemEval-S 500` parity、`PersonaMem 512` provider guard fail、三类 Persona strategy pivots 均无 keep

- `TD-038` 以 `v2.7 32k teacher-first long-run` 为目标，在**不改 `core / residual` 双银行结构**的前提下，先建立 `32k` source-level split、teacher-labeled data-quality upgrade、internal generalization test 与 `gpu2` 训练耗时基线。
  - 当前锚点：`24k train / 4k val / 4k test`
  - 当前 teacher：`MiniMax-M2.7`
  - 当前约束：不做任何 `fallback / shortcut / benchmark-specific heuristic / benchmark leakage`
  - 当前训练要求：优先使用 `gpu2`，并记录 wall-clock / throughput / memory
  - Truth boundary：先只做 `32k`；只有 internal test work well 后，才允许进入 full-data
  - 最新进展：`latest_stage2_v27_32k_split.json`、`latest_stage2_v27_32k_manifest.json`、`latest_stage2_v27_32k_audit.json`、`latest_stage2_v27_train.json`、`latest_stage2_v27_eval.json`、`latest_stage2_v27_training_timing.json`、`latest_stage2_v27_internal_test.json`、`latest_stage2_v27_holdout_summary.json`、`latest_stage2_v27_teacher_observation.json`、`latest_stage2_v27_teacher_slot_assignment.json` 与 `latest_stage2_v27_teacher_belief.json` 已落地；fresh current-head `scripts/verify_stage2_v27_longrun.py --score-only = 26`
  - 当前 teacher truth：本轮使用真实 `MiniMax-M2.7` 做了一次 sample-capped teacher pilot，caps 为 `8/2/2`、batch size 为 `1`；其中 `slot_assignment` 与 `belief` artifact 为 `completed`，`observation` artifact 为 `completed_with_failures`
  - Truth boundary：当前 `26/26` 是由 `32k split + gpu2 tiny pilot + teacher pilot artifacts` 共同满足的机械 stop condition，不代表 full `32k` teacher coverage 已完成，也不代表 teacher-conditioned `gpu2` retrain 已完成

- `TD-036` 以 `v2.5 learned core-path long-run` 为目标，在**不改 `core / residual` 双银行结构**的前提下，继续推进 `LongMemEval-S` 质量、`write / retrieve / belief` 三段的 learned 化、learned slot assignment 泛化鲁棒性、更强 latent 与 full benchmark holdout evaluation。
  - 当前起点：`TD-035 / WS-021` 已在 current HEAD `12a9a80` 上达到 `24/24`
  - 当前约束：不做任何 `fallback / shortcut / benchmark-specific heuristic`
  - 当前 managed run 状态：current HEAD `3036e3d` 已把 `stage2_v25_longrun_score` 推到 `24/24`
  - 当前边界：这轮 `24/24` 属于 baseline/package closeout；`write / retrieve / belief` 三段 artifact 当前仍显式记录 `delta_vs_v24_retained = 0`
  - 当前计划：见 [v25_plan.md](/media/storage/mingjing/workspace/CoRe_Mem/docs/v25_plan.md)

## Blocked

- `TD-012` 补齐 stage-1 外部前置条件：当前 formal benchmark 继续受 provider `HTTP 429` 阻断，且在用户明确要求前不主动继续推进。
- `TD-013` 将 PersonaMem / LongMemEval-S 从 1-sample real run 推进到正式全量 protocol 运行：当前作为 pending baseline / acceptance 项保留，直到用户要求 AI 去跑。
## Backlog

- `TD-037` 以 `v2.6 gain-first long-run` 为目标，在**不改 `core / residual` 双银行结构**的前提下，要求 `write / retrieve / belief` 至少一段出现真实正增益，并要求 `LongMemEval-S 128` 明确超过 `v2.5` retained baseline。
  - 说明：当前已在 current HEAD `cfbdc08` 上完成，`scripts/verify_stage2_v26_longrun.py --score-only = 26`

- `TD-007` 为 PersonaMem 128k / 1M 预留启动开关。
- `TD-018` 建立第二阶段 benchmark canary protocol 与结果记录。
- `TD-025` 在 `PersonaMem 64` canary 质量提升后，再扩大 stage-2 benchmark 范围与对比表。
  - 说明：`TD-025` 现视为 `TD-028 / v2.1` 的子目标之一，而不再是独立的“完整 v2 之后随手做做”项。

- `TD-030` 以 learned model / better latent 为锚点启动 `v2.1` 长跑。
  - 说明：当前机械 stop condition 已在 HEAD `d6bc4f7` 上达到 `16/16`；此项历史里程碑已完成，但不再是当前主线。

- `TD-031` 在不引入 fallback/shortcut 的前提下，把 stage-2 learned training 切到“语义优先、格式外部约束处理”主线。
  - 说明：当前机械 stop condition 已达到 `17/17`；此项历史里程碑已完成，但不再是当前主线。

- `TD-032` 以 full-data semantic-first learned memory 为目标推进 `v2.2`。
  - 说明：当前机械 stop condition 已达到 `19/19`；此项 closeout 已完成，但不再是当前主线。

## Done

- `TD-040` 以 `v2.9 learned-core-path long-run` 为目标，在**不改 `core / residual` 双银行结构**的前提下，继续以 `32k` source-level split 为锚点，依次推进 `write -> latent composition -> belief` 三段 learned 主链，并把 holdout benchmark 扩到 `LongMemEval-S 500 / PersonaMem 512`。
  - 当前结果：`scripts/verify_stage2_v29_longrun.py --score-only = 39`
  - 关键证据：`latest_stage2_v29_{write,latent,belief}_gain.json` 三段全为正，`latest_stage2_v29_holdout_summary.json` 已记录 `LongMemEval-S 500 / PersonaMem 512`，`research-results.tsv` / `autoresearch-state.json` 已记录 iteration `3 keep`

- `TD-040` 以 `v2.9 learned-core-path long-run` 为目标，在**不改 `core / residual` 双银行结构**的前提下，继续以 `32k` source-level split 为锚点，依次推进 `write -> latent composition -> belief` 三段 learned 主链，并把 holdout benchmark 扩到 `LongMemEval-S 500 / PersonaMem 512`。
  - 当前结果：`scripts/verify_stage2_v29_longrun.py --score-only = 39`
  - 关键证据：`latest_stage2_v29_{write,latent,belief}_gain.json` 三段全为正，`latest_stage2_v29_holdout_summary.json` 已记录 `LongMemEval-S 500 / PersonaMem 512`，`research-results.tsv` / `autoresearch-state.json` 已记录 iteration `3 keep`


- `TD-001` 初始化项目文档系统与根契约文件。
- `TD-002` 在默认 conda envs 目录创建并固定 conda 环境 `core_mem`（Python 3.10）。
- `TD-003` 建立基础目录结构、配置机制与输出目录规范。
- `TD-004` 确定第一阶段 pretrained embedding model 为 `sentence-transformers/all-MiniLM-L6-v2`。
- `TD-005` 实现阿里云 OpenAI-compatible provider adapter 的第一版。
- `TD-006` 接入 PersonaMem 32k 官方主任务与评测协议。
- `TD-008` 接入 LongMemEval-S 官方评测协议。
- `TD-009` 实现 vanilla CoRe Memory 核心模块与文本 embedding 接口。
- `TD-010` 建立实验总入口与分步骤脚本。
- `TD-011` 建立并扩展 unit tests 与最小 E2E smoke test。
- `TD-014` 锁定第二阶段 `V2.0 structured latent-slot memory` 方案，并将真源、设计文档和状态文档同步到 stage-2 真相。
- `TD-015` 搭建第二阶段 `V2.0` 的 observation normalization、belief JSON 目标生成、parser skeleton 与公开数据集接入骨架。
- `TD-016` 实现第二阶段 `V2.0` 主线模型骨架，并打通 direct-train 链路。
- `TD-019` 获取并接入第二阶段真实公开数据源文件。
- `TD-017` 建立第二阶段本地 intrinsic evaluation 管线、budget sweep 与 experiment registry，并在 `gpu3` 上完成 `mainline + 11` 个必做 ablation 的 tiny-backend 本地 train/eval 登记。
- `TD-021` 把第二阶段真正的 latent memory 主链路做实。
- `TD-022` 建立 stage-2 memory-mediated benchmark canary runner，并接入可用 provider 配置。
- `TD-023` 完成 stage-2 MiniMax memory canary 的 live provider 首次运行。
- `TD-026` 以系统/模型/latent 本体更强、更稳健为锚点，提升 stage-2 local intrinsic 质量，并把 `PersonaMem 64 >= 9/10` 作为不退化 guard。
- `TD-020` 为默认 `flan-t5-base` backbone 补非 tiny `gpu3` 训练证据。
- `TD-027` 推进完整的 `v2`：fresh live canary、第二 benchmark、非 tiny 训练证据，以及去除 benchmark shortcut/fallback。
- `TD-028` 把 `v2.1` 作为当前主线：提升真实质量、learned path 实际贡献、跨 benchmark 鲁棒性与系统化收口。
  - 当前结果：`scripts/verify_stage2_v21_robustness.py --score-only = 14/15`
  - 关键证据：fresh current-head `LongMemEval-S 64` 已提升到 `provider 5 / local 3`，current-head `PersonaMem 128` 已重新落地并保住 `local_exact_rate >= 0.20`，`latest_stage2_learned_online_gain.json` 已转正，且 `outputs_v2/evals_local/20260415T132109Z_stage2_local_eval.json` 对应的 `GPU3 non-tiny flan-t5-base train/eval` 仍成立
- `TD-029` 把 `v2.1` 的当前主线切到 learned-memory-first / better latent。
  - 当前结果：`scripts/verify_stage2_v21_learned_memory.py --score-only = 12/12`
  - 关键证据：online learned path、checkpoint-backed belief、online-aligned training 语义和最小 current-head learned-mode artifact 已成立
- `TD-032` 把 semantic-first 路线推进到 full-data current-head 训练、extended semantic canaries、LongMemEval-S semantic analysis 与 semantic online gain。
  - 当前结果：`scripts/verify_stage2_v22_completion.py --score-only = 19/19`
  - 关键证据：current-head full-data train/eval、`PersonaMem 128` / `LongMemEval-S 64/128` semantic canaries、`latest_longmemeval_stage2_semantic_analysis.json`、`latest_stage2_semantic_online_gain.json`

- `TD-033` 以 `LongMemEval-S` 质量提升、learned slot assignment 与 stronger latent 为目标推进 `v2.3`。
  - 当前结果：后续 `TD-034 / WS-020` 已将这条主线机械收口到 `scripts/verify_stage2_v23_longrun.py --score-only = 22/22`
  - 关键证据：`LongMemEval-S 128` current-head `provider/local = 6/128`、`PersonaMem 128` current-head `provider_exact = 38`、`local_exact = 16`、`latest_stage2_slot_assignment_gain.json`

- `TD-035` 以更远的 `v2.4 quality-first long-run` 为目标推进 `LongMemEval-S` 质量、full-data learned slot assignment 与 stronger latent。
  - 当前结果：managed autoresearch run 已在 current HEAD `12a9a80` 上达到 `scripts/verify_stage2_v24_longrun.py --score-only = 24/24`
  - 关键证据：`latest_stage2_v24_eval.json` 的 `trained_eval.token_f1 = 0.9991150844073334`、`field_f1 = 0.9976704786107581`；`latest_longmemeval_stage2_v24_canary.json` 的 `provider/local = 10/128`；`latest_personamem_stage2_v24_canary.json` 的 `provider_exact = 38`、`local_exact = 28`；`latest_stage2_v24_online_gain.json` 的 `delta_provider_exact_match = +4`、`delta_local_exact_match = +4`

- `TD-036` 以 `v2.5 learned core-path long-run` 为锚点，把当前主线推进到更强的 `LongMemEval-S`、更泛化的 learned slot assignment，以及 `write / retrieve / belief` 三段更真实的 learned 主模块。
  - 说明：这条线的重点不再是补 current-head closeout artifact，而是在不动 `core / residual` 的前提下，把 `v2.4` retained 线当作 baseline，继续做质量和泛化提升。
  - 关键边界：full benchmark 只做 holdout evaluation，不回流成训练 supervision；继续严格禁止任何 fallback / shortcut / benchmark-specific heuristic；不再把 raw JSON exactness 当作主优化目标。


## Notes

- 真实 runtime 状态以 `.agent-os/todo.md` 为准。
- 本文件用于面向人类的同步视图。
- 第一阶段剩余未过项仍主要是正式 benchmark 两项，但当前不作为默认主动执行主线。
- 第二阶段当前已从“方法与治理层锁定”推进到“完整 v2 milestone 已机械达成”的状态。
- 第二阶段 observation / belief / parser / dataset skeleton、`prepare/train/eval/canary` 脚本、主线 memory system、公开数据规范化、strict-source prepared manifest、direct-train launcher、完整 local eval 与评测文档都已落地，`stage2_readiness_score` 当前为 `50`，`stage2_acceptance` 当前为 `7/7`，`scripts/verify_stage2_latent_status.py --score-only` 当前为 `9/9`。
- 当前最大的未完成点已不再是 `v2.2` 的在线证据缺口；`TD-032` 当前已经把 `stage2_v22_completion_score` 推到 `19/19` 并达到 stop condition。
- 当前 `TD-035 / WS-021` 已机械完成；当前主线已继续前推到 `TD-036 / v2.5 long-run`。
- 第二阶段主线采用：
  - `Light Cross-Attention Resampler`
  - `Flan-T5 belief JSON decoder`
  - `public-datasets-first, synthetic-minimal`
  - `benchmark is evaluation source, not primary training source`
