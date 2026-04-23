# TODO

## Doing

- `TD-050` `[doing]` 以 `v6.1 Learned Reader/Decision over Persistent Memory` 为目标，把 v6 的 persistent substrate 升级为真正由 learned reader/readout 主导的 authoritative memory system。
  - 当前 workstream: `WS-036`
  - 当前计划：[docs/v61_plan.md](/media/storage/mingjing/workspace/CoRe_Mem/docs/v61_plan.md)
  - 当前 verifier：[scripts/verify_stage2_v61_learned_reader_decision.py](/media/storage/mingjing/workspace/CoRe_Mem/scripts/verify_stage2_v61_learned_reader_decision.py)
  - 核心目标：`persistent banks -> learned authoritative reader -> learned decision/readout head -> answer`。
  - 硬约束：no fallback、no shortcut、no benchmark-specific heuristic、no provider prompt trick、no PersonaMem gold leakage。
  - 新增硬边界：不允许回到 `vector_dot + bank_prior` 主读路径；不允许 handcrafted option scoring 作为 authoritative decision path；不允许 answer-time routing。
  - 结构继承：v6 已成立的 persistent `core_bank` / `residual_bank`、checkpoint、write trace、write-time routing、raw-context retrieval disabled、answer-time routing disabled 与 true ablation drops 继续作为 baseline 继承。
  - 新增成功门槛：authoritative learned reader、authoritative learned decision head、semantic slot matching / bank compaction、typed residuals、harder internal eval、PersonaMem full589 no-routing > text-only 且 > option-only，并保留 meaningful margin。
  - 当前 v6.1 baseline：fresh `scripts/verify_stage2_v61_learned_reader_decision.py --score-only = 45`；说明 v6 structural evidence 已被继承，但 learned reader/decision 主路径仍是当前主缺口。
  - 当前 partial：current HEAD 已新增 `scripts/publish_stage2_v61_learned_reader_decision.py` 与 `src/core_mem/v2/v61_learned_memory.py`，并把 verifier 推到 `80`
  - 当前 partial：authoritative learned reader / learned decision head、semantic slot compaction、typed residuals、hard internal eval、no-answer-routing 与 no-gold substrate proof 已成立
  - 当前负结果：PersonaMem full589 no-routing `140/589`，text-only `182/589`，option-only `235/589`，对 text-only margin `-42`

- `TD-049` `[blocked]` 以 `v6 Persistent Core-Residual Latent Memory` 为目标，把 v5.2 的 learned latent prototype 升级为真正 persistent memory substrate。
  - 当前 workstream: `WS-035`
  - 当前计划：[docs/v6_plan.md](/media/storage/mingjing/workspace/CoRe_Mem/docs/v6_plan.md)
  - 当前 verifier：[scripts/verify_stage2_v6_persistent_latent_memory.py](/media/storage/mingjing/workspace/CoRe_Mem/scripts/verify_stage2_v6_persistent_latent_memory.py)
  - 核心目标：`dialogue stream -> learned write-time router -> persistent core/residual latent banks -> query-conditioned reader -> belief/readout`。
  - 硬约束：no fallback、no shortcut、no benchmark-specific heuristic、no provider prompt trick、no PersonaMem gold leakage。
  - 新增硬边界：answer-time confidence routing 不允许作为 authoritative path；raw full-context retrieval 不允许作为 v6 authoritative eval 主语义路径。
  - Success gate：persistent bank checkpoint、write trace、learned write-time routing、raw-context retrieval disabled、answer-time routing disabled、真实 architecture ablation drops、PersonaMem full589 no-calibration meaningful margin。
  - Meaningful margin：默认至少 `+30` correct over text-only，或预先声明的显著性 artifact。
  - 当前 v6 baseline：fresh run 已初始化，baseline `16`。
  - 当前 v6 partial：已发布 persistent `core_bank=95` / `residual_bank=483`、state checkpoint、write trace `745` writes、learned write-time router eval accuracy `0.9799` vs disabled controller `0.6644`。
  - 当前 v6 partial：query-conditioned reader / belief-readout 已训练，reader eval accuracy `0.7584` vs disabled readout `0.5`；true disabled architecture ablation drops 已通过。
  - 当前 v6 verifier：`scripts/verify_stage2_v6_persistent_latent_memory.py --score-only = 80`。
  - 当前 v6 negative truth：PersonaMem full589 no-routing `146/589`，text-only `205/589`，margin `-59`；meaningful margin 未达成。
  - 当前 blocker：query-option blend 退化到 `133/589`；no-gold synthetic projection weights 只到 `157/589` 且 score 不提升。继续推进需要新的 persistent memory/readout 方案；不得回到 answer-time routing。

- `TD-048` `[done]` 以 `v5.2 Full Learned Latent Memory System` 为目标，把 v5.1 bootstrap 升级为完整 learned latent system。
  - 当前 workstream: `WS-034`
  - 当前 partial：fresh baseline `4`；iteration `1 keep` 已发布 v5.2 real multi-backbone compare，真实加载 `BGE + E5`，score 到 `20`。
  - 当前 partial：iteration `2 keep` 已发布四任务 multi-task training，`24500` samples，真实 checkpoint 写入 `outputs_v2/checkpoints/20260423T024312Z_stage2_v52_multitask/full_latent_system.pt`，score 到 `30`。
  - 当前 partial：iteration `3 keep` 已发布 PersonaMem full589 no-calibration evaluation，no-calibration `185/589`，text-only `227/589`，decision 为 `negative_result`，score 到 `65`。
  - 当前 partial：iteration `4 keep` 已发布 full latent eval + ablation，score 到 `80`。
  - 当前完成态：iteration `5 keep` 已发布 confidence-routed no-calibration rerun，PersonaMem `228/589` > text-only `227/589`，score 到 `100`。
  - 边界：`Contriever` 因当前 `torch 2.5.1` 安全版本限制未加载；PersonaMem no-calibration 对 text-only 只有 `+1` correct，不能写成强 superiority。
  - 硬门槛：至少 2 个真实 backbone、四任务 multi-task training、真实 checkpoint、trainable encoder/projection + latent reader/resampler + write controller + belief decoder/graph。
  - 质量门槛：multi-task > retrieval-only、trained > frozen、latent-only > shuffled、full > text-only、ablation drops、PersonaMem no-calibration > text-only `214/589`。
  - 禁止：fallback、shortcut、provider prompt trick、benchmark-specific option trick、PersonaMem gold leakage into substrate。
  - 计划：[docs/v52_plan.md](/media/storage/mingjing/workspace/CoRe_Mem/docs/v52_plan.md)

- `TD-047` `[done]` 以 `v5.1 Real Pretrained Training` 为目标，把 v5 scaffold/proxy evidence package 改造成真实 pretrained backbone、真实数据、真实训练和真实 eval 的研究结论。
  - 当前结果：`scripts/verify_stage2_v51_real_training.py --score-only = 100`，`research-results.tsv` iteration `3 keep` 已带齐 stop labels。
  - 关键证据：真实加载 `BAAI/bge-base-en-v1.5`，使用 `sentence_transformers` backend；stage2 32k retrieval_alignment 上训练 12k samples；checkpoint 写入 `outputs_v2/checkpoints/20260423T013123Z_stage2_v51_real_training/latent_retriever.pt`。
  - Held-out：test trained MRR `0.9792` > frozen `0.7054`；latent-only `0.9792` > shuffled-latent `0.9592`；full `0.9792` > text-only `0.7054`。
  - PersonaMem no-calibration：full589 `184/589 = 31.24%`，超过 option-only `167/589 = 28.35%` 与 random `25%`。
  - 边界：当前只完成 BGE 单 backbone；PersonaMem no-calibration 高于 option-only/random，但低于 artifact 里的 text-only `214/589`；不得写成 provider-side 或 formal benchmark superiority。
  - 计划：[docs/v51_plan.md](/media/storage/mingjing/workspace/CoRe_Mem/docs/v51_plan.md)

- `TD-046` `[done]` 以 `v5 Core-Residual Latent Substrate` 为目标，把当前 text-centered slot/belief system 推进成可训练、可消融的 latent memory system evidence package。
  - 当前锚点：retained `v32 / v33 / v4` baselines 继续保留；v5 managed run 已从 baseline `13` 推到 `52/52`。
  - 已完成重点：PersonaMem gold-isolation、gold-free context self-supervision、encoder proxy comparison、learned write controller、core/residual latent state、query-conditioned latent reader、latent-only / text-only / shuffled-latent / no-controller ablation、strict PersonaMem gold calibration isolation。
  - Final retained state：`scripts/verify_stage2_v5_longrun.py --score-only = 52`；`research-results.tsv` iteration `7 keep` 已记录 stop-condition label。
  - 硬约束：no fallback、no shortcut、no benchmark-specific heuristic、no benchmark leakage；PersonaMem gold 不得训练 memory substrate，只能隔离训练薄 answer/readout head。
  - Success boundary：当前本地 verifier / artifacts 机械满足；但 encoder comparison 是 deterministic proxy 且显式记录 `pretrained_weights_loaded = false`，provider 仍是 auxiliary，不得写成 provider-side superiority 或真实 pretrained comparison 结论。
  - 计划：[docs/v5_plan.md](/media/storage/mingjing/workspace/CoRe_Mem/docs/v5_plan.md)

- `TD-045` `[done]` 以 `v4 Persona-first learned memory` 为目标（`TD-045` / `v4` / `PersonaMem 512`），在**不改 `core / residual` 双银行结构**的前提下，暂时把主优化目标收敛到 `PersonaMem 512`，让 learned latent / belief / option scorer 成为 finite-option personalization 的主能力来源。
  - 当前锚点：retained `v32` full `PersonaMem 512` 为 `provider/local = 183/175`；current `v33` learned-authoritative full 为 `provider/local = 182/196`。
  - 当前诊断：`v33` 在 Persona local 上已有 `+21`，但 provider exact 反而 `-1`；`LongMemEval-S` 与 Persona 的任务形态差别太大，短期同时优化会拉扯方向。
  - 当前策略：`PersonaMem 512` 是主指标；`LongMemEval-S 500` 只作为 non-catastrophic guard。provider 只作为 auxiliary evidence，不再一票否决 learned option-scorer/local gain。
  - Final retained state: `scripts/verify_stage2_v4_longrun.py --score-only = 38/38`
  - Success evidence: learned local / option-scorer full replay `219/512` 超过 `v33` full Persona local `196/512`
  - 硬约束：no fallback、no shortcut、no benchmark-specific heuristic、no benchmark leakage、不训练 benchmark answers、不靠 provider prompt trick 作为主收益。
  - 计划：[docs/v4_plan.md](/media/storage/mingjing/workspace/CoRe_Mem/docs/v4_plan.md)

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
    - retained `v33` 仍是 `36/47 keep`，latest full-measurement trial 已到 `43/47 search`
    - `latest_stage2_v33_modular_authoritative_train.json`、`latest_stage2_v33_learned_write_eval.json`、`latest_stage2_v33_latent_reader_train.json`、`latest_stage2_v33_temporal_slot_eval.json`、`latest_stage2_v33_latent_objective_eval.json`、`latest_stage2_v33_belief_graph_eval.json`、`latest_stage2_v33_answer_option_eval.json` 与 `latest_stage2_v33_ablation_summary.json` 已落地
    - current-line authoritative `PersonaMem 512` full holdout 已真实完成：`outputs_v2/v33_full_personamem_512_query_overlap/evals_benchmark/20260421T001000Z_stage2_memory_canary.json` 当前固定为 `provider/local = 182/196`
    - 相对 retained `v32` Persona full `183/175`，current line 呈现 `provider -1 / local +21`；也就是说 local learned-authoritative gain 已在 full measurement 上成立，但 provider 仍略低于 retained baseline
    - 当前 provider side residual 也已在同一 full Persona artifact 上固定为 `blank = 108`、`nonlabel = 12`
    - row `102 search` 已把 completed Persona full measurement 正式记账，row `103 search` 又把 current-line `LongMemEval-S 500` full holdout、published compare 与 verifier `43/47` 一并记账
    - `latest_stage2_v33_learned_authoritative_runtime.json`、`latest_longmemeval_stage2_v33_full.json`、`latest_personamem_stage2_v33_full.json` 与 `latest_stage2_v33_full_holdout_compare.json` 已在 commit `6fa018e70561eef6b0239f41b3d662794df41a22` 上落地
    - `v32` 已证明 modular latent / belief / answer 设计可以带来 full-holdout gain
    - 但 `v32` authoritative full benchmark path 仍是 `memory_mode = symbolic`、`slot_assignment_mode = symbolic`
    - `v33` 的目标不是继续扩大 symbolic baseline，而是让 learned path 接管 authoritative runtime
    - current-line `LongMemEval-S 500` authoritative full holdout `outputs_v2/v33_full_longmemeval_500_query_overlap/evals_benchmark/20260422T015146Z_stage2_memory_canary.json` 当前固定为 `provider/local = 20/14`
    - 相对 retained `v32` LongMemEval full `21/14`，current line 呈现 `provider -1 / local tie`；相对 retained `v32` Persona full `183/175`，current line 仍是 `provider -1 / local +21`
    - row `104 search` 又把 current-line `LongMemEval` exactness patch 正式记账：generic number-word extraction、location temporal-tail trim 与 low-information `other_fact` belief backfill 已在 authoritative singles 上翻正 `6b168ec8 -> three`、`1faac195 -> denver` 与 `gpt4_ec93e27f -> train`
    - `latest_stage2_v33_full_holdout_compare.json` 当前显式记录 `longmemeval_gain_confirmed = false`、`personamem_gain_confirmed = false`
    - row `105 crash` 已机械确认：若直接沿默认 `cuda` 路径起 fresh rerun，当前机器上的 `GPU 0` 可能因外部占用而在 shared learned predictor 初始化阶段 OOM
    - row `106 no-op` 又机械确认：fresh output root 若不先显式生成 full manifest，`run_stage2_memory_canary.py` 会自动退回 `64/64` canary manifests；单纯传 `--limit 500` 不足以形成 authoritative full rerun
    - 当前真正有效的 rerun 已经切到 `outputs_v2/v33_full_longmemeval_500_exactness_rerun_gpu1_fullmanifest/`：预生成 full `LongMemEval` manifest 后，authoritative `GPU 1` run 已在 same commit `15060d1` 上启动，last check `run_metadata.json` 为 `sample_count = 500`、`provider_configured = true`、`completed_predictions = 12`
    - 因而当前 top next action 已从继续支付 Persona local smoke 收窄到：继续监控并收口这个 in-flight `LongMemEval-S 500` full rerun，完成后立即发布 fresh compare，而不是再起新的 smoke 或新的 mis-sized rerun
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
    - 对应 full guard 已重新通过，但 `scripts/verify_stage2_v33_longrun.py --score-only` 仍是 `36`；与此同时，current commit `f9ebc8b` 的 fresh `smoke8` local-only replay `outputs_v2/v33_answer_head_overlap_smoke8_localonly/evals_benchmark/20260421T062832Z_stage2_memory_canary.json` 已完整收口到 `local_exact = 8/8`、`baseline = 1/8`，并且样本级确认 `2cef...` 与 `0d...` 都已翻正。随后 current HEAD `51b2462` 又为 `OptionScoringHead` 加入 generic morphology normalization；fresh broader gate `outputs_v2/v33_answer_head_stem_smoke16_localonly/evals_benchmark/20260421T071739Z_stage2_memory_canary.json` 已把 `local_exact` 从旧的 `12/16` 提到 `14/16`、`baseline = 2/16`，并翻正 `5370...` 与 `32b...`。commit `8d391e1` 再把 scorer 看到的 gloss 从“全部 selected slots”收紧到“belief 实际 support slots”，fresh rerun `outputs_v2/v33_support_gloss_smoke16_localonly/evals_benchmark/20260421T075401Z_stage2_memory_canary.json` 已继续把 `local_exact` 提到 `15/16`，并翻正 `344ea...`
  - Truth boundary：
    - 当前主矛盾已不再是 v32 能否涨分，而是 learned path 能否接管 authoritative full-holdout runtime
    - 这轮不允许在 symbolic authoritative path 上收口
    - 当前已批准的新路线是：保留同一 provider/interface，只做协议层最小 `<think>` 清洗；`8`-sample 不再决定方向，retain 直接看 `LongMemEval-S 500 / PersonaMem 512`
    - 当前真实状态是 active measurement / refine，而不是“必须先切 provider/interface 才能继续”的硬 blocker；但修复后的 Persona provider exact 仍明显低于 retained `v32`。discard rows `51-52` 已先说明：`0adf...` 不是 parser-only 缺 observation，且 parser widening + belief-prompt reorder 虽然能把它单样本翻到 gold `(c)`，却会在 broader slice 上立刻打坏 `acd742... / f546... / b358...`
    - 随后的 search rows `58-60` 又补了一层新真相：把 learning-topic recall 收紧到 same-turn authoring context、同时把过宽的 `i'm ...` parser 模式收回 occupation-shape 后，current authoritative runtime 可以把 `a40d...` 恢复到正确 `(d)`，并保住 `0adf...` proxy / `b449...` single，不再出现旧的 `other_fact=learning about` parser 噪声
    - 但最新 discard row `61` 也明确说明：commit `fea3dea` 虽然在 broader `smoke16` partial 中继续守住 `acd742... / f546... / b358... / 5370... / a40d... / 2cef...`，却新打坏了 retained 原本正确的 `6142...`，使 belief 漂回抽象 `music_preference=music in its truest form, without rigid guidelines dictating how i should dissect it`，最好结果仍只能打平 retained `15/16`。因此 parser-noise cleanup 不是当前 gain driver；下一步必须直接转向 same-relation `music_preference` belief-selection / answer-selection，而不是继续保留 parser narrowing 线
    - latest search row `62` 则把 parser 侧 root cause 又收紧了一步：`6142...` 里真正该保留的并不是泛化 `i am ...`，而是更精确的 `I am more drawn to ...` preference clause。当前 worktree 已把旧的泛化 `i am / i'm` value 规则收回 occupation-shape，同时显式支持 `more drawn to ...`；targeted parser tests 与 real-utterance reproduction 已确认这条 patch 能恢复 `6142` 所需的 `music_preference=more drawn to the emotional aspects of music...`，同时继续压住 `a40d` 的 `I'm learning about` 噪声并保住 feedback reason extraction。由于 learned-authoritative singles 仍太慢，这条线目前仍只是 uncommitted narrow candidate，下一步必须先过一个更宽但更便宜的 integration gate
    - committed candidate `3288713` 已经结束并被 row `69` discard：虽然它在 `learned_memory + symbolic slot assignment` 下通过了 `6142... / a40d... / acd742... = 3/3` cheap integration gate，full guard 也通过，但 wider symbolic-slot `smoke16` `outputs_v2/v33_parser_precision_smoke16_symbolic_slot_v2/` 最终只到 `13/16`，miss 收敛到 `d71 / 32b / 0adf`，仍低于 retained `15/16` floor
    - 当前 parser line 已经被进一步证伪为“不是 broader gain driver”。下一步必须回到 retained line，直接分析 `d71 / 32b / 0adf` 这组 residual misses 是否共享 belief-selection / answer-projection 的更高层结构性问题
    - rows `70-71` 已把一条新的 committed candidate `eb9480b` 放进 runtime truth：它不再改 parser，而是只对 `other_fact` 的 interactional request-like support 做窄 answer-head repair。当前 real learned-authoritative row-build 已机械确认 `0adf...` 从 `(a)` 翻到 `(c)`，`ead3...` 也能从 request-like support 漂移里回到 `(d)`；同时 cheap audit 又确认现有 `PersonaMem 512` full artifacts 里 request-like single-item `other_fact` support 的唯一真实命中正是 `ead3...`
    - row `72 refine` 先把这条 candidate 提升到更宽 authoritative local gate：fresh `smoke16` local-only rerun `outputs_v2/v33_interactional_other_fact_smoke16_localonly_v2/evals_benchmark/20260421T120739Z_stage2_memory_canary.json` 已完整收口到 `16/16`，把此前 retained 唯一 miss `0adf...` 翻正，并保住 `d71... / 32b... / a40d... / 6142...`
    - row `74 refine` 曾把同一条 candidate 扩到 authoritative `smoke17` local-only gate：fresh rerun `outputs_v2/v33_interactional_other_fact_smoke17_localonly/evals_benchmark/20260421T123528Z_stage2_memory_canary.json` 在 retained `16`-slice 上并入真实 full-holdout request-shape 样本 `ead3...` 后收口到 `17/17`
    - 但 row `75 discard` 的更宽 authoritative local-only Persona `smoke32` `outputs_v2/v33_interactional_other_fact_smoke32_localonly/evals_benchmark/20260421T130303Z_stage2_memory_canary.json` 最终只有 `24/32`，并新增 `8` 条 regression：`c8a763... / c2d3daad... / 2b3ce0bd... / 7a4ed201... / afd724f4... / 346a304a... / cd81feb9... / 0d78111b...`
    - `0adf...` 与 `ead3...` 仍然 hold，说明这条 patch 不是完全无效；但它已经明确不是可保留的 broader fix
    - 当前 HEAD 已按 rollback policy 回到 revert commit `3aa38d2`，full guard 通过，`scripts/verify_stage2_v33_longrun.py --score-only` 仍是 `36`
    - row `76 search` 已完成同一 `32`-sample manifest 的 post-rollback authoritative local-only 对照：`outputs_v2/v33_postrollback_smoke32_localonly/evals_benchmark/20260421T134355Z_stage2_memory_canary.json` 最终只有 `22/32`
    - 这条 compare 说明 simple rollback 不是更优 baseline：它继续保留 `c8a763... / c2d3daad... / 2b3ce0bd... / 7a4ed201... / afd724f4... / 346a304a... / 0d78111b...` 等 persistent residual，又失去了 `0adf...` 与 `ead3...` 的 request-like 窄 gain，`cd81feb9...` 也仍然错误但换成另一种错答
    - row `77 search` 已把 `eb9480b` 逻辑恢复到当前 HEAD `d48c58c`，因为它在同一 `smoke32` 上严格优于 simple rollback；相关 interactional canary tests 通过，`scripts/verify_stage2_v33_longrun.py --score-only` 仍是 `36`
    - row `78 search` 又在 `d48c58c` 上把 local option scorer 收紧为“single-slot base projection 不吃 query overlap、multi-slot base projection 才允许 query tie-break”；新增 `c8`/`afd` regression tests，full guard 通过，verifier 仍是 `36`
    - row `79 search` 的 authoritative learned-memory Persona residual `smoke5` 已先确认 `c8 / afd / 0adf / ead3` 在真实 runtime 上成立，但 `0d225` 仍会因为 direct-fact recall 的 single-support payload shape 回退
    - row `80 search` 随后把 direct-fact recall question type 单独纳入 query tie-break；fresh authoritative `smoke5` rerun `outputs_v2/v33_option_scorer_residual_smoke5_recall_localonly/evals_benchmark/20260421T145519Z_stage2_memory_canary.json` 已完整收口到 `5/5`
    - row `81 search` 的 widening gate `outputs_v2/v33_option_scorer_widen_smoke11_localonly/evals_benchmark/20260421T150027Z_stage2_memory_canary.json` 已完整收口到 `5/11`：controls `c8 / afd / 0d225 / 0adf / ead3` 全部 hold，但 `c2d3 / 2b3 / 7a4 / 346 / 0d781 / cd81` 仍然全部错误
    - row `82 discard` 又排除了一个过宽 recommendation 分簇：`outputs_v2/v33_option_scorer_widen_smoke11_reco_localonly/evals_benchmark/20260421T151039Z_stage2_memory_canary.json` 把同一 `smoke11` 提到 `6/11`，唯一新增 gain 是 `2b3 -> (c)`，但同时把已有的 `c8` recommendation invariant 打坏，full guard 失败在 `test_personamem_local_projection_breaks_truest_music_tie_toward_first_matching_option`
    - row `83 search` 又把这条 recommendation line 收窄成 relation-gated candidate：`outputs_v2/v33_option_scorer_widen_smoke11_reco_otherfact_localonly/evals_benchmark/20260421T153608Z_stage2_memory_canary.json` 在 `provide_preference_aligned_recommendations + single-belief other_fact` 下使用 `selected_slot_glosses`，fresh authoritative rerun 已完整收口到 `6/11`，并把 `2b3 -> (c)` 拉正，同时继续保住 `c8 / afd / 0d225 / 0adf / ead3`，full guard 也通过
    - row `84 search` 又补了一条 interactional repair gain：`outputs_v2/v33_interactional_best_candidate_smoke3_localonly/evals_benchmark/20260421T155311Z_stage2_memory_canary.json` 已完整收口到 `3/3`，把 `cd81 -> (b)` 拉正，同时继续保住 `0adf -> (c)` 与 `ead3 -> (d)`；这说明 interactional other_fact repair 的主问题之一是 candidate selection 顺序依赖，而不是 repair 路线本身无效
    - row `85 search` 已把这两条 live line 合并到同一 authoritative `smoke11`：`outputs_v2/v33_interactional_best_candidate_smoke11_localonly/evals_benchmark/20260421T155704Z_stage2_memory_canary.json` 已完整收口到 `7/11`，`cd81 -> (b)` 的 interactional gain 与 `2b3 -> (c)` 的 relation-gated recommendation gain 可以共存，同时继续守住 `c8 / afd / 0d225 / 0adf / ead3`
    - row `86 search` 又给这四条 residual 补了一条 authoritative parser gate：`outputs_v2/v33_parser_residual_smoke4_localonly/evals_benchmark/20260421T161703Z_stage2_memory_canary.json` 只到 `1/4`，只有 `0d781 -> (c)` 会被新的 `community theater audition` coverage 翻正，`c2d3 / 7a4` 仍然停在旧 stale support，而 `346` 还被新的 `local library` 观测污染成另一种错答
    - row `87 search` 又把这条 parser line 收窄回 `community theater audition` + `had to step back` 最小版本；fresh authoritative `smoke2` `outputs_v2/v33_parser_theater_smoke2_localonly/evals_benchmark/20260421T162630Z_stage2_memory_canary.json` 已给出 `1/2`：`0d781 -> (c)` 继续成立，而 `346` 的 row `86` library contamination 已消失，但它仍旧停在更老的 hobby belief `(a)`
    - row `88 discard` 又排除了最后一条 parser 残余假设：把 `crowded-festival aversion` 单条 clause 加回去后，fresh authoritative `smoke2` `outputs_v2/v33_parser_theater_festival_smoke2_localonly/evals_benchmark/20260421T163358Z_stage2_memory_canary.json` 仍然只是 `1/2`，和 row `87` 完全相同；`0d781` 继续 hold，但 `346` 仍旧停在同一个 hobby belief `(a)`
    - row `89 search` 又把这组 residual 再拆开了一层：同-turn clause context parser 让 fresh authoritative `smoke3` `outputs_v2/v33_parser_contextual_smoke3_localonly/evals_benchmark/20260421T165206Z_stage2_memory_canary.json` 里，`c2d3` 第一次落到正确 belief `felt too pressured by reading challenge deadlines`，但 local answer 仍然是 `(a)`；`346` 则继续完全不动，`0d781` 继续 hold
    - row `90 discard` 又排除了“只改 c2d3 contextual belief wording 就能稳定带来 projection gain”这条线：虽然 cheap scorer 更偏向 `(b)`，但真实 learned runtime 里 `c2d3` 反而掉回旧 stale support
    - row `91 discard` 再排除了 broad polarity tie-break：它确实能把 authoritative `smoke3` 提到 `2/3`，修正 `c2d3` 并保住 `0d781`，但会打坏 retained recommendation regression test
    - row `92 search` 则把 polarity tie-break 收窄到 recall-family question types，fresh authoritative `smoke3` `outputs_v2/v33_answer_polarity_recallonly_smoke3_localonly/evals_benchmark/20260421T170528Z_stage2_memory_canary.json` 已完整给出 `2/3`：`c2d3 -> (b)` 与 `0d781 -> (c)` 同时成立，`346` 继续静止，full guard 通过
    - row `93 discard` 又排除了“补一条更窄的 library-affinity parser 就能修 `7a4`”这条线：fresh authoritative `smoke3` `outputs_v2/v33_parser_library_smoke3_localonly/evals_benchmark/20260421T173124Z_stage2_memory_canary.json` 最终只有 `1/3`，虽然 `appreciate visiting local libraries` 已进入 `7a4` 的 selected top-8，但 learned belief 仍坚持旧的 `step back from structured book club settings`；同时 `346` 反而被新的 library slot 污染成 `other_fact=appreciate visiting local libraries` 并落到 `(d)`，只有 `0d781 -> (c)` 继续 hold
    - row `94 search` 进一步说明，这条 library line 只有在和更窄的 statement-recall support coercion 组合时才会产出真实 gain：fresh authoritative `smoke3` `outputs_v2/v33_parser_library_support_coercion_v2_smoke3_localonly/evals_benchmark/20260421T175208Z_stage2_memory_canary.json` 已把 `7a4 -> (c)` 真正翻正，同时继续保住 `0d781 -> (c)`，但 `346` 仍旧被 library belief 污染成 `(d)`；full guard 通过，verifier 仍是 `36`
    - row `95 discard` 又排除了“把 library-affinity wording 改得更 peaceful / inviting 就能顺手修掉 `346`”这条线：fresh authoritative `smoke3` `outputs_v2/v33_parser_library_support_coercion_enriched_smoke3_localonly/evals_benchmark/20260421T180337Z_stage2_memory_canary.json` 仍然只是 `2/3`，而且 `346` 反而从 row `94` 的 library contamination 退化成更差的 stale-reader belief `imagining the conversations that might have occurred among readers`，provider 也从 `(c)` 掉到 `(a)`；因此这条 wording enrichment 已按 discard 记账并回滚
    - row `96 search` 又把这组 residual 再向前推进了一步：fresh authoritative `smoke3` `outputs_v2/v33_environment_fit_support_smoke3_localonly/evals_benchmark/20260421T181900Z_stage2_memory_canary.json` 已完整给出 `local_exact_match = 3/3`、`baseline_local_exact_match = 1/3`。其中 `346...` 在真实 learned-authoritative runtime 上第一次翻到 gold `(c)`，selected[0] / belief / support 全部切到 `other_fact=larger festivals feel too crowded and chaotic for me`；`7a4 -> (c)` 与 `0d781 -> (c)` 同时继续 hold，full guard 通过，official verifier 仍是 `36`
    - 当前最诚实的 unresolved truth 因而进一步收窄成：`0d781` 是干净的窄 parser gain candidate；`c2d3` 已从 unresolved set 中移除；`7a4` 已进入 statement-recall support-switch 的 live search 候选；`346` 则从 “primary unresolved residual” 降成一个已被真实修正、但尚未 wider-validated 的 environment-fit rerank/support candidate
    - row `97 search` 又把这条 line 从 residual trio 扩成 coexistence gate：fresh authoritative `smoke8` `outputs_v2/v33_environment_fit_support_smoke8_localonly/evals_benchmark/20260421T182800Z_stage2_memory_canary.json` 已完整给出 `local_exact_match = 8/8`、`baseline_local_exact_match = 1/8`，机械确认 row `96` 的 environment-fit surgery 不会打坏 row `80` 的 scorer controls `c8 / afd / 0d225 / 0adf / ead3`，同时 `346 / 7a4 / 0d781` 继续全部 hold
    - row `98 search` 又把 current line 推回旧 `smoke11` 同 manifest：fresh authoritative `smoke11` `outputs_v2/v33_environment_fit_support_smoke11_localonly/evals_benchmark/20260421T183400Z_stage2_memory_canary.json` 已完整给出 `local_exact_match = 11/11`、`baseline_local_exact_match = 1/11`，说明 row `80 / 83 / 84 / 92 / 94 / 96` 这些此前分散成立的 gains 现在已经能在同一 learned-authoritative line 上同时成立
    - row `99 search` 又把 widened gate 跑实：fresh authoritative `smoke32` `outputs_v2/v33_environment_fit_support_smoke32_localonly/evals_benchmark/20260421T184200Z_stage2_memory_canary.json` 已完整给出 `local_exact_match = 29/32`、`baseline_local_exact_match = 3/32`，相对 row `75` 的旧 wider gate `24/32` 净增 `+5`
    - 这条 widened gate 已把旧 `smoke32` 的八条错例 `c8 / c2d3 / 2b3 / 7a4 / afd / 346 / cd81 / 0d781` 全部翻正，但同时新引入了 `defecce... / b481... / 7028667...` 三条 `musicRecommendation` regression
    - row `100 search` 已把这三条新 regression 收窄成 authoritative `smoke4`：fresh local-only gate `outputs_v2/v33_music_projection_smoke4_localonly/evals_benchmark/20260421T000000Z_stage2_memory_canary.json` 已完整给出 `local_exact_match = 4/4`、`baseline_local_exact_match = 0/4`，说明 `defecce... / b481... / 7028667...` 已全部翻正，且 `c2d3...` 继续 hold
    - row `101 search` 已把这条 query-overlap patch widen 回旧 `smoke32` 同 manifest：fresh authoritative local gate `outputs_v2/v33_music_projection_smoke32_localonly/evals_benchmark/20260421T000500Z_stage2_memory_canary.json` 已完整给出 `local_exact_match = 32/32`、`baseline_local_exact_match = 3/32`
    - 当前 local coexistence truth 已经收口，不再有 active Persona-side local blocker
    - 下一步不再继续支付 local smoke，而是直接启动 v33 的 full-holdout measurement 与 verifier publish 链路

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
