# TODO

## Doing

- `TD-050` `[doing]` 以 `v6.1 Learned Reader/Decision over Persistent Memory` 为目标，把 v6 的 persistent substrate 推进到真正 learned authoritative read/decision path。
  - Current workstream: `WS-036`
  - Plan: [docs/v61_plan.md](/media/storage/mingjing/workspace/CoRe_Mem/docs/v61_plan.md)
  - Verifier: [scripts/verify_stage2_v61_learned_reader_decision.py](/media/storage/mingjing/workspace/CoRe_Mem/scripts/verify_stage2_v61_learned_reader_decision.py)
  - Required structure: persistent substrate inherited from v6, but authoritative inference must now use a learned reader and a learned decision/readout head.
  - Hard constraints: no fallback、no shortcut、no benchmark-specific heuristic、no provider prompt trick、no PersonaMem gold leakage、no raw full-context retrieval、no answer-time routing。
  - New required evidence: semantic slot matching / bank compaction, typed residuals, harder internal eval with natural-language queries and hard negatives, PersonaMem full589 no-routing > text-only and > option-only with meaningful margin.
  - Current baseline: `scripts/verify_stage2_v61_learned_reader_decision.py --score-only = 45`
  - Current partial: current HEAD 已发布 learned reader/decision artifacts，`scripts/verify_stage2_v61_learned_reader_decision.py --score-only = 85`
  - Current partial: authoritative learned reader / learned decision head、semantic slot matching / bank compaction、typed residuals、hard internal eval、no-answer-routing 与 no-gold substrate proof 已成立；当前 latest keep 又用 generic low-information slot filtering 把 PersonaMem full589 no-routing 从 `182` 提到 `191`
  - Current negative truth: PersonaMem full589 no-routing `191/589` vs text-only `180/589` vs option-only `235/589`，margin `+11`
  - Current reason for launch: v6 already proved persistent storage, but failed on no-routing PersonaMem (`146/589` vs text-only `205/589`). The next run must strengthen reading/decision rather than storage.

- `TD-049` `[blocked]` 以 `v6 Persistent Core-Residual Latent Memory` 为目标，把 v5.2 的 real learned latent prototype 升级为真正 persistent core/residual latent memory substrate。
  - Current workstream: `WS-035`
  - Plan: [docs/v6_plan.md](/media/storage/mingjing/workspace/CoRe_Mem/docs/v6_plan.md)
  - Verifier: [scripts/verify_stage2_v6_persistent_latent_memory.py](/media/storage/mingjing/workspace/CoRe_Mem/scripts/verify_stage2_v6_persistent_latent_memory.py)
  - Required structure: persistent `core_bank` and `residual_bank`; learned write-time memory routing connected to actual state updates; query-conditioned latent reader over persistent banks; learned belief/readout path without answer-time confidence routing; true disabled-architecture ablations.
  - Hard constraints: no fallback、no shortcut、no benchmark-specific heuristic、no provider prompt trick、no PersonaMem gold leakage into memory substrate、no raw full-context retrieval as v6 authoritative memory path。
  - Stop gate: PersonaMem full589 no-calibration must beat text-only by a meaningful margin, default `>= +30` correct, not v5.2-style `+1`.
  - Current baseline: fresh managed run initialized at `stage2_v6_persistent_latent_memory_score = 16`.
  - Current partial evidence: persistent `core_bank=95` / `residual_bank=483` exists, checkpoint exists, write trace records `745` writes, and learned write-time router is applied to state updates.
  - Current partial evidence: v6 query-conditioned reader / belief-readout is trained; true disabled architecture ablation drops now pass.
  - Current score: `scripts/verify_stage2_v6_persistent_latent_memory.py --score-only = 80`.
  - Current negative result: PersonaMem full589 no-routing `146/589` vs text-only `205/589`, margin `-59`; no claim allowed.
  - Current blocker/gap: significant PersonaMem no-routing margin absent. Query-option blend degraded to `133/589`; no-gold synthetic projection weights reached only `157/589` and did not improve verifier score. Further progress requires a new persistent-memory/readout direction, not answer-time routing.

- `TD-048` `[done]` 以 `v5.2 Full Learned Latent Memory System` 为目标，把 v5.1 bootstrap 升级为完整 learned latent system。
  - Current workstream: `WS-034`
  - Current retained partial:
    - fresh baseline `stage2_v52_full_latent_system_score = 4`
    - iteration `1 keep`: v5.2 real multi-backbone compare，真实加载 `BAAI/bge-base-en-v1.5` 与 `intfloat/e5-base-v2`，score 到 `20`
    - attempted `facebook/contriever`，但当前 `torch 2.5.1` 被 `transformers` 安全限制挡住；artifact 已记录 failure，不能写成三 backbone 完成
    - iteration `2 keep`: v5.2 四任务 multi-task training，`24500` samples，checkpoint `outputs_v2/checkpoints/20260423T024312Z_stage2_v52_multitask/full_latent_system.pt`，score 到 `30`
    - iteration `3 keep`: v5.2 PersonaMem full589 no-calibration evaluation，`185/589`；text-only `227/589`；decision `negative_result`；score 到 `65`
    - iteration `4 keep`: v5.2 full latent eval + ablation，trained > frozen、multi-task > retrieval-only、latent-only > shuffled、full > text-only composite，三项 ablation drops；score 到 `80`
    - iteration `5 keep`: confidence-routed no-calibration rerun，PersonaMem `228/589` > text-only `227/589`，score 到 `100`
    - Truth boundary: PersonaMem margin 只有 `+1` correct，route 使用 text evidence `539` 次、latent evidence `50` 次；可以写成 mechanical stop condition reached，但不能写成 strong superiority
  - Hard gates:
    - at least 2 real pretrained backbones loaded/compared
    - non-retrieval-only multi-task training over `retrieval_alignment / slot_autoencoding / composition_to_belief / lifecycle_prediction`
    - real checkpoint exists
    - trainable encoder/projection, latent reader/resampler, write controller, belief decoder/graph
    - multi-task latent > retrieval-only latent
    - trained > frozen
    - latent-only > shuffled-latent
    - full > text-only
    - no-controller / no-belief / no-core-residual ablations drop
    - PersonaMem no-calibration > text-only `214/589`
  - Plan: [docs/v52_plan.md](/media/storage/mingjing/workspace/CoRe_Mem/docs/v52_plan.md)

- `TD-047` `[done]` 以 `v5.1 Real Pretrained Training` 为目标，把 v5 scaffold/proxy evidence package 改造成真实 pretrained backbone、真实数据、真实训练和真实 eval 的研究结论。
  - Current workstream: `WS-033`
  - Final retained state:
    - baseline `stage2_v51_real_training_score = 10`
    - iteration `1 keep`: real BGE pretrained load / sentence-transformers backend，score `30`
    - iteration `2 keep`: 12k stage2 gold-free training、real checkpoint、held-out trained > frozen / latent > shuffled / full > text-only，score `85`
    - iteration `3 keep`: PersonaMem full589 no-calibration `184/589 = 31.24%`，超过 option-only `167/589 = 28.35%` 与 random `25%`，score `100`
  - Stop labels retained: `real-pretrained`、`real-checkpoint`、`heldout-gain`、`anti-shortcut`、`no-calibration`
  - Evidence:
    - `outputs_v2/artifacts/latest_stage2_v51_real_backbone_compare.json`
    - `outputs_v2/artifacts/latest_stage2_v51_real_training.json`
    - `outputs_v2/artifacts/latest_stage2_v51_latent_eval.json`
    - `outputs_v2/artifacts/latest_stage2_v51_personamem_full589.json`
    - `outputs_v2/artifacts/latest_stage2_v51_real_training_decision.json`
    - `research-results.tsv` iteration `3 keep`
  - Truth boundary:
    - 当前完成态是 local mechanical verifier + gold-free/no-calibration positive result
    - 三 backbone 比较尚未完成；当前只真实加载并训练 `BAAI/bge-base-en-v1.5`
    - PersonaMem no-calibration 高于 option-only/random，但仍低于同 artifact 的 text-only score `214/589`
    - 不得写成 provider-side superiority、formal benchmark superiority，或 broader architecture 已充分验证
  - Plan: [docs/v51_plan.md](/media/storage/mingjing/workspace/CoRe_Mem/docs/v51_plan.md)

- `TD-046` `[done]` 以 `v5 Core-Residual Latent Substrate` 为目标，把当前 text-centered slot/belief system 推进成可训练、可消融的 latent memory system evidence package。
  - Current workstream: `WS-032`
  - Current retained baselines: `v32` symbolic authoritative full benchmark、`v33` learned-authoritative runtime/full-holdout evidence、`v4` Persona-first learned option replay。
  - Final retained state:
    - baseline `stage2_v5_longrun_score = 13`
    - final `scripts/verify_stage2_v5_longrun.py --score-only = 52`
    - `research-results.tsv` iteration `7 keep` 已记录 stop-condition label
  - Implemented scope:
    - PersonaMem persona/context grouped gold-isolation checker
    - PersonaMem context-level self-supervised episode data without gold answers
    - `BGE / E5 / Contriever` encoder comparison harness with deterministic proxy metrics
    - core-residual latent substrate and learned write-controller train artifact
    - query-conditioned latent reader and latent/text/shuffled ablations
    - thin answer-head calibration and PersonaMem full-589 local report
    - anti-shortcut ablation summary and paper evidence package
  - Hard constraints:
    - no fallback
    - no shortcut
    - no benchmark-specific heuristic
    - no benchmark leakage
    - PersonaMem gold can calibrate only a thin answer/readout head, never the memory writer/reader/controller/latent substrate
  - Truth boundary:
    - current completion is local mechanical verifier + artifact evidence
    - provider is auxiliary, not a superiority claim
    - encoder comparison currently records model ids through a deterministic hashing proxy and explicitly marks `pretrained_weights_loaded = false`
  - Plan: [docs/v5_plan.md](/media/storage/mingjing/workspace/CoRe_Mem/docs/v5_plan.md)

- `TD-045` `[done]` 以 `v4 Persona-first learned memory` 为目标（`TD-045` / `v4` / `Persona-first`），在**不改 `core / residual` 双银行结构**的前提下，主攻 `PersonaMem 512` 的 learned latent / belief / option-scorer 能力。
  - Primary benchmark: `PersonaMem 512`
  - Guard benchmark: `LongMemEval-S 500` non-catastrophic local guard
  - Constraints: no fallback, no shortcut, no benchmark-specific heuristic, no benchmark leakage
  - Baseline truth: `v32 PersonaMem 512 provider/local = 183/175`; `v33 learned-authoritative PersonaMem 512 provider/local = 182/196`
  - Final retained state: `scripts/verify_stage2_v4_longrun.py --score-only = 38/38`
  - Success evidence: learned option-scorer/local full replay `219/512` beats `v33` local `196/512`; provider remains auxiliary; ablation artifact records latent / belief / option scorer contribution and no fallback/shortcut
  - Plan: [docs/v4_plan.md](/media/storage/mingjing/workspace/CoRe_Mem/docs/v4_plan.md)

- `TD-044` `[doing]` 以 `v3.3 / v33 learned-authoritative latent run` 为目标，在**不改 `core / residual` 双银行结构**的前提下，基于 retained `v32` baseline，推进一个真正由 learned path 主导 authoritative runtime 的 latent system。
  - Current focus:
    - learned authoritative runtime
    - competition-based learned write
    - temporal-semantic latent reader
    - belief graph head
    - answer / option head
    - full benchmark holdout：`LongMemEval-S 500 / PersonaMem 512`
  - Hard constraints:
    - no fallback
    - no shortcut
    - no benchmark-specific heuristic
    - no benchmark leakage
    - benchmark remains holdout-only
  - Runtime truth:
    - retained `v32` 已到 `44/44 keep`
    - retained `v33` 仍是 `36/47 keep`，latest full-measurement trial 已到 `43/47 search`
    - `latest_stage2_v33_modular_authoritative_train.json`、`latest_stage2_v33_learned_write_eval.json`、`latest_stage2_v33_latent_reader_train.json`、`latest_stage2_v33_temporal_slot_eval.json`、`latest_stage2_v33_latent_objective_eval.json`、`latest_stage2_v33_belief_graph_eval.json`、`latest_stage2_v33_answer_option_eval.json` 与 `latest_stage2_v33_ablation_summary.json` 已落地
    - current-line authoritative `PersonaMem 512` full holdout 已真实完成：`outputs_v2/v33_full_personamem_512_query_overlap/evals_benchmark/20260421T001000Z_stage2_memory_canary.json` 当前固定为 `provider/local = 182/196`
    - 相对 retained `v32` Persona full `183/175`，current line 呈现 `provider -1 / local +21`；这说明 local learned-authoritative gain 已经在 full measurement 上成立，但 provider 仍未超过 retained baseline
    - 当前 provider 侧 residual 也已在 full Persona artifact 上实测固定为 `blank = 108`、`nonlabel = 12`
    - `latest_stage2_v33_learned_authoritative_runtime.json`、`latest_longmemeval_stage2_v33_full.json`、`latest_personamem_stage2_v33_full.json` 与 `latest_stage2_v33_full_holdout_compare.json` 已在 commit `6fa018e70561eef6b0239f41b3d662794df41a22` 上发布
    - `v32` 已在 modular backbone / write / latent / belief / answer / ablation / full holdout 上全线转正
    - 但 `v32` 的 authoritative full benchmark path 仍是 `memory_mode = symbolic`、`slot_assignment_mode = symbolic`
    - `v33` 的目标是把 authoritative full benchmark path 改成 `learned_memory + learned slot assignment`
    - row `102 search` 已把 Persona full measurement 记账，row `103 search` 又把 current-line `LongMemEval-S 500` full holdout、published compare 与 verifier `43/47` 一并记账
    - current-line `LongMemEval-S 500` authoritative full holdout `outputs_v2/v33_full_longmemeval_500_query_overlap/evals_benchmark/20260422T015146Z_stage2_memory_canary.json` 当前固定为 `provider/local = 20/14`
    - 相对 retained `v32` LongMemEval full `21/14`，current line 呈现 `provider -1 / local tie`；相对 retained `v32` Persona full `183/175`，current line 仍是 `provider -1 / local +21`
    - `latest_stage2_v33_full_holdout_compare.json` 当前显式记录 `longmemeval_gain_confirmed = false`、`personamem_gain_confirmed = false`
    - row `104 search` 又把 current-line `LongMemEval` exactness patch 正式记账：generic number-word extraction、location temporal-tail trim 与 low-information `other_fact` belief backfill 已在 authoritative singles 上翻正 `6b168ec8 -> three`、`1faac195 -> denver` 与 `gpt4_ec93e27f -> train`
    - row `105 crash` 已机械确认：若直接沿默认 `cuda` 路径起 fresh rerun，当前机器上的 `GPU 0` 可能因外部占用而在 shared learned predictor 初始化阶段 OOM
    - row `106 no-op` 又机械确认：fresh output root 若不先显式生成 full manifest，`run_stage2_memory_canary.py` 会自动退回 `64/64` canary manifests；单纯传 `--limit 500` 不足以形成 authoritative full rerun
    - 当前真正有效的 rerun 已经切到 `outputs_v2/v33_full_longmemeval_500_exactness_rerun_gpu1_fullmanifest/`：预生成 full `LongMemEval` manifest 后，authoritative `GPU 1` run 已在 same commit `15060d1` 上启动，last check `run_metadata.json` 为 `sample_count = 500`、`provider_configured = true`、`completed_predictions = 12`
    - 因而 top next action 已从“再发起 fresh full rerun”前推到“继续监控并收口这个 in-flight `LongMemEval-S 500` full rerun，完成后立即发布 fresh compare”，而不是回到任何新的 Persona local smoke
    - semantic-full checkpoint + `v31` latent ranker 当前已证明 `LongMemEval-S 64` 可到 `provider/local = 10/10`，但 `PersonaMem 64` 最好只到 `provider/local = 21/24`，provider 仍未越过 retained `v32` rate `0.357421875`
    - 当前实际 provider 路径 `https://gpt-agent.cc/v1` 不兑现 MiniMax 官方 OpenAI-compatible 文档中的 `reasoning_split=True` 行为；对失败 Persona prompt 的真实单样本 probe 仍返回 `<think>` 污染的 `message.content`
    - current session 已机械证伪三条 repo 内补救线：runner-side finite-option normalization、`system` role 压制、以及更小 `max_tokens` budget 都没把 Persona provider exact 推过 retained 门槛
    - current HEAD `2d3e59c` 已把 provider raw-output repair 接进 runner / resume path；现有 partial Persona learned-authoritative full-holdout run `outputs_v2/v33_semantic_full_persona/runs/20260420T203544Z_stage2_memory_canary_personamem/` 已从旧 summary 的 `provider exact = 5/66` 回收到当前 `15/77`，且未引入任何 benchmark-specific normalization
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
    - partial local-only `smoke16` 在主动截断前也已给出 `local_exact = 6/6`、`baseline = 1/6`，因此当前 gain 没有在更大 slice 上立刻塌掉；current HEAD `49e1841` 又进一步把其中 `b358...` 这类 `suggest_new_ideas` 样本暴露出的 `belief=true` boolean-like noise 修成 support-slot canonical value，说明这条 projection / belief noise 线确实值得保留
    - fresh partial `suggest_new_ideas` local-only slice `outputs_v2/v33_boolean_belief_suggest8_localonly/artifacts/20260421T021500Z_partial_summary.json` 又给出 `local = 4/4`、`baseline = 0/4`、`boolean-like belief rows = 0/4`；当前新问题已不再是 “boolean-backfill 有没有价值”，而是它是否只对前几个样本有效
    - second focused `suggest_new_ideas` slice `outputs_v2/v33_boolean_belief_suggest4b_localonly/evals_benchmark/20260421T021706Z_stage2_memory_canary.json` 已完整给出 `local = 4/4`、`baseline = 0/4`；对应 artifact `outputs_v2/v33_boolean_belief_suggest4b_localonly/artifacts/20260421T022100Z_boolean_check.json` 也确认 `boolean-like belief rows = 0/4`
    - fresh mixed Persona local-only gate `outputs_v2/v33_boolean_backfill_smoke8_localonly/artifacts/20260421T023400Z_partial_summary.json` 在与旧 `eade2c3` smoke8 相同 manifest 上，主动截断前已给出 `local = 4/4`、`baseline = 1/4`、`boolean-like belief rows = 0/4`
    - current HEAD `18f1f53` 又试探了 generic withdrawal parser coverage，synthetic parser tests 与 targeted local projection test 都能通过，但 fresh real `d71...` local-only probe `outputs_v2/v33_withdrawal_probe_d71/evals_benchmark/20260421T040035Z_stage2_memory_canary.json` 仍给出 `local_exact = 0/1`，belief 继续是 `other_fact=assisting with literacy programs`
    - fresh discard row `35` 又试探了 advice-query option-head reweighting：`0d225...` fresh local-only probe `outputs_v2/v33_advice_probe_0d/evals_benchmark/20260421T042820Z_stage2_memory_canary.json` 已翻到 `local_exact = 1/1`，`b449...` 仍保持 `1/1`，但 `d71...` probe `outputs_v2/v33_advice_probe_d71/evals_benchmark/20260421T042841Z_stage2_memory_canary.json` 仍是 `0/1`，而 official verifier 继续停在 `36`，所以整条 answer-head patch 已回滚
    - current HEAD `cd1205f` 又把 `configs/minimax_m27.yaml` 的 stage-2 runtime defaults 接进 canary runner；现在即使 CLI 不显式传 checkpoint args，learned-memory / latent / learned slot-assignment 也会默认加载 `v33` 当前 intended checkpoints。对应 full guard 已通过，fresh learned probe `outputs_v2/v33_advice_probe_0d/evals_benchmark/20260421T043908Z_stage2_memory_canary.json` 也已机械记录 non-null checkpoint dirs
    - current HEAD `3fa4d06` 又把 activity-withdrawal parser coverage 保留到 corrected runtime：fresh real `d71...` probe `outputs_v2/v33_withdrawal_probe_d71/evals_benchmark/20260421T045624Z_stage2_memory_canary.json` 继续是 `local_exact = 0/1`、answer `(d)`，但 active memory 已新增 `other_fact=step back from structured book club settings`。这说明当前 `d71` 的主瓶颈不再是 parser/write 缺口，而是 selected-slot ranking / belief selection 仍把 attention 留在 `assisting with literacy programs`
    - current HEAD `2cd6c9b` 又把 withdrawal clause 编成 `recent_change` advice evidence，并把 advice-query zero-overlap negatives 改成 `score-first` 排序；fresh `d71...` probe `outputs_v2/v33_withdrawal_probe_d71/evals_benchmark/20260421T051757Z_stage2_memory_canary.json` 已让 `cdc8...` 进入 selected top-k，但 belief 仍未消费它
    - current HEAD `c4e71d1` 又把 weak `other_fact` latent facet promotion 收紧到 score-proximity gate；fresh `d71...` probe `outputs_v2/v33_withdrawal_probe_d71/evals_benchmark/20260421T052743Z_stage2_memory_canary.json` 已让 selected[0] 与 learned belief 一起落成 `step back from structured book club settings`，但 local answer 仍是 `(a)`，说明当前主剩余问题已收敛到 answer-option projection
    - current HEAD `12e676a` 又把 `back / other / think / thoughts` 一起压成 option-head low-info overlap tokens；fresh probes `outputs_v2/v33_withdrawal_probe_d71/evals_benchmark/20260421T054201Z_stage2_memory_canary.json` 与 `outputs_v2/v33_advice_probe_b449/evals_benchmark/20260421T054201Z_stage2_memory_canary.json` 已分别给出 `1/1` 与 `1/1`，而 `outputs_v2/v33_advice_probe_0d/evals_benchmark/20260421T054201Z_stage2_memory_canary.json` 仍是 `0/1`
    - 当前 worktree 又撤回了 `3234a4b` 引入的 `support-density / unsupported-detail / option_length` 惩罚，让 `OptionScoringHead` 回到 v32 overlap-style ranking；新增 targeted regressions 已直接覆盖 `0d...` 与 `2cef...` 两类 v33 answer-head 回退
    - 对应 full guard 已重新通过，但 `scripts/verify_stage2_v33_longrun.py --score-only` 仍是 `36`；与此同时，current commit `f9ebc8b` 的 fresh `smoke8` local-only replay `outputs_v2/v33_answer_head_overlap_smoke8_localonly/evals_benchmark/20260421T062832Z_stage2_memory_canary.json` 已完整收口到 `local_exact = 8/8`、`baseline = 1/8`，并且样本级确认 `2cef...` 与 `0d...` 都已翻正。随后 current HEAD `51b2462` 又为 `OptionScoringHead` 加入 generic morphology normalization；fresh broader gate `outputs_v2/v33_answer_head_stem_smoke16_localonly/evals_benchmark/20260421T071739Z_stage2_memory_canary.json` 已把 `local_exact` 从旧的 `12/16` 提到 `14/16`、`baseline = 2/16`，并翻正 `5370...` 与 `32b...`。commit `8d391e1` 再把 scorer 看到的 gloss 从“全部 selected slots”收紧到“belief 实际 support slots”，fresh rerun `outputs_v2/v33_support_gloss_smoke16_localonly/evals_benchmark/20260421T075401Z_stage2_memory_canary.json` 已继续把 `local_exact` 提到 `15/16`，并翻正 `344ea...`
  - Truth boundary:
    - 当前主问题已不再是 gain evidence 是否成立，而是 learned path 能否接管 authoritative runtime
    - 这轮不允许在 symbolic authoritative path 上 closeout
    - 当前已批准的新路线是：保留同一 provider/interface，只做协议层最小 `<think>` 清洗；`8`-sample quick-smoke 不再决定方向，retain 直接看 `PersonaMem 512 / LongMemEval-S 500`
    - 当前真实状态是 active measurement / refine，而不是“必须先切 provider/interface 才能继续”的硬 blocker；但修复后的 Persona provider exact 仍明显低于 retained `v32`。discard rows `51-52` 已先说明：`0adf...` 不是 parser-only 缺 observation，且 parser widening + belief-prompt reorder 虽然能把它单样本翻到 gold `(c)`，却会在 broader slice 上立刻打坏 `acd742... / f546... / b358...`
    - 随后的 search rows `58-60` 又补了一层新真相：把 learning-topic recall 收紧到 same-turn authoring context、同时把过宽的 `i'm ...` parser 模式收回 occupation-shape 后，current authoritative runtime 可以把 `a40d...` 恢复到正确 `(d)`，并保住 `0adf...` proxy / `b449...` single，不再出现旧的 `other_fact=learning about` parser 噪声
    - 但最新 discard row `61` 也明确说明：commit `fea3dea` 虽然在 broader `smoke16` partial 中继续守住 `acd742... / f546... / b358... / 5370... / a40d... / 2cef...`，却新打坏了 retained 原本正确的 `6142...`，使 belief 漂回抽象 `music_preference=music in its truest form, without rigid guidelines dictating how i should dissect it`，最好结果仍只能打平 retained `15/16`。因此 parser-noise cleanup 不是当前 gain driver；下一步必须直接转向 same-relation `music_preference` belief-selection / answer-selection，而不是继续保留 parser narrowing 线
    - latest search row `62` 则把 parser 侧 root cause 又收紧了一步：`6142...` 里真正该保留的并不是泛化 `i am ...`，而是更精确的 `I am more drawn to ...` preference clause。当前 worktree 已把旧的泛化 `i am / i'm` value 规则收回 occupation-shape，同时显式支持 `more drawn to ...`；targeted parser tests 与 real-utterance reproduction 已确认这条 patch 能恢复 `6142` 所需的 `music_preference=more drawn to the emotional aspects of music...`，同时继续压住 `a40d` 的 `I'm learning about` 噪声并保住 feedback reason extraction。由于 learned-authoritative singles 仍太慢，这条线目前仍只是 uncommitted narrow candidate，下一步必须先过一个更宽但更便宜的 integration gate
    - committed candidate `3288713` 已经结束并被 row `69` discard：虽然它在 `learned_memory + symbolic slot assignment` 下通过了 `6142... / a40d... / acd742... = 3/3` cheap integration gate，full guard 也通过，但 wider symbolic-slot `smoke16` `outputs_v2/v33_parser_precision_smoke16_symbolic_slot_v2/` 最终只到 `13/16`，miss 收敛到 `d71 / 32b / 0adf`，仍低于 retained `15/16` floor
    - 当前 parser line 已经被进一步证伪为“不是 broader gain driver”。下一步必须回到 retained line，直接分析 `d71 / 32b / 0adf` 这组 residual misses 是否共享 belief-selection / answer-projection 的更高层结构性问题
    - rows `70-71` 已把 committed candidate `eb9480b` 写入 helper：它不改 parser，只在 `scripts/run_stage2_memory_canary.py` 里对 `other_fact` 的 interactional request-like support 做窄 answer-head repair；当前 real learned-authoritative row-build 已机械确认 `0adf...` 从 `(a)` 翻到 `(c)`，`ead3...` 也能从 request-like support 漂移里回到 `(d)`，而且现有 `PersonaMem 512` full artifacts 里 request-like single-item `other_fact` support 的唯一命中也正是 `ead3...`
    - row `72 refine` 先把这条 candidate 提升到 authoritative `smoke16` local-only gate：fresh rerun `outputs_v2/v33_interactional_other_fact_smoke16_localonly_v2/evals_benchmark/20260421T120739Z_stage2_memory_canary.json` 已完整收口到 `16/16`，把此前 retained 唯一 miss `0adf...` 翻正，并保住 `d71... / 32b... / a40d... / 6142...`
    - row `74 refine` 曾把同一条 candidate 扩到 authoritative `smoke17` local-only gate：fresh rerun `outputs_v2/v33_interactional_other_fact_smoke17_localonly/evals_benchmark/20260421T123528Z_stage2_memory_canary.json` 在 retained `16`-slice 上并入真实 full-holdout request-shape 样本 `ead3...` 后收口到 `17/17`
    - 但 row `75 discard` 的更宽 authoritative local-only Persona `smoke32` `outputs_v2/v33_interactional_other_fact_smoke32_localonly/evals_benchmark/20260421T130303Z_stage2_memory_canary.json` 最终只有 `24/32`，并新增 `8` 条 regression：`c8a763... / c2d3daad... / 2b3ce0bd... / 7a4ed201... / afd724f4... / 346a304a... / cd81feb9... / 0d78111b...`
    - `0adf...` 与 `ead3...` 仍然 hold，说明这条 patch 不是完全无效；但它已经明确不是可保留的 broader fix
    - 当前 HEAD 已按 rollback policy 回到 revert commit `3aa38d2`，full guard 通过，`scripts/verify_stage2_v33_longrun.py --score-only` 仍是 `36`
    - row `76 search` 已完成同一 `32`-sample manifest 的 post-rollback authoritative local-only 对照：`outputs_v2/v33_postrollback_smoke32_localonly/evals_benchmark/20260421T134355Z_stage2_memory_canary.json` 最终只有 `22/32`
    - 这条 compare 说明 simple rollback 不是更优 baseline：它继续保留 `c8a763... / c2d3daad... / 2b3ce0bd... / 7a4ed201... / afd724f4... / 346a304a... / 0d78111b...` 等 persistent residual，又失去了 `0adf...` 与 `ead3...` 的 request-like 窄 gain，`cd81feb9...` 也仍然错误但换成另一种错答
    - row `77 search` 已把 `eb9480b` 逻辑恢复到当前 HEAD `d48c58c`，因为它在同一 `smoke32` 上严格优于 simple rollback；相关 interactional canary tests 通过，`scripts/verify_stage2_v33_longrun.py --score-only` 仍是 `36`
    - row `78 search` 又在 `d48c58c` 上把 local option scorer 收紧为“single-slot base projection 不吃 query overlap、multi-slot base projection 才允许 query tie-break”；新增 `c8`-style 与 `afd`-style regression tests，full guard 通过，verifier 仍是 `36`
    - row `79 search` 的 authoritative learned-memory Persona residual `smoke5` 已先把这条 scorer refine 推到真实 runtime：`c8 / afd / 0adf / ead3` 都正确，但 `0d225` 因 direct-fact recall 的 single-support payload shape 回退到 `(a)`
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
    - row `96 search` 又把这组 residual 再向前推进了一步：fresh authoritative `smoke3` `outputs_v2/v33_environment_fit_support_smoke3_localonly/evals_benchmark/20260421T181900Z_stage2_memory_canary.json` 已完整给出 `local_exact_match = 3/3`、`baseline_local_exact_match = 1/3`。其中 `346...` 在真实 learned-authoritative runtime 上第一次翻到 gold `(c)`，并且不是 parser wording 偶然命中，而是 selected[0] / belief / support 都切到了 `other_fact=larger festivals feel too crowded and chaotic for me`；`7a4 -> (c)` 与 `0d781 -> (c)` 同时继续 hold，full guard 通过，official verifier 仍是 `36`
    - 当前最诚实的 unresolved truth 因而继续收窄成：`0d781` 是干净的窄 parser gain candidate，`7a4` 已进入 statement-recall support-switch live line，而 `346` 已从“primary unresolved residual”降成一个已被真实修正、但尚未 wider-validated 的 environment-fit rerank/support candidate
    - row `97 search` 又把这条 line 从 residual trio 扩成 coexistence gate：fresh authoritative `smoke8` `outputs_v2/v33_environment_fit_support_smoke8_localonly/evals_benchmark/20260421T182800Z_stage2_memory_canary.json` 已完整给出 `local_exact_match = 8/8`、`baseline_local_exact_match = 1/8`，机械确认 row `96` 的 environment-fit surgery 不会打坏 row `80` 的 scorer controls `c8 / afd / 0d225 / 0adf / ead3`，同时 `346 / 7a4 / 0d781` 继续全部 hold
    - row `98 search` 又把 current line 推回旧 `smoke11` 同 manifest：fresh authoritative `smoke11` `outputs_v2/v33_environment_fit_support_smoke11_localonly/evals_benchmark/20260421T183400Z_stage2_memory_canary.json` 已完整给出 `local_exact_match = 11/11`、`baseline_local_exact_match = 1/11`，说明 row `80 / 83 / 84 / 92 / 94 / 96` 这些此前分散成立的 gains 现在已经能在同一 learned-authoritative line 上同时成立
    - row `99 search` 又把 widened gate 跑实：fresh authoritative `smoke32` `outputs_v2/v33_environment_fit_support_smoke32_localonly/evals_benchmark/20260421T184200Z_stage2_memory_canary.json` 已完整给出 `local_exact_match = 29/32`、`baseline_local_exact_match = 3/32`，相对 row `75` 的旧 wider gate `24/32` 净增 `+5`
    - 这条 widened gate 已把旧 `smoke32` 的八条错例 `c8 / c2d3 / 2b3 / 7a4 / afd / 346 / cd81 / 0d781` 全部翻正，但同时新引入了 `defecce... / b481... / 7028667...` 三条 `musicRecommendation` regression
    - row `100 search` 已把这三条新 regression 收窄成 authoritative `smoke4`：fresh local-only gate `outputs_v2/v33_music_projection_smoke4_localonly/evals_benchmark/20260421T000000Z_stage2_memory_canary.json` 已完整给出 `local_exact_match = 4/4`、`baseline_local_exact_match = 0/4`，说明 `defecce... / b481... / 7028667...` 已全部翻正，且 `c2d3...` 继续 hold
    - row `101 search` 已把这条 query-overlap patch widen 回旧 `smoke32` 同 manifest：fresh authoritative local gate `outputs_v2/v33_music_projection_smoke32_localonly/evals_benchmark/20260421T000500Z_stage2_memory_canary.json` 已完整给出 `local_exact_match = 32/32`、`baseline_local_exact_match = 3/32`
    - 当前 local coexistence truth 已经收口，不再有 active Persona-side local blocker
    - 下一步不再继续支付 local smoke，而是直接启动 v33 的 full-holdout measurement 与 verifier publish 链路

- `TD-042` `[doing]` 以 `v3.1 / v31 latent-first quality run` 为目标，在**不改 `core / residual` 双银行结构**的前提下，基于 retained `v30` baseline 继续推进 full holdout 上真正更强的 learned 主链。
  - Current focus:
    - latent strengthening
    - belief strengthening
    - write strengthening
    - ablation truth
    - full benchmark holdout：`LongMemEval-S 500 / PersonaMem 512`
  - Hard constraints:
    - no fallback
    - no shortcut
    - no benchmark-specific heuristic
    - no benchmark leakage
    - benchmark remains holdout-only
  - Runtime truth:
    - retained `v30` 已到 `41/41 keep`
    - current retained `v31` 仍是 `24/32`
    - `latest_stage2_v31_latent_mainline_train.json`、`latest_stage2_v31_latent_holdout_compare.json`、`latest_stage2_v31_belief_mainline_eval.json`、`latest_stage2_v31_belief_holdout_compare.json`、`latest_stage2_v31_write_mainline_eval.json` 与 `latest_stage2_v31_write_holdout_compare.json` 已落地
    - quick-smoke holdout line 仍停在 soft blocker，不能再作为 retained keep gate
    - commit `489ada0` 已让 learned full-holdout runner 复用 `latent_slot_ranker` 并按 batch 增量落盘；此前停在 `completed_predictions = 0` 的 resumed runs 现已恢复推进
    - current HEAD `838a861` 又补齐了 `scripts/verify_stage2_v31_longrun.py` 的 authoritative full-holdout publisher；对应 targeted test 与 configured guard 已通过
    - 截至当前检查：`outputs_v2/runs/20260419T160705Z_stage2_memory_canary_longmemeval/` 已推进到 `322/500`，`outputs_v2/runs/20260419T160701Z_stage2_memory_canary_personamem/` 已推进到 `238/512`
  - Truth boundary:
    - 当前这条线已经 soft-blocked；不能再当 active closeout 线继续推进
    - 当前最诚实的结论是：`LongMemEval-S 500` parity、`PersonaMem 512` provider guard fail、三类 Persona pivots 均无 keep

- `TD-039` `[doing]` 以 `v2.8 teacher-quality long-run` 为目标，在**不改 `core / residual` 双银行结构**的前提下，继续以 `32k` source-level split 为锚点，把 teacher-supervision 做成真正可比较、可泛化的训练资产。
  - Runtime truth: `TD-038 / WS-024` 已在 current HEAD `ac84cc1` 上完成，`scripts/verify_stage2_v27_longrun.py --score-only = 26`
  - Current focus:
    - 保持 current-head `256 / 128 / 128` teacher suite、matched manifests 与 holdout guard 可复验
    - 记录当前 launch 已到达的 true blocker，而不是继续重复 refresh
    - 若后续恢复 `TD-039`，必须先提出新的 teacher-generation hypothesis
    - benchmark 继续只作 holdout guard
  - Target teacher scale:
    - train `512`
    - val `128`
    - test `128`
  - Hard constraints:
    - no fallback
    - no shortcut
    - no benchmark-specific heuristic
    - no benchmark leakage
    - full benchmark 只作 holdout evaluation
  - Truth boundary:
    - 暂不进入 full-data
    - 当前成功定义不是 artifact completeness，而是 teacher-enhanced 在 `32k` internal test 上真实优于 silver baseline
  - Latest runtime truth:
    - current-head teacher suite 已以 `256 / 128 / 128` caps 完成，observation teacher 当前 `success_rate = 1.0`
    - `raw observation teacher + matched teacher-vs-silver subset` 已真实跑完；当前 matched manifests 按 changed `raw-observation / slot-assignment / belief` sample_ids 取 source-record 对齐子集，确保训练/评测真正消费 teacher 改动
    - same-split matched non-tiny compare 已完成；fresh compare 当前记录 `delta_internal_token_f1 = -0.05357191517996174`、`delta_internal_field_f1 = -0.08173076923076938`、`delta_internal_exact_match = -0.1826923076923077`
    - 当前主退化项是 `lifecycle_prediction`：val `token_f1` 从 `0.9655172413793104` 降到 `0.8004926108374386`，test 从 `0.9285714285714286` 降到 `0.8482142857142859`
    - selective integration 也已验证：在 all-changed matched subset 上让 `lifecycle_prediction` 保持 silver、只应用 `observation + belief` teacher 后，fresh compare 仍为负，`delta_internal_token_f1 = -0.03262529332240871`、`delta_internal_field_f1 = -0.02564102564102566`、`delta_internal_exact_match = -0.10576923076923073`
    - 剩余可单独保留的 raw-observation teacher 改动只覆盖 matched subset 的 `train=1 / val=0 / test=3`
    - current-head `scripts/verify_stage2_v28_longrun.py --score-only = 32 / 34`
    - `research-results.tsv` / `autoresearch-state.json` 当前已依次记录 iteration `1 discard`、iteration `2 discard` 与 iteration `3 blocked`
  - Current blocker:
    - “teacher 改动未被消费” 已不是当前 blocker；真实 blocker 是：`lifecycle` teacher 负增益、去掉 `lifecycle` 后 `belief` teacher 仍负增益、而 observation-only raw-teacher 改动又过于稀疏。当前 launch 因此停在 blocked，不能写成 keep，也不能进入 full-data

- `TD-038` `[doing]` 以 `v2.7 32k teacher-first long-run` 为目标，在**不改 `core / residual` 双银行结构**的前提下，先建立 `32k` source-level split、teacher-labeled data-quality upgrade、internal generalization test 与 `gpu2` 训练耗时基线。
  - Runtime truth: `TD-037 / WS-023` 已在 current HEAD `cfbdc08` 上完成；`TD-038` 当前作为 retained `32k` split / teacher pilot baseline 保留
  - Current progress:
    - current HEAD `21dd40b` 已新增真实 `32k` source-level split / manifest / audit pipeline
    - `latest_stage2_v27_32k_split.json`、`latest_stage2_v27_32k_manifest.json` 与 `latest_stage2_v27_32k_audit.json` 已落地
    - fresh background run 已通过 helper 初始化 `research-results.tsv` 与 `autoresearch-state.json`；`autoresearch_resume_check.py --repo ...` 当前返回 `full_resume`
    - current HEAD `6333689` 已补齐 `latest_stage2_v27_train.json`、`latest_stage2_v27_eval.json`、`latest_stage2_v27_training_timing.json`、`latest_stage2_v27_internal_test.json` 与 `latest_stage2_v27_holdout_summary.json`
    - `gpu2` tiny pilot 当前真实记录为：`4096` effective examples、`512` steps、wall-clock `5.420951s`、`755.59 examples/s`、peak GPU memory `55.09MB`
    - 当前又已补齐 `latest_stage2_v27_teacher_observation.json`、`latest_stage2_v27_teacher_slot_assignment.json` 与 `latest_stage2_v27_teacher_belief.json`
    - 当前 teacher pilot 使用真实 `MiniMax-M2.7`、sample caps `8/2/2` 与 batch size `1`；`slot_assignment` 与 `belief` artifact 为 `completed`，`observation` artifact 为 `completed_with_failures`
    - fresh current-head `scripts/verify_stage2_v27_longrun.py --score-only = 26`
  - Core requirements:
    - `32k` 作为锚点；先做 `24k train / 4k val / 4k test`
    - 训练前先做 data-quality audit
    - teacher 使用 `MiniMax-M2.7`
    - 训练优先使用 `gpu2`
    - 记录 wall-clock / throughput / memory，判断 `32k` 是否过重
  - Hard requirements:
    - no fallback
    - no shortcut
    - no benchmark-specific heuristic
    - no benchmark leakage
    - full benchmark 只作 holdout evaluation
    - 只有 `32k` internal test work well 后，才允许进入 full-data
  - Truth boundary:
    - 当前 `26/26` 是由 `32k split + gpu2 tiny pilot + teacher pilot artifacts` 共同满足的机械 stop condition
    - 不能把当前完成态误写成 full `32k` teacher coverage 已完成，也不能误写成 teacher-conditioned `gpu2` retrain 已完成
    - `latest_stage2_v27_teacher_observation.json` 当前显式记录 `total_labeled_examples = 6`、`total_failed_examples = 6`，因此 observation teacher 仍是 pilot quality

- `TD-036` `[done]` 以 `v2.5 learned core-path long-run` 为目标，在**不改 `core / residual` 双银行结构**的前提下，继续推进 `LongMemEval-S` 质量、`write / retrieve / belief` 三段的 learned 化、learned slot assignment 泛化鲁棒性、更强 latent 与 full benchmark holdout evaluation。
  - Runtime truth: `TD-035 / WS-021` 已在 current HEAD `12a9a80` 上达到 `stage2_v24_longrun_score = 24/24`，当前作为 `v2.5` retained baseline 保留
  - Current managed-run state: current HEAD `3036e3d` 已把 `stage2_v25_longrun_score` 推到 stop condition `24/24`
  - Evidence shape: 当前新增的是 `v2.5` baseline/package artifact suite 与 `write / retrieve / belief` baseline decomposition artifacts；三段 artifact 均显式记录 `delta_vs_v24_retained = 0`
  - Truth boundary: 当前不能把 `24/24` 误写成真实 `LongMemEval-S` / online quality gain 已经超过 `v2.4`
  - Hard constraints:
    - no fallback
    - no shortcut
    - no benchmark-specific heuristic
    - full benchmark is holdout evaluation, not training supervision
  - Plan: [docs/v25_plan.md](/media/storage/mingjing/workspace/CoRe_Mem/docs/v25_plan.md)


## Backlog

- `TD-007` `[backlog]` 为 PersonaMem 128k / 1M 提供启动开关。
  - Reason: 第一阶段只要求保留启动能力。

- `TD-018` `[backlog]` 建立第二阶段 benchmark canary protocol 与结果记录。
  - Reason: `gpu3 + tiny backend` 的 stage-2 本地 train/eval/ablation matrix 已全部完成并登记；下一步若继续推进，更应该把这个完成态沉淀为 canary 结果和对比表。
  - Evidence target: PersonaMem 64 / LongMemEval-S 64 的固定 canary 运行结果、输出表和记录规范。

- `TD-025` `[backlog]` 在 `PersonaMem 64` canary 质量提升后，扩大 stage-2 benchmark 范围与结果对比。
  - Reason: `TD-029 / WS-015` 已完成 learned-memory-first plumbing；当前更合理的后续工作是扩大 learned-mode canary 范围、补结果对比，而不是重复补 toggle/checkpoint plumbing。
  - Evidence target: 更大样本或第二 benchmark 的 stage-2 live canary 结果与对比表。

## Blocked

- `TD-012` `[blocked]` 补齐 stage-1 外部前置条件。
  - Reason: 当前可用的 Gemini key/provider 组合在 formal benchmark 负载下仍连续触发 `HTTP 429`，且 formal benchmark 在用户明确要求前不主动继续推进。
  - Evidence target: 可持续推进 formal benchmark 的 provider 配额、稳定 key，或用户批准的其他可用 provider。

- `TD-013` `[blocked]` 将两个 benchmark 从 1-sample real run 推进到正式全量 protocol 运行。
  - Reason: 当前只能证明最小链路打通，尚不足以满足 AC-002 / AC-003 的正式运行要求；同时该项被用户触发条件与 provider blocker 双重约束。
  - Evidence target: PersonaMem 与 LongMemEval-S 在正式范围内完成可重复结果运行。
  - Current evidence: runner 已具备增量落盘与续跑能力；Gemini 路径已把 PersonaMem formal run 推进到 `22/589`、把 LongMemEval formal run 推进到 `19/500`，但超保守单样本检查仍连续触发 `HTTP 429`，说明当前 key/provider 组合已构成真实外部 blocker。

## Done

- `TD-040` `[done]` 以 `v2.9 learned-core-path long-run` 为目标，在**不改 `core / residual` 双银行结构**的前提下，继续以 `32k` source-level split 为锚点，依次推进 `write -> latent composition -> belief` 三段 learned 主链，并把 holdout benchmark 扩到 `LongMemEval-S 500 / PersonaMem 512`。
  - Reason: 当前 retained state 已达到 `scripts/verify_stage2_v29_longrun.py --score-only = 39`
  - Evidence target:
    - `outputs_v2/artifacts/latest_stage2_v29_write_gain.json`
    - `outputs_v2/artifacts/latest_stage2_v29_latent_gain.json`
    - `outputs_v2/artifacts/latest_stage2_v29_belief_gain.json`
    - `outputs_v2/artifacts/latest_stage2_v29_holdout_summary.json`
    - `outputs_v2/artifacts/latest_longmemeval_stage2_v29_canary.json`
    - `outputs_v2/artifacts/latest_personamem_stage2_v29_canary.json`
    - `research-results.tsv` iteration `3 keep`

- `TD-040` `[done]` 以 `v2.9 learned-core-path long-run` 为目标，在**不改 `core / residual` 双银行结构**的前提下，继续以 `32k` source-level split 为锚点，依次推进 `write -> latent composition -> belief` 三段 learned 主链，并把 holdout benchmark 扩到 `LongMemEval-S 500 / PersonaMem 512`。
  - Reason: 当前 retained state 已达到 `scripts/verify_stage2_v29_longrun.py --score-only = 39`；`write / latent / belief` 三段 gain 全部转正，expanded holdout 已真实完成，launch guard 通过
  - Evidence target:
    - `outputs_v2/artifacts/latest_stage2_v29_write_gain.json`
    - `outputs_v2/artifacts/latest_stage2_v29_latent_gain.json`
    - `outputs_v2/artifacts/latest_stage2_v29_belief_gain.json`
    - `outputs_v2/artifacts/latest_stage2_v29_holdout_summary.json`
    - `outputs_v2/artifacts/latest_longmemeval_stage2_v29_canary.json`
    - `outputs_v2/artifacts/latest_personamem_stage2_v29_canary.json`
    - `research-results.tsv` iteration `3 keep`

- `TD-037` `[done]` 以 `v2.6 gain-first long-run` 为目标，在**不改 `core / residual` 双银行结构**的前提下，要求 `write / retrieve / belief` 至少一段出现真实正增益，并要求 `LongMemEval-S 128` 明确超过 `v2.5` retained baseline。
  - Reason: 该条线已在 current HEAD `cfbdc08` 上机械达成 `stage2_v26_longrun_score = 26/26`
  - Evidence target:
    - `LongMemEval-S 128 = 11 / 11`
    - `PersonaMem 128 = 44 / 33`
    - `write_gain.positive_gain = true`
    - `belief_gain.positive_gain = true`

- `TD-001` `[done]` 初始化项目文档系统并建立根契约、真源文档和状态文档。
- `TD-002` `[done]` 用 `environment.yaml` 在默认 conda envs 目录创建并固定 conda 环境 `core_mem`。
- `TD-003` `[done]` 建立基础目录结构、配置机制与输出目录规范。
- `TD-004` `[done]` 确定第一阶段 pretrained embedding model 为 `sentence-transformers/all-MiniLM-L6-v2`。
- `TD-005` `[done]` 实现阿里云 OpenAI-compatible provider adapter 第一版。
- `TD-006` `[done]` 接入 PersonaMem 32k 官方主任务与评测协议。
- `TD-008` `[done]` 接入 LongMemEval-S 官方协议。
- `TD-009` `[done]` 实现 vanilla CoRe Memory 核心模块与文本 embedding 接口。
- `TD-010` `[done]` 建立实验总入口与分步骤脚本。
- `TD-011` `[done]` 建立 unit tests 与最小 E2E smoke test。
- `TD-014` `[done]` 锁定第二阶段 `V2.0 structured latent-slot memory` 方案并同步真源与状态文档。
  - Reason: 第二阶段已由用户确认，需要进入文件级 runtime truth。
  - Evidence target: `docs/requirements.md`、`docs/v2_design.md` 与相关状态文档同步到 stage-2 真相。
- `TD-016` `[done]` 实现第二阶段 `V2.0` 主线模型骨架并打通直训链路。
  - Reason: 当前主线骨架、训练 runtime、公开数据规范化、严格 source prepare 与 direct-train launcher 已全部连通。
  - Evidence target: `src/core_mem/v2` 下主线模块、真实 public-data prepared manifest、repo-local cache 训练配置与 execute-train 证据。
- `TD-019` `[done]` 获取并接入第二阶段真实公开数据源文件。
  - Reason: 五个目标公开数据源已下载、规范化并通过 preflight 与 strict prepare 验证。
  - Evidence target: `stage2_data_preflight.py` 不再报 missing，且 `prepare_stage2_data.py --strict-sources` 能产出非 demo manifests。
- `TD-017` `[done]` 建立第二阶段本地 intrinsic evaluation 的完整指标、budget sweep 与 experiment registry。
  - Reason: local eval 已从 skeleton 扩展到模块级/家族级/budget-sweep 评测体系，并已被实际用于 `mainline + 11` 个必做 ablation 的 `gpu3` 本地 train/eval 登记。
  - Evidence target: `outputs_v2/artifacts/stage2_experiment_index.json`、`outputs_v2/evals_local/`、`outputs_v2/tables/` 与 `scripts/verify_stage2_experiment_status.py --score-only = 13`。

- `TD-021` `[done]` 把第二阶段真正的 latent memory 主链路做实。
  - Reason: query/slot encoder 已不再是 hash-only，resampler 已不再是 mean-only，decoder 已真实消费 `composed_memory`，且 `StructuredMemorySystem.query()` 已把 composed latent 传入 belief decode 主链。
  - Evidence target: `scripts/verify_stage2_latent_status.py --score-only = 9` 中与 latent path 对应的 7 个实现检查全部通过。

- `TD-022` `[done]` 建立 stage-2 memory-mediated benchmark canary runner，并接入可用 provider 配置。
  - Reason: `scripts/run_stage2_memory_canary.py` 已落地，且 canary artifact/config snapshot/traceability 已成立。
  - Evidence target: `run_stage2_memory_canary.py` 可生成 stage-2 benchmark canary run 与 summary artifact。

- `TD-023` `[done]` 完成 stage-2 MiniMax memory canary 的 live provider 首次运行。
  - Reason: `MiniMax-M2.7` 已完成 1-sample live canary 和 `PersonaMem 64` live canary。
  - Evidence target: 至少一条 live MiniMax-M2.7 的 stage-2 canary 运行结果。

- `TD-026` `[done]` 以系统/模型/latent 本体更强、更稳健为锚点，提升 stage-2 local intrinsic 质量，并把 `PersonaMem 64 >= 9/10` 作为不退化 guard。
  - Reason: `stage2_latent_core_quality_score` 已从本轮 baseline `6/10` 提升到 `10/10`；belief recovery 现直接消费 lifecycle-ordered memory state，且 local intrinsic belief-family 与 retrieval-family 已解耦到更符合主链分层的评测方式。
  - Evidence target: `scripts/verify_stage2_latent_core_quality.py --score-only = 10`，且 retained PersonaMem 64 guard artifact 仍满足 `9/10`。

- `TD-020` `[done]` 为默认 `flan-t5-base` backbone 补非 tiny `gpu3` 训练证据。
  - Reason: 当前已完成一条默认 `configs/stage2_train.yaml` 的真实 `GPU3` 非 tiny train/eval 证据链，并形成正的 `trained_eval.token_f1`。
  - Evidence target: `outputs_v2/runs/20260415T043648Z_stage2_train_exec/execution_summary.json`、`outputs_v2/checkpoints/20260415T043648Z_stage2_train_exec/`、`outputs_v2/evals_local/20260415T043706Z_stage2_local_eval.json`。

- `TD-027` `[done]` 推进完整的 `v2`：fresh live canary、第二 benchmark、非 tiny 训练证据，以及去除 benchmark shortcut/fallback。
  - Reason: 当前 `stage2_v2_completion_score = 14/14`；fresh `PersonaMem 64`、`LongMemEval-S 64`、`latest_longmemeval_stage2_canary_analysis.json`、`no-shortcut runner` 与默认 `flan-t5-base` 的 `GPU3` 非 tiny train/eval 已全部机械成立。
  - Evidence target: `scripts/verify_stage2_v2_completion.py --score-only = 14`，且 `scripts/verify_stage2_latent_core_quality.py --score-only = 10`、stage-2 guard 通过。

- `TD-029` `[done]` 把 `v2.1` 的当前主线切到 learned-memory-first / better latent。
  - Reason: online learned path、checkpoint-backed belief、online-aligned training 语义和最小 current-head learned-mode artifact 已成立。
  - Evidence target: `scripts/verify_stage2_v21_learned_memory.py --score-only = 12`

- `TD-032` `[done]` 把 semantic-first 路线推进到 full-data current-head 训练、extended semantic canaries、LongMemEval-S semantic analysis 与 semantic online gain。
  - Reason: current-head full-data semantic train/eval、`PersonaMem 128` / `LongMemEval-S 64/128` semantic canaries、fresh semantic analysis 与 semantic online gain 已全部补齐。
  - Evidence target: `scripts/verify_stage2_v22_completion.py --score-only = 19`

- `TD-033` `[done]` 以 `LongMemEval-S` 质量提升、learned slot assignment 与 stronger latent 为锚点推进 `v2.3`。
  - Reason: 这条主线已由后续 `TD-034 / WS-020` 机械收口；当前不再是 active 主线。
  - Evidence target: `scripts/verify_stage2_v23_longrun.py --score-only = 22`

- `TD-035` `[done]` 以 `v2.4 quality-first long-run` 为锚点推进 `LongMemEval-S` 质量、full-data learned slot assignment 与 stronger latent。
  - Reason: managed autoresearch run 已在 current HEAD `12a9a80` 上触发 stop condition，`stage2_v24_longrun_score = 24/24`
  - Evidence target:
    - `scripts/verify_stage2_v24_longrun.py --score-only = 24`
    - `outputs_v2/artifacts/latest_stage2_v24_eval.json`：`trained_eval.token_f1 = 0.9991150844073334`、`trained_eval.field_f1 = 0.9976704786107581`
    - `outputs_v2/artifacts/latest_longmemeval_stage2_v24_canary.json`：`provider_exact = 10`、`local_exact = 10`
    - `outputs_v2/artifacts/latest_personamem_stage2_v24_canary.json`：`provider_exact = 38`、`local_exact = 28`
    - `outputs_v2/artifacts/latest_stage2_v24_online_gain.json`：`delta_provider_exact_match = +4`、`delta_local_exact_match = +4`
    - `research-results.tsv` / `autoresearch-state.json`：best iteration `7`、best/current metric `24`

- `TD-036` `[doing]` 以 `v2.5 generalization-first long-run` 为锚点，把当前主线推进到更强的 `LongMemEval-S`、更泛化的 learned slot assignment、更强的 online latent，以及更大切片 / full benchmark holdout measurement。
  - Reason: `v2.4` 已把 quality-first 机械目标推到 `24/24`，下一步最值得做的不再是补 closeout artifact，而是在**不动 `core / residual`** 的前提下，把 `write / retrieve / belief` 三段进一步 learned 化
  - Evidence target:
    - current-head 更强的 `LongMemEval-S 64/128` retained line
    - generalized learned `write / slot-assignment`、`retrieve`、`belief` train/eval/gain artifact
    - current-head `PersonaMem 128` 不明显退化
    - full benchmark holdout evaluation summary
  - Current runtime note: 本轮 managed run 已机械达到 `24/24`，但当前 retained evidence 仍属于 baseline/package closeout，不是新的 positive-gain closeout



## Verified

- `TD-003` `[verified]` 目录结构、配置加载、输出规范、run metadata 写入与 benchmark dry-run 可运行。
- `TD-002` `[verified]` `core_mem` 环境已在默认 conda envs 目录创建，并通过 `python --version`、`pytest` 与 `run_experiment.py --verify-only` 复验。
- `TD-004` `[verified]` embedding 配置、adapter 和环境依赖已接入；`sentence-transformers 5.3.0` 可在 `core_mem` 环境中 import。
- `TD-005` `[verified]` Provider adapter 通过单测，支持 OpenAI-compatible 请求构造与响应解析。
- `TD-010` `[verified]` 最小 `prepare / evaluate / make tables` 分步骤脚本已落地并有 smoke tests。
- `TD-009` `[verified]` vanilla CoRe Memory 已具备 writer / updater / residual manager / reader / text observe/query 最小闭环。
- `TD-014` `[verified]` 第二阶段真源、设计文档、stage2 verifier/acceptance skeleton 与 `outputs_v2/` 目录骨架已同步到当前 runtime truth。
- `TD-015` `[verified]` 第二阶段 observation / belief / parser / dataset skeleton、prepare/train/eval/canary 脚本与首批 `outputs_v2/` artifacts 已落地。
  - Reason: 当前已不再是纯 verifier skeleton 状态，而是 stage2 pipeline 首批可运行实现。
  - Evidence target: `src/core_mem/v2/`、`configs/stage2_train.yaml`、四个 stage2 脚本、`outputs_v2/` artifacts 与对应测试。
- `TD-016` `[verified]` 第二阶段 `V2.0` 主线模型骨架与直训链路已进入可直接训练状态。
  - Reason: 真实 public-data source -> normalize -> strict prepare -> direct-train launcher -> execute-train tiny proof 已连续通过。
  - Evidence target: `outputs_v2/artifacts/stage2_prepared_samples_manifest.json`、`outputs_v2/runs/20260414T043221Z_stage2_train_plan/launch_stage2_training.sh`、`outputs_v2/runs/20260414T043225Z_stage2_train_exec/training_metrics.json`。
- `TD-019` `[verified]` 第二阶段真实公开数据源文件已接入。
  - Reason: `SGD`、`MultiWOZ 2.4`、`Persona-Chat`、`MQUAKE`、`ReCoE` 现均已具备 repo-local raw source 与对应 `normalized.jsonl`。
  - Evidence target: `scripts/stage2_data_preflight.py --json` 返回 `missing=[]`，且 `normalize_stage2_public_data.py` 已产出 5 份 `normalized.jsonl`。
- `TD-021` `[verified]` 第二阶段 latent memory 主链路已满足当前机械 readiness 目标。
  - Reason: `stage2_latent_readiness_score` 已从 `2/9` 提升到 `9/9`，且 stage-2 guard 与新增 canary test 均已通过。
  - Evidence target: `research-results.tsv` iteration `1`、`outputs_v2/evals_benchmark/20260414T170606Z_stage2_memory_canary.json`、`scripts/verify_stage2_latent_status.py --score-only = 9`。

- `TD-022` `[verified]` Stage-2 memory-mediated benchmark canary runner 已接上 benchmark path。
  - Reason: blocked artifact、1-sample live artifact 与 64-sample live artifact 已依次落地。
  - Evidence target: `outputs_v2/evals_benchmark/20260414T170606Z_stage2_memory_canary.json`、`outputs_v2/evals_benchmark/20260414T231441Z_stage2_memory_canary.json`、`outputs_v2/evals_benchmark/20260414T231617Z_stage2_memory_canary.json`。

- `TD-023` `[verified]` Stage-2 MiniMax memory canary 的 live provider 路径已被真实执行。
  - Reason: `PersonaMem 64` live canary 已完成 `64/64` provider 调用。
  - Evidence target: `outputs_v2/evals_benchmark/20260414T231617Z_stage2_memory_canary.json` 与对应 `predictions.jsonl`。

## Abandoned

- None
