# Acceptance Report

## Stage 2 V6.5 Facetized Observation-to-Memory Evidence

- `EV-V65-001` -> `TD-054 / WS-040` facetized observation-to-memory redesign
  - Status: partial / negative_result
  - Evidence:
    - `scripts/verify_stage2_v65_facetized_memory.py --score-only` 当前返回 `85`
    - `latest_stage2_v65_facetized_memory_eval.json` 记录 full-context `37` shared contexts 下 authoritative facetizer、`8` 类 facet schema、`facet_candidate_count = 4834` 与 `facet_rerouted_actions = 1014`
    - `latest_stage2_v65_persistent_state.json` 记录 persistent `core_bank_size = 691`、`residual_bank_size = 1732`、`stream_observations_written = 3776`、checkpoint 与 write trace
    - `latest_stage2_v65_internal_eval.json` 记录 `facet_aware_write_utility_used = true`，raw-context retrieval disabled 继续成立
    - `latest_stage2_v65_personamem_no_routing.json` 记录 answer-time routing disabled、PersonaMem gold 未用于 facetizer 或 substrate
    - `research-results.tsv` iteration `2 discard` 已记录 cue-anchored phrase retention trial：局部把 `needed_facet_missing` 改善到 `448`、把 PersonaMem full589 no-routing 改善到 `138/589`，但 verifier 仍停在 `85`，因此该 trial 未进入 retained state
    - `research-results.tsv` iteration `3 refine` 已记录 facet-type glossing + salient phrase selection trial：当前 selected slots 已主要呈现为 `preference_target=... / environment_aversion=... / update_reason=...`，`needed_facet_missing` 进一步降到 `434`，PersonaMem full589 no-routing 升到 `150/589`
    - `research-results.tsv` iteration `4 discard` 已记录 decision-feature patch：直接把 `selected slot canonical_gloss` 接入 option features 只把 `needed_facet_missing` 从 `434` 微调到 `433`，未能提升 PersonaMem no-routing 或 verifier
    - `research-results.tsv` iteration `5 discard` 已记录 reader-metadata patch：facet metadata 与 environment-query alignment 反而把 PersonaMem full589 no-routing 打回 `128/589`，并把 `needed_facet_missing` 恶化到 `456`
    - `research-results.tsv` iteration `6 pivot` 已记录当前 local feature surgery family结束，下一步改 broader question-conditioned supervision
    - `research-results.tsv` iteration `7 discard` 已记录 option-conditioned decision readout：base question readout + per-option reread 联合训练/推理会把 PersonaMem full589 no-routing 从 `150/589` 打回 `142/589`，并把 `needed_facet_missing` 恶化到 `441`
    - `research-results.tsv` iteration `8 discard` 已记录 intent-conditioned facet readout：query-intent facet priors 与 duplicate-facet crowding penalty 会把 PersonaMem full589 no-routing 从 `150/589` 打回 `147/589`，并把 `needed_facet_missing` 恶化到 `436`
    - `research-results.tsv` iteration `9 pivot` 已记录当前 broader supervision 子 family 继续失败；下一步不再直接改 slot-selection heuristics
    - `research-results.tsv` iteration `10 search` 已记录两条主来源启发：DST-as-reading-comprehension 的 question-conditioned slot QA 视角，以及 MMR 的 relevance + novelty shortlist 视角
  - Negative result:
    - `needed_facet_missing_count = 434 > parser_only 409`
    - `wrong_sibling_facet_selected_count = 0`
    - `facet_written_but_reader_missed_count = 5`
    - PersonaMem full589 no-routing `150/589`
    - text-only `234/589`
    - option-only `235/589`
    - margin vs text-only `-84`
    - margin vs option-only `-85`
  - Boundary:
    - 这一轮只证明 v6.5 的 explicit facet schema、authoritative facetizer、facet-aware write utility 与 facet-level attribution 已经进入真实 memory path。
    - latest discard 说明：短语级去重/保留可以改善局部 failure bucket，但还不足以跨过 verifier 的 retained gate。
    - latest refine 又说明：即使 `facet_type` 已真正进入 selected gloss 语义，当前 gain 仍没有自动传到 verifier，下一步必须继续打 reader / decision 消费层。
    - latest discard `4` 则进一步说明：当前 reader / decision 层的问题也不是“少接了一条 selected-gloss overlap feature”这么简单。
    - latest discard `5` 又把边界再压清一层：reader facet-metadata alignment 不是当前 gain driver，继续沿这条 family 细抠只会回退。
    - latest discard `7` 继续把边界压清：简单的 option-conditioned reread 也不是当前 gain driver，它会把选项文本过早耦合进 slot selection，结果比 restore 后的 refine line 更差。
    - latest discard `8` 又继续把边界压清：即使把 question intent 直接转成 facet prior 和重复 facet crowding penalty，也还是会在 authoritative full-context publish 上回退，说明当前缺口不是再多一点 shortlist heuristic。
    - latest pivot/search 说明下一步应优先试更 principled 的 question-conditioned facet supervision 或 diversified shortlist，而不是继续做 hand-tuned slot-selection bias。
    - 当前不能声明 `needed_facet_missing` reduction、benchmark gain、acceptance met 或 closeout。
    - 下一步必须直接修更广的 question-conditioned reader / decision 对 facet support 的消费，而不是把当前 partial、latest discard 或 latest refine 写成成功。

## Stage 2 V6.4 Learned Observation Proposal Evidence

- `EV-V64-001` -> `TD-053 / WS-039` learned observation proposer / hybrid candidate pool
  - Status: partial / negative_result
  - Evidence:
    - `scripts/verify_stage2_v64_observation_proposal.py --score-only` 当前返回 `75`
    - `latest_stage2_v64_observation_proposal_train.json` 记录 learned observation proposer、write-worthiness、attribute-validity、write router、learned reader、learned decision 全部已训练
    - `latest_stage2_v64_observation_proposal_eval.json` 记录 authoritative learned proposer、hybrid candidate pool、dedup / normalization / validation，以及 `proposal_recall = 0.08488964346349745 > parser_only 0.015280135823429542`
    - `latest_stage2_v64_persistent_state.json` 记录 persistent `core_bank_size = 726`、`residual_bank_size = 960`、`stream_observations_written = 3415`、checkpoint 与 write trace
    - `latest_stage2_v64_personamem_no_routing.json` 记录 answer-time routing disabled、raw-context retrieval disabled、PersonaMem gold 未用于 proposer 或 substrate
    - `research-results.tsv` iteration `1 keep` 已记录 `persistent-bank / write-routing / learned-reader-authoritative / learned-decision-authoritative / learned-observation-proposer-authoritative / hybrid-candidate-pool / proposal-recall-positive / dedup-normalization-validation / no-answer-routing / no-gold-leakage`
  - Negative result:
    - full589 `never_written_count = 442 > parser_only 409`
    - support recovery `0.24957555178268254 < parser_only 0.30560271646859083`
    - PersonaMem full589 no-routing `138/589`
    - text-only `199/589`
    - option-only `235/589`
    - margin vs text-only `-61`
    - margin vs option-only `-97`
    - discarded same-relation novelty gate + top-1 learned budget degraded full589 no-routing to `97/589` and worsened `never_written` to `481`
    - discarded prompt/question filtering + relation-aware value compression cleaned learned candidates but still left full589 at `135/589` with `never_written = 445`
    - discarded length-aware learned confidence calibration only reweighted long learned-only values, but full589 still regressed to `never_written = 455` and no-routing `130/589`, showing that mild proposer-side confidence shaping alone does not remove the write-path bottleneck
    - discarded relation-aware fragment extraction + rule-preferred merge preservation improved full589 `never_written` to `427` and no-routing to `155/589`, but still failed to beat parser baseline `409` or lift verifier above `75`
    - discarded candidate provenance metadata preservation + provenance-aware reader prior regressed full589 to `133/589` and `never_written = 448`, showing that reader-side provenance bias alone does not remove the write-path bottleneck
    - discarded provenance-aware pre-write features + write-policy heuristics only nudged full589 to `never_written = 441` and no-routing `140/589`, while text-only also rose to `202/589`, so the verifier remained `75`
    - discarded value-aware persistent slot matching expanded the banks to `core=1402 / residual=1866` but regressed full589 to `never_written = 445`, `written_but_reader_missed = 24`, and no-routing `120/589`, showing that naive anti-overwrite widening can bloat the state without preserving usable support
    - discarded relation-specific sibling utility competition improved full589 to `never_written = 420`, `no-routing = 161/589`, and text-only `195`, but still failed the parser-baseline and significant-margin gates
    - discarded refined abstract-clause demotion produced the strongest current near-miss at `never_written = 413` and `no-routing = 171/589`, but still did not beat parser baseline `409`
    - discarded hobby-misroute demotion regressed the same family back to `never_written = 423` and `no-routing = 159/589`, showing that the new strategy should stay narrower than a broad hobby cleanup bundle
    - discarded narrower sibling-only abstract-clause demotion plus overwrite protection only triggered `4` demotions and regressed full589 to `never_written = 450` and no-routing `130/589`, showing that turn-local sibling gating is too narrow to preserve the iteration `11` near-miss
    - discarded carryover-vs-update overwrite gate redirected blocked low-reliability updates to `new_residual`; full589 improved `never_written` to `431` but exploded the state to `core=420 / residual=2140` and `written_but_reader_missed = 18`, so the gain did not survive as usable support
    - discarded carryover-vs-update `ignore` redirect improved the same family to `never_written = 434` and PersonaMem `151/589`, but it still stayed below the retained line and parser baseline
    - discarded overwrite-only carryover gate regressed the same family back to retained-level truth (`never_written = 442`, PersonaMem `138/589`), showing that local overwrite gating does not unlock the missing v6.4 labels
  - Boundary:
    - 这一轮证明 v6.4 的 authoritative learned proposer、hybrid candidate pool、proposal recall gain、persistent banks、write trace、learned reader/decision 与 no-answer-routing / no-gold structure 已成立。
    - 当前不能声明 `never_written` reduction、support-coverage gain、benchmark gain、acceptance met 或 reproducible closeout。
    - proposer-threshold tightening、extraction-only cleanup、provenance-aware bias、naive anti-overwrite widening、过宽的 `hobby` misroute cleanup、turn-local sibling-only gating，以及 carryover-vs-update overwrite redirects 都已出现失败证据；当前 run 已在第 `3` 个 pivot 进入 soft-blocker handoff。

## Stage 2 V6.3 Confidence-Aware Write-Policy Evidence

- `EV-V63-001` -> `TD-052 / WS-038` recall-preserving confidence-aware write policy
  - Status: partial / negative_result
  - Evidence:
    - `scripts/verify_stage2_v63_write_policy.py --score-only` 当前返回 `85`
    - `latest_stage2_v63_write_policy_train.json` 记录 learned write-worthiness、attribute-validity、write router、learned reader、learned decision 与 `write_policy / confidence_aware_write / weak_keep_policy` 全部已训练
    - `latest_stage2_v63_write_policy_eval.json` 记录 authoritative four-way policy、`support_coverage_recall = 1.0` > disabled `0.0625`、`write_recall = 1.0` > disabled `0.0769`、`policy_labels = [core-worthy, residual-worthy, weak-but-keep, drop]`
    - `latest_stage2_v63_persistent_state.json` 记录 non-collapse `core_bank_size = 22`、`residual_bank_size = 476`、`stream_observations_written = 652`、checkpoint、write trace 与 `weak_but_keep_residual_count = 12`
    - `latest_stage2_v63_internal_eval.json` 记录 authoritative reader path `learned_query_slot_reader`、raw-context retrieval disabled、typed residuals 与 `full_accuracy = 0.375`
    - `latest_stage2_v63_personamem_no_routing.json` 记录 answer-time routing disabled、raw-context retrieval disabled、PersonaMem gold 未用于 memory substrate
    - `latest_stage2_v63_error_attribution.json` 记录 full589 failure attribution counts
  - Negative result:
    - PersonaMem full589 no-routing `180/589`
    - text-only `180/589`
    - option-only `235/589`
    - margin vs text-only `0`
    - margin vs option-only `-55`
    - `latest_stage2_v63_decision.json` 记录 `result_type = negative_result`
    - discarded durable-fact core-promotion trial degraded to `152/589`
    - discarded selected-support-gloss decision-feature trial degraded to `173/589`
    - discarded state-anchored support-supervision expansion degraded to `144/589`
    - discarded scenario-aware reader-semantics trial degraded to `162/589`
    - discarded question-conditioned belief-selector trial degraded to `176/589`
    - discarded query-relation-router trial degraded to `163/589`
    - discarded relation-pooled belief-composer trial degraded to `160/589`
    - discarded late-interaction reader-features trial degraded to `156/589`
  - Boundary:
    - 这一轮证明 v6.3 的 authoritative four-way write policy、weak-but-keep residual retention、support-coverage gain、non-collapse state 与 error attribution artifact 已成立。
    - 当前不能声明 benchmark gain、significant PersonaMem margin、acceptance met 或 reproducible closeout。
    - 第三次 `pivot` 已成立，当前 run 进入 soft blocker handoff：search-derived 的 pooled-belief 与 late-interaction 两条结构线都没有形成 retainable gain。
    - 若后续还要继续推进，必须比现有 question-conditioned read-path、relation-pooling 和 late-interaction reader 更广。

## Stage 2 V6.2 Learned Write-Quality Evidence

- `EV-V62-001` -> `TD-051 / WS-037` learned write-worthiness / attribute-validity before write
  - Status: partial / negative_result
  - Evidence:
    - `scripts/verify_stage2_v62_write_quality.py --score-only` 当前返回 `85`
    - `latest_stage2_v62_write_quality_train.json` 记录 message-level learned write-worthiness、observation-level attribute-validity / relation-validity、write-time router、learned reader、learned decision 全部已训练
    - 同一 train artifact 记录 `write_worthiness_eval_accuracy = 0.8890` > disabled `0.8834`，`attribute_validity_f1 = 0.9728` > disabled `0.9203`
    - `latest_stage2_v62_write_quality_eval.json` 记录 authoritative write path 使用 learned write-worthiness / attribute-validity，raw-dialogue hard negatives `68`、conflict negatives `382`，且 held-out `write_worthiness_eval_accuracy = 0.90625` > disabled `0.8984375`、`attribute_validity_f1 = 0.9865` > disabled `0.9289`
    - `latest_stage2_v62_persistent_state.json` 记录 persistent `core_bank_size = 4`、`residual_bank_size = 42`、checkpoint、write trace，以及 `bank_precision_estimate = 1.0`、`invalid_slot_rate = 0.0`、`low_information_slot_share = 0.0`
    - `latest_stage2_v62_internal_eval.json` 记录 authoritative reader path `learned_query_slot_reader`、raw-context retrieval disabled、typed residuals 与 `full_accuracy = 0.578125`
    - `latest_stage2_v62_personamem_no_routing.json` 记录 answer-time routing disabled、raw-context retrieval disabled、PersonaMem gold 未用于 memory substrate
    - `latest_stage2_v62_arch_ablation.json` 记录 true disabled architecture reruns
  - Negative result:
    - PersonaMem full589 no-routing `160/589`
    - text-only `170/589`
    - option-only `235/589`
    - margin vs text-only `-10`
    - margin vs option-only `-75`
    - `latest_stage2_v62_decision.json` 记录 `result_type = negative_result`
  - Boundary:
    - 这一轮只证明 v6.2 的 authoritative learned write-quality 主路径、hard write-quality eval 与 clean persistent-state evidence 已成立。
    - 当前不能声明 benchmark gain、acceptance met、significant PersonaMem margin，或 reproducible closeout。

## Stage 2 V6 Persistent Core-Residual Latent Memory Evidence

- `EV-V6-001` -> `TD-049 / WS-035` persistent substrate partial evidence
  - Status: partial / blocked / negative_result
  - Evidence:
    - Fresh v6 baseline was `stage2_v6_persistent_latent_memory_score = 16`
    - Current verifier returns `80`
    - `latest_stage2_v6_persistent_state.json` records persistent `core_bank_size = 95`, `residual_bank_size = 483`, checkpoint path, and write trace path
    - `latest_stage2_v6_persistent_memory_train.json` records learned write-time router accuracy `0.9798657894134521` vs disabled controller `0.6644295454025269`
    - `latest_stage2_v6_persistent_memory_eval.json` records persistent-state authoritative input and raw-context retrieval disabled
    - `latest_stage2_v6_personamem_no_routing.json` records answer-time routing disabled and PersonaMem gold not used for memory substrate
    - `latest_stage2_v6_persistent_memory_train.json` records query-conditioned reader / belief-readout training, reader eval accuracy `0.7583892345428467` vs disabled readout `0.5`
    - `latest_stage2_v6_arch_ablation.json` records true disabled architecture ablation drops
  - Negative result:
    - PersonaMem full589 no-routing `146/589`
    - text-only `205/589`
    - margin `-59`
    - `latest_stage2_v6_decision.json` records `result_type = negative_result`
    - query-option blend trial degraded no-routing to `133/589`
    - no-gold synthetic projection weights reached `157/589` but did not improve verifier score
  - Boundary:
    - This is not v6 completion and not a superiority claim.
    - Current caps remain active because PersonaMem no-routing does not beat text-only and meaningful margin is absent.

## Stage 2 V6.1 Learned Reader/Decision Evidence

- `EV-V61-001` -> `TD-050 / WS-036` learned reader/decision over persistent memory
  - Status: partial / negative_result
  - Evidence:
    - `scripts/verify_stage2_v61_learned_reader_decision.py --score-only` 当前返回 `85`
    - `latest_stage2_v61_reader_decision_train.json` 记录 learned write-time router、query-conditioned learned reader、learned decision head、semantic slot matching / slot compaction 全部已训练；reader eval accuracy `0.9609810709953308`、decision eval accuracy `0.4983498454093933`
    - `latest_stage2_v61_persistent_state.json` 记录 persistent `core_bank_size = 64`、`residual_bank_size = 510`、checkpoint、write trace、bank compaction、typed residual relations 与 `other_fact_share = 0.0`
    - `latest_stage2_v61_internal_eval.json` 记录 authoritative reader path `learned_query_slot_reader`、natural-language queries、hard negatives、`reader_support_accuracy = 0.33203125`、`full_accuracy = 0.47265625`、`disabled_reader_accuracy = 0.23828125`、`disabled_decision_accuracy = 0.359375`
    - `latest_stage2_v61_personamem_no_routing.json` 记录 `sample_count = 589`、`score_mode = learned_decision_head`、`decision_mode = learned_answer_head`、`handcrafted_option_scoring_used = false`、`lexical_jaccard_used = false`
    - `latest_stage2_v61_arch_ablation.json` 记录 true disabled architecture reruns，当前 writer / reader / decision ablation drops 为真
  - Negative result:
    - PersonaMem full589 no-routing `191/589`
    - text-only `180/589`
    - option-only `235/589`
    - margin vs text-only `+11`
    - margin vs option-only `-44`
    - `latest_stage2_v61_decision.json` 记录 `result_type = negative_result`
  - Boundary:
    - 当前只证明 v6.1 的 authoritative learned reader / learned decision、semantic slot compaction、typed residuals 与 hard internal eval 已成立，并且 no-routing PersonaMem 已从 `140` 提到 `191`。
    - 当前不能声明 benchmark gain、acceptance met 或 significant PersonaMem margin。

## Stage 2 V5.2 Full Learned Latent System Evidence

- `EV-V52-001` -> `TD-048 / WS-034` real multi-backbone compare
  - Status: partial
  - Evidence:
    - `scripts/verify_stage2_v52_full_latent_system.py --score-only` 当前返回 `20`
    - `latest_stage2_v52_backbone_compare.json` 记录 `BAAI/bge-base-en-v1.5` 与 `intfloat/e5-base-v2` 真实加载，backend 为 `sentence_transformers`
    - 同一 artifact 记录 attempted backbones 为 `BAAI/bge-base-en-v1.5 / intfloat/e5-base-v2 / facebook/contriever`
    - `facebook/contriever` 当前因 `torch 2.5.1` 安全版本限制未加载，artifact 已记录 failure
  - Boundary:
    - 当前只证明 v5.2 已达到 “at least two real pretrained backbones loaded/compared” 的 partial gate。
    - multi-task training、real v5.2 checkpoint、trainable belief/controller/resampler、ablation drops 与 PersonaMem no-calibration > text-only `214/589` 尚未完成。
    - 不得把当前 `20/100` 写成 full learned latent system verified。

- `EV-V52-002` -> `TD-048 / WS-034` four-task multi-task checkpoint
  - Status: partial
  - Evidence:
    - `scripts/verify_stage2_v52_full_latent_system.py --score-only` 当前返回 `30`
    - `latest_stage2_v52_multitask_training.json` 记录四任务训练：`retrieval_alignment / slot_autoencoding / composition_to_belief / lifecycle_prediction`
    - 同一 artifact 记录 `train_samples = 24500`，`retrieval_only = false`，`uses_personamem_gold = false`
    - checkpoint 写入 `outputs_v2/checkpoints/20260423T024312Z_stage2_v52_multitask/full_latent_system.pt`
    - trained modules 覆盖 `encoder_adapter / query_projection / slot_projection / latent_reader / latent_resampler / write_controller / lifecycle_head / belief_decoder / belief_graph`
  - Boundary:
    - 当前只证明 multi-task training 和真实 checkpoint 已存在。
    - 尚未发布 v5.2 full latent eval、ablation drops 或 PersonaMem full589 no-calibration text-only gain。
    - 当前 verifier 仍被 gold-isolation / PersonaMem / eval gates 限制，不能写成 acceptance met。

- `EV-V52-003` -> `TD-048 / WS-034` PersonaMem full589 no-calibration evaluation
  - Status: verified
  - Evidence:
    - `latest_stage2_v52_personamem_full589.json` 记录 `sample_count = 589`
    - confidence-routed no-calibration `228/589 = 38.71%`
    - option-only `168/589 = 28.52%`
    - text-only `227/589 = 38.54%`
    - score route 使用 text evidence `539` 次、latent evidence `50` 次
    - 相对 text-only improved `9`、degraded `8`
    - `gold_used_for_memory_substrate = false`
    - `latest_stage2_v52_decision.json` 记录 `result_type = positive_gain`
    - `scripts/verify_stage2_v52_full_latent_system.py --score-only` 当前返回 `100`
  - Boundary:
    - 当前 no-calibration 仅比 text-only 多 `1` 个 correct，属于极窄 mechanical pass。
    - 可以声明 v5.2 verifier stop condition reached；不得声明 broad benchmark superiority。

- `EV-V52-004` -> `TD-048 / WS-034` full latent eval and ablation
  - Status: verified
  - Evidence:
    - `latest_stage2_v52_full_latent_eval.json` 记录 trained MRR `0.9816` > frozen MRR `0.7054`
    - multi-task composite `0.9096` > retrieval-only composite `0.2454`
    - latent-only `0.9816` > shuffled-latent `0.9464`
    - full composite `0.9096` > text-only composite `0.1764`
    - `latest_stage2_v52_ablation_summary.json` 记录 no-controller、no-belief、no-core-residual ablation 全部下降
    - `scripts/verify_stage2_v52_full_latent_system.py --score-only` 当前返回 `100`
  - Boundary:
    - 当前 latent / ablation gates 已满足。
    - PersonaMem pass margin 极窄，仍需保留 truth boundary。

## Stage 2 V5.1 Real Training Evidence

- `EV-V51-001` -> `TD-047 / WS-033` real pretrained training
  - Status: verified
  - Evidence:
    - `scripts/verify_stage2_v51_real_training.py --score-only` 当前返回 `100`
    - `research-results.tsv` iteration `3 keep` 已记录 `real-pretrained / real-checkpoint / heldout-gain / anti-shortcut / no-calibration`
    - `latest_stage2_v51_real_backbone_compare.json` 记录 `BAAI/bge-base-en-v1.5` 真实加载、backend `sentence_transformers`、embedding dim `768`、parameter count `109482240`
    - `latest_stage2_v51_real_training.json` 记录 12k stage2 gold-free train samples、real checkpoint `outputs_v2/checkpoints/20260423T013123Z_stage2_v51_real_training/latent_retriever.pt`、device `cuda:2`
    - `latest_stage2_v51_latent_eval.json` 记录 held-out trained MRR `0.9792` > frozen `0.7054`，latent-only `0.9792` > shuffled-latent `0.9592`
    - `latest_stage2_v51_personamem_full589.json` 记录 no-calibration `184/589 = 31.24%` > option-only `167/589 = 28.35%` > random `25%`
    - `latest_stage2_v51_real_training_decision.json` 记录 `result_type = positive_gain`
  - Boundary:
    - 当前只完成 BGE 单 backbone，不是 BGE/E5/Contriever 三者完整比较。
    - PersonaMem no-calibration 超过 option-only/random，但低于 artifact 中 text-only `214/589`；不得误写成所有 PersonaMem text baseline 均被 latent full 超过。
    - 当前结论是 local mechanical verifier + gold-free/no-calibration positive result，不是 provider-side superiority claim。

## Stage 2 V4 Runtime Evidence

- `EV-V4-001` -> `TD-045 / WS-031` Persona-first learned memory
  - Status: verified
  - Evidence:
    - `scripts/verify_stage2_v4_longrun.py --score-only` 当前返回 `38/38`
    - `latest_stage2_v4_personamem_gap_audit.json` 记录 v33 Persona full gap：`local_correct_provider_wrong = 81`、`provider_correct_local_wrong = 67`
    - `latest_stage2_v4_persona_option_scorer_eval.json` 记录 direct-user reason-update option replay：baseline `196/512` -> option scorer `219/512`，`improved_predictions = 23`，`degraded_predictions = 0`
    - `latest_personamem_stage2_v4_full.json` 与 `latest_stage2_v4_personamem_compare.json` 记录 learned runtime `memory_mode = learned_memory`、`slot_assignment_mode = learned`，provider 为 auxiliary
    - `latest_stage2_v4_longmemeval_guard.json` 记录 non-catastrophic guard local `14/500`
    - `latest_stage2_v4_ablation_summary.json` 记录 latent / belief / option scorer contributions 与 `no_fallback_or_shortcut = true`
  - Boundary:
    - 当前完成态是 option-scorer replay + inherited learned latent/belief evidence；不得误写成 provider exact 改善或 full provider rerun 改善。

## Stage 1 Acceptance Tracking

- `EV-001` -> `AC-001` vanilla CoRe Memory 完整实现
  - Status: verified
  - Evidence:
    - `src/core_mem/memory/core.py` 已实现 writer / updater / residual manager / reader / workflow 最小闭环
    - `tests/test_core_memory_module.py` 已覆盖核心 memory module
    - 已接入 `sentence-transformers/all-MiniLM-L6-v2` embedding 配置与 `observe_text/query_text`
    - `scripts/verify_stage1_acceptance.py` 当前将 `AC-001` 判为 passed

- `EV-002` -> `AC-002` PersonaMem 官方协议跑通并产出结果
  - Status: partial
  - Evidence:
    - 官方 `32k` 数据已下载到 `data/personamem/`
    - `outputs/runs/20260405T024515Z_personamem/predictions.jsonl` 已产出真实预测
    - `outputs/runs/20260405T024515Z_personamem/evaluation.json` 显示最小 1-sample `accuracy=1.0`
    - `outputs/runs/20260405T042146Z_personamem/evaluation.json` 显示 Gemini 2.5 Flash 路径下的最小 1-sample `accuracy=1.0`
    - `outputs/runs/20260405T050000Z_personamem/predictions.jsonl` 当前仅推进到 `22/589`
    - 超保守单样本检查在目标 `22 -> 23` 上连续 4 次 child launch 均返回 `HTTP 429`，说明当前正式结果受真实外部 blocker 限制

- `EV-003` -> `AC-003` LongMemEval-S 官方协议跑通并产出结果
  - Status: partial
  - Evidence:
    - 官方 `S` 数据已下载到 `data/longmemeval/`
    - `outputs/runs/20260405T024833Z_longmemeval/predictions.jsonl` 已产出真实预测
    - `outputs/runs/20260405T024833Z_longmemeval/evaluation.json` 显示最小 1-sample `accuracy=1.0`
    - `outputs/runs/20260405T042150Z_longmemeval/evaluation.json` 显示 Gemini 2.5 Flash 路径下的最小 1-sample `accuracy=1.0`
    - `outputs/runs/20260405T042236Z_longmemeval/predictions.jsonl` 已在 formal run 下推进到 `19/500`，但尚未完成全量结果

- `EV-004` -> `AC-004` 结果数据、表格、prediction outputs 与实验元数据
  - Status: verified
  - Evidence:
    - `scripts/run_experiment.py` 在 dry-run 下会写出 `outputs/runs/.../run_metadata.json`
    - `run_metadata.json` 已包含 benchmark、record_count、model、provider、sample_ids、prompt_preview
    - 当前 `run_metadata.json` 已补充 `config_snapshot_path`、`run_timestamp`、`commit_hash`、`prompt_version`
    - 真实 run 已产生 `predictions.jsonl` 与 `evaluation.json`
    - `outputs/tables/stage1_summary.csv` 与 `outputs/tables/stage1_summary.json` 已生成

- `EV-005` -> `AC-005` 工程与复现闭环
  - Status: verified
  - Evidence:
    - `environment.yaml` 已成为唯一环境规范文件
    - `configs/defaults.yaml` 与 `src/core_mem/config.py` 已建立统一配置加载路径
    - `configs/gemini_flash.yaml` 已提供一个用户批准的替代 OpenAI-compatible provider 配置
    - `configs/gemini_flash_ultraslow.yaml` 已提供外部 blocker 判别用的超保守 provider 配置
    - `scripts/run_experiment.py` 支持 benchmark 子命令和 dry-run 复现入口
    - `scripts/prepare_data.py`、`scripts/evaluate_run.py`、`scripts/make_tables.py` 已提供最小分步骤路径
    - `scripts/stage1_preflight.py` 已提供外部前置条件检查
    - `scripts/verify_stage1_acceptance.py` 已提供机械 acceptance check
    - `scripts/run_experiment.py` 已支持 `--run-dir` / `--resume` 与增量 prediction 落盘
    - `scripts/run_until_complete.py` 已提供 crash 后自动冷却重启的外层 supervisor
    - `outputs/` 路径已被实际写入
    - 环境复现策略已更新为 `environment.yaml` + 默认 conda named env；repo-local conda 目录已清理
    - 默认 conda named env `core_mem` 已创建并复验

- `EV-006` -> `AC-006` 单元测试与 E2E smoke test
  - Status: verified
  - Evidence:
    - 当前 `pytest` 通过，共 56 个测试
    - 已包含 provider、config、benchmark adapter、verifier、E2E dry-run smoke test、step scripts smoke test、protocol alignment tests、traceability tests、core memory tests
    - 已新增 acceptance verifier test、resumable runner test、Gemini 配置 / rate-limit retry tests 与 benchmark supervisor tests

- `EV-007` -> `AC-007` 结论可信、结果可追溯、允许 negative result
  - Status: verified
  - Evidence:
    - dry-run 输出已保存结构化 run metadata
    - benchmark prompt preview 与 sample ids 已可追溯
    - `scripts/verify_stage1_acceptance.py` 当前返回 `5/7`

## Stage 2 Planning And Design Tracking

- `EV-008` -> `AC-008` 第二阶段 `V2.0` 设计文档与真源条款
  - Status: verified
  - Evidence:
    - `docs/requirements.md` 已细化第二阶段目标、范围、训练、数据、评测、输出与 pending benchmark 规则
    - `docs/v2_design.md` 已细化到 schema、dataset-to-task mapping 与 metric-to-module mapping
    - `docs/implementation_plan.md` 已将 stage-2 调整为当前执行主线
    - `scripts/verify_stage2_acceptance.py` 当前将 `AC-008` 判为 passed

- `EV-009` -> `AC-009` 第二阶段 observation / slot / belief 规范
  - Status: verified
  - Evidence:
    - `docs/requirements.md` 已锁定 `Observation JSON`、`Slot Record` 与 `Belief JSON` 的必要字段
    - `docs/v2_design.md` 已写入 observation、slot、belief 三者关系、schema、字段语义与训练样本单位
    - `scripts/verify_stage2_acceptance.py` 当前将 `AC-009` 判为 passed

- `EV-010` -> `AC-010` 第二阶段主线实现骨架
  - Status: verified
  - Evidence:
    - `src/core_mem/v2/__init__.py`、`src/core_mem/v2/schemas.py`、`src/core_mem/v2/parser.py`、`src/core_mem/v2/datasets.py` 已落地
    - `src/core_mem/v2/encoder.py`、`src/core_mem/v2/lifecycle.py`、`src/core_mem/v2/consolidation.py`、`src/core_mem/v2/resampler.py`、`src/core_mem/v2/decoder.py`、`src/core_mem/v2/projection.py`、`src/core_mem/v2/system.py` 已落地
    - `tests/test_stage2_schemas.py`、`tests/test_stage2_parser.py` 已覆盖 Observation / Slot / Belief schema 与 rule-first parser skeleton
    - `tests/test_stage2_model_skeleton.py` 已覆盖主线 memory system 的最小 observe/query smoke path，以及 belief evidence / answer projection
    - `scripts/verify_stage2_acceptance.py` 当前将 `AC-010` 判为 passed

- `EV-011` -> `AC-011` 第二阶段训练与本地 intrinsic evaluation 管线
  - Status: verified
  - Evidence:
    - `scripts/prepare_stage2_data.py` 已生成 `outputs_v2/artifacts/stage2_prepared_samples_manifest.json`
    - `scripts/prepare_stage2_data.py` 现在已支持 `configs/stage2_data_sources.json` 驱动的 source-config 模式，能够在非 demo 输入下生成 prepared manifests
    - `scripts/stage2_data_preflight.py` 已能机械报告 `data/stage2_public/` 中公开数据源的 available/missing 状态，当前已返回 `missing=[]`
    - `scripts/normalize_stage2_public_data.py` 已从真实 `SGD / MultiWOZ 2.4 / Persona-Chat / MQUAKE / ReCoE` raw sources 生成 5 份 `normalized.jsonl`
    - `prepare_stage2_data.py` 当前已支持 `--max-rows-per-dataset`，可在严格 source 模式下构建受控 public-data manifests
    - `scripts/train_stage2.py` 已生成 `outputs_v2/runs/20260407T042125Z_stage2_train_plan/training_plan.json`
    - `scripts/train_stage2.py` 现在已支持 `--execute-train`，并通过 `configs/stage2_train_tiny.yaml` 在 `outputs_v2/runs/20260414T020438Z_stage2_train_exec/training_metrics.json` 和 `outputs_v2/runs/20260414T043225Z_stage2_train_exec/training_metrics.json` 跑通最小训练执行链；后者直接消费真实 public-data prepared manifest
    - `configs/stage2_train.yaml` 已锁定 backbone / resampler / LoRA / batching 配置，并新增 repo-local Hugging Face cache root
    - `outputs_v2/runs/20260414T043221Z_stage2_train_plan/launch_stage2_training.sh` 现已是直接可执行的 `--execute-train` 启动脚本
    - `environment.yaml` 已补充 `datasets`、`peft`、`accelerate` 依赖，并已在 `core_mem` 环境实际安装验证
    - `outputs_v2/checkpoints/stage2_smoke_checkpoint.json` 已作为 smoke execution artifact 写出
    - `scripts/verify_stage2_acceptance.py` 当前将 `AC-011` 判为 passed

- `EV-012` -> `AC-012` 第二阶段本地评估、budget sweep 与核心 ablation 记录机制
  - Status: verified
  - Evidence:
    - `scripts/eval_stage2_local.py` 现已支持 `--top-k`、`--budget`、`--dataset`、`--experiment-id`
    - `src/core_mem/v2/eval_local.py` 已实现模块级、家族级、budget-sweep 和 variant-aware 评测逻辑
    - `docs/stage2_local_evaluation.md` 已完整说明接口、指标定义、模块映射与使用方式
    - `src/core_mem/v2/experiments.py` 已锁定 `mainline + 11` 个必做 ablation 的 preset variant
    - `outputs_v2/artifacts/stage2_experiment_index.json` 当前已登记 12 个 completed experiments
    - `scripts/verify_stage2_experiment_status.py --score-only` 当前返回 `13`
    - `outputs_v2/evals_local/20260414T074458Z_stage2_local_eval.json` 与同批 `outputs_v2/tables/*` 已作为最新 ablation registry 产物落地
    - `docs/requirements.md` 与 `docs/v2_design.md` 已锁定 budget sweep 与核心 ablation 集；当前 repo 已具备记录并机械验证这些结果的路径

- `EV-013` -> `AC-013` 第二阶段 benchmark canary 与正式评测隔离协议
  - Status: verified
  - Evidence:
    - `docs/requirements.md` 已写入 `local-first` 与 canary-first 协议
    - `docs/v2_design.md` 已写入 canary 设计与 benchmark 作为 evaluation source 的原则
    - `scripts/run_stage2_canary.py` 已生成 `outputs_v2/evals_benchmark/20260407T042120Z_stage2_canary_plan.json`
    - `outputs_v2/evals_benchmark/20260407T042120Z_personamem_canary.json` 与 `outputs_v2/evals_benchmark/20260407T042120Z_longmemeval_canary.json` 已落地，且二者均固定为 `64` 条

- `EV-014` -> `WS-021 / TD-035` `v2.4` 长跑阶段性证据
  - Status: verified
  - Evidence:
    - managed run 已完成 `research-results.tsv` 与 `autoresearch-state.json` 的全程记账；当前记录 best iteration `7`、best/current metric `24`
    - `scripts/verify_stage2_v24_longrun.py --score-only` 已在 current HEAD `12a9a80` 上达到 stop condition `24/24`
    - `outputs_v2/artifacts/latest_stage2_v24_train.json` 已记录 current-head full-data train，并与 retained HEAD `12a9a80` 对齐
    - `outputs_v2/artifacts/latest_stage2_v24_eval.json` 已记录 current-head full-data eval：`trained_eval.token_f1 = 0.9991150844073334`、`trained_eval.field_f1 = 0.9976704786107581`、`slot_assignment_metrics.token_f1 = 0.9961127308066084`
    - `outputs_v2/artifacts/latest_longmemeval_stage2_v24_canary.json` 与 `latest_longmemeval_stage2_v24_analysis.json` 已记录 current-head `LongMemEval-S 128`：`provider_exact_match = 10`、`provider_label_prefix_match = 10`、`local_exact_match = 10`
    - `outputs_v2/artifacts/latest_personamem_stage2_v24_canary.json` 已记录 current-head `PersonaMem 128`：`provider_exact_match = 38`、`provider_label_prefix_match = 38`、`local_exact_match = 28`
    - `outputs_v2/artifacts/latest_stage2_v24_online_gain.json` 已记录相对 `v2.3` retained baseline 的 `delta_provider_exact_match = +4`、`delta_local_exact_match = +4`
    - launch-manifest guard 已在 retained HEAD `12a9a80` 上通过：`tests/test_stage2_v24_longrun.py tests/test_stage2_model_skeleton.py tests/test_stage2_local_eval.py tests/test_stage2_memory_canary.py tests/test_stage2_memory_canary_quality.py tests/test_stage2_parser.py`
  - Boundary:
    - 当前能诚实声明的是 `v2.4 long-run` 的机械 stop condition 已达到；不能夸写成所有 benchmark 质量风险都已消失，或下一轮 stage-2 方向已经自动确定

- `EV-015` -> `WS-022 / TD-036` `v2.5` baseline/package closeout 证据
  - Status: partial
  - Evidence:
    - current HEAD `3036e3d` 上 `scripts/verify_stage2_v25_longrun.py --score-only` 已达到 stop condition `24/24`
    - `research-results.tsv` / `autoresearch-state.json` 已记录 iteration `5 keep`、best/current metric `24`
    - `outputs_v2/artifacts/latest_stage2_v25_train.json`、`latest_stage2_v25_eval.json`、`latest_longmemeval_stage2_v25_canary.json`、`latest_personamem_stage2_v25_canary.json`、`latest_longmemeval_stage2_v25_analysis.json`、`latest_stage2_v25_full_benchmark.json` 已在 current HEAD 上落地
    - `outputs_v2/artifacts/latest_stage2_v25_write_gain.json`、`latest_stage2_v25_retrieve_gain.json`、`latest_stage2_v25_belief_gain.json` 已作为 current-head baseline decomposition artifacts 落地
    - current-head `v24` baseline alias 已刷新到同一 HEAD，使 `latest_stage2_v24_eval.json`、`latest_longmemeval_stage2_v24_canary.json`、`latest_personamem_stage2_v24_canary.json` 可继续作为 `v2.5` retained baseline 复验
  - Boundary:
    - 当前 `24/24` 只说明 `v2.5` 的 artifact/verifier package 已机械齐套
    - `write / retrieve / belief` 三段 artifact 当前都显式记录 `delta_vs_v24_retained = 0`
    - 因此不能把本轮 closeout 误写成真实 `LongMemEval-S` / online quality gain 已超过 `v2.4`

- `EV-016` -> `WS-023 / TD-037` `v2.6` gain-first 长跑基线
  - Status: verified
  - Evidence:
    - `docs/v26_plan.md` 已把目标明确收紧为 gain-first：要求 current-head 的 `write / retrieve / belief` 至少一段出现真实正增益，并要求 `LongMemEval-S 128` 明确超过 `v2.5` retained baseline `10/128`
    - `scripts/verify_stage2_v26_longrun.py` 已把 mechanical target 改为 `stage2_v26_longrun_score`
    - `tests/test_stage2_v26_longrun.py` 已覆盖“只有在 positive gain 与超 baseline canary 同时出现时才可满分”的 verifier 合同
    - 当前 `v2.5` retained baseline 继续保留为对照线：`latest_stage2_v25_train.json`、`latest_stage2_v25_eval.json`、`latest_longmemeval_stage2_v25_canary.json`、`latest_personamem_stage2_v25_canary.json`、`latest_stage2_v25_write_gain.json`、`latest_stage2_v25_retrieve_gain.json`、`latest_stage2_v25_belief_gain.json`
    - fresh baseline 已通过 helper 固化：`research-results.tsv` iteration `0` 与 `autoresearch-state.json` 当前都记录 `stage2_v26_longrun_score = 9`
    - trial HEAD `077cbf3` 已加入 query-intent-aware temporal retrieval / belief scoring，并通过 stage-2 guard：`tests/test_stage2_v26_longrun.py tests/test_stage2_model_skeleton.py tests/test_stage2_local_eval.py tests/test_stage2_memory_canary.py tests/test_stage2_parser.py`
    - current session 已确认 `GPT_AGENT_API_KEY=SET`，因此历史 provider-env blocker 已被清除；新的 refine commit `a04effe` 已加入 slot-assignment prompt compaction 与 weak `other_fact` overwrite fast-path，并通过同一组 stage-2 guard
    - profiling 证据已显示 sample `51a45a95` 的 learned arbitration 次数从 `11` 压到 `1`
    - 同一 partial `LongMemEval-S 128` resumed run 已从 `2/128` 前进到 `3/128`，且 `51a45a95` 当前在 current-head 上记录 `memory_answer_local = target`、`provider_prediction = target`
    - refine commit `3051b0f` 已让 `scripts/run_stage2_memory_canary.py` 在单次 canary run 内复用 learned belief / slot-assignment predictors；fresh current-head `LongMemEval-S 128` run `outputs_v2/runs/20260417T055905Z_stage2_memory_canary_longmemeval/` 最终已完整结束在 `provider_exact_match = 11`、`local_exact_match = 10`
    - current-head `PersonaMem 128` run `outputs_v2/runs/20260417T072623Z_stage2_memory_canary_personamem/` 已完整结束在 `provider_exact_match = 44`、`local_exact_match = 33`，相对 retained `v2.5` `38 / 28` 保持正向 guard
    - refine commit `b4c997d` 已把 `v2.6` artifact 发布链并回 `scripts/verify_stage2_v26_longrun.py`，并新增 `tests/test_stage2_v26_publish.py` 覆盖 current-head publish 入口；当前 publish 入口已把 current-head `v2.5` / `v2.6` aliases、component gain artifacts、LongMemEval analysis 与 holdout full-benchmark summary 全部刷新到 current HEAD `c8c8e19`
    - trial commit `160e29f` 已在 `src/core_mem/v2/projection.py` 上加入 historical `other_fact` clause projection；对应 fresh current-head `LongMemEval-S 128` run `outputs_v2/runs/20260417T083058Z_stage2_memory_canary_longmemeval/` 在一次 `HTTP 502` 后原地 `--resume` 收口到 `provider_exact_match = 11`、`local_exact_match = 11`
    - current-head `PersonaMem 128` guard refresh `outputs_v2/runs/20260417T094824Z_stage2_memory_canary_personamem/` 在一次 read timeout 后原地 `--resume` 收口到 `provider_exact_match = 44`、`local_exact_match = 33`
    - `outputs_v2/artifacts/latest_stage2_v26_write_gain.json` 当前已记录 `positive_gain = true`、`delta_provider_exact_match = +1`、`delta_local_exact_match = +1`
    - `outputs_v2/artifacts/latest_stage2_v26_belief_gain.json` 当前已记录 `positive_gain = true`、`delta_vs_v25_retained = +1`
    - `outputs_v2/artifacts/latest_stage2_v26_full_benchmark.json` 当前已记录 `holdout_only = true`
    - `conda run -n core_mem python scripts/verify_stage2_v26_longrun.py --score-only` 当前为 `26`
  - Boundary:
    - 当前能诚实声明的是 `v2.6` 的机械 stop condition 已达成
    - 当前仍不能误写成 LongMemEval failure mass 已全面解决；remaining问题仍主要集中在 `other_fact / projection` family

- `EV-017` -> `WS-030 / TD-044` `v33` learned-authoritative full-holdout runtime 证据
  - Status: partial
  - Evidence:
    - `research-results.tsv` row `102` 已把 current-line authoritative `PersonaMem 512` full measurement 正式记账为 completed `search`；row `103` 又把 current-line `LongMemEval-S 500` full measurement 与 published full compare 正式记账为 completed `search`
    - `autoresearch-state.json` 当前 iteration 已同步到 `103`，并记录 row-103 labels `longmemeval500-full / authoritative-full / learned-runtime-established / full-compare-mixed / verifier43 / v33`
    - `outputs_v2/v33_full_personamem_512_query_overlap/evals_benchmark/20260421T001000Z_stage2_memory_canary.json` 已完整记录 current-line `PersonaMem 512` full holdout：`provider_exact_match = 182`、`local_exact_match = 196`
    - `outputs_v2/v33_full_longmemeval_500_query_overlap/evals_benchmark/20260422T015146Z_stage2_memory_canary.json` 已完整记录 current-line `LongMemEval-S 500` full holdout：`provider_exact_match = 20`、`local_exact_match = 14`
    - retained baseline `outputs_v2/artifacts/latest_personamem_stage2_v32_full.json` 当前固定为 `provider/local = 183/175`，因此 current-line Persona full measurement 已形成 `provider -1 / local +21`
    - retained baseline `outputs_v2/artifacts/latest_longmemeval_stage2_v32_full.json` 当前固定为 `provider/local = 21/14`，因此 current-line LongMemEval full measurement 已形成 `provider -1 / local +0`
    - 当前 session 已确认同一 full Persona run 的 provider failure shape 固定为 `blank = 108`、`nonlabel = 12`
    - `outputs_v2/artifacts/latest_stage2_v33_learned_authoritative_runtime.json` 已机械记录 `memory_mode = learned_memory`、`slot_assignment_mode = learned` 与 `learned_authoritative = true`
    - `outputs_v2/artifacts/latest_stage2_v33_full_holdout_compare.json` 已在 commit `6fa018e70561eef6b0239f41b3d662794df41a22` 上发布，并显式记录 `longmemeval_gain_confirmed = false`、`personamem_gain_confirmed = false`
    - `conda run -n core_mem python scripts/verify_stage2_v33_longrun.py --score-only` 当前已到 `43`
  - Boundary:
    - 当前能诚实声明的是：`v33` learned-authoritative runtime 已机械成立，且 current-line 两条 full holdout 都已完成并发布 compare
    - 当前不能声称 `v33` 已满足 keep / acceptance，因为两条 gain flag 仍都是 `false`；LongMemEval 仍是 `provider -1 / local tie`，Persona 仍是 `provider -1 / local +21`
    - 因此当前最诚实状态是 completed full-measurement mixed search result，而不是 acceptance met / keep closeout

- `EV-018` -> `WS-032 / TD-046` `v5 Core-Residual Latent Substrate` contract lock and managed-run evidence
  - Status: verified
  - Evidence:
    - `docs/v5_plan.md` 已创建，明确 core-residual latent substrate 的目标架构、训练数据、训练目标、anti-shortcut 评估和里程碑
    - `research-results.tsv` 已记录 baseline `13` 与 iteration `7 keep`；最终 `stage2_v5_longrun_score = 52`
    - `autoresearch-state.json` 当前 best/current metric 均为 `52`，last commit 为 `0c42cdc1b7cc3304cece9c7cfd30a78a938b51e9`
    - `scripts/verify_stage2_v5_longrun.py --score-only` 当前返回 `52`
    - `.agent-os/project-index.md` 已把 active workstream 前推到 `TD-046 / WS-032 / v5`
    - `.agent-os/change-decisions.md` 已记录 PersonaMem gold calibration isolation、pretrained backbone comparison、MiniMax teacher 边界
    - `.agent-os/todo.md` 与 `docs/todo.md` 已同步 `TD-046`
    - `outputs_v2/artifacts/latest_stage2_v5_personamem_isolation.json` 记录 PersonaMem persona/context grouped split、`sample_count = 589`、`no_gold_leakage = true`
    - `outputs_v2/artifacts/latest_stage2_v5_context_selfsupervised.json` 记录 gold-free context self-supervised samples，并显式记录 `no_gold_answers = true`
    - `outputs_v2/artifacts/latest_stage2_v5_encoder_compare.json` 记录 `BGE / E5 / Contriever` comparison harness、`uses_ablation_metrics = true`、`not_selected_by_personamem_only = true`
    - `outputs_v2/artifacts/latest_stage2_v5_core_residual_train.json` 与 `latest_stage2_v5_controller_ablation.json` 记录 core/residual substrate train 与 controller ablation positive gain
    - `outputs_v2/artifacts/latest_stage2_v5_latent_reader_eval.json` 与 `latest_stage2_v5_text_ablation.json` 记录 latent-only beats random、shuffled-latent drop 与 `full_beats_text_only = true`
    - `outputs_v2/artifacts/latest_stage2_v5_answer_head_calibration.json` 记录 strict gold isolation、thin answer-head-only calibration 与 no-cal / calibrated split
    - `outputs_v2/artifacts/latest_stage2_v5_personamem_full589.json` 记录 PersonaMem full-589 local report，provider 为 auxiliary 且 `provider_run_executed = false`
    - `outputs_v2/artifacts/latest_stage2_v5_ablation_summary.json` 与 `latest_stage2_v5_paper_evidence_package.json` 记录 anti-shortcut summary 与 paper evidence boundary
  - Verification:
    - `conda run -n core_mem pytest -q tests/test_stage2_v5_longrun.py` -> passed
    - `git diff --check && python scripts/verify_stage2_v5_longrun.py --score-only` -> `52`
  - Boundary:
    - `EV-018` 证明 v5 本地 mechanical evidence package 已达到 configured stop condition
    - 当前不能写成 provider-side superiority claim；provider 仍是 auxiliary
    - 当前 encoder comparison harness 显式记录 `pretrained_weights_loaded = false`，因此不能写成真实 pretrained HF weights 的模型优劣结论

- `EV-019` -> `WS-033 / TD-047` `v5.1 Real Pretrained Training` plan lock
  - Status: planned
  - Evidence:
    - `docs/v51_plan.md` 已创建，明确把 v5 proxy/scaffold 失败转成真实 pretrained training 主线
    - `.agent-os/project-index.md` 已把 active workstream 前推到 `TD-047 / WS-033 / v5.1`
    - `.agent-os/change-decisions.md` 已记录用户拒绝 placeholder closeout、接受 negative result、要求 verifier 对 proxy / pretrained false / toy training 设硬门槛
    - `docs/v5_plan.md` 已加注：v5 retained artifact 是 scaffold/evidence-package closeout，不是 scientific closeout
    - `docs/current_status.md`、`docs/implementation_plan.md`、`.agent-os/todo.md` 与 `docs/todo.md` 已同步 v5.1 next action
  - Pending acceptance:
    - `scripts/verify_stage2_v51_real_training.py` 已有 bootstrap 版；后台长跑仍需扩展并让真实 v5.1 evidence 通过它
    - 真实 pretrained backbone 尚未重新加载并训练
    - 真实 checkpoint / train log / held-out eval 尚未产出
    - trained-vs-frozen、latent-only-vs-shuffled、full-vs-text-only 与 PersonaMem full589 no-calibration 结果尚未刷新
  - Boundary:
    - `EV-019` 目前只证明计划与验收边界已锁定
    - 不能把它写成 v5.1 training 已完成，也不能写成已有真实 pretrained model gain
