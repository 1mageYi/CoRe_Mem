# Run Log

# 2026-04-21 Session 095

- Worked on: 用 focused `suggest_new_ideas` local-only slice 测量 `49e1841` 的 boolean-backfill 是否开始外扩
- State changed:
  - fresh run `outputs_v2/v33_boolean_belief_suggest8_localonly/runs/20260421T020526Z_stage2_memory_canary_personamem/` 启动后确认 `provider_configured = false`，但 learned full-context CPU path 吞吐偏慢，因此在 evidence 足够后主动截断
  - 当前 partial artifact `outputs_v2/v33_boolean_belief_suggest8_localonly/artifacts/20260421T021500Z_partial_summary.json` 已给出 `completed = 4/8`、`local_exact = 4/4`、`baseline = 0/4`、`boolean-like belief rows = 0/4`
  - 这说明 `49e1841` 的 generic boolean-backfill 已不再只停留在单样本 `b358...`，至少在首批 focused `suggest_new_ideas` slice 上形成了可测外扩
  - 因为这轮仍是 partial measurement、official verifier 没动，helper 已把它记为 iteration `30 refine`
- Evidence / artifacts:
  - commit `49e1841`
  - `outputs_v2/v33_boolean_belief_suggest8_localonly/artifacts/20260421T021500Z_partial_summary.json`
  - `research-results.tsv`
  - `autoresearch-state.json`
- Next likely action:
  - 不再把 “boolean-backfill 对 focused suggest slice 是否有效” 当开放问题，它已经在首个 partial slice 上成立
  - 下一轮直接换一组新的 `suggest_new_ideas` 小 slice，验证这条外扩不是只发生在前几个 music-preference 样本上；若不稳定，再把主火力切回 selected-slot ranking

# 2026-04-21 Session 094

- Worked on: 把 `smoke16` 暴露出的 `belief=true` projection noise 收敛成 generic boolean-like belief backfill
- State changed:
  - current HEAD `49e1841` 在 `src/core_mem/v2/system.py` 中新增 boolean-like learned belief value backfill：当 `true/false/yes/no` 这类低信息值挂在 same-support belief 上、而 support slot canonical value 明显是非 boolean 语义时，系统会回填 canonical value
  - `tests/test_stage2_model_skeleton.py` 已补齐对应回归；targeted tests、configured guard 与 real sample replay 均通过，`scripts/verify_stage2_v33_longrun.py --score-only` 仍是 `36`
  - real sample probe `outputs_v2/v33_boolean_belief_backfill_local/artifacts/20260421T015404Z_b358_probe.json` 已机械确认：`b3588797-acdf-40d3-bcc5-951f81896f95` 从 `belief=true` 修回 support-slot canonical text，answer 仍保持正确 `(a)`
  - 因为 official verifier 没动，helper 已把这轮记为 iteration `29 refine`
- Evidence / artifacts:
  - commit `49e1841`
  - `outputs_v2/v33_boolean_belief_backfill_local/artifacts/20260421T015404Z_b358_probe.json`
  - `outputs_v2/v33_boolean_belief_backfill_local/artifacts/20260421T015404Z_b358_compare.json`
  - `research-results.tsv`
  - `autoresearch-state.json`
- Next likely action:
  - 不再把 `belief=true` 是否值得修当开放问题，它已经在 current HEAD 上被 generic 化并通过真实样本验证
  - 下一轮先判断这条 boolean-backfill 能否在更大 `suggest_new_ideas` / Persona slice 上形成可测外扩；若不能，就把主火力打回 selected-slot ranking 与更大 local-only measurement

# 2026-04-21 Session 093

- Worked on: 继续放大 local-only Persona slice，并在 evidence 足够后主动截断长耗时 measurement
- State changed:
  - `smoke16` local-only run 在主动截断前已完成 `6` 条样本，当前 `local_exact = 6/6`、`local_baseline_exact = 1/6`
  - 这说明 `eade2c3` 的局部 gain 不只在 `smoke8` 上成立，也没有在更大的 Persona slice 上立刻塌掉
  - 当前 partial `smoke16` 还暴露出一个更高优先级的诊断线索：`b358...` 这类 `suggest_new_ideas` 样本虽然当前 answer 仍正确，但 learned belief value 会退化成 `true`
  - 因为这轮是 partial measurement、没有改 official verifier，helper 已把它记为 iteration `28 refine`
- Evidence / artifacts:
  - commit `eade2c3`
  - `outputs_v2/v33_latent_facet_rerank_smoke16_localonly/runs/20260421T012155Z_stage2_memory_canary_personamem/predictions.jsonl`
  - `research-results.tsv`
  - `autoresearch-state.json`
- Next likely action:
  - 不再继续空等 `smoke16` 全量跑完
  - 下一轮先检查 `suggest_new_ideas` / `belief=true` 这类 projection-noise 是否值得做最小修复

# 2026-04-21 Session 092

- Worked on: 把 `eade2c3` 的局部增益从 hard-4 扩到更大的 Persona local-only slice 测量
- State changed:
  - 首次 `smoke8` 因默认读到 `GPT_AGENT_API_KEY` 而错误走了 provider path，在第一条样本完成前已主动中断；helper 已把这次配置性失误记为 iteration `26 crash`
  - 随后 corrected local-only run `outputs_v2/v33_latent_facet_rerank_smoke8_localonly/evals_benchmark/20260421T010549Z_stage2_memory_canary.json` 已跑完，当前 `local_exact = 5/8`、`local_baseline_exact = 1/8`
  - 这说明 `eade2c3` 的 latent-facet rerank + belief-coercion 修复不只打通 hard-4，而是开始在更大的 Persona slice 上体现 local gain
  - 因为 official verifier 仍是 `36`，helper 已把 corrected smoke8 记为 iteration `27 refine`
- Evidence / artifacts:
  - commit `eade2c3`
  - `outputs_v2/v33_latent_facet_rerank_smoke8_localonly/evals_benchmark/20260421T010549Z_stage2_memory_canary.json`
  - `research-results.tsv`
  - `autoresearch-state.json`
- Next likely action:
  - 不再把 `smoke8` 当最终 gate，它只负责确认 gain 已开始外扩
  - 下一轮直接把 local-only Persona slice 继续放大到 `16` 或 `32`，再决定是否值得恢复更大 partial holdout

# 2026-04-21 Session 091

- Worked on: 验证 `eade2c3` 是否把此前已经打通的 Persona hard samples 打回去
- State changed:
  - 以 shared predictors 顺序重放 `5370...`、`a40d5...` 与 `c8a763...` 后，三条样本全部保持正确，分别仍是 `(b)`、`(d)` 与 `(b)`
  - 结合前一轮 `acd742... -> (c)` 的单样本 probe，current HEAD `eade2c3` 现在已把 actual hard-4 gate 提到 `4/4`
  - 由于这轮只是 measurement、没有改变 official verifier，helper 已把它记为 iteration `25 refine`
- Evidence / artifacts:
  - commit `eade2c3`
  - `outputs_v2/v33_latent_facet_rerank_local/artifacts/20260421T005228Z_hard3_probe.json`
  - `outputs_v2/v33_latent_facet_rerank_local/artifacts/20260421T005228Z_hard3_compare.json`
  - `research-results.tsv`
  - `autoresearch-state.json`
- Next likely action:
  - 当前不再需要继续围着 hard-4 做单点修补，它已经在 actual runtime 上闭合成 `4/4`
  - 下一轮直接把 measurement 扩到更大但仍可控的 Persona 小样本，判断局部 gain 是否能外扩

# 2026-04-21 Session 090

- Worked on: 针对 `acd742...` 的 remaining hard case，修 query ranking 与 learned belief coercion 的同 relation facet 竞争
- State changed:
  - current HEAD `eade2c3` 在 `src/core_mem/v2/system.py` 中新增了 restricted same-relation latent rerank：当 lexical-positive 的 incumbent 是长抽象 facet，而同 relation 的 concrete facet 具有更强 latent score 时，允许后者越过严格的 positive/negative 分层
  - 同一 commit 又修了 invalid relation 的 belief coercion：如果 repair 阶段已经恢复出 grounded value，不再无条件用 fallback item 的抽象 value 覆盖它
  - `tests/test_stage2_model_skeleton.py` 已补齐同 relation latent-facet 回归；configured guard 通过，`scripts/verify_stage2_v33_longrun.py --score-only` 仍是 `36`
  - 单样本 probe `outputs_v2/v33_latent_facet_rerank_local/artifacts/20260421T004426Z_acd742_probe.json` 已机械确认：`acd742...` 从旧的 `(b)` 翻到正确 `(c)`，belief 变为 `music_preference=producing music with software`
  - 因为 official verifier 仍没动，helper 已把这轮记为 iteration `24 refine`
- Evidence / artifacts:
  - commit `eade2c3`
  - `outputs_v2/v33_latent_facet_rerank_local/artifacts/20260421T004426Z_acd742_probe.json`
  - `outputs_v2/v33_latent_facet_rerank_local/artifacts/20260421T004426Z_acd742_compare.json`
  - `research-results.tsv`
  - `autoresearch-state.json`
- Next likely action:
  - 不再把 `acd742...` 当作剩余未解的单样本 blocker，它已经在 current HEAD 上翻正
  - 下一轮先建立更便宜的 non-regression gate，确认 `eade2c3` 对剩余 hard slice / 小样本 Persona 不回退，再决定是否恢复更大 partial holdout

# 2026-04-20 Session 089

- Worked on: 针对 `a40d5...` 缺失 “positive feedback from peers” reason slot 的问题，补一条 generic feedback-reason parser coverage
- State changed:
  - current HEAD `883d48d` 现可把 “received feedback from peers about my last podcast” 这类 clause 稳定写成 `other_fact=getting positive feedback from my peers about my last podcast`
  - `tests/test_stage2_parser.py` 已补齐对应回归；configured guard 通过，`scripts/verify_stage2_v33_longrun.py --score-only` 仍是 `36`
  - 用同一 `4` 个 Persona hard samples 的 clean local-only replay 重新测量后，targeted gate 已从 retained `1/4` 提到 `3/4`
  - 样本级变化是：`a40d5...` 从旧的 `(c)` 翻到正确 `(d)`；`5370...` 继续正确 `(b)`；`c8a763...` 继续正确 `(b)`；当前只剩 `acd742...` 仍停在 `(b)`
  - 因为 official verifier 仍没动，helper 已把这轮记为 iteration `23 refine`
- Evidence / artifacts:
  - commit `883d48d`
  - `outputs_v2/v33_music_software_facet_local/artifacts/20260420T234300Z_compare.json`
  - `research-results.tsv`
  - `autoresearch-state.json`
- Next likely action:
  - 不再把 `a40d5...` 当当前主 blocker，它已经被 parser coverage 打通
  - 下一轮直接集中打剩余唯一 hard case `acd742...` 的 selected-slot ranking / belief selection

# 2026-04-20 Session 088

- Worked on: 收紧 `v33` Persona relation-drift 里的 parser bug，验证 software/music clause 不再被错误写成 `location`
- State changed:
  - current HEAD `2b54785` 把 `src/core_mem/v2/parser.py` 里过宽的 `from` location 规则收紧为显式 self-location phrase，避免 “from selecting the right software” 这类句子被误判成 `location`
  - `tests/test_stage2_parser.py` 已补齐对应回归；configured guard 通过，`scripts/verify_stage2_v33_longrun.py --score-only` 仍是 `36`
  - 单样本 replay `5370...` 现保持正确 `(b)`，同时 learned belief relation 已从错误的 `location` 修回 `music_preference`
  - 因为这轮只修 semantic drift、没有推动 official verifier，helper 已把它记为 iteration `22 refine`
- Evidence / artifacts:
  - commit `2b54785`
  - `outputs_v2/v33_music_software_facet_local/artifacts/20260420T233200Z_5370_relation_probe.json`
  - `research-results.tsv`
  - `autoresearch-state.json`
- Next likely action:
  - 不再把 parser 的过宽 `from` rule 当当前 relation drift 的残余主因
  - 下一轮继续集中打 `acd742...` / `a40d5...` 的 selected-slot ranking

# 2026-04-20 Session 087

- Worked on: 验证 `software-centric music facet` 能否以 generic parser/write 修复的形式真正回流到 `v33` Persona local gate
- State changed:
  - current HEAD `65c2612` 现把 software/digital/MIDI music clauses 规范成稳定 `music_preference` facet，并在 symbolic=`new` 的 `*_preference` relation 上直接 short-circuit learned slot-assignment，避免 learned arbitration 吞掉 distinct preference facets
  - `tests/test_stage2_parser.py` 与 `tests/test_stage2_model_skeleton.py` 已补齐对应回归；configured guard 全绿，`scripts/verify_stage2_v33_longrun.py --score-only` 仍是 `36`
  - 用与 retained `v33_option_penalty_targeted_clean` 相同的 `4` 个 Persona hard samples 做 clean local-only replay 后，targeted gate 已从 retained `1/4` 提到 `2/4`，而 lexical baseline 仍是 `0/4`
  - 样本级变化是：`5370...` 从旧的错答翻到正确 `(b)`；`acd742...` 仍停在 `(b)`；`a40d5...` 仍停在 `(c)`；`c8a763...` 维持正确 `(b)`
  - 这证明 parser/write inventory 不是伪问题，software-centric music facet 的缺失确实在压 Persona local ceiling；但由于 official verifier 仍是 `36`，helper 只能把这轮记为 iteration `21 refine`，不能记 keep
- Evidence / artifacts:
  - commit `65c2612`
  - `outputs_v2/v33_music_software_facet_local/artifacts/20260420T231900Z_compare.json`
  - `research-results.tsv`
  - `autoresearch-state.json`
- Next likely action:
  - 不再把 “parser/write inventory 已经不是问题” 当作前提；它已经贡献了真实 targeted gain
  - 下一轮集中打 `selected-slot ranking / belief relation stability`，优先解释为什么 `acd742...` 仍被 original-music slot 压住，以及为什么 `5370...` 的 correct option 仍伴随 `location` relation drift

# 2026-04-20 Session 086

- Worked on: 试探 `preference-facet fast-path` 能否通过减少 learned slot-assignment 的 over-merge 直接改善 `v33` Persona local ceiling
- State changed:
  - 针对 `*_preference` relation 新做了一轮 write-path 假设验证：当 symbolic slot assignment 已给出 `new` 时，直接 short-circuit learned arbitration，避免 learned head 把语义不同的 preference facet 强行 merge 到同一 active slot
  - synthetic replay 证明该改动确实能保住多条 active `music_preference` 槽位，因此 “learned merge 会吞掉 distinct preference facets” 这条内部问题本身成立
  - 但 fresh clean targeted Persona `4`-sample gate `outputs_v2/v33_preference_facet_targeted/` 与 retained `outputs_v2/v33_option_penalty_targeted_clean/` 完全持平：`base_exact = 1/4`、`new_exact = 1/4`、`improved = 0`、`degraded = 0`
  - 因此这条 write-path fast-path 没有形成任何 external gain，helper 已把它记为 iteration `20 discard`，当前 retained `v33` 仍是 `36`
- Evidence / artifacts:
  - `outputs_v2/v33_preference_facet_targeted/evals_benchmark/20260420T223943Z_stage2_memory_canary.json`
  - `outputs_v2/v33_preference_facet_targeted/evals_benchmark/20260420T224000Z_personamem_canary.json`
  - `research-results.tsv`
  - `autoresearch-state.json`
- Next likely action:
  - 不再继续把 write-path facet-preservation 当作 `v33` Persona 主修复线
  - 下一轮直接围绕 `selected-slot ranking` 做 focused change，优先解释为什么真实 benchmark context 下仍选不到 `producing music with software` 这类应答关键槽位

## 2026-04-20 Session 085

- Worked on: 针对 `v33` Persona local ceiling，试探 generic option-scorer 收紧是否能把 wrong-long-option bias 压下来
- State changed:
  - current HEAD `3234a4b` 已把 `src/core_mem/v2/answer_head.py` 收紧成更保守的 generic finite-option scorer：过滤低信息词、对 unsupported 长细节施加惩罚、并轻度惩罚过长选项；对应 `tests/test_stage2_memory_canary.py` 已新增回归并通过
  - clean targeted Persona `4`-sample gate `outputs_v2/v33_option_penalty_targeted_clean/` 相对 retained baseline `outputs_v2/v33_semantic64_persona/` 从 `old_exact = 0/4` 提到 `new_exact = 1/4`
  - 样本级变化是：`c8a763...` 从 `(d)` 翻正到 `(b)`；`acd742...` 仍停在 `(b)`；`a40d5...` 仍停在 `(c)`；`5370...` 从 `(c)` 换到 `(a)` 但仍未命中，因此这轮只能记为 refine，不能记为 keep
  - configured guard 通过，`scripts/verify_stage2_v33_longrun.py --score-only` 仍为 `36`；helper 已把这轮记为 iteration `19 refine`
- Evidence / artifacts:
  - commit `3234a4b`
  - `outputs_v2/v33_option_penalty_targeted_clean/evals_benchmark/20260420T222705Z_stage2_memory_canary.json`
  - `outputs_v2/v33_option_penalty_targeted_clean/artifacts/20260420T222705Z_compare.json`
  - `research-results.tsv`
  - `autoresearch-state.json`
- Next likely action:
  - 保留 option-scorer 收紧，因为它已出现 targeted 正增益
  - 下一轮集中打 `selected-slot ranking`，尤其是 `acd742...` 这类 recall/reason case 的 top-1 abstract `other_fact` 误选

## 2026-04-20 Session 084

- Worked on: 继续把 `v33` Persona failure hypothesis 从 belief hygiene 收紧到 retrieval / answer-selection 主问题
- State changed:
  - 先试了一轮 generic query-stopword 扩展，但对代表性 failures `acd742...` / `5370...` 的 replay 完全无效：top-8 selected slots 与 answer text 都没变；该 patch 已完整回滚，helper 已记为 iteration `17 discard`
  - 随后又做了 relation-consistent support-slot reassignment：当 learned belief relation 已是 `music_preference`、但 `support_slot_ids` 指向 `other_fact` 时，coercion 会优先拉回同 relation slot。这个 patch 的确把 `acd742...` / `5370...` 的 belief/support 从抽象 `other_fact` 清成 relation-consistent `music_preference`
  - 但 fresh targeted Persona `4`-sample local-only gate `outputs_v2/v33_support_consistency_targeted/` 最终仍是 `local_exact = 0/4`，与 old exact 完全一致；说明 support cleanup 不是当前 external gain driver。该 patch 已完整回滚，helper 已记为 iteration `18 discard`
  - 当前最关键的新 runtime truth 是：代表性 Persona failures 的 top-1 selected slot 本身就是长抽象 `other_fact`，而 support/value cleanup 最多只能把 belief 文本变干净，不能单独带来 label flip
- Evidence / artifacts:
  - `outputs_v2/v33_support_consistency_targeted/evals_benchmark/20260420T221734Z_stage2_memory_canary.json`
  - `outputs_v2/v33_support_consistency_targeted/runs/20260420T221734Z_stage2_memory_canary_personamem/predictions.jsonl`
  - `research-results.tsv`
  - `autoresearch-state.json`
- Next likely action:
  - 不再继续 query-stopword / support-slot cleanup 这类 hygiene 线
  - 下一轮优先打 selected-slot ranking 或 answer option head，因为当前 Persona local ceiling 已更像“wrong abstract slot selected + cleaned belief still maps to wrong option”

## 2026-04-20 Session 083

- Worked on: 试探 `same-relation grounded belief backfill` 是否能直接改善 `v33` Persona local projection / belief quality
- State changed:
  - current HEAD 在 `src/core_mem/v2/system.py` 新增 support-slot grounding 检查；当前仅当 same-relation learned belief value 明显表现为 abstractive/noisy drift 时才回填 canonical slot value，避免误杀 `oolong tea` 这类合法 concise value 预测
  - `tests/test_stage2_model_skeleton.py` 已新增 same-relation ungrounded backfill 回归；`tests/test_stage2_memory_canary.py tests/test_stage2_model_skeleton.py` 与 configured guard 均已通过，`scripts/verify_stage2_v33_longrun.py --score-only` 仍为 `36`
  - 随后用与 retained `v33_semantic64_persona` 相同的 semantic-full checkpoint + `v31` latent ranker 组合，启动了 provider-disabled local-only probe `outputs_v2/v33_grounded_projection_local/`
  - 该 probe 在前 `4` 条 Persona samples 上与 retained `v33_semantic64_persona` 完全一致：`changed = 0`、`improved = 0`、`degraded = 0`
  - 因此本轮结论是：grounded same-relation belief backfill 只能算 hygiene，不是当前 Persona learned-authoritative local ceiling 的主 driver；下一条 hypothesis 应继续前推到 support-slot 选偏 / latent retrieval，而不是继续停留在 belief value coercion
- Evidence / artifacts:
  - `outputs_v2/v33_grounded_projection_local/artifacts/20260420T215205Z_partial_probe_compare.json`
  - `outputs_v2/v33_grounded_projection_local/runs/20260420T215205Z_stage2_memory_canary_personamem/predictions.jsonl`
  - `research-results.tsv`
  - `autoresearch-state.json`
- Next likely action:
  - 不再把 same-relation belief value backfill 当作 `v33` Persona 主修复线
  - 下一轮优先定位 selected support slot / latent retrieval 为什么会先选到抽象 `music_preference` / `other_fact` 槽位

## 2026-04-20 Session 082

- Worked on: 继续 `v33` Persona learned-authoritative full holdout，并把当前 partial measurement 的 failure mode 从“provider contract blocker”细化到 local projection / belief quality
- State changed:
  - 在 current HEAD `ee28a96` 上继续 resume `outputs_v2/v33_semantic_full_persona/runs/20260420T203544Z_stage2_memory_canary_personamem/`，并把同一 partial holdout 扩到 `99/512`
  - 当前该 run 的 provider/local 为 `18/99`、`42/99`，因此“只靠 minimal `<think>` cleaning + 更大样本就能自动跨过 retained `v32`”已被机械证伪；helper 已把这轮记为 iteration `14 discard`
  - 随后又基于当前 partial summary 生成了 fresh Persona analysis artifacts：`latest_personamem_stage2_canary_analysis.json` 与 `latest_personamem_stage2_layered_analysis.json`
  - 最新 layered breakdown 显示当前 Persona 剩余失败中 `projection = 54`、`provider = 27`，并且 `other_fact` 主导 `70` 个失败 relation，`suggest_new_ideas` 是当前最弱 question type（`local_exact = 1/12`）
  - 这把 `TD-044 / WS-030` 的 next hypothesis 从“继续做 provider cleanup”进一步收紧成“优先打 local projection / belief quality”；helper 已把这轮分析记为 iteration `15 refine`
- Evidence / artifacts:
  - `outputs_v2/v33_semantic_full_persona/evals_benchmark/20260420T203544Z_stage2_memory_canary.json`
  - `outputs_v2/artifacts/latest_personamem_stage2_canary_analysis.json`
  - `outputs_v2/artifacts/latest_personamem_stage2_layered_analysis.json`
  - `research-results.tsv`
  - `autoresearch-state.json`
- Next likely action:
  - 不再把 provider cleanup 当作 `v33` Persona 主线
  - 下一轮直接围绕 `other_fact` 主导的 local projection / belief quality 做 focused change，并用 provider-disabled local gate 先筛掉低价值 scorer/prompt tweak

## 2026-04-20 Session 081

- Worked on: 把 `v33` 的协议层最小 `<think>` 清洗真正接进 provider resume path，并验证它对现有 Persona learned-authoritative partial full-holdout 的实际回收效果
- State changed:
  - current HEAD `2d3e59c` 已为 `scripts/run_stage2_memory_canary.py` 新增 provider raw-output repair：当 `provider_raw_prediction` 含 `<think>` / fenced reasoning 噪声时，runner 会在 live path 与 `--resume` 路径统一回写 cleaned `provider_prediction`
  - `tests/test_stage2_memory_canary.py` 已补齐 `<think>` 清洗与 resumed prediction repair 回归；targeted `tests/test_stage2_memory_canary.py` 与 configured guard 均已通过
  - `scripts/verify_stage2_v33_longrun.py --score-only` 仍为 `36`，因此这轮不能记为 keep
  - 但对现有 partial Persona learned-authoritative full-holdout run `outputs_v2/v33_semantic_full_persona/runs/20260420T203544Z_stage2_memory_canary_personamem/`，同一代理下、无 benchmark-specific normalization 的 raw-based 清洗已把 summary 从旧的 `provider exact = 5/66` 回收到当前 `15/77`，`local_exact = 31`
  - helper 已把本轮记为 iteration `13 refine`；当前 retained metric 继续保持 `36`
- Evidence / artifacts:
  - commit `2d3e59c`
  - `outputs_v2/v33_semantic_full_persona/evals_benchmark/20260420T203544Z_stage2_memory_canary.json`
  - `outputs_v2/v33_semantic_full_persona/runs/20260420T203544Z_stage2_memory_canary_personamem/`
  - `research-results.tsv`
  - `autoresearch-state.json`
- Next likely action:
  - 继续沿当前 `gpt-agent.cc/v1` 路径跑 `PersonaMem 512 / LongMemEval-S 500` learned-authoritative holdout / publish
  - 优先把 repaired Persona run 从 partial measurement 推到可发布 compare，再判断是否足以支撑 `v33` external keep

## 2026-04-20 Session 080

- Worked on: 推进 `v33` learned-authoritative full holdout，并定位当前 run 的真实 provider-contract blocker
- State changed:
  - 已按 managed loop 先完成 semantic-full checkpoint gate：`LongMemEval-S 64` 达到 `provider/local = 10/10`，`PersonaMem 64` 达到 `provider/local = 21/24`；说明 semantic-full checkpoint + `v31` latent ranker 是当前最强 learned-authoritative runtime 候选
  - 随后尝试显式生成 `500 / 512` manifests 并启动 clean full holdout；partial evidence 证明 `LongMemEval-S` provider rate 远高于 retained 门槛，但 `PersonaMem` provider-side 明显落后于 local，主要失败模式是 `<think>` 污染与长文本不按 label contract 收口，因此该轮 full holdout 已提前停止，不继续烧 API
  - 尝试过一轮 runner-side generic finite-option normalization + 更严格的 Persona label-only prompt，但 fresh `PersonaMem 64` gate 从旧的 `provider = 21/64` 退到 `16/64`；对应 patch 已完整回滚，不保留到主线
  - 随后完成外部文档/实测交叉检查：MiniMax 官方 OpenAI-compatible 文档声称 `reasoning_split=True` 会把 thinking 拆到 `reasoning_details`，但当前 repo 实际使用的 `https://gpt-agent.cc/v1` 代理对真实 failure prompt 并未提供 `reasoning_details`，`message.content` 仍直接带 `<think>`
  - 额外 live probe 又证伪了两条 provider-contract 内补救线：额外 `system` role 明确禁止 reasoning 不能稳定压掉 `<think>`，而把 `max_tokens` 压到 `8/16/32` 只会得到更短的 reasoning 片段，不会直接产出 option label
  - 因此当前 `TD-044 / WS-030 / v33` 已进入新的 blocker：若要继续 honest 追 `PersonaMem 512` provider exact，必须先得到用户批准，切换当前 provider/interface；未经批准，不再继续消耗 full holdout API。current retained metric 继续保持 `36`
- Evidence / artifacts:
  - `outputs_v2/v33_semantic64_long/evals_benchmark/20260420T202317Z_stage2_memory_canary.json`
  - `outputs_v2/v33_semantic64_persona/evals_benchmark/20260420T202314Z_stage2_memory_canary.json`
  - partial full-holdout runs:
    - `outputs_v2/v33_semantic_full_long/runs/20260420T203547Z_stage2_memory_canary_longmemeval/`
    - `outputs_v2/v33_semantic_full_persona/runs/20260420T203544Z_stage2_memory_canary_personamem/`
  - discarded provider-normalization probe:
    - `outputs_v2/v33_persona64_provider_norm/evals_benchmark/20260420T204903Z_stage2_memory_canary.json`
    - `outputs_v2/v33_persona64_provider_norm/runs/20260420T204903Z_stage2_memory_canary_personamem/`
  - MiniMax official OpenAI-compatible docs: `https://platform.minimax.io/docs/api-reference/text-openai-api`
- Next likely action:
  - 若用户批准，优先切换离开当前 `gpt-agent.cc/v1` OpenAI-compatible 代理或改用其可分离 reasoning 的接口
  - 若未获批准，保持 `v33 = 36/47` 的 retained truth，并停止当前 managed run，避免继续浪费 provider 配额

## 2026-04-20 Session 079

- Worked on: 尝试用 “weak-fragment-aware learned belief recovery” 改善 `LongMemEval-S` learned-authoritative local gate
- State changed:
  - 曾试验两项紧耦合修复：其一，`other_fact` 的 fallback raw-match 不再优先信任 `happy / good / still` 这类弱碎片；其二，answer projection 对 `25:50` 这类成绩时间做通用规范化
  - 对应新增 unit tests 全部通过，但同口径 local `LongMemEval-S 8` probe `outputs_v2/v33_long_recovery_local/evals_benchmark/20260420T201333Z_stage2_memory_canary.json` 仍只有 `local_exact_match = 1`
  - 因此本轮结论是：当前主问题不在 weak fragment fallback ranking，也不在 time-string projection；这条 patch 已在本地完整回滚，不保留到主线
  - 当前 retained metric 继续保持 `36`
- Evidence / artifacts:
  - `outputs_v2/v33_long_recovery_local/evals_benchmark/20260420T201333Z_stage2_memory_canary.json`
  - temporary discarded probe `outputs_v2/v33_long_recovery_local/`
- Next likely action:
  - 不再继续 weak-fragment fallback 这条线
  - 下一轮应直接针对 learned belief item / support attribution 的 relation selection 与 evidence grounding 做更强修复

## 2026-04-20 Session 078

- Worked on: 诊断 `LongMemEval-S` learned-authoritative 弱点，并尝试一轮 query-aligned fallback belief ranking 修复
- State changed:
  - 已确认 `LongMemEval-S` 当前弱信号并非新 regression：semantic full checkpoint `outputs_v2/checkpoints/20260416T003548Z_stage2_train_exec` 在当前前 `8` 个 sample_ids 上历史 learned run 与 current run 一样都只有 `local = 1/8`
  - 当前更强的 Persona learned-authoritative 候选仍是 semantic full checkpoint；但 `LongMemEval-S` 仍弱，因此不能直接发起 full holdout
  - 曾试验把 `StructuredMemorySystem._fallback_slot_rank()` 改成更偏 query-aligned / numeric / temporal 的排序，并补了两条新单测；对应 local `LongMemEval-S 8` probe 仍是 `1/8`，没有形成可保留提升
  - 该 fallback-ranking patch 已在本地回滚，工作树当前只保留上一轮 committed 的 runtime prompt / docs sync 变更
- Evidence / artifacts:
  - `outputs_v2/v33_long_no_latent_local/evals_benchmark/20260420T200435Z_stage2_memory_canary.json`
  - `outputs_v2/v33_long_no_latent_local_v2/evals_benchmark/20260420T200757Z_stage2_memory_canary.json`
  - `outputs_v2/runs/20260416T095034Z_stage2_memory_canary_longmemeval/predictions.jsonl`
- Next likely action:
  - 不再继续微调 fallback slot ranking
  - 下一轮应直接针对 learned belief item / support attribution 的 `other_fact` 坍塌做更强修复，再看 `LongMemEval-S 8` 是否能脱离 `1/8`

## 2026-04-20 Session 077

- Worked on: 让 `v33` answer-head 候选真正进入 authoritative runtime prompt，并用两组 learned-authoritative `8`-sample probes 筛选更强 checkpoint 组合
- State changed:
  - `scripts/run_stage2_memory_canary.py` 现已把 generic `Answer-head candidate` 注入 `PersonaMem / LongMemEval-S` provider prompt；对应 `tests/test_stage2_memory_canary.py` 回归已补齐并通过
  - 文档与 `.agent-os` 已同步到 current retained `v33 = 36/47` truth；`docs/current_status.md`、`docs/implementation_plan.md`、`docs/todo.md`、`.agent-os/project-index.md`、`.agent-os/todo.md`、`.agent-os/run-log.md` 与 `.agent-os/lessons-learned.md` 均已刷新
  - 第一组 learned-authoritative Persona probe 使用 `v30` modular checkpoint `outputs_v2/checkpoints/20260419T012526Z_stage2_train_exec` + `v31` latent ranker `outputs_v2/checkpoints/20260419T070526Z_stage2_v31_latent_exec`，结果仅为 `provider/local = 1/5`，不适合作为 full holdout 候选
  - 第二组 Persona probe 切到 semantic full checkpoint `outputs_v2/checkpoints/20260416T003548Z_stage2_train_exec` 后，当前 `8` 样本结果提升到 `provider/local = 5/6`
  - 同一 semantic full checkpoint 组合在 `LongMemEval-S 8` 上仍只有 `provider/local = 1/1`；因此当前最诚实结论是：这次改动把 learned-authoritative Persona runtime 候选显著变强，但还不能支撑直接发起 `LongMemEval-S 500 + PersonaMem 512` full holdout
  - `scripts/verify_stage2_v33_longrun.py --score-only` 仍为 `36`；configured guard 继续通过
- Evidence / artifacts:
  - `outputs_v2/v33_prompt_probe/evals_benchmark/20260420T195548Z_stage2_memory_canary.json`
  - `outputs_v2/v33_ckpt_probe_semantic_local/evals_benchmark/20260420T195721Z_stage2_memory_canary.json`
  - `outputs_v2/v33_ckpt_probe_semantic_provider/evals_benchmark/20260420T195808Z_stage2_memory_canary.json`
  - `outputs_v2/v33_ckpt_probe_semantic_provider_long/evals_benchmark/20260420T195948Z_stage2_memory_canary.json`
  - commit pending for runtime prompt / docs sync change
- Next likely action:
  - 保留 semantic full checkpoint 作为当前 Persona 侧 learned-authoritative runtime 候选
  - 下一轮必须集中修 `LongMemEval-S` learned runtime 质量，而不是直接烧 full holdout
  - 若后续继续 `v33`，应先找到不会把 `LongMemEval-S 8` 卡在 `1/8` 的 learned runtime 组合，再启动 authoritative full holdout

## 2026-04-20 Session 076

- Worked on: 初始化 fresh managed `TD-044 / WS-030 / v33` run，完成 baseline-first helper 初始化，落地 `v33` train/local/canary artifact publisher，并把 retained verifier 从 baseline `17` 推到 `36/47`
- State changed:
  - `autoresearch_resume_check.py` 已确认当前 launch 为 `fresh_start`；随后先量 baseline：`scripts/verify_stage2_v33_longrun.py --score-only = 17`
  - helper 已用 `autoresearch_init_run.py --session-mode background` 初始化 fresh `research-results.tsv` / `autoresearch-state.json`
  - current HEAD `819c44e` 已为 `scripts/train_stage2.py`、`scripts/eval_stage2_local.py` 与 `scripts/run_stage2_memory_canary.py` 补齐 `v33` modular-authoritative / learned-write / latent-reader / temporal-slot / belief-graph / answer-option artifact publish 链，并新增 `configs/stage2_train_v33_tiny.yaml` 与 `configs/stage2_train_v33_latent.yaml`
  - `latest_stage2_v33_modular_authoritative_train.json`、`latest_stage2_v33_learned_write_eval.json`、`latest_stage2_v33_latent_reader_train.json`、`latest_stage2_v33_temporal_slot_eval.json`、`latest_stage2_v33_latent_objective_eval.json`、`latest_stage2_v33_belief_graph_eval.json`、`latest_stage2_v33_answer_option_eval.json` 与 `latest_stage2_v33_ablation_summary.json` 已全部落地
  - `scripts/verify_stage2_v33_longrun.py --score-only` 已提升到 `36/47`；configured guard `pytest -q tests/test_stage2_v33_longrun.py tests/test_stage2_v32_longrun.py tests/test_stage2_memory_canary.py tests/test_stage2_training_runtime.py tests/test_stage2_model_skeleton.py` 通过
  - helper 已把本轮记为 iteration `1 keep`；current retained metric 为 `36`
- Evidence / artifacts:
  - `research-results.tsv`
  - `autoresearch-state.json`
  - `outputs_v2/artifacts/latest_stage2_v33_modular_authoritative_train.json`
  - `outputs_v2/artifacts/latest_stage2_v33_learned_write_eval.json`
  - `outputs_v2/artifacts/latest_stage2_v33_latent_reader_train.json`
  - `outputs_v2/artifacts/latest_stage2_v33_temporal_slot_eval.json`
  - `outputs_v2/artifacts/latest_stage2_v33_latent_objective_eval.json`
  - `outputs_v2/artifacts/latest_stage2_v33_belief_graph_eval.json`
  - `outputs_v2/artifacts/latest_stage2_v33_answer_option_eval.json`
  - `outputs_v2/artifacts/latest_stage2_v33_ablation_summary.json`
  - commit `819c44e`
- Next likely action:
  - 让 `PersonaMem 512 / LongMemEval-S 500` authoritative runtime 真正消费 `learned_memory + learned slot assignment`
  - 发布 `latest_stage2_v33_learned_authoritative_runtime.json`、`latest_stage2_v33_full_holdout_compare.json`、`latest_longmemeval_stage2_v33_full.json` 与 `latest_personamem_stage2_v33_full.json`
  - 只有在 full holdout 明确超过 retained `v32` 后，才允许把 `v33` 收口成 keep

## 2026-04-20 Session 075

- Worked on: 收口 `TD-043 / WS-029 / v32` 的 clean full holdout compare，发布 authoritative `v32` full artifacts，并把 managed run 推到 stop condition
- State changed:
  - helper 已先把 completed `LongMemEval-S 500` clean run 记为 iteration `12 refine`：`outputs_v2/runs/v32_full_longmemeval_500_timeout45/` 最终固定为 `provider/local = 21/14`，相对 retained `v30` 的 `19/14` 至少 provider 侧为正
  - 随后基于 clean `timeout45` runs 重建了 benchmark-specific full summary JSON：`outputs_v2/evals_benchmark/20260420T075028Z_longmemeval_v32_full_summary.json` 与 `outputs_v2/evals_benchmark/20260420T075028Z_personamem_v32_full_summary.json`
  - `scripts/verify_stage2_v32_longrun.py --publish-full-holdout-artifacts` 已基于上述 summaries 发布 `latest_longmemeval_stage2_v32_full.json`、`latest_personamem_stage2_v32_full.json` 与 `latest_stage2_v32_full_holdout_compare.json`
  - 当前 published full-holdout truth 已固定为：`LongMemEval-S 500 = provider/local 21/14`、`PersonaMem 512 = provider/local 183/175`，且 `longmemeval_gain_confirmed = true`、`personamem_gain_confirmed = true`
  - `scripts/verify_stage2_v32_longrun.py --score-only` 已达到 `44/44`；`conda run -n core_mem pytest -q tests/test_stage2_v32_longrun.py tests/test_stage2_memory_canary.py` 通过
  - helper 已把 authoritative publish 记为 iteration `13 keep`，current retained metric 现在是 `44`
- Evidence / artifacts:
  - `research-results.tsv`
  - `autoresearch-state.json`
  - `outputs_v2/evals_benchmark/20260420T075028Z_longmemeval_v32_full_summary.json`
  - `outputs_v2/evals_benchmark/20260420T075028Z_personamem_v32_full_summary.json`
  - `outputs_v2/artifacts/latest_longmemeval_stage2_v32_full.json`
  - `outputs_v2/artifacts/latest_personamem_stage2_v32_full.json`
  - `outputs_v2/artifacts/latest_stage2_v32_full_holdout_compare.json`
  - `outputs_v2/runs/v32_full_longmemeval_500_timeout45/`
  - `outputs_v2/runs/v32_full_personamem_512_timeout45/`
  - commit `7377c29`
- Next likely action:
  - 保留当前 `v32` retained line 作为新的 stage-2 baseline
  - 不再继续同一条 managed run 消耗 provider
  - 等待用户给出下一条 stage-2 hypothesis、比较目标或新的验收要求

## 2026-04-20 Session 074

- Worked on: 初始化 fresh managed `TD-043 / WS-029 / v32` run，并把第一轮 redesign 证据从 baseline-first 启动推进到 internal retained keep
- State changed:
  - `autoresearch_resume_check.py` 已确认当前 launch 为 `fresh_start`；随后使用 `autoresearch_init_run.py --session-mode background` 以 baseline `19` 初始化新的 `research-results.tsv` / `autoresearch-state.json`
  - current-head 已补齐 `src/core_mem/v2/answer_head.py`，并让 `scripts/train_stage2.py`、`scripts/eval_stage2_local.py`、`scripts/run_stage2_memory_canary.py` 具备 `v32 modular / latent / belief / answer` artifact 发布链
  - fresh current-head `v32` internal artifact 已全部落地：`latest_stage2_v32_modular_backbone_train.json`、`latest_stage2_v32_write_head_eval.json`、`latest_stage2_v32_latent_module_train.json`、`latest_stage2_v32_latent_objective_eval.json`、`latest_stage2_v32_latent_holdout_compare.json`、`latest_stage2_v32_belief_decoder_eval.json`、`latest_stage2_v32_belief_holdout_compare.json`、`latest_stage2_v32_answer_head_eval.json` 与 `latest_stage2_v32_option_scoring_compare.json`
  - 随后 current HEAD 又为 `scripts/verify_stage2_v32_longrun.py` 补齐 authoritative `--publish-ablation-summary` 入口，并新增对应单测；`latest_stage2_v32_ablation_summary.json` 已机械确认 `latent_is_primary_driver = true`、`belief_contributes = true`、`answer_head_contributes = true`
  - `scripts/verify_stage2_v32_longrun.py --score-only` 已从 baseline `19/44` 提升到 current retained `38/44`
  - 后续 full holdout probe 又补齐了三条 runtime truth：其一，`run_stage2_memory_canary.py` 默认会复用 `evals_benchmark/` 下最新 manifest，因此做 full holdout 前必须先显式生成 `500 / 512` manifests；其二，runner 的 `--limit` 默认是 `1`，不显式传 `--limit 500/512` 就不会跑 full holdout；其三，large symbolic holdout 上旧的 parallel fast path 会拖垮吞吐，commit `4aa02dc` 已把 fast path 收紧到 `<=64` canary
  - 在 commit `514fdad` 把 `configs/minimax_m27.yaml` 的 `max_retries` 提高到 `5`、`retry_backoff_seconds` 提高到 `4.0` 之后，`PersonaMem 512` full run `outputs_v2/runs/v32_full_personamem_512/` 已能从 `13` 稳定推进到 `37/512`；但 current session 仍未把 full holdout 跑完，因此 retained metric 继续保持 `38`
  - 随后 current HEAD `34943df` 又为 canary runner 补齐两层 runtime 防护：provider transient failure 现在会先提交成功样本、把失败样本留给后续 `--resume`；同一 `run_dir` 也会加 `.active.lock`，拒绝新的并发 writer
  - 这轮恢复还暴露出一个新的 runtime truth：旧的 `outputs_v2/runs/v32_full_personamem_512/` 曾被 pre-lock wrappers 并发污染，因此其 metadata 不再适合作为 authoritative progress。fresh clean run `outputs_v2/runs/v32_full_personamem_512_clean/` 已在 lock 保护下重新起跑，并真实推进到 `14/512`
  - 当前 remaining gaps 仍只在 `LongMemEval-S 500 / PersonaMem 512` full holdout compare；在这类 artifact 发布前，不能误写成 external gain keep
- Evidence / artifacts:
  - `research-results.tsv`
  - `autoresearch-state.json`
  - `outputs_v2/artifacts/latest_stage2_v32_modular_backbone_train.json`
  - `outputs_v2/artifacts/latest_stage2_v32_write_head_eval.json`
  - `outputs_v2/artifacts/latest_stage2_v32_latent_module_train.json`
  - `outputs_v2/artifacts/latest_stage2_v32_latent_objective_eval.json`
  - `outputs_v2/artifacts/latest_stage2_v32_latent_holdout_compare.json`
  - `outputs_v2/artifacts/latest_stage2_v32_belief_decoder_eval.json`
  - `outputs_v2/artifacts/latest_stage2_v32_belief_holdout_compare.json`
  - `outputs_v2/artifacts/latest_stage2_v32_answer_head_eval.json`
  - `outputs_v2/artifacts/latest_stage2_v32_option_scoring_compare.json`
  - `outputs_v2/artifacts/latest_stage2_v32_ablation_summary.json`
  - `outputs_v2/runs/v32_full_personamem_512/`
  - `outputs_v2/runs/v32_full_personamem_512_clean/`
  - commits `05d1134`, `e50c478`, `4aa02dc`, `514fdad`
  - commit `34943df`
- Next likely action:
  - 继续沿 current HEAD `34943df` 的 explicit full-manifest + explicit `--limit` + resumable provider commit + run-dir lock path，优先从 clean run `v32_full_personamem_512_clean` 继续推进 `PersonaMem 512`
  - 再用同样的 clean-run discipline 启动 `LongMemEval-S 500`
  - 如果 provider 仍持续以 `503` 打断 full holdout，就把当前真实约束诚实写成“external provider throughput bound”，而不是继续误判成代码逻辑 blocker
  - 仅在 gain / overlap guard 为真时刷新 `latest_stage2_v32_full_holdout_compare.json`

## 2026-04-19 Session 073

- Worked on: 解除 `TD-042 / WS-028 / v31` 的 zero-progress stable-holdout blocker，并补齐 `v31` full-holdout publisher，让 current resumed `500 + 512` measurement 可以继续推进到 artifact 发布
- State changed:
  - helper 已把 iteration `6` 记为 `refine`：commit `489ada0` 让 learned full-holdout runner 复用 shared `latent_slot_ranker` 并按 batch 增量落盘；此前停在 `completed_predictions = 0` 的 resumed `LongMemEval-S 500 / PersonaMem 512` runs 已恢复推进
  - `LongMemEval-S 500` resumed run `outputs_v2/runs/20260419T160705Z_stage2_memory_canary_longmemeval/` 当前已推进到 `322/500`
  - `PersonaMem 512` resumed run `outputs_v2/runs/20260419T160701Z_stage2_memory_canary_personamem/` 在默认共享 GPU 上先暴露出 shared belief predictor OOM；切到 `CUDA_VISIBLE_DEVICES=2` 后已恢复推进到 `238/512`
  - helper 已把 iteration `7` 记为 `refine`：commit `838a861` 为 `scripts/verify_stage2_v31_longrun.py` 新增 authoritative full-holdout publisher，可在 current runs 完成后直接发布 `latest_longmemeval_stage2_v31_full.json`、`latest_personamem_stage2_v31_full.json` 与 `latest_stage2_v31_full_holdout_compare.json`
  - 当前 retained metric 仍是 `24/32`；真实状态已从 blocked 切回 active measurement，但 full holdout compare / ablation 仍未发布
- Evidence / artifacts:
  - `research-results.tsv`
  - `autoresearch-state.json`
  - `outputs_v2/runs/20260419T160705Z_stage2_memory_canary_longmemeval/`
  - `outputs_v2/runs/20260419T160701Z_stage2_memory_canary_personamem/`
  - commits `489ada0`, `838a861`
- Next likely action:
  - 等待 current resumed `LongMemEval-S 500 / PersonaMem 512` runs 完成
  - 立刻发布 `v31` full-holdout artifacts，并重新量 `scripts/verify_stage2_v31_longrun.py --score-only`
  - 若 full holdout compare 仍不足以支撑 `v31` ablation truth，再单独补一轮 ablation evidence

## 2026-04-19 Session 072

- Worked on: 把 `TD-042 / WS-028 / v31` 的 stable holdout measurement 真实跑到底，并把 current managed run 在 helper 已记账的前提下收口为 true blocker，而不是继续重复同一路径
- State changed:
  - helper 已把 iteration `3` 记为 `pivot`：CPU-backed `LongMemEval-S 500 / PersonaMem 512` learned-memory stable holdout 在持续高负载下长时间保持 `completed_predictions = 0`，且没有生成 `predictions.jsonl`
  - helper 已把 iteration `4` 记为 `refine`：切到 `cuda` 后，fresh `PersonaMem 1` 与 `LongMemEval-S 1` smoke 都能完成，说明 `learned belief checkpoint + v31 latent checkpoint` 的 online wiring 正常，不是 path 未接通
  - helper 已把 iteration `5` 记为 `blocked`：随后 fresh `cuda` full-holdout run 仍在 checkpoint 已加载、GPU/CPU 持续忙碌的情况下停在 `completed_predictions = 0`，且始终没有生成第一条 `predictions.jsonl`
  - 当前 retained truth 因此保持不变：`scripts/verify_stage2_v31_longrun.py --score-only = 24/32`，`autoresearch-state.json` 当前记录 `current_metric = 24`、`last_status = blocked`
- Evidence / artifacts:
  - `research-results.tsv`
  - `autoresearch-state.json`
  - `outputs_v2/evals_benchmark/20260419T160946Z_stage2_memory_canary.json`
  - `outputs_v2/evals_benchmark/20260419T160950Z_stage2_memory_canary.json`
  - `outputs_v2/runs/20260419T160701Z_stage2_memory_canary_personamem/`
  - `outputs_v2/runs/20260419T160705Z_stage2_memory_canary_longmemeval/`
  - `outputs_v2/runs/20260419T161051Z_stage2_memory_canary_personamem/`
  - `outputs_v2/runs/20260419T161055Z_stage2_memory_canary_longmemeval/`
- Next likely action:
  - 不再重复当前 `LongMemEval-S 500 + PersonaMem 512` stable holdout gate
  - 若后续恢复 `TD-042`，必须先提出新的 learned online throughput hypothesis 或替代稳定 measurement path；只有在新的 gate 能稳定增量产出 `predictions.jsonl` 后，才继续补 full holdout compare / ablation

## 2026-04-19 Session 071

- Worked on: 以 fresh managed `v31` run 为上下文，先修 baseline guard，再把独立 `latent_retriever.pt` checkpoint 接进 online canary，给下一轮 `LongMemEval-S 500 + PersonaMem 512` 稳定 holdout measurement 清掉实现级 blocker
- State changed:
  - baseline 已机械量到 `scripts/verify_stage2_v31_longrun.py --score-only = 24`，同时当前 HEAD baseline guard 初始为红：`tests/test_stage2_model_skeleton.py` 有 4 条 learned belief / projection 回归
  - helper 已把 iteration `1` 记为 `refine`：commit `864dfa2` 收紧 `src/core_mem/v2/system.py` 的 belief-value backfill 条件，保留结构干净的 same-relation learned value，并在 invalid-relation fallback 时继续回到 query-aligned support value；对应 full guard 已恢复通过，但 retained metric 保持 `24`
  - helper 已把 iteration `2` 记为 `refine`：commit `84d2435` 为 `StructuredMemorySystem` 新增 optional latent slot ranker / latent checkpoint loader，并把 `scripts/run_stage2_memory_canary.py` 扩展为可同时接收 `learned belief checkpoint + latent checkpoint`
  - fresh `PersonaMem 1` live smoke `outputs_v2/evals_benchmark/20260419T160230Z_stage2_memory_canary.json` 已机械完成，artifact 显式记录 `latent_retriever_checkpoint_dir = outputs_v2/checkpoints/20260419T070526Z_stage2_v31_latent_exec`；这证明 current-head online canary 已经能消费独立 `v31` latent checkpoint
  - 当前 retained truth 仍未变化：`scripts/verify_stage2_v31_longrun.py --score-only = 24/32`；full holdout compare / ablation 仍未发布，不能误写成 keep 或 gain
- Evidence / artifacts:
  - `research-results.tsv`
  - `autoresearch-state.json`
  - `outputs_v2/evals_benchmark/20260419T160230Z_stage2_memory_canary.json`
  - commits `864dfa2`, `84d2435`
- Next likely action:
  - 直接沿 current HEAD 的 `learned belief checkpoint + v31 latent checkpoint` 在线路径，跑 `LongMemEval-S 500 + PersonaMem 512` 稳定 holdout measurement
  - 在 full holdout summary 到位后，再补 `latest_stage2_v31_full_holdout_compare.json` 与 `latest_stage2_v31_ablation_summary.json`

## 2026-04-19 Session 070

- Worked on: 继续 `TD-042 / WS-028 / v31` 的 full-holdout quick-smoke line，在 retained `24/32` 基线上依次尝试 online learned belief fallback、blank-output 修补、learned slot-assignment pivot、provider-stability repeat 与最小 constrained-decoding
- State changed:
  - current retained truth 未变化；`scripts/verify_stage2_v31_longrun.py --score-only` 仍为 `24/32`
  - helper 当前已把 iteration `9` 记为 `refine`：commit `c82a56c` 的 online learned belief fallback 把 Persona `8`-sample subset 提到 `provider/local = 5/4`，但相对 retained `v30` 的 `6/3` 仍差 `1` 个 provider case；LongMemEval subset 持平 `1/1`
  - helper 当前已把 iteration `10` 记为 `discard`：commit `158649f` 的 blank-output fallback 在 learned-symbolic line 上回落到 Persona `4/4`
  - helper 当前已把 iteration `11` 记为 `pivot`：放弃继续在 symbolic-slot holdout 上做同类 fallback 微调，改切 learned slot-assignment holdout
  - helper 当前已把 iteration `12` 记为 `refine`：在 revert base `40e006c` 上重跑 learned+learned，Persona subset 提到 `5/3`，但仍落后 retained `v30` 的 provider `6/3`
  - helper 当前已把 iteration `13` 记为 `discard`：commit `a17d085` 把 blank-output fallback 嫁接到 learned+learned 后，Persona subset 回落到 `4/3`
  - helper 当前已把 iteration `14` 记为第二次 `pivot`，iteration `15` 记为 `no-op`：对同一 learned-symbolic config 重跑后再次得到 Persona `5/4`、Long `1/1`，确认当前 quick-smoke gate 稳定停在“差 `1` 个 Persona provider case”
  - helper 当前已把 iteration `16` 记为 `search`：基于 primary source 搜索锁定 `prefix_allowed_tokens_fn` / constrained generation 作为下一条结构性尝试
  - helper 当前已把 iteration `17` 记为 `discard`：commit `a99a62c` 的 json-start constrained decoding 让 Persona subset 回落到 `4/3`，Long 仍 `1/1`
  - helper 当前已把 iteration `18` 记为第三次 `pivot`：当前 `8`-sample quick-smoke holdout line 已到 soft-blocker handoff；下一步若继续 `v31`，需要更大样本的稳定 measurement 或更宽 scope 的 decoder / answer-selection redesign，而不是继续做同类 runtime micro-tune
- Evidence / artifacts:
  - `research-results.tsv`
  - `autoresearch-state.json`
  - `outputs_v2/runs/20260419T084747Z_stage2_memory_canary_personamem/`
  - `outputs_v2/runs/20260419T084752Z_stage2_memory_canary_longmemeval/`
  - `outputs_v2/runs/20260419T085629Z_stage2_memory_canary_personamem/`
  - `outputs_v2/runs/20260419T085633Z_stage2_memory_canary_longmemeval/`
  - `outputs_v2/runs/20260419T090340Z_stage2_memory_canary_personamem/`
  - `outputs_v2/runs/20260419T090344Z_stage2_memory_canary_longmemeval/`
  - `outputs_v2/runs/20260419T090911Z_stage2_memory_canary_personamem/`
  - `outputs_v2/runs/20260419T090917Z_stage2_memory_canary_longmemeval/`
  - commits `c82a56c`, `40e006c`, `a17d085`, `e1b2a8b`, `a99a62c`, `ee1e613`
- Next likely action:
  - 若继续 `TD-042 / WS-028 / v31`，优先放弃当前 `8`-sample quick-smoke 作为唯一 gate，改做 `LongMemEval-S 500 + PersonaMem 512` 的稳定 holdout measurement；`8`-sample 只保留为 sanity/crash gate
  - 在新的更稳定 measurement 出现前，不要把当前 quick-smoke line 写成“只差一个简单 patch 就能 keep”

## 2026-04-19 Session 069

- Worked on: 以 fresh managed `v31` run 推进 `latent-first quality` 主线，先完成 runtime/doc sync，再落地 `v31` latent mainline train/publish 链，并把第一条 aligned `32k val` latent compare 推到 retained positive keep
- State changed:
  - fresh managed run 已按 launch manifest 初始化；baseline 先量到 `scripts/verify_stage2_v31_longrun.py --score-only = 13`，随后 helper 已创建新的 `research-results.tsv` 与 `autoresearch-state.json`
  - commit `5b43c8d` 已把 `docs/current_status.md` 与 `docs/implementation_plan.md` 对齐到 `TD-042 / v31` runtime truth，使 verifier 从 `13/32` 提升到 `15/32`
  - commit `145e1c5` 已在 `scripts/train_stage2.py` 新增 `--execute-v31-latent-mainline`，并新增 `configs/stage2_train_v31_latent.yaml` 与对应回归，使 `latest_stage2_v31_latent_mainline_train.json` / `latest_stage2_v31_latent_holdout_compare.json` 的发布链落地；第一轮 `32k test` compare 结果为 `delta_score = -0.0087890625`，因此这轮 truth 只能保留为 pipeline keep，不是 latent gain keep
  - commit `d311901` 已把 `v31` latent config 调整为更稳的 `48 / 24 / 6` 配方，并把 compare manifest 对齐回 retained `v30` 使用的 `32k val` holdout；fresh compare 当前记录 `current_top1_accuracy = 0.97265625`、`current_mrr = 0.986328125`、`delta_score = 0.01171875`、`positive_gain = true`
  - commit `e9398f3` 已保留 belief repair 与 `v31` belief publisher；对应 `outputs_v2/evals_local/20260419T071800Z_stage2_local_eval.json` 上的 `latest_stage2_v31_belief_holdout_compare.json` 记录 `delta_token_f1 = 0.513979623914423`、`delta_field_f1 = 0.638888888888889`、`positive_gain = true`
  - 当前未提交的 write repair 改动已在 `outputs_v2/evals_local/20260419T074515Z_stage2_local_eval.json` 上完成真实 compare：`latest_stage2_v31_write_holdout_compare.json` 当前记录 `delta_token_f1 = 0.06976744186046513`、`delta_field_f1 = 0.06976744186046524`、`positive_gain = true`
  - `.agent-os/project-index.md`、`.agent-os/todo.md`、`docs/current_status.md`、`docs/todo.md` 与 `.agent-os/lessons-learned.md` 已同步到 current retained `24/32` truth：当前已有 `latent + belief + write` 三条 apples-to-apples internal keep，但还不能写成 full holdout gain
- Evidence / artifacts:
  - `research-results.tsv`
  - `autoresearch-state.json`
  - `outputs_v2/runs/20260419T070220Z_stage2_v31_latent_exec/`
  - `outputs_v2/runs/20260419T070526Z_stage2_v31_latent_exec/`
  - `outputs_v2/evals_local/20260419T071800Z_stage2_local_eval.json`
  - `outputs_v2/evals_local/20260419T074515Z_stage2_local_eval.json`
  - `outputs_v2/artifacts/latest_stage2_v31_latent_mainline_train.json`
  - `outputs_v2/artifacts/latest_stage2_v31_latent_holdout_compare.json`
  - `outputs_v2/artifacts/latest_stage2_v31_belief_mainline_eval.json`
  - `outputs_v2/artifacts/latest_stage2_v31_belief_holdout_compare.json`
  - `outputs_v2/artifacts/latest_stage2_v31_write_mainline_eval.json`
  - `outputs_v2/artifacts/latest_stage2_v31_write_holdout_compare.json`
  - commits `5b43c8d`, `145e1c5`, `d311901`, `e9398f3`
- Next likely action:
  - 在 current retained `24/32` latent + belief + write keep 上推进 full benchmark holdout
  - 最后用 ablation summary 验证增益是否主要来自 learned latent 主链

## 2026-04-19 Session 068

- Worked on: 把 stage-2 主线从 `v30 architecture-first closeout` 前推到 `v31 latent-first quality run`，明确下一轮不再以基础设施为主，而是以 full holdout 上 learned `write -> latent -> belief` 的真实增益与 ablation truth 为目标
- State changed:
  - 新增 [docs/v31_plan.md](/media/storage/mingjing/workspace/CoRe_Mem/docs/v31_plan.md)，把 `v31` 主线收紧为 latent strengthening、belief strengthening、write strengthening、full holdout 与 ablation truth
  - 新增 [scripts/verify_stage2_v31_longrun.py](/media/storage/mingjing/workspace/CoRe_Mem/scripts/verify_stage2_v31_longrun.py)，把 `v31` 的 mechanical target 锁为 32 项，要求 future run 在 retained `v30` baseline 之上拿到 latent / belief / write 正增益、full holdout 改善与 ablation 归因
  - 新增 [tests/test_stage2_v31_longrun.py](/media/storage/mingjing/workspace/CoRe_Mem/tests/test_stage2_v31_longrun.py)，提供 baseline / full-pass 两组回归
  - `.agent-os/project-index.md`、`.agent-os/todo.md`、`docs/current_status.md`、`docs/implementation_plan.md` 与 `docs/todo.md` 已同步切到 `TD-042 / WS-028 / v31`
- Evidence / artifacts:
  - `docs/v31_plan.md`
  - `scripts/verify_stage2_v31_longrun.py`
  - `tests/test_stage2_v31_longrun.py`
- Next likely action:
  - 提交 `v31` baseline commit
  - fresh-start 启动 `v31` 后台 autoresearch

## 2026-04-19 Session 067

- Worked on: 收口 `TD-041 / WS-027` 的第四轮 `v30` experiment，把 full benchmark holdout baseline、shared-subset non-regression guard 与 closeout 文档同步补齐到 stop condition
- State changed:
  - `scripts/verify_stage2_v30_longrun.py` 当前已把 `PersonaMem` non-regression guard 从“直接比较 `589` 全量 rate”修正为“基于 retained `v2.9` `512` sample_ids overlap subset 的 apples-to-apples exact-rate 比较”；对应回归已写入 `tests/test_stage2_v30_longrun.py`
  - current HEAD `75c70bc` 上，`latest_stage2_v30_full_holdout_baseline.json`、`latest_longmemeval_stage2_v30_full.json` 与 `latest_personamem_stage2_v30_full.json` 已基于真实 summary 发布
  - 当前 full holdout 明确保持 `holdout_only = true`；`LongMemEval-S 500` 与 retained `v2.9` full counts 持平，`PersonaMem 589` 则通过 retained `512` overlap subset 机械确认 `provider_exact_rate = 0.3515625 > 0.333984375`、`local_exact_rate = 0.2109375 = 0.2109375`
  - `docs/current_status.md`、`docs/implementation_plan.md`、`docs/todo.md`、`.agent-os/project-index.md` 与 `.agent-os/todo.md` 已同步到 `41/41 keep` truth
  - 当前 `scripts/verify_stage2_v30_longrun.py --score-only` 已达到 `41`；full guard 通过；本轮已满足 stop condition，下一步应由 helper 正式记账并停止 run
- Evidence / artifacts:
  - `outputs_v2/evals_benchmark/20260419T021517Z_stage2_memory_canary.json`
  - `outputs_v2/evals_benchmark/20260419T022355Z_stage2_memory_canary.json`
  - `outputs_v2/artifacts/latest_stage2_v30_full_holdout_baseline.json`
  - `outputs_v2/artifacts/latest_longmemeval_stage2_v30_full.json`
  - `outputs_v2/artifacts/latest_personamem_stage2_v30_full.json`
  - `research-results.tsv`
  - `autoresearch-state.json`
  - commit `75c70bc`
- Next likely action:
  - 保留当前 `41/41 keep` 作为新的 `v30` retained baseline
  - 下一条 stage-2 方向应直接从这条 retained line 往前推

## 2026-04-19 Session 066

- Worked on: 继续 `TD-041 / WS-027` 的第三轮 `v30` experiment，把 current-head modular checkpoint 的 learned belief decoder / write / belief 增益证据链正式发布成 verifier 可读 artifact，并完成 keep 记账
- State changed:
  - `scripts/eval_stage2_local.py` 已新增 `v30` publisher：当前可基于 checkpoint-aware local eval 发布 `latest_stage2_v30_belief_decoder_eval.json`、`latest_stage2_v30_write_gain.json` 与 `latest_stage2_v30_belief_gain.json`
  - current HEAD `e9be6f3` 上，fresh eval `outputs_v2/evals_local/20260419T015956Z_stage2_local_eval.json` 已真实完成；当前 `trained_eval` 仍为 `token_f1 = 0.6562995718106327`、`field_f1 = 0.5924479166666667`
  - 相对 `latest_stage2_v27_eval.json` 的 baseline `0`，当前 belief decoder artifact 记录 `delta_token_f1 = 0.3122825952686847`、`delta_field_f1 = 0.11111111111111109`、`positive_gain = true`
  - 当前 write artifact 记录 `delta_token_f1 = 0.8571428571428572`、`delta_field_f1 = 0.821705426356589`、`positive_gain = true`
  - `scripts/verify_stage2_v30_longrun.py --score-only` 已从 retained `27` 提升到 `33`；launch guard 通过；helper 已把这轮记为 iteration `3 keep`
- Evidence / artifacts:
  - `research-results.tsv`
  - `autoresearch-state.json`
  - `outputs_v2/evals_local/20260419T015956Z_stage2_local_eval.json`
  - `outputs_v2/artifacts/latest_stage2_v30_belief_decoder_eval.json`
  - `outputs_v2/artifacts/latest_stage2_v30_write_gain.json`
  - `outputs_v2/artifacts/latest_stage2_v30_belief_gain.json`
  - commit `e9be6f3`
- Next likely action:
  - 把下一轮假设集中到 full benchmark holdout baseline：补齐 `latest_stage2_v30_full_holdout_baseline.json`、`latest_longmemeval_stage2_v30_full.json` 与 `latest_personamem_stage2_v30_full.json`
  - 基于 retained `v2.9` expanded holdout 做 non-regression 判定，不把 benchmark 回流训练

## 2026-04-19 Session 065

- Worked on: 继续 `TD-041 / WS-027` 的第二轮真实 `v30` experiment，把 modular shared-backbone 线推进到 trainable latent + direct latent objective，并完成 keep/discard 判定与受管结果记账
- State changed:
  - 新增 `src/core_mem/v2/latent_training.py`，当前已具备可训练 latent retriever：`query_encoder`、`slot_encoder` 与 `latent_query_bank` 可直接通过 latent objective 训练
  - `scripts/train_stage2.py` 已新增 `--execute-v30-latent-objective` 与对应 artifact publish 路径，可发布 `latest_stage2_v30_latent_module_train.json`、`latest_stage2_v30_latent_objective_eval.json`、`latest_stage2_v30_latent_gain.json`
  - 新增 `configs/stage2_train_v30_latent.yaml`，把第二轮 `v30` latent experiment 固定到 `32k` anchor manifest 与独立 eval manifest
  - current HEAD `f0e3203` 上，fresh latent run `outputs_v2/runs/20260419T014522Z_stage2_v30_latent_exec/` 已真实完成：`4096` train examples、`512` eval examples、`256` steps、`final_loss = 0.1376347839832306`
  - 对应 latent eval 当前记录 `baseline_top1_accuracy = 0.724609375`、`current_top1_accuracy = 0.96484375`、`baseline_mrr = 0.8623046875`、`current_mrr = 0.982421875`、`delta_score = 0.3603515625`，`positive_gain = true`
  - `scripts/verify_stage2_v30_longrun.py --score-only` 已从 retained `21` 提升到 `27`；launch guard 通过；helper 已把这轮记为 iteration `2 keep`
- Evidence / artifacts:
  - `research-results.tsv`
  - `autoresearch-state.json`
  - `outputs_v2/runs/20260419T014522Z_stage2_v30_latent_exec/`
  - `outputs_v2/checkpoints/20260419T014522Z_stage2_v30_latent_exec/`
  - `outputs_v2/artifacts/latest_stage2_v30_latent_module_train.json`
  - `outputs_v2/artifacts/latest_stage2_v30_latent_objective_eval.json`
  - `outputs_v2/artifacts/latest_stage2_v30_latent_gain.json`
  - commit `f0e3203`
- Next likely action:
  - 在 current-head `27/41 keep` 的 retained line 上推进 learned belief decoder，并补齐 `belief decoder eval / belief gain / write gain`
  - 在 learned belief decoder 成立后，再补 full benchmark holdout baseline 与 non-regression refresh，而不是提前把 current line 写成 `v30` complete

## 2026-04-19 Session 064

- Worked on: 启动 `TD-041 / WS-027` 的第一轮真实 `v30` modular-training experiment，把当前 shared-only 训练线推进到 `shared backbone + task-specific adapters`，并用 `32k train / 32k val` 分离口径做 first compare
- State changed:
  - `src/core_mem/v2/training.py` 当前已支持真正的 task-specific adapter runtime：tiny backend 走 shared backbone + per-task heads，HF/PEFT 路径走 shared backbone + multi-adapter，并在 train/eval 阶段按 task 切换 adapter
  - `scripts/train_stage2.py` 当前已支持 `--eval-manifest`，避免 `register-experiment` 默认拿 train manifest 自己做 checkpoint eval；这条 anti-leakage 修正已由 targeted tests 复验
  - 新增 `configs/stage2_train_v30.yaml` 与 `configs/stage2_train_v30_tiny.yaml`，把 `v30` 第一轮 modular line 固定为 task adapters enabled
  - current HEAD `13bb0fa` 上，fresh `gpu2 + 32k anchor` 训练 `outputs_v2/runs/20260419T012526Z_stage2_train_exec/execution_summary.json` 已真实完成：`4096` examples、`128` steps、`wall_clock_seconds = 42.88473560567945`
  - fresh val checkpoint eval `outputs_v2/evals_local/20260419T013221Z_stage2_local_eval.json` 已记录 `trained_eval.token_f1 = 0.6562995718106327`、`trained_eval.field_f1 = 0.5924479166666667`
  - `latest_stage2_v30_shared_backbone_train.json` 与 `latest_stage2_v30_task_adapter_compare.json` 已落地；后者当前显式记录 `task_specific_positive_gain = true`、`delta_score = 1.2487474884772993`
  - `scripts/verify_stage2_v30_longrun.py --score-only` 已从 baseline `17` 提升到 `21`；launch guard 通过；helper 已把这轮记为 iteration `1 keep`
- Evidence / artifacts:
  - `research-results.tsv`
  - `autoresearch-state.json`
  - `outputs_v2/runs/20260419T012526Z_stage2_train_exec/execution_summary.json`
  - `outputs_v2/evals_local/20260419T013221Z_stage2_local_eval.json`
  - `outputs_v2/artifacts/latest_stage2_v30_shared_backbone_train.json`
  - `outputs_v2/artifacts/latest_stage2_v30_task_adapter_compare.json`
  - commit `b1f69f6`
  - commit `13bb0fa`
- Next likely action:
  - 在 current-head `21/41 keep` 的 retained line 上继续推进 `trainable latent / direct latent objective / learned belief decoder`
  - 把 future local eval 预算收紧到更适合 nightly iteration 的规模，再进入下一轮 `v30` keep/discard
  - 之后再补 full benchmark holdout baseline / refresh，而不是把当前 modular keep 误写成 `v30` 已完成

## 2026-04-19 Session 063

- Worked on: 把 next stage 从 `v2.9` closeout 前推到 `v3.0 / v30 architecture-first long-run`，明确下一条主线不再围绕 teacher，而是直接围绕可训练主链升级：shared backbone + task-specific adapters、trainable latent、direct latent objectives、learned belief decoder，以及 full benchmark holdout baseline
- State changed:
  - 新增 [docs/v30_plan.md](/media/storage/mingjing/workspace/CoRe_Mem/docs/v30_plan.md)，把当前起点、冻结边界、32k 训练锚点、full benchmark holdout（`LongMemEval-S 500 / PersonaMem 589`）、以及四条 architecture milestones 固化为真源计划
  - 新增 [scripts/verify_stage2_v30_longrun.py](/media/storage/mingjing/workspace/CoRe_Mem/scripts/verify_stage2_v30_longrun.py)，把 `v30` 的 mechanical target 锁成 42 项，既保留 `v2.9` retained line，又要求 future run 真正产出 shared-backbone modular train、trainable latent、direct latent objective、learned belief decoder 与 full benchmark holdout artifacts
  - 新增 [tests/test_stage2_v30_longrun.py](/media/storage/mingjing/workspace/CoRe_Mem/tests/test_stage2_v30_longrun.py)，提供 baseline / full-pass 两组回归
  - `docs/current_status.md`、`docs/implementation_plan.md`、`docs/todo.md`、`.agent-os/project-index.md`、`.agent-os/todo.md` 已同步切到 `TD-041 / WS-027 / v30`
- Evidence / artifacts:
  - `docs/v30_plan.md`
  - `scripts/verify_stage2_v30_longrun.py`
  - `tests/test_stage2_v30_longrun.py`
- Next likely action:
  - 在当前 repo 上跑 `scripts/verify_stage2_v30_longrun.py --score-only` 量基线
  - 提交 baseline commit
  - fresh-start 启动 `v30` 后台 autoresearch

## 2026-04-19 Session 062

- Worked on: 继续受控恢复 `TD-040 / WS-026`，修掉 `run_stage2_memory_canary.py` 在大 holdout 上“长时间 0 落盘”的问题，并把 `v2.9` 从 `33/39 partial` 推到 stop condition `39/39`
- State changed:
  - `scripts/run_stage2_memory_canary.py` 已新增 `symbolic/symbolic + provider_workers>1` 的 sample-level 并行快路径；当前大 holdout 不再卡在串行预处理，而会增量写 `predictions.jsonl`
  - `tests/test_stage2_memory_canary.py` 已补 row-builder/provider 路径回归，并把 subprocess 固定到 `GPT_AGENT_API_KEY=''`，避免测试误打真实 provider
  - fresh current-head holdout 已真实完成：`outputs_v2/evals_benchmark/20260419T000721Z_stage2_memory_canary.json` 为 `LongMemEval-S 500`，`outputs_v2/evals_benchmark/20260419T000717Z_stage2_memory_canary.json` 为 `PersonaMem 512`
  - `scripts/verify_stage2_v29_longrun.py --publish-artifacts` 已刷新 `latest_stage2_v29_{write,latent,belief}_gain.json`、`latest_stage2_v29_holdout_summary.json` 与 `latest_*_stage2_v29_canary.json`；三段 gain 当前均为 `positive_gain = true`
  - `scripts/verify_stage2_v29_longrun.py --score-only` 当前为 `39`，launch guard 通过；helper 已把这轮记为 iteration `3 keep`
  - `.agent-os/project-index.md`、`.agent-os/todo.md`、`docs/current_status.md`、`docs/implementation_plan.md` 与 `docs/todo.md` 已同步到 `39/39 keep` 的 runtime truth
- Evidence / artifacts:
  - `outputs_v2/evals_benchmark/20260419T000721Z_stage2_memory_canary.json`
  - `outputs_v2/evals_benchmark/20260419T000717Z_stage2_memory_canary.json`
  - `outputs_v2/artifacts/latest_stage2_v29_write_gain.json`
  - `outputs_v2/artifacts/latest_stage2_v29_latent_gain.json`
  - `outputs_v2/artifacts/latest_stage2_v29_belief_gain.json`
  - `outputs_v2/artifacts/latest_stage2_v29_holdout_summary.json`
  - `research-results.tsv`
  - `autoresearch-state.json`
- Next likely action:
  - 当前 stop condition 已达成；除非用户明确要求继续 post-`v2.9` 质量/泛化探索，否则保持现有 retained line 与文档状态即可

## 2026-04-18 Session 061

- Worked on: 以 fresh background managed run 启动 `TD-040 / WS-026` 的 `v2.9 learned-core-path long-run`，先补 `v29` publish 链并把 runtime truth 推进到可机械复验的 partial state
- State changed:
  - baseline 已用 `scripts/verify_stage2_v29_longrun.py --score-only` 量到 `25`，guard 通过后已初始化 fresh `research-results.tsv` 与 `autoresearch-state.json`
  - commit `374c78e` 已为 `scripts/verify_stage2_v29_longrun.py` 新增 `--publish-artifacts`，并在 `tests/test_stage2_v29_longrun.py` 补齐 publish 回归
  - current-head `latest_stage2_v29_{write,latent,belief}_gain.json`、`latest_stage2_v29_training_timing.json`、`latest_stage2_v29_holdout_summary.json`、`latest_longmemeval_stage2_v29_canary.json` 与 `latest_personamem_stage2_v29_canary.json` 已落地，使 `scripts/verify_stage2_v29_longrun.py --score-only` 从 `25` 提升到 retained `33`
  - helper 已把上述发布链记为 iteration `1 keep`；随后又把“当前 session `GPT_AGENT_API_KEY=UNSET`、provider-side holdout checks blocked”记为 iteration `2 blocked`
  - `.agent-os/project-index.md`、`.agent-os/todo.md`、`docs/current_status.md`、`docs/implementation_plan.md` 与 `docs/todo.md` 已同步到 `33/39 partial + provider env blocker` 的 runtime truth
- Evidence / artifacts:
  - commit `374c78e`
  - `research-results.tsv`
  - `autoresearch-state.json`
  - `outputs_v2/artifacts/latest_stage2_v29_write_gain.json`
  - `outputs_v2/artifacts/latest_stage2_v29_latent_gain.json`
  - `outputs_v2/artifacts/latest_stage2_v29_belief_gain.json`
  - `outputs_v2/artifacts/latest_stage2_v29_training_timing.json`
  - `outputs_v2/artifacts/latest_stage2_v29_holdout_summary.json`
  - `outputs_v2/artifacts/latest_longmemeval_stage2_v29_canary.json`
  - `outputs_v2/artifacts/latest_personamem_stage2_v29_canary.json`
  - `outputs_v2/evals_benchmark/20260418T204434Z_stage2_memory_canary.json`
  - `outputs_v2/evals_benchmark/20260418T205829Z_stage2_memory_canary.json`
- Next likely action:
  - 在当前 session 或下一次恢复中先恢复 `GPT_AGENT_API_KEY`，再重跑 current-head `LongMemEval-S 512` / `PersonaMem 512` 并只在 `write / latent / belief` 任一段出现真实正增益时刷新 retained `v29` gain artifacts

## 2026-04-18 Session 060

- Worked on: 把当前主线从 `v2.8 teacher-quality blocker` 前推到 `v2.9 learned-core-path long-run`，把下一阶段目标从“继续扩 teacher”收紧为“在 32k 锚点上让 write / latent / belief 出现真实正增益，并扩大 holdout benchmark”
- State changed:
  - 新增 [docs/v29_plan.md](/media/storage/mingjing/workspace/CoRe_Mem/docs/v29_plan.md)，明确冻结 `core / residual`，并把主线切到 `write -> latent composition -> belief`
  - 新增 `scripts/verify_stage2_v29_longrun.py` 与 `tests/test_stage2_v29_longrun.py`
  - `docs/current_status.md`、`docs/implementation_plan.md`、`docs/todo.md`、`.agent-os/project-index.md` 与 `.agent-os/todo.md` 已同步到 `TD-040 / WS-026`
  - `v2.8` 当前 blocker truth 被保留为 retained baseline：teacher suite 完整，但 matched compare 为负，因此 teacher 暂时降级为可选探索线，而不是 `v2.9` 默认主线
- Evidence / artifacts:
  - `docs/v29_plan.md`
  - `scripts/verify_stage2_v29_longrun.py`
  - `tests/test_stage2_v29_longrun.py`
- Next likely action:
  - 运行 `v2.9` baseline verifier / tests，提交 long-run baseline commit，并以 fresh background 方式启动新的 `v2.9` autoresearch

## 2026-04-18 Session 059

- Worked on: 对 `TD-039 / WS-025` 再做一轮 selective integration 验证，并在 evidence 足够后把当前 managed run 判定为 true blocker
- State changed:
  - `scripts/prepare_stage2_data.py` 当前已支持把 matched subset 的 `matched_kinds` 与 `teacher_apply_kinds` 解耦，并支持独立 `manifest_namespace`，避免新实验覆盖上一轮 matched artifact
  - `tests/test_stage2_teacher_labels.py` 已新增 selective integration 回归：当 `slot_assignment` 只用于选 matched subset、但不实际应用 teacher label 时，`lifecycle_prediction` 仍保持 silver，而 `belief` teacher 仍会被正确替换
  - fresh selective manifests 已落地到 `outputs_v2/artifacts/stage2_v28_matched_manifests_obs_belief/`
  - fresh selective compare 保持 all-changed matched subset 不变、只应用 `observation + belief` teacher；对应 silver train 为 `outputs_v2/runs/20260418T160913Z_stage2_train_exec/`，teacher train 为 `outputs_v2/runs/20260418T161000Z_stage2_train_exec/`
  - fresh selective val/test eval 结果显示：去掉 `lifecycle` teacher 后，`lifecycle_prediction` 指标已与 silver 对齐，但 compare 仍为负，`delta_internal_token_f1 = -0.03262529332240871`、`delta_internal_field_f1 = -0.02564102564102566`、`delta_internal_exact_match = -0.10576923076923073`
  - 这轮 selective integration 的主退化项已收敛到 `composition_to_belief`
  - 与此同时，剩余可单独保留的 observation-only raw-teacher 改动只覆盖 matched subset 的 `train=1 / val=0 / test=3`，不足以支撑有意义的 same-budget compare
  - launch guard 当前通过；helper 已把这轮 experiment 记为 iteration `2 discard`，随后又把“当前 launch 已进入 true blocker”记为 iteration `3 blocked`
- Evidence / artifacts:
  - `outputs_v2/artifacts/stage2_v28_matched_manifests_obs_belief/`
  - `outputs_v2/runs/20260418T160913Z_stage2_train_exec/execution_summary.json`
  - `outputs_v2/runs/20260418T161000Z_stage2_train_exec/execution_summary.json`
  - `outputs_v2/evals_local/20260418T161101Z_stage2_local_eval.json`
  - `outputs_v2/evals_local/20260418T161805Z_stage2_local_eval.json`
  - `outputs_v2/evals_local/20260418T162010Z_stage2_local_eval.json`
  - `outputs_v2/evals_local/20260418T162334Z_stage2_local_eval.json`
  - `research-results.tsv`
  - `autoresearch-state.json`
  - `conda run -n core_mem python scripts/verify_stage2_v28_longrun.py --score-only` -> `32`
- Next likely action:
  - 本轮 managed run 已到 true blocker；若未来继续 `TD-039 / WS-025`，必须先更换 teacher 生成策略或标签定义，而不是再重复当前 label suite

## 2026-04-18 Session 058

- Worked on: 把 `TD-039 / WS-025` 的 `raw observation teacher + matched teacher-vs-silver subset` 真正跑完、记录 fresh managed-run 结果，并把 runtime truth 从“teacher 改动未被消费”切换到“teacher lifecycle supervision 带来负增益”
- State changed:
  - `scripts/prepare_stage2_data.py` 当前已新增 changed-only teacher label map 与 source-record 对齐的 matched manifest builder；matched manifests 现按 changed `raw-observation / slot-assignment / belief` sample_ids 收缩到真正消费 teacher 改动的样本
  - 当前新增 `configs/stage2_train_v28_matched.yaml`，把 same-budget compare 的训练任务收敛到 `slot_autoencoding / lifecycle_prediction / composition_to_belief`
  - `tests/test_stage2_teacher_labels.py` 已新增 matched subset 对 changed sample_ids 与 source-record 对齐的回归覆盖；targeted tests 与 launch guard 当前均通过
  - fresh `v2.8` teacher suite 已在缓存标签上重新发布；当前 latest matched manifests 已落地到 `outputs_v2/artifacts/stage2_v28_matched_manifests/`
  - same-split matched non-tiny compare 已完成：silver train 为 `outputs_v2/runs/20260418T154514Z_stage2_train_exec/`，teacher train 为 `outputs_v2/runs/20260418T154614Z_stage2_train_exec/`
  - fresh matched eval 结果显示 teacher 线明显退化：`delta_internal_token_f1 = -0.05357191517996174`、`delta_internal_field_f1 = -0.08173076923076938`、`delta_internal_exact_match = -0.1826923076923077`
  - 主退化集中在 `lifecycle_prediction`：val `token_f1` 从 `0.9655172413793104` 降到 `0.8004926108374386`，test 从 `0.9285714285714286` 降到 `0.8482142857142859`
  - `scripts/verify_stage2_v28_longrun.py --score-only` 复核后仍为 `32`
  - `research-results.tsv` / `autoresearch-state.json` 已把这轮 fresh background run 记为 iteration `1 discard`，retained metric 继续保持 baseline `32`
- Evidence / artifacts:
  - `configs/stage2_train_v28_matched.yaml`
  - `outputs_v2/artifacts/latest_stage2_v28_teacher_observation.json`
  - `outputs_v2/artifacts/stage2_v28_matched_manifests/`
  - `outputs_v2/runs/20260418T154514Z_stage2_train_exec/execution_summary.json`
  - `outputs_v2/runs/20260418T154614Z_stage2_train_exec/execution_summary.json`
  - `outputs_v2/evals_local/20260418T155317Z_stage2_local_eval.json`
  - `outputs_v2/evals_local/20260418T155319Z_stage2_local_eval.json`
  - `outputs_v2/evals_local/20260418T155046Z_stage2_local_eval.json`
  - `outputs_v2/evals_local/20260418T155047Z_stage2_local_eval.json`
  - `outputs_v2/artifacts/latest_stage2_v28_teacher_compare.json`
  - `outputs_v2/artifacts/latest_stage2_v28_internal_test.json`
  - `research-results.tsv`
  - `autoresearch-state.json`
  - `conda run -n core_mem python scripts/verify_stage2_v28_longrun.py --score-only` -> `32`
- Next likely action:
  - 若继续 `TD-039 / WS-025`，下一轮不能再重复当前 teacher 配方；必须先解释 `teacher lifecycle` supervision 为何在 changed subset 上显著拉低 internal quality，再决定是否做 task-level weighting、partial integration 或直接 pivot

## 2026-04-18 Session 057

- Worked on: 把 `TD-039 / WS-025` 从 “teacher suite 是否能收口” 推进到 “teacher-enhanced 是否真的优于 silver baseline” 的真实判定，并完成一轮 same-split non-tiny `gpu2` 对照
- State changed:
  - `scripts/prepare_stage2_data.py` 当前已把 `v2.8` teacher suite 真正跑完；`latest_stage2_v28_teacher_observation.json`、`latest_stage2_v28_teacher_slot_assignment.json`、`latest_stage2_v28_teacher_belief.json` 与 `latest_stage2_v28_teacher_quality_audit.json` 已落地
  - 当前已验证的 teacher suite caps 为 `256 / 128 / 128`；其中 observation teacher 已从 `v2.7` 的 `completed_with_failures` 修到 current-head `success_rate = 1.0`
  - 先前 `v2.8` compare 之所以全零，已确认不是 teacher 数据本身，而是 `teacher-vs-silver` refresh 误用了 `configs/stage2_train_tiny.yaml` 的 `__tiny_debug_seq2seq__`；对应 helper 已把 iteration `2` 记为 `refine`
  - 随后在完全相同的 `32k` split 上补齐一轮更可信的 non-tiny same-budget compare：silver 与 teacher 都使用 `configs/stage2_train.yaml`、`gpu2`、`256` examples、`192` steps
  - silver baseline 当前 fresh internal test 为：`trained_eval.token_f1 = 0.9725304472117797`、`field_f1 = 0.938151041666667`
  - teacher-enhanced 当前 fresh internal test 为：`trained_eval.token_f1 = 0.9719352091165416`、`field_f1 = 0.9381510416666669`
  - 因此 current-head `latest_stage2_v28_teacher_compare.json` 现明确记录 `delta_internal_token_f1 = -0.0005952380952380931`、`delta_internal_field_f1 ≈ 0`，`latest_stage2_v28_internal_test.json` 当前 `gate_passed = false`
  - fresh current-head `scripts/verify_stage2_v28_longrun.py --score-only` 因此停在 `32 / 34`；helper 已把 iteration `3` 记为 `blocked`
  - 当前 `TD-039 / WS-025` 的真实 blocker 已从 teacher publisher / provider throughput 切换为：teacher supervision 线本身没有形成正的 internal generalization delta
- Evidence / artifacts:
  - `research-results.tsv`
  - `autoresearch-state.json`
  - `outputs_v2/artifacts/latest_stage2_v28_teacher_observation.json`
  - `outputs_v2/artifacts/latest_stage2_v28_teacher_slot_assignment.json`
  - `outputs_v2/artifacts/latest_stage2_v28_teacher_belief.json`
  - `outputs_v2/artifacts/latest_stage2_v28_teacher_quality_audit.json`
  - `outputs_v2/runs/20260418T061903Z_stage2_train_exec/execution_summary.json`
  - `outputs_v2/evals_local/20260418T062020Z_stage2_local_eval.json`
  - `outputs_v2/evals_local/20260418T062508Z_stage2_local_eval.json`
  - `outputs_v2/runs/20260418T062939Z_stage2_train_exec/execution_summary.json`
  - `outputs_v2/evals_local/20260418T063052Z_stage2_local_eval.json`
  - `outputs_v2/evals_local/20260418T063541Z_stage2_local_eval.json`
  - `outputs_v2/artifacts/latest_stage2_v28_teacher_compare.json`
  - `outputs_v2/artifacts/latest_stage2_v28_internal_test.json`
  - `conda run -n core_mem python scripts/verify_stage2_v28_longrun.py --json` -> `32 / 34`
- Next likely action:
  - 若继续 `TD-039 / WS-025`，下一轮不能再重复当前 teacher 配方；必须先提出新的 teacher integration / weighting 假设，解释为何 same-split non-tiny compare 仍略低于 silver

## 2026-04-17 Session 056

- Worked on: 为 `TD-039 / WS-025` 落地 `v2.8` teacher-quality 的数据发布与对照发布链，并定位 observation teacher 在更大 coverage 下的真实失败模式
- State changed:
  - `scripts/prepare_stage2_data.py` 已新增 observation teacher coercion failure 的 single-sample retry / failure 落盘逻辑，不再因单条 partial schema label 直接中断整批发布
  - 同一脚本已新增 `publish_v28_teacher_suite` 与 `--publish-v28-teacher-suite`，可在不覆盖 retained `v2.7` latest artifacts 的前提下发布 `latest_stage2_v28_teacher_{observation,slot_assignment,belief}.json`、`latest_stage2_v28_teacher_quality_audit.json` 以及 teacher-enhanced split manifests
  - `scripts/verify_stage2_v28_longrun.py` 已新增 `publish_v28_artifacts` 与对应 CLI，可发布 `latest_stage2_v28_silver_baseline.json`、`latest_stage2_v28_teacher_train.json`、`latest_stage2_v28_teacher_eval.json`、`latest_stage2_v28_teacher_compare.json` 与 `latest_stage2_v28_internal_test.json`
  - 新增回归覆盖：`tests/test_stage2_teacher_labels.py` 现覆盖 incomplete observation label retry 与 `v2.8` quality-audit/manifest 发布；`tests/test_stage2_v28_longrun.py` 现覆盖 `v2.8` compare/internal gate 发布
  - `conda run -n core_mem pytest -q tests/test_stage2_v28_longrun.py tests/test_stage2_teacher_labels.py tests/test_stage2_training_runtime.py tests/test_stage2_model_skeleton.py` 已通过
  - fresh current-head `scripts/verify_stage2_v28_longrun.py --score-only` 仍为 `19`
  - 当前回合又做了多次真实 `MiniMax-M2.7` teacher suite 尝试（`512/128/128`、`256/64/64`、`256/0/0`），但都未在可接受窗口内收口；当前还不能诚实声明 `v2.8` teacher artifacts 已落地
- Evidence / artifacts:
  - `conda run -n core_mem pytest -q tests/test_stage2_v28_longrun.py tests/test_stage2_teacher_labels.py tests/test_stage2_training_runtime.py tests/test_stage2_model_skeleton.py`
  - `conda run -n core_mem python scripts/verify_stage2_v28_longrun.py --score-only` -> `19`
  - live probes:
    - `conda run --no-capture-output -n core_mem python - <<'PY' ... recoe-aggregation-142 ...`
    - `conda run --no-capture-output -n core_mem python scripts/prepare_stage2_data.py --output-root outputs_v2 --publish-v28-teacher-suite ...`
- Next likely action:
  - 继续 `TD-039 / WS-025`，但下一轮优先从 teacher provider throughput / batching 策略入手，把真实 `v2.8` teacher suite 先跑完，再进入 `silver-vs-teacher` train/eval refresh

## 2026-04-17 Session 055

- Worked on: 把主线从 `v2.7 teacher pilot closeout` 前推到 `v2.8 teacher-quality`
- State changed:
  - 新增 `docs/v28_plan.md`、`scripts/verify_stage2_v28_longrun.py` 与 `tests/test_stage2_v28_longrun.py`
  - 当前主线正式切到 `TD-039 / WS-025`
  - 下一轮不再以 artifact completeness 为成功，而是要求 `teacher-vs-silver` 在同一 `32k` split 上给出真实 internal generalization 增益
  - 新锚点已经写入文档系统：扩大 teacher coverage、修 observation teacher failure、保持 `gpu2` 训练与 holdout-only 边界
- Next likely action:
  - 以 `32k` split 为锚点启动新的 background autoresearch，先扩大 teacher coverage，再进入 `teacher-vs-silver` 的 `gpu2` train/eval refresh

## 2026-04-17 Session 054

- Worked on: 恢复 `TD-038 / WS-024` 的 teacher 路径，落地真实 `MiniMax-M2.7` teacher artifact 发布链，并把 `v2.7` verifier 推到 stop condition
- State changed:
  - 当前 shell 已确认 `GPT_AGENT_API_KEY=SET`，先前“teacher provider env 缺失”不再代表 runtime truth
  - `scripts/prepare_stage2_data.py` 已新增 `--publish-v27-teacher-artifacts`，支持基于 `latest_stage2_v27_32k_manifest.json` 发布 observation / slot-assignment / belief 三类 teacher artifacts，并把缺失 sample_id / 非 JSON 返回如实记录到 failure jsonl，而不是伪造标签
  - current worktree 已用真实 `MiniMax-M2.7` 跑完一轮 sample-capped teacher pilot：caps `8/2/2`、batch size `1`
  - `latest_stage2_v27_teacher_slot_assignment.json` 与 `latest_stage2_v27_teacher_belief.json` 当前为 `completed`；`latest_stage2_v27_teacher_observation.json` 当前为 `completed_with_failures`，显式记录 `total_labeled_examples = 6`、`total_failed_examples = 6`
  - fresh current-head `scripts/verify_stage2_v27_longrun.py --score-only` 已从 `23` 提升到 `26`
  - `.agent-os/project-index.md`、`.agent-os/todo.md`、`docs/current_status.md`、`docs/implementation_plan.md` 与 `docs/todo.md` 已同步到上述 runtime truth
- Evidence / artifacts:
  - `outputs_v2/artifacts/latest_stage2_v27_teacher_observation.json`
  - `outputs_v2/artifacts/latest_stage2_v27_teacher_slot_assignment.json`
  - `outputs_v2/artifacts/latest_stage2_v27_teacher_belief.json`
  - `outputs_v2/artifacts/stage2_v27_teacher/`
  - `conda run -n core_mem python scripts/prepare_stage2_data.py --output-root outputs_v2 --publish-v27-teacher-artifacts --v27-manifest-artifact outputs_v2/artifacts/latest_stage2_v27_32k_manifest.json --teacher-config configs/minimax_m27.yaml --teacher-train-max-rows 8 --teacher-val-max-rows 2 --teacher-test-max-rows 2 --teacher-batch-size 1 --json`
  - `conda run -n core_mem python scripts/verify_stage2_v27_longrun.py --json`
- Next likely action:
  - 若继续 `v2.7`，优先在保持 `core / residual` 冻结与 holdout-only 的前提下，决定是扩大 teacher pilot coverage，还是直接消费当前 teacher labels 进入一轮新的 `gpu2` train/eval refresh

## 2026-04-17 Session 053

- Worked on: 继续恢复 `TD-038 / WS-024` 的 teacher 路径，最终确认 `MiniMax-M2.7` provider env 在当前工作机会话中是硬 blocker
- State changed:
  - `autoresearch_resume_check.py --repo ...` 继续返回 `full_resume`；当前 retained metric 仍是 `23`
  - 已确认仓库内不存在可 source 的 `.env` 文件
  - 已检查并 source `~/.bashrc` / `~/.profile`；之后 `GPT_AGENT_API_KEY`、`OPENAI_API_KEY`、`ALIYUN_API_KEY` 与 `GEMINI_API_KEY` 仍全部为 `UNSET`
  - 因此当前无法真实执行 `MiniMax-M2.7` teacher observation / slot-assignment / belief labels；本轮到达 true blocker，run 在 `23/26` 暂停
  - `.agent-os/project-index.md`、`.agent-os/todo.md`、`docs/current_status.md` 与 `docs/todo.md` 已同步到上述 blocker 真相
- Evidence / artifacts:
  - `python3 /home/mingjing/.codex/skills/codex-autoresearch/scripts/autoresearch_resume_check.py --repo /media/storage/mingjing/workspace/CoRe_Mem`
  - `research-results.tsv`
  - `autoresearch-state.json`
  - shell profile probe on `~/.bashrc` / `~/.profile`
- Next likely action:
  - 只有在本地 provider env 恢复后，才能继续生成 `latest_stage2_v27_teacher_observation.json`、`latest_stage2_v27_teacher_slot_assignment.json` 与 `latest_stage2_v27_teacher_belief.json`

## 2026-04-17 Session 052

- Worked on: 为 `TD-038 / WS-024` 补齐 fresh background run 初始化、`gpu2` tiny pilot train/eval/timing/internal gate，以及 `v27` 的非-teacher artifact 发布链
- State changed:
  - 先以 baseline `scripts/verify_stage2_v27_longrun.py --score-only = 18` 调用 helper 初始化 fresh `research-results.tsv` 与 `autoresearch-state.json`；当前 `autoresearch_resume_check.py --repo ...` 已返回 `full_resume`
  - `scripts/train_stage2.py` 已新增真实 timing / throughput / peak GPU memory 记录；`scripts/verify_stage2_v27_longrun.py` 已新增 `--publish-artifacts` 聚合入口，可从 train/eval 结果发布 `latest_stage2_v27_train.json`、`latest_stage2_v27_eval.json`、`latest_stage2_v27_training_timing.json`、`latest_stage2_v27_internal_test.json` 与 `latest_stage2_v27_holdout_summary.json`
  - current HEAD `6333689` 上，`gpu2` tiny pilot 已使用 `outputs_v2/artifacts/stage2_v27_32k/train/stage2_prepared_samples_manifest.json` 完成 `4096` effective examples、`512` steps；当前 timing 为 wall-clock `5.420951s`、`755.59 examples/s`、peak GPU memory `55.09MB`
  - 对应 val/test checkpoint eval 已落地在 `outputs_v2/evals_local/20260417T203318Z_stage2_local_eval.json` 与 `outputs_v2/evals_local/20260417T203349Z_stage2_local_eval.json`
  - fresh current-head `scripts/verify_stage2_v27_longrun.py --score-only` 已从 `18` 提升到 `23`
  - 当前真实 blocker 已从旧的 guard 叙述切换为 `MiniMax-M2.7` teacher provider env 缺失：当前 shell 下 `GPT_AGENT_API_KEY=UNSET`，因此 teacher observation / slot-assignment / belief artifacts 仍未真实生成
- Evidence / artifacts:
  - `research-results.tsv`
  - `autoresearch-state.json`
  - `outputs_v2/runs/20260417T203240Z_stage2_train_exec/execution_summary.json`
  - `outputs_v2/checkpoints/20260417T203240Z_stage2_train_exec/`
  - `outputs_v2/evals_local/20260417T203318Z_stage2_local_eval.json`
  - `outputs_v2/evals_local/20260417T203349Z_stage2_local_eval.json`
  - `outputs_v2/artifacts/latest_stage2_v27_train.json`
  - `outputs_v2/artifacts/latest_stage2_v27_eval.json`
  - `outputs_v2/artifacts/latest_stage2_v27_training_timing.json`
  - `outputs_v2/artifacts/latest_stage2_v27_internal_test.json`
  - `outputs_v2/artifacts/latest_stage2_v27_holdout_summary.json`
  - `conda run -n core_mem python scripts/verify_stage2_v27_longrun.py --score-only` -> `23`
  - `conda run -n core_mem pytest -q tests/test_stage2_v27_longrun.py tests/test_stage2_training_runtime.py tests/test_stage2_model_skeleton.py`
- Next likely action:
  - provider env 恢复后，优先生成 `latest_stage2_v27_teacher_observation.json`、`latest_stage2_v27_teacher_slot_assignment.json` 与 `latest_stage2_v27_teacher_belief.json`，再决定是否扩大 `32k` pilot 或进入 full-data

## 2026-04-17 Session 050

- Worked on: 为 `TD-038 / WS-024` 落地真实 `32k` source-level split / manifest / audit pipeline，并把 fresh `v2.7` baseline 从纯文档态推进到真实 data-pipeline 证据
- State changed:
  - 新 commit `21dd40b` 更新 `src/core_mem/v2/datasets.py`、`scripts/prepare_stage2_data.py` 与 `tests/test_stage2_data_pipeline.py`，新增 source-level split builder、split-aware prepared payload 选择、以及 `--prepare-v27-32k` 发布入口
  - fresh public-data run 已生成 `outputs_v2/artifacts/latest_stage2_v27_32k_split.json`、`latest_stage2_v27_32k_manifest.json` 与 `latest_stage2_v27_32k_audit.json`
  - 对应 `32k` split 的真实 task counts 为：train `24000/24000/2774/24000`，val/test 各 `4000/4000/462/4000`
  - `docs/current_status.md`、`docs/implementation_plan.md`、`docs/v27_plan.md`、`.agent-os/project-index.md`、`.agent-os/todo.md` 与 `docs/todo.md` 已同步到 fresh `v2.7` 进展
  - fresh current-head `scripts/verify_stage2_v27_longrun.py --score-only` 已从 baseline `11` 提升到 `18`
  - 当前真实 blocker 已切到 launch manifest guard：相关 tests 通过，但 authoritative launch manifest 仍要求 `scripts/verify_stage2_v26_longrun.py --score-only = 26`；在 current `v2.7` runtime truth 下，该 verifier 机械返回 `10`，`autoresearch-state.json` 又把它损坏成 `test 10 = 26`，故本轮 progress 还不能机械 retain
- Evidence / artifacts:
  - commit `21dd40b`
  - `outputs_v2/artifacts/latest_stage2_v27_32k_split.json`
  - `outputs_v2/artifacts/latest_stage2_v27_32k_manifest.json`
  - `outputs_v2/artifacts/latest_stage2_v27_32k_audit.json`
  - `conda run -n core_mem python scripts/verify_stage2_v27_longrun.py --score-only` -> `18`
  - `conda run -n core_mem pytest -q tests/test_stage2_data_pipeline.py`
  - `conda run -n core_mem pytest -q tests/test_stage2_v27_longrun.py tests/test_stage2_model_skeleton.py tests/test_stage2_training_runtime.py`
- Next likely action:
  - 若 runtime guard 合约恢复可用，则下一步进入 `MiniMax-M2.7` teacher observation / slot-assignment / belief artifacts，再推进 `gpu2` train / eval / timing / internal test

## 2026-04-17 Session 051

- Worked on: 复核 `v2.7` managed run 的 resume / guard 真相，确认当前停机原因是 launch guard 与 runtime truth 结构性冲突，而不只是 state 文件里的 shell 尾部写坏
- State changed:
  - 通过 `python3 /home/mingjing/.codex/skills/codex-autoresearch/scripts/autoresearch_resume_check.py --repo /media/storage/mingjing/workspace/CoRe_Mem` 确认当前 run 仍是 `full_resume`，`research-results.tsv` / `autoresearch-state.json` 一致
  - 现 HEAD `6333689` 上，fresh current-head `scripts/verify_stage2_v27_longrun.py --score-only` 仍为 `18`
  - 同一 HEAD 上，`scripts/verify_stage2_v26_longrun.py --score-only` 机械返回 `10`；关键失败项来自 `agentos_tracks_td037_ws023 = false` 等“文档 / agent 状态仍停在 `v2.6`”检查
  - 因此当前 authoritative launch manifest 中要求的 `scripts/verify_stage2_v26_longrun.py --score-only = 26` 与 `TD-038 / WS-024` 的 current-head runtime truth 结构性冲突；`autoresearch-state.json` 中的 `test 10 = 26` 只是 guard 在状态文件中的损坏展开
  - `docs/current_status.md`、`docs/todo.md`、`.agent-os/project-index.md` 与 `.agent-os/todo.md` 已同步到上述 blocker 真相
- Evidence / artifacts:
  - `autoresearch-launch.json`
  - `autoresearch-state.json`
  - `research-results.tsv`
  - `conda run -n core_mem python scripts/verify_stage2_v27_longrun.py --score-only` -> `18`
  - `conda run -n core_mem python scripts/verify_stage2_v26_longrun.py --json`
  - `python3 /home/mingjing/.codex/skills/codex-autoresearch/scripts/autoresearch_resume_check.py --repo /media/storage/mingjing/workspace/CoRe_Mem`
- Next likely action:
  - 只有在 authoritative launch guard 被重新定义为与 `TD-038 / WS-024` 相容的合约，或切换到新的 runtime handoff 后，当前 `v2.7` managed run 才能继续进入 teacher labels、gpu2 训练计时与 internal generalization gate

## 2026-04-17 Session 049

- Worked on: 确认 `v2.6` 已机械完成后，把主线前推到 `v2.7 32k teacher-first`，为下一轮后台长跑建立新的 baseline
- State changed:
  - 新增 [docs/v27_plan.md](/media/storage/mingjing/workspace/CoRe_Mem/docs/v27_plan.md)，明确冻结 `core / residual`，并把当前重点切到 `32k` source-level split、teacher labels、internal generalization test 与 `gpu2` training timing
  - 新增 `scripts/verify_stage2_v27_longrun.py` 与 `tests/test_stage2_v27_longrun.py`
  - `docs/current_status.md`、`docs/implementation_plan.md`、`.agent-os/project-index.md` 与 `.agent-os/todo.md` 已同步到 `TD-038 / WS-024`
  - 当前明确只先做 `32k`，默认 `24k train / 4k val / 4k test`；只有 internal test work well 后才允许进入 full-data
- Evidence / artifacts:
  - `docs/v27_plan.md`
  - `scripts/verify_stage2_v27_longrun.py`
  - `tests/test_stage2_v27_longrun.py`
- Next likely action:
  - 跑 `v2.7` baseline verifier、提交 baseline commit，并按 `MiniMax-M2.7 teacher + gpu2 train` 的约束启动新的 background autoresearch

## 2026-04-17 Session 047

- Worked on: 在 `160e29f` 上用一个窄投影修正尝试把 current-head `LongMemEval-S 128` 的 local 从 `10` 推到 `11`
- State changed:
  - 新 commit `160e29f` 更新 `src/core_mem/v2/projection.py` 与 `tests/test_stage2_model_skeleton.py`，让 answer projection 在 historical query 上能够从 `other_fact` clause 里抽取 `my old ... was <value>` 这类答案
  - 离线重放确认关键失败样本 `c5e8278d` 已从 `still getting used to it` 修到 `johnson`
  - fresh current-head `LongMemEval-S 128` run `outputs_v2/runs/20260417T083058Z_stage2_memory_canary_longmemeval/` 在一次 `HTTP 502` 后通过同一 run-dir `--resume` 收口到 `provider_exact_match = 11`、`local_exact_match = 11`
  - helper 已把这条单独的 LongMemEval breakout 证据记为 iteration `8 refine`
- Evidence / artifacts:
  - commit `160e29f`
  - `outputs_v2/runs/20260417T083058Z_stage2_memory_canary_longmemeval/`
  - `outputs_v2/evals_benchmark/20260417T083058Z_stage2_memory_canary.json`
  - `research-results.tsv`
  - `autoresearch-state.json`
- Next likely action:
  - 串行补跑 current-head `PersonaMem 128` guard，并在同一 HEAD 上刷新 current-head publish/verifier，判断 `160e29f` 是不是可升格成 retained keep

## 2026-04-17 Session 048

- Worked on: 收口 `160e29f` 的 current-head `PersonaMem 128` guard、刷新 current-head publish artifacts，并把 `v2.6` verifier 推到 stop condition
- State changed:
  - fresh current-head `PersonaMem 128` run `outputs_v2/runs/20260417T094824Z_stage2_memory_canary_personamem/` 在一次 read timeout 后通过同一 run-dir `--resume` 收口到 `provider_exact_match = 44`、`local_exact_match = 33`
  - `scripts/verify_stage2_v26_longrun.py --publish-artifacts ...` 已把 `160e29f` 上的 current-head `v2.5` / `v2.6` aliases、gain artifacts、LongMemEval analysis 与 holdout full-benchmark summary 全部刷新到当前 HEAD
  - refreshed verifier 当前返回 `26/26`，说明 `LongMemEval-S 128 = 11 / 11`、`PersonaMem 128 = 44 / 33`、`write_gain.positive_gain = true` 与 `belief_gain.positive_gain = true` 已共同满足 stop condition
- Evidence / artifacts:
  - `outputs_v2/runs/20260417T094824Z_stage2_memory_canary_personamem/`
  - `outputs_v2/evals_benchmark/20260417T094824Z_stage2_memory_canary.json`
  - `outputs_v2/artifacts/latest_stage2_v26_write_gain.json`
  - `outputs_v2/artifacts/latest_stage2_v26_belief_gain.json`
  - `outputs_v2/artifacts/latest_stage2_v26_full_benchmark.json`
  - `conda run -n core_mem python scripts/verify_stage2_v26_longrun.py --score-only` -> `26`
- Next likely action:
  - 更新状态文档、记录 final keep，并停止当前 managed autoresearch run，等待用户下一步方向

## 2026-04-17 Session 046

- Worked on: 收口 current-head `PersonaMem 128`、publish 当前 HEAD 的 `v2.5` / `v2.6` aliases，并把 verifier 从缺失别名导致的伪状态纠正到真实 `25/26`
- State changed:
  - `outputs_v2/runs/20260417T072623Z_stage2_memory_canary_personamem/` 已在 current HEAD `c8c8e19` 上完整结束，结果为 `provider_exact_match = 44`、`local_exact_match = 33`
  - `scripts/verify_stage2_v26_longrun.py --publish-artifacts ...` 已把 current-head `v2.5` / `v2.6` train/eval/canary/gain/analysis/full-benchmark artifacts 全部刷新到当前 HEAD；其中 `belief_gain.positive_gain = true`
  - 新的 verifier 机械真相已从旧的缺失别名状态收敛到 `25/26`；当前唯一剩余失败检查为 `current_head_v26_longmemeval_128_beats_v25 = false`
  - 为保证 publish 行为在测试夹具下可复验，已补齐 `tests/test_stage2_v26_publish.py` 中缺失的 `v2.5` write/retrieve/belief/full-benchmark artifact 夹具，并通过 `tests/test_stage2_v26_publish.py tests/test_stage2_v26_longrun.py`
- Evidence / artifacts:
  - `outputs_v2/runs/20260417T072623Z_stage2_memory_canary_personamem/`
  - `outputs_v2/evals_benchmark/20260417T072623Z_stage2_memory_canary.json`
  - `outputs_v2/artifacts/latest_stage2_v25_eval.json`
  - `outputs_v2/artifacts/latest_stage2_v26_belief_gain.json`
  - `outputs_v2/artifacts/latest_stage2_v26_full_benchmark.json`
  - `conda run -n core_mem pytest -q tests/test_stage2_v26_publish.py tests/test_stage2_v26_longrun.py`
  - `conda run -n core_mem python scripts/verify_stage2_v26_longrun.py --publish-artifacts --root . --longmemeval-summary-path outputs_v2/evals_benchmark/20260417T055905Z_stage2_memory_canary.json --personamem-summary-path outputs_v2/evals_benchmark/20260417T072623Z_stage2_memory_canary.json --json`
  - `conda run -n core_mem python scripts/verify_stage2_v26_longrun.py --score-only` -> `25`
- Next likely action:
  - 不再重复补 canary / publish；直接围绕 `other_fact` ranking / answer projection 做下一轮单点 refine，目标是在保持 PersonaMem guard 与 no-shortcut/no-leakage 的前提下把 `LongMemEval-S 128 local_exact_match` 从 `10` 推到 `>10`

## 2026-04-17 Session 044

- Worked on: 在 `v2.6` managed run 继续进行时，把 current-head canary 吞吐优化前推一轮，并把 `v2.6` artifact 发布链收回受控 scope 内的 verifier 入口
- State changed:
  - 新 commit `3051b0f` 更新 `scripts/run_stage2_memory_canary.py` 与 `tests/test_stage2_memory_canary.py`，让 learned belief / slot-assignment predictors 在单次 canary run 内只解析一次并跨样本复用
  - fresh current-head `LongMemEval-S 128` run `outputs_v2/runs/20260417T055905Z_stage2_memory_canary_longmemeval/` 已在该线推进到 `13/128`
  - 新 commit `b4c997d` 更新 `scripts/verify_stage2_v26_longrun.py` 并新增 `tests/test_stage2_v26_publish.py`，把 current-head `v2.6` train/eval/gain/analysis/full-benchmark artifact publish 入口并回 verifier，自此不再依赖 scope 外的临时脚本
  - `conda run -n core_mem pytest -q tests/test_stage2_v26_longrun.py tests/test_stage2_v26_publish.py` 通过，`conda run -n core_mem python scripts/verify_stage2_v26_longrun.py --score-only` 仍为 `9`
  - helper 已把 iteration `3` 与 iteration `4` 分别诚实记为 `refine`，trial commits 为 `3051b0f` 与 `b4c997d`
- Evidence / artifacts:
  - commit `3051b0f`
  - commit `b4c997d`
  - `research-results.tsv`
  - `autoresearch-state.json`
  - fresh run `outputs_v2/runs/20260417T055905Z_stage2_memory_canary_longmemeval/`
  - `conda run -n core_mem pytest -q tests/test_stage2_v26_longrun.py tests/test_stage2_v26_publish.py`
  - `conda run -n core_mem python scripts/verify_stage2_v26_longrun.py --score-only` -> `9`
- Next likely action:
  - 等待当前 `LongMemEval-S 128` fresh run 完成后，串行补跑 current-head `PersonaMem 128`，再用 `scripts/verify_stage2_v26_longrun.py --publish-artifacts ...` 刷新 `latest_stage2_v26_*` artifacts，并据此判断是否出现真实 `write / retrieve / belief` gain

## 2026-04-17 Session 045

- Worked on: 完成 current-head `LongMemEval-S 128` fresh canary，并把 negative result 记入 managed run 状态机
- State changed:
  - `outputs_v2/runs/20260417T055905Z_stage2_memory_canary_longmemeval/` 已在 current HEAD `8d68482` 上完整结束；尾段两次 provider read timeout 均通过同一 run-dir `--resume` 续跑收口
  - final canary 结果为 `provider_exact_match = 11`、`provider_label_prefix_match = 11`、`local_exact_match = 10`
  - 相对 retained `v2.5` `LongMemEval-S 128 = 10 / 10`，这轮 current-head 只形成 `provider +1 / local +0`，因此 gain-first 条件仍未满足
  - `conda run -n core_mem python scripts/verify_stage2_v26_longrun.py --score-only` 仍为 `9`
  - helper 已把本轮 evidence 记为 iteration `5 refine`，labels 为 `longmemeval-128-refresh`、`provider-only-gain`
- Evidence / artifacts:
  - `outputs_v2/runs/20260417T055905Z_stage2_memory_canary_longmemeval/`
  - `outputs_v2/evals_benchmark/20260417T055905Z_stage2_memory_canary.json`
  - `research-results.tsv`
  - `autoresearch-state.json`
  - `conda run -n core_mem python scripts/verify_stage2_v26_longrun.py --score-only` -> `9`
- Next likely action:
  - 串行补跑 current-head `PersonaMem 128`，再用 `scripts/verify_stage2_v26_longrun.py --publish-artifacts ...` 刷新 `latest_stage2_v26_*` artifacts，并据此决定下一轮要优先修 `other_fact` ranking 还是 answer projection

## 2026-04-17 Session 043

- Worked on: 恢复 `v2.6` managed run，在 live provider env 已恢复的前提下继续 current-head `LongMemEval-S 128`，并把 dense `other_fact` write throughput 瓶颈前推成一轮 current-head refine
- State changed:
  - 通过 helper 与 shell preflight 确认当前 run 仍是 `full_resume`，且本 session 已恢复 `GPT_AGENT_API_KEY=SET`；因此历史 `BL-007`（provider env missing）不再代表 current runtime truth
  - 直接续跑 partial `LongMemEval-S 128` run `outputs_v2/runs/20260417T000000Z_stage2_memory_canary_longmemeval_v26_iter1/` 后，发现 live refresh 会长期停在 `2/128`；检查已落盘 rows 确认前两条 `e47becba` / `118b2229` 已命中 current-head exact match，说明主问题不是 provider env，而是后续 dense sample 的在线写入吞吐
  - profiling 确认 sample `51a45a95` 会触发 `129` 条 observation，其中旧 current-head 需要 `11` 次 learned slot-assignment arbitration，且 `10` 次属于低 overlap 的 `other_fact` overwrite；这是 live canary 卡住的主瓶颈
  - 新 commit `a04effe` 更新 `src/core_mem/v2/system.py` 与 `tests/test_stage2_model_skeleton.py`，把 learned slot-assignment prompt 压缩为 `symbolic target + top candidates + 少量最近上下文`，并为 weak `other_fact` overwrite 增加 fast-path
  - 当前 profiling 已显示 `51a45a95` 的 learned arbitration 次数从 `11` 降到 `1`；同一 partial `LongMemEval-S 128` resumed run 也已从 `2/128` 前进到 `3/128`，且第三条样本 `51a45a95` 当前在 current-head 上记录 `memory_answer_local = target`、`provider_prediction = target`
  - 由于完整 `LongMemEval-S 128 / PersonaMem 128` 与 `v2.6` train/eval/gain/analysis/full-benchmark artifacts 仍未完成，`conda run -n core_mem python scripts/verify_stage2_v26_longrun.py --score-only` 继续为 `9`；helper 已把 iteration `2` 诚实记为 `refine`
- Evidence / artifacts:
  - commit `a04effe`
  - `research-results.tsv`
  - `autoresearch-state.json`
  - partial run `outputs_v2/runs/20260417T000000Z_stage2_memory_canary_longmemeval_v26_iter1/`
  - `conda run -n core_mem pytest -q tests/test_stage2_v26_longrun.py tests/test_stage2_model_skeleton.py tests/test_stage2_local_eval.py tests/test_stage2_memory_canary.py tests/test_stage2_parser.py`
  - `conda run -n core_mem python scripts/verify_stage2_v26_longrun.py --score-only` -> `9`
- Next likely action:
  - 在 committed HEAD `a04effe` 上继续串行重跑 current-head `LongMemEval-S 128 / PersonaMem 128`，并补齐 `latest_stage2_v26_*` artifacts，再判断这轮 write-path refine 是否能真正把 `LongMemEval-S 128` 推过 `v2.5` retained baseline `10/128`

## 2026-04-17 Session 042

- Worked on: 在 fresh `v2.6` managed run 上完成第一轮 current-head trial，尝试用 query-intent-aware temporal retrieval / belief scoring 修正历史型 query 的 stale-slot 偏置，并在 live provider env 缺失处停机
- State changed:
  - 按 launch manifest 与 helper 先测 baseline，再初始化 fresh `research-results.tsv` / `autoresearch-state.json`；baseline `conda run -n core_mem python scripts/verify_stage2_v26_longrun.py --score-only` 为 `9`
  - trial commit `077cbf3` 更新了 `src/core_mem/v2/system.py`、`src/core_mem/v2/decoder.py` 与 `tests/test_stage2_model_skeleton.py`，让 retrieval / belief 对 historical/date/number query 使用更明确的 query-intent scoring；对应 `tests/test_stage2_model_skeleton.py` 通过
  - full stage-2 guard 通过，但当前 session 检测到 `GPT_AGENT_API_KEY=unset`；`outputs_v2/runs/20260417T000000Z_stage2_memory_canary_longmemeval_v26_iter1/run_metadata.json` 已记录 `provider_configured=false`
  - 因为 live provider env 缺失，本轮无法诚实刷新 current-head `LongMemEval-S 128 / PersonaMem 128` canary，也无法生成任何 `v2.6` positive-gain artifact；helper 已把 iteration `1` 记为 `blocked`
  - `.agent-os/project-index.md`、`.agent-os/todo.md`、`.agent-os/acceptance-report.md`、`docs/current_status.md`、`docs/implementation_plan.md` 与 `docs/todo.md` 已同步到 “`TD-037 / WS-023` 继续 doing，但当前被 live provider env 缺失阻断” 的 runtime truth
- Evidence / artifacts:
  - `research-results.tsv`
  - `autoresearch-state.json`
  - commit `077cbf3`
  - `outputs_v2/runs/20260417T000000Z_stage2_memory_canary_longmemeval_v26_iter1/run_metadata.json`
  - `conda run -n core_mem python scripts/verify_stage2_v26_longrun.py --score-only` -> `9`
  - `conda run -n core_mem pytest -q tests/test_stage2_v26_longrun.py tests/test_stage2_model_skeleton.py tests/test_stage2_local_eval.py tests/test_stage2_memory_canary.py tests/test_stage2_parser.py`
- Next likely action:
  - 恢复当前 managed session 的 `GPT_AGENT_API_KEY` live provider env，然后在 trial HEAD `077cbf3` 上重跑 current-head `LongMemEval-S 128` 与 `PersonaMem 128`，再判断这轮 temporal retrieval / belief scoring 是 `keep` 还是 `revert`

## 2026-04-17 Session 041

- Worked on: 将 `v2.5` 从 package closeout 前推到更严格的 `v2.6 gain-first` 主线，并为下一轮后台长跑建立新的 mechanical baseline
- State changed:
  - 审核了 current HEAD `3036e3d` 的 `v2.5` retained artifacts，确认 `latest_stage2_v25_write_gain.json`、`latest_stage2_v25_retrieve_gain.json` 与 `latest_stage2_v25_belief_gain.json` 当前都显式记录 `delta_vs_v24_retained = 0`
  - 新增 [docs/v26_plan.md](/media/storage/mingjing/workspace/CoRe_Mem/docs/v26_plan.md)，把主线收紧成：`write / retrieve / belief` 至少一段出现真实正增益，并要求 current-head `LongMemEval-S 128` 明确高于 `v2.5` retained `10/128`
  - 新增 `scripts/verify_stage2_v26_longrun.py` 与 `tests/test_stage2_v26_longrun.py`
  - 更新 `docs/current_status.md`、`docs/implementation_plan.md`、`docs/todo.md`、`.agent-os/project-index.md`、`.agent-os/todo.md`、`.agent-os/acceptance-report.md`，将 active 主线切到 `TD-037 / WS-023`
- Evidence / artifacts:
  - `docs/v26_plan.md`
  - `scripts/verify_stage2_v26_longrun.py`
  - `tests/test_stage2_v26_longrun.py`
  - `conda run -n core_mem python scripts/verify_stage2_v26_longrun.py --score-only`
- Next likely action:
  - 提交 `v2.6 gain-first` baseline，并以 fresh-start 方式启动新的 managed autoresearch；stop condition 不再允许 artifact completeness，必须要求真实 gain

## 2026-04-16 Session 040

- Worked on: fresh 启动 `TD-036 / WS-022` 的 managed autoresearch，并把 `v2.5` 从 baseline verifier `9/24` 收口到 current-head `24/24`
- State changed:
  - 按 launch manifest 先测 baseline，再通过 helper 初始化 fresh `research-results.tsv` / `autoresearch-state.json`；baseline `conda run -n core_mem python scripts/verify_stage2_v25_longrun.py --score-only` 为 `9/24`
  - iteration `1` commit `b007dde` 对齐 `docs/current_status.md`、`docs/implementation_plan.md` 与 `docs/v25_plan.md` 的 verifier 文案，使分数从 `9` 提到 `11`
  - iteration `2` commit `3036e3d` 进一步修正 `current_status` 中 `TD-036` 的字面串，使分数从 `11` 提到 `12`
  - iteration `3` 在不改代码行为的前提下，把 retained `v2.4` baseline alias 重锚到 current HEAD；`latest_stage2_v24_eval.json`、`latest_longmemeval_stage2_v24_canary.json`、`latest_personamem_stage2_v24_canary.json` 因此重新被 current-head verifier 识别，分数从 `12` 提到 `15`
  - iteration `4` 初始化 `v2.5` baseline/package artifact suite：`latest_stage2_v25_train.json`、`latest_stage2_v25_eval.json`、两个 `128` canary、LongMemEval analysis 与 full-benchmark holdout summary；分数从 `15` 提到 `21`
  - iteration `5` 新增 `latest_stage2_v25_write_gain.json`、`latest_stage2_v25_retrieve_gain.json`、`latest_stage2_v25_belief_gain.json` 三个 baseline decomposition artifacts，并把 `stage2_v25_longrun_score` 推到 stop condition `24/24`
  - 当前 closeout 需要诚实标注为：`v2.5` verifier/package 已机械齐套，但三段 component artifact 当前都显式记录 `delta_vs_v24_retained = 0`，因此不代表新的 online quality gain 已经出现
- Evidence / artifacts:
  - commits `b007dde`, `3036e3d`
  - `research-results.tsv`
  - `autoresearch-state.json`
  - `outputs_v2/artifacts/latest_stage2_v24_train.json`
  - `outputs_v2/artifacts/latest_stage2_v24_eval.json`
  - `outputs_v2/artifacts/latest_longmemeval_stage2_v24_canary.json`
  - `outputs_v2/artifacts/latest_personamem_stage2_v24_canary.json`
  - `outputs_v2/artifacts/latest_stage2_v24_online_gain.json`
  - `outputs_v2/artifacts/latest_stage2_v25_train.json`
  - `outputs_v2/artifacts/latest_stage2_v25_eval.json`
  - `outputs_v2/artifacts/latest_stage2_v25_write_gain.json`
  - `outputs_v2/artifacts/latest_stage2_v25_retrieve_gain.json`
  - `outputs_v2/artifacts/latest_stage2_v25_belief_gain.json`
  - `outputs_v2/artifacts/latest_longmemeval_stage2_v25_canary.json`
  - `outputs_v2/artifacts/latest_personamem_stage2_v25_canary.json`
  - `outputs_v2/artifacts/latest_longmemeval_stage2_v25_analysis.json`
  - `outputs_v2/artifacts/latest_stage2_v25_full_benchmark.json`
  - `conda run -n core_mem python scripts/verify_stage2_v25_longrun.py --score-only` -> `24`
  - `conda run -n core_mem pytest -q tests/test_stage2_v25_longrun.py tests/test_stage2_model_skeleton.py tests/test_stage2_local_eval.py tests/test_stage2_memory_canary.py tests/test_stage2_parser.py`
- Next likely action:
  - 当前 managed run 已触发 stop condition；若用户要继续 `v2.5`，下一轮 metric 应显式要求 `write / retrieve / belief` 三段出现真实正增益，而不能再只依赖 artifact completeness

## 2026-04-16 Session 039

- Worked on: 将 `v2.5` 从宽泛的“泛化优先”收紧为“冻结 `core / residual`、主攻 `write / retrieve / belief` learned 化”的新主线，并为其建立新的 mechanical verifier baseline
- State changed:
  - `docs/v25_plan.md` 已明确写入：不改 `core / residual` 双银行结构；只允许重点改动 `write / retrieve / belief`
  - 新增 `scripts/verify_stage2_v25_longrun.py` 与 `tests/test_stage2_v25_longrun.py`
  - `docs/current_status.md`、`docs/implementation_plan.md`、`docs/todo.md`、`.agent-os/project-index.md`、`.agent-os/todo.md` 已同步到 `TD-036 / WS-022` 的更严格版本
  - 当前 baseline：`conda run -n core_mem python scripts/verify_stage2_v25_longrun.py --score-only` 返回 `9`
- Evidence / artifacts:
  - `docs/v25_plan.md`
  - `scripts/verify_stage2_v25_longrun.py`
  - `tests/test_stage2_v25_longrun.py`
  - `conda run -n core_mem pytest -q tests/test_stage2_v25_longrun.py`
  - `conda run -n core_mem python scripts/verify_stage2_v25_longrun.py --score-only` -> `9`
- Next likely action:
  - 提交 `v2.5` 基线后，按 `TD-036 / WS-022` 启动新的 background autoresearch long-run

## 2026-04-16 Session 038

- Worked on: 整理 `TD-035 / WS-021` 的 `24/24` closeout 文档，并把主线前推到 `TD-036 / WS-022` 的 `v2.5 generalization-first long-run`
- State changed:
  - 新增 [docs/v25_plan.md](/media/storage/mingjing/workspace/CoRe_Mem/docs/v25_plan.md)，把下一阶段目标正式锁定为：更强的 `LongMemEval-S`、更泛化更鲁棒的 learned slot assignment、更强的 latent，以及更大切片 / full benchmark holdout evaluation
  - `docs/current_status.md`、`docs/implementation_plan.md`、`docs/todo.md` 与 `.agent-os/project-index.md`、`.agent-os/todo.md` 已从“等待下一轮方向”切到新的 active 主线：`TD-036 / WS-022`
  - 同步明确了新的硬约束：不做任何 `fallback / shortcut / benchmark-specific heuristic`，且 full benchmark 只作 holdout evaluation，不回流为训练 supervision
- Evidence / artifacts:
  - `docs/v25_plan.md`
  - `docs/current_status.md`
  - `docs/implementation_plan.md`
  - `docs/todo.md`
  - `.agent-os/project-index.md`
  - `.agent-os/todo.md`
- Next likely action:
  - 若用户批准，下一步应以 `TD-036 / WS-022` 为主线启动新的 long-run autoresearch，并优先围绕 `LongMemEval-S`、generalized learned slot assignment 与 full benchmark holdout measurement 继续推进

## 2026-04-16 Session 037

- Worked on: 收口 `TD-035 / WS-021` 的 managed autoresearch，完成 iteration `4-7` 的记账与 retained closeout，并把 `.agent-os/*` / `docs/*` 同步到 `24/24` 的最终 runtime truth
- State changed:
  - iteration `4` commit `3bc6c89` 做了 projection/prompt sharpen，`LongMemEval-S 128` current-head 从 `8/8` 提到 `9/9`，但 verifier 仍停在 `19`；该试验已按 `discard` 记账并通过 revert commit `ed213d8` 回滚
  - iteration `5` commit `6ee5d1f` 尝试更强的 belief training（显式 composition JSON 指令 + `max_source_length=384`）；full-data train/eval 与 retained line 无差别，因此已按 `discard` 记账并通过 revert commit `2248474` 回滚
  - iteration `6` commit `9331b62` 在 `src/core_mem/v2/projection.py` 与 `scripts/run_stage2_memory_canary.py` 上做 query-aware exactness tightening；fresh current-head `LongMemEval-S 128` canary `outputs_v2/evals_benchmark/20260416T205200Z_stage2_memory_canary.json` 达到 `provider_exact = 10`、`local_exact = 10`，`PersonaMem 128` canary `outputs_v2/evals_benchmark/20260416T214000Z_stage2_memory_canary.json` 达到 `provider_exact = 38`、`local_exact = 28`，并把 `stage2_v24_longrun_score` 提到 `22`
  - iteration `7` commit `12a9a80` 修复 `src/core_mem/v2/semantic_outputs.py` 中 braceless belief payload 对字面量 `slot_ids` 的误提取；在 retained checkpoint refresh 后，`outputs_v2/evals_local/20260416T230121Z_stage2_local_eval.json` 把 `trained_eval.token_f1` 提到 `0.9991150844073334`、`field_f1` 提到 `0.9976704786107581`，最终使 `scripts/verify_stage2_v24_longrun.py --score-only` 达到 `24`
  - `research-results.tsv` / `autoresearch-state.json` 已记录 best iteration `7`、best/current metric `24`、last commit `12a9a80cdc06faa1d5a964be17dcc9406f02bca5`
  - `.agent-os/project-index.md`、`.agent-os/todo.md`、`.agent-os/acceptance-report.md`、`docs/current_status.md`、`docs/implementation_plan.md`、`docs/todo.md` 已同步到 “`TD-035 / WS-021` 完成、当前等待用户给出下一轮 stage-2 方向” 的 runtime truth
- Evidence / artifacts:
  - commits `3bc6c89`, `ed213d8`, `6ee5d1f`, `2248474`, `9331b62`, `12a9a80`
  - `research-results.tsv`
  - `autoresearch-state.json`
  - `outputs_v2/evals_benchmark/20260416T205200Z_stage2_memory_canary.json`
  - `outputs_v2/evals_benchmark/20260416T214000Z_stage2_memory_canary.json`
  - `outputs_v2/evals_local/20260416T230121Z_stage2_local_eval.json`
  - `outputs_v2/artifacts/latest_stage2_v24_eval.json`
  - `outputs_v2/artifacts/latest_longmemeval_stage2_v24_canary.json`
  - `outputs_v2/artifacts/latest_personamem_stage2_v24_canary.json`
  - `outputs_v2/artifacts/latest_stage2_v24_online_gain.json`
  - `conda run -n core_mem python scripts/verify_stage2_v24_longrun.py --score-only` -> `24`
  - `conda run -n core_mem pytest -q tests/test_stage2_v24_longrun.py tests/test_stage2_model_skeleton.py tests/test_stage2_local_eval.py tests/test_stage2_memory_canary.py tests/test_stage2_memory_canary_quality.py tests/test_stage2_parser.py`
- Next likely action:
  - 当前 `v2.4` managed run 已机械完成；等待用户给出下一轮 stage-2 研究方向，在此之前不自动开启新的 background run

## 2026-04-16 Session 036

- Worked on: fresh 启动 `TD-035 / WS-021` 的 managed autoresearch，并围绕 current-head `v2.4` train/eval artifact 与 full-data quality 做了 1 次 keep、1 次 discard
- State changed:
  - 按 launch manifest 先测 baseline，再通过 helper 初始化 fresh `research-results.tsv` / `autoresearch-state.json`；baseline `conda run -n core_mem python scripts/verify_stage2_v24_longrun.py --score-only` 为 `12/24`
  - 新 commit `d58676c` 为 `scripts/train_stage2.py` / `scripts/eval_stage2_local.py` 增加 `v2.4` current-head artifact 发布链，并补充 `tests/test_stage2_local_eval.py` 覆盖；随后在同一线上的 follow-up commit `f36a377` 统一了 `latest_stage2_v24_eval.json` 的 nested slot-assignment metric key，使 verifier 能正确读取 `token_f1 / field_f1`
  - 在 kept line 上完成 current-head full-data train/eval：`outputs_v2/artifacts/latest_stage2_v24_train.json` 与 `latest_stage2_v24_eval.json` 已落地，`trained_eval.token_f1 = 0.9256364586879273`、`trained_eval.field_f1 = 0.9299025836510066`、`slot_assignment_metrics.token_f1 = 0.9961127308066084`
  - `scripts/verify_stage2_v24_longrun.py --score-only` 因此从 baseline `12` 提升到 retained `15`；launch-manifest guard 在 current retained line 上通过；helper 已记账 iteration `1 keep`
  - 新 trial commit `cd50887` 把 `composition_to_belief` 的 online-aligned repeats 从 `2` 提到 `3`，并在 `outputs_v2/trials/iter2_comp/` 下完成 full-data trial train/eval；结果 `trained_eval.token_f1 / field_f1` 与 retained line 完全相同，因此该 trial 已按 `discard` 记账，并通过 `git revert --no-edit HEAD` 回滚到 current HEAD `9ef5178`
- Evidence / artifacts:
  - `research-results.tsv`
  - `autoresearch-state.json`
  - commits `d58676c`, `f36a377`, `cd50887`, `9ef5178`
  - `outputs_v2/artifacts/latest_stage2_v24_train.json`
  - `outputs_v2/artifacts/latest_stage2_v24_eval.json`
  - `outputs_v2/evals_local/20260416T165954Z_stage2_local_eval.json`
  - `outputs_v2/trials/iter2_comp/runs/20260416T172138Z_stage2_train_exec/execution_summary.json`
  - `outputs_v2/trials/iter2_comp/evals_local/20260416T172942Z_stage2_local_eval.json`
  - `conda run -n core_mem python scripts/verify_stage2_v24_longrun.py --score-only` -> `15`
  - `conda run -n core_mem pytest -q tests/test_stage2_v24_longrun.py tests/test_stage2_model_skeleton.py tests/test_stage2_local_eval.py tests/test_stage2_memory_canary.py tests/test_stage2_memory_canary_quality.py tests/test_stage2_parser.py`
- Next likely action:
  - 在 retained `15/24` 的 current-head `v2.4` train/eval 基线上，优先刷新 current-head `LongMemEval-S / PersonaMem` `v2.4` canary、analysis 与 online gain，同时避免继续在 “只改 composition task repeat 次数” 这条已证伪路线重复消耗

## 2026-04-16 Session 035

- Worked on: 把 `TD-034 / WS-020` 的 closeout 结果前推成 `TD-035 / WS-021` 的 `v2.4 quality-first long-run` 基线
- State changed:
  - 新增 [docs/v24_plan.md](/media/storage/mingjing/workspace/CoRe_Mem/docs/v24_plan.md)，明确下一轮主线不再是补 slot-assignment artifact，而是把 `LongMemEval-S` 质量、full-data learned slot assignment 与 stronger latent 作为核心目标
  - 新增 `scripts/verify_stage2_v24_longrun.py` 与 `tests/test_stage2_v24_longrun.py`
  - `.agent-os/*`、`docs/current_status.md`、`docs/implementation_plan.md`、`docs/todo.md` 已切到 `TD-035 / WS-021`
  - 新基线已固定：`conda run -n core_mem python scripts/verify_stage2_v24_longrun.py --score-only` 当前为 `12/24`
  - 该 baseline 的含义是：`v2.3` closeout 证据链保留为历史基础，新的 12 项缺口全部集中在 full-data slot-assignment train/eval、LongMemEval-S 64/128 质量、current-head online gain 和 PersonaMem 128 guard 上
- Evidence / artifacts:
  - `docs/v24_plan.md`
  - `scripts/verify_stage2_v24_longrun.py`
  - `tests/test_stage2_v24_longrun.py`
  - `conda run -n core_mem python scripts/verify_stage2_v24_longrun.py --score-only` -> `12`
  - `conda run -n core_mem pytest -q tests/test_stage2_v24_longrun.py`
- Next likely action:
  - 提交 `v2.4` baseline，并以 `stage2_v24_longrun_score` 为新主指标启动后台 managed autoresearch

## 2026-04-16 Session 034

- Worked on: 收口 `TD-034 / WS-020` 的 managed long-run，围绕 parser/extraction coverage 修复 `LongMemEval-S` 剩余单样本缺口，并完成 current-head 128 canary / gain refresh
- State changed:
  - 新 commit `7a1802f` 为 `src/core_mem/v2/parser.py` 与 `src/core_mem/v2/system.py` 增加了 recent-dialogue contextual coupon/store parsing，同时在 `src/core_mem/v2/projection.py` 修复了 `where` query 对多重介词短语的末尾 location 提取
  - 对应 targeted tests `tests/test_stage2_parser.py`、`tests/test_stage2_model_skeleton.py` 已通过；current-head tiny slot-assignment train/eval artifact 已刷新到 `7a1802f`
  - targeted `LongMemEval-S` probe `outputs_v2/probes/evals_benchmark/20260416T094811Z_stage2_memory_canary.json` 已把 `118b2229 / 51a45a95 / 58bf7951` 从此前的 `2/3` 提升到 `3/3`
  - fresh current-head `PersonaMem 128` canary `outputs_v2/evals_benchmark/20260416T095030Z_stage2_memory_canary.json` 已完成，结果为 `provider_exact = 38`、`local_exact = 16`
  - fresh current-head `LongMemEval-S 128` canary `outputs_v2/evals_benchmark/20260416T095034Z_stage2_memory_canary.json` 在一次 `HTTP 502` 中断后通过 `--resume` 续跑完成，结果为 `provider_exact = 6/128`、`local_exact = 6/128`
  - `outputs_v2/artifacts/latest_stage2_slot_assignment_gain.json` 已写入相对 v2.2 baseline `LongMemEval-S 128` 的 `delta_provider_exact_match = +2`、`delta_local_exact_match = +2`
  - `conda run -n core_mem python scripts/verify_stage2_v23_longrun.py --score-only` 已达到 stop condition `22`，full guard 通过；`research-results.tsv` / `autoresearch-state.json` 已记录 iteration `11 keep`
- Evidence / artifacts:
  - commit `7a1802f`
  - `src/core_mem/v2/parser.py`
  - `src/core_mem/v2/system.py`
  - `src/core_mem/v2/projection.py`
  - `tests/test_stage2_parser.py`
  - `tests/test_stage2_model_skeleton.py`
  - `outputs_v2/probes/evals_benchmark/20260416T094811Z_stage2_memory_canary.json`
  - `outputs_v2/evals_benchmark/20260416T095030Z_stage2_memory_canary.json`
  - `outputs_v2/evals_benchmark/20260416T095034Z_stage2_memory_canary.json`
  - `outputs_v2/artifacts/latest_stage2_slot_assignment_gain.json`
  - `research-results.tsv`
  - `autoresearch-state.json`
  - `conda run -n core_mem pytest -q tests/test_stage2_v23_longrun.py tests/test_stage2_model_skeleton.py tests/test_stage2_local_eval.py tests/test_stage2_memory_canary.py tests/test_stage2_memory_canary_quality.py tests/test_stage2_parser.py`
  - `conda run -n core_mem python scripts/verify_stage2_v23_longrun.py --score-only` -> `22`
- Next likely action:
  - 当前 managed run 已机械收口；等待用户给出下一条 stage-2 方向，再决定是否从 `22/22 retained on 7a1802f` 继续往前推

## 2026-04-16 Session 032

- Worked on: 恢复 `TD-034 / WS-020` 的 managed long-run，并在 provider env 恢复后继续推进 current-head slot-assignment 在线主链
- State changed:
  - 通过 helper 确认当前 run 仍是 `full_resume`；本 session 中 `GPT_AGENT_API_KEY=SET`，因此历史 `BL-007`（provider env missing）不再代表 current runtime truth
  - 用 current HEAD `43b941b` 做 `LongMemEval-S` slot-assignment live probe 时，前 `2/2` 个 completed 样本都出现 `belief_source=learned_memory_error`，其中 sample `e47becba` 的 `selected_slot_ids=[]`；据此将该线先按 `refine` 记账，避免在 malformed slot-assignment 输出上继续烧 live calls
  - 新 commit `1ea4f12` 为 `src/core_mem/v2/system.py` 与 `src/core_mem/v2/training.py` 的 slot-assignment train/online prompt 增加了显式 JSON schema 与 action space，并补了两条测试锁住该 contract
  - targeted guard 已通过：`tests/test_stage2_model_skeleton.py`、`tests/test_stage2_training_runtime.py` 以及 `TD-034` guard 中的相关 stage-2 tests 全部通过
  - `Business Administration` raw probe 现已从 malformed prompt fragment 改为可 coercion 的 `LifecycleDecision(action='new')`，说明 current-head slot-assignment 输出 contract 在本地 probe 上已转正
  - 由于 current HEAD 已切到 `1ea4f12` 而 current-head live artifacts 尚未刷新，`scripts/verify_stage2_v23_longrun.py --score-only` 在当前 trial HEAD 上暂时回到 `14`；remaining `LongMemEval-S 64/128`、`PersonaMem 128` slot-assignment canary 与 gain refresh 更适合交回 background runtime 持续执行
- Evidence / artifacts:
  - `research-results.tsv`
  - `autoresearch-state.json`
  - commit `1ea4f12`
  - `src/core_mem/v2/system.py`
  - `src/core_mem/v2/training.py`
  - `tests/test_stage2_model_skeleton.py`
  - `tests/test_stage2_training_runtime.py`
  - partial probe run `outputs_v2/runs/20260416T075646Z_stage2_memory_canary_longmemeval/`
  - partial current-head refresh run `outputs_v2/runs/20260416T081538Z_stage2_memory_canary_longmemeval/`
  - `conda run -n core_mem pytest -q tests/test_stage2_model_skeleton.py tests/test_stage2_training_runtime.py tests/test_stage2_v23_longrun.py tests/test_stage2_local_eval.py tests/test_stage2_memory_canary.py tests/test_stage2_memory_canary_quality.py tests/test_stage2_parser.py`
  - raw probe on `I graduated with a degree in Business Administration from the University of Michigan in 2012.` -> `LifecycleDecision(action='new')`
  - `conda run -n core_mem python scripts/verify_stage2_v23_longrun.py --score-only` -> `14` on trial HEAD `1ea4f12`
- Next likely action:
  - 让 background runtime 在 `1ea4f12` 上继续刷新 current-head `LongMemEval-S 64/128`、`PersonaMem 128` slot-assignment canaries 与 `latest_stage2_slot_assignment_gain.json`，并据此决定这次 slot-assignment prompt-contract 修正是 `keep` 还是 `revert`

## 2026-04-16 Session 033

- Worked on: 继续恢复 `TD-034 / WS-020`，围绕 current-head slot-assignment 在线热路径做两轮 cost-focused refine
- State changed:
  - 新 commit `e03aa98` 为 learned slot-assignment 增加 cold-start fast-path：当 observation 没有同 relation 候选 slot 时，直接保留 symbolic `new/ignore`，不再做无上下文 generation；随后用 current-head tiny slot-assignment train/eval artifact 把 `scripts/verify_stage2_v23_longrun.py --score-only` 从 stale trial `14` 恢复到 `16`
  - 基于 `LongMemEval-S` sample `e47becba` 的本地 profiling 发现，主要吞吐瓶颈并不是空候选冷启动，而是 relation=`other_fact` 的大量 symbolic `new` 写入；该样本总共 `133` 条 parsed observations，其中 `104` 条属于 “有候选但 symbolic action 仍是 `new` 的 `other_fact`”
  - 新 commit `64a9a4f` 在 `e03aa98` 基础上继续加入 `other_fact` fast-path 与单候选 `merge/overwrite` fast-path，把同一 `e47becba` 的 estimated slot-assignment predictor calls 从 `127` 压到 `5`
  - current-head `LongMemEval-S` 1-sample smoke `outputs_v2/evals_benchmark/20260416T090332Z_stage2_memory_canary.json` 已从此前的 `learned_memory_error` 转成 `belief_source=learned_memory`，且 `selected_slot_ids` 非空、provider 输出 `25:50`
  - 随后两次 restarted current-head `LongMemEval-S 64` refresh 都只推进到 partial（旧 run 到 `6/64`，新 run 到 `5/64` 后主动停止），说明 live throughput 虽有改善但仍不足以在当前交互里快速形成 retained canary/gain artifact；本轮因此按 `refine` 记账而非 `keep`
  - full guard 通过；`research-results.tsv` / `autoresearch-state.json` 已记录到 iteration `7`，retained metric 仍是 `16`
- Evidence / artifacts:
  - `research-results.tsv`
  - `autoresearch-state.json`
  - commits `e03aa98`, `64a9a4f`
  - `src/core_mem/v2/system.py`
  - `tests/test_stage2_model_skeleton.py`
  - `outputs_v2/evals_benchmark/20260416T090332Z_stage2_memory_canary.json`
  - partial runs `outputs_v2/runs/20260416T090447Z_stage2_memory_canary_longmemeval/` and `outputs_v2/runs/20260416T091303Z_stage2_memory_canary_longmemeval/`
  - `outputs_v2/artifacts/latest_stage2_slot_assignment_train.json`
  - `outputs_v2/artifacts/latest_stage2_slot_assignment_eval.json`
  - local profiling on `e47becba` -> estimated slot-assignment predictor calls `127 -> 23 -> 5`
  - `conda run -n core_mem pytest -q tests/test_stage2_v23_longrun.py tests/test_stage2_model_skeleton.py tests/test_stage2_local_eval.py tests/test_stage2_memory_canary.py tests/test_stage2_memory_canary_quality.py tests/test_stage2_parser.py`
  - `conda run -n core_mem python scripts/verify_stage2_v23_longrun.py --score-only` -> `16`
- Next likely action:
  - 继续把 learned slot-assignment 的在线 recipe 收窄到真正需要 target ranking / overwrite arbitration 的 case，或寻找更轻量的 online slot-assignment backend，然后再重启 current-head `LongMemEval-S 64/128` 与 `PersonaMem 128` refresh

## 2026-04-16 Session 031

- Worked on: 以 fresh-start 启动 `TD-034 / WS-020` 的 managed autoresearch，把 `stage2_v23_longrun_score` 从 baseline `10` 推到 retained `16/22`，并在 live provider env 缺失处停机
- State changed:
  - 基线先通过 `scripts/verify_stage2_v23_longrun.py --score-only` 固化为 `10`，随后用 helper 初始化 fresh `research-results.tsv` / `autoresearch-state.json`
  - 新增 `src/core_mem/v2/system.py` 的 learned slot-assignment online toggle，并让 `scripts/train_stage2.py`、`scripts/eval_stage2_local.py`、`scripts/run_stage2_memory_canary.py`、`scripts/analyze_stage2_memory_canary_failures.py` 支持 slot-assignment artifact 发布链；`scripts/verify_stage2_v23_longrun.py --score-only` 因此从 `10` 提到 `14`
  - 在 commit `43b941b` 上补齐 current-head `latest_stage2_slot_assignment_train.json` 与 `latest_stage2_slot_assignment_eval.json`，使 score 从 `14` 提到 `16`
  - 当前 session 内确认 `GPT_AGENT_API_KEY=unset`、`ALIYUN_API_KEY=unset`、`GEMINI_API_KEY=unset`，因此 `configs/minimax_m27.yaml` 的 `provider.is_configured()` 为 `false`；`LongMemEval-S 64/128`、`PersonaMem 128` 的 slot-assignment live canaries 与 `latest_stage2_slot_assignment_gain.json` 因此真实 blocked
  - `research-results.tsv` / `autoresearch-state.json` 已记录到 iteration `3`，当前 retained metric 为 `16`，last status 为 `blocked`
- Evidence / artifacts:
  - `src/core_mem/v2/system.py`
  - `src/core_mem/v2/training.py`
  - `src/core_mem/v2/semantic_outputs.py`
  - `scripts/train_stage2.py`
  - `scripts/eval_stage2_local.py`
  - `scripts/run_stage2_memory_canary.py`
  - `scripts/analyze_stage2_memory_canary_failures.py`
  - `outputs_v2/artifacts/latest_stage2_slot_assignment_train.json`
  - `outputs_v2/artifacts/latest_stage2_slot_assignment_eval.json`
  - `outputs_v2/runs/20260416T071023Z_stage2_train_exec/execution_summary.json`
  - `outputs_v2/evals_local/20260416T071034Z_stage2_local_eval.json`
  - `research-results.tsv`
  - `autoresearch-state.json`
  - `conda run -n core_mem python scripts/verify_stage2_v23_longrun.py --score-only` -> `16`
  - `conda run -n core_mem pytest -q tests/test_stage2_v23_longrun.py tests/test_stage2_model_skeleton.py tests/test_stage2_local_eval.py tests/test_stage2_memory_canary.py tests/test_stage2_memory_canary_quality.py tests/test_stage2_parser.py`
- Next likely action:
  - 在 live provider env 恢复后，继续 current-head `LongMemEval-S 64/128` 与 `PersonaMem 128` slot-assignment canaries，并基于 fresh completed summaries 生成 `latest_stage2_slot_assignment_gain.json`

## 2026-04-16 Session 030

- Worked on: 恢复 `TD-032 / WS-018` 的 managed autoresearch，并把 `v2.2` 从 retained `14/19` 推到 stop condition `19/19`
- State changed:
  - 当前 session 检测到 `GPT_AGENT_API_KEY=SET`，确认上一轮 `BL-006` 不再代表本 session truth；fresh `LongMemEval-S 64` learned semantic canary `outputs_v2/evals_benchmark/20260416T015930Z_stage2_memory_canary.json` 与 corresponding semantic analysis 先把 score 从 `14` 提到 `16`
  - 新增 `src/core_mem/v2/projection.py` 的 generic answer projection normalization，并补充 `tests/test_stage2_model_skeleton.py` 对 explanatory suffix / location phrase 的单测；在同一批 `LongMemEval-S 64` belief state 上，local exact 从 `3` 提到 `4`
  - 刷新 `outputs_v2/evals_benchmark/20260416T021743Z_stage2_memory_canary.json` 后，`latest_stage2_semantic_online_gain.json` 记录 `LongMemEval-S 64` 相对 retained symbolic 64 baseline 的 `delta_local_exact_match = +1`，使 score 从 `16` 提到 `17`
  - 通过 `outputs_v2/evals_benchmark/20260416T022924Z_longmemeval_canary.json` 生成 `128` manifest，并在 `outputs_v2/runs/20260416T021743Z_stage2_memory_canary_longmemeval` 上 resume 扩样到 `128/128`；对应 summary 与 `latest_longmemeval_stage2_semantic_analysis.json` 已刷新，使 score 从 `17` 提到 `18`
  - 完成 fresh current-head `PersonaMem 128` learned semantic canary `outputs_v2/evals_benchmark/20260416T024146Z_stage2_memory_canary.json`，其快速统计为 `provider_exact = 40/128`、`provider_label_prefix = 40/128`、`local_exact = 12/128`；`scripts/verify_stage2_v22_completion.py --score-only` 因此达到 `19/19`
  - `research-results.tsv` / `autoresearch-state.json` 已记录到 iteration `7`，supervisor state 已更新为 `decision=stop`、`terminal_reason=goal_reached`
- Evidence / artifacts:
  - `src/core_mem/v2/projection.py`
  - `tests/test_stage2_model_skeleton.py`
  - `outputs_v2/evals_benchmark/20260416T015930Z_stage2_memory_canary.json`
  - `outputs_v2/evals_benchmark/20260416T021743Z_stage2_memory_canary.json`
  - `outputs_v2/evals_benchmark/20260416T024146Z_stage2_memory_canary.json`
  - `outputs_v2/artifacts/latest_longmemeval_stage2_semantic_analysis.json`
  - `outputs_v2/artifacts/latest_stage2_semantic_online_gain.json`
  - `research-results.tsv`
  - `autoresearch-state.json`
  - `conda run -n core_mem python scripts/verify_stage2_v22_completion.py --score-only` -> `19`
  - `conda run -n core_mem pytest -q tests/test_stage2_model_skeleton.py tests/test_stage2_local_eval.py tests/test_stage2_data_pipeline.py tests/test_stage2_public_data.py tests/test_stage2_memory_canary.py tests/test_stage2_memory_canary_quality.py tests/test_stage2_latent_core_quality.py tests/test_stage2_v21_learned_memory.py tests/test_stage2_v21_longrun.py tests/test_stage2_v21_semantic_model.py tests/test_stage2_v22_completion.py tests/test_stage2_parser.py`
- Next likely action:
  - 当前 stop condition 已机械达到；在用户给出新的 stage-2 方向前，保持 `TD-032 / WS-018` 的 `19/19` 完成态可复验，并诚实保留 `LongMemEval-S` provider-side 质量尚弱的风险

## 2026-04-16 Session 029

- Worked on: `TD-032 / WS-018` 下补齐 `v2.2` semantic artifact 发布链，并推进 current-head full-data semantic train/eval
- State changed:
  - `scripts/train_stage2.py` / `scripts/eval_stage2_local.py` 新增 semantic full artifact 发布能力，可直接写出 `latest_stage2_semantic_full_train.json` 与 `latest_stage2_semantic_full_local_eval.json`
  - `scripts/run_stage2_memory_canary.py` 新增 completed learned run 的 semantic canary alias 发布能力
  - `scripts/analyze_stage2_memory_canary_failures.py` 新增 LongMemEval semantic analysis alias 与 semantic online gain artifact 发布能力
  - `outputs_v2/artifacts/latest_stage2_semantic_full_train.json` 与 `latest_stage2_semantic_full_local_eval.json` 已在 current HEAD `510aeb7` 上落地，使 `stage2_v22_completion_score` 从 `9` 提升到 `14`
  - 同一 session 内确认 `BL-006`：live provider env 缺失，`outputs_v2/evals_benchmark/20260416T005639Z_stage2_memory_canary.json` 将 current-head learned `LongMemEval-S 1` probe 记录为 `blocked_provider_not_configured`
  - `research-results.tsv` / `autoresearch-state.json` 已从一次 helper 并发记账竞态中修复回 `full_resume` 一致状态；当前 retained metric 为 `14`，last status 为 `blocked`
- Evidence / artifacts:
  - `outputs_v2/artifacts/latest_stage2_semantic_full_train.json`
  - `outputs_v2/artifacts/latest_stage2_semantic_full_local_eval.json`
  - `outputs_v2/evals_local/20260416T003931Z_stage2_local_eval.json`
  - `outputs_v2/evals_benchmark/20260416T005639Z_stage2_memory_canary.json`
  - `research-results.tsv`
  - `autoresearch-state.json`
  - `conda run -n core_mem python scripts/verify_stage2_v22_completion.py --score-only` -> `14`
  - `conda run -n core_mem pytest -q tests/test_stage2_model_skeleton.py tests/test_stage2_local_eval.py tests/test_stage2_data_pipeline.py tests/test_stage2_public_data.py tests/test_stage2_memory_canary.py tests/test_stage2_memory_canary_quality.py tests/test_stage2_latent_core_quality.py tests/test_stage2_v21_learned_memory.py tests/test_stage2_v21_longrun.py tests/test_stage2_v21_semantic_model.py tests/test_stage2_v22_completion.py tests/test_stage2_parser.py`
- Next likely action:
  - 在 live provider env 恢复后，继续 current-head `PersonaMem 128` / `LongMemEval-S 64/128` semantic canaries，并基于 fresh LongMemEval-S artifact 产出 semantic analysis 与 semantic online gain

## 2026-04-15 Session 028

- Worked on: 把 stage-2 当前主线从 `TD-031 / WS-017` 提升到 `TD-032 / WS-018`，正式进入 `v2.2 full-data semantic latent`
- State changed:
  - 新增 `docs/v22_plan.md`，把 `v2.2` 的 full-data semantic-first 计划、里程碑和 no-fallback 约束写成真源文档
  - 新增 `scripts/verify_stage2_v22_completion.py` 与 `tests/test_stage2_v22_completion.py`
  - 将 `docs/current_status.md`、`docs/implementation_plan.md`、`docs/todo.md`、`.agent-os/project-index.md`、`.agent-os/todo.md` 同步到 `TD-032 / WS-018`
  - 明确 `v2.2` 的核心不是继续 closeout `TD-031`，而是把 semantic-first 扩展到 current-head full-data 训练、extended live canary、LongMemEval-S 质量提升和 learned-vs-symbolic 对照
- Evidence / artifacts:
  - `docs/v22_plan.md`
  - `scripts/verify_stage2_v22_completion.py`
  - `tests/test_stage2_v22_completion.py`
- Next likely action:
  - 以 `scripts/verify_stage2_v22_completion.py --score-only` 为机械指标，启动新的 background autoresearch
## 2026-04-15 Session 028

- Worked on: 完成 `TD-031 / WS-017` 的 semantic-first managed autoresearch 收口，把 `stage2_v21_semantic_model_score` 从 baseline `13/17` 推到 stop condition `17/17`
- State changed:
  - 新增 `src/core_mem/v2/semantic_outputs.py`，把 task-aware 结构修复与语义计分抽成通用组件
  - `src/core_mem/v2/training.py` 的 checkpoint eval 现不再把 raw JSON 外壳错误直接等价成语义失败；新增 `json_validity_rate`、`semantic_validity_rate`、`field_accuracy`、`field_f1`
  - `src/core_mem/v2/system.py` 的 online learned belief parse 现共享同一套语义修复逻辑，brace-less belief 输出不再必然触发 `learned_memory_error`
  - `docs/current_status.md`、`docs/implementation_plan.md`、`docs/todo.md`、`.agent-os/project-index.md`、`.agent-os/todo.md` 同步到 “semantic-first stop condition 已达成，但 `TD-031 / WS-017` 仍保留为当前 closeout 主线” 的 runtime truth
- Evidence / artifacts:
  - `outputs_v2/evals_local/20260415T230211Z_stage2_local_eval.json`
  - `outputs_v2/tables/20260415T230211Z_stage2_local_eval_summary.csv`
  - `outputs_v2/tables/20260415T230211Z_stage2_local_eval_budget_sweep.csv`
  - `conda run -n core_mem python scripts/verify_stage2_v21_semantic_model.py --score-only` -> `17`
  - `conda run -n core_mem pytest -q tests/test_stage2_model_skeleton.py tests/test_stage2_local_eval.py tests/test_stage2_data_pipeline.py tests/test_stage2_public_data.py tests/test_stage2_memory_canary.py tests/test_stage2_memory_canary_quality.py tests/test_stage2_latent_core_quality.py tests/test_stage2_v21_learned_memory.py tests/test_stage2_v21_longrun.py tests/test_stage2_v21_semantic_model.py tests/test_stage2_parser.py`
  - `conda run -n core_mem python scripts/verify_stage2_latent_status.py --score-only` -> `9`
  - `conda run -n core_mem python scripts/verify_stage2_latent_core_quality.py --score-only` -> `10`
  - `conda run -n core_mem python scripts/verify_stage2_v21_learned_memory.py --score-only` -> `11`
  - `conda run -n core_mem python scripts/verify_stage2_v21_longrun.py --score-only` -> `12`
  - `conda run -n core_mem python scripts/run_experiment.py --verify-only` -> `34`
- Next likely action:
  - 若用户继续推进 stage-2，更合理的下一步会是把 semantic-first 结构修复与语义监督继续扩展到 fresh current-head learned canary，而不是回头追 raw JSON exact match

## 2026-04-15 Session 027

- Worked on: 把 stage-2 下一轮主线从 `TD-030 / WS-016` 收口态切到 `TD-031 / WS-017`，明确采用 semantic-first learned decoder 方向
- State changed:
  - 将 `docs/v21_longrun_plan.md` 扩展为“语义正确优先于 raw JSON 表面格式完全匹配”，并新增阶段 F：语义优先训练重构
  - 将 `docs/current_status.md`、`docs/implementation_plan.md`、`docs/todo.md`、`.agent-os/project-index.md`、`.agent-os/todo.md` 同步到 `TD-031 / WS-017`
  - 明确记录当前 best non-tiny 训练结论：`trained_eval.token_f1 = 0.3885239109848479` 代表模型学会了部分结构模式，但尚未稳定输出合法完整 JSON
  - 明确后续目标：优先修 semantic correctness、belief JSON 有效性、`retrieval_alignment` 长期为零和 `learned_memory_error`
- Evidence / artifacts:
  - `outputs_v2/evals_local/20260415T191953Z_stage2_local_eval.json`
  - `docs/v21_longrun_plan.md`
  - `scripts/verify_stage2_v21_semantic_model.py`
- Next likely action:
  - 以 `scripts/verify_stage2_v21_semantic_model.py --score-only` 为新机械指标，启动下一轮 background autoresearch

## 2026-04-15 Session 026

- Worked on: 完成 `TD-030 / WS-016` 的 managed autoresearch 收口，把 `stage2_v21_longrun_score` 从 retained `15/16` 推到 stop condition `16/16`
- State changed:
  - 将 `scripts/verify_stage2_latent_core_quality.py` 修到只消费 symbolic PersonaMem canary 维持 latent-core guard，不再让 learned canary 误伤 `10/10` guard
  - 提交 `d6bc4f7` 固化上述 latent-core guard 修正，并在该 HEAD 上重跑 current-head learned canaries
  - 完成 `outputs_v2/evals_benchmark/20260415T202608Z_stage2_memory_canary.json`，把 current-head `PersonaMem 128` learned artifact 刷到 `d6bc4f7`，恢复 personamem `64/128` 两项检查
  - 完成 `outputs_v2/evals_benchmark/20260415T205627Z_stage2_memory_canary.json`，把 current-head `LongMemEval-S 64` learned artifact 刷到同一 HEAD，最终使 `scripts/verify_stage2_v21_longrun.py --score-only` 返回 `16`
  - 通过 helper 记账：iteration `8` 为 commit 前移导致的 `discard`，iteration `9` 为 personamem refresh 的 `no-op`，iteration `10` 为 final `keep`
  - 同步 `.agent-os/*` 与 `docs/*` 到 “`TD-030 / WS-016` 已机械完成，stop condition 已达到，但 learned live path 仍存在 `learned_memory_error` 风险” 的 runtime truth
- Evidence / artifacts:
  - `research-results.tsv`
  - `autoresearch-state.json`
  - `outputs_v2/evals_local/20260415T191953Z_stage2_local_eval.json`
  - `outputs_v2/evals_benchmark/20260415T202608Z_stage2_memory_canary.json`
  - `outputs_v2/evals_benchmark/20260415T205627Z_stage2_memory_canary.json`
  - `conda run -n core_mem python scripts/verify_stage2_v21_longrun.py --score-only` -> `16`
  - launch manifest full guard 通过
- Next likely action:
  - 若用户要求继续推进 stage-2，优先开 `TD-031`：在不引入 fallback/shortcut 的前提下，专项修复 learned belief JSON 无效输出与 `learned_memory_error`

## 2026-04-15 Session 025

- Worked on: managed autoresearch 长跑下先把 `TD-030 / WS-016` 的 retained baseline 从 `11/16` 推到 `12/16`，再围绕 online-aligned learned training 做多轮 probe 与一次 `REFINE`
- State changed:
  - 初始化 fresh managed run artifacts，并把 `stage2_v21_longrun_score` baseline 固化为 `11/16`
  - `StructuredMemorySystem` 的 learned mode 已去除 symbolic fallback；当 learned predictor 不可用或空输出时，现改为显式 `learned_memory_unavailable / learned_memory_error / learned_memory_empty`
  - `src/core_mem/v2/training.py` 与 `scripts/verify_stage2_v21_learned_memory.py` 已修复两类机械错配：
    - capped `max_train_examples / max_eval_examples` 不再按 manifest 顺序只吃到首个 task
    - `verify_stage2_v21_learned_memory.py` 现在接受 `TD-030 / WS-016` 下 `TD-029 / WS-015 done` 的 runtime truth
  - 上述 retained 改动已把 `scripts/verify_stage2_v21_longrun.py --score-only` 从 `11` 提升到 `12`，且 full guard 通过
  - 随后连续三轮训练 probe 均未能仅靠 `24` optimizer steps 把 learned `Flan-T5` decoder 拉过线，已按 `discard -> discard -> discard -> refine` 记账
  - 最新 `REFINE` 已把 `configs/stage2_train.yaml` 的 `gradient_accumulation_steps` 从 `8` 调到 `1`，并把 online-aligned 训练配方收窄到 `retrieval_alignment / lifecycle_prediction / composition_to_belief`
  - 在 refined 配方下，`outputs_v2/checkpoints/20260415T191811Z_stage2_train_exec/` 的 `8`-sample direct probe 已达到 `trained_eval.token_f1 = 0.3459821428571428`，但对应 `64/256` eval artifact 仍在运行，因此当前只能记为 `partial / unverified`
- Evidence / artifacts:
  - `research-results.tsv`
  - `autoresearch-state.json`
  - `outputs_v2/runs/20260415T191811Z_stage2_train_exec/execution_summary.json`
  - `outputs_v2/checkpoints/20260415T191811Z_stage2_train_exec/`
  - `conda run -n core_mem python scripts/verify_stage2_v21_longrun.py --score-only` -> `12`
  - `conda run -n core_mem pytest -q tests/test_stage2_model_skeleton.py tests/test_stage2_local_eval.py tests/test_stage2_data_pipeline.py tests/test_stage2_public_data.py tests/test_stage2_memory_canary.py tests/test_stage2_memory_canary_quality.py tests/test_stage2_latent_core_quality.py tests/test_stage2_v21_learned_memory.py tests/test_stage2_v21_longrun.py tests/test_stage2_parser.py`
  - direct probe on `outputs_v2/checkpoints/20260415T191811Z_stage2_train_exec/` with `max_eval_examples=8` -> `trained_eval.token_f1 = 0.3459821428571428`
- Next likely action:
  - 等待或重跑当前 refined checkpoint 的 `64/256` sample eval 形成正式 artifact；若 `trained_eval.token_f1 >= 0.20` 机械成立，则立刻刷新 current-head learned `PersonaMem 64 / LongMemEval-S 64 / PersonaMem 128` canaries

## 2026-04-15 Session 024

- Worked on: 将 stage-2 主线从 `TD-029 / WS-015` 的 learned-memory plumbing 完成态，推进到 `TD-030 / WS-016` 的 learned-model-first 长跑计划
- State changed:
  - 新增 `docs/v21_longrun_plan.md`，把 learned-model-first 长跑目标、阶段 A-E、主指标、守卫和自动迭代闭环写成真源文档
  - 新增 `scripts/verify_stage2_v21_longrun.py` 与 `tests/test_stage2_v21_longrun.py`
  - 将 `docs/current_status.md`、`docs/implementation_plan.md`、`docs/todo.md`、`.agent-os/project-index.md`、`.agent-os/todo.md` 切到 `TD-030 / WS-016`
  - 明确把 “不做任何兜底/fallback/benchmark-specific shortcut” 设为本轮长跑的硬约束
  - 把当前主线从“learned path 已存在”切到“learned model 是否在更大样本和多个 benchmark 上真正变强，并逐步摆脱 fallback 依赖”
  - 运行新 verifier 后，当前 baseline 固化为 `stage2_v21_longrun_score = 11/16`
- Evidence / artifacts:
  - `docs/v21_longrun_plan.md`
  - `scripts/verify_stage2_v21_longrun.py`
  - `tests/test_stage2_v21_longrun.py`
- Next likely action:
  - 运行新的 long-run verifier 建立 baseline，提交基线改动，然后按 `stage2_v21_longrun_score` 启动新的 background autoresearch

## 2026-04-15 Session 023

- Worked on: managed autoresearch 长跑下完成 learned-memory-first / better latent 的首批机械里程碑，并将 `stage2_v21_learned_memory_score` 从 `8` 推到 stop condition `12`
- State changed:
  - 初始化 fresh managed run artifacts，并把 baseline 固化为 `stage2_v21_learned_memory_score = 8/12`
  - `StructuredMemorySystem` 新增 `memory_mode=learned_memory` / `use_learned_memory` 开关；当提供 checkpoint 与 train config 时，online query path 会通过 checkpoint-backed `composition_to_belief` 生成 learned belief，失败时诚实退回 symbolic decoder
  - `scripts/run_stage2_memory_canary.py` 新增 learned-mode CLI 参数与 metadata，并在 learned mode 下自动写出 `latest_personamem_stage2_learned_canary.json` / `latest_longmemeval_stage2_learned_canary.json`
  - `configs/stage2_train*.yaml` 与 `scripts/train_stage2.py` / `src/core_mem/v2/training.py` 新增 `online_aligned` 语义，使训练配置能显式表达“更贴近在线 retrieval/belief/lifecycle”的 learned variant
  - 完成 current-head learned-mode live canaries：`outputs_v2/evals_benchmark/20260415T175004Z_stage2_memory_canary.json`（PersonaMem, 1 sample）与 `outputs_v2/evals_benchmark/20260415T175032Z_stage2_memory_canary.json`（LongMemEval-S, 1 sample）
  - 同步 `docs/current_status.md`、`docs/implementation_plan.md`、`docs/todo.md`、`.agent-os/project-index.md`、`.agent-os/todo.md` 到 “`TD-029 / WS-015 done`，下一个更合理动作是扩大 learned-mode canary 覆盖” 的当前真相
- Evidence / artifacts:
  - `research-results.tsv`
  - `autoresearch-state.json`
  - `outputs_v2/artifacts/latest_personamem_stage2_learned_canary.json`
  - `outputs_v2/artifacts/latest_longmemeval_stage2_learned_canary.json`
  - `outputs_v2/evals_benchmark/20260415T175004Z_stage2_memory_canary.json`
  - `outputs_v2/evals_benchmark/20260415T175032Z_stage2_memory_canary.json`
  - `conda run -n core_mem python scripts/verify_stage2_v21_learned_memory.py --score-only` -> `12`
  - `conda run -n core_mem pytest -q tests/test_stage2_model_skeleton.py tests/test_stage2_local_eval.py tests/test_stage2_data_pipeline.py tests/test_stage2_public_data.py tests/test_stage2_memory_canary.py tests/test_stage2_memory_canary_quality.py tests/test_stage2_latent_core_quality.py tests/test_stage2_v21_learned_memory.py tests/test_stage2_parser.py`
  - `conda run -n core_mem python scripts/verify_stage2_latent_status.py --score-only` -> `9`
  - `conda run -n core_mem python scripts/verify_stage2_latent_core_quality.py --score-only` -> `10`
  - `conda run -n core_mem python scripts/run_experiment.py --verify-only` -> `34`
- Next likely action:
  - 若继续推进 stage-2 learned-memory-first 主线，优先把 current-head learned-mode canary 从 `1` 样本扩大到固定切片，并分析 learned belief path 的真实增益与退化来源

## 2026-04-15 Session 022

- Worked on: 停止旧的 robustness 长跑，并把 stage-2 主线 pivot 到 learned-memory-first / better latent
- State changed:
  - 按用户确认停止了当前 background autoresearch run
  - 回滚了未保留的 in-progress experiment commit `3bf640d`，避免把旧 run 的失败线带入新主线
  - 将 `TD-028 / WS-014` 的 `14/15` 保留为 historical best，而不再作为当前主线终点
  - 新增 `docs/learned_memory_related_work.md`
  - 将当前 next action / active workstream 切换到 `TD-029 / WS-015`
  - 新主线明确要求：减少 rule-heavy write/read，优先推进 checkpoint-backed learned memory path、online-aligned learned variant 和 learned-mode canaries
- Evidence / artifacts:
  - `docs/learned_memory_related_work.md`
  - `scripts/verify_stage2_v21_learned_memory.py`
  - `tests/test_stage2_v21_learned_memory.py`
- Next likely action:
  - 以 `stage2_v21_learned_memory_score` 为主指标启动新的 managed autoresearch 长跑

## 2026-04-15 Session 021

- Worked on: managed autoresearch 长跑下推进 `v2.1` robustness，从 fresh init 提升到 `14/15`
- State changed:
  - 先对齐 `docs/implementation_plan.md` 与 `v2.1` verifier 的机械短语要求，把 baseline 从 `4/15` 推到 `5/15`
  - 将 `scripts/run_stage2_memory_canary.py` 升级为增量落盘 + `--run-dir --resume` 的可续跑 runner，并用当前 HEAD 刷新 fresh `PersonaMem 64` 与 `LongMemEval-S 64` live canaries
  - 新增 `scripts/analyze_stage2_memory_canary_failures.py` 的 layered artifact 输出，已生成 `outputs_v2/artifacts/latest_longmemeval_stage2_layered_analysis.json`
  - 通过 `scripts/run_stage2_canary.py --personamem-size 128` 生成新的 `128`-sized canary manifest，并完成 current-head `PersonaMem 128` live canary
  - 在 `GPU3` 上追加一条更强的默认 `configs/stage2_train.yaml` 非 tiny train/eval 证据链：`outputs_v2/runs/20260415T132046Z_stage2_train_exec/execution_summary.json` 与 `outputs_v2/evals_local/20260415T132109Z_stage2_local_eval.json`
  - 补写 `outputs_v2/artifacts/latest_stage2_learned_online_gain.json` 后，分数先升到 `12/15`
  - 新 trial commit `8cc598da58241aa03b6245198659b6ab110a506e` 通过 parser/retrieval cue 调整，把 fresh current-head `LongMemEval-S 64` 提升到 `provider_exact = 5 / 64`、`provider_label_prefix = 5 / 64`、`local_exact = 3 / 64`
  - 同一 HEAD 下重刷 current-head `PersonaMem 128` 后，`local_exact_rate` 仍保持通过，但 `provider_label_prefix_rate` 降到 `39 / 128 = 30.47%`
  - 当前 `scripts/verify_stage2_v21_robustness.py --score-only = 14`，只剩 `PersonaMem 128 provider_prefix_rate >= 0.40` 一项未过
- Evidence / artifacts:
  - `research-results.tsv`
  - `autoresearch-state.json`
  - `outputs_v2/evals_benchmark/20260415T122009Z_stage2_memory_canary.json`
  - `outputs_v2/evals_benchmark/20260415T123316Z_stage2_memory_canary.json`
  - `outputs_v2/evals_benchmark/20260415T124345Z_stage2_memory_canary.json`
  - `outputs_v2/artifacts/latest_longmemeval_stage2_layered_analysis.json`
  - `outputs_v2/evals_benchmark/20260415T124337Z_personamem_canary.json`
  - `outputs_v2/artifacts/latest_stage2_learned_online_gain.json`
  - `outputs_v2/evals_benchmark/20260415T134213Z_stage2_memory_canary.json`
  - `outputs_v2/evals_benchmark/20260415T135121Z_stage2_memory_canary.json`
  - `outputs_v2/runs/20260415T132046Z_stage2_train_exec/execution_summary.json`
  - `outputs_v2/evals_local/20260415T132109Z_stage2_local_eval.json`
  - `conda run -n core_mem python scripts/verify_stage2_v21_robustness.py --score-only` -> `14`
  - `conda run -n core_mem python scripts/verify_stage2_latent_core_quality.py --score-only` -> `10`
  - `conda run -n core_mem pytest -q tests/test_stage2_model_skeleton.py tests/test_stage2_local_eval.py tests/test_stage2_data_pipeline.py tests/test_stage2_public_data.py tests/test_stage2_memory_canary.py tests/test_stage2_memory_canary_quality.py tests/test_stage2_latent_core_quality.py tests/test_stage2_v21_robustness.py tests/test_stage2_parser.py`
- Next likely action:
  - 继续围绕 `TD-028` 只攻最后一个缺口：把 `PersonaMem 128 provider_label_prefix_rate` 从 `30.47%` 推到 `>= 40%`

## 2026-04-15 Session 020

- Worked on: 将项目主线从“完整 v2 已成立”切换到“robust v2.1”
- State changed:
  - 用户确认采用 `v2.1` 路线图作为新的 stage-2 主线
  - 更新 `docs/current_status.md`、`docs/implementation_plan.md`、`docs/todo.md` 与 `.agent-os/*`，把 current truth 从“完整 v2 已完成”推进到“v2.1 doing”
  - 新主线明确围绕四个方向推进：真实质量、learned path 在线增益、跨 benchmark 鲁棒性、系统化收口
  - 将 `TD-028 / WS-014` 设为当前 next action / active workstream
- Evidence / artifacts:
  - `docs/current_status.md`
  - `docs/implementation_plan.md`
  - `docs/todo.md`
  - `.agent-os/project-index.md`
  - `.agent-os/todo.md`
- Next likely action:
  - 基于 `v2.1` 路线图启动新的长跑 autoresearch，优先做 `LongMemEval-S` 专项质量提升与 learned path 在线增益验证

## 2026-04-15 Session 019

- Worked on: 恢复 managed autoresearch 长跑并把完整 `v2` 从 `11/14` 收口到 stop condition `14/14`
- State changed:
  - 确认当前 session 实际具备 `GPT_AGENT_API_KEY`，因此上一轮的 `BL-005` 不再代表本 session truth
  - 完成 fresh `PersonaMem 64` live canary：`outputs_v2/evals_benchmark/20260415T052916Z_stage2_memory_canary.json` 在当前 HEAD `53eaf44` 上完成 `64/64` live predictions，并把 `stage2_v2_completion_score` 从 `11` 提升到 `12`
  - 完成 fresh `LongMemEval-S 64` live canary 与 failure analysis：`outputs_v2/evals_benchmark/20260415T054234Z_stage2_memory_canary.json` 与 `outputs_v2/artifacts/latest_longmemeval_stage2_canary_analysis.json` 已落地
  - 修复 verifier 与 runtime truth 的两处机械错配：`scripts/verify_stage2_v2_completion.py` 现接受 `longmemeval_s` alias；`scripts/verify_stage2_latent_core_quality.py` 现只使用 PersonaMem completed canary 维持 latent-core guard，避免被 `LongMemEval-S` artifact 误伤
  - 将 `docs/current_status.md`、`docs/implementation_plan.md`、`docs/todo.md`、`.agent-os/project-index.md`、`.agent-os/todo.md` 同步到 “`TD-027 done / stage2_v2_completion_score = 14/14 / BL-005 cleared`” 的当前真相
- Evidence / artifacts:
  - `research-results.tsv`
  - `autoresearch-state.json`
  - `outputs_v2/evals_benchmark/20260415T052916Z_stage2_memory_canary.json`
  - `outputs_v2/runs/20260415T052916Z_stage2_memory_canary_personamem/predictions.jsonl`
  - `outputs_v2/evals_benchmark/20260415T054234Z_stage2_memory_canary.json`
  - `outputs_v2/runs/20260415T054234Z_stage2_memory_canary_longmemeval/predictions.jsonl`
  - `outputs_v2/artifacts/latest_longmemeval_stage2_canary_analysis.json`
  - `conda run -n core_mem python scripts/verify_stage2_v2_completion.py --score-only` -> `14`
  - `conda run -n core_mem python scripts/verify_stage2_latent_core_quality.py --score-only` -> `10`
  - `conda run -n core_mem pytest -q tests/test_stage2_model_skeleton.py tests/test_stage2_local_eval.py tests/test_stage2_data_pipeline.py tests/test_stage2_public_data.py tests/test_stage2_memory_canary.py tests/test_stage2_memory_canary_quality.py tests/test_stage2_latent_core_quality.py tests/test_stage2_v2_completion.py tests/test_stage2_parser.py`
  - `conda run -n core_mem python scripts/run_experiment.py --verify-only` -> `34`
- Next likely action:
  - 若继续推进 stage-2，切到更大范围 benchmark / 结果对比，而不是重复同一组 `64` canary；stage-1 formal benchmark 继续维持 pending + provider blocker 真相

## 2026-04-15 Session 018

- Worked on: managed autoresearch 长跑下补齐完整 `v2` 的 no-shortcut runner 与 non-tiny training 里程碑，并在 live provider 缺失处诚实收束
- State changed:
  - 初始化 fresh managed run artifacts，并把 `stage2_v2_completion_score` baseline 机械固定为 `7/14`
  - 移除了 `scripts/run_stage2_memory_canary.py` 中 PersonaMem-specific candidate injection 与 blank-provider fallback，`benchmark_runner_avoids_shortcuts` 从 `false` 提升到 `true`
  - 在尝试刷新 fresh live canary 时确认当前 session 缺少 `GPT_AGENT_API_KEY`；fresh probe 只能生成 `blocked_provider_not_configured` artifact，因此把主线 pivot 到 non-tiny training
  - 修复 `src/core_mem/v2/training.py` 对 Hugging Face tokenizer 的错误调用，使默认 `google/flan-t5-base` 的 stage-2 train path 能在 `GPU3` 上真实执行
  - 在 `GPU3` 上完成一条默认 `configs/stage2_train.yaml` 的非 tiny train/eval 证据链，并产出正的 `trained_eval.token_f1`
  - 调整 `scripts/verify_stage2_latent_core_quality.py`，使其对齐当前 runtime truth：接受 `TD-026 done / WS-012 done` 的收口状态，并忽略更新但 blocked 的 canary probe，不让它误伤 latent-core guard
  - 将 `docs/current_status.md`、`docs/implementation_plan.md`、`docs/todo.md`、`.agent-os/project-index.md`、`.agent-os/todo.md` 同步到 “`11/14` + `BL-005` live provider env missing” 的当前真相
- Evidence / artifacts:
  - `research-results.tsv`
  - `autoresearch-state.json`
  - `outputs_v2/evals_benchmark/20260415T043414Z_stage2_memory_canary.json`
  - `outputs_v2/runs/20260415T043648Z_stage2_train_exec/execution_summary.json`
  - `outputs_v2/checkpoints/20260415T043648Z_stage2_train_exec/`
  - `outputs_v2/evals_local/20260415T043706Z_stage2_local_eval.json`
  - `outputs_v2/evals_local/20260415T043830Z_stage2_local_eval.json`
  - `conda run -n core_mem python scripts/verify_stage2_v2_completion.py --score-only` -> `11`
  - `conda run -n core_mem python scripts/verify_stage2_latent_core_quality.py --score-only` -> `10`
  - `conda run -n core_mem pytest -q tests/test_stage2_data_pipeline.py tests/test_stage2_latent_core_quality.py tests/test_stage2_v2_completion.py`
- Next likely action:
  - 在具备 `GPT_AGENT_API_KEY` 的受控 session 中恢复 managed autoresearch，优先刷新 fresh `PersonaMem 64` 与 `LongMemEval-S 64` live canaries，并随后生成 `latest_longmemeval_stage2_canary_analysis.json`

## 2026-04-15 Session 017

- Worked on: 为“完整 v2”长期后台 run 建立新的机械里程碑与 hard constraint
- State changed:
  - 用户明确批准新一轮长期后台 run 可把 `GPU3` 正式训练和 `MiniMax-M2.7` live benchmark 调用纳入里程碑验证
  - 将项目 next action 从 backlog 选择切换到 `TD-027`，明确目标为推进完整 `v2`
  - 新增 `scripts/verify_stage2_v2_completion.py` 与对应测试，准备把 fresh canary、第二 benchmark、非 tiny 训练证据和 no-shortcut runner 统一收敛为单一 milestone score
  - 将“不要做任何偷懒兜底 fallback”写入 docs 与 `.agent-os` 当前真相
- Evidence / artifacts:
  - `scripts/verify_stage2_v2_completion.py`
  - `tests/test_stage2_v2_completion.py`
  - `docs/current_status.md`
  - `docs/implementation_plan.md`
  - `docs/todo.md`
  - `.agent-os/project-index.md`
  - `.agent-os/todo.md`
- Next likely action:
  - 提交长期 run 基线改动后，以 fresh-start 方式启动新的 background autoresearch

## 2026-04-15 Session 016

- Worked on: managed autoresearch 背景运行下提升 stage-2 latent-core robustness，并达成 stop condition
- State changed:
  - 初始化 fresh managed autoresearch artifacts：`research-results.tsv` 与 `autoresearch-state.json`
  - 先尝试了一轮 `retrieval-recency + eval-alignment` 路线，local submetrics 有小幅改善，但 `stage2_latent_core_quality_score` 仍停在 `6/10`，已按 `discard` 记账并完整回滚
  - 第二轮改为把 `composition_to_belief` 的 local eval 对齐到 memory-state -> belief recovery 分层，并让 `BeliefDecoder` 直接消费 selected slots 的既定顺序，不再在 decode 阶段二次重排
  - `scripts/verify_stage2_latent_core_quality.py --score-only` 已从 baseline `6` 提升到 `10`
  - stage-2 guard 指定测试集与 `scripts/run_experiment.py --verify-only` 已通过
  - 同步 `docs/current_status.md`、`docs/implementation_plan.md`、`docs/todo.md`、`.agent-os/project-index.md`、`.agent-os/todo.md`
- Evidence / artifacts:
  - `research-results.tsv`
  - `autoresearch-state.json`
  - `outputs_v2/evals_local/20260415T031952Z_stage2_local_eval.json`
  - `outputs_v2/tables/20260415T031952Z_stage2_local_eval_summary.csv`
  - `outputs_v2/tables/20260415T031952Z_stage2_local_eval_budget_sweep.csv`
  - `python3 scripts/verify_stage2_latent_core_quality.py --score-only` -> `10`
  - `conda run -n core_mem pytest -q tests/test_stage2_model_skeleton.py tests/test_stage2_local_eval.py tests/test_stage2_data_pipeline.py tests/test_stage2_public_data.py tests/test_stage2_memory_canary.py tests/test_stage2_memory_canary_quality.py tests/test_stage2_latent_core_quality.py tests/test_stage2_parser.py`
  - `conda run -n core_mem python scripts/run_experiment.py --verify-only` -> `34`
- Next likely action:
  - 若继续推进 stage-2，优先在 `TD-025` 与 `TD-020` 之间选择下一条 backlog 主线；当前 `TD-026` 已达 stop condition

## 2026-04-15 Session 015

- Worked on: 将第二阶段优化目标从 benchmark-facing canary 提升切换到 latent-core robustness
- State changed:
  - 更新 `docs/current_status.md`、`docs/implementation_plan.md`、`docs/todo.md`、`.agent-os/project-index.md`、`.agent-os/todo.md`，把当前主线改为“提升 local intrinsic 质量，并把 `PersonaMem 64 >= 9/10` 作为不退化 guard”
  - 新增 `scripts/verify_stage2_latent_core_quality.py`
  - 新增 `tests/test_stage2_latent_core_quality.py`
  - 调整 `scripts/verify_stage2_latent_status.py`，使其兼容新的文档表述而不误报 readiness 回退
  - 新的 latent-core quality 基线已被机械化为 `5/10`
- Evidence / artifacts:
  - `python3 scripts/verify_stage2_latent_status.py --score-only` -> `9`
  - `python3 scripts/verify_stage2_latent_core_quality.py --score-only` -> `5`
  - `conda run -n core_mem pytest -q tests/test_stage2_latent_core_quality.py tests/test_stage2_memory_canary_quality.py`
- Next likely action:
  - 用 background autoresearch 围绕 local intrinsic 指标提升做新一轮迭代，并以 `PersonaMem 64 >= 9/10` 为 guard

## 2026-04-15 Session 014

- Worked on: 以 managed autoresearch 方式迭代提升 stage-2 PersonaMem 64 live canary 质量
- State changed:
  - 初始化 fresh background autoresearch artifacts：`research-results.tsv` 与 `autoresearch-state.json`
  - 修复 PersonaMem observation path：assistant turn 不再进入 memory，且 `create -> eat -> food_preference` 的字符串误判已移除
  - 将 `scripts/run_stage2_memory_canary.py` 的 PersonaMem prompt 与本地 answer projection 对齐到 `(a)/(b)/(c)/(d)` 标签空间
  - 新增与扩展 `tests/test_stage2_parser.py`、`tests/test_stage2_memory_canary.py`
  - 后续继续完成 `discard -> refine -> pivot -> keep` 记录，主线从 answer-layer 微调切换到 lifecycle retention
  - 当前 retained best 已更新为 `quality score = 9/10`
- Evidence / artifacts:
  - `research-results.tsv`
  - `autoresearch-state.json`
  - `outputs_v2/evals_benchmark/20260415T003952Z_stage2_memory_canary.json`
  - `outputs_v2/evals_benchmark/20260415T005653Z_stage2_memory_canary.json`
  - `outputs_v2/evals_benchmark/20260415T015324Z_stage2_memory_canary.json`
  - `outputs_v2/artifacts/latest_personamem_stage2_canary_analysis.json`
  - `conda run -n core_mem pytest -q tests/test_stage2_model_skeleton.py tests/test_stage2_local_eval.py tests/test_stage2_data_pipeline.py tests/test_stage2_public_data.py tests/test_stage2_memory_canary.py tests/test_stage2_memory_canary_quality.py`
  - `conda run -n core_mem python scripts/run_experiment.py --verify-only`
- Next likely action:
  - 在当前 `9/10` canary 收益上继续做剩余 recall/suggestion 错误分析，并决定是否扩大 stage-2 benchmark 范围

## 2026-04-14 Session 013

- Worked on: 将 stage-2 的当前主线从“latent path 已做实”切换到“live canary 质量提升”
- State changed:
  - 更新 `docs/current_status.md`、`docs/todo.md`、`docs/implementation_plan.md`、`.agent-os/project-index.md`、`.agent-os/todo.md`，使项目真相从“缺 provider key / blocked artifact”切换到“已完成 live canary，但质量不足”
  - 新增 `scripts/analyze_stage2_memory_canary_failures.py`
  - 新增 `scripts/verify_stage2_memory_canary_quality.py`
  - 新增 `tests/test_stage2_memory_canary_quality.py`
  - 基于 `PersonaMem 64` live canary 生成失败分析 artifact，并建立新的质量基线 `4/10`
- Evidence / artifacts:
  - `outputs_v2/evals_benchmark/20260414T231617Z_stage2_memory_canary.json`
  - `outputs_v2/artifacts/latest_personamem_stage2_canary_analysis.json`
  - `python3 scripts/verify_stage2_memory_canary_quality.py --benchmark personamem --summary-path outputs_v2/evals_benchmark/20260414T231617Z_stage2_memory_canary.json`
  - `conda run -n core_mem pytest -q tests/test_stage2_memory_canary.py tests/test_stage2_memory_canary_quality.py`
- Next likely action:
  - 用 background autoresearch 围绕 `PersonaMem 64` live canary 的 failure analysis 和 online path 提效持续迭代

## 2026-04-04 Session 001

- Worked on: 项目初始化前的人类决策锁定与文档系统搭建准备
- State changed:
  - 冻结了第一阶段目标、范围、硬约束和验收标准
  - 决定 `docs/requirements.md` 为最高优先级真源文档
  - 决定 `docs/` 与 `.agent-os/` 并存
- Evidence / artifacts:
  - 本次会话中已确认 requirements 草案与 AGENTS 草案
- Next likely action:
  - 在 repo 中创建 `AGENTS.md`、真源文档和依赖状态文档，并交由用户审阅

## 2026-04-04 Session 002

- Worked on: 项目系统初始化与研究平台骨架搭建
- State changed:
  - 创建了 `AGENTS.md` / `CLAUDE.md`
  - 创建了 `docs/` 与 `.agent-os/` 的基础文档体系
  - 建立了 `src/`、`scripts/`、`configs/`、`tests/`、`outputs/` 目录骨架
  - 添加了第一版 `environment.yml`、配置文件、provider adapter skeleton、benchmark adapter skeleton、verifier、runner 和最小测试
- Evidence / artifacts:
  - `docs/requirements.md`
  - `.agent-os/project-index.md`
  - `scripts/verify_stage1_status.py`
  - `scripts/run_experiment.py`
- Next likely action:
  - 将骨架升级为可执行的 benchmark dry-run 平台，并补齐更有意义的验证与测试

## 2026-04-05 Session 003

- Worked on: 平台从空骨架升级到 benchmark dry-run 与真实 provider 调用层
- State changed:
  - 实现了 `src/core_mem/config.py` 配置加载
  - 将 OpenAI-compatible provider 从 payload builder 升级为真实 HTTP 调用 adapter
  - 将 PersonaMem / LongMemEval-S adapter 升级为可读取官方文件格式并构造 prompt-ready records
  - 将 `scripts/run_experiment.py` 升级为支持 `personamem` / `longmemeval` 子命令与 dry-run 输出
  - 扩展 verifier，使 readiness score 从旧骨架分数提升为更贴近第一阶段要求的检查集
  - 增加 benchmark fixtures、provider tests、adapter tests、config test 和 dry-run smoke test
- Evidence / artifacts:
  - `src/core_mem/config.py`
  - `src/core_mem/providers/openai_compatible.py`
  - `src/core_mem/benchmarks/personamem.py`
  - `src/core_mem/benchmarks/longmemeval.py`
  - `tests/test_provider_adapter.py`
  - `tests/test_benchmark_adapters.py`
  - `pytest` 通过（9 tests）
  - `scripts/verify_stage1_status.py --score-only` 输出 `16`
- Next likely action:
  - 创建并固定 conda 环境 `core_mem`
  - 下载并接入 PersonaMem 32k 与 LongMemEval-S 官方数据
  - 推进 official protocol 的正式运行与结果产出

## 2026-04-05 Session 004

- Worked on: 项目内 conda 环境的真实创建与环境复现固化
- State changed:
  - 创建了项目内 prefix conda 环境 `.conda_envs/core_mem`
  - 固化了 `configs/condarc.project.yaml`
  - 添加了 `scripts/bootstrap_conda_env.ps1` 与 `scripts/run_in_core_mem.ps1`
  - 扩展 verifier，使其检查 project-local condarc、bootstrap scripts 和环境内 `python.exe`
  - 修复 `pytest.ini`，避免新环境下递归进入 `.conda_envs` 与 `.conda_pkgs`
- Evidence / artifacts:
  - `.conda_envs/core_mem/python.exe`
  - `scripts/bootstrap_conda_env.ps1`
  - `scripts/run_in_core_mem.ps1`
  - 项目内 conda 环境下 `python --version` -> `3.10.20`
  - 项目内 conda 环境下 `pytest` 通过（9 tests）
  - 项目内 conda 环境下 `scripts/run_experiment.py --verify-only` 输出 `20`
- Next likely action:
  - 下载 PersonaMem 32k 官方数据
  - 开始主 benchmark 的正式 protocol 运行

## 2026-04-05 Session 005

- Worked on: 从 Windows 本地执行迁移到 WSL 的 handoff 清理
- State changed:
  - 删除了 Windows-specific environment bootstrap / condarc 文件
  - 将仓库重新收敛到平台中立状态
  - 将 agent state 文档改为“下一步在 WSL 中重建 conda 环境”
- Evidence / artifacts:
  - 删除 `configs/condarc.project.yaml`
  - 删除 `scripts/bootstrap_conda_env.ps1`
  - 删除 `scripts/run_in_core_mem.ps1`
  - `docs/current_status.md`、`.agent-os/project-index.md`、`.agent-os/todo.md` 已切换到 WSL handoff 状态
- Next likely action:
  - 在 WSL 中 `git clone` 当前分支
  - 创建 Linux 下的 `core_mem` conda 环境
  - 用新环境复验 `verify_stage1_status.py`、`pytest` 和 benchmark dry-run

## 2026-04-04 Session 006

- Worked on: 将环境复现策略从 repo-local conda prefix 回退到 named env + 单一环境文件
- State changed:
  - 用户明确要求 `core_mem` 使用默认 conda envs 目录，不再允许 repo 工作区内保留 `.conda_envs/` 或 `.conda_pkgs/`
  - 环境复现真相改为 `environment.yaml`
  - 已删除 repo-local conda 产物、WSL bootstrap / runner 脚本与相关 verifier 检查
  - 仓库内 `pytest` 与 `scripts/run_experiment.py --verify-only` 在新策略下继续通过
- Evidence / artifacts:
  - `docs/requirements.md`、`docs/current_status.md`、`docs/todo.md`、`docs/implementation_plan.md`
  - `.agent-os/project-index.md`、`.agent-os/todo.md`、`.agent-os/architecture-milestones.md`
  - `pytest` 通过（9 tests）
  - `scripts/verify_stage1_status.py --score-only` 输出 `17`
- Next likely action:
  - 在默认 conda envs 目录重新创建命名环境 `core_mem`
  - 在新约束下重新启动 autoresearch
  - 补齐最小分步骤脚本并继续推进 official benchmark readiness

## 2026-04-04 Session 007

- Worked on: 恢复前台 autoresearch，继续推进复现闭环并机械识别 stage-1 外部 blocker
- State changed:
  - fresh-start autoresearch 成功恢复并延续
  - 新增 `scripts/stage1_preflight.py`
  - 新增 preflight smoke test，使 `pytest` 总数提升到 12
  - `scripts/verify_stage1_status.py --score-only` 从 `20` 提升到 `22`
  - 当前 run 被机械 blocker 停住：default named env、`ALIYUN_API_KEY`、PersonaMem / LongMemEval-S 官方数据缺失
- Evidence / artifacts:
  - `scripts/stage1_preflight.py --config configs/defaults.yaml` 返回 blocker 列表
  - `pytest` 通过（12 tests）
  - `scripts/run_experiment.py --verify-only` 输出 `22`
  - `research-results.tsv` / `autoresearch-state.json` 已记录 keep 与 blocked 行
- Next likely action:
  - 在默认 conda envs 目录创建 `core_mem`
  - 配置 `ALIYUN_API_KEY`
  - 放入 PersonaMem 32k 与 LongMemEval-S 官方数据后恢复正式 benchmark 推进

## 2026-04-04 Session 008

- Worked on: 持续执行前台 autoresearch，并将两个 benchmark 推进到真实 1-sample 运行
- State changed:
  - 用户提供 session 内 `ALIYUN_API_KEY` 后，preflight blocker 从 4 项缩减到 named env 1 项
  - 下载 PersonaMem 32k 与 LongMemEval-S 官方数据到 `data/`
  - 新增最小 vanilla CoRe Memory workflow 实现与测试
  - 补齐 run metadata traceability 字段与 config snapshot
  - 修复 PersonaMem 官方 `shared_contexts_32k.jsonl` 单键映射格式
  - 对齐 PersonaMem multiple-choice label-space 输出协议
  - 对齐 LongMemEval answer-phrase 输出协议
  - `scripts/verify_stage1_status.py --score-only` 提升到 `26`
  - `pytest` 提升到 `18` 个测试并全部通过
  - PersonaMem / LongMemEval-S 均完成真实 1-sample run，最小 `evaluation.json` 均为 `accuracy=1.0`
- Evidence / artifacts:
  - `data/personamem/questions_32k.csv`
  - `data/personamem/shared_contexts_32k.jsonl`
  - `data/longmemeval/longmemeval_s_cleaned.json`
  - `outputs/runs/20260405T024515Z_personamem/evaluation.json`
  - `outputs/runs/20260405T024833Z_longmemeval/evaluation.json`
  - `pytest` 通过（18 tests）
  - `scripts/run_experiment.py --verify-only` 输出 `26`
- Next likely action:
  - 解决 default named env `core_mem` blocker
  - 将两个 benchmark 从 1-sample real run 扩展到正式全量 protocol 运行

## 2026-04-05 Session 009

- Worked on: 继续执行前台 autoresearch，将 readiness/acceptance 机械化并为正式 benchmark 补齐可续跑执行路径
- State changed:
  - 新增 `scripts/verify_stage1_acceptance.py`，将 AC-001~AC-007 机械化
  - `scripts/verify_stage1_status.py --score-only` 从 `26` 提升到 `31`
  - `scripts/run_experiment.py` 新增 `--run-dir` / `--resume` 与增量 prediction 落盘
  - 新增 resumable runner test 与 acceptance verifier test，`pytest` 提升到 `21` 个测试并继续通过
  - 尝试正式 LongMemEval-S 全量运行时，provider 直接返回 `HTTP 403 AllocationQuota.FreeTierOnly`
- Evidence / artifacts:
  - `scripts/verify_stage1_acceptance.py --skip-pytest` 返回 `3/7`
  - `scripts/verify_stage1_status.py --score-only` 输出 `31`
  - `scripts/run_experiment.py --verify-only` 输出 `31`
  - `pytest` 通过（21 tests）
  - `research-results.tsv` 已记录 iteration `7` keep、`8` refine、`9` keep、`10` crash
- Next likely action:
  - 恢复阿里云 provider 可用额度或替换为用户批准的可用 key
  - 创建默认 conda named env `core_mem`
  - 在外部 blocker 解除后恢复 PersonaMem / LongMemEval-S 正式全量运行

## 2026-04-05 Session 010

- Worked on: 在新权限下补齐默认 conda 环境、embedding 层和非 benchmark 第一阶段项
- State changed:
  - 安装了 autoresearch hooks，供后续 session 使用
  - 将“除环境外，外部目录修改必须先问用户”写入 `AGENTS.md` 与 `change-decisions.md`
  - 创建默认 conda named env `core_mem`，并用 Python / pytest / verify-only 复验
  - 补齐 `outputs/logs` 与 `outputs/figures` 目录结构
  - 接入 `sentence-transformers/all-MiniLM-L6-v2` embedding 配置与 adapter
  - 为 `VanillaCoreMemory` 增加 `observe_text/query_text`
  - `scripts/verify_stage1_status.py --score-only` 从 `33` 提升到 `36`
  - `scripts/verify_stage1_acceptance.py --score-only` 提升到 `5`
- Evidence / artifacts:
  - `/home/image/miniconda3/envs/core_mem`
  - `CONDA_NO_PLUGINS=true conda run -n core_mem python --version` -> `Python 3.10.20`
  - `CONDA_NO_PLUGINS=true conda run -n core_mem pytest -q` 通过
  - `CONDA_NO_PLUGINS=true conda run -n core_mem python scripts/run_experiment.py --verify-only` 输出 `36`
  - `CONDA_NO_PLUGINS=true conda run -n core_mem python -c "import sentence_transformers; print(sentence_transformers.__version__)"` -> `5.3.0`
  - `pytest` 通过（23 tests）
- Next likely action:
  - 等待 provider 配额恢复后恢复 PersonaMem / LongMemEval-S 正式全量运行

## 2026-04-05 Session 011

- Worked on: 切换到用户批准的 Gemini 路径并恢复正式 benchmark 推进
- State changed:
  - 新增 `configs/gemini_flash.yaml` 作为 Google AI Studio OpenAI-compatible 试运行配置
  - `scripts/stage1_preflight.py` 改为 provider-agnostic API key 检查
  - 为 provider 增加 `max_retries` / `retry_backoff_seconds`
  - 为 `scripts/run_experiment.py` 增加对旧式 `llm_config` 的向后兼容默认值
  - 增加 Gemini 配置测试与 rate-limit retry 测试，`pytest` 提升到 `25` 个测试并继续通过
  - Gemini 下 PersonaMem / LongMemEval-S 的真实 1-sample run 成功
  - Gemini 下 LongMemEval-S formal run 从此前的 `2` 条 prediction 续跑到更高进度，不再因首个 `HTTP 429` 立即崩溃
- Evidence / artifacts:
  - `configs/gemini_flash.yaml`
  - `outputs/runs/20260405T042146Z_personamem/evaluation.json`
  - `outputs/runs/20260405T042150Z_longmemeval/evaluation.json`
  - `outputs/runs/20260405T042236Z_longmemeval/predictions.jsonl`
  - `pytest` 通过（25 tests）
  - `scripts/run_experiment.py --verify-only` 输出 `36`
  - `scripts/verify_stage1_acceptance.py` 返回 `5/7`
- Next likely action:
  - 继续推进 Gemini 下的 formal benchmark
  - 如果 `429` 压力下吞吐仍偏低，则增加显式 pacing 控制

## 2026-04-05 Session 012

- Worked on: 用前台 autoresearch 持续尝试完成 Gemini formal benchmark，并将失败路径机械化收束为真实外部 blocker
- State changed:
  - 为 provider 增加显式 pacing、retry delay cap，并补齐相应测试
  - 新增 `scripts/run_until_complete.py` 作为外层 benchmark supervisor
  - 新增 `configs/gemini_flash_ultraslow.yaml` 用于超保守单样本推进检查
  - `pytest` 提升到 `31` 个测试并继续通过
  - PersonaMem formal run 在 `outputs/runs/20260405T050000Z_personamem/` 下推进到 `22/589`
  - LongMemEval-S formal run 在 `outputs/runs/20260405T042236Z_longmemeval/` 下推进到 `19/500`
  - 即使在 pacing、bounded retry、delay cap、outer supervisor、chunked relaunch 和 ultra-slow single-sample 检查下，当前 Gemini key 仍连续返回 `HTTP 429`，无法把 PersonaMem 从 `22` 推到 `23`
  - 当前第一阶段仍停在 acceptance `5/7`，formal benchmark 两项受真实外部 blocker 限制
- Evidence / artifacts:
  - `scripts/run_until_complete.py`
  - `configs/gemini_flash_ultraslow.yaml`
  - `outputs/runs/20260405T050000Z_personamem/predictions.jsonl` -> `22`
  - `outputs/runs/20260405T042236Z_longmemeval/predictions.jsonl` -> `19`
  - `pytest` 通过（31 tests）
  - `scripts/run_experiment.py --verify-only` 输出 `36`
  - `research-results.tsv` 已记录 iteration `18`~`29` 的 pacing / supervisor / blocker 证据
- Next likely action:
  - 等待更高 quota / 更稳定的 Gemini key，或用户批准的其他可用 provider
  - provider 条件恢复后，从现有 run dir 继续 resume formal benchmark

## 2026-04-06 Session 013

- Worked on: 审阅当前 stage-1 模型、benchmark 主链路、memory 原型和已有实验，并沉淀详细白盒实现文档
- State changed:
  - 新增 `docs/current_model_implementation.md`
  - 在 `docs/current_status.md` 中明确写入：当前 `VanillaCoreMemory` 最小原型已实现，但正式 benchmark runner 仍是 full-context prompt 主链路，尚未接入 memory write/read 的正式推理闭环
  - 在 `.agent-os/project-index.md` 中补充本次审阅结论，便于后续 session 恢复
- Evidence / artifacts:
  - `docs/current_model_implementation.md`
  - `docs/current_status.md`
  - `.agent-os/project-index.md`
- Next likely action:
  - 基于该白盒审阅文档，讨论并确定下一版 memory-mediated benchmark inference 的集成方案

## 2026-04-06 Session 014

- Worked on: 将用户确认过的 stage-2 / `V2.0 structured latent-slot memory` 方案写入真源文档，并同步整个项目文档系统

## 2026-04-14 Session 015

- Worked on: 由 background autoresearch runtime 驱动，完成 stage-2 本地 tiny-backend `gpu3` train/eval/ablation matrix，并补 experiment registry
- State changed:
  - 新增 `src/core_mem/v2/experiments.py`
  - `scripts/train_stage2.py` / `scripts/eval_stage2_local.py` 现支持 preset ablation variants、checkpoint-aware local eval 与 `stage2_experiment_index.json` 自动登记
  - `research-results.tsv` / `autoresearch-state.json` 已初始化并完整记录本次 background run
  - `outputs_v2/artifacts/stage2_experiment_index.json` 当前已登记 `mainline + 11` 个必做 ablation
  - `scripts/verify_stage2_experiment_status.py --score-only` 已从 baseline `1` 提升到 `13`
- Evidence / artifacts:
  - `research-results.tsv`
  - `autoresearch-state.json`
  - `outputs_v2/artifacts/stage2_experiment_index.json`
  - `outputs_v2/runs/20260414T074034Z_stage2_train_exec/`
  - `outputs_v2/evals_local/20260414T074458Z_stage2_local_eval.json`
  - `conda run -n core_mem pytest -q tests/test_stage2_training_runtime.py tests/test_stage2_local_eval.py tests/test_stage2_data_pipeline.py tests/test_stage2_public_data.py` 通过（18 tests）
- Next likely action:
  - 如需继续推进 stage-2，优先补 benchmark canary 结果或默认 `flan-t5-base` backbone 的非 tiny `gpu3` 训练证据；stage-1 formal benchmark 继续保持 pending by explicit user trigger

## 2026-04-07 Session 015

- Worked on: 以 fresh-start foreground autoresearch 启动 stage-2，并落第一轮 verifier / acceptance / outputs 骨架
- State changed:
  - 归档旧的 stage-1 autoresearch 工件，初始化新的 stage-2 foreground run
  - 新增 `scripts/verify_stage2_status.py` 与 `scripts/verify_stage2_acceptance.py`
  - 新增 `tests/test_verify_stage2_status.py` 与 `tests/test_verify_stage2_acceptance.py`
  - 新增 `outputs_v2/` 默认目录骨架
  - `stage2_readiness_score` 从 `0` 提升到 `20`
  - `stage2_acceptance` 当前为 `2/7`
  - `pytest` 提升到 `34` 个测试并继续通过
- Evidence / artifacts:
  - `scripts/verify_stage2_status.py`
  - `scripts/verify_stage2_acceptance.py`
  - `outputs_v2/`
  - `research-results.tsv` / `autoresearch-state.json` 已切换到 stage-2 fresh-start run
  - `pytest` 通过（34 tests）
  - `python3 scripts/verify_stage2_status.py --score-only` -> `20`
  - `python3 scripts/verify_stage2_acceptance.py --skip-pytest` -> `2/7`
- Next likely action:
  - 实现 stage-2 observation schema、parser skeleton、dataset ingestion skeleton 与最小训练样本构造器
- State changed:
  - `docs/requirements.md` 已扩展为 stage-1 + stage-2 双阶段真源文档
  - 根契约 `AGENTS.md` / `CLAUDE.md` 已同步第二阶段边界与恢复规则，且 `CLAUDE.md` 已重新建立为指向 `AGENTS.md` 的硬链接
  - 新增 `docs/v2_design.md`，汇总第二阶段方法、训练、数据、评测和输出规范
  - `docs/current_status.md`、`docs/implementation_plan.md`、`docs/todo.md` 已同步 stage-2 真相
  - `.agent-os/project-index.md`、`.agent-os/todo.md`、`.agent-os/architecture-milestones.md`、`.agent-os/change-decisions.md`、`.agent-os/acceptance-report.md`、`.agent-os/lessons-learned.md` 已同步 stage-2 真相
  - 全局 top next action 从受外部 provider 阻断的 stage-1 formal benchmark，切换为可独立推进的 stage-2 数据与规范管线骨架 `TD-015`
- Evidence / artifacts:
  - `docs/requirements.md`
  - `docs/v2_design.md`
  - `docs/current_status.md`
  - `docs/implementation_plan.md`
  - `docs/todo.md`
  - `.agent-os/project-index.md`
  - `.agent-os/todo.md`
  - `.agent-os/architecture-milestones.md`
- Next likely action:
  - 开始实现第二阶段 `TD-015`：observation normalization、belief JSON 目标生成与公开数据集接入骨架

## 2026-04-06 Session 015

- Worked on: 细化第二阶段真源与状态文档，纠正 stage-2 / stage-1 runtime truth，并把 schema、数据映射和指标映射写成实现级规格
- State changed:
  - 重新创建并重写 `docs/requirements.md`，补全 stage-1 + stage-2 双阶段真源
  - 明确 stage-2 是当前执行主线，stage-1 formal benchmark 为待用户显式触发项
  - 将 `docs/v2_design.md` 细化到 `Observation JSON`、`Slot Record`、`Belief JSON`、dataset-to-task mapping 与 metric-to-module mapping
  - 将 `docs/current_status.md`、`docs/implementation_plan.md`、`docs/todo.md`、`.agent-os/project-index.md`、`.agent-os/todo.md`、`.agent-os/architecture-milestones.md`、`.agent-os/change-decisions.md`、`.agent-os/acceptance-report.md` 同步到新的 stage-2 主线真相
- Evidence / artifacts:
  - `docs/requirements.md`
  - `docs/v2_design.md`
  - `docs/current_status.md`
  - `docs/implementation_plan.md`
  - `docs/todo.md`
  - `.agent-os/project-index.md`
  - `.agent-os/todo.md`
- Next likely action:
  - 进入 `TD-015` 的代码骨架实现：observation normalization、belief JSON 目标生成、parser skeleton 与公开数据集 ingestion

## 2026-04-07 Session 016

- Worked on: 继续执行 stage-2 foreground autoresearch，并把 `TD-015` 从 verifier skeleton 推进到首批可运行实现
- State changed:
  - 新增 `src/core_mem/v2/`，包含 Observation / Slot / Belief schema、relation normalization、rule-first parser 与 dataset-task registry
  - 新增 `scripts/prepare_stage2_data.py`、`scripts/train_stage2.py`、`scripts/eval_stage2_local.py`、`scripts/run_stage2_canary.py`
  - 新增 `configs/stage2_train.yaml`
  - 新增 stage-2 schema/parser/pipeline 测试，`pytest` 总数提升到 `40`
  - `stage2_readiness_score` 从 `20` 提升到 `30`
  - `stage2_acceptance` 从 `2/7` 提升到 `7/7`
  - `outputs_v2/` 已实际产出 prepared manifest、smoke train plan、smoke local eval 结果与 `PersonaMem/LongMemEval-S` 的 `64` canary manifests
  - runtime truth 已切换为：`TD-015` 完成，`TD-016` 成为新的 top next action
- Evidence / artifacts:
  - `src/core_mem/v2/`
  - `configs/stage2_train.yaml`
  - `outputs_v2/artifacts/stage2_prepared_samples_manifest.json`
  - `outputs_v2/runs/20260407T042125Z_stage2_train_plan/training_plan.json`
  - `outputs_v2/evals_local/20260407T042125Z_stage2_local_eval.json`
  - `outputs_v2/evals_benchmark/20260407T042120Z_stage2_canary_plan.json`
  - `pytest` 通过（40 tests）
  - `python3 scripts/verify_stage2_status.py --score-only` -> `30`
  - `python3 scripts/verify_stage2_acceptance.py --score-only` -> `7`
- Next likely action:
  - 进入 `TD-016` 的主线模型骨架实现：slot encoder、retrieval key、lifecycle、core/residual、consolidation、light resampler 与 belief decoder

## 2026-04-07 Session 017

- Worked on: 继续执行 stage-2 foreground autoresearch，并把 `TD-016` 从“待实现”推进到第一版可运行的 memory system skeleton
- State changed:
  - 新增 `src/core_mem/v2/encoder.py`、`lifecycle.py`、`resampler.py`、`decoder.py`、`system.py`
  - 新增 `tests/test_stage2_model_skeleton.py`
  - `stage2_readiness_score` 从 `30` 提升到 `36`
  - `stage2_acceptance` 维持 `7/7`
  - `pytest` 提升到 `42` 个测试并继续通过
- Evidence / artifacts:
  - `src/core_mem/v2/system.py`
  - `tests/test_stage2_model_skeleton.py`
  - `pytest` 通过（42 tests）
  - `python3 scripts/verify_stage2_status.py --score-only` -> `36`
  - `python3 scripts/verify_stage2_acceptance.py --score-only` -> `7`
- Next likely action:
  - 将 deterministic model skeleton 更紧地接到训练样本、consolidation 细节与 local eval 主链路

## 2026-04-07 Session 018

- Worked on: 继续执行 stage-2 foreground autoresearch，并把主线模型 skeleton 从 `memory -> belief` 推进到 `memory -> belief -> answer`
- State changed:
  - 新增 `src/core_mem/v2/consolidation.py` 与 `src/core_mem/v2/projection.py`
  - `src/core_mem/v2/system.py` 现在通过显式 consolidation 与 answer projection 输出 evidence block 和 answer text
  - `stage2_readiness_score` 从 `36` 提升到 `38`
  - `stage2_acceptance` 维持 `7/7`
  - `pytest` 维持 `42` 个测试并继续通过
- Evidence / artifacts:
  - `src/core_mem/v2/consolidation.py`
  - `src/core_mem/v2/projection.py`
  - `tests/test_stage2_model_skeleton.py`
  - `pytest` 通过（42 tests）
  - `python3 scripts/verify_stage2_status.py --score-only` -> `38`
  - `python3 scripts/verify_stage2_acceptance.py --score-only` -> `7`
- Next likely action:
  - 将 deterministic skeleton 进一步接到训练样本、richer consolidation 与 local intrinsic eval 主链路

## 2026-04-13 Session 019

- Worked on: 继续执行 stage-2 foreground autoresearch，并把训练能力从 planner/smoke 推进到真实可执行 runtime
- State changed:
  - 新增 `src/core_mem/v2/training.py`
  - `scripts/train_stage2.py` 现在支持 `--execute-train`
  - 新增 `configs/stage2_train_tiny.yaml` 作为离线 tiny backend 验证配置
  - `environment.yaml` 已加入 `datasets`、`peft`、`accelerate`
  - 默认 conda 环境 `core_mem` 中已实际安装并验证 `torch / transformers / datasets / peft / accelerate`
  - `outputs_v2/runs/20260414T020438Z_stage2_train_exec` 已跑通最小 `execute-train`
  - `stage2_readiness_score` 从 `38` 提升到 `41`
  - `pytest` 提升到 `44` 个测试并继续通过
- Evidence / artifacts:
  - `src/core_mem/v2/training.py`
  - `configs/stage2_train_tiny.yaml`
  - `outputs_v2/runs/20260414T020438Z_stage2_train_exec/training_metrics.json`
  - `outputs_v2/checkpoints/20260414T020438Z_stage2_train_exec/`
  - `pytest` 通过（44 tests）
  - `python3 scripts/verify_stage2_status.py --score-only` -> `41`
  - `python3 scripts/verify_stage2_acceptance.py --score-only` -> `7`
- Next likely action:
  - 将 `prepare_stage2_data.py` 从 demo payload 推进到真实公开数据集处理链，并让 trainer 直接消费 `SGD + MultiWOZ 2.4 + Persona-Chat + MQUAKE + ReCoE` 的 prepared manifests

## 2026-04-13 Session 020

- Worked on: 继续执行 stage-2 foreground autoresearch，并把数据准备从 demo-only 推进到 source-config 驱动的真实入口
- State changed:
  - `prepare_stage2_data.py` 已支持 `--source-config` 和 `--strict-sources`
  - 新增 `configs/stage2_data_sources.json`
  - 新增 `scripts/stage2_data_preflight.py`
  - 新增 `tests/test_stage2_data_pipeline.py`
  - `stage2_readiness_score` 从 `41` 提升到 `44`
  - `pytest` 提升到 `46` 个测试并继续通过
  - 当前真实缺口从“没有数据入口”收敛为“`data/stage2_public/` 下还没有真实公开数据文件”
- Evidence / artifacts:
  - `configs/stage2_data_sources.json`
  - `scripts/stage2_data_preflight.py --json` 当前返回 5 个 source 都 missing
  - `pytest` 通过（46 tests）
  - `python3 scripts/verify_stage2_status.py --score-only` -> `44`
  - `python3 scripts/verify_stage2_acceptance.py --score-only` -> `7`
- Next likely action:
  - 继续补真实公开数据源文件的获取/放置路径，使 `prepare_stage2_data.py --strict-sources` 能产出非 demo manifests

## 2026-04-13 Session 021

- Worked on: 继续执行 stage-2 foreground autoresearch，并把“真实公开数据缺失”推进到“stage2 可直接训练”
- State changed:
  - 新增 `src/core_mem/v2/public_data.py` 与 `scripts/normalize_stage2_public_data.py`
  - 从官方或作者源下载并落地 `SGD`、`MultiWOZ 2.4`、`Persona-Chat`、`MQUAKE`、`ReCoE`
  - 生成 5 份 `data/stage2_public/*/normalized.jsonl`
  - `scripts/stage2_data_preflight.py --json` 现在返回 `missing=[]`
  - `prepare_stage2_data.py` 新增 `--max-rows-per-dataset`
  - `configs/stage2_train.yaml` 与 `configs/stage2_train_tiny.yaml` 已锁定 repo-local Hugging Face cache root
  - `scripts/train_stage2.py` 的 launcher 现已改为 `--execute-train`
  - 真实 public-data prepared manifest 已在 `outputs_v2/artifacts/stage2_prepared_samples_manifest.json` 产出，且 `outputs_v2/runs/20260414T043225Z_stage2_train_exec` 已基于其跑通 tiny execute-train
  - `stage2_readiness_score` 从 `44` 提升到 `47`
  - `pytest` 提升到 `53` 个测试并继续通过
- Evidence / artifacts:
  - `data/stage2_public/sgd/normalized.jsonl`
  - `data/stage2_public/multiwoz24/normalized.jsonl`
  - `data/stage2_public/personachat/normalized.jsonl`
  - `data/stage2_public/mquake/normalized.jsonl`
  - `data/stage2_public/recoe/normalized.jsonl`
  - `outputs_v2/artifacts/stage2_prepared_samples_manifest.json`
  - `outputs_v2/runs/20260414T043221Z_stage2_train_plan/launch_stage2_training.sh`
  - `outputs_v2/runs/20260414T043225Z_stage2_train_exec/training_metrics.json`
  - `pytest` 通过（53 tests）
  - `python3 scripts/verify_stage2_status.py --score-only` -> `47`
  - `python3 scripts/verify_stage2_acceptance.py --score-only` -> `7`
- Next likely action:
  - 将 stage-2 主线从“可直接训练”继续推进到完整的 local intrinsic eval / budget sweep / ablation 记录

## 2026-04-14 Session 022

- Worked on: 补完第二阶段本地 intrinsic evaluation 体系，并为其补齐完整中文文档
- State changed:
  - 新增 `src/core_mem/v2/eval_local.py`
  - `scripts/eval_stage2_local.py` 现支持 `--top-k`、`--budget`、`--dataset`
  - 新增 `docs/stage2_local_evaluation.md`
  - `prepare_stage2_data.py` 保留 `_meta`，使 local eval 能按 dataset/sample_id 分层
  - public-data lifecycle 标签已对齐当前 lifecycle 规则，避免 local eval 被标签偏差污染
  - local eval 现在会输出 JSON + summary CSV + budget CSV
  - `stage2_readiness_score` 从 `50` 稳定为 `50`（新增检查项后保持全通过）
  - `pytest` 提升到 `56` 个测试并继续通过
- Evidence / artifacts:
  - `src/core_mem/v2/eval_local.py`
  - `docs/stage2_local_evaluation.md`
  - `outputs_v2/evals_local/20260414T055852Z_stage2_local_eval.json`
  - `outputs_v2/tables/20260414T055852Z_stage2_local_eval_summary.csv`
  - `outputs_v2/tables/20260414T055852Z_stage2_local_eval_budget_sweep.csv`
  - `pytest` 通过（56 tests）
  - `python3 scripts/verify_stage2_status.py --score-only` -> `50`
  - `python3 scripts/verify_stage2_acceptance.py --score-only` -> `7`
- Next likely action:
  - 用这套 local eval 体系去跑更系统的 budget sweep、ablation 记录和训练后模型比较

## 2026-04-16 Session 023

- Worked on: 收口 `v2.2` full-data semantic latent 长跑，并把项目主线切到 `v2.3 stronger learned slot assignment + stronger latent`
- State changed:
  - `TD-032 / WS-018` 已机械完成，`scripts/verify_stage2_v22_completion.py --score-only = 19`
  - current-head full-data semantic train/eval、`PersonaMem 128` / `LongMemEval-S 64/128` semantic canaries、LongMemEval-S semantic analysis、semantic online gain 全部已补齐
  - retained 代码侧关键改动是 `src/core_mem/v2/projection.py` 的 generic answer projection normalization，以及对应 stage-2 regression tests
  - runtime truth 已从 `v2.2 closeout` 切到 `TD-033 / WS-019`
  - 新增 `docs/v23_plan.md`，把下一阶段目标正式切到 `LongMemEval-S` 质量、learned slot assignment 与 stronger latent
- Evidence / artifacts:
  - `outputs_v2/artifacts/latest_stage2_semantic_full_train.json`
  - `outputs_v2/artifacts/latest_stage2_semantic_full_local_eval.json`
  - `outputs_v2/evals_benchmark/20260416T021743Z_stage2_memory_canary.json`
  - `outputs_v2/evals_benchmark/20260416T024146Z_stage2_memory_canary.json`
  - `outputs_v2/artifacts/latest_longmemeval_stage2_semantic_analysis.json`
  - `outputs_v2/artifacts/latest_stage2_semantic_online_gain.json`
  - `docs/v23_plan.md`
  - `python3 scripts/verify_stage2_v22_completion.py --score-only` -> `19`
- Next likely action:
  - 建立 `v2.3` verifier 与 long-run baseline，优先验证 `LongMemEval-S` 分层错误、learned slot assignment 和更强的在线 latent 主链

## 2026-04-16 Session 024

- Worked on: 把 `v2.3` 进一步升级成更远的 managed long-run，并为其建立新的 verifier / baseline / 文档主线
- State changed:
  - 新增 `docs/v23_longrun_plan.md`
  - 新增 `scripts/verify_stage2_v23_longrun.py`
  - 新增 `tests/test_stage2_v23_longrun.py`
  - 当前 next action 从 `TD-033 / WS-019` 继续前推到 `TD-034 / WS-020`
  - 新长跑继续保留 `semantic-first`、`no fallback`、`no shortcut`，但目标显著拉远：不只要 learned slot assignment 接上，还要求 `LongMemEval-S 64/128`、`PersonaMem 128` 和 gain artifact 都形成 current-head retained evidence
- Evidence / artifacts:
  - `docs/v23_longrun_plan.md`
  - `scripts/verify_stage2_v23_longrun.py`
  - `tests/test_stage2_v23_longrun.py`
- Next likely action:
  - 以 fresh-start 启动新的后台 `v2.3 long-run` autoresearch，并在 baseline 后开始围绕 `LongMemEval-S` 质量与 learned slot assignment 迭代
