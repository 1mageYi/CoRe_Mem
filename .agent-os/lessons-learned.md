# Lessons Learned

## Purpose

本文件用于记录失败探索、陷阱、未采用方案及其重试条件。

## Entries

- 2026-04-05:
  - `conda` 在当前机器上直接运行会触发 plugin / CUDA virtual package 的权限异常；后续应优先尝试 `--no-plugins` 或 `CONDA_NO_PLUGINS=true`。
  - Windows 下超大 `apply_patch` 可能触发 `CreateProcessAsUserW failed: 206`；需要将补丁拆成更小批次提交。
  - 在当前机器上，named env 注册依赖用户目录写权限，容易失败；若需要避开只读家目录缓存，应优先重定向缓存目录，但最终仍以默认 conda named env 作为项目真相，不把环境目录放进 repo。
  - 如果目标执行平台最终是 WSL，就不要继续把 bootstrap 逻辑写死到 Windows；应尽快把仓库收敛成平台中立状态后迁移。
  - 在缺少 named env、API key 或官方数据时，不要继续假装能推进正式 benchmark；应先用机械 preflight 明确 blocker，再同步回 `.agent-os/` 与 `docs/`。
  - 正式 benchmark 一旦进入数百样本规模，没有进度输出和增量落盘的 runner 会让排障成本很高；应优先提供 `--run-dir` / `--resume` 与边跑边写 prediction 的能力。
  - 阿里云兼容接口即使 key 语法有效，也可能因免费额度耗尽返回 `HTTP 403 AllocationQuota.FreeTierOnly`；遇到 403 时应先读取响应体，不要先假设是本地代码 bug。
  - 在这台机器上，`conda env create -f environment.yaml` 的 classic solver 元数据求解明显慢于直接 `conda create -n core_mem python=3.10 ...`；优先用后者建最小环境更稳。
  - `sentence-transformers` 的默认 pip 安装会把大体积 `torch` / CUDA 依赖一起拉进环境；后续如果只需要 CPU，应该考虑更精细的依赖约束。
  - Google AI Studio 的 OpenAI-compatible 端点可以直接复用当前 provider adapter，但 formal benchmark 下仍可能触发 `HTTP 429`；加入 retry/backoff 后，runner 能从“立即崩溃”改善为“低吞吐但可续跑”的状态。
  - 当 `HTTP 429` 持续发生时，单纯增加进程内 retry/backoff 不一定能换来 completion；更可靠的做法是外层 supervisor + resumed run，但如果超保守单样本检查仍连续 429，就应诚实判定为外部 provider blocker，而不是继续假装通过调参就能跑完。

- 2026-04-06:
  - 如果第二阶段方法要训练 latent memory，benchmark-derived proxy supervision 更适合做 intrinsic eval、ablation 和 failure analysis，而不是主要训练信号；否则很容易产生 benchmark leakage 争议。
  - 对第二阶段来说，`public-datasets-first, synthetic-minimal` 比“大量自生成 memory worlds”更有说服力，也更符合论文写作中的数据可信度预期。
  - 如果方法主线是 `structured latent slots`，那么“直接取 gloss 当答案证据”会过于接近压缩版 RAG；必须保留 `latent composition -> belief decoding` 这一层，才能维持方法边界。

- 2026-04-13:
  - `prepare_stage2_data.py` 先做 source-config 入口和 preflight，再做真实数据抓取，比一开始就把下载逻辑写死进 trainer 更稳；这样可以把“代码没写完”和“数据源没到位”分开。
  - 在当前 `datasets` 版本下，测试过的多个常见脚本型 HF dataset IDs（如 `gem/schema_guided_dialog`、`schema_guided_dstc8`、`multi_woz_v22`、`bavard/personachat_truecased`）会直接返回 “dataset scripts are no longer supported”；后续如果要自动取数，不能默认依赖旧式脚本型 HF loader，需要更明确的抓取策略或直接提供原始数据文件。
  - 真实公开数据一旦全部规范化，`prepare_stage2_data.py` 直接吃全量 `normalized.jsonl` 会让 smoke 级验证变得过重；补一个 `--max-rows-per-dataset` 开关，可以同时保留全量真实语料与轻量、可机械复验的 public-data slice。
  - 如果 stage-2 要遵守“repo 外目录修改需先确认”的契约，那么首次 Hugging Face 权重和 tokenizer 下载就不能默认落到 `~/.cache`；应在训练配置中显式锁定 repo-local cache root，再把 launcher 直接写成 `--execute-train`。
- 2026-04-14:
  - local eval 如果只输出一个混合均值，很快就会失去诊断价值；必须至少同时保留 `families`、`modules`、`budget_sweep` 三个视图，才能回答“到底是哪个模块出了问题”。
  - prepared task rows 如果不保留 `dataset/sample_id` 元信息，update/progression 和 dataset-specific 分层评测基本做不起来；所以 `_meta` 必须在 prepare 阶段保留下来，而不是等 eval 再猜。
  - public-data lifecycle 标签必须和当前 lifecycle 规则一致，否则 local eval 会把“标签构造偏差”误判成模块退化。
  - 如果 verifier 只检查 experiment registry，而没有任何代码路径去维护它，那么 completion score 会永远卡死在 baseline；应先补 `outputs_v2/artifacts/stage2_experiment_index.json` 的 authoritative writer，再谈自动运行矩阵。
  - background autoresearch 的 shell loop 很容易在复杂 quoting 上出错，导致“实验已完成但未记账”；对需要“每完成一个实验立刻记账”的序列任务，用 Python driver 调 subprocess 更稳。
- 2026-04-16:
  - `v2.4` 的 nested `slot_assignment_metrics` 如果只保留 `slot_assignment_token_f1` 这类前缀字段，当前 verifier 不会把它记进 `token_f1/field_f1` 阈值检查；发布 current-head artifact 时需要同时带通用键名。
  - 单纯把 `composition_to_belief` 的 online-aligned repeats 从 `2` 提到 `3`，在 full-data train/eval 上不会自动换来更高的 `trained_eval.token_f1 / field_f1`；当前瓶颈不是 task repeat 数量本身。
  - `LongMemEval-S` 的一部分 retained 收益来自 answer exactness，而不是更重的训练配方；对 `How many / How often / where-did-you-buy` 这类 query-aware exactness case，先修 projection/prompt contract 比盲目加训练更有效。
  - `semantic_outputs.py` 里直接用 `_SLOT_ID_RE.findall(text)` 去扫 malformed belief payload 会把字面量 `slot_ids` 当成真的 support slot id；对 braceless / partially malformed belief JSON，必须先做真实 slot-id 值过滤，再做 support evidence 恢复。
  - 一旦进入 `v2.5` 这类更激进阶段，full benchmark 应只当 holdout measurement；如果把 benchmark slice 的错误模式直接回灌成 task supervision、prompt special case 或 benchmark-specific heuristic，短期分数可能会上升，但会直接破坏“更泛化、更鲁棒”的目标。
- 2026-04-17:
  - 对 background managed run 来说，历史 artifact 曾经在另一 session 成功跑过 live provider，并不等于当前 session 仍然带着同样的 env；在任何 live canary / full benchmark 刷新前，必须先机械确认 `GPT_AGENT_API_KEY` 已进入当前进程环境，否则会白白烧掉 slot-assignment / model 预处理时间，最后只得到 `provider_configured=false`。
  - query-intent-aware temporal retrieval / belief scoring 可以先用 unit tests 锁住“历史型 query 不被当前槽位压掉”的行为，但没有 live provider env 时，不能把这种 trial commit 误写成 `v2.6 gain`；应诚实停在 `blocked`，把 commit 留作未验证 trial，而不是伪造 canary/gain artifact。
  - learned slot-assignment 在线 prompt 如果直接吃全量 memory context，会在 dense `other_fact` 样本上把 live canary 拖成分钟级；即使真正需要 arbitration 的 observation 只有十来条，T5 generation 也会被 60+ 候选 prompt 放大到不可用。对 `other_fact` 这类 open-world relation，应该优先把 online prompt 收缩到 `symbolic target + top candidates + 少量最近上下文`，并对低 lexical-overlap 的 overwrite 直接 fast-path 为 `new`，再把剩余 live refresh 交回 background runtime。
- 2026-04-18:
  - `v2.8` 这条线里，“teacher 改动没有被训练/评测真正消费” 和 “teacher supervision 本身有害” 是两件不同的事；必须先用 changed-sample matched subset 把前者排除掉，才能诚实断定后者。
  - 在当前 `256 / 128 / 128` teacher suite 下，能覆盖 enough samples 的 teacher supervision 只有 `slot_assignment` 和 `belief`；如果这两条线都在 matched internal compare 上显式退化，而 observation-only raw-teacher 改动又只剩 `train=1 / val=0 / test=3`，那就应把 run 判成 true blocker，而不是继续靠更窄的 refresh 刷随机噪声。
  - `train_stage2.py` 与 `eval_stage2_local.py` 当前都用秒级 UTC 时间戳生成 artifact 路径；并行启动时会直接撞到同一个 `run_dir` / `result_path`。后续若还要做并行 compare，必须先加更细粒度时间戳或显式 run namespace，否则结果不可审计。
  - `v2.9` 这条线里，先补 authoritative publisher 再跑长实验是值得的：如果 verifier 只认 `latest_stage2_v29_*` artifacts，而 repo 里还没有 publish path，那么就算 current-head 已经产出了 train/eval/canary 原始结果，metric 也只会停在 baseline。
  - launch manifest ready 并不等于当前 session 带着 live provider env；在任何 `512` holdout 之前，先用 `echo ${GPT_AGENT_API_KEY:+SET}` 或 `run_stage2_memory_canary.py --limit 1` 机械确认 `provider_configured=true`，能避免把 run 浪费在注定只能得到 `blocked_provider_not_configured` 的 probe 上。
- 2026-04-19:
  - 如果 `run_stage2_memory_canary.py` 只把 provider 调用并行化、却把 memory build / prompt precompute 保持串行，那么大 holdout 会长时间停在 `completed_predictions=0`；对 `memory_mode=symbolic` 且 `slot_assignment_mode=symbolic` 的路径，应该直接做 sample-level 并行并增量写 `predictions.jsonl`
  - `LongMemEval-S` 当前官方 cleaned 数据集实际上只有 `500` 条；`v2.9` 的 expanded holdout 目标和 verifier 不应再写成 `512`
  - `v30` 这种 non-tiny modular 训练不适合把 `train -> checkpoint eval -> register -> publish` 全塞进一次长命令里；训练 summary 往往很快就落盘，但 checkpoint eval 可能继续占住 GPU 很久。更稳的做法是先让训练独立完成，再单独跑 val eval 和 artifact publish。
  - 一旦进入 `v30` 这种强调“holdout-only / no leakage”的阶段，`scripts/train_stage2.py --register-experiment` 不能默认重用 train manifest 做 checkpoint eval；必须显式分离 `--prepared-manifest` 和 `--eval-manifest`，否则 compare 口径会直接漂移。
  - `v31` latent compare 如果直接拿 retained `v30` 的 `val` 指标去对 current-head `test` 指标，数值变化会混进 split 差异而不是方法差异；在判断 latent-first 是否真的转正之前，必须先把 compare manifest 对齐到与 retained baseline 相同的 held-out split，再决定 keep/discard。
  - 对 `composition_to_belief`，如果模型输出只保留 relation / slot 片段而丢失合法 JSON 壳，不应立刻判成语义失败；只要 `input_text` 里还带着 `query` 与 `memory_slots`，就可以基于 relation hint、slot rank 与 `canonical_gloss` 把 value / support 修回去。这类修复可以显著提升 belief field-level 指标，但它只代表 internal semantic recovery 变强，不等于 full holdout gain 已成立。
  - 对 `lifecycle_prediction` 也存在类似模式：少数失败样本不是 action 决策完全错误，而是模型退回去复述旧 slot 的 `canonical_gloss`。如果 `memory_context` 与 `new_observation` 已经明确同 relation 的 active overwrite/new 条件，就可以在 parse 失败时基于输入上下文把 `target_action / target_flags` 修回去。这能把 write internal compare 从 tie 推到正增益，但同样只代表 internal semantic recovery，不代表 full holdout 已成立。
  - 在 `v31` 的 `8`-sample holdout quick smoke 上，`acd74206-37dc-4756-94a8-b99a395d9a21` 曾在**完全相同的 prompt 与 evidence block**下跨 run 出现 provider `(c)` 与 `(b)` 翻转；后续若还要用这类小样本 smoke 做 gate，必须先把 provider 波动视为真实风险，而不是默认把 `±1` 的 provider exact 变化全归因到代码。
  - `v31` 当前 quick-smoke line 里，belief fallback 微调、blank-output 专项修补、learned slot-assignment pivot、以及最小 json-start constrained decoding 都没能把 retained `24/32` 推成新的 keep；这说明下一步若要继续提升，已经不适合再做同类 runtime micro-tune，更可能需要更大样本 measurement 或更宽 scope 的 decoder / answer-selection redesign。
  - `v31` 的 `latent_retriever.pt` 单独训练 artifact 如果不显式接进 `StructuredMemorySystem` / `run_stage2_memory_canary.py`，那么 online holdout 根本不会消费这条 latent mainline；只刷新 internal latent compare 并不能证明 learned latent 是 external gain driver。后续做 full holdout 前，必须先确认 summary artifact 显式记录 `latent_retriever_checkpoint_dir`，否则 measurement 口径不成立。
  - `v31` 的 full-holdout learned-memory gate 不能因为 `PersonaMem 1 / LongMemEval-S 1` cuda smoke 成功就默认视为可扩到 `500 + 512`。如果 CPU-backed 和 cuda-backed stable holdout 都长时间停在 `completed_predictions = 0`、且没有生成第一条 `predictions.jsonl`，就应诚实把它判成 true throughput blocker，而不是继续重复同一路径。
  - `v31` learned full-holdout 的 zero-progress stall 不一定来自 belief checkpoint 本身；如果每条样本都重复解析 `latent_retriever.pt` 并且只在所有 pending rows 都拿到 provider 结果后才第一次落盘，run 会看起来像“完全卡死”。把 `latent_slot_ranker` 升级成 run-level shared predictor，并按 provider batch drain `pending_rows` 后，即使 full `500 + 512` gate 也能恢复成稳定增量推进。
  - `PersonaMem 512` learned full-holdout 在共享繁忙 GPU 上可能会在 shared belief predictor 初始化阶段 OOM；这不等于 learned path 本身坏掉。优先把 `CUDA_VISIBLE_DEVICES` 切到空闲卡，再复用相同 checkpoint / config，更符合“先排设备竞争、再判断代码是否退化”的顺序。
- 2026-04-20:
  - `v30` / `v31` 的 publish 入口当前仍会刷新各自的 `latest_stage2_v30_*` / `latest_stage2_v31_*` aliases；如果直接在 repo `outputs_v2/` 上运行某些 `v32` 中间 publisher，旧 retained aliases 会被连带覆盖。更稳的做法是先把这类 `v32` 中间 run 放到隔离的 `--output-root`，只在结果确认后把新的 `latest_stage2_v32_*` artifact 拷回主 `outputs_v2/artifacts/`。
  - `v32` answer / option-scoring 路线最好把 label normalization、candidate body 解析与 lexical projection 抽成通用模块，而不是继续把 Persona-specific 选项逻辑散落在 canary runner 里；这样后续做 full holdout compare 与 ablation 时，answer head 证据才是模块级，而不是 benchmark-side patch。
  - `run_stage2_memory_canary.py` 当前会无条件复用 `outputs_v2/evals_benchmark/` 下最新的 `*_canary.json` manifest；如果仓库里已经有旧的 `12`-sample 或 `64`-sample manifest，而这轮目标其实是 `500 / 512` full holdout，就必须先显式生成新的 full manifests。否则即使你把 run dir 命名成 `full_*`，实际 sample_count 也可能完全不对。
  - 同一个 runner 的 `--limit` 默认值是 `1`，所以“manifest 已经是 full size”并不等于 run 会自动吃满；做 full holdout 时必须把 `--limit 500` 或 `--limit 512` 显式写进命令，否则只会得到一个误导性的 `1/1 completed`。
  - symbolic + provider-configured + `provider_workers > 1` 的旧 fast path 会把“memory build”与“provider request”一起扔进并行 worker；在 `500 / 512` 这种 large holdout 上，这条 fast path 比顺序 build + batched provider 更容易表现成长时间不落盘。把 fast path 限制回 `<=64` canary 后，大 holdout 才重新恢复到增量推进。
  - 对当前 `MiniMax-M2.7` provider 来说，large holdout 的主要外部约束不是单次请求完全不可用，而是会周期性返回 `HTTP 503`。提高 `max_retries` / `retry_backoff_seconds` 不能直接消灭这个问题，但可以显著提升一次 resumed run 的有效推进长度；在 current session 里，它已经把 `PersonaMem 512` partial progress 从 `13` 提到 `37`。
