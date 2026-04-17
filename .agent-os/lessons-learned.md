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
