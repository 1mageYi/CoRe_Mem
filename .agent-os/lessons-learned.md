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
