# TODO

## Doing

- `TD-006` `[doing]` 接入 PersonaMem 32k 官方主任务与评测协议。
  - Reason: PersonaMem 是第一阶段主 benchmark，也是最接近 personalized memory 设定的硬要求。
  - Evidence target: 官方文件可读、主任务入口可运行、官方 protocol 对齐。
  - Current evidence: 官方 `32k` 数据已下载；真实 1-sample run 成功，最小 `evaluation.json` 为 `accuracy=1.0`。

- `TD-008` `[doing]` 接入 LongMemEval-S 官方协议。
  - Reason: LongMemEval-S 是第一阶段次 benchmark，也是 memory update / temporal reasoning 的关键诊断集。
  - Evidence target: 官方 S 版文件可读、评测流程可运行。
  - Current evidence: 官方 `S` 数据已下载；阿里云路径曾被 `HTTP 403 AllocationQuota.FreeTierOnly` 阻断；用户批准的 Gemini 2.5 Flash 路径已完成真实 1-sample run，formal run 在 `outputs/runs/20260405T042236Z_longmemeval/` 下推进到 `19/500` 后也受到持续 `HTTP 429` 外部限制。

- `TD-010` `[doing]` 建立实验总入口与分步骤脚本。
  - Reason: 第一阶段必须具备复现实验的统一入口和分步骤脚本。
  - Evidence target: 总入口和分步骤脚本都可用于复现实验。
  - Current evidence: `scripts/prepare_data.py`、`scripts/evaluate_run.py`、`scripts/make_tables.py` 已落地；`scripts/run_experiment.py` 已支持 `--run-dir` / `--resume` 与增量 prediction 落盘，仍待正式 protocol 全量结果。

- `TD-011` `[doing]` 建立 unit tests 与最小 E2E smoke test。
  - Reason: 初始测试已建立，但需要随核心模块扩展持续补齐。
  - Evidence target: 核心模块覆盖更完整，回归测试可持续使用。
  - Current evidence: `pytest` 通过（31 tests）。

## Ready

- `TD-015` `[ready]` 搭建第二阶段 `V2.0` 的 observation normalization、belief JSON 目标生成与公开数据集接入骨架。
  - Reason: 第二阶段设计已锁定，下一步必须把数据规范、parser、样本构造和公开数据集接入落成最小实现骨架。
  - Evidence target: observation schema、belief schema、数据 loader、parser skeleton、`outputs_v2/` 目录与对应测试骨架落地。

## Backlog

- `TD-007` `[backlog]` 为 PersonaMem 128k / 1M 提供启动开关。
  - Reason: 第一阶段只要求保留启动能力。

- `TD-016` `[backlog]` 实现第二阶段 `V2.0` 主线模型骨架。
  - Reason: 需要把已锁定的方法设计落成可训练代码。
  - Evidence target: slot encoder、retrieval key head、lifecycle module、core/residual banks、consolidation、light resampler、Flan-T5 belief decoder 就位。

- `TD-017` `[backlog]` 建立第二阶段本地 intrinsic evaluation 管线与 budget sweep。
  - Reason: 第二阶段采用 local-first protocol，需要先在本地完成 memory-specific 指标和 ablation。
  - Evidence target: local eval 脚本、指标表格、budget sweep 配置与输出。

- `TD-018` `[backlog]` 建立第二阶段 benchmark canary protocol 与结果记录。
  - Reason: 正式 benchmark API 昂贵，需要先走 fixed canary subset。
  - Evidence target: PersonaMem 64 / LongMemEval-S 64 的固定 canary 子集、脚本、输出表和记录规范。

## Blocked

- `TD-012` `[blocked]` 补齐 stage-1 外部前置条件。
  - Reason: 当前可用的 Gemini key/provider 组合在 formal benchmark 负载下仍连续触发 `HTTP 429`，无法继续推进正式结果。
  - Evidence target: 可持续推进 formal benchmark 的 provider 配额、稳定 key，或用户批准的其他可用 provider。

- `TD-013` `[blocked]` 将两个 benchmark 从 1-sample real run 推进到正式全量 protocol 运行。
  - Reason: 当前只能证明最小链路打通，尚不足以满足 AC-002 / AC-003 的正式运行要求。
  - Evidence target: PersonaMem 与 LongMemEval-S 在正式范围内完成可重复结果运行。
  - Current evidence: runner 已具备增量落盘与续跑能力；Gemini 路径已把 PersonaMem formal run 推进到 `22/589`、把 LongMemEval formal run 推进到 `19/500`，但超保守单样本检查仍连续触发 `HTTP 429`，说明当前 key/provider 组合已构成真实外部 blocker。

## Done

- `TD-001` `[done]` 初始化项目文档系统并建立根契约、真源文档和状态文档。
- `TD-002` `[done]` 用 `environment.yaml` 在默认 conda envs 目录创建并固定 conda 环境 `core_mem`。
- `TD-003` `[done]` 建立基础目录结构、配置机制与输出目录规范。
- `TD-004` `[done]` 确定第一阶段 pretrained embedding model 为 `sentence-transformers/all-MiniLM-L6-v2`。
- `TD-005` `[done]` 实现阿里云 OpenAI-compatible provider adapter 第一版。
- `TD-009` `[done]` 实现 vanilla CoRe Memory 核心模块与文本 embedding 接口。
- `TD-014` `[done]` 锁定第二阶段 `V2.0 structured latent-slot memory` 方案并同步真源与状态文档。
  - Reason: 第二阶段已由用户确认，需要进入文件级 runtime truth。
  - Evidence target: `docs/requirements.md`、`docs/v2_design.md` 与相关状态文档同步到 stage-2 真相。

## Verified

- `TD-003` `[verified]` 目录结构、配置加载、输出规范、run metadata 写入与 benchmark dry-run 可运行。
- `TD-002` `[verified]` `core_mem` 环境已在默认 conda envs 目录创建，并通过 `python --version`、`pytest` 与 `run_experiment.py --verify-only` 复验。
- `TD-004` `[verified]` embedding 配置、adapter 和环境依赖已接入；`sentence-transformers 5.3.0` 可在 `core_mem` 环境中 import。
- `TD-005` `[verified]` Provider adapter 通过单测，支持 OpenAI-compatible 请求构造与响应解析。
- `TD-010` `[verified]` 最小 `prepare / evaluate / make tables` 分步骤脚本已落地并有 smoke tests。
- `TD-009` `[verified]` vanilla CoRe Memory 已具备 writer / updater / residual manager / reader / text observe/query 最小闭环。

## Abandoned

- None
