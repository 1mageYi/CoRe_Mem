# TODO

## Doing

- `TD-006` 接入 PersonaMem 32k 官方主任务与评测协议。
- `TD-008` 接入 LongMemEval-S 官方评测协议。
- `TD-010` 建立实验总入口与分步骤脚本。
- `TD-011` 建立并扩展 unit tests 与最小 E2E smoke test。

## Ready

- `TD-015` 搭建第二阶段 `V2.0` 的 observation normalization、belief JSON 目标生成与公开数据集接入骨架。

## Blocked

- `TD-012` 补齐 stage-1 外部前置条件：当前 formal benchmark 继续受 provider `HTTP 429` 阻断。
- `TD-013` 将 PersonaMem / LongMemEval-S 从 1-sample real run 推进到正式全量 protocol 运行（当前分别停在 `22/589` 和 `19/500`）。

## Backlog

- `TD-007` 为 PersonaMem 128k / 1M 预留启动开关。
- `TD-016` 实现第二阶段 `V2.0` 主线模型骨架：slot encoder、lifecycle、core/residual、resampler、belief decoder。
- `TD-017` 建立第二阶段本地 intrinsic evaluation 管线与 budget sweep。
- `TD-018` 建立第二阶段 benchmark canary protocol 与结果记录。

## Done

- `TD-001` 初始化项目文档系统与根契约文件。
- `TD-002` 在默认 conda envs 目录创建并固定 conda 环境 `core_mem`（Python 3.10）。
- `TD-003` 建立基础目录结构、配置机制与输出目录规范。
- `TD-004` 确定第一阶段 pretrained embedding model 为 `sentence-transformers/all-MiniLM-L6-v2`。
- `TD-005` 实现阿里云 OpenAI-compatible provider adapter 的第一版。
- `TD-009` 实现 vanilla CoRe Memory 核心模块与文本 embedding 接口。
- `TD-014` 锁定第二阶段 `V2.0 structured latent-slot memory` 方案，并将真源、设计文档和状态文档同步到 stage-2 真相。

## Notes

- 真实 runtime 状态以 `.agent-os/todo.md` 为准。
- 本文件用于面向人类的同步视图。
- 第一阶段剩余未过项仍只包括正式 benchmark 两项。
- 第二阶段当前已完成方法与治理层锁定，尚未开始代码实现。
- 第二阶段主线采用：
  - `Light Cross-Attention Resampler`
  - `Flan-T5 belief JSON decoder`
  - `public-datasets-first, synthetic-minimal`
  - `benchmark is evaluation source, not primary training source`
