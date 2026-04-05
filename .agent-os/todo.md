# TODO

## Doing

- `TD-002` `[doing]` 创建并固定 conda 环境 `core_mem`，Python 版本为 `3.10`。
  - Reason: 第一阶段要求固定专属环境，且后续 benchmark 运行必须从冻结环境启动。
  - Evidence target: `core_mem` 环境存在，可安装项目依赖，可运行基础测试。

- `TD-006` `[doing]` 接入 PersonaMem 32k 官方主任务与评测协议。
  - Reason: PersonaMem 是第一阶段主 benchmark，也是最接近 personalized memory 设定的硬要求。
  - Evidence target: 官方文件可读、主任务入口可运行、官方 protocol 对齐。

- `TD-008` `[doing]` 接入 LongMemEval-S 官方协议。
  - Reason: LongMemEval-S 是第一阶段次 benchmark，也是 memory update / temporal reasoning 的关键诊断集。
  - Evidence target: 官方 S 版文件可读、评测流程可运行。

- `TD-010` `[doing]` 建立实验总入口与分步骤脚本。
  - Reason: 目前已有总入口与 benchmark dry-run，但还缺 prepare/eval 等分步骤脚本。
  - Evidence target: 总入口和分步骤脚本都可用于复现实验。

- `TD-011` `[doing]` 建立 unit tests 与最小 E2E smoke test。
  - Reason: 初始测试已建立，但需要随核心模块扩展持续补齐。
  - Evidence target: 核心模块覆盖更完整，回归测试可持续使用。

## Ready

- `TD-004` `[ready]` 调研并确定第一阶段使用的 pretrained embedding model。

## Backlog

- `TD-007` `[backlog]` 为 PersonaMem 128k / 1M 提供启动开关。
- `TD-009` `[backlog]` 实现 vanilla CoRe Memory 核心模块。

## Blocked

- None

## Done

- `TD-001` `[done]` 初始化项目文档系统并建立根契约、真源文档和状态文档。
- `TD-003` `[done]` 建立基础目录结构、配置机制与输出目录规范。
- `TD-005` `[done]` 实现阿里云 OpenAI-compatible provider adapter 第一版。

## Verified

- `TD-003` `[verified]` 目录结构、配置加载、输出规范、run metadata 写入与 benchmark dry-run 可运行。
- `TD-005` `[verified]` Provider adapter 通过单测，支持 OpenAI-compatible 请求构造与响应解析。

## Abandoned

- None
