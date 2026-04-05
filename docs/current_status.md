# Current Status

## 当前结论

- 项目已完成需求冻结与项目系统初始化准备。
- 第一阶段方法已锁定为 `vanilla CoRe Memory`。
- 第一阶段 benchmark 已锁定为：
  - 主：`PersonaMem`
  - 次：`LongMemEval-S`
  - 未来扩展：`LoCoMo`

## 当前状态

- 文档状态：initialized
- 代码状态：platform skeleton + benchmark dry-run ready
- 环境状态：specified but not yet materialized as conda env
- benchmark 适配状态：PersonaMem / LongMemEval-S official file formats can be loaded in dry-run mode
- 测试状态：initial unit tests + E2E smoke test passing

## 当前最重要的下一步

- 创建并固定 conda 环境 `core_mem`，然后继续推进 PersonaMem 32k 的正式数据接入与官方协议运行。

## 关键约束

- 真源文档为 `docs/requirements.md`
- 第一阶段不跑 baseline
- 第一阶段接受 negative result
- 第一阶段重点是系统完整、实验可运行、结论可信

## 风险与待定项

- 具体 embedding model 尚未锁定
- conda 在当前机器上存在 plugin/权限异常，后续环境创建需要绕过插件或调整命令方式
- PersonaMem 与 LongMemEval-S 目前仅完成官方文件格式 dry-run，尚未完成正式评测运行
- PersonaMem `128k / 1M` 只要求保留启动能力，第一阶段不做正式结果
- 图不是第一阶段硬要求，但结果数据与表格是硬要求
