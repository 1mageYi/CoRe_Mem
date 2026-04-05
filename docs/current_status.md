# Current Status

## 当前结论

- 项目已完成需求冻结与项目系统初始化准备。
- 第一阶段方法已锁定为 `vanilla CoRe Memory`。
- 第一阶段 benchmark 已锁定为：
  - 主：`PersonaMem`
  - 次：`LongMemEval-S`
  - 未来扩展：`LoCoMo`

## 当前状态

- 文档状态：initializing
- 代码状态：not started
- 环境状态：not started
- benchmark 适配状态：not started
- 测试状态：not started

## 当前最重要的下一步

- 初始化 repo 文档系统并建立第一批执行项，使后续实现可以直接围绕 `docs/requirements.md` 推进。

## 关键约束

- 真源文档为 `docs/requirements.md`
- 第一阶段不跑 baseline
- 第一阶段接受 negative result
- 第一阶段重点是系统完整、实验可运行、结论可信

## 风险与待定项

- 具体 embedding model 尚未锁定
- PersonaMem `128k / 1M` 只要求保留启动能力，第一阶段不做正式结果
- 图不是第一阶段硬要求，但结果数据与表格是硬要求
