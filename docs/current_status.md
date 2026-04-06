# Current Status

## 当前结论

- 项目系统初始化已完成（文档系统、目录结构、依赖管理、模块骨架）。
- 环境管理改用 venv（原 conda，用户决策 `CD-005`）。
- 第一阶段方法已锁定为 `vanilla CoRe Memory`。
- 第一阶段 benchmark 已锁定为：
  - 主：`PersonaMem`
  - 次：`LongMemEval-S`
  - 未来扩展：`LoCoMo`

## 当前状态

- 文档状态：done
- 目录与工程骨架：done
- 环境依赖文件：done（`pyproject.toml` + `requirements.txt`）
- 代码状态：skeleton only（占位模块已创建，逻辑未实现）
- benchmark 适配状态：not started
- 测试状态：smoke test 占位已创建

## 当前最重要的下一步

- 调研并确定第一阶段使用的 pretrained embedding model（`TD-004` / `RSK-001`）。

## 关键约束

- 真源文档为 `docs/requirements.md`
- 第一阶段不跑 baseline
- 第一阶段接受 negative result
- 第一阶段重点是系统完整、实验可运行、结论可信

## 风险与待定项

- 具体 embedding model 尚未锁定
- PersonaMem `128k / 1M` 只要求保留启动能力，第一阶段不做正式结果
- 图不是第一阶段硬要求，但结果数据与表格是硬要求
