# TODO（人类可读视图）

> 与 `.agent-os/todo.md` 同步。

## 进行中

- Writer fact 提取逻辑（LLM prompt-based extraction）

## 就绪

- 实现阿里云 OpenAI-compatible provider adapter
- 接入 PersonaMem 32k 官方主任务与评测协议

## 待办

- PersonaMem 128k / 1M 启动开关
- 接入 LongMemEval-S 官方协议
- 建立实验总入口与分步骤脚本
- 实现 Reader（query → top-k core + top-j residual 检索）
- 训练 vec2text 模型解码 latent 信息

## 已完成

- 初始化项目文档系统
- 创建环境管理方案（venv，Python 3.10）
- 建立基础目录结构、配置机制
- 确定 embedding model（`all-MiniLM-L6-v2`）
- **实现 write path 核心模块**：slot、embedding、residual_manager、core_updater、writer
- **建立 unit tests**：28 个测试全部通过
