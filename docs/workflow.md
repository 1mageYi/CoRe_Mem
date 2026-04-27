# Project Workflow (Coding SOP)

## 目的
这是本项目的固定改代码工作流文档。  
每次实现、重构、加实验、改参数，都按此流程执行。

## 1) 开工前必读文档顺序
1. `docs/requirements.md`（目标、约束、验收标准）
2. `docs/todo.md`（当前要做的任务与优先级）
3. `docs/current_status.md`（已做决策、风险、最近进展）
4. `docs/structure.md`（当前整体架构）
5. `docs/implementation_plan.md`（实现策略与参数）

## 2) 开发执行规则
- 先从 `todo` 中选一项最小可交付任务，不跨太多模块同时改。
- 优先保持模块解耦：
  - 写入（Add）与检索（Search）分开
  - Extractor 与 Graph Store 分开
  - Rerank 与 Prompt Builder 分开
- 任何新逻辑必须可消融（可开关/可对照）。
- 新增配置必须参数化，避免硬编码在主流程。

## 3) 改动后文档回写（强制）
完成代码改动后，必须同步更新：
- `docs/current_status.md`
  - 写明本次改了什么、为什么改、结果如何、下一步是什么
- `docs/todo.md`
  - 已完成项打勾，新增任务补充，失效任务移除
- `docs/structure.md`（仅当架构变化时）
  - 模块/链路改变才更新
- `docs/implementation_plan.md`（仅当方案变化时）
  - 新规则、新参数、流程变更时更新

## 4) 依赖管理规则（强制）
若新增/升级依赖，必须同步更新三处：
1. `requirements.txt`
2. `pyproject.toml`
3. `uv.lock`

并在 `current_status` 记录：
- 依赖变更原因
- 影响模块
- 是否需要迁移/重装环境

## 5) 实验与评测规则
- 主评测：`LoCoMo`（长上下文对话记忆）
- 对照 / 回归：`PERMA`
- 最终迁移验证：`PersonaMem`

每次实验至少记录：
- 配置（关键参数）
- 指标结果
- 与 baseline 对比
- 失败样本与可能原因

主评测默认以 **LoCoMo（长对话记忆）** 为准；PERMA 可作对照跑法。

## 6) 提交前检查清单
- [ ] 代码能运行（最小路径）
- [ ] 新增逻辑有对应文档回写
- [ ] 参数有默认值与说明
- [ ] 依赖变更已三处同步
- [ ] todo/current_status 已更新

## 7) 禁止项
- 不记录“只改代码不改文档”的提交
- 不允许把实验临时脚本混入主流程
- 不允许把 benchmark-specific hack 写成通用逻辑且不标注
