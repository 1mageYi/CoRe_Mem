# CoRe_Mem Agent Contract

本文件是项目根目录的运行契约。任何新 session 或新 agent 进入仓库后，必须先读本文件，再按规定顺序恢复上下文。

## 项目定位

本项目是一个以研究为核心的代码仓库。

- 第一阶段目标：实现并验证一个 bounded cross-session personalized memory framework，其具体实例为 `vanilla CoRe Memory`，并在指定 benchmark 上完成可复现的实验运行。
- 第二阶段目标：在保留第一阶段平台和 v1 naive baseline 参考价值的前提下，推进 `V2.0 structured latent-slot memory` 研究路线。

## 最高优先级真源文档

项目最高优先级真源文档为：

- [`docs/requirements.md`](/home/image/workspace/CoRe_Mem/docs/requirements.md)

规则：

1. `docs/requirements.md` 是关于目标、范围、硬约束、分阶段验收标准与非目标的唯一最高真源。
2. 若 `docs/requirements.md` 与其他任何文档冲突，以 `docs/requirements.md` 为准。
3. 未经用户明确要求，不得擅自修改其中的最终目标、第一阶段目标、第二阶段目标、任一阶段验收标准、非目标和硬约束。
4. 若当前任务与 `docs/requirements.md` 未对齐，或该文档尚未覆盖相关目标与验收标准，必须先指出缺口，不得自行扩展范围。

## 恢复顺序

1. 读取本文件 `AGENTS.md`
2. 读取 `docs/requirements.md`
3. 读取 `.agent-os/project-index.md`
4. 读取当前活跃项指向的相关文档
5. 读取 `.agent-os/run-log.md`

不得默认依赖聊天历史进行恢复。

## 文档系统职责划分

### `docs/`

面向人类的主文档目录，至少包含：

- `requirements.md`
- `current_status.md`
- `implementation_plan.md`
- `todo.md`

职责：

- `docs/requirements.md`
  - 项目真源文档
  - 定义目标、范围、约束、验收标准
- `docs/current_status.md`
  - 面向人类的当前状态摘要
- `docs/implementation_plan.md`
  - 面向人类的实现计划和阶段划分
- `docs/todo.md`
  - 面向人类的任务视图
  - 必须与 `.agent-os/todo.md` 保持同步

### `.agent-os/`

面向 agent 的状态目录，至少包含：

- `project-index.md`
- `change-decisions.md`
- `architecture-milestones.md`
- `todo.md`
- `acceptance-report.md`
- `lessons-learned.md`
- `run-log.md`

职责：

- `.agent-os/project-index.md`
  - agent 恢复主入口
  - 必须始终暴露一个全局 top next action
- `.agent-os/todo.md`
  - agent 真实任务状态机来源
- `.agent-os/run-log.md`
  - 最近工作记录
- `.agent-os/acceptance-report.md`
  - 验收证据与通过/未通过状态
- `.agent-os/lessons-learned.md`
  - 失败探索、陷阱、重试条件
- `.agent-os/change-decisions.md`
  - 后续人工决策追加记录
- `.agent-os/architecture-milestones.md`
  - 当前架构、工作流、里程碑

## 双 TODO / 双状态文档同步规则

1. `.agent-os/todo.md` 是 agent 的真实状态机来源。
2. `docs/todo.md` 是面向人类的可读视图，必须与 `.agent-os/todo.md` 同步。
3. `.agent-os/project-index.md` 是 agent 恢复主入口。
4. `docs/current_status.md` 是面向人类的解释型摘要。
5. 若这些文档冲突：
   - 先以 `docs/requirements.md` 为准
   - 再以 `.agent-os/project-index.md` 和 `.agent-os/todo.md` 作为 runtime truth
   - 面向人类的文档必须随后同步修正

## 文档更新纪律

出现以下情况时，必须更新对应文档，而不是只改代码：

- 新 TODO 创建
- TODO 状态变化
- blocker 出现或解除
- milestone 增减、完成或放弃
- 产生新证据
- 发生失败探索
- 一次工作 session 结束

最低要求：

- 状态变化更新 `.agent-os/todo.md`
- 重要变化更新 `.agent-os/project-index.md`
- 新证据更新 `.agent-os/acceptance-report.md`
- 失败探索更新 `.agent-os/lessons-learned.md`
- session 结束更新 `.agent-os/run-log.md`
- 面向人类的重要状态变化同步到 `docs/current_status.md` 与 `docs/todo.md`

## Truthfulness Guardrail

不得无证据声称以下状态：

- complete
- verified
- benchmark passed
- acceptance met
- reproducible
- fixed

若尚未验证，必须明确标注为：

- hypothesis
- partial
- unverified
- blocked
- suspected

失败探索是项目知识的一部分，不得因为结果不好看而省略。

## 阶段边界

第一阶段只做：

- 实现 `vanilla CoRe Memory`
- 跑通 `PersonaMem` 官方主任务，优先级最高
- 跑通 `LongMemEval-S`，优先级次之
- 建立可复现实验流程
- 产出结果数据、主结果表和可追溯实验记录
- 建立测试与恢复体系

第一阶段明确不做：

- baseline 在本 repo 内的运行与整合
- performance superiority claim
- assignment randomization ablation
- spectral / multi-timescale extension
- 重型训练
- 训练独立 embedding model
- LongMemEval-M
- LoCoMo 正式实验
- 报告写作作为硬验收

第二阶段当前主线为：

- `V2.0 structured latent-slot memory`
- `structured latent slots` 而非文本 chunk RAG
- `core + residual` 双银行 memory
- `latent composition -> belief decoding -> answer projection`
- 主 decoder 路线为 `Flan-T5-based belief JSON decoder`
- 训练策略为 `public-datasets-first, synthetic-minimal`
- benchmark 继续作为 evaluation source，而非 primary training source

第二阶段默认保留：

- 第一阶段 v1 作为 naive baseline 与可追溯参考
- 第一阶段平台、脚本、结果与状态文档

当前执行规则：

1. 当前执行主线为第二阶段 `V2.0 structured latent-slot memory`。
2. 第一阶段 formal benchmark 在用户明确要求 AI 去跑之前，默认视为 pending，不主动继续消耗 API。
3. `.agent-os/project-index.md` 与 `.agent-os/todo.md` 必须反映“stage-2 主线 / stage-1 待触发”的 runtime truth。

## 环境与外部依赖规则

1. 固定使用 conda 环境：`core_mem`
2. Python 版本固定：`3.10`
3. 默认外部 LLM provider 为阿里云兼容 OpenAI API
4. 默认 `base_url`：`https://dashscope.aliyuncs.com/compatible-mode/v1`
5. 默认模型：`qwen3-235b-a22b-instruct-2507`
6. API key 不得写入仓库文件；仅允许通过环境变量或本地 `.env` 管理
7. 若需更换 provider、主要模型或核心环境版本，必须先得到用户明确批准

## 外部目录修改规则

1. 允许为了创建、更新或删除 `core_mem` conda 环境而修改默认 conda envs 目录中的环境路径与 conda 自身的必要元数据路径。
2. 除上述环境相关路径外，任何 repo 工作区之外的目录一律视为外部目录。
3. 对任何外部目录的写入、删除、重命名、移动或配置修改，必须先得到用户明确批准。
4. 若外部路径修改并非创建或管理 `core_mem` 环境所必需，也不得因为当前 sandbox 已放开而擅自执行。

## 测试纪律

1. 核心模块必须有 unit tests
2. 至少保留一条最小 E2E smoke test
3. 每次改动后必须通过相关测试
4. 未通过测试的改动不得视为完成

## 结果与输出规范

1. 所有实验结果统一放在 `outputs/`
2. 子目录约定：
   - `outputs/runs/`
   - `outputs/tables/`
   - `outputs/figures/`
   - `outputs/logs/`
3. 必须保存：
   - 原始 prediction outputs
   - config snapshot
   - prompt version
   - model name
   - run timestamp
   - commit hash
4. 图不是第一阶段硬要求，但结果数据和表格是硬要求
5. 第二阶段训练、评测与分析产物统一放在 `outputs_v2/`

## 语言规则

- 回答和文档默认使用中文，风格直接、准确、可追溯
- 所有代码注释、`print` 和日志输出默认使用英文
- 汇报代码改动时，优先说明行为变化、验证方式和未覆盖风险，不要只罗列文件名

## 需要向用户升级确认的情况

以下情况必须先问用户，不得自行决定：

1. 需要修改 `docs/requirements.md` 的核心条款
2. 需要扩大任何已锁定阶段的范围
3. 需要切换主要 benchmark 目标
4. 需要切换主要 API provider 或主模型
5. 需要引入重型训练或超出当前资源约束的方案
6. 需要改变任一阶段验收标准
7. 遇到无法从文件和现有代码中推断的人类判断问题
8. 需要修改除 conda 环境相关路径外的任何外部目录

## 默认执行原则

在不触发升级条件时，agent 应主动推进项目，包括：

- 环境搭建
- 代码实现
- 配置完善
- 测试补齐
- 文档同步
- 实验运行
- 结果整理

但所有推进必须严格受 `docs/requirements.md` 限制。
