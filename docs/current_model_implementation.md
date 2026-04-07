# 当前模型实现白盒说明（Stage-1 / Vanilla CoRe Memory）

## 文档目的

这份文档用于把当前 repo 中已经实现的 stage-1 模型、benchmark 主链路、memory 原型、实验入口和真实使用方式写清楚。

本文只描述当前代码库里的真实实现，不把“设计目标”“未来计划”和“已经接入正式推理的部分”混在一起。

结论先行：

- 当前 repo 已经有一个最小可运行的 `VanillaCoreMemory` 原型。
- 当前 repo 也已经有一条能真实调用外部 LLM、能生成预测、能评测、能续跑的 benchmark 主链路。
- 但这两部分目前还是“并列存在”，而不是“完全打通”。
- 更具体地说：当前正式 benchmark runner 主要还是把官方上下文直接拼进 prompt 发送给 LLM；`VanillaCoreMemory` 目前还没有接入 `run_experiment.py` 的正式推理循环。

这也是为什么我会把当前实现定义为：

- 系统平台已成型
- naive CoRe Memory 原型已落地
- benchmark live pipeline 已落地
- 但“memory-mediated benchmark inference” 仍是下一步最值得做的算法集成工作

## 1. 当前系统的真实结构

从 repo 的运行角度看，当前系统可以分成五层：

1. 配置层
- 负责固定 provider、模型、embedding、benchmark 路径、输出目录。
- 入口在 `src/core_mem/config.py`。

2. Benchmark adapter 层
- 负责把官方数据文件读进来，转换成统一的 `PromptRecord`。
- 入口在 `src/core_mem/benchmarks/personamem.py` 和 `src/core_mem/benchmarks/longmemeval.py`。

3. Provider 调用层
- 负责把 prompt 发给 OpenAI-compatible 接口，处理 rate limit retry 和 pacing。
- 入口在 `src/core_mem/providers/openai_compatible.py`。

4. Memory 原型层
- 负责最小版 writer、core updater、residual slots、reader，以及文本 embedding 接口。
- 入口在 `src/core_mem/memory/core.py`。

5. 运行与评测层
- 负责 dry-run、正式 run、增量落盘、resume、evaluation、summary table。
- 入口在 `scripts/run_experiment.py`、`scripts/evaluate_run.py`、`scripts/make_tables.py`、`scripts/run_until_complete.py`。

这五层里，1/2/3/5 已经组成了当前正式 benchmark 主链路；第 4 层已经实现，但还主要以独立模块和单元测试形式存在。

## 2. 当前 benchmark 主链路到底怎么跑

### 2.1 配置加载

`load_project_config()` 会从 YAML 读出 `ProjectConfig`，其中包括：

- `llm`
  - provider
  - api key 环境变量名
  - base url
  - model
  - temperature
  - max tokens
  - timeout
  - retry / pacing 参数
- `embedding`
  - 当前固定为 `sentence-transformers/all-MiniLM-L6-v2`
- `benchmarks`
  - PersonaMem 数据根目录、variant、task
  - LongMemEval-S 数据根目录、subset
- `outputs`
  - 输出根目录

默认配置在 `configs/defaults.yaml`，默认 provider 是阿里云 OpenAI-compatible 路径；用户批准后又增加了 `configs/gemini_flash.yaml` 和 `configs/gemini_flash_ultraslow.yaml` 作为 Google AI Studio 的替代路径。

### 2.2 Benchmark adapter 的职责

当前 benchmark adapter 只做两件事：

1. 读取官方数据
2. 把每个样本渲染成一个可直接送给 LLM 的 prompt

统一输出结构是 `PromptRecord`，字段包括：

- `sample_id`
- `benchmark`
- `prompt`
- `expected_answer`
- `metadata`

这意味着当前主链路本质上是：

官方样本 -> adapter 渲染 prompt -> provider 调 LLM -> 存 prediction -> exact match 评测

这里并没有出现“先写入 memory，再从 memory 读出 supporting evidence，再构造 answer prompt”的中间层。

### 2.3 PersonaMem 当前实现

`PersonaMemAdapter` 的工作方式非常直接：

- 读 `questions_32k.csv`
- 读 `shared_contexts_32k.jsonl`
- 用 `shared_context_id` 找到共享上下文
- 根据 `end_index_in_shared_context` 截断上下文
- 把上下文渲染成文本
- 把 multiple-choice 选项拼到 prompt 里

当前 PersonaMem prompt 的核心结构是：

- 背景说明：这是一个 PersonaMem multiple-choice personalization 问题
- `Context`
- `User query`
- `Options`
- 一个回答约束

回答约束分两种：

- 如果选项是普通文本，就要求“返回 exact best option text”
- 如果官方选项是 `(a)`、`(b)` 这种 label 格式，就要求“只返回 option label”

这一点是当前 repo 很重要的 protocol 对齐点，因为官方数据里确实存在 label-answer setting，测试也专门检查了这件事。

### 2.4 LongMemEval-S 当前实现

`LongMemEvalAdapter` 也很直接：

- 读 `longmemeval_s_cleaned.json`
- 把 `haystack_sessions` 渲染成带 session 分段的 history
- 拼入 question date 和 question
- 明确要求模型“只返回 exact answer phrase，不要加 article、verb 或额外词”

这代表当前 LongMemEval-S 也是典型的 full-history prompt，而不是 memory retrieval prompt。

### 2.5 Provider 调用层

`OpenAICompatibleProvider` 负责：

- 读取 API key 环境变量
- 构造 `/chat/completions` 请求
- 发送单轮 user prompt
- 可选插入 system prompt
- 支持 `max_tokens`
- 对 `429` 和 `5xx` 做有限重试
- 支持 `Retry-After`
- 支持最小请求间隔 `min_request_interval_seconds`
- 支持 retry delay cap

所以从工程角度看，当前平台已经考虑到了：

- 接口兼容性
- 配额抖动
- 节流
- 长跑 benchmark 的稳定性

这也是当前 formal run 能推进到部分样本的原因。

### 2.6 Runner、resume 和输出

`scripts/run_experiment.py` 是总入口。它会：

1. 读取 benchmark records
2. 创建或解析 `run_dir`
3. 拷贝 `config_snapshot.yaml`
4. 生成 `run_metadata.json`
5. 若 `dry_run=true`，只打印元数据
6. 若正式运行，则逐条调用 provider
7. 逐条把结果追加写入 `predictions.jsonl`
8. 若启用 `--resume`，会跳过已完成的 `sample_id`

输出元数据目前包括：

- benchmark
- record_count
- dry_run
- config path
- config snapshot path
- model
- provider
- run timestamp
- commit hash
- sample id preview
- prompt preview
- expected preview
- prompt version
- resumed prediction count

这是当前 repo 在“可追溯实验”方面最成熟的一块。

`scripts/evaluate_run.py` 目前采用最简单的 exact-match accuracy。

`scripts/make_tables.py` 会收集所有带 `evaluation.json` 的 run，生成：

- `outputs/tables/stage1_summary.json`
- `outputs/tables/stage1_summary.csv`

`scripts/run_until_complete.py` 是外层 supervisor，用于：

- 调起 `run_experiment.py`
- crash 或 rate limit 后冷却
- 从同一个 `run_dir` 继续续跑
- 在 chunked limit 下缓慢推进

对于当前这种 provider 容量不稳定的环境，这个 supervisor 是有实际价值的。

## 3. 当前 Vanilla CoRe Memory 原型到底是什么

### 3.1 当前实现的定位

当前 `VanillaCoreMemory` 不是一个完整论文级 memory framework，而是一个非常清楚的、最小闭环的 stage-1 原型。

它已经具备以下结构：

- writer
- core updater
- residual slot manager
- reader
- text embedding interface

但它还没有这些东西：

- benchmark 内真正的 memory write/read orchestration
- learned writer policy
- learned gating / scorer / projector
- temporal decay
- cross-session identity management
- retrieval-conditioned final prompt builder
- memory state persistence / reload for official runs

所以当前实现更准确地说，是“memory kernel 已实现，benchmark orchestration 未接上”。

### 3.2 数据结构

当前 memory 层的主要数据结构有：

`MemoryObservation`
- 一次写入事件
- 字段：`latent`、`timestamp`、`salience`、`support`

`ResidualSlot`
- 一条残差记忆槽
- 字段与 observation 基本一致：`latent`、`timestamp`、`salience`、`support`

`CoreMemoryState`
- 当前 memory 状态
- 包含：
  - `core_vector`
  - `residual_slots`

`MemoryReadResult`
- 一次读取结果
- 包含：
  - `core_vector`
  - `supporting_slots`

这里的设计是非常“latent + evidence pointer”导向的：

- `core_vector` 代表压缩后的主记忆
- `residual_slots` 代表保留下来的显性记忆碎片
- `support` 代表这条 latent 对应的文本证据或标签

### 3.3 Core updater

当前 `CoreMemoryUpdater` 是一个固定学习率的指数平滑更新器，不是训练出来的。

更新公式可以写成：

`c_t = (1 - lr) * c_(t-1) + lr * z_t`

其中：

- `c_t` 是新的 core vector
- `z_t` 是当前 observation latent
- 默认 `learning_rate = 0.5`

如果向量维度不一致，代码会先 pad 到同一维度再更新。

这件事很朴素，但也说明了当前 stage-1 原型的性质：

- 它不是一个复杂的记忆选择器
- 它没有显式区分“应该进 core”还是“不应该进 core”
- 它默认每次 observation 都会影响 core

这很适合作为第一阶段的 naive baseline implementation，但不适合作为最终算法形态。

### 3.4 Residual manager

`ResidualMemoryManager` 的逻辑是：

- 把新 slot 加入已有 slots
- 按 `(salience, timestamp)` 降序排序
- 保留前 `max_slots`

默认：

- `max_slots = 4`

这说明当前 residual memory 是一个固定容量、按显著性优先的 top-k 容器。

它的优点是：

- 简单
- 可控
- deterministic

它的局限也很明显：

- 没有 novelty 判断
- 没有 redundancy merge
- 没有 topic diversity
- 没有 recency-decay tradeoff
- 没有 session-aware budget allocation

### 3.5 Writer

`MemoryWriter.write()` 的行为是：

1. 用 updater 更新 `core_vector`
2. 如果 `observation.salience >= residual_salience_floor`
3. 就把 observation 复制成一个 `ResidualSlot`
4. 交给 residual manager 截断

默认：

- `residual_salience_floor = 0.0`

因此在默认配置下，所有 observation 都会进 residual ranking，只是最后可能因预算上限被挤掉。

这体现出当前 writer 还没有真正的“写入决策能力”，只是：

- core 永远更新
- residual 近似永远候选

### 3.6 Reader

`MemoryReader.read()` 的逻辑是：

- 计算 query latent 与每个 residual slot latent 的点积
- 再用 `(dot product, salience, timestamp)` 排序
- 返回 top-k supporting slots

默认：

- `top_k = 3`

所以当前 reader 不是：

- cross-attention reader
- learned retriever
- multi-hop retriever

而是一个最小的 dense retrieval 近似器。

### 3.7 Text interface

`VanillaCoreMemory` 支持：

- `observe_text(text, timestamp, salience, support)`
- `query_text(text)`

这两者依赖 `TextEmbedder` 接口。

当前真正接好的 embedder 是 `SentenceTransformerEmbedder`，默认模型：

- `sentence-transformers/all-MiniLM-L6-v2`

工作方式是：

- 第一次调用时 lazy load 模型
- `encode(text, normalize_embeddings=True)`
- 输出 float list

也就是说，当前 memory latent 的来源已经明确对齐到：

- pretrained embedding model
- training-light
- 非自训练 embedding

这与第一阶段 requirements 是一致的。

## 4. 当前实现和“真正的 CoRe Mem benchmark 推理”之间的差距

这是这份文档里最重要的一节。

### 4.1 当前 benchmark 不是 memory-mediated inference

现在的正式 benchmark 主链路没有实例化 `VanillaCoreMemory`，也没有在 runner 里做以下步骤：

1. 把多轮历史拆成 observation
2. 为 observation 计算 salience
3. 调 `observe` 或 `observe_text`
4. 在问题到来时调 `query` 或 `query_text`
5. 用 read result 构造最终 answer prompt

相反，当前主链路是：

- adapter 直接把官方上下文渲染成大 prompt
- provider 直接对大 prompt 推理

因此当前 benchmark 成绩更多反映的是：

- prompt rendering 是否对齐协议
- provider 是否稳定
- 模型对长上下文问答的直接能力

而不是：

- bounded personalized memory 框架本身的效果

### 4.2 当前 `VanillaCoreMemory` 更像“核”，不是“系统”

如果把一个完整 CoRe Mem 系统拆开，至少要有：

- observation parser
- write policy
- core update policy
- residual retention policy
- query-to-memory retrieval
- memory-to-prompt synthesis
- benchmark-specific answer head

当前 repo 已经有的是：

- core update policy 的最简版
- residual retention policy 的最简版
- query-to-memory retrieval 的最简版

还缺的是：

- 把 benchmark conversation 变成 memory observations 的那一层
- 把 memory read result 再变成 final answer prompt 的那一层

这也是我认为后续最该优先推进的地方。

## 5. 现在已有的实验和它们能说明什么

### 5.1 当前已经真实发生的实验

从 `outputs/` 和状态文档看，当前已经完成：

- PersonaMem 真实 1-sample run
- LongMemEval-S 真实 1-sample run
- Gemini 路径下的真实 1-sample run
- formal run 的部分续跑
  - PersonaMem 停在 `22/589`
  - LongMemEval-S 停在 `19/500`

### 5.2 这些实验能说明什么

它们可以说明：

- benchmark adapter 可读官方文件
- provider 调用链可用
- 输出元数据和追溯机制可用
- evaluation 和 make tables 链路可用
- resume 机制可用
- 当前 prompt 设计至少能在最小 live run 中返回符合协议的答案

### 5.3 这些实验还不能说明什么

它们还不能说明：

- vanilla CoRe Memory 已经接入正式 benchmark
- bounded memory 比 full-context 更好
- 当前 memory 设计已经有效提升 benchmark 表现
- 当前 repo 已完成 CoRe Mem 算法验证

这几个判断现在都不能做。

## 6. 如果今天要把当前 memory 用到真实推理里，应该怎么用

这里我分成“当前代码能直接做的用法”和“真正推荐的 benchmark 用法”。

### 6.1 当前代码能直接做的最小用法

当前可以这样理解 `VanillaCoreMemory`：

1. 准备一个 text embedder
2. 对历史中的每条用户事实或对话片段调用 `observe_text`
3. 问题到来时对问题调用 `query_text`
4. 取返回的 `supporting_slots`
5. 把这些 supporting evidence 和问题一起拼给 LLM 生成答案

换句话说，当前最合理的实际用法不是“让 core vector 直接回答问题”，而是：

- `core_vector` 作为压缩状态保留
- `supporting_slots` 作为最终 prompt 的显性证据

因为当前 provider/runner 根本不会消费 `core_vector` 本身。

### 6.2 在真实 PersonaMem 里更合理的用法

如果我们要把它真正接进 PersonaMem，我建议流程是：

1. 先把 shared context 按 turn 或按句子拆成 observation
2. 为每个 observation 计算 salience
3. 顺序写入 `VanillaCoreMemory`
4. 到问题时，用 question 去 query memory
5. 只把：
   - 问题
   - top-k supporting slots
   - 必要的 persona summary
   拼进最终 answer prompt

这样才能真正体现：

- bounded memory budget
- retrieval before answer
- personalized evidence selection

否则现在这种 full-context prompt，本质上还是“直接喂长上下文”。

### 6.3 在真实 LongMemEval-S 里更合理的用法

LongMemEval-S 更适合检验 temporal memory，因此建议流程是：

1. 按 session 写入
2. 给每个 observation 保留时间戳和 session 标识
3. query 时优先做：
   - semantic match
   - 再加时间一致性重排
4. 最终 prompt 只展示最相关 session span，而不是整段 haystack

如果还是直接喂整个 haystack，那么 memory 模块基本没有被用到。

### 6.4 当前实现下应该怎样看待 `core_vector`

当前 `core_vector` 更像是一个压缩统计态，而不是可直接解释的 summary。

所以在真实推理中：

- 不建议直接把 `core_vector` 当作自然语言证据用
- 更适合把它用于：
  - 粗召回
  - 变化检测
  - 是否需要读 residual 的 gating

但当前代码还没有把这些用途接出来。

## 7. 我对当前实现的审阅判断

### 7.1 当前实现做对了什么

1. 先把平台、协议、输出和可追溯性建起来了。
- 这让后续算法迭代不会陷入“改了很多，但不知道是不是协议问题”的混乱。

2. memory 原型保持得足够简单。
- 这对第一阶段是优点，因为它把问题收敛到最关键的几个机制。

3. embedding 路线和 training-light 约束保持一致。
- 没有越界到重训练。

4. resume / pacing / retry / supervisor 很实用。
- 对真实 benchmark 运行尤其重要。

### 7.2 当前实现最关键的不足

1. benchmark 主链路和 memory 主链路没有打通。
- 这是目前最核心的算法缺口。

2. writer 还不是“选择性写入”。
- 现在几乎所有 observation 都更新 core，也都进入 residual 候选。

3. residual 管理过于静态。
- 只按 salience 和 timestamp 排序，不考虑去重、覆盖、topic balance。

4. reader 只做单次点积排序。
- 对复杂 personalized reasoning 还太弱。

5. 当前没有把 retrieved support 重新组织成 answer-time context。
- 没有 memory-to-prompt synthesis，就很难说 memory 真正参与了推理。

## 8. 我建议的改进优先级

下面这些建议都尽量保持在第一阶段边界内，不引入重训练。

### 优先级 A：先把 memory 接进 benchmark 主循环

这是最重要的一步。

建议新增一个显式 orchestration 层，负责：

- 历史切分
- salience 估计
- memory 写入
- query 检索
- final prompt 构造

在这一步之前，任何“memory 机制改进”都很难真正体现在 benchmark 结果里。

### 优先级 B：把 writer 从“全写入”改成“选择性写入”

我更推荐先做 heuristic 版，而不是立刻训练 scorer：

- 用户显式偏好
- 稳定 profile fact
- 长期目标
- 持续性约束
- 被 assistant 重复确认的事实

这类 observation 的 salience 应该显著高于一次性闲聊内容。

如果不做这一步，residual slots 很快就会被噪声占满。

### 优先级 C：把 residual budget 从“全局 top-k”改成“带覆盖意识的 top-k”

当前只按 salience 排序，会导致：

- 同主题信息挤占预算
- 新近但低 salience 的关键信息消失

更好的 stage-1 版本不一定要训练，可以先做：

- 相似槽去重
- 每个主题或事实类型的最小配额
- recency bonus
- contradiction overwrite

### 优先级 D：把 reader 输出真正变成 answer-time evidence

当前 `query()` 只返回：

- `core_vector`
- `supporting_slots`

下一步应该补一个 synthesis 层，把 supporting slots 变成：

- 精简证据块
- 带来源的 profile summary
- benchmark-specific answer prompt

如果没有这层，memory 读出来和没读出来差别不大。

### 优先级 E：把 `core_vector` 用作“预算内摘要状态”，而不只是被动存着

可以考虑两个轻量方向：

1. 用 `core_vector` 做 coarse recall gate
- 如果 query 与 core 很不相关，就减少 residual 读取

2. 定期把高频稳定事实回写成 textual summary
- 让 core 不只存在于 latent 空间，也能生成少量可读 summary

这两条都比“把 core 做成复杂神经模块”更适合第一阶段。

## 9. 我认为下一版更好的实现方式

如果我们继续沿第一阶段的边界推进，我建议下一版采用下面这个结构：

### 9.1 三段式推理

第一段：observe
- 把历史 turn 转成 observation
- 做 fact typing
- 做 salience estimation
- 写入 core + residual

第二段：retrieve
- 问题到来时用 query embedding 检索 residual
- 同时参考 core 的整体状态
- 返回 top-k evidence

第三段：answer
- 只把 top-k evidence、必要约束和问题送入 LLM
- 不再把完整长历史直接塞给模型

### 9.2 推荐的 observation 粒度

PersonaMem：
- 以用户事实、偏好、约束、长期目标为主
- assistant 的确认句可以作为 support，但不一定单独写入

LongMemEval-S：
- 以事件、时间、实体关系为主
- 保留 session 和日期

### 9.3 推荐的 salience 组成

一个实用但仍然轻量的 heuristic 可以由以下部分线性组合：

- personal relevance
- future usefulness
- explicit preference / constraint marker
- repetition count
- recency
- contradiction penalty or overwrite bonus

这已经足够比“全量平均写入”好很多。

### 9.4 推荐的 prompt 形态

最终 answer prompt 应该长这样：

- 任务说明
- memory evidence block
- optional concise persona summary
- current question
- 输出约束

而不应该继续是：

- 全量原始上下文
- current question

因为后一种形态没有真正体现 bounded memory framework。

## 10. 一句话总结

当前 repo 的真实状态不是“CoRe Mem 已经完整进入正式 benchmark 推理”，而是：

- benchmark 平台已经能跑
- memory kernel 已经实现
- 真实 live runs 已经发生
- 但最关键的下一步，是把 memory kernel 真正接到 benchmark inference loop 里

如果要做研究上的有效改进，我认为最高优先级不是继续微调 prompt，而是先完成下面这条闭环：

`history -> observation -> memory write -> query -> supporting evidence -> final answer prompt`

只有这条闭环真正建立起来，后续关于 salience、budget、reader、core/residual 分工的改进，才会变成有意义的算法改进，而不是平台层面的错觉增益。
