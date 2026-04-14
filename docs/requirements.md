# Requirements

## 文档定位

本文档是项目最高优先级真源文档，定义项目目标、阶段边界、硬约束、验收标准、运行时真相和第二阶段设计要求。

规则：

1. 本文档优先级高于其他所有项目文档。
2. 若本文件与 `docs/current_status.md`、`docs/implementation_plan.md`、`docs/todo.md`、`.agent-os/*` 冲突，以本文件为准。
3. 第一阶段和第二阶段都受本文件约束；阶段状态变化必须先体现在本文件，再同步到其他文档。
4. 本文件只记录已经由用户明确确认、或可由代码与状态文档直接支持的事实；不得把假设写成既定事实。

## 0. 真源摘要

### 0.1 全局目标

- `OBJ-002`：实现一个 bounded cross-session personalized memory framework。
- `OBJ-003`：第一阶段具体实例为 `vanilla CoRe Memory`，目标是在指定 benchmark 上完成可复现的 benchmark 运行，并保留 v1 作为 naive baseline。
- `OBJ-004`：第二阶段具体实例为 `V2.0 structured latent-slot memory`，目标是在保留 v1 baseline 参考价值的前提下，建立更强的 latent memory 研究主线。

### 0.2 当前主线

- 当前主线是 **第二阶段 `V2.0 structured latent-slot memory`**。
- 第一阶段 `v1 / vanilla CoRe Memory` 继续保留为：
  - naive baseline
  - 代码与实验对照组
  - stage-1 验收目标
- **第一阶段正式 benchmark 运行当前不是主线执行项**；在用户明确要求 AI 去跑之前，PersonaMem 与 LongMemEval-S 的 formal benchmark 均处于待触发状态。

### 0.3 当前阶段状态

- `Stage-1`: 已有最小实现、测试和部分真实运行证据，但正式 benchmark 尚未完成；formal run 继续保留为 pending until explicit user trigger。
- `Stage-2`: 已完成方法论锁定，当前进入实现级规格与最小骨架阶段。

## 1. 第一阶段：目标、边界与验收

### 1.1 第一阶段目标

第一阶段只做以下事项：

1. 实现 `vanilla CoRe Memory`
2. 跑通 `PersonaMem` 官方主任务，优先级最高
3. 跑通 `LongMemEval-S`，优先级次之
4. 建立可复现实验流程
5. 产出结果数据、主结果表和可追溯实验记录
6. 建立测试与恢复体系

### 1.2 第一阶段非目标

第一阶段明确不做：

1. baseline 在本 repo 内的运行与整合
2. performance superiority claim
3. assignment randomization ablation
4. spectral / multi-timescale extension
5. 重型训练
6. 训练独立 embedding model
7. `LongMemEval-M`
8. `LoCoMo` 正式实验
9. 报告写作作为硬验收

### 1.3 第一阶段当前真实实现

当前 v1 实现是：

- `VanillaCoreMemory`
- observation 通过 pretrained embedding model 编码
- memory 内部是：
  - 一个 `core_vector`
  - 若干 `residual slots`
- reader 使用 query embedding 对 residual slots 做相似度排序

重要真实约束：

- 正式 benchmark 主链路当前仍然主要是 benchmark adapter 直接构造 full-context prompt。
- memory 模块虽已实现、已测试，但尚未构成 stage-1 benchmark 的主推理闭环。
- 因此 stage-1 的 formal benchmark 能说明：
  - repo 对 benchmark protocol 的接入状态
  - provider / runner / output traceability 的成熟度
  - v1 minimal memory module 的实现可用性
- 但 **不能把当前 formal benchmark 结果误称为 memory-mediated inference superiority**。

### 1.4 第一阶段验收标准

- `AC-001`：`vanilla CoRe Memory` 核心模块存在、可配置、可单测。
- `AC-002`：`PersonaMem` 官方主任务可在 repo 内按统一流程运行，并产生可追溯输出。
- `AC-003`：`LongMemEval-S` 可在 repo 内按统一流程运行，并产生可追溯输出。
- `AC-004`：存在统一的复现实验入口、分步骤脚本和输出目录规范。
- `AC-005`：核心模块有 unit tests，且至少有一条最小 E2E smoke test。

### 1.5 第一阶段执行规则

1. stage-1 的 formal benchmark 在用户明确下达运行指令前，不主动消耗 API 去继续 formal run。
2. stage-1 代码、测试、文档和 baseline 地位必须保留。
3. stage-2 不得覆盖或删除 stage-1 的 baseline 参考价值。

## 2. 第二阶段：总目标与研究定位

### 2.1 第二阶段定位

第二阶段不是对 full-context prompt 再做微调，而是建立一个新的 latent memory 主线：

- memory 基本形式是 `structured latent slots`
- memory 内容是开放域的
- answer-time 不回放原始文本 chunk
- 主链路是 `observation -> slot -> belief -> answer`

### 2.2 第二阶段当前目标

第二阶段目标是实现并验证一个更强的 memory system，使其具备：

1. bounded memory budget
2. update-aware writing
3. core/residual 双银行
4. query-conditioned latent composition
5. belief-state recovery
6. local-first evaluation protocol

### 2.3 第二阶段不要求立即达成的事项

第二阶段当前不要求：

1. 立即替换所有 stage-1 代码路径
2. 立即跑 full benchmark
3. 立即证明显著优于所有 baseline
4. 立即做重型 end-to-end 全参数训练

## 3. 第二阶段方法总览

### 3.1 总体设计

`V2.0 structured latent-slot memory` 的主线锁定为：

- `Resampler`: Light Cross-Attention Resampler
- `Decoder`: Flan-T5-based belief JSON decoder
- `训练方式`: LoRA/adapter + resampler/projection 可训练
- `输出目标`: Belief JSON

其他路线处理规则：

- `Optimus-like latent decoder`：保留为 ablation 或 `V2.1` 升级候选
- `Perceiver-style heavier resampler`：保留为 ablation 或 `V2.1`
- `mean pooling resampler`：只作为更弱 baseline ablation

### 3.2 V2.0 不是以下路线

V2.0 不属于：

1. 原始文本 chunk RAG
2. 单一全局向量 memory
3. 直接取 retrieval gloss 作为回答证据
4. 把 benchmark eval/test 数据作为主训练源

### 3.3 V2.0 必须满足的系统主链路

1. 原始对话先变成 `Observation`
2. Observation 被编码为 `Slot`
3. Query 触发 retrieval + latent composition
4. Decoder 从 composed latent 恢复 `Belief JSON`
5. 最终回答从 belief 层投影，而不是从 raw history 直接回放

## 4. 第二阶段结构化对象定义

### 4.1 Observation JSON v1

Observation 是从原始对话中抽出来的 memory-bearing unit。

必须字段：

```json
{
  "obs_id": "string",
  "source_dataset": "string",
  "source_dialogue_id": "string",
  "source_turn_id": "string",
  "session_id": "string",
  "speaker": "user|assistant|system",
  "entity": "user",
  "relation": "string",
  "value": "string",
  "value_type": "preference|constraint|profile|goal|event|location|occupation|other",
  "time_scope": "current|past|future|recent_change|unknown",
  "status_hint": "active|stale|conflicted|unknown",
  "polarity": "positive|negative|neutral",
  "confidence": 1.0,
  "evidence_text": "string",
  "canonical_gloss": "string",
  "metadata": {}
}
```

解释：

- `evidence_text` 只用于离线审计，不作为 answer-time 主 memory 载体。
- `canonical_gloss` 只用于解释、错误分析和人工检查，不等于原始文本。
- `value_type` 是训练与分析时的弱标签，不是封闭 ontology。
- Observation parser 采用 **混合策略**：
  - `rule-first`
  - `model-second`

### 4.2 Slot Record v1

Slot 是 observation 经编码并写入 memory 后的基本单位。

必须字段：

```json
{
  "slot_id": "string",
  "bank": "core|residual",
  "entity": "user",
  "relation": "string",
  "retrieval_key": "float[d]",
  "latent_tokens": "float[m,d]",
  "soft_role_scores": {
    "stable": 0.0,
    "preference": 0.0,
    "constraint": 0.0,
    "goal": 0.0,
    "temporal": 0.0,
    "social": 0.0
  },
  "confidence": 0.0,
  "first_seen_ts": "string",
  "last_update_ts": "string",
  "revision_count": 0,
  "active_flag": true,
  "revision_parent": "string|null",
  "canonical_gloss": "string"
}
```

默认实现要求：

- slot 使用 **连续 latent tokens**
- 默认 `m = 8`
- retrieval key 与 decoder latent 分离
- `bank` 不是语义本体分类，而是生命周期位置

### 4.3 Belief JSON v1

Belief 是 query 条件下从若干 slots 恢复出的当前状态。

必须字段：

```json
{
  "query_id": "string",
  "entity": "user",
  "query_type": "single_fact|multi_fact|update_sensitive|temporal",
  "belief_items": [
    {
      "relation": "string",
      "value": "string",
      "status": "active|stale|conflicted|unknown",
      "time_scope": "current|past|future|recent_change|unknown",
      "confidence": 0.0,
      "support_slot_ids": ["string"]
    }
  ],
  "global_consistency": "high|medium|low"
}
```

字段规则：

1. `support_slot_ids` 必须保留，用于分离 retrieval 错误与 decoder 错误。
2. `query_type` 固定为四类：
   - `single_fact`
   - `multi_fact`
   - `update_sensitive`
   - `temporal`
3. `global_consistency` 用于诊断 memory 冲突程度。

## 5. 第二阶段 memory 结构与更新机制

### 5.1 Core / Residual 双银行

#### Core

Core 负责：

1. 长期稳定 persona 状态
2. 当前有效且被多次支持的信息
3. 经 consolidation 确认后的 active state

#### Residual

Residual 负责：

1. 新信息
2. 近期事件
3. 候选更新
4. 冲突状态
5. 等待晋升或淘汰的局部记忆

### 5.2 Lifecycle

状态锁定为：

- `active`
- `stale`
- `inactive`

动作锁定为：

- `new`
- `merge`
- `overwrite`
- `evict`

### 5.3 更新原则

新 observation 先进入 residual 视角判断：

1. 无高相似且写入价值高 -> `new`
2. 高相似且语义一致 -> `merge`
3. 高相似但表示状态变化 -> `overwrite`
4. 低价值 -> `ignore`

### 5.4 Core 的变化方式

Core 不做高频抖动更新。Core 的变化主要通过：

1. `reinforce_core`
2. `promote_residual_to_core`
3. `replace_core_via_update`

### 5.5 Consolidation

Consolidation 是显式模块，负责：

1. residual -> core promotion
2. core stale handling
3. duplicate merge
4. eviction

### 5.6 关键原则

必须保持以下逻辑成立：

> residual 负责吸收变化，core 负责维持稳定状态；更新先在 residual 发酵，再通过 consolidation 决定是否重写 core。

## 6. 第二阶段 latent composition 与 decoder

### 6.1 Composition 目标

composition 负责把 retrieval 返回的多个相关 slots 聚合为一个 query-conditioned latent workspace。

它不是：

- 简单拼接 raw text
- 只做 nearest slot gloss 输出

它必须做到：

1. 接受 top-k slots 的 latent tokens
2. 产生固定长度的 composed latent representation
3. 为 belief decoder 提供统一输入

### 6.2 主线 resampler

主线 resampler 采用 `Light Cross-Attention Resampler`。

默认结构：

1. query 编码为 `q`
2. top-k slots 的 latent tokens 拼成 memory token set
3. 一组固定的 latent queries 对 memory token set 做 `1~2` 层 cross-attention
4. 得到固定长度 composed latent block `Z_q`
5. `Z_q` 条件化 Flan-T5 decoder 输出 Belief JSON

### 6.3 主线 decoder

decoder 采用 `Flan-T5-base` 主线。

要求：

1. belief JSON 是主输出目标
2. decoder 通过 LoRA/adapter 与 resampler/projection 协同训练
3. 最终 answer-time 可以把 belief JSON 进一步投影成短 evidence block，再交给 benchmark answering path

### 6.4 训练参数冻结策略

默认策略：

1. 大部分 backbone encoder 冻结
2. 大部分 backbone decoder 冻结
3. resampler / slot projection / lifecycle head 可训练
4. decoder 采用 LoRA/adapter 方式做轻量训练

## 7. 第二阶段默认超参数

默认值不是不可更改的理论约束，但在未被实验否定前，视为主线默认配置。

### 7.1 Backbone

- `decoder backbone`: `google/flan-t5-base`
- `encoder side`: 与 decoder 同 backbone 的 encoder 初始化
- `retrieval encoder`: 默认与主 encoder 共享参数

### 7.2 Resampler

- `top_k slots = 8`
- `latent tokens per slot (m) = 8`
- `composed latent tokens (L) = 8`
- `hidden dim (d) = 768`
- `cross-attention layers = 2`
- `attention heads = 12`
- `dropout = 0.1`
- `slot projection = 2-layer MLP`

### 7.3 LoRA

- `rank = 16`
- `alpha = 32`
- `dropout = 0.05`

## 8. 第二阶段训练任务

V2.0 训练拆成四个任务：

1. `Slot Autoencoding`
2. `Retrieval Alignment`
3. `Lifecycle Prediction`
4. `Composition-to-Belief Decoding`

### 8.1 Slot Autoencoding

目标：

- 让单条 observation 能压缩成 slot latent tokens
- 再恢复为规范化 semantic record

### 8.2 Retrieval Alignment

目标：

- 让 query 能找到正确 slot
- 错误 slot 尤其是 stale / temporal mismatch slot 被压低

### 8.3 Lifecycle Prediction

目标：

- 学会 `new / merge / overwrite / ignore`
- 学会 stale / promote 等辅助状态

### 8.4 Composition-to-Belief Decoding

目标：

- 从 query + selected slots 恢复 Belief JSON

### 8.5 训练方式

采用 staged training：

1. 先训 `Slot Autoencoding`
2. 再训 `Retrieval Alignment`
3. 再训 `Lifecycle Prediction`
4. 最后训 `Composition-to-Belief Decoding`
5. 之后再做 joint fine-tuning

## 9. 第二阶段训练数据策略

### 9.1 顶层原则

训练数据策略锁定为：

- `public-datasets-first`
- `synthetic-minimal`

并且：

- **benchmark 是 evaluation source，不是 primary training source**
- **benchmark-derived proxy supervision 不作为主要训练信号**

### 9.2 数据池定义

V2.0 使用四个数据池：

1. `State / Belief Pool`
2. `Persona / Preference Pool`
3. `Update / Propagation Pool`
4. `Minimal Synthetic Pool`

### 9.3 数据池与公开数据集

#### State / Belief Pool

主数据集：

- `Schema-Guided Dialogue (SGD)`
- `MultiWOZ 2.4`

作用：

- belief/state 结构化监督主来源
- 训练 slot-value 风格状态恢复

#### Persona / Preference Pool

主数据集：

- `Persona-Chat`
- `Training Millions of Personalized Dialogue Agents`

作用：

- 训练 stable persona / preference / profile 相关 observation 和 retrieval

#### Update / Propagation Pool

主数据集：

- `MQUAKE`
- `ReCoE`

补充来源：

- 从 `SGD` / `MultiWOZ` 自动抽取 state-change turns

作用：

- 训练 overwrite / stale suppression / update propagation

#### Minimal Synthetic Pool

只允许补这四类 corner cases：

1. overwrite stress cases
2. budget stress cases
3. conflict bundles
4. multi-fact composition

规则：

- synthetic 不是主粮
- 占比默认不超过总训练样本的 `10%`

### 9.4 第二优先级候选数据集

以下数据集保留为可选增强，但不是 V2.0 第一轮主训练必需项：

- `DuLeMon`
- `REALTALK`
- `MS-TOD`

## 10. 数据集到训练任务的精确映射

本节是第二阶段最重要的 runtime truth 之一：**必须明确哪些数据训练哪些模块，而不是把所有数据和任务混在一起。**

### 10.1 `Slot Autoencoding` 使用哪些数据

主训练数据：

- `SGD`
- `MultiWOZ 2.4`
- `Persona-Chat`
- `Training Millions of Personalized Dialogue Agents`

可选少量补充：

- `Minimal Synthetic Pool` 中的规范化 observation 样本

训练对象：

- slot encoder
- slot projection
- decoder 的 observation reconstruction 路径

不主要依赖：

- `MQUAKE`
- `ReCoE`

原因：

- `Slot Autoencoding` 的核心是学 observation -> latent -> normalized record，而不是复杂 update reasoning。

### 10.2 `Retrieval Alignment` 使用哪些数据

主训练数据：

- `Persona-Chat`
- `Training Millions of Personalized Dialogue Agents`
- `SGD`
- `MultiWOZ 2.4`

hard negatives 增强来源：

- `MQUAKE`
- `ReCoE`

训练对象：

- retrieval key head
- query encoder
- retrieval scoring path

负样本规则必须包含：

1. same relation, wrong value
2. same entity, wrong relation
3. stale version
4. temporal mismatch

### 10.3 `Lifecycle Prediction` 使用哪些数据

主训练数据：

- `MQUAKE`
- `ReCoE`
- 从 `SGD` / `MultiWOZ` 自动抽取的 state-change turns

最小补充：

- `Minimal Synthetic Pool`

训练对象：

- lifecycle classifier
- stale / promote flags
- overwrite handling path

为什么不能只靠 persona/state 数据：

- 因为 overwrite、propagation 和 conflict-sensitive update 是该任务的核心监督来源。

### 10.4 `Composition-to-Belief Decoding` 使用哪些数据

主训练数据：

- `SGD`
- `MultiWOZ 2.4`
- `Persona-Chat`
- `Training Millions of Personalized Dialogue Agents`
- `MQUAKE`
- `ReCoE`

最小补充：

- `Minimal Synthetic Pool` 中的 multi-fact composition 样本

训练对象：

- light resampler
- composed latent projection
- belief JSON decoder

训练样本规则：

1. 早期阶段允许使用 gold support slot set
2. 后期阶段必须混入 noisy retrieved slot set
3. belief decoder 的主要目标是结构化 belief state，不是 final answer

## 11. 第二阶段本地评估策略

### 11.1 顶层原则

V2.0 采用 `local-first` protocol：

1. 先跑本地 intrinsic evaluation
2. 再决定哪些版本值得上 benchmark canary
3. full benchmark 只在用户要求后执行

### 11.2 指标家族

本地评估固定采用：

- `4 个现成指标家族`
- `2 个少量自定义指标`

### 11.3 现成指标家族

#### Family A: Retrieval metrics

来源对齐：

- `LongMemEval`
- `LMEB`

记录指标：

- `Recall@k`
- `NDCG@k`

用于评估：

- retrieval key head
- query encoder
- top-k slot selection

主要评估数据：

- retrieval 样本集
- belief decode 样本中的 support slot 恢复

#### Family B: Belief recovery metrics

来源对齐：

- `Dialogue State Tracking`

记录指标：

- `Joint Belief Accuracy`
- slot/value F1

用于评估：

- belief decoder
- composition 之后的 state recovery

主要评估数据：

- composition-to-belief 样本
- belief JSON gold targets

#### Family C: Update / propagation metrics

来源对齐：

- `MQUAKE`
- `ReCoE`

记录指标：

- `Update Success`
- `Propagation Accuracy`

用于评估：

- lifecycle module
- overwrite / stale suppression
- update-aware retrieval and belief recovery

主要评估数据：

- update / propagation 样本

#### Family D: Specificity / locality metrics

来源对齐：

- `CounterFact`
- `CounterFact+`

记录指标：

- `Specificity / Locality`

用于评估：

- overwrite 后的非目标扰动
- memory edit 的局部性

主要评估数据：

- overwrite 样本
- conflict bundles

### 11.4 自定义指标

#### Custom-1: `Compression Fidelity`

定义：

- 在固定 slot/token budget 下，gold belief 或 answer-relevant facts 的保留率

用于评估：

- 整个 memory system 的压缩有效性
- budget sweep 主结果

#### Custom-2: `Edit Locality at Slot Level`

定义：

- 更新目标 relation 后，被无关更新误伤的 slot 比例

用于评估：

- overwrite 是否只影响局部 slots
- core/residual 与 consolidation 的副作用控制

### 11.5 指标到模块的精确映射

必须按如下映射记录，而不是只给总分：

1. `Slot Autoencoding`
   - 主看：reconstruction quality
   - 辅看：后续 belief decoder 的 slot usability
2. `Retrieval Alignment`
   - 主看：`Recall@k`、`NDCG@k`
3. `Lifecycle Prediction`
   - 主看：`Update Success`
   - 辅看：`Specificity / Locality`、`Edit Locality at Slot Level`
4. `Composition-to-Belief Decoding`
   - 主看：`Joint Belief Accuracy`、slot/value F1
   - 辅看：`Propagation Accuracy`
5. `Whole system under budget`
   - 主看：`Compression Fidelity`

## 12. Benchmark 与 canary 规则

### 12.1 Benchmark 使用原则

1. benchmark 是 evaluation source，不是 primary training source
2. benchmark-derived proxy supervision 只用于：
   - intrinsic eval
   - ablation
   - failure analysis
3. benchmark eval/test 不得作为主要训练数据

### 12.2 Stage-1 formal benchmark 当前规则

1. `PersonaMem` formal run
2. `LongMemEval-S` formal run

二者当前都记为：

- baseline / acceptance 相关任务
- **pending until explicit user trigger**

### 12.3 Stage-2 canary 规则

在未来需要上 benchmark 时，默认先走 canary：

- `PersonaMem canary = 64`
- `LongMemEval-S canary = 64`

当前阶段记录协议即可，不强制立即执行。

## 13. 第二阶段 ablation 要求

第二阶段至少保留以下 ablation 设计位：

1. `T5 route vs Optimus-like`
2. `mean pooling vs light cross-attention resampler`
3. `belief JSON decoding vs direct answer decoding`
4. `merge-only vs merge+overwrite`
5. `single-bank vs core+residual`
6. `core/residual assignment randomization`
7. `budget sweep`
8. `with/without update pool`
9. `with/without persona pool`
10. `with/without state pool`
11. `with/without synthetic pool`

## 14. 输出与目录规范

### 14.1 Stage-1 输出

stage-1 结果统一放在：

- `outputs/`

### 14.2 Stage-2 输出

stage-2 结果统一放在：

- `outputs_v2/`

必须至少包含：

- `outputs_v2/checkpoints/`
- `outputs_v2/runs/`
- `outputs_v2/evals_local/`
- `outputs_v2/evals_benchmark/`
- `outputs_v2/tables/`
- `outputs_v2/figures/`
- `outputs_v2/logs/`
- `outputs_v2/cache_manifests/`
- `outputs_v2/artifacts/`

### 14.3 Stage-2 artifact 保存要求

必须保存：

1. config snapshot
2. model/backbone name
3. LoRA / resampler config
4. run timestamp
5. commit hash
6. local eval results
7. belief dumps
8. slot dumps
9. canary / benchmark eval outputs（若有）

## 15. 环境与资源约束

### 15.1 环境约束

1. conda 环境固定为 `core_mem`
2. Python 固定为 `3.10`

### 15.2 Stage-1 provider 约束

1. 默认外部 LLM provider 为阿里云兼容 OpenAI API
2. 默认 `base_url`：`https://dashscope.aliyuncs.com/compatible-mode/v1`
3. 默认模型：`qwen3-235b-a22b-instruct-2507`
4. API key 不得写入仓库，只能走环境变量或本地 `.env`

### 15.3 Stage-2 训练资源约束

1. 当前按 **单卡训练** 规划
2. 允许下载额外 checkpoint
3. 当前阶段不把显存预算作为先验 blocker
4. 训练规模仍应优先保持轻量化，不默认扩展为重型全参数训练

## 16. 文档与状态同步要求

### 16.1 必须同步的文档

当第二阶段方法、训练、数据、评估、TODO 状态变化时，至少同步：

- `docs/current_status.md`
- `docs/implementation_plan.md`
- `docs/todo.md`
- `.agent-os/project-index.md`
- `.agent-os/todo.md`
- `.agent-os/architecture-milestones.md`
- `.agent-os/change-decisions.md`
- `.agent-os/acceptance-report.md`
- `.agent-os/run-log.md`

### 16.2 Stage-2 主线要求

在未收到新的用户指令前：

1. `docs/` 与 `.agent-os/` 中的 top next action 应指向 stage-2 主线任务
2. stage-1 formal benchmark 任务应显示为 pending / blocked-by-user-trigger，而不是当前执行主线

## 17. 升级确认条件

以下事项必须先经用户确认：

1. 修改本真源文档的核心目标、硬约束或验收标准
2. 把 benchmark eval/test 变成主要训练源
3. 把 synthetic 升级为主训练源
4. 切换主要 backbone、主要 provider 或核心环境版本
5. 引入重型训练或超出当前资源边界的方案
6. 改变 stage-1 baseline 的保留策略

## 18. 当前已确认但必须持续遵守的事实

1. stage-2 是当前主线
2. stage-1 v1 继续保留为 naive baseline
3. stage-1 formal benchmark 在用户明确要求前不主动继续跑
4. stage-2 主线是 `Light Cross-Attention Resampler + Flan-T5-base belief JSON decoder`
5. `public-datasets-first, synthetic-minimal`
6. benchmark 只作评测源，不作主要训练源
7. 数据集到训练任务、指标到模块的映射必须明确记录，不能只做“全混合训练 / 全混合评估”的含糊描述
