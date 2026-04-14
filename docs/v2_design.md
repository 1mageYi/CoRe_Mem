# V2.0 Design

## 文档定位

本文档是第二阶段 `V2.0 structured latent-slot memory` 的实现级设计文档。

规则：

1. 本文档从属于真源 [requirements.md](/home/image/workspace/CoRe_Mem/docs/requirements.md)。
2. 若本文档与真源冲突，以真源为准。
3. 本文档的职责不是再次讨论方向，而是把用户已经确认的 stage-2 方案写成可直接实现、可直接评估、可直接做 ablation 的执行规格。

## 1. V2.0 一句话定义

V2.0 是一个 **structured latent-slot memory system**：

- 不做原始文本 chunk RAG
- 不做单一全局向量记忆
- 不把 `canonical_gloss` 当作主回答证据
- 主链路固定为：
  - `observation -> slot -> latent composition -> belief JSON -> answer`

## 2. 方法边界

### 2.1 与 stage-1 的关系

- stage-1 `vanilla CoRe Memory` 保留为 naive baseline
- stage-2 `V2.0` 是当前主线
- stage-1 formal benchmark 不会被删除，但在用户明确要求前不主动继续跑

### 2.2 V2.0 的核心研究 claim

V2.0 试图证明：

1. 在固定 budget 下，可以用 **structured latent slots** 代替 full-context history
2. memory 必须是 **stateful** 的，而不是文本块检索
3. personalized memory 的关键不是“找回一段原文”，而是“恢复当前有效 belief state”
4. update / overwrite / stale suppression 是 memory 方法的核心，而不是边角功能

## 3. 核心对象与数据流

### 3.1 三层对象

V2.0 必须清楚区分三层对象：

1. `Observation`
2. `Slot`
3. `Belief`

对应的数据流为：

`raw dialogue -> observation parser -> observation -> slot encoder/write -> slot banks -> retrieval + composition -> belief decoder -> belief JSON -> answer projection`

### 3.2 为什么必须区分这三层

如果不区分：

- parser 噪声
- memory 写入错误
- retrieval 错误
- belief decoder 错误

会全部混成一个黑箱，无法做 intrinsic eval，也无法做结构化 ablation。

## 4. Observation JSON v1

Observation 是从原始对话里抽出来的 memory-bearing unit，是 memory 的唯一写入入口。

### 4.1 Schema

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

### 4.2 字段语义

- `obs_id`: observation 唯一 ID
- `source_*`: 用于回溯来源，不进入 answer-time 主证据
- `speaker`: 用于区分用户事实和 assistant 引导文本
- `entity`: 当前 V2.0 默认先固定为 `user`
- `relation`: 规范化关系名
- `value`: 关系值
- `value_type`: 训练/分析弱标签，不是封闭 ontology
- `time_scope`: 当前 / 过去 / 未来 / 最近变化
- `status_hint`: parser 对 active/stale/conflict 的弱提示
- `polarity`: 正向、否定或中性表述
- `confidence`: parser 置信度
- `evidence_text`: 只用于离线审计
- `canonical_gloss`: 只用于解释和错误分析

### 4.3 Parser 规则

Observation parser 采用：

- `rule-first`
- `model-second`

默认流程：

1. 先用规则抽：
   - 显式 preference
   - constraint
   - update cue
   - temporal cue
2. 规则置信度低时，回退到同一 `Flan-T5-base` backbone 的结构化抽取分支
3. parser 输出必须带 `confidence`

## 5. Slot Record v1

Slot 是 observation 被编码并写入 memory 后的状态单元。

### 5.1 Schema

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

### 5.2 关键设计点

1. `retrieval_key` 与 `latent_tokens` 分离
2. `bank` 是生命周期位置，不是语义本体类别
3. `latent_tokens` 使用连续 latent，不是离散 VQ token
4. `soft_role_scores` 是控制层，不是封闭类型全集

### 5.3 默认值

- `m = 8`
- `d = 768`
- `bank` 默认由 lifecycle / consolidation 决定，不由 parser 直接拍板

## 6. Belief JSON v1

Belief JSON 是 decoder 的主输出目标，也是 answer projection 的上游输入。

### 6.1 Schema

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

### 6.2 为什么必须输出 Belief JSON

因为 V2.0 要恢复的是 **state**，不是一堆文本片段。

Belief JSON 的作用：

1. 分离 memory 与 answer head
2. 让 local intrinsic eval 变得可能
3. 让 retrieval 错误和 decoding 错误可诊断
4. 让 update / stale / temporal reasoning 有显式监督目标

### 6.3 `support_slot_ids` 必须保留

原因：

1. 可以直接核对 retrieval 是否正确
2. 可以把“slot 找错”和“decoder 读错”分开
3. 对 ablation 与 failure analysis 至关重要

## 7. Relation normalization v1

V2.0 不采用封闭 ontology 限制 memory 内容，但训练与评估仍需要规范化 relation 名。

### 7.1 默认 relation set

- `drink_preference`
- `food_preference`
- `music_preference`
- `tool_preference`
- `hobby`
- `location`
- `occupation`
- `education`
- `goal`
- `constraint`
- `relationship`
- `episodic_event`
- `recent_change`
- `temporal_fact`
- `profile_trait`
- `other_fact`

### 7.2 规则

1. OOD 内容一律允许落到 `other_fact`
2. relation normalization 只用于 parser、训练和评估，不是 memory 内容上限

## 8. Memory 结构

### 8.1 双银行

#### Core

Core 保存：

1. 稳定 persona 状态
2. 当前有效且被多次支持的信息
3. consolidation 后确认的 active state

#### Residual

Residual 保存：

1. 新 observation
2. 近期事件
3. 候选更新
4. 冲突状态
5. 等待晋升或淘汰的局部记忆

### 8.2 Lifecycle

状态：

- `active`
- `stale`
- `inactive`

动作：

- `new`
- `merge`
- `overwrite`
- `evict`

### 8.3 更新逻辑

新 observation 一律先从 residual 视角处理：

1. 低相似但高价值 -> `new`
2. 高相似且一致 -> `merge`
3. 高相似但 value 变化 -> `overwrite`
4. 低价值 -> `ignore`

Core 变化仅通过：

1. `reinforce_core`
2. `promote_residual_to_core`
3. `replace_core_via_update`

### 8.4 Consolidation

Consolidation 负责：

1. residual -> core promotion
2. core stale handling
3. duplicate merge
4. eviction

### 8.5 必须保持的设计原则

> residual 负责吸收变化，core 负责维持稳定状态；更新先在 residual 发酵，再通过 consolidation 决定是否重写 core。

## 9. Latent Composition

### 9.1 问题定义

retrieval 返回的是若干离散 slot；decoder 需要的是固定长度、面向当前 query 的 latent workspace。

composition 的职责就是把：

`top-k slots -> composed latent memory`

### 9.2 主线结构

V2.0 主线采用 `Light Cross-Attention Resampler`：

1. query 编码为 `q`
2. 根据 retrieval score 选出 `top-k` slots
3. 取出这些 slots 的 `latent_tokens`
4. 使用 `L` 个 learnable latent queries 对 memory tokens 做 `2` 层 cross-attention
5. 输出固定长度 composed latent block `Z_q`

### 9.3 默认超参数

- `top_k = 8`
- `m = 8`
- `L = 8`
- `d = 768`
- `heads = 12`
- `layers = 2`
- `dropout = 0.1`

### 9.4 为什么不是 mean pooling

mean pooling 只作为 baseline ablation。

主线不用它的原因：

1. multi-slot interaction 太弱
2. 很容易退化成“高级平均”
3. 不足以体现 query-conditioned latent composition 的研究点

## 10. Belief Decoder

### 10.1 主线 decoder

主线 decoder 为：

- `google/flan-t5-base`

### 10.2 训练方式

可训练部分：

- resampler
- slot projection
- lifecycle head
- decoder LoRA / adapter

默认冻结：

- 大部分 encoder backbone
- 大部分 decoder backbone

### 10.3 输出方式

主输出是 `Belief JSON`，不是 direct answer。

后续回答路径：

`Belief JSON -> short evidence block -> benchmark answer path`

### 10.4 为什么不用 direct answer 作为主目标

因为 direct answer 会把：

- retrieval
- composition
- belief recovery
- reasoning style

混在一起，不利于做结构化 memory 研究。

## 11. 训练任务与样本定义

V2.0 训练固定拆成四个任务。

## 11.1 `Slot Autoencoding`

### 目标

学 observation -> latent slot -> normalized record

### 输入样本

```json
{
  "input_observation": { "...Observation JSON..." },
  "target_record": {
    "entity": "user",
    "relation": "drink_preference",
    "value": "matcha",
    "time_scope": "current",
    "status": "active"
  }
}
```

### 训练对象

- slot encoder
- slot projection
- decoder 的 reconstruction 路径

## 11.2 `Retrieval Alignment`

### 目标

让 query 命中正确 slot

### 输入样本

```json
{
  "query": "What does the user currently like to drink?",
  "positive_slot": { "...slot..." },
  "negative_slots": [{ "...slot..." }, { "...slot..." }]
}
```

### 训练对象

- retrieval key head
- query encoder
- retrieval scoring path

### 负样本必须包含

1. same relation, wrong value
2. same entity, wrong relation
3. stale version
4. temporal mismatch

## 11.3 `Lifecycle Prediction`

### 目标

学习：

- `new`
- `merge`
- `overwrite`
- `ignore`

### 输入样本

```json
{
  "memory_context": [{ "...slot..." }],
  "new_observation": { "...observation..." },
  "target_action": "overwrite",
  "target_flags": {
    "promote": false,
    "stale_old": true
  }
}
```

### 训练对象

- lifecycle classifier
- stale/promotion flags

## 11.4 `Composition-to-Belief Decoding`

### 目标

从 query + selected slots 恢复 Belief JSON

### 输入样本

```json
{
  "query": "What is the user's current drink preference?",
  "memory_slots": [{ "...slot..." }, { "...slot..." }],
  "target_belief_json": { "...Belief JSON..." }
}
```

### 训练对象

- resampler
- decoder LoRA/adapter
- belief reconstruction path

### 训练规则

1. 早期允许 gold support set
2. 后期必须混入 noisy retrieved set

## 12. 数据策略：public-datasets-first, synthetic-minimal

### 12.1 数据池

V2.0 用四个数据池：

1. `State / Belief Pool`
2. `Persona / Preference Pool`
3. `Update / Propagation Pool`
4. `Minimal Synthetic Pool`

### 12.2 公开数据集

#### State / Belief Pool

- `SGD`
- `MultiWOZ 2.4`

#### Persona / Preference Pool

- `Persona-Chat`
- `Training Millions of Personalized Dialogue Agents`

#### Update / Propagation Pool

- `MQUAKE`
- `ReCoE`
- 从 `SGD` / `MultiWOZ` 自动抽取的 state-change turns

#### Minimal Synthetic Pool

只补：

1. overwrite stress
2. budget stress
3. conflict bundles
4. multi-fact composition

## 13. 数据集到训练任务的精确映射

本节是 V2.0 的关键约束：不能只写“把所有数据都用来训练模型”，必须明确 **哪个数据池训练哪个子模块**。

## 13.1 `Slot Autoencoding`

### 主数据

- `SGD`
- `MultiWOZ 2.4`
- `Persona-Chat`
- `Training Millions of Personalized Dialogue Agents`

### 可选补充

- `Minimal Synthetic Pool` 中的 observation-level 样本

### 训练模块

- slot encoder
- slot projection
- reconstruction path

### 不作为主数据

- `MQUAKE`
- `ReCoE`

原因：该任务主要学习 observation 压缩，而不是 update propagation。

## 13.2 `Retrieval Alignment`

### 主数据

- `Persona-Chat`
- `Training Millions of Personalized Dialogue Agents`
- `SGD`
- `MultiWOZ 2.4`

### hard negative 增强

- `MQUAKE`
- `ReCoE`

### 训练模块

- query encoder
- retrieval key head
- retrieval scorer

## 13.3 `Lifecycle Prediction`

### 主数据

- `MQUAKE`
- `ReCoE`
- 从 `SGD` / `MultiWOZ` 自动抽出的 state-change turns

### 最小补充

- `Minimal Synthetic Pool`

### 训练模块

- lifecycle head
- overwrite / stale logic 路径

## 13.4 `Composition-to-Belief Decoding`

### 主数据

- `SGD`
- `MultiWOZ 2.4`
- `Persona-Chat`
- `Training Millions of Personalized Dialogue Agents`
- `MQUAKE`
- `ReCoE`

### 最小补充

- `Minimal Synthetic Pool` 中的 multi-fact composition 样本

### 训练模块

- resampler
- decoder adapter / LoRA
- belief reconstruction path

## 14. 指标体系

V2.0 本地评估固定采用：

- `4 个现成指标家族`
- `2 个少量自定义指标`

## 14.1 Family A: Retrieval

来源对齐：

- `LongMemEval`
- `LMEB`

指标：

- `Recall@k`
- `NDCG@k`

评估模块：

- query encoder
- retrieval key head
- top-k selection

评估数据：

- retrieval 样本集
- belief 样本中的 support slot set

## 14.2 Family B: Belief recovery

来源对齐：

- `Dialogue State Tracking`

指标：

- `Joint Belief Accuracy`
- slot/value F1

评估模块：

- resampler
- belief decoder

评估数据：

- composition-to-belief 样本

## 14.3 Family C: Update / propagation

来源对齐：

- `MQUAKE`
- `ReCoE`

指标：

- `Update Success`
- `Propagation Accuracy`

评估模块：

- lifecycle head
- overwrite / stale suppression path
- update-aware belief recovery

评估数据：

- update/propagation 样本

## 14.4 Family D: Specificity / locality

来源对齐：

- `CounterFact`
- `CounterFact+`

指标：

- `Specificity / Locality`

评估模块：

- overwrite 后的副作用控制
- consolidation 过程的局部性

评估数据：

- overwrite 样本
- conflict bundle 样本

## 14.5 Custom-1: `Compression Fidelity`

定义：

- 固定 slot/token budget 下，gold belief 保留率或 answer-relevant fact retention

评估模块：

- whole memory system
- budget sweep 主结果

## 14.6 Custom-2: `Edit Locality at Slot Level`

定义：

- 一次更新后，被无关影响的 slot 比例

评估模块：

- lifecycle
- consolidation
- core/residual 分工

## 14.7 指标到模块映射表

### `Slot Autoencoding`

- 主看：reconstruction quality
- 辅看：后续 belief decoder 可用性

### `Retrieval Alignment`

- 主看：`Recall@k`
- 主看：`NDCG@k`

### `Lifecycle Prediction`

- 主看：`Update Success`
- 辅看：`Specificity / Locality`
- 辅看：`Edit Locality at Slot Level`

### `Composition-to-Belief Decoding`

- 主看：`Joint Belief Accuracy`
- 主看：slot/value F1
- 辅看：`Propagation Accuracy`

### `Whole system under budget`

- 主看：`Compression Fidelity`

## 15. 本地优先评估协议

### 15.1 Protocol

1. 先跑本地 intrinsic evaluation
2. 再决定是否进入 benchmark canary
3. full benchmark 只在用户要求后执行

### 15.2 当前阶段记录规则

当前先记录指标，不设自动门槛。

每次实验至少记录：

- `Recall@k`
- `NDCG@k`
- `Joint Belief Accuracy`
- slot/value F1
- `Update Success`
- `Propagation Accuracy`
- `Specificity / Locality`
- `Compression Fidelity`
- `Edit Locality at Slot Level`

并记录：

- `3 seeds`
- `mean ± std`

## 16. Benchmark canary

### 16.1 当前原则

benchmark 只作评测源，不作主要训练源。

### 16.2 Canary 设计

- `PersonaMem canary = 64`
- `LongMemEval-S canary = 64`

分层采样要求：

1. PersonaMem 按 question type 分层
2. LongMemEval-S 按 temporal / update / recall 类型分层

### 16.3 当前执行状态

当前阶段只记录协议，不主动运行 full benchmark。

## 17. 默认网络与训练超参数

### 17.1 Backbone

- `google/flan-t5-base`
- retrieval encoder 默认与主 encoder 共享参数

### 17.2 Resampler

- `top_k = 8`
- `m = 8`
- `L = 8`
- `d = 768`
- `layers = 2`
- `heads = 12`
- `dropout = 0.1`

### 17.3 LoRA

- `rank = 16`
- `alpha = 32`
- `dropout = 0.05`

## 18. 输出目录

stage-2 结果统一放在：

- `outputs_v2/`

默认结构：

- `outputs_v2/checkpoints/`
- `outputs_v2/runs/`
- `outputs_v2/evals_local/`
- `outputs_v2/evals_benchmark/`
- `outputs_v2/tables/`
- `outputs_v2/figures/`
- `outputs_v2/logs/`
- `outputs_v2/cache_manifests/`
- `outputs_v2/artifacts/`

## 19. V2.0 必做 ablation

1. `T5 route vs Optimus-like`
2. `mean pooling vs light resampler`
3. `belief JSON decoding vs direct answer decoding`
4. `merge-only vs merge+overwrite`
5. `single-bank vs core+residual`
6. `core/residual assignment randomization`
7. `budget sweep`
8. `with/without update pool`
9. `with/without persona pool`
10. `with/without state pool`
11. `with/without synthetic pool`

## 20. 当前未决但不阻塞实现的事项

以下事项允许在实现期继续微调，但不构成方法空洞：

1. `DuLeMon / REALTALK / MS-TOD` 是否纳入第二轮训练
2. answer projection 的最终 prompt 细节
3. 各 loss 权重的最终微调值

## 21. 当前设计结论

V2.0 当前已经完成方法级闭环：

1. 有明确对象定义
2. 有明确 memory 结构
3. 有明确数据-任务映射
4. 有明确指标-模块映射
5. 有明确 parser / training / eval / output 规则

后续工作重点不再是继续抽象讨论，而是把这些设计落成最小实现骨架和本地 evaluation 管线。
