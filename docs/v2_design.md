# V2.0 Design

## 文档定位

本文档是第二阶段 `V2.0 structured latent-slot memory` 的实现级设计文档。

规则：

- 本文档从属于真源 [requirements.md](/home/image/workspace/CoRe_Mem/docs/requirements.md)。
- 若本文档与真源冲突，以真源为准。
- 本文档目标是把已经由用户确认过的第二阶段方案收束成可执行规格，而不是保留讨论过程。

## 1. 第二阶段目标

第二阶段目标不是继续在原始长上下文 prompt 上做工程修补，而是建立一个更强的 latent memory 主线：

- memory 基本形式是 `structured latent slots`
- memory 内容是开放域的
- answer-time 不回放原始文本 chunk
- 主链路是 `observation -> slot -> belief -> answer`

第二阶段默认保留第一阶段 v1 作为 naive baseline 与可追溯参考。

## 2. 方法总览

### 2.1 核心思想

V2.0 不是：

- 文本 chunk RAG
- 单一全局向量记忆
- 直接用 gloss 当答案证据

V2.0 是：

- `structured latent slots`
- `core + residual` 双银行
- query-conditioned latent composition
- belief JSON decoding

### 2.2 主线模块

1. observation parser
2. slot encoder
3. retrieval key head
4. residual bank
5. core bank
6. lifecycle module
7. consolidation
8. light cross-attention resampler
9. Flan-T5 belief decoder
10. answer projection

## 3. Observation / Slot / Belief

### 3.1 Observation

Observation 是从原始对话中抽出来的 memory-bearing unit。

Observation JSON 必须至少包含：

- `obs_id`
- `source_dataset`
- `source_dialogue_id`
- `source_turn_id`
- `session_id`
- `speaker`
- `entity`
- `relation`
- `value`
- `value_type`
- `time_scope`
- `status_hint`
- `polarity`
- `confidence`
- `evidence_text`
- `canonical_gloss`
- `metadata`

Observation parser 默认采用混合策略：

- `rule-first`
- `model-second`

优先用规则抽显式 preference / constraint / update / temporal cue；规则低置信时，回退到同一 backbone 的结构化抽取分支。

### 3.2 Slot

Slot 是 observation 经编码后写入 memory 的单位。

每个 slot 默认包含：

- `slot_id`
- `bank`
- `relation`
- `entity`
- `retrieval_key`
- `latent_tokens`
- `soft_role_scores`
- `confidence`
- `first_seen_ts`
- `last_update_ts`
- `revision_count`
- `active_flag`
- `revision_parent`
- `canonical_gloss`

默认：

- 每个 slot 使用连续 latent tokens
- 默认 `m = 8` 个 latent tokens

### 3.3 Belief

Belief 是在 query 条件下，从若干 slots 恢复出的当前状态。

Belief JSON 必须至少包含：

- `query_id`
- `entity`
- `query_type`
- `belief_items`
- `global_consistency`

其中 `belief_items` 至少包含：

- `relation`
- `value`
- `status`
- `time_scope`
- `confidence`
- `support_slot_ids`

默认 query_type：

- `single_fact`
- `multi_fact`
- `update_sensitive`
- `temporal`

## 4. Memory 结构

### 4.1 双银行

#### Core

Core 负责：

- 长期稳定 persona 状态
- 当前有效且被多次支持的信息
- 被 consolidation 确认后的 active state

#### Residual

Residual 负责：

- 新信息
- 近期事件
- 候选更新
- 冲突状态
- 等待晋升或淘汰的局部记忆

### 4.2 Lifecycle

状态：

- `active`
- `stale`
- `inactive`

动作：

- `new`
- `merge`
- `overwrite`
- `evict`

### 4.3 更新原则

新 observation 先进入 residual 视角判断：

- 无高相似且 write score 高 -> `new`
- 高相似且一致 -> `merge`
- 高相似且冲突 -> `overwrite`
- 低价值 -> `ignore`

Core 不做高频抖动更新。Core 的变化主要通过：

- `reinforce_core`
- `promote_residual_to_core`
- `replace_core_via_update`

### 4.4 Consolidation

Consolidation 是显式模块，负责：

- residual -> core promotion
- core stale handling
- duplicate merge
- eviction

## 5. Latent Composition

### 5.1 为什么需要 composition

retrieval 得到的是多个相关 slots；decoder 需要的是一个适合当前 query 的固定长度临时工作记忆。

composition 的职责是：

- 从多个相关 slots 中聚合信息
- 形成 query-conditioned latent workspace
- 为 belief decoder 提供固定长度输入

### 5.2 主线结构

主线采用 `Light Cross-Attention Resampler`：

1. query 编码为 `q`
2. 检索 top-k slots
3. 将这些 slots 的 latent tokens 拼成 memory token set
4. 使用固定数量的 learnable latent queries 做 2 层 cross-attention
5. 得到固定长度的 composed latent tokens `Z_q`
6. `Z_q` 条件化 belief decoder

### 5.3 默认超参数

- `top_k slots = 8`
- `latent tokens per slot (m) = 8`
- `composed latent tokens (L) = 8`
- `hidden dim = 768`
- `cross-attention layers = 2`
- `attention heads = 12`
- `dropout = 0.1`

## 6. Belief Decoder

### 6.1 主线

主线 decoder 锁定为：

- `google/flan-t5-base`
- belief JSON 作为主输出目标

### 6.2 训练方式

可训练部分：

- resampler
- slot projection
- retrieval key head
- lifecycle head
- decoder LoRA / adapter

默认 LoRA：

- `rank = 16`
- `alpha = 32`
- `dropout = 0.05`

尽量冻结：

- backbone 主体参数

### 6.3 答案投影

默认做法：

- 本地 intrinsic eval：直接评 belief JSON
- benchmark inference：belief JSON -> 短 evidence block -> benchmark answer model

## 7. 训练目标

V2.0 默认分四个训练任务：

1. `Slot Autoencoding`
2. `Retrieval Alignment`
3. `Lifecycle Prediction`
4. `Composition-to-Belief Decoding`

训练顺序：

1. `Slot Autoencoding`
2. `Retrieval Alignment`
3. `Lifecycle Prediction`
4. `Composition-to-Belief Decoding`
5. `Joint fine-tuning`

## 8. 数据策略

### 8.1 总原则

- `public-datasets-first`
- `synthetic-minimal`
- benchmark 不是 primary training source

### 8.2 主训练数据池

- `State / Belief Pool`
  - `SGD`
  - `MultiWOZ 2.4`
- `Persona / Preference Pool`
  - `Persona-Chat`
  - `Training Millions of Personalized Dialogue Agents`
- `Update / Propagation Pool`
  - `MQUAKE`
  - `ReCoE`
- `Minimal Synthetic Pool`
  - 仅补 corner cases

### 8.3 第二优先级候选

- `DuLeMon`
- `REALTALK`
- `MS-TOD`

默认不进入第一版主训练，除非后续确认接入成本可控。

### 8.4 Synthetic 的用途

仅用于：

- overwrite stress cases
- budget stress cases
- conflict bundles
- multi-fact composition

## 9. 本地评估

### 9.1 指标家族

采用 `4 个现成指标家族 + 2 个少量自定义指标`。

默认至少记录：

- `Recall@k`
- `NDCG@k`
- `Joint Belief Accuracy`
- slot/value F1
- `Update Success`
- `Propagation Accuracy`
- `Specificity / Locality`
- `Compression Fidelity`
- `Edit Locality at Slot Level`

### 9.2 运行规则

- 默认 `3 seeds`
- 汇报 `mean ± std`
- 先本地 intrinsic，后 benchmark canary

## 10. Benchmark 协议

### 10.1 基本原则

- benchmark 是 evaluation source，不是 primary training source
- benchmark-derived proxy supervision 只用于 intrinsic eval、ablation、failure analysis

### 10.2 Canary

默认固定：

- PersonaMem canary: `64`
- LongMemEval-S canary: `64`

所有方法共享同一 canary 子集。

### 10.3 正式 benchmark

只有在方法完成本地评估和 canary 记录后，才推进正式 benchmark。

## 11. 主要 ablation

### 必做

- `T5` vs `Optimus-like`
- mean pooling vs light resampler
- direct answer decoding vs belief JSON decoding
- with / without retrieval loss
- with / without lifecycle loss
- with / without field auxiliary loss
- single-bank vs core+residual
- core/residual assignment randomization
- budget sweep

### 说明

advisor 提出的 `core vs residual assignment randomization` 已正式纳入第二阶段 ablation 集，用于检验双银行设计本身是否真有方法价值。

## 12. 输出目录

第二阶段所有产物统一放在 `outputs_v2/`。

默认子目录：

- `checkpoints/`
- `runs/`
- `evals_local/`
- `evals_benchmark/`
- `tables/`
- `figures/`
- `logs/`
- `cache_manifests/`
- `artifacts/`

其中 `artifacts/` 应保存：

- belief dumps
- slot dumps
- attention / resampler 分析
- 关键错误案例

## 13. 与第一阶段的关系

- 第一阶段真相、结果、平台与文档继续保留
- 第二阶段是 stage-2 研究主线
- v1 作为 naive baseline 尽量保留
- 如未来需要让 V2.0 成为默认主线，应在不破坏第一阶段可追溯性的前提下进行

## 14. 当前下一步

当前全局 top next action 为：

- 实现第二阶段 `TD-015`
- 内容包括：
  - observation normalization
  - belief JSON target generation
  - public datasets ingestion skeleton
  - `outputs_v2/` 目录与最小测试骨架
