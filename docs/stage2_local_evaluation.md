# Stage-2 本地评测体系

## 文档目的

本文档说明第二阶段 `V2.0 structured latent-slot memory` 的本地 intrinsic evaluation 如何使用、评什么、每个接口是什么，以及每个模块分别对应哪些评测指标。

本地评测的目标不是替代 benchmark，而是：

1. 在不消耗外部 API 的前提下先验证方法主链
2. 把 parser / slot / retrieval / update / belief / answer 这些错误拆开
3. 为后续 budget sweep、ablation、canary 和 full benchmark 提供低成本筛选依据

---

## 1. 当前评测对象

当前 stage-2 本地评测覆盖 10 个模块：

1. `parser`
2. `slot_encoder`
3. `query_encoder`
4. `retrieval_key_head`
5. `lifecycle`
6. `consolidation`
7. `core_residual_memory`
8. `resampler`
9. `belief_decoder`
10. `answer_projection`

这些模块对应当前代码位置：

- parser: [parser.py](/home/image/workspace/CoRe_Mem/src/core_mem/v2/parser.py)
- slot/query encoder: [encoder.py](/home/image/workspace/CoRe_Mem/src/core_mem/v2/encoder.py)
- lifecycle: [lifecycle.py](/home/image/workspace/CoRe_Mem/src/core_mem/v2/lifecycle.py)
- consolidation: [consolidation.py](/home/image/workspace/CoRe_Mem/src/core_mem/v2/consolidation.py)
- resampler: [resampler.py](/home/image/workspace/CoRe_Mem/src/core_mem/v2/resampler.py)
- belief decoder: [decoder.py](/home/image/workspace/CoRe_Mem/src/core_mem/v2/decoder.py)
- answer projection: [projection.py](/home/image/workspace/CoRe_Mem/src/core_mem/v2/projection.py)
- end-to-end system: [system.py](/home/image/workspace/CoRe_Mem/src/core_mem/v2/system.py)

---

## 2. 输入产物

本地评测直接消费 stage-2 prepared manifest：

- 默认输入：[stage2_prepared_samples_manifest.json](/home/image/workspace/CoRe_Mem/outputs_v2/artifacts/stage2_prepared_samples_manifest.json)

该 manifest 由 [prepare_stage2_data.py](/home/image/workspace/CoRe_Mem/scripts/prepare_stage2_data.py) 生成，内部会引用 4 类 task files：

1. `slot_autoencoding.jsonl`
2. `retrieval_alignment.jsonl`
3. `lifecycle_prediction.jsonl`
4. `composition_to_belief.jsonl`

每一类 task row 现在都保留 `_meta`：

- `dataset`
- `dataset_key`
- `sample_id`
- `task_name`

所以本地评测可以按：

- 全量
- 指定数据集
- 指定预算

做分层分析。

---

## 3. 命令行接口

主入口是：

- [eval_stage2_local.py](/home/image/workspace/CoRe_Mem/scripts/eval_stage2_local.py)

### 3.1 最常用命令

全量本地评测：

```bash
python scripts/eval_stage2_local.py \
  --prepared-manifest outputs_v2/artifacts/stage2_prepared_samples_manifest.json \
  --output-root outputs_v2 \
  --json
```

指定 retrieval top-k 与 budget sweep：

```bash
python scripts/eval_stage2_local.py \
  --prepared-manifest outputs_v2/artifacts/stage2_prepared_samples_manifest.json \
  --output-root outputs_v2 \
  --top-k 8 \
  --budget 1 \
  --budget 2 \
  --budget 4 \
  --budget 8 \
  --json
```

只评某个数据集：

```bash
python scripts/eval_stage2_local.py \
  --prepared-manifest outputs_v2/artifacts/stage2_prepared_samples_manifest.json \
  --output-root outputs_v2 \
  --dataset sgd \
  --json
```

### 3.2 CLI 参数

- `--prepared-manifest`
  - 输入 manifest 路径
  - 默认：`outputs_v2/artifacts/stage2_prepared_samples_manifest.json`

- `--output-root`
  - 输出根目录
  - 默认：`outputs_v2`

- `--top-k`
  - retrieval / belief recovery 时选多少个 slot
  - 默认：`8`

- `--budget`
  - budget sweep 的 slot budget
  - 可重复传入
  - 默认：`1, 2, 4, 8`

- `--dataset`
  - 只评指定数据集
  - 可重复传入
  - 例如：`sgd`、`multiwoz24`、`personachat`、`mquake`、`recoe`

- `--json`
  - 打印 JSON 结果

---

## 4. 输出接口

运行后会写出 3 份主要产物：

1. 详细 JSON 报告
  - `outputs_v2/evals_local/<timestamp>_stage2_local_eval.json`

2. 汇总表
  - `outputs_v2/tables/<timestamp>_stage2_local_eval_summary.csv`

3. budget sweep 表
  - `outputs_v2/tables/<timestamp>_stage2_local_eval_budget_sweep.csv`

### 4.1 JSON 报告结构

顶层字段：

- `prepared_manifest`
- `selected_datasets`
- `top_k`
- `budgets`
- `task_counts`
- `metrics`
- `families`
- `modules`
- `module_inventory`
- `per_dataset`
- `budget_sweep`

### 4.2 `metrics`

这是所有主指标的平铺总览，用于快速看一版模型当前强弱。

### 4.3 `families`

这是按指标家族聚类后的视图，和真源文档中的 4 大家族 + 2 个自定义指标对齐。

### 4.4 `modules`

这是按模块聚类后的视图，用于回答“某个模块到底是被什么指标评的”。

### 4.5 `budget_sweep`

这是按 budget 聚合后的压缩保真结果。键是 budget，值里至少包含：

- `joint_belief_accuracy`
- `belief_item_recall`
- `support_slot_recall`

---

## 5. 指标家族

本地评测现在分成 6 组：

1. `parser_slot_family`
2. `retrieval_family`
3. `belief_recovery_family`
4. `update_propagation_family`
5. `specificity_locality_family`
6. `compression_family`

其中：

- `retrieval / belief / update / specificity-locality` 对齐真源里的 4 个主要指标家族
- `compression` 对齐自定义 `Compression Fidelity`
- `edit_locality_at_slot_level` 作为自定义局部性指标保留在 locality family 中
- `parser / slot` 属于本地补充诊断家族，用来解释“错误是发生在 observation/slot 之前，还是发生在 retrieval/belief 之后”

---

## 6. 每个指标测什么

### 6.1 Parser / Slot 诊断指标

- `parser_replay_coverage`
  - 重新用 parser 去解析 `input_observation.evidence_text`
  - 统计能否至少解析出一个 observation
  - 反映 parser 是否“有抓取能力”

- `parser_replay_exact`
  - 重新解析后，是否能恢复出与 gold observation 相同的 `relation + value`
  - 反映 parser 是否“抓得对”

- `slot_relation_accuracy`
  - 用 `slot_encoder` 对 gold observation 编码后，slot 的 `relation` 是否和 target record 一致
  - 反映 observation 到 slot 的结构保持能力

- `slot_value_recoverability`
  - 编码后 slot 的 `canonical_gloss` 是否仍可恢复 target value
  - 反映 slot 表示是否还保留关键值信息

### 6.2 Retrieval 指标

- `recall_at_k`
  - 查询后 top-k slots 是否命中任一 gold support slot
  - 主看 retrieval 能不能把“对的 slot”拉回来

- `ndcg_at_k`
  - 命中的 slot 排名越靠前，分数越高
  - 主看 retrieval 排序质量，不只是有没有命中

- `support_slot_recall`
  - 在 belief decode 样本里，预测 belief 使用的 support slots 覆盖了多少 gold support slots
  - 用来连接 retrieval 和 belief 两个阶段

### 6.3 Belief Recovery 指标

- `joint_belief_accuracy`
  - 预测 belief 的 `relation + value` 集合是否与 gold 完全一致
  - 这是 belief 层最硬的指标

- `slot_value_f1`
  - 预测与 gold 的 `relation + value` 集合 F1
  - 比 `joint_belief_accuracy` 更平滑，适合看部分恢复情况

- `answer_exact_match`
  - 用 `answer_projection` 从预测 belief 生成 answer，再和 gold belief 投影出来的 answer 比较
  - 反映 belief 到 answer 的末端可用性

### 6.4 Update / Propagation 指标

- `update_action_accuracy`
  - lifecycle 模块预测的 action 是否等于 gold `target_action`
  - 当前 action 包括：`overwrite / merge / new / ignore`

- `update_flag_accuracy`
  - lifecycle 预测的 flags 是否等于 gold flags
  - 当前主要看：
    - `promote`
    - `stale_old`

- `update_success`
  - `update_action_accuracy` 和 `update_flag_accuracy` 的平均
  - 用于给 update 成功与否一个单值概览

- `propagation_accuracy`
  - 对 lifecycle 样本，先执行 update，再用对应 query 做 belief recovery
  - 只有当 post-update belief 与 gold 完整一致时才算成功
  - 反映“更新是否真的传播到了可查询状态”

### 6.5 Specificity / Locality 指标

- `specificity_locality`
  - 一次 update 后，不相关 slots 是否保持不变
  - 反映更新是否局部、是否污染了无关记忆

- `edit_locality_at_slot_level`
  - 以 slot 为单位计算 locality
  - 当前实现看：
    - `active_flag`
    - `bank`
    - `canonical_gloss`
  - 如果这些在无关 slots 上都没变，说明 edit 比较局部

### 6.6 Compression 指标

- `compression_fidelity`
  - 在给定 budget sweep 下，预测 belief item 对 gold belief item 的平均保留率
  - 这是“固定 memory budget 下还保住多少信息”的主指标

- `avg_composed_token_count`
  - resampler 最终生成的 composed latent token 数量均值
  - 更偏诊断，不是最终优先指标

---

## 7. Budget Sweep 是怎么做的

对每个 `composition_to_belief` 样本：

1. 用 `query_encoder + retrieval_key` 对 `memory_slots` 排序
2. 依次取 budget = `1 / 2 / 4 / 8 / ...`
3. 用该 budget 下的 top-b slots 做 belief decode
4. 记录：
  - `joint_belief_accuracy`
  - `belief_item_recall`
  - `support_slot_recall`

最终：

- 每个 budget 会有一组聚合结果
- `compression_fidelity` 取整个 budget sweep 中 `belief_item_recall` 的平均值

所以这里评的不是“模型有没有更大参数”，而是：

- 同一套 memory 表示在更小预算下掉信息掉得快不快

---

## 8. 每个模块分别怎么测

### 8.1 `parser`

主看：

- `parser_replay_coverage`
- `parser_replay_exact`

解释：

- 如果 coverage 低，说明 parser 连 observation 都抓不出来
- 如果 coverage 高但 exact 低，说明 parser 有输出但 relation/value 常抓错

### 8.2 `slot_encoder`

主看：

- `slot_relation_accuracy`
- `slot_value_recoverability`

解释：

- relation 对不上，说明 observation 压缩后语义类型已经漂了
- value recover 不出来，说明 slot 表示保真不够

### 8.3 `query_encoder`

主看：

- `recall_at_k`
- `ndcg_at_k`

解释：

- 它和 `retrieval_key_head` 一起决定“query 能不能找到对的 slot”

### 8.4 `retrieval_key_head`

主看：

- `recall_at_k`
- `ndcg_at_k`
- `support_slot_recall`

解释：

- 如果 retrieval 指标高但 belief 很差，问题更可能在 resampler / decoder
- 如果 retrieval 指标低，后面 belief 再强也救不回来

### 8.5 `lifecycle`

主看：

- `update_action_accuracy`
- `update_flag_accuracy`
- `update_success`

解释：

- action 错，说明更新策略方向错了
- flags 错，说明 stale/promote 等细粒度控制错了

### 8.6 `consolidation`

主看：

- `specificity_locality`
- `edit_locality_at_slot_level`

解释：

- consolidation 的重点不是“改得多”，而是“只改该改的”

### 8.7 `core_residual_memory`

主看：

- `compression_fidelity`
- `edit_locality_at_slot_level`

解释：

- 它衡量双银行 memory 在有限 budget 下是否 still usable

### 8.8 `resampler`

主看：

- `joint_belief_accuracy`
- `slot_value_f1`
- `avg_composed_token_count`

解释：

- retrieval 已经把相关 slot 找回来后，resampler 决定这些 slot 能否被组织成一个可读 latent workspace

### 8.9 `belief_decoder`

主看：

- `joint_belief_accuracy`
- `slot_value_f1`
- `propagation_accuracy`

解释：

- 它不只是读单事实，还要在 update / propagation 场景里恢复正确 belief

### 8.10 `answer_projection`

主看：

- `answer_exact_match`

解释：

- 如果 belief 层已经正确但 answer 仍错，那问题就在 projection，而不是 retrieval 或 decoder

---

## 9. 结果怎么解读

一个常见的排障顺序是：

1. 先看 `parser_replay_exact` 和 `slot_value_recoverability`
2. 再看 `recall_at_k / ndcg_at_k`
3. 再看 `joint_belief_accuracy / slot_value_f1`
4. 最后看 `update_success / propagation_accuracy / specificity_locality`

经验上：

- `retrieval` 低 + `belief` 低
  - 优先排 query encoder / retrieval key

- `retrieval` 高 + `belief` 低
  - 优先排 resampler / decoder

- `belief` 高 + `answer_exact_match` 低
  - 优先排 answer projection

- `update_success` 高 + `specificity_locality` 低
  - 说明能改对目标，但副作用太大

- `compression_fidelity` 随 budget 很快下降
  - 说明 memory 表示还不够紧凑，或 retrieval 排序在小 budget 下不稳

---

## 10. 当前实现的边界

这套 local eval 现在已经不是 placeholder，但仍有边界需要明确：

1. 它是 stage-2 intrinsic eval，不是 benchmark 最终成绩
2. 它当前主要评估 repo 内的结构化模块链路，不直接替代 full benchmark
3. 如果后面引入真正训练后的 checkpoint，建议继续复用同一套指标和输出格式
4. 当前 locality 仍受 prepared sample 结构影响；如果后续加入更强的 conflict bundles 和 richer context，它会更有判别力

---

## 11. 相关代码与测试

- 入口脚本：[eval_stage2_local.py](/home/image/workspace/CoRe_Mem/scripts/eval_stage2_local.py)
- 核心实现：[eval_local.py](/home/image/workspace/CoRe_Mem/src/core_mem/v2/eval_local.py)
- pipeline 测试：[test_stage2_pipeline.py](/home/image/workspace/CoRe_Mem/tests/test_stage2_pipeline.py)
- local eval 测试：[test_stage2_local_eval.py](/home/image/workspace/CoRe_Mem/tests/test_stage2_local_eval.py)

如果只想快速确认这套评测能跑，最直接的命令就是：

```bash
python scripts/eval_stage2_local.py --json
```
