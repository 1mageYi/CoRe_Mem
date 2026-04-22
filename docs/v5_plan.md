# V5 Plan: Core-Residual Latent Substrate

## 0. 版本定位

`v5` 的目标不是继续围绕 benchmark 做局部修补，而是把 CoRe_Mem 推进成一个真正可训练、可验证、可发表的 personalized latent memory system。

核心判断：

- 当前 `v32 / v33 / v4` 已经证明 learned path 能产生局部和 full-holdout 层面的正向信号。
- 但当前主系统仍然偏 text-centered：slot / canonical_gloss / belief JSON 是主要语义载体，latent 更多承担 retrieval / rerank / auxiliary readout。
- 顶会级贡献不能停留在“更好的规则 + 更好的 option scorer”，必须证明 `core + residual` latent memory 本身承载了可迁移、可读出、可干预的个性化知识。

因此 `v5` 的主题是：

> 用 pretrained encoder 作为感知底座，训练一个 core-residual latent substrate，让 stable persona traits 进入 core，让 episodic / update-sensitive facts 进入 residual，并用 query-conditioned latent reader 直接产出 belief / answer evidence。

## 1. 硬约束

这些约束优先级高于短期指标：

- 不允许 fallback。
- 不允许 shortcut。
- 不允许 benchmark-specific heuristic。
- 不允许 benchmark leakage。
- 不允许为了 PersonaMem 的 `a/b/c/d` 形式训练 memory writer / reader / controller / latent substrate。
- 不允许把 provider prompt trick 写成模型能力提升。
- 所有最终结论必须同时报告 `no-calibration` 和 `calibrated answer head` 两条线。
- `PersonaMem` gold 只允许在严格隔离的 train split 上校准薄 answer/readout head。
- split 必须按 `shared_context_id` / persona 分组，不能让同一 persona 的 context 同时出现在 train 与 eval。

## 2. 当前系统结论

### 2.1 Local / Provider 结论

`local/readout` 是 memory probe，用来判断 latent / belief / answer head 能不能自己从 memory state 读出正确答案。

`provider/API` 是 downstream consumer，用来判断外部 LLM 在看到 CoRe_Mem 选出的 memory view 后能否答对。它不是 latent substrate 的主训练目标。

当前 API 实际看到的是 selected memory view，而不是完整原始历史：

- `Question`
- `Belief JSON`
- `Evidence block`
- `Answer-head candidate`
- `Options`

它看不到完整 raw dialogue history、全部 memory slots、全部 residual/core state 或未选中的 memory。因此 provider gap 不能直接解释成“API 比 local 弱”，更准确的说法是：provider 是另一个消费 selected memory view 的下游决策器。

### 2.2 PersonaMem 结论

当前本地 PersonaMem 32k 变体包含：

- `589` 条 question
- `37` 个 shared contexts
- 全部为四选一 `a/b/c/d`
- `512` 是我们此前的 capped subset，不是官方全量

这会带来真实风险：如果直接用 gold answer 训练主模型，模型可能学习 option geometry，而不是真正学习 memory。因此 `v5` 允许 gold calibration，但只允许训练薄 answer head，且必须严格按 persona/context 隔离。

### 2.3 Stage2 32k 结论

`32k` prepared stage2 数据不能被称为大规模预训练数据。它更准确的角色是 domain warm-up / task adaptation：

- 可用于 warm-up pretrained encoder adaptor、write controller、belief decoder。
- 不足以从零训练强 latent model。
- 不能作为顶会级“数据规模充分”的唯一证据。

## 3. 目标架构

目标路径：

```text
dialogue history
  -> pretrained encoder
  -> learned write controller
  -> core latent memory + residual latent slots
  -> query-conditioned latent reader
  -> belief decoder
  -> answer/readout head
```

模块职责：

- `Pretrained encoder`: 比较 `BGE`、`E5`、`Contriever`。先 frozen + adapters，若显著不足再做 LoRA / light finetune。
- `Core latent memory`: 保存稳定 persona traits、长期 preferences、跨场景一致事实。
- `Residual latent slots`: 保存 episodic facts、recent updates、conflicts、temporally sensitive evidence。
- `Write controller`: 学习 `new / merge / overwrite / stale / promote_to_core / keep_residual`。
- `Latent reader`: query-conditioned cross-attention over core + residual，不依赖 text slot 作为唯一语义源。
- `Belief decoder`: 作为 interpretable bottleneck，不再是 memory 的唯一载体。
- `Answer/readout head`: 用于 probe 和 downstream task；PersonaMem 的 calibrated 版本必须单独报告。

## 4. 训练数据计划

### 4.1 Pretrained Backbone

允许下载和使用 HuggingFace pretrained models。第一批候选：

- `BGE` 系列
- `E5` 系列
- `Contriever` 系列

比较原则：

- frozen encoder 下的 retrieval / latent-read 基线。
- adapters / LoRA 后的 domain adaptation 增益。
- 对 text ablation / shuffled latent 的鲁棒性。
- 不以单一 PersonaMem 分数选 backbone。

### 4.2 Stage2 32k Warm-Up

用途：

- 写入控制器 warm-up。
- latent query-memory contrastive alignment。
- belief reconstruction / denoising。
- core-residual promotion / stale detection 的弱监督。

边界：

- 这是 domain warm-up，不是大规模预训练。
- 不允许把 benchmark answer label 混入此阶段。

### 4.3 PersonaMem Context Self-Supervision

可使用 PersonaMem 的 raw shared context/dialogue 来构造无 gold-answer 的训练任务：

- masked memory reconstruction
- future query relevant evidence prediction
- core vs residual assignment
- contradiction / update / stale detection
- query-conditioned latent retrieval

禁止事项：

- 不使用 `a/b/c/d` gold answer 训练 writer / reader / controller / latent substrate。
- 不用 option text 构造 memory writer 的监督目标。

### 4.4 PersonaMem Gold Calibration

允许但严格隔离：

- 只训练薄 answer/readout head。
- 只使用 train split。
- split 必须按 `shared_context_id` / persona 分组。
- latent writer / latent reader / controller / belief decoder 不得读取 gold answer。
- 最终报告必须同时给 `no-calibration` 与 `calibrated`。

### 4.5 LLM Teacher

允许使用 MiniMax-M2.7 作为 teacher，但只用于从 raw dialogue 生成 memory/belief/update 类监督：

- teacher 输入必须是 raw dialogue / context / query。
- teacher 不允许条件化在现有 silver label 上。
- teacher 不允许看到 PersonaMem gold answer。
- teacher 输出必须记录 provenance、model、prompt version、filter result。

## 5. 训练目标

第一批 objective：

- `contrastive query-memory alignment`: query 与 relevant latent memory 对齐。
- `write action loss`: 学习 new / merge / overwrite / stale / promote。
- `slot target loss`: 写入哪个 core/residual slot。
- `latent denoising`: 在 canonical text dropout 下恢复 memory semantics。
- `belief reconstruction`: 从 latent reader 输出恢复 interpretable belief。
- `answer/readout CE`: 只在隔离 calibration split 上训练薄 answer head。
- `core-residual consistency`: stable traits 更偏 core，recent/update-sensitive facts 更偏 residual。
- `staleness regularization`: 过期事实应降低 read priority。

## 6. 反 Shortcut 评估协议

所有 v5 结果必须至少报告这些视图：

- `full`: text + latent。
- `latent-only`: mask canonical_gloss / textual value，只保留 latent。
- `text-only`: 禁用 latent reader。
- `shuffled-latent`: memory text 不变，latent 随机打乱。
- `shuffled-memory/persona`: query 与他人 memory 配对。
- `core-only`: 只读 core。
- `residual-only`: 只读 residual。
- `no-controller`: 禁用 learned write controller。
- `option-only baseline`: 只给 question + options，不给 memory。
- `answer-label permutation`: 检测是否学了 option prior。
- `option paraphrase`: 检测是否依赖固定 option 表面形式。
- `gold-memory oracle`: 标定 answer head 上界，不作为系统结果。

关键成功标准：

- `latent-only` 必须显著高于 random。
- `full` 必须高于 `text-only`。
- `shuffled-latent` 必须明显低于 `latent-only`。
- `core-only` 与 `residual-only` 应呈现可解释分工。
- learned controller 必须优于 disabled / symbolic-only control。
- no-calibration 结果必须独立成立；calibration 只能作为薄 readout 上界。

## 7. 里程碑

### M1: 文档和契约锁定

交付：

- `docs/v5_plan.md`
- `.agent-os` project-index / todo / decision / milestone 同步
- 明确 no-shortcut 和 PersonaMem gold isolation contract

成功标准：

- 后续 agent 可以只从项目文档恢复 v5 目标、边界和验收方式。

### M2: Backbone Comparison Harness

交付：

- BGE / E5 / Contriever 的统一 encoder wrapper。
- frozen baseline。
- adapter / LoRA 开关。
- retrieval / latent read / text ablation eval。

成功标准：

- 给出 backbone 选择表，不只看 PersonaMem exact。

### M3: PersonaMem Context Episode Builder

交付：

- 按 `shared_context_id` 生成 episode。
- gold-answer-free self-supervised samples。
- persona-grouped split。

成功标准：

- 能明确证明 memory training 未接触 PersonaMem answer labels。

### M4: Trainable Core-Residual Latent State

交付：

- learned write controller。
- core/residual latent bank update。
- promote / stale / overwrite action。

成功标准：

- controller ablation 显示 learned > disabled / symbolic control。

### M5: Query-Conditioned Latent Reader

交付：

- cross-attention latent reader。
- latent-only readout path。
- text dropout / shuffled latent ablations。

成功标准：

- latent-only 明显强于 random，shuffled-latent 明显下降。

### M6: Warm-Up + Adaptation Training

交付：

- 32k stage2 warm-up。
- PersonaMem context self-supervised adaptation。
- 训练耗时、GPU、吞吐记录。

成功标准：

- warm-up 后在 held-out self-supervised metrics 上正增益。

### M7: PersonaMem Full 589 Evaluation

交付：

- local/readout no-calibration。
- local/readout calibrated。
- provider auxiliary。
- full ablation report。

成功标准：

- 不只看 exact，还要报告 anti-shortcut matrix。

### M8: Paper-Level Evidence Package

交付：

- architecture diagram。
- ablation table。
- data isolation table。
- failure taxonomy。
- top-conference related-work positioning。

成功标准：

- 能回答“为什么这不是 text slot RAG / rule system / benchmark trick”。

## 8. 下一步执行顺序

建议下一轮先执行：

1. 建立 v5 data isolation checker，验证 PersonaMem split 和 gold-answer 使用边界。
2. 建立 BGE / E5 / Contriever encoder harness。
3. 实现最小 latent-only readout eval，先不碰 provider。
4. 在 stage2 32k warm-up + PersonaMem context self-supervision 上跑首轮 frozen encoder。
5. 只有当 latent-only / shuffled-latent 对比成立后，再进入 calibrated answer head。

## 9. 当前不做的事情

- 不继续把主目标放在 LongMemEval 与 PersonaMem 同时刷分。
- 不把 provider prompt 规范化当作主创新。
- 不靠 PersonaMem `a/b/c/d` gold 训练 memory substrate。
- 不继续用更复杂的 rule parser 作为核心改进。
- 不把 8-sample smoke 当成策略依据。

## 10. 预期结论形式

理想的 v5 结论不是：

> 我们在 PersonaMem 上调参后 exact 更高。

而是：

> CoRe_Mem learns a core-residual latent memory substrate that separates stable persona traits from episodic residuals, supports query-conditioned latent readout under text ablation, and improves personalized decision making without benchmark-label leakage.
