# V5.1 Plan: Real Pretrained Training For Core-Residual Latent Memory

## 0. 为什么需要 v5.1

`v5` managed run 达到了 `scripts/verify_stage2_v5_longrun.py --score-only = 52`，但这不是我们想要的科学结论。它主要完成了 scaffold / evidence-package 形状：

- PersonaMem gold isolation checker。
- gold-free context self-supervision builder。
- BGE / E5 / Contriever 名义上的 encoder comparison harness。
- core/residual toy train artifact。
- latent reader / ablation artifact。
- thin answer-head calibration artifact。
- PersonaMem full-589 local report。

但关键事实是：

- `latest_stage2_v5_encoder_compare.json` 显式记录 `evaluation_backend = deterministic_hashing_proxy`。
- `pretrained_weights_loaded = false`。
- `sentence_transformers / transformers` 没有在 v5 run 里实际参与 backbone comparison。
- `core_residual_train` 只用了 `200 train / 96 eval` 的 toy-scale CPU training。
- `PersonaMem full589` no-calibration accuracy 约 `24.3%`，calibrated overall 约 `26.8%`，接近四选一随机水平。

因此 v5.1 的第一原则是：

> scaffold 可以作为工程准备，但不能作为 research gain。真实目标必须是 pretrained weights + real data + real training + held-out eval + anti-shortcut ablation。

## 1. 本轮最高目标

v5.1 的目标是得到一个诚实的 real-training conclusion：

- 若真实 pretrained + real training 产生增益，保留并量化。
- 若训练后仍接近随机，必须如实记录 negative result，并分析瓶颈。
- 不允许再用 proxy / artifact completeness / toy train / option geometry 伪装成模型能力。

目标句：

> Train and evaluate a real pretrained-backbone core-residual latent memory system using real stage2 / PersonaMem context data, and determine whether learned latent memory improves over frozen, text-only, shuffled-latent, option-only, and random baselines.

## 2. 不可协商硬门槛

v5.1 verifier 必须有硬门槛。任何一条失败，都不能 stop / closeout 为 success。

- `pretrained_weights_loaded = true`。
- `evaluation_backend` 必须是 `sentence_transformers`、`transformers` 或明确的 HF model backend。
- 至少一个真实 backbone 成功加载：`BGE`、`E5`、`Contriever` 三选一是最低线，三者比较是目标线。
- 必须记录 model id、embedding dim、parameter count 或 module count、HF/cache path、load_seconds、device。
- 必须有真实训练 checkpoint。
- checkpoint 必须来自 real training script，而不是直接写 artifact。
- 主训练数据不能少于 `10k` stage2 train samples；目标用 full `24k` retrieval / slot / belief warm-up。
- PersonaMem gold answers 不得进入 encoder / writer / reader / controller / latent substrate。
- PersonaMem gold 只允许用于 strict context/persona split 的 thin answer-head calibration。
- PersonaMem no-calibration full589 必须显著高于 random / option-only baseline，或者明确记为 negative result。
- 若 no-calibration 仍约等于 `25%`，不能写成 latent memory improvement。
- Provider/API 只能作为 auxiliary，不参与 keep/stop 主门槛。

## 3. 数据计划

### 3.1 Stage2 32k Real Warm-Up

当前可用 split：

- `outputs_v2/artifacts/stage2_v27_32k/train/retrieval_alignment.jsonl`: `24000`
- `outputs_v2/artifacts/stage2_v27_32k/train/slot_autoencoding.jsonl`: `24000`
- `outputs_v2/artifacts/stage2_v27_32k/train/composition_to_belief.jsonl`: `24000`
- `outputs_v2/artifacts/stage2_v27_32k/train/lifecycle_prediction.jsonl`: `2774`
- validation / test 分别为 `4000 / 4000` 主任务样本，lifecycle 为 `462 / 462`

使用方式：

- `retrieval_alignment`: 训练 query-memory contrastive objective。
- `slot_autoencoding`: 训练 latent reconstruction / denoising。
- `composition_to_belief`: 训练 latent reader -> belief bottleneck。
- `lifecycle_prediction`: 训练 write controller action / stale / overwrite。

边界：

- `32k` 是 domain warm-up / adaptation，不称为大规模预训练。
- 它可以训练 adapter/projector/controller/reader，但不能替代外部 pretrained backbone。

### 3.2 PersonaMem Raw Context Self-Supervision

当前可用：

- `data/personamem/shared_contexts_32k.jsonl`: `37` contexts
- `data/personamem/questions_32k.csv`: `589` questions

使用方式：

- 用 shared context 构造 gold-answer-free episode tasks。
- 用 context/persona grouped split 防止同一 persona 泄漏。
- 只生成 masked evidence、future relevant evidence、core/residual assignment、staleness/update supervision。

禁止：

- 不用 `answer` 字段训练 substrate。
- 不用 `a/b/c/d` label 训练 substrate。
- 不用 option text 构造 memory writer / latent reader 监督。

### 3.3 PersonaMem Gold Calibration

仅允许：

- 训练 thin answer/readout head。
- 使用 grouped train split。
- 单独报告 calibrated。

必须同时报告：

- no-calibration full589
- calibrated full589
- option-only baseline
- random baseline
- gold-memory oracle 上界

## 4. 模型计划

### 4.1 Backbone Loading

第一阶段尝试真实加载：

- `BAAI/bge-base-en-v1.5`
- `intfloat/e5-base-v2`
- `facebook/contriever`

运行要求：

- 优先用 `conda run -n core_mem`。
- 优先 GPU `2`，但必须记录实际 device。
- 若网络 / cache / version 不兼容导致加载失败，记录 crash/blocker，不能回退到 hash proxy。
- 如果三者中只有一个能加载，就先用能加载者推进 real training，但 artifact 必须写清未完成三者比较。

### 4.2 Trainable Modules

最低真实训练模块：

- pretrained encoder frozen embedding。
- trainable projection from embedding to latent memory.
- core/residual bank classifier or controller。
- query-conditioned latent reader。
- contrastive readout head。

目标训练模块：

- adapter / LoRA on backbone 或 lightweight transformer reader。
- learned write controller for `new / merge / overwrite / stale / promote_to_core`。
- core/residual latent update module。
- belief bottleneck decoder。

禁止：

- 不能只训练 logistic/linear toy head 就声称 latent system trained。
- 不能只用 `_lexical_features` / hash features 声称 pretrained backbone。
- 不能只发布 artifact 而没有 checkpoint。

## 5. 训练目标

### 5.1 Gold-Free Objectives

主训练目标必须 gold-free：

- retrieval contrastive loss on stage2 retrieval_alignment。
- slot denoising / reconstruction loss on slot_autoencoding。
- lifecycle/write controller loss on lifecycle_prediction。
- belief reconstruction loss on composition_to_belief。
- PersonaMem context masked evidence recovery。
- query-conditioned latent read objective。

### 5.2 Calibration Objective

PersonaMem answer CE 只允许作用于 thin answer head：

- substrate frozen。
- encoder/reader/controller frozen。
- split by context/persona。
- calibration 结果单独报告。

## 6. Eval Plan

### 6.1 Backbone Eval

必须比较：

- frozen pretrained backbone。
- trained/adapted backbone or adapter。
- current v5 deterministic proxy baseline，作为 negative/scaffold reference。

指标：

- retrieval top1 / recall@k / MRR。
- shuffled target gap。
- text ablation gap。
- train-vs-frozen gain。

### 6.2 Latent Eval

必须报告：

- latent-only。
- text-only。
- full latent+text。
- shuffled-latent。
- shuffled-persona / shuffled-memory。
- core-only。
- residual-only。
- no-controller。
- random baseline。

硬判定：

- `latent-only > random` 只是最低线。
- `latent-only > shuffled-latent` 必须有明确 margin。
- `full > text-only` 必须成立。
- `trained > frozen` 必须成立，否则只能写 negative / no gain。

### 6.3 PersonaMem Full589 Eval

必须报告：

- no-calibration exact。
- calibrated exact。
- option-only exact。
- random baseline。
- context/persona grouped split。
- failure taxonomy。

硬判定：

- no-calibration 若不明显高于 `25%`，不能 claim memory ability。
- calibrated 若提升但 no-calibration 不提升，只能说明 answer head 学到 option mapping，不能说明 latent substrate 变强。

## 7. Mechanical Verifier 设计

新增 verifier：

```bash
python scripts/verify_stage2_v51_real_training.py --score-only
```

建议满分 `100`，但必须带 score cap：

- 若 `pretrained_weights_loaded != true`，最高分 `20`。
- 若 `evaluation_backend = deterministic_hashing_proxy`，最高分 `20`。
- 若没有 real checkpoint，最高分 `35`。
- 若 train samples `< 10000`，最高分 `45`。
- 若没有 held-out eval，最高分 `50`。
- 若 trained 不超过 frozen，最高分 `65`。
- 若 latent-only 不超过 shuffled-latent，最高分 `70`。
- 若 full 不超过 text-only，最高分 `75`。
- 若 PersonaMem no-calibration 不超过 random / option-only，最高分 `80`，且结论必须标为 negative。
- 若 gold isolation 不通过，最高分 `30`，无论其他指标如何。

Stop condition 不再是 artifacts 齐全，而是：

- real pretrained loaded
- real checkpoint exists
- gold isolation passes
- trained > frozen on gold-free held-out metric
- latent-only > shuffled-latent
- full > text-only
- no-calibration PersonaMem full589 > option-only/random by meaningful margin
- anti-shortcut summary passes

## 8. Required Artifacts

### 8.1 Real Backbone

`latest_stage2_v51_real_backbone_compare.json`

必须包含：

- `pretrained_weights_loaded`
- `backend`
- `loaded_backbones`
- `failed_backbones`
- `selected_backbone`
- `embedding_dim`
- `device`
- `load_seconds`
- `cache_or_model_path`
- `frozen_metrics`
- `uses_personamem_gold = false`

### 8.2 Real Training

`latest_stage2_v51_real_training.json`

必须包含：

- `checkpoint_path`
- `checkpoint_exists`
- `train_samples`
- `val_samples`
- `test_samples`
- `train_seconds`
- `device`
- `epochs`
- `batch_size`
- `loss_curve`
- `trained_modules`
- `frozen_modules`
- `uses_personamem_gold = false`

### 8.3 Latent Eval

`latest_stage2_v51_latent_eval.json`

必须包含：

- `frozen_metrics`
- `trained_metrics`
- `trained_beats_frozen`
- `latent_only`
- `shuffled_latent`
- `text_only`
- `full`
- `core_only`
- `residual_only`
- `no_controller`

### 8.4 PersonaMem

`latest_stage2_v51_personamem_full589.json`

必须包含：

- `sample_count = 589`
- `no_calibration_accuracy`
- `calibrated_accuracy`
- `option_only_accuracy`
- `random_baseline`
- `gold_used_for_memory_substrate = false`
- `provider_is_auxiliary = true`

### 8.5 Decision

`latest_stage2_v51_real_training_decision.json`

必须包含：

- `result_type`: `positive_gain` / `negative_result` / `blocked`
- `claim_allowed`
- `claim_text`
- `blockers`
- `next_recommendation`

## 9. Autoresearch Execution Plan

### M1: Verifier Repair

目标：

- 写 `verify_stage2_v51_real_training.py`。
- 将 proxy artifacts 识别为 scaffold baseline。
- 确保 proxy 不能满分，不能 stop。

Keep 条件：

- 当前 v5 proxy closeout 在 v5.1 verifier 下必须低分。
- `pretrained_weights_loaded=false` 必须触发 cap。

### M2: Real Backbone Load

目标：

- 真实加载 BGE/E5/Contriever。
- 至少一个 backbone 成功。
- 记录失败原因，不 fallback。

Keep 条件：

- artifact 证明真实 pretrained weights loaded。
- frozen eval 可复现。

### M3: Real Data Builder

目标：

- 建立 stage2 32k training dataloader。
- 建立 PersonaMem gold-free context dataloader。
- 建立 grouped split checker。

Keep 条件：

- train samples >= `10000`。
- val/test 非空。
- gold leakage checker pass。

### M4: Real Training

目标：

- 训练 adapter/projector/controller/reader。
- 至少产出一个真实 checkpoint。
- 优先 GPU 2。

Keep 条件：

- checkpoint exists。
- train log / loss curve exists。
- train_seconds / device / sample counts recorded。

### M5: Held-Out Eval

目标：

- 比较 trained vs frozen。
- 比较 latent-only vs shuffled-latent。
- 比较 full vs text-only。

Keep 条件：

- 至少一个 gold-free held-out metric trained > frozen。
- anti-shortcut ablation 有真实 margin。

### M6: PersonaMem Full589

目标：

- downstream validation。
- 不把 PersonaMem 当 substrate training source。

Keep 条件：

- no-calibration full589 明确超过 random / option-only，才允许 positive claim。
- 若不超过，必须输出 negative_result，并继续分析瓶颈。

### M7: Scientific Closeout

目标：

- 写 decision artifact 和 docs。

Keep 条件：

- 结论与证据一致。
- positive / negative / blocked 三者必须选一个，不能用模糊措辞。

## 10. 自我审视

### Pass 1: 是否解决 v5 的失败？

是。v5 失败点是 verifier 接受 proxy。v5.1 把真实 pretrained loading、真实 checkpoint、真实 sample size 和真实 held-out gain 设成 score cap / stop gate。

### Pass 2: 是否仍可能被 artifact 形状刷满？

风险降低但不为零。解决方式是 artifact 必须引用真实 checkpoint path、train log、model backend、sample counts，并由 verifier 检查文件存在、数值阈值和 `pretrained_weights_loaded=true`。

### Pass 3: 是否可能学 PersonaMem option shortcut？

仍有风险。解决方式是 substrate 禁止使用 gold answer 和 option text；PersonaMem gold 只校准 thin answer head；no-calibration 必须单独过 baseline；option-only baseline 必须报告。

### Pass 4: 是否目标过于激进？

激进但合理。环境具备 CUDA、torch、transformers、sentence-transformers、peft、datasets、accelerate。真实数据也已存在。若网络或 HF 权重不可用，应该记录 blocker，而不是回退到 proxy。

### Pass 5: 如果真实训练还是随机怎么办？

这也是有效结果。v5.1 允许 negative result，但必须诚实写明：当前 architecture / data / objective 不足以让 latent substrate 超过 baseline。然后下一步再选择更强 backbone、更大数据、不同 objective 或架构重设计。

## 11. 启动前必须写给 autoresearch 的一句话

> Do not optimize artifact completeness. Optimize real-training evidence. If pretrained weights are not loaded, if no checkpoint is trained, or if no-calibration PersonaMem remains near random, record the run as blocked or negative instead of claiming success.
