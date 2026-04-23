# V5.2 Plan: Full Learned Latent Memory System

## 0. 为什么 v5.1 不够

`v5.1` 证明了 real pretrained + real checkpoint + real held-out eval 可以跑通，但它仍只是 bootstrap：

- 只加载并训练了 `BAAI/bge-base-en-v1.5` 一条 backbone。
- backbone frozen，只训练了 query / slot projection、temperature、bank controller。
- 训练目标主要是 `retrieval_alignment`，不是完整 multi-task latent memory。
- `composition_to_belief`、`slot_autoencoding`、`lifecycle_prediction` 没有一起训练。
- PersonaMem no-calibration `184/589` 只超过 random / option-only，仍低于 text-only `214/589`。

所以 v5.2 不再接受“小步真实训练”作为 closeout。目标是完整 learned latent system。

## 1. v5.2 最高目标

训练并验证一个真正的 full learned latent memory system：

- real multi-backbone comparison
- shared pretrained backbone + task-specific adapters 或等价 trainable module
- multi-task latent objectives
- trainable encoder/projection + latent reader/resampler + write controller + belief decoder/graph
- PersonaMem no-calibration 超过 text-only baseline
- full system 超过 retrieval-only system

## 2. 硬门槛

v5.2 verifier 必须阻止以下弱完成态：

- 只训练一个 BGE retriever
- 只训练 retrieval_alignment
- 只超过 random / option-only
- artifact completeness 但没有真实 checkpoint
- 只靠 provider / prompt / option trick
- PersonaMem gold answer 进入 memory substrate

成功必须同时满足：

- 至少 `2` 个真实 pretrained backbones 被加载并比较；目标是 `BGE / E5 / Contriever` 三者完整比较。
- 真实 checkpoint 存在。
- 训练样本不少于 `20000`，并且至少覆盖 `retrieval_alignment`、`slot_autoencoding`、`composition_to_belief`、`lifecycle_prediction` 四个任务。
- trainable modules 覆盖 encoder adapter/projection、latent reader/resampler、write controller、belief decoder/graph。
- multi-task latent system 优于 retrieval-only latent baseline。
- trained system 优于 frozen baseline。
- latent-only 优于 shuffled-latent。
- full system 优于 text-only。
- no-controller、no-belief、no-core-residual 等 ablation 明确下降。
- PersonaMem full589 no-calibration 必须超过 text-only `214/589`。

## 3. Required Artifacts

- `latest_stage2_v52_backbone_compare.json`
- `latest_stage2_v52_multitask_training.json`
- `latest_stage2_v52_full_latent_eval.json`
- `latest_stage2_v52_personamem_full589.json`
- `latest_stage2_v52_ablation_summary.json`
- `latest_stage2_v52_decision.json`

## 4. Autoresearch 方向

M1: 强 verifier 与 tests。

M2: 真实 multi-backbone comparison，至少 `BGE + E5`，目标加上 `Contriever`。

M3: multi-task dataloader，统一四个 stage2 tasks，并保证 PersonaMem gold-free。

M4: 训练 shared backbone/task-adapter 或等价模块化 latent system。

M5: 训练 belief decoder / graph head，让 latent 直接服务 belief，而不是只做 retrieval。

M6: full system eval：trained vs frozen、multi-task vs retrieval-only、latent vs shuffled、full vs text-only。

M7: PersonaMem full589 no-calibration，目标超过 text-only `214/589`。

M8: closeout。如果没有超过 text-only，记录 `negative_result`，不能声称强 latent system。

## 5. 给 autoresearch 的约束句

> Do not stop at real-training bootstrap. The retained result must be a full learned latent memory system: multi-backbone, multi-task, trainable reader/controller/belief, and PersonaMem no-calibration above text-only. If this cannot be achieved, record a negative result instead of claiming success.
