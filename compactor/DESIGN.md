# Compactor：记忆 Embedding 合并与分存 — 设计说明

本文档描述在 `compactor/` 目录内要实现的**记忆压缩（compaction）**能力：在固定维度的 sentence embedding 空间内，对「历史记忆向量」与「新观测向量」做**路由决策**与**语义融合**，并与现有 **gtr-t5-base + vec2text** 管线对齐。实现阶段**以本文档为约定**，代码按后续步骤迭代添加。

**实现进度**：Router / Compactor 已在 **Wikitext-2 + gtr-t5-base** 上完成可复现训练（见 **§1.3**、**§12**）；checkpoint 位于 `compactor/checkpoints/`。待续项主要是 **对话域数据**、**vec2text 一键评估**、**联合微调** 等（见 **§10**）。

**架构原则（推荐）**：将「选哪条记忆 / 是否新开槽位」与「如何把两条向量合成一条」**分开实现**——**Router** 负责前者，**Compactor** 负责后者。二者可独立训练与替换，接口清晰，也便于消融实验。

---

## 1. 背景与约束

### 1.1 现有能力（`vec2text_test/demo.py`）

- **编码器**：`sentence-transformers/gtr-t5-base`，输出固定维度 embedding（与 ST 默认一致，通常为 768 维）。
- **解码（可逆近似）**：`ielabgroup/vec2text_gtr-base-st_inversion` + `ielabgroup/vec2text_gtr-base-st_corrector`，通过 vec2text 将 embedding **还原为文本**。
- **序列长度**：该配套模型存在 **约 32 token** 的上下文上限；长文本需截断或在外层分段，不在本阶段强行突破，但 compactor 设计需**意识到**还原阶段的信息瓶颈。

### 1.2 环境与依赖

- **代码根目录**：`compactor/`（本目录），作为 Python 包与 `src/core_mem` 并列；`pyproject.toml` 中 **setuptools** 已包含 `compactor*`（可与 `core_mem` 一并 `pip install -e .`，开发时亦可在仓库根设置 **`PYTHONPATH=.`**）。
- **Python 环境**：沿用项目当前 venv（例如 `/root/.venvs/CoRe_Mem` 或项目 `.venv` 指向的同一环境），需 **Python 3.10**（与 `pyproject` 一致）。
- **依赖**：训练与推理依赖仓库已有 **`torch`**、**`sentence-transformers`**、`datasets`（见根目录 `requirements.txt`）；`compactor` **未新增**单独依赖包。

### 1.3 训练与数据（当前落地）

本节描述**仓库里已接好、且已用于产出 checkpoint** 的设定；与设计章 §4 / §5 的差异以本节为准。

**编码器**

- `sentence-transformers/gtr-t5-base`（`encoding.load_encoder()`），输出维度通常 **768**，L2 归一化后送入 Router / Compactor。

**语料（弱监督）**

- **来源**：Hugging Face **`wikitext`** / **`wikitext-2-raw-v1`**，**`train`** 切分。
- **行结构**：该集 **一行一条样本**；**正样本（应 merge）** 取 **数据集里相邻两行**（`row_i` 与 `row_{i+1}`），且两行长度均 ≥ `min_line_chars`（默认 **20**）。
- **负样本（new_slot）**：从已收集的「长行池」中 **随机两行** 配对（弱负例；与正例均衡采样进 batch）。
- **说明**：相邻行不保证语义上总应合并（标题/换段噪声）；适合 **冷启动**。对话记忆等目标域需另接语料（§4.5 仍为扩展参考）。

**Router 训练**

- **任务**：**K=1**；**K+1 类交叉熵**（类 0 = 并入唯一槽，类 1 = new_slot）；`RouterV1`：`pair_mlp`（拼接 `[e_new;e_old;e_new⊙e_old;|e_new−e_old|]`）+ `reject_mlp(e_new)`。
- **优化器**：AdamW（脚本默认 `lr=1e-3`）；GPU 下可用 **AMP**。
- **已实现权重示例**（一次完整跑法）：`n_train=8000`，`n_val=1000`，**6 epochs**，batch **64** → 保存为 `compactor/checkpoints/router_wikitext.pt`（验证准确率约 **0.96** 量级，弱标签下偏高属常见）。

**Compactor 训练**

- **教师**：**`e_target = encode(a + " " + b)`**（两句空格拼接再编码），与 §4.3 一致。
- **损失**：batch 内平均 **\(1 - \cos(e_{sum}, e_{target})\)**。
- **已实现权重示例**：同上数据规模下 **12 epochs**，batch **64** → `compactor/checkpoints/compactor_wikitext.pt`（验证 **1−cos** 约 **0.018** 量级）。

**评估与测试**

- 训练日志中的 **val acc / val loss** 为主指标。
- **`tests/test_compactor_trained.py`**：有 checkpoint 时**同时**检查 Router 与 Compactor（结构、随机前向）；**`-m slow`** 会分别在 Wikitext 验证集上跑 **Router 准确率** 与 **Compactor 1−cos**（默认 `pytest` 不跑 slow，见 `pyproject.toml`）。
- **批量、与训练数据协议一致**：`python -m compactor.eval_wikitext_pipeline`（默认 `n_val=1000`，打印 batched Router acc、Compactor val loss，以及「单槽 memory → ingest」下相邻行 merge 率 / 随机行 new_slot 率）。**不要用**该指标解释任意英文短句上的 `e2e_text_pipeline` 行为（后者常为分布外）。

**端到端：文本 → embedding → Router/Compactor → vec2text 还原文本**

- Router/Compactor **不参与** vec2text 反传；若要**主观**看 merge 后单向量是否仍可解码为可读摘要，在仓库根执行：

```bash
export PYTHONPATH=.:src
python -m compactor.e2e_text_pipeline \
  --memory "The user prefers dark mode." \
  --new "They also use vim keybindings."
```

- 默认加载 `compactor/checkpoints/router_wikitext.pt` 与 `compactor/checkpoints/compactor_wikitext.pt`；可用 `--router` / `--compactor` 指定路径。首次运行会从 Hugging Face 拉 **gtr-t5-base**、**ielabgroup/vec2text_*** 等权重；**推荐 CUDA**（CPU 可能很慢或显存/内存吃紧）。
- 仅向量层面的自动化测试见上文；**完整 E2E 含 vec2text** 未纳入默认 `pytest`（依赖大模型与设备），按需本地运行上述命令。

---

## 2. 目标功能（问题定义）

系统维护一条或多条「记忆」的 **embedding 表示**（与 gtr-t5-base 同维度）。当到来一个新的观测并得到其 embedding `e_new` 时，需要两类行为：

| 情况 | 行为 |
|------|------|
| **可合并（merge）** | 将 Router 选中的「旧记忆」`e_old` 与 `e_new` 交给 **Compactor**，融合为**一个** summary embedding `e_sum`，**维度不变**，语义上尽量覆盖二者合适的信息。 |
| **不可合并（split / store）** | Router 判定**不应**并入当前候选记忆；**单独保存** `e_new`（新 slot 或新类别），避免污染原记忆。 |

### 2.1 模块划分：Router 与 Compactor

| 模块 | 职责 | 典型输入 / 输出 |
|------|------|-----------------|
| **Router** | **不融合向量**。当前版本（**Router v1**，见 §2.2）：在记忆库中 **至多并入一条**——要么并入 **与 `e_new` 最相关的那一个** slot，要么 **新开一条**；不做多槽同时更新。 | 输入：`e_new`，候选 `e_old^{(1)},…,e_old^{(K)}`。输出：**`new_slot`** 或 **`merge_into_k`**（`k` 为唯一胜者）。 |
| **Compactor** | **只做融合**。给定已配对好的 `(e_old, e_new)`，输出 `e_sum`（同维、建议 L2 normalize）。**不负责**判断该不该合并；仅在 Router 决定 `merge_into_k` 且已取出对应 `e_old` 后调用。 | 输入：`(e_old, e_new)`。输出：`e_sum`。 |

**数据流（推理）**：`e_new` → **Router** → 若 `new_slot`：持久化 `e_new`；若 `merge_into_k`：取 `e_old = slot[k]` → **Compactor**(`e_old`, `e_new`) → 写回 `slot[k] ← e_sum`。

**为何拆开**：路由（相似度结构、开新槽）与融合（单向量语义合成）优化目标不同；分开后可用**不同数据**训练（例如路由用弱标注的「是否同一主题」，融合用「合并句的教师 embedding」），也避免一个头同时学两件事导致梯度纠缠。

### 2.2 Router v1（当前范围）：单胜者 + 新开槽

推理规则（与实现顺序一致）：

1. 记忆库中有 **K 个候选** embedding：`e_old^{(1)},…,e_old^{(K)}`（K 可随时间增长；实现时可设上限或截断）。
2. 对每个候选算 **相关度分数** `s_k`（先可用余弦相似度 + 小 MLP，再迭代）。
3. 令 `k^* = \operatorname{argmax}_k s_k`（**只选最相关的一条**）。
4. **新开槽判定**：若 `s_{k^*} < τ`（阈值，验证集上标定），或单独学一个「无匹配」logit 且最大概率落在 **new_slot** 类 → 输出 **`new_slot`**，只写入 `e_new`。
5. 否则 → 输出 **`merge_into_{k^*}`**，再交给 Compactor 与 `e_old^{(k^*)}` 融合。

**特例 `K=1`**：只有一个候选时，Router 退化为 **merge vs new_slot 二分类**（「是否与当前唯一记忆合并」），无需 argmax，逻辑最简单，适合先写通数据与训练脚本。

**明确不做（留待后期）**：同时更新多个 slot、对多个记忆做软混合、或 Router 与 Compactor 联合输出一条以外的结构。

---

## 3. 方案设想

### 3.1 Router v1：打分 → argmax → 阈值

- **打分**：`s_k = f_\theta(e_new, e_old^{(k)})`，`f` 可为 `cos`、双线性、`MLP([e_new; e_old^{(k)}; e_new \odot e_old^{(k)}])` 等。
- **决策**：`k^* = argmax_k s_k`；若 `s_{k^*} < τ` → **new_slot**，否则 **merge_into k\***。也可把 **new_slot** 设为第 `K+1` 个 logits，与 `K` 个槽一并 softmax（见 §5.1），省掉单独调 `τ`。
- **类别不均衡 / 易全预测 new_slot**：focal loss、对 merge 类加权、或提高「应合并」样本比例。

**「相差很大就单独存」**：对应 **new_slot** 分支，Compactor **不参与**。

### 3.2 Compactor：仅做语义融合（Gated / MLP）

| 方案 | 思路 | 适用 |
|------|------|------|
| **Gated network（推荐优先尝试）** | 例如 `g = σ(W[e_old; e_new])`，`e_sum = normalize(g ⊙ e_old + (1-g) ⊙ e_new)` 或带残差与线性投回 `d` 维；门控表示「保留旧 vs 注入新」的比例。 | 成对融合参数少、训练稳。 |
| **MLP** | `[e_old; e_new]` 与交互特征拼接，隐藏层后投回 `d` 维，再 L2 normalize。 | 表达力强，可作对照基线。 |

**预设想**：先实现 **门控融合**，必要时加深 MLP 或加 LayerNorm / dropout。

### 3.3 与 vec2text 的衔接（训练与推理）

- **训练时**不必每次反传 vec2text（成本高）；可用 **代理目标**（见下节）预训练 compactor，再在子集上用 **decode 质量** 做抽检或微调。
- **推理时**：merge 路径上得到 `e_sum` 后，仍可走现有 `invert_embeddings` 得到 summary 文本，用于可读性或下游任务。

---

## 4. 数据设计：Router 与 Compactor 各要什么

二者监督信号不同：**Router** 学「该不该并、并到哪」；**Compactor** 学「并完之后向量长什么样」。数据可以来自同一批原始文本，经 **gtr-t5-base** 离线编码后存盘，训练时只读向量，加快迭代。

### 4.1 共同约定：embedding 与归一化

- 所有 `e` 均用 **同一套** `SentenceTransformer("sentence-transformers/gtr-t5-base")` 编码得到，**dtype / 设备**与线上一致。
- 若 ST 输出已 L2 归一化，训练时保持；若否，建议在进入 Router / Compactor **前统一 `F.normalize(e, dim=-1)`**，使余弦相似度与损失公式一致。

### 4.2 Router v1 的数据形态与标签

**目标**：与推理一致——每个样本要么标 **应并入第 `k` 槽**，要么标 **new_slot**（不与任一候选合并）。

**通用形式（K 个候选，推荐与线上一致的固定 K 训练）**

每条样本：

- **query** → `e_new`；
- **K 个候选** → `e_old^{(1)},…,e_old^{(K)}`（不足可 padding，mask 掉无效位）；
- **标签 `t ∈ {0,…,K-1, K}`**：`t=k` 表示应 **merge_into k**；**`t=K` 表示 new_slot**（不与任一候选合并）。

构造示例：

- **正槽位**：同一话题簇 / 同文档中，把「应归属」的那条放进候选 `k`，其余候选用 in-batch 其他话题、随机句、或硬负例（相似但不属同一记忆）填满。
- **new_slot**：query 与所有候选都不该绑定时标 `t=K`（跨主题、新实体、刻意采的「都不相关」）。

**硬负例（推荐）**：若干候选与 `e_new` 余弦相似度中高但标签仍应是 **new_slot** 或 **错误槽**，迫使模型学 **argmax + 拒绝**。

**特例 K=1（先跑通）**

每条样本只有 **一个** `e_old`，标签退化为 **二分类**：`y=1` → 并入该条；`y=0` → new_slot。数据即 §4.2 旧版「一对一句」的 merge / split 表（同文档相邻句 vs 跨主题随机对等）。与 **Router v1** 的 `K=1` 推理完全一致。

**采样比例**：负例 / new_slot 易过多，需 **下采样** 或 **focal / 加权 BCE（K=1）**、**加权 CE（K>1）**。

### 4.3 Compactor 的数据形态与教师信号

Compactor **只使用「真该 merge」的对**（可由人工规则构造，**不依赖** Router 预测）。

对每条 merge 正例，需要 `(e_old, e_new)` 与 **融合目标向量** `e_target`（教师）：

| 教师类型 | 做法 | 备注 |
|----------|------|------|
| **合并文本编码（主推荐）** | 把两句合成一句或一小段「摘要句」（规则拼接、或 LLM 生成一句 summary），再 `encode → e_target` | 与「单向量摘要」语义对齐，便于后面 vec2text |
| **池化近似（弱）** | `e_target = normalize(α e_old + (1-α) e_new)` | 无文本标签时的下界基线，易训但上限低 |
| **仅第一句 / 主句** | `e_target = encode(concat_text)` 或选较长句 | 实现快，适合先跑通 |

**不要**用 split 负例训练 Compactor（除非做显式正则，见下节），否则会把「不该并」也学成一种融合，与推理管线冲突。

### 4.4 与 Router 数据的复用关系

- **简单流水线**：从同一语料先筛出 merge 对 → 一部分只标 Router 的 merge/split，一部分再生成 `e_target` 训 Compactor；**split 对仅进 Router**。
- **避免泄漏**：验证集 / 测试集按「对话或文档 id」划分，不要把同一文档的句对同时出现在 train 和 val。

### 4.5 训练数据来源（扩展候选）

**当前默认已接入**：**Wikitext-2-raw-v1 相邻行 + 随机行对**（实现见 `weak_supervision.py`，概述见 **§1.3**）。

下列为**尚未默认接入**、可作换域或增强的弱监督来源（需自行写数据构造脚本）：

- **百科 / 网页长文档**：同段相邻句、跨段硬负例；CC-News、C4 等。
- **对话**：同 session 相邻轮为正，跨 session 为负（DailyDialog、MultiWOZ 等）。
- **释义 / NLI**：Quora、PAWS、entailment 对等（需与 gtr 语种一致）。

**Compactor 教师**：落地实现采用 **拼接句 encode**；**LLM 摘要句**、**插值 `αe_old+(1-α)e_new`** 仅作 `train_* --synthetic` 或实验对照。

**体量**：当前示例为 **8k/1k** train/val 对级；扩到十万级对通常需 **向量缓存**（`.npz` / sqlite）以免重复编码。

---

## 5. 损失函数、训练阶段与评估

### 5.1 Router v1：损失形式

**K=1（二分类）**：logits `z`，标签 `y∈{0,1}`（1=并入唯一候选，0=new_slot）。

- \( \mathcal{L}_{\mathrm{router}} = \mathrm{BCEWithLogits}(z, y) \)；不均衡时用 **focal** 或 **weighted BCE**。
- 若推理用 **阈值** 而非可学习「new_slot」类：在验证集上扫 `τ`，使 F1 / 业务指标最优。

**K≥1（与 v1 推理一致，推荐一种实现）**

- **多类交叉熵（K+1 类）**：类别 `0…K-1` 为「并入对应槽」，类别 **`K` = new_slot**。对每条样本算 logits `g_0,…,g_K`，\( \mathcal{L} = \mathrm{CE}(g, t) \)。padding 候选不参与 softmax（mask）。
- 该形式与「**argmax 再在最高分与 new_slot 间抉择**」可等价实现：例如令 `g_K` 为「无匹配」logit，`g_k` 为与各候选相关度；训练时直接优化分类准确率即可，推理时对 `softmax(g)` 取 argmax，若为 `K` 则 new_slot。

**两阶段分解（可选，调参友好）**

- 先 **BCE**：是否应并入「某一候选」vs new_slot；再 **CE**：在应并入子集上对 K 个槽分类。两项加权和。

**辅助损失（可选）**

- **边际损失**：对正槽 `s^+` 与负槽 `s^-`，要求 \( s^+ > s^- + m \)（hinge），增强分离度。
- **对比学习**：InfoNCE 式，使 `e_new` 与正确 `e_old` 拉近、与其他候选推远（需 batch 内多负样本）。

### 5.2 Compactor：损失形式

设 Compactor 输出 `e_sum`（建议与教师同尺度，均 L2 normalize）。

- **主损失（余弦回归）**：  
  \( \mathcal{L}_{\mathrm{compact}} = 1 - \cos(e_{\mathrm{sum}}, e_{\mathrm{target}}) \)  
  或等价地 **负余弦相似度**（batch 内平均）。

- **MSE 在球面上（备选）**：对归一化向量可用 \( \| e_{\mathrm{sum}} - e_{\mathrm{target}} \|_2^2 \)，与 cosine 相关但数值尺度不同，二选一为主即可。

- **不要**在 split 对上算主损失；若担心 merge 输出「漂移」，可加轻量 **正则**：`λ ||e_sum - e_old||^2` 或鼓励与 `e_new` 保持合理混合（小权重，避免压过主项）。

### 5.3 训练阶段怎么排（推荐默认）

| 阶段 | 内容 | 目的 |
|------|------|------|
| **A. Router 预训练** | 仅 `L_router`，数据见 §4.2 | 先稳定「像不像 / 该不该并 / 选哪槽」 |
| **B. Compactor 预训练** | 仅 `L_compact`，**oracle merge 对** + 教师 `e_target`（§4.3） | 融合头不受 Router 噪声影响 |
| **C. 联合微调（可选）** | 小学习率同时训两者；batch 内先 **冻结 Router 只训 Compactor** 若干 step 再解冻，或交替优化 | 缩小训练–推理分布差（推理时 bad route 会喂错对） |

**联合训练注意**：若 Router 输出离散槽位，端到端反传需 **straight-through / Gumbel-Softmax** 或 **REINFORCE**，工程成本高；更实用的 **伪联合**：用当前 Router 在训练集上 **hard 或 soft 选样**，再更新 Compactor（类似蒸馏迭代）。

**推理时一致性**：最终评估应以 **Router 选出的 `(e_old, e_new)`** 喂 Compactor，而不是全程 oracle 对；否则 Compactor 分数会虚高。

### 5.4 离线指标（与损失对应）

- **Router v1**：整体 **(K+1)-way 准确率**；拆分为 **new_slot 召回/精确率**、**在应 merge 子集上的槽位准确率**；K=1 时用 merge vs new_slot 的 F1、ROC-AUC。
- **Compactor**：`mean cos(e_sum, e_target)` 或 `L_compact`；分 strata 看（仅 oracle 对 / 经 Router 后的对）。
- **端到端（后期）**：固定 Router 策略下，对 merge 路径做 vec2text 可读性抽检或用户任务指标。

---

## 6. 存储与运行流程（逻辑）

1. **读**：记忆库中多个 slot 的 `e_old^{(k)}`（及元数据）。
2. **写**：输入 `e_new` → **Router** →  
   - **merge_into_k**：取 `e_old = slot[k]` → **Compactor**(`e_old`, `e_new`) → `slot[k] ← e_sum`；  
   - **new_slot**：追加新 slot 存 `e_new`，不调用 Compactor。
3. **维度**：全程 `d = dim(gtr-t5-base)`；Compactor 输出维与之一致。

**冷启动（K=0）**：记忆库为空时**不进行 Router 前向**，直接 **new_slot** 写入首条 embedding（实现见 `inference.ingest_embedding`）。

持久化格式（JSON / numpy / sqlite）在实现阶段再定，本文档只约束语义；当前代码仅提供 **内存中** `MemoryBank`。

---

## 7. 分阶段实施（里程碑）

| 里程碑 | 状态 |
|--------|------|
| Router v1 + `MemoryCompactor` + `MemoryBank` + `ingest_embedding` / `route_only` | **已完成** |
| 合成数据 + **Wikitext 弱监督** + **`train_* --wikitext`** + checkpoint | **已完成**（§1.3） |
| 示例权重 `compactor/checkpoints/*_wikitext.pt` | **已训练产出**（可提交 Git LFS / HF，勿必交仓库） |
| pytest：`test_compactor_router` + `test_compactor_trained`（含可选 slow） | **已完成** |
| 对话域 / 业务语料、预计算向量库 | **未做** |
| vec2text 一键解码评估脚本、联合微调（§5.3 C） | **未做** |

---

## 8. 风险与待决问题

- **32 token 上限**：合并多轮后 decode 可能丢失细节；产品层是否限制单次写入长度或分段记忆，需与业务对齐。
- **域偏移**：Router 阈值与 Compactor 融合质量可能同时漂移；上线前应用目标域做校准或小样本微调。
- **候选数 K 较大**：Router v1 需对 K 个候选各算一次分数，注意延迟与 batch 实现；二期可做候选剪枝（先粗排再精排）或更小模型。

---

## 9. 文档维护

重大行为或训练约定变更时，请同步更新 **§1.3**、**§10**、**§12**，并在 Git 提交说明中写清。

---

## 10. TODO（后续工作）

**已完成内容**（无需再作为待办跟踪）：可导入包与模块（§12.1）；`encoding` / `RouterV1` / `MemoryCompactor` / `MemoryBank` / `inference`；**`--synthetic`** 与 **`--wikitext`** 训练脚本；**`weak_supervision.py`**（Wikitext 相邻行）；**checkpoint 保存**；**`tests/test_compactor_router.py`** 与 **`test_compactor_trained.py`**；**`pyproject.toml`** 中 `pytest` 的 `slow` 标记与 `pythonpath`。

**待办与可选增强**

| 优先级 | 项 |
|--------|-----|
| 中 | **对话 / 业务域**语料：按 §4.4 做 **文档或 session 级划分**，替换或混合 Wikitext。 |
| 中 | **预计算向量**（`.npz` / sqlite）：大规模训练时避免重复 `encode`。 |
| 中 | **Router**：**focal / 加权 CE**、验证集 **按类 F1 / new_slot 召回** 报表；推理侧可选 **temperature / 校准**（当前为 softmax+argmax，无单独阈值 τ）。 |
| 中 | **评估**：Compactor 在 **经 Router 选中的 (e_old, e_new)** 上算损失（§5.3），避免仅 oracle 虚高。 |
| 低 | **联合微调**（§5.3 阶段 C）或 **伪联合**。 |
| 低 | **vec2text**：封装「读入 `e_sum` → `invert_embeddings` → 文本」的**小脚本**，便于合并质量抽检。 |
| 低 | 训练脚本 **`--seed`** 与 dtype 说明（当前默认 **float32**）。 |

**不计划在本仓库单独实现**（除非产品要求）：为 `MemoryCompactor.forward` 再包一层无状态的 `compact()` 薄 API（调用方可直接用类实例）。

---

## 11. 设计边界与待细化项

下列内容**不阻碍**按 §7 / §10 开工，但在落地或上线前应逐步补齐或写清约定。

| 项 | 说明 |
|----|------|
| **空记忆库（K=0）** | 首条观测无候选可并，应 **直接 new_slot**；**§6** 与 `ingest_embedding` 已实现该分支。 |
| **训练 K vs 线上 K** | 训练常用 **固定 K**；线上槽位数变化时，需约定 **候选截断 / 最大 K / 粗排预筛**（§8 已提延迟，实现时要定策略）。 |
| **多轮 merge 的分布偏移** | Compactor 训练多为「原始句对」的 `(e_old, e_new)`；线上 `e_old` 可能是 **多次融合后的 summary 向量**，分布可能不同；缓解：**回放合成**（用 Compactor 输出再喂 Router/Compactor 混合训练）、或周期性 **微调**。 |
| **槽位顺序与元数据** | Router v1 只输出索引 `k`；**slot 的语义顺序、时间戳、是否过期** 属产品/存储层，本文档不展开；若需「遗忘」或 LRU，在记忆 API 层扩展。 |
| **模型版本** | 更换 **gtr-t5-base** 或 vec2text 权重时，旧 embedding **与新版几何可能不一致**；需 **重编码** 或 **版本字段** 隔离不同批次向量。 |
| **错误级联** | Router 误判 merge 时，Compactor 仍会融合；评估与线上需接受该风险，或通过 **置信度 + 人工审核队列**（后期）缓解。 |
| **监控与回滚** | 上线后可记 **merge 率、new_slot 率、cos(e_sum,e_target)**、decode 抽检；异常时回退到 **仅 new_slot** 或旧阈值 `τ`。 |

---

## 12. 当前实现概要（代码对照）

本节记录**已写入仓库**的实现，便于与设计对照；文件名均相对于 `CoRe_Mem/compactor/`（及仓库根 `tests/`）。

### 12.1 模块与文件

| 文件 | 说明 |
|------|------|
| `config.py` | `EMBED_MODEL_NAME`（gtr-t5-base）、全局 `embed_dim`（由 encoder 首次加载时 `set_embed_dim`） |
| `encoding.py` | 懒加载 `SentenceTransformer`，`encode_texts` / `encode_texts_numpy` |
| `types.py` | `RouteDecision`、`IngestResult` |
| `router.py` | **`RouterV1`**：对每候选拼接特征 `[e_new; e_old; e_new⊙e_old; \|e_new−e_old\|]` 经共享 MLP 得 **K** 个 logit；**`reject_mlp(e_new)`** 得第 **K+1** 个 logit（**new_slot**）；输出 `(B, K+1)`；**`apply_cand_mask`** 屏蔽 padding；**`predict_decision`** 对全类 **softmax + argmax**（与设计 §5.1 的 K+1 类 CE 一致） |
| `fusion.py` | **`MemoryCompactor`**：门控 `σ(W[e_old;e_new])`，逐维混合后 **L2 normalize** |
| `memory_bank.py` | **`MemoryBank`**：`append` / `replace` / `get` / **`stack_for_router`**（形状 `(1,K,d)` + mask） |
| `inference.py` | **`route_only`**、**`ingest_embedding`**（空库直接新槽；merge 路径调用 Compactor 并写回 slot） |
| `datasets.py` | **`RouterTensorDataset`**、**`CompactorTensorDataset`**；**`make_synthetic_*`**（随机向量 / 弱教师） |
| `weak_supervision.py` | **Wikitext-2-raw-v1**：相邻 **数据行** 为正样本对；负样本为随机两行；Compactor 教师为 **`encode(a + " " + b)`** |
| `train_router.py` | **`--synthetic`** 或 **`--wikitext`**（K=1，CE）；`--min-line-chars`；`--save` |
| `train_compactor.py` | **`--synthetic`** 或 **`--wikitext`**（`1−cos`）；`--save` |
| `checkpoints/*.pt` | 示例训练产出（如 `router_wikitext.pt`、`compactor_wikitext.pt`）；体积约数 MB，建议 **`.gitignore`** 或 **Git LFS / HF Hub** |
| `__init__.py` | 对外导出 Router / Compactor / Bank / 推理与编码辅助函数 |

仓库根 **`tests/test_compactor_router.py`**：张量形状、输出单位范数、**空库 ingest**、mask 行为（**不依赖**下载 HF，便于 CI）。

**`tests/test_compactor_trained.py`**（可选）：若存在 `compactor/checkpoints/*_wikitext.pt`，加载权重做前向；**`@pytest.mark.slow`** 在小型 Wikitext 验证集上检查 Router **acc** / Compactor **val loss** 是否高于宽松阈值。默认 **`pytest` 不跑 slow**（见 `pyproject.toml`）；完整检查：`pytest tests/test_compactor_trained.py -m slow`。

### 12.2 实现与设计说明

- **决策形式**：**K+1 类 softmax + argmax**（含 new_slot 类），**未**实现单独可调阈值 **τ**（见 §10）。
- **合成 vs 真实维**：`--synthetic` 用 **64 维**随机向量；**`--wikitext`** 与线上一致为 **gtr-t5-base** 维（通常 **768**）。
- **vec2text**：仅推理侧可读性；训练不反传 vec2text（§1.3）。

### 12.3 运行命令（开发）

在仓库根 **`CoRe_Mem/`**（Python 3.10，`torch` + `sentence-transformers` + `datasets`）：

```bash
export PYTHONPATH=.:src
python -m pytest tests/test_compactor_router.py tests/test_compactor_trained.py -q
# pytest tests/test_compactor_trained.py -m slow -q   # 需已有 checkpoints，较慢

python -m compactor.train_router --wikitext --n-train 8000 --n-val 1000 --epochs 6 --batch 64 --device cuda \
  --save compactor/checkpoints/router_wikitext.pt
python -m compactor.train_compactor --wikitext --n-train 8000 --n-val 1000 --epochs 12 --batch 64 --device cuda \
  --save compactor/checkpoints/compactor_wikitext.pt
```

---

*与 `vec2text_test/demo.py` 中 gtr-t5-base + vec2text 设定一致；**§1.3** 为当前训练约定，**§12** 为代码与命令快照。*
