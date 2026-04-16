# Compactor：记忆 Embedding 合并与分存 — 设计说明

本文档描述在 `compactor/` 目录内要实现的**记忆压缩（compaction）**能力：在固定维度的 sentence embedding 空间内，对「历史记忆向量」与「新观测向量」做**路由决策**与**语义融合**，并与现有 **gtr-t5-base + vec2text** 管线对齐。实现阶段**以本文档为约定**，代码按后续步骤迭代添加。

**实现进度**：Router v1、门控 Compactor、内存 `MemoryBank`、推理入口 `ingest_embedding` / `route_only`、合成与 **Wikitext-2 弱监督 + gtr-t5-base** 训练路径（见 **§12**）；**vec2text 解码评估脚本**、联合微调、更大/目标域语料等待续。

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
- **依赖**：训练与推理依赖仓库已有 **`torch`**、**`sentence-transformers`**（见根目录 `requirements.txt`）；`compactor` **未新增**单独依赖包。

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

### 4.5 训练数据来源（可选用）

不要求一开始就有「真实记忆库」标注；用 **弱监督** 从公开文本构造 `(句/段 A, 句/段 B)` 再经 **gtr-t5-base** 编码即可。**目标域是对话记忆时**，应用同分布数据 **微调 Router**（少量标注或继续弱监督）。

**Router（merge / new_slot 或选槽）**

| 类型 | 正例（应 merge 或应并入某槽） | 负例（应 new_slot） |
|------|------------------------------|---------------------|
| **百科 / 新闻 / 网页** | 同一 **section / 段落 / 文档** 内 **相邻句** 或 **相邻短句窗口**（Wikipedia dump、CC-News、C4/OpenWebText 子集等） | 从 **不同文章 / 不同随机文档** 各抽一句配对 |
| **对话** | 同一 **session** 内 **相邻轮次**（如 DailyDialog、Persona-Chat、MultiWOZ 等；可过滤过短轮次） | 从 **不同 session** 各抽一句配对 |
| **语义重复 / 复述**（加强「像且该并」） | Quora Question Pairs **重复问**、PAWS **释义对**、NLI 中 **entailment** 对（句对同主题） | PAWS **非释义**、**contradiction**、或随机错配 |

**硬负例**：同一文档内但 **跨段落**、或 **高余弦相似但标注为不同实体** 的对（需规则或辅助模型），用于 §4.2 所述「中高相似仍 new_slot」。

**Compactor**

- 与 Router 的 **merge 正例** 共用同一批句对即可。
- **教师 `e_target`**（推荐顺序）：① 将两句 **用空格或句号拼成一句** 再 `encode`；② 同一窗口内 **人工摘要句**（若有）；③ 弱基线 `normalize(αe_old+(1-α)e_new)`。

**语种**：与线上一致即可（英文模型则用英文语料）；若记忆为中文，应用 **中文句向量模型** 或 **中文语料** 做域适配（否则仅作冷启动）。

**体量**：先 **十万～百万对量级** 即可跑通；再按验证集过拟合/欠拟合增减。

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

## 7. 分阶段实施（建议顺序）

| 步骤 | 内容 | 状态 |
|------|------|------|
| 1 | 文档与接口约定 | **已完成**（本文件 + §12 代码对照） |
| 2 | 数据与张量管线 | **部分完成**：合成数据（`datasets.py`）；**Wikitext-2 相邻行弱监督**（`weak_supervision.py`）+ gtr 编码已接；对话/业务域语料仍待接 |
| 3 | Router v1 + 门控 Compactor + `MemoryBank` + 推理 API | **已完成**（见 §12） |
| 4 | 训练脚本 | **部分完成**：`--synthetic` 冒烟；**`--wikitext`** 为真实 **gtr-t5-base** 维训练；`--save` 存 checkpoint |
| 5 | 推理与 `demo.py` 对齐 | **部分完成**：embedding 与 `encoding.py` 对齐 gtr-t5-base；**vec2text 解码抽检脚本** 未写 |
| 6 | 联合微调、增广、端到端指标 | **未做** |

---

## 8. 风险与待决问题

- **32 token 上限**：合并多轮后 decode 可能丢失细节；产品层是否限制单次写入长度或分段记忆，需与业务对齐。
- **域偏移**：Router 阈值与 Compactor 融合质量可能同时漂移；上线前应用目标域做校准或小样本微调。
- **候选数 K 较大**：Router v1 需对 K 个候选各算一次分数，注意延迟与 batch 实现；二期可做候选剪枝（先粗排再精排）或更小模型。

---

## 9. 文档修订

实现过程中若对方案有重大调整，应更新「模块划分」「方案设想」与「分阶段实施」，并在 **Changelog** 中记一笔。

### Changelog

- **初版**：合并判别 + 融合合述。
- **修订**：明确 **Router（选槽 / merge vs split）** 与 **Compactor（成对融合）** 分模块实现；更新数据流、训练与实施步骤。
- **修订**：扩充 **§4 数据设计**（归一化、单槽/多槽样本构造、教师 `e_target`、硬负例、数据划分）与 **§5 损失与训练阶段**（BCE/focal/多类 CE、余弦损失、A→B→C 分阶段与联合微调注意、评估分层）。
- **修订**：锁定 **Router v1**：候选中 **只并入最相关的一条** 或 **new_slot**；§2.2 / §3.1 / §4.2 / §5.1 与之对齐；**K=1** 为二分类特例。
- **修订**：增加 **§10 TODO 清单**。
- **修订**：增加 **§4.5 训练数据来源**（弱监督语料与正负例构造）。
- **修订**：增加 **§11 设计边界与待细化项**（K=0、训练/线上 K、多轮融合分布、模型版本、错误级联等）。
- **实现**：`compactor/` 初版代码（`RouterV1`、`MemoryCompactor`、`MemoryBank`、`ingest_embedding`、合成数据训练脚本、`tests/test_compactor_router.py`）；真实语料编码管线见训练脚本 TODO。
- **实现**：更新 **§1.2**（包路径与依赖）、**§6**（K=0 冷启动）、**§7** 为状态表、**§10** TODO 勾选、新增 **§12 当前实现概要**；页脚说明与 §12 同步。
- **实现**：**`weak_supervision.py`**（Wikitext 相邻行 + gtr 编码）、训练脚本 **`--wikitext`**；修正 Wikitext「单行一条」应用**连续行**配对；DESIGN §7/§10/§12 同步。

---

## 10. TODO 清单

以下为实现时的检查项；**已完成**标为 `[x]`，仍待办为 `[ ]`。

### 10.1 工程与目录

- [x] 在 `compactor/` 下建立可导入包（扁平模块：`router.py`、`fusion.py` 等，见 §12）。
- [ ] 训练脚本增加 **可配置随机种子**、与线上一致的 **dtype** 文档说明（当前依赖 PyTorch 默认与 `float32` 张量）。
- [x] 从项目根可运行 **`python -m compactor.train_router`** / **`train_compactor`**。

### 10.2 数据管线

- [x] 封装 **统一 embed**（`encoding.py`，`gtr-t5-base` + L2 normalize 选项）。
- [x] **Router 弱监督数据**：Wikitext 相邻行 + 随机负例（`weak_supervision.py`）；**按文档 id 的精细划分**仍待业务语料（§4.4）。
- [x] **合成 Router 数据**：`make_synthetic_router_data`（固定 K、mask 支持）。
- [x] **Compactor Wikitext 路径**：**`e_target` = encode(a + " " + b)**（§4.3）；**对话域 / 人工摘要** 仍可选增强。
- [x] **合成 Compactor 数据**：弱教师 `normalize(α e_old + (1−α) e_new)`。
- [ ] （可选）**预计算 .npz / sqlite** 向量缓存。

### 10.3 Router v1

- [x] 打分 **MLP** 与 **reject** 头；**K+1** logits；**K=1** 时等价二分类（两个 logit）。
- [x] 训练：**(K+1) 类 CE**；推理：**argmax**；**padding mask**（`apply_cand_mask`）。
- [ ] **focal / 加权 CE**、验证集 **F1 / new_slot 召回** 报表化。
- [x] **`--save` checkpoint**（`train_router.py`）。

### 10.4 Compactor

- [x] **门控** 融合 + **L2 normalize**（`MemoryCompactor`）。
- [x] 损失：**`1 - cos(e_sum, e_target)`**；合成数据上训练循环。
- [x] **`--save` checkpoint**（`train_compactor.py`）。

### 10.5 训练流程

- [x] 可分别只训 Router / 只训 Compactor（两脚本 + `--synthetic`）。
- [ ] **阶段 C**：联合或伪联合（§5.3）。
- [ ] Compactor 验证集上 **经 Router 选对** 的 stratified 评估（§5.3）。

### 10.6 推理与集成

- [x] **`route_only`**、**`ingest_embedding`** + **`MemoryBank`**（§6，含 **K=0**）。
- [ ] 独立 **`compact(e_old, e_new)`** 单函数导出（当前通过 **`MemoryCompactor.forward`** 直接调用即可）。
- [ ] 对接 **`vec2text_test/demo.py`** 的解码抽检脚本。

### 10.7 依赖与文档

- [x] **无新增 pip 依赖**（沿用仓库 `requirements.txt`）；本文件 **§12** 记录实现状态。
- [x] **Changelog** 随本版本更新。

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
| `__init__.py` | 对外导出 Router / Compactor / Bank / 推理与编码辅助函数 |

仓库根 **`tests/test_compactor_router.py`**：张量形状、输出单位范数、**空库 ingest**、mask 行为（**不依赖**下载 HF，便于 CI）。

### 12.2 与设计的差异与未实现项

- **Router**：实现为 **K+1 类 softmax + argmax**，未单独提供可调的标量阈值 **τ**（若需可与 `logits[:,K]` 对比手工后处理，或后续加 **temperature / 校准**）。
- **训练**：**`--synthetic`** 仍可用 **64 维**随机向量冒烟；**`--wikitext`** 使用 **`load_encoder()`** 后真实 **gtr-t5-base 维度（通常 768）** 与 Wikitext 弱标签。
- **弱监督语义**：相邻 Wikitext 行**并不保证**总是应 merge（标题/换段噪声）；适合 **冷启动**，换域建议用 §4.5 的对话数据或人工校准。
- **Compactor 教师**：**`--wikitext`** 下为 **`encode(concat(a,b))`**（§4.3）；**`--synthetic`** 仍为插值弱教师。
- **vec2text**：训练**不依赖** vec2text；解码质量抽检可复用 `vec2text_test/demo.py`，**尚未**封装一键评估脚本。

### 12.3 运行命令（开发）

在仓库根 **`CoRe_Mem/`** 执行（需已安装 `torch`、**`sentence-transformers`**，Python 3.10）：

```bash
export PYTHONPATH=.
python -m pytest tests/test_compactor_router.py -q
# 冒烟（随机向量，无 HF）
python -m compactor.train_router --synthetic --epochs 2 --save /tmp/router.pt
python -m compactor.train_compactor --synthetic --epochs 2 --save /tmp/compactor.pt
# 真实 gtr + Wikitext（首次会下载 gtr-t5-base 与数据集）
python -m compactor.train_router --wikitext --n-train 2000 --n-val 400 --epochs 3 --save router_wiki.pt
python -m compactor.train_compactor --wikitext --n-train 2000 --n-val 400 --epochs 5 --save compactor_wiki.pt
```

---

*与 `vec2text_test/demo.py` 中 gtr-t5-base + vec2text 设定一致；**§12** 为当前代码快照，后续迭代请同步更新本节与 Changelog。*
