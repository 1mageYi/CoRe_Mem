# Learned-Memory Related Work Notes

## Purpose

当前 stage-2 的下一步不再是继续沿着 rule-heavy 的 robustness 小修补推进，而是把主线重新锚定到：

- 更 learned 的 memory write/read
- 更强的 latent slots / memory tokens
- 更少的 hand-written rule 依赖

这份笔记只记录对当前 repo 最有启发的相关工作，以及它们对 CoRe_Mem 的直接设计含义。

## Key Inspirations

### End-To-End Memory Networks

- Source: https://papers.nips.cc/paper_files/paper/2015/hash/8fb21ee7a2207526da55a679f0332de2-Abstract.html
- Why it matters:
  - 强调 end-to-end multi-hop attention over external memory，而不是依赖大量显式监督或手工定义推理路径。
- CoRe_Mem implication:
  - `memory -> belief` 主链不应长期停留在 rule constructor。
  - 应该优先探索 learned read / rerank / multi-hop composition。

### Memorizing Transformers

- Source: https://arxiv.org/abs/2203.08913
- Why it matters:
  - 用 external memory 存过去 hidden states，并通过近邻读取增强长程建模。
- CoRe_Mem implication:
  - memory 不应只保留 symbolic gloss/value。
  - 需要更明确地支持 dense latent memory entries。

### RETRO

- Source: https://arxiv.org/abs/2112.04426
- Why it matters:
  - retrieval 不是简单拼回 prompt，而是进入模型内部作为 reader path 的一部分。
- CoRe_Mem implication:
  - 不应再把主要收益放在 prompt / option mapping。
  - 更值得做的是 learned retrieval-conditioned reader。

### LongMem

- Source: https://arxiv.org/abs/2306.07174
- Why it matters:
  - 将 backbone 与 long-term memory reader 解耦，让 memory encoder / retriever / reader 成为可训练、可替换模块。
- CoRe_Mem implication:
  - 当前最值得 learned 化的是：
    - `query encoder`
    - `slot encoder`
    - retrieval / ranking
    - belief selection / composition
  - online system 应逐步支持 checkpoint-backed memory components。

### Slot Attention

- Source: https://arxiv.org/abs/2006.15055
- Why it matters:
  - 用 slots 做竞争式绑定，得到可组合的 object-like latent representations。
- CoRe_Mem implication:
  - `structured latent-slot memory` 的 slot 不应只理解成 symbolic record。
  - 更理想的方向是 “traceable symbolic fields + compositional latent slots” 共存。

## Design Direction For CoRe_Mem

基于这些工作，当前 repo 的更合理主线是：

1. 减少 write/read 主链对手工规则的依赖。
2. 让 online `StructuredMemorySystem` 支持 learned memory path。
3. 让训练目标直接服务 online chain：
   - memory write
   - retrieval / rerank
   - belief selection / composition
   - answer support
4. 用 learned-mode benchmark evidence 证明系统智能程度提升，而不是用 prompt tricks 证明。

## Explicit Non-Goals

- 不把 benchmark-specific shortcut / fallback 重新加回主链。
- 不把 provider prompt 小修小补当成 learned-memory gain。
- 不继续把 rule-only parser / selector 微调当成当前主收益来源，除非它们明确服务 learned path。
