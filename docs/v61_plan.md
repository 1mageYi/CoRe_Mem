# V6.1 Plan: Learned Reader/Decision over Persistent Memory

## 0. Why V6 Blocked

`v6` proved that CoRe_Mem can build a real persistent substrate:

- explicit `core_bank` / `residual_bank`
- state checkpoint
- write trace
- learned write-time routing
- raw-context retrieval disabled
- answer-time routing disabled
- true disabled-architecture ablations

But it also exposed the real bottleneck:

- PersonaMem no-routing is only `146/589`
- text-only is `205/589`
- the persistent bank exists, but the authoritative read/decision path is still too weak

The main issue is not storage anymore. The issue is that **the model still has not learned to read persistent memory and turn it into correct decisions**.

## 1. Highest-Level Goal

Upgrade `v6` into a system where the authoritative path is:

```text
dialogue stream
  -> learned write-time routing
  -> persistent core/residual latent banks
  -> learned reader over persistent state
  -> learned belief/readout / decision head
  -> answer
```

Not:

```text
persistent bank exists
  -> dot-product retrieval + bank prior
  -> handcrafted option scoring
  -> answer
```

## 2. Hard Constraints

These remain non-negotiable:

- no fallback
- no shortcut
- no benchmark-specific heuristic
- no provider prompt trick as a model claim
- no PersonaMem gold leakage into substrate / writer / reader / controller / state
- no answer-time confidence routing
- no raw full-context retrieval as the authoritative memory path

## 3. Required V6.1 Upgrades

### 3.1 Learned Reader In The Authoritative Path

The trained reader must actually rank slots during inference.

Required direction:

- remove `vector_dot + bank_prior` as the main authoritative reader
- use a learned query-slot scorer or equivalent learned reader path
- report explicit evidence that authoritative inference consumes the learned reader

### 3.2 Learned Decision / Readout Head

The final answer path must be learned.

Required direction:

- remove handcrafted `semantic + lexical_jaccard` option scoring from authoritative use
- replace it with a learned decision / answer head over persistent-memory readout
- keep calibrated thin heads isolated if they use benchmark labels

### 3.3 Semantic Slot Matching And Bank Cleanup

The bank must become easier to read.

Required direction:

- learned semantic matching for merge / overwrite / promote decisions
- dedup / compaction / revision chains
- reduce same-relation duplicate clutter

### 3.4 Typed Residuals

`other_fact` is too coarse.

Required direction:

- split residual content into typed subrelations when possible
- reduce `other_fact` dominance
- improve support specificity for belief/readout

### 3.5 Harder Internal Evaluation

Internal eval must become benchmark-aligned.

Required direction:

- natural-language queries, not direct `relation + value` echo queries
- hard negatives from same persona / same relation / similar topic
- evidence that the learned reader/decision path, not hand-written ranking, drives the metric

## 4. Training Data

Allowed:

- stage2 `32k` train split
- PersonaMem raw context without gold answers
- generated NL query / support / slot-matching supervision from raw context
- isolated calibrated thin head using benchmark gold only when clearly separated

Not allowed:

- training substrate / writer / reader / persistent state on PersonaMem `a/b/c/d` gold
- using option text to supervise writing or state construction
- using benchmark labels to choose routing or memory state

## 5. Success Criteria

V6.1 may stop only when all are true:

- persistent banks and write-time routing still work
- authoritative read path uses the learned reader
- authoritative answer path uses a learned decision/readout head
- semantic slot matching / bank compaction evidence exists
- typed residual evidence exists
- harder internal eval exists and is positive
- no answer-time routing
- no raw-context retrieval
- no PersonaMem gold leakage into substrate
- PersonaMem full589 no-routing beats both text-only and option-only
- meaningful margin over text-only remains required, default `>= +30`

## 6. Milestones For Autoresearch

M1: v6.1 verifier and tests.

M2: authoritative learned reader path wired into inference.

M3: authoritative learned decision/readout head replaces handcrafted scorer.

M4: semantic slot matching + bank compaction + revision-chain evidence.

M5: typed residuals and reduced `other_fact` dominance.

M6: harder internal eval with natural-language queries and hard negatives.

M7: full PersonaMem no-routing rerun.

M8: meaningful margin or honest negative result.

## 7. Autoresearch Constraint Sentence

> Do not go back to answer-time routing, handcrafted option tricks, or raw-context retrieval. Keep the persistent substrate from v6, but force the learned reader and learned decision head to become the authoritative path. If PersonaMem still cannot beat text-only and option-only without shortcuts, record the negative result honestly.
