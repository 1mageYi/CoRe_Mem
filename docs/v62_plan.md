# V6.2 Plan: Learned Write-Worthiness / Attribute-Validity Before Extraction

## 0. Why V6.1 Blocked

`v6.1` proved that CoRe_Mem can now:

- build and persist explicit `core_bank` / `residual_bank`
- use learned reader and learned decision heads in the authoritative no-routing path
- beat `text-only` on PersonaMem full589 (`191/589` vs `180/589`)

But it still failed the real benchmark gate:

- `option-only = 235/589`
- `v6.1 learned no-routing = 191/589`
- multiple late-stage fixes (`decision tweak`, `bank simplification`, `fragment suppression`, `parser self-state precision`) did not produce a retained keep

The bottleneck is no longer storage or basic readout wiring. The bottleneck is that the write side still injects too much weak or low-value content into the persistent state, so downstream learned reader/decision modules do not receive a clean enough substrate.

## 1. Highest-Level Goal

Upgrade the authoritative path from:

```text
raw dialogue
  -> heuristic/weak extraction
  -> persistent banks
  -> learned reader
  -> learned decision
  -> answer
```

to:

```text
raw dialogue
  -> learned write-worthiness / attribute-validity filter
  -> learned write action over valid observations
  -> cleaner persistent core/residual banks
  -> learned reader
  -> learned decision
  -> answer
```

## 2. Hard Constraints

These remain fixed:

- no fallback
- no shortcut
- no benchmark-specific heuristic
- no provider prompt trick as a model claim
- no PersonaMem gold leakage into substrate / writer / reader / controller / state
- no answer-time routing
- no raw full-context retrieval as the authoritative memory path
- no post-hoc heuristic cleanup masquerading as learned write quality

## 3. Required V6.2 Upgrades

### 3.1 Learned Write-Worthiness In The Authoritative Write Path

Before writing anything into memory, the system must learn whether the candidate observation is worth writing at all.

Required direction:

- train a `write-worthiness` model on raw-dialogue-derived supervision
- connect it to the actual authoritative write path
- replace broad heuristic "always write parsed fact" behavior with learned gating

### 3.2 Attribute-Validity / Relation-Validity Before Write

The system must learn whether the extracted attribute is valid enough to enter the bank.

Required direction:

- train `attribute-validity` and/or `relation-validity` models before write
- suppress low-information fragments, malformed self-state, and weak pseudo-facts before they become bank entries
- keep this as a learned pre-write stage, not a benchmark-specific patch

### 3.3 Cleaner Persistent State As A First-Class Metric

The bank quality itself must become measurable.

Required direction:

- expose bank quality metrics such as noise rate, low-information slot share, duplicate/stale share, write precision, or comparable equivalents
- require evidence that the cleaned write path improves state quality, not only downstream score

### 3.4 Harder Write-Quality Internal Evaluation

Internal write evaluation must resemble the real failure mode.

Required direction:

- evaluate on raw dialogue observations
- include hard negatives: same persona, same relation family, conflicting updates, low-information fragments
- require learned write-worthiness / validity to outperform disabled or heuristic baselines

### 3.5 Reader / Decision Retraining On Cleaner Banks

The write-side upgrade is not enough unless it produces a better downstream path.

Required direction:

- regenerate persistent state with the new write path
- retrain or refresh learned reader/decision on the cleaner banks
- keep the no-routing constraint for PersonaMem evaluation

## 4. Training Data

Allowed:

- stage2 `32k` train split
- PersonaMem raw context without gold answers
- raw-dialogue-derived write-worthiness / fact-validity / conflict / update supervision
- synthetic or public no-gold query-support / decision supervision

Not allowed:

- PersonaMem `a/b/c/d` gold in substrate / writer / extraction / persistent-state supervision
- option text supervising write-worthiness or state construction
- post-hoc score tuning using benchmark gold as the main retained path

## 5. Success Criteria

`v6.2` may stop only when all are true:

- persistent banks, checkpoint, write trace, learned write routing, learned reader, and learned decision still work
- authoritative write path uses a learned write-worthiness gate
- authoritative pre-write path uses learned attribute-validity / relation-validity
- hard write-quality internal eval exists and is positive
- persistent-state cleanliness evidence exists and is positive
- no answer-time routing
- no raw-context retrieval
- no PersonaMem gold leakage into substrate
- PersonaMem full589 no-routing still beats `text-only`
- PersonaMem full589 no-routing also beats `option-only`
- meaningful margin remains required; default is `>= +30` over `text-only` or a significance artifact

## 6. Milestones For Autoresearch

M1: `v6.2` verifier and tests.

M2: learned write-worthiness model trained and wired into authoritative write path.

M3: learned attribute-validity / relation-validity model trained and wired before write.

M4: write-quality internal eval with raw dialogue, hard negatives, and disabled-baseline comparison.

M5: persistent-state cleanliness metrics and evidence.

M6: regenerate banks and retrain/refresh reader + decision on cleaner state.

M7: full PersonaMem 589 no-routing rerun.

M8: meaningful margin or honest negative result.

## 7. Autoresearch Constraint Sentence

> Do not go back to answer-time routing, handcrafted option tricks, raw-context retrieval, or broad heuristic cleanup. Keep the persistent substrate and learned reader/decision from `v6.1`, but move the bottleneck to the right place: learned write-worthiness and learned attribute-validity before extraction and before write. If cleaner writing still does not beat `option-only`, record the negative result honestly.
