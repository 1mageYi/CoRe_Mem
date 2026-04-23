# V6.3 Plan: Recall-Preserving Confidence-Aware Write Policy

## 0. Why V6.2 Blocked

`v6.2` proved that CoRe_Mem can train and connect:

- learned write-worthiness
- learned attribute-validity / relation-validity
- hard write-quality eval on raw dialogue
- persistent-state cleanliness metrics

But it also exposed the failure mode clearly:

- PersonaMem full589 no-routing fell to `160/589`
- text-only was `170/589`
- option-only stayed `235/589`
- persistent state collapsed to `core_bank = 4`, `residual_bank = 42`, `stream_observations_written = 64`

So the problem is not "write quality is impossible". The problem is that `v6.2` implemented write quality as a high-precision hard filter, and the system wrote too little useful memory.

## 1. Highest-Level Goal

Upgrade the write path from:

```text
turn -> write-worthiness -> write / drop
observation -> attribute-validity -> keep / drop
```

to:

```text
turn / observation
  -> confidence-aware write policy
  -> core-worthy / residual-worthy / weak-but-keep / drop
  -> persistent state with non-collapsed support coverage
  -> learned reader
  -> learned decision
```

The main claim for `v6.3` is:

> Better memory-state dynamics, not a cleaner-but-emptier bank.

## 2. Hard Constraints

These remain fixed:

- no fallback
- no shortcut
- no benchmark-specific heuristic
- no provider prompt trick as a model claim
- no PersonaMem gold leakage into substrate / writer / reader / controller / state
- no answer-time routing
- no raw full-context retrieval as the authoritative memory path
- no broad heuristic cleanup disguised as write-policy innovation

## 3. Required V6.3 Upgrades

### 3.1 Confidence-Aware Four-Way Write Policy

The authoritative write path must stop being binary.

Required direction:

- predict `core-worthy`
- predict `residual-worthy`
- predict `weak-but-keep`
- predict `drop`

`weak-but-keep` must write into a controlled low-confidence residual path instead of being discarded.

### 3.2 Weak-But-Keep Residual Buffer

The system must preserve uncertain but potentially useful information.

Required direction:

- weak-but-keep entries must be stored in `residual`
- they must carry lower confidence / softer update strength
- later merge / promote / stale dynamics may revise them
- they cannot be treated as immediate core facts

### 3.3 Recall-Preserving State Quality

State quality must no longer be evaluated as cleanliness alone.

Required direction:

- add support-coverage / write-recall metrics
- prove that the new write policy improves or preserves coverage relative to the hard-filter baseline
- explicitly forbid degenerate "tiny clean bank" wins

### 3.4 Error Attribution By Pipeline Stage

The next run must explain where benchmark failures come from.

Required direction:

- classify misses into at least:
  - `never_written`
  - `written_but_compacted_away`
  - `written_but_reader_missed`
  - `reader_selected_but_decision_failed`
- use this to verify that the run is really fixing write-side recall

### 3.5 Keep V6.1/V6.2 Structural Gains

The write-policy upgrade is not allowed to regress core structure.

Required direction:

- persistent banks, checkpoint, write trace stay live
- learned reader stays authoritative
- learned decision stays authoritative
- no-routing PersonaMem remains the benchmark gate

## 4. Training Data

Allowed:

- stage2 `32k` train split
- PersonaMem raw context without gold answers
- raw-dialogue-derived supervision for write policy / support coverage / update consistency
- synthetic or public no-gold reader / decision supervision

Not allowed:

- PersonaMem `a/b/c/d` gold in substrate / writer / extraction / persistent-state supervision
- option text supervising write policy or state construction
- benchmark gold used to hand-tune thresholding or route selection

## 5. Success Criteria

`v6.3` may stop only when all are true:

- persistent banks, checkpoint, write trace, learned write routing, learned reader, and learned decision still work
- authoritative write path uses a confidence-aware four-way write policy
- weak-but-keep writes into residual instead of direct drop
- support coverage / write recall evidence exists and is positive
- state does not collapse into a tiny bank
- error attribution artifact exists
- no answer-time routing
- no raw-context retrieval
- no PersonaMem gold leakage into substrate
- PersonaMem full589 no-routing beats `text-only`
- PersonaMem full589 no-routing beats `option-only`
- meaningful margin remains required; default is `>= +30` over `text-only` or a significance artifact

## 6. Milestones For Autoresearch

M1: `v6.3` verifier and tests.

M2: confidence-aware four-way write policy trained and wired into authoritative write path.

M3: weak-but-keep residual buffer and confidence-aware state update dynamics.

M4: support-coverage / write-recall eval and non-collapse state metrics.

M5: error attribution artifact for full589 failures.

M6: reader + decision refresh on the new state dynamics.

M7: full PersonaMem 589 no-routing rerun.

M8: meaningful margin or honest negative result.

## 7. Autoresearch Constraint Sentence

> Do not make the bank cleaner by deleting most of memory. Keep the v6.1 learned reader/decision and the v6.2 write-quality signal, but convert write quality into a confidence-aware memory policy: `core-worthy`, `residual-worthy`, `weak-but-keep`, `drop`. The system must preserve useful weak evidence in residual memory and prove better support coverage, not just lower noise. If this still does not beat `option-only`, record the negative result honestly.
