# V6 Plan: Persistent Core-Residual Latent Memory

## 0. Why v5.2 Is Not Enough

`v5.2` moved CoRe_Mem from proxy/scaffold evidence to real pretrained encoders, real checkpoints, and multi-task trainable latent modules. That is useful, but it is not yet the target system.

The remaining gap is structural:

- PersonaMem crossed the verifier through answer-time text-vs-latent confidence routing with only `+1` correct over text-only.
- The v5.2 path still builds latent candidates from raw context chunks at evaluation time.
- There is no explicit persistent `core_bank` / `residual_bank` state that is written over a dialogue stream and later queried as the authoritative memory substrate.
- Ablations are partly composite-score ablations rather than true disabled-architecture reruns.

Therefore `v6` must target the actual research claim: **a persistent learned latent memory state, not a better scorer over text chunks**.

## 1. Highest-Level Goal

Build and verify a persistent core-residual latent memory system:

```text
dialogue / observation stream
  -> learned write-time router
  -> persistent core latent bank + residual latent bank
  -> query-conditioned latent reader
  -> learned belief / readout
  -> answer evidence
```

The authoritative v6 path must consume persistent memory state. Raw text may be used as source evidence, supervision, or audit trail, but it must not remain the primary retrieval substrate at evaluation time.

## 2. Hard Constraints

These constraints outrank short-term benchmark score:

- No fallback.
- No shortcut.
- No benchmark-specific heuristic.
- No provider prompt trick as a model claim.
- No PersonaMem gold leakage into memory writer, reader, controller, bank state, or belief decoder.
- No answer-time confidence routing as the authoritative path.
- No direct full raw-context retrieval at evaluation time.
- PersonaMem answer labels may only train a strictly isolated thin readout head, and calibrated results must be reported separately from no-calibration results.

## 3. Required System Changes

### 3.1 Persistent Core/Residual Banks

Implement explicit persistent bank state:

- `core_bank`: stable persona traits, durable preferences, repeated facts.
- `residual_bank`: recent facts, updates, conflicts, episodic evidence, temporary state.
- state snapshots with bank sizes, timestamps, provenance, and write traces.
- serialized checkpoint that can be loaded independently of raw dialogue.

### 3.2 Learned Write-Time Routing

The write controller must decide memory-state updates:

- `new_core`
- `new_residual`
- `merge_core`
- `merge_residual`
- `overwrite`
- `mark_stale`
- `promote_to_core`
- `ignore`

This routing is the good routing. It decides how memory is stored. It is different from v5.2 answer-time confidence routing, which is disallowed for authoritative v6 scoring.

### 3.3 Query-Conditioned Latent Reader

The reader must attend over persistent core/residual latent banks:

- core-only readout.
- residual-only readout.
- core+residual readout.
- shuffled-bank negative control.
- other-persona shuffled-bank negative control.

### 3.4 Learned Belief/Readout

The answer path should be:

```text
query + persistent latent memory -> belief/readout -> answer evidence
```

It should not be:

```text
if text margin > latent margin, use text answer; otherwise use latent answer
```

## 4. Training Data

Allowed:

- Stage2 `32k` train split for domain warm-up and multi-task objectives.
- PersonaMem raw context/dialogue without gold answers for self-supervised memory writing and readout.
- LLM teacher labels generated from raw dialogue/context/query only, with provenance and without PersonaMem gold.
- PersonaMem gold only for an optional isolated calibrated thin answer head.

Not allowed:

- training memory writer/reader/controller/bank state on PersonaMem `a/b/c/d` gold.
- using option text to supervise memory writing.
- using benchmark answer labels to choose routing or memory state.

## 5. Objectives

Minimum v6 objectives:

- stream-to-bank write loss.
- core/residual assignment and promotion loss.
- stale/overwrite/update loss.
- latent memory denoising under text dropout.
- query-to-persistent-bank contrastive alignment.
- belief reconstruction from persistent latent readout.
- bank consistency regularization across dialogue prefixes.
- optional calibrated answer-head CE on isolated PersonaMem train contexts only.

## 6. Evaluation And Anti-Shortcut Matrix

All retained v6 claims must report:

- no-calibration full589 PersonaMem.
- text-only baseline.
- latent-only persistent-bank path.
- core-only.
- residual-only.
- shuffled latent bank.
- shuffled persona bank.
- disabled write controller rerun.
- disabled belief/readout rerun.
- disabled core/residual split rerun.
- answer-time routing disabled.
- raw-context retrieval disabled in authoritative path.
- option paraphrase / answer-label permutation diagnostic when available.

The ablations must be true disabled-architecture reruns or separately executed scripts, not only composite score arithmetic.

## 7. Success Criteria

Mechanical stop is allowed only when all are true:

- persistent `core_bank` and `residual_bank` state exists and is checkpointed.
- query-time authoritative eval consumes persistent state, not full raw context retrieval.
- learned write-time routing updates state.
- answer-time confidence routing is disabled.
- trainable reader/controller/belief/readout are present.
- true architecture ablations drop.
- no PersonaMem gold is used for substrate.
- PersonaMem no-calibration beats text-only by a meaningful margin, target `>= +5%` absolute sample margin on full589, or an equivalent pre-declared significance artifact.

For full589, `+5%` means at least `30` more correct answers than text-only.

## 8. Milestones For Autoresearch

M1: v6 verifier and tests.

M2: persistent memory state schema and checkpoint format.

M3: stream writer that produces core/residual bank snapshots.

M4: learned write-time router connected to actual state updates.

M5: query-conditioned latent reader over persistent banks.

M6: belief/readout head that does not use answer-time routing.

M7: true disabled-architecture ablations.

M8: PersonaMem full589 no-calibration with raw-context retrieval disabled.

M9: significant margin or honest negative result.

M10: documentation and evidence package.

## 9. Autoresearch Constraint Sentence

> Do not optimize answer-time routing, provider prompting, or benchmark-specific option tricks. Build a persistent learned core/residual latent memory state with learned write-time routing and query-time latent readout. If it cannot beat text-only by a meaningful margin without raw-context retrieval or answer-time routing, record a negative result instead of claiming success.
