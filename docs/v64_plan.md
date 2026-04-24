# V6.4 Plan: Learned Observation Proposal / Extraction Coverage

## 0. Why V6.3 Blocked

`v6.3` proved that CoRe_Mem can sustain:

- persistent `core_bank` / `residual_bank`
- confidence-aware four-way write policy
- non-collapse state
- authoritative learned reader / learned decision

But the retained negative result made the upstream bottleneck explicit:

- PersonaMem full589 no-routing only reached `180/589`
- text-only was also `180/589`
- option-only remained `235/589`
- full589 error attribution recorded `never_written_count = 409`
- the other failure buckets were effectively `0`

So the current dominant problem is no longer "writer collapses the bank" or "decision head is disconnected".
The dominant problem is:

> the system still fails to propose enough needed observations from raw dialogue before write-time memory policy even starts.

## 1. Highest-Level Goal

Upgrade the observation path from:

```text
raw turn
  -> rule-first parser
  -> candidate observations
  -> write policy
  -> persistent memory
```

to:

```text
raw turn + local context
  -> rule proposer
  -> learned proposer
  -> merged candidate pool
  -> dedup / normalization / validation
  -> existing confidence-aware write policy
  -> persistent core/residual memory
```

The main claim for `v6.4` is not "we invented a new IE problem".
The main claim is:

> use a stronger learned observation proposal layer to remove the write-side coverage bottleneck, so the real `Core-Residual latent memory` contribution can continue.

## 2. Hard Constraints

These remain fixed:

- `Core-Residual persistent latent memory` stays the primary contribution
- no fallback
- no shortcut
- no benchmark-specific heuristic
- no provider prompt trick as a model claim
- no PersonaMem gold leakage into proposer / validator / writer / reader / controller / substrate
- no answer-time routing
- no raw full-context retrieval as the authoritative path
- do not turn observation extraction into a separate paper-claim track unless later evidence justifies it

## 3. Required V6.4 Upgrades

### 3.1 Learned Observation Proposer In The Authoritative Candidate Path

The system must no longer depend on `rule-first parser only`.

Required direction:

- add a learned observation proposer in the authoritative candidate path
- the proposer may be initialized from strong existing extraction-style solutions
- the proposer must read raw turn text plus bounded local context
- output must be normalized into the existing `Observation` schema or a lossless intermediate form

### 3.2 Hybrid Candidate Pool

The authoritative path must become:

- `rule proposer + learned proposer`
- merged into one candidate pool
- with provenance recorded per candidate

Required direction:

- keep rule candidates as a conservative floor
- let learned candidates increase recall
- log rule-only / learned-only / merged counts

### 3.3 Proposal Recall And Support Recovery

This run must measure the bottleneck directly.

Required direction:

- add parser-only vs hybrid proposer recall metrics
- measure recovered support for benchmark-needed memory
- require positive improvement in proposal recall or support recovery

### 3.4 Never-Written Reduction

The next run must prove that the dominant v6.3 failure bucket is shrinking.

Required direction:

- compare full589 `never_written_count` against parser-only baseline
- report absolute and relative reduction
- do not accept internal-only gains that fail to reduce this bucket

### 3.5 Dedup / Normalization / Validation Layer

Higher recall is allowed to add noise, but not uncontrolled noise.

Required direction:

- merge duplicates across rule and learned candidates
- normalize relation/value/time/polarity where needed
- keep the existing validator / write-policy path in control of final memory writes

### 3.6 Keep V6.3 Structural Gains

The proposer upgrade is not allowed to regress existing system structure.

Required direction:

- persistent banks, checkpoint, write trace stay live
- confidence-aware write policy stays authoritative after the candidate pool
- learned reader stays authoritative
- learned decision stays authoritative
- no-routing PersonaMem remains the benchmark gate

## 4. Allowed Borrowing Strategy

`v6.4` explicitly allows the use of strong existing solution families for the proposer layer, as long as they respect current project constraints.

Allowed:

- pretrained extraction / span / IE style backbones
- schema-conditioned extraction or instruction-conditioned extraction
- teacher-assisted proposal generation from raw dialogue only
- public no-gold extraction corpora or synthetic no-gold supervision

Not allowed:

- PersonaMem gold answer supervising the proposer
- option text supervising proposer outputs
- benchmark-specific handcrafted extraction rules presented as the scientific gain

## 5. Training Data

Allowed:

- stage2 `32k` train split
- current parser/silver observations as warm-start supervision only
- raw dialogue with local context for learned proposal targets
- teacher labels generated from raw dialogue only
- public IE / extraction data if schema-normalizable
- PersonaMem raw contexts without gold answers

Not allowed:

- PersonaMem `a/b/c/d` gold for proposer / validator / writer / substrate
- using benchmark answer correctness to tune proposer thresholds

## 6. Success Criteria

`v6.4` may stop only when all are true:

- persistent banks, checkpoint, write trace, learned write routing, learned reader, and learned decision still work
- authoritative candidate path uses a learned observation proposer
- hybrid rule + learned candidate pool exists
- proposal recall / support recovery is positive over parser-only
- full589 `never_written_count` is reduced
- dedup / normalization / validation evidence exists
- non-collapse state is preserved
- no answer-time routing
- no raw-context retrieval
- no PersonaMem gold leakage into proposer / substrate
- PersonaMem full589 no-routing beats `text-only`
- PersonaMem full589 no-routing beats `option-only`
- meaningful margin remains required; default is `>= +30` over `text-only` or a significance artifact

## 7. Milestones For Autoresearch

M1: `v6.4` verifier and tests.

M2: learned proposer scaffold wired into authoritative candidate path.

M3: hybrid candidate pool with provenance.

M4: proposal recall / support recovery artifact.

M5: full589 error attribution refresh with never-written comparison.

M6: validator + v6.3 write policy refresh on merged candidates.

M7: full PersonaMem 589 no-routing rerun.

M8: meaningful margin or honest negative result.

## 8. Autoresearch Constraint Sentence

> Do not treat observation extraction as a new parallel paper track. Use a strong learned observation proposer to remove the current parser-coverage bottleneck: hybridize rule candidates with learned candidates, measure proposal recall and support recovery directly, and reduce full589 `never_written` without regressing the persistent `Core-Residual` memory architecture. Keep no-answer-routing, no raw-context retrieval, and no-gold substrate rules fixed.
