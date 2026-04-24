# V6.5 Plan: Facetized Observation-to-Memory Redesign

## 0. Why V6.4 Soft-Blocked

`v6.4` established a real learned observation proposer and a real `rule + learned` hybrid candidate path.
The retained line is real:

- proposal recall `0.08488964346349745 > parser_only 0.015280135823429542`
- persistent `core_bank = 726`
- persistent `residual_bank = 960`
- `stream_observations_written = 3415`

But the benchmark result stayed negative:

- full589 no-routing `138/589`
- text-only `199/589`
- option-only `235/589`
- `never_written_count = 442 > parser_only 409`

The key structural diagnosis is now different from `v6.3` and different from early `v6.4`:

> the system is no longer failing only because it cannot propose candidates; it is failing because clause-level observations are too coarse to become stable, query-useful memory units.

The learned proposer can increase candidate recall.
It still does not reliably produce memory units that survive write-time competition and later become useful support.

## 1. Highest-Level Goal

Upgrade the memory-unit design from clause-level observations / slots:

```text
raw turn
  -> proposer
  -> clause-level observation
  -> write policy
  -> slot memory
```

to facetized memory units:

```text
raw turn + local context
  -> proposer
  -> facetizer / canonicalizer
  -> facet records
  -> facet-aware write utility
  -> persistent core/residual memory
  -> learned reader
  -> learned decision
```

`v6.5` is not a retreat from latent memory.
The intended direction is:

- `facet` defines a better memory unit
- `latent` remains the representation and dynamics layer

So the target is **facet-structured latent memory**, not a return to pure symbolic text slots.

## 2. Hard Constraints

These remain fixed:

- `Core-Residual persistent latent memory` remains the main contribution
- no fallback
- no shortcut
- no benchmark-specific heuristic
- no provider prompt trick as a scientific claim
- no PersonaMem gold leakage into facetizer / writer / reader / substrate
- no answer-time routing
- no raw full-context retrieval as the authoritative path
- do not turn extraction into a separate parallel paper track
- do not replace latent memory with text-only facet matching

## 3. Required V6.5 Upgrades

### 3.1 Facet Schema

The repo must stop treating the clause-level observation as the only memory unit.

Required direction:

- add an explicit facet schema
- each candidate observation may yield `1..k` facets
- facets must preserve provenance back to the source observation / clause

Minimal expected fields:

- `facet_type`
- `facet_value`
- `facet_scope`
- `facet_polarity`
- `source_observation_id`

### 3.2 Authoritative Facetizer In Path

The authoritative path must become:

- `rule proposer + learned proposer`
- merged candidate pool
- facetizer / canonicalizer
- existing write policy upgraded to facet-aware write utility

The facetizer may use existing strong extraction / IE methods.
It does not need to be presented as the paper's main contribution.

### 3.3 Facet-Aware Write Utility

Current same-relation clause competition is too coarse.

Required direction:

- write / merge / overwrite / preserve should reason over `(relation_family, facet_type)`
- broad abstract clauses must stop crowding out concrete needed facets
- state growth is allowed only if facet utility improves rather than merely increasing bank size

### 3.4 Facet-Level Error Attribution

`never_written` is no longer enough by itself.

Required direction:

- add facet-level failure buckets
- expected buckets include:
  - `needed_facet_missing`
  - `wrong_sibling_facet_selected`
  - `facet_written_but_reader_missed`
  - `facet_selected_but_decision_failed`

### 3.5 Needed-Facet Reduction

`v6.5` must prove that the redesign is fixing the specific clause-level bottleneck.

Required direction:

- compare facet-level missing-support counts against the `v6.4` baseline
- require reduction in `needed_facet_missing`
- if the redesign only reshapes state without reducing missing needed facets, it is not enough

### 3.6 Keep V6.4 Structural Gains

The redesign is not allowed to regress existing system structure.

Required direction:

- persistent banks, checkpoint, write trace remain live
- write routing remains authoritative
- learned reader remains authoritative
- learned decision remains authoritative
- no-routing PersonaMem remains the external gate

## 4. Allowed Borrowing Strategy

`v6.5` explicitly allows reuse of strong existing solution families where they help unblock the memory-unit bottleneck.

Allowed:

- learned observation proposal from strong extraction / IE style systems
- teacher-assisted clause-to-facet conversion from raw dialogue only
- pretrained extraction / normalization components
- public no-gold extraction corpora if schema-normalizable

Not allowed:

- PersonaMem gold answer supervising the facetizer or substrate
- option text supervising facet extraction
- benchmark-specific hand-coded facet rules presented as the scientific gain

## 5. Training Data

Allowed:

- stage2 `32k` train split
- current rule/silver observations as warm-start only
- raw dialogue with local context for clause-to-facet supervision
- teacher-generated facet labels from raw dialogue only
- PersonaMem raw contexts without answer labels

Not allowed:

- PersonaMem `a/b/c/d` gold for facetizer / writer / substrate
- using benchmark correctness to tune facet thresholds

## 6. Success Criteria

`v6.5` may stop only when all are true:

- persistent banks, checkpoint, write trace, learned write routing, learned reader, and learned decision still work
- an explicit facet schema exists
- an authoritative facetizer exists in the memory path
- write utility is facet-aware rather than relation-only
- facet-level error attribution exists
- `needed_facet_missing` is reduced relative to the retained `v6.4` line
- state remains non-collapsed
- no answer-time routing
- no raw-context retrieval
- no PersonaMem gold leakage into facetizer / substrate
- PersonaMem full589 no-routing beats `text-only`
- PersonaMem full589 no-routing beats `option-only`
- meaningful margin remains required

## 7. Milestones For Autoresearch

M1: `v6.5` verifier and tests.

M2: explicit facet schema and facet record plumbing.

M3: facetizer added after proposer and before write policy.

M4: facet-aware write utility / merge competition.

M5: facet-level error attribution artifact.

M6: full589 `needed_facet_missing` comparison against retained `v6.4`.

M7: full PersonaMem 589 no-routing rerun.

M8: meaningful margin or honest negative result.

## 8. Autoresearch Constraint Sentence

> Do not treat `v6.5` as a new extraction paper. Use facetized observation-to-memory conversion to fix the clause-level memory-unit bottleneck exposed by `v6.4`: keep the hybrid proposer as upstream input, add an authoritative facetizer and facet-aware write utility, measure facet-level error attribution directly, reduce `needed_facet_missing`, and preserve the existing Core-Residual latent-memory path without answer-time routing, raw-context retrieval, or gold leakage.
