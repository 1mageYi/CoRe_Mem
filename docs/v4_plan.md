# v4 Persona-First Learned Memory Plan

Persona-first scope: `PersonaMem 512` is the primary benchmark, while
`LongMemEval-S` is a guard.

## Thesis

`v4` narrows the research environment to `PersonaMem` because the current `v33`
evidence shows two different failure modes:

- `PersonaMem` already has a strong learned-authoritative local signal
  (`v33 local 196/512` versus retained `v32 local 175/512`), but the provider
  final-label channel loses part of that gain.
- `LongMemEval-S` currently behaves like an open-ended exact-span extraction
  problem and pulls optimization toward a different architecture.

The goal is not to overfit PersonaMem templates. The goal is to make the
learned `write -> latent -> belief -> option scorer` path strong enough that
the system can answer finite-option personalization questions without relying
on provider-side label execution.

## Hard Constraints

- Do not change the `core / residual` memory substrate.
- No fallback.
- No shortcut.
- No benchmark-specific label mapping.
- No benchmark leakage.
- Do not train on benchmark answers.
- Do not claim provider-rendered improvements as model capability unless the
  learned local/option-scorer path improves too.

## Primary Target

`PersonaMem 512` is the main optimization target for this run.

Primary success means:

- learned authoritative runtime remains enabled:
  - `memory_mode = learned_memory`
  - `slot_assignment_mode = learned`
- learned option-scorer/local answer exact beats the current `v33` full
  PersonaMem local score (`196/512`)
- provider exact is reported but not allowed to be the only source of a keep
- ablation attributes the gain to learned latent / belief / option scoring

## Secondary Guard

`LongMemEval-S 500` is not the main optimization target in `v4`.

It is retained as a non-catastrophic regression guard:

- do not let the learned path collapse below the current `v33` local score
  (`14/500`)
- record provider behavior, but do not let open-ended exact-span behavior pull
  the main optimization away from PersonaMem

## Milestones

### M1. Gap Audit

Publish a `PersonaMem 512` provider-vs-local gap audit from the latest `v33`
full run:

- local-correct/provider-wrong count
- provider-correct/local-wrong count
- provider blank/non-label counts
- question-type/topic buckets

This should make the local/provider gap explicit before new modeling work.

### M2. Learned Option Scorer Authority

Move final PersonaMem answer authority toward the learned option scorer:

- no provider final-label dependency for the main learned metric
- option scoring consumes belief graph and selected supports
- answer-head output is a model artifact, not a post-hoc mapping table

### M3. Persona Latent Reader Strengthening

Train/evaluate a Persona-focused latent reader:

- query-option-support contrastive objective
- hard negative option ranking
- support attribution objective
- selected-slot gloss calibration

### M4. Persona Belief Graph Strengthening

Train/evaluate a Persona-focused belief graph head:

- preference/persona relation selection
- support selection
- stale support rejection
- multi-facet preference support

### M5. Full PersonaMem 512 Evaluation

Run full `PersonaMem 512` under learned authoritative runtime and publish:

- learned local exact
- learned option-scorer exact
- provider exact as auxiliary evidence
- comparison against `v32` and `v33`

### M6. Ablation Truth

Publish ablations:

- without learned latent reader
- without belief graph head
- without option scorer
- with symbolic runtime baseline

The run succeeds only if the PersonaMem gain is attributed to learned modules,
not to runtime repair layers.

## Non-Goals

- Do not chase `LongMemEval-S` quality in this run.
- Do not add Persona-specific answer rewrite rules.
- Do not add new provider prompt tricks as the main gain mechanism.
- Do not expand teacher labeling unless a matched learned-training objective
  consumes it.

## Stop Condition

`v4` is successful when:

- `PersonaMem 512` learned local / option-scorer exact beats `v33`
- the learned authoritative path is the main path
- ablation confirms latent + belief + option scorer contribution
- LongMemEval-S does not catastrophically regress
- all constraints above remain true
