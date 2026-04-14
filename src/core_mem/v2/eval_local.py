"""Local intrinsic evaluation helpers for stage-2."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from core_mem.v2.consolidation import ConsolidationManager
from core_mem.v2.decoder import BeliefDecoder, _infer_query_type
from core_mem.v2.encoder import QueryEncoder, SlotEncoder
from core_mem.v2.lifecycle import LifecycleDecision, LifecycleManager
from core_mem.v2.parser import Stage2ObservationParser
from core_mem.v2.projection import AnswerProjection
from core_mem.v2.resampler import LightResampler
from core_mem.v2.experiments import dataset_allowed_for_variant
from core_mem.v2.schemas import BeliefItem, BeliefState, Observation, SlotRecord
from core_mem.v2.system import StructuredMemoryState, StructuredMemorySystem
from core_mem.v2.vector_ops import dot_product, mean_vectors


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _dcg(hit_positions: list[int]) -> float:
    return sum(1.0 / math.log2(position + 2.0) for position in hit_positions)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _meta(row: dict[str, Any]) -> dict[str, Any]:
    return dict(row.get("_meta", {}) or {})


def _dataset_of(row: dict[str, Any]) -> str:
    return str(_meta(row).get("dataset", "unknown"))


def _sample_id_of(row: dict[str, Any]) -> str:
    return str(_meta(row).get("sample_id", ""))


def _belief_pairs(belief: BeliefState | dict[str, Any]) -> set[tuple[str, str]]:
    payload = belief.to_dict() if isinstance(belief, BeliefState) else belief
    return {
        (str(item.get("relation", "")), str(item.get("value", "")))
        for item in payload.get("belief_items", [])
    }


def _belief_support_ids(belief: BeliefState | dict[str, Any]) -> set[str]:
    payload = belief.to_dict() if isinstance(belief, BeliefState) else belief
    support: set[str] = set()
    for item in payload.get("belief_items", []):
        support.update(str(slot_id) for slot_id in item.get("support_slot_ids", []))
    return support


def _belief_value_f1(predicted: set[tuple[str, str]], gold: set[tuple[str, str]]) -> float:
    if not predicted and not gold:
        return 1.0
    precision = _safe_divide(len(predicted & gold), len(predicted))
    recall = _safe_divide(len(predicted & gold), len(gold))
    return _safe_divide(2 * precision * recall, precision + recall)


def _rank_slots(query_encoder: QueryEncoder, query: str, slots: list[SlotRecord]) -> list[SlotRecord]:
    query_vector = query_encoder.encode(query)
    return sorted(slots, key=lambda slot: dot_product(query_vector, slot.retrieval_key), reverse=True)


def _rank_slots_with_variant(
    query_encoder: QueryEncoder,
    query: str,
    slots: list[SlotRecord],
    *,
    assignment_mode: str,
) -> list[SlotRecord]:
    ranked = _rank_slots(query_encoder, query, slots)
    if assignment_mode != "randomized":
        return ranked
    return sorted(
        ranked,
        key=lambda slot: f"{slot.slot_id}|{query}",
    )


def _gold_answer(query: str, belief_payload: dict[str, Any], projection: AnswerProjection) -> str:
    belief = BeliefState.from_dict(belief_payload)
    return projection.project_answer(query, belief)


def _reactivate_target_slots(slots: list[SlotRecord], affected_slot_ids: list[str]) -> list[SlotRecord]:
    affected = set(affected_slot_ids)
    adjusted: list[SlotRecord] = []
    for slot in slots:
        if slot.slot_id in affected:
            adjusted.append(
                SlotRecord(
                    slot_id=slot.slot_id,
                    bank=slot.bank,
                    entity=slot.entity,
                    relation=slot.relation,
                    retrieval_key=list(slot.retrieval_key),
                    latent_tokens=[list(token) for token in slot.latent_tokens],
                    soft_role_scores=slot.soft_role_scores,
                    confidence=slot.confidence,
                    first_seen_ts=slot.first_seen_ts,
                    last_update_ts=slot.last_update_ts,
                    revision_count=slot.revision_count,
                    active_flag=True,
                    revision_parent=slot.revision_parent,
                    canonical_gloss=slot.canonical_gloss,
                )
            )
        else:
            adjusted.append(slot)
    return adjusted


def _state_from_slots(slots: list[SlotRecord]) -> StructuredMemoryState:
    return StructuredMemoryState(
        core_slots=[slot for slot in slots if slot.bank == "core"],
        residual_slots=[slot for slot in slots if slot.bank != "core"],
    )


def _state_from_slots_with_variant(
    slots: list[SlotRecord],
    *,
    bank_mode: str,
    assignment_mode: str,
) -> StructuredMemoryState:
    adjusted = list(slots)
    if assignment_mode == "randomized":
        randomized: list[SlotRecord] = []
        for slot in adjusted:
            digest = hashlib.sha1(slot.slot_id.encode("utf-8")).hexdigest()
            randomized.append(replace(slot, bank="core" if int(digest, 16) % 2 == 0 else "residual"))
        adjusted = randomized
    if bank_mode == "single":
        adjusted = [replace(slot, bank="residual") for slot in adjusted]
    return _state_from_slots(adjusted)


@dataclass(frozen=True)
class MeanPoolingResampler:
    latent_queries: int = 8

    def compose(self, query_vector: list[float], slots: list[SlotRecord]) -> list[list[float]]:
        del query_vector
        if not slots:
            return []
        all_tokens = [token for slot in slots for token in slot.latent_tokens]
        anchor = mean_vectors(all_tokens)
        return [list(anchor) for _ in range(self.latent_queries)]


@dataclass(frozen=True)
class DirectAnswerBeliefDecoder:
    def decode(self, query_id: str, query_text: str, slots: list[SlotRecord]) -> BeliefState:
        del query_text
        if not slots:
            return BeliefState(query_id=query_id, entity="user", query_type="single_fact", belief_items=[], global_consistency="low")
        top_slot = slots[0]
        return BeliefState(
            query_id=query_id,
            entity="user",
            query_type="single_fact",
            belief_items=[
                BeliefItem(
                    relation="other_fact",
                    value=top_slot.canonical_gloss,
                    status="active",
                    time_scope="current",
                    confidence=top_slot.confidence,
                    support_slot_ids=[top_slot.slot_id],
                )
            ],
            global_consistency="medium",
        )


@dataclass(frozen=True)
class OptimusLikeBeliefDecoder:
    max_items: int = 1

    def decode(self, query_id: str, query_text: str, slots: list[SlotRecord]) -> BeliefState:
        belief_items: list[BeliefItem] = []
        for slot in reversed(slots):
            if not slot.active_flag:
                continue
            belief_items.append(
                BeliefItem(
                    relation=slot.relation,
                    value=slot.canonical_gloss.split("=")[-1].strip() if "=" in slot.canonical_gloss else slot.canonical_gloss,
                    status="active",
                    time_scope="current",
                    confidence=max(slot.confidence - 0.1, 0.0),
                    support_slot_ids=[slot.slot_id],
                )
            )
            if len(belief_items) >= self.max_items:
                break
        return BeliefState(
            query_id=query_id,
            entity="user",
            query_type=_infer_query_type(query_text),
            belief_items=belief_items,
            global_consistency="medium" if belief_items else "low",
        )


@dataclass(frozen=True)
class VariantLifecycleManager:
    base: LifecycleManager
    overwrite_mode: str

    def decide(self, observation: Observation, slots: list[SlotRecord]) -> LifecycleDecision:
        decision = self.base.decide(observation, slots)
        if self.overwrite_mode == "merge_only" and decision.action == "overwrite":
            return LifecycleDecision(
                action="merge",
                matched_slot_id=decision.matched_slot_id,
                promote=decision.promote,
                stale_old=False,
            )
        return decision


@dataclass(frozen=True)
class VariantConsolidationManager:
    base: ConsolidationManager
    bank_mode: str
    assignment_mode: str

    def apply(
        self,
        core_slots: list[SlotRecord],
        residual_slots: list[SlotRecord],
        decision: LifecycleDecision,
    ) -> tuple[list[SlotRecord], list[SlotRecord]]:
        next_core, next_residual = self.base.apply(core_slots, residual_slots, decision)
        slots = [*next_core, *next_residual]
        if self.assignment_mode == "randomized":
            randomized: list[SlotRecord] = []
            for slot in slots:
                digest = hashlib.sha1(slot.slot_id.encode("utf-8")).hexdigest()
                randomized.append(replace(slot, bank="core" if int(digest, 16) % 2 == 0 else "residual"))
            slots = randomized
        if self.bank_mode == "single":
            return [], [replace(slot, bank="residual") for slot in slots]
        return _state_from_slots(slots).core_slots, _state_from_slots(slots).residual_slots


def _variant_components(top_k: int, variant: dict[str, Any]) -> tuple[Any, Any, Any, Any]:
    resampler_type = str(variant.get("resampler_type", "light"))
    decoder_type = str(variant.get("decoder_type", "belief_json"))
    overwrite_mode = str(variant.get("overwrite_mode", "merge_overwrite"))
    bank_mode = str(variant.get("bank_mode", "dual"))
    assignment_mode = str(variant.get("assignment_mode", "default"))

    lifecycle = VariantLifecycleManager(LifecycleManager(), overwrite_mode=overwrite_mode)
    consolidation = VariantConsolidationManager(
        ConsolidationManager(),
        bank_mode=bank_mode,
        assignment_mode=assignment_mode,
    )
    if resampler_type == "mean_pooling":
        resampler: Any = MeanPoolingResampler(latent_queries=top_k)
    else:
        resampler = LightResampler(latent_queries=top_k)

    if decoder_type == "direct_answer":
        decoder: Any = DirectAnswerBeliefDecoder()
    elif decoder_type == "optimus_like":
        decoder = OptimusLikeBeliefDecoder(max_items=max(1, min(2, top_k)))
    else:
        decoder = BeliefDecoder(max_items=top_k)
    return lifecycle, consolidation, resampler, decoder


@dataclass(frozen=True)
class ModuleSpec:
    name: str
    role: str
    evaluated_by: list[str]


MODULE_INVENTORY = [
    ModuleSpec("parser", "observation extraction", ["parser_replay_coverage", "parser_replay_exact"]),
    ModuleSpec("slot_encoder", "observation -> slot encoding", ["slot_relation_accuracy", "slot_value_recoverability"]),
    ModuleSpec("query_encoder", "query representation", ["recall_at_k", "ndcg_at_k"]),
    ModuleSpec("retrieval_key_head", "slot retrieval scoring", ["recall_at_k", "ndcg_at_k", "support_slot_recall"]),
    ModuleSpec("lifecycle", "update action and flags", ["update_action_accuracy", "update_flag_accuracy"]),
    ModuleSpec("consolidation", "core/residual overwrite locality", ["specificity_locality", "edit_locality_at_slot_level"]),
    ModuleSpec("core_residual_memory", "budgeted memory state", ["compression_fidelity", "budget_joint_belief_accuracy"]),
    ModuleSpec("resampler", "query-conditioned composition", ["joint_belief_accuracy", "slot_value_f1"]),
    ModuleSpec("belief_decoder", "belief JSON recovery", ["joint_belief_accuracy", "slot_value_f1", "propagation_accuracy"]),
    ModuleSpec("answer_projection", "belief -> answer text", ["answer_exact_match"]),
]


def module_inventory_payload() -> list[dict[str, Any]]:
    return [{"name": item.name, "role": item.role, "evaluated_by": list(item.evaluated_by)} for item in MODULE_INVENTORY]


def evaluate_local(
    prepared_manifest_path: Path,
    output_root: Path,
    *,
    top_k: int = 8,
    budgets: list[int] | None = None,
    datasets: list[str] | None = None,
    variant: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = _load_json(prepared_manifest_path)
    variant = dict(variant or {})
    selected_datasets = set(datasets or [])
    requested_budgets = sorted({budget for budget in (budgets or [1, 2, 4, 8]) if budget > 0})

    slot_rows = _load_jsonl(Path(manifest["task_files"]["slot_autoencoding"]))
    retrieval_rows = _load_jsonl(Path(manifest["task_files"]["retrieval_alignment"]))
    lifecycle_rows = _load_jsonl(Path(manifest["task_files"]["lifecycle_prediction"]))
    belief_rows = _load_jsonl(Path(manifest["task_files"]["composition_to_belief"]))

    if selected_datasets:
        slot_rows = [row for row in slot_rows if _dataset_of(row) in selected_datasets]
        retrieval_rows = [row for row in retrieval_rows if _dataset_of(row) in selected_datasets]
        lifecycle_rows = [row for row in lifecycle_rows if _dataset_of(row) in selected_datasets]
        belief_rows = [row for row in belief_rows if _dataset_of(row) in selected_datasets]

    disabled_pools = [str(pool) for pool in variant.get("disabled_pools", [])]
    if disabled_pools:
        slot_rows = [row for row in slot_rows if dataset_allowed_for_variant(_dataset_of(row), disabled_pools)]
        retrieval_rows = [row for row in retrieval_rows if dataset_allowed_for_variant(_dataset_of(row), disabled_pools)]
        lifecycle_rows = [row for row in lifecycle_rows if dataset_allowed_for_variant(_dataset_of(row), disabled_pools)]
        belief_rows = [row for row in belief_rows if dataset_allowed_for_variant(_dataset_of(row), disabled_pools)]

    parser = Stage2ObservationParser()
    slot_encoder = SlotEncoder()
    query_encoder = QueryEncoder()
    lifecycle, consolidation, resampler, decoder = _variant_components(top_k, variant)
    projection = AnswerProjection()

    parser_coverage: list[float] = []
    parser_exact: list[float] = []
    slot_relation_hits: list[float] = []
    slot_value_hits: list[float] = []
    for row in slot_rows:
        observation = Observation.from_dict(row["input_observation"])
        target = row["target_record"]
        encoded = slot_encoder.encode(observation, timestamp="2026-04-14T00:00:00Z")
        slot_relation_hits.append(float(encoded.relation == str(target.get("relation", ""))))
        slot_value_hits.append(float(str(target.get("value", "")).lower() in encoded.canonical_gloss.lower()))

        if observation.speaker == "user" and observation.evidence_text:
            parsed = parser.parse_turn(
                observation.evidence_text,
                source_dataset=observation.source_dataset,
                source_dialogue_id=observation.source_dialogue_id,
                source_turn_id=observation.source_turn_id,
                session_id=observation.session_id,
                speaker=observation.speaker,
            )
            parser_coverage.append(float(bool(parsed)))
            exact_match = any(
                item.relation == observation.relation and item.value.lower() == observation.value.lower()
                for item in parsed
            )
            parser_exact.append(float(exact_match))

    retrieval_hits: list[float] = []
    ndcg_values: list[float] = []
    for row in retrieval_rows:
        slots = [SlotRecord.from_dict(row["positive_slot"])] + [SlotRecord.from_dict(item) for item in row["negative_slots"]]
        ranked = _rank_slots_with_variant(
            query_encoder,
            row["query"],
            slots,
            assignment_mode=str(variant.get("assignment_mode", "default")),
        )[:top_k]
        gold = set(row["gold_support_slot_ids"])
        hits = [index for index, slot in enumerate(ranked) if slot.slot_id in gold]
        retrieval_hits.append(float(bool(hits)))
        ndcg_values.append(_safe_divide(_dcg(hits), _dcg(list(range(len(gold)))) if gold else 0.0))

    belief_joint: list[float] = []
    belief_f1: list[float] = []
    support_recall: list[float] = []
    answer_exact: list[float] = []
    composed_token_counts: list[float] = []
    budget_rows: list[dict[str, Any]] = []
    composition_by_sample = {_sample_id_of(row): row for row in belief_rows if _sample_id_of(row)}

    for row in belief_rows:
        slots = [SlotRecord.from_dict(item) for item in row["memory_slots"]]
        ranked = _rank_slots_with_variant(
            query_encoder,
            row["query"],
            slots,
            assignment_mode=str(variant.get("assignment_mode", "default")),
        )
        selected = ranked[:top_k]
        composed = resampler.compose(query_encoder.encode(row["query"]), selected)
        predicted = decoder.decode(
            _sample_id_of(row) or "belief_eval",
            row["query"],
            selected,
        )
        gold_pairs = _belief_pairs(row["target_belief_json"])
        predicted_pairs = _belief_pairs(predicted)
        belief_joint.append(float(predicted_pairs == gold_pairs))
        belief_f1.append(_belief_value_f1(predicted_pairs, gold_pairs))
        gold_support = _belief_support_ids(row["target_belief_json"])
        predicted_support = _belief_support_ids(predicted)
        support_recall.append(_safe_divide(len(predicted_support & gold_support), len(gold_support)))
        gold_answer = _gold_answer(row["query"], row["target_belief_json"], projection)
        predicted_answer = projection.project_answer(row["query"], predicted)
        answer_exact.append(float(predicted_answer == gold_answer))
        composed_token_counts.append(float(len(composed)))

        for budget in requested_budgets:
            budget_selected = ranked[: min(budget, len(ranked))]
            budget_predicted = decoder.decode(
                f"{_sample_id_of(row) or 'belief_eval'}_budget_{budget}",
                row["query"],
                budget_selected,
            )
            budget_pairs = _belief_pairs(budget_predicted)
            budget_support = _belief_support_ids(budget_predicted)
            budget_rows.append(
                {
                    "dataset": _dataset_of(row),
                    "sample_id": _sample_id_of(row),
                    "budget": budget,
                    "joint_belief_accuracy": float(budget_pairs == gold_pairs),
                    "belief_item_recall": _safe_divide(len(budget_pairs & gold_pairs), len(gold_pairs)),
                    "support_slot_recall": _safe_divide(len(budget_support & gold_support), len(gold_support)),
                }
            )

    update_action: list[float] = []
    update_flags: list[float] = []
    propagation_values: list[float] = []
    locality_values: list[float] = []
    edit_locality_values: list[float] = []
    for row in lifecycle_rows:
        raw_context = [SlotRecord.from_dict(item) for item in row["memory_context"]]
        target_action = str(row["target_action"])
        affected_slot_ids = [str(slot_id) for slot_id in row.get("affected_slot_ids", [])]
        context = _reactivate_target_slots(raw_context, affected_slot_ids)
        observation = Observation.from_dict(row["new_observation"])
        decision = lifecycle.decide(observation, context)
        update_action.append(float(decision.action == target_action))
        target_flags = dict(row.get("target_flags", {}))
        update_flags.append(
            float(
                decision.promote == bool(target_flags.get("promote", False))
                and decision.stale_old == bool(target_flags.get("stale_old", False))
            )
        )

        sample_id = _sample_id_of(row)
        paired_belief = composition_by_sample.get(sample_id)
        if paired_belief is None:
            continue

        system = StructuredMemorySystem(
            parser=parser,
            slot_encoder=slot_encoder,
            query_encoder=query_encoder,
            lifecycle=lifecycle,
            consolidation=consolidation,
            resampler=resampler,
            decoder=decoder,
            projection=projection,
            top_k=top_k,
            state=_state_from_slots_with_variant(
                context,
                bank_mode=str(variant.get("bank_mode", "dual")),
                assignment_mode=str(variant.get("assignment_mode", "default")),
            ),
        )
        before_slots = {slot.slot_id: slot for slot in [*system.state.core_slots, *system.state.residual_slots]}
        system.observe_observation(observation, timestamp="2026-04-14T00:00:00Z")
        after_slots = {slot.slot_id: slot for slot in [*system.state.core_slots, *system.state.residual_slots]}
        result = system.query(sample_id or "update_eval", paired_belief["query"])
        gold_pairs = _belief_pairs(paired_belief["target_belief_json"])
        propagation_values.append(float(_belief_pairs(result.belief_state) == gold_pairs))

        unaffected_ids = [slot_id for slot_id in before_slots if slot_id not in set(affected_slot_ids)]
        unchanged = 0
        for slot_id in unaffected_ids:
            before = before_slots[slot_id]
            after = after_slots.get(slot_id)
            if after is None:
                continue
            if (
                before.active_flag == after.active_flag
                and before.bank == after.bank
                and before.canonical_gloss == after.canonical_gloss
            ):
                unchanged += 1
        locality_score = 1.0 if not unaffected_ids else _safe_divide(unchanged, len(unaffected_ids))
        locality_values.append(locality_score)
        edit_locality_values.append(locality_score)

    budget_summary: dict[int, dict[str, float]] = {}
    for budget in requested_budgets:
        scoped = [row for row in budget_rows if row["budget"] == budget]
        budget_summary[budget] = {
            "joint_belief_accuracy": _mean([row["joint_belief_accuracy"] for row in scoped]),
            "belief_item_recall": _mean([row["belief_item_recall"] for row in scoped]),
            "support_slot_recall": _mean([row["support_slot_recall"] for row in scoped]),
        }

    metrics = {
        "parser_replay_coverage": _mean(parser_coverage),
        "parser_replay_exact": _mean(parser_exact),
        "slot_relation_accuracy": _mean(slot_relation_hits),
        "slot_value_recoverability": _mean(slot_value_hits),
        "recall_at_k": _mean(retrieval_hits),
        "ndcg_at_k": _mean(ndcg_values),
        "joint_belief_accuracy": _mean(belief_joint),
        "slot_value_f1": _mean(belief_f1),
        "support_slot_recall": _mean(support_recall),
        "update_action_accuracy": _mean(update_action),
        "update_flag_accuracy": _mean(update_flags),
        "update_success": _mean([(action + flag) / 2.0 for action, flag in zip(update_action, update_flags)]),
        "propagation_accuracy": _mean(propagation_values),
        "specificity_locality": _mean(locality_values),
        "compression_fidelity": _mean([item["belief_item_recall"] for item in budget_rows]),
        "edit_locality_at_slot_level": _mean(edit_locality_values),
        "answer_exact_match": _mean(answer_exact),
        "avg_composed_token_count": _mean(composed_token_counts),
    }

    family_scores = {
        "parser_slot_family": {
            "parser_replay_coverage": metrics["parser_replay_coverage"],
            "parser_replay_exact": metrics["parser_replay_exact"],
            "slot_relation_accuracy": metrics["slot_relation_accuracy"],
            "slot_value_recoverability": metrics["slot_value_recoverability"],
        },
        "retrieval_family": {
            "recall_at_k": metrics["recall_at_k"],
            "ndcg_at_k": metrics["ndcg_at_k"],
            "support_slot_recall": metrics["support_slot_recall"],
        },
        "belief_recovery_family": {
            "joint_belief_accuracy": metrics["joint_belief_accuracy"],
            "slot_value_f1": metrics["slot_value_f1"],
            "answer_exact_match": metrics["answer_exact_match"],
        },
        "update_propagation_family": {
            "update_action_accuracy": metrics["update_action_accuracy"],
            "update_flag_accuracy": metrics["update_flag_accuracy"],
            "update_success": metrics["update_success"],
            "propagation_accuracy": metrics["propagation_accuracy"],
        },
        "specificity_locality_family": {
            "specificity_locality": metrics["specificity_locality"],
            "edit_locality_at_slot_level": metrics["edit_locality_at_slot_level"],
        },
        "compression_family": {
            "compression_fidelity": metrics["compression_fidelity"],
            "avg_composed_token_count": metrics["avg_composed_token_count"],
        },
    }

    module_scores = {
        "parser": {
            "parser_replay_coverage": metrics["parser_replay_coverage"],
            "parser_replay_exact": metrics["parser_replay_exact"],
        },
        "slot_encoder": {
            "slot_relation_accuracy": metrics["slot_relation_accuracy"],
            "slot_value_recoverability": metrics["slot_value_recoverability"],
        },
        "query_encoder": {
            "recall_at_k": metrics["recall_at_k"],
            "ndcg_at_k": metrics["ndcg_at_k"],
        },
        "retrieval_key_head": {
            "recall_at_k": metrics["recall_at_k"],
            "ndcg_at_k": metrics["ndcg_at_k"],
            "support_slot_recall": metrics["support_slot_recall"],
        },
        "lifecycle": {
            "update_action_accuracy": metrics["update_action_accuracy"],
            "update_flag_accuracy": metrics["update_flag_accuracy"],
            "update_success": metrics["update_success"],
        },
        "consolidation": {
            "specificity_locality": metrics["specificity_locality"],
            "edit_locality_at_slot_level": metrics["edit_locality_at_slot_level"],
        },
        "core_residual_memory": {
            "compression_fidelity": metrics["compression_fidelity"],
            "edit_locality_at_slot_level": metrics["edit_locality_at_slot_level"],
        },
        "resampler": {
            "joint_belief_accuracy": metrics["joint_belief_accuracy"],
            "slot_value_f1": metrics["slot_value_f1"],
            "avg_composed_token_count": metrics["avg_composed_token_count"],
        },
        "belief_decoder": {
            "joint_belief_accuracy": metrics["joint_belief_accuracy"],
            "slot_value_f1": metrics["slot_value_f1"],
            "propagation_accuracy": metrics["propagation_accuracy"],
        },
        "answer_projection": {
            "answer_exact_match": metrics["answer_exact_match"],
        },
    }

    per_dataset: dict[str, dict[str, Any]] = {}
    for dataset in sorted({_dataset_of(row) for row in [*slot_rows, *retrieval_rows, *belief_rows, *lifecycle_rows]}):
        if not dataset:
            continue
        dataset_retrieval = [row for row in retrieval_rows if _dataset_of(row) == dataset]
        dataset_belief = [row for row in belief_rows if _dataset_of(row) == dataset]
        dataset_update = [row for row in lifecycle_rows if _dataset_of(row) == dataset]
        per_dataset[dataset] = {
            "counts": {
                "slot_autoencoding": sum(1 for row in slot_rows if _dataset_of(row) == dataset),
                "retrieval_alignment": len(dataset_retrieval),
                "lifecycle_prediction": len(dataset_update),
                "composition_to_belief": len(dataset_belief),
            }
        }

    summary = {
        "prepared_manifest": str(prepared_manifest_path),
        "selected_datasets": sorted(selected_datasets) if selected_datasets else [],
        "variant": variant,
        "top_k": top_k,
        "budgets": requested_budgets,
        "task_counts": {
            "slot_autoencoding": len(slot_rows),
            "retrieval_alignment": len(retrieval_rows),
            "lifecycle_prediction": len(lifecycle_rows),
            "composition_to_belief": len(belief_rows),
        },
        "metrics": metrics,
        "families": family_scores,
        "modules": module_scores,
        "module_inventory": module_inventory_payload(),
        "per_dataset": per_dataset,
        "budget_sweep": {str(key): value for key, value in budget_summary.items()},
    }
    return {
        "summary": summary,
        "budget_rows": budget_rows,
        "module_inventory": module_inventory_payload(),
    }
