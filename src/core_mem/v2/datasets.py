"""Stage-2 dataset/task registry and prepared payload builders."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from core_mem.v2.schemas import BeliefItem, BeliefState, Observation, SlotRecord, SoftRoleScores

STAGE2_DATA_POOLS = {
    "state_belief": ["SGD", "MultiWOZ 2.4"],
    "persona_preference": ["Persona-Chat", "Training Millions of Personalized Dialogue Agents"],
    "update_propagation": ["MQUAKE", "ReCoE", "SGD-derived state changes", "MultiWOZ-derived state changes"],
    "minimal_synthetic": ["overwrite stress", "budget stress", "conflict bundles", "multi-fact composition"],
}

STAGE2_TASK_DATASETS = {
    "slot_autoencoding": ["SGD", "MultiWOZ 2.4", "Persona-Chat", "Training Millions of Personalized Dialogue Agents"],
    "retrieval_alignment": [
        "Persona-Chat",
        "Training Millions of Personalized Dialogue Agents",
        "SGD",
        "MultiWOZ 2.4",
        "MQUAKE",
        "ReCoE",
    ],
    "lifecycle_prediction": ["MQUAKE", "ReCoE", "SGD-derived state changes", "MultiWOZ-derived state changes", "Minimal Synthetic Pool"],
    "composition_to_belief": [
        "SGD",
        "MultiWOZ 2.4",
        "Persona-Chat",
        "Training Millions of Personalized Dialogue Agents",
        "MQUAKE",
        "ReCoE",
        "Minimal Synthetic Pool",
    ],
}

STAGE2_TASK_MODULES = {
    "slot_autoencoding": ["slot_encoder", "slot_projection", "reconstruction_path"],
    "retrieval_alignment": ["query_encoder", "retrieval_key_head", "retrieval_scorer"],
    "lifecycle_prediction": ["lifecycle_head", "overwrite_logic", "stale_promotion_flags"],
    "composition_to_belief": ["resampler", "decoder_adapter", "belief_reconstruction_path"],
}

STAGE2_PRIMARY_DATASETS = {
    "sgd": "SGD",
    "multiwoz24": "MultiWOZ 2.4",
    "personachat": "Persona-Chat",
    "mquake": "MQUAKE",
    "recoe": "ReCoE",
}


@dataclass(frozen=True)
class TaskRegistryEntry:
    task_name: str
    datasets: list[str]
    modules: list[str]
    primary_metrics: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_name": self.task_name,
            "datasets": list(self.datasets),
            "modules": list(self.modules),
            "primary_metrics": list(self.primary_metrics),
        }


@dataclass(frozen=True)
class SourceRecordRef:
    dataset_key: str
    dataset: str
    sample_id: str
    has_lifecycle: bool
    relation: str
    parser_bucket: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_key": self.dataset_key,
            "dataset": self.dataset,
            "sample_id": self.sample_id,
            "has_lifecycle": self.has_lifecycle,
            "relation": self.relation,
            "parser_bucket": self.parser_bucket,
        }


def build_task_registry() -> list[TaskRegistryEntry]:
    return [
        TaskRegistryEntry(
            task_name="slot_autoencoding",
            datasets=STAGE2_TASK_DATASETS["slot_autoencoding"],
            modules=STAGE2_TASK_MODULES["slot_autoencoding"],
            primary_metrics=["reconstruction_quality", "slot_usability"],
        ),
        TaskRegistryEntry(
            task_name="retrieval_alignment",
            datasets=STAGE2_TASK_DATASETS["retrieval_alignment"],
            modules=STAGE2_TASK_MODULES["retrieval_alignment"],
            primary_metrics=["recall_at_k", "ndcg_at_k"],
        ),
        TaskRegistryEntry(
            task_name="lifecycle_prediction",
            datasets=STAGE2_TASK_DATASETS["lifecycle_prediction"],
            modules=STAGE2_TASK_MODULES["lifecycle_prediction"],
            primary_metrics=["update_success", "edit_locality_at_slot_level"],
        ),
        TaskRegistryEntry(
            task_name="composition_to_belief",
            datasets=STAGE2_TASK_DATASETS["composition_to_belief"],
            modules=STAGE2_TASK_MODULES["composition_to_belief"],
            primary_metrics=["joint_belief_accuracy", "slot_value_f1", "propagation_accuracy"],
        ),
    ]


def build_demo_prepared_payload() -> dict[str, list[dict[str, Any]]]:
    observation = Observation(
        obs_id="obs_demo_matcha",
        source_dataset="synthetic",
        source_dialogue_id="dialog_demo_001",
        source_turn_id="turn_001",
        session_id="session_demo_001",
        speaker="user",
        entity="user",
        relation="drink_preference",
        value="matcha",
        value_type="preference",
        time_scope="current",
        status_hint="active",
        polarity="positive",
        confidence=0.95,
        evidence_text="I like matcha latte.",
        canonical_gloss="drink_preference=matcha",
        metadata={"pool": "minimal_synthetic"},
    )
    positive_slot = SlotRecord(
        slot_id="slot_matcha_current",
        bank="residual",
        entity="user",
        relation="drink_preference",
        retrieval_key=[0.1, 0.9, 0.0],
        latent_tokens=[[0.1, 0.4, 0.7], [0.2, 0.5, 0.8]],
        soft_role_scores=SoftRoleScores(stable=0.2, preference=0.9),
        confidence=0.94,
        first_seen_ts="2026-04-07T00:00:00Z",
        last_update_ts="2026-04-07T00:00:00Z",
        revision_count=0,
        active_flag=True,
        revision_parent=None,
        canonical_gloss="current drink preference is matcha",
    )
    negative_slot = SlotRecord(
        slot_id="slot_coffee_stale",
        bank="core",
        entity="user",
        relation="drink_preference",
        retrieval_key=[0.6, 0.2, 0.1],
        latent_tokens=[[0.7, 0.1, 0.1]],
        soft_role_scores=SoftRoleScores(stable=0.7, preference=0.6, temporal=0.3),
        confidence=0.81,
        first_seen_ts="2025-12-07T00:00:00Z",
        last_update_ts="2026-03-07T00:00:00Z",
        revision_count=1,
        active_flag=False,
        revision_parent="slot_matcha_current",
        canonical_gloss="past drink preference was coffee",
    )
    belief = BeliefState(
        query_id="query_demo_drink",
        entity="user",
        query_type="single_fact",
        belief_items=[
            BeliefItem(
                relation="drink_preference",
                value="matcha",
                status="active",
                time_scope="current",
                confidence=0.96,
                support_slot_ids=[positive_slot.slot_id],
            )
        ],
        global_consistency="high",
    )
    return {
        "slot_autoencoding": [
            {
                "input_observation": observation.to_dict(),
                "target_record": {
                    "entity": "user",
                    "relation": "drink_preference",
                    "value": "matcha",
                    "time_scope": "current",
                    "status": "active",
                },
            }
        ],
        "retrieval_alignment": [
            {
                "query": "What does the user currently like to drink?",
                "positive_slot": positive_slot.to_dict(),
                "negative_slots": [negative_slot.to_dict()],
                "gold_support_slot_ids": [positive_slot.slot_id],
            }
        ],
        "lifecycle_prediction": [
            {
                "memory_context": [negative_slot.to_dict()],
                "new_observation": observation.to_dict(),
                "target_action": "overwrite",
                "target_flags": {"promote": False, "stale_old": True},
                "affected_slot_ids": [negative_slot.slot_id],
            }
        ],
        "composition_to_belief": [
            {
                "query": "What is the user's current drink preference?",
                "memory_slots": [positive_slot.to_dict(), negative_slot.to_dict()],
                "target_belief_json": belief.to_dict(),
                "retrieved_slot_ids": [positive_slot.slot_id, negative_slot.slot_id],
            }
        ],
    }


def load_stage2_source_config(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Stage-2 source config must be a mapping.")
    return payload


def summarize_stage2_sources(path: str | Path) -> dict[str, Any]:
    payload = load_stage2_source_config(path)
    datasets = payload.get("datasets", {})
    summary = {"enabled": [], "missing": [], "available": []}
    for dataset_key, spec in datasets.items():
        if not spec.get("enabled", True):
            continue
        dataset_path = Path(spec["path"])
        summary["enabled"].append(dataset_key)
        if dataset_path.exists():
            summary["available"].append(dataset_key)
        else:
            summary["missing"].append(dataset_key)
    return summary


def _record_relation(record: dict[str, Any]) -> str:
    tasks = record.get("tasks", {}) or {}
    slot_payload = tasks.get("slot_autoencoding") or {}
    target_record = slot_payload.get("target_record") or {}
    if isinstance(target_record, dict) and target_record.get("relation"):
        return str(target_record["relation"])

    input_observation = slot_payload.get("input_observation") or {}
    if isinstance(input_observation, dict) and input_observation.get("relation"):
        return str(input_observation["relation"])

    lifecycle_payload = tasks.get("lifecycle_prediction") or {}
    new_observation = lifecycle_payload.get("new_observation") or {}
    if isinstance(new_observation, dict) and new_observation.get("relation"):
        return str(new_observation["relation"])

    belief_payload = tasks.get("composition_to_belief") or {}
    target_belief = belief_payload.get("target_belief_json") or {}
    belief_items = target_belief.get("belief_items", []) if isinstance(target_belief, dict) else []
    for item in belief_items:
        if isinstance(item, dict) and item.get("relation"):
            return str(item["relation"])

    retrieval_payload = tasks.get("retrieval_alignment") or {}
    positive_slot = retrieval_payload.get("positive_slot") or {}
    if isinstance(positive_slot, dict) and positive_slot.get("relation"):
        return str(positive_slot["relation"])
    return "unknown"


def _record_parser_bucket(record: dict[str, Any]) -> str:
    tasks = record.get("tasks", {}) or {}
    slot_payload = tasks.get("slot_autoencoding") or {}
    input_observation = slot_payload.get("input_observation") or {}
    metadata = input_observation.get("metadata", {}) if isinstance(input_observation, dict) else {}
    if isinstance(metadata, dict):
        for key in ("parser_source", "parser_bucket", "source_type", "normalizer"):
            value = metadata.get(key)
            if value:
                return str(value)
    return "unknown"


def collect_stage2_source_records(path: str | Path) -> list[SourceRecordRef]:
    payload = load_stage2_source_config(path)
    datasets = payload.get("datasets", {})
    records: list[SourceRecordRef] = []
    for dataset_key, spec in datasets.items():
        if not spec.get("enabled", True):
            continue
        dataset_path = Path(spec["path"])
        if not dataset_path.exists():
            continue
        with dataset_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                tasks = record.get("tasks", {}) or {}
                records.append(
                    SourceRecordRef(
                        dataset_key=str(dataset_key),
                        dataset=str(record.get("dataset", dataset_key)),
                        sample_id=str(record.get("sample_id", "")),
                        has_lifecycle="lifecycle_prediction" in tasks,
                        relation=_record_relation(record),
                        parser_bucket=_record_parser_bucket(record),
                    )
                )
    return records


def _allocate_split_quota(available_counts: dict[tuple[str, bool], int], target_total: int) -> dict[tuple[str, bool], int]:
    total_available = sum(available_counts.values())
    if total_available < target_total:
        raise ValueError(
            f"Requested split of {target_total} source rows but only {total_available} rows remain."
        )
    quotas = {key: int(target_total * count / total_available) for key, count in available_counts.items()}
    assigned = sum(quotas.values())
    if assigned < target_total:
        remainders = sorted(
            (
                (target_total * count / total_available) - quotas[key],
                key,
            )
            for key, count in available_counts.items()
        )
        for _, key in reversed(remainders):
            if assigned >= target_total:
                break
            if quotas[key] >= available_counts[key]:
                continue
            quotas[key] += 1
            assigned += 1
    return quotas


def build_source_record_split_from_sources(
    path: str | Path,
    *,
    split_counts: dict[str, int],
    split_seed: int = 27,
) -> dict[str, Any]:
    records = collect_stage2_source_records(path)
    total_requested = sum(split_counts.values())
    if len(records) < total_requested:
        raise ValueError(
            f"Requested {total_requested} source rows but only {len(records)} rows are available."
        )

    grouped: dict[tuple[str, bool], list[SourceRecordRef]] = defaultdict(list)
    for record in records:
        grouped[(record.dataset_key, record.has_lifecycle)].append(record)

    for items in grouped.values():
        items.sort(
            key=lambda item: hashlib.sha1(
                f"{split_seed}:{item.dataset_key}:{item.sample_id}".encode("utf-8")
            ).hexdigest()
        )

    offsets = {key: 0 for key in grouped}
    split_rows: dict[str, list[dict[str, Any]]] = {}
    for split_name, target_count in split_counts.items():
        available_counts = {
            key: len(items) - offsets[key]
            for key, items in grouped.items()
            if len(items) - offsets[key] > 0
        }
        quotas = _allocate_split_quota(available_counts, target_count)
        selected: list[dict[str, Any]] = []
        for key in sorted(quotas):
            count = quotas[key]
            if count <= 0:
                continue
            start = offsets[key]
            end = start + count
            selected.extend(item.to_dict() for item in grouped[key][start:end])
            offsets[key] = end
        split_rows[split_name] = selected

    return {
        "split_seed": split_seed,
        "requested_counts": {name: int(count) for name, count in split_counts.items()},
        "available_source_rows": len(records),
        "split_rows": split_rows,
        "available_by_dataset": {
            dataset_key: sum(1 for item in records if item.dataset_key == dataset_key)
            for dataset_key in sorted({item.dataset_key for item in records})
        },
    }


def build_prepared_payload_from_sources(
    path: str | Path,
    *,
    max_rows_per_dataset: int | None = None,
    selected_sample_ids: dict[str, set[str]] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    payload = load_stage2_source_config(path)
    datasets = payload.get("datasets", {})
    prepared = {
        "slot_autoencoding": [],
        "retrieval_alignment": [],
        "lifecycle_prediction": [],
        "composition_to_belief": [],
    }
    for dataset_key, spec in datasets.items():
        if not spec.get("enabled", True):
            continue
        dataset_path = Path(spec["path"])
        if not dataset_path.exists():
            continue
        seen_rows = 0
        with dataset_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                sample_id = str(record.get("sample_id", ""))
                selected_ids = (selected_sample_ids or {}).get(str(dataset_key))
                if selected_ids is not None and sample_id not in selected_ids:
                    continue
                for task_name, task_payload in record.get("tasks", {}).items():
                    if task_name in prepared:
                        payload = dict(task_payload)
                        payload["_meta"] = {
                            "dataset_key": dataset_key,
                            "dataset": str(record.get("dataset", dataset_key)),
                            "sample_id": sample_id,
                            "task_name": task_name,
                        }
                        prepared[task_name].append(payload)
                seen_rows += 1
                if max_rows_per_dataset is not None and seen_rows >= max_rows_per_dataset:
                    break
    return prepared
