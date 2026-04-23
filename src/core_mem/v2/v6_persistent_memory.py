"""Persistent core/residual memory substrate for stage-2 v6."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import hashlib
import math
import re
from typing import Any, Iterable

import torch
from torch import nn

from core_mem.v2.encoder import QueryEncoder, SlotEncoder, _lexical_features
from core_mem.v2.schemas import Observation, SlotRecord


V6_WRITE_ACTIONS = (
    "new_core",
    "new_residual",
    "merge_core",
    "merge_residual",
    "overwrite",
    "mark_stale",
    "promote_to_core",
    "ignore",
)

_ACTION_TO_BANK = {
    "new_core": "core",
    "merge_core": "core",
    "promote_to_core": "core",
    "new_residual": "residual",
    "merge_residual": "residual",
    "overwrite": "residual",
    "mark_stale": "residual",
    "ignore": "residual",
}


def stable_timestamp(turn_index: int, obs_index: int) -> str:
    return f"turn-{turn_index:05d}-obs-{obs_index:03d}"


def vector_dot(left: list[float], right: list[float]) -> float:
    return float(sum(a * b for a, b in zip(left, right)))


def mean_vector(vectors: Iterable[list[float]]) -> list[float]:
    rows = [row for row in vectors if row]
    if not rows:
        return []
    dim = len(rows[0])
    merged = [sum(row[idx] for row in rows) / len(rows) for idx in range(dim)]
    norm = math.sqrt(sum(value * value for value in merged)) or 1.0
    return [value / norm for value in merged]


def observation_features(observation: Observation) -> list[float]:
    text = f"{observation.relation} {observation.value} {observation.canonical_gloss}"
    relation = observation.relation
    value_type = observation.value_type
    return [
        *_lexical_features(text),
        float(value_type == "preference" or relation.endswith("preference") or relation == "hobby"),
        float(value_type == "profile" or relation in {"occupation", "education_degree", "location"}),
        float(value_type == "constraint" or relation == "constraint"),
        float(value_type == "goal" or relation == "goal"),
        float(observation.time_scope == "recent_change"),
        float(observation.time_scope == "past"),
        float(observation.status_hint == "stale"),
        float(observation.polarity == "negative"),
        float(observation.confidence),
    ]


def silver_action_for_observation(observation: Observation) -> str:
    relation = observation.relation
    durable = (
        observation.value_type in {"preference", "profile", "occupation", "location"}
        or relation.endswith("preference")
        or relation in {"hobby", "occupation", "education_degree", "location"}
    )
    if observation.status_hint == "stale" or observation.time_scope == "past":
        return "mark_stale"
    if observation.time_scope == "recent_change" or observation.polarity == "negative":
        return "overwrite"
    if durable and observation.confidence >= 0.78:
        return "promote_to_core"
    if durable:
        return "new_core"
    return "new_residual"


class V6WriteRouter(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, len(V6_WRITE_ACTIONS)),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


@dataclass(frozen=True)
class V6RouterTrainingResult:
    router: V6WriteRouter
    action_accuracy: float
    disabled_controller_accuracy: float
    train_examples: int
    eval_examples: int
    learned_update_actions: list[str]
    loss_curve: list[dict[str, float]]


def train_write_router(
    observations: list[Observation],
    *,
    epochs: int = 80,
    learning_rate: float = 0.03,
    seed: int = 606,
) -> V6RouterTrainingResult:
    if len(observations) < 8:
        raise ValueError("v6 write-router training requires at least 8 observations.")
    torch.manual_seed(seed)
    features = torch.tensor([observation_features(obs) for obs in observations], dtype=torch.float32)
    labels = torch.tensor([V6_WRITE_ACTIONS.index(silver_action_for_observation(obs)) for obs in observations], dtype=torch.long)
    split = max(4, int(len(observations) * 0.8))
    split = min(split, len(observations) - 2)
    train_x, eval_x = features[:split], features[split:]
    train_y, eval_y = labels[:split], labels[split:]
    router = V6WriteRouter(input_dim=features.shape[1])
    optimizer = torch.optim.AdamW(router.parameters(), lr=learning_rate, weight_decay=1e-4)
    loss_curve: list[dict[str, float]] = []
    for epoch in range(epochs):
        logits = router(train_x)
        loss = nn.functional.cross_entropy(logits, train_y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if epoch in {0, epochs - 1}:
            loss_curve.append({"epoch": float(epoch + 1), "loss": float(loss.item())})
    with torch.no_grad():
        predictions = router(eval_x).argmax(dim=1)
    action_accuracy = float((predictions == eval_y).float().mean().item())
    counts = Counter(int(item) for item in train_y.tolist())
    majority = counts.most_common(1)[0][0]
    disabled = torch.full_like(eval_y, majority)
    disabled_accuracy = float((disabled == eval_y).float().mean().item())
    learned_update_actions = sorted({V6_WRITE_ACTIONS[int(item)] for item in router(features).argmax(dim=1).tolist()})
    return V6RouterTrainingResult(
        router=router,
        action_accuracy=action_accuracy,
        disabled_controller_accuracy=disabled_accuracy,
        train_examples=len(train_y),
        eval_examples=len(eval_y),
        learned_update_actions=learned_update_actions,
        loss_curve=loss_curve,
    )


def route_observation(router: V6WriteRouter, observation: Observation) -> str:
    features = torch.tensor([observation_features(observation)], dtype=torch.float32)
    with torch.no_grad():
        action = int(router(features).argmax(dim=1).item())
    return V6_WRITE_ACTIONS[action]


def relation_family(relation: str) -> str:
    if relation.endswith("_preference"):
        return "preference"
    return relation


@dataclass
class PersistentCoreResidualMemory:
    slot_encoder: SlotEncoder = field(default_factory=SlotEncoder)
    core_bank: list[SlotRecord] = field(default_factory=list)
    residual_bank: list[SlotRecord] = field(default_factory=list)
    write_trace: list[dict[str, Any]] = field(default_factory=list)

    def active_slots(self, *, context_id: str | None = None, max_turn_index: int | None = None) -> list[SlotRecord]:
        rows: list[SlotRecord] = []
        for slot in [*self.core_bank, *self.residual_bank]:
            if not slot.active_flag:
                continue
            metadata = self._slot_metadata(slot)
            if context_id is not None and metadata.get("source_dialogue_id") != context_id:
                continue
            if max_turn_index is not None and int(metadata.get("source_turn_index", -1)) > max_turn_index:
                continue
            rows.append(slot)
        return rows

    def write(self, observation: Observation, action: str, *, turn_index: int, obs_index: int) -> SlotRecord | None:
        if action not in V6_WRITE_ACTIONS:
            raise ValueError(f"unknown v6 write action: {action}")
        timestamp = stable_timestamp(turn_index, obs_index)
        target_bank = _ACTION_TO_BANK[action]
        before_core = len([slot for slot in self.core_bank if slot.active_flag])
        before_residual = len([slot for slot in self.residual_bank if slot.active_flag])
        matched = self._find_match(observation, bank=target_bank)
        written: SlotRecord | None = None
        if action == "ignore":
            written = None
        elif action == "mark_stale":
            if matched is not None:
                written = self._replace_slot(matched, active=False)
        elif action == "promote_to_core":
            if matched is not None and matched.bank == "residual":
                self._replace_slot(matched, active=False)
            written = self._append_slot(observation, bank="core", timestamp=timestamp, revision_parent=matched.slot_id if matched else None)
        elif action in {"merge_core", "merge_residual", "overwrite"} and matched is not None:
            self._replace_slot(matched, active=False)
            written = self._append_slot(
                observation,
                bank=target_bank,
                timestamp=timestamp,
                existing_slot=matched,
                revision_parent=matched.slot_id,
            )
        else:
            written = self._append_slot(observation, bank=target_bank, timestamp=timestamp)
        after_core = len([slot for slot in self.core_bank if slot.active_flag])
        after_residual = len([slot for slot in self.residual_bank if slot.active_flag])
        self.write_trace.append(
            {
                "obs_id": observation.obs_id,
                "source_dialogue_id": observation.source_dialogue_id,
                "source_turn_id": observation.source_turn_id,
                "source_turn_index": turn_index,
                "action": action,
                "bank": target_bank,
                "slot_id": written.slot_id if written is not None else None,
                "revision_parent": written.revision_parent if written is not None else (matched.slot_id if matched else None),
                "core_bank_size_before": before_core,
                "residual_bank_size_before": before_residual,
                "core_bank_size_after": after_core,
                "residual_bank_size_after": after_residual,
            }
        )
        return written

    def read(self, query: str, *, slots: list[SlotRecord] | None = None, top_k: int = 8) -> dict[str, Any]:
        query_encoder = QueryEncoder(dimension=self.slot_encoder.config.retrieval_dim)
        query_key = query_encoder.encode(query)
        candidates = slots if slots is not None else self.active_slots()
        scored = sorted(
            ((vector_dot(query_key, slot.retrieval_key), slot) for slot in candidates),
            key=lambda item: item[0],
            reverse=True,
        )
        selected = scored[:top_k]
        composed = mean_vector(slot.retrieval_key for _, slot in selected)
        return {
            "query_key": query_key,
            "selected": [{"score": score, "slot": slot} for score, slot in selected],
            "composed_key": composed,
            "belief_items": [
                {
                    "relation": slot.relation,
                    "value": slot.canonical_gloss.split("=", 1)[-1],
                    "support_slot_id": slot.slot_id,
                    "confidence": slot.confidence,
                    "bank": slot.bank,
                }
                for _, slot in selected[:3]
            ],
        }

    def to_checkpoint(self) -> dict[str, Any]:
        return {
            "format": "stage2_v6_persistent_core_residual_state_v1",
            "core_bank": [self._slot_with_metadata(slot) for slot in self.core_bank],
            "residual_bank": [self._slot_with_metadata(slot) for slot in self.residual_bank],
            "write_trace_length": len(self.write_trace),
        }

    def _append_slot(
        self,
        observation: Observation,
        *,
        bank: str,
        timestamp: str,
        existing_slot: SlotRecord | None = None,
        revision_parent: str | None = None,
    ) -> SlotRecord:
        slot = self.slot_encoder.encode(
            observation,
            timestamp=timestamp,
            existing_slot=existing_slot,
            bank=bank,
            revision_parent=revision_parent,
        )
        slot = self._attach_metadata(slot, observation)
        if bank == "core":
            self.core_bank.append(slot)
        else:
            self.residual_bank.append(slot)
        return slot

    def _find_match(self, observation: Observation, *, bank: str) -> SlotRecord | None:
        candidates = self.core_bank if bank == "core" else self.residual_bank
        family = relation_family(observation.relation)
        source_dialogue_id = observation.source_dialogue_id
        for slot in reversed(candidates):
            if not slot.active_flag:
                continue
            metadata = self._slot_metadata(slot)
            if metadata.get("source_dialogue_id") != source_dialogue_id:
                continue
            if relation_family(slot.relation) == family:
                return slot
        return None

    def _replace_slot(self, slot: SlotRecord, *, active: bool) -> SlotRecord:
        payload = slot.to_dict()
        payload["active_flag"] = active
        updated = SlotRecord.from_dict(payload)
        target = self.core_bank if slot.bank == "core" else self.residual_bank
        for idx, item in enumerate(target):
            if item.slot_id == slot.slot_id:
                target[idx] = updated
                break
        return updated

    @staticmethod
    def _slot_metadata(slot: SlotRecord) -> dict[str, Any]:
        try:
            return dict(getattr(slot, "_v6_metadata"))  # type: ignore[arg-type]
        except AttributeError:
            payload = slot.to_dict()
            return dict(payload.get("metadata", {}) or {})

    @staticmethod
    def _attach_metadata(slot: SlotRecord, observation: Observation) -> SlotRecord:
        metadata = {
            "source_dataset": observation.source_dataset,
            "source_dialogue_id": observation.source_dialogue_id,
            "source_turn_id": observation.source_turn_id,
            "source_turn_index": int(observation.source_turn_id) if str(observation.source_turn_id).isdigit() else -1,
            "obs_id": observation.obs_id,
        }
        object.__setattr__(slot, "_v6_metadata", metadata)
        return slot

    def _slot_with_metadata(self, slot: SlotRecord) -> dict[str, Any]:
        payload = slot.to_dict()
        payload["metadata"] = self._slot_metadata(slot)
        return payload


def option_label(option: str) -> str:
    match = re.match(r"\s*(\([a-d]\))", option)
    return match.group(1) if match else option


def score_options_from_persistent_read(readout: dict[str, Any], options: list[str]) -> tuple[int, list[float]]:
    composed = readout.get("composed_key") or []
    if not composed:
        return 0, [0.0 for _ in options]
    option_encoder = QueryEncoder(dimension=len(composed))
    belief_text = " ".join(f"{item['relation']} {item['value']}" for item in readout.get("belief_items", []))
    scores: list[float] = []
    for option in options:
        option_key = option_encoder.encode(option)
        semantic_score = vector_dot(composed, option_key)
        lexical_score = _jaccard(belief_text, option)
        scores.append(0.75 * semantic_score + 0.25 * lexical_score)
    best = max(range(len(scores)), key=lambda idx: scores[idx]) if scores else 0
    return best, scores


def _jaccard(left: str, right: str) -> float:
    left_tokens = set(re.findall(r"[a-z0-9']+", left.lower()))
    right_tokens = set(re.findall(r"[a-z0-9']+", right.lower()))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def stable_slot_fingerprint(slots: Iterable[SlotRecord]) -> str:
    payload = "|".join(sorted(f"{slot.slot_id}:{slot.bank}:{slot.active_flag}" for slot in slots))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
