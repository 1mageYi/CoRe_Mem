from __future__ import annotations

import torch

from core_mem.v2.schemas import Observation, SlotRecord, SoftRoleScores
from core_mem.v2.v61_learned_memory import (
    _best_matching_slot,
    _read_with_model,
    filter_v61_observations,
    typed_observation,
)


def _observation(**overrides: object) -> Observation:
    payload = {
        "obs_id": "obs-1",
        "source_dataset": "unit",
        "source_dialogue_id": "dialog-1",
        "source_turn_id": "3",
        "session_id": "dialog-1",
        "speaker": "user",
        "entity": "user",
        "relation": "other_fact",
        "value": "I changed plans because the feedback felt rushed.",
        "value_type": "other",
        "time_scope": "recent_change",
        "status_hint": "active",
        "polarity": "neutral",
        "confidence": 0.9,
        "evidence_text": "I changed plans because the feedback felt rushed.",
        "canonical_gloss": "other_fact=I changed plans because the feedback felt rushed.",
    }
    payload.update(overrides)
    return Observation.from_dict(payload)


def _slot(slot_id: str, relation: str, canonical_gloss: str, *, bank: str = "residual") -> SlotRecord:
    return SlotRecord(
        slot_id=slot_id,
        bank=bank,
        entity="user",
        relation=relation,
        retrieval_key=[0.1] * 8,
        latent_tokens=[[0.1] * 8 for _ in range(8)],
        soft_role_scores=SoftRoleScores(
            stable=0.2,
            preference=0.95 if relation.endswith("preference") else 0.05,
            constraint=0.0,
            goal=0.0,
            temporal=0.9 if relation == "reason_fact" else 0.1,
            social=0.0,
        ),
        confidence=0.8,
        first_seen_ts="turn-00001-obs-001",
        last_update_ts="turn-00001-obs-001",
        revision_count=0,
        active_flag=True,
        revision_parent=None,
        canonical_gloss=canonical_gloss,
    )


def test_typed_observation_promotes_reason_fact() -> None:
    observation = _observation(
        value="I paused because the feedback felt rushed.",
        time_scope="current",
        canonical_gloss="other_fact=I paused because the feedback felt rushed.",
    )

    typed = typed_observation(observation)

    assert typed.relation == "reason_fact"
    assert typed.canonical_gloss.startswith("reason_fact=")


def test_best_matching_slot_prefers_same_relation_family_and_overlap() -> None:
    observation = _observation(
        relation="music_preference",
        value="music with emotional storytelling",
        value_type="preference",
        time_scope="current",
        canonical_gloss="music_preference=music with emotional storytelling",
    )
    slots = [
        _slot("slot-profile", "profile_trait", "profile_trait=structured and analytical", bank="residual"),
        _slot("slot-music-a", "music_preference", "music_preference=music with emotional storytelling", bank="core"),
        _slot("slot-music-b", "music_preference", "music_preference=instrumental ambient loops", bank="core"),
    ]

    matched = _best_matching_slot(observation, slots)

    assert matched is not None
    assert matched.slot_id == "slot-music-a"


def test_filter_v61_observations_removes_low_information_goal_and_hobby_values() -> None:
    kept = _observation(
        obs_id="obs-keep",
        relation="goal",
        value="avoid technical jargon",
        value_type="goal",
        time_scope="current",
        canonical_gloss="goal=avoid technical jargon",
    )
    dropped_goal = _observation(
        obs_id="obs-drop-goal",
        relation="goal",
        value="check out",
        value_type="goal",
        time_scope="future",
        canonical_gloss="goal=check out",
    )
    dropped_hobby = _observation(
        obs_id="obs-drop-hobby",
        relation="hobby",
        value="myself",
        value_type="preference",
        time_scope="current",
        canonical_gloss="hobby=myself",
    )

    filtered = filter_v61_observations([kept, dropped_goal, dropped_hobby])

    assert [item.obs_id for item in filtered] == ["obs-keep"]


class _DummyReader:
    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        return torch.tensor([4.0, 3.0, -1.5], dtype=torch.float32)


def test_read_with_model_pools_same_relation_beliefs() -> None:
    slots = [
        _slot("slot-music-a", "music_preference", "music_preference=music with emotional storytelling", bank="core"),
        _slot("slot-music-b", "music_preference", "music_preference=gentle acoustic live sessions", bank="residual"),
        _slot("slot-profile", "profile_trait", "profile_trait=reflective and analytical", bank="residual"),
    ]

    readout = _read_with_model(
        "What music would the user most likely prefer right now?",
        slots=slots,
        reader=_DummyReader(),
        top_k=3,
    )

    assert len(readout["belief_items"]) == 2
    top_belief = readout["belief_items"][0]
    assert top_belief["relation"] == "music_preference"
    assert top_belief["support_count"] == 2
    assert top_belief["support_slot_ids"] == ["slot-music-a", "slot-music-b"]
    assert "emotional storytelling" in top_belief["value"]
    assert "gentle acoustic live sessions" in top_belief["value"]
    assert len(readout["composed_key"]) == len(slots[0].retrieval_key)
