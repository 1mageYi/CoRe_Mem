from __future__ import annotations

from core_mem.v2.schemas import Observation
from core_mem.v2.v62_write_memory import (
    DialogueTurn,
    estimate_state_cleanliness,
    observation_attribute_valid_label,
    turn_text_features,
)
from core_mem.v2.v6_persistent_memory import PersistentCoreResidualMemory


def _observation(**overrides: object) -> Observation:
    payload = {
        "obs_id": "obs_test",
        "source_dataset": "unit",
        "source_dialogue_id": "dialogue",
        "source_turn_id": "0",
        "session_id": "dialogue",
        "speaker": "user",
        "entity": "user",
        "relation": "profile_trait",
        "value": "about me",
        "value_type": "other",
        "time_scope": "current",
        "status_hint": "active",
        "polarity": "neutral",
        "confidence": 0.8,
        "evidence_text": "I am thinking about me.",
        "canonical_gloss": "profile_trait=about me",
        "metadata": {},
    }
    payload.update(overrides)
    return Observation.from_dict(payload)


def test_observation_attribute_validity_rejects_low_information_profile() -> None:
    assert observation_attribute_valid_label(_observation()) is False


def test_turn_text_features_encode_user_message_shape() -> None:
    features = turn_text_features(
        DialogueTurn(
            source_dialogue_id="dialogue",
            source_turn_id="1",
            session_id="dialogue",
            speaker="user",
            text="I recently decided to step back from crowded festivals.",
            context_text="",
        )
    )

    assert len(features) > 12
    assert features[12] == 1.0


def test_estimate_state_cleanliness_flags_invalid_slots() -> None:
    memory = PersistentCoreResidualMemory()
    low_info = _observation(
        relation="profile_trait",
        value="about me",
        canonical_gloss="profile_trait=about me",
        evidence_text="I am about me.",
    )
    memory.write(low_info, "new_residual", turn_index=0, obs_index=0)

    cleanliness = estimate_state_cleanliness(memory)

    assert cleanliness["invalid_slot_rate"] > 0.0
    assert cleanliness["bank_precision_estimate"] < 1.0
