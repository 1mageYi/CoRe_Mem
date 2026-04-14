from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core_mem.v2 import BeliefItem, BeliefState, Observation, SlotRecord, SoftRoleScores


def test_observation_round_trip_normalizes_relation():
    observation = Observation(
        obs_id="obs_1",
        source_dataset="synthetic",
        source_dialogue_id="dlg_1",
        source_turn_id="turn_1",
        session_id="sess_1",
        speaker="user",
        entity="user",
        relation="drink",
        value="matcha",
        value_type="preference",
        time_scope="current",
        status_hint="active",
        polarity="positive",
        confidence=0.9,
        evidence_text="I like matcha.",
        canonical_gloss="drink_preference=matcha",
    )
    restored = Observation.from_dict(observation.to_dict())
    assert restored.relation == "drink_preference"
    assert restored.value == "matcha"


def test_slot_and_belief_round_trip_keeps_support_ids():
    slot = SlotRecord(
        slot_id="slot_1",
        bank="core",
        entity="user",
        relation="music",
        retrieval_key=[0.1, 0.2],
        latent_tokens=[[0.1, 0.2]],
        soft_role_scores=SoftRoleScores(preference=0.8, stable=0.6),
        confidence=0.85,
        first_seen_ts="2026-04-07T00:00:00Z",
        last_update_ts="2026-04-07T00:00:00Z",
        revision_count=1,
        active_flag=True,
        revision_parent=None,
        canonical_gloss="music_preference=jazz",
    )
    belief = BeliefState(
        query_id="query_1",
        entity="user",
        query_type="single_fact",
        belief_items=[
            BeliefItem(
                relation="music",
                value="jazz",
                status="active",
                time_scope="current",
                confidence=0.88,
                support_slot_ids=[slot.slot_id],
            )
        ],
        global_consistency="high",
    )
    restored_slot = SlotRecord.from_dict(slot.to_dict())
    restored_belief = BeliefState.from_dict(belief.to_dict())
    assert restored_slot.relation == "music_preference"
    assert restored_belief.belief_items[0].support_slot_ids == ["slot_1"]
