from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core_mem.v2 import Stage2ObservationParser


def test_stage2_parser_extracts_preference_and_location():
    parser = Stage2ObservationParser()
    observations = parser.parse_turn(
        "I like matcha latte and I live in Boston.",
        source_dataset="synthetic",
        source_dialogue_id="dlg",
        source_turn_id="turn",
        session_id="sess",
    )
    assert len(observations) == 2
    assert observations[0].relation == "drink_preference"
    assert observations[0].value == "matcha latte"
    assert observations[1].relation == "location"
    assert observations[1].value == "boston"


def test_stage2_parser_marks_used_to_as_past():
    parser = Stage2ObservationParser()
    observations = parser.parse_turn(
        "I used to drink coffee.",
        source_dataset="synthetic",
        source_dialogue_id="dlg",
        source_turn_id="turn",
        session_id="sess",
    )
    assert len(observations) == 1
    assert observations[0].time_scope == "past"
    assert observations[0].status_hint == "stale"


def test_stage2_parser_ignores_assistant_turns():
    parser = Stage2ObservationParser()
    observations = parser.parse_turn(
        "Assistant: You should keep creating music because it suits you.",
        source_dataset="synthetic",
        source_dialogue_id="dlg",
        source_turn_id="turn",
        session_id="sess",
        speaker="assistant",
    )
    assert observations == []


def test_stage2_parser_does_not_treat_create_as_food_signal():
    parser = Stage2ObservationParser()
    observations = parser.parse_turn(
        "I enjoy creating digital music remixes.",
        source_dataset="synthetic",
        source_dialogue_id="dlg",
        source_turn_id="turn",
        session_id="sess",
    )
    assert observations
    assert observations[0].relation == "music_preference"
    assert observations[0].value == "producing music with software"
    assert all(item.relation != "food_preference" for item in observations)


def test_stage2_parser_extracts_music_technology_preference_from_software_context():
    parser = Stage2ObservationParser()
    observations = parser.parse_turn(
        "I started creating digital music remixes with new software tools.",
        source_dataset="synthetic",
        source_dialogue_id="dlg",
        source_turn_id="turn",
        session_id="sess",
    )

    assert len(observations) == 1
    assert observations[0].relation == "music_preference"
    assert observations[0].value == "producing music with software"


def test_stage2_parser_does_not_treat_from_phrase_inside_music_clause_as_location():
    parser = Stage2ObservationParser()
    observations = parser.parse_turn(
        "Every aspect of this process has been fascinating, from selecting the right software to getting the nuances of mixing different sounds to create something unique.",
        source_dataset="synthetic",
        source_dialogue_id="dlg",
        source_turn_id="turn",
        session_id="sess",
    )

    assert observations
    assert any(item.relation == "music_preference" for item in observations)
    assert all(item.relation != "location" for item in observations)


def test_stage2_parser_extracts_feedback_reason_clause():
    parser = Stage2ObservationParser()
    observations = parser.parse_turn(
        "I recorded a new episode after receiving some great feedback from my peers about my last podcast.",
        source_dataset="synthetic",
        source_dialogue_id="dlg",
        source_turn_id="turn",
        session_id="sess",
    )

    assert observations
    assert any(item.value == "getting positive feedback from my peers about my last podcast" for item in observations)


def test_stage2_parser_extracts_step_back_withdrawal_clause():
    parser = Stage2ObservationParser()
    observations = parser.parse_turn(
        "However, I've decided to step back from structured book club settings.",
        source_dataset="synthetic",
        source_dialogue_id="dlg",
        source_turn_id="turn",
        session_id="sess",
    )

    assert len(observations) == 1
    assert observations[0].relation == "other_fact"
    assert observations[0].value == "step back from structured book club settings"


def test_stage2_parser_extracts_opted_out_withdrawal_clause():
    parser = Stage2ObservationParser()
    observations = parser.parse_turn(
        "I also opted out of cooking classes that I once enjoyed.",
        source_dataset="synthetic",
        source_dialogue_id="dlg",
        source_turn_id="turn",
        session_id="sess",
    )

    assert len(observations) == 1
    assert observations[0].relation == "other_fact"
    assert observations[0].value == "opted out of cooking classes that i once enjoyed"


def test_stage2_parser_extracts_store_location_from_shopping_turn():
    parser = Stage2ObservationParser()
    observations = parser.parse_turn(
        "I shop at Target pretty frequently, maybe every other week.",
        source_dataset="synthetic",
        source_dialogue_id="dlg",
        source_turn_id="turn",
        session_id="sess",
    )

    assert len(observations) == 1
    assert observations[0].relation == "location"
    assert observations[0].value == "target"


def test_stage2_parser_uses_context_for_coupon_redemption_event():
    parser = Stage2ObservationParser()
    observations = parser.parse_turn(
        "I actually redeemed a $5 coupon on coffee creamer last Sunday.",
        source_dataset="synthetic",
        source_dialogue_id="dlg",
        source_turn_id="turn",
        session_id="sess",
        context_text="I have been using the Cartwheel app from Target for household items.",
    )

    assert len(observations) == 1
    assert observations[0].relation == "episodic_event"
    assert observations[0].value == "redeemed a $5 coupon on coffee creamer last sunday at target"
