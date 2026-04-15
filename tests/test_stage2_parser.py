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
    assert all(item.relation != "food_preference" for item in observations)
