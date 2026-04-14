from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core_mem.v2.system import StructuredMemorySystem


def test_structured_memory_system_answers_from_active_slot():
    system = StructuredMemorySystem()
    system.observe_turn(
        "I like matcha latte.",
        source_dataset="synthetic",
        source_dialogue_id="dlg-1",
        source_turn_id="turn-1",
        session_id="sess-1",
        timestamp="2026-04-07T05:00:00Z",
    )

    result = system.query("query-1", "What does the user currently like to drink?")
    assert result.belief_state.belief_items[0].relation == "drink_preference"
    assert result.belief_state.belief_items[0].value == "matcha latte"
    assert result.selected_slots
    assert result.composed_memory
    assert result.answer_text == "matcha latte"
    assert "drink_preference" in result.evidence_block


def test_structured_memory_system_overwrite_marks_old_slot_inactive():
    system = StructuredMemorySystem()
    system.observe_turn(
        "I like coffee.",
        source_dataset="synthetic",
        source_dialogue_id="dlg-1",
        source_turn_id="turn-1",
        session_id="sess-1",
        timestamp="2026-04-07T05:00:00Z",
    )
    system.observe_turn(
        "Now I prefer matcha.",
        source_dataset="synthetic",
        source_dialogue_id="dlg-1",
        source_turn_id="turn-2",
        session_id="sess-1",
        timestamp="2026-04-07T05:05:00Z",
    )

    result = system.query("query-2", "What does the user now like to drink?")
    active_values = [item.value for item in result.belief_state.belief_items]
    all_slots = [*system.state.core_slots, *system.state.residual_slots]
    assert "matcha" in active_values
    assert any((not slot.active_flag) and "coffee" in slot.canonical_gloss for slot in all_slots)
    assert result.answer_text == "matcha"
    assert any(slot.bank == "core" and slot.active_flag and "matcha" in slot.canonical_gloss for slot in all_slots)
