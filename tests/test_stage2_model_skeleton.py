from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core_mem.v2.system import StructuredMemoryState, StructuredMemorySystem
from core_mem.v2.schemas import Observation


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


def test_structured_memory_system_encoder_and_composed_memory_shapes_are_stable():
    system = StructuredMemorySystem()
    system.observe_turn(
        "I like matcha latte.",
        source_dataset="synthetic",
        source_dialogue_id="dlg-1",
        source_turn_id="turn-1",
        session_id="sess-1",
        timestamp="2026-04-07T05:00:00Z",
    )

    slot = [*system.state.core_slots, *system.state.residual_slots][0]
    query_vector = system.query_encoder.encode("What drink does the user like now?")
    composed = system.resampler.compose(query_vector, [slot])
    belief = system.decoder.decode("query-shape", "What drink does the user like now?", [slot], composed_memory=composed)

    assert len(query_vector) == system.query_encoder.dimension
    assert len(slot.retrieval_key) == system.slot_encoder.config.retrieval_dim
    assert len(slot.latent_tokens) == system.slot_encoder.config.slot_tokens
    assert all(len(token) == system.slot_encoder.config.token_dim for token in slot.latent_tokens)
    assert len(composed) == system.resampler.latent_queries
    assert all(len(token) == system.resampler.token_dim for token in composed)
    assert belief.belief_items[0].support_slot_ids == [slot.slot_id]


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


def test_structured_memory_system_keeps_distinct_music_facets_active():
    system = StructuredMemorySystem()
    system.observe_turn(
        "I like producing music with software.",
        source_dataset="synthetic",
        source_dialogue_id="dlg-1",
        source_turn_id="turn-1",
        session_id="sess-1",
        timestamp="2026-04-07T05:00:00Z",
    )
    system.observe_turn(
        "I prefer unique blends of electronic Pacific music.",
        source_dataset="synthetic",
        source_dialogue_id="dlg-1",
        source_turn_id="turn-2",
        session_id="sess-1",
        timestamp="2026-04-07T05:05:00Z",
    )

    active_music_slots = [
        slot for slot in [*system.state.core_slots, *system.state.residual_slots] if slot.active_flag and slot.relation == "music_preference"
    ]
    assert len(active_music_slots) >= 2


def test_belief_decoder_preserves_selected_slot_order():
    system = StructuredMemorySystem()
    system.observe_turn(
        "I enjoy telling jokes on stage.",
        source_dataset="synthetic",
        source_dialogue_id="dlg-1",
        source_turn_id="turn-1",
        session_id="sess-1",
        timestamp="2026-04-07T05:00:00Z",
    )
    system.observe_turn(
        "I enjoy quiet nights at home.",
        source_dataset="synthetic",
        source_dialogue_id="dlg-1",
        source_turn_id="turn-2",
        session_id="sess-1",
        timestamp="2026-04-07T05:05:00Z",
    )

    slots = list(reversed(system.state.active_slots()))
    composed = system.resampler.compose(system.query_encoder.encode("What is the latest fact about the user?"), slots)
    belief = system.decoder.decode(
        "query-ordered",
        "What is the latest fact about the user?",
        slots,
        composed_memory=composed,
    )
    assert belief.belief_items[0].value == "quiet nights at home"


def test_structured_memory_system_prefers_query_lexical_overlap_for_other_facts():
    system = StructuredMemorySystem()
    system.observe_turn(
        "I've actually been using Down Dog for my home practice, especially on days when I can't make it to Serenity Yoga.",
        source_dataset="synthetic",
        source_dialogue_id="dlg-1",
        source_turn_id="turn-1",
        session_id="sess-1",
        timestamp="2026-04-07T05:00:00Z",
    )
    system.observe_turn(
        "I am excited to recreate a vegan quinoa salad I tried at a cafe.",
        source_dataset="synthetic",
        source_dialogue_id="dlg-1",
        source_turn_id="turn-2",
        session_id="sess-1",
        timestamp="2026-04-07T05:05:00Z",
    )

    result = system.query("query-lexical", "What is the name of the yoga studio where I take classes?")
    assert result.selected_slots[0].canonical_gloss.endswith("=serenity yoga")
    assert result.answer_text == "serenity yoga"


def test_structured_memory_system_can_switch_to_learned_memory_belief_predictor():
    def _predict(query_id: str, query_text: str, slots):
        assert query_id == "query-learned"
        assert "drink" in query_text
        assert slots
        return {
            "query_id": query_id,
            "query_type": "single_fact",
            "global_consistency": "high",
            "belief_items": [
                {
                    "relation": "drink_preference",
                    "value": "oolong tea",
                    "support_slot_ids": [slots[0].slot_id],
                }
            ],
        }

    system = StructuredMemorySystem(memory_mode="learned_memory", use_learned_memory=True, learned_belief_predictor=_predict)
    system.observe_turn(
        "I like matcha latte.",
        source_dataset="synthetic",
        source_dialogue_id="dlg-1",
        source_turn_id="turn-1",
        session_id="sess-1",
        timestamp="2026-04-07T05:00:00Z",
    )

    result = system.query("query-learned", "What drink does the user like?")
    assert result.belief_source == "learned_memory"
    assert result.belief_state.belief_items[0].value == "oolong tea"
    assert result.answer_text == "oolong tea"


def test_structured_memory_system_does_not_fallback_to_symbolic_in_learned_mode():
    def _raise(*_args, **_kwargs):
        raise RuntimeError("learned predictor unavailable")

    system = StructuredMemorySystem(
        memory_mode="learned_memory",
        use_learned_memory=True,
        learned_belief_predictor=_raise,
    )
    system.observe_turn(
        "I like matcha latte.",
        source_dataset="synthetic",
        source_dialogue_id="dlg-1",
        source_turn_id="turn-1",
        session_id="sess-1",
        timestamp="2026-04-07T05:00:00Z",
    )

    result = system.query("query-learned-error", "What drink does the user like?")
    assert result.belief_source == "learned_memory_error"
    assert result.belief_state.belief_items == []
    assert result.answer_text == "unknown"


def test_structured_memory_system_can_switch_to_learned_slot_assignment():
    calls = {"count": 0}

    def _predict(_observation, _slots):
        calls["count"] += 1
        return {"target_action": "ignore", "target_flags": {"promote": False, "stale_old": False}}

    system = StructuredMemorySystem(
        slot_assignment_mode="learned",
        use_learned_slot_assignment=True,
        learned_slot_assignment_predictor=_predict,
    )
    boston = Observation.from_dict(
        {
            "obs_id": "obs-1",
            "source_dataset": "synthetic",
            "source_dialogue_id": "dlg-1",
            "source_turn_id": "turn-1",
            "session_id": "sess-1",
            "speaker": "user",
            "entity": "user",
            "relation": "location",
            "value": "Boston",
            "value_type": "location",
            "time_scope": "current",
            "status_hint": "active",
            "polarity": "neutral",
            "confidence": 1.0,
            "evidence_text": "I live in Boston.",
            "canonical_gloss": "location=Boston",
        }
    )
    chicago = Observation.from_dict(
        {
            "obs_id": "obs-2",
            "source_dataset": "synthetic",
            "source_dialogue_id": "dlg-1",
            "source_turn_id": "turn-2",
            "session_id": "sess-1",
            "speaker": "user",
            "entity": "user",
            "relation": "location",
            "value": "Chicago",
            "value_type": "location",
            "time_scope": "current",
            "status_hint": "active",
            "polarity": "neutral",
            "confidence": 1.0,
            "evidence_text": "I moved to Chicago.",
            "canonical_gloss": "location=Chicago",
        }
    )
    system.state = StructuredMemoryState(
        residual_slots=[
            system.slot_encoder.encode(boston, timestamp="2026-04-07T05:00:00Z", bank="residual"),
            system.slot_encoder.encode(chicago, timestamp="2026-04-07T05:02:00Z", bank="residual"),
        ]
    )
    system.observe_observation(
        Observation.from_dict(
            {
                "obs_id": "obs-3",
                "source_dataset": "synthetic",
                "source_dialogue_id": "dlg-1",
                "source_turn_id": "turn-3",
                "session_id": "sess-1",
                "speaker": "user",
                "entity": "user",
                "relation": "location",
                "value": "Seattle",
                "value_type": "location",
                "time_scope": "current",
                "status_hint": "active",
                "polarity": "neutral",
                "confidence": 1.0,
                "evidence_text": "Now I live in Seattle.",
                "canonical_gloss": "location=Seattle",
            }
        ),
        timestamp="2026-04-07T05:05:00Z",
    )

    assert calls["count"] == 1
    assert len(system.state.active_slots()) == 2


def test_structured_memory_system_short_circuits_single_candidate_overwrite():
    calls = {"count": 0}

    def _predict(_observation, _slots):
        calls["count"] += 1
        return {"target_action": "overwrite", "target_flags": {"promote": False, "stale_old": True}}

    system = StructuredMemorySystem(
        slot_assignment_mode="learned",
        use_learned_slot_assignment=True,
        learned_slot_assignment_predictor=_predict,
    )
    system.observe_observation(
        Observation.from_dict(
            {
                "obs_id": "obs-1",
                "source_dataset": "synthetic",
                "source_dialogue_id": "dlg-1",
                "source_turn_id": "turn-1",
                "session_id": "sess-1",
                "speaker": "user",
                "entity": "user",
                "relation": "location",
                "value": "Boston",
                "value_type": "location",
                "time_scope": "current",
                "status_hint": "active",
                "polarity": "neutral",
                "confidence": 1.0,
                "evidence_text": "I live in Boston.",
                "canonical_gloss": "location=Boston",
            }
        ),
        timestamp="2026-04-07T05:00:00Z",
    )
    system.observe_observation(
        Observation.from_dict(
            {
                "obs_id": "obs-2",
                "source_dataset": "synthetic",
                "source_dialogue_id": "dlg-1",
                "source_turn_id": "turn-2",
                "session_id": "sess-1",
                "speaker": "user",
                "entity": "user",
                "relation": "location",
                "value": "Seattle",
                "value_type": "location",
                "time_scope": "current",
                "status_hint": "active",
                "polarity": "neutral",
                "confidence": 1.0,
                "evidence_text": "Now I live in Seattle.",
                "canonical_gloss": "location=Seattle",
            }
        ),
        timestamp="2026-04-07T05:05:00Z",
    )

    result = system.query("query-slot-assignment", "Where does the user live?")
    assert calls["count"] == 0
    assert result.answer_text == "Seattle"


def test_structured_memory_system_short_circuits_slot_assignment_without_candidates():
    calls = {"count": 0}

    def _predict(_observation, _slots):
        calls["count"] += 1
        return {"target_action": "ignore", "target_flags": {"promote": False, "stale_old": False}}

    system = StructuredMemorySystem(
        slot_assignment_mode="learned",
        use_learned_slot_assignment=True,
        learned_slot_assignment_predictor=_predict,
    )
    system.observe_turn(
        "I graduated with a degree in Business Administration.",
        source_dataset="synthetic",
        source_dialogue_id="dlg-1",
        source_turn_id="turn-1",
        session_id="sess-1",
        timestamp="2026-04-07T05:00:00Z",
    )

    assert calls["count"] == 0
    assert len(system.state.active_slots()) == 1
    assert "business administration" in system.state.active_slots()[0].canonical_gloss.lower()


def test_structured_memory_system_short_circuits_other_fact_slot_assignment_with_candidates():
    calls = {"count": 0}

    def _predict(_observation, _slots):
        calls["count"] += 1
        return {"target_action": "ignore", "target_flags": {"promote": False, "stale_old": False}}

    system = StructuredMemorySystem(
        slot_assignment_mode="learned",
        use_learned_slot_assignment=True,
        learned_slot_assignment_predictor=_predict,
    )
    system.observe_observation(
        Observation.from_dict(
            {
                "obs_id": "obs-1",
                "source_dataset": "synthetic",
                "source_dialogue_id": "dlg-1",
                "source_turn_id": "turn-1",
                "session_id": "sess-1",
                "speaker": "user",
                "entity": "user",
                "relation": "other_fact",
                "value": "curious about yoga poses for sleep",
                "value_type": "other",
                "time_scope": "current",
                "status_hint": "active",
                "polarity": "neutral",
                "confidence": 1.0,
                "evidence_text": "I am curious about yoga poses for sleep.",
                "canonical_gloss": "other_fact=curious about yoga poses for sleep",
            }
        ),
        timestamp="2026-04-07T05:00:00Z",
    )
    system.observe_observation(
        Observation.from_dict(
            {
                "obs_id": "obs-2",
                "source_dataset": "synthetic",
                "source_dialogue_id": "dlg-1",
                "source_turn_id": "turn-2",
                "session_id": "sess-1",
                "speaker": "user",
                "entity": "user",
                "relation": "other_fact",
                "value": "experimenting with essential oils",
                "value_type": "other",
                "time_scope": "current",
                "status_hint": "active",
                "polarity": "neutral",
                "confidence": 1.0,
                "evidence_text": "I am experimenting with essential oils.",
                "canonical_gloss": "other_fact=experimenting with essential oils",
            }
        ),
        timestamp="2026-04-07T05:05:00Z",
    )

    assert calls["count"] == 0
    assert len(system.state.active_slots()) == 2


def test_structured_memory_system_slot_assignment_prompt_uses_explicit_schema():
    system = StructuredMemorySystem()
    observation = system.parser.parse_turn(
        "I graduated with a degree in Business Administration.",
        source_dataset="synthetic",
        source_dialogue_id="dlg-1",
        source_turn_id="turn-1",
        session_id="sess-1",
        speaker="user",
    )[0]

    prompt = system._render_slot_assignment_example(
        observation,
        [],
        lambda payload: payload,
        lambda payload: payload,
    )

    assert "Choose exactly one target_action from [merge, overwrite, new, ignore]." in prompt
    assert 'Return JSON only with the schema {"target_action":"new","target_flags":{"promote":false,"stale_old":false}}.' in prompt


def test_structured_memory_system_learned_mode_repairs_braceless_belief_payload():
    def _predict(_query_id: str, _query_text: str, slots):
        return '"belief_items": ["relation": "drink_preference", "support_slot_ids": ["%s"], "value": "oolong tea"]' % slots[0].slot_id

    system = StructuredMemorySystem(
        memory_mode="learned_memory",
        use_learned_memory=True,
        learned_belief_predictor=_predict,
    )
    system.observe_turn(
        "I like matcha latte.",
        source_dataset="synthetic",
        source_dialogue_id="dlg-1",
        source_turn_id="turn-1",
        session_id="sess-1",
        timestamp="2026-04-07T05:00:00Z",
    )

    result = system.query("query-learned-semantic", "What drink does the user like?")
    assert result.belief_source == "learned_memory"
    assert result.belief_state.belief_items[0].relation == "drink_preference"
    assert result.belief_state.belief_items[0].value == "oolong tea"


def test_answer_projection_trims_explanatory_suffixes_for_single_fact_queries():
    def _predict(query_id: str, _query_text: str, _slots):
        return {
            "query_id": query_id,
            "query_type": "single_fact",
            "belief_items": [
                {
                    "relation": "paint_color",
                    "value": "a lighter shade of gray - it's made the room feel so much brighter",
                    "support_slot_ids": [],
                }
            ],
        }

    system = StructuredMemorySystem(memory_mode="learned_memory", use_learned_memory=True, learned_belief_predictor=_predict)
    system.observe_turn(
        "I repainted my bedroom walls a lighter shade of gray.",
        source_dataset="synthetic",
        source_dialogue_id="dlg-1",
        source_turn_id="turn-1",
        session_id="sess-1",
        timestamp="2026-04-07T05:00:00Z",
    )

    result = system.query("query-projection-color", "What color did I repaint my bedroom walls?")
    assert result.answer_text == "a lighter shade of gray"


def test_answer_projection_extracts_location_phrase_for_where_queries():
    def _predict(query_id: str, _query_text: str, _slots):
        return {
            "query_id": query_id,
            "query_type": "single_fact",
            "belief_items": [
                {
                    "relation": "location",
                    "value": "I got my new tennis racket from a sports store downtown.",
                    "support_slot_ids": [],
                }
            ],
        }

    system = StructuredMemorySystem(memory_mode="learned_memory", use_learned_memory=True, learned_belief_predictor=_predict)
    system.observe_turn(
        "I got my new tennis racket from a sports store downtown.",
        source_dataset="synthetic",
        source_dialogue_id="dlg-1",
        source_turn_id="turn-1",
        session_id="sess-1",
        timestamp="2026-04-07T05:00:00Z",
    )

    result = system.query("query-projection-where", "Where did I buy my new tennis racket from?")
    assert result.answer_text == "a sports store downtown"


def test_learned_belief_example_uses_compact_slot_view():
    def _compact(slots):
        return slots

    system = StructuredMemorySystem()
    system.observe_turn(
        "I like matcha latte.",
        source_dataset="synthetic",
        source_dialogue_id="dlg-1",
        source_turn_id="turn-1",
        session_id="sess-1",
        timestamp="2026-04-07T05:00:00Z",
    )
    slot = [*system.state.core_slots, *system.state.residual_slots][0]
    rendered = system._render_learned_belief_example("What drink does the user like?", [slot], _compact)
    assert "Semantic correctness matters more than raw JSON surface matching." in rendered
    assert "latent_tokens" in rendered

    from core_mem.v2.training import compact_slot_list

    compact_rendered = system._render_learned_belief_example(
        "What drink does the user like?",
        [slot],
        compact_slot_list,
    )
    assert "latent_tokens" not in compact_rendered
    assert "canonical_gloss" in compact_rendered
