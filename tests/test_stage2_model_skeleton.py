from pathlib import Path
import sys
from unittest.mock import patch


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


def test_structured_memory_system_keeps_recent_change_other_fact_distinct():
    system = StructuredMemorySystem()
    system.observe_observation(
        Observation.from_dict(
            {
                "obs_id": "obs-other-fact-current",
                "source_dataset": "synthetic",
                "source_dialogue_id": "dlg-1",
                "source_turn_id": "turn-1",
                "session_id": "sess-1",
                "speaker": "user",
                "entity": "user",
                "relation": "other_fact",
                "value": "assisting with literacy programs",
                "value_type": "other",
                "time_scope": "current",
                "status_hint": "active",
                "polarity": "positive",
                "confidence": 0.8,
                "evidence_text": "I am assisting with literacy programs.",
                "canonical_gloss": "other_fact=assisting with literacy programs",
            }
        ),
        timestamp="2026-04-07T05:00:00Z",
    )
    system.observe_observation(
        Observation.from_dict(
            {
                "obs_id": "obs-other-fact-recent-change",
                "source_dataset": "synthetic",
                "source_dialogue_id": "dlg-1",
                "source_turn_id": "turn-2",
                "session_id": "sess-1",
                "speaker": "user",
                "entity": "user",
                "relation": "other_fact",
                "value": "step back from structured book club settings",
                "value_type": "other",
                "time_scope": "recent_change",
                "status_hint": "active",
                "polarity": "negative",
                "confidence": 0.8,
                "evidence_text": "I've decided to step back from structured book club settings.",
                "canonical_gloss": "other_fact=step back from structured book club settings",
            }
        ),
        timestamp="2026-04-07T05:05:00Z",
    )

    active_other_fact_slots = [
        slot for slot in [*system.state.core_slots, *system.state.residual_slots] if slot.active_flag and slot.relation == "other_fact"
    ]
    assert len(active_other_fact_slots) >= 2
    assert any("assisting with literacy programs" in slot.canonical_gloss for slot in active_other_fact_slots)
    assert any("step back from structured book club settings" in slot.canonical_gloss for slot in active_other_fact_slots)


def test_structured_memory_system_prefers_recent_change_negative_slot_for_advice_queries():
    def _latent_ranker(_query_text: str, slots):
        scores = {}
        for slot in slots:
            if "assisting with literacy programs" in slot.canonical_gloss:
                scores[slot.slot_id] = 1.0452
            elif "step back from structured book club settings" in slot.canonical_gloss:
                scores[slot.slot_id] = 1.0328
            elif "future events" in slot.canonical_gloss:
                scores[slot.slot_id] = 0.9789
            else:
                scores[slot.slot_id] = 0.5
        return scores

    system = StructuredMemorySystem(latent_slot_ranker=_latent_ranker)
    system.observe_observation(
        Observation.from_dict(
            {
                "obs_id": "obs-advice-events",
                "source_dataset": "synthetic",
                "source_dialogue_id": "dlg-1",
                "source_turn_id": "turn-1",
                "session_id": "sess-1",
                "speaker": "user",
                "entity": "user",
                "relation": "other_fact",
                "value": "particularly thrilled about the prospect of showcasing these items at our future events",
                "value_type": "other",
                "time_scope": "current",
                "status_hint": "active",
                "polarity": "positive",
                "confidence": 0.8,
                "evidence_text": "I feel thrilled about showcasing these items at future events.",
                "canonical_gloss": "other_fact=particularly thrilled about the prospect of showcasing these items at our future events",
            }
        ),
        timestamp="2026-04-07T05:00:00Z",
    )
    system.observe_observation(
        Observation.from_dict(
            {
                "obs_id": "obs-advice-literacy",
                "source_dataset": "synthetic",
                "source_dialogue_id": "dlg-1",
                "source_turn_id": "turn-2",
                "session_id": "sess-1",
                "speaker": "user",
                "entity": "user",
                "relation": "other_fact",
                "value": "assisting with literacy programs",
                "value_type": "other",
                "time_scope": "current",
                "status_hint": "active",
                "polarity": "positive",
                "confidence": 0.8,
                "evidence_text": "I am assisting with literacy programs.",
                "canonical_gloss": "other_fact=assisting with literacy programs",
            }
        ),
        timestamp="2026-04-07T05:01:00Z",
    )
    system.observe_observation(
        Observation.from_dict(
            {
                "obs_id": "obs-advice-step-back",
                "source_dataset": "synthetic",
                "source_dialogue_id": "dlg-1",
                "source_turn_id": "turn-3",
                "session_id": "sess-1",
                "speaker": "user",
                "entity": "user",
                "relation": "other_fact",
                "value": "step back from structured book club settings",
                "value_type": "other",
                "time_scope": "recent_change",
                "status_hint": "active",
                "polarity": "negative",
                "confidence": 0.8,
                "evidence_text": "I've decided to step back from structured book club settings.",
                "canonical_gloss": "other_fact=step back from structured book club settings",
            }
        ),
        timestamp="2026-04-07T05:02:00Z",
    )

    result = system.query(
        "query-advice",
        "I've been involved in planning events for my community lately, but I'm not sure if I should continue with it. What do you think?",
    )
    selected_glosses = [slot.canonical_gloss for slot in result.selected_slots]
    assert any("step back from structured book club settings" in gloss for gloss in selected_glosses[:3])


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


def test_structured_memory_system_can_prioritize_slots_with_latent_ranker():
    def _latent_ranker(_query_text: str, slots):
        return {
            slot.slot_id: (10.0 if "other_fact=more descriptive comments" in slot.canonical_gloss else 0.0)
            for slot in slots
        }

    system = StructuredMemorySystem(latent_slot_ranker=_latent_ranker)
    system.observe_observation(
        Observation.from_dict(
            {
                "obs_id": "obs-latent-ranker-recipes",
                "source_dataset": "synthetic",
                "source_dialogue_id": "dlg-1",
                "source_turn_id": "turn-1",
                "session_id": "sess-1",
                "speaker": "user",
                "entity": "user",
                "relation": "hobby",
                "value": "experimenting with different recipes",
                "value_type": "other",
                "time_scope": "current",
                "status_hint": "active",
                "polarity": "positive",
                "confidence": 0.75,
                "evidence_text": "I enjoy experimenting with different recipes.",
                "canonical_gloss": "hobby=experimenting with different recipes",
            }
        ),
        timestamp="2026-04-07T05:00:00Z",
    )
    system.observe_observation(
        Observation.from_dict(
            {
                "obs_id": "obs-latent-ranker-comments",
                "source_dataset": "synthetic",
                "source_dialogue_id": "dlg-1",
                "source_turn_id": "turn-2",
                "session_id": "sess-1",
                "speaker": "user",
                "entity": "user",
                "relation": "other_fact",
                "value": "more descriptive comments that highlight the purpose of code blocks",
                "value_type": "other",
                "time_scope": "current",
                "status_hint": "active",
                "polarity": "neutral",
                "confidence": 0.95,
                "evidence_text": "I prefer more descriptive comments that highlight the purpose of code blocks.",
                "canonical_gloss": "other_fact=more descriptive comments that highlight the purpose of code blocks",
            }
        ),
        timestamp="2026-04-07T05:01:00Z",
    )

    result = system.query("query-latent-ranker", "What should I remember?")
    assert result.selected_slots[0].canonical_gloss == "other_fact=more descriptive comments that highlight the purpose of code blocks"


def test_structured_memory_system_keeps_query_anchored_slot_ahead_of_zero_overlap_latent_outlier():
    def _latent_ranker(_query_text: str, slots):
        return {
            slot.slot_id: (
                10.0 if "bachelor's degree in business administration" in slot.canonical_gloss else 0.0
            )
            for slot in slots
        }

    system = StructuredMemorySystem(latent_slot_ranker=_latent_ranker)
    system.observe_turn(
        "I earned a bachelor's degree in business administration from the University of Michigan in 2012.",
        source_dataset="synthetic",
        source_dialogue_id="dlg-latent-guard",
        source_turn_id="turn-1",
        session_id="sess-latent-guard",
        timestamp="2026-04-07T05:00:00Z",
    )
    system.observe_turn(
        "I have been using the Cartwheel app from Target for household items.",
        source_dataset="synthetic",
        source_dialogue_id="dlg-latent-guard",
        source_turn_id="turn-2",
        session_id="sess-latent-guard",
        timestamp="2026-04-07T05:05:00Z",
    )
    system.observe_turn(
        "I actually redeemed a $5 coupon on coffee creamer last Sunday.",
        source_dataset="synthetic",
        source_dialogue_id="dlg-latent-guard",
        source_turn_id="turn-3",
        session_id="sess-latent-guard",
        timestamp="2026-04-07T05:10:00Z",
    )

    result = system.query("query-latent-guard", "Where did I redeem a $5 coupon on coffee creamer?")
    assert "redeemed a $5 coupon on coffee creamer" in result.selected_slots[0].canonical_gloss
    assert result.answer_text == "target"


def test_structured_memory_system_prefers_higher_query_score_within_lexical_positive_latent_candidates():
    def _latent_ranker(_query_text: str, slots):
        return {
            slot.slot_id: (10.0 if "local italian restaurants" in slot.canonical_gloss else 0.0)
            for slot in slots
        }

    system = StructuredMemorySystem(latent_slot_ranker=_latent_ranker)
    system.observe_observation(
        Observation.from_dict(
            {
                "obs_id": "obs-latent-play",
                "source_dataset": "synthetic",
                "source_dialogue_id": "dlg-latent-play",
                "source_turn_id": "turn-1",
                "session_id": "sess-latent-play",
                "speaker": "user",
                "entity": "user",
                "relation": "attended_play",
                "value": "The Glass Menagerie at the local community theater",
                "value_type": "event",
                "time_scope": "past",
                "status_hint": "active",
                "polarity": "positive",
                "confidence": 1.0,
                "evidence_text": "I attended a local community theater production of The Glass Menagerie last weekend.",
                "canonical_gloss": "attended_play=the glass menagerie at the local community theater",
            }
        ),
        timestamp="2026-04-07T05:00:00Z",
    )
    system.observe_observation(
        Observation.from_dict(
            {
                "obs_id": "obs-latent-restaurants",
                "source_dataset": "synthetic",
                "source_dialogue_id": "dlg-latent-play",
                "source_turn_id": "turn-2",
                "session_id": "sess-latent-play",
                "speaker": "user",
                "entity": "user",
                "relation": "other_fact",
                "value": "looking for some recommendations for local italian restaurants",
                "value_type": "other",
                "time_scope": "current",
                "status_hint": "active",
                "polarity": "neutral",
                "confidence": 1.0,
                "evidence_text": "I'm looking for some recommendations for local Italian restaurants.",
                "canonical_gloss": "other_fact=looking for some recommendations for local italian restaurants",
            }
        ),
        timestamp="2026-04-07T05:05:00Z",
    )

    result = system.query("query-latent-play", "What play did I attend at the local community theater?")
    assert "glass menagerie" in result.selected_slots[0].canonical_gloss
    assert "glass menagerie" in result.answer_text


def test_structured_memory_system_promotes_concrete_latent_facet_over_weak_same_relation_overlap():
    def _latent_ranker(_query_text: str, slots):
        scores: dict[str, float] = {}
        for slot in slots:
            if "producing music with software" in slot.canonical_gloss:
                scores[slot.slot_id] = 1.13
            elif "unique sound rather than merely reinterpreting" in slot.canonical_gloss:
                scores[slot.slot_id] = 1.07
            else:
                scores[slot.slot_id] = 0.1
        return scores

    system = StructuredMemorySystem(
        memory_mode="learned_memory",
        use_learned_memory=True,
        learned_belief_predictor=lambda *_args, **_kwargs: '"belief_items": ["good"]["good"]',
        latent_slot_ranker=_latent_ranker,
    )
    system.observe_observation(
        Observation.from_dict(
            {
                "obs_id": "obs-music-abstract",
                "source_dataset": "synthetic",
                "source_dialogue_id": "dlg-latent-facet",
                "source_turn_id": "turn-1",
                "session_id": "sess-latent-facet",
                "speaker": "user",
                "entity": "user",
                "relation": "music_preference",
                "value": "committing to invest my time in the creation of original music, focusing on my unique sound rather than merely reinterpreting the pieces of other artists",
                "value_type": "other",
                "time_scope": "current",
                "status_hint": "active",
                "polarity": "positive",
                "confidence": 0.9,
                "evidence_text": "I am committing to invest my time in the creation of original music, focusing on my unique sound rather than merely reinterpreting the pieces of other artists.",
                "canonical_gloss": "music_preference=committing to invest my time in the creation of original music, focusing on my unique sound rather than merely reinterpreting the pieces of other artists",
            }
        ),
        timestamp="2026-04-07T05:00:00Z",
    )
    system.observe_observation(
        Observation.from_dict(
            {
                "obs_id": "obs-music-software",
                "source_dataset": "synthetic",
                "source_dialogue_id": "dlg-latent-facet",
                "source_turn_id": "turn-2",
                "session_id": "sess-latent-facet",
                "speaker": "user",
                "entity": "user",
                "relation": "music_preference",
                "value": "producing music with software",
                "value_type": "other",
                "time_scope": "current",
                "status_hint": "active",
                "polarity": "positive",
                "confidence": 0.9,
                "evidence_text": "I like producing music with software.",
                "canonical_gloss": "music_preference=producing music with software",
            }
        ),
        timestamp="2026-04-07T05:01:00Z",
    )

    result = system.query(
        "query-latent-facet",
        "I recently attended an event where there was a unique blend of modern beats with Pacific sounds.",
    )

    assert result.selected_slots[0].canonical_gloss == "music_preference=producing music with software"
    assert result.belief_state.belief_items[0].relation == "music_preference"
    assert result.belief_state.belief_items[0].value == "producing music with software"


def test_structured_memory_system_uses_recent_dialogue_context_for_coupon_redemption():
    system = StructuredMemorySystem()
    system.observe_turn(
        "I have been using the Cartwheel app from Target for household items.",
        source_dataset="synthetic",
        source_dialogue_id="dlg-1",
        source_turn_id="turn-1",
        session_id="sess-1",
        timestamp="2026-04-07T05:00:00Z",
    )
    system.observe_turn(
        "I actually redeemed a $5 coupon on coffee creamer last Sunday.",
        source_dataset="synthetic",
        source_dialogue_id="dlg-1",
        source_turn_id="turn-2",
        session_id="sess-1",
        timestamp="2026-04-07T05:05:00Z",
    )

    active_glosses = [slot.canonical_gloss for slot in system.state.active_slots()]
    assert any("redeemed a $5 coupon on coffee creamer last sunday at target" in gloss for gloss in active_glosses)

    result = system.query("query-coupon", "Where did I redeem a $5 coupon on coffee creamer?")
    assert result.answer_text == "target"


def test_structured_memory_system_prefers_temporal_slot_for_previous_occupation_query():
    system = StructuredMemorySystem()
    current_observation = Observation.from_dict(
        {
            "obs_id": "obs-current-occupation",
            "source_dataset": "synthetic",
            "source_dialogue_id": "dlg-1",
            "source_turn_id": "turn-1",
            "session_id": "sess-1",
            "speaker": "user",
            "entity": "user",
            "relation": "occupation",
            "value": "lpc associate",
            "value_type": "occupation",
            "time_scope": "current",
            "status_hint": "active",
            "polarity": "neutral",
            "confidence": 1.0,
            "evidence_text": "I work as an LPC associate.",
            "canonical_gloss": "occupation=lpc associate",
        }
    )
    past_observation = Observation.from_dict(
        {
            "obs_id": "obs-past-occupation",
            "source_dataset": "synthetic",
            "source_dialogue_id": "dlg-1",
            "source_turn_id": "turn-2",
            "session_id": "sess-1",
            "speaker": "user",
            "entity": "user",
            "relation": "occupation",
            "value": "marketing specialist at a small startup",
            "value_type": "occupation",
            "time_scope": "past",
            "status_hint": "stale",
            "polarity": "neutral",
            "confidence": 1.0,
            "evidence_text": "I used to work as a marketing specialist at a small startup.",
            "canonical_gloss": "occupation=marketing specialist at a small startup",
        }
    )
    system.state = StructuredMemoryState(
        residual_slots=[
            system.slot_encoder.encode(current_observation, timestamp="2026-04-07T05:00:00Z", bank="residual"),
            system.slot_encoder.encode(past_observation, timestamp="2026-04-07T05:01:00Z", bank="residual"),
        ]
    )

    result = system.query("query-previous-occupation", "What was my previous occupation?")
    assert result.answer_text == "marketing specialist at a small startup"
    assert result.belief_state.belief_items[0].time_scope == "past"


def test_structured_memory_system_prefers_numeric_slot_for_how_much_query():
    system = StructuredMemorySystem()
    price_observation = Observation.from_dict(
        {
            "obs_id": "obs-price",
            "source_dataset": "synthetic",
            "source_dialogue_id": "dlg-1",
            "source_turn_id": "turn-1",
            "session_id": "sess-1",
            "speaker": "user",
            "entity": "user",
            "relation": "other_fact",
            "value": "designer handbag for $800",
            "value_type": "other",
            "time_scope": "current",
            "status_hint": "active",
            "polarity": "neutral",
            "confidence": 1.0,
            "evidence_text": "I recently bought a designer handbag for $800.",
            "canonical_gloss": "other_fact=designer handbag for $800",
        }
    )
    distractor_observation = Observation.from_dict(
        {
            "obs_id": "obs-distractor",
            "source_dataset": "synthetic",
            "source_dialogue_id": "dlg-1",
            "source_turn_id": "turn-2",
            "session_id": "sess-1",
            "speaker": "user",
            "entity": "user",
            "relation": "other_fact",
            "value": "still new to instagram ads",
            "value_type": "other",
            "time_scope": "current",
            "status_hint": "active",
            "polarity": "neutral",
            "confidence": 1.0,
            "evidence_text": "I am still new to Instagram ads and need help creating an ad account.",
            "canonical_gloss": "other_fact=still new to instagram ads",
        }
    )
    system.state = StructuredMemoryState(
        residual_slots=[
            system.slot_encoder.encode(price_observation, timestamp="2026-04-07T05:00:00Z", bank="residual"),
            system.slot_encoder.encode(distractor_observation, timestamp="2026-04-07T05:01:00Z", bank="residual"),
        ]
    )

    result = system.query("query-price", "How much did I spend on a designer handbag?")
    assert result.answer_text == "800"


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


def test_structured_memory_system_repairs_learned_belief_from_context():
    system = StructuredMemorySystem(
        memory_mode="learned_memory",
        use_learned_memory=True,
        learned_belief_predictor=lambda *_args, **_kwargs: '"food_preference" is the current preference of the user.',
    )
    system.observe_observation(
        Observation.from_dict(
            {
                "obs_id": "obs-learned-food",
                "source_dataset": "synthetic",
                "source_dialogue_id": "dlg-1",
                "source_turn_id": "turn-1",
                "session_id": "sess-1",
                "speaker": "user",
                "entity": "user",
                "relation": "food_preference",
                "value": "burgers",
                "value_type": "other",
                "time_scope": "current",
                "status_hint": "active",
                "polarity": "positive",
                "confidence": 1.0,
                "evidence_text": "I currently prefer burgers.",
                "canonical_gloss": "food_preference=burgers",
            }
        ),
        timestamp="2026-04-07T05:00:00Z",
    )

    result = system.query("query-learned-repaired", "What food does the user currently prefer?")
    assert result.belief_source == "learned_memory"
    assert result.belief_state.belief_items[0].relation == "food_preference"
    assert result.belief_state.belief_items[0].value == "burgers"
    assert result.answer_text == "burgers"


def test_structured_memory_system_backfills_relation_name_placeholder_from_support_slot():
    system = StructuredMemorySystem(
        memory_mode="learned_memory",
        use_learned_memory=True,
        learned_belief_predictor=lambda query_id, *_args, **_kwargs: {
            "query_id": query_id,
            "query_type": "single_fact",
            "belief_items": [
                {
                    "relation": "class_location",
                    "value": "class_location",
                }
            ],
        },
    )
    system.observe_observation(
        Observation.from_dict(
            {
                "obs_id": "obs-class-location",
                "source_dataset": "synthetic",
                "source_dialogue_id": "dlg-class",
                "source_turn_id": "turn-1",
                "session_id": "sess-class",
                "speaker": "user",
                "entity": "user",
                "relation": "class_location",
                "value": "Serenity Yoga",
                "value_type": "other",
                "time_scope": "current",
                "status_hint": "active",
                "polarity": "positive",
                "confidence": 1.0,
                "evidence_text": "I take classes at Serenity Yoga.",
                "canonical_gloss": "class_location=serenity yoga",
            }
        ),
        timestamp="2026-04-07T05:00:00Z",
    )

    result = system.query("query-class-location", "Where do I take yoga classes?")
    assert result.belief_source == "learned_memory"
    assert result.belief_state.belief_items[0].relation == "class_location"
    assert result.belief_state.belief_items[0].value == "serenity yoga"
    assert result.answer_text == "serenity yoga"


def test_structured_memory_system_falls_back_from_raw_value_fragment_in_learned_belief():
    system = StructuredMemorySystem(
        memory_mode="learned_memory",
        use_learned_memory=True,
        learned_belief_predictor=lambda *_args, **_kwargs: 'modern beats with Pacific sounds"]',
    )
    system.observe_observation(
        Observation.from_dict(
            {
                "obs_id": "obs-learned-music",
                "source_dataset": "synthetic",
                "source_dialogue_id": "dlg-1",
                "source_turn_id": "turn-1",
                "session_id": "sess-1",
                "speaker": "user",
                "entity": "user",
                "relation": "music_preference",
                "value": "producing music with software",
                "value_type": "other",
                "time_scope": "current",
                "status_hint": "active",
                "polarity": "positive",
                "confidence": 1.0,
                "evidence_text": "I like producing music with software.",
                "canonical_gloss": "music_preference=producing music with software",
            }
        ),
        timestamp="2026-04-07T05:00:00Z",
    )

    result = system.query("query-learned-fragment", "I recently attended an event where there was a unique blend of modern beats with Pacific sounds.")
    assert result.belief_source == "learned_memory"
    assert result.belief_state.belief_items[0].relation == "music_preference"
    assert result.belief_state.belief_items[0].value == "producing music with software"


def test_structured_memory_system_uses_query_aligned_slot_for_raw_belief_fallback():
    system = StructuredMemorySystem(
        memory_mode="learned_memory",
        use_learned_memory=True,
        learned_belief_predictor=lambda *_args, **_kwargs: '"context_to_belief": "context_to_belief"',
    )
    system.observe_observation(
        Observation.from_dict(
            {
                "obs_id": "obs-learned-recipes",
                "source_dataset": "synthetic",
                "source_dialogue_id": "dlg-1",
                "source_turn_id": "turn-1",
                "session_id": "sess-1",
                "speaker": "user",
                "entity": "user",
                "relation": "hobby",
                "value": "experimenting with different recipes",
                "value_type": "other",
                "time_scope": "current",
                "status_hint": "active",
                "polarity": "positive",
                "confidence": 0.75,
                "evidence_text": "I enjoy experimenting with different recipes.",
                "canonical_gloss": "hobby=experimenting with different recipes",
            }
        ),
        timestamp="2026-04-07T05:00:00Z",
    )
    system.observe_observation(
        Observation.from_dict(
            {
                "obs_id": "obs-learned-comments",
                "source_dataset": "synthetic",
                "source_dialogue_id": "dlg-1",
                "source_turn_id": "turn-2",
                "session_id": "sess-1",
                "speaker": "user",
                "entity": "user",
                "relation": "other_fact",
                "value": "more descriptive comments that highlight the purpose of code blocks",
                "value_type": "other",
                "time_scope": "current",
                "status_hint": "active",
                "polarity": "neutral",
                "confidence": 0.95,
                "evidence_text": "I prefer more descriptive comments that highlight the purpose of code blocks.",
                "canonical_gloss": "other_fact=more descriptive comments that highlight the purpose of code blocks",
            }
        ),
        timestamp="2026-04-07T05:01:00Z",
    )

    result = system.query("query-learned-cooking", "Can you suggest some new cooking techniques or recipes I might enjoy exploring?")
    assert result.belief_source == "learned_memory"
    assert result.belief_state.belief_items[0].relation == "hobby"
    assert result.belief_state.belief_items[0].value == "experimenting with different recipes"


def test_structured_memory_system_backfills_malformed_learned_belief_value_from_support_slot():
    system = StructuredMemorySystem(
        memory_mode="learned_memory",
        use_learned_memory=True,
        learned_belief_predictor=lambda *_args, **_kwargs: 'other_fact=on a journey to redefine how i approach group collaborations, aiming for a more structured", "confidence": 0.82, "',
    )
    system.observe_observation(
        Observation.from_dict(
            {
                "obs_id": "obs-learned-music-pref",
                "source_dataset": "synthetic",
                "source_dialogue_id": "dlg-1",
                "source_turn_id": "turn-1",
                "session_id": "sess-1",
                "speaker": "user",
                "entity": "user",
                "relation": "music_preference",
                "value": "music in its truest form, without rigid guidelines dictating how i should dissect it",
                "value_type": "other",
                "time_scope": "current",
                "status_hint": "active",
                "polarity": "positive",
                "confidence": 0.9,
                "evidence_text": "I want to enjoy music in its truest form.",
                "canonical_gloss": "music_preference=music in its truest form, without rigid guidelines dictating how i should dissect it",
            }
        ),
        timestamp="2026-04-07T05:00:00Z",
    )
    system.observe_observation(
        Observation.from_dict(
            {
                "obs_id": "obs-learned-other-fact",
                "source_dataset": "synthetic",
                "source_dialogue_id": "dlg-1",
                "source_turn_id": "turn-2",
                "session_id": "sess-1",
                "speaker": "user",
                "entity": "user",
                "relation": "other_fact",
                "value": "on a journey to redefine how i approach group collaborations, aiming for a more structured approach",
                "value_type": "other",
                "time_scope": "current",
                "status_hint": "active",
                "polarity": "neutral",
                "confidence": 0.82,
                "evidence_text": "I am rethinking how I approach group collaborations.",
                "canonical_gloss": "other_fact=on a journey to redefine how i approach group collaborations, aiming for a more structured approach",
            }
        ),
        timestamp="2026-04-07T05:01:00Z",
    )

    result = system.query("query-learned-music-support", "How can I find a more fulfilling way to express my love for music?")
    assert result.belief_source == "learned_memory"
    assert result.belief_state.belief_items[0].relation == "music_preference"
    assert result.belief_state.belief_items[0].value == "music in its truest form, without rigid guidelines dictating how i should dissect it"


def test_structured_memory_system_backfills_boolean_like_learned_belief_value_from_support_slot():
    system = StructuredMemorySystem(
        memory_mode="learned_memory",
        use_learned_memory=True,
        learned_belief_predictor=lambda *_args, **_kwargs: "true",
    )
    system.observe_observation(
        Observation.from_dict(
            {
                "obs_id": "obs-learned-music-boolean",
                "source_dataset": "synthetic",
                "source_dialogue_id": "dlg-1",
                "source_turn_id": "turn-1",
                "session_id": "sess-1",
                "speaker": "user",
                "entity": "user",
                "relation": "music_preference",
                "value": "music in its truest form, without rigid guidelines dictating how i should dissect it",
                "value_type": "other",
                "time_scope": "current",
                "status_hint": "active",
                "polarity": "positive",
                "confidence": 0.9,
                "evidence_text": "I want to enjoy music in its truest form.",
                "canonical_gloss": "music_preference=music in its truest form, without rigid guidelines dictating how i should dissect it",
            }
        ),
        timestamp="2026-04-07T05:00:00Z",
    )

    result = system.query("query-learned-bool-support", "How can I find a more fulfilling way to express my love for music?")
    assert result.belief_source == "learned_memory"
    assert result.belief_state.belief_items[0].relation == "music_preference"
    assert result.belief_state.belief_items[0].value == "music in its truest form, without rigid guidelines dictating how i should dissect it"


def test_structured_memory_system_backfills_same_relation_belief_value_when_ungrounded():
    def _predict(query_id: str, _query_text: str, _slots):
        return {
            "query_id": query_id,
            "query_type": "single_fact",
            "belief_items": [
                {
                    "relation": "music_preference",
                    "value": "able to create a unique sound that pays homage to my roots while also pushing the boundaries of what is traditionally expected",
                    "support_slot_ids": [],
                }
            ],
        }

    system = StructuredMemorySystem(
        memory_mode="learned_memory",
        use_learned_memory=True,
        learned_belief_predictor=_predict,
    )
    system.observe_observation(
        Observation.from_dict(
            {
                "obs_id": "obs-grounded-music-pref",
                "source_dataset": "synthetic",
                "source_dialogue_id": "dlg-1",
                "source_turn_id": "turn-1",
                "session_id": "sess-1",
                "speaker": "user",
                "entity": "user",
                "relation": "music_preference",
                "value": "producing music with software",
                "value_type": "other",
                "time_scope": "current",
                "status_hint": "active",
                "polarity": "positive",
                "confidence": 0.95,
                "evidence_text": "I like producing music with software.",
                "canonical_gloss": "music_preference=producing music with software",
            }
        ),
        timestamp="2026-04-07T05:00:00Z",
    )

    result = system.query("query-grounded-music-support", "How can I find a more fulfilling way to express my love for music?")
    assert result.belief_source == "learned_memory"
    assert result.belief_state.belief_items[0].relation == "music_preference"
    assert result.belief_state.belief_items[0].value == "producing music with software"


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


def test_structured_memory_system_preserves_distinct_preference_facets_under_learned_slot_assignment():
    calls = {"count": 0}

    def _predict(_observation, _slots):
        calls["count"] += 1
        return {"target_action": "merge", "target_flags": {"promote": False, "stale_old": False}}

    system = StructuredMemorySystem(
        slot_assignment_mode="learned",
        use_learned_slot_assignment=True,
        learned_slot_assignment_predictor=_predict,
    )
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
        slot
        for slot in [*system.state.core_slots, *system.state.residual_slots]
        if slot.active_flag and slot.relation == "music_preference"
    ]
    assert len(active_music_slots) >= 2
    assert calls["count"] == 0


def test_structured_memory_system_repairs_learned_slot_assignment_from_context():
    system = StructuredMemorySystem(
        slot_assignment_mode="learned",
        use_learned_slot_assignment=True,
        learned_slot_assignment_predictor=lambda *_args, **_kwargs: 'temporal_fact=half past 7 in the evening"',
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
                "relation": "temporal_fact",
                "value": "half past 7 in the evening",
                "value_type": "other",
                "time_scope": "current",
                "status_hint": "active",
                "polarity": "neutral",
                "confidence": 1.0,
                "evidence_text": "Dinner is at half past 7 in the evening.",
                "canonical_gloss": "temporal_fact=half past 7 in the evening",
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
                "relation": "temporal_fact",
                "value": "7:30 pm",
                "value_type": "other",
                "time_scope": "recent_change",
                "status_hint": "active",
                "polarity": "neutral",
                "confidence": 1.0,
                "evidence_text": "Actually, make it 7:30 pm.",
                "canonical_gloss": "temporal_fact=7:30 pm",
            }
        ),
        timestamp="2026-04-07T05:05:00Z",
    )

    active_slots = system.state.active_slots()
    assert len(active_slots) == 1
    assert active_slots[0].canonical_gloss == "temporal_fact=7:30 pm"


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


def test_structured_memory_system_short_circuits_weak_other_fact_overwrite():
    calls = {"count": 0}

    def _predict(_observation, _slots):
        calls["count"] += 1
        return {"target_action": "overwrite", "target_flags": {"promote": False, "stale_old": True}}

    system = StructuredMemorySystem(
        slot_assignment_mode="learned",
        use_learned_slot_assignment=True,
        learned_slot_assignment_predictor=_predict,
    )
    for index, value in enumerate(["updating my email address", "considering a vacation cabin"]):
        system.observe_observation(
            Observation.from_dict(
                {
                    "obs_id": f"obs-weak-{index}",
                    "source_dataset": "synthetic",
                    "source_dialogue_id": "dlg-1",
                    "source_turn_id": f"turn-{index}",
                    "session_id": "sess-1",
                    "speaker": "user",
                    "entity": "user",
                    "relation": "other_fact",
                    "value": value,
                    "value_type": "other",
                    "time_scope": "current",
                    "status_hint": "active",
                    "polarity": "neutral",
                    "confidence": 1.0,
                    "evidence_text": value,
                    "canonical_gloss": f"other_fact={value}",
                }
            ),
            timestamp=f"2026-04-07T05:0{index}:00Z",
        )

    system.observe_observation(
        Observation.from_dict(
            {
                "obs_id": "obs-weak-overwrite",
                "source_dataset": "synthetic",
                "source_dialogue_id": "dlg-1",
                "source_turn_id": "turn-overwrite",
                "session_id": "sess-1",
                "speaker": "user",
                "entity": "user",
                "relation": "other_fact",
                "value": "designing steering column stalks",
                "value_type": "other",
                "time_scope": "past",
                "status_hint": "unknown",
                "polarity": "neutral",
                "confidence": 1.0,
                "evidence_text": "I was designing steering column stalks.",
                "canonical_gloss": "other_fact=designing steering column stalks",
            }
        ),
        timestamp="2026-04-07T06:00:00Z",
    )

    assert calls["count"] == 0
    assert any("designing steering column stalks" in slot.canonical_gloss for slot in system.state.active_slots())


def test_structured_memory_system_preserves_strong_other_fact_overwrite_candidates():
    system = StructuredMemorySystem()
    candidate = system.slot_encoder.encode(
        Observation.from_dict(
            {
                "obs_id": "obs-strong-0",
                "source_dataset": "synthetic",
                "source_dialogue_id": "dlg-1",
                "source_turn_id": "turn-0",
                "session_id": "sess-1",
                "speaker": "user",
                "entity": "user",
                "relation": "other_fact",
                "value": "learning about a data analysis course for work",
                "value_type": "other",
                "time_scope": "current",
                "status_hint": "active",
                "polarity": "neutral",
                "confidence": 1.0,
                "evidence_text": "learning about a data analysis course for work",
                "canonical_gloss": "other_fact=learning about a data analysis course for work",
            }
        ),
        timestamp="2026-04-07T05:10:00Z",
        bank="residual",
    )
    overwrite_observation = Observation.from_dict(
        {
            "obs_id": "obs-strong-overwrite",
            "source_dataset": "synthetic",
            "source_dialogue_id": "dlg-1",
            "source_turn_id": "turn-strong-overwrite",
            "session_id": "sess-1",
            "speaker": "user",
            "entity": "user",
            "relation": "other_fact",
            "value": "interested in a data analysis course for work",
            "value_type": "other",
            "time_scope": "past",
            "status_hint": "unknown",
            "polarity": "neutral",
            "confidence": 1.0,
            "evidence_text": "I was interested in a data analysis course for work.",
            "canonical_gloss": "other_fact=interested in a data analysis course for work",
        }
    )

    decision = system._fast_path_weak_other_fact_overwrite(
        overwrite_observation,
        candidates=[candidate],
        symbolic_decision=type("Decision", (), {"action": "overwrite", "matched_slot_id": candidate.slot_id, "promote": False})(),
    )

    assert decision is None


def test_structured_memory_system_caps_slot_assignment_decode_length(tmp_path: Path):
    config_path = tmp_path / "train.yaml"
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    config_path.write_text(
        "training:\n  batching:\n    max_source_length: 256\n    max_target_length: 192\n",
        encoding="utf-8",
    )

    captured: dict[str, int] = {}

    def _generate(_model, _tokenizer, _example, *, max_source_length: int, max_target_length: int, device: str) -> str:
        captured["max_source_length"] = max_source_length
        captured["max_target_length"] = max_target_length
        captured["device"] = device
        return '{"target_action":"new","target_flags":{"promote":false,"stale_old":false}}'

    system = StructuredMemorySystem(
        slot_assignment_mode="learned",
        use_learned_slot_assignment=True,
        learned_slot_assignment_checkpoint_dir=str(checkpoint_dir),
        learned_slot_assignment_train_config_path=str(config_path),
        learned_slot_assignment_device="cpu",
    )
    observation = Observation.from_dict(
        {
            "obs_id": "obs-1",
            "source_dataset": "synthetic",
            "source_dialogue_id": "dlg-1",
            "source_turn_id": "turn-1",
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
            "evidence_text": "I live in Seattle.",
            "canonical_gloss": "location=Seattle",
        }
    )

    with patch("core_mem.v2.training.load_runtime_components", return_value=(object(), object())), patch(
        "core_mem.v2.training.generate_prediction_text",
        side_effect=_generate,
    ):
        predictor = system._resolve_slot_assignment_predictor()
        assert predictor is not None
        payload = predictor(observation, [])

    assert '"target_action":"new"' in payload
    assert captured["max_source_length"] == 256
    assert captured["max_target_length"] == 48
    assert captured["device"] == "cpu"


def test_structured_memory_system_compacts_slot_assignment_prompt_slots_for_dense_other_fact_overwrite():
    system = StructuredMemorySystem()
    for index in range(15):
        value = f"seed fact {index}"
        if index == 0:
            value = "planning a longer backpacking trip"
        system.observe_observation(
            Observation.from_dict(
                {
                    "obs_id": f"obs-seed-{index}",
                    "source_dataset": "synthetic",
                    "source_dialogue_id": "dlg-1",
                    "source_turn_id": f"turn-{index}",
                    "session_id": "sess-1",
                    "speaker": "user",
                    "entity": "user",
                    "relation": "other_fact",
                    "value": value,
                    "value_type": "other",
                    "time_scope": "current",
                    "status_hint": "active",
                    "polarity": "neutral",
                    "confidence": 1.0,
                    "evidence_text": value,
                    "canonical_gloss": f"other_fact={value}",
                }
            ),
            timestamp=f"2026-04-07T05:{index:02d}:00Z",
        )

    overwrite_observation = Observation.from_dict(
        {
            "obs_id": "obs-overwrite",
            "source_dataset": "synthetic",
            "source_dialogue_id": "dlg-1",
            "source_turn_id": "turn-overwrite",
            "session_id": "sess-1",
            "speaker": "user",
            "entity": "user",
            "relation": "other_fact",
            "value": "planning a longer backpacking trip through europe soon",
            "value_type": "other",
            "time_scope": "past",
            "status_hint": "unknown",
            "polarity": "neutral",
            "confidence": 1.0,
            "evidence_text": "I was planning a longer backpacking trip soon.",
            "canonical_gloss": "other_fact=planning a longer backpacking trip through europe soon",
        }
    )
    symbolic_target = system.state.active_slots()[0]
    prompt_slots = system._slot_assignment_prompt_slots(
        overwrite_observation,
        system.state.active_slots(),
        candidates=system._slot_assignment_candidates(overwrite_observation, system.state.active_slots()),
        symbolic_decision=type(
            "Decision",
            (),
            {"action": "overwrite", "matched_slot_id": symbolic_target.slot_id, "promote": False},
        )(),
    )

    assert len(prompt_slots) <= 12
    assert prompt_slots[0].slot_id == symbolic_target.slot_id
    assert len(prompt_slots) < len(system.state.active_slots())


def test_structured_memory_system_compacted_slot_assignment_prompt_keeps_topical_candidate():
    system = StructuredMemorySystem()
    seed_values = [
        "tracking coupon redemptions",
        "learning more about wildlife management",
        "social media growth tips for my bakery business",
        "thinking about a vacation cabin",
        "reading about composting",
        "shopping for hiking boots",
        "updating my email address",
        "planning a small herb garden",
        "trying yoga before bed",
    ]
    for index, value in enumerate(seed_values):
        system.observe_observation(
            Observation.from_dict(
                {
                    "obs_id": f"obs-topical-{index}",
                    "source_dataset": "synthetic",
                    "source_dialogue_id": "dlg-1",
                    "source_turn_id": f"turn-{index}",
                    "session_id": "sess-1",
                    "speaker": "user",
                    "entity": "user",
                    "relation": "other_fact",
                    "value": value,
                    "value_type": "other",
                    "time_scope": "current",
                    "status_hint": "active",
                    "polarity": "neutral",
                    "confidence": 1.0,
                    "evidence_text": value,
                    "canonical_gloss": f"other_fact={value}",
                }
            ),
            timestamp=f"2026-04-07T05:{index:02d}:00Z",
        )

    overwrite_observation = Observation.from_dict(
        {
            "obs_id": "obs-social-media",
            "source_dataset": "synthetic",
            "source_dialogue_id": "dlg-1",
            "source_turn_id": "turn-social-media",
            "session_id": "sess-1",
            "speaker": "user",
            "entity": "user",
            "relation": "other_fact",
            "value": "looking for more social media advice for my bakery business",
            "value_type": "other",
            "time_scope": "past",
            "status_hint": "unknown",
            "polarity": "neutral",
            "confidence": 1.0,
            "evidence_text": "I was looking for more social media advice for my bakery.",
            "canonical_gloss": "other_fact=looking for more social media advice for my bakery business",
        }
    )

    prompt_slots = system._slot_assignment_prompt_slots(
        overwrite_observation,
        system.state.active_slots(),
        candidates=system._slot_assignment_candidates(overwrite_observation, system.state.active_slots()),
        symbolic_decision=type(
            "Decision",
            (),
            {"action": "overwrite", "matched_slot_id": system.state.active_slots()[0].slot_id, "promote": False},
        )(),
    )

    assert any("social media growth tips for my bakery" in slot.canonical_gloss for slot in prompt_slots)
    assert len(prompt_slots) <= 12


def test_structured_memory_system_query_terms_normalize_question_noise():
    terms = StructuredMemorySystem._query_terms("How long is my daily commute to work?")
    assert "how" not in terms
    assert "long" not in terms
    assert "daily" not in terms
    assert "commute" in terms
    assert "work" in terms


def test_structured_memory_system_query_terms_drop_generic_advice_words():
    terms = StructuredMemorySystem._query_terms(
        "Can you suggest some new cooking techniques or recipes I might enjoy exploring?"
    )
    assert "can" not in terms
    assert "might" not in terms
    assert "enjoy" not in terms
    assert "suggest" not in terms
    assert "explor" not in terms
    assert "cook" in terms
    assert "recip" in terms


def test_structured_memory_system_detects_open_ended_advice_queries():
    assert StructuredMemorySystem._query_seeks_open_ended_advice(
        "I've been involved in planning events for my community lately, but I'm not sure if I should continue with it. What do you think?"
    )
    assert StructuredMemorySystem._query_seeks_open_ended_advice(
        "Can you suggest some new cooking techniques or recipes I might enjoy exploring?"
    )
    assert not StructuredMemorySystem._query_seeks_open_ended_advice(
        "What is the name of the yoga studio where I take classes?"
    )


def test_structured_memory_system_advice_queries_do_not_overweight_irrelevant_enjoy_overlap():
    def _latent_ranker(_query_text: str, slots):
        return {
            slot.slot_id: (3.0 if "experimenting with different recipes" in slot.canonical_gloss else 0.1)
            for slot in slots
        }

    system = StructuredMemorySystem(latent_slot_ranker=_latent_ranker)
    system.observe_observation(
        Observation.from_dict(
            {
                "obs_id": "obs-meditation",
                "source_dataset": "synthetic",
                "source_dialogue_id": "dlg-advice",
                "source_turn_id": "turn-1",
                "session_id": "sess-advice",
                "speaker": "user",
                "entity": "user",
                "relation": "hobby",
                "value": "not enjoy meditation",
                "value_type": "other",
                "time_scope": "current",
                "status_hint": "active",
                "polarity": "negative",
                "confidence": 0.8,
                "evidence_text": "I used to not enjoy meditation.",
                "canonical_gloss": "hobby=not enjoy meditation",
            }
        ),
        timestamp="2026-04-07T05:00:00Z",
    )
    system.observe_observation(
        Observation.from_dict(
            {
                "obs_id": "obs-recipes",
                "source_dataset": "synthetic",
                "source_dialogue_id": "dlg-advice",
                "source_turn_id": "turn-2",
                "session_id": "sess-advice",
                "speaker": "user",
                "entity": "user",
                "relation": "hobby",
                "value": "experimenting with different recipes",
                "value_type": "other",
                "time_scope": "current",
                "status_hint": "active",
                "polarity": "positive",
                "confidence": 0.9,
                "evidence_text": "I enjoy experimenting with different recipes.",
                "canonical_gloss": "hobby=experimenting with different recipes",
            }
        ),
        timestamp="2026-04-07T05:01:00Z",
    )

    result = system.query(
        "query-advice-cooking",
        "Can you suggest some new cooking techniques or recipes I might enjoy exploring?",
    )
    assert "experimenting with different recipes" in result.selected_slots[0].canonical_gloss


def test_structured_memory_system_advice_queries_do_not_overweight_scenario_terms():
    def _latent_ranker(_query_text: str, slots):
        return {
            slot.slot_id: (3.0 if "imagining the conversations" in slot.canonical_gloss else 0.1)
            for slot in slots
        }

    system = StructuredMemorySystem(latent_slot_ranker=_latent_ranker)
    system.observe_observation(
        Observation.from_dict(
            {
                "obs_id": "obs-literacy",
                "source_dataset": "synthetic",
                "source_dialogue_id": "dlg-advice",
                "source_turn_id": "turn-1",
                "session_id": "sess-advice",
                "speaker": "user",
                "entity": "user",
                "relation": "other_fact",
                "value": "assisting with literacy programs",
                "value_type": "other",
                "time_scope": "current",
                "status_hint": "active",
                "polarity": "neutral",
                "confidence": 0.82,
                "evidence_text": "I'm assisting with literacy programs and organizing community events to promote reading.",
                "canonical_gloss": "other_fact=assisting with literacy programs",
            }
        ),
        timestamp="2026-04-07T05:00:00Z",
    )
    system.observe_observation(
        Observation.from_dict(
            {
                "obs_id": "obs-conversations",
                "source_dataset": "synthetic",
                "source_dialogue_id": "dlg-advice",
                "source_turn_id": "turn-2",
                "session_id": "sess-advice",
                "speaker": "user",
                "entity": "user",
                "relation": "other_fact",
                "value": "imagining the conversations that might have occurred among readers",
                "value_type": "other",
                "time_scope": "current",
                "status_hint": "active",
                "polarity": "neutral",
                "confidence": 0.82,
                "evidence_text": "I enjoy imagining the conversations that might have occurred among readers.",
                "canonical_gloss": "other_fact=imagining the conversations that might have occurred among readers",
            }
        ),
        timestamp="2026-04-07T05:01:00Z",
    )

    result = system.query(
        "query-advice-community",
        "I've been involved in planning events for my community lately, but I'm not sure if I should continue with it. What do you think?",
    )
    assert "imagining the conversations" in result.selected_slots[0].canonical_gloss


def test_structured_memory_system_slot_terms_apply_light_stemming():
    observation = Observation.from_dict(
        {
            "obs_id": "obs-stem",
            "source_dataset": "synthetic",
            "source_dialogue_id": "dlg-1",
            "source_turn_id": "turn-1",
            "session_id": "sess-1",
            "speaker": "user",
            "entity": "user",
            "relation": "attended_play",
            "value": "The Glass Menagerie",
            "value_type": "event",
            "time_scope": "past",
            "status_hint": "active",
            "polarity": "neutral",
            "confidence": 1.0,
            "evidence_text": "I attended The Glass Menagerie.",
            "canonical_gloss": "attended_play=the glass menagerie",
        }
    )
    system = StructuredMemorySystem()
    slot = system.slot_encoder.encode(observation, timestamp="2026-04-07T05:00:00Z", bank="residual")
    terms = system._slot_terms(slot)

    assert "attend" in terms
    assert "attended" not in terms
    assert "play" in terms


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
    assert result.belief_state.belief_items[0].support_slot_ids == [result.selected_slots[0].slot_id]


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
    assert result.answer_text == "the sports store downtown"


def test_answer_projection_extracts_historical_name_from_other_fact_clause():
    def _predict(query_id: str, _query_text: str, _slots):
        return {
            "query_id": query_id,
            "query_type": "single_fact",
            "belief_items": [
                {
                    "relation": "other_fact",
                    "value": "still getting used to it - it's funny, my old name was johnson",
                    "support_slot_ids": [],
                }
            ],
        }

    system = StructuredMemorySystem(memory_mode="learned_memory", use_learned_memory=True, learned_belief_predictor=_predict)
    system.observe_turn(
        "I'm still getting used to it, but it's funny because my old name was Johnson.",
        source_dataset="synthetic",
        source_dialogue_id="dlg-1",
        source_turn_id="turn-1",
        session_id="sess-1",
        timestamp="2026-04-07T05:00:00Z",
    )

    result = system.query("query-projection-old-name", "What was my last name before I changed it?")
    assert result.answer_text == "johnson"


def test_answer_projection_prefers_last_location_phrase_for_event_answers():
    def _predict(query_id: str, _query_text: str, _slots):
        return {
            "query_id": query_id,
            "query_type": "single_fact",
            "belief_items": [
                {
                    "relation": "episodic_event",
                    "value": "redeemed a $5 coupon on coffee creamer last sunday at target",
                    "support_slot_ids": [],
                }
            ],
        }

    system = StructuredMemorySystem(memory_mode="learned_memory", use_learned_memory=True, learned_belief_predictor=_predict)
    system.observe_turn(
        "I redeemed a $5 coupon on coffee creamer last Sunday.",
        source_dataset="synthetic",
        source_dialogue_id="dlg-1",
        source_turn_id="turn-1",
        session_id="sess-1",
        timestamp="2026-04-07T05:00:00Z",
    )

    result = system.query("query-projection-event-where", "Where did I redeem a $5 coupon on coffee creamer?")
    assert result.answer_text == "target"


def test_answer_projection_extracts_page_count_for_how_many_queries():
    def _predict(query_id: str, _query_text: str, _slots):
        return {
            "query_id": query_id,
            "query_type": "single_fact",
            "belief_items": [
                {
                    "relation": "other_fact",
                    "value": (
                        'interested in learning more about renewable energy, and i just finished reading '
                        'about the discovery of dna structure in "a short history of nearly everything" '
                        "- i'm now on page 220"
                    ),
                    "support_slot_ids": [],
                }
            ],
        }

    system = StructuredMemorySystem(memory_mode="learned_memory", use_learned_memory=True, learned_belief_predictor=_predict)
    system.observe_turn(
        "I am now on page 220 of A Short History of Nearly Everything.",
        source_dataset="synthetic",
        source_dialogue_id="dlg-1",
        source_turn_id="turn-1",
        session_id="sess-1",
        timestamp="2026-04-07T05:00:00Z",
    )

    result = system.query("query-projection-pages", "How many pages of A Short History of Nearly Everything have I read so far?")
    assert result.answer_text == "220"


def test_answer_projection_extracts_frequency_phrase_for_how_often_queries():
    def _predict(query_id: str, _query_text: str, _slots):
        return {
            "query_id": query_id,
            "query_type": "single_fact",
            "belief_items": [
                {
                    "relation": "other_fact",
                    "value": "more focused on days when i attend yoga classes, which is three times a week - it really helps me clear my head",
                    "support_slot_ids": [],
                }
            ],
        }

    system = StructuredMemorySystem(memory_mode="learned_memory", use_learned_memory=True, learned_belief_predictor=_predict)
    system.observe_turn(
        "I attend yoga classes three times a week to help with my anxiety.",
        source_dataset="synthetic",
        source_dialogue_id="dlg-1",
        source_turn_id="turn-1",
        session_id="sess-1",
        timestamp="2026-04-07T05:00:00Z",
    )

    result = system.query("query-projection-frequency", "How often do I attend yoga classes to help with my anxiety?")
    assert result.answer_text == "three times a week"


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
