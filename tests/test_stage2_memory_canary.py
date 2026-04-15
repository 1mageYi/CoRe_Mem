from pathlib import Path
import json
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from core_mem.v2.system import StructuredMemorySystem
from core_mem.benchmarks.personamem import PersonaMemQuestion
from run_stage2_memory_canary import (
    _best_personamem_option_label,
    _finalize_personamem_provider_prediction,
    _observe_personamem_context,
    _project_personamem_local_answer,
    _rewrite_persona_summary,
    _render_personamem_options,
    _render_personamem_prompt,
)


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_stage2_memory_canary_writes_honest_blocked_artifact_without_provider(tmp_path: Path):
    output_root = tmp_path / "outputs_v2"
    result = _run(
        "scripts/run_stage2_memory_canary.py",
        "--config",
        "configs/minimax_m27.yaml",
        "--benchmark",
        "personamem",
        "--output-root",
        str(output_root),
        "--limit",
        "1",
        "--json",
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert Path(payload["summary_path"]).exists()
    assert Path(payload["run_dir"]).exists()
    assert payload["model"] == "MiniMax-M2.7"
    assert payload["status"] in {"completed", "blocked_provider_not_configured"}


def test_personamem_context_observer_skips_assistant_turns():
    system = StructuredMemorySystem()
    observed = _observe_personamem_context(
        system,
        "system: User persona summary.\n"
        "user: User: I like matcha latte.\n"
        "assistant: Assistant: You should keep exploring new drinks.",
        sample_id="sample-1",
    )
    assert observed == 3
    slots = [*system.state.core_slots, *system.state.residual_slots]
    assert any("matcha latte" in slot.canonical_gloss for slot in slots)
    assert all("keep exploring new drinks" not in slot.canonical_gloss for slot in slots)


def test_personamem_prompt_keeps_label_space():
    question = PersonaMemQuestion(
        persona_id="p",
        question_id="q",
        question_type="recall_user_shared_facts",
        topic="music",
        user_question_or_message="What fits best?",
        correct_answer="(b)",
        all_options=["(a) Wrong", "(b) Right"],
        shared_context_id="ctx",
        end_index_in_shared_context=1,
    )
    prompt = _render_personamem_prompt(
        question,
        {
            "belief_state": {"belief_items": [{"relation": "music_preference", "value": "digital music"}]},
            "evidence_block": "- music_preference: digital music",
        },
        candidate_answer="(b)",
    )
    assert "(a) Wrong" in prompt
    assert "(b) Right" in prompt
    assert "Latent matcher candidate" in prompt
    assert "Return only the best option label" in prompt


def test_personamem_local_projection_maps_belief_text_to_option_label():
    question = PersonaMemQuestion(
        persona_id="p",
        question_id="q",
        question_type="recall_user_shared_facts",
        topic="music",
        user_question_or_message="What fits best?",
        correct_answer="(c)",
        all_options=[
            "(a) The user dislikes software tools.",
            "(b) The user prefers painting.",
            "(c) The user likes producing music with software.",
        ],
        shared_context_id="ctx",
        end_index_in_shared_context=1,
    )
    projected = _project_personamem_local_answer(
        {
            "answer_text": "producing music with software",
            "belief_state": {"belief_items": [{"relation": "music_preference", "value": "producing music with software"}]},
            "evidence_block": "- music_preference: producing music with software",
        },
        question,
    )
    assert projected == "(c)"


def test_personamem_option_renderer_preserves_raw_labels():
    rendered = _render_personamem_options(["(a) First", "(b) Second"])
    assert rendered == "(a) First\n(b) Second"


def test_persona_summary_rewriter_extracts_structured_first_person_facts():
    rewrites = _rewrite_persona_summary(
        "Current user persona: Alex is a 32-year-old software engineer with a passion for music and technology. "
        "Currently, he is deeply involved in a project where he's experimenting with MIDI files, aiming to create a fusion of electronic and traditional music. "
        "Alex spends his weekends tinkering with various software tools and musical instruments. "
        "His ultimate goal is to develop an app that helps musicians."
    )
    assert "I am a software engineer." in rewrites
    assert any("I enjoy" in item for item in rewrites)
    assert any("I want to develop an app that helps musicians." == item for item in rewrites)


def test_personamem_option_scorer_prefers_support_overlap():
    question = PersonaMemQuestion(
        persona_id="p",
        question_id="q",
        question_type="recalling_facts_mentioned_by_the_user",
        topic="travel",
        user_question_or_message="What kind of trips fit me best?",
        correct_answer="(a)",
        all_options=[
            "(a) Quiet hiking trips through mountain trails.",
            "(b) Loud nightclub tours in busy cities.",
        ],
        shared_context_id="ctx",
        end_index_in_shared_context=1,
    )
    label = _best_personamem_option_label(
        {
            "belief_state": {"belief_items": [{"relation": "hobby", "value": "hiking mountain trails"}]},
            "evidence_block": "- hobby: hiking mountain trails",
        },
        question,
    )
    assert label == "(a)"


def test_personamem_provider_blank_falls_back_to_option_scorer():
    question = PersonaMemQuestion(
        persona_id="p",
        question_id="q",
        question_type="recalling_facts_mentioned_by_the_user",
        topic="travel",
        user_question_or_message="What kind of trips fit me best?",
        correct_answer="(a)",
        all_options=[
            "(a) Quiet hiking trips through mountain trails.",
            "(b) Loud nightclub tours in busy cities.",
        ],
        shared_context_id="ctx",
        end_index_in_shared_context=1,
    )
    final_prediction = _finalize_personamem_provider_prediction(
        "",
        memory_payload={
            "belief_state": {"belief_items": [{"relation": "hobby", "value": "hiking mountain trails"}]},
            "evidence_block": "- hobby: hiking mountain trails",
        },
        question=question,
    )
    assert final_prediction == "(a)"
