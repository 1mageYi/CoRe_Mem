from pathlib import Path
import json
import subprocess
import sys
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from core_mem.v2.system import StructuredMemorySystem
from core_mem.benchmarks.longmemeval import LongMemEvalQuestion
from core_mem.benchmarks.personamem import PersonaMemQuestion
from run_stage2_memory_canary import (
    _observe_personamem_context,
    _project_personamem_local_answer,
    _resolve_shared_predictors,
    _rewrite_persona_summary,
    _render_longmemeval_prompt,
    _render_personamem_options,
    _render_personamem_prompt,
    run_personamem_canary,
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


def test_stage2_memory_canary_writes_learned_alias_artifact(tmp_path: Path):
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
        "--memory-mode",
        "learned_memory",
        "--json",
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    learned_alias = output_root / "artifacts" / "latest_personamem_stage2_learned_canary.json"
    assert learned_alias.exists()
    alias_payload = json.loads(learned_alias.read_text(encoding="utf-8"))
    assert alias_payload["memory_mode"] == "learned_memory"
    assert alias_payload["summary_path"] == payload["summary_path"]


def test_stage2_memory_canary_writes_slot_assignment_alias_artifact(tmp_path: Path):
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
        "--slot-assignment-mode",
        "learned",
        "--json",
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    slot_alias = output_root / "artifacts" / "latest_personamem_stage2_slot_assignment_canary.json"
    if payload["status"] == "completed":
        assert slot_alias.exists()
        alias_payload = json.loads(slot_alias.read_text(encoding="utf-8"))
        assert alias_payload["slot_assignment_mode"] == "learned"
        assert "provider_exact_match" in alias_payload


def test_stage2_memory_canary_writes_v24_alias_artifact(tmp_path: Path):
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
        "--slot-assignment-mode",
        "learned",
        "--json",
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    v24_alias = output_root / "artifacts" / "latest_personamem_stage2_v24_canary.json"
    if payload["status"] == "completed":
        assert v24_alias.exists()
        alias_payload = json.loads(v24_alias.read_text(encoding="utf-8"))
        assert alias_payload["slot_assignment_mode"] == "learned"
        assert alias_payload["summary_path"] == payload["summary_path"]


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
    )
    assert "(a) Wrong" in prompt
    assert "(b) Right" in prompt
    assert "Latent matcher candidate" not in prompt
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


def test_personamem_local_projection_uses_evidence_overlap_when_answer_text_is_empty():
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
    projected = _project_personamem_local_answer(
        {
            "answer_text": "",
            "belief_state": {"belief_items": [{"relation": "hobby", "value": "hiking mountain trails"}]},
            "evidence_block": "- hobby: hiking mountain trails",
        },
        question,
    )
    assert projected == "(a)"


def test_personamem_prompt_has_no_candidate_injection():
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
    prompt = _render_personamem_prompt(
        question,
        {
            "belief_state": {"belief_items": [{"relation": "hobby", "value": "hiking mountain trails"}]},
            "evidence_block": "- hobby: hiking mountain trails",
        },
    )
    assert "Latent matcher candidate" not in prompt


def test_longmemeval_prompt_adds_query_specific_exact_answer_instruction():
    question = LongMemEvalQuestion(
        question_id="q",
        question_type="single-session-user",
        question="Where did I buy my new tennis racket from?",
        answer="the sports store downtown",
        question_date="2023/05/30 (Tue) 23:39",
        haystack_sessions=[],
        answer_session_ids=[],
    )
    prompt = _render_longmemeval_prompt(
        question,
        {
            "belief_state": {"belief_items": [{"relation": "location", "value": "really happy with my new tennis racket, which i got from a sports store downtown"}]},
            "evidence_block": "- location: really happy with my new tennis racket, which i got from a sports store downtown",
        },
    )
    assert "Return only the shortest exact answer phrase supported by the belief state." in prompt
    assert "Omit any leading preposition" in prompt
    assert "rewrite it as 'the ...'" in prompt


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeProvider:
    def is_configured(self) -> bool:
        return True

    def chat(self, prompt: str, temperature: float, max_tokens: int) -> _FakeResponse:
        return _FakeResponse("(a)")


def test_stage2_memory_canary_writes_semantic_alias_for_completed_learned_run(monkeypatch, tmp_path: Path):
    output_root = tmp_path / "outputs_v2"
    run_dir = output_root / "runs" / "semantic_personamem"
    questions = [
        PersonaMemQuestion(
            persona_id="p1",
            question_id="q1",
            question_type="recall_user_shared_facts",
            topic="food",
            user_question_or_message="What food do I like?",
            correct_answer="(a)",
            all_options=["(a) sushi", "(b) pasta"],
            shared_context_id="ctx",
            end_index_in_shared_context=1,
        )
    ]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"sample_ids": ["q1"]}), encoding="utf-8")

    monkeypatch.setattr(
        "run_stage2_memory_canary.load_project_config",
        lambda path: SimpleNamespace(
            llm=SimpleNamespace(
                api_key_env="GPT_AGENT_API_KEY",
                base_url="https://example.com/v1",
                model="fake-model",
                temperature=0.0,
                max_tokens=16,
                timeout_seconds=1,
                max_retries=0,
                retry_backoff_seconds=0.0,
                min_request_interval_seconds=0.0,
                max_retry_delay_seconds=0.0,
            ),
            benchmarks=SimpleNamespace(personamem=SimpleNamespace(data_root="unused")),
        ),
    )
    monkeypatch.setattr("run_stage2_memory_canary._provider_from_llm", lambda llm: _FakeProvider())
    monkeypatch.setattr("run_stage2_memory_canary._ensure_canary_manifest", lambda output_root, benchmark: manifest_path)

    class _FakeAdapter:
        def load_shared_contexts(self):
            return {"ctx": "user: I like sushi."}

        def load_questions(self):
            return questions

        def render_context_for_question(self, question, contexts):
            return contexts[question.shared_context_id]

    monkeypatch.setattr("run_stage2_memory_canary.PersonaMemAdapter", lambda data_root: _FakeAdapter())

    payload = run_personamem_canary(
        output_root=output_root,
        config_path=REPO_ROOT / "configs" / "minimax_m27.yaml",
        limit=1,
        memory_mode="learned_memory",
        requested_run_dir=str(run_dir),
    )

    assert payload["status"] == "completed"
    semantic_alias = output_root / "artifacts" / "latest_personamem_stage2_semantic_canary.json"
    assert semantic_alias.exists()
    alias_payload = json.loads(semantic_alias.read_text(encoding="utf-8"))
    assert alias_payload["memory_mode"] == "learned_memory"
    assert alias_payload["summary_path"] == payload["summary_path"]


def test_stage2_memory_canary_resume_skips_completed_predictions(monkeypatch, tmp_path: Path):
    output_root = tmp_path / "outputs_v2"
    run_dir = output_root / "runs" / "resume_personamem"
    run_dir.mkdir(parents=True)
    predictions_path = run_dir / "predictions.jsonl"
    predictions_path.write_text(
        json.dumps(
            {
                "sample_id": "q1",
                "benchmark": "personamem",
                "question_type": "recall_user_shared_facts",
                "topic": "food",
                "expected_answer": "(a)",
                "memory_answer_local": "(a)",
                "provider_prediction": "(a)",
                "provider_raw_prediction": "(a)",
                "provider_status": "completed",
                "provider_configured": True,
                "observed_turns": 1,
                "selected_slot_ids": [],
                "belief_state": {"belief_items": []},
                "evidence_block": "",
                "prompt": "done",
                "prompt_version": "stage2_memory_canary_v1",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    questions = [
        PersonaMemQuestion(
            persona_id="p1",
            question_id="q1",
            question_type="recall_user_shared_facts",
            topic="food",
            user_question_or_message="What food do I like?",
            correct_answer="(a)",
            all_options=["(a) sushi", "(b) pasta"],
            shared_context_id="ctx",
            end_index_in_shared_context=1,
        ),
        PersonaMemQuestion(
            persona_id="p1",
            question_id="q2",
            question_type="recall_user_shared_facts",
            topic="food",
            user_question_or_message="What food do I like?",
            correct_answer="(a)",
            all_options=["(a) sushi", "(b) pasta"],
            shared_context_id="ctx",
            end_index_in_shared_context=1,
        ),
    ]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"sample_ids": ["q1", "q2"]}), encoding="utf-8")

    monkeypatch.setattr(
        "run_stage2_memory_canary.load_project_config",
        lambda path: SimpleNamespace(
            llm=SimpleNamespace(
                api_key_env="GPT_AGENT_API_KEY",
                base_url="https://example.com/v1",
                model="fake-model",
                temperature=0.0,
                max_tokens=16,
                timeout_seconds=1,
                max_retries=0,
                retry_backoff_seconds=0.0,
                min_request_interval_seconds=0.0,
                max_retry_delay_seconds=0.0,
            ),
            benchmarks=SimpleNamespace(personamem=SimpleNamespace(data_root="unused")),
        ),
    )
    monkeypatch.setattr("run_stage2_memory_canary._provider_from_llm", lambda llm: _FakeProvider())
    monkeypatch.setattr("run_stage2_memory_canary._ensure_canary_manifest", lambda output_root, benchmark: manifest_path)

    class _FakeAdapter:
        def load_shared_contexts(self):
            return {"ctx": "user: I like sushi."}

        def load_questions(self):
            return questions

        def render_context_for_question(self, question, contexts):
            return contexts[question.shared_context_id]

    monkeypatch.setattr("run_stage2_memory_canary.PersonaMemAdapter", lambda data_root: _FakeAdapter())

    payload = run_personamem_canary(
        output_root=output_root,
        config_path=REPO_ROOT / "configs" / "minimax_m27.yaml",
        limit=2,
        requested_run_dir=str(run_dir),
        resume=True,
    )

    assert payload["status"] == "completed"
    lines = predictions_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    rows = [json.loads(line) for line in lines]
    assert [row["sample_id"] for row in rows] == ["q1", "q2"]
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["resumed_prediction_count"] == 1
    assert metadata["completed_predictions"] == 2
    assert metadata["live_predictions_completed"] == 2


def test_stage2_memory_canary_resolves_shared_predictors_once(monkeypatch):
    calls = {"build": 0, "belief": 0, "slot": 0}
    shared_belief = object()
    shared_slot = object()

    class _TemplateSystem:
        def _resolve_learned_belief_predictor(self):
            calls["belief"] += 1
            return shared_belief

        def _resolve_slot_assignment_predictor(self):
            calls["slot"] += 1
            return shared_slot

    def _fake_build_memory_system(**_kwargs):
        calls["build"] += 1
        return _TemplateSystem()

    monkeypatch.setattr("run_stage2_memory_canary._build_memory_system", _fake_build_memory_system)

    belief_predictor, slot_predictor = _resolve_shared_predictors(
        memory_mode="learned_memory",
        slot_assignment_mode="learned",
        learned_memory_checkpoint_dir="checkpoint",
        learned_memory_train_config_path="config.yaml",
        learned_memory_device="cpu",
        learned_slot_assignment_checkpoint_dir="checkpoint",
        learned_slot_assignment_train_config_path="config.yaml",
        learned_slot_assignment_device="cpu",
    )

    assert belief_predictor is shared_belief
    assert slot_predictor is shared_slot
    assert calls == {"build": 1, "belief": 1, "slot": 1}
