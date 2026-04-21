from pathlib import Path
import json
import os
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
    _acquire_run_lock,
    _build_personamem_row,
    _iter_provider_predictions,
    _memory_payload,
    _observe_personamem_context,
    _project_personamem_local_answer,
    _release_run_lock,
    _resolve_shared_predictors,
    _rewrite_persona_summary,
    _render_longmemeval_prompt,
    _render_personamem_options,
    _render_personamem_prompt,
    _stage2_runtime_defaults,
    _should_use_symbolic_parallel_fast_path,
    _strip_explicit_think_blocks,
    run_personamem_canary,
)


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["GPT_AGENT_API_KEY"] = ""
    return subprocess.run(
        [sys.executable, *args],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        env=env,
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


def test_iter_provider_predictions_supports_parallel_workers(monkeypatch):
    class FakeProvider:
        def chat(self, prompt: str, *, temperature: float = 0.0, max_tokens: int | None = None):
            return SimpleNamespace(content=f"echo::{prompt}")

    monkeypatch.setattr("run_stage2_memory_canary._provider_from_llm", lambda llm: FakeProvider())
    llm = SimpleNamespace(temperature=0.0, max_tokens=32)
    results, failed = _iter_provider_predictions(llm, [(0, "a"), (1, "b"), (2, "c")], workers=3)
    assert failed == []
    assert sorted(results) == [
        (0, "echo::a", "echo::a"),
        (1, "echo::b", "echo::b"),
        (2, "echo::c", "echo::c"),
    ]


def test_stage2_runtime_defaults_load_from_config(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "project:",
                "  name: core_mem",
                "  stage: stage2",
                "llm:",
                "  provider: openai_compatible",
                "  api_key_env: GPT_AGENT_API_KEY",
                "  base_url: https://example.com/v1",
                "  model: fake-model",
                "benchmarks:",
                "  primary: personamem",
                "  secondary: longmemeval_s",
                "stage2_runtime:",
                "  learned_memory_checkpoint_dir: outputs_v2/checkpoints/learned",
                "  learned_memory_train_config_path: configs/stage2_train.yaml",
                "  latent_retriever_checkpoint_dir: outputs_v2/checkpoints/latent",
                "  learned_slot_assignment_checkpoint_dir: outputs_v2/checkpoints/slot",
                "  learned_slot_assignment_train_config_path: configs/stage2_slot.yaml",
            ]
        ),
        encoding="utf-8",
    )

    defaults = _stage2_runtime_defaults(config_path)

    assert defaults["learned_memory_checkpoint_dir"] == "outputs_v2/checkpoints/learned"
    assert defaults["learned_memory_train_config_path"] == "configs/stage2_train.yaml"
    assert defaults["latent_retriever_checkpoint_dir"] == "outputs_v2/checkpoints/latent"
    assert defaults["learned_slot_assignment_checkpoint_dir"] == "outputs_v2/checkpoints/slot"
    assert defaults["learned_slot_assignment_train_config_path"] == "configs/stage2_slot.yaml"


def test_strip_explicit_think_blocks_only_removes_protocol_noise():
    assert _strip_explicit_think_blocks("<think>reasoning</think>\n(a)") == "(a)"
    assert _strip_explicit_think_blocks("```text\n<think>hidden</think>\n(b)\n```") == "(b)"
    assert _strip_explicit_think_blocks("<think>hidden</think>\n\n(c)") == "(c)"
    assert _strip_explicit_think_blocks("plain answer") == "plain answer"


def test_symbolic_parallel_fast_path_only_applies_to_small_canaries():
    assert _should_use_symbolic_parallel_fast_path(
        provider_configured=True,
        memory_mode="symbolic",
        slot_assignment_mode="symbolic",
        provider_workers=4,
        sample_count=64,
    )
    assert not _should_use_symbolic_parallel_fast_path(
        provider_configured=True,
        memory_mode="symbolic",
        slot_assignment_mode="symbolic",
        provider_workers=4,
        sample_count=512,
    )


def test_run_lock_rejects_active_other_process(monkeypatch, tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    lock_path = run_dir / ".active.lock"
    lock_path.write_text(json.dumps({"pid": 999999, "acquired_at": "2026-04-20T00:00:00Z"}), encoding="utf-8")
    monkeypatch.setattr("run_stage2_memory_canary._pid_is_running", lambda pid: pid == 999999)
    try:
        _acquire_run_lock(run_dir)
    except RuntimeError as exc:
        assert "run_dir already active" in str(exc)
    else:
        raise AssertionError("expected active lock to block acquisition")


def test_run_lock_clears_stale_lock(monkeypatch, tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    lock_path = run_dir / ".active.lock"
    lock_path.write_text(json.dumps({"pid": 999999, "acquired_at": "2026-04-20T00:00:00Z"}), encoding="utf-8")
    monkeypatch.setattr("run_stage2_memory_canary._pid_is_running", lambda pid: False)
    acquired = _acquire_run_lock(run_dir)
    assert acquired.exists()
    _release_run_lock(acquired)
    assert not acquired.exists()


def test_build_personamem_row_can_complete_provider_prediction(monkeypatch):
    class FakeAdapter:
        def render_context_for_question(self, question, contexts):
            assert contexts == {"ctx": "system: User persona summary.\nuser: User: I like matcha latte."}
            return contexts[question.shared_context_id]

    monkeypatch.setattr("run_stage2_memory_canary._provider_chat_request", lambda llm, prompt: ("(b)", f"raw::{prompt}"))
    row = _build_personamem_row(
        PersonaMemQuestion(
            persona_id="p",
            question_id="q",
            question_type="recall_user_shared_facts",
            topic="food",
            user_question_or_message="Which option matches the user?",
            correct_answer="(b)",
            all_options=["(a) Tea", "(b) Matcha latte"],
            shared_context_id="ctx",
            end_index_in_shared_context=1,
        ),
        adapter=FakeAdapter(),
        contexts={"ctx": "system: User persona summary.\nuser: User: I like matcha latte."},
        memory_mode="symbolic",
        slot_assignment_mode="symbolic",
        learned_memory_checkpoint_dir=None,
        learned_memory_train_config_path=None,
        learned_memory_device="cpu",
        latent_retriever_checkpoint_dir=None,
        latent_retriever_device="cpu",
        learned_slot_assignment_checkpoint_dir=None,
        learned_slot_assignment_train_config_path=None,
        learned_slot_assignment_device="cpu",
        provider_configured=True,
        llm=SimpleNamespace(temperature=0.0, max_tokens=32),
    )
    assert row["provider_status"] == "completed"
    assert row["provider_prediction"] == "(b)"
    assert row["provider_raw_prediction"].startswith("raw::")
    assert row["memory_answer_local"] == "(b)"
    assert "Answer-head candidate:" in row["prompt"]
    assert "(b)" in row["prompt"]


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


def test_run_personamem_canary_uses_stage2_runtime_defaults_from_config(monkeypatch, tmp_path: Path):
    output_root = tmp_path / "outputs_v2"
    run_dir = output_root / "runs" / "defaulted_personamem"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "project:",
                "  name: core_mem",
                "  stage: stage2",
                "llm:",
                "  provider: openai_compatible",
                "  api_key_env: GPT_AGENT_API_KEY",
                "  base_url: https://example.com/v1",
                "  model: fake-model",
                "benchmarks:",
                "  primary: personamem",
                "  secondary: longmemeval_s",
                "  personamem:",
                "    data_root: unused",
                "stage2_runtime:",
                "  learned_memory_checkpoint_dir: outputs_v2/checkpoints/learned",
                "  learned_memory_train_config_path: configs/stage2_train.yaml",
                "  latent_retriever_checkpoint_dir: outputs_v2/checkpoints/latent",
                "  learned_slot_assignment_checkpoint_dir: outputs_v2/checkpoints/slot",
                "  learned_slot_assignment_train_config_path: configs/stage2_slot.yaml",
            ]
        ),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"sample_ids": ["q1"]}), encoding="utf-8")
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

    class _NoProvider:
        def is_configured(self) -> bool:
            return False

    class _FakeAdapter:
        def load_shared_contexts(self):
            return {"ctx": "user: I like sushi."}

        def load_questions(self):
            return questions

        def render_context_for_question(self, question, contexts):
            return contexts[question.shared_context_id]

    captured: dict[str, str | None] = {}

    def _fake_resolve_shared_predictors(**kwargs):
        captured.update(kwargs)
        return None, None, None

    def _fake_build_personamem_row(*args, **kwargs):
        question = args[0]
        return {
            "sample_id": question.question_id,
            "benchmark": "personamem",
            "question_type": question.question_type,
            "topic": question.topic,
            "expected_answer": question.correct_answer,
            "memory_answer_local": question.correct_answer,
            "memory_answer_local_baseline": "baseline",
            "provider_prediction": None,
            "provider_raw_prediction": None,
            "provider_status": "provider_not_configured",
            "provider_configured": False,
            "observed_turns": 1,
            "belief_source": "learned_memory",
            "selected_slot_ids": [],
            "belief_state": {"belief_items": []},
            "evidence_block": "",
            "prompt": "prompt",
            "prompt_version": "stage2_memory_canary_v2",
        }

    monkeypatch.setattr("run_stage2_memory_canary._provider_from_llm", lambda llm: _NoProvider())
    monkeypatch.setattr("run_stage2_memory_canary._ensure_canary_manifest", lambda output_root, benchmark: manifest_path)
    monkeypatch.setattr("run_stage2_memory_canary.PersonaMemAdapter", lambda data_root: _FakeAdapter())
    monkeypatch.setattr("run_stage2_memory_canary._resolve_shared_predictors", _fake_resolve_shared_predictors)
    monkeypatch.setattr("run_stage2_memory_canary._build_personamem_row", _fake_build_personamem_row)

    payload = run_personamem_canary(
        output_root=output_root,
        config_path=config_path,
        limit=1,
        memory_mode="learned_memory",
        slot_assignment_mode="learned",
        requested_run_dir=str(run_dir),
    )

    assert payload["status"] == "blocked_provider_not_configured"
    assert captured["learned_memory_checkpoint_dir"] == "outputs_v2/checkpoints/learned"
    assert captured["learned_memory_train_config_path"] == "configs/stage2_train.yaml"
    assert captured["latent_retriever_checkpoint_dir"] == "outputs_v2/checkpoints/latent"
    assert captured["learned_slot_assignment_checkpoint_dir"] == "outputs_v2/checkpoints/slot"
    assert captured["learned_slot_assignment_train_config_path"] == "configs/stage2_slot.yaml"


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
            "answer_head_candidate": "(b)",
        },
    )
    assert "(a) Wrong" in prompt
    assert "(b) Right" in prompt
    assert "Answer-head candidate:" in prompt
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


def test_personamem_local_projection_penalizes_unsupported_long_option_details():
    question = PersonaMemQuestion(
        persona_id="p",
        question_id="q",
        question_type="suggest_new_ideas",
        topic="music",
        user_question_or_message="How can I find a more fulfilling way to express my love for music?",
        correct_answer="(b)",
        all_options=[
            "(a) Consider getting involved in music criticism by writing album reviews after attending workshops and articulating detailed contexts and intentions behind albums.",
            "(b) You might consider exploring different avenues like writing about your musical journey or experimenting with performing live in settings that inspire you.",
            "(c) Collaborating with others who share your musical interests can also be a rewarding path, mixing traditional and electronic elements to expand your creative horizons.",
            "(d) Exploring sound engineering might offer a fulfilling way to express your love for music through digital remixes and chance meetings with audio engineers.",
        ],
        shared_context_id="ctx",
        end_index_in_shared_context=1,
    )
    projected = _project_personamem_local_answer(
        {
            "answer_text": "on a journey to redefine how i approach group collaborations, aiming for a more structured",
            "belief_state": {"belief_items": [{"relation": "other_fact", "value": "on a journey to redefine how i approach group collaborations, aiming for a more structured"}]},
            "evidence_block": "- other_fact: on a journey to redefine how i approach group collaborations, aiming for a more structured",
            "selected_slot_glosses": [
                "other_fact=on a journey to redefine how i approach group collaborations, aiming for a more structured"
            ],
        },
        question,
    )
    assert projected == "(b)"


def test_personamem_local_projection_downweights_generic_back_other_tokens_for_withdrawal_advice():
    question = PersonaMemQuestion(
        persona_id="p",
        question_id="q",
        question_type="generalizing_to_new_scenarios",
        topic="bookRecommendation",
        user_question_or_message="I've been involved in planning events for my community lately, but I'm not sure if I should continue with it. What do you think?",
        correct_answer="(c)",
        all_options=[
            "(a) It seems like you're questioning whether the time commitment aligns with your other responsibilities. Balancing event planning with personal activities can be challenging. It might be worthwhile to evaluate if this is the right time to continue, or if easing back could give you more freedom. What are your thoughts on adjusting your involvement?",
            "(b) It appears you're pondering your passion for event planning in your community. Community engagement is valuable, but it's also important to feel motivated and interested in what you're doing. Maybe explore different roles or activities that might reignite your enthusiasm. How does that sound to you?",
            "(c) It sounds like you might be feeling a bit overwhelmed or finding the process less enjoyable than before. If the structured planning is becoming tedious, perhaps focusing on more spontaneous or informal engagement within the community could be a refreshing change. Consider activities that require less meticulous planning and more personal interaction. How do you feel about trying something like that?",
            "(d) It sounds like you're in a period of reflection about what you enjoy most. Community events can be rewarding and stressful at the same time. Perhaps exploring other interests or roles can provide a sense of fulfillment without the pressure. Do you think trying out different activities might help you decide?",
        ],
        shared_context_id="ctx",
        end_index_in_shared_context=1,
    )
    projected = _project_personamem_local_answer(
        {
            "answer_text": "step back from structured book club settings",
            "belief_state": {
                "belief_items": [
                    {"relation": "other_fact", "value": "step back from structured book club settings"}
                ]
            },
            "evidence_block": "- other_fact: step back from structured book club settings",
            "selected_slot_glosses": [
                "other_fact=step back from structured book club settings",
                "other_fact=being boxed into a particular format that doesn't allow my thoughts to flow freely",
                "other_fact=opted out of future reading marathons",
            ],
        },
        question,
    )
    assert projected == "(c)"


def test_personamem_local_projection_prefers_recipe_expansion_over_generic_markets():
    question = PersonaMemQuestion(
        persona_id="p",
        question_id="q",
        question_type="recalling_facts_mentioned_by_the_user",
        topic="foodRecommendation",
        user_question_or_message="Can you suggest some new cooking techniques or recipes I might enjoy exploring?",
        correct_answer="(c)",
        all_options=[
            "(a) Since you've recently taken a cooking class and enjoyed learning about both the techniques and cultural histories behind recipes, you might appreciate exploring local farmers' markets. Discovering fresh, seasonal produce and learning about the farmers' stories could enhance your appreciation of ingredients. Have you tried visiting different markets to find unique ingredients?",
            "(b) Since you've recently taken a cooking class and enjoyed learning about both the techniques and cultural histories behind recipes, you might appreciate starting a food blog. Sharing your culinary journey and exchanging ideas with fellow enthusiasts could further enrich your experience. Have you tried writing about your own interpretations of the dishes you've learned?",
            "(c) Since you've recently taken a cooking class and enjoyed learning about both the techniques and cultural histories behind recipes, you might appreciate delving into fusion cuisines. Exploring how different cultures use similar ingredients in unique ways could further expand your culinary skills. Have you tried looking into how you can combine elements from the cuisines you learned about with others to create something entirely new?",
            "(d) Since you've recently taken a cooking class and enjoyed learning about both the techniques and cultural histories behind recipes, you might appreciate experimenting with molecular gastronomy. Understanding the science behind cooking could add a new dimension to your skills. Have you tried using techniques like spherification or sous vide to transform classic dishes?",
        ],
        shared_context_id="ctx",
        end_index_in_shared_context=1,
    )
    projected = _project_personamem_local_answer(
        {
            "answer_text": "opted out of cooking classes that i once enjoyed",
            "belief_state": {
                "belief_items": [
                    {"relation": "other_fact", "value": "opted out of cooking classes that i once enjoyed"}
                ]
            },
            "evidence_block": "- other_fact: opted out of cooking classes that i once enjoyed",
            "selected_slot_glosses": [
                "other_fact=opted out of cooking classes that i once enjoyed",
                "hobby=experimenting with different recipes",
                "other_fact=excited to learn various techniques that can facilitate this process",
            ],
        },
        question,
    )
    assert projected == "(c)"


def test_personamem_local_projection_repairs_interactional_other_fact_with_non_support_gloss():
    question = PersonaMemQuestion(
        persona_id="p",
        question_id="q",
        question_type="recalling_facts_mentioned_by_the_user",
        topic="movieRecommendation",
        user_question_or_message="What are some new trends in the film industry recently?",
        correct_answer="(c)",
        all_options=[
            "(a) It’s excellent to see your enthusiasm for cinema through your blog! Recently, the film industry has seen a surge in interest towards sustainable production methods, aiming to reduce the carbon footprint of movie-making. Additionally, with groundbreaking advancements in artificial intelligence, filmmakers are exploring how AI can be used in both production and screenplay writing. These subjects could lead to vibrant discussions in your blog’s comments section!",
            "(b) Your passion for movies is truly inspiring, especially as seen in your blog! Lately, there has been an increased focus on the use of blockchain technology in film financing and distribution, providing new avenues for indie filmmakers. Furthermore, partnerships between filmmakers and video game developers have been fruitful, leading to unique cross-medium stories. These trends might stimulate some engaging dialogue in the comments section of your blog!",
            "(c) It's great to see your passion for film shining through your blog! Recently, there's been a lot of buzz about the rise of virtual reality in filmmaking and its potential to revolutionize storytelling. Also, with the increasing focus on diversity, many studios are striving to bring more inclusive stories and perspectives to the forefront. I imagine these trends could spark some interesting discussions in the comments section of your blog!",
            "(d) Seeing your love for film come through in your blog is truly wonderful! Recently, there’s been an exciting shift towards interactive storytelling in cinema, allowing audiences to influence the narrative. Another trend is the expanding influence of mobile filmmaking, where filmmakers are leveraging smartphone technology for creativity. These topics could certainly generate lively conversations in your blog's comments section!",
        ],
        shared_context_id="ctx",
        end_index_in_shared_context=1,
    )
    projected = _project_personamem_local_answer(
        {
            "answer_text": "looking for some movie recommendations based on my recent activities",
            "belief_state": {
                "belief_items": [
                    {
                        "relation": "other_fact",
                        "value": "looking for some movie recommendations based on my recent activities",
                    }
                ]
            },
            "evidence_block": "- other_fact: looking for some movie recommendations based on my recent activities",
            "selected_slot_glosses": [
                "other_fact=looking for some movie recommendations based on my recent activities",
                "other_fact=accustomed to straightforward, logical approaches to organization",
                "hobby=unearthing these narratives",
                "other_fact=seeing how different people interpret the same material in unique ways",
                "other_fact=feeling particularly enthusiastic about this journey because i believe that visual storytelling is crucial in today's media landscape",
                "other_fact=getting positive feedback from my peers",
            ],
            "support_slot_glosses": [
                "other_fact=looking for some movie recommendations based on my recent activities"
            ],
        },
        question,
    )
    assert projected == "(c)"


def test_personamem_local_projection_skips_interactional_alternatives_before_non_support_repair():
    question = PersonaMemQuestion(
        persona_id="p",
        question_id="q",
        question_type="track_full_preference_evolution",
        topic="datingConsultation",
        user_question_or_message=(
            "I also stopped my photography classes since they became too pressured to capture perfect moments, "
            "which turned into a stressor. The structured environment of those classes seemed to weigh heavily "
            "on my creativity, making it feel like an obligation rather than a joyful pursuit. I found that each "
            "session was less about exploring my artistic eye and more about meeting expected standards, which "
            "just drained the excitement out of the whole experience. It was disheartening, particularly when I "
            "would spend countless hours preparing only to feel inadequate after each critique. Ultimately, I "
            "realized that I could freely explore my passion without the constraints of formal education, which "
            "kindled a new sense of liberty in how I engage with photography. Now, I am now focusing more on "
            "spontaneous moments, allowing my surroundings and mood to guide my lens."
        ),
        correct_answer="(d)",
        all_options=[
            "(a) I understand that your journey with photography has evolved from never having any particular interest in it to suddenly developing a deep preference due to challenges. In the beginning, you were indifferent to photography, but the structured nature of photography classes piqued your interest, leading you to embrace capturing moments. Now, it seems you've found a renewed sense of freedom by moving away from formal instruction, allowing you to reconnect with photography on your own terms, focusing on spontaneous moments driven by your surroundings and mood.",
            "(b) I understand that your journey with photography has evolved from initially disliking it to experiencing challenges that shifted your preference. In the beginning, you did not enjoy photography, but the structured nature of photography classes introduced new perspectives that allowed you to appreciate it more, leading you to find enjoyment. Now, it seems you've found a renewed sense of freedom by moving away from formal instruction, allowing you to reconnect with photography on your own terms, focusing on spontaneous moments driven by your surroundings and mood.",
            "(c) I understand that your journey with photography has evolved from initially liking it to continuously enjoying the experience without challenges that shifted your preference. In the beginning, you embraced photography, finding joy in capturing moments. Over time, the structured nature of photography classes reinforced this joy and appreciation for the art. Now, it seems you've found a renewed sense of freedom by moving away from formal instruction, allowing you to reconnect with photography on your own terms, focusing on spontaneous moments driven by your surroundings and mood.",
            "(d) I understand that your journey with photography has evolved from initially liking it to experiencing challenges that shifted your preference. In the beginning, you embraced photography, finding joy in capturing moments. However, over time, the structured nature of photography classes introduced pressures and expectations that detracted from your artistic enjoyment, leading you to dislike the constraints it imposed. Now, it seems you've found a renewed sense of freedom by moving away from formal instruction, allowing you to reconnect with photography on your own terms, focusing on spontaneous moments driven by your surroundings and mood.",
        ],
        shared_context_id="ctx",
        end_index_in_shared_context=1,
    )
    projected = _project_personamem_local_answer(
        {
            "answer_text": "really hoping that through this experience, i can find like-minded individuals who share similar values",
            "belief_state": {
                "belief_items": [
                    {
                        "relation": "other_fact",
                        "value": "really hoping that through this experience, i can find like-minded individuals who share similar values",
                    }
                ]
            },
            "evidence_block": (
                "- other_fact: really hoping that through this experience, i can find like-minded individuals who share similar values"
            ),
            "selected_slot_glosses": [
                "other_fact=looking to refactor my code to make it more readable",
                "other_fact=face-to-face discussions where ideas can be exchanged more dynamically, allowing for a richer understanding of one another's viewpoints",
                "other_fact=more practical applications of law, where i can see how policies affect everyday lives",
                "other_fact=to keep things straightforward",
                "other_fact=actively analyzing trends",
                "other_fact=really hoping that through this experience, i can find like-minded individuals who share similar values",
            ],
            "support_slot_glosses": [
                "other_fact=really hoping that through this experience, i can find like-minded individuals who share similar values"
            ],
        },
        question,
    )
    assert projected == "(d)"


def test_personamem_local_projection_recovers_preference_evolution_option_order():
    question = PersonaMemQuestion(
        persona_id="p",
        question_id="q",
        question_type="track_full_preference_evolution",
        topic="musicRecommendation",
        user_question_or_message=(
            "After several disagreements over the artistic direction, I felt stifled, which really discouraged me from that collaborative process. "
            "It was as if the creative vision I had in mind was at odds with the direction my bandmates wanted to pursue. "
            "I remember those discussions often being emotionally charged, where opinions clashed and passion ran high, but ultimately, it was clear that I couldn't contribute to something that did not resonate with my artistic essence. "
            "Feeling restricted in that environment made me question not just my contributions, but also my abilities and instincts as an artist. "
            "This inner turmoil prompted a desire for more autonomy over my own creative endeavors, leading to a pivotal decision to explore my artistry in a more personal and independent setting."
        ),
        correct_answer="(a)",
        all_options=[
            "(a) I understand that initially, you enjoyed collaborating with other musicians, finding it an enriching creative experience. However, over time, your preferences evolved due to feeling stifled by disagreements over artistic direction. This led to a shift away from collaboration, as you felt constrained and questioned your artistic abilities and instincts. Your journey reflects a transition from a collaborative approach to seeking independence in order to pursue your creative endeavors in a manner that resonates with your artistic essence.",
            "(b) You initially felt constrained by disagreements over artistic direction, feeling stifled and questioning your artistic abilities and instincts. Afterwards, you went on to enjoy collaborating with other musicians, as it was an enriching creative experience, before ultimately seeking independence to pursue your creative endeavors.",
            "(c) At first, you questioned your artistic abilities and instincts, feeling stifled by disagreements over artistic direction in collaborations. Nevertheless, over time, you began enjoying collaboration with other musicians as an enriching creative experience, before eventually transitioning to independence to fully resonate with your true creative essence.",
            "(d) Initially, you felt constrained by disagreements over artistic direction, which made you question your artistic abilities and instincts. However, over time, you found collaborating with other musicians to be an enriching creative experience. This led to a shift towards collaboration, as you sought creative endeavors that resonate with your artistic essence.",
        ],
        shared_context_id="ctx",
        end_index_in_shared_context=1,
    )
    projected = _project_personamem_local_answer(
        {
            "answer_text": "able to create a unique sound that pays homage to my roots while also pushing the boundaries of what is traditionally expected",
            "belief_state": {
                "belief_items": [
                    {
                        "relation": "other_fact",
                        "value": "able to create a unique sound that pays homage to my roots while also pushing the boundaries of what is traditionally expected",
                    }
                ]
            },
            "evidence_block": "- other_fact: able to create a unique sound that pays homage to my roots while also pushing the boundaries of what is traditionally expected",
            "selected_slot_glosses": [
                "music_preference=enjoy collaborating with other musicians, finding it an enriching creative experience",
                "other_fact=able to create a unique sound that pays homage to my roots while also pushing the boundaries of what is traditionally expected",
                "music_preference=seeking independence in order to pursue my creative endeavors",
            ],
        },
        question,
    )
    assert projected == "(a)"


def test_personamem_local_projection_matches_music_production_morphology():
    question = PersonaMemQuestion(
        persona_id="p",
        question_id="q",
        question_type="suggest_new_ideas",
        topic="musicRecommendation",
        user_question_or_message=(
            "I'm exploring new creative outlets and would love some suggestions. "
            "Any ideas on activities that can capture emotions and ideas effectively?"
        ),
        correct_answer="(b)",
        all_options=[
            "(a) Another route you might consider is engaging in a collaborative project that involves exploring and blending different musical traditions. Much like how conversations with like-minded individuals can spark creativity and lead to new artistic endeavors, working with others on fusion music projects can blend unique perspectives, offering a fresh tapestry of sound that captures emotions and ideas effectively. Such collaborations might lead to artistic growth and open the door to new projects where diverse influences intertwine, creating a vibrant expression that resonates with audiences.",
            "(b) Have you considered diving deeper into the various aspects of music production or learning about different genres and styles? Experimenting with new techniques or instruments could also offer fresh ways to express yourself creatively.",
            "(c) Consider trying your hand at musical storytelling through documentaries. While some might feel lacking, there are incredible narratives to explore that highlight the authentic experiences of artists and the depth of their artistry. You could create a documentary that goes beyond spectacle to spotlight genuine stories, making space for underrepresented voices. Embarking on such a project could not only capture emotions and ideas effectively but also offer viewers a richer, more impactful connection to music and its cultural context.",
            "(d) Exploring curated playlists or creating your own can be a creative outlet that allows you to appreciate and highlight artists from underrepresented genres or regions. By spotlighting talented individuals who bring unique cultural elements through fusion genres, you engage emotionally with the diversity and innovation in music. This approach not only helps capture emotions and ideas but also deepens your understanding of the music industry's rich landscape. It's a rewarding way to connect with the artistry and reflect these emotions in your musical journey.",
        ],
        shared_context_id="ctx",
        end_index_in_shared_context=1,
    )
    projected = _project_personamem_local_answer(
        {
            "answer_text": "producing music with software",
            "belief_state": {
                "belief_items": [
                    {"relation": "music_preference", "value": "producing music with software"}
                ]
            },
            "evidence_block": "- music_preference: producing music with software",
            "selected_slot_glosses": ["music_preference=producing music with software"],
        },
        question,
    )
    assert projected == "(b)"


def test_personamem_local_projection_generalizes_structured_withdrawal_to_race_scenario():
    question = PersonaMemQuestion(
        persona_id="p",
        question_id="q",
        question_type="generalizing_to_new_scenarios",
        topic="bookRecommendation",
        user_question_or_message=(
            "I've been thinking about joining a 5k race that my friends are signing up for, "
            "but I'm not sure if it's something I should do. What do you think?"
        ),
        correct_answer="(d)",
        all_options=[
            "(a) Before deciding, consider how you respond to competition and social settings. A race can be a fun and rewarding goal, offering a chance to connect with your friends, but if that doesn't sound appealing, focusing on personal achievements and fitness milestones might be more suitable.",
            "(b) Thinking about joining a 5k race is a great opportunity to challenge yourself and have fun with friends. If the idea excites you, it might be a good way to bond while also setting personal fitness goals. Remember, it's all about participating and enjoying the experience more than anything else.",
            "(c) If you're feeling unsure, it might help to think about what you enjoy most in activities. Participating in an event with friends can be motivating, and trying something new could bring a fresh and exciting change to your routine. Enjoyment and camaraderie often make the experience worthwhile.",
            "(d) If you're feeling hesitant, maybe it's worth considering how you like to approach physical activities. If being part of a structured event feels overwhelming or stressful, it might be better to enjoy the activity at your own pace, like running at your preferred time and setting your own goals. Finding enjoyment in the journey without the pressure to keep up with others can be really fulfilling.",
        ],
        shared_context_id="ctx",
        end_index_in_shared_context=1,
    )
    projected = _project_personamem_local_answer(
        {
            "answer_text": "step back from structured book club settings",
            "belief_state": {
                "belief_items": [
                    {"relation": "other_fact", "value": "step back from structured book club settings"}
                ]
            },
            "evidence_block": "- other_fact: step back from structured book club settings",
            "selected_slot_glosses": ["other_fact=step back from structured book club settings"],
        },
        question,
    )
    assert projected == "(d)"


def test_personamem_local_projection_breaks_truest_music_tie_toward_first_matching_option():
    question = PersonaMemQuestion(
        persona_id="p",
        question_id="q",
        question_type="provide_preference_aligned_recommendations",
        topic="musicRecommendation",
        user_question_or_message=(
            "I'm working on a new project blending cultural elements with modern music styles and "
            "I'm curious what fresh approaches or techniques I could explore to really captivate an audience. "
            "Any creative recommendations?"
        ),
        correct_answer="(a)",
        all_options=[
            "(a) To truly captivate your audience with a fusion of traditional and modern sounds, consider experimenting with a symphonic electronic style. This approach can elevate traditional melodies by embedding them within sweeping orchestral arrangements, paired with rhythmic electronic beats that offer a contemporary edge. Try collaborating with artists who specialize in electronic symphony-a genre that perfectly marries the depth of classical compositions with the energy of electronic music. This could not only highlight the richness of your cultural heritage but also attract a wide audience that appreciates intricate, harmonious blends. By focusing on a cohesive narrative within your remix, you could address creative differences by setting a clear artistic direction and shared aesthetic goal from the outset.",
            "(b) To truly captivate your audience with a fusion of traditional and modern sounds, consider experimenting with a symphonic electronic style. This approach can elevate traditional melodies by embedding them within sweeping orchestral arrangements, paired with rhythmic electronic beats that offer a contemporary edge. Try collaborating with artists who specialize in electronic symphony-a genre that perfectly marries the depth of classical compositions with the energy of electronic music. This could not only highlight the richness of your cultural heritage but also attract a wide audience that appreciates intricate, harmonious blends. By focusing on a cohesive narrative within your remix, you could address creative differences by setting a clear artistic direction and shared aesthetic goal from the outset.",
            "(c) One approach could be to dive into traditional Polynesian dances as the central element of your project. Use these movements to create a visually captivating performance, incorporating them with gentle acoustic guitar melodies that capture the essence of island life. Collaborate with other Pacific Island artists who focus on traditional dance music, ensuring the cultural integrity of your work is preserved. This approach not only emphasizes your connection to the rich heritage of the Pacific Islands but can also draw in audiences interested in cultural diversity. By keeping the performance rooted in authentic expressions, you can establish a clear, engaging narrative that respects and celebrates traditional practices.",
            "(d) Consider hosting intimate live performances focusing on acoustic instruments and unplugged sets. This tactile experience can enhance the authenticity of blending traditional and modern sounds, offering audiences a raw, unaffected connection to your music. Collaborate with folk and jazz artists who excel in spontaneous, improvisational performances to infuse a natural, storytelling element into your songs. This method may help uncover new nuances within your music as you capture the spontaneous magic of live session recordings. By prioritizing acoustic soundscapes, you could navigate creative differences by embracing organic, real-time creation as a shared objective.",
        ],
        shared_context_id="ctx",
        end_index_in_shared_context=1,
    )
    projected = _project_personamem_local_answer(
        {
            "answer_text": "music in its truest form, without rigid guidelines dictating how i should dissect it",
            "belief_state": {
                "belief_items": [
                    {
                        "relation": "music_preference",
                        "value": "music in its truest form, without rigid guidelines dictating how i should dissect it",
                    }
                ]
            },
            "evidence_block": (
                "- music_preference: music in its truest form, without rigid guidelines dictating how i should dissect it"
            ),
            "selected_slot_glosses": [
                "music_preference=more drawn to the emotional aspects of music, like the storytelling elements in lyrics or the feelings evoked by melodies",
                "other_fact=on a journey to redefine how i approach group collaborations, aiming for a more structured",
                "other_fact=genuinely hopeful about the potential outcomes of these interactions, as they could open doors to opportunities i never considered before",
                "music_preference=committing to invest my time in the creation of original music, focusing on my unique sound rather than merely reinterpreting the pieces of other artists",
                "music_preference=music in its truest form, without rigid guidelines dictating how i should dissect it"
            ],
            "support_slot_glosses": [
                "music_preference=music in its truest form, without rigid guidelines dictating how i should dissect it"
            ],
        },
        question,
    )
    assert projected == "(a)"


def test_memory_payload_collects_only_belief_support_slot_glosses():
    belief_state = {
        "belief_items": [
            {
                "relation": "music_preference",
                "value": "music in its truest form, without rigid guidelines dictating how i should dissect it",
                "support_slot_ids": ["slot_keep"],
            }
        ]
    }

    class FakeBeliefState:
        def to_dict(self):
            return belief_state

    fake_system = SimpleNamespace(
        query=lambda query_id, query_text: SimpleNamespace(
            belief_state=FakeBeliefState(),
            belief_source="learned_memory",
            evidence_block="- music_preference: music in its truest form, without rigid guidelines dictating how i should dissect it",
            answer_text="music in its truest form, without rigid guidelines dictating how i should dissect it",
            selected_slots=[
                SimpleNamespace(slot_id="slot_noise", canonical_gloss="other_fact=on a journey to redefine how i approach group collaborations, aiming for a more structured"),
                SimpleNamespace(slot_id="slot_keep", canonical_gloss="music_preference=music in its truest form, without rigid guidelines dictating how i should dissect it"),
            ],
            composed_memory="",
        )
    )

    payload = _memory_payload(fake_system, "q", "Which approach fits best?")

    assert payload["selected_slot_ids"] == ["slot_noise", "slot_keep"]
    assert payload["support_slot_glosses"] == [
        "music_preference=music in its truest form, without rigid guidelines dictating how i should dissect it"
    ]


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
            "answer_head_candidate": "(a)",
        },
    )
    assert "Latent matcher candidate" not in prompt
    assert "Answer-head candidate:" in prompt


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
            "answer_head_candidate": "the sports store downtown",
        },
    )
    assert "Return only the shortest exact answer phrase supported by the belief state." in prompt
    assert "Do not add trailing punctuation." in prompt
    assert "Answer-head candidate:" in prompt
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


def test_stage2_memory_canary_writes_v32_answer_head_aliases(monkeypatch, tmp_path: Path):
    output_root = tmp_path / "outputs_v2"
    run_dir = output_root / "runs" / "v32_answer_personamem"
    questions = [
        PersonaMemQuestion(
            persona_id="p1",
            question_id="q1",
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
    ]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"sample_ids": ["q1"]}), encoding="utf-8")

    class _NoProvider:
        def is_configured(self) -> bool:
            return False

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
    monkeypatch.setattr("run_stage2_memory_canary._provider_from_llm", lambda llm: _NoProvider())
    monkeypatch.setattr("run_stage2_memory_canary._ensure_canary_manifest", lambda output_root, benchmark: manifest_path)

    class _FakeAdapter:
        def load_shared_contexts(self):
            return {"ctx": "user: I like hiking mountain trails."}

        def load_questions(self):
            return questions

        def render_context_for_question(self, question, contexts):
            return contexts[question.shared_context_id]

    monkeypatch.setattr("run_stage2_memory_canary.PersonaMemAdapter", lambda data_root: _FakeAdapter())

    payload = run_personamem_canary(
        output_root=output_root,
        config_path=REPO_ROOT / "configs" / "minimax_m27.yaml",
        limit=1,
        requested_run_dir=str(run_dir),
    )

    assert payload["status"] == "blocked_provider_not_configured"
    assert payload["local_exact_match"] == 1
    assert payload["local_baseline_exact_match"] == 0
    answer_eval = output_root / "artifacts" / "latest_stage2_v32_answer_head_eval.json"
    compare = output_root / "artifacts" / "latest_stage2_v32_option_scoring_compare.json"
    assert answer_eval.exists()
    assert compare.exists()
    compare_payload = json.loads(compare.read_text(encoding="utf-8"))
    assert compare_payload["positive_gain"] is True
    assert compare_payload["delta_local_exact_match"] == 1


def test_stage2_memory_canary_writes_v33_answer_option_aliases_for_learned_runtime(monkeypatch, tmp_path: Path):
    output_root = tmp_path / "outputs_v2"
    run_dir = output_root / "runs" / "learned_personamem"
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

    class _NoProviderForV33:
        def is_configured(self) -> bool:
            return False

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
    monkeypatch.setattr("run_stage2_memory_canary._provider_from_llm", lambda llm: _NoProviderForV33())
    monkeypatch.setattr("run_stage2_memory_canary._ensure_canary_manifest", lambda output_root, benchmark: manifest_path)

    class _FakeAdapter:
        def load_shared_contexts(self):
            return {"ctx": "user: I like sushi."}

        def load_questions(self):
            return questions

        def render_context_for_question(self, question, contexts):
            return contexts[question.shared_context_id]

    monkeypatch.setattr("run_stage2_memory_canary.PersonaMemAdapter", lambda data_root: _FakeAdapter())
    monkeypatch.setattr("run_stage2_memory_canary._resolve_shared_predictors", lambda **kwargs: (None, None, None))

    def _fake_build_personamem_row(*args, **kwargs):
        question = args[0]
        return {
            "sample_id": question.question_id,
            "benchmark": "personamem",
            "question_type": question.question_type,
            "topic": question.topic,
            "expected_answer": question.correct_answer,
            "memory_answer_local": question.correct_answer,
            "memory_answer_local_baseline": "baseline",
            "provider_prediction": None,
            "provider_raw_prediction": None,
            "provider_status": "provider_not_configured",
            "provider_configured": False,
            "observed_turns": 1,
            "belief_source": "learned_memory",
            "selected_slot_ids": [],
            "belief_state": {"belief_items": []},
            "evidence_block": "",
            "prompt": "prompt",
            "prompt_version": "stage2_memory_canary_v2",
        }

    monkeypatch.setattr("run_stage2_memory_canary._build_personamem_row", _fake_build_personamem_row)

    payload = run_personamem_canary(
        output_root=output_root,
        config_path=REPO_ROOT / "configs" / "minimax_m27.yaml",
        limit=1,
        memory_mode="learned_memory",
        slot_assignment_mode="learned",
        requested_run_dir=str(run_dir),
    )

    assert payload["status"] == "blocked_provider_not_configured"
    artifact = output_root / "artifacts" / "latest_stage2_v33_answer_option_eval.json"
    assert artifact.exists()
    artifact_payload = json.loads(artifact.read_text(encoding="utf-8"))
    assert artifact_payload["memory_mode"] == "learned_memory"
    assert artifact_payload["slot_assignment_mode"] == "learned"
    assert artifact_payload["positive_gain"] is True


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
                "provider_prediction": "<think>draft</think>\n(a)",
                "provider_raw_prediction": "<think>draft</think>\n(a)",
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
    assert rows[0]["provider_prediction"] == "(a)"
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["resumed_prediction_count"] == 1
    assert metadata["completed_predictions"] == 2
    assert metadata["live_predictions_completed"] == 2
    assert payload["provider_exact_match"] == 2


def test_stage2_memory_canary_partial_provider_error_commits_successes_and_resume_recovers(monkeypatch, tmp_path: Path):
    output_root = tmp_path / "outputs_v2"
    run_dir = output_root / "runs" / "partial_personamem"
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

    call_count = {"value": 0}

    def _flaky_provider_chat_request(llm, prompt):
        call_count["value"] += 1
        if call_count["value"] >= 2:
            raise RuntimeError("transient provider failure")
        return "(a)", "raw"

    monkeypatch.setattr("run_stage2_memory_canary._provider_chat_request", _flaky_provider_chat_request)

    first = run_personamem_canary(
        output_root=output_root,
        config_path=REPO_ROOT / "configs" / "minimax_m27.yaml",
        limit=2,
        requested_run_dir=str(run_dir),
        resume=True,
    )

    assert first["status"] == "partial_provider_error"
    assert first["completed_predictions"] == 1
    assert first["failed_predictions_pending"] == 1
    predictions_path = run_dir / "predictions.jsonl"
    assert len(predictions_path.read_text(encoding="utf-8").strip().splitlines()) == 1

    monkeypatch.setattr("run_stage2_memory_canary._provider_chat_request", lambda llm, prompt: ("(a)", "raw"))
    second = run_personamem_canary(
        output_root=output_root,
        config_path=REPO_ROOT / "configs" / "minimax_m27.yaml",
        limit=2,
        requested_run_dir=str(run_dir),
        resume=True,
    )

    assert second["status"] == "completed"
    assert second["completed_predictions"] == 2
    assert second["failed_predictions_pending"] == 0


def test_stage2_memory_canary_resolves_shared_predictors_once(monkeypatch):
    calls = {"build": 0, "belief": 0, "slot": 0, "latent": 0}
    shared_belief = object()
    shared_slot = object()
    shared_latent = object()

    class _TemplateSystem:
        def _resolve_learned_belief_predictor(self):
            calls["belief"] += 1
            return shared_belief

        def _resolve_slot_assignment_predictor(self):
            calls["slot"] += 1
            return shared_slot

        def _resolve_latent_slot_ranker(self):
            calls["latent"] += 1
            return shared_latent

    def _fake_build_memory_system(**_kwargs):
        calls["build"] += 1
        return _TemplateSystem()

    monkeypatch.setattr("run_stage2_memory_canary._build_memory_system", _fake_build_memory_system)

    belief_predictor, slot_predictor, latent_ranker = _resolve_shared_predictors(
        memory_mode="learned_memory",
        slot_assignment_mode="learned",
        learned_memory_checkpoint_dir="checkpoint",
        learned_memory_train_config_path="config.yaml",
        learned_memory_device="cpu",
        latent_retriever_checkpoint_dir="latent-checkpoint",
        latent_retriever_device="cpu",
        learned_slot_assignment_checkpoint_dir="checkpoint",
        learned_slot_assignment_train_config_path="config.yaml",
        learned_slot_assignment_device="cpu",
    )

    assert belief_predictor is shared_belief
    assert slot_predictor is shared_slot
    assert latent_ranker is shared_latent
    assert calls == {"build": 1, "belief": 1, "slot": 1, "latent": 1}
