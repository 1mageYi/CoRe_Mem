"""Run a stage-2 memory-mediated benchmark canary with honest artifact logging."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core_mem.benchmarks.longmemeval import LongMemEvalAdapter, LongMemEvalQuestion
from core_mem.benchmarks.personamem import PersonaMemAdapter, PersonaMemQuestion
from core_mem.config import LLMConfig, load_project_config
from core_mem.providers.openai_compatible import OpenAICompatibleConfig, OpenAICompatibleProvider
from core_mem.v2.system import StructuredMemorySystem
from run_stage2_canary import build_canary_manifests


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _current_commit_hash() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _append_jsonl_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def _exact_match(expected: str, prediction: str | None) -> bool:
    return _normalize_answer(expected) == _normalize_answer(prediction)


def _label_prefix_match(expected: str, prediction: str | None) -> bool:
    normalized_expected = _normalize_answer(expected)
    normalized_prediction = _normalize_answer(prediction)
    return bool(normalized_expected and normalized_prediction.startswith(normalized_expected))


def _prediction_metrics(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "provider_exact_match": sum(
            1 for row in rows if _exact_match(str(row.get("expected_answer", "")), row.get("provider_prediction"))
        ),
        "provider_label_prefix_match": sum(
            1 for row in rows if _label_prefix_match(str(row.get("expected_answer", "")), row.get("provider_prediction"))
        ),
        "local_exact_match": sum(
            1 for row in rows if _exact_match(str(row.get("expected_answer", "")), row.get("memory_answer_local"))
        ),
    }


def _copy_config_snapshot(config_path: Path, run_dir: Path) -> Path:
    snapshot = run_dir / "config_snapshot.yaml"
    snapshot.write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    return snapshot


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def _latest_manifest(output_root: Path, benchmark: str) -> Path | None:
    candidates = sorted((output_root / "evals_benchmark").glob(f"*_{benchmark}_canary.json"))
    return candidates[-1] if candidates else None


def _ensure_canary_manifest(output_root: Path, benchmark: str) -> Path:
    latest = _latest_manifest(output_root, benchmark)
    if latest is not None:
        return latest
    payload = build_canary_manifests(
        output_root,
        personamem_size=64,
        longmemeval_size=64,
        personamem_questions=REPO_ROOT / "data/personamem/questions_32k.csv",
        personamem_contexts=REPO_ROOT / "data/personamem/shared_contexts_32k.jsonl",
        longmemeval_dataset=REPO_ROOT / "data/longmemeval/longmemeval_s_cleaned.json",
    )
    if benchmark == "personamem":
        return Path(str(payload["personamem_manifest"]))
    return Path(str(payload["longmemeval_manifest"]))


def _resolve_run_dir(output_root: Path, benchmark: str, requested_run_dir: str | None = None) -> Path:
    if requested_run_dir:
        run_dir = Path(requested_run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir
    stamp = _timestamp()
    run_dir = output_root / "runs" / f"{stamp}_stage2_memory_canary_{benchmark}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _provider_from_llm(llm: LLMConfig) -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        OpenAICompatibleConfig(
            api_key_env=llm.api_key_env,
            base_url=llm.base_url,
            model=llm.model,
            timeout_seconds=llm.timeout_seconds,
            max_retries=llm.max_retries,
            retry_backoff_seconds=llm.retry_backoff_seconds,
            min_request_interval_seconds=llm.min_request_interval_seconds,
            max_retry_delay_seconds=llm.max_retry_delay_seconds,
        )
    )


def _provider_chat_request(
    llm: LLMConfig,
    prompt: str,
) -> tuple[str, str | None]:
    provider = _provider_from_llm(llm)
    response = provider.chat(
        prompt,
        temperature=llm.temperature,
        max_tokens=llm.max_tokens,
    )
    return response.content, getattr(response, "content", None)


def _iter_provider_predictions(
    llm: LLMConfig,
    prompts: list[tuple[int, str]],
    *,
    workers: int,
) -> list[tuple[int, str, str | None]]:
    if workers <= 1:
        results: list[tuple[int, str, str | None]] = []
        for index, prompt in prompts:
            prediction, raw_prediction = _provider_chat_request(llm, prompt)
            results.append((index, prediction, raw_prediction))
        return results

    results = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {
            executor.submit(_provider_chat_request, llm, prompt): index
            for index, prompt in prompts
        }
        for future in as_completed(future_map):
            index = future_map[future]
            prediction, raw_prediction = future.result()
            results.append((index, prediction, raw_prediction))
    return results


def _normalize_answer(text: str | None) -> str:
    if text is None:
        return ""
    return " ".join(text.strip().lower().split())


def _options_use_labels(options: list[str]) -> bool:
    return bool(options) and all(option.startswith("(") and ")" in option[:4] for option in options)


def _option_label(option: str) -> str:
    closing = option.find(")")
    return option[: closing + 1].strip() if option.startswith("(") and closing > 0 else option.strip()


def _option_body(option: str) -> str:
    label = _option_label(option)
    remainder = option[len(label) :].strip()
    return remainder or option.strip()

def _observe_personamem_context(system: StructuredMemorySystem, context_text: str, *, sample_id: str) -> int:
    observed = 0
    for turn_idx, line in enumerate(context_text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        lowered = stripped.lower()
        if lowered.startswith("assistant:"):
            speaker = "assistant"
        elif lowered.startswith("system:"):
            speaker = "system"
        else:
            speaker = "user"
        content = stripped.split(":", 1)[1].strip() if ":" in stripped else stripped
        if not content:
            continue
        system.observe_turn(
            content,
            source_dataset="personamem",
            source_dialogue_id=sample_id,
            source_turn_id=f"turn-{turn_idx}",
            session_id=sample_id,
            timestamp=f"2026-04-14T00:{turn_idx:02d}:00Z",
            speaker=speaker,
        )
        observed += 1
        if speaker == "system":
            for rewrite_idx, rewrite in enumerate(_rewrite_persona_summary(content), start=1):
                system.observe_turn(
                    rewrite,
                    source_dataset="personamem",
                    source_dialogue_id=sample_id,
                    source_turn_id=f"turn-{turn_idx}-persona-{rewrite_idx}",
                    session_id=sample_id,
                    timestamp=f"2026-04-14T00:{turn_idx:02d}:{rewrite_idx:02d}Z",
                    speaker="user",
                )
                observed += 1
    return observed


def _rewrite_persona_summary(text: str) -> list[str]:
    rewrites: list[str] = []
    normalized = " ".join(text.split())

    occupation = re.search(r"is a \d+-year-old ([^.]+?)(?: with|\.|,)", normalized, flags=re.IGNORECASE)
    if occupation:
        rewrites.append(f"I am a {occupation.group(1).strip()}.")

    passion = re.search(r"with a passion for ([^.]+?)(?:\.|,| and )", normalized, flags=re.IGNORECASE)
    if passion:
        rewrites.append(f"I enjoy {passion.group(1).strip()}.")

    experimenting = re.search(r"experimenting with ([^.]+?)(?:,|\.| aiming)", normalized, flags=re.IGNORECASE)
    if experimenting:
        rewrites.append(f"I enjoy experimenting with {experimenting.group(1).strip()}.")

    weekends = re.search(r"spends (?:his|her|their) weekends ([^.]+?)(?:,|\.| always)", normalized, flags=re.IGNORECASE)
    if weekends:
        rewrites.append(f"In my free time I {weekends.group(1).strip()}.")

    goal = re.search(r"ultimate goal is to ([^.]+)", normalized, flags=re.IGNORECASE)
    if goal:
        rewrites.append(f"I want to {goal.group(1).strip()}.")

    deduped: list[str] = []
    seen: set[str] = set()
    for rewrite in rewrites:
        compact = " ".join(rewrite.lower().split())
        if compact in seen:
            continue
        seen.add(compact)
        deduped.append(rewrite)
    return deduped


def _observe_longmemeval_context(system: StructuredMemorySystem, sessions: list[list[dict[str, Any]]], *, sample_id: str) -> int:
    observed = 0
    turn_idx = 0
    for session_idx, session in enumerate(sessions, start=1):
        for message in session:
            content = str(message.get("content", "")).strip()
            if not content:
                continue
            turn_idx += 1
            system.observe_turn(
                content,
                source_dataset="longmemeval_s",
                source_dialogue_id=sample_id,
                source_turn_id=f"session-{session_idx}-turn-{turn_idx}",
                session_id=sample_id,
                timestamp=f"2026-04-14T01:{turn_idx:02d}:00Z",
                speaker=str(message.get("role", "user")),
            )
            observed += 1
    return observed


def _resolve_personamem_prediction(local_answer: str, options: list[str]) -> str:
    normalized_local = _normalize_answer(local_answer)
    if not options:
        return local_answer
    if _options_use_labels(options):
        for option in options:
            label = _option_label(option)
            if normalized_local == _normalize_answer(label):
                return label
    for option in options:
        normalized_option = _normalize_answer(option)
        normalized_body = _normalize_answer(_option_body(option))
        if normalized_local and (
            normalized_local in normalized_option
            or normalized_local in normalized_body
            or normalized_option in normalized_local
            or normalized_body in normalized_local
        ):
            return _option_label(option) if _options_use_labels(options) else option
    query_terms = {token for token in normalized_local.split() if len(token) >= 4}
    if query_terms:
        best_option = ""
        best_overlap = 0
        for option in options:
            body_terms = set(_normalize_answer(_option_body(option)).split())
            overlap = len(query_terms & body_terms)
            if overlap > best_overlap:
                best_overlap = overlap
                best_option = option
        if best_overlap > 0:
            return _option_label(best_option) if _options_use_labels(options) else best_option
    return local_answer


def _project_personamem_local_answer(memory_payload: dict[str, Any], question: PersonaMemQuestion) -> str:
    projected = _resolve_personamem_prediction(memory_payload["answer_text"], question.all_options)
    if projected in question.all_options:
        return _option_label(projected) if _options_use_labels(question.all_options) else projected
    if projected != memory_payload["answer_text"]:
        return projected
    belief_text = " ".join(
        str(item.get("value", ""))
        for item in memory_payload["belief_state"].get("belief_items", [])
        if isinstance(item, dict)
    )
    evidence_text = f"{belief_text} {memory_payload['evidence_block']}".strip()
    return _resolve_personamem_prediction(evidence_text, question.all_options)


def _render_personamem_options(options: list[str]) -> str:
    if _options_use_labels(options):
        return "\n".join(options)
    return "\n".join(f"{idx + 1}. {option}" for idx, option in enumerate(options))


def _personamem_answer_instruction(options: list[str]) -> str:
    if _options_use_labels(options):
        return "Return only the best option label, for example (a)."
    return "Return only the best option text."


def _render_personamem_query_type_hint(question: PersonaMemQuestion) -> str:
    mapping = {
        "recall_user_shared_facts": "Identify the single user fact that best matches the query.",
        "recalling_facts_mentioned_by_the_user": "Focus on facts directly stated by the user.",
        "recalling_the_reasons_behind_previous_updates": "Use belief items to recover the user's stated reason for the update.",
        "provide_preference_aligned_recommendations": "Choose the option most aligned with the user's active preferences and goals.",
        "suggest_new_ideas": "Choose the option that best extends the user's interests without contradicting active beliefs.",
        "track_full_preference_evolution": "Prefer the option that reflects how the user's preferences changed over time.",
        "generalizing_to_new_scenarios": "Choose the option that best generalizes the active preference to the new scenario.",
    }
    return mapping.get(question.question_type, "Choose the option best supported by the active belief state.")


def _render_personamem_prompt(
    question: PersonaMemQuestion,
    memory_payload: dict[str, Any],
) -> str:
    options_block = _render_personamem_options(question.all_options)
    return (
        "You are answering a PersonaMem question using only the structured memory state below.\n\n"
        f"Question:\n{question.user_question_or_message}\n\n"
        f"Question type hint:\n{_render_personamem_query_type_hint(question)}\n\n"
        f"Belief JSON:\n{json.dumps(memory_payload['belief_state'], ensure_ascii=False, indent=2)}\n\n"
        f"Evidence:\n{memory_payload['evidence_block']}\n\n"
        f"Options:\n{options_block}\n\n"
        f"{_personamem_answer_instruction(question.all_options)} Do not use any raw history beyond the belief state and evidence."
    )


def _render_longmemeval_prompt(question: LongMemEvalQuestion, memory_payload: dict[str, Any]) -> str:
    lowered_question = question.question.strip().lower()
    instruction = "Return only the shortest exact answer phrase supported by the belief state."
    if lowered_question.startswith("where "):
        instruction += " Omit any leading preposition such as 'at', 'in', 'on', or 'from'."
        if any(token in lowered_question for token in (" buy ", " bought ", " purchase ", " purchased ", " got ")):
            instruction += " For purchase-source questions, return the concrete place phrase; if the place is described as 'a' or 'an' store/shop/market, rewrite it as 'the ...'."
    elif lowered_question.startswith("how many") or "what number" in lowered_question:
        instruction += " Return only the bare number, with no unit words or explanation."
    elif lowered_question.startswith("how much"):
        instruction += " Return only the exact amount phrase, including any needed currency or percent symbol, with no explanation."
    elif lowered_question.startswith("how often") or "how frequently" in lowered_question:
        instruction += " Return only the frequency phrase."
    return (
        "You are answering a LongMemEval question using only the structured memory state below.\n\n"
        f"Question date: {question.question_date}\n"
        f"Question:\n{question.question}\n\n"
        f"Belief JSON:\n{json.dumps(memory_payload['belief_state'], ensure_ascii=False, indent=2)}\n\n"
        f"Evidence:\n{memory_payload['evidence_block']}\n\n"
        f"{instruction}"
    )


def _memory_payload(system: StructuredMemorySystem, query_id: str, query_text: str) -> dict[str, Any]:
    result = system.query(query_id, query_text)
    return {
        "belief_state": result.belief_state.to_dict(),
        "belief_source": result.belief_source,
        "evidence_block": result.evidence_block,
        "answer_text": result.answer_text,
        "selected_slot_ids": [slot.slot_id for slot in result.selected_slots],
        "selected_slot_glosses": [slot.canonical_gloss for slot in result.selected_slots],
        "composed_memory": result.composed_memory,
    }


def _build_memory_system(
    *,
    memory_mode: str,
    slot_assignment_mode: str,
    learned_memory_checkpoint_dir: str | None,
    learned_memory_train_config_path: str | None,
    learned_memory_device: str,
    learned_slot_assignment_checkpoint_dir: str | None,
    learned_slot_assignment_train_config_path: str | None,
    learned_slot_assignment_device: str,
    learned_belief_predictor: Any | None = None,
    learned_slot_assignment_predictor: Any | None = None,
) -> StructuredMemorySystem:
    return StructuredMemorySystem(
        memory_mode=memory_mode,
        use_learned_memory=memory_mode == "learned_memory",
        slot_assignment_mode=slot_assignment_mode,
        use_learned_slot_assignment=slot_assignment_mode == "learned",
        learned_memory_checkpoint_dir=learned_memory_checkpoint_dir,
        learned_memory_train_config_path=learned_memory_train_config_path,
        learned_memory_device=learned_memory_device,
        learned_belief_predictor=learned_belief_predictor,
        learned_slot_assignment_checkpoint_dir=learned_slot_assignment_checkpoint_dir or learned_memory_checkpoint_dir,
        learned_slot_assignment_train_config_path=learned_slot_assignment_train_config_path or learned_memory_train_config_path,
        learned_slot_assignment_device=learned_slot_assignment_device,
        learned_slot_assignment_predictor=learned_slot_assignment_predictor,
    )


def _resolve_shared_predictors(
    *,
    memory_mode: str,
    slot_assignment_mode: str,
    learned_memory_checkpoint_dir: str | None,
    learned_memory_train_config_path: str | None,
    learned_memory_device: str,
    learned_slot_assignment_checkpoint_dir: str | None,
    learned_slot_assignment_train_config_path: str | None,
    learned_slot_assignment_device: str,
) -> tuple[Any | None, Any | None]:
    if memory_mode != "learned_memory" and slot_assignment_mode != "learned":
        return None, None
    template = _build_memory_system(
        memory_mode=memory_mode,
        slot_assignment_mode=slot_assignment_mode,
        learned_memory_checkpoint_dir=learned_memory_checkpoint_dir,
        learned_memory_train_config_path=learned_memory_train_config_path,
        learned_memory_device=learned_memory_device,
        learned_slot_assignment_checkpoint_dir=learned_slot_assignment_checkpoint_dir,
        learned_slot_assignment_train_config_path=learned_slot_assignment_train_config_path,
        learned_slot_assignment_device=learned_slot_assignment_device,
    )
    belief_predictor = None
    slot_assignment_predictor = None
    if memory_mode == "learned_memory":
        belief_predictor = template._resolve_learned_belief_predictor()
    if slot_assignment_mode == "learned":
        slot_assignment_predictor = template._resolve_slot_assignment_predictor()
    return belief_predictor, slot_assignment_predictor


def _maybe_write_semantic_alias(output_root: Path, benchmark: str, summary: dict[str, Any]) -> None:
    if summary.get("memory_mode") != "learned_memory" or summary.get("status") != "completed":
        return
    alias_name = (
        "latest_personamem_stage2_semantic_canary.json"
        if benchmark == "personamem"
        else "latest_longmemeval_stage2_semantic_canary.json"
    )
    _write_json(output_root / "artifacts" / alias_name, summary)


def _maybe_write_learned_alias(output_root: Path, benchmark: str, summary: dict[str, Any]) -> None:
    if summary.get("memory_mode") != "learned_memory":
        return
    alias_name = (
        "latest_personamem_stage2_learned_canary.json"
        if benchmark == "personamem"
        else "latest_longmemeval_stage2_learned_canary.json"
    )
    _write_json(output_root / "artifacts" / alias_name, summary)
    _maybe_write_semantic_alias(output_root, benchmark, summary)


def _maybe_write_slot_assignment_alias(output_root: Path, benchmark: str, summary: dict[str, Any]) -> None:
    if summary.get("slot_assignment_mode") != "learned" or summary.get("status") != "completed":
        return
    alias_name = (
        "latest_personamem_stage2_slot_assignment_canary.json"
        if benchmark == "personamem"
        else "latest_longmemeval_stage2_slot_assignment_canary.json"
    )
    _write_json(output_root / "artifacts" / alias_name, summary)


def _maybe_write_v24_canary_alias(output_root: Path, benchmark: str, summary: dict[str, Any]) -> None:
    if summary.get("slot_assignment_mode") != "learned" or summary.get("status") != "completed":
        return
    alias_name = (
        "latest_personamem_stage2_v24_canary.json"
        if benchmark == "personamem"
        else "latest_longmemeval_stage2_v24_canary.json"
    )
    _write_json(output_root / "artifacts" / alias_name, summary)


def _should_use_symbolic_parallel_fast_path(
    *,
    provider_configured: bool,
    memory_mode: str,
    slot_assignment_mode: str,
    provider_workers: int,
) -> bool:
    return (
        provider_configured
        and memory_mode == "symbolic"
        and slot_assignment_mode == "symbolic"
        and max(int(provider_workers), 1) > 1
    )


def _build_personamem_row(
    question: PersonaMemQuestion,
    *,
    adapter: PersonaMemAdapter,
    contexts: dict[str, str],
    memory_mode: str,
    slot_assignment_mode: str,
    learned_memory_checkpoint_dir: str | None,
    learned_memory_train_config_path: str | None,
    learned_memory_device: str,
    learned_slot_assignment_checkpoint_dir: str | None,
    learned_slot_assignment_train_config_path: str | None,
    learned_slot_assignment_device: str,
    provider_configured: bool,
    llm: LLMConfig | None = None,
    learned_belief_predictor: Any | None = None,
    learned_slot_assignment_predictor: Any | None = None,
) -> dict[str, Any]:
    system = _build_memory_system(
        memory_mode=memory_mode,
        slot_assignment_mode=slot_assignment_mode,
        learned_memory_checkpoint_dir=learned_memory_checkpoint_dir,
        learned_memory_train_config_path=learned_memory_train_config_path,
        learned_memory_device=learned_memory_device,
        learned_slot_assignment_checkpoint_dir=learned_slot_assignment_checkpoint_dir,
        learned_slot_assignment_train_config_path=learned_slot_assignment_train_config_path,
        learned_slot_assignment_device=learned_slot_assignment_device,
        learned_belief_predictor=learned_belief_predictor,
        learned_slot_assignment_predictor=learned_slot_assignment_predictor,
    )
    observed_turns = _observe_personamem_context(
        system,
        adapter.render_context_for_question(question, contexts),
        sample_id=question.question_id,
    )
    memory_payload = _memory_payload(system, question.question_id, question.user_question_or_message)
    local_projection = _project_personamem_local_answer(memory_payload, question)
    prompt = _render_personamem_prompt(question, memory_payload)
    row = {
        "sample_id": question.question_id,
        "benchmark": "personamem",
        "question_type": question.question_type,
        "topic": question.topic,
        "expected_answer": question.correct_answer,
        "memory_answer_local": local_projection,
        "provider_prediction": None,
        "provider_raw_prediction": None,
        "provider_status": "provider_not_configured",
        "provider_configured": provider_configured,
        "observed_turns": observed_turns,
        "belief_source": memory_payload["belief_source"],
        "selected_slot_ids": memory_payload["selected_slot_ids"],
        "belief_state": memory_payload["belief_state"],
        "evidence_block": memory_payload["evidence_block"],
        "prompt": prompt,
        "prompt_version": "stage2_memory_canary_v2",
    }
    if provider_configured:
        if llm is None:
            raise ValueError("llm is required when provider_configured is True.")
        provider_prediction, raw_provider_prediction = _provider_chat_request(llm, prompt)
        row["provider_prediction"] = provider_prediction
        row["provider_raw_prediction"] = raw_provider_prediction
        row["provider_status"] = "completed"
    return row


def _build_longmemeval_row(
    question: LongMemEvalQuestion,
    *,
    memory_mode: str,
    slot_assignment_mode: str,
    learned_memory_checkpoint_dir: str | None,
    learned_memory_train_config_path: str | None,
    learned_memory_device: str,
    learned_slot_assignment_checkpoint_dir: str | None,
    learned_slot_assignment_train_config_path: str | None,
    learned_slot_assignment_device: str,
    provider_configured: bool,
    llm: LLMConfig | None = None,
    learned_belief_predictor: Any | None = None,
    learned_slot_assignment_predictor: Any | None = None,
) -> dict[str, Any]:
    system = _build_memory_system(
        memory_mode=memory_mode,
        slot_assignment_mode=slot_assignment_mode,
        learned_memory_checkpoint_dir=learned_memory_checkpoint_dir,
        learned_memory_train_config_path=learned_memory_train_config_path,
        learned_memory_device=learned_memory_device,
        learned_slot_assignment_checkpoint_dir=learned_slot_assignment_checkpoint_dir,
        learned_slot_assignment_train_config_path=learned_slot_assignment_train_config_path,
        learned_slot_assignment_device=learned_slot_assignment_device,
        learned_belief_predictor=learned_belief_predictor,
        learned_slot_assignment_predictor=learned_slot_assignment_predictor,
    )
    observed_turns = _observe_longmemeval_context(system, question.haystack_sessions, sample_id=question.question_id)
    memory_payload = _memory_payload(system, question.question_id, question.question)
    prompt = _render_longmemeval_prompt(question, memory_payload)
    row = {
        "sample_id": question.question_id,
        "benchmark": "longmemeval_s",
        "question_type": question.question_type,
        "expected_answer": question.answer,
        "memory_answer_local": memory_payload["answer_text"],
        "provider_prediction": None,
        "provider_raw_prediction": None,
        "provider_status": "provider_not_configured",
        "provider_configured": provider_configured,
        "observed_turns": observed_turns,
        "belief_source": memory_payload["belief_source"],
        "selected_slot_ids": memory_payload["selected_slot_ids"],
        "belief_state": memory_payload["belief_state"],
        "evidence_block": memory_payload["evidence_block"],
        "prompt": prompt,
        "prompt_version": "stage2_memory_canary_v2",
    }
    if provider_configured:
        if llm is None:
            raise ValueError("llm is required when provider_configured is True.")
        provider_prediction, raw_provider_prediction = _provider_chat_request(llm, prompt)
        row["provider_prediction"] = provider_prediction
        row["provider_raw_prediction"] = raw_provider_prediction
        row["provider_status"] = "completed"
    return row


def run_personamem_canary(
    *,
    output_root: Path,
    config_path: Path,
    limit: int,
    memory_mode: str = "symbolic",
    slot_assignment_mode: str = "symbolic",
    learned_memory_checkpoint_dir: str | None = None,
    learned_memory_train_config_path: str | None = None,
    learned_memory_device: str = "cpu",
    learned_slot_assignment_checkpoint_dir: str | None = None,
    learned_slot_assignment_train_config_path: str | None = None,
    learned_slot_assignment_device: str = "cpu",
    provider_workers: int = 1,
    requested_run_dir: str | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    config = load_project_config(config_path)
    provider = _provider_from_llm(config.llm)
    adapter = PersonaMemAdapter(data_root=REPO_ROOT / config.benchmarks.personamem.data_root)
    contexts = adapter.load_shared_contexts()
    manifest = _load_json(_ensure_canary_manifest(output_root, "personamem"))
    selected_ids = set(str(sample_id) for sample_id in manifest["sample_ids"][:limit])
    questions = [item for item in adapter.load_questions() if item.question_id in selected_ids]

    run_dir = _resolve_run_dir(output_root, "personamem", requested_run_dir=requested_run_dir)
    stamp = run_dir.name.split("_", 1)[0]
    config_snapshot = _copy_config_snapshot(config_path, run_dir)
    predictions_path = run_dir / "predictions.jsonl"
    existing_rows = _load_jsonl(predictions_path) if resume else []
    completed_ids = {str(row.get("sample_id", "")) for row in existing_rows}

    rows: list[dict[str, Any]] = list(existing_rows)
    live_completed = sum(1 for row in existing_rows if row.get("provider_status") == "completed")
    metadata = {
        "benchmark": "personamem",
        "sample_count": len(questions),
        "live_predictions_completed": live_completed,
        "completed_predictions": len(rows),
        "resumed_prediction_count": len(existing_rows),
        "provider_configured": provider.is_configured(),
        "config_path": str(config_path),
        "config_snapshot_path": str(config_snapshot),
        "model": config.llm.model,
        "api_key_env": config.llm.api_key_env,
        "memory_mode": memory_mode,
        "use_learned_memory": memory_mode == "learned_memory",
        "slot_assignment_mode": slot_assignment_mode,
        "use_learned_slot_assignment": slot_assignment_mode == "learned",
        "learned_memory_checkpoint_dir": learned_memory_checkpoint_dir,
        "learned_memory_train_config_path": learned_memory_train_config_path,
        "learned_slot_assignment_checkpoint_dir": learned_slot_assignment_checkpoint_dir or learned_memory_checkpoint_dir,
        "learned_slot_assignment_train_config_path": learned_slot_assignment_train_config_path or learned_memory_train_config_path,
        "run_timestamp": stamp,
        "commit_hash": _current_commit_hash(),
        "provider_workers": max(int(provider_workers), 1),
        "canary_manifest": str(_ensure_canary_manifest(output_root, "personamem")),
        "predictions_path": str(predictions_path),
    }
    _write_json(run_dir / "run_metadata.json", metadata)
    shared_belief_predictor, shared_slot_assignment_predictor = _resolve_shared_predictors(
        memory_mode=memory_mode,
        slot_assignment_mode=slot_assignment_mode,
        learned_memory_checkpoint_dir=learned_memory_checkpoint_dir,
        learned_memory_train_config_path=learned_memory_train_config_path,
        learned_memory_device=learned_memory_device,
        learned_slot_assignment_checkpoint_dir=learned_slot_assignment_checkpoint_dir,
        learned_slot_assignment_train_config_path=learned_slot_assignment_train_config_path,
        learned_slot_assignment_device=learned_slot_assignment_device,
    )
    pending_questions = [question for question in questions if question.question_id not in completed_ids]
    pending_rows: list[dict[str, Any]] = []
    if _should_use_symbolic_parallel_fast_path(
        provider_configured=provider.is_configured(),
        memory_mode=memory_mode,
        slot_assignment_mode=slot_assignment_mode,
        provider_workers=provider_workers,
    ):
        with ThreadPoolExecutor(max_workers=max(int(provider_workers), 1)) as executor:
            future_map = {
                executor.submit(
                    _build_personamem_row,
                    question,
                    adapter=adapter,
                    contexts=contexts,
                    memory_mode=memory_mode,
                    slot_assignment_mode=slot_assignment_mode,
                    learned_memory_checkpoint_dir=learned_memory_checkpoint_dir,
                    learned_memory_train_config_path=learned_memory_train_config_path,
                    learned_memory_device=learned_memory_device,
                    learned_slot_assignment_checkpoint_dir=learned_slot_assignment_checkpoint_dir,
                    learned_slot_assignment_train_config_path=learned_slot_assignment_train_config_path,
                    learned_slot_assignment_device=learned_slot_assignment_device,
                    provider_configured=provider.is_configured(),
                    llm=config.llm,
                    learned_belief_predictor=shared_belief_predictor,
                    learned_slot_assignment_predictor=shared_slot_assignment_predictor,
                ): question.question_id
                for question in pending_questions
            }
            for future in as_completed(future_map):
                row = future.result()
                if row["provider_status"] == "completed":
                    live_completed += 1
                rows.append(row)
                _append_jsonl_row(predictions_path, row)
                metadata["live_predictions_completed"] = live_completed
                metadata["completed_predictions"] = len(rows)
                _write_json(run_dir / "run_metadata.json", metadata)
    else:
        for question in pending_questions:
            pending_rows.append(
                _build_personamem_row(
                    question,
                    adapter=adapter,
                    contexts=contexts,
                    memory_mode=memory_mode,
                    slot_assignment_mode=slot_assignment_mode,
                    learned_memory_checkpoint_dir=learned_memory_checkpoint_dir,
                    learned_memory_train_config_path=learned_memory_train_config_path,
                    learned_memory_device=learned_memory_device,
                    learned_slot_assignment_checkpoint_dir=learned_slot_assignment_checkpoint_dir,
                    learned_slot_assignment_train_config_path=learned_slot_assignment_train_config_path,
                    learned_slot_assignment_device=learned_slot_assignment_device,
                    provider_configured=False,
                    learned_belief_predictor=shared_belief_predictor,
                    learned_slot_assignment_predictor=shared_slot_assignment_predictor,
                )
            )

        if provider.is_configured():
            provider_results = _iter_provider_predictions(
                config.llm,
                [(index, row["prompt"]) for index, row in enumerate(pending_rows)],
                workers=max(int(provider_workers), 1),
            )
            for index, provider_prediction, raw_provider_prediction in provider_results:
                row = pending_rows[index]
                row["provider_prediction"] = provider_prediction
                row["provider_raw_prediction"] = raw_provider_prediction
                row["provider_status"] = "completed"
    for row in pending_rows:
        if row["provider_status"] == "completed":
            live_completed += 1
        rows.append(row)
        _append_jsonl_row(predictions_path, row)
        metadata["live_predictions_completed"] = live_completed
        metadata["completed_predictions"] = len(rows)
        _write_json(run_dir / "run_metadata.json", metadata)

    metadata["live_predictions_completed"] = live_completed
    metadata["completed_predictions"] = len(rows)
    _write_json(run_dir / "run_metadata.json", metadata)

    summary_path = output_root / "evals_benchmark" / f"{stamp}_stage2_memory_canary.json"
    summary = {
        **metadata,
        "status": "completed" if provider.is_configured() else "blocked_provider_not_configured",
        "run_dir": str(run_dir),
        "summary_path": str(summary_path),
        **_prediction_metrics(rows),
    }
    _write_json(summary_path, summary)
    _maybe_write_learned_alias(output_root, "personamem", summary)
    _maybe_write_slot_assignment_alias(output_root, "personamem", summary)
    _maybe_write_v24_canary_alias(output_root, "personamem", summary)
    return summary


def run_longmemeval_canary(
    *,
    output_root: Path,
    config_path: Path,
    limit: int,
    memory_mode: str = "symbolic",
    slot_assignment_mode: str = "symbolic",
    learned_memory_checkpoint_dir: str | None = None,
    learned_memory_train_config_path: str | None = None,
    learned_memory_device: str = "cpu",
    learned_slot_assignment_checkpoint_dir: str | None = None,
    learned_slot_assignment_train_config_path: str | None = None,
    learned_slot_assignment_device: str = "cpu",
    provider_workers: int = 1,
    requested_run_dir: str | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    config = load_project_config(config_path)
    provider = _provider_from_llm(config.llm)
    adapter = LongMemEvalAdapter(data_root=REPO_ROOT / config.benchmarks.longmemeval.data_root)
    manifest = _load_json(_ensure_canary_manifest(output_root, "longmemeval"))
    selected_ids = set(str(sample_id) for sample_id in manifest["sample_ids"][:limit])
    questions = [item for item in adapter.load_questions() if item.question_id in selected_ids]

    run_dir = _resolve_run_dir(output_root, "longmemeval", requested_run_dir=requested_run_dir)
    stamp = run_dir.name.split("_", 1)[0]
    config_snapshot = _copy_config_snapshot(config_path, run_dir)
    predictions_path = run_dir / "predictions.jsonl"
    existing_rows = _load_jsonl(predictions_path) if resume else []
    completed_ids = {str(row.get("sample_id", "")) for row in existing_rows}

    rows: list[dict[str, Any]] = list(existing_rows)
    live_completed = sum(1 for row in existing_rows if row.get("provider_status") == "completed")
    metadata = {
        "benchmark": "longmemeval_s",
        "sample_count": len(questions),
        "live_predictions_completed": live_completed,
        "completed_predictions": len(rows),
        "resumed_prediction_count": len(existing_rows),
        "provider_configured": provider.is_configured(),
        "config_path": str(config_path),
        "config_snapshot_path": str(config_snapshot),
        "model": config.llm.model,
        "api_key_env": config.llm.api_key_env,
        "memory_mode": memory_mode,
        "use_learned_memory": memory_mode == "learned_memory",
        "slot_assignment_mode": slot_assignment_mode,
        "use_learned_slot_assignment": slot_assignment_mode == "learned",
        "learned_memory_checkpoint_dir": learned_memory_checkpoint_dir,
        "learned_memory_train_config_path": learned_memory_train_config_path,
        "learned_slot_assignment_checkpoint_dir": learned_slot_assignment_checkpoint_dir or learned_memory_checkpoint_dir,
        "learned_slot_assignment_train_config_path": learned_slot_assignment_train_config_path or learned_memory_train_config_path,
        "run_timestamp": stamp,
        "commit_hash": _current_commit_hash(),
        "provider_workers": max(int(provider_workers), 1),
        "canary_manifest": str(_ensure_canary_manifest(output_root, "longmemeval")),
        "predictions_path": str(predictions_path),
    }
    _write_json(run_dir / "run_metadata.json", metadata)
    shared_belief_predictor, shared_slot_assignment_predictor = _resolve_shared_predictors(
        memory_mode=memory_mode,
        slot_assignment_mode=slot_assignment_mode,
        learned_memory_checkpoint_dir=learned_memory_checkpoint_dir,
        learned_memory_train_config_path=learned_memory_train_config_path,
        learned_memory_device=learned_memory_device,
        learned_slot_assignment_checkpoint_dir=learned_slot_assignment_checkpoint_dir,
        learned_slot_assignment_train_config_path=learned_slot_assignment_train_config_path,
        learned_slot_assignment_device=learned_slot_assignment_device,
    )
    pending_questions = [question for question in questions if question.question_id not in completed_ids]
    pending_rows: list[dict[str, Any]] = []
    if _should_use_symbolic_parallel_fast_path(
        provider_configured=provider.is_configured(),
        memory_mode=memory_mode,
        slot_assignment_mode=slot_assignment_mode,
        provider_workers=provider_workers,
    ):
        with ThreadPoolExecutor(max_workers=max(int(provider_workers), 1)) as executor:
            future_map = {
                executor.submit(
                    _build_longmemeval_row,
                    question,
                    memory_mode=memory_mode,
                    slot_assignment_mode=slot_assignment_mode,
                    learned_memory_checkpoint_dir=learned_memory_checkpoint_dir,
                    learned_memory_train_config_path=learned_memory_train_config_path,
                    learned_memory_device=learned_memory_device,
                    learned_slot_assignment_checkpoint_dir=learned_slot_assignment_checkpoint_dir,
                    learned_slot_assignment_train_config_path=learned_slot_assignment_train_config_path,
                    learned_slot_assignment_device=learned_slot_assignment_device,
                    provider_configured=provider.is_configured(),
                    llm=config.llm,
                    learned_belief_predictor=shared_belief_predictor,
                    learned_slot_assignment_predictor=shared_slot_assignment_predictor,
                ): question.question_id
                for question in pending_questions
            }
            for future in as_completed(future_map):
                row = future.result()
                if row["provider_status"] == "completed":
                    live_completed += 1
                rows.append(row)
                _append_jsonl_row(predictions_path, row)
                metadata["live_predictions_completed"] = live_completed
                metadata["completed_predictions"] = len(rows)
                _write_json(run_dir / "run_metadata.json", metadata)
    else:
        for question in pending_questions:
            pending_rows.append(
                _build_longmemeval_row(
                    question,
                    memory_mode=memory_mode,
                    slot_assignment_mode=slot_assignment_mode,
                    learned_memory_checkpoint_dir=learned_memory_checkpoint_dir,
                    learned_memory_train_config_path=learned_memory_train_config_path,
                    learned_memory_device=learned_memory_device,
                    learned_slot_assignment_checkpoint_dir=learned_slot_assignment_checkpoint_dir,
                    learned_slot_assignment_train_config_path=learned_slot_assignment_train_config_path,
                    learned_slot_assignment_device=learned_slot_assignment_device,
                    provider_configured=False,
                    learned_belief_predictor=shared_belief_predictor,
                    learned_slot_assignment_predictor=shared_slot_assignment_predictor,
                )
            )

        if provider.is_configured():
            provider_results = _iter_provider_predictions(
                config.llm,
                [(index, row["prompt"]) for index, row in enumerate(pending_rows)],
                workers=max(int(provider_workers), 1),
            )
            for index, provider_prediction, raw_provider_prediction in provider_results:
                row = pending_rows[index]
                row["provider_prediction"] = provider_prediction
                row["provider_raw_prediction"] = raw_provider_prediction
                row["provider_status"] = "completed"
    for row in pending_rows:
        if row["provider_status"] == "completed":
            live_completed += 1
        rows.append(row)
        _append_jsonl_row(predictions_path, row)
        metadata["live_predictions_completed"] = live_completed
        metadata["completed_predictions"] = len(rows)
        _write_json(run_dir / "run_metadata.json", metadata)

    metadata["live_predictions_completed"] = live_completed
    metadata["completed_predictions"] = len(rows)
    _write_json(run_dir / "run_metadata.json", metadata)

    summary_path = output_root / "evals_benchmark" / f"{stamp}_stage2_memory_canary.json"
    summary = {
        **metadata,
        "status": "completed" if provider.is_configured() else "blocked_provider_not_configured",
        "run_dir": str(run_dir),
        "summary_path": str(summary_path),
        **_prediction_metrics(rows),
    }
    _write_json(summary_path, summary)
    _maybe_write_learned_alias(output_root, "longmemeval", summary)
    _maybe_write_slot_assignment_alias(output_root, "longmemeval", summary)
    _maybe_write_v24_canary_alias(output_root, "longmemeval", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/minimax_m27.yaml")
    parser.add_argument("--benchmark", choices=["personamem", "longmemeval"], default="personamem")
    parser.add_argument("--output-root", default="outputs_v2")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--memory-mode", choices=["symbolic", "learned_memory"], default="symbolic")
    parser.add_argument("--slot-assignment-mode", choices=["symbolic", "learned"], default="symbolic")
    parser.add_argument("--learned-memory-checkpoint-dir")
    parser.add_argument("--learned-memory-train-config")
    parser.add_argument("--learned-memory-device", default="cpu")
    parser.add_argument("--learned-slot-assignment-checkpoint-dir")
    parser.add_argument("--learned-slot-assignment-train-config")
    parser.add_argument("--learned-slot-assignment-device", default="cpu")
    parser.add_argument("--provider-workers", type=int, default=1)
    parser.add_argument("--run-dir")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.benchmark == "personamem":
        payload = run_personamem_canary(
            output_root=Path(args.output_root),
            config_path=Path(args.config),
            limit=args.limit,
            memory_mode=args.memory_mode,
            slot_assignment_mode=args.slot_assignment_mode,
            learned_memory_checkpoint_dir=args.learned_memory_checkpoint_dir,
            learned_memory_train_config_path=args.learned_memory_train_config,
            learned_memory_device=args.learned_memory_device,
            learned_slot_assignment_checkpoint_dir=args.learned_slot_assignment_checkpoint_dir,
            learned_slot_assignment_train_config_path=args.learned_slot_assignment_train_config,
            learned_slot_assignment_device=args.learned_slot_assignment_device,
            provider_workers=args.provider_workers,
            requested_run_dir=args.run_dir,
            resume=args.resume,
        )
    else:
        payload = run_longmemeval_canary(
            output_root=Path(args.output_root),
            config_path=Path(args.config),
            limit=args.limit,
            memory_mode=args.memory_mode,
            slot_assignment_mode=args.slot_assignment_mode,
            learned_memory_checkpoint_dir=args.learned_memory_checkpoint_dir,
            learned_memory_train_config_path=args.learned_memory_train_config,
            learned_memory_device=args.learned_memory_device,
            learned_slot_assignment_checkpoint_dir=args.learned_slot_assignment_checkpoint_dir,
            learned_slot_assignment_train_config_path=args.learned_slot_assignment_train_config,
            learned_slot_assignment_device=args.learned_slot_assignment_device,
            provider_workers=args.provider_workers,
            requested_run_dir=args.run_dir,
            resume=args.resume,
        )
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"summary_path={payload['summary_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
