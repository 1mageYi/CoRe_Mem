"""Run a stage-2 memory-mediated benchmark canary with honest artifact logging."""

from __future__ import annotations

import argparse
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


def _copy_config_snapshot(config_path: Path, run_dir: Path) -> Path:
    snapshot = run_dir / "config_snapshot.yaml"
    snapshot.write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    return snapshot


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _normalize_answer(text: str) -> str:
    return " ".join(text.strip().lower().split())


_OPTION_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "to",
    "of",
    "in",
    "on",
    "for",
    "with",
    "that",
    "this",
    "it",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "as",
    "at",
    "by",
    "from",
    "about",
    "into",
    "through",
    "your",
    "you",
    "i",
    "my",
    "me",
    "we",
    "our",
    "their",
    "they",
    "them",
    "he",
    "she",
    "his",
    "her",
}


def _options_use_labels(options: list[str]) -> bool:
    return bool(options) and all(option.startswith("(") and ")" in option[:4] for option in options)


def _option_label(option: str) -> str:
    closing = option.find(")")
    return option[: closing + 1].strip() if option.startswith("(") and closing > 0 else option.strip()


def _option_body(option: str) -> str:
    label = _option_label(option)
    remainder = option[len(label) :].strip()
    return remainder or option.strip()


def _tokenize_option_text(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z]+", text.lower())
        if len(token) >= 3 and token not in _OPTION_STOPWORDS
    ]


def _best_personamem_option_label(memory_payload: dict[str, Any], question: PersonaMemQuestion) -> str:
    support_parts = [
        question.user_question_or_message,
        memory_payload.get("evidence_block", ""),
        " ".join(str(item) for item in memory_payload.get("selected_slot_glosses", [])),
        " ".join(
            str(item.get("value", ""))
            for item in memory_payload.get("belief_state", {}).get("belief_items", [])
            if isinstance(item, dict)
        ),
    ]
    support_tokens = set(_tokenize_option_text(" ".join(part for part in support_parts if part)))
    best_label = ""
    best_score = float("-inf")
    for option in question.all_options:
        body = _option_body(option)
        option_tokens = _tokenize_option_text(body)
        overlap = sum(1 for token in option_tokens if token in support_tokens)
        penalty = int(any(token in body.lower() for token in (" not ", " dislike", " avoid")))
        score = overlap - penalty
        label = _option_label(option)
        if score > best_score:
            best_score = score
            best_label = label
    return best_label


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
    scored_label = _best_personamem_option_label(memory_payload, question)
    if scored_label:
        return scored_label
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


def _finalize_personamem_provider_prediction(
    provider_prediction: str | None,
    *,
    memory_payload: dict[str, Any],
    question: PersonaMemQuestion,
) -> str | None:
    normalized = (provider_prediction or "").strip()
    if normalized:
        return provider_prediction
    fallback = _best_personamem_option_label(memory_payload, question)
    return fallback or provider_prediction


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
    *,
    candidate_answer: str | None = None,
) -> str:
    options_block = _render_personamem_options(question.all_options)
    candidate_block = ""
    if candidate_answer:
        candidate_block = (
            f"Latent matcher candidate:\n{candidate_answer}\n\n"
            "Prefer this candidate when it is consistent with the belief state and evidence. "
            "Only override it when another option is more strongly supported.\n\n"
        )
    return (
        "You are answering a PersonaMem question using only the structured memory state below.\n\n"
        f"Question:\n{question.user_question_or_message}\n\n"
        f"Question type hint:\n{_render_personamem_query_type_hint(question)}\n\n"
        f"Belief JSON:\n{json.dumps(memory_payload['belief_state'], ensure_ascii=False, indent=2)}\n\n"
        f"Evidence:\n{memory_payload['evidence_block']}\n\n"
        f"{candidate_block}"
        f"Options:\n{options_block}\n\n"
        f"{_personamem_answer_instruction(question.all_options)} Do not use any raw history beyond the belief state and evidence."
    )


def _render_longmemeval_prompt(question: LongMemEvalQuestion, memory_payload: dict[str, Any]) -> str:
    return (
        "You are answering a LongMemEval question using only the structured memory state below.\n\n"
        f"Question date: {question.question_date}\n"
        f"Question:\n{question.question}\n\n"
        f"Belief JSON:\n{json.dumps(memory_payload['belief_state'], ensure_ascii=False, indent=2)}\n\n"
        f"Evidence:\n{memory_payload['evidence_block']}\n\n"
        "Return only the shortest exact answer phrase supported by the belief state."
    )


def _memory_payload(system: StructuredMemorySystem, query_id: str, query_text: str) -> dict[str, Any]:
    result = system.query(query_id, query_text)
    return {
        "belief_state": result.belief_state.to_dict(),
        "evidence_block": result.evidence_block,
        "answer_text": result.answer_text,
        "selected_slot_ids": [slot.slot_id for slot in result.selected_slots],
        "selected_slot_glosses": [slot.canonical_gloss for slot in result.selected_slots],
        "composed_memory": result.composed_memory,
    }


def run_personamem_canary(
    *,
    output_root: Path,
    config_path: Path,
    limit: int,
) -> dict[str, Any]:
    config = load_project_config(config_path)
    provider = _provider_from_llm(config.llm)
    adapter = PersonaMemAdapter(data_root=REPO_ROOT / config.benchmarks.personamem.data_root)
    contexts = adapter.load_shared_contexts()
    manifest = _load_json(_ensure_canary_manifest(output_root, "personamem"))
    selected_ids = set(str(sample_id) for sample_id in manifest["sample_ids"][:limit])
    questions = [item for item in adapter.load_questions() if item.question_id in selected_ids]

    stamp = _timestamp()
    run_dir = output_root / "runs" / f"{stamp}_stage2_memory_canary_personamem"
    run_dir.mkdir(parents=True, exist_ok=True)
    config_snapshot = _copy_config_snapshot(config_path, run_dir)

    rows: list[dict[str, Any]] = []
    live_completed = 0
    for question in questions:
        system = StructuredMemorySystem()
        observed_turns = _observe_personamem_context(
            system,
            adapter.render_context_for_question(question, contexts),
            sample_id=question.question_id,
        )
        memory_payload = _memory_payload(system, question.question_id, question.user_question_or_message)
        local_projection = _project_personamem_local_answer(memory_payload, question)
        prompt = _render_personamem_prompt(question, memory_payload, candidate_answer=local_projection)
        provider_prediction = None
        raw_provider_prediction = None
        status = "provider_not_configured"
        if provider.is_configured():
            raw_provider_prediction = provider.chat(
                prompt,
                temperature=config.llm.temperature,
                max_tokens=config.llm.max_tokens,
            ).content
            provider_prediction = _finalize_personamem_provider_prediction(
                raw_provider_prediction,
                memory_payload=memory_payload,
                question=question,
            )
            status = "completed"
            live_completed += 1
        rows.append(
            {
                "sample_id": question.question_id,
                "benchmark": "personamem",
                "question_type": question.question_type,
                "topic": question.topic,
                "expected_answer": question.correct_answer,
                "memory_answer_local": local_projection,
                "provider_prediction": provider_prediction,
                "provider_raw_prediction": raw_provider_prediction,
                "provider_status": status,
                "provider_configured": provider.is_configured(),
                "observed_turns": observed_turns,
                "selected_slot_ids": memory_payload["selected_slot_ids"],
                "belief_state": memory_payload["belief_state"],
                "evidence_block": memory_payload["evidence_block"],
                "prompt": prompt,
                "prompt_version": "stage2_memory_canary_v1",
            }
        )

    predictions_path = run_dir / "predictions.jsonl"
    _write_jsonl(predictions_path, rows)
    metadata = {
        "benchmark": "personamem",
        "sample_count": len(rows),
        "live_predictions_completed": live_completed,
        "provider_configured": provider.is_configured(),
        "config_path": str(config_path),
        "config_snapshot_path": str(config_snapshot),
        "model": config.llm.model,
        "api_key_env": config.llm.api_key_env,
        "run_timestamp": stamp,
        "commit_hash": _current_commit_hash(),
        "canary_manifest": str(_ensure_canary_manifest(output_root, "personamem")),
        "predictions_path": str(predictions_path),
    }
    _write_json(run_dir / "run_metadata.json", metadata)

    summary_path = output_root / "evals_benchmark" / f"{stamp}_stage2_memory_canary.json"
    summary = {
        **metadata,
        "status": "completed" if provider.is_configured() else "blocked_provider_not_configured",
        "run_dir": str(run_dir),
        "summary_path": str(summary_path),
    }
    _write_json(summary_path, summary)
    return summary


def run_longmemeval_canary(
    *,
    output_root: Path,
    config_path: Path,
    limit: int,
) -> dict[str, Any]:
    config = load_project_config(config_path)
    provider = _provider_from_llm(config.llm)
    adapter = LongMemEvalAdapter(data_root=REPO_ROOT / config.benchmarks.longmemeval.data_root)
    manifest = _load_json(_ensure_canary_manifest(output_root, "longmemeval"))
    selected_ids = set(str(sample_id) for sample_id in manifest["sample_ids"][:limit])
    questions = [item for item in adapter.load_questions() if item.question_id in selected_ids]

    stamp = _timestamp()
    run_dir = output_root / "runs" / f"{stamp}_stage2_memory_canary_longmemeval"
    run_dir.mkdir(parents=True, exist_ok=True)
    config_snapshot = _copy_config_snapshot(config_path, run_dir)

    rows: list[dict[str, Any]] = []
    live_completed = 0
    for question in questions:
        system = StructuredMemorySystem()
        observed_turns = _observe_longmemeval_context(system, question.haystack_sessions, sample_id=question.question_id)
        memory_payload = _memory_payload(system, question.question_id, question.question)
        prompt = _render_longmemeval_prompt(question, memory_payload)
        provider_prediction = None
        status = "provider_not_configured"
        if provider.is_configured():
            provider_prediction = provider.chat(
                prompt,
                temperature=config.llm.temperature,
                max_tokens=config.llm.max_tokens,
            ).content
            status = "completed"
            live_completed += 1
        rows.append(
            {
                "sample_id": question.question_id,
                "benchmark": "longmemeval_s",
                "question_type": question.question_type,
                "expected_answer": question.answer,
                "memory_answer_local": memory_payload["answer_text"],
                "provider_prediction": provider_prediction,
                "provider_status": status,
                "provider_configured": provider.is_configured(),
                "observed_turns": observed_turns,
                "selected_slot_ids": memory_payload["selected_slot_ids"],
                "belief_state": memory_payload["belief_state"],
                "evidence_block": memory_payload["evidence_block"],
                "prompt": prompt,
                "prompt_version": "stage2_memory_canary_v1",
            }
        )

    predictions_path = run_dir / "predictions.jsonl"
    _write_jsonl(predictions_path, rows)
    metadata = {
        "benchmark": "longmemeval_s",
        "sample_count": len(rows),
        "live_predictions_completed": live_completed,
        "provider_configured": provider.is_configured(),
        "config_path": str(config_path),
        "config_snapshot_path": str(config_snapshot),
        "model": config.llm.model,
        "api_key_env": config.llm.api_key_env,
        "run_timestamp": stamp,
        "commit_hash": _current_commit_hash(),
        "canary_manifest": str(_ensure_canary_manifest(output_root, "longmemeval")),
        "predictions_path": str(predictions_path),
    }
    _write_json(run_dir / "run_metadata.json", metadata)

    summary_path = output_root / "evals_benchmark" / f"{stamp}_stage2_memory_canary.json"
    summary = {
        **metadata,
        "status": "completed" if provider.is_configured() else "blocked_provider_not_configured",
        "run_dir": str(run_dir),
        "summary_path": str(summary_path),
    }
    _write_json(summary_path, summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/minimax_m27.yaml")
    parser.add_argument("--benchmark", choices=["personamem", "longmemeval"], default="personamem")
    parser.add_argument("--output-root", default="outputs_v2")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.benchmark == "personamem":
        payload = run_personamem_canary(
            output_root=Path(args.output_root),
            config_path=Path(args.config),
            limit=args.limit,
        )
    else:
        payload = run_longmemeval_canary(
            output_root=Path(args.output_root),
            config_path=Path(args.config),
            limit=args.limit,
        )
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"summary_path={payload['summary_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
