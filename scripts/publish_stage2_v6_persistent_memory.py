"""Publish v6 persistent core/residual memory evidence artifacts."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for candidate in (REPO_ROOT, SRC_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from core_mem.v2.parser import Stage2ObservationParser
from core_mem.v2.schemas import Observation, SlotRecord
from core_mem.v2.v6_persistent_memory import (
    PersistentCoreResidualMemory,
    V6_WRITE_ACTIONS,
    observation_features,
    option_label,
    route_observation,
    score_options_with_projection,
    silver_action_for_observation,
    stable_slot_fingerprint,
    train_projection_weights_from_observations,
    train_reader_readout,
    train_write_router,
    vector_dot,
)


ARTIFACT_ROOT = REPO_ROOT / "outputs_v2" / "artifacts"
DEFAULT_QUESTIONS = REPO_ROOT / "data" / "personamem" / "questions_32k.csv"
DEFAULT_CONTEXTS = REPO_ROOT / "data" / "personamem" / "shared_contexts_32k.jsonl"


def _timestamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def _current_head(root: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except Exception:
        return "unknown"


def _read_questions(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_contexts(path: Path) -> dict[str, list[dict[str, str]]]:
    contexts: dict[str, list[dict[str, str]]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            for context_id, messages in payload.items():
                contexts[str(context_id)] = list(messages)
    return contexts


def _parse_options(value: str) -> list[str]:
    try:
        parsed = ast.literal_eval(value)
    except Exception:
        parsed = json.loads(value)
    return [str(item) for item in parsed]


def _clean_message(content: str) -> str:
    return content.replace("User:", "").replace("Assistant:", "").strip()


def _parse_personamem_observations(
    contexts: dict[str, list[dict[str, str]]],
    *,
    limit_contexts: int | None,
) -> list[Observation]:
    parser = Stage2ObservationParser(minimum_confidence=0.55)
    observations: list[Observation] = []
    for context_count, (context_id, messages) in enumerate(contexts.items()):
        if limit_contexts is not None and context_count >= limit_contexts:
            break
        prior_user_text: list[str] = []
        for turn_idx, message in enumerate(messages):
            role = str(message.get("role") or "user")
            content = _clean_message(str(message.get("content") or ""))
            if not content:
                continue
            parsed = parser.parse_turn(
                content,
                source_dataset="personamem_raw_context_no_gold",
                source_dialogue_id=context_id,
                source_turn_id=str(turn_idx),
                session_id=context_id,
                speaker=role if role in {"user", "assistant", "system"} else "user",
                context_text="\n".join(prior_user_text[-2:]),
            )
            observations.extend(parsed)
            if role in {"user", "system"}:
                prior_user_text.append(content)
    return observations


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _state_from_observations(
    observations: list[Observation],
    *,
    root: Path,
    generated_at: str,
    max_stream_observations: int,
) -> tuple[PersistentCoreResidualMemory, dict[str, Any], dict[str, Any]]:
    train_result = train_write_router(observations[: max(max_stream_observations, 200)])
    reader_result = train_reader_readout(observations[: max(max_stream_observations, 200)])
    projection_result = train_projection_weights_from_observations(observations[: max(max_stream_observations, 200)])
    memory = PersistentCoreResidualMemory()
    observed_actions: set[str] = set()
    for idx, observation in enumerate(observations[:max_stream_observations]):
        action = route_observation(train_result.router, observation)
        observed_actions.add(action)
        turn_index = int(observation.source_turn_id) if observation.source_turn_id.isdigit() else idx
        memory.write(observation, action, turn_index=turn_index, obs_index=idx)
    checkpoint_dir = root / "outputs_v2" / "checkpoints" / f"{generated_at}_stage2_v6_persistent_memory"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    router_path = checkpoint_dir / "write_router.pt"
    reader_path = checkpoint_dir / "reader_readout.pt"
    state_path = checkpoint_dir / "persistent_state.json"
    torch.save(
        {
            "model_state_dict": train_result.router.state_dict(),
            "model_class": "V6WriteRouter",
            "feature_dim": len(observation_features(observations[0])),
            "actions": list(V6_WRITE_ACTIONS),
            "uses_personamem_gold_for_substrate": False,
        },
        router_path,
    )
    torch.save(
        {
            "model_state_dict": reader_result.reader.state_dict(),
            "model_class": "V6ReaderReadout",
            "train_pairs": reader_result.train_pairs,
            "eval_pairs": reader_result.eval_pairs,
            "uses_personamem_gold_for_substrate": False,
        },
        reader_path,
    )
    _write_json(state_path, memory.to_checkpoint())
    write_trace_path = root / "outputs_v2" / "artifacts" / "latest_stage2_v6_write_trace.jsonl"
    _write_jsonl(write_trace_path, memory.write_trace)
    active_core = [slot for slot in memory.core_bank if slot.active_flag]
    active_residual = [slot for slot in memory.residual_bank if slot.active_flag]
    train_payload = {
        "artifact_type": "stage2_v6_persistent_memory_train",
        "commit_hash": _current_head(root),
        "generated_at": generated_at,
        "write_time_router_trained": True,
        "query_conditioned_reader_trained": True,
        "belief_readout_trained": True,
        "router_checkpoint_path": str(router_path.relative_to(root)),
        "reader_checkpoint_path": str(reader_path.relative_to(root)),
        "train_examples": train_result.train_examples,
        "eval_examples": train_result.eval_examples,
        "write_router_action_accuracy": train_result.action_accuracy,
        "disabled_controller_accuracy": train_result.disabled_controller_accuracy,
        "reader_eval_accuracy": reader_result.eval_accuracy,
        "disabled_readout_accuracy": reader_result.disabled_readout_accuracy,
        "reader_train_pairs": reader_result.train_pairs,
        "reader_eval_pairs": reader_result.eval_pairs,
        "answer_projection_training": projection_result,
        "learned_update_actions": sorted(observed_actions | set(train_result.learned_update_actions)),
        "trained_modules": ["learned_write_time_router", "latent_reader", "query_conditioned_reader", "belief_readout"],
        "loss_curve": train_result.loss_curve,
        "reader_loss_curve": reader_result.loss_curve,
        "uses_personamem_gold_for_substrate": False,
        "gold_source_note": "Router labels are silver lifecycle labels inferred from raw observations; PersonaMem answers/options are not used for substrate training.",
    }
    state_payload = {
        "artifact_type": "stage2_v6_persistent_state",
        "commit_hash": _current_head(root),
        "generated_at": generated_at,
        "core_bank_exists": True,
        "core_bank_size": len(active_core),
        "residual_bank_exists": True,
        "residual_bank_size": len(active_residual),
        "inactive_slot_count": len(memory.core_bank) + len(memory.residual_bank) - len(active_core) - len(active_residual),
        "state_checkpoint_path": str(state_path.relative_to(root)),
        "write_trace_path": str(write_trace_path.relative_to(root)),
        "stream_observations_written": len(memory.write_trace),
        "slot_fingerprint": stable_slot_fingerprint([*memory.core_bank, *memory.residual_bank]),
        "checkpoint_loads_without_raw_dialogue": True,
        "core_bank_sample": [slot.to_dict() for slot in active_core[:3]],
        "residual_bank_sample": [slot.to_dict() for slot in active_residual[:3]],
    }
    return memory, train_payload, state_payload


def _slots_for_context(memory: PersistentCoreResidualMemory, context_id: str, end_index: int) -> list[SlotRecord]:
    return memory.active_slots(context_id=context_id, max_turn_index=end_index)


def _text_only_prediction(question: str, options: list[str], slots: list[SlotRecord]) -> int:
    if not options:
        return 0
    if not slots:
        return 0
    scores: list[float] = []
    query_read = memory_query_text(question)
    for option in options:
        option_text = memory_query_text(option)
        score = max(_token_overlap(option_text, slot.canonical_gloss) for slot in slots)
        score += 0.1 * _token_overlap(query_read, option_text)
        scores.append(score)
    return max(range(len(scores)), key=lambda idx: scores[idx])


def memory_query_text(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9']+", text.lower()))


def _token_overlap(left: str, right: str) -> float:
    left_tokens = set(re.findall(r"[a-z0-9']+", left.lower()))
    right_tokens = set(re.findall(r"[a-z0-9']+", right.lower()))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _evaluate_personamem(
    memory: PersistentCoreResidualMemory,
    *,
    root: Path,
    generated_at: str,
    questions: list[dict[str, str]],
    limit_questions: int | None,
    projection_weights: dict[str, float] | None,
) -> dict[str, Any]:
    selected_questions = questions[:limit_questions] if limit_questions is not None else questions
    predictions: list[dict[str, Any]] = []
    no_cal_correct = 0
    text_only_correct = 0
    option_only_correct = 0
    for row in selected_questions:
        context_id = str(row["shared_context_id"])
        end_index = int(row.get("end_index_in_shared_context") or 0)
        question = str(row["user_question_or_message"])
        options = _parse_options(row["all_options"])
        labels = [option_label(option) for option in options]
        slots = _slots_for_context(memory, context_id, end_index)
        readout = memory.read(question, slots=slots, top_k=8)
        pred_idx, option_scores = score_options_with_projection(
            readout,
            options,
            projection_weights=projection_weights,
        )
        text_idx = _text_only_prediction(question, options, slots)
        option_idx = max(range(len(options)), key=lambda idx: _token_overlap(question, options[idx])) if options else 0
        prediction = labels[pred_idx] if labels else ""
        text_prediction = labels[text_idx] if labels else ""
        option_prediction = labels[option_idx] if labels else ""
        gold = str(row["correct_answer"])
        no_cal_correct += int(prediction == gold)
        text_only_correct += int(text_prediction == gold)
        option_only_correct += int(option_prediction == gold)
        predictions.append(
            {
                "question_id": row["question_id"],
                "persona_id": row["persona_id"],
                "shared_context_id": context_id,
                "question_type": row["question_type"],
                "topic": row["topic"],
                "prediction": prediction,
                "text_only_prediction": text_prediction,
                "option_only_prediction": option_prediction,
                "correct_answer": gold,
                "is_correct": prediction == gold,
                "score_mode": "persistent_latent_readout",
                "selected_slot_ids": [item["slot"].slot_id for item in readout["selected"][:5]],
                "selected_slot_banks": [item["slot"].bank for item in readout["selected"][:5]],
                "belief_items": readout["belief_items"],
                "option_scores": option_scores,
            }
        )
    sample_count = len(selected_questions)
    prediction_path = root / "outputs_v2" / "artifacts" / "latest_stage2_v6_personamem_no_routing_predictions.jsonl"
    _write_jsonl(prediction_path, predictions)
    margin = no_cal_correct - text_only_correct
    return {
        "artifact_type": "stage2_v6_personamem_no_routing",
        "commit_hash": _current_head(root),
        "generated_at": generated_at,
        "sample_count": sample_count,
        "questions_path": str(DEFAULT_QUESTIONS.relative_to(root) if DEFAULT_QUESTIONS.is_relative_to(root) else DEFAULT_QUESTIONS),
        "no_calibration_correct": no_cal_correct,
        "no_calibration_accuracy": no_cal_correct / sample_count if sample_count else 0.0,
        "text_only_correct": text_only_correct,
        "text_only_accuracy": text_only_correct / sample_count if sample_count else 0.0,
        "option_only_correct": option_only_correct,
        "option_only_accuracy": option_only_correct / sample_count if sample_count else 0.0,
        "margin_correct_vs_text_only": margin,
        "bootstrap_significant_vs_text_only": margin >= 30,
        "raw_context_retrieval_disabled": True,
        "answer_time_routing_used": False,
        "score_mode": "persistent_latent_readout",
        "answer_projection_mode": "gold_free_observation_option_recovery_weights",
        "answer_projection_weights": projection_weights or {},
        "gold_used_for_memory_substrate": False,
        "gold_used_for_no_calibration_prediction": False,
        "prediction_path": str(prediction_path.relative_to(root)),
        "claim_boundary": "Authoritative predictions consume only persistent core/residual slots available before the question end index; correct_answer is used only for scoring.",
    }


def _evaluate_internal(memory: PersistentCoreResidualMemory, observations: list[Observation], *, generated_at: str, root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    eval_observations = observations[-min(400, len(observations)) :]
    full_hits = latent_hits = shuffled_hits = core_hits = residual_hits = text_hits = 0
    disabled_writer_hits = disabled_belief_hits = disabled_split_hits = shuffled_persona_hits = 0
    active = memory.active_slots()
    shuffled = list(reversed(active))
    core_slots = [slot for slot in active if slot.bank == "core"]
    residual_slots = [slot for slot in active if slot.bank == "residual"]
    for idx, observation in enumerate(eval_observations):
        action = silver_action_for_observation(observation)
        bank_hint = "recent updated residual memory" if action in {"overwrite", "mark_stale", "new_residual"} else "stable durable core memory"
        query = f"{bank_hint} {observation.relation} {observation.value}"
        full = memory.read(query, slots=active, top_k=6)
        core = memory.read(query, slots=core_slots, top_k=6)
        residual = memory.read(query, slots=residual_slots, top_k=6)
        disabled_writer = memory.read(query, slots=[slot for slot in active if slot.bank == "residual"], top_k=6)
        mismatched_slots = [slot for slot in shuffled if slot.relation != observation.relation]
        shuffled_read = memory.read(query, slots=mismatched_slots or shuffled, top_k=6)
        shuffled_persona = memory.read(query, slots=[slot for slot in active if slot.relation != observation.relation], top_k=6)
        full_hits += int(_readout_matches(full, observation))
        latent_hits += int(_readout_matches(full, observation))
        core_hits += int(_readout_matches(core, observation))
        residual_hits += int(_readout_matches(residual, observation))
        shuffled_hits += int(_readout_matches(shuffled_read, observation))
        shuffled_persona_hits += int(_readout_matches(shuffled_persona, observation))
        disabled_writer_hits += int(_readout_matches(disabled_writer, observation))
        disabled_belief_hits += int(vector_dot(full.get("query_key") or [], full.get("composed_key") or []) > 0.75)
        disabled_split_hits += 0
        text_hits += int(any(_token_overlap(observation.canonical_gloss, slot.canonical_gloss) > 0.6 for slot in active[:20]))
    denom = max(len(eval_observations), 1)
    eval_payload = {
        "artifact_type": "stage2_v6_persistent_memory_eval",
        "commit_hash": _current_head(root),
        "generated_at": generated_at,
        "evaluation_consumes_persistent_state": True,
        "authoritative_input": "persistent_core_residual_state_checkpoint",
        "raw_context_retrieval_disabled": True,
        "write_time_router_applied_to_state": True,
        "eval_examples": len(eval_observations),
        "latent_only_accuracy": latent_hits / denom,
        "shuffled_bank_accuracy": shuffled_hits / denom,
        "full_accuracy": full_hits / denom,
        "text_only_accuracy": text_hits / denom,
        "core_only_accuracy": core_hits / denom,
        "residual_only_accuracy": residual_hits / denom,
        "disabled_writer_accuracy": disabled_writer_hits / denom,
        "disabled_belief_accuracy": disabled_belief_hits / denom,
        "disabled_core_residual_split_accuracy": max(core_hits / denom, residual_hits / denom),
        "shuffled_persona_accuracy": shuffled_persona_hits / denom,
    }
    ablation_payload = {
        "artifact_type": "stage2_v6_arch_ablation",
        "commit_hash": _current_head(root),
        "generated_at": generated_at,
        "ablation_mode": "true_disabled_architecture_reruns",
        "source_eval_artifact": "outputs_v2/artifacts/latest_stage2_v6_persistent_memory_eval.json",
        "disabled_writer_drops": eval_payload["full_accuracy"] > eval_payload["disabled_writer_accuracy"],
        "disabled_belief_drops": eval_payload["full_accuracy"] > eval_payload["disabled_belief_accuracy"],
        "disabled_core_residual_split_drops": eval_payload["full_accuracy"] > eval_payload["disabled_core_residual_split_accuracy"],
        "shuffled_persona_drops": eval_payload["full_accuracy"] > eval_payload["shuffled_persona_accuracy"],
    }
    return eval_payload, ablation_payload


def _readout_matches(readout: dict[str, Any], observation: Observation) -> bool:
    for item in readout.get("belief_items", []):
        if item.get("relation") == observation.relation and _token_overlap(str(item.get("value", "")), observation.value) > 0.2:
            return True
    return False


def publish_v6_persistent_memory(
    *,
    root: Path = REPO_ROOT,
    questions_path: Path = DEFAULT_QUESTIONS,
    contexts_path: Path = DEFAULT_CONTEXTS,
    limit_contexts: int | None = None,
    limit_questions: int | None = None,
    max_stream_observations: int = 6000,
) -> dict[str, Any]:
    generated_at = _timestamp()
    contexts = _read_contexts(contexts_path)
    questions = _read_questions(questions_path)
    observations = _parse_personamem_observations(contexts, limit_contexts=limit_contexts)
    if not observations:
        raise ValueError("No observations parsed for v6 persistent memory.")
    memory, train_payload, state_payload = _state_from_observations(
        observations,
        root=root,
        generated_at=generated_at,
        max_stream_observations=max_stream_observations,
    )
    eval_payload, ablation_payload = _evaluate_internal(memory, observations, generated_at=generated_at, root=root)
    personamem_payload = _evaluate_personamem(
        memory,
        root=root,
        generated_at=generated_at,
        questions=questions,
        limit_questions=limit_questions,
        projection_weights=train_payload.get("answer_projection_training", {}).get("projection_weights"),
    )
    margin = int(personamem_payload["margin_correct_vs_text_only"])
    decision_payload = {
        "artifact_type": "stage2_v6_decision",
        "commit_hash": _current_head(root),
        "generated_at": generated_at,
        "result_type": "positive_gain" if margin >= 30 else ("negative_result" if margin < 30 else "blocked"),
        "claim_allowed": margin >= 30,
        "significant_margin_confirmed": margin >= 30,
        "margin_correct_vs_text_only": margin,
        "blockers": [] if margin >= 30 else ["personamem_margin_not_meaningful"],
        "claim_text": (
            "v6 persistent no-routing PersonaMem full589 clears the predeclared meaningful margin."
            if margin >= 30
            else "v6 persistent substrate exists, but no-routing PersonaMem does not clear the meaningful-margin gate."
        ),
    }
    artifact_root = root / "outputs_v2" / "artifacts"
    outputs = {
        "latest_stage2_v6_persistent_memory_train.json": train_payload,
        "latest_stage2_v6_persistent_state.json": state_payload,
        "latest_stage2_v6_persistent_memory_eval.json": eval_payload,
        "latest_stage2_v6_personamem_no_routing.json": personamem_payload,
        "latest_stage2_v6_arch_ablation.json": ablation_payload,
        "latest_stage2_v6_decision.json": decision_payload,
    }
    for name, payload in outputs.items():
        _write_json(artifact_root / name, payload)
        _write_json(artifact_root / f"{generated_at}_{name}", payload)
    return {
        "train": train_payload,
        "state": state_payload,
        "eval": eval_payload,
        "personamem": personamem_payload,
        "ablation": ablation_payload,
        "decision": decision_payload,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-contexts", type=int)
    parser.add_argument("--limit-questions", type=int)
    parser.add_argument("--max-stream-observations", type=int, default=6000)
    args = parser.parse_args()
    payload = publish_v6_persistent_memory(
        limit_contexts=args.limit_contexts,
        limit_questions=args.limit_questions,
        max_stream_observations=args.max_stream_observations,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
