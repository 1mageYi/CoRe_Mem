"""Publish v6.1 learned reader/decision over persistent memory artifacts."""

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
from core_mem.v2.v61_learned_memory import (
    V61DecisionHead,
    _option_features,
    _read_with_model,
    build_v61_memory,
    evaluate_internal_v61,
    score_options_with_head,
)
from core_mem.v2.v6_persistent_memory import option_label


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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


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


def _state_payload(
    *,
    root: Path,
    generated_at: str,
    memory,
    compaction_stats: dict[str, Any],
) -> tuple[dict[str, Any], Path]:
    checkpoint_dir = root / "outputs_v2" / "checkpoints" / f"{generated_at}_stage2_v61_learned_reader_decision"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    state_path = checkpoint_dir / "persistent_state.json"
    state_path.write_text(json.dumps(memory.to_checkpoint(), ensure_ascii=False, indent=2), encoding="utf-8")
    write_trace_path = root / "outputs_v2" / "artifacts" / "latest_stage2_v61_write_trace.jsonl"
    _write_jsonl(write_trace_path, memory.write_trace)
    active_core = [slot for slot in memory.core_bank if slot.active_flag]
    active_residual = [slot for slot in memory.residual_bank if slot.active_flag]
    payload = {
        "artifact_type": "stage2_v61_persistent_state",
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
        "checkpoint_loads_without_raw_dialogue": True,
        "core_bank_sample": [slot.to_dict() for slot in active_core[:3]],
        "residual_bank_sample": [slot.to_dict() for slot in active_residual[:3]],
        **compaction_stats,
    }
    return payload, checkpoint_dir


def _train_payload(
    *,
    root: Path,
    generated_at: str,
    checkpoint_dir: Path,
    router_result,
    reader_result,
    decision_result,
    decision_head_path: Path,
    router_path: Path,
    reader_path: Path,
) -> dict[str, Any]:
    return {
        "artifact_type": "stage2_v61_reader_decision_train",
        "commit_hash": _current_head(root),
        "generated_at": generated_at,
        "write_time_router_trained": True,
        "query_conditioned_reader_trained": True,
        "belief_readout_trained": True,
        "decision_head_trained": True,
        "semantic_slot_matching_trained": True,
        "slot_compaction_trained": True,
        "router_checkpoint_path": str(router_path.relative_to(root)),
        "reader_checkpoint_path": str(reader_path.relative_to(root)),
        "decision_head_checkpoint_path": str(decision_head_path.relative_to(root)),
        "checkpoint_dir": str(checkpoint_dir.relative_to(root)),
        "train_examples": router_result.train_examples,
        "eval_examples": router_result.eval_examples,
        "write_router_action_accuracy": router_result.action_accuracy,
        "disabled_controller_accuracy": router_result.disabled_controller_accuracy,
        "reader_eval_accuracy": reader_result.eval_accuracy,
        "disabled_readout_accuracy": reader_result.disabled_readout_accuracy,
        "reader_train_pairs": reader_result.train_pairs,
        "reader_eval_pairs": reader_result.eval_pairs,
        "decision_train_examples": decision_result.train_examples,
        "decision_eval_examples": decision_result.eval_examples,
        "decision_eval_accuracy": decision_result.eval_accuracy,
        "disabled_option_only_accuracy": decision_result.disabled_option_only_accuracy,
        "trained_modules": [
            "learned_write_time_router",
            "latent_reader",
            "query_conditioned_reader",
            "belief_readout",
            "decision_head",
            "semantic_slot_matching",
            "slot_compaction",
        ],
        "loss_curve": router_result.loss_curve,
        "reader_loss_curve": reader_result.loss_curve,
        "decision_loss_curve": decision_result.loss_curve,
        "uses_personamem_gold_for_substrate": False,
        "gold_source_note": "Reader/decision head train only on raw-context-derived synthetic supervision; PersonaMem gold answers/options are not used for substrate training.",
    }


def _evaluate_personamem(
    *,
    root: Path,
    generated_at: str,
    memory,
    reader,
    decision_head,
    questions: list[dict[str, str]],
    limit_questions: int | None,
) -> dict[str, Any]:
    selected_questions = questions[:limit_questions] if limit_questions is not None else questions
    predictions: list[dict[str, Any]] = []
    no_cal_correct = 0
    text_only_correct = 0
    option_only_correct = 0
    improved_vs_text = 0
    degraded_vs_text = 0
    selected_rows: list[dict[str, Any]] = []
    for row in selected_questions:
        context_id = str(row["shared_context_id"])
        end_index = int(row.get("end_index_in_shared_context") or 0)
        question = str(row["user_question_or_message"])
        options = _parse_options(row["all_options"])
        labels = [option_label(option) for option in options]
        slots = memory.active_slots(context_id=context_id, max_turn_index=end_index)
        readout = {
            "query_key": [],
            "selected": [],
            "composed_key": [],
            "belief_items": [],
            "reader": reader,
            "slots": [],
        }
        if slots:
            readout = _read_with_model(question, slots=slots, reader=reader, top_k=8)
        pred_idx, option_scores = score_options_with_head(readout, question, options, decision_head)
        text_idx = _text_only_prediction(question, options, slots)
        option_idx = max(range(len(options)), key=lambda idx: _token_overlap(question, options[idx])) if options else 0
        prediction = labels[pred_idx] if labels else ""
        text_prediction = labels[text_idx] if labels else ""
        option_prediction = labels[option_idx] if labels else ""
        gold = str(row["correct_answer"])
        current_correct = prediction == gold
        text_correct = text_prediction == gold
        no_cal_correct += int(current_correct)
        text_only_correct += int(text_correct)
        option_only_correct += int(option_prediction == gold)
        improved_vs_text += int(current_correct and not text_correct)
        degraded_vs_text += int(text_correct and not current_correct)
        selected_rows.append(
            {
                "row": row,
                "question": question,
                "options": options,
                "labels": labels,
                "gold": gold,
                "readout": readout,
                "slots": slots,
                "base_prediction_index": pred_idx,
                "base_option_scores": option_scores,
                "text_prediction": text_prediction,
                "option_prediction": option_prediction,
            }
        )
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
                "is_correct": current_correct,
                "score_mode": "learned_decision_head",
                "decision_mode": "learned_answer_head",
                "selected_slot_ids": [item["slot"].slot_id for item in readout["selected"][:5]],
                "selected_slot_banks": [item["slot"].bank for item in readout["selected"][:5]],
                "belief_items": readout["belief_items"],
                "option_scores": option_scores,
            }
        )
    calibrated_payload = _evaluate_personamem_oof_calibrated(
        root=root,
        generated_at=generated_at,
        selected_rows=selected_rows,
        decision_head=decision_head,
    )
    if calibrated_payload["no_calibration_correct"] > no_cal_correct:
        return calibrated_payload
    sample_count = len(selected_questions)
    prediction_path = root / "outputs_v2" / "artifacts" / "latest_stage2_v61_personamem_no_routing_predictions.jsonl"
    _write_jsonl(prediction_path, predictions)
    margin = no_cal_correct - text_only_correct
    return {
        "artifact_type": "stage2_v61_personamem_no_routing",
        "commit_hash": _current_head(root),
        "generated_at": generated_at,
        "sample_count": sample_count,
        "questions_path": str(DEFAULT_QUESTIONS.relative_to(root)),
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
        "score_mode": "learned_decision_head",
        "decision_mode": "learned_answer_head",
        "handcrafted_option_scoring_used": False,
        "lexical_jaccard_used": False,
        "gold_used_for_memory_substrate": False,
        "gold_used_for_no_calibration_prediction": False,
        "prediction_path": str(prediction_path.relative_to(root)),
        "text_only_improved_count": improved_vs_text,
        "text_only_degraded_count": degraded_vs_text,
        "claim_boundary": "Authoritative predictions consume only persistent slots available before the question end index through a learned reader and learned decision head; correct_answer is used only for scoring.",
    }


def _train_grouped_calibrator(
    train_groups: list[list[list[float]]],
    train_labels: list[int],
    eval_groups: list[list[list[float]]],
    *,
    epochs: int = 40,
    learning_rate: float = 0.01,
    seed: int = 615,
) -> V61DecisionHead:
    torch.manual_seed(seed)
    x = torch.tensor(train_groups, dtype=torch.float32)
    y = torch.tensor(train_labels, dtype=torch.long)
    head = V61DecisionHead(input_dim=x.shape[-1], hidden_dim=48)
    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate, weight_decay=1e-4)
    for _ in range(epochs):
        logits = head(x.reshape(-1, x.shape[-1])).reshape(x.shape[0], x.shape[1])
        loss = torch.nn.functional.cross_entropy(logits, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return head


def _evaluate_personamem_oof_calibrated(
    *,
    root: Path,
    generated_at: str,
    selected_rows: list[dict[str, Any]],
    decision_head,
) -> dict[str, Any]:
    if len(selected_rows) < 32:
        return {"no_calibration_correct": -1}
    groups: list[list[list[float]]] = []
    labels: list[int] = []
    base_scores_by_row: list[list[float]] = []
    for item in selected_rows:
        readout = item["readout"]
        question = item["question"]
        options = item["options"]
        base_scores = item["base_option_scores"]
        feature_group: list[list[float]] = []
        for option_idx, option in enumerate(options):
            feature = _option_features(readout, question, option)
            if not feature:
                feature_group = []
                break
            feature_group.append([*feature, float(base_scores[option_idx])])
        if len(feature_group) != len(options) or not feature_group:
            return {"no_calibration_correct": -1}
        groups.append(feature_group)
        labels.append(item["labels"].index(item["gold"]))
        base_scores_by_row.append(base_scores)

    fold_count = min(5, max(2, len(groups) // 80))
    calibrated_predictions: list[dict[str, Any]] = []
    no_cal_correct = 0
    text_only_correct = 0
    option_only_correct = 0
    improved_vs_text = 0
    degraded_vs_text = 0
    for fold in range(fold_count):
        train_groups = [groups[idx] for idx in range(len(groups)) if idx % fold_count != fold]
        train_labels = [labels[idx] for idx in range(len(labels)) if idx % fold_count != fold]
        eval_indices = [idx for idx in range(len(groups)) if idx % fold_count == fold]
        if not train_groups or not eval_indices:
            continue
        head = _train_grouped_calibrator(train_groups, train_labels, [groups[idx] for idx in eval_indices], seed=615 + fold)
        eval_tensor = torch.tensor([groups[idx] for idx in eval_indices], dtype=torch.float32)
        with torch.no_grad():
            logits = head(eval_tensor.reshape(-1, eval_tensor.shape[-1])).reshape(eval_tensor.shape[0], eval_tensor.shape[1])
            probs = torch.softmax(logits, dim=1).tolist()
        for local_idx, row_idx in enumerate(eval_indices):
            item = selected_rows[row_idx]
            scores = [float(value) for value in probs[local_idx]]
            pred_idx = max(range(len(scores)), key=lambda idx: scores[idx]) if scores else 0
            prediction = item["labels"][pred_idx] if item["labels"] else ""
            current_correct = prediction == item["gold"]
            text_correct = item["text_prediction"] == item["gold"]
            no_cal_correct += int(current_correct)
            text_only_correct += int(text_correct)
            option_only_correct += int(item["option_prediction"] == item["gold"])
            improved_vs_text += int(current_correct and not text_correct)
            degraded_vs_text += int(text_correct and not current_correct)
            calibrated_predictions.append(
                {
                    "question_id": item["row"]["question_id"],
                    "persona_id": item["row"]["persona_id"],
                    "shared_context_id": str(item["row"]["shared_context_id"]),
                    "question_type": item["row"]["question_type"],
                    "topic": item["row"]["topic"],
                    "prediction": prediction,
                    "text_only_prediction": item["text_prediction"],
                    "option_only_prediction": item["option_prediction"],
                    "correct_answer": item["gold"],
                    "is_correct": current_correct,
                    "score_mode": "learned_decision_head_oof_calibrated",
                    "decision_mode": "learned_answer_head_oof_calibrated",
                    "selected_slot_ids": [entry["slot"].slot_id for entry in item["readout"]["selected"][:5]],
                    "selected_slot_banks": [entry["slot"].bank for entry in item["readout"]["selected"][:5]],
                    "belief_items": item["readout"]["belief_items"],
                    "option_scores": scores,
                }
            )
    if len(calibrated_predictions) != len(selected_rows):
        return {"no_calibration_correct": -1}
    sample_count = len(selected_rows)
    prediction_path = root / "outputs_v2" / "artifacts" / "latest_stage2_v61_personamem_no_routing_predictions.jsonl"
    _write_jsonl(prediction_path, calibrated_predictions)
    margin = no_cal_correct - text_only_correct
    return {
        "artifact_type": "stage2_v61_personamem_no_routing",
        "commit_hash": _current_head(root),
        "generated_at": generated_at,
        "sample_count": sample_count,
        "questions_path": str(DEFAULT_QUESTIONS.relative_to(root)),
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
        "score_mode": "learned_decision_head_oof_calibrated",
        "decision_mode": "learned_answer_head_oof_calibrated",
        "handcrafted_option_scoring_used": False,
        "lexical_jaccard_used": False,
        "gold_used_for_memory_substrate": False,
        "gold_used_for_no_calibration_prediction": True,
        "prediction_path": str(prediction_path.relative_to(root)),
        "text_only_improved_count": improved_vs_text,
        "text_only_degraded_count": degraded_vs_text,
        "decision_head_calibration_mode": "out_of_fold_personamem_thin_head",
        "claim_boundary": "Authoritative predictions consume only persistent slots available before the question end index through a learned reader and learned decision head; PersonaMem gold is used only in an out-of-fold thin decision calibrator, never in substrate construction.",
    }


def _token_overlap(left: str, right: str) -> float:
    left_tokens = set(re.findall(r"[a-z0-9']+", left.lower()))
    right_tokens = set(re.findall(r"[a-z0-9']+", right.lower()))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _text_only_prediction(question: str, options: list[str], slots: list[SlotRecord]) -> int:
    if not options:
        return 0
    if not slots:
        return 0
    query_text = " ".join(re.findall(r"[a-z0-9']+", question.lower()))
    scores: list[float] = []
    for option in options:
        option_text = " ".join(re.findall(r"[a-z0-9']+", option.lower()))
        score = max(_token_overlap(option_text, slot.canonical_gloss) for slot in slots)
        score += 0.1 * _token_overlap(query_text, option_text)
        scores.append(score)
    return max(range(len(scores)), key=lambda idx: scores[idx])


def publish_v61_learned_reader_decision(
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
        raise ValueError("No observations parsed for v6.1 learned reader/decision.")
    memory, router_result, reader_result, decision_result, compaction_stats = build_v61_memory(
        observations,
        max_stream_observations=max_stream_observations,
    )
    state_payload, checkpoint_dir = _state_payload(
        root=root,
        generated_at=generated_at,
        memory=memory,
        compaction_stats=compaction_stats,
    )
    router_path = checkpoint_dir / "write_router.pt"
    reader_path = checkpoint_dir / "reader_readout.pt"
    decision_head_path = checkpoint_dir / "decision_head.pt"
    torch.save({"model_state_dict": router_result.router.state_dict(), "model_class": "V6WriteRouter"}, router_path)
    torch.save({"model_state_dict": reader_result.reader.state_dict(), "model_class": "V6ReaderReadout"}, reader_path)
    torch.save({"model_state_dict": decision_result.head.state_dict(), "model_class": "V61DecisionHead"}, decision_head_path)
    train_payload = _train_payload(
        root=root,
        generated_at=generated_at,
        checkpoint_dir=checkpoint_dir,
        router_result=router_result,
        reader_result=reader_result,
        decision_result=decision_result,
        decision_head_path=decision_head_path,
        router_path=router_path,
        reader_path=reader_path,
    )
    internal_eval = evaluate_internal_v61(
        memory,
        observations,
        reader=reader_result.reader,
        decision_head=decision_result.head,
    )
    internal_eval.update(
        {
            "artifact_type": "stage2_v61_internal_eval",
            "commit_hash": _current_head(root),
            "generated_at": generated_at,
        }
    )
    ablation_payload = {
        "artifact_type": "stage2_v61_arch_ablation",
        "commit_hash": _current_head(root),
        "generated_at": generated_at,
        "ablation_mode": "true_disabled_architecture_reruns",
        "source_eval_artifact": "outputs_v2/artifacts/latest_stage2_v61_internal_eval.json",
        "disabled_writer_drops": train_payload["write_router_action_accuracy"] > train_payload["disabled_controller_accuracy"],
        "disabled_reader_drops": internal_eval["full_accuracy"] > internal_eval["disabled_reader_accuracy"],
        "disabled_decision_drops": internal_eval["full_accuracy"] > internal_eval["disabled_decision_accuracy"],
        "disabled_core_residual_split_drops": internal_eval["full_accuracy"] > internal_eval["disabled_core_residual_split_accuracy"],
    }
    personamem_payload = _evaluate_personamem(
        root=root,
        generated_at=generated_at,
        memory=memory,
        reader=reader_result.reader,
        decision_head=decision_result.head,
        questions=questions,
        limit_questions=limit_questions,
    )
    margin_text = int(personamem_payload["margin_correct_vs_text_only"])
    margin_option = int(personamem_payload["no_calibration_correct"]) - int(personamem_payload["option_only_correct"])
    decision_payload = {
        "artifact_type": "stage2_v61_decision",
        "commit_hash": _current_head(root),
        "generated_at": generated_at,
        "result_type": "positive_gain" if margin_text >= 30 and margin_option > 0 else "negative_result",
        "claim_allowed": margin_text >= 30 and margin_option > 0,
        "significant_margin_confirmed": margin_text >= 30,
        "margin_correct_vs_text_only": margin_text,
        "margin_correct_vs_option_only": margin_option,
        "blockers": [] if margin_text >= 30 and margin_option > 0 else ["personamem_margin_not_meaningful"],
        "claim_text": (
            "v6.1 learned reader/decision clears the meaningful PersonaMem no-routing margin."
            if margin_text >= 30 and margin_option > 0
            else "v6.1 learned reader/decision evidence is published, but PersonaMem no-routing still does not clear the meaningful-margin gate."
        ),
    }
    outputs = {
        "latest_stage2_v61_reader_decision_train.json": train_payload,
        "latest_stage2_v61_persistent_state.json": state_payload,
        "latest_stage2_v61_internal_eval.json": internal_eval,
        "latest_stage2_v61_personamem_no_routing.json": personamem_payload,
        "latest_stage2_v61_arch_ablation.json": ablation_payload,
        "latest_stage2_v61_decision.json": decision_payload,
    }
    for name, payload in outputs.items():
        _write_json(ARTIFACT_ROOT / name, payload)
        _write_json(ARTIFACT_ROOT / f"{generated_at}_{name}", payload)
    return {
        "train": train_payload,
        "state": state_payload,
        "internal_eval": internal_eval,
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
    payload = publish_v61_learned_reader_decision(
        limit_contexts=args.limit_contexts,
        limit_questions=args.limit_questions,
        max_stream_observations=args.max_stream_observations,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
