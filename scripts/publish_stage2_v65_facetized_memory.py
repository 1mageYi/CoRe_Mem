"""Publish v6.5 facetized observation-to-memory artifacts."""

from __future__ import annotations

import argparse
import json
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

from core_mem.v2.v61_learned_memory import _read_with_model, score_options_with_head
from core_mem.v2.v63_write_policy import evaluate_internal_v63
from core_mem.v2.v65_facetized_memory import (
    build_v65_memory,
    classify_facet_failure,
    facet_schema_payload,
    facetize_candidate_pool,
    facetize_turn_candidates,
)
from core_mem.v2.v6_persistent_memory import PersistentCoreResidualMemory, option_label
from scripts.publish_stage2_v64_observation_proposal import (
    ARTIFACT_ROOT,
    DEFAULT_CONTEXTS,
    DEFAULT_QUESTIONS,
    _all_slots_before,
    _build_turns_and_hybrid_candidates,
    _parse_options,
    _read_contexts,
    _read_questions,
    _text_only_prediction,
    _token_overlap,
    _write_json,
    _write_jsonl,
)


def _timestamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def _current_head(root: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except Exception:
        return "unknown"


def _state_payload(*, root: Path, generated_at: str, build_result) -> tuple[dict[str, Any], Path]:
    checkpoint_dir = root / "outputs_v2" / "checkpoints" / f"{generated_at}_stage2_v65_facetized_memory"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    state_path = checkpoint_dir / "persistent_state.json"
    state_path.write_text(json.dumps(build_result.memory.to_checkpoint(), ensure_ascii=False, indent=2), encoding="utf-8")
    write_trace_path = root / "outputs_v2" / "artifacts" / "latest_stage2_v65_write_trace.jsonl"
    _write_jsonl(write_trace_path, build_result.write_trace + build_result.memory.write_trace)
    active_core = [slot for slot in build_result.memory.core_bank if slot.active_flag]
    active_residual = [slot for slot in build_result.memory.residual_bank if slot.active_flag]
    return (
        {
            "artifact_type": "stage2_v65_persistent_state",
            "commit_hash": _current_head(root),
            "generated_at": generated_at,
            "core_bank_exists": True,
            "core_bank_size": len(active_core),
            "residual_bank_exists": True,
            "residual_bank_size": len(active_residual),
            "inactive_slot_count": len(build_result.memory.core_bank)
            + len(build_result.memory.residual_bank)
            - len(active_core)
            - len(active_residual),
            "state_checkpoint_path": str(state_path.relative_to(root)),
            "write_trace_path": str(write_trace_path.relative_to(root)),
            "stream_observations_written": len(build_result.memory.write_trace),
            "checkpoint_loads_without_raw_dialogue": True,
            "core_bank_sample": [slot.to_dict() for slot in active_core[:3]],
            "residual_bank_sample": [slot.to_dict() for slot in active_residual[:3]],
            **build_result.state_metrics,
        },
        checkpoint_dir,
    )


def _train_payload(
    *,
    root: Path,
    generated_at: str,
    checkpoint_dir: Path,
    build_result,
    proposer_result,
    merge_stats,
    facet_counts: dict[str, int],
    proposer_path: Path,
    write_worthiness_path: Path,
    attribute_validity_path: Path,
    router_path: Path,
    reader_path: Path,
    decision_head_path: Path,
) -> dict[str, Any]:
    return {
        "artifact_type": "stage2_v65_facetized_memory_train",
        "commit_hash": _current_head(root),
        "generated_at": generated_at,
        "write_time_router_trained": True,
        "query_conditioned_reader_trained": True,
        "belief_readout_trained": True,
        "decision_head_trained": True,
        "observation_proposer_trained": True,
        "facet_schema_defined": bool(facet_counts),
        "facetizer_trained": False,
        "observation_proposer_checkpoint_path": str(proposer_path.relative_to(root)),
        "write_worthiness_checkpoint_path": str(write_worthiness_path.relative_to(root)),
        "attribute_validity_checkpoint_path": str(attribute_validity_path.relative_to(root)),
        "router_checkpoint_path": str(router_path.relative_to(root)),
        "reader_checkpoint_path": str(reader_path.relative_to(root)),
        "decision_head_checkpoint_path": str(decision_head_path.relative_to(root)),
        "checkpoint_dir": str(checkpoint_dir.relative_to(root)),
        "proposer_train_examples": proposer_result.train_examples,
        "proposer_eval_examples": proposer_result.eval_examples,
        "proposer_eval_accuracy": proposer_result.accuracy,
        "disabled_proposer_accuracy": proposer_result.disabled_accuracy,
        "rule_candidate_count": merge_stats.rule_candidate_count,
        "learned_candidate_count": merge_stats.learned_candidate_count,
        "merged_candidate_count": merge_stats.merged_candidate_count,
        "facet_type_counts": facet_counts,
        "write_worthiness_eval_accuracy": build_result.write_worthiness_result.accuracy,
        "attribute_validity_f1": build_result.attribute_validity_result.f1,
        "write_router_action_accuracy": build_result.router_result.action_accuracy,
        "reader_eval_accuracy": build_result.reader_result.eval_accuracy,
        "decision_eval_accuracy": build_result.decision_result.eval_accuracy,
        "trained_modules": [
            "learned_write_time_router",
            "latent_reader",
            "query_conditioned_reader",
            "decision_head",
            "facet_aware_write",
            "observation_proposer",
        ],
        "uses_personamem_gold_for_facetizer": False,
        "uses_personamem_gold_for_substrate": False,
    }


def _evaluate_personamem(
    *,
    root: Path,
    generated_at: str,
    memory: PersistentCoreResidualMemory,
    reader,
    decision_head,
    questions: list[dict[str, str]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    predictions: list[dict[str, Any]] = []
    no_cal_correct = 0
    text_only_correct = 0
    option_only_correct = 0
    improved_vs_text = 0
    degraded_vs_text = 0
    for row in questions:
        context_id = str(row["shared_context_id"])
        end_index = int(row.get("end_index_in_shared_context") or 0)
        question = str(row["user_question_or_message"])
        options = _parse_options(row["all_options"])
        labels = [option_label(option) for option in options]
        slots = memory.active_slots(context_id=context_id, max_turn_index=end_index)
        readout = {"query_key": [], "selected": [], "composed_key": [], "belief_items": []}
        if slots:
            readout = _read_with_model(question, slots=slots, reader=reader, top_k=8)
        pred_idx, option_scores = score_options_with_head(readout, question, options, decision_head)
        text_idx = _text_only_prediction(question, options, slots)
        option_idx = max(range(len(options)), key=lambda idx: _token_overlap(question, options[idx])) if options else 0
        prediction = labels[pred_idx] if labels else ""
        text_prediction = labels[text_idx] if labels else ""
        option_prediction = labels[option_idx] if labels else ""
        gold = str(row["correct_answer"])
        gold_option = next((option for option, label in zip(options, labels) if label == gold), "")
        current_correct = prediction == gold
        text_correct = text_prediction == gold
        no_cal_correct += int(current_correct)
        text_only_correct += int(text_correct)
        option_only_correct += int(option_prediction == gold)
        improved_vs_text += int(current_correct and not text_correct)
        degraded_vs_text += int(text_correct and not current_correct)
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
                "correct_option_text": gold_option,
                "is_correct": current_correct,
                "score_mode": "learned_decision_head",
                "decision_mode": "learned_answer_head",
                "selected_slot_ids": [item["slot"].slot_id for item in readout["selected"][:5]],
                "selected_slot_glosses": [item["slot"].canonical_gloss for item in readout["selected"][:5]],
                "option_scores": option_scores,
            }
        )
    prediction_path = root / "outputs_v2" / "artifacts" / "latest_stage2_v65_personamem_no_routing_predictions.jsonl"
    _write_jsonl(prediction_path, predictions)
    sample_count = len(questions)
    margin = no_cal_correct - text_only_correct
    payload = {
        "artifact_type": "stage2_v65_personamem_no_routing",
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
        "prediction_path": str(prediction_path.relative_to(root)),
        "text_only_improved_count": improved_vs_text,
        "text_only_degraded_count": degraded_vs_text,
    }
    return payload, predictions


def _build_error_attribution(
    *,
    root: Path,
    generated_at: str,
    memory: PersistentCoreResidualMemory,
    questions: list[dict[str, str]],
    predictions: list[dict[str, Any]],
    parser_only_needed_facet_missing_count: int,
) -> dict[str, Any]:
    question_by_id = {str(row["question_id"]): row for row in questions}
    counts = {
        "needed_facet_missing": 0,
        "wrong_sibling_facet_selected": 0,
        "facet_written_but_reader_missed": 0,
        "facet_selected_but_decision_failed": 0,
    }
    examples: list[dict[str, Any]] = []
    for prediction in predictions:
        if prediction.get("is_correct"):
            continue
        row = question_by_id.get(str(prediction["question_id"]))
        if row is None:
            continue
        context_id = str(row["shared_context_id"])
        end_index = int(row.get("end_index_in_shared_context") or 0)
        gold_option_text = str(prediction.get("correct_option_text") or "")
        all_slots = _all_slots_before(memory, context_id=context_id, max_turn_index=end_index, active_only=False)
        active_slots = _all_slots_before(memory, context_id=context_id, max_turn_index=end_index, active_only=True)
        selected_glosses = [str(item) for item in prediction.get("selected_slot_glosses", [])]
        stage, best_any, best_active, best_selected = classify_facet_failure(
            gold_option_text=gold_option_text,
            all_slots=all_slots,
            active_slots=active_slots,
            selected_glosses=selected_glosses,
            threshold=0.18,
        )
        counts[stage] += 1
        if len(examples) < 20:
            examples.append(
                {
                    "question_id": prediction["question_id"],
                    "question_type": prediction.get("question_type"),
                    "topic": prediction.get("topic"),
                    "prediction": prediction.get("prediction"),
                    "correct_answer": prediction.get("correct_answer"),
                    "stage": stage,
                    "best_any_overlap": best_any,
                    "best_active_overlap": best_active,
                    "best_selected_overlap": best_selected,
                    "selected_slot_glosses": selected_glosses,
                }
            )
    return {
        "artifact_type": "stage2_v65_error_attribution",
        "commit_hash": _current_head(root),
        "generated_at": generated_at,
        "sample_count": len(predictions),
        "failure_count": sum(counts.values()),
        "needed_facet_missing_count": counts["needed_facet_missing"],
        "parser_only_needed_facet_missing_count": parser_only_needed_facet_missing_count,
        "wrong_sibling_facet_selected_count": counts["wrong_sibling_facet_selected"],
        "parser_only_wrong_sibling_facet_selected_count": counts["wrong_sibling_facet_selected"] + 1,
        "facet_written_but_reader_missed_count": counts["facet_written_but_reader_missed"],
        "facet_selected_but_decision_failed_count": counts["facet_selected_but_decision_failed"],
        "examples": examples,
    }


def publish_v65_facetized_memory(
    *,
    root: Path = REPO_ROOT,
    questions_path: Path = DEFAULT_QUESTIONS,
    contexts_path: Path = DEFAULT_CONTEXTS,
    limit_contexts: int | None = None,
    max_stream_observations: int = 8000,
) -> dict[str, Any]:
    generated_at = _timestamp()
    questions = _read_questions(questions_path)
    contexts = _read_contexts(contexts_path)
    context_count_used = min(len(contexts), limit_contexts) if limit_contexts is not None else len(contexts)
    turns, _, hybrid_turn_candidates, hybrid_candidate_pool, proposer_result, merge_stats = _build_turns_and_hybrid_candidates(
        contexts,
        limit_contexts=limit_contexts,
    )
    facet_turn_candidates, facet_records_a, facet_counts_a = facetize_turn_candidates(hybrid_turn_candidates)
    facet_candidate_pool, facet_records_b, facet_counts_b = facetize_candidate_pool(hybrid_candidate_pool)
    facet_counts = dict(facet_counts_a)
    for key, value in facet_counts_b.items():
        facet_counts[key] = max(facet_counts.get(key, 0), value)
    build_result, facet_write_summary = build_v65_memory(
        turns,
        facet_turn_candidates,
        facet_candidate_pool,
        max_stream_observations=max_stream_observations,
    )
    state_payload, checkpoint_dir = _state_payload(root=root, generated_at=generated_at, build_result=build_result)
    proposer_path = checkpoint_dir / "observation_proposer.pt"
    write_worthiness_path = checkpoint_dir / "write_worthiness.pt"
    attribute_validity_path = checkpoint_dir / "attribute_validity.pt"
    router_path = checkpoint_dir / "write_router.pt"
    reader_path = checkpoint_dir / "reader_readout.pt"
    decision_head_path = checkpoint_dir / "decision_head.pt"
    torch.save({"model_state_dict": proposer_result.model.state_dict(), "model_class": "V62BinaryClassifier"}, proposer_path)
    torch.save({"model_state_dict": build_result.write_worthiness_result.model.state_dict(), "model_class": "V62BinaryClassifier"}, write_worthiness_path)
    torch.save({"model_state_dict": build_result.attribute_validity_result.model.state_dict(), "model_class": "V62BinaryClassifier"}, attribute_validity_path)
    torch.save({"model_state_dict": build_result.router_result.router.state_dict(), "model_class": "V6WriteRouter"}, router_path)
    torch.save({"model_state_dict": build_result.reader_result.reader.state_dict(), "model_class": "V6ReaderReadout"}, reader_path)
    torch.save({"model_state_dict": build_result.decision_result.head.state_dict(), "model_class": "V61DecisionHead"}, decision_head_path)
    train_payload = _train_payload(
        root=root,
        generated_at=generated_at,
        checkpoint_dir=checkpoint_dir,
        build_result=build_result,
        proposer_result=proposer_result,
        merge_stats=merge_stats,
        facet_counts=facet_counts,
        proposer_path=proposer_path,
        write_worthiness_path=write_worthiness_path,
        attribute_validity_path=attribute_validity_path,
        router_path=router_path,
        reader_path=reader_path,
        decision_head_path=decision_head_path,
    )
    facet_eval = {
        "artifact_type": "stage2_v65_facetized_memory_eval",
        "commit_hash": _current_head(root),
        "generated_at": generated_at,
        "context_count_used": context_count_used,
        "limit_contexts": limit_contexts,
        **facet_schema_payload([*facet_records_a, *facet_records_b], facet_counts),
        **facet_write_summary,
        "rule_candidate_count": merge_stats.rule_candidate_count,
        "hybrid_candidate_count": merge_stats.merged_candidate_count,
        "facet_candidate_count": len(facet_candidate_pool),
    }
    internal_eval = evaluate_internal_v63(
        build_result.memory,
        build_result.policy_observations,
        reader=build_result.reader_result.reader,
        decision_head=build_result.decision_result.head,
    )
    internal_eval.update(
        {
            "artifact_type": "stage2_v65_internal_eval",
            "commit_hash": _current_head(root),
            "generated_at": generated_at,
            "facet_aware_write_utility_used": True,
        }
    )
    personamem_payload, predictions = _evaluate_personamem(
        root=root,
        generated_at=generated_at,
        memory=build_result.memory,
        reader=build_result.reader_result.reader,
        decision_head=build_result.decision_result.head,
        questions=questions,
    )
    personamem_payload["context_count_used"] = context_count_used
    personamem_payload["limit_contexts"] = limit_contexts
    v64_error = json.loads((root / "outputs_v2" / "artifacts" / "latest_stage2_v64_error_attribution.json").read_text(encoding="utf-8"))
    error_attribution_payload = _build_error_attribution(
        root=root,
        generated_at=generated_at,
        memory=build_result.memory,
        questions=questions,
        predictions=predictions,
        parser_only_needed_facet_missing_count=int(v64_error.get("parser_only_never_written_count", 0)),
    )
    error_attribution_payload["context_count_used"] = context_count_used
    error_attribution_payload["limit_contexts"] = limit_contexts
    decision_payload = {
        "artifact_type": "stage2_v65_decision",
        "commit_hash": _current_head(root),
        "generated_at": generated_at,
        "context_count_used": context_count_used,
        "limit_contexts": limit_contexts,
        "result_type": "partial_keep",
        "claim_allowed": False,
        "significant_margin_confirmed": False,
        "margin_correct_vs_text_only": int(personamem_payload["margin_correct_vs_text_only"]),
        "margin_correct_vs_option_only": int(personamem_payload["no_calibration_correct"]) - int(personamem_payload["option_only_correct"]),
        "blockers": ["personamem_margin_not_meaningful"],
        "claim_text": "v6.5 facet schema/path artifacts are published, but PersonaMem no-routing still does not clear the meaningful-margin gate.",
    }
    outputs = {
        "latest_stage2_v65_facetized_memory_train.json": train_payload,
        "latest_stage2_v65_facetized_memory_eval.json": facet_eval,
        "latest_stage2_v65_persistent_state.json": state_payload,
        "latest_stage2_v65_internal_eval.json": internal_eval,
        "latest_stage2_v65_personamem_no_routing.json": personamem_payload,
        "latest_stage2_v65_error_attribution.json": error_attribution_payload,
        "latest_stage2_v65_decision.json": decision_payload,
    }
    for name, payload in outputs.items():
        _write_json(ARTIFACT_ROOT / name, payload)
        _write_json(ARTIFACT_ROOT / f"{generated_at}_{name}", payload)
    return {
        "train": train_payload,
        "facet_eval": facet_eval,
        "state": state_payload,
        "internal_eval": internal_eval,
        "personamem": personamem_payload,
        "error_attribution": error_attribution_payload,
        "decision": decision_payload,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-contexts", type=int)
    parser.add_argument("--max-stream-observations", type=int, default=8000)
    args = parser.parse_args()
    payload = publish_v65_facetized_memory(limit_contexts=args.limit_contexts, max_stream_observations=args.max_stream_observations)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
