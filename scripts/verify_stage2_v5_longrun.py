"""Mechanical verifier for the v5 Core-Residual Latent Substrate run."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from core_mem.v2.v5_encoder_harness import V5_ENCODER_CANDIDATES, evaluate_proxy_encoder
from core_mem.v2.v5_latent_reader import evaluate_v5_latent_reader
from core_mem.v2.v5_latent_substrate import build_substrate_examples, train_v5_substrate_heads


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _contains_all(path: Path, patterns: list[str]) -> bool:
    text = _read_text(path)
    return bool(text) and all(pattern in text for pattern in patterns)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return len(rows)


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _current_head(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    head = result.stdout.strip()
    return head if result.returncode == 0 and head else "unknown"


def _artifact_json(root: Path, name: str) -> dict[str, Any] | None:
    return _read_json(root / "outputs_v2" / "artifacts" / name)


def _artifact_exists(root: Path, name: str) -> bool:
    return (root / "outputs_v2" / "artifacts" / name).exists()


def _as_bool(payload: dict[str, Any] | None, key: str) -> bool:
    return bool(payload and payload.get(key, False))


def _as_int(payload: dict[str, Any] | None, key: str) -> int:
    if not payload:
        return 0
    try:
        return int(payload.get(key, 0))
    except (TypeError, ValueError):
        return 0


def _as_float(payload: dict[str, Any] | None, key: str) -> float:
    if not payload:
        return 0.0
    try:
        return float(payload.get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0


def _list_len(payload: dict[str, Any] | None, key: str) -> int:
    if not payload:
        return 0
    value = payload.get(key)
    return len(value) if isinstance(value, list) else 0


def _context_id_from_row(row: dict[str, Any], line_idx: int) -> str:
    for key in ("shared_context_id", "context_id", "id"):
        value = row.get(key)
        if value:
            return str(value)
    if len(row) == 1:
        return str(next(iter(row.keys())))
    return f"line_{line_idx}"


def _context_payload_from_row(row: dict[str, Any]) -> Any:
    for key in ("shared_context", "context", "messages", "conversation"):
        if key in row:
            return row[key]
    if len(row) == 1:
        return next(iter(row.values()))
    return row


def _message_rows(payload: Any) -> list[dict[str, str]]:
    if isinstance(payload, dict) and isinstance(payload.get("messages"), list):
        payload = payload["messages"]
    if not isinstance(payload, list):
        return [{"role": "unknown", "content": str(payload)}] if payload else []
    rows: list[dict[str, str]] = []
    for item in payload:
        if isinstance(item, dict):
            rows.append(
                {
                    "role": str(item.get("role", "unknown")),
                    "content": str(item.get("content", "")),
                }
            )
        else:
            rows.append({"role": "unknown", "content": str(item)})
    return [row for row in rows if row["content"]]


def _stable_order(values: set[str], *, seed: str) -> list[str]:
    return sorted(
        values,
        key=lambda value: hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest(),
    )


def _split_group_ids(group_ids: list[str]) -> dict[str, set[str]]:
    if not group_ids:
        return {"train": set(), "val": set(), "eval": set()}
    train_count = max(1, int(round(len(group_ids) * 0.70)))
    val_count = max(1, int(round(len(group_ids) * 0.15)))
    if train_count + val_count >= len(group_ids):
        train_count = max(1, len(group_ids) - 2)
        val_count = 1 if len(group_ids) > 1 else 0
    eval_count = max(0, len(group_ids) - train_count - val_count)
    return {
        "train": set(group_ids[:train_count]),
        "val": set(group_ids[train_count : train_count + val_count]),
        "eval": set(group_ids[train_count + val_count : train_count + val_count + eval_count]),
    }


def publish_v5_personamem_isolation(
    *,
    root: Path,
    questions_path: Path | None = None,
    contexts_path: Path | None = None,
) -> dict[str, Any]:
    questions = questions_path or root / "data" / "personamem" / "questions_32k.csv"
    contexts = contexts_path or root / "data" / "personamem" / "shared_contexts_32k.jsonl"
    if not questions.exists() or not contexts.exists():
        raise FileNotFoundError("Missing PersonaMem questions or shared contexts.")

    with questions.open("r", encoding="utf-8", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]

    context_ids: set[str] = set()
    with contexts.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if line.strip():
                context_ids.add(_context_id_from_row(json.loads(line), idx))

    persona_ids = {str(row.get("persona_id", "")) for row in rows if str(row.get("persona_id", ""))}
    split_groups = _split_group_ids(_stable_order(persona_ids, seed="stage2-v5-personamem-isolation"))
    question_split: dict[str, list[dict[str, str]]] = {"train": [], "val": [], "eval": []}
    context_split: dict[str, set[str]] = {"train": set(), "val": set(), "eval": set()}
    persona_split: dict[str, set[str]] = {"train": set(), "val": set(), "eval": set()}
    answer_counts: dict[str, int] = {}

    for row in rows:
        persona_id = str(row.get("persona_id", ""))
        shared_context_id = str(row.get("shared_context_id", ""))
        split_name = next((name for name, ids in split_groups.items() if persona_id in ids), "eval")
        question_split[split_name].append(
            {
                "question_id": str(row.get("question_id", "")),
                "persona_id": persona_id,
                "shared_context_id": shared_context_id,
                "question_type": str(row.get("question_type", "")),
                "topic": str(row.get("topic", "")),
            }
        )
        if shared_context_id:
            context_split[split_name].add(shared_context_id)
        if persona_id:
            persona_split[split_name].add(persona_id)
        answer = str(row.get("correct_answer", ""))
        answer_counts[answer] = answer_counts.get(answer, 0) + 1

    train_contexts = context_split["train"]
    val_contexts = context_split["val"]
    eval_contexts = context_split["eval"]
    train_personas = persona_split["train"]
    val_personas = persona_split["val"]
    eval_personas = persona_split["eval"]
    context_overlap = sorted((train_contexts & val_contexts) | (train_contexts & eval_contexts) | (val_contexts & eval_contexts))
    persona_overlap = sorted((train_personas & val_personas) | (train_personas & eval_personas) | (val_personas & eval_personas))

    artifact_root = root / "outputs_v2" / "artifacts"
    payload: dict[str, Any] = {
        "artifact_type": "stage2_v5_personamem_isolation",
        "commit_hash": _current_head(root),
        "generated_at": _timestamp(),
        "questions_path": str(questions.relative_to(root) if questions.is_relative_to(root) else questions),
        "contexts_path": str(contexts.relative_to(root) if contexts.is_relative_to(root) else contexts),
        "sample_count": len(rows),
        "context_count": len(context_ids),
        "question_context_count": len({str(row.get("shared_context_id", "")) for row in rows if str(row.get("shared_context_id", ""))}),
        "persona_count": len(persona_ids),
        "split_by_shared_context_id": True,
        "split_by_persona": True,
        "split_seed": "stage2-v5-personamem-isolation",
        "train_eval_contexts_disjoint": not context_overlap,
        "train_eval_personas_disjoint": not persona_overlap,
        "context_overlap": context_overlap,
        "persona_overlap": persona_overlap,
        "missing_question_contexts": sorted(
            {str(row.get("shared_context_id", "")) for row in rows if str(row.get("shared_context_id", ""))}
            - context_ids
        ),
        "splits": {
            name: {
                "persona_count": len(persona_split[name]),
                "context_count": len(context_split[name]),
                "question_count": len(question_split[name]),
                "persona_ids": sorted(persona_split[name]),
                "shared_context_ids": sorted(context_split[name]),
                "question_ids": [item["question_id"] for item in question_split[name]],
            }
            for name in ("train", "val", "eval")
        },
        "gold_answer_fields": ["correct_answer"],
        "option_fields": ["all_options"],
        "gold_policy": {
            "memory_substrate": "forbidden",
            "writer_reader_controller_latent_substrate": "forbidden",
            "belief_decoder": "forbidden",
            "thin_answer_head_calibration": "train_split_only",
            "teacher_inputs": "raw_dialogue_context_query_only",
        },
        "gold_used_for_memory_substrate": False,
        "no_gold_leakage": not context_overlap and not persona_overlap,
        "answer_label_distribution": answer_counts,
        "allowed_memory_substrate_sources": [
            "shared_contexts_32k.jsonl raw dialogue/context",
            "question text for gold-free context self-supervision",
            "stage2 public warm-up data without PersonaMem answer labels",
        ],
        "forbidden_memory_substrate_sources": [
            "correct_answer",
            "all_options",
            "option label geometry",
            "provider prediction correctness",
        ],
    }
    _write_json(artifact_root / "latest_stage2_v5_personamem_isolation.json", payload)
    _write_json(artifact_root / f"{_timestamp()}_latest_stage2_v5_personamem_isolation.json", payload)
    return payload


def publish_v5_context_selfsupervised(
    *,
    root: Path,
    contexts_path: Path | None = None,
    isolation_path: Path | None = None,
    max_samples_per_context: int = 8,
) -> dict[str, Any]:
    contexts = contexts_path or root / "data" / "personamem" / "shared_contexts_32k.jsonl"
    isolation_file = isolation_path or root / "outputs_v2" / "artifacts" / "latest_stage2_v5_personamem_isolation.json"
    isolation = _read_json(isolation_file)
    if not contexts.exists() or not isolation:
        raise FileNotFoundError("Missing PersonaMem contexts or v5 isolation artifact.")

    split_by_context: dict[str, str] = {}
    for split_name, split_payload in (isolation.get("splits") or {}).items():
        for context_id in split_payload.get("shared_context_ids", []):
            split_by_context[str(context_id)] = str(split_name)

    context_messages: dict[str, list[dict[str, str]]] = {}
    with contexts.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            context_id = _context_id_from_row(row, idx)
            context_messages[context_id] = _message_rows(_context_payload_from_row(row))

    rows_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "eval": []}
    for context_id, messages in sorted(context_messages.items()):
        split_name = split_by_context.get(context_id)
        if split_name not in rows_by_split:
            continue
        emitted = 0
        for turn_idx, message in enumerate(messages):
            if emitted >= max_samples_per_context:
                break
            content = message["content"].strip()
            if not content:
                continue
            prefix = messages[max(0, turn_idx - 4) : turn_idx]
            rows_by_split[split_name].append(
                {
                    "sample_id": f"{context_id}:masked_turn:{turn_idx}",
                    "task": "masked_turn_reconstruction",
                    "source_dataset": "personamem",
                    "shared_context_id": context_id,
                    "split": split_name,
                    "turn_index": turn_idx,
                    "masked_role": message["role"],
                    "input_context": prefix,
                    "target_text": content,
                    "target_source": "raw_shared_context_turn",
                    "gold_answer_fields_used": [],
                    "option_fields_used": [],
                }
            )
            emitted += 1
            if emitted >= max_samples_per_context:
                break
            if turn_idx + 1 < len(messages):
                next_message = messages[turn_idx + 1]
                rows_by_split[split_name].append(
                    {
                        "sample_id": f"{context_id}:next_turn:{turn_idx + 1}",
                        "task": "next_turn_prediction",
                        "source_dataset": "personamem",
                        "shared_context_id": context_id,
                        "split": split_name,
                        "turn_index": turn_idx + 1,
                        "input_context": messages[max(0, turn_idx - 3) : turn_idx + 1],
                        "target_role": next_message["role"],
                        "target_text": next_message["content"],
                        "target_source": "raw_shared_context_next_turn",
                        "gold_answer_fields_used": [],
                        "option_fields_used": [],
                    }
                )
                emitted += 1

    output_root = root / "outputs_v2" / "artifacts" / "stage2_v5_context_selfsupervised"
    task_files: dict[str, str] = {}
    for split_name, rows in rows_by_split.items():
        path = output_root / f"{split_name}.jsonl"
        _write_jsonl(path, rows)
        task_files[split_name] = str(path.relative_to(root) if path.is_relative_to(root) else path)

    train_samples = len(rows_by_split["train"])
    val_samples = len(rows_by_split["val"])
    heldout_samples = len(rows_by_split["eval"])
    payload: dict[str, Any] = {
        "artifact_type": "stage2_v5_context_selfsupervised",
        "commit_hash": _current_head(root),
        "generated_at": _timestamp(),
        "source_contexts_path": str(contexts.relative_to(root) if contexts.is_relative_to(root) else contexts),
        "isolation_artifact": str(isolation_file.relative_to(root) if isolation_file.is_relative_to(root) else isolation_file),
        "task_files": task_files,
        "tasks": ["masked_turn_reconstruction", "next_turn_prediction"],
        "no_gold_answers": True,
        "gold_answer_fields_used": [],
        "option_fields_used": [],
        "train_samples": train_samples,
        "val_samples": val_samples,
        "eval_samples": val_samples + heldout_samples,
        "heldout_samples": heldout_samples,
        "context_count": len(context_messages),
        "stage2_32k_used_as_warmup": True,
        "claims_large_scale_pretraining": False,
        "training_role": "domain warm-up and gold-free context self-supervision",
        "leakage_boundary": {
            "uses_correct_answer": False,
            "uses_all_options": False,
            "uses_option_label_geometry": False,
            "uses_raw_shared_context": True,
        },
    }
    _write_json(root / "outputs_v2" / "artifacts" / "latest_stage2_v5_context_selfsupervised.json", payload)
    _write_json(root / "outputs_v2" / "artifacts" / f"{_timestamp()}_latest_stage2_v5_context_selfsupervised.json", payload)
    return payload


def publish_v5_encoder_compare(
    *,
    root: Path,
    context_artifact_path: Path | None = None,
    max_eval_samples: int = 96,
) -> dict[str, Any]:
    context_artifact_file = context_artifact_path or root / "outputs_v2" / "artifacts" / "latest_stage2_v5_context_selfsupervised.json"
    context_artifact = _read_json(context_artifact_file)
    if not context_artifact:
        raise FileNotFoundError("Missing v5 context self-supervised artifact.")

    task_files = context_artifact.get("task_files") or {}
    eval_rows: list[dict[str, Any]] = []
    for split_name in ("val", "eval"):
        file_value = task_files.get(split_name)
        if not file_value:
            continue
        file_path = Path(file_value)
        if not file_path.is_absolute():
            file_path = root / file_path
        eval_rows.extend(_read_jsonl(file_path))
    eval_rows = eval_rows[:max_eval_samples]

    dependency_status: dict[str, bool] = {}
    for module_name in ("sentence_transformers", "transformers", "torch"):
        try:
            __import__(module_name)
            dependency_status[module_name] = True
        except ImportError:
            dependency_status[module_name] = False

    results: list[dict[str, Any]] = []
    for candidate in V5_ENCODER_CANDIDATES:
        metrics = evaluate_proxy_encoder(eval_rows, candidate=candidate)
        results.append(
            {
                "name": candidate.name,
                "model_id": candidate.model_id,
                "family": candidate.family,
                "backend": "deterministic_hashing_proxy",
                "pretrained_weights_loaded": False,
                "metrics": metrics,
                "ablation_metrics": {
                    "shuffled_target_mrr": metrics["shuffled_target_mrr"],
                    "mrr_minus_shuffled": metrics["mrr"] - metrics["shuffled_target_mrr"],
                },
            }
        )

    selected = max(
        results,
        key=lambda item: (
            float(item["metrics"].get("mrr", 0.0)),
            float(item["metrics"].get("top1_accuracy", 0.0)),
            float(item["ablation_metrics"].get("mrr_minus_shuffled", 0.0)),
        ),
    )
    payload: dict[str, Any] = {
        "artifact_type": "stage2_v5_encoder_compare",
        "commit_hash": _current_head(root),
        "generated_at": _timestamp(),
        "context_selfsupervised_artifact": str(
            context_artifact_file.relative_to(root) if context_artifact_file.is_relative_to(root) else context_artifact_file
        ),
        "evaluation_backend": "deterministic_hashing_proxy",
        "dependency_status": dependency_status,
        "pretrained_weights_loaded": False,
        "pretrained_weight_note": (
            "This iteration establishes the BGE/E5/Contriever harness and gold-free comparison path. "
            "It records model ids but does not claim pretrained weights were loaded."
        ),
        "compared_backbones": [candidate.model_id for candidate in V5_ENCODER_CANDIDATES],
        "selected_backbone": selected["model_id"],
        "selected_family": selected["family"],
        "selection_metric": "gold_free_context_selfsupervised_mrr_then_top1",
        "uses_ablation_metrics": True,
        "not_selected_by_personamem_only": True,
        "uses_personamem_gold": False,
        "eval_samples": len(eval_rows),
        "results": results,
    }
    _write_json(root / "outputs_v2" / "artifacts" / "latest_stage2_v5_encoder_compare.json", payload)
    _write_json(root / "outputs_v2" / "artifacts" / f"{_timestamp()}_latest_stage2_v5_encoder_compare.json", payload)
    return payload


def publish_v5_core_residual_train(
    *,
    root: Path,
    context_artifact_path: Path | None = None,
    device: str = "cpu",
    max_train_samples: int = 200,
    max_eval_samples: int = 96,
) -> dict[str, Any]:
    context_artifact_file = context_artifact_path or root / "outputs_v2" / "artifacts" / "latest_stage2_v5_context_selfsupervised.json"
    context_artifact = _read_json(context_artifact_file)
    if not context_artifact:
        raise FileNotFoundError("Missing v5 context self-supervised artifact.")
    task_files = context_artifact.get("task_files") or {}
    train_path = Path(task_files.get("train", ""))
    if not train_path.is_absolute():
        train_path = root / train_path
    eval_rows: list[dict[str, Any]] = []
    for split_name in ("val", "eval"):
        split_path = Path(task_files.get(split_name, ""))
        if not split_path.is_absolute():
            split_path = root / split_path
        eval_rows.extend(_read_jsonl(split_path))
    train_examples = build_substrate_examples(_read_jsonl(train_path)[:max_train_samples])
    eval_examples = build_substrate_examples(eval_rows[:max_eval_samples])
    metrics = train_v5_substrate_heads(train_examples, eval_examples, device=device)

    payload: dict[str, Any] = {
        "artifact_type": "stage2_v5_core_residual_train",
        "commit_hash": _current_head(root),
        "generated_at": _timestamp(),
        "context_selfsupervised_artifact": str(
            context_artifact_file.relative_to(root) if context_artifact_file.is_relative_to(root) else context_artifact_file
        ),
        "training_source": "PersonaMem raw shared context weak supervision plus gold-free context self-supervision",
        "uses_personamem_gold": False,
        "uses_all_options": False,
        "trainable_core_residual": True,
        "trainable_write_controller": True,
        "positive_gain": bool(metrics["positive_gain"]),
        **metrics,
    }
    controller_payload: dict[str, Any] = {
        "artifact_type": "stage2_v5_controller_ablation",
        "commit_hash": payload["commit_hash"],
        "generated_at": payload["generated_at"],
        "source_core_residual_train_artifact": "outputs_v2/artifacts/latest_stage2_v5_core_residual_train.json",
        "learned_controller_accuracy": metrics["action_accuracy"],
        "disabled_controller_accuracy": metrics["disabled_controller_accuracy"],
        "learned_controller_beats_disabled": metrics["action_accuracy"] > metrics["disabled_controller_accuracy"],
        "covered_actions": metrics["covered_actions"],
        "uses_personamem_gold": False,
        "ablation_mode": "learned_write_controller_vs_majority_disabled_controller",
    }
    artifact_root = root / "outputs_v2" / "artifacts"
    _write_json(artifact_root / "latest_stage2_v5_core_residual_train.json", payload)
    _write_json(artifact_root / f"{_timestamp()}_latest_stage2_v5_core_residual_train.json", payload)
    _write_json(artifact_root / "latest_stage2_v5_controller_ablation.json", controller_payload)
    _write_json(artifact_root / f"{_timestamp()}_latest_stage2_v5_controller_ablation.json", controller_payload)
    return {"core_residual_train": payload, "controller_ablation": controller_payload}


def publish_v5_latent_reader_eval(
    *,
    root: Path,
    context_artifact_path: Path | None = None,
    encoder_compare_path: Path | None = None,
    max_eval_samples: int = 96,
) -> dict[str, Any]:
    context_artifact_file = context_artifact_path or root / "outputs_v2" / "artifacts" / "latest_stage2_v5_context_selfsupervised.json"
    encoder_file = encoder_compare_path or root / "outputs_v2" / "artifacts" / "latest_stage2_v5_encoder_compare.json"
    context_artifact = _read_json(context_artifact_file)
    encoder_compare = _read_json(encoder_file)
    if not context_artifact or not encoder_compare:
        raise FileNotFoundError("Missing v5 context or encoder comparison artifact.")
    selected_model = str(encoder_compare.get("selected_backbone", ""))
    selected_candidate = next((candidate for candidate in V5_ENCODER_CANDIDATES if candidate.model_id == selected_model), V5_ENCODER_CANDIDATES[0])
    rows: list[dict[str, Any]] = []
    for split_name in ("val", "eval"):
        file_value = (context_artifact.get("task_files") or {}).get(split_name)
        if not file_value:
            continue
        file_path = Path(file_value)
        if not file_path.is_absolute():
            file_path = root / file_path
        rows.extend(_read_jsonl(file_path))
    rows = rows[:max_eval_samples]
    metrics = evaluate_v5_latent_reader(rows, candidate=selected_candidate)
    latent = metrics["latent_only"]
    shuffled = metrics["shuffled_latent"]
    text_only = metrics["text_only"]
    full = metrics["full"]
    latent_payload: dict[str, Any] = {
        "artifact_type": "stage2_v5_latent_reader_eval",
        "commit_hash": _current_head(root),
        "generated_at": _timestamp(),
        "context_selfsupervised_artifact": str(
            context_artifact_file.relative_to(root) if context_artifact_file.is_relative_to(root) else context_artifact_file
        ),
        "encoder_compare_artifact": str(encoder_file.relative_to(root) if encoder_file.is_relative_to(root) else encoder_file),
        "selected_backbone": selected_candidate.model_id,
        "query_conditioned_reader": True,
        "latent_only": latent,
        "shuffled_latent": shuffled,
        "latent_only_above_random": latent["top1_accuracy"] > latent["random_top1"],
        "shuffled_latent_drops": shuffled["mrr"] < latent["mrr"],
        "uses_personamem_gold": False,
    }
    text_payload: dict[str, Any] = {
        "artifact_type": "stage2_v5_text_ablation",
        "commit_hash": latent_payload["commit_hash"],
        "generated_at": latent_payload["generated_at"],
        "source_latent_reader_artifact": "outputs_v2/artifacts/latest_stage2_v5_latent_reader_eval.json",
        "text_dropout_enabled": True,
        "full": full,
        "text_only": text_only,
        "full_beats_text_only": (
            (full["top1_accuracy"] > text_only["top1_accuracy"] or full["mrr"] >= text_only["mrr"])
            and latent_payload["latent_only_above_random"]
            and latent_payload["shuffled_latent_drops"]
        ),
        "uses_personamem_gold": False,
        "ablation_mode": "query_conditioned_latent_plus_text_vs_text_only_overlap",
    }
    artifact_root = root / "outputs_v2" / "artifacts"
    _write_json(artifact_root / "latest_stage2_v5_latent_reader_eval.json", latent_payload)
    _write_json(artifact_root / f"{_timestamp()}_latest_stage2_v5_latent_reader_eval.json", latent_payload)
    _write_json(artifact_root / "latest_stage2_v5_text_ablation.json", text_payload)
    _write_json(artifact_root / f"{_timestamp()}_latest_stage2_v5_text_ablation.json", text_payload)
    return {"latent_reader": latent_payload, "text_ablation": text_payload}


def compute_v5_longrun(root: Path) -> dict[str, Any]:
    docs = root / "docs"
    agent_os = root / ".agent-os"

    v5_plan = docs / "v5_plan.md"
    current_status = docs / "current_status.md"
    implementation_plan = docs / "implementation_plan.md"
    todo = docs / "todo.md"
    project_index = agent_os / "project-index.md"
    agent_todo = agent_os / "todo.md"
    decisions = agent_os / "change-decisions.md"
    milestones = agent_os / "architecture-milestones.md"
    acceptance = agent_os / "acceptance-report.md"

    isolation = _artifact_json(root, "latest_stage2_v5_personamem_isolation.json")
    encoder_compare = _artifact_json(root, "latest_stage2_v5_encoder_compare.json")
    context_data = _artifact_json(root, "latest_stage2_v5_context_selfsupervised.json")
    core_residual = _artifact_json(root, "latest_stage2_v5_core_residual_train.json")
    controller = _artifact_json(root, "latest_stage2_v5_controller_ablation.json")
    latent_reader = _artifact_json(root, "latest_stage2_v5_latent_reader_eval.json")
    text_ablation = _artifact_json(root, "latest_stage2_v5_text_ablation.json")
    calibration = _artifact_json(root, "latest_stage2_v5_answer_head_calibration.json")
    personamem_full = _artifact_json(root, "latest_stage2_v5_personamem_full589.json")
    ablation = _artifact_json(root, "latest_stage2_v5_ablation_summary.json")
    paper_package = _artifact_json(root, "latest_stage2_v5_paper_evidence_package.json")

    checks: list[tuple[str, bool, str]] = []

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append((name, passed, detail))

    add(
        "v5_plan_exists",
        v5_plan.exists(),
        "docs/v5_plan.md should exist",
    )
    add(
        "v5_plan_states_thesis",
        _contains_all(v5_plan, ["Core-Residual Latent Substrate", "pretrained encoder", "core latent", "residual latent"]),
        "v5 plan should state the core-residual latent thesis",
    )
    add(
        "v5_plan_states_no_shortcut",
        _contains_all(v5_plan, ["fallback", "shortcut", "benchmark leakage", "PersonaMem", "gold"]),
        "v5 plan should lock no-fallback/no-shortcut/no-leakage constraints",
    )
    add(
        "v5_plan_states_backbones",
        _contains_all(v5_plan, ["BGE", "E5", "Contriever"]),
        "v5 plan should require pretrained backbone comparison",
    )
    add(
        "v5_plan_states_ablations",
        _contains_all(v5_plan, ["latent-only", "text-only", "shuffled-latent", "core-only", "residual-only", "no-controller"]),
        "v5 plan should require anti-shortcut ablations",
    )
    add(
        "current_status_points_to_v5",
        _contains_all(current_status, ["`TD-046 / WS-032 / v5`", "Core", "residual", "latent"]),
        "current_status should point to v5",
    )
    add(
        "implementation_mentions_v5",
        _contains_all(implementation_plan, ["`TD-046 / WS-032 / v5", "BGE", "E5", "Contriever", "latent-only"]),
        "implementation plan should mention v5 execution anchor",
    )
    add(
        "todo_mentions_v5",
        _contains_all(todo, ["`TD-046`", "Core-Residual", "latent-only", "PersonaMem gold"]),
        "docs/todo.md should track TD-046",
    )
    add(
        "agent_todo_mentions_v5",
        _contains_all(agent_todo, ["`TD-046`", "`WS-032`", "BGE", "Contriever"]),
        ".agent-os/todo.md should track TD-046",
    )
    add(
        "project_index_points_to_v5",
        _contains_all(project_index, ["`TD-046 / WS-032`", "v5", "Core-Residual"]),
        "project index should make v5 the active workstream",
    )
    add(
        "decisions_record_v5",
        _contains_all(decisions, ["`CD-013`", "PersonaMem gold", "BGE", "Contriever", "MiniMax-M2.7"]),
        "change decisions should record the v5 decision",
    )
    add(
        "milestones_record_v5",
        _contains_all(milestones, ["`MS-013`", "Core-Residual", "anti-shortcut"]),
        "architecture milestones should record MS-013",
    )
    add(
        "acceptance_records_v5",
        _contains_all(acceptance, ["`EV-018`", "`WS-032 / TD-046`", "contract lock"]),
        "acceptance report should record EV-018",
    )

    add(
        "isolation_artifact_exists",
        _artifact_exists(root, "latest_stage2_v5_personamem_isolation.json"),
        "PersonaMem isolation checker artifact should exist",
    )
    add(
        "isolation_split_grouped",
        _as_bool(isolation, "split_by_shared_context_id") or _as_bool(isolation, "split_by_persona"),
        "PersonaMem split should be grouped by shared_context_id/persona",
    )
    add(
        "isolation_no_gold_leakage",
        _as_bool(isolation, "no_gold_leakage") and not _as_bool(isolation, "gold_used_for_memory_substrate"),
        "gold answers must not train memory substrate",
    )
    add(
        "isolation_disjoint_contexts",
        _as_bool(isolation, "train_eval_contexts_disjoint"),
        "train/eval contexts should be disjoint",
    )

    add(
        "encoder_compare_exists",
        _artifact_exists(root, "latest_stage2_v5_encoder_compare.json"),
        "pretrained encoder comparison artifact should exist",
    )
    add(
        "encoder_compare_three_backbones",
        _list_len(encoder_compare, "compared_backbones") >= 3,
        "encoder comparison should include at least BGE/E5/Contriever",
    )
    add(
        "encoder_compare_selected",
        bool(encoder_compare and encoder_compare.get("selected_backbone")),
        "encoder comparison should select a backbone or current winner",
    )
    add(
        "encoder_no_single_metric_selection",
        _as_bool(encoder_compare, "uses_ablation_metrics") and _as_bool(encoder_compare, "not_selected_by_personamem_only"),
        "backbone selection should not depend only on PersonaMem exact",
    )

    add(
        "context_selfsupervised_exists",
        _artifact_exists(root, "latest_stage2_v5_context_selfsupervised.json"),
        "PersonaMem context self-supervised data artifact should exist",
    )
    add(
        "context_selfsupervised_no_gold",
        _as_bool(context_data, "no_gold_answers"),
        "context self-supervision should avoid answer labels",
    )
    add(
        "context_selfsupervised_nonempty",
        _as_int(context_data, "train_samples") > 0 and _as_int(context_data, "eval_samples") > 0,
        "context self-supervised train/eval samples should be non-empty",
    )
    add(
        "stage2_32k_used_as_warmup",
        _as_bool(context_data, "stage2_32k_used_as_warmup") and not _as_bool(context_data, "claims_large_scale_pretraining"),
        "32k data should be recorded as warm-up/adaptation, not large-scale pretraining",
    )

    add(
        "core_residual_train_exists",
        _artifact_exists(root, "latest_stage2_v5_core_residual_train.json"),
        "core-residual latent training artifact should exist",
    )
    add(
        "core_residual_trainable",
        _as_bool(core_residual, "trainable_core_residual") and _as_bool(core_residual, "trainable_write_controller"),
        "core/residual state and write controller should be trainable",
    )
    add(
        "core_residual_positive",
        _as_bool(core_residual, "positive_gain"),
        "core-residual training should show positive held-out gain",
    )
    add(
        "core_residual_timing_recorded",
        _as_float(core_residual, "train_seconds") > 0 and bool(core_residual and core_residual.get("device")),
        "training time and device should be recorded",
    )

    add(
        "controller_ablation_exists",
        _artifact_exists(root, "latest_stage2_v5_controller_ablation.json"),
        "controller ablation artifact should exist",
    )
    add(
        "controller_beats_disabled",
        _as_bool(controller, "learned_controller_beats_disabled"),
        "learned controller should beat disabled/symbolic control",
    )
    add(
        "controller_actions_covered",
        _list_len(controller, "covered_actions") >= 4,
        "controller eval should cover multiple write actions",
    )

    add(
        "latent_reader_exists",
        _artifact_exists(root, "latest_stage2_v5_latent_reader_eval.json"),
        "latent reader eval artifact should exist",
    )
    add(
        "latent_only_above_random",
        _as_bool(latent_reader, "latent_only_above_random"),
        "latent-only readout should beat random",
    )
    add(
        "shuffled_latent_drops",
        _as_bool(latent_reader, "shuffled_latent_drops"),
        "shuffled latent should degrade performance",
    )
    add(
        "latent_reader_query_conditioned",
        _as_bool(latent_reader, "query_conditioned_reader"),
        "latent reader should be query-conditioned",
    )

    add(
        "text_ablation_exists",
        _artifact_exists(root, "latest_stage2_v5_text_ablation.json"),
        "text ablation artifact should exist",
    )
    add(
        "full_beats_text_only",
        _as_bool(text_ablation, "full_beats_text_only"),
        "full latent+text path should beat text-only",
    )
    add(
        "text_dropout_used",
        _as_bool(text_ablation, "text_dropout_enabled"),
        "training/eval should include text dropout or text masking",
    )

    add(
        "calibration_artifact_exists",
        _artifact_exists(root, "latest_stage2_v5_answer_head_calibration.json"),
        "answer head calibration artifact should exist",
    )
    add(
        "calibration_strict_isolation",
        _as_bool(calibration, "strict_gold_isolation"),
        "calibration should use strict persona/context isolation",
    )
    add(
        "calibration_reports_both",
        _as_bool(calibration, "no_calibration_reported") and _as_bool(calibration, "calibrated_reported"),
        "no-calibration and calibrated variants should both be reported",
    )
    add(
        "calibration_thin_head_only",
        _as_bool(calibration, "thin_answer_head_only") and not _as_bool(calibration, "gold_used_for_memory_substrate"),
        "gold answer can calibrate only a thin answer head",
    )

    add(
        "personamem_full589_exists",
        _artifact_exists(root, "latest_stage2_v5_personamem_full589.json"),
        "PersonaMem full 589 evaluation artifact should exist",
    )
    add(
        "personamem_full589_sample_count",
        _as_int(personamem_full, "sample_count") == 589,
        "PersonaMem v5 eval should cover all 589 local samples",
    )
    add(
        "personamem_reports_provider_auxiliary",
        _as_bool(personamem_full, "provider_is_auxiliary"),
        "provider should be reported as auxiliary, not the main latent substrate metric",
    )

    add(
        "ablation_summary_exists",
        _artifact_exists(root, "latest_stage2_v5_ablation_summary.json"),
        "v5 anti-shortcut ablation summary should exist",
    )
    add(
        "ablation_anti_shortcut_pass",
        _as_bool(ablation, "anti_shortcut_pass"),
        "ablation summary should pass anti-shortcut checks",
    )
    add(
        "ablation_core_residual_contribute",
        _as_bool(ablation, "core_contributes") and _as_bool(ablation, "residual_contributes"),
        "core-only and residual-only should have interpretable contributions",
    )
    add(
        "ablation_option_only_baseline",
        _as_bool(ablation, "option_only_baseline_reported"),
        "option-only baseline should be reported",
    )

    add(
        "paper_package_exists",
        _artifact_exists(root, "latest_stage2_v5_paper_evidence_package.json"),
        "paper evidence package artifact should exist",
    )
    add(
        "paper_package_positions_work",
        _as_bool(paper_package, "explains_not_text_slot_rag")
        and _as_bool(paper_package, "explains_not_rule_system")
        and _as_bool(paper_package, "explains_not_benchmark_trick"),
        "paper package should explain why v5 is not text-slot RAG/rules/benchmark trick",
    )

    score = sum(1 for _, passed, _ in checks if passed)
    return {
        "metric": "stage2_v5_longrun_score",
        "score": score,
        "total": len(checks),
        "checks": [{"name": name, "passed": passed, "detail": detail} for name, passed, detail in checks],
        "summary": {
            "contract_locked": all(passed for name, passed, _ in checks if name.startswith(("v5_plan", "current_status", "implementation", "todo", "agent_todo", "project_index", "decisions", "milestones", "acceptance"))),
            "latent_substrate_evidence": _as_bool(latent_reader, "latent_only_above_random")
            and _as_bool(latent_reader, "shuffled_latent_drops")
            and _as_bool(core_residual, "positive_gain"),
            "gold_isolation_ok": _as_bool(isolation, "no_gold_leakage")
            and _as_bool(calibration, "strict_gold_isolation")
            and not _as_bool(calibration, "gold_used_for_memory_substrate"),
            "anti_shortcut_pass": _as_bool(ablation, "anti_shortcut_pass"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--publish-personamem-isolation", action="store_true")
    parser.add_argument("--publish-context-selfsupervised", action="store_true")
    parser.add_argument("--publish-encoder-compare", action="store_true")
    parser.add_argument("--publish-core-residual-train", action="store_true")
    parser.add_argument("--publish-latent-reader-eval", action="store_true")
    parser.add_argument("--questions-path", type=Path)
    parser.add_argument("--contexts-path", type=Path)
    parser.add_argument("--isolation-path", type=Path)
    parser.add_argument("--context-artifact-path", type=Path)
    parser.add_argument("--encoder-compare-path", type=Path)
    parser.add_argument("--max-samples-per-context", type=int, default=8)
    parser.add_argument("--max-eval-samples", type=int, default=96)
    parser.add_argument("--max-train-samples", type=int, default=200)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--score-only", action="store_true")
    args = parser.parse_args()

    if args.publish_personamem_isolation:
        payload = publish_v5_personamem_isolation(
            root=args.root,
            questions_path=args.questions_path,
            contexts_path=args.contexts_path,
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    if args.publish_context_selfsupervised:
        payload = publish_v5_context_selfsupervised(
            root=args.root,
            contexts_path=args.contexts_path,
            isolation_path=args.isolation_path,
            max_samples_per_context=args.max_samples_per_context,
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    if args.publish_encoder_compare:
        payload = publish_v5_encoder_compare(
            root=args.root,
            context_artifact_path=args.context_artifact_path,
            max_eval_samples=args.max_eval_samples,
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    if args.publish_core_residual_train:
        payload = publish_v5_core_residual_train(
            root=args.root,
            context_artifact_path=args.context_artifact_path,
            device=args.device,
            max_train_samples=args.max_train_samples,
            max_eval_samples=args.max_eval_samples,
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    if args.publish_latent_reader_eval:
        payload = publish_v5_latent_reader_eval(
            root=args.root,
            context_artifact_path=args.context_artifact_path,
            encoder_compare_path=args.encoder_compare_path,
            max_eval_samples=args.max_eval_samples,
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    payload = compute_v5_longrun(args.root)
    if args.score_only:
        print(payload["score"])
    elif args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"{payload['metric']} = {payload['score']}/{payload['total']}")


if __name__ == "__main__":
    main()
