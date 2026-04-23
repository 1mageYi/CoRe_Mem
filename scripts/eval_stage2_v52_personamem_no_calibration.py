"""Evaluate v5.2 full latent checkpoint on PersonaMem full589 without calibration."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval_stage2_v51_personamem_no_calibration import (
    DEFAULT_CONTEXTS,
    DEFAULT_QUESTIONS,
    _context_chunk_records,
    _current_head,
    _label,
    _parse_options,
    _read_contexts,
    _read_questions,
    _timestamp,
)
from scripts.publish_stage2_v51_real_backbone import _prepared_text
from scripts.train_stage2_v52_multitask import V52FullLatentSystem


ARTIFACT_ROOT = REPO_ROOT / "outputs_v2" / "artifacts"
DEFAULT_TRAINING = ARTIFACT_ROOT / "latest_stage2_v52_multitask_training.json"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _encode(encoder: Any, texts: list[str], *, model_id: str, role: str, batch_size: int) -> torch.Tensor:
    prepared = [_prepared_text(text, model_id=model_id, role=role) for text in texts]
    vectors = encoder.encode(
        prepared,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return torch.tensor(np.asarray(vectors, dtype=np.float32))


def _load_model(training: dict[str, Any], *, root: Path, device: torch.device) -> V52FullLatentSystem:
    checkpoint_path = Path(str(training["checkpoint_path"]))
    if not checkpoint_path.is_absolute():
        checkpoint_path = root / checkpoint_path
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = V52FullLatentSystem(
        embedding_dim=int(checkpoint["embedding_dim"]),
        latent_dim=int(checkpoint["latent_dim"]),
        action_count=len(checkpoint.get("action_vocab") or []),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def _query_latent(model: V52FullLatentSystem, vectors: torch.Tensor, device: torch.device) -> torch.Tensor:
    with torch.no_grad():
        return torch.nn.functional.normalize(model.query_projection(vectors.to(device)), dim=-1).cpu()


def _slot_latent(model: V52FullLatentSystem, vectors: torch.Tensor, device: torch.device) -> torch.Tensor:
    with torch.no_grad():
        projected = model.slot_projection(vectors.to(device))
        read = model.latent_reader(projected)
        return torch.nn.functional.normalize(read, dim=-1).cpu()


def _belief_vector(
    model: V52FullLatentSystem,
    query_embedding: torch.Tensor,
    memory_embeddings: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    with torch.no_grad():
        _, belief = model.belief_outputs(
            query_embedding.to(device),
            memory_embeddings.to(device),
        )
        return torch.nn.functional.normalize(belief, dim=-1).cpu()[0]


def publish_v52_personamem_no_calibration(
    *,
    root: Path = REPO_ROOT,
    questions_path: Path = DEFAULT_QUESTIONS,
    contexts_path: Path = DEFAULT_CONTEXTS,
    training_path: Path = DEFAULT_TRAINING,
    top_k: int = 8,
    batch_size: int = 128,
    device_name: str = "cuda:2",
) -> tuple[dict[str, Any], dict[str, Any]]:
    from sentence_transformers import SentenceTransformer

    training = _read_json(training_path)
    model_id = str(training["model_id"])
    device = torch.device(device_name if torch.cuda.is_available() or not device_name.startswith("cuda") else "cpu")
    encoder = SentenceTransformer(
        model_id,
        device=str(device),
        cache_folder=os.environ.get("HF_HOME", str(root / "outputs_v2" / "hf_cache")),
    )
    model = _load_model(training, root=root, device=device)
    questions = _read_questions(questions_path)
    contexts = _read_contexts(contexts_path)

    context_cache: dict[str, dict[str, Any]] = {}
    for context_id, messages in contexts.items():
        records = _context_chunk_records(messages, end_index=len(messages) - 1)
        chunks = [text for _, text in records] or [""]
        message_indices = torch.tensor([idx for idx, _ in records] or [0], dtype=torch.long)
        embeddings = _encode(encoder, chunks, model_id=model_id, role="passage", batch_size=batch_size)
        context_cache[context_id] = {
            "chunks": chunks,
            "message_indices": message_indices,
            "embeddings": embeddings,
            "latent": _slot_latent(model, embeddings, device),
        }

    predictions: list[dict[str, Any]] = []
    no_cal_correct = 0
    option_only_correct = 0
    text_only_correct = 0
    label_counts = {"(a)": 0, "(b)": 0, "(c)": 0, "(d)": 0}

    for row in questions:
        context_id = str(row["shared_context_id"])
        context_entry = context_cache[context_id]
        end_index = int(row.get("end_index_in_shared_context") or len(context_entry["chunks"]) - 1)
        prefix_mask = context_entry["message_indices"] <= end_index
        if not bool(prefix_mask.any()):
            prefix_mask = torch.ones_like(prefix_mask, dtype=torch.bool)
        chunk_embeddings = context_entry["embeddings"][prefix_mask]
        chunk_latent = context_entry["latent"][prefix_mask]
        raw_chunks = [chunk for chunk, keep in zip(context_entry["chunks"], prefix_mask.tolist()) if keep]

        question = str(row["user_question_or_message"])
        options = _parse_options(row["all_options"])
        labels = [_label(option) for option in options]
        query_embedding = _encode(encoder, [question], model_id=model_id, role="query", batch_size=batch_size)
        option_embeddings = _encode(encoder, options, model_id=model_id, role="passage", batch_size=batch_size)
        query_latent = _query_latent(model, query_embedding, device)[0]
        option_latent = _slot_latent(model, option_embeddings, device)

        retrieval_scores = chunk_latent @ query_latent
        top_indices = torch.topk(retrieval_scores, k=min(top_k, len(retrieval_scores))).indices
        top_latent = chunk_latent[top_indices]
        top_embeddings = chunk_embeddings[top_indices].unsqueeze(0)
        belief_vector = _belief_vector(model, query_embedding, top_embeddings, device)

        option_only_scores = option_latent @ query_latent
        latent_memory_scores = torch.max(option_latent @ top_latent.T, dim=1).values
        text_memory_scores = torch.max(option_embeddings @ chunk_embeddings[top_indices].T, dim=1).values
        belief_scores = option_embeddings @ belief_vector
        full_scores = (
            0.50 * latent_memory_scores
            + 0.20 * belief_scores
            + 0.20 * option_only_scores
            + 0.10 * text_memory_scores
        )
        no_cal_idx = int(torch.argmax(full_scores).item())
        option_only_idx = int(torch.argmax(option_only_scores).item())
        text_only_idx = int(torch.argmax(text_memory_scores).item())
        no_cal_label = labels[no_cal_idx]
        option_only_label = labels[option_only_idx]
        text_only_label = labels[text_only_idx]
        gold = str(row["correct_answer"])
        no_cal_correct += int(no_cal_label == gold)
        option_only_correct += int(option_only_label == gold)
        text_only_correct += int(text_only_label == gold)
        label_counts[no_cal_label] = label_counts.get(no_cal_label, 0) + 1
        predictions.append(
            {
                "question_id": row["question_id"],
                "persona_id": row["persona_id"],
                "shared_context_id": context_id,
                "question_type": row["question_type"],
                "topic": row["topic"],
                "prediction": no_cal_label,
                "option_only_prediction": option_only_label,
                "text_only_prediction": text_only_label,
                "correct_answer": gold,
                "is_correct": no_cal_label == gold,
                "top_memory_indices": [int(idx) for idx in top_indices.tolist()],
                "top_memory_preview": [raw_chunks[int(idx)][:240] for idx in top_indices[:3].tolist()],
            }
        )

    sample_count = len(questions)
    no_cal_accuracy = no_cal_correct / sample_count if sample_count else 0.0
    option_only_accuracy = option_only_correct / sample_count if sample_count else 0.0
    text_only_accuracy = text_only_correct / sample_count if sample_count else 0.0
    payload: dict[str, Any] = {
        "artifact_type": "stage2_v52_personamem_full589",
        "commit_hash": _current_head(root),
        "generated_at": _timestamp(),
        "sample_count": sample_count,
        "questions_path": str(questions_path.relative_to(root) if questions_path.is_relative_to(root) else questions_path),
        "contexts_path": str(contexts_path.relative_to(root) if contexts_path.is_relative_to(root) else contexts_path),
        "training_artifact": str(training_path.relative_to(root) if training_path.is_relative_to(root) else training_path),
        "checkpoint_path": training.get("checkpoint_path"),
        "model_id": model_id,
        "evaluation_backend": "sentence_transformers_plus_v52_full_latent_checkpoint",
        "no_calibration_accuracy": no_cal_accuracy,
        "no_calibration_correct": no_cal_correct,
        "calibrated_accuracy": None,
        "option_only_accuracy": option_only_accuracy,
        "option_only_correct": option_only_correct,
        "text_only_accuracy": text_only_accuracy,
        "text_only_correct": text_only_correct,
        "random_baseline": 0.25,
        "gold_used_for_memory_substrate": False,
        "gold_used_for_no_calibration_prediction": False,
        "provider_is_auxiliary": True,
        "answer_head_trained": False,
        "label_counts": label_counts,
        "top_k": top_k,
        "claim_boundary": "No-calibration prediction uses PersonaMem context/query/options at eval time only; correct_answer is used only for scoring.",
    }
    beats_text = no_cal_correct > max(text_only_correct, 214)
    decision: dict[str, Any] = {
        "artifact_type": "stage2_v52_decision",
        "commit_hash": _current_head(root),
        "generated_at": payload["generated_at"],
        "result_type": "positive_gain" if beats_text else "negative_result",
        "claim_allowed": beats_text,
        "claim_text": (
            "v5.2 no-calibration PersonaMem full589 beats text-only."
            if beats_text
            else "v5.2 no-calibration PersonaMem full589 does not yet beat text-only; continue as partial/negative evidence."
        ),
        "blockers": [] if beats_text else ["personamem_no_calibration_not_above_text_only"],
        "next_recommendation": "Publish latent eval and ablations, then improve PersonaMem memory/projection if text-only remains higher.",
    }

    artifact_root = root / "outputs_v2" / "artifacts"
    prediction_path = artifact_root / "latest_stage2_v52_personamem_full589_predictions.jsonl"
    with prediction_path.open("w", encoding="utf-8") as handle:
        for item in predictions:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    payload["prediction_path"] = str(prediction_path.relative_to(root))
    for name, item in (
        ("latest_stage2_v52_personamem_full589.json", payload),
        ("latest_stage2_v52_decision.json", decision),
    ):
        (artifact_root / name).write_text(json.dumps(item, ensure_ascii=False, indent=2), encoding="utf-8")
        (artifact_root / f"{payload['generated_at']}_{name}").write_text(
            json.dumps(item, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return payload, decision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda:2")
    args = parser.parse_args()
    payload, decision = publish_v52_personamem_no_calibration(
        top_k=args.top_k,
        batch_size=args.batch_size,
        device_name=args.device,
    )
    print(json.dumps({"personamem": payload, "decision": decision}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
