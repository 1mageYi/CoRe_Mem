"""Evaluate v5.1 trained latent memory on PersonaMem full589 without calibration."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.publish_stage2_v51_real_backbone import _prepared_text
from scripts.train_stage2_v51_real_training import V51LatentRetriever


ARTIFACT_ROOT = REPO_ROOT / "outputs_v2" / "artifacts"
DEFAULT_QUESTIONS = REPO_ROOT / "data" / "personamem" / "questions_32k.csv"
DEFAULT_CONTEXTS = REPO_ROOT / "data" / "personamem" / "shared_contexts_32k.jsonl"
DEFAULT_TRAINING = ARTIFACT_ROOT / "latest_stage2_v51_real_training.json"


def _timestamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def _current_head(root: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except Exception:
        return "unknown"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _label(option: str) -> str:
    match = re.match(r"\s*(\([a-d]\))", option)
    return match.group(1) if match else ""


def _clean_message(content: str) -> str:
    return re.sub(r"\s+", " ", content.replace("User:", "").replace("Assistant:", "")).strip()


def _context_chunk_records(messages: list[dict[str, str]], *, end_index: int) -> list[tuple[int, str]]:
    chunks: list[tuple[int, str]] = []
    for idx, message in enumerate(messages[: max(end_index + 1, 0)]):
        role = str(message.get("role") or "unknown")
        content = _clean_message(str(message.get("content") or ""))
        if not content:
            continue
        if role == "assistant" and len(content) > 900:
            content = content[:900]
        if role == "system":
            chunks.extend((idx, part.strip()) for part in re.split(r"\n+|  +", content) if len(part.strip()) > 30)
        else:
            chunks.append((idx, f"{role}: {content[:900]}"))
    return chunks


def _context_chunks(messages: list[dict[str, str]], *, end_index: int) -> list[str]:
    return [text for _, text in _context_chunk_records(messages, end_index=end_index)]


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


def _load_latent(training: dict[str, Any], *, root: Path, device: torch.device) -> V51LatentRetriever:
    checkpoint_path = Path(str(training["checkpoint_path"]))
    if not checkpoint_path.is_absolute():
        checkpoint_path = root / checkpoint_path
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = V51LatentRetriever(
        embedding_dim=int(checkpoint["embedding_dim"]),
        latent_dim=int(checkpoint["latent_dim"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def _project_query(model: V51LatentRetriever, vectors: torch.Tensor, device: torch.device) -> torch.Tensor:
    with torch.no_grad():
        return torch.nn.functional.normalize(model.query_projection(vectors.to(device)), dim=-1).cpu()


def _project_slots(model: V51LatentRetriever, vectors: torch.Tensor, device: torch.device) -> torch.Tensor:
    with torch.no_grad():
        return torch.nn.functional.normalize(model.slot_projection(vectors.to(device)), dim=-1).cpu()


def publish_personamem_no_calibration(
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
    latent_model = _load_latent(training, root=root, device=device)
    questions = _read_questions(questions_path)
    contexts = _read_contexts(contexts_path)

    context_cache: dict[str, dict[str, Any]] = {}
    for context_id, messages in contexts.items():
        records = _context_chunk_records(messages, end_index=len(messages) - 1)
        chunks = [text for _, text in records]
        message_indices = torch.tensor([idx for idx, _ in records], dtype=torch.long)
        if not chunks:
            chunks = [""]
            message_indices = torch.tensor([0], dtype=torch.long)
        embeddings = _encode(encoder, chunks, model_id=model_id, role="passage", batch_size=batch_size)
        context_cache[context_id] = {
            "chunks": chunks,
            "message_indices": message_indices,
            "embeddings": embeddings,
            "latent": _project_slots(latent_model, embeddings, device),
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
        query_latent = _project_query(latent_model, query_embedding, device)[0]
        option_latent = _project_slots(latent_model, option_embeddings, device)
        retrieval_scores = chunk_latent @ query_latent
        top_indices = torch.topk(retrieval_scores, k=min(top_k, len(retrieval_scores))).indices
        top_latent = chunk_latent[top_indices]
        top_text = chunk_embeddings[top_indices]

        option_only_scores = option_latent @ query_latent
        latent_memory_scores = torch.max(option_latent @ top_latent.T, dim=1).values
        text_memory_scores = torch.max(option_embeddings @ top_text.T, dim=1).values
        full_scores = 0.65 * latent_memory_scores + 0.25 * option_only_scores + 0.10 * text_memory_scores
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
        "artifact_type": "stage2_v51_personamem_full589",
        "commit_hash": _current_head(root),
        "generated_at": _timestamp(),
        "sample_count": sample_count,
        "questions_path": str(questions_path.relative_to(root) if questions_path.is_relative_to(root) else questions_path),
        "contexts_path": str(contexts_path.relative_to(root) if contexts_path.is_relative_to(root) else contexts_path),
        "training_artifact": str(training_path.relative_to(root) if training_path.is_relative_to(root) else training_path),
        "checkpoint_path": training.get("checkpoint_path"),
        "model_id": model_id,
        "evaluation_backend": "sentence_transformers_plus_trained_latent_projection",
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
    decision: dict[str, Any]
    if no_cal_accuracy > max(option_only_accuracy, 0.25):
        decision = {
            "artifact_type": "stage2_v51_real_training_decision",
            "commit_hash": _current_head(root),
            "generated_at": payload["generated_at"],
            "result_type": "positive_gain",
            "claim_allowed": True,
            "claim_text": "Real pretrained v5.1 training produced held-out stage2 gains and PersonaMem full589 no-calibration accuracy above option-only/random baselines.",
            "blockers": [],
            "next_recommendation": "Repeat with broader backbone comparison and stronger PersonaMem memory construction before making broader benchmark superiority claims.",
        }
    else:
        decision = {
            "artifact_type": "stage2_v51_real_training_decision",
            "commit_hash": _current_head(root),
            "generated_at": payload["generated_at"],
            "result_type": "negative_result",
            "claim_allowed": False,
            "claim_text": "Real pretrained v5.1 training did not produce PersonaMem no-calibration accuracy above option-only/random baselines.",
            "blockers": ["personamem_no_calibration_not_above_baselines"],
            "next_recommendation": "Treat the current architecture/objective as insufficient for PersonaMem no-calibration and redesign memory construction or answer projection.",
        }

    artifact_root = root / "outputs_v2" / "artifacts"
    prediction_path = artifact_root / "latest_stage2_v51_personamem_full589_predictions.jsonl"
    with prediction_path.open("w", encoding="utf-8") as handle:
        for item in predictions:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    payload["prediction_path"] = str(prediction_path.relative_to(root))
    for name, item in (
        ("latest_stage2_v51_personamem_full589.json", payload),
        ("latest_stage2_v51_real_training_decision.json", decision),
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
    payload, decision = publish_personamem_no_calibration(
        top_k=args.top_k,
        batch_size=args.batch_size,
        device_name=args.device,
    )
    print(json.dumps({"personamem": payload, "decision": decision}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
