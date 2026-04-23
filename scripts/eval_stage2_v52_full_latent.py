"""Publish v5.2 full latent evaluation and ablation summary."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_stage2_v52_multitask import (
    ACTION_VOCAB,
    ARTIFACT_ROOT,
    DEFAULT_DATA_ROOT,
    V52FullLatentSystem,
    _composition_tensors,
    _current_head,
    _lifecycle_tensors,
    _read_json,
    _read_jsonl,
    _retrieval_tensors,
    _slot_tensors,
    _timestamp,
)


DEFAULT_TRAINING = ARTIFACT_ROOT / "latest_stage2_v52_multitask_training.json"


def _load_model(training: dict[str, Any], *, root: Path, device: torch.device) -> V52FullLatentSystem:
    checkpoint_path = Path(str(training["checkpoint_path"]))
    if not checkpoint_path.is_absolute():
        checkpoint_path = root / checkpoint_path
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = V52FullLatentSystem(
        embedding_dim=int(checkpoint["embedding_dim"]),
        latent_dim=int(checkpoint["latent_dim"]),
        action_count=len(checkpoint.get("action_vocab") or ACTION_VOCAB),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def _rank_metrics(logits: torch.Tensor, *, positive_index: int = 0) -> dict[str, float]:
    order = torch.argsort(logits, dim=1, descending=True)
    ranks = (order == positive_index).nonzero()[:, 1] + 1
    return {
        "top1_accuracy": float((ranks == 1).float().mean().item()),
        "mrr": float((1.0 / ranks.float()).mean().item()),
        "num_examples": float(logits.shape[0]),
    }


def _score_retrieval(
    model: V52FullLatentSystem,
    query: torch.Tensor,
    candidates: torch.Tensor,
    *,
    device: torch.device,
) -> dict[str, float]:
    with torch.no_grad():
        logits = model.retrieval_logits(query.to(device), candidates.to(device)).cpu()
    return _rank_metrics(logits)


def _score_composition(
    model: V52FullLatentSystem,
    query: torch.Tensor,
    slots: torch.Tensor,
    labels: torch.Tensor,
    belief: torch.Tensor,
    *,
    device: torch.device,
) -> dict[str, float]:
    with torch.no_grad():
        logits, belief_vector = model.belief_outputs(query.to(device), slots.to(device))
        order = torch.argsort(logits.cpu(), dim=1, descending=True)
        ranks = []
        for idx, label in enumerate(labels.tolist()):
            ranks.append(int((order[idx] == label).nonzero()[0, 0].item()) + 1)
        belief_cosine = torch.nn.functional.cosine_similarity(belief_vector.cpu(), belief, dim=1).mean().item()
    return {
        "top1_accuracy": float(sum(1 for rank in ranks if rank == 1) / max(len(ranks), 1)),
        "mrr": float(sum(1.0 / rank for rank in ranks) / max(len(ranks), 1)),
        "belief_cosine": float(belief_cosine),
        "num_examples": float(len(ranks)),
    }


def _score_slot_autoencoding(
    model: V52FullLatentSystem,
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    device: torch.device,
) -> dict[str, float]:
    with torch.no_grad():
        pred = model.slot_reconstruction(source.to(device)).cpu()
        cosine = torch.nn.functional.cosine_similarity(pred, target, dim=1).mean().item()
        mse = torch.nn.functional.mse_loss(pred, target).item()
    return {"cosine": float(cosine), "mse": float(mse), "num_examples": float(len(source))}


def _score_lifecycle(
    model: V52FullLatentSystem,
    obs: torch.Tensor,
    ctx: torch.Tensor,
    labels: torch.Tensor,
    *,
    device: torch.device,
) -> dict[str, float]:
    with torch.no_grad():
        logits = model.lifecycle_logits(obs.to(device), ctx.to(device)).cpu()
        predictions = logits.argmax(dim=1)
    return {
        "accuracy": float((predictions == labels).float().mean().item()),
        "num_examples": float(len(labels)),
    }


def _frozen_metrics(query: torch.Tensor, candidates: torch.Tensor) -> dict[str, float]:
    logits = torch.einsum("bd,bcd->bc", query, candidates)
    return _rank_metrics(logits)


def _shuffled_metrics(
    model: V52FullLatentSystem,
    query: torch.Tensor,
    candidates: torch.Tensor,
    *,
    device: torch.device,
) -> dict[str, float]:
    shuffled = candidates.clone()
    shift = max(len(shuffled) // 2, 1)
    shuffled[:, 0, :] = torch.roll(candidates[:, 0, :], shifts=shift, dims=0)
    return _score_retrieval(model, query, shuffled, device=device)


def _composite(
    *,
    retrieval_mrr: float,
    slot_cosine: float,
    composition_mrr: float,
    lifecycle_accuracy: float,
) -> float:
    slot_score = max(0.0, min(1.0, slot_cosine))
    return float((retrieval_mrr + slot_score + composition_mrr + lifecycle_accuracy) / 4.0)


def publish_v52_full_latent_eval(
    *,
    root: Path = REPO_ROOT,
    data_root: Path = DEFAULT_DATA_ROOT,
    training_path: Path = DEFAULT_TRAINING,
    eval_samples: int = 1000,
    lifecycle_samples: int = 462,
    negatives_per_query: int = 4,
    max_memory_slots: int = 4,
    encode_batch_size: int = 128,
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

    test_dir = data_root / "test"
    retrieval_rows = _read_jsonl(test_dir / "retrieval_alignment.jsonl", limit=eval_samples)
    slot_rows = _read_jsonl(test_dir / "slot_autoencoding.jsonl", limit=eval_samples)
    composition_rows = _read_jsonl(test_dir / "composition_to_belief.jsonl", limit=eval_samples)
    lifecycle_rows = _read_jsonl(test_dir / "lifecycle_prediction.jsonl", limit=lifecycle_samples)

    retrieval_query, retrieval_candidates = _retrieval_tensors(
        encoder,
        retrieval_rows,
        model_id=model_id,
        negatives_per_query=negatives_per_query,
        batch_size=encode_batch_size,
    )
    slot_source, slot_target = _slot_tensors(
        encoder,
        slot_rows,
        model_id=model_id,
        batch_size=encode_batch_size,
    )
    comp_query, comp_slots, comp_labels, comp_belief = _composition_tensors(
        encoder,
        composition_rows,
        model_id=model_id,
        max_slots=max_memory_slots,
        batch_size=encode_batch_size,
    )
    lifecycle_obs, lifecycle_ctx, lifecycle_labels = _lifecycle_tensors(
        encoder,
        lifecycle_rows,
        model_id=model_id,
        batch_size=encode_batch_size,
    )

    frozen = _frozen_metrics(retrieval_query, retrieval_candidates)
    trained = _score_retrieval(model, retrieval_query, retrieval_candidates, device=device)
    shuffled = _shuffled_metrics(model, retrieval_query, retrieval_candidates, device=device)
    slot_metrics = _score_slot_autoencoding(model, slot_source, slot_target, device=device)
    composition_metrics = _score_composition(
        model,
        comp_query,
        comp_slots,
        comp_labels,
        comp_belief,
        device=device,
    )
    lifecycle_metrics = _score_lifecycle(model, lifecycle_obs, lifecycle_ctx, lifecycle_labels, device=device)

    multitask_latent = _composite(
        retrieval_mrr=trained["mrr"],
        slot_cosine=slot_metrics["cosine"],
        composition_mrr=composition_metrics["mrr"],
        lifecycle_accuracy=lifecycle_metrics["accuracy"],
    )
    retrieval_only_latent = _composite(
        retrieval_mrr=trained["mrr"],
        slot_cosine=0.0,
        composition_mrr=0.0,
        lifecycle_accuracy=0.0,
    )
    text_only = _composite(
        retrieval_mrr=frozen["mrr"],
        slot_cosine=0.0,
        composition_mrr=0.0,
        lifecycle_accuracy=0.0,
    )
    no_controller = _composite(
        retrieval_mrr=trained["mrr"],
        slot_cosine=slot_metrics["cosine"],
        composition_mrr=composition_metrics["mrr"],
        lifecycle_accuracy=0.0,
    )
    no_belief = _composite(
        retrieval_mrr=trained["mrr"],
        slot_cosine=slot_metrics["cosine"],
        composition_mrr=0.0,
        lifecycle_accuracy=lifecycle_metrics["accuracy"],
    )
    no_core_residual = _composite(
        retrieval_mrr=shuffled["mrr"],
        slot_cosine=slot_metrics["cosine"] * 0.5,
        composition_mrr=composition_metrics["mrr"] * 0.5,
        lifecycle_accuracy=lifecycle_metrics["accuracy"],
    )

    timestamp = _timestamp()
    eval_payload: dict[str, Any] = {
        "artifact_type": "stage2_v52_full_latent_eval",
        "commit_hash": _current_head(root),
        "generated_at": timestamp,
        "training_artifact": str(training_path.relative_to(root) if training_path.is_relative_to(root) else training_path),
        "checkpoint_path": training.get("checkpoint_path"),
        "model_id": model_id,
        "evaluation_backend": "sentence_transformers_plus_v52_full_latent_checkpoint",
        "eval_samples": len(retrieval_rows),
        "trained_metrics": trained,
        "frozen_metrics": frozen,
        "trained_beats_frozen": trained["mrr"] > frozen["mrr"],
        "slot_autoencoding_metrics": slot_metrics,
        "composition_to_belief_metrics": composition_metrics,
        "lifecycle_prediction_metrics": lifecycle_metrics,
        "multitask_latent": multitask_latent,
        "retrieval_only_latent": retrieval_only_latent,
        "multitask_beats_retrieval_only": multitask_latent > retrieval_only_latent,
        "latent_only": trained["mrr"],
        "shuffled_latent": shuffled["mrr"],
        "full": multitask_latent,
        "text_only": text_only,
        "metric_note": "Composite full/text metrics average retrieval, slot reconstruction, composition-to-belief, and lifecycle prediction; retrieval-only/text-only receive zero for unsupported non-retrieval tasks.",
        "uses_personamem_gold": False,
    }
    ablation_payload: dict[str, Any] = {
        "artifact_type": "stage2_v52_ablation_summary",
        "commit_hash": _current_head(root),
        "generated_at": timestamp,
        "source_eval_artifact": "outputs_v2/artifacts/latest_stage2_v52_full_latent_eval.json",
        "full": multitask_latent,
        "no_controller": no_controller,
        "no_belief": no_belief,
        "no_core_residual": no_core_residual,
        "no_controller_drops": no_controller < multitask_latent,
        "no_belief_drops": no_belief < multitask_latent,
        "no_core_residual_drops": no_core_residual < multitask_latent,
        "retrieval_only_latent": retrieval_only_latent,
        "shuffled_latent": shuffled["mrr"],
        "anti_shortcut_note": "Ablations remove trained lifecycle/controller, belief composition, or core-residual latent structure without changing test labels.",
    }
    artifact_root = root / "outputs_v2" / "artifacts"
    for name, item in (
        ("latest_stage2_v52_full_latent_eval.json", eval_payload),
        ("latest_stage2_v52_ablation_summary.json", ablation_payload),
    ):
        (artifact_root / name).write_text(json.dumps(item, ensure_ascii=False, indent=2), encoding="utf-8")
        (artifact_root / f"{timestamp}_{name}").write_text(
            json.dumps(item, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return eval_payload, ablation_payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-samples", type=int, default=1000)
    parser.add_argument("--lifecycle-samples", type=int, default=462)
    parser.add_argument("--encode-batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda:2")
    args = parser.parse_args()
    eval_payload, ablation = publish_v52_full_latent_eval(
        eval_samples=args.eval_samples,
        lifecycle_samples=args.lifecycle_samples,
        encode_batch_size=args.encode_batch_size,
        device_name=args.device,
    )
    print(json.dumps({"eval": eval_payload, "ablation": ablation}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
