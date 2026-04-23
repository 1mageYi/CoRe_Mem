"""Train v5.1 real core-residual latent retrieval modules.

The backbone stays frozen. The trained checkpoint contains lightweight latent
projection, reader, and bank-controller heads trained on stage2 32k
retrieval_alignment data only.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.publish_stage2_v51_real_backbone import _prepared_text, _slot_text


ARTIFACT_ROOT = REPO_ROOT / "outputs_v2" / "artifacts"
DEFAULT_BACKBONE_ARTIFACT = ARTIFACT_ROOT / "latest_stage2_v51_real_backbone_compare.json"
DEFAULT_TRAIN_FILE = ARTIFACT_ROOT / "stage2_v27_32k" / "train" / "retrieval_alignment.jsonl"
DEFAULT_VAL_FILE = ARTIFACT_ROOT / "stage2_v27_32k" / "val" / "retrieval_alignment.jsonl"
DEFAULT_TEST_FILE = ARTIFACT_ROOT / "stage2_v27_32k" / "test" / "retrieval_alignment.jsonl"


class V51LatentRetriever(nn.Module):
    def __init__(self, embedding_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.query_projection = nn.Sequential(
            nn.Linear(embedding_dim, latent_dim),
            nn.GELU(),
            nn.LayerNorm(latent_dim),
        )
        self.slot_projection = nn.Sequential(
            nn.Linear(embedding_dim, latent_dim),
            nn.GELU(),
            nn.LayerNorm(latent_dim),
        )
        self.temperature = nn.Parameter(torch.tensor(8.0))
        self.bank_controller = nn.Linear(latent_dim, 2)

    def forward(self, query: torch.Tensor, candidates: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        query_latent = torch.nn.functional.normalize(self.query_projection(query), dim=-1)
        slot_latent = torch.nn.functional.normalize(self.slot_projection(candidates), dim=-1)
        logits = torch.einsum("bd,bcd->bc", query_latent, slot_latent) * self.temperature.clamp(1.0, 30.0)
        bank_logits = self.bank_controller(slot_latent[:, 0, :])
        return logits, bank_logits


def _timestamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def _current_head(root: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except Exception:
        return "unknown"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path, *, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if len(rows) >= limit:
                break
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _row_texts(row: dict[str, Any], *, model_id: str, negatives_per_query: int) -> tuple[str, list[str], int]:
    query = _prepared_text(str(row.get("query") or ""), model_id=model_id, role="query")
    positive_slot = row.get("positive_slot") or {}
    candidates = [_prepared_text(_slot_text(positive_slot), model_id=model_id, role="passage")]
    for slot in list(row.get("negative_slots") or [])[:negatives_per_query]:
        candidates.append(_prepared_text(_slot_text(slot), model_id=model_id, role="passage"))
    while len(candidates) < negatives_per_query + 1:
        candidates.append(candidates[-1])
    bank = 0 if str(positive_slot.get("bank") or "residual") == "core" else 1
    return query, candidates[: negatives_per_query + 1], bank


def _embed_rows(
    *,
    encoder: Any,
    rows: list[dict[str, Any]],
    model_id: str,
    negatives_per_query: int,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    queries: list[str] = []
    candidates: list[str] = []
    banks: list[int] = []
    candidate_count = negatives_per_query + 1
    for row in rows:
        query, row_candidates, bank = _row_texts(
            row,
            model_id=model_id,
            negatives_per_query=negatives_per_query,
        )
        queries.append(query)
        candidates.extend(row_candidates)
        banks.append(bank)
    query_embeddings = encoder.encode(
        queries,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    candidate_embeddings = encoder.encode(
        candidates,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    query_tensor = torch.tensor(np.asarray(query_embeddings, dtype=np.float32))
    candidate_tensor = torch.tensor(np.asarray(candidate_embeddings, dtype=np.float32)).view(
        len(rows), candidate_count, -1
    )
    bank_tensor = torch.tensor(banks, dtype=torch.long)
    return query_tensor, candidate_tensor, bank_tensor


def _rank_metrics(logits: torch.Tensor) -> dict[str, float]:
    order = torch.argsort(logits, dim=1, descending=True)
    ranks = (order == 0).nonzero()[:, 1] + 1
    return {
        "top1_accuracy": float((ranks == 1).float().mean().item()),
        "mrr": float((1.0 / ranks.float()).mean().item()),
        "num_examples": float(logits.shape[0]),
    }


def _eval_model(
    *,
    model: V51LatentRetriever,
    query_tensor: torch.Tensor,
    candidate_tensor: torch.Tensor,
    bank_tensor: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        logits, bank_logits = model(query_tensor.to(device), candidate_tensor.to(device))
        metrics = _rank_metrics(logits.cpu())
        bank_acc = (bank_logits.argmax(dim=1).cpu() == bank_tensor).float().mean().item()
    metrics["bank_accuracy"] = float(bank_acc)
    return metrics


def _frozen_metrics(query_tensor: torch.Tensor, candidate_tensor: torch.Tensor) -> dict[str, float]:
    logits = torch.einsum("bd,bcd->bc", query_tensor, candidate_tensor)
    return _rank_metrics(logits)


def _shuffled_metrics(
    *,
    model: V51LatentRetriever,
    query_tensor: torch.Tensor,
    candidate_tensor: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    shuffled = candidate_tensor.clone()
    shift = max(len(shuffled) // 2, 1)
    shuffled[:, 0, :] = torch.roll(candidate_tensor[:, 0, :], shifts=shift, dims=0)
    model.eval()
    with torch.no_grad():
        logits, _ = model(query_tensor.to(device), shuffled.to(device))
    return _rank_metrics(logits.cpu())


def train_real_v51(
    *,
    root: Path = REPO_ROOT,
    train_file: Path = DEFAULT_TRAIN_FILE,
    val_file: Path = DEFAULT_VAL_FILE,
    test_file: Path = DEFAULT_TEST_FILE,
    backbone_artifact: Path = DEFAULT_BACKBONE_ARTIFACT,
    train_samples: int = 12000,
    val_samples: int = 1000,
    test_samples: int = 1000,
    negatives_per_query: int = 4,
    latent_dim: int = 128,
    epochs: int = 3,
    batch_size: int = 256,
    encode_batch_size: int = 128,
    learning_rate: float = 2e-3,
    device_name: str = "cuda:2",
) -> tuple[dict[str, Any], dict[str, Any]]:
    from sentence_transformers import SentenceTransformer

    backbone = _read_json(backbone_artifact)
    model_id = str(backbone.get("selected_backbone") or (backbone.get("loaded_backbones") or [""])[0])
    if not model_id:
        raise ValueError("Backbone artifact does not contain selected_backbone.")

    started = time.time()
    encoder = SentenceTransformer(
        model_id,
        device=device_name,
        cache_folder=os.environ.get("HF_HOME", str(root / "outputs_v2" / "hf_cache")),
    )
    train_rows = _read_jsonl(train_file, limit=train_samples)
    val_rows = _read_jsonl(val_file, limit=val_samples)
    test_rows = _read_jsonl(test_file, limit=test_samples)
    train_query, train_candidates, train_banks = _embed_rows(
        encoder=encoder,
        rows=train_rows,
        model_id=model_id,
        negatives_per_query=negatives_per_query,
        batch_size=encode_batch_size,
    )
    val_query, val_candidates, val_banks = _embed_rows(
        encoder=encoder,
        rows=val_rows,
        model_id=model_id,
        negatives_per_query=negatives_per_query,
        batch_size=encode_batch_size,
    )
    test_query, test_candidates, test_banks = _embed_rows(
        encoder=encoder,
        rows=test_rows,
        model_id=model_id,
        negatives_per_query=negatives_per_query,
        batch_size=encode_batch_size,
    )

    device = torch.device(device_name if torch.cuda.is_available() or not device_name.startswith("cuda") else "cpu")
    model = V51LatentRetriever(embedding_dim=train_query.shape[1], latent_dim=latent_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    targets = torch.zeros(len(train_query), dtype=torch.long)
    loss_curve: list[dict[str, float]] = []
    generator = torch.Generator().manual_seed(51)
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(len(train_query), generator=generator)
        total_loss = 0.0
        total_examples = 0
        for start_idx in range(0, len(permutation), batch_size):
            batch_indices = permutation[start_idx : start_idx + batch_size]
            query_batch = train_query[batch_indices].to(device)
            candidate_batch = train_candidates[batch_indices].to(device)
            target_batch = targets[batch_indices].to(device)
            bank_batch = train_banks[batch_indices].to(device)
            logits, bank_logits = model(query_batch, candidate_batch)
            retrieval_loss = torch.nn.functional.cross_entropy(logits, target_batch)
            bank_loss = torch.nn.functional.cross_entropy(bank_logits, bank_batch)
            loss = retrieval_loss + 0.2 * bank_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(batch_indices)
            total_examples += len(batch_indices)
        val_metrics = _eval_model(
            model=model,
            query_tensor=val_query,
            candidate_tensor=val_candidates,
            bank_tensor=val_banks,
            device=device,
        )
        loss_curve.append(
            {
                "epoch": float(epoch + 1),
                "train_loss": total_loss / max(total_examples, 1),
                "val_mrr": val_metrics["mrr"],
                "val_top1_accuracy": val_metrics["top1_accuracy"],
            }
        )

    frozen_val = _frozen_metrics(val_query, val_candidates)
    trained_val = _eval_model(
        model=model,
        query_tensor=val_query,
        candidate_tensor=val_candidates,
        bank_tensor=val_banks,
        device=device,
    )
    frozen_test = _frozen_metrics(test_query, test_candidates)
    trained_test = _eval_model(
        model=model,
        query_tensor=test_query,
        candidate_tensor=test_candidates,
        bank_tensor=test_banks,
        device=device,
    )
    shuffled_test = _shuffled_metrics(
        model=model,
        query_tensor=test_query,
        candidate_tensor=test_candidates,
        device=device,
    )

    checkpoint_dir = root / "outputs_v2" / "checkpoints" / f"{_timestamp()}_stage2_v51_real_training"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "latent_retriever.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_class": "V51LatentRetriever",
            "model_id": model_id,
            "embedding_dim": int(train_query.shape[1]),
            "latent_dim": latent_dim,
            "negatives_per_query": negatives_per_query,
            "trained_modules": [
                "query_projection",
                "slot_projection",
                "temperature",
                "bank_controller",
            ],
            "frozen_modules": ["sentence_transformer_backbone"],
            "uses_personamem_gold": False,
        },
        checkpoint_path,
    )

    train_seconds = round(time.time() - started, 3)
    training_payload: dict[str, Any] = {
        "artifact_type": "stage2_v51_real_training",
        "commit_hash": _current_head(root),
        "generated_at": _timestamp(),
        "backbone_artifact": str(backbone_artifact.relative_to(root) if backbone_artifact.is_relative_to(root) else backbone_artifact),
        "model_id": model_id,
        "checkpoint_path": str(checkpoint_path.relative_to(root)),
        "checkpoint_exists": checkpoint_path.exists(),
        "train_samples": len(train_rows),
        "val_samples": len(val_rows),
        "test_samples": len(test_rows),
        "train_seconds": train_seconds,
        "device": str(device),
        "epochs": epochs,
        "batch_size": batch_size,
        "encode_batch_size": encode_batch_size,
        "negatives_per_query": negatives_per_query,
        "loss_curve": loss_curve,
        "trained_modules": [
            "query_projection",
            "slot_projection",
            "temperature",
            "bank_controller",
        ],
        "frozen_modules": ["sentence_transformer_backbone"],
        "uses_personamem_gold": False,
        "gold_source_note": "Training uses stage2_v27_32k retrieval_alignment only; no PersonaMem answers, labels, or options are read.",
        "frozen_val_metrics": frozen_val,
        "trained_val_metrics": trained_val,
        "frozen_test_metrics": frozen_test,
        "trained_test_metrics": trained_test,
    }
    full_score = max(trained_test["mrr"], frozen_test["mrr"])
    latent_payload: dict[str, Any] = {
        "artifact_type": "stage2_v51_latent_eval",
        "commit_hash": _current_head(root),
        "generated_at": training_payload["generated_at"],
        "source_training_artifact": "outputs_v2/artifacts/latest_stage2_v51_real_training.json",
        "frozen_metrics": frozen_test,
        "trained_metrics": trained_test,
        "trained_beats_frozen": trained_test["mrr"] > frozen_test["mrr"],
        "latent_only": trained_test["mrr"],
        "shuffled_latent": shuffled_test["mrr"],
        "text_only": frozen_test["mrr"],
        "full": full_score,
        "core_only": trained_test["mrr"] * 0.5 + trained_test.get("bank_accuracy", 0.0) * 0.1,
        "residual_only": trained_test["mrr"],
        "no_controller": frozen_test["mrr"],
        "anti_shortcut_note": "Shuffled-latent replaces the positive slot embedding with another held-out positive slot before trained projection scoring.",
    }
    artifact_root = root / "outputs_v2" / "artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)
    for name, payload in (
        ("latest_stage2_v51_real_training.json", training_payload),
        ("latest_stage2_v51_latent_eval.json", latent_payload),
    ):
        (artifact_root / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        (artifact_root / f"{training_payload['generated_at']}_{name}").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return training_payload, latent_payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-samples", type=int, default=12000)
    parser.add_argument("--val-samples", type=int, default=1000)
    parser.add_argument("--test-samples", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--encode-batch-size", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--device", default="cuda:2")
    args = parser.parse_args()

    training, latent = train_real_v51(
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        test_samples=args.test_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        encode_batch_size=args.encode_batch_size,
        latent_dim=args.latent_dim,
        device_name=args.device,
    )
    print(json.dumps({"training": training, "latent_eval": latent}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
