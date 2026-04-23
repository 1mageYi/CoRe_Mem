"""Train v5.2 lightweight full learned latent modules on four stage-2 tasks.

The backbone stays pretrained and frozen, but the v5.2 latent substrate is
multi-task: retrieval alignment, slot autoencoding, composition-to-belief, and
lifecycle prediction all update trainable heads in one checkpoint.
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
DEFAULT_BACKBONE_ARTIFACT = ARTIFACT_ROOT / "latest_stage2_v52_backbone_compare.json"
DEFAULT_DATA_ROOT = ARTIFACT_ROOT / "stage2_v27_32k"
TASKS = (
    "retrieval_alignment",
    "slot_autoencoding",
    "composition_to_belief",
    "lifecycle_prediction",
)
ACTION_VOCAB = ("append", "keep", "merge", "new", "overwrite", "promote", "ignore")


class V52FullLatentSystem(nn.Module):
    def __init__(self, embedding_dim: int, latent_dim: int, action_count: int) -> None:
        super().__init__()
        self.encoder_adapter = nn.Sequential(
            nn.Linear(embedding_dim, latent_dim),
            nn.GELU(),
            nn.LayerNorm(latent_dim),
        )
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
        self.latent_reader = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.LayerNorm(latent_dim),
        )
        self.latent_resampler = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=4,
            batch_first=True,
        )
        self.write_controller = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, action_count),
        )
        self.belief_decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, embedding_dim),
        )
        self.belief_graph = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.slot_decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, embedding_dim),
        )
        self.temperature = nn.Parameter(torch.tensor(8.0))

    def retrieval_logits(self, query: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        query_latent = torch.nn.functional.normalize(self.query_projection(query), dim=-1)
        slot_latent = torch.nn.functional.normalize(self.slot_projection(candidates), dim=-1)
        slot_latent = self.latent_reader(slot_latent)
        return torch.einsum("bd,bcd->bc", query_latent, slot_latent) * self.temperature.clamp(1.0, 30.0)

    def slot_reconstruction(self, observation: torch.Tensor) -> torch.Tensor:
        latent = self.encoder_adapter(observation)
        return self.slot_decoder(latent)

    def belief_outputs(
        self,
        query: torch.Tensor,
        memory_slots: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query_latent = self.query_projection(query).unsqueeze(1)
        slot_latent = self.latent_reader(self.slot_projection(memory_slots))
        composed, _ = self.latent_resampler(query_latent, slot_latent, slot_latent)
        graph_state = self.belief_graph(composed.squeeze(1))
        logits = torch.einsum("bd,bcd->bc", graph_state, slot_latent) * self.temperature.clamp(1.0, 30.0)
        belief_vector = self.belief_decoder(graph_state)
        return logits, belief_vector

    def lifecycle_logits(self, observation: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        obs_latent = self.encoder_adapter(observation)
        ctx_latent = self.encoder_adapter(context)
        return self.write_controller(torch.cat([obs_latent, ctx_latent], dim=-1))


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


def _obs_text(obs: dict[str, Any]) -> str:
    gloss = str(obs.get("canonical_gloss") or "").strip()
    if gloss:
        return gloss
    relation = str(obs.get("relation") or "").strip()
    value = str(obs.get("value") or "").strip()
    evidence = str(obs.get("evidence_text") or "").strip()
    return " ".join(part for part in (relation, value, evidence) if part)


def _target_record_text(record: dict[str, Any]) -> str:
    relation = str(record.get("relation") or "").strip()
    value = str(record.get("value") or "").strip()
    status = str(record.get("status") or "").strip()
    time_scope = str(record.get("time_scope") or "").strip()
    return " ".join(part for part in (relation, value, status, time_scope) if part)


def _belief_text(payload: dict[str, Any]) -> str:
    items = payload.get("belief_items") or []
    if items:
        first = items[0]
        return _target_record_text(
            {
                "relation": first.get("relation"),
                "value": first.get("value"),
                "status": first.get("status"),
                "time_scope": first.get("time_scope"),
            }
        )
    return json.dumps(payload, sort_keys=True)


def _lifecycle_context_text(slots: list[dict[str, Any]]) -> str:
    if not slots:
        return "empty memory context"
    return " ; ".join(_slot_text(slot) for slot in slots[:4])


def _encode_texts(
    encoder: Any,
    texts: list[str],
    *,
    model_id: str,
    role: str,
    batch_size: int,
) -> torch.Tensor:
    prepared = [_prepared_text(text, model_id=model_id, role=role) for text in texts]
    encoded = encoder.encode(
        prepared,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return torch.tensor(np.asarray(encoded, dtype=np.float32))


def _retrieval_tensors(
    encoder: Any,
    rows: list[dict[str, Any]],
    *,
    model_id: str,
    negatives_per_query: int,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    queries: list[str] = []
    candidates: list[str] = []
    candidate_count = negatives_per_query + 1
    for row in rows:
        queries.append(str(row.get("query") or ""))
        row_candidates = [_slot_text(row.get("positive_slot") or {})]
        for slot in list(row.get("negative_slots") or [])[:negatives_per_query]:
            row_candidates.append(_slot_text(slot))
        while len(row_candidates) < candidate_count:
            row_candidates.append(row_candidates[-1] if row_candidates else "")
        candidates.extend(row_candidates[:candidate_count])
    query_tensor = _encode_texts(encoder, queries, model_id=model_id, role="query", batch_size=batch_size)
    candidate_tensor = _encode_texts(encoder, candidates, model_id=model_id, role="passage", batch_size=batch_size)
    return query_tensor, candidate_tensor.view(len(rows), candidate_count, -1)


def _slot_tensors(
    encoder: Any,
    rows: list[dict[str, Any]],
    *,
    model_id: str,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    obs_texts = [_obs_text(row.get("input_observation") or {}) for row in rows]
    target_texts = [_target_record_text(row.get("target_record") or {}) for row in rows]
    return (
        _encode_texts(encoder, obs_texts, model_id=model_id, role="passage", batch_size=batch_size),
        _encode_texts(encoder, target_texts, model_id=model_id, role="passage", batch_size=batch_size),
    )


def _composition_tensors(
    encoder: Any,
    rows: list[dict[str, Any]],
    *,
    model_id: str,
    max_slots: int,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    queries: list[str] = []
    slot_texts: list[str] = []
    labels: list[int] = []
    belief_texts: list[str] = []
    for row in rows:
        queries.append(str(row.get("query") or ""))
        slots = list(row.get("memory_slots") or [])[:max_slots]
        if not slots:
            slots = [{"canonical_gloss": ""}]
        gold_ids = set(str(item) for item in row.get("retrieved_slot_ids") or [])
        label = 0
        row_texts: list[str] = []
        for idx, slot in enumerate(slots):
            row_texts.append(_slot_text(slot))
            if str(slot.get("slot_id")) in gold_ids:
                label = idx
        while len(row_texts) < max_slots:
            row_texts.append(row_texts[-1])
        slot_texts.extend(row_texts[:max_slots])
        labels.append(label)
        belief_texts.append(_belief_text(row.get("target_belief_json") or {}))
    query_tensor = _encode_texts(encoder, queries, model_id=model_id, role="query", batch_size=batch_size)
    slot_tensor = _encode_texts(encoder, slot_texts, model_id=model_id, role="passage", batch_size=batch_size)
    belief_tensor = _encode_texts(encoder, belief_texts, model_id=model_id, role="passage", batch_size=batch_size)
    return query_tensor, slot_tensor.view(len(rows), max_slots, -1), torch.tensor(labels), belief_tensor


def _lifecycle_tensors(
    encoder: Any,
    rows: list[dict[str, Any]],
    *,
    model_id: str,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    obs_texts = [_obs_text(row.get("new_observation") or {}) for row in rows]
    context_texts = [_lifecycle_context_text(list(row.get("memory_context") or [])) for row in rows]
    labels = [ACTION_VOCAB.index(str(row.get("target_action") or "ignore")) if str(row.get("target_action") or "ignore") in ACTION_VOCAB else ACTION_VOCAB.index("ignore") for row in rows]
    return (
        _encode_texts(encoder, obs_texts, model_id=model_id, role="passage", batch_size=batch_size),
        _encode_texts(encoder, context_texts, model_id=model_id, role="passage", batch_size=batch_size),
        torch.tensor(labels),
    )


def _rank_metrics(logits: torch.Tensor) -> dict[str, float]:
    order = torch.argsort(logits, dim=1, descending=True)
    ranks = (order == 0).nonzero()[:, 1] + 1
    return {
        "top1_accuracy": float((ranks == 1).float().mean().item()),
        "mrr": float((1.0 / ranks.float()).mean().item()),
        "num_examples": float(logits.shape[0]),
    }


def _batched_indices(length: int, batch_size: int, *, generator: torch.Generator) -> list[torch.Tensor]:
    if length <= 0:
        return []
    perm = torch.randperm(length, generator=generator)
    return [perm[start : start + batch_size] for start in range(0, length, batch_size)]


def _eval_retrieval(
    model: V52FullLatentSystem,
    query: torch.Tensor,
    candidates: torch.Tensor,
    *,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        logits = model.retrieval_logits(query.to(device), candidates.to(device)).cpu()
    return _rank_metrics(logits)


def train_v52_multitask(
    *,
    root: Path = REPO_ROOT,
    data_root: Path = DEFAULT_DATA_ROOT,
    backbone_artifact: Path = DEFAULT_BACKBONE_ARTIFACT,
    retrieval_samples: int = 10000,
    slot_samples: int = 6000,
    composition_samples: int = 6000,
    lifecycle_samples: int = 2500,
    val_samples: int = 1000,
    negatives_per_query: int = 4,
    max_memory_slots: int = 4,
    latent_dim: int = 128,
    epochs: int = 2,
    batch_size: int = 256,
    encode_batch_size: int = 128,
    learning_rate: float = 2e-3,
    device_name: str = "cuda:2",
) -> dict[str, Any]:
    from sentence_transformers import SentenceTransformer

    started = time.time()
    backbone = _read_json(backbone_artifact)
    model_id = str(backbone.get("selected_backbone") or (backbone.get("loaded_backbones") or [""])[0])
    if not model_id:
        raise ValueError("Backbone artifact does not contain selected_backbone or loaded_backbones.")
    encoder = SentenceTransformer(
        model_id,
        device=device_name,
        cache_folder=os.environ.get("HF_HOME", str(root / "outputs_v2" / "hf_cache")),
    )

    train_dir = data_root / "train"
    val_dir = data_root / "val"
    retrieval_rows = _read_jsonl(train_dir / "retrieval_alignment.jsonl", limit=retrieval_samples)
    slot_rows = _read_jsonl(train_dir / "slot_autoencoding.jsonl", limit=slot_samples)
    composition_rows = _read_jsonl(train_dir / "composition_to_belief.jsonl", limit=composition_samples)
    lifecycle_rows = _read_jsonl(train_dir / "lifecycle_prediction.jsonl", limit=lifecycle_samples)
    val_retrieval_rows = _read_jsonl(val_dir / "retrieval_alignment.jsonl", limit=val_samples)

    retrieval_query, retrieval_candidates = _retrieval_tensors(
        encoder,
        retrieval_rows,
        model_id=model_id,
        negatives_per_query=negatives_per_query,
        batch_size=encode_batch_size,
    )
    slot_input, slot_target = _slot_tensors(
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
    val_query, val_candidates = _retrieval_tensors(
        encoder,
        val_retrieval_rows,
        model_id=model_id,
        negatives_per_query=negatives_per_query,
        batch_size=encode_batch_size,
    )

    device = torch.device(device_name if torch.cuda.is_available() or not device_name.startswith("cuda") else "cpu")
    model = V52FullLatentSystem(
        embedding_dim=int(retrieval_query.shape[1]),
        latent_dim=latent_dim,
        action_count=len(ACTION_VOCAB),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    target_zero = torch.zeros(len(retrieval_query), dtype=torch.long)
    generator = torch.Generator().manual_seed(52)
    loss_curve: list[dict[str, float]] = []

    for epoch in range(epochs):
        model.train()
        retrieval_loss_total = 0.0
        slot_loss_total = 0.0
        composition_loss_total = 0.0
        lifecycle_loss_total = 0.0
        retrieval_count = slot_count = composition_count = lifecycle_count = 0

        for batch_indices in _batched_indices(len(retrieval_query), batch_size, generator=generator):
            logits = model.retrieval_logits(
                retrieval_query[batch_indices].to(device),
                retrieval_candidates[batch_indices].to(device),
            )
            loss = torch.nn.functional.cross_entropy(logits, target_zero[batch_indices].to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            retrieval_loss_total += float(loss.item()) * len(batch_indices)
            retrieval_count += len(batch_indices)

        for batch_indices in _batched_indices(len(slot_input), batch_size, generator=generator):
            pred = model.slot_reconstruction(slot_input[batch_indices].to(device))
            loss = torch.nn.functional.mse_loss(pred, slot_target[batch_indices].to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            slot_loss_total += float(loss.item()) * len(batch_indices)
            slot_count += len(batch_indices)

        for batch_indices in _batched_indices(len(comp_query), batch_size, generator=generator):
            logits, belief_vector = model.belief_outputs(
                comp_query[batch_indices].to(device),
                comp_slots[batch_indices].to(device),
            )
            rank_loss = torch.nn.functional.cross_entropy(logits, comp_labels[batch_indices].to(device))
            belief_loss = torch.nn.functional.mse_loss(
                belief_vector,
                comp_belief[batch_indices].to(device),
            )
            loss = rank_loss + 0.5 * belief_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            composition_loss_total += float(loss.item()) * len(batch_indices)
            composition_count += len(batch_indices)

        for batch_indices in _batched_indices(len(lifecycle_obs), batch_size, generator=generator):
            logits = model.lifecycle_logits(
                lifecycle_obs[batch_indices].to(device),
                lifecycle_ctx[batch_indices].to(device),
            )
            loss = torch.nn.functional.cross_entropy(logits, lifecycle_labels[batch_indices].to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            lifecycle_loss_total += float(loss.item()) * len(batch_indices)
            lifecycle_count += len(batch_indices)

        val_metrics = _eval_retrieval(model, val_query, val_candidates, device=device)
        loss_curve.append(
            {
                "epoch": float(epoch + 1),
                "retrieval_alignment_loss": retrieval_loss_total / max(retrieval_count, 1),
                "slot_autoencoding_loss": slot_loss_total / max(slot_count, 1),
                "composition_to_belief_loss": composition_loss_total / max(composition_count, 1),
                "lifecycle_prediction_loss": lifecycle_loss_total / max(lifecycle_count, 1),
                "val_retrieval_mrr": val_metrics["mrr"],
                "val_retrieval_top1_accuracy": val_metrics["top1_accuracy"],
            }
        )

    checkpoint_dir = root / "outputs_v2" / "checkpoints" / f"{_timestamp()}_stage2_v52_multitask"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "full_latent_system.pt"
    trained_modules = [
        "encoder_adapter",
        "query_projection",
        "slot_projection",
        "latent_reader",
        "latent_resampler",
        "write_controller",
        "lifecycle_head",
        "belief_decoder",
        "belief_graph",
        "slot_decoder",
        "temperature",
    ]
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_class": "V52FullLatentSystem",
            "model_id": model_id,
            "embedding_dim": int(retrieval_query.shape[1]),
            "latent_dim": latent_dim,
            "trained_tasks": list(TASKS),
            "trained_modules": trained_modules,
            "action_vocab": list(ACTION_VOCAB),
            "uses_personamem_gold": False,
        },
        checkpoint_path,
    )
    task_counts = {
        "retrieval_alignment": len(retrieval_rows),
        "slot_autoencoding": len(slot_rows),
        "composition_to_belief": len(composition_rows),
        "lifecycle_prediction": len(lifecycle_rows),
    }
    payload: dict[str, Any] = {
        "artifact_type": "stage2_v52_multitask_training",
        "commit_hash": _current_head(root),
        "generated_at": _timestamp(),
        "backbone_artifact": str(backbone_artifact.relative_to(root) if backbone_artifact.is_relative_to(root) else backbone_artifact),
        "model_id": model_id,
        "checkpoint_path": str(checkpoint_path.relative_to(root)),
        "checkpoint_exists": checkpoint_path.exists(),
        "train_samples": int(sum(task_counts.values())),
        "task_train_samples": task_counts,
        "val_samples": len(val_retrieval_rows),
        "train_seconds": round(time.time() - started, 3),
        "device": str(device),
        "epochs": epochs,
        "batch_size": batch_size,
        "encode_batch_size": encode_batch_size,
        "negatives_per_query": negatives_per_query,
        "latent_dim": latent_dim,
        "trained_tasks": list(TASKS),
        "trained_modules": trained_modules,
        "frozen_modules": ["sentence_transformer_backbone"],
        "retrieval_only": False,
        "uses_personamem_gold": False,
        "gold_source_note": "Training uses stage2_v27_32k public-data task files only; no PersonaMem answer labels or options are read.",
        "loss_curve": loss_curve,
        "final_val_retrieval_metrics": _eval_retrieval(model, val_query, val_candidates, device=device),
    }
    artifact_root = root / "outputs_v2" / "artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)
    latest = artifact_root / "latest_stage2_v52_multitask_training.json"
    latest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (artifact_root / f"{payload['generated_at']}_latest_stage2_v52_multitask_training.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval-samples", type=int, default=10000)
    parser.add_argument("--slot-samples", type=int, default=6000)
    parser.add_argument("--composition-samples", type=int, default=6000)
    parser.add_argument("--lifecycle-samples", type=int, default=2500)
    parser.add_argument("--val-samples", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--encode-batch-size", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--device", default="cuda:2")
    args = parser.parse_args()
    payload = train_v52_multitask(
        retrieval_samples=args.retrieval_samples,
        slot_samples=args.slot_samples,
        composition_samples=args.composition_samples,
        lifecycle_samples=args.lifecycle_samples,
        val_samples=args.val_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        encode_batch_size=args.encode_batch_size,
        latent_dim=args.latent_dim,
        device_name=args.device,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
