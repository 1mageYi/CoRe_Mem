"""Trainable latent objective utilities for v30 experiments."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from core_mem.v2.encoder import _lexical_features


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _slot_text(slot_payload: dict[str, Any]) -> str:
    parts = [
        str(slot_payload.get("relation", "")),
        str(slot_payload.get("canonical_gloss", "")),
        str(slot_payload.get("bank", "")),
        str(slot_payload.get("entity", "")),
    ]
    return " ".join(part for part in parts if part)


def _slot_features(slot_payload: dict[str, Any]) -> list[float]:
    return _lexical_features(_slot_text(slot_payload))


@dataclass(frozen=True)
class LatentObjectiveExample:
    query_features: list[float]
    candidate_features: list[list[float]]
    target_index: int


def build_latent_objective_examples(
    prepared_manifest_path: Path,
    *,
    max_examples: int | None = None,
) -> list[LatentObjectiveExample]:
    manifest = _load_json(prepared_manifest_path)
    rows = _load_jsonl(Path(manifest["task_files"]["retrieval_alignment"]))
    examples: list[LatentObjectiveExample] = []
    for row in rows[:max_examples]:
        candidates = [row["positive_slot"], *row["negative_slots"]]
        examples.append(
            LatentObjectiveExample(
                query_features=_lexical_features(str(row["query"])),
                candidate_features=[_slot_features(candidate) for candidate in candidates],
                target_index=0,
            )
        )
    return examples


class LatentObjectiveDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, examples: list[LatentObjectiveExample]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        example = self.examples[index]
        return {
            "query_features": torch.tensor(example.query_features, dtype=torch.float32),
            "candidate_features": torch.tensor(example.candidate_features, dtype=torch.float32),
            "target_index": torch.tensor(example.target_index, dtype=torch.long),
        }


class TrainableLatentRetriever(nn.Module):
    def __init__(
        self,
        *,
        feature_dim: int = 12,
        hidden_dim: int = 32,
        latent_dim: int = 16,
        latent_queries: int = 4,
    ) -> None:
        super().__init__()
        self.query_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.slot_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.latent_query_bank = nn.Parameter(torch.randn(latent_queries, latent_dim) * 0.02)
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        *,
        query_features: torch.Tensor,
        candidate_features: torch.Tensor,
    ) -> torch.Tensor:
        query_hidden = nn.functional.normalize(self.query_encoder(query_features), dim=-1)
        slot_hidden = nn.functional.normalize(self.slot_encoder(candidate_features), dim=-1)
        latent_queries = nn.functional.normalize(
            query_hidden.unsqueeze(1) + self.latent_query_bank.unsqueeze(0),
            dim=-1,
        )
        attention_logits = torch.einsum("bqd,bkd->bqk", latent_queries, slot_hidden)
        pooled_logits = attention_logits.mean(dim=1)
        scale = torch.clamp(self.temperature.abs(), min=0.1)
        return pooled_logits / scale


def evaluate_latent_retriever(
    model: TrainableLatentRetriever,
    examples: list[LatentObjectiveExample],
    *,
    device: str = "cpu",
) -> dict[str, float]:
    if not examples:
        raise ValueError("Latent objective eval requires at least one example.")
    model.eval()
    hits = 0.0
    reciprocal_ranks: list[float] = []
    with torch.no_grad():
        for example in examples:
            logits = model(
                query_features=torch.tensor([example.query_features], dtype=torch.float32, device=device),
                candidate_features=torch.tensor([example.candidate_features], dtype=torch.float32, device=device),
            )[0]
            ranked = torch.argsort(logits, descending=True).tolist()
            predicted = ranked[0]
            hits += float(predicted == example.target_index)
            reciprocal_ranks.append(1.0 / (ranked.index(example.target_index) + 1))
    return {
        "top1_accuracy": hits / len(examples),
        "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks),
        "num_examples": float(len(examples)),
    }


def train_latent_retriever(
    train_examples: list[LatentObjectiveExample],
    *,
    device: str = "cpu",
    batch_size: int = 32,
    lr: float = 1e-3,
    num_epochs: int = 5,
    max_steps: int | None = None,
    hidden_dim: int = 32,
    latent_dim: int = 16,
    latent_queries: int = 4,
) -> tuple[TrainableLatentRetriever, dict[str, Any]]:
    if not train_examples:
        raise ValueError("Latent objective training requires at least one example.")
    torch.manual_seed(30)
    model = TrainableLatentRetriever(
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        latent_queries=latent_queries,
    ).to(device)
    loader = DataLoader(LatentObjectiveDataset(train_examples), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_history: list[float] = []
    step = 0
    for _ in range(num_epochs):
        for batch in loader:
            step += 1
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(query_features=batch["query_features"], candidate_features=batch["candidate_features"])
            loss = nn.functional.cross_entropy(logits, batch["target_index"])
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            loss_history.append(float(loss.detach().cpu().item()))
            if max_steps is not None and step >= max_steps:
                break
        if max_steps is not None and step >= max_steps:
            break
    summary = {
        "num_examples": len(train_examples),
        "num_steps": step,
        "final_loss": loss_history[-1] if loss_history else None,
        "hidden_dim": hidden_dim,
        "latent_dim": latent_dim,
        "latent_queries": latent_queries,
        "loss_history": loss_history,
    }
    return model, summary
