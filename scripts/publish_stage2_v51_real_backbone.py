"""Publish v5.1 real pretrained backbone loading evidence.

This script is intentionally limited to gold-free encoder loading and held-out
retrieval scoring. It does not create training checkpoints or claim substrate
training success.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = REPO_ROOT / "outputs_v2" / "artifacts"
DEFAULT_RETRIEVAL_FILE = ARTIFACT_ROOT / "stage2_v27_32k" / "val" / "retrieval_alignment.jsonl"
DEFAULT_MODEL_IDS = ("BAAI/bge-base-en-v1.5",)


def _timestamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def _current_head(root: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except Exception:
        return "unknown"


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


def _slot_text(slot: dict[str, Any]) -> str:
    gloss = str(slot.get("canonical_gloss") or "").strip()
    relation = str(slot.get("relation") or "").strip()
    bank = str(slot.get("bank") or "").strip()
    if gloss and relation and relation not in gloss:
        return f"{relation}: {gloss}"
    if gloss:
        return gloss
    return " ".join(part for part in (bank, relation, str(slot.get("slot_id") or "")) if part)


def _prepared_text(text: str, *, model_id: str, role: str) -> str:
    lowered_id = model_id.lower()
    if "e5" in lowered_id:
        return f"{'query' if role == 'query' else 'passage'}: {text}"
    if "bge" in lowered_id and role == "query":
        return f"Represent this sentence for searching relevant passages: {text}"
    return text


def _embedding_dimension(model: Any) -> int | None:
    if hasattr(model, "get_embedding_dimension"):
        return int(model.get_embedding_dimension())
    if hasattr(model, "get_sentence_embedding_dimension"):
        return int(model.get_sentence_embedding_dimension())
    return None


def _parameter_count(model: Any) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def _module_count(model: Any) -> int:
    return int(sum(1 for _ in model.modules()))


def _cache_path(model_id: str) -> str | None:
    hf_home = Path(os.environ.get("HF_HOME", REPO_ROOT / "outputs_v2" / "hf_cache"))
    candidate = hf_home / "hub" / f"models--{model_id.replace('/', '--')}"
    return str(candidate) if candidate.exists() else str(hf_home)


def _encode(model: Any, texts: list[str], *, batch_size: int) -> np.ndarray:
    encoded = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return np.asarray(encoded, dtype=np.float32)


def _evaluate_model(
    *,
    model: Any,
    model_id: str,
    rows: list[dict[str, Any]],
    negatives_per_query: int,
    batch_size: int,
) -> dict[str, float]:
    query_texts: list[str] = []
    candidate_texts: list[str] = []
    spans: list[tuple[int, int]] = []
    positive_texts: list[str] = []

    for row in rows:
        query_texts.append(_prepared_text(str(row.get("query") or ""), model_id=model_id, role="query"))
        positive = _slot_text(row.get("positive_slot") or {})
        positive_texts.append(_prepared_text(positive, model_id=model_id, role="passage"))
        start = len(candidate_texts)
        candidate_texts.append(positive_texts[-1])
        for slot in list(row.get("negative_slots") or [])[:negatives_per_query]:
            candidate_texts.append(_prepared_text(_slot_text(slot), model_id=model_id, role="passage"))
        spans.append((start, len(candidate_texts)))

    query_vectors = _encode(model, query_texts, batch_size=batch_size)
    candidate_vectors = _encode(model, candidate_texts, batch_size=batch_size)
    shuffled_positive_vectors = _encode(
        model,
        positive_texts[max(len(positive_texts) // 2, 1) :] + positive_texts[: max(len(positive_texts) // 2, 1)],
        batch_size=batch_size,
    )

    top1 = 0
    reciprocal_ranks: list[float] = []
    shuffled_ranks: list[float] = []
    for idx, (start, end) in enumerate(spans):
        scores = candidate_vectors[start:end] @ query_vectors[idx]
        rank_order = list(np.argsort(-scores))
        rank = rank_order.index(0) + 1
        top1 += int(rank == 1)
        reciprocal_ranks.append(1.0 / rank)

        shuffled_candidates = candidate_vectors[start:end].copy()
        shuffled_candidates[0] = shuffled_positive_vectors[idx]
        shuffled_scores = shuffled_candidates @ query_vectors[idx]
        shuffled_rank = list(np.argsort(-shuffled_scores)).index(0) + 1
        shuffled_ranks.append(1.0 / shuffled_rank)

    return {
        "num_examples": float(len(rows)),
        "top1_accuracy": float(top1 / len(rows)) if rows else 0.0,
        "mrr": float(sum(reciprocal_ranks) / len(reciprocal_ranks)) if reciprocal_ranks else 0.0,
        "shuffled_target_mrr": float(sum(shuffled_ranks) / len(shuffled_ranks)) if shuffled_ranks else 0.0,
    }


def publish_real_backbone_compare(
    *,
    root: Path = REPO_ROOT,
    retrieval_file: Path = DEFAULT_RETRIEVAL_FILE,
    model_ids: tuple[str, ...] = DEFAULT_MODEL_IDS,
    eval_samples: int = 128,
    negatives_per_query: int = 4,
    batch_size: int = 32,
    device: str = "cpu",
) -> dict[str, Any]:
    from sentence_transformers import SentenceTransformer

    rows = _read_jsonl(retrieval_file, limit=eval_samples)
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for model_id in model_ids:
        started = time.time()
        try:
            model = SentenceTransformer(
                model_id,
                device=device,
                cache_folder=os.environ.get("HF_HOME", str(root / "outputs_v2" / "hf_cache")),
            )
            metrics = _evaluate_model(
                model=model,
                model_id=model_id,
                rows=rows,
                negatives_per_query=negatives_per_query,
                batch_size=batch_size,
            )
            results.append(
                {
                    "model_id": model_id,
                    "backend": "sentence_transformers",
                    "pretrained_weights_loaded": True,
                    "embedding_dim": _embedding_dimension(model),
                    "parameter_count": _parameter_count(model),
                    "module_count": _module_count(model),
                    "hf_cache_path": _cache_path(model_id),
                    "load_seconds": round(time.time() - started, 3),
                    "device": device,
                    "metrics": metrics,
                    "ablation_metrics": {
                        "shuffled_target_mrr": metrics["shuffled_target_mrr"],
                        "mrr_minus_shuffled": metrics["mrr"] - metrics["shuffled_target_mrr"],
                    },
                }
            )
        except Exception as exc:
            failures.append(
                {
                    "model_id": model_id,
                    "backend": "sentence_transformers",
                    "pretrained_weights_loaded": False,
                    "device": device,
                    "load_seconds": round(time.time() - started, 3),
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:1000],
                }
            )

    selected = max(
        results,
        key=lambda item: (
            float(item["metrics"].get("mrr", 0.0)),
            float(item["metrics"].get("top1_accuracy", 0.0)),
            float(item["ablation_metrics"].get("mrr_minus_shuffled", 0.0)),
        ),
        default=None,
    )
    payload: dict[str, Any] = {
        "artifact_type": "stage2_v51_real_backbone_compare",
        "commit_hash": _current_head(root),
        "generated_at": _timestamp(),
        "backend": "sentence_transformers",
        "evaluation_backend": "sentence_transformers",
        "pretrained_weights_loaded": bool(results),
        "uses_personamem_gold": False,
        "gold_source_note": "Stage2 32k validation retrieval_alignment only; no PersonaMem answer or option label is read.",
        "retrieval_eval_file": str(retrieval_file.relative_to(root) if retrieval_file.is_relative_to(root) else retrieval_file),
        "eval_samples": len(rows),
        "negatives_per_query": negatives_per_query,
        "attempted_backbones": list(model_ids),
        "loaded_backbones": [str(item["model_id"]) for item in results],
        "failed_backbones": failures,
        "selected_backbone": selected["model_id"] if selected else None,
        "selected_metric": "gold_free_stage2_val_mrr_then_top1",
        "three_backbone_compare_complete": len(results) == len(model_ids) == 3,
        "note": "This artifact proves real pretrained loading/eval only; v5.1 training and checkpoint evidence are separate gates.",
        "results": results,
    }
    artifact_root = root / "outputs_v2" / "artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)
    latest = artifact_root / "latest_stage2_v51_real_backbone_compare.json"
    latest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (artifact_root / f"{payload['generated_at']}_latest_stage2_v51_real_backbone_compare.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval-file", type=Path, default=DEFAULT_RETRIEVAL_FILE)
    parser.add_argument("--model-id", action="append", dest="model_ids")
    parser.add_argument("--eval-samples", type=int, default=128)
    parser.add_argument("--negatives-per-query", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    payload = publish_real_backbone_compare(
        retrieval_file=args.retrieval_file,
        model_ids=tuple(args.model_ids or DEFAULT_MODEL_IDS),
        eval_samples=args.eval_samples,
        negatives_per_query=args.negatives_per_query,
        batch_size=args.batch_size,
        device=args.device,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not payload["pretrained_weights_loaded"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
