"""Publish v5.2 real multi-backbone loading evidence.

This is the first v5.2 gate only: it proves that multiple real pretrained
backbones can be loaded and compared on gold-free stage-2 retrieval data. It
does not publish training or downstream benchmark claims.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.publish_stage2_v51_real_backbone import (
    DEFAULT_RETRIEVAL_FILE,
    _cache_path,
    _current_head,
    _embedding_dimension,
    _evaluate_model,
    _module_count,
    _parameter_count,
    _read_jsonl,
    _timestamp,
)


DEFAULT_MODEL_IDS = (
    "BAAI/bge-base-en-v1.5",
    "intfloat/e5-base-v2",
    "facebook/contriever",
)


def publish_v52_backbone_compare(
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
        "artifact_type": "stage2_v52_backbone_compare",
        "commit_hash": _current_head(root),
        "generated_at": _timestamp(),
        "backend": "sentence_transformers",
        "evaluation_backend": "sentence_transformers",
        "pretrained_weights_loaded": len(results) >= 2,
        "uses_personamem_gold": False,
        "gold_source_note": "Stage2 32k validation retrieval_alignment only; no PersonaMem answers, options, or gold labels are read.",
        "retrieval_eval_file": str(
            retrieval_file.relative_to(root) if retrieval_file.is_relative_to(root) else retrieval_file
        ),
        "eval_samples": len(rows),
        "negatives_per_query": negatives_per_query,
        "attempted_backbones": list(model_ids),
        "loaded_backbones": [str(item["model_id"]) for item in results],
        "failed_backbones": failures,
        "selected_backbone": selected["model_id"] if selected else None,
        "selected_metric": "gold_free_stage2_val_mrr_then_top1",
        "three_backbone_compare_complete": len(results) == len(model_ids) == 3,
        "minimum_two_real_backbones_loaded": len(results) >= 2,
        "note": "This artifact proves v5.2 real multi-backbone loading/eval only; multi-task training and downstream benchmark evidence are separate gates.",
        "results": results,
    }
    artifact_root = root / "outputs_v2" / "artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)
    latest = artifact_root / "latest_stage2_v52_backbone_compare.json"
    latest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (artifact_root / f"{payload['generated_at']}_latest_stage2_v52_backbone_compare.json").write_text(
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

    payload = publish_v52_backbone_compare(
        retrieval_file=args.retrieval_file,
        model_ids=tuple(args.model_ids or DEFAULT_MODEL_IDS),
        eval_samples=args.eval_samples,
        negatives_per_query=args.negatives_per_query,
        batch_size=args.batch_size,
        device=args.device,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not payload["minimum_two_real_backbones_loaded"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
