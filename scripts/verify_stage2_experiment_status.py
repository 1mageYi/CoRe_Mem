"""Mechanical verifier for stage-2 training and ablation completion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED_EXPERIMENTS = [
    {"id": "mainline", "description": "Default stage-2 mainline training plus local eval."},
    {"id": "ablation_t5_vs_optimus", "description": "T5 route vs Optimus-like."},
    {"id": "ablation_mean_pooling_vs_light_resampler", "description": "Mean pooling vs light resampler."},
    {"id": "ablation_belief_json_vs_direct_answer", "description": "Belief JSON decoding vs direct answer decoding."},
    {"id": "ablation_merge_only_vs_merge_overwrite", "description": "Merge-only vs merge+overwrite."},
    {"id": "ablation_single_bank_vs_core_residual", "description": "Single-bank vs core+residual."},
    {"id": "ablation_assignment_randomization", "description": "Core/residual assignment randomization."},
    {"id": "ablation_budget_sweep", "description": "Budget sweep."},
    {"id": "ablation_without_update_pool", "description": "Without update pool."},
    {"id": "ablation_without_persona_pool", "description": "Without persona pool."},
    {"id": "ablation_without_state_pool", "description": "Without state pool."},
    {"id": "ablation_without_synthetic_pool", "description": "Without synthetic pool."},
]


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping JSON in {path}")
    return payload


def _artifact_exists(root: Path, candidate: str | None) -> bool:
    if not candidate:
        return False
    path = Path(candidate)
    if not path.is_absolute():
        path = root / path
    return path.exists()


def compute_stage2_experiment_status(root: Path) -> dict[str, Any]:
    artifacts_root = root / "outputs_v2" / "artifacts"
    index_path = artifacts_root / "stage2_experiment_index.json"
    prepared_manifest = artifacts_root / "stage2_prepared_samples_manifest.json"

    completed = 0
    checks: dict[str, bool] = {
        "prepared_manifest_ready": prepared_manifest.exists(),
    }
    if checks["prepared_manifest_ready"]:
        completed += 1

    experiments_payload: dict[str, Any] = {}
    if index_path.exists():
        experiments_payload = _load_json(index_path).get("experiments", {})
        if not isinstance(experiments_payload, dict):
            experiments_payload = {}

    experiment_rows: list[dict[str, Any]] = []
    for spec in REQUIRED_EXPERIMENTS:
        record = experiments_payload.get(spec["id"], {})
        if not isinstance(record, dict):
            record = {}
        is_complete = bool(record.get("completed")) and _artifact_exists(root, record.get("checkpoint_dir")) and _artifact_exists(root, record.get("local_eval_path"))
        checks[spec["id"]] = is_complete
        if is_complete:
            completed += 1
        experiment_rows.append(
            {
                "id": spec["id"],
                "description": spec["description"],
                "completed": is_complete,
                "checkpoint_dir": record.get("checkpoint_dir"),
                "local_eval_path": record.get("local_eval_path"),
                "notes": record.get("notes"),
            }
        )

    total = 1 + len(REQUIRED_EXPERIMENTS)
    return {
        "score": completed,
        "total": total,
        "index_path": str(index_path),
        "checks": checks,
        "experiments": experiment_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--score-only", action="store_true")
    args = parser.parse_args()

    payload = compute_stage2_experiment_status(Path(args.root).resolve())
    if args.score_only:
        print(payload["score"])
        return 0

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
