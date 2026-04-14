"""Stage-2 training planner and execution entrypoint."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core_mem.v2.training import save_training_artifacts, train_stage2_model


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("Stage-2 config must be a mapping.")
    return payload


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _count_jsonl_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def stage2_train_plan(config_path: Path, prepared_manifest_path: Path, output_root: Path, *, execute_smoke: bool) -> dict[str, Any]:
    config = _load_yaml(config_path)
    manifest = _load_json(prepared_manifest_path)
    run_dir = output_root / "runs" / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_stage2_train_plan"
    run_dir.mkdir(parents=True, exist_ok=True)

    config_snapshot = run_dir / "config_snapshot.yaml"
    shutil.copyfile(config_path, config_snapshot)

    task_files = {name: Path(path) for name, path in manifest["task_files"].items()}
    task_rows = {name: _count_jsonl_rows(path) for name, path in task_files.items()}
    plan = {
        "stage": "stage2",
        "execution_mode": "execute_smoke" if execute_smoke else "plan_only",
        "backbone": config.get("model", {}).get("backbone", "google/flan-t5-base"),
        "train_tasks": config.get("training", {}).get("tasks", []),
        "prepared_manifest": str(prepared_manifest_path),
        "task_rows": task_rows,
        "resampler": config.get("model", {}).get("resampler", {}),
        "lora": config.get("model", {}).get("lora", {}),
        "batching": config.get("training", {}).get("batching", {}),
    }
    (run_dir / "training_plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "launch_stage2_training.sh").write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                f"cd {REPO_ROOT}",
                "conda run -n core_mem python scripts/train_stage2.py \\",
                f"  --config {config_path} \\",
                f"  --prepared-manifest {prepared_manifest_path} \\",
                f"  --output-root {output_root} \\",
                "  --execute-train \\",
                "  --device cpu",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    if execute_smoke:
        checkpoint_payload = {
            "epoch": 1,
            "seen_examples": sum(task_rows.values()),
            "note": "Smoke execution only; no heavy model training performed.",
        }
        (output_root / "checkpoints").mkdir(parents=True, exist_ok=True)
        checkpoint_path = output_root / "checkpoints" / "stage2_smoke_checkpoint.json"
        checkpoint_path.write_text(json.dumps(checkpoint_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        plan["smoke_checkpoint"] = str(checkpoint_path)
    return {"run_dir": str(run_dir), "training_plan": plan}


def stage2_train_execute(
    config_path: Path,
    prepared_manifest_path: Path,
    output_root: Path,
    *,
    max_steps: int | None,
    max_train_examples: int | None,
    device: str,
    cuda_visible_devices: str | None,
) -> dict[str, Any]:
    if cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    config = _load_yaml(config_path)
    manifest = _load_json(prepared_manifest_path)
    run_dir = output_root / "runs" / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_stage2_train_exec"
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config_path, run_dir / "config_snapshot.yaml")
    (run_dir / "prepared_manifest_snapshot.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    metrics = train_stage2_model(
        config,
        prepared_manifest_path,
        max_steps=max_steps,
        max_train_examples=max_train_examples,
        device=device,
    )
    artifact_paths = save_training_artifacts(
        run_dir=run_dir,
        output_root=output_root,
        config=config,
        metrics=metrics,
    )
    payload = {
        "run_dir": str(run_dir),
        "metrics_path": artifact_paths["metrics_path"],
        "checkpoint_dir": artifact_paths["checkpoint_dir"],
        "num_examples": metrics["num_examples"],
        "num_steps": metrics["num_steps"],
        "optimizer_steps": metrics["optimizer_steps"],
        "final_loss": metrics["final_loss"],
        "device": device,
        "cuda_visible_devices": cuda_visible_devices,
    }
    (run_dir / "execution_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage2_train.yaml")
    parser.add_argument("--prepared-manifest", default="outputs_v2/artifacts/stage2_prepared_samples_manifest.json")
    parser.add_argument("--output-root", default="outputs_v2")
    parser.add_argument("--execute-smoke", action="store_true")
    parser.add_argument("--execute-train", action="store_true")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-train-examples", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cuda-visible-devices")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.execute_train:
        payload = stage2_train_execute(
            Path(args.config),
            Path(args.prepared_manifest),
            Path(args.output_root),
            max_steps=args.max_steps,
            max_train_examples=args.max_train_examples,
            device=args.device,
            cuda_visible_devices=args.cuda_visible_devices,
        )
    else:
        payload = stage2_train_plan(
            Path(args.config),
            Path(args.prepared_manifest),
            Path(args.output_root),
            execute_smoke=args.execute_smoke,
        )
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"run_dir={payload['run_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
