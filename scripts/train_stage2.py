"""Stage-2 training planner and execution entrypoint."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

import torch
import yaml

from eval_stage2_local import run_local_eval

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core_mem.v2.experiments import (
    register_stage2_experiment,
    spec_for_experiment,
    variant_payload_for_experiment,
)
from core_mem.v2.latent_training import (
    build_latent_objective_examples,
    evaluate_latent_retriever,
    train_latent_retriever,
)
from core_mem.v2.training import save_training_artifacts, train_stage2_model


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("Stage-2 config must be a mapping.")
    return payload


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")


def _count_jsonl_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def _current_commit_hash() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _publish_semantic_full_train_artifact(
    *,
    output_root: Path,
    config_path: Path,
    prepared_manifest_path: Path,
    summary: dict[str, Any],
) -> str:
    artifact_path = output_root / "artifacts" / "latest_stage2_semantic_full_train.json"
    payload = {
        "artifact_type": "stage2_semantic_full_train",
        "commit_hash": _current_commit_hash(),
        "config_path": str(config_path),
        "prepared_manifest": str(prepared_manifest_path),
        **summary,
    }
    _write_json(artifact_path, payload)
    return str(artifact_path)


def _publish_slot_assignment_train_artifact(
    *,
    output_root: Path,
    config_path: Path,
    prepared_manifest_path: Path,
    summary: dict[str, Any],
) -> str:
    artifact_path = output_root / "artifacts" / "latest_stage2_slot_assignment_train.json"
    payload = {
        "artifact_type": "stage2_slot_assignment_train",
        "commit_hash": _current_commit_hash(),
        "slot_assignment_task": "lifecycle_prediction",
        "config_path": str(config_path),
        "prepared_manifest": str(prepared_manifest_path),
        **summary,
    }
    _write_json(artifact_path, payload)
    return str(artifact_path)


def _publish_v24_train_artifact(
    *,
    output_root: Path,
    config_path: Path,
    prepared_manifest_path: Path,
    summary: dict[str, Any],
) -> str:
    artifact_path = output_root / "artifacts" / "latest_stage2_v24_train.json"
    payload = {
        "artifact_type": "stage2_v24_train",
        "commit_hash": _current_commit_hash(),
        "config_path": str(config_path),
        "prepared_manifest": str(prepared_manifest_path),
        "slot_assignment_task": "lifecycle_prediction",
        **summary,
    }
    _write_json(artifact_path, payload)
    return str(artifact_path)


def _publish_v30_shared_backbone_train_artifact(
    *,
    output_root: Path,
    config_path: Path,
    prepared_manifest_path: Path,
    summary: dict[str, Any],
) -> str:
    artifact_path = output_root / "artifacts" / "latest_stage2_v30_shared_backbone_train.json"
    payload = {
        "artifact_type": "stage2_v30_shared_backbone_train",
        "commit_hash": _current_commit_hash(),
        "config_path": str(config_path),
        "prepared_manifest": str(prepared_manifest_path),
        "task_adapters_enabled": bool(summary.get("task_adapters_enabled", False)),
        "task_adapter_names": summary.get("task_adapter_names", {}),
        **summary,
    }
    _write_json(artifact_path, payload)
    return str(artifact_path)


def _publish_v30_task_adapter_compare_artifact(
    *,
    output_root: Path,
    baseline_eval_path: Path,
    summary: dict[str, Any],
) -> str:
    baseline_payload = _load_json(baseline_eval_path)
    baseline_eval = baseline_payload.get("trained_eval", baseline_payload)
    current_eval = summary.get("trained_eval")
    if not isinstance(current_eval, dict):
        raise ValueError("v30 task-adapter compare requires trained_eval in the current summary.")
    baseline_metrics = baseline_eval.get("metrics", {}) if isinstance(baseline_eval, dict) else {}
    current_metrics = current_eval.get("metrics", {})
    baseline_per_task = baseline_eval.get("per_task", {}) if isinstance(baseline_eval, dict) else {}
    current_per_task = current_eval.get("per_task", {})
    task_deltas = {}
    for task_name, current_task_payload in current_per_task.items():
        if not isinstance(current_task_payload, dict):
            continue
        baseline_task_payload = baseline_per_task.get(task_name, {}) if isinstance(baseline_per_task, dict) else {}
        current_task_token_f1 = float(current_task_payload.get("token_f1", 0.0))
        baseline_task_token_f1 = float((baseline_task_payload or {}).get("token_f1", 0.0))
        current_task_field_f1 = float(current_task_payload.get("field_f1", 0.0))
        baseline_task_field_f1 = float((baseline_task_payload or {}).get("field_f1", 0.0))
        task_deltas[task_name] = {
            "delta_token_f1": current_task_token_f1 - baseline_task_token_f1,
            "delta_field_f1": current_task_field_f1 - baseline_task_field_f1,
            "current_token_f1": current_task_token_f1,
            "baseline_token_f1": baseline_task_token_f1,
            "current_field_f1": current_task_field_f1,
            "baseline_field_f1": baseline_task_field_f1,
        }
    baseline_score = float(baseline_metrics.get("token_f1", 0.0)) + float(baseline_metrics.get("field_f1", 0.0))
    current_score = float(current_metrics.get("token_f1", 0.0)) + float(current_metrics.get("field_f1", 0.0))
    artifact_path = output_root / "artifacts" / "latest_stage2_v30_task_adapter_compare.json"
    payload = {
        "artifact_type": "stage2_v30_task_adapter_compare",
        "commit_hash": _current_commit_hash(),
        "baseline_eval_path": str(baseline_eval_path),
        "current_eval_path": str(summary.get("result_path", "")),
        "baseline_score": baseline_score,
        "current_score": current_score,
        "delta_score": current_score - baseline_score,
        "task_specific_positive_gain": current_score > baseline_score,
        "task_deltas": task_deltas,
        "task_adapters_enabled": bool(summary.get("task_adapters_enabled", False)),
        "task_adapter_names": summary.get("task_adapter_names", {}),
    }
    _write_json(artifact_path, payload)
    return str(artifact_path)


def _publish_v30_latent_module_train_artifact(
    *,
    output_root: Path,
    config_path: Path,
    train_manifest_path: Path,
    summary: dict[str, Any],
) -> str:
    artifact_path = output_root / "artifacts" / "latest_stage2_v30_latent_module_train.json"
    payload = {
        "artifact_type": "stage2_v30_latent_module_train",
        "commit_hash": _current_commit_hash(),
        "config_path": str(config_path),
        "prepared_manifest": str(train_manifest_path),
        "trainable_encoder_resampler": True,
        **summary,
    }
    _write_json(artifact_path, payload)
    return str(artifact_path)


def _publish_v30_latent_objective_eval_artifact(
    *,
    output_root: Path,
    train_manifest_path: Path,
    eval_manifest_path: Path,
    summary: dict[str, Any],
) -> str:
    artifact_path = output_root / "artifacts" / "latest_stage2_v30_latent_objective_eval.json"
    payload = {
        "artifact_type": "stage2_v30_latent_objective_eval",
        "commit_hash": _current_commit_hash(),
        "train_manifest": str(train_manifest_path),
        "eval_manifest": str(eval_manifest_path),
        **summary,
    }
    _write_json(artifact_path, payload)
    return str(artifact_path)


def _publish_v30_latent_gain_artifact(
    *,
    output_root: Path,
    summary: dict[str, Any],
) -> str:
    artifact_path = output_root / "artifacts" / "latest_stage2_v30_latent_gain.json"
    payload = {
        "artifact_type": "stage2_v30_latent_gain",
        "commit_hash": _current_commit_hash(),
        **summary,
    }
    _write_json(artifact_path, payload)
    return str(artifact_path)


def _publish_v31_latent_mainline_train_artifact(
    *,
    output_root: Path,
    config_path: Path,
    train_manifest_path: Path,
    eval_manifest_path: Path,
    summary: dict[str, Any],
) -> str:
    artifact_path = output_root / "artifacts" / "latest_stage2_v31_latent_mainline_train.json"
    payload = {
        "artifact_type": "stage2_v31_latent_mainline_train",
        "commit_hash": _current_commit_hash(),
        "config_path": str(config_path),
        "train_manifest": str(train_manifest_path),
        "heldout_manifest": str(eval_manifest_path),
        "trainable_latent": True,
        **summary,
    }
    _write_json(artifact_path, payload)
    return str(artifact_path)


def _publish_v31_latent_holdout_compare_artifact(
    *,
    output_root: Path,
    baseline_eval_path: Path,
    eval_manifest_path: Path,
    summary: dict[str, Any],
) -> str:
    baseline_payload = _load_json(baseline_eval_path)
    baseline_score = float(baseline_payload.get("current_top1_accuracy", 0.0)) + float(
        baseline_payload.get("current_mrr", 0.0)
    )
    current_score = float(summary.get("current_top1_accuracy", 0.0)) + float(summary.get("current_mrr", 0.0))
    artifact_path = output_root / "artifacts" / "latest_stage2_v31_latent_holdout_compare.json"
    payload = {
        "artifact_type": "stage2_v31_latent_holdout_compare",
        "commit_hash": _current_commit_hash(),
        "baseline_eval_path": str(baseline_eval_path),
        "heldout_manifest": str(eval_manifest_path),
        "baseline_top1_accuracy": float(baseline_payload.get("current_top1_accuracy", 0.0)),
        "baseline_mrr": float(baseline_payload.get("current_mrr", 0.0)),
        "current_top1_accuracy": float(summary.get("current_top1_accuracy", 0.0)),
        "current_mrr": float(summary.get("current_mrr", 0.0)),
        "delta_score": current_score - baseline_score,
        "positive_gain": current_score > baseline_score,
    }
    _write_json(artifact_path, payload)
    return str(artifact_path)


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
    variant: dict[str, Any] | None = None,
    online_aligned: bool = False,
) -> dict[str, Any]:
    if cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    config = _load_yaml(config_path)
    variant = dict(variant or {})
    if variant:
        config.setdefault("experiment", {})
        config["experiment"]["variant"] = dict(variant)
    if online_aligned:
        config.setdefault("training", {})
        config["training"]["online_aligned"] = True
    manifest = _load_json(prepared_manifest_path)
    run_dir = output_root / "runs" / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_stage2_train_exec"
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_yaml(run_dir / "config_snapshot.yaml", config)
    (run_dir / "prepared_manifest_snapshot.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    peak_gpu_memory_mb = None
    gpu_name = None
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())

    started_at = datetime.now(timezone.utc)
    started_perf = time.perf_counter()
    metrics = train_stage2_model(
        config,
        prepared_manifest_path,
        max_steps=max_steps,
        max_train_examples=max_train_examples,
        device=device,
        disabled_pools=variant.get("disabled_pools"),
    )
    wall_clock_seconds = max(time.perf_counter() - started_perf, 0.0)
    finished_at = datetime.now(timezone.utc)
    if device.startswith("cuda") and torch.cuda.is_available():
        peak_gpu_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

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
        "variant": variant,
        "online_aligned": bool(config.get("training", {}).get("online_aligned", False)),
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "wall_clock_seconds": wall_clock_seconds,
        "examples_per_second": (metrics["num_examples"] / wall_clock_seconds) if wall_clock_seconds else None,
        "steps_per_second": (metrics["num_steps"] / wall_clock_seconds) if wall_clock_seconds else None,
        "peak_gpu_memory_mb": peak_gpu_memory_mb,
        "gpu_name": gpu_name,
        "task_adapters_enabled": bool(metrics.get("task_adapters_enabled", False)),
        "task_adapter_names": metrics.get("task_adapter_names", {}),
    }
    (run_dir / "execution_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def run_v30_latent_objective_experiment(
    config_path: Path,
    train_manifest_path: Path,
    eval_manifest_path: Path,
    output_root: Path,
    *,
    max_steps: int | None,
    max_train_examples: int | None,
    max_eval_examples: int | None,
    device: str,
) -> dict[str, Any]:
    config = _load_yaml(config_path)
    latent_cfg = ((config.get("model", {}) or {}).get("latent_objective", {}) or {})
    training_cfg = ((config.get("training", {}) or {}).get("latent_objective", {}) or {})
    train_examples = build_latent_objective_examples(train_manifest_path, max_examples=max_train_examples)
    eval_examples = build_latent_objective_examples(eval_manifest_path, max_examples=max_eval_examples)
    baseline_model, _ = train_latent_retriever(
        train_examples[:1],
        device=device,
        batch_size=1,
        lr=0.0,
        num_epochs=0,
        max_steps=0,
        hidden_dim=int(latent_cfg.get("hidden_dim", 32)),
        latent_dim=int(latent_cfg.get("latent_dim", 16)),
        latent_queries=int(latent_cfg.get("latent_queries", 4)),
    )
    baseline_metrics = evaluate_latent_retriever(baseline_model, eval_examples, device=device)

    run_dir = output_root / "runs" / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_stage2_v30_latent_exec"
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_yaml(run_dir / "config_snapshot.yaml", config)
    model, train_summary = train_latent_retriever(
        train_examples,
        device=device,
        batch_size=int(training_cfg.get("batch_size", 64)),
        lr=float(training_cfg.get("lr", 1e-3)),
        num_epochs=int(training_cfg.get("num_train_epochs", 5)),
        max_steps=max_steps,
        hidden_dim=int(latent_cfg.get("hidden_dim", 32)),
        latent_dim=int(latent_cfg.get("latent_dim", 16)),
        latent_queries=int(latent_cfg.get("latent_queries", 4)),
    )
    current_metrics = evaluate_latent_retriever(model, eval_examples, device=device)

    checkpoints_dir = output_root / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = checkpoints_dir / run_dir.name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_dir / "latent_retriever.pt")

    delta_score = (current_metrics["top1_accuracy"] + current_metrics["mrr"]) - (
        baseline_metrics["top1_accuracy"] + baseline_metrics["mrr"]
    )
    train_payload = {
        "run_dir": str(run_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "device": device,
        "train_examples": len(train_examples),
        "eval_examples": len(eval_examples),
        **train_summary,
    }
    eval_payload = {
        "baseline_top1_accuracy": baseline_metrics["top1_accuracy"],
        "baseline_mrr": baseline_metrics["mrr"],
        "current_top1_accuracy": current_metrics["top1_accuracy"],
        "current_mrr": current_metrics["mrr"],
        "delta_score": delta_score,
        "positive_gain": delta_score > 0.0,
    }
    (run_dir / "latent_train_summary.json").write_text(json.dumps(train_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "latent_eval_summary.json").write_text(json.dumps(eval_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    train_artifact = _publish_v30_latent_module_train_artifact(
        output_root=output_root,
        config_path=config_path,
        train_manifest_path=train_manifest_path,
        summary=train_payload,
    )
    eval_artifact = _publish_v30_latent_objective_eval_artifact(
        output_root=output_root,
        train_manifest_path=train_manifest_path,
        eval_manifest_path=eval_manifest_path,
        summary=eval_payload,
    )
    gain_artifact = _publish_v30_latent_gain_artifact(output_root=output_root, summary=eval_payload)
    return {
        **train_payload,
        **eval_payload,
        "v30_latent_module_train_artifact": train_artifact,
        "v30_latent_objective_eval_artifact": eval_artifact,
        "v30_latent_gain_artifact": gain_artifact,
    }


def run_v31_latent_mainline_experiment(
    config_path: Path,
    train_manifest_path: Path,
    eval_manifest_path: Path,
    output_root: Path,
    *,
    baseline_eval_path: Path,
    max_steps: int | None,
    max_train_examples: int | None,
    max_eval_examples: int | None,
    device: str,
) -> dict[str, Any]:
    config = _load_yaml(config_path)
    latent_cfg = ((config.get("model", {}) or {}).get("latent_objective", {}) or {})
    training_cfg = ((config.get("training", {}) or {}).get("latent_objective", {}) or {})
    train_examples = build_latent_objective_examples(train_manifest_path, max_examples=max_train_examples)
    eval_examples = build_latent_objective_examples(eval_manifest_path, max_examples=max_eval_examples)

    run_dir = output_root / "runs" / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_stage2_v31_latent_exec"
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_yaml(run_dir / "config_snapshot.yaml", config)
    model, train_summary = train_latent_retriever(
        train_examples,
        device=device,
        batch_size=int(training_cfg.get("batch_size", 64)),
        lr=float(training_cfg.get("lr", 1e-3)),
        num_epochs=int(training_cfg.get("num_train_epochs", 5)),
        max_steps=max_steps,
        hidden_dim=int(latent_cfg.get("hidden_dim", 32)),
        latent_dim=int(latent_cfg.get("latent_dim", 16)),
        latent_queries=int(latent_cfg.get("latent_queries", 4)),
    )
    current_metrics = evaluate_latent_retriever(model, eval_examples, device=device)

    checkpoints_dir = output_root / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = checkpoints_dir / run_dir.name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_dir / "latent_retriever.pt")

    train_payload = {
        "run_dir": str(run_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "device": device,
        "train_examples": len(train_examples),
        "heldout_examples": len(eval_examples),
        **train_summary,
    }
    compare_summary = {
        "current_top1_accuracy": current_metrics["top1_accuracy"],
        "current_mrr": current_metrics["mrr"],
    }
    (run_dir / "latent_train_summary.json").write_text(json.dumps(train_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "latent_holdout_summary.json").write_text(json.dumps(compare_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    train_artifact = _publish_v31_latent_mainline_train_artifact(
        output_root=output_root,
        config_path=config_path,
        train_manifest_path=train_manifest_path,
        eval_manifest_path=eval_manifest_path,
        summary=train_payload,
    )
    compare_artifact = _publish_v31_latent_holdout_compare_artifact(
        output_root=output_root,
        baseline_eval_path=baseline_eval_path,
        eval_manifest_path=eval_manifest_path,
        summary=compare_summary,
    )
    compare_payload = _load_json(Path(compare_artifact))
    return {
        **train_payload,
        **(compare_payload or {}),
        "v31_latent_mainline_train_artifact": train_artifact,
        "v31_latent_holdout_compare_artifact": compare_artifact,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage2_train.yaml")
    parser.add_argument("--prepared-manifest", default="outputs_v2/artifacts/stage2_prepared_samples_manifest.json")
    parser.add_argument("--eval-manifest")
    parser.add_argument("--output-root", default="outputs_v2")
    parser.add_argument("--execute-smoke", action="store_true")
    parser.add_argument("--execute-train", action="store_true")
    parser.add_argument("--execute-v30-latent-objective", action="store_true")
    parser.add_argument("--execute-v31-latent-mainline", action="store_true")
    parser.add_argument("--experiment-id")
    parser.add_argument("--register-experiment", action="store_true")
    parser.add_argument("--resampler-type", choices=["light", "mean_pooling"])
    parser.add_argument("--decoder-type", choices=["belief_json", "direct_answer", "optimus_like"])
    parser.add_argument("--overwrite-mode", choices=["merge_overwrite", "merge_only"])
    parser.add_argument("--bank-mode", choices=["dual", "single"])
    parser.add_argument("--assignment-mode", choices=["default", "randomized"])
    parser.add_argument("--disable-pool", action="append", default=[])
    parser.add_argument("--eval-top-k", type=int)
    parser.add_argument("--eval-budget", type=int, action="append")
    parser.add_argument("--eval-device", default="cpu")
    parser.add_argument("--max-eval-examples", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-train-examples", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cuda-visible-devices")
    parser.add_argument("--online-aligned", action="store_true")
    parser.add_argument("--publish-semantic-full-train", action="store_true")
    parser.add_argument("--publish-slot-assignment-train", action="store_true")
    parser.add_argument("--publish-v24-train", action="store_true")
    parser.add_argument("--publish-v30-shared-backbone-train", action="store_true")
    parser.add_argument("--publish-v30-task-adapter-compare", action="store_true")
    parser.add_argument("--v30-shared-baseline-eval")
    parser.add_argument("--v31-latent-baseline-artifact")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    variant: dict[str, Any] = {}
    notes = ""
    budgets = args.eval_budget or [1, 2, 4, 8]
    if args.experiment_id:
        spec = spec_for_experiment(args.experiment_id)
        variant.update(variant_payload_for_experiment(args.experiment_id))
        notes = spec.notes
        if not args.eval_budget:
            budgets = list(spec.budgets)
    for key, value in (
        ("resampler_type", args.resampler_type),
        ("decoder_type", args.decoder_type),
        ("overwrite_mode", args.overwrite_mode),
        ("bank_mode", args.bank_mode),
        ("assignment_mode", args.assignment_mode),
    ):
        if value is not None:
            variant[key] = value
    if args.disable_pool:
        variant["disabled_pools"] = sorted({*variant.get("disabled_pools", []), *args.disable_pool})

    if args.execute_v30_latent_objective:
        payload = run_v30_latent_objective_experiment(
            Path(args.config),
            Path(args.prepared_manifest),
            Path(args.eval_manifest) if args.eval_manifest else Path(args.prepared_manifest),
            Path(args.output_root),
            max_steps=args.max_steps,
            max_train_examples=args.max_train_examples,
            max_eval_examples=args.max_eval_examples,
            device=args.device,
        )
    elif args.execute_v31_latent_mainline:
        if not args.v31_latent_baseline_artifact:
            raise ValueError("--execute-v31-latent-mainline requires --v31-latent-baseline-artifact")
        payload = run_v31_latent_mainline_experiment(
            Path(args.config),
            Path(args.prepared_manifest),
            Path(args.eval_manifest) if args.eval_manifest else Path(args.prepared_manifest),
            Path(args.output_root),
            baseline_eval_path=Path(args.v31_latent_baseline_artifact),
            max_steps=args.max_steps,
            max_train_examples=args.max_train_examples,
            max_eval_examples=args.max_eval_examples,
            device=args.device,
        )
    elif args.execute_train:
        config_path = Path(args.config)
        prepared_manifest_path = Path(args.prepared_manifest)
        output_root = Path(args.output_root)
        payload = stage2_train_execute(
            config_path,
            prepared_manifest_path,
            output_root,
            max_steps=args.max_steps,
            max_train_examples=args.max_train_examples,
            device=args.device,
            cuda_visible_devices=args.cuda_visible_devices,
            variant=variant,
            online_aligned=args.online_aligned,
        )
        if args.register_experiment:
            if not args.experiment_id:
                raise ValueError("--register-experiment requires --experiment-id")
            eval_payload = run_local_eval(
                Path(args.eval_manifest) if args.eval_manifest else Path(args.prepared_manifest),
                Path(args.output_root),
                top_k=args.eval_top_k or 8,
                budgets=budgets,
                datasets=None,
                variant=variant,
                checkpoint_dir=Path(payload["checkpoint_dir"]),
                train_config_path=Path(payload["run_dir"]) / "config_snapshot.yaml",
                eval_device=args.eval_device,
                max_eval_examples=args.max_eval_examples,
            )
            index_path = register_stage2_experiment(
                Path(args.output_root),
                experiment_id=args.experiment_id,
                train_run_dir=payload["run_dir"],
                checkpoint_dir=payload["checkpoint_dir"],
                local_eval_path=eval_payload["result_path"],
                summary_table_path=eval_payload["summary_table_path"],
                budget_table_path=eval_payload["budget_table_path"],
                metrics_path=payload["metrics_path"],
                variant=variant,
                budgets=budgets,
                notes=notes,
            )
            payload["local_eval_path"] = eval_payload["result_path"]
            payload["summary_table_path"] = eval_payload["summary_table_path"]
            payload["budget_table_path"] = eval_payload["budget_table_path"]
            if "trained_eval" in eval_payload:
                payload["trained_eval"] = eval_payload["trained_eval"]
            payload["experiment_index_path"] = str(index_path)
        if args.publish_semantic_full_train:
            payload["semantic_full_train_artifact"] = _publish_semantic_full_train_artifact(
                output_root=output_root,
                config_path=config_path,
                prepared_manifest_path=prepared_manifest_path,
                summary=payload,
            )
        if args.publish_slot_assignment_train:
            payload["slot_assignment_train_artifact"] = _publish_slot_assignment_train_artifact(
                output_root=output_root,
                config_path=config_path,
                prepared_manifest_path=prepared_manifest_path,
                summary=payload,
            )
        if args.publish_v24_train:
            payload["v24_train_artifact"] = _publish_v24_train_artifact(
                output_root=output_root,
                config_path=config_path,
                prepared_manifest_path=prepared_manifest_path,
                summary=payload,
            )
        if args.publish_v30_shared_backbone_train:
            payload["v30_shared_backbone_train_artifact"] = _publish_v30_shared_backbone_train_artifact(
                output_root=output_root,
                config_path=config_path,
                prepared_manifest_path=prepared_manifest_path,
                summary=payload,
            )
        if args.publish_v30_task_adapter_compare:
            if "trained_eval" not in payload:
                raise ValueError("--publish-v30-task-adapter-compare requires --register-experiment so trained_eval is available.")
            if not args.v30_shared_baseline_eval:
                raise ValueError("--publish-v30-task-adapter-compare requires --v30-shared-baseline-eval")
            payload["v30_task_adapter_compare_artifact"] = _publish_v30_task_adapter_compare_artifact(
                output_root=output_root,
                baseline_eval_path=Path(args.v30_shared_baseline_eval),
                summary=payload,
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
