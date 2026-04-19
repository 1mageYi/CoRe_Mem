"""Stage-2 local intrinsic evaluation entrypoint."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import csv
import json
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core_mem.v2.eval_local import evaluate_local
from core_mem.v2.experiments import variant_payload_for_experiment
from core_mem.v2.training import evaluate_stage2_checkpoint, slot_assignment_metrics_from_eval_payload
import yaml


def _write_summary_csv(path: Path, metrics: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        for name, value in metrics.items():
            writer.writerow([name, value])


def _write_budget_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["dataset", "sample_id", "budget", "joint_belief_accuracy", "belief_item_recall", "support_slot_recall"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _load_yaml(path: Path) -> dict[str, object]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("Stage-2 config must be a mapping.")
    return payload


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _current_commit_hash() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _publish_semantic_full_eval_artifact(
    *,
    prepared_manifest_path: Path,
    output_root: Path,
    payload: dict[str, object],
) -> str:
    artifact_path = output_root / "artifacts" / "latest_stage2_semantic_full_local_eval.json"
    artifact_payload = {
        "artifact_type": "stage2_semantic_full_local_eval",
        "commit_hash": _current_commit_hash(),
        "prepared_manifest": str(prepared_manifest_path),
        **payload,
    }
    _write_json(artifact_path, artifact_payload)
    return str(artifact_path)


def _publish_slot_assignment_eval_artifact(
    *,
    prepared_manifest_path: Path,
    output_root: Path,
    payload: dict[str, object],
) -> str:
    trained_eval = payload.get("trained_eval") if isinstance(payload, dict) else None
    slot_assignment_metrics = slot_assignment_metrics_from_eval_payload(trained_eval if isinstance(trained_eval, dict) else None)
    artifact_path = output_root / "artifacts" / "latest_stage2_slot_assignment_eval.json"
    artifact_payload = {
        "artifact_type": "stage2_slot_assignment_eval",
        "commit_hash": _current_commit_hash(),
        "prepared_manifest": str(prepared_manifest_path),
        "trained_eval": trained_eval,
        **slot_assignment_metrics,
        "result_path": payload.get("result_path"),
    }
    _write_json(artifact_path, artifact_payload)
    return str(artifact_path)


def _publish_v24_eval_artifact(
    *,
    prepared_manifest_path: Path,
    output_root: Path,
    payload: dict[str, object],
) -> str:
    trained_eval = payload.get("trained_eval") if isinstance(payload, dict) else None
    raw_slot_assignment_metrics = slot_assignment_metrics_from_eval_payload(trained_eval if isinstance(trained_eval, dict) else None)
    slot_assignment_metrics = {
        "task": raw_slot_assignment_metrics.get("slot_assignment_task"),
        "count": raw_slot_assignment_metrics.get("slot_assignment_count", 0),
        "accuracy": raw_slot_assignment_metrics.get("slot_assignment_accuracy", 0.0),
        "field_f1": raw_slot_assignment_metrics.get("slot_assignment_field_f1", 0.0),
        "exact_match": raw_slot_assignment_metrics.get("slot_assignment_exact_match", 0.0),
        "token_f1": raw_slot_assignment_metrics.get("slot_assignment_token_f1", 0.0),
        **raw_slot_assignment_metrics,
    }
    artifact_path = output_root / "artifacts" / "latest_stage2_v24_eval.json"
    artifact_payload = {
        "artifact_type": "stage2_v24_eval",
        "commit_hash": _current_commit_hash(),
        "prepared_manifest": str(prepared_manifest_path),
        **payload,
        "trained_eval": trained_eval,
        "slot_assignment_metrics": slot_assignment_metrics,
        "slot_assignment": slot_assignment_metrics,
    }
    _write_json(artifact_path, artifact_payload)
    return str(artifact_path)


def _load_eval_payload(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Eval payload at {path} must be a JSON object.")
    return payload


def _trained_eval_payload(payload: dict[str, object]) -> dict[str, object]:
    trained = payload.get("trained_eval")
    if isinstance(trained, dict):
        return trained
    return payload


def _task_metrics(payload: dict[str, object], task_name: str) -> dict[str, object]:
    per_task = payload.get("per_task")
    if not isinstance(per_task, dict):
        return {}
    task_payload = per_task.get(task_name)
    return task_payload if isinstance(task_payload, dict) else {}


def _positive_task_gain(
    baseline_payload: dict[str, object],
    current_payload: dict[str, object],
    *,
    task_name: str,
) -> bool:
    baseline_task = _task_metrics(baseline_payload, task_name)
    current_task = _task_metrics(current_payload, task_name)
    return (
        float(current_task.get("token_f1", 0.0)) > float(baseline_task.get("token_f1", 0.0))
        or float(current_task.get("field_f1", 0.0)) > float(baseline_task.get("field_f1", 0.0))
    )


def _publish_v30_belief_decoder_eval_artifact(
    *,
    prepared_manifest_path: Path,
    baseline_eval_path: Path,
    output_root: Path,
    payload: dict[str, object],
) -> str:
    baseline_eval = _trained_eval_payload(_load_eval_payload(baseline_eval_path))
    current_eval = _trained_eval_payload(payload)
    baseline_task = _task_metrics(baseline_eval, "composition_to_belief")
    current_task = _task_metrics(current_eval, "composition_to_belief")
    artifact_path = output_root / "artifacts" / "latest_stage2_v30_belief_decoder_eval.json"
    artifact_payload = {
        "artifact_type": "stage2_v30_belief_decoder_eval",
        "commit_hash": _current_commit_hash(),
        "prepared_manifest": str(prepared_manifest_path),
        "baseline_eval_path": str(baseline_eval_path),
        "current_eval_path": payload.get("result_path"),
        "baseline_composition_to_belief": baseline_task,
        "current_composition_to_belief": current_task,
        "belief_recovery_family": payload.get("families", {}).get("belief_recovery_family", {}),
        "belief_decoder_module": payload.get("modules", {}).get("belief_decoder", {}),
        "delta_token_f1": float(current_task.get("token_f1", 0.0)) - float(baseline_task.get("token_f1", 0.0)),
        "delta_field_f1": float(current_task.get("field_f1", 0.0)) - float(baseline_task.get("field_f1", 0.0)),
        "delta_exact_match": float(current_task.get("exact_match", 0.0)) - float(baseline_task.get("exact_match", 0.0)),
        "positive_gain": _positive_task_gain(baseline_eval, current_eval, task_name="composition_to_belief"),
    }
    _write_json(artifact_path, artifact_payload)
    return str(artifact_path)


def _publish_v30_write_gain_artifact(
    *,
    prepared_manifest_path: Path,
    baseline_eval_path: Path,
    output_root: Path,
    payload: dict[str, object],
) -> str:
    baseline_eval = _trained_eval_payload(_load_eval_payload(baseline_eval_path))
    current_eval = _trained_eval_payload(payload)
    baseline_task = _task_metrics(baseline_eval, "lifecycle_prediction")
    current_task = _task_metrics(current_eval, "lifecycle_prediction")
    artifact_path = output_root / "artifacts" / "latest_stage2_v30_write_gain.json"
    artifact_payload = {
        "artifact_type": "stage2_v30_write_gain",
        "commit_hash": _current_commit_hash(),
        "prepared_manifest": str(prepared_manifest_path),
        "baseline_eval_path": str(baseline_eval_path),
        "current_eval_path": payload.get("result_path"),
        "baseline_lifecycle_prediction": baseline_task,
        "current_lifecycle_prediction": current_task,
        "delta_token_f1": float(current_task.get("token_f1", 0.0)) - float(baseline_task.get("token_f1", 0.0)),
        "delta_field_f1": float(current_task.get("field_f1", 0.0)) - float(baseline_task.get("field_f1", 0.0)),
        "delta_exact_match": float(current_task.get("exact_match", 0.0)) - float(baseline_task.get("exact_match", 0.0)),
        "positive_gain": _positive_task_gain(baseline_eval, current_eval, task_name="lifecycle_prediction"),
    }
    _write_json(artifact_path, artifact_payload)
    return str(artifact_path)


def _publish_v30_belief_gain_artifact(
    *,
    prepared_manifest_path: Path,
    baseline_eval_path: Path,
    output_root: Path,
    payload: dict[str, object],
) -> str:
    baseline_eval = _trained_eval_payload(_load_eval_payload(baseline_eval_path))
    current_eval = _trained_eval_payload(payload)
    baseline_task = _task_metrics(baseline_eval, "composition_to_belief")
    current_task = _task_metrics(current_eval, "composition_to_belief")
    artifact_path = output_root / "artifacts" / "latest_stage2_v30_belief_gain.json"
    artifact_payload = {
        "artifact_type": "stage2_v30_belief_gain",
        "component": "belief",
        "commit_hash": _current_commit_hash(),
        "prepared_manifest": str(prepared_manifest_path),
        "baseline_eval_path": str(baseline_eval_path),
        "current_eval_path": payload.get("result_path"),
        "baseline_composition_to_belief": baseline_task,
        "current_composition_to_belief": current_task,
        "belief_recovery_family": payload.get("families", {}).get("belief_recovery_family", {}),
        "belief_decoder_module": payload.get("modules", {}).get("belief_decoder", {}),
        "delta_token_f1": float(current_task.get("token_f1", 0.0)) - float(baseline_task.get("token_f1", 0.0)),
        "delta_field_f1": float(current_task.get("field_f1", 0.0)) - float(baseline_task.get("field_f1", 0.0)),
        "delta_exact_match": float(current_task.get("exact_match", 0.0)) - float(baseline_task.get("exact_match", 0.0)),
        "positive_gain": _positive_task_gain(baseline_eval, current_eval, task_name="composition_to_belief"),
    }
    _write_json(artifact_path, artifact_payload)
    return str(artifact_path)


def run_local_eval(
    prepared_manifest_path: Path,
    output_root: Path,
    *,
    top_k: int,
    budgets: list[int],
    datasets: list[str] | None,
    variant: dict[str, object] | None,
    checkpoint_dir: Path | None,
    train_config_path: Path | None,
    eval_device: str,
    max_eval_examples: int | None,
) -> dict[str, object]:
    payload = evaluate_local(
        prepared_manifest_path,
        output_root,
        top_k=top_k,
        budgets=budgets,
        datasets=datasets,
        variant=variant,
    )
    summary = payload["summary"]
    eval_dir = output_root / "evals_local"
    table_dir = output_root / "tables"
    eval_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result_path = eval_dir / f"{timestamp}_stage2_local_eval.json"
    summary_path = table_dir / f"{timestamp}_stage2_local_eval_summary.csv"
    budget_path = table_dir / f"{timestamp}_stage2_local_eval_budget_sweep.csv"
    if checkpoint_dir is not None:
        config_path = train_config_path or checkpoint_dir.parent.parent / "runs" / checkpoint_dir.name / "config_snapshot.yaml"
        if not Path(config_path).exists():
            raise FileNotFoundError(f"Missing train config for checkpoint eval: {config_path}")
        trained_eval = evaluate_stage2_checkpoint(
            _load_yaml(Path(config_path)),
            prepared_manifest_path,
            checkpoint_dir,
            device=eval_device,
            max_eval_examples=max_eval_examples,
        )
        summary["trained_eval"] = trained_eval

    flattened_metrics = dict(summary["metrics"])
    trained_eval_payload = summary.get("trained_eval")
    if isinstance(trained_eval_payload, dict):
        for name, value in trained_eval_payload.get("metrics", {}).items():
            flattened_metrics[f"trained::{name}"] = value

    result_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_summary_csv(summary_path, flattened_metrics)
    _write_budget_csv(budget_path, payload["budget_rows"])
    response = {
        "metrics": summary["metrics"],
        "families": summary["families"],
        "modules": summary["modules"],
        "budget_sweep": summary["budget_sweep"],
        "variant": summary.get("variant", {}),
        "result_path": str(result_path),
        "summary_table_path": str(summary_path),
        "budget_table_path": str(budget_path),
    }
    if "trained_eval" in summary:
        response["trained_eval"] = summary["trained_eval"]
    return response


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-manifest", default="outputs_v2/artifacts/stage2_prepared_samples_manifest.json")
    parser.add_argument("--output-root", default="outputs_v2")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--budget", type=int, action="append", help="Repeat to override default budget sweep.")
    parser.add_argument("--dataset", action="append", help="Optional dataset filter, e.g. sgd or personachat.")
    parser.add_argument("--experiment-id", help="Optional stage-2 experiment preset.")
    parser.add_argument("--variant-json", help="Optional JSON object with eval variant overrides.")
    parser.add_argument("--checkpoint-dir", help="Optional trained checkpoint dir for checkpoint-aware local eval.")
    parser.add_argument("--train-config", help="Optional config snapshot path used to train the checkpoint.")
    parser.add_argument("--eval-device", default="cpu")
    parser.add_argument("--max-eval-examples", type=int)
    parser.add_argument("--publish-semantic-full-eval", action="store_true")
    parser.add_argument("--publish-slot-assignment-eval", action="store_true")
    parser.add_argument("--publish-v24-eval", action="store_true")
    parser.add_argument("--publish-v30-belief-decoder-eval", action="store_true")
    parser.add_argument("--publish-v30-write-gain", action="store_true")
    parser.add_argument("--publish-v30-belief-gain", action="store_true")
    parser.add_argument("--v30-baseline-eval")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    variant: dict[str, object] = {}
    if args.experiment_id:
        variant.update(variant_payload_for_experiment(args.experiment_id))
    if args.variant_json:
        variant.update(json.loads(args.variant_json))

    prepared_manifest_path = Path(args.prepared_manifest)
    output_root = Path(args.output_root)
    payload = run_local_eval(
        prepared_manifest_path,
        output_root,
        top_k=args.top_k,
        budgets=args.budget or [1, 2, 4, 8],
        datasets=args.dataset,
        variant=variant,
        checkpoint_dir=Path(args.checkpoint_dir) if args.checkpoint_dir else None,
        train_config_path=Path(args.train_config) if args.train_config else None,
        eval_device=args.eval_device,
        max_eval_examples=args.max_eval_examples,
    )
    if args.publish_semantic_full_eval:
        payload["semantic_full_eval_artifact"] = _publish_semantic_full_eval_artifact(
            prepared_manifest_path=prepared_manifest_path,
            output_root=output_root,
            payload=payload,
        )
    if args.publish_slot_assignment_eval:
        payload["slot_assignment_eval_artifact"] = _publish_slot_assignment_eval_artifact(
            prepared_manifest_path=prepared_manifest_path,
            output_root=output_root,
            payload=payload,
        )
    if args.publish_v24_eval:
        payload["v24_eval_artifact"] = _publish_v24_eval_artifact(
            prepared_manifest_path=prepared_manifest_path,
            output_root=output_root,
            payload=payload,
        )
    if args.publish_v30_belief_decoder_eval or args.publish_v30_write_gain or args.publish_v30_belief_gain:
        if not args.v30_baseline_eval:
            raise ValueError("v30 eval publishers require --v30-baseline-eval")
    if args.publish_v30_belief_decoder_eval:
        payload["v30_belief_decoder_eval_artifact"] = _publish_v30_belief_decoder_eval_artifact(
            prepared_manifest_path=prepared_manifest_path,
            baseline_eval_path=Path(args.v30_baseline_eval),
            output_root=output_root,
            payload=payload,
        )
    if args.publish_v30_write_gain:
        payload["v30_write_gain_artifact"] = _publish_v30_write_gain_artifact(
            prepared_manifest_path=prepared_manifest_path,
            baseline_eval_path=Path(args.v30_baseline_eval),
            output_root=output_root,
            payload=payload,
        )
    if args.publish_v30_belief_gain:
        payload["v30_belief_gain_artifact"] = _publish_v30_belief_gain_artifact(
            prepared_manifest_path=prepared_manifest_path,
            baseline_eval_path=Path(args.v30_baseline_eval),
            output_root=output_root,
            payload=payload,
        )
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"result_path={payload['result_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
