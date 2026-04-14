"""Stage-2 local intrinsic evaluation entrypoint."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import csv
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core_mem.v2.eval_local import evaluate_local
from core_mem.v2.training import evaluate_stage2_checkpoint
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


def run_local_eval(
    prepared_manifest_path: Path,
    output_root: Path,
    *,
    top_k: int,
    budgets: list[int],
    datasets: list[str] | None,
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
    parser.add_argument("--checkpoint-dir", help="Optional trained checkpoint dir for checkpoint-aware local eval.")
    parser.add_argument("--train-config", help="Optional config snapshot path used to train the checkpoint.")
    parser.add_argument("--eval-device", default="cpu")
    parser.add_argument("--max-eval-examples", type=int)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = run_local_eval(
        Path(args.prepared_manifest),
        Path(args.output_root),
        top_k=args.top_k,
        budgets=args.budget or [1, 2, 4, 8],
        datasets=args.dataset,
        checkpoint_dir=Path(args.checkpoint_dir) if args.checkpoint_dir else None,
        train_config_path=Path(args.train_config) if args.train_config else None,
        eval_device=args.eval_device,
        max_eval_examples=args.max_eval_examples,
    )
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"result_path={payload['result_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
