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


def run_local_eval(
    prepared_manifest_path: Path,
    output_root: Path,
    *,
    top_k: int,
    budgets: list[int],
    datasets: list[str] | None,
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
    result_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_summary_csv(summary_path, summary["metrics"])
    _write_budget_csv(budget_path, payload["budget_rows"])
    return {
        "metrics": summary["metrics"],
        "families": summary["families"],
        "modules": summary["modules"],
        "budget_sweep": summary["budget_sweep"],
        "result_path": str(result_path),
        "summary_table_path": str(summary_path),
        "budget_table_path": str(budget_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-manifest", default="outputs_v2/artifacts/stage2_prepared_samples_manifest.json")
    parser.add_argument("--output-root", default="outputs_v2")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--budget", type=int, action="append", help="Repeat to override default budget sweep.")
    parser.add_argument("--dataset", action="append", help="Optional dataset filter, e.g. sgd or personachat.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = run_local_eval(
        Path(args.prepared_manifest),
        Path(args.output_root),
        top_k=args.top_k,
        budgets=args.budget or [1, 2, 4, 8],
        datasets=args.dataset,
    )
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"result_path={payload['result_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
