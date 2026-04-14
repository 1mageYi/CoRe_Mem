"""Aggregate evaluation artifacts into a stage-1 summary table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-root", default=str(REPO_ROOT / "outputs"))
    parser.add_argument("--evaluation-file", default="evaluation.json")
    args = parser.parse_args()

    outputs_root = Path(args.outputs_root)
    runs_root = outputs_root / "runs"
    tables_root = outputs_root / "tables"
    tables_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    if runs_root.exists():
        for run_dir in sorted(runs_root.iterdir()):
            evaluation_path = run_dir / args.evaluation_file
            if not evaluation_path.exists():
                continue
            payload = load_json(evaluation_path)
            rows.append(
                {
                    "run_dir": str(run_dir),
                    "accuracy": payload.get("accuracy", 0.0),
                    "correct": payload.get("correct", 0),
                    "total": payload.get("total", 0),
                }
            )

    json_path = tables_root / "stage1_summary.json"
    csv_path = tables_root / "stage1_summary.csv"
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["run_dir", "accuracy", "correct", "total"])
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps({"rows": len(rows), "json_path": str(json_path), "csv_path": str(csv_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
