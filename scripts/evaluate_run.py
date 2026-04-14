"""Evaluate a predictions file with exact-match accuracy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_predictions(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def normalize_answer(value: object) -> str:
    return str(value or "").strip().lower()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--predictions-file", default="predictions.jsonl")
    parser.add_argument("--output-file", default="evaluation.json")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    predictions_path = run_dir / args.predictions_file
    rows = load_predictions(predictions_path)
    correct = sum(
        1 for row in rows if normalize_answer(row.get("prediction")) == normalize_answer(row.get("expected_answer"))
    )
    total = len(rows)
    accuracy = correct / total if total else 0.0
    payload = {
        "run_dir": str(run_dir),
        "predictions_file": str(predictions_path),
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
    }
    output_path = run_dir / args.output_file
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"evaluation_file": str(output_path), **payload}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
