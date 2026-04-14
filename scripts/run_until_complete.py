"""Resume-aware benchmark supervisor for slow or rate-limited providers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[1]


def count_predictions(run_dir: Path) -> int:
    predictions_path = run_dir / "predictions.jsonl"
    if not predictions_path.exists():
        return 0
    with predictions_path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def build_runner_command(args: argparse.Namespace, completed_before: int) -> list[str]:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_experiment.py"),
        args.benchmark,
        "--config",
        args.config,
        "--run-dir",
        args.run_dir,
        "--resume",
    ]
    effective_limit = args.limit
    if args.chunk_size is not None:
        desired_limit = completed_before + args.chunk_size
        effective_limit = desired_limit if effective_limit is None else min(effective_limit, desired_limit)
    if effective_limit is not None:
        cmd.extend(["--limit", str(effective_limit)])
    if args.benchmark == "personamem":
        if args.questions_csv:
            cmd.extend(["--questions-csv", args.questions_csv])
        if args.contexts_jsonl:
            cmd.extend(["--contexts-jsonl", args.contexts_jsonl])
    if args.benchmark == "longmemeval" and args.dataset_json:
        cmd.extend(["--dataset-json", args.dataset_json])
    return cmd


def supervise_run(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    target_count = args.expected_count
    launch_index = 0
    idle_launches = 0

    while True:
        completed_before = count_predictions(run_dir)
        if completed_before >= target_count:
            print(
                json.dumps(
                    {
                        "run_dir": str(run_dir),
                        "status": "completed",
                        "completed_predictions": completed_before,
                        "expected_count": target_count,
                        "launches": launch_index,
                    },
                    ensure_ascii=False,
                )
            )
            return 0

        launch_index += 1
        result = subprocess.run(build_runner_command(args, completed_before), cwd=REPO_ROOT, check=False)
        completed_after = count_predictions(run_dir)
        progressed = completed_after > completed_before
        idle_launches = 0 if progressed else idle_launches + 1

        print(
            json.dumps(
                {
                    "run_dir": str(run_dir),
                    "status": "retrying" if completed_after < target_count else "completed",
                    "launch": launch_index,
                    "completed_before": completed_before,
                    "completed_after": completed_after,
                    "expected_count": target_count,
                    "exit_code": result.returncode,
                    "progressed": progressed,
                    "idle_launches": idle_launches,
                },
                ensure_ascii=False,
            )
        )

        if completed_after >= target_count:
            return 0

        if args.max_launches is not None and launch_index >= args.max_launches:
            return 1

        if args.max_idle_launches is not None and idle_launches >= args.max_idle_launches:
            return 1

        time.sleep(args.cooldown_seconds)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark", choices=["personamem", "longmemeval"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--cooldown-seconds", type=float, default=60.0)
    parser.add_argument("--max-launches", type=int)
    parser.add_argument("--max-idle-launches", type=int)
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument("--questions-csv")
    parser.add_argument("--contexts-jsonl")
    parser.add_argument("--dataset-json")
    args = parser.parse_args()
    return supervise_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
