"""Top-level experiment entrypoint for the CoRe Memory stage-1 platform."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_stage1_verifier(repo_root: Path) -> int:
    verifier = repo_root / "scripts" / "verify_stage1_status.py"
    cmd = [sys.executable, str(verifier), "--score-only"]
    result = subprocess.run(cmd, cwd=repo_root, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(result.returncode)
    print(result.stdout.strip())
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    if args.verify_only:
        return run_stage1_verifier(repo_root)

    print("Stage-1 experiment runner scaffold is ready.")
    print("Use --verify-only for the current mechanical status check.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

