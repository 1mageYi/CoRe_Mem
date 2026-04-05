"""Mechanical verifier for stage-1 platform readiness."""

from __future__ import annotations

import argparse
from pathlib import Path


CHECKS = {
    "environment_spec": "environment.yml",
    "default_config": "configs/defaults.yaml",
    "provider_adapter": "src/core_mem/providers/openai_compatible.py",
    "personamem_adapter": "src/core_mem/benchmarks/personamem.py",
    "longmemeval_adapter": "src/core_mem/benchmarks/longmemeval.py",
    "core_memory_module": "src/core_mem/memory/core.py",
    "run_entrypoint": "scripts/run_experiment.py",
    "stage1_verifier": "scripts/verify_stage1_status.py",
    "unit_test": "tests/test_verify_stage1_status.py",
    "smoke_test": "tests/test_smoke_pipeline.py",
}


def compute_status(root: Path) -> tuple[int, dict[str, bool]]:
    checks = {name: (root / rel_path).exists() for name, rel_path in CHECKS.items()}
    return sum(checks.values()), checks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--score-only", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    score, checks = compute_status(root)
    if args.score_only:
        print(score)
        return 0

    print(f"stage1_platform_score={score}")
    for name, passed in checks.items():
        print(f"{name}={'pass' if passed else 'fail'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

