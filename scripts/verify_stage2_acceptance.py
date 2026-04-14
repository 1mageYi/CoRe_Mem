"""Mechanical verifier for stage-2 acceptance evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


REQUIRED_DOCS = [
    "AGENTS.md",
    "docs/requirements.md",
    "docs/current_status.md",
    "docs/implementation_plan.md",
    "docs/todo.md",
    "docs/v2_design.md",
    ".agent-os/project-index.md",
    ".agent-os/todo.md",
    ".agent-os/acceptance-report.md",
    ".agent-os/run-log.md",
    ".agent-os/architecture-milestones.md",
]

REQUIRED_OUTPUT_DIRS = [
    "outputs_v2/checkpoints",
    "outputs_v2/runs",
    "outputs_v2/evals_local",
    "outputs_v2/evals_benchmark",
    "outputs_v2/tables",
    "outputs_v2/figures",
    "outputs_v2/logs",
    "outputs_v2/cache_manifests",
    "outputs_v2/artifacts",
]

REQUIRED_STAGE2_STRINGS = {
    "requirements_stage2_mainline": (
        "docs/requirements.md",
        [
            "V2.0 structured latent-slot memory",
            "Stage-2",
            "public-datasets-first, synthetic-minimal",
        ],
    ),
    "v2_design_schema": (
        "docs/v2_design.md",
        [
            "Observation",
            "Slot",
            "Belief",
            "outputs_v2/",
        ],
    ),
    "project_index_stage2": (
        ".agent-os/project-index.md",
        [
            "Stage-2",
            "Top Next Action",
        ],
    ),
}

REQUIRED_STAGE2_FILES = {
    "stage2_module_root": [
        "src/core_mem/v2/__init__.py",
        "src/core_mem/v2/schemas.py",
        "src/core_mem/v2/parser.py",
        "src/core_mem/v2/datasets.py",
    ],
    "stage2_training_pipeline": [
        "scripts/prepare_stage2_data.py",
        "scripts/train_stage2.py",
        "configs/stage2_train.yaml",
    ],
    "stage2_eval_pipeline": [
        "scripts/eval_stage2_local.py",
    ],
    "stage2_canary_pipeline": [
        "scripts/run_stage2_canary.py",
    ],
}


def file_contains_all(path: Path, patterns: list[str]) -> bool:
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8")
    return all(pattern in text for pattern in patterns)


def pytest_passes(root: Path) -> bool:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def compute_acceptance_status(root: Path, *, run_pytest: bool = True) -> dict[str, Any]:
    docs_complete = all((root / rel_path).exists() for rel_path in REQUIRED_DOCS)
    outputs_complete = all((root / rel_path).exists() for rel_path in REQUIRED_OUTPUT_DIRS)
    string_checks = {
        name: file_contains_all(root / rel_path, patterns)
        for name, (rel_path, patterns) in REQUIRED_STAGE2_STRINGS.items()
    }
    file_checks = {
        name: all((root / rel_path).exists() for rel_path in rel_paths)
        for name, rel_paths in REQUIRED_STAGE2_FILES.items()
    }

    criteria = {
        "AC-008": {
            "passed": docs_complete and string_checks["requirements_stage2_mainline"] and string_checks["project_index_stage2"],
            "evidence": {
                "requirements": str(root / "docs" / "requirements.md"),
                "v2_design": str(root / "docs" / "v2_design.md"),
                "project_index": str(root / ".agent-os" / "project-index.md"),
            },
        },
        "AC-009": {
            "passed": string_checks["v2_design_schema"],
            "evidence": {
                "schema_doc": str(root / "docs" / "v2_design.md"),
            },
        },
        "AC-010": {
            "passed": file_checks["stage2_module_root"],
            "evidence": {
                "stage2_module_root": str(root / "src" / "core_mem" / "v2"),
            },
        },
        "AC-011": {
            "passed": file_checks["stage2_training_pipeline"],
            "evidence": {
                "prepare_stage2_data": str(root / "scripts" / "prepare_stage2_data.py"),
                "train_stage2": str(root / "scripts" / "train_stage2.py"),
                "train_config": str(root / "configs" / "stage2_train.yaml"),
            },
        },
        "AC-012": {
            "passed": file_checks["stage2_eval_pipeline"] and outputs_complete,
            "evidence": {
                "eval_stage2_local": str(root / "scripts" / "eval_stage2_local.py"),
                "outputs_v2_root": str(root / "outputs_v2"),
            },
        },
        "AC-013": {
            "passed": file_checks["stage2_canary_pipeline"],
            "evidence": {
                "run_stage2_canary": str(root / "scripts" / "run_stage2_canary.py"),
            },
        },
        "AC-014": {
            "passed": pytest_passes(root) if run_pytest else False,
            "evidence": {
                "pytest_executed": run_pytest,
                "stage2_status_test": str(root / "tests" / "test_verify_stage2_status.py"),
            },
        },
    }

    passed = sum(1 for item in criteria.values() if item["passed"])
    return {
        "passed": passed,
        "total": len(criteria),
        "criteria": criteria,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument("--skip-pytest", action="store_true")
    args = parser.parse_args()

    payload = compute_acceptance_status(Path(args.root).resolve(), run_pytest=not args.skip_pytest)
    if args.score_only:
        print(payload["passed"])
        return 0 if payload["passed"] == payload["total"] else 1

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["passed"] == payload["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
