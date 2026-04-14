"""Mechanical verifier for stage-1 acceptance evidence."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core_mem.memory import (  # noqa: E402
    CoreMemoryUpdater,
    MemoryReader,
    MemoryWriter,
    ResidualMemoryManager,
    VanillaCoreMemory,
)


REQUIRED_DOCS = [
    "AGENTS.md",
    "docs/requirements.md",
    "docs/current_status.md",
    "docs/implementation_plan.md",
    "docs/todo.md",
    ".agent-os/project-index.md",
    ".agent-os/todo.md",
    ".agent-os/acceptance-report.md",
    ".agent-os/run-log.md",
]

REQUIRED_STEP_SCRIPTS = [
    "scripts/run_experiment.py",
    "scripts/prepare_data.py",
    "scripts/evaluate_run.py",
    "scripts/make_tables.py",
    "scripts/stage1_preflight.py",
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def dataset_size_personamem(root: Path) -> int:
    path = root / "data" / "personamem" / "questions_32k.csv"
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def dataset_size_longmemeval(root: Path) -> int:
    path = root / "data" / "longmemeval" / "longmemeval_s_cleaned.json"
    if not path.exists():
        return 0
    return len(json.loads(path.read_text(encoding="utf-8")))


def benchmark_name_for_run(run_dir: Path, metadata: dict[str, Any]) -> str:
    benchmark = metadata.get("benchmark")
    if benchmark:
        return str(benchmark)
    return run_dir.name.split("_", 1)[-1]


def iter_run_artifacts(root: Path) -> list[dict[str, Any]]:
    runs_root = root / "outputs" / "runs"
    artifacts: list[dict[str, Any]] = []
    if not runs_root.exists():
        return artifacts

    for run_dir in sorted(runs_root.iterdir()):
        metadata_path = run_dir / "run_metadata.json"
        if not metadata_path.exists():
            continue
        metadata = load_json(metadata_path)
        evaluation_path = run_dir / "evaluation.json"
        prediction_path = run_dir / "predictions.jsonl"
        config_snapshot = run_dir / "config_snapshot.yaml"
        evaluation = load_json(evaluation_path) if evaluation_path.exists() else None
        artifacts.append(
            {
                "run_dir": run_dir,
                "benchmark": benchmark_name_for_run(run_dir, metadata),
                "metadata": metadata,
                "evaluation": evaluation,
                "prediction_path": prediction_path,
                "config_snapshot_path": config_snapshot,
            }
        )
    return artifacts


def find_formal_run(root: Path, benchmark: str, expected_total: int) -> dict[str, Any] | None:
    if expected_total <= 0:
        return None
    for artifact in iter_run_artifacts(root):
        if artifact["benchmark"] != benchmark:
            continue
        evaluation = artifact["evaluation"]
        if evaluation is None:
            continue
        metadata = artifact["metadata"]
        if int(metadata.get("record_count", 0) or 0) != expected_total:
            continue
        if int(evaluation.get("total", 0) or 0) != expected_total:
            continue
        if not artifact["prediction_path"].exists():
            continue
        if not artifact["config_snapshot_path"].exists():
            continue
        return artifact
    return None


def conda_env_exists(name: str) -> bool:
    result = subprocess.run(
        ["conda", "env", "list", "--json"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return False
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return False
    envs = payload.get("envs", [])
    for env_path in envs:
        if Path(env_path).name == name:
            return True
    return False


def pytest_passes(root: Path) -> bool:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def traceability_complete(artifact: dict[str, Any] | None) -> bool:
    if artifact is None:
        return False
    metadata = artifact["metadata"]
    required_fields = [
        "config_snapshot_path",
        "model",
        "provider",
        "run_timestamp",
        "commit_hash",
        "prompt_version",
    ]
    return all(bool(metadata.get(field)) for field in required_fields)


def compute_acceptance_status(root: Path, *, run_pytest: bool = True) -> dict[str, Any]:
    personamem_total = dataset_size_personamem(root)
    longmemeval_total = dataset_size_longmemeval(root)
    personamem_formal = find_formal_run(root, "personamem", personamem_total)
    longmemeval_formal = find_formal_run(root, "longmemeval", longmemeval_total)
    any_evaluated_run = next((artifact for artifact in iter_run_artifacts(root) if artifact["evaluation"] is not None), None)

    core_memory_complete = all(
        item is not None
        for item in (
            MemoryWriter,
            CoreMemoryUpdater,
            ResidualMemoryManager,
            MemoryReader,
            VanillaCoreMemory,
        )
    )
    docs_complete = all((root / rel_path).exists() for rel_path in REQUIRED_DOCS)
    step_scripts_complete = all((root / rel_path).exists() for rel_path in REQUIRED_STEP_SCRIPTS)
    outputs_dirs_complete = all(
        (root / "outputs" / child).exists() for child in ("runs", "tables", "logs", "figures")
    )
    summary_table_exists = (root / "outputs" / "tables" / "stage1_summary.csv").exists()

    criteria = {
        "AC-001": {
            "passed": core_memory_complete and (root / "tests" / "test_core_memory_module.py").exists(),
            "evidence": {
                "core_memory_module": str(root / "src" / "core_mem" / "memory" / "core.py"),
                "core_memory_test": str(root / "tests" / "test_core_memory_module.py"),
            },
        },
        "AC-002": {
            "passed": personamem_formal is not None,
            "evidence": {
                "expected_total": personamem_total,
                "run_dir": str(personamem_formal["run_dir"]) if personamem_formal else None,
            },
        },
        "AC-003": {
            "passed": longmemeval_formal is not None,
            "evidence": {
                "expected_total": longmemeval_total,
                "run_dir": str(longmemeval_formal["run_dir"]) if longmemeval_formal else None,
            },
        },
        "AC-004": {
            "passed": summary_table_exists and any_evaluated_run is not None,
            "evidence": {
                "summary_table": str(root / "outputs" / "tables" / "stage1_summary.csv"),
                "sample_run_dir": str(any_evaluated_run["run_dir"]) if any_evaluated_run else None,
            },
        },
        "AC-005": {
            "passed": docs_complete and step_scripts_complete and outputs_dirs_complete and (root / "environment.yaml").exists() and conda_env_exists("core_mem"),
            "evidence": {
                "environment_yaml": str(root / "environment.yaml"),
                "conda_env_core_mem": conda_env_exists("core_mem"),
                "docs_complete": docs_complete,
                "step_scripts_complete": step_scripts_complete,
                "outputs_dirs_complete": outputs_dirs_complete,
            },
        },
        "AC-006": {
            "passed": pytest_passes(root) if run_pytest else False,
            "evidence": {
                "pytest_executed": run_pytest,
                "smoke_test": str(root / "tests" / "test_smoke_pipeline.py"),
            },
        },
        "AC-007": {
            "passed": traceability_complete(any_evaluated_run),
            "evidence": {
                "traceability_run_dir": str(any_evaluated_run["run_dir"]) if any_evaluated_run else None,
            },
        },
    }

    passed = sum(1 for payload in criteria.values() if payload["passed"])
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

    root = Path(args.root).resolve()
    payload = compute_acceptance_status(root, run_pytest=not args.skip_pytest)
    if args.score_only:
        print(payload["passed"])
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["passed"] == payload["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
