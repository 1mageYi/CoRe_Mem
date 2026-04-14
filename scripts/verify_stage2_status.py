"""Mechanical verifier for stage-2 readiness."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.verify_stage2_acceptance import compute_acceptance_status


CHECKS = {
    "scripts_package": "scripts/__init__.py",
    "stage2_acceptance_verifier": "scripts/verify_stage2_acceptance.py",
    "stage2_status_verifier": "scripts/verify_stage2_status.py",
    "stage2_acceptance_test": "tests/test_verify_stage2_acceptance.py",
    "stage2_status_test": "tests/test_verify_stage2_status.py",
    "stage2_v2_package": "src/core_mem/v2",
    "stage2_v2_init": "src/core_mem/v2/__init__.py",
    "stage2_v2_schemas": "src/core_mem/v2/schemas.py",
    "stage2_v2_parser": "src/core_mem/v2/parser.py",
    "stage2_v2_datasets": "src/core_mem/v2/datasets.py",
    "stage2_v2_encoder": "src/core_mem/v2/encoder.py",
    "stage2_v2_lifecycle": "src/core_mem/v2/lifecycle.py",
    "stage2_v2_consolidation": "src/core_mem/v2/consolidation.py",
    "stage2_v2_resampler": "src/core_mem/v2/resampler.py",
    "stage2_v2_decoder": "src/core_mem/v2/decoder.py",
    "stage2_v2_projection": "src/core_mem/v2/projection.py",
    "stage2_v2_system": "src/core_mem/v2/system.py",
    "stage2_v2_training": "src/core_mem/v2/training.py",
    "stage2_v2_public_data": "src/core_mem/v2/public_data.py",
    "stage2_v2_eval_local": "src/core_mem/v2/eval_local.py",
    "stage2_prepare_script": "scripts/prepare_stage2_data.py",
    "stage2_data_preflight_script": "scripts/stage2_data_preflight.py",
    "stage2_data_normalize_script": "scripts/normalize_stage2_public_data.py",
    "stage2_train_script": "scripts/train_stage2.py",
    "stage2_data_sources_config": "configs/stage2_data_sources.json",
    "stage2_train_tiny_config": "configs/stage2_train_tiny.yaml",
    "stage2_eval_script": "scripts/eval_stage2_local.py",
    "stage2_canary_script": "scripts/run_stage2_canary.py",
    "stage2_train_config": "configs/stage2_train.yaml",
    "stage2_model_test": "tests/test_stage2_model_skeleton.py",
    "stage2_training_runtime_test": "tests/test_stage2_training_runtime.py",
    "stage2_data_pipeline_test": "tests/test_stage2_data_pipeline.py",
    "stage2_public_data_test": "tests/test_stage2_public_data.py",
    "stage2_local_eval_test": "tests/test_stage2_local_eval.py",
    "v2_design_doc": "docs/v2_design.md",
    "stage2_local_eval_doc": "docs/stage2_local_evaluation.md",
    "requirements_doc": "docs/requirements.md",
    "architecture_milestones": ".agent-os/architecture-milestones.md",
    "acceptance_report": ".agent-os/acceptance-report.md",
    "outputs_v2_checkpoints": "outputs_v2/checkpoints/.gitkeep",
    "outputs_v2_runs": "outputs_v2/runs/.gitkeep",
    "outputs_v2_evals_local": "outputs_v2/evals_local/.gitkeep",
    "outputs_v2_evals_benchmark": "outputs_v2/evals_benchmark/.gitkeep",
    "outputs_v2_tables": "outputs_v2/tables/.gitkeep",
    "outputs_v2_figures": "outputs_v2/figures/.gitkeep",
    "outputs_v2_logs": "outputs_v2/logs/.gitkeep",
    "outputs_v2_cache_manifests": "outputs_v2/cache_manifests/.gitkeep",
    "outputs_v2_artifacts": "outputs_v2/artifacts/.gitkeep",
}


def compute_status(root: Path) -> tuple[int, dict[str, bool]]:
    checks = {name: (root / rel_path).exists() for name, rel_path in CHECKS.items()}
    acceptance = compute_acceptance_status(root, run_pytest=False)
    checks["stage2_docs_ready"] = acceptance["criteria"]["AC-008"]["passed"]
    checks["stage2_schema_ready"] = acceptance["criteria"]["AC-009"]["passed"]
    return sum(checks.values()), checks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--score-only", action="store_true")
    args = parser.parse_args()

    score, checks = compute_status(Path(args.root).resolve())
    if args.score_only:
        print(score)
        return 0

    print(f"stage2_readiness_score={score}")
    for name, passed in checks.items():
        print(f"{name}={'pass' if passed else 'fail'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
