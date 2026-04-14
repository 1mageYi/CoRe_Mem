"""Mechanical verifier for stage-1 platform readiness."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.verify_stage1_acceptance import compute_acceptance_status


CHECKS = {
    "environment_spec": "environment.yaml",
    "default_config": "configs/defaults.yaml",
    "scripts_package": "scripts/__init__.py",
    "prepare_step_script": "scripts/prepare_data.py",
    "evaluate_step_script": "scripts/evaluate_run.py",
    "make_tables_script": "scripts/make_tables.py",
    "stage1_preflight_script": "scripts/stage1_preflight.py",
    "config_loader": "src/core_mem/config.py",
    "embedding_interface": "src/core_mem/embeddings/base.py",
    "embedding_adapter": "src/core_mem/embeddings/sentence_transformer.py",
    "provider_adapter": "src/core_mem/providers/openai_compatible.py",
    "common_benchmark_schema": "src/core_mem/benchmarks/common.py",
    "personamem_adapter": "src/core_mem/benchmarks/personamem.py",
    "longmemeval_adapter": "src/core_mem/benchmarks/longmemeval.py",
    "core_memory_module": "src/core_mem/memory/core.py",
    "run_entrypoint": "scripts/run_experiment.py",
    "stage1_verifier": "scripts/verify_stage1_status.py",
    "unit_test": "tests/test_verify_stage1_status.py",
    "smoke_test": "tests/test_smoke_pipeline.py",
    "adapter_test": "tests/test_benchmark_adapters.py",
    "provider_test": "tests/test_provider_adapter.py",
    "step_scripts_test": "tests/test_step_scripts.py",
    "core_memory_test": "tests/test_core_memory_module.py",
    "traceability_test": "tests/test_run_metadata_traceability.py",
    "run_resume_support_test": "tests/test_run_resume_support.py",
    "embedding_pipeline_test": "tests/test_embedding_pipeline.py",
    "personamem_protocol_test": "tests/test_personamem_protocol_alignment.py",
    "longmemeval_protocol_test": "tests/test_longmemeval_protocol_alignment.py",
    "acceptance_verifier": "scripts/verify_stage1_acceptance.py",
    "acceptance_verifier_test": "tests/test_verify_stage1_acceptance.py",
    "fixture_personamem_questions": "tests/fixtures/personamem/questions_32k.csv",
    "fixture_longmemeval": "tests/fixtures/longmemeval/longmemeval_s_cleaned.json",
}


def compute_status(root: Path) -> tuple[int, dict[str, bool]]:
    checks = {name: (root / rel_path).exists() for name, rel_path in CHECKS.items()}
    acceptance = compute_acceptance_status(root, run_pytest=False)
    run_entrypoint_text = (root / "scripts" / "run_experiment.py").read_text(encoding="utf-8")
    checks["summary_table"] = (root / "outputs" / "tables" / "stage1_summary.csv").exists()
    checks["resumable_runner_support"] = "--run-dir" in run_entrypoint_text and "--resume" in run_entrypoint_text
    checks["core_mem_named_env"] = bool(acceptance["criteria"]["AC-005"]["evidence"]["conda_env_core_mem"])
    checks["outputs_dir_structure"] = bool(acceptance["criteria"]["AC-005"]["evidence"]["outputs_dirs_complete"])
    checks["formal_personamem_result"] = acceptance["criteria"]["AC-002"]["passed"]
    checks["formal_longmemeval_result"] = acceptance["criteria"]["AC-003"]["passed"]
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
