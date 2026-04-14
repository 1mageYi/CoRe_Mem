"""Check external readiness blockers for stage-1 runs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core_mem.config import load_project_config


def conda_env_exists(env_name: str) -> bool | None:
    result = subprocess.run(
        ["conda", "env", "list", "--json"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    payload = json.loads(result.stdout)
    env_paths = payload.get("envs", [])
    suffix = f"/{env_name}"
    return any(path.endswith(suffix) for path in env_paths)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "defaults.yaml"))
    args = parser.parse_args()

    config = load_project_config(args.config)
    personamem_questions = REPO_ROOT / config.benchmarks.personamem.data_root / "questions_32k.csv"
    personamem_contexts = REPO_ROOT / config.benchmarks.personamem.data_root / "shared_contexts_32k.jsonl"
    longmemeval_dataset = REPO_ROOT / config.benchmarks.longmemeval.data_root / "longmemeval_s_cleaned.json"

    checks = {
        "environment_yaml": (REPO_ROOT / "environment.yaml").exists(),
        "conda_env_core_mem": conda_env_exists("core_mem"),
        "llm_api_key": bool(os.getenv(config.llm.api_key_env)),
        "personamem_questions": personamem_questions.exists(),
        "personamem_contexts": personamem_contexts.exists(),
        "longmemeval_dataset": longmemeval_dataset.exists(),
    }
    blockers = [name for name, passed in checks.items() if passed is not True]
    payload = {
        "config_path": str(Path(args.config)),
        "checks": checks,
        "ready": not blockers,
        "blockers": blockers,
    }
    print(json.dumps(payload, ensure_ascii=False))
    return 0 if payload["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
