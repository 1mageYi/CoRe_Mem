"""Validate benchmark inputs for the staged prepare step."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core_mem.config import load_project_config


def resolve_personamem_paths(args: argparse.Namespace, config_path: Path) -> dict[str, str]:
    config = load_project_config(str(config_path))
    data_root = Path(args.data_root or config.benchmarks.personamem.data_root)
    questions = Path(args.questions_csv or data_root / "questions_32k.csv")
    contexts = Path(args.contexts_jsonl or data_root / "shared_contexts_32k.jsonl")
    return {
        "questions_csv": str(questions),
        "contexts_jsonl": str(contexts),
    }


def resolve_longmemeval_paths(args: argparse.Namespace, config_path: Path) -> dict[str, str]:
    config = load_project_config(str(config_path))
    data_root = Path(args.data_root or config.benchmarks.longmemeval.data_root)
    dataset = Path(args.dataset_json or data_root / "longmemeval_s_cleaned.json")
    return {"dataset_json": str(dataset)}


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    personamem_parser = subparsers.add_parser("personamem")
    personamem_parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "defaults.yaml"))
    personamem_parser.add_argument("--data-root")
    personamem_parser.add_argument("--questions-csv")
    personamem_parser.add_argument("--contexts-jsonl")

    longmemeval_parser = subparsers.add_parser("longmemeval")
    longmemeval_parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "defaults.yaml"))
    longmemeval_parser.add_argument("--data-root")
    longmemeval_parser.add_argument("--dataset-json")

    args = parser.parse_args()
    config_path = Path(args.config)

    if args.command == "personamem":
        paths = resolve_personamem_paths(args, config_path)
    else:
        paths = resolve_longmemeval_paths(args, config_path)

    status = {name: Path(raw_path).exists() for name, raw_path in paths.items()}
    payload = {
        "benchmark": args.command,
        "config_path": str(config_path),
        "paths": paths,
        "status": status,
        "ready": all(status.values()),
    }
    print(json.dumps(payload, ensure_ascii=False))
    return 0 if payload["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
