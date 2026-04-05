"""Top-level experiment entrypoint for the CoRe Memory stage-1 platform."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core_mem.benchmarks.longmemeval import LongMemEvalAdapter
from core_mem.benchmarks.personamem import PersonaMemAdapter
from core_mem.config import load_project_config
from core_mem.providers.openai_compatible import OpenAICompatibleConfig, OpenAICompatibleProvider


def run_stage1_verifier(repo_root: Path) -> int:
    verifier = repo_root / "scripts" / "verify_stage1_status.py"
    cmd = [sys.executable, str(verifier), "--score-only"]
    result = subprocess.run(cmd, cwd=repo_root, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(result.returncode)
    print(result.stdout.strip())
    return 0


def build_output_dir(outputs_root: Path, benchmark: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = outputs_root / "runs" / f"{stamp}_{benchmark}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def execute_records(
    *,
    benchmark: str,
    records: list,
    config_path: Path,
    outputs_root: Path,
    llm_config,
    dry_run: bool,
) -> int:
    run_dir = build_output_dir(outputs_root, benchmark)
    metadata = {
        "benchmark": benchmark,
        "record_count": len(records),
        "dry_run": dry_run,
        "config_path": str(config_path),
        "model": llm_config.model,
        "provider": llm_config.provider,
        "sample_ids": [record.sample_id for record in records[:5]],
    }
    if records:
        metadata["prompt_preview"] = records[0].prompt[:1200]
        metadata["expected_preview"] = records[0].expected_answer
    write_json(run_dir / "run_metadata.json", metadata)

    if dry_run:
        print(json.dumps({"run_dir": str(run_dir), **metadata}, ensure_ascii=False))
        return 0

    provider = OpenAICompatibleProvider(
        OpenAICompatibleConfig(
            api_key_env=llm_config.api_key_env,
            base_url=llm_config.base_url,
            model=llm_config.model,
            timeout_seconds=llm_config.timeout_seconds,
        )
    )
    predictions: list[dict[str, object]] = []
    for record in records:
        response = provider.chat(
            record.prompt,
            temperature=llm_config.temperature,
            max_tokens=llm_config.max_tokens,
        )
        predictions.append(
            {
                "sample_id": record.sample_id,
                "benchmark": record.benchmark,
                "prediction": response.content,
                "expected_answer": record.expected_answer,
                "metadata": record.metadata,
            }
        )
    write_jsonl(run_dir / "predictions.jsonl", predictions)
    print(json.dumps({"run_dir": str(run_dir), "prediction_count": len(predictions)}, ensure_ascii=False))
    return 0


def run_personamem(args: argparse.Namespace) -> int:
    config = load_project_config(args.config)
    adapter = PersonaMemAdapter(data_root=args.data_root or config.benchmarks.personamem.data_root)
    records = adapter.build_records(
        questions_path=args.questions_csv,
        contexts_path=args.contexts_jsonl,
        limit=args.limit,
    )
    return execute_records(
        benchmark="personamem",
        records=records,
        config_path=Path(args.config),
        outputs_root=REPO_ROOT / config.outputs.root,
        llm_config=config.llm,
        dry_run=args.dry_run,
    )


def run_longmemeval(args: argparse.Namespace) -> int:
    config = load_project_config(args.config)
    adapter = LongMemEvalAdapter(data_root=args.data_root or config.benchmarks.longmemeval.data_root)
    records = adapter.build_records(dataset_path=args.dataset_json, limit=args.limit)
    return execute_records(
        benchmark="longmemeval",
        records=records,
        config_path=Path(args.config),
        outputs_root=REPO_ROOT / config.outputs.root,
        llm_config=config.llm,
        dry_run=args.dry_run,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true")
    subparsers = parser.add_subparsers(dest="command")

    personamem_parser = subparsers.add_parser("personamem")
    personamem_parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "defaults.yaml"))
    personamem_parser.add_argument("--data-root")
    personamem_parser.add_argument("--questions-csv")
    personamem_parser.add_argument("--contexts-jsonl")
    personamem_parser.add_argument("--limit", type=int, default=1)
    personamem_parser.add_argument("--dry-run", action="store_true")

    longmemeval_parser = subparsers.add_parser("longmemeval")
    longmemeval_parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "defaults.yaml"))
    longmemeval_parser.add_argument("--data-root")
    longmemeval_parser.add_argument("--dataset-json")
    longmemeval_parser.add_argument("--limit", type=int, default=1)
    longmemeval_parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()
    if args.verify_only:
        return run_stage1_verifier(REPO_ROOT)
    if args.command == "personamem":
        return run_personamem(args)
    if args.command == "longmemeval":
        return run_longmemeval(args)

    print("Stage-1 experiment runner supports PersonaMem and LongMemEval dry runs.")
    print("Use --verify-only or a benchmark subcommand.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
