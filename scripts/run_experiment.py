"""Top-level experiment entrypoint for the CoRe Memory stage-1 platform."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


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


def resolve_run_dir(outputs_root: Path, benchmark: str, requested_run_dir: str | None = None) -> Path:
    if requested_run_dir:
        run_dir = Path(requested_run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir
    return build_output_dir(outputs_root, benchmark)


def current_commit_hash(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return "unknown"
    return result.stdout.strip()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def copy_config_snapshot(config_path: Path, run_dir: Path) -> Path:
    snapshot_path = run_dir / "config_snapshot.yaml"
    snapshot_path.write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    return snapshot_path


def execute_records(
    *,
    benchmark: str,
    records: list,
    config_path: Path,
    outputs_root: Path,
    llm_config,
    dry_run: bool,
    requested_run_dir: str | None = None,
    resume: bool = False,
) -> int:
    run_dir = resolve_run_dir(outputs_root, benchmark, requested_run_dir=requested_run_dir)
    run_timestamp = run_dir.name.split("_", 1)[0]
    config_snapshot = copy_config_snapshot(config_path, run_dir)
    predictions_path = run_dir / "predictions.jsonl"
    existing_predictions = load_jsonl(predictions_path) if resume else []
    completed_ids = {str(row.get("sample_id", "")) for row in existing_predictions}
    metadata = {
        "benchmark": benchmark,
        "record_count": len(records),
        "dry_run": dry_run,
        "config_path": str(config_path),
        "config_snapshot_path": str(config_snapshot),
        "model": llm_config.model,
        "provider": llm_config.provider,
        "run_timestamp": run_timestamp,
        "commit_hash": current_commit_hash(REPO_ROOT),
        "sample_ids": [record.sample_id for record in records[:5]],
        "resumed_prediction_count": len(existing_predictions),
    }
    if records:
        metadata["prompt_preview"] = records[0].prompt[:1200]
        metadata["expected_preview"] = records[0].expected_answer
        metadata["prompt_version"] = records[0].metadata.get("prompt_version", "unknown")
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
            max_retries=getattr(llm_config, "max_retries", 0),
            retry_backoff_seconds=getattr(llm_config, "retry_backoff_seconds", 2.0),
            min_request_interval_seconds=getattr(llm_config, "min_request_interval_seconds", 0.0),
            max_retry_delay_seconds=getattr(llm_config, "max_retry_delay_seconds", 60.0),
        )
    )
    predictions: list[dict[str, object]] = list(existing_predictions)
    with predictions_path.open("a", encoding="utf-8") as handle:
        for record in records:
            if record.sample_id in completed_ids:
                continue
            response = provider.chat(
                record.prompt,
                temperature=llm_config.temperature,
                max_tokens=llm_config.max_tokens,
            )
            row = {
                "sample_id": record.sample_id,
                "benchmark": record.benchmark,
                "prediction": response.content,
                "expected_answer": record.expected_answer,
                "metadata": record.metadata,
            }
            predictions.append(row)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            if len(predictions) == len(records) or len(predictions) % 25 == 0:
                print(
                    json.dumps(
                        {
                            "run_dir": str(run_dir),
                            "completed_predictions": len(predictions),
                            "total_records": len(records),
                            "last_sample_id": record.sample_id,
                        },
                        ensure_ascii=False,
                    )
                )
    metadata["completed_predictions"] = len(predictions)
    write_json(run_dir / "run_metadata.json", metadata)
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
        requested_run_dir=args.run_dir,
        resume=args.resume,
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
        requested_run_dir=args.run_dir,
        resume=args.resume,
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
    personamem_parser.add_argument("--run-dir")
    personamem_parser.add_argument("--resume", action="store_true")
    personamem_parser.add_argument("--dry-run", action="store_true")

    longmemeval_parser = subparsers.add_parser("longmemeval")
    longmemeval_parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "defaults.yaml"))
    longmemeval_parser.add_argument("--data-root")
    longmemeval_parser.add_argument("--dataset-json")
    longmemeval_parser.add_argument("--limit", type=int, default=1)
    longmemeval_parser.add_argument("--run-dir")
    longmemeval_parser.add_argument("--resume", action="store_true")
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
