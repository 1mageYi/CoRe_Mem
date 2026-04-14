"""Prepare stage-2 task manifests and tiny demo samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core_mem.v2 import STAGE2_TASK_DATASETS, STAGE2_TASK_MODULES, build_demo_prepared_payload, build_task_registry
from core_mem.v2.datasets import STAGE2_DATA_POOLS, build_prepared_payload_from_sources, summarize_stage2_sources


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def prepare_stage2_data(
    output_root: Path,
    *,
    source_config: Path | None = None,
    strict_sources: bool = False,
    max_rows_per_dataset: int | None = None,
) -> dict[str, object]:
    registry_entries = [entry.to_dict() for entry in build_task_registry()]
    source_summary = None
    if source_config is not None:
        source_summary = summarize_stage2_sources(source_config)
        prepared_payload = build_prepared_payload_from_sources(
            source_config,
            max_rows_per_dataset=max_rows_per_dataset,
        )
    else:
        prepared_payload = {}
    if not any(prepared_payload.values()):
        if strict_sources:
            raise FileNotFoundError("No stage-2 source datasets were available under the configured paths.")
        prepared_payload = build_demo_prepared_payload()
        mode = "demo_fallback"
    else:
        mode = "configured_sources"

    cache_manifest_path = output_root / "cache_manifests" / "stage2_dataset_registry.json"
    task_registry_path = output_root / "artifacts" / "stage2_task_registry.json"
    prepared_manifest_path = output_root / "artifacts" / "stage2_prepared_samples_manifest.json"

    _write_json(
        cache_manifest_path,
        {
            "data_pools": STAGE2_DATA_POOLS,
            "task_datasets": STAGE2_TASK_DATASETS,
            "task_modules": STAGE2_TASK_MODULES,
            "source_config": str(source_config) if source_config is not None else None,
            "source_summary": source_summary,
            "max_rows_per_dataset": max_rows_per_dataset,
        },
    )
    _write_json(task_registry_path, {"tasks": registry_entries})

    task_files: dict[str, str] = {}
    task_counts: dict[str, int] = {}
    for task_name, rows in prepared_payload.items():
        task_path = output_root / "artifacts" / f"{task_name}.jsonl"
        _write_jsonl(task_path, rows)
        task_files[task_name] = str(task_path)
        task_counts[task_name] = len(rows)

    manifest = {
        "prepared_at": "2026-04-07T00:00:00Z",
        "mode": mode,
        "task_files": task_files,
        "task_counts": task_counts,
        "task_registry": str(task_registry_path),
        "dataset_registry": str(cache_manifest_path),
        "source_config": str(source_config) if source_config is not None else None,
        "source_summary": source_summary,
        "max_rows_per_dataset": max_rows_per_dataset,
    }
    _write_json(prepared_manifest_path, manifest)
    return {
        "prepared_manifest": str(prepared_manifest_path),
        "task_registry": str(task_registry_path),
        "dataset_registry": str(cache_manifest_path),
        "task_counts": task_counts,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="outputs_v2")
    parser.add_argument("--source-config")
    parser.add_argument("--strict-sources", action="store_true")
    parser.add_argument("--max-rows-per-dataset", type=int)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = prepare_stage2_data(
        Path(args.output_root),
        source_config=Path(args.source_config) if args.source_config else None,
        strict_sources=args.strict_sources,
        max_rows_per_dataset=args.max_rows_per_dataset,
    )
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"prepared_manifest={payload['prepared_manifest']}")
        print(f"task_registry={payload['task_registry']}")
        print(f"dataset_registry={payload['dataset_registry']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
