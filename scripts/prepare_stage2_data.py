"""Prepare stage-2 task manifests and tiny demo samples."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core_mem.v2 import STAGE2_TASK_DATASETS, STAGE2_TASK_MODULES, build_demo_prepared_payload, build_task_registry
from core_mem.v2.datasets import (
    STAGE2_DATA_POOLS,
    build_prepared_payload_from_sources,
    build_source_record_split_from_sources,
    summarize_stage2_sources,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def _materialize_prepared_payload(
    output_root: Path,
    *,
    prepared_payload: dict[str, list[dict]],
    source_config: Path | None,
    source_summary: dict[str, object] | None,
    max_rows_per_dataset: int | None,
    mode: str,
    namespace: str | None = None,
    extra_manifest: dict[str, object] | None = None,
) -> dict[str, object]:
    registry_entries = [entry.to_dict() for entry in build_task_registry()]
    artifact_root = output_root / "artifacts"
    cache_root = output_root / "cache_manifests"
    if namespace:
        artifact_root = artifact_root / namespace
        cache_root = cache_root / namespace

    cache_manifest_path = cache_root / "stage2_dataset_registry.json"
    task_registry_path = artifact_root / "stage2_task_registry.json"
    prepared_manifest_path = artifact_root / "stage2_prepared_samples_manifest.json"

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
        task_path = artifact_root / f"{task_name}.jsonl"
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
    if extra_manifest:
        manifest.update(extra_manifest)
    _write_json(prepared_manifest_path, manifest)
    return {
        "prepared_manifest": str(prepared_manifest_path),
        "task_registry": str(task_registry_path),
        "dataset_registry": str(cache_manifest_path),
        "task_counts": task_counts,
    }


def prepare_stage2_data(
    output_root: Path,
    *,
    source_config: Path | None = None,
    strict_sources: bool = False,
    max_rows_per_dataset: int | None = None,
) -> dict[str, object]:
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
    return _materialize_prepared_payload(
        output_root,
        prepared_payload=prepared_payload,
        source_config=source_config,
        source_summary=source_summary,
        max_rows_per_dataset=max_rows_per_dataset,
        mode=mode,
    )


def _selection_map(rows: list[dict[str, object]]) -> dict[str, set[str]]:
    selection: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        selection[str(row["dataset_key"])].add(str(row["sample_id"]))
    return dict(selection)


def _top_counter(rows: list[dict[str, object]], field: str, *, limit: int = 10) -> list[dict[str, object]]:
    counts = Counter(str(row.get(field, "unknown")) for row in rows)
    return [{"value": value, "count": count} for value, count in counts.most_common(limit)]


def prepare_v27_source_split(
    output_root: Path,
    *,
    source_config: Path,
    train_source_rows: int = 24000,
    val_source_rows: int = 4000,
    test_source_rows: int = 4000,
    split_seed: int = 27,
) -> dict[str, object]:
    source_summary = summarize_stage2_sources(source_config)
    split_payload = build_source_record_split_from_sources(
        source_config,
        split_counts={
            "train": train_source_rows,
            "val": val_source_rows,
            "test": test_source_rows,
        },
        split_seed=split_seed,
    )

    split_root = output_root / "artifacts" / "stage2_v27_32k"
    split_root.mkdir(parents=True, exist_ok=True)
    split_manifests: dict[str, dict[str, object]] = {}
    audit_splits: dict[str, dict[str, object]] = {}
    for split_name, rows in split_payload["split_rows"].items():
        selection = _selection_map(rows)
        prepared_payload = build_prepared_payload_from_sources(
            source_config,
            selected_sample_ids=selection,
        )
        source_rows_path = split_root / f"{split_name}_source_rows.jsonl"
        _write_jsonl(source_rows_path, rows)
        manifest_payload = _materialize_prepared_payload(
            output_root,
            prepared_payload=prepared_payload,
            source_config=source_config,
            source_summary=source_summary,
            max_rows_per_dataset=None,
            mode="source_level_split",
            namespace=f"stage2_v27_32k/{split_name}",
            extra_manifest={
                "source_split": split_name,
                "source_row_count": len(rows),
                "source_rows_path": str(source_rows_path),
                "split_seed": split_seed,
            },
        )
        split_manifests[split_name] = manifest_payload
        audit_splits[split_name] = {
            "source_row_count": len(rows),
            "task_counts": manifest_payload["task_counts"],
            "dataset_counts": _top_counter(rows, "dataset_key", limit=16),
            "relation_counts": _top_counter(rows, "relation", limit=16),
            "parser_bucket_counts": _top_counter(rows, "parser_bucket", limit=16),
            "lifecycle_source_rows": sum(1 for row in rows if bool(row.get("has_lifecycle"))),
        }

    latest_split_path = output_root / "artifacts" / "latest_stage2_v27_32k_split.json"
    latest_manifest_path = output_root / "artifacts" / "latest_stage2_v27_32k_manifest.json"
    latest_audit_path = output_root / "artifacts" / "latest_stage2_v27_32k_audit.json"

    _write_json(
        latest_split_path,
        {
            "artifact_type": "stage2_v27_32k_source_split",
            "source_config": str(source_config),
            "split_seed": split_seed,
            "requested_counts": split_payload["requested_counts"],
            "available_source_rows": split_payload["available_source_rows"],
            "available_by_dataset": split_payload["available_by_dataset"],
            "split_counts": {name: len(rows) for name, rows in split_payload["split_rows"].items()},
            "split_rows_paths": {
                split_name: str(split_root / f"{split_name}_source_rows.jsonl")
                for split_name in split_payload["split_rows"]
            },
        },
    )
    _write_json(
        latest_manifest_path,
        {
            "artifact_type": "stage2_v27_32k_manifest",
            "source_config": str(source_config),
            "split_seed": split_seed,
            "requested_counts": split_payload["requested_counts"],
            "splits": split_manifests,
        },
    )
    _write_json(
        latest_audit_path,
        {
            "artifact_type": "stage2_v27_32k_audit",
            "source_config": str(source_config),
            "split_seed": split_seed,
            "requested_counts": split_payload["requested_counts"],
            "available_source_rows": split_payload["available_source_rows"],
            "available_by_dataset": split_payload["available_by_dataset"],
            "splits": audit_splits,
        },
    )
    return {
        "v27_split_artifact": str(latest_split_path),
        "v27_manifest_artifact": str(latest_manifest_path),
        "v27_audit_artifact": str(latest_audit_path),
        "splits": {
            split_name: {
                "source_rows_path": str(split_root / f"{split_name}_source_rows.jsonl"),
                **manifest_payload,
            }
            for split_name, manifest_payload in split_manifests.items()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="outputs_v2")
    parser.add_argument("--source-config")
    parser.add_argument("--strict-sources", action="store_true")
    parser.add_argument("--max-rows-per-dataset", type=int)
    parser.add_argument("--prepare-v27-32k", action="store_true")
    parser.add_argument("--v27-train-source-rows", type=int, default=24000)
    parser.add_argument("--v27-val-source-rows", type=int, default=4000)
    parser.add_argument("--v27-test-source-rows", type=int, default=4000)
    parser.add_argument("--v27-split-seed", type=int, default=27)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.prepare_v27_32k:
        if not args.source_config:
            raise ValueError("--prepare-v27-32k requires --source-config")
        payload = prepare_v27_source_split(
            Path(args.output_root),
            source_config=Path(args.source_config),
            train_source_rows=args.v27_train_source_rows,
            val_source_rows=args.v27_val_source_rows,
            test_source_rows=args.v27_test_source_rows,
            split_seed=args.v27_split_seed,
        )
    else:
        payload = prepare_stage2_data(
            Path(args.output_root),
            source_config=Path(args.source_config) if args.source_config else None,
            strict_sources=args.strict_sources,
            max_rows_per_dataset=args.max_rows_per_dataset,
        )
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        if args.prepare_v27_32k:
            print(f"v27_split_artifact={payload['v27_split_artifact']}")
            print(f"v27_manifest_artifact={payload['v27_manifest_artifact']}")
            print(f"v27_audit_artifact={payload['v27_audit_artifact']}")
        else:
            print(f"prepared_manifest={payload['prepared_manifest']}")
            print(f"task_registry={payload['task_registry']}")
            print(f"dataset_registry={payload['dataset_registry']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
