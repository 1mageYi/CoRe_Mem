"""Normalize repo-local public datasets into stage-2 prepared-source JSONL files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core_mem.v2.public_data import (
    normalize_mquake_dataset,
    normalize_multiwoz24_dataset,
    normalize_personachat_dataset,
    normalize_recoe_dataset,
    normalize_sgd_dataset,
)


def _default_specs(source_root: Path) -> dict[str, dict[str, Path]]:
    return {
        "sgd": {
            "raw": source_root / "sgd" / "raw" / "dstc8-schema-guided-dialogue",
            "normalized": source_root / "sgd" / "normalized.jsonl",
        },
        "multiwoz24": {
            "raw": source_root / "multiwoz24" / "raw" / "MultiWOZ2.4" / "data" / "MULTIWOZ2.4.zip",
            "normalized": source_root / "multiwoz24" / "normalized.jsonl",
        },
        "personachat": {
            "raw": source_root / "personachat" / "raw" / "convai2_fix_723.tgz",
            "normalized": source_root / "personachat" / "normalized.jsonl",
        },
        "mquake": {
            "raw": source_root / "mquake" / "raw" / "MQuAKE",
            "normalized": source_root / "mquake" / "normalized.jsonl",
        },
        "recoe": {
            "raw": source_root / "recoe" / "raw" / "ReCoE",
            "normalized": source_root / "recoe" / "normalized.jsonl",
        },
    }


def normalize_public_datasets(
    *,
    source_root: Path,
    datasets: list[str] | None,
    max_rows_per_dataset: int | None,
) -> dict[str, dict[str, object]]:
    specs = _default_specs(source_root)
    selected = datasets or list(specs)
    results: dict[str, dict[str, object]] = {}
    for dataset in selected:
        spec = specs[dataset]
        raw_path = spec["raw"]
        output_path = spec["normalized"]
        if dataset == "sgd":
            result = normalize_sgd_dataset(raw_path, output_path, max_rows=max_rows_per_dataset)
        elif dataset == "multiwoz24":
            result = normalize_multiwoz24_dataset(raw_path, output_path, max_rows=max_rows_per_dataset)
        elif dataset == "personachat":
            result = normalize_personachat_dataset(raw_path, output_path, max_rows=max_rows_per_dataset)
        elif dataset == "mquake":
            result = normalize_mquake_dataset(raw_path, output_path, max_rows=max_rows_per_dataset)
        elif dataset == "recoe":
            result = normalize_recoe_dataset(raw_path, output_path, max_rows=max_rows_per_dataset)
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")
        results[dataset] = {
            **result,
            "raw_path": str(raw_path),
        }
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default="data/stage2_public")
    parser.add_argument(
        "--dataset",
        action="append",
        choices=["sgd", "multiwoz24", "personachat", "mquake", "recoe"],
        help="Repeat to normalize a subset; defaults to all five datasets.",
    )
    parser.add_argument("--max-rows-per-dataset", type=int)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = normalize_public_datasets(
        source_root=Path(args.source_root),
        datasets=args.dataset,
        max_rows_per_dataset=args.max_rows_per_dataset,
    )
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        for dataset, result in payload.items():
            print(f"{dataset}: rows={result['rows']} output={result['output_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
