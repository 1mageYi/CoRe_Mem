"""Check whether stage-2 public dataset sources are present."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core_mem.v2.datasets import summarize_stage2_sources


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-config", default="configs/stage2_data_sources.json")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    summary = summarize_stage2_sources(Path(args.source_config))
    summary["source_config"] = args.source_config
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(f"enabled={len(summary['enabled'])}")
        print(f"available={len(summary['available'])}")
        print(f"missing={len(summary['missing'])}")
    return 0 if not summary["missing"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
