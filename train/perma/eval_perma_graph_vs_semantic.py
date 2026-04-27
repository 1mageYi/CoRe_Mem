from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from graph_mem.perma_eval import PermaEvalConfig, run_perma_eval_with_timestamped_output


def run(args: argparse.Namespace) -> None:
    cfg = PermaEvalConfig(user_id=args.user_id, variant=args.variant, limit=args.limit)
    summary, out_dir = run_perma_eval_with_timestamped_output(cfg, root_dir=ROOT, show_progress=True)

    print("\n=== PERMA Graph vs Semantic ===")
    print(json.dumps(summary, indent=2))
    print(f"saved: {out_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--user-id", default="user108")
    p.add_argument("--variant", default="c", choices=["c", "n", "s"])
    p.add_argument("--limit", type=int, default=20)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
