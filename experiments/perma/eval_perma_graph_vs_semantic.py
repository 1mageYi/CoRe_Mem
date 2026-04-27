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
    cfg = PermaEvalConfig(
        user_id=args.user_id,
        variant=args.variant,
        limit=args.limit,
        lambda_core=args.lambda_core,
        write_debug=not args.no_debug,
        graph_edge_mode=args.graph_edge_mode,
        merge_strategy=args.merge_strategy,
        co_usage_decay=args.co_usage_decay,
        co_usage_min_count=args.co_usage_min_count,
        co_usage_prune_threshold=args.co_usage_prune,
        expand_min_co_usage_usage=args.expand_min_co_usage,
        temporal_probe_thirds=args.temporal_probe,
    )
    summary, out_dir = run_perma_eval_with_timestamped_output(cfg, root_dir=ROOT, show_progress=True)

    print("\n=== PERMA Graph vs Semantic ===")
    print(json.dumps(summary, indent=2))
    print(f"saved: {out_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--user-id", default="user108")
    p.add_argument("--variant", default="c", choices=["c", "n", "s"])
    p.add_argument("--limit", type=int, default=20)
    p.add_argument(
        "--lambda-core",
        type=float,
        default=None,
        dest="lambda_core",
        help="Override RankingConfig.lambda_core (omit for default).",
    )
    p.add_argument("--no-debug", action="store_true", help="Skip debug_samples.jsonl export.")
    p.add_argument(
        "--graph-edge-mode",
        default="full",
        choices=["full", "semantic_only", "semantic_temporal", "semantic_co_usage"],
    )
    p.add_argument(
        "--merge-strategy",
        default="hybrid",
        choices=["hybrid", "latest_wins"],
    )
    p.add_argument("--co-usage-decay", type=float, default=1.0, dest="co_usage_decay")
    p.add_argument(
        "--co-usage-min-count",
        type=int,
        default=1,
        help="Joint co-occurrence events before solidifying co_usage edges.",
        dest="co_usage_min_count",
    )
    p.add_argument(
        "--co-usage-prune",
        type=float,
        default=0.05,
        dest="co_usage_prune",
        help="Drop co_usage edges weaker than this after decay.",
    )
    p.add_argument(
        "--expand-min-co-usage",
        type=int,
        default=1,
        dest="expand_min_co_usage",
        help="Traverse co_usage neighbors only if usage_count >= this.",
    )
    p.add_argument(
        "--temporal-probe",
        action="store_true",
        help="Add early/mid/late slice accuracies (by sample order).",
        dest="temporal_probe",
    )
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
