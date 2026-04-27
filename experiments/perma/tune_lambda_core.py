"""
Grid search over `RankingConfig.lambda_core` on PERMA (single user, fixed limit).

Includes λ=0 as "no core prior" vs positive λ. Writes one summary JSON under
`outputs/perma_eval/lambda_grid_<timestamp>/`.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from graph_mem.perma_eval import PermaEvalConfig, run_perma_eval  # noqa: E402


def parse_lambdas(s: str) -> list[float]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return [float(x) for x in parts]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--user-id", default="user108")
    p.add_argument("--variant", default="c", choices=["c", "n", "s"])
    p.add_argument("--limit", type=int, default=10)
    p.add_argument(
        "--lambdas",
        default="0.0,0.1,0.2,0.3",
        help="Comma-separated lambda_core values (0 = no core prior multiplier).",
    )
    p.add_argument(
        "--no-debug",
        action="store_true",
        help="Skip per-sample debug JSONL for faster sweeps.",
    )
    args = p.parse_args()

    lambdas = parse_lambdas(args.lambdas)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = ROOT / "outputs" / "perma_eval" / f"lambda_grid_{ts}"
    out_root.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    for lam in lambdas:
        sub = out_root / f"lambda_{lam:.4f}".replace(".", "_")
        cfg = PermaEvalConfig(
            user_id=args.user_id,
            variant=args.variant,
            limit=args.limit,
            lambda_core=lam,
            write_debug=not args.no_debug,
        )
        summary = run_perma_eval(cfg, root_dir=ROOT, out_dir=sub, show_progress=False)
        summary["out_dir"] = str(sub)
        results.append(summary)
        print(json.dumps(summary, ensure_ascii=False))

    grid_path = out_root / "lambda_grid_summary.json"
    meta = {
        "user_id": args.user_id,
        "variant": args.variant,
        "limit": args.limit,
        "lambdas": lambdas,
        "runs": results,
    }
    grid_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nsaved grid summary: {grid_path}")


if __name__ == "__main__":
    main()
