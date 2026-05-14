"""
Phase 3: run a small matrix of PERMA ablations and write a single JSON summary.

Default: all `graph_edge_mode` presets (hybrid merge), then optional merge-strategy row.
Supports multiple users via `--user-ids` (directories: `<out>/<user_id>/edge_<mode>/`).

Does not start the LLM by itself — requires valid `.env` like other eval scripts.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import cast


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from graph_mem.ablation_presets import GraphEdgeMode  # noqa: E402
from graph_mem.perma_eval import PermaEvalConfig, run_perma_eval  # noqa: E402


DEFAULT_EDGE_MODES: tuple[GraphEdgeMode, ...] = (
    "full",
    "semantic_only",
    "semantic_temporal",
    "semantic_co_usage",
)

# Suggested stratified mini-PERMA spread (6 of 10 users); override with --user-ids.
DEFAULT_MULTI_USER_IDS = (
    "user108",
    "user109",
    "user112",
    "user123",
    "user334",
    "user914",
)


def _aggregate_by_mode(runs: list[dict]) -> dict:
    by_mode: dict[str, list[dict]] = defaultdict(list)
    for r in runs:
        mode = r.get("graph_edge_mode")
        if mode is not None:
            by_mode[str(mode)].append(r)

    out: dict[str, dict] = {}
    for mode, lst in sorted(by_mode.items()):
        acc_g = sum(x["acc_graph_full"] for x in lst) / len(lst)
        acc_s = sum(x["acc_semantic_only"] for x in lst) / len(lst)
        dlt = sum(x["acc_delta_graph_minus_semantic"] for x in lst) / len(lst)
        out[mode] = {
            "n_users": len(lst),
            "mean_acc_graph_full": round(acc_g, 4),
            "mean_acc_semantic_only": round(acc_s, 4),
            "mean_acc_delta_graph_minus_semantic": round(dlt, 4),
            "per_user": [
                {
                    "user_id": x["user_id"],
                    "acc_graph_full": x["acc_graph_full"],
                    "acc_semantic_only": x["acc_semantic_only"],
                    "acc_delta_graph_minus_semantic": x["acc_delta_graph_minus_semantic"],
                    "n_samples": x["n_samples"],
                }
                for x in lst
            ],
        }
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--user-id", default="user108", help="Used when --user-ids is omitted.")
    p.add_argument(
        "--user-ids",
        default=None,
        help="Comma-separated user_ids. If set, runs each mode for each user (e.g. 6 users × 4 modes).",
    )
    p.add_argument(
        "--use-default-six-users",
        action="store_true",
        help=f"Shortcut: use {len(DEFAULT_MULTI_USER_IDS)} preset user ids (see script constant).",
    )
    p.add_argument("--variant", default="c", choices=["c", "n", "s"])
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--lambda-core", type=float, default=None, dest="lambda_core")
    p.add_argument(
        "--modes",
        default=",".join(DEFAULT_EDGE_MODES),
        help="Comma-separated graph_edge_mode values.",
    )
    p.add_argument(
        "--also-merge-ablation",
        action="store_true",
        help="After edge sweep, run merge_strategy=latest_wins with graph_edge_mode=full.",
    )
    p.add_argument("--temporal-probe", action="store_true", dest="temporal_probe")
    args = p.parse_args()

    if args.use_default_six_users:
        users = list(DEFAULT_MULTI_USER_IDS)
    elif args.user_ids:
        users = [u.strip() for u in args.user_ids.split(",") if u.strip()]
    else:
        users = [args.user_id]

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = ROOT / "outputs" / "perma_eval" / f"phase3_ablation_{ts}"
    out_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    allowed: set[str] = set(DEFAULT_EDGE_MODES)

    for user in users:
        base = PermaEvalConfig(
            user_id=user,
            variant=args.variant,
            limit=args.limit,
            lambda_core=args.lambda_core,
            write_debug=False,
            temporal_probe_thirds=args.temporal_probe,
        )

        for mode in modes:
            if mode not in allowed:
                raise SystemExit(f"Unknown graph_edge_mode: {mode!r}. Expected one of {sorted(allowed)}")
            cfg = replace(
                base,
                graph_edge_mode=cast(GraphEdgeMode, mode),
                merge_strategy="hybrid",
            )
            sub = out_root / user / f"edge_{mode}"
            summary = run_perma_eval(cfg, root_dir=ROOT, out_dir=sub, show_progress=False)
            summary["artifact_dir"] = str(sub)
            rows.append(summary)
            print(
                json.dumps(
                    {"user_id": user, "graph_edge_mode": mode, **{k: summary[k] for k in summary if k != "artifact_dir"}},
                    ensure_ascii=False,
                )
            )

        if args.also_merge_ablation:
            cfg = replace(base, graph_edge_mode="full", merge_strategy="latest_wins")
            sub = out_root / user / "merge_latest_wins_full"
            summary = run_perma_eval(cfg, root_dir=ROOT, out_dir=sub, show_progress=False)
            summary["artifact_dir"] = str(sub)
            rows.append(summary)
            print(json.dumps({"user_id": user, "merge_ablation": "latest_wins", **summary}, ensure_ascii=False))

    edge_only_runs = [r for r in rows if r.get("merge_strategy") == "hybrid" and r.get("graph_edge_mode")]
    manifest = {
        "created_at": ts,
        "user_ids": users,
        "n_users": len(users),
        "variant": args.variant,
        "limit_per_user": args.limit,
        "modes": modes,
        "runs": rows,
        "aggregate_by_graph_edge_mode": _aggregate_by_mode(edge_only_runs),
    }
    manifest_path = out_root / "phase3_ablation_summary.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nsaved: {manifest_path}")
    if len(users) > 1 and edge_only_runs:
        print("\n=== mean delta (graph - semantic) by graph_edge_mode ===")
        agg = manifest["aggregate_by_graph_edge_mode"]
        for mode in modes:
            if mode in agg:
                print(f"  {mode}: {agg[mode]['mean_acc_delta_graph_minus_semantic']}")


if __name__ == "__main__":
    main()
