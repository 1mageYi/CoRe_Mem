"""Quick failure analysis for LoCoMo eval debug output.

Usage (from CoRe_Mem/ root):

    # Latest run:
    uv run python experiments/locomo/analyze_results.py

    # Specific run:
    uv run python experiments/locomo/analyze_results.py --run outputs/locomo_eval/20260427_031615

    # Show N failing samples per category (default 3):
    uv run python experiments/locomo/analyze_results.py --show-n 5

    # Only show graph failures where semantic succeeded (retrieval gap):
    uv run python experiments/locomo/analyze_results.py --mode gap

    # Check whether gold evidence node was recalled at all:
    uv run python experiments/locomo/analyze_results.py --mode recall
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------

def _latest_run(base: Path) -> Path:
    runs = sorted(base.glob("*"), key=lambda p: p.name, reverse=True)
    if not runs:
        raise FileNotFoundError(f"No runs found in {base}")
    return runs[0]


def _load(run_dir: Path) -> tuple[list[dict], dict]:
    dbg_path = run_dir / "debug_samples.jsonl"
    sum_path = run_dir / "summary.json"
    rows: list[dict] = []
    with dbg_path.open(encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    summary = json.loads(sum_path.read_text(encoding="utf-8"))
    return rows, summary


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def _node_id_from_ref(ref: str) -> str | None:
    """LoCoMo evidence refs look like 'D2:1' meaning dialog 2 turn 1.
    We can't directly match to internal node IDs, so we check if the
    evidence text appears in the retrieved evidence instead."""
    return None  # raw refs are dialog-level, not node IDs


def _evidence_contains_hint(evidence_list: list[str], gold: str) -> bool:
    """Rough check: does any evidence snippet contain key words from the gold answer?"""
    gold_lower = gold.lower()
    gold_words = set(gold_lower.split())
    for ev in evidence_list:
        ev_lower = ev.lower()
        # Check if 3+ gold words appear in the evidence snippet
        matches = sum(1 for w in gold_words if len(w) > 2 and w in ev_lower)
        if matches >= min(3, max(1, len(gold_words) // 2)):
            return True
    return False


def _print_sample(row: dict, idx: int, *, show_evidence: int = 2) -> None:
    gd = row.get("graph_debug", {})
    print(f"  [{idx}] cat={row['category']} | ok_graph={row['ok_graph']} ok_sem={row['ok_semantic']}"
          + (f" ok_ctx={row.get('ok_full_ctx','?')}" if 'ok_full_ctx' in row else ""))
    print(f"       Q : {row['question']}")
    print(f"       Gold: {row['gold']}")
    print(f"       Graph pred : {row['pred_graph']}")
    print(f"       Sem pred   : {row['pred_semantic']}")
    if 'pred_full_ctx' in row:
        print(f"       FullCtx pred: {row['pred_full_ctx']}")
    ev_ref = row.get("evidence_refs", [])
    if ev_ref:
        print(f"       Gold evidence refs: {ev_ref}")
    skip = gd.get("skipped_expand")
    n_seeds = len(gd.get("seeds", []))
    n_exp = len(gd.get("expanded", []))
    print(f"       Graph: {n_seeds} seeds -> {n_exp} expanded | skipped_expand={skip}")

    gold_in_graph = _evidence_contains_hint(row.get("graph_top_evidence", []), row["gold"])
    gold_in_sem = _evidence_contains_hint(row.get("semantic_top_evidence", []), row["gold"])
    print(f"       Gold hint in graph evidence: {gold_in_graph} | in semantic evidence: {gold_in_sem}")

    if show_evidence > 0:
        print(f"       Top graph evidence ({show_evidence}):")
        for ev in row.get("graph_top_evidence", [])[:show_evidence]:
            print(f"         > {ev[:150]}")
        print(f"       Top semantic evidence ({show_evidence}):")
        for ev in row.get("semantic_top_evidence", [])[:show_evidence]:
            print(f"         > {ev[:150]}")
    print()


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def mode_overview(rows: list[dict], summary: dict) -> None:
    """Print summary + per-category breakdown."""
    print("=" * 72)
    print(f"Run summary  n={summary['n_samples']}  judge={summary['judge_method']}")
    print(f"  graph-full   : {summary['acc_graph_full']:.1%}")
    print(f"  semantic-only: {summary['acc_semantic_only']:.1%}")
    if summary.get("acc_full_context") is not None:
        print(f"  full-context : {summary['acc_full_context']:.1%}")
    print(f"  delta graph-sem: {summary['acc_delta_graph_minus_semantic']:+.1%}")
    print()
    print("Per-category:")
    for cat, s in summary["per_category"].items():
        parts = (f"graph={s['acc_graph']:.1%}  sem={s['acc_semantic']:.1%}"
                 + (f"  ctx={s.get('acc_full_context', '?'):.1%}" if 'acc_full_context' in s else ""))
        print(f"  cat {cat}  n={s['n']:3d}  {parts}")
    print()

    # Error pattern tallies
    n_all_wrong = sum(1 for r in rows if not r["ok_graph"] and not r["ok_semantic"])
    n_graph_only_wrong = sum(1 for r in rows if not r["ok_graph"] and r["ok_semantic"])
    n_sem_only_wrong = sum(1 for r in rows if r["ok_graph"] and not r["ok_semantic"])
    n_all_correct = sum(1 for r in rows if r["ok_graph"] and r["ok_semantic"])
    print("Error patterns:")
    print(f"  both correct         : {n_all_correct:3d}  ({n_all_correct/len(rows):.1%})")
    print(f"  both wrong           : {n_all_wrong:3d}  ({n_all_wrong/len(rows):.1%})")
    print(f"  graph wrong, sem OK  : {n_graph_only_wrong:3d}  ({n_graph_only_wrong/len(rows):.1%})  [graph retrieval/rerank issue]")
    print(f"  sem wrong, graph OK  : {n_sem_only_wrong:3d}  ({n_sem_only_wrong/len(rows):.1%})  [graph helps here]")
    print()


def mode_failures(rows: list[dict], show_n: int) -> None:
    """Show failing graph samples grouped by category."""
    by_cat: dict[int, list[dict]] = {}
    for r in rows:
        if not r["ok_graph"]:
            by_cat.setdefault(r["category"], []).append(r)

    for cat in sorted(by_cat):
        cat_rows = by_cat[cat]
        print(f"--- Category {cat} failures ({len(cat_rows)} total) ---")
        for i, row in enumerate(cat_rows[:show_n]):
            _print_sample(row, i + 1)
        if len(cat_rows) > show_n:
            print(f"  ... {len(cat_rows) - show_n} more not shown\n")


def mode_gap(rows: list[dict], show_n: int) -> None:
    """Show samples where graph failed but semantic succeeded (retrieval gap)."""
    gap_rows = [r for r in rows if not r["ok_graph"] and r["ok_semantic"]]
    print(f"--- Graph-failed / Semantic-succeeded ({len(gap_rows)} samples) ---\n")
    for i, row in enumerate(gap_rows[:show_n]):
        _print_sample(row, i + 1)
    if len(gap_rows) > show_n:
        print(f"... {len(gap_rows) - show_n} more not shown")


def mode_recall(rows: list[dict], show_n: int) -> None:
    """For each failing graph sample, check if the gold answer appeared in the evidence at all."""
    print("--- Evidence recall analysis (gold hint in retrieved evidence) ---\n")
    recall_graph = sum(1 for r in rows if _evidence_contains_hint(r.get("graph_top_evidence", []), r["gold"]))
    recall_sem = sum(1 for r in rows if _evidence_contains_hint(r.get("semantic_top_evidence", []), r["gold"]))
    n = len(rows)
    print(f"  Gold hint in graph top-evidence : {recall_graph}/{n} ({recall_graph/n:.1%})")
    print(f"  Gold hint in semantic top-evidence: {recall_sem}/{n} ({recall_sem/n:.1%})")
    print()

    # Where graph has gold in evidence but still wrong -- LLM reasoning failure
    ev_ok_pred_wrong = [
        r for r in rows
        if not r["ok_graph"] and _evidence_contains_hint(r.get("graph_top_evidence", []), r["gold"])
    ]
    print(f"  Graph has gold in evidence but pred wrong: {len(ev_ok_pred_wrong)} (LLM reasoning failures)")
    for i, row in enumerate(ev_ok_pred_wrong[:show_n]):
        _print_sample(row, i + 1, show_evidence=2)

    # Where graph doesn't have gold in evidence -- retrieval failure
    ev_miss = [
        r for r in rows
        if not r["ok_graph"] and not _evidence_contains_hint(r.get("graph_top_evidence", []), r["gold"])
    ]
    print(f"  Graph missing gold in evidence: {len(ev_miss)} (retrieval failures)")
    for i, row in enumerate(ev_miss[:show_n]):
        _print_sample(row, i + 1, show_evidence=1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze LoCoMo eval debug output")
    p.add_argument(
        "--run",
        default="",
        help="Path to run directory (default: latest in outputs/locomo_eval/)",
    )
    p.add_argument(
        "--mode",
        choices=["overview", "failures", "gap", "recall"],
        default="overview",
        help=(
            "overview: summary + error patterns | "
            "failures: all graph failures by category | "
            "gap: cases where graph fails but semantic succeeds | "
            "recall: check if gold answer was even in retrieved evidence"
        ),
    )
    p.add_argument(
        "--show-n",
        type=int,
        default=3,
        help="Max samples to show per group (default: 3)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    base = ROOT / "outputs" / "locomo_eval"
    run_dir = Path(args.run) if args.run else _latest_run(base)
    print(f"Run dir: {run_dir}\n")

    rows, summary = _load(run_dir)

    if args.mode == "overview":
        mode_overview(rows, summary)
        mode_failures(rows, show_n=args.show_n)
    elif args.mode == "failures":
        mode_overview(rows, summary)
        mode_failures(rows, show_n=args.show_n)
    elif args.mode == "gap":
        mode_overview(rows, summary)
        mode_gap(rows, show_n=args.show_n)
    elif args.mode == "recall":
        mode_overview(rows, summary)
        mode_recall(rows, show_n=args.show_n)


if __name__ == "__main__":
    main()
