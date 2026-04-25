"""End-to-end RAG evaluation on PersonaMem 32k benchmark.

Usage (from CoRe_Mem/ or RAG_test/):
    python RAG_test/e2e_test.py          # run from CoRe_Mem/
    python e2e_test.py                   # run from RAG_test/

What this script does:
  1. Downloads questions_32k.csv + shared_contexts_32k.jsonl from HuggingFace
     (cached locally on the first run).
  2. For each of the 589 questions:
       a. Slices the conversation history to the relevant prefix.
       b. Retrieves the top-5 messages most similar to the user query (BGE-M3).
       c. Queries GPT-4o-mini with the retrieved context + 4-choice question.
       d. Parses the answer letter and records accuracy + latency.
  3. Aggregates accuracy and average query time by question_type.
  4. Writes results under RAG_test/results/{timestamp}/:
       predictions.csv       — per-question raw output
       summary_by_type.csv   — accuracy + timing per question type
       report.md             — human-readable markdown report

Environment variables (read from CoRe_Mem/.env):
    LLM_API_KEY   — OpenAI-compatible API key
    LLM_BASE_URL  — API base URL
    LLM_MODEL     — model name (default: gpt-4o-mini)
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Ensure RAG_test/ is on sys.path so sibling modules are importable
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# Load .env from CoRe_Mem/
from dotenv import load_dotenv
load_dotenv(_HERE.parent / ".env")

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

from data_utils import ensure_data, iter_questions_with_context, count_questions
from rag_pipeline import retrieve_top_k, query_llm, extract_answer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TOP_K = 5
MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
API_KEY = os.getenv("LLM_API_KEY", "")
BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")

RESULTS_DIR = _HERE / "results"

# Human-readable display names aligned with Figure 5 in the paper
_TYPE_DISPLAY: dict[str, str] = {
    "recall_user_shared_facts": "Recall User Shared Facts",
    "suggest_new_ideas": "Suggest New Ideas",
    "acknowledge_latest_preferences": "Acknowledge Latest User Preference",
    "track_full_preference_evolution": "Tracking Full Preference Evolution",
    "revisit_reasons_behind_preference_updates": "Revisit Reasons Behind Preference Updates",
    "provide_preference_aligned_recommendations": "Provide Preference Aligned Recommendations",
    "generalize_to_new_scenarios": "Generalize Reasons to New Scenarios",
}

_PRED_COLUMNS = [
    "persona_id",
    "question_id",
    "question_type",
    "topic",
    "correct_answer",
    "predicted_answer",
    "score",
    "query_time_s",
    "context_length_in_tokens",
    "distance_to_ref_in_blocks",
    "distance_to_ref_proportion_in_context",
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = RESULTS_DIR / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "model": MODEL,
        "base_url": BASE_URL,
        "top_k": TOP_K,
        "embedding_model": "BAAI/bge-m3",
        "benchmark": "PersonaMem-32k",
        "timestamp": timestamp,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    print("Checking PersonaMem 32k data …")
    q_path, ctx_path = ensure_data()
    total = count_questions(q_path)
    print(f"  questions        : {q_path}  ({total} rows)")
    print(f"  shared_contexts  : {ctx_path}")

    # ------------------------------------------------------------------
    # LLM client
    # ------------------------------------------------------------------
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    print(f"  model            : {MODEL}")
    print(f"  base_url         : {BASE_URL}\n")

    # ------------------------------------------------------------------
    # Evaluation loop
    # ------------------------------------------------------------------
    pred_path = out_dir / "predictions.csv"
    all_rows: list[dict] = []
    errors: list[dict] = []

    with open(pred_path, "w", newline="", encoding="utf-8") as pred_f:
        writer = csv.DictWriter(pred_f, fieldnames=_PRED_COLUMNS)
        writer.writeheader()

        pbar = tqdm(
            iter_questions_with_context(q_path, ctx_path),
            total=total,
            desc="Evaluating",
            unit="q",
        )
        for row, context_msgs in pbar:
            try:
                question = row["user_question_or_message"]
                all_options = row["all_options"]
                correct_answer = row["correct_answer"]
                q_type = row["question_type"]

                # RAG: retrieve top-k relevant messages
                retrieved = retrieve_top_k(context_msgs, question, k=TOP_K)

                # LLM inference
                raw_response, elapsed = query_llm(
                    client, MODEL, question, all_options, retrieved
                )

                # Score
                is_correct, pred_letter = extract_answer(raw_response, correct_answer)

                result = {
                    "persona_id": row["persona_id"],
                    "question_id": row["question_id"],
                    "question_type": q_type,
                    "topic": row["topic"],
                    "correct_answer": correct_answer,
                    "predicted_answer": pred_letter,
                    "score": int(is_correct),
                    "query_time_s": round(elapsed, 4),
                    "context_length_in_tokens": row["context_length_in_tokens"],
                    "distance_to_ref_in_blocks": row["distance_to_ref_in_blocks"],
                    "distance_to_ref_proportion_in_context": row[
                        "distance_to_ref_proportion_in_context"
                    ],
                }

                writer.writerow(result)
                pred_f.flush()
                all_rows.append(result)

                # Live progress update
                if all_rows:
                    acc_so_far = sum(r["score"] for r in all_rows) / len(all_rows)
                    pbar.set_postfix(acc=f"{acc_so_far*100:.1f}%", type=q_type[:12])

            except Exception as exc:
                errors.append(
                    {
                        "persona_id": row.get("persona_id", "?"),
                        "question_id": row.get("question_id", "?"),
                        "error": str(exc),
                    }
                )
                tqdm.write(f"[ERROR] {exc}")
                continue

    if errors:
        err_path = out_dir / "errors.json"
        err_path.write_text(json.dumps(errors, indent=2), encoding="utf-8")
        print(f"\n{len(errors)} errors logged to {err_path}")

    if not all_rows:
        print("No successful predictions — check errors above.")
        return

    # ------------------------------------------------------------------
    # Summarize
    # ------------------------------------------------------------------
    df = pd.DataFrame(all_rows)
    df["score"] = df["score"].astype(int)
    df["query_time_s"] = df["query_time_s"].astype(float)

    overall_acc = df["score"].mean()
    overall_avg_time = df["query_time_s"].mean()
    overall_total_time = df["query_time_s"].sum()

    summary = (
        df.groupby("question_type", sort=False)
        .agg(
            n=("score", "count"),
            accuracy=("score", "mean"),
            avg_query_time_s=("query_time_s", "mean"),
            total_query_time_s=("query_time_s", "sum"),
        )
        .reset_index()
    )
    summary["accuracy_pct"] = (summary["accuracy"] * 100).round(1)
    summary["avg_query_time_s"] = summary["avg_query_time_s"].round(3)
    summary["total_query_time_s"] = summary["total_query_time_s"].round(1)
    summary["display_name"] = summary["question_type"].map(
        lambda t: _TYPE_DISPLAY.get(t, t)
    )

    summary_path = out_dir / "summary_by_type.csv"
    summary.to_csv(summary_path, index=False)

    # ------------------------------------------------------------------
    # Markdown report
    # ------------------------------------------------------------------
    report = _build_report(config, df, summary, overall_acc, overall_avg_time,
                           overall_total_time, errors, timestamp)
    report_path = out_dir / "report.md"
    report_path.write_text(report, encoding="utf-8")

    # ------------------------------------------------------------------
    # Console output
    # ------------------------------------------------------------------
    print(f"\n{'='*65}")
    print(f"  PersonaMem 32k — RAG ({MODEL})  |  {timestamp}")
    print(f"{'='*65}")
    print(f"  Overall accuracy  : {overall_acc*100:.1f}%")
    print(f"  Avg query time    : {overall_avg_time:.3f}s / question")
    print(f"  Total questions   : {len(df)}  |  Errors: {len(errors)}")
    print()
    _WIDTH_TYPE = 50
    print(f"  {'Query Type':<{_WIDTH_TYPE}} {'N':>5}  {'Acc%':>6}  {'AvgT(s)':>8}")
    print(f"  {'-'*_WIDTH_TYPE}  -----  ------  --------")
    for _, r in summary.iterrows():
        label = _TYPE_DISPLAY.get(r["question_type"], r["question_type"])
        print(
            f"  {label:<{_WIDTH_TYPE}} {int(r['n']):>5}  "
            f"{r['accuracy_pct']:>5.1f}%  {r['avg_query_time_s']:>8.3f}"
        )
    print(f"  {'Average':<{_WIDTH_TYPE}} {len(df):>5}  {overall_acc*100:>5.1f}%  {overall_avg_time:>8.3f}")
    print(f"\n  Results → {out_dir}")
    print(f"{'='*65}\n")


def _build_report(
    config: dict,
    df: pd.DataFrame,
    summary: pd.DataFrame,
    overall_acc: float,
    overall_avg_time: float,
    overall_total_time: float,
    errors: list[dict],
    timestamp: str,
) -> str:
    """Build the markdown report string."""
    lines = [
        "# PersonaMem 32k — RAG Evaluation Report",
        "",
        f"**Run timestamp** : `{timestamp}`  ",
        f"**Model**         : `{config['model']}`  ",
        f"**Embedding**     : `{config['embedding_model']}`  ",
        f"**Retrieval**     : dense cosine similarity, top-{config['top_k']} messages  ",
        f"**Benchmark**     : PersonaMem 32k (N={len(df)})  ",
        "",
        "---",
        "",
        "## Overall Results",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Overall accuracy | **{overall_acc*100:.1f}%** |",
        f"| Avg query time (per question) | {overall_avg_time:.3f} s |",
        f"| Total LLM query time | {overall_total_time:.1f} s |",
        f"| Total questions evaluated | {len(df)} |",
        f"| Errors / skipped | {len(errors)} |",
        "",
        "---",
        "",
        "## Accuracy & Query Time by Question Type",
        "",
        "Matches the query-type breakdown shown in Figure 5 of the PersonaMem paper.",
        "",
        "| Question Type | N | Accuracy (%) | Avg Query Time (s) |",
        "|--------------|--:|-------------:|-------------------:|",
    ]

    for _, r in summary.iterrows():
        label = _TYPE_DISPLAY.get(r["question_type"], r["question_type"])
        lines.append(
            f"| {label} | {int(r['n'])} | {r['accuracy_pct']:.1f} | {r['avg_query_time_s']:.3f} |"
        )

    lines += [
        f"| **Average** | **{len(df)}** | **{overall_acc*100:.1f}** | **{overall_avg_time:.3f}** |",
        "",
        "---",
        "",
        "## Run Config",
        "",
        "```json",
        json.dumps(config, indent=2),
        "```",
        "",
    ]

    if errors:
        lines += [
            "## Errors",
            "",
            f"{len(errors)} questions failed. See `errors.json` for details.",
            "",
        ]

    return "\n".join(lines)


if __name__ == "__main__":
    run()
