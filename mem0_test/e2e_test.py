"""PersonaMem 32k — Mem0-style evaluation (OSS Memory + FAISS).

Run from repo root:
    python mem0_test/e2e_test.py

Requires:
    pip install mem0ai faiss-cpu   # see CoRe_Mem/requirements.txt

Design (aligned with PersonaMem §4.4, details where the paper is silent):
  - Build memory by feeding the sliced dialog **turn-wise** via ``Memory.add``
    (pairs user+assistant; ``async_mode=False`` so extraction finishes).
  - Retrieve **top-5** facts with ``Memory.search`` for the current user question.
  - Answer the same 4-way MCQ with **gpt-4o-mini** via your OpenAI-compatible API.

Env (CoRe_Mem/.env):
    LLM_API_KEY, LLM_BASE_URL, LLM_MODEL=gpt-4o-mini
    MEM0_EMBED_PROVIDER — ``huggingface`` (default) or ``openai``
    MEM0_HF_EMBED_MODEL — HF model id when using huggingface embedder
    MEM0_EMBEDDING_MODEL / MEM0_EMBEDDING_DIMS — when MEM0_EMBED_PROVIDER=openai

Note: Mem0 extraction + embedding traffic is heavy; expect long wall-clock
(similar order-of-magnitude to the paper's Mem0 latency discussion).
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for p in (_HERE, _ROOT / "RAG_test"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from dotenv import load_dotenv

load_dotenv(_ROOT / ".env")

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

from llm_utils import extract_answer, format_retrieved_memories, query_llm
from mem0_pipeline import build_memory_config, ingest_history_turnwise, search_top_memories

# Re-use HF download helpers from RAG_test (same cache dir → no duplicate data).
from data_utils import ensure_data, iter_questions_with_context, count_questions

TOP_K = 5
MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
API_KEY = os.getenv("LLM_API_KEY", "")
BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")

RESULTS_DIR = _HERE / "results"

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
    "mem0_ingest_time_s",
    "mem0_search_time_s",
    "query_time_s",
    "context_length_in_tokens",
    "distance_to_ref_in_blocks",
    "distance_to_ref_proportion_in_context",
]


def _load_memory():
    try:
        from mem0 import Memory
    except ImportError as exc:
        raise SystemExit(
            "mem0 is not installed. Run: uv pip install mem0ai faiss-cpu"
        ) from exc

    cfg = build_memory_config(
        api_key=API_KEY,
        base_url=BASE_URL,
        llm_model=MODEL,
    )
    return Memory.from_config(cfg)


def run() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = RESULTS_DIR / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "approach": "mem0_oss_Memory_faiss",
        "llm_model": MODEL,
        "llm_base_url": BASE_URL,
        "top_k_memories": TOP_K,
        "ingest": "turnwise_dialog_pairs_infer_true",
        "mem0_embed_provider": os.getenv("MEM0_EMBED_PROVIDER", "huggingface"),
        "mem0_hf_embed_model": os.getenv(
            "MEM0_HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        "mem0_openai_embed_model": os.getenv(
            "MEM0_EMBEDDING_MODEL", "text-embedding-3-small"
        ),
        "benchmark": "PersonaMem-32k",
        "timestamp": timestamp,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    print("Initializing Mem0 (OSS) …")
    memory = _load_memory()
    llm_client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    print("Loading PersonaMem 32k …")
    q_path, ctx_path = ensure_data()
    total = count_questions(q_path)
    print(f"  rows: {total}  |  model: {MODEL}\n")

    pred_path = out_dir / "predictions.csv"
    all_rows: list[dict] = []
    errors: list[dict] = []

    with open(pred_path, "w", newline="", encoding="utf-8") as pred_f:
        writer = csv.DictWriter(pred_f, fieldnames=_PRED_COLUMNS)
        writer.writeheader()

        pbar = tqdm(
            iter_questions_with_context(q_path, ctx_path),
            total=total,
            desc="Mem0 eval",
            unit="q",
        )
        for row, context_msgs in pbar:
            qid = row["question_id"]
            user_id = f"pm_{qid}"
            question = row["user_question_or_message"]
            all_options = row["all_options"]
            correct = row["correct_answer"]

            try:
                t_ingest = ingest_history_turnwise(
                    memory, context_msgs, user_id=user_id
                )
                mem_texts, t_search = search_top_memories(
                    memory, question, user_id=user_id, top_k=TOP_K
                )
                prefix = format_retrieved_memories(mem_texts)

                raw, t_llm = query_llm(
                    llm_client,
                    MODEL,
                    question,
                    all_options,
                    prefix,
                )

                ok, pred = extract_answer(raw, correct)
                rec = {
                    "persona_id": row["persona_id"],
                    "question_id": qid,
                    "question_type": row["question_type"],
                    "topic": row["topic"],
                    "correct_answer": correct,
                    "predicted_answer": pred,
                    "score": int(ok),
                    "mem0_ingest_time_s": round(t_ingest, 3),
                    "mem0_search_time_s": round(t_search, 4),
                    "query_time_s": round(t_llm, 4),
                    "context_length_in_tokens": row["context_length_in_tokens"],
                    "distance_to_ref_in_blocks": row["distance_to_ref_in_blocks"],
                    "distance_to_ref_proportion_in_context": row[
                        "distance_to_ref_proportion_in_context"
                    ],
                }
                writer.writerow(rec)
                pred_f.flush()
                all_rows.append(rec)

                if all_rows:
                    acc = sum(r["score"] for r in all_rows) / len(all_rows)
                    pbar.set_postfix(
                        acc=f"{acc*100:.0f}%",
                        ing=f"{t_ingest:.0f}s",
                        llm=f"{t_llm:.1f}s",
                    )

            except Exception as exc:
                errors.append({"question_id": qid, "error": str(exc)})
                pbar.write(f"[ERROR] {qid}: {exc}")
                continue

    if errors:
        (out_dir / "errors.json").write_text(
            json.dumps(errors, indent=2), encoding="utf-8"
        )

    if not all_rows:
        print("No successful rows. See errors.json / stderr.")
        return

    df = pd.DataFrame(all_rows)
    df["score"] = df["score"].astype(int)
    overall_acc = df["score"].mean()
    overall_llm = df["query_time_s"].astype(float).mean()
    overall_ingest = df["mem0_ingest_time_s"].astype(float).mean()
    overall_search = df["mem0_search_time_s"].astype(float).mean()
    overall_e2e = (
        df["mem0_ingest_time_s"].astype(float)
        + df["mem0_search_time_s"].astype(float)
        + df["query_time_s"].astype(float)
    ).mean()

    summary = (
        df.groupby("question_type", sort=False)
        .agg(
            n=("score", "count"),
            accuracy=("score", "mean"),
            avg_ingest_s=("mem0_ingest_time_s", "mean"),
            avg_search_s=("mem0_search_time_s", "mean"),
            avg_llm_s=("query_time_s", "mean"),
        )
        .reset_index()
    )
    summary["accuracy_pct"] = (summary["accuracy"] * 100).round(1)
    for c in ("avg_ingest_s", "avg_search_s", "avg_llm_s"):
        summary[c] = summary[c].round(3)
    summary["display_name"] = summary["question_type"].map(
        lambda t: _TYPE_DISPLAY.get(t, t)
    )
    summary_path = out_dir / "summary_by_type.csv"
    summary.to_csv(summary_path, index=False)

    report_lines = [
        "# PersonaMem 32k — Mem0 (OSS) Evaluation",
        "",
        f"- **Timestamp**: `{timestamp}`",
        f"- **LLM**: `{MODEL}` @ `{BASE_URL}`",
        f"- **Mem0**: open-source `Memory`, FAISS, top-{TOP_K} memories",
        f"- **Overall accuracy**: **{overall_acc*100:.1f}%**",
        f"- **Avg ingest (Mem0 add)**: {overall_ingest:.2f} s / question",
        f"- **Avg search (Mem0)**: {overall_search:.3f} s / question",
        f"- **Avg final LLM**: {overall_llm:.3f} s / question",
        f"- **Avg end-to-end**: {overall_e2e:.2f} s / question",
        "",
        "## By question type",
        "",
        "| Type | N | Acc % | Avg ingest (s) | Avg search (s) | Avg LLM (s) |",
        "|------|--:|------:|---------------:|---------------:|------------:|",
    ]
    for _, r in summary.iterrows():
        report_lines.append(
            f"| {_TYPE_DISPLAY.get(r['question_type'], r['question_type'])} | "
            f"{int(r['n'])} | {r['accuracy_pct']:.1f} | "
            f"{r['avg_ingest_s']:.2f} | {r['avg_search_s']:.3f} | {r['avg_llm_s']:.3f} |"
        )
    report_lines.append(
        f"| **Average** | {len(df)} | {overall_acc*100:.1f} | "
        f"{overall_ingest:.2f} | {overall_search:.3f} | {overall_llm:.3f} |"
    )
    report_lines.extend(["", "```json", json.dumps(config, indent=2), "```"])
    (out_dir / "report.md").write_text("\n".join(report_lines), encoding="utf-8")

    print(f"\nDone. Results: {out_dir}")
    print(f"  accuracy={overall_acc*100:.1f}%  n={len(df)}  errors={len(errors)}")


if __name__ == "__main__":
    run()
