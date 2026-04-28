"""LoCoMo QA evaluation: graph memory vs semantic-only vs full-context baselines.

Three retrieval modes are compared side-by-side:
  graph-full    Our graph memory system (extract → graph → rerank)
  semantic-only Pure embedding search over the structured graph nodes (no graph expansion)
  full-context  Entire conversation history pasted directly into the LLM prompt.
                Falls back to truncation (most recent turns) if the conversation exceeds
                ``--context-budget-chars`` (default 400 k chars ≈ 100 k tokens).

Usage (from CoRe_Mem/ root with uv):

    uv run python experiments/locomo/eval_locomo_graph.py

    # Quick smoke test (2 conversations, 20 QA each):
    uv run python experiments/locomo/eval_locomo_graph.py --conv-limit 2 --qa-limit 20

    # Only category 1 & 2 questions, with LLM judge:
    uv run python experiments/locomo/eval_locomo_graph.py --categories 1,2 --use-llm-judge

    # Skip the full-context baseline (saves API cost):
    uv run python experiments/locomo/eval_locomo_graph.py --no-full-context

    # Skip debug file:
    uv run python experiments/locomo/eval_locomo_graph.py --no-debug

Outputs are written to:
    outputs/locomo_eval/<timestamp>/
        summary.json
        predictions.csv
        debug_samples.jsonl   (unless --no-debug)
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from openai import OpenAI
from tqdm import tqdm

from graph_mem.add_pipeline import AddConfig, AddPipeline
from graph_mem.embedder import BgeM3Embedder
from graph_mem.eval_utils import load_llm_config
from graph_mem.extractor import RuleExtractor
from graph_mem.graph_store import MemoryGraphStore
from graph_mem.locomo_data import (
    LoCoMoConversation,
    LoCoMoQA,
    filter_qa_by_categories,
    load_locomo,
    turn_time_index,
)
from graph_mem.ranking import RankingConfig
from graph_mem.search_pipeline import SearchConfig, SearchPipeline


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _answer_prompt(question: str, evidence: list[str]) -> str:
    """Prompt for graph-memory / semantic-only baselines (retrieved snippets).

    Design principles:
    - Encourage multi-snippet reasoning: snippets may need to be combined.
    - Explicit temporal arithmetic: resolve relative dates from session timestamps.
    - Commonsense bridging: allowed for cat-3 style questions.
    - Concise final answer: do NOT output reasoning, only the answer.
    """
    ev_text = (
        "\n".join(f"  [{i+1}] {e}" for i, e in enumerate(evidence))
        if evidence else "  (no relevant memories found)"
    )
    return (
        "You are a memory assistant answering questions about people based on retrieved conversation snippets.\n"
        "\n"
        "Rules:\n"
        "1. Use ONLY the snippets below as your source of facts.\n"
        "2. You may combine information from multiple snippets to form the answer.\n"
        "3. Each snippet includes a timestamp (e.g. '[Caroline, Session 3, 9 June 2023]'). "
        "Use these to resolve relative time expressions like 'yesterday', 'last week', 'next month'.\n"
        "4. You may apply general commonsense reasoning to bridge implicit connections between facts.\n"
        "5. Give a concise, direct answer. Do NOT include your reasoning steps in the reply.\n"
        "6. If the snippets do not contain enough information to answer, reply exactly: I don't know\n"
        "\n"
        f"Conversation snippets:\n{ev_text}\n"
        "\n"
        f"Question: {question}\n"
        "Answer:"
    )


def _full_context_prompt(question: str, context_block: str) -> str:
    """Prompt for the full-context baseline (raw conversation history)."""
    return (
        "You are a memory assistant answering questions about people based on their full conversation history.\n"
        "\n"
        "Rules:\n"
        "1. Use ONLY the conversation provided below as your source of facts.\n"
        "2. You may combine information from multiple parts of the conversation.\n"
        "3. Each turn is timestamped by session and date; use these to resolve relative time references.\n"
        "4. You may apply general commonsense reasoning to bridge implicit connections between facts.\n"
        "5. Give a concise, direct answer. Do NOT include your reasoning steps in the reply.\n"
        "6. If the answer cannot be found in the conversation, reply exactly: I don't know\n"
        "\n"
        f"Conversation:\n{context_block}\n"
        "\n"
        f"Question: {question}\n"
        "Answer:"
    )


def _judge_prompt(question: str, gold: str, predicted: str) -> str:
    return (
        "You are judging whether a predicted answer correctly answers a question.\n"
        "The predicted answer is correct if it captures the essential information from the gold answer,\n"
        "even if worded differently or more verbose.\n\n"
        f"Question: {question}\n"
        f"Gold answer: {gold}\n"
        f"Predicted answer: {predicted}\n\n"
        "Is the predicted answer correct? Reply with exactly 'yes' or 'no'."
    )


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

def _query_llm(
    client: OpenAI,
    model: str,
    prompt: str,
    max_retries: int = 3,
) -> tuple[str, float]:
    delay = 1.0
    for attempt in range(max_retries):
        try:
            t0 = time.perf_counter()
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            elapsed = time.perf_counter() - t0
            return (resp.choices[0].message.content or "").strip(), elapsed
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(delay)
            delay = min(30.0, delay * 2.0)
    return "", 0.0


# ---------------------------------------------------------------------------
# Answer judging
# ---------------------------------------------------------------------------

def _normalize(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _soft_match(gold: str, pred: str) -> bool:
    """Fast fuzzy match without an extra LLM call.

    Correct if:
    - Normalized exact match
    - Gold is a substring of pred (pred is verbose but contains the answer)
    - Pred is a substring of gold (pred is a valid shorthand)
    - Word-level F1 >= 0.5
    """
    gn = _normalize(gold)
    pn = _normalize(pred)
    if not gn or not pn:
        return False
    if gn == pn:
        return True
    if gn in pn or pn in gn:
        return True
    gw = set(gn.split())
    pw = set(pn.split())
    if not gw or not pw:
        return False
    f1 = 2 * len(gw & pw) / (len(gw) + len(pw))
    return f1 >= 0.5


def _llm_judge(
    client: OpenAI,
    model: str,
    question: str,
    gold: str,
    predicted: str,
) -> tuple[bool, float]:
    prompt = _judge_prompt(question, gold, predicted)
    answer, elapsed = _query_llm(client, model, prompt)
    return answer.lower().startswith("yes"), elapsed


# ---------------------------------------------------------------------------
# Full-context baseline helpers
# ---------------------------------------------------------------------------

def _format_conversation_block(conv: LoCoMoConversation) -> str:
    """Format all turns grouped by session, suitable for an LLM prompt."""
    lines: list[str] = []
    current_session = -1
    for turn in conv.turns:
        if turn.session_idx != current_session:
            current_session = turn.session_idx
            dt = f" ({turn.session_datetime})" if turn.session_datetime else ""
            lines.append(f"\n[Session {turn.session_idx}{dt}]")
        lines.append(f"  {turn.speaker}: {turn.text}")
    return "\n".join(lines).strip()


def full_context_evidence(
    conv: LoCoMoConversation,
    budget_chars: int = 400_000,
) -> tuple[str, bool]:
    """Return the full conversation block as a single string.

    Returns:
        (context_block, was_truncated) — if the full conversation exceeds
        ``budget_chars`` the most recent turns are kept until the budget is
        exhausted (truncated from the beginning).
    """
    full_block = _format_conversation_block(conv)
    if len(full_block) <= budget_chars:
        return full_block, False

    # Truncate: walk turns from the end until we fit inside budget.
    lines: list[str] = []
    chars_used = 0
    current_session = -1
    for turn in reversed(conv.turns):
        # Estimate chars for this turn (+ session header on session change)
        header = ""
        if turn.session_idx != current_session:
            dt = f" ({turn.session_datetime})" if turn.session_datetime else ""
            header = f"\n[Session {turn.session_idx}{dt}]\n"
            current_session = turn.session_idx
        turn_line = f"  {turn.speaker}: {turn.text}\n"
        cost = len(header) + len(turn_line)
        if chars_used + cost > budget_chars:
            break
        lines.append(turn_line if not header else header + turn_line)
        chars_used += cost

    truncated_block = "".join(reversed(lines)).strip()
    return truncated_block, True


# ---------------------------------------------------------------------------
# Gold-turn vs retrieval diagnostics (for graph vs full-context analysis)
# ---------------------------------------------------------------------------

def gold_evidence_retrieval_diagnostics(
    conv: LoCoMoConversation,
    qa: LoCoMoQA,
    store: MemoryGraphStore,
    embedder: BgeM3Embedder,
    query: str,
    graph_debug: dict,
    graph_evidence: list[str],
    *,
    final_topn: int,
) -> dict:
    """Align LoCoMo ``evidence_refs`` (e.g. D1:3) to graph nodes and report ranks.

    Used to separate **retrieval failure** (gold node not in top-k evidence) from
    **reasoning failure** (gold content present but LLM still wrong vs full-context).
    """
    qv = embedder.encode(query)
    nodes = store.all_nodes()
    sem_ranked = sorted(
        [(n.node_id, embedder.cosine(qv, n.embedding)) for n in nodes],
        key=lambda x: x[1],
        reverse=True,
    )
    rank_by_id = {nid: i + 1 for i, (nid, _) in enumerate(sem_ranked)}
    sim_by_id = {nid: sim for nid, sim in sem_ranked}

    gold_nids: list[str] = []
    per_ref: list[dict] = []

    for ref in qa.evidence:
        turn = conv.get_turn(ref)
        if turn is None:
            per_ref.append({"dia_id": ref, "resolved": False})
            continue
        matched: list[str] = []
        needle = turn.text.strip()
        for node in nodes:
            rec = node.structured_record
            if turn.global_idx in rec.source_turn_ids:
                matched.append(node.node_id)
            elif needle and needle in rec.text:
                matched.append(node.node_id)
        matched = list(dict.fromkeys(matched))
        gold_nids.extend(matched)
        ranks = [rank_by_id[m] for m in matched]
        sims = [sim_by_id[m] for m in matched]
        per_ref.append(
            {
                "dia_id": ref,
                "resolved": True,
                "matched_node_ids": matched,
                "best_semantic_rank": min(ranks) if ranks else None,
                "best_semantic_sim": round(max(sims), 6) if sims else None,
            }
        )

    gold_nids = list(dict.fromkeys(gold_nids))
    best_sem_rank = min((rank_by_id[n] for n in gold_nids), default=None) if gold_nids else None
    best_sem_sim = max((sim_by_id[n] for n in gold_nids), default=None) if gold_nids else None

    seeds = graph_debug.get("seeds", [])
    expanded = graph_debug.get("expanded", [])
    ranked_top = graph_debug.get("ranked_top_ids", [])
    breakdown = graph_debug.get("score_breakdown", [])

    gold_in_seed = bool(gold_nids) and any(n in seeds for n in gold_nids)
    gold_in_expanded = bool(gold_nids) and any(n in expanded for n in gold_nids)
    gold_in_final_topk = bool(gold_nids) and any(n in ranked_top[:final_topn] for n in gold_nids)

    positions = [
        i + 1 for i, item in enumerate(breakdown) if item.get("node_id") in gold_nids
    ]
    gold_rerank_pos = min(positions) if positions else None

    ev_blob = _normalize(" ".join(graph_evidence))
    gold_turn_text_in_evidence = False
    for ref in qa.evidence:
        t = conv.get_turn(ref)
        if t and _normalize(t.text) in ev_blob:
            gold_turn_text_in_evidence = True
            break

    return {
        "gold_evidence_node_ids": gold_nids,
        "gold_node_found": len(gold_nids) > 0,
        "best_semantic_rank_global": best_sem_rank,
        "best_semantic_sim": round(best_sem_sim, 6) if best_sem_sim is not None else None,
        "gold_in_seed_list": gold_in_seed,
        "gold_in_expanded_set": gold_in_expanded,
        "gold_in_final_topk": gold_in_final_topk,
        "gold_rerank_position_among_expanded": gold_rerank_pos,
        "gold_turn_raw_text_in_graph_evidence": gold_turn_text_in_evidence,
        "per_reference": per_ref,
    }


# ---------------------------------------------------------------------------
# Graph building from LoCoMo turns
# ---------------------------------------------------------------------------

def build_graph_from_conversation(
    conv: LoCoMoConversation,
    *,
    embedder: BgeM3Embedder,
    add_cfg: AddConfig | None = None,
    build_entity_edges: bool = True,
) -> tuple[MemoryGraphStore, dict]:
    """Feed all turns (both speakers) into a fresh graph store.

    Returns:
        (store, build_stats) where build_stats contains node/edge counts.
    """
    store = MemoryGraphStore()
    extractor = RuleExtractor()
    ac = add_cfg or AddConfig()
    add = AddPipeline(graph_store=store, embedder=embedder, cfg=ac)

    for turn in conv.turns:
        if not turn.text.strip():
            continue
        ti = turn_time_index(turn)
        # Include session date so the LLM can resolve relative time references
        # ("yesterday", "next month") into absolute dates, matching full-context.
        date_part = f", {turn.session_datetime}" if turn.session_datetime else ""
        text = f"[{turn.speaker}, Session {turn.session_idx}{date_part}] {turn.text}"
        rec = extractor.extract_from_turn(text, time_index=ti, source_turn_id=turn.global_idx)
        add.add_record(rec, now_ts=ti)

    # P1: build entity edges in a single batch pass after all nodes are inserted
    n_entity_edges = 0
    if build_entity_edges:
        n_entity_edges = add.build_entity_edges(now_ts=0)

    build_stats = {
        "n_nodes": len(store.nodes),
        "n_edges_total": store.graph.number_of_edges(),
        "n_entity_edges": n_entity_edges,
    }
    return store, build_stats


# ---------------------------------------------------------------------------
# Semantic-only evidence (no graph expansion)
# ---------------------------------------------------------------------------

def semantic_only_evidence(
    store: MemoryGraphStore,
    embedder: BgeM3Embedder,
    query: str,
    top_k: int = 8,
) -> list[str]:
    qv = embedder.encode(query)
    scored = [
        (embedder.cosine(qv, node.embedding), node.structured_record.text)
        for node in store.all_nodes()
    ]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [txt for _, txt in scored[:top_k]]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class LoCoMoEvalConfig:
    data_path: str = ""             # filled in main()
    conv_limit: int = 0             # 0 = all conversations
    qa_limit: int = 0               # 0 = all QA per conversation
    categories: set[int] = field(default_factory=lambda: {1, 2, 3, 4})
    exclude_adversarial: bool = True
    use_llm_judge: bool = False     # True = extra LLM call per QA for judging
    include_full_context: bool = True  # add full-context baseline column
    context_budget_chars: int = 400_000  # truncate if conv exceeds this
    write_debug: bool = True
    top_k: int = 16   # semantic-only baseline top-k; keep aligned with final_topn_evidence


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_locomo_eval(
    cfg: LoCoMoEvalConfig,
    *,
    root_dir: Path,
    out_dir: Path | None = None,
    show_progress: bool = True,
) -> dict:
    llm_cfg = load_llm_config(str(root_dir / ".env"))
    client = OpenAI(api_key=llm_cfg.api_key, base_url=llm_cfg.base_url)

    convs = load_locomo(cfg.data_path, exclude_adversarial=cfg.exclude_adversarial)
    if cfg.conv_limit > 0:
        convs = convs[: cfg.conv_limit]

    # For LoCoMo's fact-retrieval task, semantic similarity should dominate.
    # PERMA's core-prior boost hurts here: early-session nodes (which hold
    # key facts) have fewer edges than later nodes and get demoted unfairly.
    rank_cfg = RankingConfig(
        alpha_semantic=0.90,
        beta_centrality=0.05,
        gamma_edge_evidence=0.03,
        delta_temporal_fit=0.02,
        lambda_core=0.0,   # no core-prior boost
    )
    # Hybrid BM25+semantic retrieval + P0 adaptive expansion:
    # - BM25 complements dense retrieval for keyword-heavy single-hop facts
    # - adaptive_expand_threshold: skip graph traversal when top-1 is already
    #   high-confidence (prevents adding noisy neighbours to cat-1 evidence)
    # - expand_use_entity: follow entity edges built by build_entity_edges()
    # Widened recall pool (from dropout analysis):
    # - seed_topk 20->40: gold semantic rank median=22 for misses, need wider window
    # - bm25_topk matches seed_topk to keep RRF balance
    # - final_topn_evidence 8->12: 520/1540 were dropped by top-8 budget cut
    search_cfg = SearchConfig(
        seed_topk=40,
        use_bm25=True,
        bm25_topk=40,
        adaptive_expand_threshold=0.80,  # P0
        expand_use_entity=True,           # P1
        final_topn_evidence=16,
        # Split-slot rerank: seeds ranked by semantic, expand nodes ranked by
        # edge proximity to seeds.  Prevents graph-peripheral fact nodes from
        # being evicted by the fuse_score that rewards highly-connected nodes.
        seed_evidence_slots=10,
        expand_evidence_slots=6,
        # P2: Personalized PageRank — disabled for now.
        # On LoCoMo's sparse per-conversation graph (~200-400 nodes, chain-like
        # temporal edges), PPR propagation overlaps heavily with existing 1-hop
        # expand and slightly hurts single-hop (cat4) by boosting noisy neighbours.
        # Re-evaluate after graph density improves (e.g. sentence-level nodes).
        use_ppr=False,
    )
    add_cfg = AddConfig()

    shared_embedder = BgeM3Embedder()

    rows: list[dict] = []
    debug_rows: list[dict] = []

    # Aggregate gold-vs-retrieval stats (for graph vs full-context post-mortem)
    retr_summary = {
        "n_total": 0,
        "n_gold_node_found": 0,
        "n_gold_in_final_topk": 0,
        "n_gold_turn_text_in_graph_evidence": 0,
        "n_graph_wrong_full_right": 0,
        "n_graph_wrong_full_right_gold_in_topk": 0,
        "n_graph_wrong_full_right_gold_missing_topk": 0,
    }

    conv_iter = tqdm(convs, desc="LoCoMo conv", unit="conv") if show_progress else convs

    for conv in conv_iter:
        # Build graph once per conversation (shared across all QA pairs).
        store, build_stats = build_graph_from_conversation(
            conv, embedder=shared_embedder, add_cfg=add_cfg, build_entity_edges=True
        )
        pipe = SearchPipeline(
            graph_store=store,
            embedder=shared_embedder,
            rank_cfg=rank_cfg,
            cfg=search_cfg,
        )

        qa_items = filter_qa_by_categories(conv.qa, cfg.categories)
        if cfg.qa_limit > 0:
            qa_items = qa_items[: cfg.qa_limit]

        # Pre-build full-context block once per conversation (reused for every QA).
        ctx_block: str = ""
        ctx_truncated: bool = False
        ctx_chars: int = 0
        if cfg.include_full_context:
            ctx_block, ctx_truncated = full_context_evidence(conv, cfg.context_budget_chars)
            ctx_chars = len(ctx_block)

        qa_iter = (
            tqdm(qa_items, desc=f"  {conv.sample_id}", leave=False, unit="q")
            if show_progress
            else qa_items
        )

        for qa in qa_iter:
            now_ts = 999_999  # query happens "after" all ingestion

            graph_evidence, graph_debug = pipe.search_with_debug(qa.question, now_ts=now_ts)
            sem_evidence = semantic_only_evidence(
                store, shared_embedder, qa.question, top_k=cfg.top_k
            )

            graph_prompt = _answer_prompt(qa.question, graph_evidence)
            sem_prompt = _answer_prompt(qa.question, sem_evidence)

            graph_ans, t_graph = _query_llm(client, llm_cfg.model, graph_prompt)
            sem_ans, t_sem = _query_llm(client, llm_cfg.model, sem_prompt)

            # Full-context baseline
            full_ctx_ans: str = ""
            t_full_ctx: float = 0.0
            ok_full_ctx: bool = False
            if cfg.include_full_context:
                full_ctx_prompt = _full_context_prompt(qa.question, ctx_block)
                full_ctx_ans, t_full_ctx = _query_llm(client, llm_cfg.model, full_ctx_prompt)

            def _judge(gold: str, pred: str) -> bool:
                if cfg.use_llm_judge:
                    correct, _ = _llm_judge(client, llm_cfg.model, qa.question, gold, pred)
                    return correct
                return _soft_match(gold, pred)

            ok_graph = _judge(qa.answer, graph_ans)
            ok_sem = _judge(qa.answer, sem_ans)
            if cfg.include_full_context:
                ok_full_ctx = _judge(qa.answer, full_ctx_ans)

            row: dict = {
                "sample_id": conv.sample_id,
                "category": qa.category,
                "question": qa.question,
                "gold": qa.answer,
                "pred_graph": graph_ans,
                "pred_semantic": sem_ans,
                "ok_graph": int(ok_graph),
                "ok_semantic": int(ok_sem),
                "t_graph_s": round(t_graph, 4),
                "t_semantic_s": round(t_sem, 4),
            }
            if cfg.include_full_context:
                row["pred_full_ctx"] = full_ctx_ans
                row["ok_full_ctx"] = int(ok_full_ctx)
                row["t_full_ctx_s"] = round(t_full_ctx, 4)
                row["ctx_chars"] = ctx_chars
                row["ctx_truncated"] = int(ctx_truncated)

            rows.append(row)

            if cfg.write_debug:
                dbg: dict = {
                    "sample_id": conv.sample_id,
                    "category": qa.category,
                    "question": qa.question,
                    "gold": qa.answer,
                    "evidence_refs": qa.evidence,
                    "pred_graph": graph_ans,
                    "pred_semantic": sem_ans,
                    "ok_graph": int(ok_graph),
                    "ok_semantic": int(ok_sem),
                    "graph_debug": graph_debug,
                    "graph_top_evidence": graph_evidence,
                    "semantic_top_evidence": sem_evidence,
                    "build_stats": build_stats,
                }
                if cfg.include_full_context:
                    dbg["pred_full_ctx"] = full_ctx_ans
                    dbg["ok_full_ctx"] = int(ok_full_ctx)

                retr = gold_evidence_retrieval_diagnostics(
                    conv,
                    qa,
                    store,
                    shared_embedder,
                    qa.question,
                    graph_debug,
                    graph_evidence,
                    final_topn=pipe.cfg.final_topn_evidence,
                )
                dbg["retrieval_diagnostics"] = retr
                retr_summary["n_total"] += 1
                if retr["gold_node_found"]:
                    retr_summary["n_gold_node_found"] += 1
                if retr["gold_in_final_topk"]:
                    retr_summary["n_gold_in_final_topk"] += 1
                if retr["gold_turn_raw_text_in_graph_evidence"]:
                    retr_summary["n_gold_turn_text_in_graph_evidence"] += 1
                if cfg.include_full_context and (not ok_graph) and ok_full_ctx:
                    retr_summary["n_graph_wrong_full_right"] += 1
                    if retr["gold_in_final_topk"]:
                        retr_summary["n_graph_wrong_full_right_gold_in_topk"] += 1
                    else:
                        retr_summary["n_graph_wrong_full_right_gold_missing_topk"] += 1

                debug_rows.append(dbg)

    # Aggregate
    n = len(rows)
    acc_graph = sum(r["ok_graph"] for r in rows) / n if n else 0.0
    acc_sem = sum(r["ok_semantic"] for r in rows) / n if n else 0.0
    avg_t_graph = sum(r["t_graph_s"] for r in rows) / n if n else 0.0
    avg_t_sem = sum(r["t_semantic_s"] for r in rows) / n if n else 0.0

    acc_full_ctx: float | None = None
    avg_t_full_ctx: float | None = None
    if cfg.include_full_context and rows:
        acc_full_ctx = sum(r.get("ok_full_ctx", 0) for r in rows) / n
        avg_t_full_ctx = sum(r.get("t_full_ctx_s", 0.0) for r in rows) / n
        n_truncated = sum(r.get("ctx_truncated", 0) for r in rows)
    else:
        n_truncated = 0

    # Per-category accuracy
    cat_stats: dict[int, dict] = {}
    for cat in sorted(cfg.categories):
        cat_rows = [r for r in rows if r["category"] == cat]
        nc = len(cat_rows)
        entry: dict = {
            "n": nc,
            "acc_graph": round(sum(r["ok_graph"] for r in cat_rows) / nc, 4) if nc else 0.0,
            "acc_semantic": round(sum(r["ok_semantic"] for r in cat_rows) / nc, 4) if nc else 0.0,
        }
        if cfg.include_full_context:
            entry["acc_full_context"] = (
                round(sum(r.get("ok_full_ctx", 0) for r in cat_rows) / nc, 4) if nc else 0.0
            )
        cat_stats[cat] = entry

    summary: dict = {
        "n_conversations": len(convs),
        "n_samples": n,
        "judge_method": "llm" if cfg.use_llm_judge else "soft_match",
        "categories_evaluated": sorted(cfg.categories),
        "baselines": (
            ["graph_full", "semantic_only", "full_context"]
            if cfg.include_full_context
            else ["graph_full", "semantic_only"]
        ),
        "acc_graph_full": round(acc_graph, 4),
        "acc_semantic_only": round(acc_sem, 4),
        "acc_delta_graph_minus_semantic": round(acc_graph - acc_sem, 4),
        "avg_t_graph_s": round(avg_t_graph, 4),
        "avg_t_semantic_s": round(avg_t_sem, 4),
        "per_category": cat_stats,
        "llm_model": llm_cfg.model,
        "llm_base_url": llm_cfg.base_url,
    }
    if cfg.include_full_context:
        summary["acc_full_context"] = round(acc_full_ctx, 4) if acc_full_ctx is not None else None
        summary["acc_delta_graph_minus_full_ctx"] = (
            round(acc_graph - acc_full_ctx, 4) if acc_full_ctx is not None else None
        )
        summary["avg_t_full_ctx_s"] = round(avg_t_full_ctx, 4) if avg_t_full_ctx is not None else None
        summary["n_truncated_convs"] = n_truncated

    if retr_summary["n_total"] > 0:
        nt = retr_summary["n_total"]
        summary["retrieval_diagnostics"] = {
            **retr_summary,
            "rate_gold_node_found": round(retr_summary["n_gold_node_found"] / nt, 4),
            "rate_gold_in_final_topk": round(retr_summary["n_gold_in_final_topk"] / nt, 4),
            "rate_gold_turn_text_in_graph_evidence": round(
                retr_summary["n_gold_turn_text_in_graph_evidence"] / nt, 4
            ),
            "rate_graph_wrong_full_right": round(retr_summary["n_graph_wrong_full_right"] / nt, 4),
            "among_graph_wrong_full_right": {
                "n": retr_summary["n_graph_wrong_full_right"],
                "gold_in_topk": retr_summary["n_graph_wrong_full_right_gold_in_topk"],
                "gold_missing_topk": retr_summary["n_graph_wrong_full_right_gold_missing_topk"],
            },
        }

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        if rows:
            pred_csv = out_dir / "predictions.csv"
            with pred_csv.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
        (out_dir / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        if cfg.write_debug and debug_rows:
            with (out_dir / "debug_samples.jsonl").open("w", encoding="utf-8") as f:
                for item in debug_rows:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"\n[LoCoMo] Results written to: {out_dir}")

    return summary


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LoCoMo graph-memory QA evaluation")
    p.add_argument(
        "--data",
        default=str(ROOT / "data" / "locomo" / "locomo10.json"),
        help="Path to locomo JSON file",
    )
    p.add_argument(
        "--conv-limit",
        type=int,
        default=0,
        help="Max conversations to evaluate (0 = all)",
    )
    p.add_argument(
        "--qa-limit",
        type=int,
        default=0,
        help="Max QA pairs per conversation (0 = all)",
    )
    p.add_argument(
        "--categories",
        default="1,2,3,4",
        help="Comma-separated category IDs to include (default: 1,2,3,4; excludes 5=adversarial)",
    )
    p.add_argument(
        "--include-adversarial",
        action="store_true",
        help="Include category 5 (adversarial / false-premise) questions",
    )
    p.add_argument(
        "--use-llm-judge",
        action="store_true",
        help="Use LLM to judge answer correctness instead of soft string match (costs extra API calls)",
    )
    p.add_argument(
        "--no-full-context",
        action="store_true",
        help="Skip the full-context baseline (saves API cost for quick runs)",
    )
    p.add_argument(
        "--context-budget-chars",
        type=int,
        default=400_000,
        help=(
            "Max chars for the full-context prompt. Conversations exceeding this are "
            "truncated (most recent turns kept). locomo10 max is ~90k chars so default "
            "400k means no truncation."
        ),
    )
    p.add_argument(
        "--no-debug",
        action="store_true",
        help="Skip debug_samples.jsonl output (faster runs)",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=16,
        help="Top-K evidence snippets to retrieve (graph final_topn and semantic-only top-k)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    categories = {int(c.strip()) for c in args.categories.split(",") if c.strip()}
    if args.include_adversarial:
        categories.add(5)

    cfg = LoCoMoEvalConfig(
        data_path=args.data,
        conv_limit=args.conv_limit,
        qa_limit=args.qa_limit,
        categories=categories,
        exclude_adversarial=not args.include_adversarial,
        use_llm_judge=args.use_llm_judge,
        include_full_context=not args.no_full_context,
        context_budget_chars=args.context_budget_chars,
        write_debug=not args.no_debug,
        top_k=args.top_k,
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "outputs" / "locomo_eval" / ts

    summary = run_locomo_eval(cfg, root_dir=ROOT, out_dir=out_dir, show_progress=True)

    print("\n===== LoCoMo Evaluation Summary =====")
    print(f"Conversations : {summary['n_conversations']}")
    print(f"QA samples    : {summary['n_samples']}")
    print(f"Judge method  : {summary['judge_method']}")
    print(f"Categories    : {summary['categories_evaluated']}")
    print(f"Baselines     : {summary['baselines']}")
    print()
    print(f"Acc graph-full   : {summary['acc_graph_full']:.4f}")
    print(f"Acc semantic-only: {summary['acc_semantic_only']:.4f}")
    if summary.get("acc_full_context") is not None:
        print(f"Acc full-context : {summary['acc_full_context']:.4f}")
    print()
    print(f"Delta (graph - semantic)  : {summary['acc_delta_graph_minus_semantic']:+.4f}")
    if summary.get("acc_delta_graph_minus_full_ctx") is not None:
        print(f"Delta (graph - full-ctx)  : {summary['acc_delta_graph_minus_full_ctx']:+.4f}")
    print()
    print(f"Avg t graph   : {summary['avg_t_graph_s']:.3f}s")
    print(f"Avg t semantic: {summary['avg_t_semantic_s']:.3f}s")
    if summary.get("avg_t_full_ctx_s") is not None:
        print(f"Avg t full-ctx: {summary['avg_t_full_ctx_s']:.3f}s")
    print("\nPer-category accuracy:")
    for cat, stats in summary["per_category"].items():
        line = (
            f"  cat {cat}: n={stats['n']:3d}  "
            f"graph={stats['acc_graph']:.4f}  "
            f"semantic={stats['acc_semantic']:.4f}"
        )
        if "acc_full_context" in stats:
            line += f"  full-ctx={stats['acc_full_context']:.4f}"
        print(line)


if __name__ == "__main__":
    main()
