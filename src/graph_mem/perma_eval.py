from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

from .add_pipeline import AddPipeline
from .embedder import BgeM3Embedder
from .eval_utils import build_eval_prompt, load_llm_config, query_option_letter
from .extractor import RuleExtractor
from .graph_store import MemoryGraphStore
from .perma_data import load_perma_user_samples
from .search_pipeline import SearchPipeline


@dataclass(slots=True)
class PermaEvalConfig:
    user_id: str = "user108"
    variant: str = "c"
    limit: int = 10


def semantic_only_evidence(
    store: MemoryGraphStore,
    embedder: BgeM3Embedder,
    query: str,
    top_k: int = 8,
) -> list[str]:
    qv = embedder.encode(query)
    scored = []
    for node in store.all_nodes():
        sim = embedder.cosine(qv, node.embedding)
        scored.append((sim, node.structured_text))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [txt for _, txt in scored[:top_k]]


def build_graph_from_context(
    context_messages: list[dict],
    *,
    embedder: BgeM3Embedder,
) -> tuple[MemoryGraphStore, BgeM3Embedder]:
    store = MemoryGraphStore()
    extractor = RuleExtractor()
    add = AddPipeline(graph_store=store, embedder=embedder)

    t = 0
    for msg in context_messages:
        if msg.get("role") != "user":
            continue
        t += 1
        rec = extractor.extract_from_turn(msg["content"], time_index=t, source_turn_id=t)
        add.add_record(rec, now_ts=t)
    return store, embedder


def run_perma_eval(
    cfg: PermaEvalConfig,
    *,
    root_dir: Path,
    out_dir: Path | None = None,
    show_progress: bool = True,
) -> dict:
    llm_cfg = load_llm_config(str(root_dir / ".env"))
    client = OpenAI(api_key=llm_cfg.api_key, base_url=llm_cfg.base_url)
    samples = load_perma_user_samples(cfg.user_id, split_variant=cfg.variant)
    if cfg.limit > 0:
        samples = samples[: cfg.limit]

    shared_embedder = BgeM3Embedder()
    search_cache: dict[str, SearchPipeline] = {}
    graph_cache: dict[str, tuple[MemoryGraphStore, BgeM3Embedder]] = {}

    rows = []
    iterator = tqdm(samples, desc="PERMA eval", unit="q") if show_progress else samples
    for idx, sample in enumerate(iterator, start=1):
        key = f"{sample.user_id}:{sample.task_id}:{sample.task_type}"
        if key not in graph_cache:
            store, embedder = build_graph_from_context(sample.context_messages, embedder=shared_embedder)
            graph_cache[key] = (store, embedder)
            search_cache[key] = SearchPipeline(graph_store=store, embedder=embedder)
        else:
            store, embedder = graph_cache[key]

        graph_evidence = search_cache[key].search(sample.question, now_ts=10_000 + idx)
        semantic_evidence = semantic_only_evidence(store, embedder, sample.question, top_k=8)

        graph_prompt = build_eval_prompt(sample.question, sample.options_text, graph_evidence)
        sem_prompt = build_eval_prompt(sample.question, sample.options_text, semantic_evidence)

        graph_pred, t_graph = query_option_letter(client, llm_cfg, graph_prompt)
        sem_pred, t_sem = query_option_letter(client, llm_cfg, sem_prompt)

        rows.append(
            {
                "user_id": sample.user_id,
                "task_id": sample.task_id,
                "task_type": sample.task_type,
                "gold": sample.gold_label,
                "pred_graph": graph_pred,
                "pred_semantic": sem_pred,
                "ok_graph": int(graph_pred == sample.gold_label),
                "ok_semantic": int(sem_pred == sample.gold_label),
                "t_graph_s": round(t_graph, 4),
                "t_semantic_s": round(t_sem, 4),
            }
        )

    n = len(rows)
    acc_graph = (sum(r["ok_graph"] for r in rows) / n) if n else 0.0
    acc_sem = (sum(r["ok_semantic"] for r in rows) / n) if n else 0.0
    avg_t_graph = (sum(r["t_graph_s"] for r in rows) / n) if n else 0.0
    avg_t_sem = (sum(r["t_semantic_s"] for r in rows) / n) if n else 0.0

    summary = {
        "user_id": cfg.user_id,
        "variant": cfg.variant,
        "n_samples": n,
        "acc_graph_full": round(acc_graph, 4),
        "acc_semantic_only": round(acc_sem, 4),
        "acc_delta_graph_minus_semantic": round(acc_graph - acc_sem, 4),
        "avg_t_graph_s": round(avg_t_graph, 4),
        "avg_t_semantic_s": round(avg_t_sem, 4),
        "llm_model": llm_cfg.model,
        "llm_base_url": llm_cfg.base_url,
    }

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        pred_csv = out_dir / "predictions.csv"
        with pred_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
            if rows:
                writer.writeheader()
                writer.writerows(rows)
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return summary


def run_perma_eval_with_timestamped_output(
    cfg: PermaEvalConfig,
    *,
    root_dir: Path,
    show_progress: bool = True,
) -> tuple[dict, Path]:
    out_root = root_dir / "outputs" / "perma_eval"
    out_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = out_root / timestamp
    summary = run_perma_eval(cfg, root_dir=root_dir, out_dir=out_dir, show_progress=show_progress)
    return summary, out_dir
