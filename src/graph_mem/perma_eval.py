from __future__ import annotations

import csv
import json
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

from .ablation_presets import (
    GraphEdgeMode,
    add_config_for_merge,
    search_config_for_edge_mode,
)
from .add_pipeline import AddConfig, AddPipeline, MergeStrategy
from .embedder import BgeM3Embedder
from .eval_utils import build_eval_prompt, load_llm_config, query_option_letter
from .extractor import RuleExtractor
from .graph_store import MemoryGraphStore
from .perma_data import load_perma_user_samples
from .ranking import RankingConfig
from .search_pipeline import SearchConfig, SearchPipeline


def _search_cfg_sig(sc: SearchConfig) -> str:
    return (
        f"s{int(sc.expand_use_semantic)}{int(sc.expand_use_temporal)}{int(sc.expand_use_co_usage)}"
        f"emu{sc.expand_min_co_usage_usage}cms{sc.co_usage_min_count}"
        f"cd{sc.co_usage_decay}cp{sc.co_usage_prune_threshold}"
    )


@dataclass(slots=True)
class PermaEvalConfig:
    user_id: str = "user108"
    variant: str = "c"
    limit: int = 10
    # None -> use `RankingConfig.lambda_core` in `ranking.py`
    lambda_core: float | None = None
    # False skips debug_samples.jsonl (faster parameter sweeps)
    write_debug: bool = True
    graph_edge_mode: GraphEdgeMode = "full"
    merge_strategy: MergeStrategy = "hybrid"
    co_usage_decay: float = 1.0
    co_usage_min_count: int = 1
    co_usage_prune_threshold: float = 0.05
    expand_min_co_usage_usage: int = 1
    temporal_probe_thirds: bool = False


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
        scored.append((sim, node.structured_record.text))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [txt for _, txt in scored[:top_k]]


def build_graph_from_context(
    context_messages: list[dict],
    *,
    embedder: BgeM3Embedder,
    add_cfg: AddConfig | None = None,
) -> tuple[MemoryGraphStore, BgeM3Embedder]:
    store = MemoryGraphStore()
    extractor = RuleExtractor()
    ac = add_cfg or AddConfig()
    add = AddPipeline(graph_store=store, embedder=embedder, cfg=ac)

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

    rank_cfg = RankingConfig()
    if cfg.lambda_core is not None:
        rank_cfg = replace(rank_cfg, lambda_core=float(cfg.lambda_core))

    search_cfg = search_config_for_edge_mode(cfg.graph_edge_mode)
    search_cfg = replace(
        search_cfg,
        co_usage_decay=float(cfg.co_usage_decay),
        co_usage_min_count=int(cfg.co_usage_min_count),
        co_usage_prune_threshold=float(cfg.co_usage_prune_threshold),
        expand_min_co_usage_usage=int(cfg.expand_min_co_usage_usage),
    )
    add_cfg = add_config_for_merge(cfg.merge_strategy)

    shared_embedder = BgeM3Embedder()
    search_cache: dict[str, SearchPipeline] = {}
    graph_cache: dict[str, tuple[MemoryGraphStore, BgeM3Embedder]] = {}

    rows = []
    debug_rows = []
    iterator = tqdm(samples, desc="PERMA eval", unit="q") if show_progress else samples
    for idx, sample in enumerate(iterator, start=1):
        gk = f"{sample.user_id}:{sample.task_id}:{sample.task_type}:m{cfg.merge_strategy}"
        if gk not in graph_cache:
            store, embedder = build_graph_from_context(
                sample.context_messages,
                embedder=shared_embedder,
                add_cfg=add_cfg,
            )
            graph_cache[gk] = (store, embedder)
        else:
            store, embedder = graph_cache[gk]

        sk = f"{gk}|λ{rank_cfg.lambda_core}|{_search_cfg_sig(search_cfg)}"
        if sk not in search_cache:
            search_cache[sk] = SearchPipeline(
                graph_store=store,
                embedder=embedder,
                rank_cfg=rank_cfg,
                cfg=search_cfg,
            )
        pipe = search_cache[sk]

        graph_evidence, graph_debug = pipe.search_with_debug(sample.question, now_ts=10_000 + idx)
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
        if cfg.write_debug:
            debug_rows.append(
                {
                    "sample_index": idx,
                    "user_id": sample.user_id,
                    "task_id": sample.task_id,
                    "task_type": sample.task_type,
                    "question": sample.question,
                    "gold": sample.gold_label,
                    "pred_graph": graph_pred,
                    "pred_semantic": sem_pred,
                    "ok_graph": int(graph_pred == sample.gold_label),
                    "ok_semantic": int(sem_pred == sample.gold_label),
                    "graph_debug": graph_debug,
                    "graph_top_evidence": graph_evidence,
                    "semantic_top_evidence": semantic_evidence,
                    "graph_prompt_preview": graph_prompt[:800],
                    "semantic_prompt_preview": sem_prompt[:800],
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
        "lambda_core": rank_cfg.lambda_core,
        "graph_edge_mode": cfg.graph_edge_mode,
        "merge_strategy": cfg.merge_strategy,
        "co_usage_decay": cfg.co_usage_decay,
        "co_usage_min_count": cfg.co_usage_min_count,
        "co_usage_prune_threshold": cfg.co_usage_prune_threshold,
        "expand_min_co_usage_usage": cfg.expand_min_co_usage_usage,
        "n_samples": n,
        "acc_graph_full": round(acc_graph, 4),
        "acc_semantic_only": round(acc_sem, 4),
        "acc_delta_graph_minus_semantic": round(acc_graph - acc_sem, 4),
        "avg_t_graph_s": round(avg_t_graph, 4),
        "avg_t_semantic_s": round(avg_t_sem, 4),
        "llm_model": llm_cfg.model,
        "llm_base_url": llm_cfg.base_url,
    }

    if cfg.temporal_probe_thirds and n >= 1:
        t1 = n // 3
        t2 = 2 * n // 3

        def _third(acc_fn, start: int, end: int) -> dict:
            sl = rows[start:end]
            if not sl:
                return {"n": 0, "acc_graph": 0.0, "acc_semantic": 0.0}
            ng = sum(acc_fn(r)[0] for r in sl) / len(sl)
            ns = sum(acc_fn(r)[1] for r in sl) / len(sl)
            return {"n": len(sl), "acc_graph": round(ng, 4), "acc_semantic": round(ns, 4)}

        summary["temporal_probe_thirds"] = {
            "early": _third(lambda r: (r["ok_graph"], r["ok_semantic"]), 0, t1),
            "mid": _third(lambda r: (r["ok_graph"], r["ok_semantic"]), t1, t2),
            "late": _third(lambda r: (r["ok_graph"], r["ok_semantic"]), t2, n),
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
        if cfg.write_debug:
            debug_path = out_dir / "debug_samples.jsonl"
            with debug_path.open("w", encoding="utf-8") as f:
                for item in debug_rows:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

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
