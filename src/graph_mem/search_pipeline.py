from __future__ import annotations

from dataclasses import dataclass, field

from .embedder import BgeM3Embedder
from .graph_store import MemoryGraphStore
from .ranking import RankingConfig, fuse_score


@dataclass(slots=True)
class SearchConfig:
    seed_topk: int = 12
    expand_hop: int = 1
    expand_max_neighbors_per_type: int = 6
    final_topn_evidence: int = 8
    # 1-hop expansion: which edge types to follow (seeds are always kept)
    expand_use_semantic: bool = True
    expand_use_temporal: bool = True
    expand_use_co_usage: bool = True
    # Only traverse co_usage edges with at least this usage_count
    expand_min_co_usage_usage: int = 1
    # Solidify new co_usage pair edges only after this many joint top-k co-occurrence events
    co_usage_min_count: int = 1
    co_usage_decay: float = 1.0
    co_usage_prune_threshold: float = 0.05
    core_top_ratio: float = 0.2
    # Hybrid retrieval: BM25 + semantic RRF fusion for seed candidates
    use_bm25: bool = False      # off by default; enable per-eval (e.g. LoCoMo)
    bm25_topk: int = 20         # BM25 candidates before RRF fusion
    rrf_k: int = 60             # RRF constant (standard: 60)
    # P0: adaptive expansion - skip graph expansion when semantic confidence is high
    adaptive_expand_threshold: float = 0.0  # 0.0 = always expand; e.g. 0.80 = skip if top-1 sim >= 0.80
    # P1: entity edge traversal during graph expansion
    expand_use_entity: bool = True


@dataclass(slots=True)
class SearchPipeline:
    graph_store: MemoryGraphStore
    embedder: BgeM3Embedder
    rank_cfg: RankingConfig = field(default_factory=RankingConfig)
    cfg: SearchConfig = field(default_factory=SearchConfig)

    def search(self, query: str, *, now_ts: int) -> list[str]:
        qvec = self.embedder.encode(query)
        seeds = self._seed_retrieve(qvec, query)
        expanded = self._maybe_expand(seeds, qvec)
        ranked = self._rank(expanded, qvec)
        top_ids = [nid for nid, _ in ranked[: self.cfg.final_topn_evidence]]
        self._update_co_usage(top_ids, now_ts)
        return [self.graph_store.get_node(nid).structured_record.text for nid in top_ids]

    def search_with_debug(self, query: str, *, now_ts: int) -> tuple[list[str], dict]:
        qvec = self.embedder.encode(query)
        seeds = self._seed_retrieve(qvec, query)
        expanded, skipped_expand = self._maybe_expand(seeds, qvec, return_skip_flag=True)
        ranked, breakdown = self._rank_with_breakdown(expanded, qvec)
        top_ids = [nid for nid, _ in ranked[: self.cfg.final_topn_evidence]]
        self._update_co_usage(top_ids, now_ts)
        evidence = [self.graph_store.get_node(nid).structured_record.text for nid in top_ids]
        debug = {
            "seeds": seeds,
            "expanded": expanded,
            "skipped_expand": skipped_expand,
            "ranked_top_ids": top_ids,
            "score_breakdown": breakdown,
        }
        return evidence, debug

    def _seed_retrieve(self, qvec: list[float], query_str: str = "") -> list[str]:
        """Return seed node IDs.

        When ``cfg.use_bm25`` is True, performs Reciprocal Rank Fusion (RRF)
        over dense (semantic) and sparse (BM25) rankings before taking top-k.
        Falls back to pure semantic when BM25 is disabled.
        """
        # --- Dense (semantic) ranking ---
        sem_scores = [
            (node.node_id, self.embedder.cosine(qvec, node.embedding))
            for node in self.graph_store.all_nodes()
        ]
        sem_scores.sort(key=lambda x: x[1], reverse=True)

        if not self.cfg.use_bm25 or not query_str:
            return [nid for nid, _ in sem_scores[: self.cfg.seed_topk]]

        # --- Sparse (BM25) ranking ---
        bm25_results = self.graph_store.bm25_search(query_str, top_k=self.cfg.bm25_topk)

        # --- RRF fusion ---
        k = self.cfg.rrf_k
        rrf: dict[str, float] = {}
        for rank, (nid, _) in enumerate(sem_scores):
            rrf[nid] = rrf.get(nid, 0.0) + 1.0 / (k + rank + 1)
        for rank, (nid, _) in enumerate(bm25_results):
            rrf[nid] = rrf.get(nid, 0.0) + 1.0 / (k + rank + 1)

        fused = sorted(rrf.items(), key=lambda x: x[1], reverse=True)
        return [nid for nid, _ in fused[: self.cfg.seed_topk]]

    def _maybe_expand(
        self,
        seeds: list[str],
        qvec: list[float],
        *,
        return_skip_flag: bool = False,
    ) -> list[str] | tuple[list[str], bool]:
        """Expand seeds, optionally skipping if top-1 semantic confidence exceeds threshold."""
        skipped = False
        if self.cfg.adaptive_expand_threshold > 0.0 and seeds:
            top_sim = self.embedder.cosine(qvec, self.graph_store.get_node(seeds[0]).embedding)
            if top_sim >= self.cfg.adaptive_expand_threshold:
                skipped = True
        result = seeds if skipped else self._expand(seeds)
        if return_skip_flag:
            return result, skipped
        return result

    def _expand(self, seeds: list[str]) -> list[str]:
        edge_types: list[str] = []
        if self.cfg.expand_use_semantic:
            edge_types.append("semantic")
        if self.cfg.expand_use_temporal:
            edge_types.append("temporal")
        if self.cfg.expand_use_co_usage:
            edge_types.append("co_usage")
        if self.cfg.expand_use_entity:
            edge_types.append("entity")
        visited = set(seeds)
        frontier = list(seeds)
        for _ in range(self.cfg.expand_hop):
            nxt_frontier: list[str] = []
            for nid in frontier:
                for edge_type in edge_types:
                    if edge_type == "co_usage":
                        neighbors = self.graph_store.neighbors_by_type(
                            nid,
                            edge_type=edge_type,
                            limit=self.cfg.expand_max_neighbors_per_type,
                            min_usage_count=self.cfg.expand_min_co_usage_usage,
                        )
                    else:
                        neighbors = self.graph_store.neighbors_by_type(
                            nid, edge_type=edge_type, limit=self.cfg.expand_max_neighbors_per_type
                        )
                    for nb in neighbors:
                        if nb not in visited:
                            visited.add(nb)
                            nxt_frontier.append(nb)
            frontier = nxt_frontier
        return list(visited)

    def _rank(self, candidates: list[str], qvec: list[float]) -> list[tuple[str, float]]:
        centrality = self.graph_store.centrality()
        core_scores = self.graph_store.compute_core_scores()
        ranked: list[tuple[str, float]] = []
        for nid in candidates:
            node = self.graph_store.get_node(nid)
            semantic = self.embedder.cosine(qvec, node.embedding)
            cent = float(core_scores.get(nid, 0.0))
            edge_ev = self._edge_evidence_score(nid)
            temporal_fit = 1.0 if node.state == "active" else 0.5
            is_core = cent >= self._core_threshold(core_scores)
            score = fuse_score(
                semantic_score=semantic,
                centrality_score=cent,
                edge_evidence_score=edge_ev,
                temporal_fit_score=temporal_fit,
                is_core=is_core,
                is_active=node.state == "active",
                cfg=self.rank_cfg,
            )
            ranked.append((nid, score))
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked

    def _rank_with_breakdown(self, candidates: list[str], qvec: list[float]) -> tuple[list[tuple[str, float]], list[dict]]:
        centrality = self.graph_store.centrality()
        core_scores = self.graph_store.compute_core_scores()
        ranked: list[tuple[str, float]] = []
        breakdown: list[dict] = []
        cth = self._core_threshold(core_scores)
        for nid in candidates:
            node = self.graph_store.get_node(nid)
            semantic = self.embedder.cosine(qvec, node.embedding)
            cent = float(core_scores.get(nid, 0.0))
            edge_ev = self._edge_evidence_score(nid)
            temporal_fit = 1.0 if node.state == "active" else 0.5
            is_core = cent >= cth
            score = fuse_score(
                semantic_score=semantic,
                centrality_score=cent,
                edge_evidence_score=edge_ev,
                temporal_fit_score=temporal_fit,
                is_core=is_core,
                is_active=node.state == "active",
                cfg=self.rank_cfg,
            )
            ranked.append((nid, score))
            breakdown.append(
                {
                    "node_id": nid,
                    "semantic": round(semantic, 6),
                    "centrality": round(float(centrality.get(nid, 0.0)), 6),
                    "core_score": round(cent, 6),
                    "retrieve_count": node.retrieve_count,
                    "edge_evidence": round(edge_ev, 6),
                    "temporal_fit": round(temporal_fit, 6),
                    "is_core": is_core,
                    "is_active": node.state == "active",
                    "final_score": round(score, 6),
                }
            )
        ranked.sort(key=lambda x: x[1], reverse=True)
        breakdown.sort(key=lambda x: x["final_score"], reverse=True)
        return ranked, breakdown

    def _edge_evidence_score(self, node_id: str) -> float:
        out_degree = self.graph_store.graph.out_degree(node_id)
        if out_degree == 0:
            return 0.0
        # Normalize with a simple bounded transform.
        return min(1.0, out_degree / 10.0)

    def _core_threshold(self, core_scores: dict[str, float]) -> float:
        if not core_scores:
            return 0.0
        vals = sorted(core_scores.values(), reverse=True)
        top_n = max(1, int(len(vals) * self.cfg.core_top_ratio))
        return vals[top_n - 1]

    def _update_co_usage(self, top_ids: list[str], now_ts: int) -> None:
        self.graph_store.bump_retrieve_counts(top_ids)
        for i in range(len(top_ids)):
            for j in range(i + 1, len(top_ids)):
                a, b = top_ids[i], top_ids[j]
                self.graph_store.register_co_usage_between(
                    a,
                    b,
                    now_ts=now_ts,
                    min_count_to_solidify=self.cfg.co_usage_min_count,
                )
        if self.cfg.co_usage_decay < 1.0 or self.cfg.co_usage_prune_threshold > 0.0:
            self.graph_store.decay_co_usage_edges(
                self.cfg.co_usage_decay,
                self.cfg.co_usage_prune_threshold,
                now_ts,
            )
