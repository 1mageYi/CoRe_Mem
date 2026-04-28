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
    # Split-slot evidence budget:
    #   seed_evidence_slots  – filled from seed nodes ranked by semantic similarity
    #   expand_evidence_slots – filled from expansion-only nodes ranked by edge proximity to seeds
    # When both are 0 (default), falls back to original unified fuse_score ranking.
    seed_evidence_slots: int = 0
    expand_evidence_slots: int = 0
    # Personalized PageRank (PPR) as a third retrieval signal.
    # PPR propagates semantic-anchor relevance across graph edges, enabling
    # multi-hop candidates to surface even when they have low direct semantic sim.
    # When enabled, PPR scores are fused via RRF alongside semantic and BM25.
    use_ppr: bool = False
    ppr_n_anchors: int = 5        # top-N semantic hits used as PPR personalization anchors
    ppr_alpha: float = 0.85       # PageRank restart probability (higher = more local)
    ppr_topk: int = 40            # how many top-PPR candidates to include in RRF


@dataclass(slots=True)
class SearchPipeline:
    graph_store: MemoryGraphStore
    embedder: BgeM3Embedder
    rank_cfg: RankingConfig = field(default_factory=RankingConfig)
    cfg: SearchConfig = field(default_factory=SearchConfig)

    def _use_split_slots(self) -> bool:
        return self.cfg.seed_evidence_slots > 0 or self.cfg.expand_evidence_slots > 0

    def search(self, query: str, *, now_ts: int) -> list[str]:
        qvec = self.embedder.encode(query)
        seeds = self._seed_retrieve(qvec, query)
        expanded = self._maybe_expand(seeds, qvec)
        if self._use_split_slots():
            ranked = self._split_slot_rank(seeds, expanded, qvec)
        else:
            ranked = self._rank(expanded, qvec)
        top_ids = [nid for nid, _ in ranked[: self.cfg.final_topn_evidence]]
        self._update_co_usage(top_ids, now_ts)
        return [self.graph_store.get_node(nid).structured_record.text for nid in top_ids]

    def search_with_debug(self, query: str, *, now_ts: int) -> tuple[list[str], dict]:
        qvec = self.embedder.encode(query)
        seeds = self._seed_retrieve(qvec, query)
        expanded, skipped_expand = self._maybe_expand(seeds, qvec, return_skip_flag=True)
        if self._use_split_slots():
            ranked = self._split_slot_rank(seeds, expanded, qvec)
            breakdown: list[dict] = []   # full breakdown not computed in split-slot mode
        else:
            ranked, breakdown = self._rank_with_breakdown(expanded, qvec)
        top_ids = [nid for nid, _ in ranked[: self.cfg.final_topn_evidence]]
        self._update_co_usage(top_ids, now_ts)
        evidence = [self.graph_store.get_node(nid).structured_record.text for nid in top_ids]
        top_seed_sim = 0.0
        if seeds:
            top_seed_sim = float(self.embedder.cosine(qvec, self.graph_store.get_node(seeds[0]).embedding))
        debug = {
            "seeds": seeds,
            "expanded": expanded,
            "skipped_expand": skipped_expand,
            "ranked_top_ids": top_ids,
            "score_breakdown": breakdown,
            "top_seed_semantic_sim": round(top_seed_sim, 6),
            "n_expanded": len(expanded),
            "n_candidates_ranked": len(breakdown),
        }
        return evidence, debug

    def _seed_retrieve(self, qvec: list[float], query_str: str = "") -> list[str]:
        """Return seed node IDs via RRF over up to three signals:
        1. Dense semantic (always on)
        2. Sparse BM25 (cfg.use_bm25)
        3. Personalized PageRank (cfg.use_ppr) — propagates anchor relevance
           across graph edges, helping multi-hop candidates surface.
        """
        # --- Dense (semantic) ranking ---
        sem_scores = [
            (node.node_id, self.embedder.cosine(qvec, node.embedding))
            for node in self.graph_store.all_nodes()
        ]
        sem_scores.sort(key=lambda x: x[1], reverse=True)

        use_bm25 = self.cfg.use_bm25 and bool(query_str)
        use_ppr = self.cfg.use_ppr

        if not use_bm25 and not use_ppr:
            return [nid for nid, _ in sem_scores[: self.cfg.seed_topk]]

        # --- RRF accumulator ---
        k = self.cfg.rrf_k
        rrf: dict[str, float] = {}
        for rank, (nid, _) in enumerate(sem_scores):
            rrf[nid] = rrf.get(nid, 0.0) + 1.0 / (k + rank + 1)

        # --- Sparse (BM25) ---
        if use_bm25:
            bm25_results = self.graph_store.bm25_search(query_str, top_k=self.cfg.bm25_topk)
            for rank, (nid, _) in enumerate(bm25_results):
                rrf[nid] = rrf.get(nid, 0.0) + 1.0 / (k + rank + 1)

        # --- Personalized PageRank ---
        if use_ppr:
            # Use top-N semantic hits as personalization anchors
            anchors = {
                nid: max(0.0, score)
                for nid, score in sem_scores[: self.cfg.ppr_n_anchors]
            }
            ppr = self.graph_store.ppr_scores(anchors, alpha=self.cfg.ppr_alpha)
            ppr_sorted = sorted(ppr.items(), key=lambda x: x[1], reverse=True)
            for rank, (nid, _) in enumerate(ppr_sorted[: self.cfg.ppr_topk]):
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

    def _split_slot_rank(
        self, seeds: list[str], expanded: list[str], qvec: list[float]
    ) -> list[tuple[str, float]]:
        """Split-slot ranking: semantic-ranked seed slots + graph-proximity expand slots.

        seed_evidence_slots positions are filled by the best-semantic seed nodes.
        expand_evidence_slots positions are filled by expansion-only nodes ranked by
        their maximum edge weight to any seed node (graph proximity), not by semantic
        similarity to the query.  This prevents bridge nodes from being evicted by the
        unified fuse_score which unfairly favours graph-central nodes over fact-bearing
        periphery nodes.
        """
        seed_set = set(seeds)
        expand_only = [nid for nid in expanded if nid not in seed_set]

        # --- seed slots: ranked by semantic similarity ---
        seed_ranked = sorted(
            seeds,
            key=lambda nid: self.embedder.cosine(qvec, self.graph_store.get_node(nid).embedding),
            reverse=True,
        )
        top_seeds = seed_ranked[: self.cfg.seed_evidence_slots]

        # --- expand slots: ranked by max edge weight to any selected seed ---
        taken = set(top_seeds)

        def edge_proximity(nid: str) -> float:
            best = 0.0
            for seed_nid in top_seeds:
                if self.graph_store.graph.has_edge(seed_nid, nid):
                    w = self.graph_store.graph[seed_nid][nid].get("weight", 1.0)
                    best = max(best, float(w))
                if self.graph_store.graph.has_edge(nid, seed_nid):
                    w = self.graph_store.graph[nid][seed_nid].get("weight", 1.0)
                    best = max(best, float(w))
            # fall back to semantic if no direct edge (e.g. 2-hop)
            if best == 0.0:
                best = self.embedder.cosine(qvec, self.graph_store.get_node(nid).embedding) * 0.5
            return best

        expand_ranked = sorted(expand_only, key=edge_proximity, reverse=True)
        top_expand = [nid for nid in expand_ranked if nid not in taken][: self.cfg.expand_evidence_slots]

        # Combine: seeds first (highest semantic), then expand nodes
        result: list[tuple[str, float]] = []
        for rank, nid in enumerate(top_seeds):
            sem = self.embedder.cosine(qvec, self.graph_store.get_node(nid).embedding)
            result.append((nid, 1.0 - rank * 1e-6))   # preserve seed order via tiny tiebreak
            _ = sem
        for rank, nid in enumerate(top_expand):
            result.append((nid, 0.5 - rank * 1e-6))   # expand nodes always after seeds
        return result

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
