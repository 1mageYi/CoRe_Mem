"""
Weak supervision from raw wiki text: adjacent lines as merge positives, random cross-line pairs
as new_slot negatives (Router K=1). Compactor uses merge positives with teacher = encode(a + " " + b).

Requires: ``datasets``, ``sentence-transformers`` (see project ``requirements.txt``).
First run downloads Wikitext-2 (~4MB) from Hugging Face Hub.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import torch

from compactor.datasets import CompactorTensorDataset, RouterTensorDataset

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


def _collect_pairs_from_wikitext(
    *,
    split_rows: str = "train",
    max_positive_pairs: int = 20000,
    min_line_chars: int = 20,
    seed: int = 42,
) -> tuple[list[tuple[str, str]], list[str]]:
    """
    Return positive (prev_line, next_line) pairs and a line pool for negatives.

    Wikitext-2 raw has **one wiki line per row**; consecutive rows are often adjacent
    lines in the article — we pair ``(row[i], row[i+1])`` when both pass length filter.
    """
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split_rows)
    positives: list[tuple[str, str]] = []
    pool: list[str] = []
    prev: str | None = None

    for ex in ds:
        line = (ex.get("text") or "").strip()
        if not line:
            prev = None
            continue
        if len(line) >= min_line_chars:
            pool.append(line)
        if prev is not None and len(prev) >= min_line_chars and len(line) >= min_line_chars:
            positives.append((prev, line))
            if len(positives) >= max_positive_pairs:
                break
        prev = line

    if len(positives) < 50:
        raise RuntimeError(
            f"too few adjacent pairs ({len(positives)}) from wikitext; try lowering min_line_chars "
            "or check dataset access."
        )
    if len(pool) < 100:
        raise RuntimeError("too few lines in pool for negatives; check network / dataset access")
    random.Random(seed).shuffle(positives)
    positives = positives[:max_positive_pairs]
    return positives, pool


def _make_negative_pairs(
    positives: list[tuple[str, str]],
    pool: list[str],
    n: int,
    seed: int,
) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    neg: list[tuple[str, str]] = []
    for _ in range(n):
        a, _ = rng.choice(positives) if positives else (rng.choice(pool), rng.choice(pool))
        if not positives:
            a = rng.choice(pool)
        b = rng.choice(pool)
        t = 0
        while a == b and t < 15:
            b = rng.choice(pool)
            t += 1
        neg.append((a, b))
    return neg


def _encode_all(
    model: "SentenceTransformer",
    texts: list[str],
    *,
    batch_size: int = 32,
) -> torch.Tensor:
    import numpy as np

    out: list[torch.Tensor] = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        emb = model.encode(
            chunk,
            batch_size=min(batch_size, len(chunk)),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        t = torch.from_numpy(np.asarray(emb, dtype=np.float32))
        out.append(t)
    return torch.cat(out, dim=0)


def _router_tensors_from_pairs(
    model: "SentenceTransformer",
    pos: list[tuple[str, str]],
    neg: list[tuple[str, str]],
    *,
    encode_batch_size: int,
) -> RouterTensorDataset:
    texts_old: list[str] = []
    texts_new: list[str] = []
    labels: list[int] = []
    for a, b in pos:
        texts_old.append(a)
        texts_new.append(b)
        labels.append(0)
    for a, b in neg:
        texts_old.append(a)
        texts_new.append(b)
        labels.append(1)
    e_old = _encode_all(model, texts_old, batch_size=encode_batch_size)
    e_new = _encode_all(model, texts_new, batch_size=encode_batch_size)
    y = torch.tensor(labels, dtype=torch.long)
    e_cands = e_old.unsqueeze(1)
    mask = torch.ones(e_new.shape[0], 1, dtype=torch.bool)
    return RouterTensorDataset(e_new, e_cands, mask, y)


def build_router_k1_wikitext(
    model: "SentenceTransformer",
    *,
    n_train: int,
    n_val: int,
    min_line_chars: int = 20,
    encode_batch_size: int = 32,
    seed: int = 42,
) -> tuple[RouterTensorDataset, RouterTensorDataset]:
    """
    Router K=1: ``e_cands[:,0]`` = memory, ``e_new`` = incoming; label 0=merge, 1=new_slot.

    Balanced: half merge (adjacent Wikitext **rows**), half negative (random cross lines).
    """
    need_pairs = max(n_train, n_val) + n_train + n_val + 500
    pos_all, pool = _collect_pairs_from_wikitext(
        max_positive_pairs=need_pairs,
        min_line_chars=min_line_chars,
        seed=seed,
    )
    rng = random.Random(seed)
    rng.shuffle(pos_all)

    n_pos_tr = n_train // 2
    n_neg_tr = n_train - n_pos_tr
    n_pos_va = n_val // 2
    n_neg_va = n_val - n_pos_va

    pos_tr = pos_all[:n_pos_tr]
    pos_va = pos_all[n_pos_tr : n_pos_tr + n_pos_va]
    neg_tr = _make_negative_pairs(pos_all, pool, n_neg_tr, seed=seed + 1)
    neg_va = _make_negative_pairs(pos_all, pool, n_neg_va, seed=seed + 2)

    train_ds = _router_tensors_from_pairs(model, pos_tr, neg_tr, encode_batch_size=encode_batch_size)
    val_ds = _router_tensors_from_pairs(model, pos_va, neg_va, encode_batch_size=encode_batch_size)
    return train_ds, val_ds


def build_compactor_wikitext(
    model: "SentenceTransformer",
    *,
    n_train: int,
    n_val: int,
    min_line_chars: int = 20,
    encode_batch_size: int = 32,
    seed: int = 42,
) -> tuple[CompactorTensorDataset, CompactorTensorDataset]:
    """Merge-only triples: ``e_target = encode(a + " " + b)`` (DESIGN §4.3)."""
    need = n_train + n_val + 100
    pos_all, _ = _collect_pairs_from_wikitext(
        max_positive_pairs=need,
        min_line_chars=min_line_chars,
        seed=seed,
    )
    rng = random.Random(seed)
    rng.shuffle(pos_all)
    pos_tr = pos_all[:n_train]
    pos_va = pos_all[n_train : n_train + n_val]

    def triples(pairs: list[tuple[str, str]]) -> CompactorTensorDataset:
        texts_old = [a for a, _ in pairs]
        texts_new = [b for _, b in pairs]
        texts_tgt = [f"{a} {b}" for a, b in pairs]
        e_old = _encode_all(model, texts_old, batch_size=encode_batch_size)
        e_new = _encode_all(model, texts_new, batch_size=encode_batch_size)
        e_tgt = _encode_all(model, texts_tgt, batch_size=encode_batch_size)
        return CompactorTensorDataset(e_old, e_new, e_tgt)

    return triples(pos_tr), triples(pos_va)
