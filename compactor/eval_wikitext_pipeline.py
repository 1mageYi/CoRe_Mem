"""
在 **Wikitext-2 弱监督切分** 上批量评估 Router + Compactor（与 ``train_*.py`` 同一数据协议）。

与 ``e2e_text_pipeline`` 的区别：

- **本脚本**：用训练时的相邻行 / 随机行对，衡量「记忆库单槽 + ingest」下的 merge / new_slot 与 Compactor 的 1−cos。
- **e2e_text_pipeline**：任意两句英文；Router 只在 **百科相邻行 vs 随机行** 上训练，对话/偏好句常被判 **new_slot**（分布外），不代表模型坏了。

用法（仓库根 ``CoRe_Mem/``）::

    export PYTHONPATH=.:src
    python -m compactor.eval_wikitext_pipeline --n-val 1000

若未传 ``--router`` / ``--compactor``，默认 ``compactor/checkpoints/*_wikitext.pt``。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from compactor.encoding import load_encoder
from compactor.fusion import MemoryCompactor
from compactor.inference import ingest_embedding
from compactor.memory_bank import MemoryBank
from compactor.router import RouterV1
from compactor.train_compactor import collate_compact
from compactor.train_router import collate_router, eval_epoch
from compactor.weak_supervision import (
    build_compactor_wikitext,
    build_router_k1_wikitext,
    wikitext_compactor_text_splits,
    wikitext_router_k1_text_splits,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_router(path: Path, device: torch.device) -> RouterV1:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = RouterV1(embed_dim=ck["embed_dim"], hidden_dim=ck["hidden_dim"])
    m.load_state_dict(ck["state_dict"])
    return m.to(device).eval()


def load_compactor(path: Path, device: torch.device) -> MemoryCompactor:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = MemoryCompactor(embed_dim=ck["embed_dim"], hidden_dim=ck["hidden_dim"])
    m.load_state_dict(ck["state_dict"])
    return m.to(device).eval()


@torch.no_grad()
def router_batch_accuracy(
    router: RouterV1,
    *,
    n_train: int,
    n_val: int,
    min_line_chars: int,
    seed: int,
    device: torch.device,
) -> float:
    """Same batched CE accuracy as training ``eval_epoch`` (tensor dataset)."""
    st = load_encoder()
    _, val_ds = build_router_k1_wikitext(
        st,
        n_train=n_train,
        n_val=n_val,
        min_line_chars=min_line_chars,
        seed=seed,
    )
    loader = DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=collate_router)
    return eval_epoch(router, loader, device)


@torch.no_grad()
def router_streaming_metrics(
    router: RouterV1,
    compactor: MemoryCompactor,
    *,
    pos_va: list[tuple[str, str]],
    neg_va: list[tuple[str, str]],
    device: torch.device,
    max_pairs: int | None,
) -> tuple[float, float]:
    """
    模拟线上：槽 0 先写入 ``a``，再 ingest ``b``。

    Returns
    -------
    merge_recall_on_adjacent, new_slot_rate_on_random
        相邻行对上 **merge 比例**（越高越好）；随机对上 **new_slot 比例**（越高越好）。
    """
    from compactor.encoding import encode_texts

    def run_pairs(pairs: list[tuple[str, str]], expect_merge: bool) -> tuple[int, int]:
        n_ok, n_tot = 0, 0
        limit = len(pairs) if max_pairs is None else min(len(pairs), max_pairs)
        for i in range(limit):
            a, b = pairs[i]
            e_a = encode_texts([a], device=device).squeeze(0)
            e_b = encode_texts([b], device=device)  # (1, d)
            bank = MemoryBank(device=torch.device("cpu"))
            bank.append(e_a.cpu())
            res = ingest_embedding(e_b, bank, router, compactor, device=device)
            merged = not res.decision.is_new_slot
            if expect_merge:
                n_ok += int(merged)
            else:
                n_ok += int(res.decision.is_new_slot)
            n_tot += 1
        return n_ok, n_tot

    p_pos, t_pos = run_pairs(pos_va, expect_merge=True)
    p_neg, t_neg = run_pairs(neg_va, expect_merge=False)
    merge_recall = p_pos / max(t_pos, 1)
    new_slot_recall = p_neg / max(t_neg, 1)
    return merge_recall, new_slot_recall


@torch.no_grad()
def compactor_val_loss(
    compactor: MemoryCompactor,
    *,
    n_train: int,
    n_val: int,
    min_line_chars: int,
    seed: int,
    device: torch.device,
) -> float:
    st = load_encoder()
    _, val_ds = build_compactor_wikitext(
        st,
        n_train=n_train,
        n_val=n_val,
        min_line_chars=min_line_chars,
        seed=seed,
    )
    loader = DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=collate_compact)
    total, n = 0.0, 0
    for e_old, e_new, e_tgt in loader:
        e_old = e_old.to(device)
        e_new = e_new.to(device)
        e_tgt = e_tgt.to(device)
        pred = compactor(e_old, e_new)
        loss = (1.0 - (pred * e_tgt).sum(dim=-1)).mean().item()
        total += loss * e_old.shape[0]
        n += e_old.shape[0]
    return total / max(n, 1)


def main() -> None:
    root = _repo_root()
    p = argparse.ArgumentParser(
        description="Wikitext-aligned eval: Router + Compactor (same splits as training)."
    )
    p.add_argument("--n-train", type=int, default=8000, help="must match training if comparing checkpoints")
    p.add_argument("--n-val", type=int, default=1000, help="validation size (same protocol as train_* )")
    p.add_argument("--min-line-chars", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--router",
        type=Path,
        default=root / "compactor" / "checkpoints" / "router_wikitext.pt",
    )
    p.add_argument(
        "--compactor",
        type=Path,
        default=root / "compactor" / "checkpoints" / "compactor_wikitext.pt",
    )
    p.add_argument("--device", type=str, default=None)
    p.add_argument(
        "--max-streaming-pairs",
        type=int,
        default=None,
        help="cap per-class pairs for streaming sim (default: all val merge / neg pairs)",
    )
    args = p.parse_args()

    if not args.router.is_file():
        raise SystemExit(f"missing router checkpoint: {args.router}")
    if not args.compactor.is_file():
        raise SystemExit(f"missing compactor checkpoint: {args.compactor}")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    load_encoder()
    router = load_router(args.router, device)
    compactor = load_compactor(args.compactor, device)

    print("=== Wikitext eval (same weak labels as training) ===")
    print(f"device={device}, n_train={args.n_train}, n_val={args.n_val}, seed={args.seed}")

    acc = router_batch_accuracy(
        router,
        n_train=args.n_train,
        n_val=args.n_val,
        min_line_chars=args.min_line_chars,
        seed=args.seed,
        device=device,
    )
    print(f"\n[Router] batched val accuracy (merge vs new_slot CE): {acc:.4f}")

    c_loss = compactor_val_loss(
        compactor,
        n_train=args.n_train,
        n_val=args.n_val,
        min_line_chars=args.min_line_chars,
        seed=args.seed,
        device=device,
    )
    print(f"[Compactor] val mean 1−cos vs encode(a+' '+b): {c_loss:.6f}")

    _, _, pos_va, neg_va = wikitext_router_k1_text_splits(
        n_train=args.n_train,
        n_val=args.n_val,
        min_line_chars=args.min_line_chars,
        seed=args.seed,
    )
    mr, nr = router_streaming_metrics(
        router,
        compactor,
        pos_va=pos_va,
        neg_va=neg_va,
        device=device,
        max_pairs=args.max_streaming_pairs,
    )
    print(
        "\n[Router+ingest streaming] single-slot memory then ingest b:\n"
        f"  merge rate on adjacent-line val pairs: {mr:.4f}  (want high)\n"
        f"  new_slot rate on random-line val pairs: {nr:.4f}  (want high)"
    )

    _, pos_c_va = wikitext_compactor_text_splits(
        n_train=args.n_train,
        n_val=args.n_val,
        min_line_chars=args.min_line_chars,
        seed=args.seed,
    )
    print(
        f"\nNote: Compactor val pairs are a **different** slice of positives than Router val "
        f"(router val merge count={len(pos_va)}, compactor val merge count={len(pos_c_va)})."
    )


if __name__ == "__main__":
    main()
