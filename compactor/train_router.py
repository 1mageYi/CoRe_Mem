"""Train RouterV1 with CE on (K+1)-way logits.

Examples::

    python -m compactor.train_router --synthetic
    python -m compactor.train_router --wikitext --n-train 2000 --n-val 400 --epochs 3 --save router.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from compactor.config import embed_dim as cfg_embed_dim
from compactor.datasets import make_synthetic_router_data
from compactor.encoding import load_encoder
from compactor.router import RouterV1


def collate_router(batch):
    e_new = torch.stack([b[0] for b in batch], dim=0)
    e_cands = torch.stack([b[1] for b in batch], dim=0)
    cand_mask = torch.stack([b[2] for b in batch], dim=0)
    labels = torch.stack([b[3] for b in batch], dim=0)
    return e_new, e_cands, cand_mask, labels


def train_epoch(model, loader, optim, device, scaler):
    model.train()
    total, correct = 0.0, 0
    for e_new, e_cands, cand_mask, labels in loader:
        e_new = e_new.to(device)
        e_cands = e_cands.to(device)
        cand_mask = cand_mask.to(device)
        labels = labels.to(device)
        optim.zero_grad(set_to_none=True)
        logits = model(e_new, e_cands, cand_mask=cand_mask)
        loss = F.cross_entropy(logits, labels)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss.backward()
            optim.step()
        total += loss.item() * labels.shape[0]
        correct += (logits.argmax(dim=-1) == labels).sum().item()
    n = len(loader.dataset)
    return total / n, correct / n


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    correct = 0
    n = 0
    for e_new, e_cands, cand_mask, labels in loader:
        e_new = e_new.to(device)
        e_cands = e_cands.to(device)
        cand_mask = cand_mask.to(device)
        labels = labels.to(device)
        logits = model(e_new, e_cands, cand_mask=cand_mask)
        correct += (logits.argmax(dim=-1) == labels).sum().item()
        n += labels.shape[0]
    return correct / max(n, 1)


def main():
    p = argparse.ArgumentParser(description="Train RouterV1")
    p.add_argument("--synthetic", action="store_true", help="random unit vectors (smoke test)")
    p.add_argument(
        "--wikitext",
        action="store_true",
        help="weak supervision from Wikitext-2 + gtr-t5-base encodings (Router K=1, real training)",
    )
    p.add_argument("--k", type=int, default=3, help="number of candidate slots (not counting new_slot); forced to 1 with --wikitext")
    p.add_argument("--n-train", type=int, default=512)
    p.add_argument("--n-val", type=int, default=128)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--save", type=Path, default=None, help="path to save router.pt")
    p.add_argument(
        "--min-line-chars",
        type=int,
        default=20,
        help="Wikitext line length filter (weak supervision; see weak_supervision.py)",
    )
    args = p.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    if args.synthetic and args.wikitext:
        raise SystemExit("choose only one of --synthetic or --wikitext")

    k_meta: int

    if args.synthetic:
        d = 64
        from compactor.config import set_embed_dim

        set_embed_dim(d)
        train_ds = make_synthetic_router_data(n=args.n_train, k=args.k, d=d, seed=1, device=torch.device("cpu"))
        val_ds = make_synthetic_router_data(n=args.n_val, k=args.k, d=d, seed=2, device=torch.device("cpu"))
        k_meta = args.k
    elif args.wikitext:
        from compactor.weak_supervision import build_router_k1_wikitext

        st_model = load_encoder()
        d = cfg_embed_dim()
        train_ds, val_ds = build_router_k1_wikitext(
            st_model,
            n_train=args.n_train,
            n_val=args.n_val,
            min_line_chars=args.min_line_chars,
            seed=42,
        )
        k_meta = 1
    else:
        raise SystemExit("Specify --synthetic (smoke) or --wikitext (gtr-t5-base + wiki weak labels).")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, collate_fn=collate_router)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, collate_fn=collate_router)

    model = RouterV1(embed_dim=d, hidden_dim=args.hidden).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    for ep in range(1, args.epochs + 1):
        loss, acc = train_epoch(model, train_loader, optim, device, scaler)
        vacc = eval_epoch(model, val_loader, device)
        print(f"epoch {ep}: loss={loss:.4f} train_acc={acc:.4f} val_acc={vacc:.4f}")

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "embed_dim": d,
                "k_candidates": k_meta,
                "hidden_dim": args.hidden,
            },
            args.save,
        )
        print(f"saved {args.save}")


if __name__ == "__main__":
    main()
