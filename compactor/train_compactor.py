"""Train MemoryCompactor with cosine loss. Example: ``python -m compactor.train_compactor --synthetic``."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from compactor.config import embed_dim as cfg_embed_dim
from compactor.datasets import make_synthetic_compactor_data
from compactor.encoding import load_encoder
from compactor.fusion import MemoryCompactor


def compact_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean 1 - cos(pred, target) for row-wise unit vectors."""
    return (1.0 - (pred * target).sum(dim=-1)).mean()


def collate_compact(batch):
    a = torch.stack([b[0] for b in batch], dim=0)
    b_ = torch.stack([b[1] for b in batch], dim=0)
    t = torch.stack([b[2] for b in batch], dim=0)
    return a, b_, t


def train_epoch(model, loader, optim, device, scaler):
    model.train()
    total = 0.0
    for e_old, e_new, e_tgt in loader:
        e_old = e_old.to(device)
        e_new = e_new.to(device)
        e_tgt = e_tgt.to(device)
        optim.zero_grad(set_to_none=True)
        pred = model(e_old, e_new)
        loss = compact_loss(pred, e_tgt)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss.backward()
            optim.step()
        total += loss.item() * e_old.shape[0]
    return total / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total = 0.0
    for e_old, e_new, e_tgt in loader:
        e_old = e_old.to(device)
        e_new = e_new.to(device)
        e_tgt = e_tgt.to(device)
        pred = model(e_old, e_new)
        total += compact_loss(pred, e_tgt).item() * e_old.shape[0]
    return total / len(loader.dataset)


def main():
    p = argparse.ArgumentParser(description="Train MemoryCompactor")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument(
        "--wikitext",
        action="store_true",
        help="Wikitext-2 adjacent lines + gtr-t5-base; teacher = encode(a + ' ' + b)",
    )
    p.add_argument("--n-train", type=int, default=512)
    p.add_argument("--n-val", type=int, default=128)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--save", type=Path, default=None)
    p.add_argument("--min-line-chars", type=int, default=20, help="Wikitext weak supervision (see weak_supervision.py)")
    args = p.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    if args.synthetic and args.wikitext:
        raise SystemExit("choose only one of --synthetic or --wikitext")

    if args.synthetic:
        d = 64
        from compactor.config import set_embed_dim

        set_embed_dim(d)
        train_ds = make_synthetic_compactor_data(n=args.n_train, d=d, seed=1)
        val_ds = make_synthetic_compactor_data(n=args.n_val, d=d, seed=2)
    elif args.wikitext:
        from compactor.weak_supervision import build_compactor_wikitext

        st_model = load_encoder()
        d = cfg_embed_dim()
        train_ds, val_ds = build_compactor_wikitext(
            st_model,
            n_train=args.n_train,
            n_val=args.n_val,
            min_line_chars=args.min_line_chars,
            seed=42,
        )
    else:
        raise SystemExit("Specify --synthetic (smoke) or --wikitext (gtr + merged-sentence teacher).")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, collate_fn=collate_compact)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, collate_fn=collate_compact)

    hidden = args.hidden or d
    model = MemoryCompactor(embed_dim=d, hidden_dim=hidden).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    for ep in range(1, args.epochs + 1):
        tr = train_epoch(model, train_loader, optim, device, scaler)
        va = eval_epoch(model, val_loader, device)
        print(f"epoch {ep}: train_loss={tr:.6f} val_loss={va:.6f}")

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"state_dict": model.state_dict(), "embed_dim": d, "hidden_dim": hidden},
            args.save,
        )
        print(f"saved {args.save}")


if __name__ == "__main__":
    main()
