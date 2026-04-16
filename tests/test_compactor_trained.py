"""
Tests for **trained** Router / Compactor checkpoints (optional).

- **Fast** (default): if ``compactor/checkpoints/*.pt`` exist, load weights and run forward
  (shape, finite, unit norm for compactor output).
- **Slow** (``-m slow``): rebuild a small Wikitext val split and check accuracy / loss
  against loose thresholds (needs gtr encoder + network cache).

How to judge training quality in general:

1. **Offline metrics** (already printed during ``train_*.py``): Router **val acc**, Compactor **val loss**.
2. **This file**: regression-style checks so a bad export or broken load fails CI.
3. **Downstream**: decode merged embeddings with vec2text (``vec2text_test/demo.py``) and read text quality — not automated here.

See ``compactor/DESIGN.md`` §5.4.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
ROUTER_CKPT = REPO_ROOT / "compactor" / "checkpoints" / "router_wikitext.pt"
COMPACTOR_CKPT = REPO_ROOT / "compactor" / "checkpoints" / "compactor_wikitext.pt"

has_checkpoints = ROUTER_CKPT.is_file() and COMPACTOR_CKPT.is_file()

skip_no_ckpt = pytest.mark.skipif(
    not has_checkpoints,
    reason="missing compactor/checkpoints/router_wikitext.pt or compactor_wikitext.pt",
)


@pytest.fixture(scope="module")
def router_ckpt():
    if not ROUTER_CKPT.is_file():
        pytest.skip("no router checkpoint")
    return torch.load(ROUTER_CKPT, map_location="cpu", weights_only=False)


@pytest.fixture(scope="module")
def compactor_ckpt():
    if not COMPACTOR_CKPT.is_file():
        pytest.skip("no compactor checkpoint")
    return torch.load(COMPACTOR_CKPT, map_location="cpu", weights_only=False)


@skip_no_ckpt
def test_router_checkpoint_structure(router_ckpt):
    assert "state_dict" in router_ckpt
    assert "embed_dim" in router_ckpt
    assert "hidden_dim" in router_ckpt
    assert "k_candidates" in router_ckpt
    assert router_ckpt["k_candidates"] == 1
    sd = router_ckpt["state_dict"]
    assert any(k.startswith("pair_mlp.") for k in sd)
    assert any(k.startswith("reject_mlp.") for k in sd)


@skip_no_ckpt
def test_compactor_checkpoint_structure(compactor_ckpt):
    assert "state_dict" in compactor_ckpt
    assert "embed_dim" in compactor_ckpt
    assert "hidden_dim" in compactor_ckpt
    assert any(k.startswith("gate.") for k in compactor_ckpt["state_dict"])


@skip_no_ckpt
def test_router_and_compactor_same_embed_dim(router_ckpt, compactor_ckpt):
    assert router_ckpt["embed_dim"] == compactor_ckpt["embed_dim"]


@skip_no_ckpt
def test_router_loaded_forward(router_ckpt):
    from compactor.router import RouterV1

    d = router_ckpt["embed_dim"]
    h = router_ckpt["hidden_dim"]
    m = RouterV1(embed_dim=d, hidden_dim=h)
    m.load_state_dict(router_ckpt["state_dict"])
    m.eval()
    b, k = 4, 1
    e_new = F.normalize(torch.randn(b, d), dim=-1)
    e_cands = F.normalize(torch.randn(b, k, d), dim=-1)
    mask = torch.ones(b, k, dtype=torch.bool)
    with torch.no_grad():
        logits = m(e_new, e_cands, cand_mask=mask)
    assert logits.shape == (b, k + 1)
    assert torch.isfinite(logits).all()


@skip_no_ckpt
def test_compactor_loaded_forward(compactor_ckpt):
    from compactor.fusion import MemoryCompactor

    d = compactor_ckpt["embed_dim"]
    h = compactor_ckpt["hidden_dim"]
    c = MemoryCompactor(embed_dim=d, hidden_dim=h)
    c.load_state_dict(compactor_ckpt["state_dict"])
    c.eval()
    a = F.normalize(torch.randn(2, d), dim=-1)
    b = F.normalize(torch.randn(2, d), dim=-1)
    with torch.no_grad():
        out = c(a, b)
    assert out.shape == (2, d)
    assert torch.isfinite(out).all()
    assert torch.allclose(out.norm(dim=-1), torch.ones(2), atol=1e-4)


def _compact_loss(pred: torch.Tensor, target: torch.Tensor) -> float:
    with torch.no_grad():
        return (1.0 - (pred * target).sum(dim=-1)).mean().item()


@skip_no_ckpt
@pytest.mark.slow
def test_trained_router_val_accuracy_wikitext(router_ckpt):
    """Rebuild val set (same pipeline as training) and require accuracy above random."""
    if os.environ.get("COMPACTOR_SKIP_SLOW", "").strip() == "1":
        pytest.skip("COMPACTOR_SKIP_SLOW=1")

    from compactor.encoding import load_encoder
    from compactor.router import RouterV1
    from compactor.train_router import collate_router, eval_epoch
    from compactor.weak_supervision import build_router_k1_wikitext

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st = load_encoder()
    _, val_ds = build_router_k1_wikitext(
        st,
        n_train=512,
        n_val=800,
        min_line_chars=20,
        seed=42,
    )
    d = router_ckpt["embed_dim"]
    h = router_ckpt["hidden_dim"]
    m = RouterV1(embed_dim=d, hidden_dim=h).to(device)
    m.load_state_dict(router_ckpt["state_dict"])
    loader = DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=collate_router)
    acc = eval_epoch(m, loader, device)
    # Weak labels are noisy; trained model was ~0.95+ on similar val size
    assert acc >= 0.88, f"router val acc too low: {acc:.4f}"


@skip_no_ckpt
@pytest.mark.slow
def test_trained_compactor_val_loss_wikitext(compactor_ckpt):
    if os.environ.get("COMPACTOR_SKIP_SLOW", "").strip() == "1":
        pytest.skip("COMPACTOR_SKIP_SLOW=1")

    from compactor.encoding import load_encoder
    from compactor.fusion import MemoryCompactor
    from compactor.train_compactor import collate_compact
    from compactor.weak_supervision import build_compactor_wikitext

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st = load_encoder()
    _, val_ds = build_compactor_wikitext(
        st,
        n_train=512,
        n_val=800,
        min_line_chars=20,
        seed=42,
    )
    d = compactor_ckpt["embed_dim"]
    h = compactor_ckpt["hidden_dim"]
    c = MemoryCompactor(embed_dim=d, hidden_dim=h).to(device)
    c.load_state_dict(compactor_ckpt["state_dict"])
    c.eval()
    loader = DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=collate_compact)
    total, n = 0.0, 0
    with torch.no_grad():
        for e_old, e_new, e_tgt in loader:
            e_old = e_old.to(device)
            e_new = e_new.to(device)
            e_tgt = e_tgt.to(device)
            pred = c(e_old, e_new)
            total += _compact_loss(pred, e_tgt) * e_old.shape[0]
            n += e_old.shape[0]
    loss = total / max(n, 1)
    assert loss <= 0.035, f"compactor val loss too high: {loss:.6f}"
