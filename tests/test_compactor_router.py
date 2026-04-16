"""Smoke tests for compactor Router + MemoryCompactor (no HF download)."""

from __future__ import annotations

import pytest
import torch

from compactor.fusion import MemoryCompactor
from compactor.inference import ingest_embedding
from compactor.memory_bank import MemoryBank
from compactor.router import RouterV1, apply_cand_mask, predict_decision


def test_router_forward_and_mask():
    d, k, b = 32, 4, 8
    m = RouterV1(embed_dim=d, hidden_dim=64)
    e_new = torch.randn(b, d)
    e_cands = torch.randn(b, k, d)
    mask = torch.tensor([[True, True, False, False]] * b)
    logits = m(e_new, e_cands, cand_mask=mask)
    assert logits.shape == (b, k + 1)
    pred, probs = predict_decision(logits)
    assert pred.shape == (b,)
    assert probs.shape == logits.shape


def test_compactor_normalize():
    d = 32
    c = MemoryCompactor(embed_dim=d, hidden_dim=64)
    a = torch.nn.functional.normalize(torch.randn(4, d), dim=-1)
    b = torch.nn.functional.normalize(torch.randn(4, d), dim=-1)
    out = c(a, b)
    assert torch.allclose(out.norm(dim=-1), torch.ones(4), atol=1e-5)


def test_ingest_empty_bank_cpu():
    d = 32
    router = RouterV1(embed_dim=d, hidden_dim=64)
    compactor = MemoryCompactor(embed_dim=d, hidden_dim=64)
    bank = MemoryBank(device=torch.device("cpu"))
    e = torch.nn.functional.normalize(torch.randn(d), dim=-1)
    res = ingest_embedding(e, bank, router, compactor, device=torch.device("cpu"))
    assert res.new_slot_appended
    assert not res.memory_updated
    assert len(bank) == 1


def test_apply_cand_mask_infs():
    logits = torch.zeros(2, 4)  # K=3 + new_slot
    mask = torch.tensor([[True, False, False], [True, True, False]])
    out = apply_cand_mask(logits, mask)
    assert out[0, 1] < -1e5
