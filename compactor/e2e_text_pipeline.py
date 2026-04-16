"""
端到端：文本 → gtr-t5-base 向量 → Router + Compactor 写入记忆 → vec2text 还原为文本。

依赖与 ``vec2text_test/demo.py`` 相同（vec2text + gtr + CUDA 推荐）。用于**主观**查看 merge 后单向量是否仍可解码为可读摘要。

用法（在仓库根 ``CoRe_Mem/``）::

    export PYTHONPATH=.:src
    python -m compactor.e2e_text_pipeline \\
      --memory \"The user prefers dark mode.\" \\
      --new \"They also use vim keybindings.\"

若未传 ``--router`` / ``--compactor``，会尝试加载 ``compactor/checkpoints/*_wikitext.pt``；不存在则报错。

**为何两句「语义相关」仍被判 new_slot**：Router 只在 **Wikitext 相邻行 vs 随机行** 上训练；任意英文对话/偏好句属于 **分布外**。要看与训练一致的行为，请用 ``python -m compactor.eval_wikitext_pipeline``（整段验证集指标 + 单槽 ingest 统计）。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import transformers

from compactor.encoding import encode_texts, load_encoder
from compactor.fusion import MemoryCompactor
from compactor.inference import ingest_embedding
from compactor.memory_bank import MemoryBank
from compactor.router import RouterV1

import vec2text


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


def build_vec2text_corrector(device: torch.device):
    """Same stack as ``vec2text_test/demo.py`` (gtr-t5-base compatible)."""
    inversion_model = vec2text.models.InversionModel.from_pretrained(
        "ielabgroup/vec2text_gtr-base-st_inversion"
    ).to(device)
    corrector_model = vec2text.models.CorrectorEncoderModel.from_pretrained(
        "ielabgroup/vec2text_gtr-base-st_corrector"
    ).to(device)
    inversion_trainer = vec2text.trainers.InversionTrainer(
        model=inversion_model,
        train_dataset=None,
        eval_dataset=None,
        data_collator=transformers.DataCollatorForSeq2Seq(
            inversion_model.tokenizer,
            label_pad_token_id=-100,
        ),
    )
    corrector_model.config.dispatch_batches = None
    corrector = vec2text.trainers.Corrector(
        model=corrector_model,
        inversion_trainer=inversion_trainer,
        args=None,
        data_collator=vec2text.collator.DataCollatorForCorrection(
            tokenizer=inversion_trainer.model.tokenizer
        ),
    )
    return corrector


@torch.no_grad()
def decode_embeddings(
    embeddings: torch.Tensor,
    corrector,
    *,
    num_steps: int = 20,
    sequence_beam_width: int = 4,
) -> list[str]:
    """``embeddings``: (n, d) on same device as corrector."""
    return vec2text.invert_embeddings(
        embeddings=embeddings,
        corrector=corrector,
        num_steps=num_steps,
        sequence_beam_width=sequence_beam_width,
    )


def main():
    root = _repo_root()
    p = argparse.ArgumentParser(description="E2E: text → embed → router/compactor → vec2text")
    p.add_argument("--memory", type=str, required=True, help="Initial memory sentence (fills slot 0 after first encode)")
    p.add_argument("--new", dest="new_text", type=str, required=True, help="Incoming sentence to route/merge")
    p.add_argument(
        "--router",
        type=Path,
        default=root / "compactor" / "checkpoints" / "router_wikitext.pt",
        help="router checkpoint",
    )
    p.add_argument(
        "--compactor",
        type=Path,
        default=root / "compactor" / "checkpoints" / "compactor_wikitext.pt",
        help="compactor checkpoint",
    )
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--num-steps", type=int, default=20)
    p.add_argument("--beam", type=int, default=4)
    args = p.parse_args()

    if not args.router.is_file():
        raise SystemExit(f"missing router checkpoint: {args.router}")
    if not args.compactor.is_file():
        raise SystemExit(f"missing compactor checkpoint: {args.compactor}")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type != "cuda":
        print("warning: vec2text demo is intended for CUDA; CPU will be very slow or OOM.")

    load_encoder()
    router = load_router(args.router, device)
    compactor = load_compactor(args.compactor, device)

    # Encode (CPU tensors from encode_texts; move to device)
    e_mem = encode_texts([args.memory], device=device)
    e_new = encode_texts([args.new_text], device=device)

    bank = MemoryBank(device=torch.device("cpu"))
    bank.append(e_mem.squeeze(0).cpu())

    print("=== inputs ===")
    print(f"memory text: {args.memory!r}")
    print(f"new text:    {args.new_text!r}")

    res = ingest_embedding(e_new, bank, router, compactor, device=device)
    print("\n=== router/compactor ===")
    print(f"new_slot: {res.decision.is_new_slot}, slot_index: {res.decision.slot_index}")
    print(f"memory_updated: {res.memory_updated}, new_slot_appended: {res.new_slot_appended}")

    print("\n=== vec2text decode (each slot) ===")
    corrector = build_vec2text_corrector(device)
    for i in range(len(bank)):
        emb = bank.get(i).to(device).unsqueeze(0)
        out = decode_embeddings(
            emb,
            corrector,
            num_steps=args.num_steps,
            sequence_beam_width=args.beam,
        )
        print(f"\n--- slot {i} ---\n{out[0]}")


if __name__ == "__main__":
    main()
