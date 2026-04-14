import torch
import transformers
from sentence_transformers import SentenceTransformer

import vec2text

DEVICE = "cuda"

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA 不可用。请确认：\n"
        "  1. 安装了与驱动匹配的 PyTorch：python -m pip install torch --index-url https://download.pytorch.org/whl/cu124\n"
        "  2. Transformers 已降版：python -m pip install 'transformers==4.44.2'"
    )

inversion_model = vec2text.models.InversionModel.from_pretrained(
    "ielabgroup/vec2text_gtr-base-st_inversion"
).to(DEVICE)
corrector_model = vec2text.models.CorrectorEncoderModel.from_pretrained(
    "ielabgroup/vec2text_gtr-base-st_corrector"
).to(DEVICE)

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

embedder = SentenceTransformer("sentence-transformers/gtr-t5-base")
texts = [
    "Jack Morris is a PhD student at Cornell Tech in New York City",
    "It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness, it was the epoch of belief, it was the epoch of incredulity",
]
embeddings = embedder.encode(texts, convert_to_tensor=True).to(DEVICE).clone()

reconstructed = vec2text.invert_embeddings(
    embeddings=embeddings,
    corrector=corrector,
    num_steps=20,
    sequence_beam_width=4,  # 每步保留 4 个候选，更大搜索空间；显存不足可改为 0
)
for i, (src, out) in enumerate(zip(texts, reconstructed)):
    print(f"\n--- [{i}] ---\n原文:\n{src}\n\n还原:\n{out}")
