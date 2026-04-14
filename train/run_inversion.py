#!/usr/bin/env python3
"""
Train vec2text zero-step (hypothesizer / inversion) model for
sentence-transformers/all-MiniLM-L6-v2 embeddings.

Uses vec2text programmatically (avoids broken CLI --help on some transformers versions).

Example:
  uv run python train/run_inversion.py --output-dir outputs/vec2text/minilm/inversion \\
    --dataset nq --use-less-data 50000 --per-device-train-batch-size 64

Distributed (multi-GPU):
  torchrun --nproc_per_node=4 train/run_inversion.py ...
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from train._nltk import ensure_nltk_punkt
from train._paths import default_inversion_dir, repo_root

MINILM_EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train vec2text inversion model for MiniLM-L6-v2.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(default_inversion_dir()),
        help="Directory for checkpoints and model_args.bin / data_args.bin",
    )
    parser.add_argument(
        "--dataset",
        choices=("nq", "msmarco"),
        default="nq",
        help="Training corpus (msmarco is larger; nq is lighter for debugging)",
    )
    parser.add_argument("--model-name-or-path", type=str, default="t5-base")
    parser.add_argument("--max-seq-length", type=int, default=128)
    parser.add_argument("--num-repeat-tokens", type=int, default=16)
    parser.add_argument("--per-device-train-batch-size", type=int, default=64)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=64)
    parser.add_argument("--num-train-epochs", type=float, default=30.0)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="If > 0, overrides num_train_epochs (early stop by steps)",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--warmup-steps", type=int, default=10_000)
    parser.add_argument("--eval-steps", type=int, default=20_000)
    parser.add_argument("--save-steps", type=int, default=2000)
    parser.add_argument("--max-eval-samples", type=int, default=500)
    parser.add_argument(
        "--use-less-data",
        type=int,
        default=-1,
        help="Cap train/val rows per split; -1 = use full dataset",
    )
    parser.add_argument("--bf16", action="store_true", help="Force bf16 on CUDA (default: auto)")
    parser.add_argument("--no-bf16", action="store_true", help="Disable bf16")
    parser.add_argument("--overwrite-output-dir", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--vec2text-cache",
        type=str,
        default="",
        help="If set, sets VEC2TEXT_CACHE for dataset tokenization cache",
    )
    args = parser.parse_args()

    from train._paths import default_vec2text_cache_dir

    cache_dir = Path(args.vec2text_cache) if args.vec2text_cache else default_vec2text_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["VEC2TEXT_CACHE"] = str(cache_dir.resolve())

    os.chdir(repo_root())
    ensure_nltk_punkt()

    if args.no_bf16:
        use_bf16 = False
    elif args.bf16:
        use_bf16 = torch.cuda.is_available()
    else:
        use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    from vec2text.experiments import experiment_from_args
    from vec2text.run_args import DataArguments, ModelArguments, TrainingArguments

    out_dir = os.path.abspath(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    model_args = ModelArguments(
        model_name_or_path=args.model_name_or_path,
        embedder_model_name=MINILM_EMBEDDER,
        max_seq_length=args.max_seq_length,
        num_repeat_tokens=args.num_repeat_tokens,
        embedder_no_grad=True,
        embedder_torch_dtype="float32",
        use_frozen_embeddings_as_input=False,
        embedder_fake_with_zeros=False,
        freeze_strategy="none",
    )
    data_args = DataArguments(
        dataset_name=args.dataset,
        max_eval_samples=args.max_eval_samples,
        use_less_data=args.use_less_data,
    )

    train_kw: dict = dict(
        output_dir=out_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        use_wandb=False,
        bf16=use_bf16,
        fp16=False,
        experiment="inversion",
        exp_group_name="minilm",
        exp_name="inversion",
        seed=args.seed,
        logging_steps=100,
        save_total_limit=3,
        load_best_model_at_end=False,
        report_to=[],
    )

    training_args = TrainingArguments(**train_kw)
    # transformers>=5 removed this ctor arg; vec2text still reads the attribute in experiments.py
    training_args.overwrite_output_dir = args.overwrite_output_dir

    experiment = experiment_from_args(model_args, data_args, training_args)
    experiment.run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
