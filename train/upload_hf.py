#!/usr/bin/env python3
"""
Upload a vec2text training output folder (or a single checkpoint-* directory) to the Hugging Face Hub.

Requires: pip install huggingface_hub
Auth: set HF_TOKEN, or run `huggingface-cli login` once.

Examples:
  # Default: upload latest inversion checkpoint to bob013/minilm-vec2text-inversion
  uv run python train/upload_hf.py --use-latest-checkpoint

  # Explicit repo and folder
  uv run python train/upload_hf.py \\
    --repo-id bob013/minilm-vec2text-inversion \\
    --local-dir outputs/vec2text/minilm/inversion \\
    --use-latest-checkpoint

  # Upload inversion + corrector to two repos
  uv run python train/upload_hf.py both --use-latest-checkpoint
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from train._paths import default_corrector_dir, default_inversion_dir, repo_root

HF_USER_DEFAULT = "bob013"


def _resolve_upload_dir(local_dir: Path, use_latest_checkpoint: bool) -> Path:
    local_dir = local_dir.resolve()
    if not local_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {local_dir}")
    if not use_latest_checkpoint:
        return local_dir
    # If user pointed at a checkpoint-* folder, use it directly.
    if local_dir.name.startswith("checkpoint-"):
        return local_dir
    from transformers.trainer_utils import get_last_checkpoint

    last = get_last_checkpoint(str(local_dir))
    if last is not None:
        return Path(last)
    # No checkpoint-*; upload root (e.g. contains pytorch_model.bin at end of training)
    return local_dir


def _upload_folder(*, repo_id: str, folder_path: Path, private: bool, commit_message: str) -> None:
    try:
        from huggingface_hub import HfApi, whoami
    except ImportError as e:
        raise SystemExit(
            "Missing huggingface_hub. Install with: uv pip install huggingface_hub"
        ) from e

    api = HfApi()
    try:
        whoami()
    except Exception:
        if not os.environ.get("HF_TOKEN") and not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
            print(
                "Warning: not logged in and no HF_TOKEN/HUGGING_FACE_HUB_TOKEN set. "
                "Run `huggingface-cli login` or export HF_TOKEN.",
                file=sys.stderr,
            )

    api.create_repo(repo_id, repo_type="model", exist_ok=True, private=private)
    print(f"Uploading {folder_path} -> https://huggingface.co/{repo_id}")
    api.upload_folder(
        folder_path=str(folder_path),
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_message,
    )
    print("Done.")


def _cmd_single(args: argparse.Namespace) -> None:
    os.chdir(repo_root())
    folder = _resolve_upload_dir(Path(args.local_dir), args.use_latest_checkpoint)
    _upload_folder(
        repo_id=args.repo_id,
        folder_path=folder,
        private=args.private,
        commit_message=args.commit_message,
    )


def _cmd_both(args: argparse.Namespace) -> None:
    os.chdir(repo_root())
    inv_root = Path(args.inversion_dir)
    cor_root = Path(args.corrector_dir)
    inv_upload = _resolve_upload_dir(inv_root, args.use_latest_checkpoint)
    cor_upload = _resolve_upload_dir(cor_root, args.use_latest_checkpoint)
    _upload_folder(
        repo_id=args.inversion_repo,
        folder_path=inv_upload,
        private=args.private,
        commit_message=args.commit_message + " (inversion)",
    )
    _upload_folder(
        repo_id=args.corrector_repo,
        folder_path=cor_upload,
        private=args.private,
        commit_message=args.commit_message + " (corrector)",
    )


def main() -> None:
    argv = sys.argv[1:]
    if argv and argv[0] == "both":
        sys.argv = [sys.argv[0]] + argv[1:]
        _main_both()
        return
    _main_single()


def _main_single() -> None:
    parser = argparse.ArgumentParser(description="Upload one vec2text folder to a Hugging Face model repo.")
    parser.add_argument(
        "--repo-id",
        type=str,
        default=f"{HF_USER_DEFAULT}/minilm-vec2text-inversion",
        help="Hub model repo id (default: %(default)s)",
    )
    parser.add_argument(
        "--local-dir",
        type=str,
        default=str(default_inversion_dir()),
        help="Checkpoint folder or training output_dir",
    )
    parser.add_argument(
        "--use-latest-checkpoint",
        action="store_true",
        help="If local-dir is output_dir, upload latest checkpoint-* instead of whole tree",
    )
    parser.add_argument("--private", action="store_true", help="Create/use a private repo")
    parser.add_argument(
        "--commit-message",
        type=str,
        default="Upload vec2text MiniLM checkpoint",
    )
    args = parser.parse_args()
    _cmd_single(args)


def _main_both() -> None:
    parser = argparse.ArgumentParser(description="Upload inversion + corrector to two Hugging Face repos.")
    parser.add_argument(
        "--inversion-repo",
        type=str,
        default=f"{HF_USER_DEFAULT}/minilm-vec2text-inversion",
    )
    parser.add_argument(
        "--corrector-repo",
        type=str,
        default=f"{HF_USER_DEFAULT}/minilm-vec2text-corrector",
    )
    parser.add_argument(
        "--inversion-dir",
        type=str,
        default=str(default_inversion_dir()),
    )
    parser.add_argument(
        "--corrector-dir",
        type=str,
        default=str(default_corrector_dir()),
    )
    parser.add_argument("--use-latest-checkpoint", action="store_true")
    parser.add_argument("--private", action="store_true")
    parser.add_argument(
        "--commit-message",
        type=str,
        default="Upload vec2text MiniLM",
    )
    args = parser.parse_args()
    _cmd_both(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
