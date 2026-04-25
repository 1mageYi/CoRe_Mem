"""PersonaMem 32k benchmark data: download from HuggingFace and load efficiently.

Data layout on HF (bowen-upenn/PersonaMem):
  questions_32k.csv          -- 589 question rows
  shared_contexts_32k.jsonl  -- one JSON object per line: {shared_context_id: [msg, ...]}

Each shared context is referenced by multiple questions via shared_context_id;
questions slice it with end_index_in_shared_context.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Generator

from huggingface_hub import hf_hub_download

REPO_ID = "bowen-upenn/PersonaMem"
_CACHE_DIR = Path(__file__).parent / ".hf_cache"


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def ensure_data(cache_dir: Path = _CACHE_DIR) -> tuple[Path, Path]:
    """Download questions_32k.csv and shared_contexts_32k.jsonl if not cached.

    Returns (questions_path, shared_contexts_path).
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    q_path = Path(
        hf_hub_download(
            repo_id=REPO_ID,
            filename="questions_32k.csv",
            repo_type="dataset",
            local_dir=str(cache_dir),
        )
    )
    ctx_path = Path(
        hf_hub_download(
            repo_id=REPO_ID,
            filename="shared_contexts_32k.jsonl",
            repo_type="dataset",
            local_dir=str(cache_dir),
        )
    )
    return q_path, ctx_path


# ---------------------------------------------------------------------------
# JSONL index  (scan once, then seek by offset)
# ---------------------------------------------------------------------------

def build_context_index(jsonl_path: Path) -> dict[str, int]:
    """Return {shared_context_id: byte_offset} by scanning the JSONL once."""
    index: dict[str, int] = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        while True:
            offset = f.tell()
            line = f.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            key = next(iter(json.loads(line).keys()))
            index[key] = offset
    return index


def load_context_at(jsonl_path: Path, offset: int) -> list[dict]:
    """Seek to offset in JSONL and return the message list for that context."""
    with open(jsonl_path, "r", encoding="utf-8") as f:
        f.seek(offset)
        item = json.loads(f.readline())
    return next(iter(item.values()))


# ---------------------------------------------------------------------------
# Main iterator
# ---------------------------------------------------------------------------

def iter_questions_with_context(
    q_path: Path,
    ctx_path: Path,
) -> Generator[tuple[dict, list[dict]], None, None]:
    """Yield (row_dict, sliced_context_messages) for every question row.

    sliced_context_messages is context[:end_index_in_shared_context] —
    the same slice the original inference.py uses before passing to the LLM.
    Consecutive questions sharing the same shared_context_id reuse the cached
    context object (avoids repeated disk reads).
    """
    index = build_context_index(ctx_path)
    prev_sid: str | None = None
    prev_full_context: list[dict] | None = None

    with open(q_path, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = row["shared_context_id"]
            end = int(row["end_index_in_shared_context"])

            if sid != prev_sid:
                prev_full_context = load_context_at(ctx_path, index[sid])
                prev_sid = sid

            yield row, prev_full_context[:end]  # type: ignore[index]


def count_questions(q_path: Path) -> int:
    """Return the number of data rows (header excluded)."""
    with open(q_path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f) - 1
