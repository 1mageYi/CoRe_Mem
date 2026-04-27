from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files


@dataclass(slots=True)
class PermaEvalSample:
    user_id: str
    task_id: str
    task_type: int
    question: str
    options_text: str
    gold_label: str
    context_messages: list[dict]


def _download_perma_file(filename: str) -> Path:
    path = hf_hub_download("ustclsc/PERMA", repo_type="dataset", filename=filename)
    return Path(path)


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_perma_user_samples(user_id: str, *, split_variant: str = "c") -> list[PermaEvalSample]:
    """
    Build evaluation samples for one PERMA user from:
      - tasks/{user}/input_data_{variant}.json (context source)
      - evaluation/{user}/meta/overall/*.json (question/options/label)
    """
    tasks_file = _download_perma_file(f"tasks/{user_id}/input_data_{split_variant}.json")
    task_data = _load_json(tasks_file)
    rows = task_data.get("overall", [])

    by_key: dict[tuple[str, int], dict] = {}
    for row in rows:
        key = (row["task_id"], int(row.get("type", 1)))
        by_key[key] = row

    all_files = list_repo_files("ustclsc/PERMA", repo_type="dataset")
    eval_prefix = f"evaluation/{user_id}/meta/overall/"
    eval_files = sorted([f for f in all_files if f.startswith(eval_prefix) and f.endswith(".json")])

    samples: list[PermaEvalSample] = []
    for ef in eval_files:
        epath = _download_perma_file(ef)
        meta = _load_json(epath)
        key = (meta["task_id"], int(meta["task_type"]))
        task_row = by_key.get(key)
        if not task_row:
            continue

        messages = _flatten_context_messages(task_row.get("context", []))
        samples.append(
            PermaEvalSample(
                user_id=user_id,
                task_id=meta["task_id"],
                task_type=int(meta["task_type"]),
                question=meta["question"],
                options_text=meta["options"],
                gold_label=str(meta["gold_label"]).strip().upper(),
                context_messages=messages,
            )
        )
    return samples


def _flatten_context_messages(context_obj: list) -> list[dict]:
    """
    PERMA context format is a nested list:
      [ [ [ {role, content}, ...], "timestamp"], ... ]
    """
    out: list[dict] = []
    for session in context_obj:
        if not isinstance(session, list) or not session:
            continue
        turns = session[0]
        if not isinstance(turns, list):
            continue
        for msg in turns:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            content = (msg.get("content") or "").strip()
            if role in ("user", "assistant") and content:
                out.append({"role": role, "content": content})
    return out
