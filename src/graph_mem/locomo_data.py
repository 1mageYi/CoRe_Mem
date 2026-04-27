from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class LoCoMoTurn:
    """Single dialogue turn from a LoCoMo conversation."""

    dia_id: str           # e.g. "D1:3"
    speaker: str
    text: str
    session_idx: int      # 1-based session number
    turn_idx: int         # 1-based turn index within the session
    session_datetime: str = ""
    global_idx: int = 0   # sequential index across all sessions (1-based)


@dataclass
class LoCoMoQA:
    """A single QA pair from a LoCoMo conversation."""

    question: str
    answer: str           # always str (original ints are cast)
    evidence: list[str]   # list of dia_id references, e.g. ["D1:3", "D2:5"]
    category: int         # 1=single-hop, 2=temporal, 3=commonsense, 4=multi-hop, 5=adversarial


@dataclass
class LoCoMoConversation:
    """Parsed LoCoMo conversation with all turns and QA pairs."""

    sample_id: str
    speaker_a: str
    speaker_b: str
    turns: list[LoCoMoTurn]
    qa: list[LoCoMoQA]

    def __post_init__(self) -> None:
        self._turn_lookup: dict[str, LoCoMoTurn] = {t.dia_id: t for t in self.turns}

    def get_turn(self, dia_id: str) -> LoCoMoTurn | None:
        return self._turn_lookup.get(dia_id)

    def get_evidence_texts(self, evidence: list[str]) -> list[str]:
        """Return formatted text for a list of dia_id references."""
        texts = []
        for eid in evidence:
            t = self._turn_lookup.get(eid)
            if t is not None:
                texts.append(f"[{t.speaker}] {t.text}")
        return texts

    @property
    def n_sessions(self) -> int:
        if not self.turns:
            return 0
        return max(t.session_idx for t in self.turns)

    @property
    def n_turns(self) -> int:
        return len(self.turns)


# ---------------------------------------------------------------------------
# Time index convention
# ---------------------------------------------------------------------------
# time_index = (session_idx - 1) * 10_000 + turn_idx
# This gives each session a distinct "epoch" while preserving
# intra-session ordering, consistent with MemoryNode temporal edges.

def turn_time_index(turn: LoCoMoTurn) -> int:
    return (turn.session_idx - 1) * 10_000 + turn.turn_idx


# ---------------------------------------------------------------------------
# Internal parsing helpers
# ---------------------------------------------------------------------------

def _parse_conversation(raw: dict, *, exclude_adversarial: bool) -> LoCoMoConversation:
    conv_raw = raw["conversation"]
    speaker_a = conv_raw.get("speaker_a", "A")
    speaker_b = conv_raw.get("speaker_b", "B")

    all_turns: list[LoCoMoTurn] = []
    global_idx = 0
    sess_idx = 1

    while True:
        sess_key = f"session_{sess_idx}"
        if sess_key not in conv_raw:
            break
        dt_key = f"session_{sess_idx}_date_time"
        datetime_str = conv_raw.get(dt_key, "")

        for turn_data in conv_raw[sess_key]:
            dia_id = turn_data.get("dia_id", f"D{sess_idx}:{global_idx + 1}")
            parts = dia_id.split(":")
            turn_idx = int(parts[1]) if len(parts) == 2 and parts[1].isdigit() else global_idx + 1
            global_idx += 1
            all_turns.append(
                LoCoMoTurn(
                    dia_id=dia_id,
                    speaker=turn_data.get("speaker", ""),
                    text=turn_data.get("text", ""),
                    session_idx=sess_idx,
                    turn_idx=turn_idx,
                    session_datetime=datetime_str,
                    global_idx=global_idx,
                )
            )
        sess_idx += 1

    qa_list: list[LoCoMoQA] = []
    for q in raw.get("qa", []):
        if exclude_adversarial and q.get("category") == 5:
            continue
        qa_list.append(
            LoCoMoQA(
                question=q["question"],
                answer=str(q["answer"]),
                evidence=q.get("evidence", []),
                category=q.get("category", 1),
            )
        )

    return LoCoMoConversation(
        sample_id=raw.get("sample_id", ""),
        speaker_a=speaker_a,
        speaker_b=speaker_b,
        turns=all_turns,
        qa=qa_list,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_locomo(
    path: str | Path,
    *,
    exclude_adversarial: bool = True,
) -> list[LoCoMoConversation]:
    """Load all conversations from a LoCoMo JSON file.

    Args:
        path: Path to the LoCoMo JSON file (e.g. ``locomo10.json``).
        exclude_adversarial: If True, QA pairs with category 5
            (false-premise / adversarial) are dropped.

    Returns:
        List of parsed :class:`LoCoMoConversation` objects.
    """
    raw_list: list[dict] = json.loads(Path(path).read_text(encoding="utf-8"))
    return [_parse_conversation(r, exclude_adversarial=exclude_adversarial) for r in raw_list]


def filter_qa_by_categories(
    qa: list[LoCoMoQA],
    categories: set[int] | None = None,
) -> list[LoCoMoQA]:
    """Optionally restrict QA pairs to a subset of category IDs.

    LoCoMo categories:
      1 = single-hop
      2 = temporal reasoning
      3 = commonsense / multi-hop inference
      4 = multi-hop without commonsense
      5 = adversarial (false premise)
    """
    if categories is None:
        return qa
    return [q for q in qa if q.category in categories]
