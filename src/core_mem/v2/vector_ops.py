"""Small deterministic vector helpers for stage-2 skeletons."""

from __future__ import annotations

import hashlib
import math


def hash_text_to_vector(text: str, dimension: int) -> list[float]:
    if dimension <= 0:
        return []
    digest = hashlib.sha1(text.encode("utf-8")).digest()
    values = [((digest[idx % len(digest)] / 255.0) * 2.0) - 1.0 for idx in range(dimension)]
    norm = math.sqrt(sum(value * value for value in values)) or 1.0
    return [value / norm for value in values]


def dot_product(left: list[float], right: list[float]) -> float:
    size = max(len(left), len(right))
    padded_left = list(left) + [0.0] * (size - len(left))
    padded_right = list(right) + [0.0] * (size - len(right))
    return sum(lhs * rhs for lhs, rhs in zip(padded_left, padded_right))


def mean_vectors(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []
    width = max(len(vector) for vector in vectors)
    totals = [0.0] * width
    for vector in vectors:
        padded = list(vector) + [0.0] * (width - len(vector))
        totals = [left + right for left, right in zip(totals, padded)]
    count = float(len(vectors))
    return [value / count for value in totals]
