from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


def token_edit_distance(left: Sequence[int], right: Sequence[int]) -> int:
    """Levenshtein distance over token IDs using linear memory."""

    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_token in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_token in enumerate(right, start=1):
            insertion = current[right_index - 1] + 1
            deletion = previous[right_index] + 1
            substitution = previous[right_index - 1] + (left_token != right_token)
            current.append(min(insertion, deletion, substitution))
        previous = current
    return previous[-1]


@dataclass(frozen=True)
class TokenDiagnostics:
    edit_distance: int
    normalized_edit_distance: float
    token_count_delta: int
    token_id_jaccard: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "token_edit_distance": self.edit_distance,
            "normalized_token_edit_distance": self.normalized_edit_distance,
            "token_count_delta": self.token_count_delta,
            "token_id_jaccard": self.token_id_jaccard,
        }


def token_sequence_diagnostics(
    base_ids: Sequence[int], transformed_ids: Sequence[int]
) -> TokenDiagnostics:
    distance = token_edit_distance(base_ids, transformed_ids)
    scale = max(len(base_ids), len(transformed_ids), 1)
    left = set(base_ids)
    right = set(transformed_ids)
    union = left | right
    jaccard = len(left & right) / len(union) if union else 1.0
    return TokenDiagnostics(
        edit_distance=distance,
        normalized_edit_distance=distance / scale,
        token_count_delta=len(transformed_ids) - len(base_ids),
        token_id_jaccard=jaccard,
    )
