import pytest

from spectra_v3.token_diagnostics import (
    token_edit_distance,
    token_sequence_diagnostics,
)


def test_token_edit_distance_counts_substitution_insertion_and_deletion() -> None:
    assert token_edit_distance([1, 2, 3], [1, 4, 3]) == 1
    assert token_edit_distance([1, 2], [1, 2, 3]) == 1
    assert token_edit_distance([1, 2, 3], [1, 3]) == 1


def test_token_sequence_diagnostics_are_normalized_and_symmetric() -> None:
    diagnostic = token_sequence_diagnostics([1, 2, 3], [1, 4])
    reverse = token_sequence_diagnostics([1, 4], [1, 2, 3])

    assert diagnostic.edit_distance == 2
    assert diagnostic.normalized_edit_distance == pytest.approx(2 / 3)
    assert diagnostic.token_count_delta == -1
    assert diagnostic.token_id_jaccard == pytest.approx(1 / 4)
    assert reverse.edit_distance == diagnostic.edit_distance
