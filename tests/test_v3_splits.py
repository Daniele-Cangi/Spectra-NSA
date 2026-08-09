import pytest

from spectra_v3.splits import validate_disjoint_groups


def test_disjoint_group_validation_accepts_clean_split() -> None:
    validate_disjoint_groups({"train": ["a", "b"], "test": ["c"]})


def test_disjoint_group_validation_rejects_leakage() -> None:
    with pytest.raises(ValueError, match="group leakage"):
        validate_disjoint_groups({"train": ["shared"], "test": ["shared"]})
