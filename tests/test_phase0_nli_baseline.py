import pytest

from experiments.phase0_nli_baseline import find_entailment_index


def test_find_entailment_index_uses_semantic_label() -> None:
    assert find_entailment_index({0: "contradiction", 1: "entailment", 2: "neutral"}) == 1


def test_find_entailment_index_rejects_ambiguous_labels() -> None:
    with pytest.raises(ValueError, match="cannot identify"):
        find_entailment_index({0: "LABEL_0", 1: "LABEL_1"})
