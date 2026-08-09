import numpy as np
import pytest

from experiments.phase0_nli_baseline import (
    find_entailment_index,
    labelled_probabilities,
)


def test_find_entailment_index_uses_semantic_label() -> None:
    assert find_entailment_index({0: "contradiction", 1: "entailment", 2: "neutral"}) == 1


def test_find_entailment_index_rejects_ambiguous_labels() -> None:
    with pytest.raises(ValueError, match="cannot identify"):
        find_entailment_index({0: "LABEL_0", 1: "LABEL_1"})


def test_labelled_probabilities_uses_model_labels() -> None:
    result = labelled_probabilities(
        np.asarray([0.7, 0.2, 0.1]),
        {0: "CONTRADICTION", 1: "ENTAILMENT", 2: "NEUTRAL"},
    )

    assert result == {"contradiction": 0.7, "entailment": 0.2, "neutral": 0.1}
