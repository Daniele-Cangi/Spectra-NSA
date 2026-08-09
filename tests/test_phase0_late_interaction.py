import pytest
import torch

from experiments.phase0_late_interaction import maxsim_score


def test_maxsim_is_one_for_identical_orthogonal_token_sets() -> None:
    tokens = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    assert maxsim_score(tokens, tokens) == pytest.approx(1.0)


def test_maxsim_rejects_dimension_mismatch() -> None:
    with pytest.raises(ValueError, match="dimensions"):
        maxsim_score(torch.ones((2, 3)), torch.ones((2, 4)))
