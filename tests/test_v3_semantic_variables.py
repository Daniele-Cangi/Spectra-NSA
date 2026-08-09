import pytest

from spectra_v3.interventions import ExpectedRelation
from spectra_v3.semantic_variables import (
    SemanticAxis,
    SemanticVariableChange,
    relational_state_from_probabilities,
)


def test_semantic_change_derives_task_relative_relation() -> None:
    critical = SemanticVariableChange(
        axis=SemanticAxis.POLARITY,
        frame_id="shipment",
        query_relevant=True,
        value_changed=True,
        before_value="affirmed",
        after_value="negated",
    )
    control = SemanticVariableChange(
        axis=SemanticAxis.POLARITY,
        frame_id="report",
        query_relevant=False,
        value_changed=True,
        before_value="affirmed",
        after_value="negated",
    )

    assert critical.expected_relation == ExpectedRelation.CHANGE
    assert control.expected_relation == ExpectedRelation.PRESERVE
    critical.validate_relation(ExpectedRelation.CHANGE)
    with pytest.raises(ValueError, match="contract disagrees"):
        control.validate_relation(ExpectedRelation.CHANGE)


def test_relational_state_has_interpretable_coordinates() -> None:
    base = relational_state_from_probabilities(
        {"contradiction": 0.1, "entailment": 0.8, "neutral": 0.1}
    )
    changed = relational_state_from_probabilities(
        {"contradiction": 0.8, "entailment": 0.1, "neutral": 0.1}
    )

    assert base.support == pytest.approx(0.7)
    assert base.commitment == pytest.approx(0.9)
    assert changed.delta_from(base)["support_delta"] == pytest.approx(-1.4)
    assert 0.0 < base.entropy < 1.0


def test_relational_state_rejects_unlabelled_distribution() -> None:
    with pytest.raises(ValueError, match="cannot identify"):
        relational_state_from_probabilities(
            {"LABEL_0": 0.2, "LABEL_1": 0.3, "LABEL_2": 0.5}
        )
