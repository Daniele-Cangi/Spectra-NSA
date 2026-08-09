import pytest

from spectra_v3.lexical_variables import (
    lexical_state_distance,
    lexical_variable_state,
    select_relevant_span,
)

QUERY = "A shipment of 1,000 units to Berlin was approved by Atlas in 2025."
BASE = (
    "Analyst Alice filed report 1, which was approved for publication in 2018. "
    "A shipment of 1,000 units to Berlin was approved by Atlas in 2025."
)


def test_select_relevant_span_ignores_irrelevant_report() -> None:
    span = select_relevant_span(QUERY, BASE)

    assert span.startswith("A shipment")
    assert "Analyst" not in span


@pytest.mark.parametrize(
    "transformed,coordinate",
    [
        (BASE.replace("Atlas", "Delta"), "entity_coverage"),
        (BASE.replace("1,000", "1,500"), "numeric_coverage"),
        (BASE.replace("in 2025", "in 2026"), "numeric_coverage"),
        (BASE.replace("was approved by", "was not approved by"), "polarity_alignment"),
    ],
)
def test_lexical_state_exposes_relevant_variable_change(
    transformed: str, coordinate: str
) -> None:
    base_state = lexical_variable_state(QUERY, BASE)
    changed_state = lexical_variable_state(QUERY, transformed)

    assert getattr(changed_state, coordinate) < getattr(base_state, coordinate)
    assert lexical_state_distance(base_state, changed_state) > 0.0


def test_irrelevant_control_does_not_move_selected_state() -> None:
    base_state = lexical_variable_state(QUERY, BASE)
    control = BASE.replace(
        "was approved for publication", "was not approved for publication"
    )
    control_state = lexical_variable_state(QUERY, control)

    assert lexical_state_distance(base_state, control_state) == 0.0


def test_numeric_formatting_is_canonicalized() -> None:
    base_state = lexical_variable_state(QUERY, BASE)
    formatting_control = BASE.replace("1,000", "1000")
    control_state = lexical_variable_state(QUERY, formatting_control)

    assert lexical_state_distance(base_state, control_state) == 0.0
