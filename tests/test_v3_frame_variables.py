import pytest

from spectra_v3.frame_variables import (
    extract_semantic_frame,
    fixed_frame_distance,
    frame_compatibility,
)


def _exact(left: str, right: str) -> float:
    return float(left.casefold() == right.casefold())


def test_extracts_active_and_passive_frames_with_the_same_direction() -> None:
    query = "Aster Labs approved Project Juniper."
    active = extract_semantic_frame(query, query)
    passive = extract_semantic_frame(
        query,
        "Project Juniper was authorized by Aster Labs.",
    )

    assert active.actor == passive.actor == "Aster Labs"
    assert active.patient == passive.patient == "Project Juniper"
    assert active.predicate == "approved"
    assert passive.predicate == "authorized"
    assert passive.voice == "passive"
    assert passive.reliability == 1.0


def test_scope_is_attached_to_the_relation_not_a_trailing_adjunct() -> None:
    query = "Aster Labs approved Project Juniper."
    negated = extract_semantic_frame(
        query, "Aster Labs did not approve Project Juniper."
    )
    harmless = extract_semantic_frame(
        query, "Aster Labs approved Project Juniper without delay."
    )

    assert negated.predicate == "approve"
    assert negated.polarity == "negated"
    assert harmless.polarity == "affirmed"


def test_argument_and_direction_coordinates_are_separate() -> None:
    query = extract_semantic_frame(
        "Aster Labs approved Project Juniper.",
        "Aster Labs approved Project Juniper.",
    )
    swapped = extract_semantic_frame(
        query.span, "Project Juniper approved Aster Labs."
    )
    state = frame_compatibility(query, swapped, _exact)

    assert state.argument_alignment == 1.0
    assert state.direction_alignment == 0.0


def test_fixed_frame_distance_is_monotone_and_checks_coordinates() -> None:
    drops = {
        "predicate_alignment_drop": 0.4,
        "argument_alignment_drop": 0.0,
        "direction_alignment_drop": 1.0,
        "scope_alignment_drop": 0.0,
        "modality_alignment_drop": 0.0,
    }
    assert fixed_frame_distance(drops) == 1.0
    with pytest.raises(ValueError, match="missing frame drops"):
        fixed_frame_distance({})
