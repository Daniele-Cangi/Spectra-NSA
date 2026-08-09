from collections import Counter

import pytest

from experiments.pilot_corpus import (
    generate_matched_pilot_orbits,
    generate_pilot_orbits,
)
from spectra_v3.interventions import ExpectedRelation


def test_pilot_generation_is_deterministic_and_balanced() -> None:
    left = generate_pilot_orbits(4, seed=17)
    right = generate_pilot_orbits(4, seed=17)

    assert [orbit.to_dict() for orbit in left] == [orbit.to_dict() for orbit in right]
    assert len({orbit.base_id for orbit in left}) == 4

    for orbit in left:
        assert orbit.metadata["development_only"] is True
        assert len(orbit.interventions) == 12
        families = Counter(item.family for item in orbit.interventions)
        assert set(families.values()) == {2}
        relations = Counter(item.expected_relation for item in orbit.interventions)
        assert relations == {
            ExpectedRelation.PRESERVE: 8,
            ExpectedRelation.CHANGE: 4,
        }
        assert len({item.transformed_text for item in orbit.interventions}) == 12


def test_pilot_generation_rejects_invalid_count() -> None:
    with pytest.raises(ValueError, match="positive"):
        generate_pilot_orbits(0, seed=1)


def test_matched_pilot_uses_parallel_relevant_and_control_fields() -> None:
    orbit = generate_matched_pilot_orbits(1, seed=23)[0]

    assert orbit.metadata["task_scope"] == "shipment_event"
    assert orbit.context_text.startswith("A shipment of")
    assert orbit.source == "synthetic-token-matched-pilot-v2"
    assert len(orbit.interventions) == 12
    assert all(item.metadata["matched_design"] is True for item in orbit.interventions)
    relations = Counter(item.expected_relation for item in orbit.interventions)
    assert relations == {
        ExpectedRelation.PRESERVE: 8,
        ExpectedRelation.CHANGE: 4,
    }
