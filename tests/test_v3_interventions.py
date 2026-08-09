import pytest

from spectra_v3.interventions import (
    ExpectedRelation,
    Intervention,
    InterventionOrbit,
    VerificationStatus,
    make_exact_replacement,
    make_surface_intervention,
)


def test_exact_replacement_changes_only_requested_occurrence() -> None:
    intervention = make_exact_replacement(
        "red then red",
        target="red",
        replacement="blue",
        family="entity",
        expected_relation=ExpectedRelation.CHANGE,
        generator_id="entity-color-v1",
        occurrence=1,
    )

    assert intervention.transformed_text == "red then blue"
    assert intervention.metadata["occurrence"] == 1


def test_exact_replacement_rejects_missing_occurrence() -> None:
    with pytest.raises(ValueError, match="does not exist"):
        make_exact_replacement(
            "one item",
            target="item",
            replacement="object",
            family="lexical",
            expected_relation=ExpectedRelation.PRESERVE,
            generator_id="synonym-v1",
            occurrence=1,
        )


def test_surface_intervention_has_preservation_contract() -> None:
    intervention = make_surface_intervention(
        "Mixed Case.",
        transformed_text="mixed case.",
        generator_id="lowercase-v1",
    )

    assert intervention.expected_relation is ExpectedRelation.PRESERVE
    assert intervention.verification_status is VerificationStatus.AUTOMATIC


def test_orbit_round_trip_and_base_validation() -> None:
    intervention = Intervention(
        base_text="The valve is open.",
        transformed_text="The valve is not open.",
        family="negation",
        expected_relation=ExpectedRelation.CHANGE,
        strength=1.0,
        generator_id="negation-v1",
        verification_status=VerificationStatus.HUMAN,
    )
    orbit = InterventionOrbit(
        base_id="valve-1",
        base_text=intervention.base_text,
        interventions=(intervention,),
        source="unit-test",
    )

    assert InterventionOrbit.from_dict(orbit.to_dict()) == orbit

    with pytest.raises(ValueError, match="orbit base_text"):
        InterventionOrbit(
            base_id="bad",
            base_text="A",
            interventions=(
                Intervention(
                    base_text="B",
                    transformed_text="C",
                    family="surface",
                    expected_relation=ExpectedRelation.PRESERVE,
                    strength=0.1,
                    generator_id="surface-v1",
                ),
            ),
        )
