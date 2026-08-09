import numpy as np
import pytest

from spectra_v3.features import extract_orbit_features, pair_response_features
from spectra_v3.interventions import ExpectedRelation, Intervention


def _intervention(
    transformed_text: str,
    relation: ExpectedRelation,
    family: str,
    strength: float,
) -> Intervention:
    return Intervention(
        base_text="base",
        transformed_text=transformed_text,
        family=family,
        expected_relation=relation,
        strength=strength,
        generator_id=f"{family}-{transformed_text}",
    )


def test_orbit_features_separate_relation_and_family_measurements() -> None:
    interventions = (
        _intervention("surface", ExpectedRelation.PRESERVE, "surface", 0.25),
        _intervention("negated", ExpectedRelation.CHANGE, "negation", 1.0),
    )
    features = extract_orbit_features(
        np.array([1.0, 0.0, 0.0]),
        np.array([[1.0, 0.1, 0.0], [1.0, 0.0, 1.0]]),
        interventions,
        rank=2,
    )

    assert set(features.relation_measurements) == {
        ExpectedRelation.PRESERVE,
        ExpectedRelation.CHANGE,
    }
    assert set(features.family_measurements) == {"negation", "surface"}
    assert features.features["response.critical_to_invariant_energy_ratio"] > 1.0


def test_pair_features_are_identity_consistent() -> None:
    interventions = (
        _intervention("a", ExpectedRelation.PRESERVE, "surface", 1.0),
        _intervention("b", ExpectedRelation.CHANGE, "negation", 1.0),
    )
    orbit = extract_orbit_features(
        np.array([1.0, 0.0, 0.0]),
        np.array([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]]),
        interventions,
        rank=2,
    )

    pair = pair_response_features(orbit, orbit)

    assert pair["pair.spectrum_energy_l1"] == pytest.approx(0.0)
    assert pair["pair.effective_rank_gap"] == pytest.approx(0.0)
    assert pair["pair.response_subspace_compatibility"] == pytest.approx(1.0)


def test_energy_contrast_stays_finite_when_invariant_energy_is_zero() -> None:
    interventions = (
        _intervention("same-direction", ExpectedRelation.PRESERVE, "surface", 1.0),
        _intervention("changed", ExpectedRelation.CHANGE, "negation", 1.0),
    )
    orbit = extract_orbit_features(
        np.array([1.0, 0.0, 0.0]),
        np.array([[2.0, 0.0, 0.0], [1.0, 0.0, 1.0]]),
        interventions,
        rank=2,
    )

    values = orbit.features
    assert np.isfinite(values["response.critical_to_invariant_energy_ratio"])
    assert np.isfinite(values["response.critical_to_invariant_log_energy_ratio"])
    assert 0.0 < values["response.critical_invariant_energy_contrast"] <= 1.0


def test_strength_weighting_is_an_explicit_ablation() -> None:
    interventions = (
        _intervention("a", ExpectedRelation.PRESERVE, "surface", 0.01),
        _intervention("b", ExpectedRelation.CHANGE, "negation", 1.0),
    )
    base = np.array([1.0, 0.0, 0.0])
    transformed = np.array([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]])

    uniform = extract_orbit_features(base, transformed, interventions, weighting="uniform")
    weighted = extract_orbit_features(base, transformed, interventions, weighting="strength")

    assert uniform.global_measurement.spectrum.total_energy > (
        weighted.global_measurement.spectrum.total_energy
    )
