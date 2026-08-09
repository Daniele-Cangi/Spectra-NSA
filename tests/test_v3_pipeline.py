import numpy as np

from spectra_v3.encoders import EncoderSpec
from spectra_v3.interventions import (
    ExpectedRelation,
    Intervention,
    InterventionOrbit,
)
from spectra_v3.pipeline import measure_orbit, measure_orbit_from_embeddings


class FixedEncoder:
    spec = EncoderSpec("fixed", "commit-1")

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = {
            "base": [1.0, 0.0, 0.0],
            "surface": [1.0, 0.1, 0.0],
            "critical": [1.0, 0.0, 1.0],
        }
        return np.asarray([vectors[text] for text in texts], dtype=np.float32)


def test_measure_orbit_exports_provenance_without_text_by_default() -> None:
    orbit = InterventionOrbit(
        base_id="base-1",
        base_text="base",
        source="unit-test",
        interventions=(
            Intervention(
                base_text="base",
                transformed_text="surface",
                family="surface",
                expected_relation=ExpectedRelation.PRESERVE,
                strength=0.1,
                generator_id="surface-v1",
            ),
            Intervention(
                base_text="base",
                transformed_text="critical",
                family="negation",
                expected_relation=ExpectedRelation.CHANGE,
                strength=1.0,
                generator_id="negation-v1",
            ),
        ),
    )

    measurement = measure_orbit(orbit, FixedEncoder(), rank=2)
    exported = measurement.to_dict()

    assert exported["base_id"] == "base-1"
    assert exported["encoder"]["revision"] == "commit-1"
    assert "base_text" not in exported
    assert "transformed_text" not in exported["interventions"][0]
    assert exported["interventions"][0]["response_norm"] > 0.0
    assert exported["interventions"][0]["measurement_response_norm"] > 0.0
    assert exported["measurement"]["weighting"] == "uniform"
    assert "token_edit_distance" not in exported["interventions"][0]
    assert "response.global.total_energy" in exported["features"]


def test_precomputed_measurement_exports_token_diagnostics() -> None:
    orbit = InterventionOrbit(
        base_id="base-token",
        base_text="base",
        interventions=(
            Intervention(
                base_text="base",
                transformed_text="critical",
                family="negation",
                expected_relation=ExpectedRelation.CHANGE,
                strength=1.0,
                generator_id="negation-v1",
            ),
        ),
    )
    measurement = measure_orbit_from_embeddings(
        orbit,
        FixedEncoder.spec,
        np.asarray([[1.0, 0.0, 0.0], [1.0, 0.0, 1.0]]),
        token_ids=[[1, 2], [1, 3, 2]],
    )

    intervention = measurement.to_dict()["interventions"][0]
    assert intervention["token_edit_distance"] == 1
    assert intervention["token_count_delta"] == 1


def test_precomputed_measurement_exports_query_conditioned_delta() -> None:
    orbit = InterventionOrbit(
        base_id="base-context",
        base_text="base",
        context_text="query",
        interventions=(
            Intervention(
                base_text="base",
                transformed_text="critical",
                family="negation",
                expected_relation=ExpectedRelation.CHANGE,
                strength=1.0,
                generator_id="negation-v1",
            ),
        ),
    )
    measurement = measure_orbit_from_embeddings(
        orbit,
        FixedEncoder.spec,
        np.asarray([[1.0, 0.0], [0.0, 1.0]]),
        context_embedding=np.asarray([1.0, 0.0]),
    )

    exported = measurement.to_dict()
    assert exported["context"]["base_similarity"] == 1.0
    assert exported["interventions"][0]["context_similarity_delta"] == -1.0
