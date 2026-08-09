from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import numpy.typing as npt

from .interventions import ExpectedRelation, Intervention
from .response import (
    ResponseMeasurement,
    measure_response,
    principal_angles,
    subspace_compatibility,
    tangent_responses,
)


@dataclass(frozen=True)
class OrbitFeatures:
    global_measurement: ResponseMeasurement
    relation_measurements: Mapping[ExpectedRelation, ResponseMeasurement]
    family_measurements: Mapping[str, ResponseMeasurement]
    unweighted_response_norms: npt.NDArray[np.float64]
    features: Mapping[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "features": dict(self.features),
            "spectrum": {
                "singular_values": self.global_measurement.spectrum.singular_values.tolist(),
                "normalized_energy": self.global_measurement.spectrum.normalized_energy.tolist(),
                "right_subspace": self.global_measurement.spectrum.right_subspace.tolist(),
            },
        }


def _select_rows(
    matrix: npt.NDArray[np.float64], indices: Iterable[int]
) -> npt.NDArray[np.float64]:
    selected = list(indices)
    if not selected:
        raise ValueError("cannot measure an empty response group")
    return matrix[selected]


def extract_orbit_features(
    base_embedding: npt.ArrayLike,
    transformed_embeddings: npt.ArrayLike,
    interventions: Sequence[Intervention],
    *,
    rank: int = 8,
    weighting: str = "uniform",
) -> OrbitFeatures:
    transformed = np.asarray(transformed_embeddings, dtype=np.float64)
    if transformed.ndim != 2:
        raise ValueError("transformed_embeddings must be two-dimensional")
    if transformed.shape[0] != len(interventions):
        raise ValueError("one transformed embedding is required per intervention")
    if not interventions:
        raise ValueError("interventions cannot be empty")

    if weighting not in {"strength", "uniform"}:
        raise ValueError("weighting must be 'strength' or 'uniform'")
    strengths = np.asarray(
        [intervention.strength for intervention in interventions], dtype=np.float64
    )
    measurement_weights = strengths if weighting == "strength" else np.ones_like(strengths)
    unweighted_response_norms = np.linalg.norm(
        tangent_responses(base_embedding, transformed), axis=1
    )
    global_measurement = measure_response(
        base_embedding,
        transformed,
        weights=measurement_weights,
        rank=rank,
    )
    features = global_measurement.feature_dict(prefix="response.global")

    relation_measurements: dict[ExpectedRelation, ResponseMeasurement] = {}
    for relation in ExpectedRelation:
        indices = [
            index
            for index, intervention in enumerate(interventions)
            if intervention.expected_relation == relation
        ]
        if not indices:
            continue
        measurement = measure_response(
            base_embedding,
            _select_rows(transformed, indices),
            weights=measurement_weights[indices],
            rank=rank,
        )
        relation_measurements[relation] = measurement
        features.update(
            measurement.feature_dict(prefix=f"response.relation.{relation.value}")
        )

    family_measurements: dict[str, ResponseMeasurement] = {}
    families = sorted({intervention.family for intervention in interventions})
    for family in families:
        indices = [
            index
            for index, intervention in enumerate(interventions)
            if intervention.family == family
        ]
        measurement = measure_response(
            base_embedding,
            _select_rows(transformed, indices),
            weights=measurement_weights[indices],
            rank=rank,
        )
        family_measurements[family] = measurement
        features.update(measurement.feature_dict(prefix=f"response.family.{family}"))

    preserving = relation_measurements.get(ExpectedRelation.PRESERVE)
    changing = relation_measurements.get(ExpectedRelation.CHANGE)
    if preserving is not None and changing is not None:
        invariant_energy = preserving.spectrum.total_energy
        critical_energy = changing.spectrum.total_energy
        energy_scale = max(
            global_measurement.spectrum.total_energy,
            invariant_energy + critical_energy,
            1e-12,
        )
        stabilizer = 1e-6 * energy_scale
        denominator = invariant_energy + stabilizer
        features["response.critical_to_invariant_energy_ratio"] = (
            (critical_energy + stabilizer) / denominator
        )
        features["response.critical_to_invariant_log_energy_ratio"] = float(
            np.log((critical_energy + stabilizer) / denominator)
        )
        features["response.critical_invariant_energy_contrast"] = (
            (critical_energy - invariant_energy)
            / (critical_energy + invariant_energy + stabilizer)
        )
        features["response.energy_ratio_stabilizer"] = stabilizer
        left = preserving.spectrum.right_subspace
        right = changing.spectrum.right_subspace
        if left.shape[1] > 0 and right.shape[1] > 0:
            angles = principal_angles(left, right)
            features["response.invariant_critical_mean_angle"] = float(angles.mean())
            features["response.invariant_critical_compatibility"] = (
                subspace_compatibility(left, right)
            )

    features["interventions.count"] = float(len(interventions))
    features["interventions.mean_strength"] = float(
        np.mean([intervention.strength for intervention in interventions])
    )

    return OrbitFeatures(
        global_measurement=global_measurement,
        relation_measurements=relation_measurements,
        family_measurements=family_measurements,
        unweighted_response_norms=unweighted_response_norms,
        features=features,
    )


def pair_response_features(
    left: OrbitFeatures, right: OrbitFeatures
) -> dict[str, float]:
    """Build pairwise response-geometry features without using pair labels."""

    left_spectrum = left.global_measurement.spectrum
    right_spectrum = right.global_measurement.spectrum
    width = min(
        left_spectrum.normalized_energy.shape[0],
        right_spectrum.normalized_energy.shape[0],
    )
    result = {
        "pair.spectrum_energy_l1": float(
            np.abs(
                left_spectrum.normalized_energy[:width]
                - right_spectrum.normalized_energy[:width]
            ).sum()
        ),
        "pair.effective_rank_gap": abs(
            left_spectrum.effective_rank - right_spectrum.effective_rank
        ),
        "pair.total_energy_log_gap": abs(
            np.log1p(left_spectrum.total_energy) - np.log1p(right_spectrum.total_energy)
        ),
    }
    left_subspace = left_spectrum.right_subspace
    right_subspace = right_spectrum.right_subspace
    if left_subspace.shape[1] > 0 and right_subspace.shape[1] > 0:
        angles = principal_angles(left_subspace, right_subspace)
        result["pair.response_subspace_mean_angle"] = float(angles.mean())
        result["pair.response_subspace_compatibility"] = subspace_compatibility(
            left_subspace, right_subspace
        )
    return result
