from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.floating[Any]]
DEFAULT_SPECTRUM_ATOL = 1e-6
DEFAULT_SPECTRUM_RTOL = 1e-5


def _as_vector(value: npt.ArrayLike, *, name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.size == 0:
        raise ValueError(f"{name} cannot be empty")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _as_matrix(value: npt.ArrayLike, *, name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} cannot have an empty dimension")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def l2_normalize(
    value: npt.ArrayLike, *, axis: int = -1, eps: float = 1e-12
) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("value must contain only finite values")
    norms = np.linalg.norm(array, axis=axis, keepdims=True)
    if np.any(norms <= eps):
        raise ValueError("cannot normalize a zero-length vector")
    return array / norms


def tangent_project(
    base_embedding: npt.ArrayLike, responses: npt.ArrayLike
) -> FloatArray:
    """Project response rows onto the tangent plane at a normalized base embedding."""

    base = l2_normalize(_as_vector(base_embedding, name="base_embedding"))
    matrix = _as_matrix(responses, name="responses")
    if matrix.shape[1] != base.shape[0]:
        raise ValueError(
            "responses and base_embedding must share the embedding dimension"
        )
    radial = matrix @ base
    return matrix - radial[:, None] * base[None, :]


def tangent_responses(
    base_embedding: npt.ArrayLike,
    transformed_embeddings: npt.ArrayLike,
) -> FloatArray:
    base = l2_normalize(_as_vector(base_embedding, name="base_embedding"))
    transformed = l2_normalize(
        _as_matrix(transformed_embeddings, name="transformed_embeddings"), axis=1
    )
    if transformed.shape[1] != base.shape[0]:
        raise ValueError(
            "transformed_embeddings and base_embedding dimensions must match"
        )
    raw = transformed - base[None, :]
    return tangent_project(base, raw)


def _energy_distribution(singular_values: FloatArray, eps: float) -> FloatArray:
    energy = np.square(singular_values)
    total = float(energy.sum())
    if total <= eps:
        return np.zeros_like(energy)
    return energy / total


@dataclass(frozen=True)
class ResponseSpectrum:
    singular_values: FloatArray
    normalized_energy: FloatArray
    right_subspace: FloatArray
    total_energy: float
    effective_rank: float
    spectral_entropy: float
    top_concentration: float
    numerical_rank: int

    def feature_dict(self, *, prefix: str = "response") -> dict[str, float]:
        result = {
            f"{prefix}.total_energy": self.total_energy,
            f"{prefix}.effective_rank": self.effective_rank,
            f"{prefix}.spectral_entropy": self.spectral_entropy,
            f"{prefix}.top_concentration": self.top_concentration,
            f"{prefix}.numerical_rank": float(self.numerical_rank),
        }
        for index, value in enumerate(self.singular_values):
            result[f"{prefix}.singular_value_{index + 1}"] = float(value)
        for index, value in enumerate(self.normalized_energy):
            result[f"{prefix}.energy_ratio_{index + 1}"] = float(value)
        return result


@dataclass(frozen=True)
class ResponseMeasurement:
    response_matrix: FloatArray
    spectrum: ResponseSpectrum
    response_norms: FloatArray
    mean_response_norm: float
    max_response_norm: float
    response_norm_variance: float

    def feature_dict(self, *, prefix: str = "response") -> dict[str, float]:
        result = self.spectrum.feature_dict(prefix=prefix)
        result.update(
            {
                f"{prefix}.mean_norm": self.mean_response_norm,
                f"{prefix}.max_norm": self.max_response_norm,
                f"{prefix}.norm_variance": self.response_norm_variance,
            }
        )
        return result


def response_spectrum(
    response_matrix: npt.ArrayLike,
    *,
    rank: int | None = None,
    eps: float = 1e-12,
    noise_atol: float = DEFAULT_SPECTRUM_ATOL,
    noise_rtol: float = DEFAULT_SPECTRUM_RTOL,
) -> ResponseSpectrum:
    matrix = _as_matrix(response_matrix, name="response_matrix")
    if rank is not None and rank <= 0:
        raise ValueError("rank must be positive")
    if noise_atol < 0 or noise_rtol < 0:
        raise ValueError("noise tolerances must be non-negative")

    _, singular_values, vh = np.linalg.svd(matrix, full_matrices=False)
    leading_value = float(singular_values[0])
    threshold = max(noise_atol, noise_rtol * leading_value)
    numerical_rank = int(np.count_nonzero(singular_values > threshold))
    denoised_values = singular_values.copy()
    denoised_values[numerical_rank:] = 0.0
    available = singular_values.shape[0]
    keep = available if rank is None else min(rank, available)
    kept_values = denoised_values[:keep]
    full_energy = _energy_distribution(denoised_values, eps)
    energy = full_energy[:keep]
    positive = full_energy[full_energy > eps]
    entropy = float(-(positive * np.log(positive)).sum()) if positive.size else 0.0
    effective_rank = float(np.exp(entropy)) if positive.size else 0.0
    total_energy = float(np.square(denoised_values).sum())
    top_concentration = float(full_energy[0]) if full_energy.size else 0.0

    subspace_keep = min(keep, numerical_rank)
    return ResponseSpectrum(
        singular_values=kept_values,
        normalized_energy=energy,
        right_subspace=vh[:subspace_keep].T,
        total_energy=total_energy,
        effective_rank=effective_rank,
        spectral_entropy=entropy,
        top_concentration=top_concentration,
        numerical_rank=numerical_rank,
    )


def measure_response(
    base_embedding: npt.ArrayLike,
    transformed_embeddings: npt.ArrayLike,
    *,
    weights: Sequence[float] | None = None,
    rank: int | None = None,
) -> ResponseMeasurement:
    matrix = tangent_responses(base_embedding, transformed_embeddings)
    if weights is not None:
        weight_array = np.asarray(weights, dtype=np.float64)
        if weight_array.shape != (matrix.shape[0],):
            raise ValueError("weights must contain one value per intervention")
        if not np.isfinite(weight_array).all() or np.any(weight_array < 0):
            raise ValueError("weights must be finite and non-negative")
        matrix = matrix * np.sqrt(weight_array)[:, None]

    norms = np.linalg.norm(matrix, axis=1)
    spectrum = response_spectrum(matrix, rank=rank)
    return ResponseMeasurement(
        response_matrix=matrix,
        spectrum=spectrum,
        response_norms=norms,
        mean_response_norm=float(norms.mean()),
        max_response_norm=float(norms.max()),
        response_norm_variance=float(norms.var()),
    )


def _orthonormal_basis(
    value: npt.ArrayLike, *, name: str, eps: float = 1e-12
) -> FloatArray:
    matrix = _as_matrix(value, name=name)
    q, r = np.linalg.qr(matrix)
    diagonal = np.abs(np.diag(r))
    if diagonal.size == 0 or np.any(diagonal <= eps):
        raise ValueError(f"{name} must contain linearly independent columns")
    return q[:, : matrix.shape[1]]


def principal_angles(
    left_subspace: npt.ArrayLike, right_subspace: npt.ArrayLike
) -> FloatArray:
    left = _orthonormal_basis(left_subspace, name="left_subspace")
    right = _orthonormal_basis(right_subspace, name="right_subspace")
    if left.shape[0] != right.shape[0]:
        raise ValueError("subspaces must share the ambient dimension")
    cosines = np.linalg.svd(left.T @ right, compute_uv=False)
    return np.arccos(np.clip(cosines, -1.0, 1.0))


def subspace_compatibility(
    left_subspace: npt.ArrayLike, right_subspace: npt.ArrayLike
) -> float:
    angles = principal_angles(left_subspace, right_subspace)
    return float(np.square(np.cos(angles)).mean())
