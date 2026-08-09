import numpy as np
import pytest

from spectra_v3.response import (
    measure_response,
    principal_angles,
    response_spectrum,
    subspace_compatibility,
    tangent_responses,
)


def test_tangent_responses_are_orthogonal_to_base() -> None:
    base = np.array([1.0, 0.0, 0.0])
    transformed = np.array([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]])

    responses = tangent_responses(base, transformed)

    np.testing.assert_allclose(responses @ base, 0.0, atol=1e-12)


def test_spectrum_is_invariant_to_response_row_order() -> None:
    matrix = np.array(
        [
            [1.0, 2.0, 0.0, 0.0],
            [0.0, 1.0, 3.0, 0.0],
            [1.0, 0.0, 0.0, 2.0],
        ]
    )
    original = response_spectrum(matrix)
    permuted = response_spectrum(matrix[[2, 0, 1]])

    np.testing.assert_allclose(original.singular_values, permuted.singular_values)
    original_projection = original.right_subspace @ original.right_subspace.T
    permuted_projection = permuted.right_subspace @ permuted.right_subspace.T
    np.testing.assert_allclose(original_projection, permuted_projection, atol=1e-12)


def test_singular_values_are_invariant_to_orthogonal_coordinate_change() -> None:
    matrix = np.array([[1.0, 2.0, 0.0], [0.0, 1.0, 3.0]])
    q, _ = np.linalg.qr(np.array([[1.0, 2.0, 3.0], [0.0, 2.0, 1.0], [3.0, 1.0, 0.0]]))

    left = response_spectrum(matrix)
    rotated = response_spectrum(matrix @ q)

    np.testing.assert_allclose(left.singular_values, rotated.singular_values)


def test_truncated_energy_ratios_still_use_full_spectrum() -> None:
    spectrum = response_spectrum(np.diag([3.0, 1.0]), rank=1)

    np.testing.assert_allclose(spectrum.normalized_energy, [0.9])
    assert spectrum.effective_rank > 1.0


def test_zero_response_has_no_artificial_subspace() -> None:
    spectrum = response_spectrum(np.zeros((2, 3)))

    assert spectrum.numerical_rank == 0
    assert spectrum.effective_rank == 0.0
    assert spectrum.right_subspace.shape == (3, 0)


def test_float32_scale_noise_does_not_create_response_modes() -> None:
    spectrum = response_spectrum(np.diag([1.0, 5e-7]), rank=2)

    assert spectrum.numerical_rank == 1
    np.testing.assert_allclose(spectrum.singular_values, [1.0, 0.0])
    assert spectrum.total_energy == pytest.approx(1.0)


def test_strength_weights_scale_response_energy() -> None:
    base = np.array([1.0, 0.0])
    transformed = np.array([[1.0, 1.0]])
    full = measure_response(base, transformed, weights=[1.0])
    quarter = measure_response(base, transformed, weights=[0.25])

    assert quarter.spectrum.total_energy == pytest.approx(
        0.25 * full.spectrum.total_energy
    )


def test_principal_angles_and_compatibility_for_partially_shared_subspaces() -> None:
    left = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    right = np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]])

    np.testing.assert_allclose(principal_angles(left, right), [0.0, np.pi / 2])
    assert subspace_compatibility(left, right) == pytest.approx(0.5)
