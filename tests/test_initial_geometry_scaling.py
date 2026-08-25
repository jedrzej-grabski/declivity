"""Regression tests for the transform/scaling split on :class:`InitialGeometry`.

``HandoffTransform`` now covers *shape* only (INVERSE / IDENTITY) and
``HessianScaling`` covers *magnitude* (NONE / SIGMA / UNIT / IDENTITY_NORM).
These tests pin down that ``transform=INVERSE, scaling=SIGMA`` reproduces the
old fused ``sigma_inverse`` transform, and that UNIT / IDENTITY_NORM scale the
resulting matrix to the expected Frobenius norm.
"""

import numpy as np

from declivity.benchmarking.algorithm_run import initial_hessian_from_cmaes
from declivity.utils.initial_geometry import (
    HandoffTransform,
    HessianScaling,
    InitialGeometry,
)


def _random_covariance_eigendecomposition(dim: int, seed: int):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(dim, dim))
    covariance = a @ a.T + dim * np.eye(dim)  # SPD
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues_sqrt = np.sqrt(eigenvalues)
    return eigenvectors, eigenvalues_sqrt, covariance


def test_sigma_scaling_reproduces_old_fused_sigma_inverse():
    dim = 6
    sigma = 2.5
    eigenvectors, eigenvalues_sqrt, covariance = _random_covariance_eigendecomposition(
        dim, seed=0
    )

    geometry = InitialGeometry.from_covariance(
        eigenvectors,
        eigenvalues_sqrt,
        sigma,
        transform=HandoffTransform.INVERSE,
        scaling=HessianScaling.SIGMA,
    )

    expected = np.linalg.inv(covariance) / (sigma * sigma)
    assert geometry.mode.value == "dense"
    actual = geometry._matrix
    assert actual is not None
    assert np.allclose(actual, expected, rtol=1e-8, atol=1e-10)


def test_none_scaling_is_plain_inverse():
    dim = 5
    eigenvectors, eigenvalues_sqrt, covariance = _random_covariance_eigendecomposition(
        dim, seed=1
    )
    geometry = InitialGeometry.from_covariance(
        eigenvectors,
        eigenvalues_sqrt,
        sigma=3.0,
        transform=HandoffTransform.INVERSE,
        scaling=HessianScaling.NONE,
    )
    expected = np.linalg.inv(covariance)
    actual = geometry._matrix
    assert actual is not None
    assert np.allclose(actual, expected, rtol=1e-8, atol=1e-10)


def test_unit_scaling_gives_unit_frobenius_norm():
    dim = 7
    eigenvectors, eigenvalues_sqrt, _ = _random_covariance_eigendecomposition(
        dim, seed=2
    )
    geometry = InitialGeometry.from_covariance(
        eigenvectors,
        eigenvalues_sqrt,
        sigma=1.0,
        transform=HandoffTransform.INVERSE,
        scaling=HessianScaling.UNIT,
    )
    actual = geometry._matrix
    assert actual is not None
    assert np.isclose(np.linalg.norm(actual), 1.0, rtol=1e-8, atol=1e-10)


def test_identity_norm_scaling_matches_identity_frobenius_norm():
    dim = 8
    eigenvectors, eigenvalues_sqrt, _ = _random_covariance_eigendecomposition(
        dim, seed=3
    )
    geometry = InitialGeometry.from_covariance(
        eigenvectors,
        eigenvalues_sqrt,
        sigma=1.0,
        transform=HandoffTransform.INVERSE,
        scaling=HessianScaling.IDENTITY_NORM,
    )
    actual = geometry._matrix
    assert actual is not None
    assert np.isclose(np.linalg.norm(actual), np.sqrt(dim), rtol=1e-8, atol=1e-10)


def test_identity_norm_scaling_on_identity_geometry():
    # transform=IDENTITY -> matrix is None -> B_0 stored as diag(ones), whose
    # implied dense matrix has Frobenius norm sqrt(dim); IDENTITY_NORM should
    # leave it at norm sqrt(dim) (factor 1), i.e. B_0 stays the identity.
    dim = 4
    geometry = InitialGeometry.identity(dim, scaling=HessianScaling.IDENTITY_NORM)
    assert np.allclose(geometry.diagonal, np.ones(dim))


def test_scaling_does_not_perturb_forward_geometry():
    # Scaling is magnitude-only: principal directions / scales (used by
    # Powell / Nelder-Mead) must be identical regardless of scaling.
    dim = 5
    eigenvectors, eigenvalues_sqrt, _ = _random_covariance_eigendecomposition(
        dim, seed=4
    )

    unscaled = InitialGeometry.from_covariance(
        eigenvectors, eigenvalues_sqrt, sigma=1.7, transform=HandoffTransform.INVERSE
    )
    scaled = InitialGeometry.from_covariance(
        eigenvectors,
        eigenvalues_sqrt,
        sigma=1.7,
        transform=HandoffTransform.INVERSE,
        scaling=HessianScaling.UNIT,
    )
    assert np.allclose(unscaled.principal_directions(), scaled.principal_directions())
    assert np.allclose(unscaled.principal_scales(), scaled.principal_scales())


def test_raw_handoff_seam_matches_from_covariance():
    # initial_hessian_from_cmaes (raw-matrix seam feeding LBFGSBConfig) must
    # bake in the same scaling as InitialGeometry.from_covariance for every
    # HessianScaling, so the one-shot / interleaved L-BFGS-B handoffs agree
    # numerically with the geometry-object path.
    dim = 6
    sigma = 2.5
    eigenvectors, eigenvalues_sqrt, _ = _random_covariance_eigendecomposition(
        dim, seed=5
    )
    for scaling in HessianScaling:
        raw = initial_hessian_from_cmaes(
            HandoffTransform.INVERSE,
            eigenvectors,
            eigenvalues_sqrt,
            scaling=scaling,
            sigma=sigma,
        )
        geometry = InitialGeometry.from_covariance(
            eigenvectors,
            eigenvalues_sqrt,
            sigma,
            transform=HandoffTransform.INVERSE,
            scaling=scaling,
        )
        expected = geometry._matrix
        assert raw is not None
        assert expected is not None
        assert np.allclose(raw, expected, rtol=1e-8, atol=1e-10)

    # IDENTITY transform drops the covariance (B_0 = I) regardless of scaling.
    assert (
        initial_hessian_from_cmaes(
            HandoffTransform.IDENTITY,
            eigenvectors,
            eigenvalues_sqrt,
            scaling=HessianScaling.UNIT,
            sigma=sigma,
        )
        is None
    )


def test_adaptive_scaling_without_prev_norm_is_a_no_op():
    # No previous burst yet (a one-shot handoff, or the first probe of an
    # interleaved run): ADAPTIVE must fall back to NONE rather than raise.
    dim = 5
    eigenvectors, eigenvalues_sqrt, covariance = _random_covariance_eigendecomposition(
        dim, seed=6
    )
    geometry = InitialGeometry.from_covariance(
        eigenvectors,
        eigenvalues_sqrt,
        sigma=1.0,
        transform=HandoffTransform.INVERSE,
        scaling=HessianScaling.ADAPTIVE,
        prev_norm=None,
    )
    expected = np.linalg.inv(covariance)
    actual = geometry._matrix
    assert actual is not None
    assert np.allclose(actual, expected, rtol=1e-8, atol=1e-10)


def test_adaptive_scaling_matches_prev_norm():
    dim = 6
    eigenvectors, eigenvalues_sqrt, _ = _random_covariance_eigendecomposition(
        dim, seed=7
    )
    prev_norm = 3.75
    geometry = InitialGeometry.from_covariance(
        eigenvectors,
        eigenvalues_sqrt,
        sigma=1.0,
        transform=HandoffTransform.INVERSE,
        scaling=HessianScaling.ADAPTIVE,
        prev_norm=prev_norm,
    )
    actual = geometry._matrix
    assert actual is not None
    assert np.isclose(np.linalg.norm(actual), prev_norm, rtol=1e-8, atol=1e-10)


if __name__ == "__main__":
    tests = [
        test_sigma_scaling_reproduces_old_fused_sigma_inverse,
        test_none_scaling_is_plain_inverse,
        test_unit_scaling_gives_unit_frobenius_norm,
        test_identity_norm_scaling_matches_identity_frobenius_norm,
        test_identity_norm_scaling_on_identity_geometry,
        test_scaling_does_not_perturb_forward_geometry,
        test_raw_handoff_seam_matches_from_covariance,
        test_adaptive_scaling_without_prev_norm_is_a_no_op,
        test_adaptive_scaling_matches_prev_norm,
    ]
    for test in tests:
        test()
        print(f"OK: {test.__name__}")
    print(f"{len(tests)} tests passed")
