"""Finite-difference Hessian estimation.

For quadratic objectives (e.g. CEC 2017 F1, the shifted-rotated Bent Cigar)
the central-difference Hessian is exact up to rounding, so it can stand in
for the analytic Hessian when the objective is a compiled black box.  For
non-quadratic objectives it is only the local curvature at ``x``.
"""

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray


def numerical_hessian(
    func: Callable[[NDArray[np.float64]], float],
    x: NDArray[np.float64],
    eps: float | None = None,
) -> NDArray[np.float64]:
    """Central-difference Hessian of ``func`` at ``x`` (``2n^2 + 2n + 1`` evals)."""
    x = np.asarray(x, dtype=float)
    n = x.size
    if eps is None:
        # Fourth-root step balances truncation and rounding error for
        # second-order central differences.
        eps = float(np.finfo(float).eps ** 0.25) * max(1.0, float(np.max(np.abs(x))))

    f0 = float(func(x))
    hessian = np.empty((n, n), dtype=float)

    f_plus = np.empty(n)
    f_minus = np.empty(n)
    for i in range(n):
        step = np.zeros(n)
        step[i] = eps
        f_plus[i] = float(func(x + step))
        f_minus[i] = float(func(x - step))
        hessian[i, i] = (f_plus[i] - 2.0 * f0 + f_minus[i]) / (eps * eps)

    for i in range(n):
        for j in range(i + 1, n):
            step_i = np.zeros(n)
            step_j = np.zeros(n)
            step_i[i] = eps
            step_j[j] = eps
            f_pp = float(func(x + step_i + step_j))
            f_mm = float(func(x - step_i - step_j))
            value = (
                f_pp - f_plus[i] - f_plus[j] + 2.0 * f0 - f_minus[i] - f_minus[j] + f_mm
            ) / (2.0 * eps * eps)
            hessian[i, j] = value
            hessian[j, i] = value

    return 0.5 * (hessian + hessian.T)


def spd_regularize(
    matrix: NDArray[np.float64],
    floor_ratio: float = 1e-12,
) -> NDArray[np.float64]:
    """Nearest-eigenvalue-floored SPD version of a symmetric matrix.

    Eigenvalues below ``floor_ratio * max(eigenvalue)`` (or below an absolute
    tiny floor when the spectrum is degenerate) are raised to that floor, so
    the result is safe for a Cholesky factorization.
    """
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    top = float(np.max(eigenvalues)) if eigenvalues.size else 1.0
    floor = max(abs(top) * floor_ratio, np.finfo(float).tiny)
    eigenvalues = np.maximum(eigenvalues, floor)
    return (eigenvectors * eigenvalues) @ eigenvectors.T
