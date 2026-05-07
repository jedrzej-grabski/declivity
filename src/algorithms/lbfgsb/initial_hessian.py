"""
Initial Hessian approximation for L-BFGS-B.

Encapsulates the base Hessian B_0 and provides the operations needed by the
compact L-BFGS representation. Dispatches to element-wise operations for
diagonal B_0 or matrix operations for dense B_0, so the optimizer code
can call the same interface regardless of the representation.
"""

from enum import Enum

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import cho_factor, cho_solve


class InitialHessianMode(Enum):
    """How the initial Hessian B_0 is stored and applied."""

    DIAGONAL = "diagonal"
    """B_0 = diag(h). All operations are element-wise, O(n) per vector."""

    DENSE = "dense"
    """B_0 is a full symmetric positive definite matrix. Operations use
    matrix-vector products O(n^2) and Cholesky solves O(n^2)."""


class InitialHessian:
    """Base Hessian approximation B_0 used in the compact representation
    B = theta * B_0 - W * M * W'.

    Constructed from user input which may be None (identity), a scalar,
    a 1D array (diagonal), or a 2D array (full matrix). Provides operations
    used throughout the L-BFGS-B algorithm:

    - multiply(v): B_0 * v
    - solve(v): B_0^{-1} * v
    - quadratic_form(S): S' * B_0 * S
    - scale_columns(S): B_0 * S
    - weighted_dot(d): d' * B_0 * d
    """

    def __init__(
        self,
        initial_hessian,
        num_dimensions: int,
    ):
        if initial_hessian is None:
            self._mode = InitialHessianMode.DIAGONAL
            self._diagonal = np.ones(num_dimensions)
            self._diagonal_inverse = np.ones(num_dimensions)
            self._matrix = None
            self._cholesky_factor = None

        elif np.isscalar(initial_hessian):
            value = float(initial_hessian)
            if value <= 0:
                raise ValueError("Scalar initial_hessian must be positive")
            self._mode = InitialHessianMode.DIAGONAL
            self._diagonal = np.full(num_dimensions, value)
            self._diagonal_inverse = np.full(num_dimensions, 1.0 / value)
            self._matrix = None
            self._cholesky_factor = None

        elif np.ndim(initial_hessian) == 1:
            diagonal = np.asarray(initial_hessian, dtype=float)
            if diagonal.shape != (num_dimensions,):
                raise ValueError(
                    f"initial_hessian vector must have length {num_dimensions}, "
                    f"got {diagonal.shape}"
                )
            if np.any(diagonal <= 0):
                raise ValueError("initial_hessian diagonal entries must be positive")
            self._mode = InitialHessianMode.DIAGONAL
            self._diagonal = diagonal
            self._diagonal_inverse = 1.0 / diagonal
            self._matrix = None
            self._cholesky_factor = None

        elif np.ndim(initial_hessian) == 2:
            matrix = np.asarray(initial_hessian, dtype=float)
            if matrix.shape != (num_dimensions, num_dimensions):
                raise ValueError(
                    f"initial_hessian matrix must be ({num_dimensions}, {num_dimensions}), "
                    f"got {matrix.shape}"
                )
            matrix = 0.5 * (matrix + matrix.T)
            self._mode = InitialHessianMode.DENSE
            self._diagonal = np.diag(matrix)
            self._matrix = matrix
            try:
                self._cholesky_factor = cho_factor(matrix)
            except np.linalg.LinAlgError:
                raise ValueError(
                    "initial_hessian matrix must be symmetric positive definite"
                )
            self._diagonal_inverse = None

        else:
            raise ValueError(
                "initial_hessian must be None, a scalar, a 1D array, or a 2D array"
            )

    @property
    def mode(self) -> InitialHessianMode:
        return self._mode

    @property
    def diagonal(self) -> NDArray[np.float64]:
        """The diagonal of B_0, available in both modes."""
        return self._diagonal

    def multiply(self, v: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute B_0 * v."""
        if self._mode == InitialHessianMode.DIAGONAL:
            return self._diagonal * v
        else:
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                return self._matrix @ v

    def solve(self, v: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute B_0^{-1} * v."""
        if self._mode == InitialHessianMode.DIAGONAL:
            return self._diagonal_inverse * v
        else:
            return cho_solve(self._cholesky_factor, v)

    def scale_columns(self, S: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute B_0 * S where S is n x m (column-wise multiplication).

        Used for building W = [Y | theta * B_0 * S].
        """
        if self._mode == InitialHessianMode.DIAGONAL:
            return self._diagonal[:, np.newaxis] * S
        else:
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                return self._matrix @ S

    def quadratic_form(self, S: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute S' * B_0 * S where S is n x m. Returns m x m.

        Used for the M^{-1} and T matrices in the compact representation.
        """
        if self._mode == InitialHessianMode.DIAGONAL:
            return S.T @ (self._diagonal[:, np.newaxis] * S)
        else:
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                return S.T @ (self._matrix @ S)

    def weighted_dot(self, d: NDArray[np.float64]) -> float:
        """Compute d' * B_0 * d (scalar).

        Used in the Cauchy point for the second derivative of the quadratic model.
        """
        if self._mode == InitialHessianMode.DIAGONAL:
            return float(np.dot(self._diagonal * d, d))
        else:
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                return float(d @ self._matrix @ d)

    def diagonal_element(self, i: int) -> float:
        """Return B_0[i, i].

        Used in the Cauchy point breakpoint updates. For dense B_0, the
        off-diagonal contributions at breakpoints are captured by the
        L-BFGS correction terms when correction pairs are available.
        """
        return float(self._diagonal[i])
