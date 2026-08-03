"""
Shared "learned local geometry" object for the local optimizers.

Originally the L-BFGS-B initial Hessian ``B_0``; now generalized into a single
curvature/quadratic model that seeds the single-point local optimizers from the
same learned geometry (typically a CMA-ES covariance):

- **L-BFGS-B** consumes it as the initial Hessian ``B_0`` (curvature ops
  ``multiply`` / ``solve`` / ``quadratic_form`` / ...), exactly as before.
- **Powell** consumes :meth:`InitialGeometry.principal_directions` — the
  eigenvectors of the covariance — as its initial search-direction set.

Canonical stored quantity is **curvature** ``B_0`` (large eigenvalue = steep),
so the ~15 L-BFGS-B call sites stay byte-identical.  The covariance -> curvature
inversion happens **once**, in :meth:`InitialGeometry.from_covariance`.  When
built from a covariance the object *also* stores the forward eigendecomposition
``(B, D, sigma)`` as first-class fields, so the direction accessors are exact
and cheap (no re-``eigh``) and do not double-invert.

``InitialHessian`` is kept as an alias of ``InitialGeometry`` for backwards
compatibility (L-BFGS-B and its exports import the old name).
"""

from enum import Enum, StrEnum

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import cho_factor, cho_solve

# Numeric floor applied to (squared) eigenvalues before inversion, so a
# collapsed covariance direction does not blow the reciprocal up to inf.
_EIGENVALUE_FLOOR = 1e-30


class HandoffTransform(StrEnum):
    """How to turn a source covariance ``C`` into an initial curvature ``B_0``.

    The CMA-ES covariance ``C`` is proportional to ``B^{-1}`` (large variance =
    flat = small curvature), so the useful transforms invert it.  Used by
    :meth:`InitialGeometry.from_covariance`, the one-shot ``CMAESLBFGSBHandoff``
    / ``CMAESLocalHandoff``, and the ``InterleavedCMAESLBFGSB`` scheme.
    """

    INVERSE = "inverse"
    """Use ``C^{-1}`` directly. The L-BFGS-B model becomes a true quadratic
    approximation of the CMA-ES posterior around the warm-up mean."""

    SIGMA_INVERSE = "sigma_inverse"
    """Use ``(sigma^2 C)^{-1}`` — accounts for the CMA-ES global step-size
    scaling. Sometimes more conservative than the bare inverse."""

    IDENTITY = "identity"
    """Drop the covariance and use the isotropic default (``B_0 = I``). Mainly a
    control experiment: isolates the value of *passing covariance information*
    from the value of *sharing a starting point* with CMA-ES."""


def covariance_to_hessian_matrix(
    transform: "HandoffTransform | str",
    eigenvectors: NDArray[np.float64],
    eigenvalues_sqrt: NDArray[np.float64],
    sigma: float,
) -> NDArray[np.float64] | None:
    """Turn a covariance eigendecomposition ``(B, D)`` into a curvature matrix.

    ``C = B @ diag(D**2) @ B.T``; the returned matrix is the L-BFGS-B initial
    Hessian ``B_0`` (proportional to ``C^{-1}``).  Single source of truth shared
    by :meth:`InitialGeometry.from_covariance` and the benchmarking
    ``initial_hessian_from_cmaes`` delegate.  ``IDENTITY`` returns ``None`` (the
    L-BFGS-B default ``B_0 = I``).
    """
    transform = str(transform)
    if transform == HandoffTransform.IDENTITY:
        return None

    eigenvalues = np.maximum(np.asarray(eigenvalues_sqrt) ** 2, _EIGENVALUE_FLOOR)

    # Floored 1/eigenvalues can be huge (up to 1e30); the matmul values are
    # still valid but numpy raises spurious divide/overflow warnings on the
    # intermediates.
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        if transform == HandoffTransform.INVERSE:
            return (eigenvectors * (1.0 / eigenvalues)) @ eigenvectors.T
        if transform == HandoffTransform.SIGMA_INVERSE:
            inverse = (eigenvectors * (1.0 / eigenvalues)) @ eigenvectors.T
            return inverse / (sigma * sigma)

    valid = ", ".join(repr(value.value) for value in HandoffTransform)
    raise ValueError(f"Unknown handoff transform: {transform!r}. Use one of {valid}.")


class InitialHessianMode(Enum):
    """How the initial Hessian B_0 is stored and applied."""

    DIAGONAL = "diagonal"
    """B_0 = diag(h). All operations are element-wise, O(n) per vector."""

    DENSE = "dense"
    """B_0 is a full symmetric positive definite matrix. Operations use
    matrix-vector products O(n^2) and Cholesky solves O(n^2)."""


class InitialGeometry:
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

    and the geometry accessors consumed by the derivative-free methods:

    - principal_directions(): eigenvectors of the geometry (Powell directions)
    - principal_scales(): per-axis forward std-devs

    Use :meth:`identity`, :meth:`from_curvature`, or :meth:`from_covariance` to
    construct from a source covariance; the plain constructor accepts a raw
    curvature (None / scalar / 1D / 2D) as before.
    """

    def __init__(
        self,
        initial_hessian,
        num_dimensions: int,
    ):
        # Forward-geometry fields — populated eagerly by ``from_covariance`` (so
        # the direction / scale accessors are exact and need no re-``eigh``), or
        # lazily by ``_forward_geometry`` for a raw curvature.
        self._eigenvectors: NDArray[np.float64] | None = None
        self._forward_scales: NDArray[np.float64] | None = None
        self._sigma: float = 1.0

        if initial_hessian is None:
            self._mode = InitialHessianMode.DIAGONAL
            self._diagonal = np.ones(num_dimensions)
            self._diagonal_inverse = np.ones(num_dimensions)
            self._matrix = None
            self._cholesky_factor = None

        elif np.isscalar(initial_hessian):
            value = float(initial_hessian)  # type: ignore[arg-type]
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

    # ------------------------------------------------------------------
    # Constructors.
    # ------------------------------------------------------------------

    @classmethod
    def identity(cls, num_dimensions: int) -> "InitialGeometry":
        """Isotropic geometry — ``B_0 = I``, unit directions, uniform scales.

        The neutral control: L-BFGS-B behaves as ``config.initial_hessian=None``,
        and Powell gets the coordinate directions.
        """
        return cls(None, num_dimensions)

    @classmethod
    def from_curvature(cls, curvature, num_dimensions: int) -> "InitialGeometry":
        """Build directly from a curvature ``B_0`` (None / scalar / 1D / 2D).

        Alias of the plain constructor, named for symmetry with
        :meth:`from_covariance` at call sites.
        """
        return cls(curvature, num_dimensions)

    @classmethod
    def from_covariance(
        cls,
        eigenvectors: NDArray[np.float64],
        eigenvalues_sqrt: NDArray[np.float64],
        sigma: float = 1.0,
        transform: "HandoffTransform | str" = HandoffTransform.INVERSE,
    ) -> "InitialGeometry":
        """Build from a covariance eigendecomposition ``(B, D)``.

        ``C = B @ diag(D**2) @ B.T``.  ``transform`` selects how ``C`` becomes
        the curvature ``B_0`` (:class:`HandoffTransform`); the L-BFGS-B matrix is
        built once, byte-identical to the pre-existing handoff.  The **forward**
        eigendecomposition ``(B, D, sigma)`` is stored directly (columns ordered
        by descending variance) so the direction / scale accessors are exact and
        do not re-invert.  ``transform=IDENTITY`` discards the covariance shape
        and returns an isotropic :meth:`identity` geometry.
        """
        eigenvectors = np.asarray(eigenvectors, dtype=float)
        eigenvalues_sqrt = np.asarray(eigenvalues_sqrt, dtype=float)
        num_dimensions = eigenvectors.shape[0]

        matrix = covariance_to_hessian_matrix(
            transform, eigenvectors, eigenvalues_sqrt, sigma
        )
        if matrix is None:
            return cls.identity(num_dimensions)

        geometry = cls(matrix, num_dimensions)

        # Store the forward geometry of the *source covariance* (independent of
        # the transform used for the matrix): columns ordered by descending
        # variance (largest std-dev first). CMA-ES hands eigenvalues back
        # ascending, so impose our own order rather than passing it through.
        order = np.argsort(eigenvalues_sqrt)[::-1]
        geometry._eigenvectors = eigenvectors[:, order].copy()
        geometry._forward_scales = np.maximum(
            eigenvalues_sqrt[order], np.sqrt(_EIGENVALUE_FLOOR)
        ).copy()
        geometry._sigma = float(sigma)
        return geometry

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
            assert self._matrix is not None
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                return self._matrix @ v

    def solve(self, v: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute B_0^{-1} * v."""
        if self._mode == InitialHessianMode.DIAGONAL:
            assert self._diagonal_inverse is not None
            return self._diagonal_inverse * v
        else:
            assert self._cholesky_factor is not None
            return cho_solve(self._cholesky_factor, v)

    def scale_columns(self, S: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute B_0 * S where S is n x m (column-wise multiplication).

        Used for building W = [Y | theta * B_0 * S].
        """
        if self._mode == InitialHessianMode.DIAGONAL:
            return self._diagonal[:, np.newaxis] * S
        else:
            assert self._matrix is not None
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                return self._matrix @ S

    def quadratic_form(self, S: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute S' * B_0 * S where S is n x m. Returns m x m.

        Used for the M^{-1} and T matrices in the compact representation.
        """
        if self._mode == InitialHessianMode.DIAGONAL:
            return S.T @ (self._diagonal[:, np.newaxis] * S)
        else:
            assert self._matrix is not None
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                return S.T @ (self._matrix @ S)

    def weighted_dot(self, d: NDArray[np.float64]) -> float:
        """Compute d' * B_0 * d (scalar).

        Used in the Cauchy point for the second derivative of the quadratic model.
        """
        if self._mode == InitialHessianMode.DIAGONAL:
            return float(np.dot(self._diagonal * d, d))
        else:
            assert self._matrix is not None
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                return float(d @ self._matrix @ d)

    def diagonal_element(self, i: int) -> float:
        """Return B_0[i, i].

        Used in the Cauchy point breakpoint updates. For dense B_0, the
        off-diagonal contributions at breakpoints are captured by the
        L-BFGS correction terms when correction pairs are available.
        """
        return float(self._diagonal[i])

    # ------------------------------------------------------------------
    # Geometry accessors — consumed by the derivative-free methods.
    # ------------------------------------------------------------------

    def _forward_geometry(
        self,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return ``(eigenvectors, forward_scales)`` of the source geometry.

        ``eigenvectors`` are unit columns; ``forward_scales`` are the per-axis
        standard deviations of the *covariance* (shape only, no ``sigma``), i.e.
        ``1/sqrt(curvature eigenvalue)``.  Populated eagerly by
        :meth:`from_covariance`; otherwise derived once (identity for a diagonal
        curvature, a cached ``eigh`` for a raw dense curvature) and cached.
        """
        if self._eigenvectors is not None and self._forward_scales is not None:
            return self._eigenvectors, self._forward_scales

        n = len(self._diagonal)
        if self._mode == InitialHessianMode.DIAGONAL:
            eigenvectors = np.eye(n)
            scales = 1.0 / np.sqrt(np.maximum(self._diagonal, _EIGENVALUE_FLOOR))
        else:
            assert self._matrix is not None
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                curvatures, eigenvectors = np.linalg.eigh(self._matrix)
            curvatures = np.maximum(curvatures, _EIGENVALUE_FLOOR)
            scales = 1.0 / np.sqrt(curvatures)
            order = np.argsort(scales)[::-1]  # descending variance
            eigenvectors = eigenvectors[:, order]
            scales = scales[order]

        self._eigenvectors = eigenvectors
        self._forward_scales = scales
        return eigenvectors, scales

    def principal_directions(self) -> NDArray[np.float64]:
        """Eigenvectors of the geometry as **unit columns** (defensive copy).

        For Powell these are the initial search directions (the columns must be
        transposed to rows to match Powell's per-row direction convention).
        Equal to the eigenvectors of the source covariance, since inversion
        preserves eigenvectors.
        """
        directions, _ = self._forward_geometry()
        return directions.copy()

    def principal_scales(self, include_sigma: bool = False) -> NDArray[np.float64]:
        """Per-axis forward standard deviations (defensive copy).

        ``include_sigma=False`` (default) returns the covariance *shape*
        std-devs ``D``; ``include_sigma=True`` returns the full search-distribution
        std-devs ``sigma * D``.  Ordered to match :meth:`principal_directions`.
        Returned directly from the stored forward geometry — never by inverting
        ``B_0`` — so there is no spurious factor of ``sigma`` and no double
        inversion.
        """
        _, scales = self._forward_geometry()
        scales = scales.copy()
        if include_sigma:
            scales = scales * self._sigma
        return scales

    def principal_curvatures(self) -> NDArray[np.float64]:
        """Shape curvature per axis (``1 / D**2``), ordered like the directions.

        Diagnostic reciprocal of the squared shape std-devs; not used for the
        L-BFGS-B matrix math (that uses the stored ``B_0`` directly).
        """
        _, scales = self._forward_geometry()
        return 1.0 / np.maximum(scales, np.sqrt(_EIGENVALUE_FLOOR)) ** 2


# Backwards-compatible aliases: the object was previously named after its
# L-BFGS-B role. Existing imports (`InitialHessian`, `InitialHessianMode`) and
# call sites keep working unchanged.
InitialHessian = InitialGeometry
GeometryMode = InitialHessianMode
