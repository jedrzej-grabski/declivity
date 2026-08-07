"""
Shared learned-geometry object for the single-point local optimizers.

One curvature/quadratic model, typically built from a CMA-ES covariance, that
seeds all three local optimizers:

- **L-BFGS-B** consumes it as the initial Hessian ``B_0`` (``multiply`` /
  ``solve`` / ``quadratic_form`` / ...).
- **Powell** consumes :meth:`InitialGeometry.principal_directions`, the
  eigenvectors of the covariance, as its initial search-direction set.
- **Nelder-Mead** consumes :meth:`InitialGeometry.axis_steps` /
  :meth:`InitialGeometry.principal_scales` to shape its initial simplex.

The stored quantity is curvature ``B_0`` (large eigenvalue = steep).  The
covariance-to-curvature inversion happens once, in
:meth:`InitialGeometry.from_covariance`, which also stores the forward
eigendecomposition ``(B, D, sigma)`` so the direction / simplex accessors need
no second ``eigh``.

``InitialHessian`` is an alias of ``InitialGeometry`` kept for backwards
compatibility.
"""

from enum import Enum, StrEnum

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import cho_factor, cho_solve

# Floor applied to (squared) eigenvalues before inversion, so a collapsed
# covariance direction does not send the reciprocal to inf.  Distinct from the
# simplex edge floor (``min_step``) in ``CovarianceSimplexInitializer``.
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
    """Use ``(sigma^2 C)^{-1}``, which accounts for the CMA-ES global
    step-size scaling."""

    IDENTITY = "identity"
    """Drop the covariance and use the isotropic default ``B_0 = I``.  The
    control case: same starting point, no covariance information."""


def covariance_to_hessian_matrix(
    transform: "HandoffTransform | str",
    eigenvectors: NDArray[np.float64],
    eigenvalues_sqrt: NDArray[np.float64],
    sigma: float,
) -> NDArray[np.float64] | None:
    """Turn a covariance eigendecomposition ``(B, D)`` into a curvature matrix.

    ``C = B @ diag(D**2) @ B.T``; the returned matrix is the L-BFGS-B initial
    Hessian ``B_0``, proportional to ``C^{-1}``.  ``IDENTITY`` returns ``None``
    (the L-BFGS-B default ``B_0 = I``).
    """
    transform = str(transform)
    if transform == HandoffTransform.IDENTITY:
        return None

    eigenvalues = np.maximum(np.asarray(eigenvalues_sqrt) ** 2, _EIGENVALUE_FLOOR)

    # Floored 1/eigenvalues can reach 1e30, which raises spurious numpy
    # divide/overflow warnings on the intermediates.
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

    and the geometry accessors consumed by Powell / Nelder-Mead:

    - principal_directions(): eigenvectors of the geometry (Powell directions)
    - principal_scales(): per-axis forward std-devs (Nelder-Mead simplex extent)
    - axis_steps(...): ready-to-use simplex edge vectors

    Use :meth:`identity`, :meth:`from_curvature`, or :meth:`from_covariance`;
    the plain constructor accepts a raw curvature (None / scalar / 1D / 2D).
    """

    def __init__(
        self,
        initial_hessian,
        num_dimensions: int,
    ):
        # Forward-geometry fields: populated eagerly by ``from_covariance``,
        # or lazily by ``_forward_geometry`` for a raw curvature.
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

    # Constructors.

    @classmethod
    def identity(cls, num_dimensions: int) -> "InitialGeometry":
        """Isotropic geometry: ``B_0 = I``, unit directions, uniform scales.

        L-BFGS-B behaves as ``config.initial_hessian=None``, Powell gets the
        coordinate directions, Nelder-Mead an isotropic simplex.
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
        the curvature ``B_0`` (:class:`HandoffTransform`).  The forward
        eigendecomposition ``(B, D, sigma)`` is stored as-is, with columns
        ordered by descending variance, so the direction / scale accessors do
        not re-invert.  ``transform=IDENTITY`` returns an isotropic
        :meth:`identity` geometry.
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

        # Store the forward geometry of the source covariance, independent of
        # the transform, with columns ordered by descending variance.  CMA-ES
        # returns eigenvalues ascending.
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

    # Geometry accessors, consumed by Powell / Nelder-Mead.

    def _forward_geometry(
        self,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return ``(eigenvectors, forward_scales)`` of the source geometry.

        ``eigenvectors`` are unit columns; ``forward_scales`` are the per-axis
        standard deviations of the covariance (shape only, no ``sigma``), i.e.
        ``1/sqrt(curvature eigenvalue)``.  Populated by
        :meth:`from_covariance`, otherwise derived once and cached.
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
        """Eigenvectors of the geometry as unit columns (defensive copy).

        Powell uses these as its initial search directions, transposed to rows.
        Equal to the eigenvectors of the source covariance, since inversion
        preserves eigenvectors.
        """
        directions, _ = self._forward_geometry()
        return directions.copy()

    def principal_scales(self, include_sigma: bool = False) -> NDArray[np.float64]:
        """Per-axis forward standard deviations (defensive copy).

        ``include_sigma=False`` (default) returns the covariance shape std-devs
        ``D``; ``include_sigma=True`` returns the full search-distribution
        std-devs ``sigma * D``.  Ordered to match :meth:`principal_directions`.
        """
        _, scales = self._forward_geometry()
        scales = scales.copy()
        if include_sigma:
            scales = scales * self._sigma
        return scales

    def principal_curvatures(self) -> NDArray[np.float64]:
        """Shape curvature per axis (``1 / D**2``), ordered like the directions.

        Diagnostic only; the L-BFGS-B matrix math uses the stored ``B_0``.
        """
        _, scales = self._forward_geometry()
        return 1.0 / np.maximum(scales, np.sqrt(_EIGENVALUE_FLOOR)) ** 2

    def axis_steps(
        self,
        base_size: float,
        normalize: bool = True,
        ratio_floor: float = 1e-3,
        min_step: float = 0.0,
        absolute: bool = False,
    ) -> NDArray[np.float64]:
        """Simplex edge vectors as columns ``(n, n)``: column ``k`` = edge ``k``.

        Each column is a principal direction scaled to a length:

        - ``absolute=True``: ``length_k = sigma * D_k``, the true CMA-ES
          search-distribution extent, which collapses as CMA-ES converges.
        - ``normalize=True`` (default): ``length_k = base_size *
          clip(D_k/max(D), ratio_floor, 1)``, relative anisotropy with the
          absolute size decoupled into ``base_size``.
        - otherwise: ``length_k = base_size * D_k``.

        Every length is floored at ``min_step``.
        """
        directions, scales = self._forward_geometry()
        if absolute:
            lengths = self._sigma * scales
        elif normalize:
            scale_max = float(np.max(scales)) if scales.size else 1.0
            if scale_max <= 0:
                scale_max = 1.0
            shape = np.clip(scales / scale_max, ratio_floor, 1.0)
            lengths = base_size * shape
        else:
            lengths = base_size * scales
        lengths = np.maximum(lengths, min_step)
        return directions * lengths


# Backwards-compatible aliases from when the object was named after its
# L-BFGS-B role.
InitialHessian = InitialGeometry
GeometryMode = InitialHessianMode
