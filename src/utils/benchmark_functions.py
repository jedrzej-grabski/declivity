"""
A collection of benchmark functions for testing optimization algorithms.
"""

import numpy as np
from numpy.typing import NDArray

from opfunu.cec_based import cec2017


class BenchmarkFunction:
    """Base class for benchmark functions."""

    def __init__(self, dimensions: int):
        """
        Initialize a benchmark function.

        Args:
            dimensions: Number of dimensions for the function
        """
        self.dimensions = dimensions

    def __call__(self, x: NDArray[np.float64]) -> float:
        """
        Evaluate the function at point x.

        Args:
            x: Input vector of length self.dimensions

        Returns:
            Function value at x
        """
        raise NotImplementedError("Subclasses must implement this method")

    @property
    def bounds(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Get the bounds of the function.

        Returns:
            Tuple of (lower_bounds, upper_bounds)
        """
        raise NotImplementedError("Subclasses must implement this method")

    @property
    def global_minimum(self) -> tuple[NDArray[np.float64], float]:
        """
        Get the global minimum of the function.

        Returns:
            Tuple of (optimal_solution, optimal_value)
        """
        raise NotImplementedError("Subclasses must implement this method")


class Sphere(BenchmarkFunction):
    """
    Sphere function.
    f(x) = sum(x_i^2)
    Global minimum: f(0, 0, ..., 0) = 0
    """

    def __call__(self, x: NDArray[np.float64]) -> float:
        return np.sum(x**2)

    @property
    def bounds(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return -100.0 * np.ones(self.dimensions), 100.0 * np.ones(self.dimensions)

    @property
    def global_minimum(self) -> tuple[NDArray[np.float64], float]:
        return np.zeros(self.dimensions), 0.0


class Ellipsoid(BenchmarkFunction):
    """
    Ellipsoid function.
    f(x) = sum(10^(6*(i-1)/(n-1)) * x_i^2)
    Global minimum: f(0, 0, ..., 0) = 0
    """

    def __call__(self, x: NDArray[np.float64]) -> float:
        n = len(x)
        if n == 1:
            return float(x[0] ** 2)
        scales = 10.0 ** (6.0 * np.arange(n) / (n - 1))
        return float(np.sum(scales * x**2))

    @property
    def bounds(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return -100.0 * np.ones(self.dimensions), 100.0 * np.ones(self.dimensions)

    @property
    def global_minimum(self) -> tuple[NDArray[np.float64], float]:
        return np.zeros(self.dimensions), 0.0


class Rosenbrock(BenchmarkFunction):
    """
    Rosenbrock function (Valley function).
    f(x) = sum(100 * (x_{i+1} - x_i^2)^2 + (1 - x_i)^2)
    Global minimum: f(1, 1, ..., 1) = 0
    """

    def __call__(self, x: NDArray[np.float64]) -> float:
        return np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)

    @property
    def bounds(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return -5.0 * np.ones(self.dimensions), 10.0 * np.ones(self.dimensions)

    @property
    def global_minimum(self) -> tuple[NDArray[np.float64], float]:
        return np.ones(self.dimensions), 0.0


class Rastrigin(BenchmarkFunction):
    """
    Rastrigin function.
    f(x) = 10*d + sum(x_i^2 - 10*cos(2*pi*x_i))
    Global minimum: f(0, 0, ..., 0) = 0
    """

    def __call__(self, x: NDArray[np.float64]) -> float:
        return 10 * self.dimensions + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

    def gradient(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Analytical gradient: 2*x_i + 20*pi*sin(2*pi*x_i)."""
        return 2.0 * x + 20.0 * np.pi * np.sin(2.0 * np.pi * x)

    @property
    def bounds(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return -5.12 * np.ones(self.dimensions), 5.12 * np.ones(self.dimensions)

    @property
    def global_minimum(self) -> tuple[NDArray[np.float64], float]:
        return np.zeros(self.dimensions), 0.0


class Griewank(BenchmarkFunction):
    """
    Griewank function.
    f(x) = 1 + sum(x_i^2 / 4000) - prod(cos(x_i / sqrt(i)))
    Global minimum: f(0, 0, ..., 0) = 0

    Multimodal with many local minima but a soft global trend that points
    towards the origin — well suited for testing whether a hybrid
    CMA-ES -> L-BFGS-B handoff can locate a basin and then descend it.
    """

    def __call__(self, x: NDArray[np.float64]) -> float:
        indices = np.arange(1, len(x) + 1)
        sum_term = np.sum(x**2) / 4000.0
        prod_term = np.prod(np.cos(x / np.sqrt(indices)))
        return float(1.0 + sum_term - prod_term)

    def gradient(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Analytical gradient.

        df/dx_i = x_i / 2000 + sin(x_i/sqrt(i)) / sqrt(i) * prod_{j != i} cos(x_j/sqrt(j))
        """
        indices = np.arange(1, len(x) + 1)
        sqrt_indices = np.sqrt(indices)
        cosines = np.cos(x / sqrt_indices)
        sines = np.sin(x / sqrt_indices)

        # prod_{j != i} cos(...) = total_product / cos(...) when cos is non-zero;
        # when a cosine is exactly zero the leave-one-out product collapses,
        # so handle it directly.
        zeros = cosines == 0.0
        if np.any(zeros):
            leave_one_out = np.empty_like(cosines)
            for i in range(len(cosines)):
                mask = np.ones_like(cosines, dtype=bool)
                mask[i] = False
                leave_one_out[i] = np.prod(cosines[mask])
        else:
            total_product = np.prod(cosines)
            leave_one_out = total_product / cosines

        return x / 2000.0 + sines / sqrt_indices * leave_one_out

    @property
    def bounds(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return -600.0 * np.ones(self.dimensions), 600.0 * np.ones(self.dimensions)

    @property
    def global_minimum(self) -> tuple[NDArray[np.float64], float]:
        return np.zeros(self.dimensions), 0.0


class Ackley(BenchmarkFunction):
    """
    Ackley function.
    f(x) = -20*exp(-0.2*sqrt(1/d * sum(x_i^2))) - exp(1/d * sum(cos(2*pi*x_i))) + 20 + e
    Global minimum: f(0, 0, ..., 0) = 0
    """

    def __call__(self, x: NDArray[np.float64]) -> float:
        term1 = -20.0 * np.exp(-0.2 * np.sqrt(np.mean(x**2)))
        term2 = -np.exp(np.mean(np.cos(2 * np.pi * x)))
        return term1 + term2 + 20.0 + np.e

    def gradient(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Analytical gradient. Returns zero at the (degenerate) origin."""
        n = len(x)
        rms = np.sqrt(np.mean(x**2))
        if rms == 0.0:
            return np.zeros_like(x)
        term1 = (4.0 / n) * np.exp(-0.2 * rms) * x / rms
        cos_mean = np.mean(np.cos(2 * np.pi * x))
        term2 = (2.0 * np.pi / n) * np.exp(cos_mean) * np.sin(2 * np.pi * x)
        return term1 + term2

    @property
    def bounds(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return -32.768 * np.ones(self.dimensions), 32.768 * np.ones(self.dimensions)

    @property
    def global_minimum(self) -> tuple[NDArray[np.float64], float]:
        return np.zeros(self.dimensions), 0.0


class Schwefel(BenchmarkFunction):
    """
    Schwefel function.
    f(x) = 418.9829*d - sum(x_i * sin(sqrt(abs(x_i))))
    Global minimum: f(420.9687, 420.9687, ..., 420.9687) = 0
    """

    def __call__(self, x: NDArray[np.float64]) -> float:
        return 418.9829 * self.dimensions - np.sum(x * np.sin(np.sqrt(np.abs(x))))

    @property
    def bounds(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return -500.0 * np.ones(self.dimensions), 500.0 * np.ones(self.dimensions)

    @property
    def global_minimum(self) -> tuple[NDArray[np.float64], float]:
        return 420.9687 * np.ones(self.dimensions), 0.0


class RotatedEllipsoid(BenchmarkFunction):
    """Ellipsoid function with a rotation applied to the input.

    f(x) = sum(scale_i * z_i^2)  where z = R x and R is an orthogonal matrix.

    The Hessian is H = 2 R' diag(scales) R, which is a full (non-diagonal) matrix
    whenever R is not a permutation. This makes the problem coordinate-system
    dependent and tests whether an optimizer can handle off-diagonal curvature.

    Four rotation modes are provided:

    - "uniform_45": chain of 45-degree Givens rotations in consecutive planes
      (1,2), (2,3), ..., (n-1,n). Produces a single dense orthogonal matrix
      that couples all variables with symmetric mixing.

    - "golden": chain of Givens rotations using the golden angle (137.5 degrees)
      multiplied by the plane index. Avoids periodic alignment, producing an
      irregular coupling pattern with no repeating block structure.

    - "random": a uniformly random orthogonal matrix from the QR decomposition
      of a random Gaussian matrix. Maximally unstructured; no coordinate
      direction is privileged.

    - A user-supplied orthogonal matrix passed directly.
    """

    def __init__(
        self,
        dimensions: int,
        rotation: str | NDArray[np.float64] = "uniform_45",
        seed: int = 0,
    ):
        super().__init__(dimensions)

        if isinstance(rotation, str):
            self.rotation_matrix = self._build_rotation(rotation, seed)
            self.rotation_name = rotation
        else:
            self.rotation_matrix = np.asarray(rotation, dtype=float)
            self.rotation_name = "custom"

        n = dimensions
        self._scales = 10.0 ** (6.0 * np.arange(n) / max(n - 1, 1))
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            self._hessian = 2.0 * (
                self.rotation_matrix.T @ np.diag(self._scales) @ self.rotation_matrix
            )

    def _build_rotation(self, mode: str, seed: int) -> NDArray[np.float64]:
        n = self.dimensions
        if mode == "uniform_45":
            return self._givens_chain(
                [np.radians(45)] * (n - 1)
            )
        elif mode == "golden":
            golden_angle = np.radians(137.5)
            angles = [(k + 1) * golden_angle for k in range(n - 1)]
            return self._givens_chain(angles)
        elif mode == "random":
            rng = np.random.default_rng(seed)
            random_matrix = rng.standard_normal((n, n))
            q, r = np.linalg.qr(random_matrix)
            # Ensure proper rotation (det = +1)
            q *= np.sign(np.diag(r))
            return q
        else:
            raise ValueError(
                f"Unknown rotation mode: {mode}. "
                f"Use 'uniform_45', 'golden', 'random', or pass a matrix."
            )

    def _givens_chain(self, angles: list[float]) -> NDArray[np.float64]:
        """Build a rotation matrix from a chain of Givens rotations in
        consecutive planes (0,1), (1,2), ..., (n-2,n-1)."""
        n = self.dimensions
        rotation = np.eye(n)
        for k, angle in enumerate(angles):
            givens = np.eye(n)
            c, s = np.cos(angle), np.sin(angle)
            givens[k, k] = c
            givens[k, k + 1] = -s
            givens[k + 1, k] = s
            givens[k + 1, k + 1] = c
            rotation = givens @ rotation
        return rotation

    def __call__(self, x: NDArray[np.float64]) -> float:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            z = self.rotation_matrix @ x
        return float(np.sum(self._scales * z**2))

    def gradient(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Analytical gradient: 2 R' diag(scales) R x = H x."""
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            return self._hessian @ x

    @property
    def hessian(self) -> NDArray[np.float64]:
        """The full Hessian matrix H = 2 R' diag(scales) R."""
        return self._hessian

    @property
    def hessian_diagonal(self) -> NDArray[np.float64]:
        """Diagonal of the Hessian (the best a diagonal approximation can do)."""
        return np.diag(self._hessian)

    @property
    def bounds(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return -100.0 * np.ones(self.dimensions), 100.0 * np.ones(self.dimensions)

    @property
    def global_minimum(self) -> tuple[NDArray[np.float64], float]:
        return np.zeros(self.dimensions), 0.0


class RippledEllipsoid(BenchmarkFunction):
    """Anisotropic multimodal benchmark: ill-conditioned ellipsoid + Rastrigin-style ripples.

    ``f(x) = sum(scale_i * x_i**2 + amplitude * (1 - cos(2 pi x_i)))``

    The quadratic term gives an anisotropic basin with prescribed condition
    number; the cosine ripples add many local minima centred at integer
    coordinates. At the global optimum x = 0 the Hessian is
    ``diag(2 scale_i + 4 pi**2 amplitude)`` — the ratio of largest to
    smallest eigenvalue is

        (2 * condition + 4 pi**2 amplitude) / (2 + 4 pi**2 amplitude)

    so for ``condition >> 2 pi**2 amplitude`` the eigenvalues span the
    full conditioning, while a tighter amplitude/condition ratio keeps
    multimodality alive across more dimensions.

    Wrap with ``RotatedFunction(...)`` to also couple the curvature
    off-axis — that is the regime where L-BFGS-B with B_0 = I has the
    hardest time and where the CMA-ES C^{-1} handoff should win.
    """

    def __init__(
        self,
        dimensions: int,
        condition: float = 100.0,
        amplitude: float = 10.0,
        bound: float = 5.12,
    ):
        super().__init__(dimensions)
        self.condition = condition
        self.amplitude = amplitude
        self.bound = bound
        n = dimensions
        if n == 1:
            self._scales = np.array([1.0])
        else:
            self._scales = condition ** (np.arange(n) / (n - 1))

    def __call__(self, x: NDArray[np.float64]) -> float:
        quad = float(np.sum(self._scales * x**2))
        ripples = float(self.amplitude * np.sum(1.0 - np.cos(2.0 * np.pi * x)))
        return quad + ripples

    def gradient(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        quad_grad = 2.0 * self._scales * x
        ripple_grad = 2.0 * np.pi * self.amplitude * np.sin(2.0 * np.pi * x)
        return quad_grad + ripple_grad

    @property
    def bounds(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return -self.bound * np.ones(self.dimensions), self.bound * np.ones(self.dimensions)

    @property
    def global_minimum(self) -> tuple[NDArray[np.float64], float]:
        return np.zeros(self.dimensions), 0.0


class RotatedFunction(BenchmarkFunction):
    """Generic rotation wrapper for any BenchmarkFunction with a gradient.

    ``f_rotated(x) = f_base(R @ x)``

    so the global optimum at z=0 in the base function (when present) maps
    back to x=0 in the rotated function — origin-centred problems stay
    origin-centred. Bounds are inherited from the base function and
    interpreted in *x-space*; the rotated z = R @ x may go outside the
    base's typical range, which is fine for analytic functions defined
    everywhere.

    The local Hessian of the rotated function at x is
    ``R^T H_base(R x) R``, so any anisotropy in the base function's
    curvature becomes coupled (off-diagonal) in coordinates. This is what
    makes the rotation useful for demonstrating that L-BFGS-B with
    B_0 = I cannot recover the coupling without enough corrections.

    Rotation modes: same as ``RotatedEllipsoid``
    ("uniform_45" / "golden" / "random" / supplied matrix).
    """

    def __init__(
        self,
        base: BenchmarkFunction,
        rotation: str | NDArray[np.float64] = "random",
        seed: int = 0,
        name_suffix: str | None = None,
    ):
        super().__init__(base.dimensions)
        if not hasattr(base, "gradient"):
            raise ValueError(
                f"Base function {type(base).__name__} has no gradient method; "
                f"RotatedFunction needs it to compute the rotated gradient."
            )
        self.base = base
        self.rotation_matrix = self._build_rotation(rotation, seed)
        self.rotation_name = (
            name_suffix or (rotation if isinstance(rotation, str) else "custom")
        )

    def _build_rotation(
        self, rotation: str | NDArray[np.float64], seed: int
    ) -> NDArray[np.float64]:
        if isinstance(rotation, np.ndarray):
            matrix = np.asarray(rotation, dtype=float)
            if matrix.shape != (self.dimensions, self.dimensions):
                raise ValueError(
                    f"Rotation matrix shape {matrix.shape} doesn't match "
                    f"dimension {self.dimensions}."
                )
            return matrix
        n = self.dimensions
        if rotation == "uniform_45":
            angles = [np.radians(45)] * (n - 1)
            return self._givens_chain(angles)
        if rotation == "golden":
            golden = np.radians(137.5)
            angles = [(k + 1) * golden for k in range(n - 1)]
            return self._givens_chain(angles)
        if rotation == "random":
            rng = np.random.default_rng(seed)
            q, r = np.linalg.qr(rng.standard_normal((n, n)))
            q *= np.sign(np.diag(r))  # det = +1
            return q
        raise ValueError(
            f"Unknown rotation mode {rotation!r}; use 'uniform_45', 'golden', "
            f"'random', or pass an explicit matrix."
        )

    def _givens_chain(self, angles: list[float]) -> NDArray[np.float64]:
        n = self.dimensions
        rotation = np.eye(n)
        for k, angle in enumerate(angles):
            givens = np.eye(n)
            c, s = np.cos(angle), np.sin(angle)
            givens[k, k] = c
            givens[k, k + 1] = -s
            givens[k + 1, k] = s
            givens[k + 1, k + 1] = c
            rotation = givens @ rotation
        return rotation

    def __call__(self, x: NDArray[np.float64]) -> float:
        z = self.rotation_matrix @ x
        return float(self.base(z))

    def gradient(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        z = self.rotation_matrix @ x
        return self.rotation_matrix.T @ self.base.gradient(z)

    @property
    def bounds(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return self.base.bounds

    @property
    def global_minimum(self) -> tuple[NDArray[np.float64], float]:
        opt_z, opt_v = self.base.global_minimum
        # R is orthogonal, so x* = R^T @ z*. For z* = 0 this stays at 0.
        return self.rotation_matrix.T @ opt_z, opt_v


class CEC17Function(BenchmarkFunction):

    def __init__(self, dimensions: int, function_id: int):
        """
        Initialize a CEC benchmark function.

        Args:
            dimensions: Number of dimensions for the function
            function_id: ID of the CEC function to use
        """
        super().__init__(dimensions)

        if function_id < 1 or function_id > 30:
            raise ValueError("Function ID must be between 1 and 29.")

        self.function_id = function_id

        fname = f"F{function_id}2017"
        self.func = getattr(cec2017, fname)(dimensions)

    def __call__(self, x: NDArray[np.float64]) -> float:
        """
        Evaluate the CEC function at point x.

        Args:
            x: Input vector of length self.dimensions

        Returns:
            Function value at x
        """
        return self.func.evaluate(x)

    @property
    def bounds(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Get the bounds of the CEC function.

        Returns:
            Tuple of (lower_bounds, upper_bounds)
        """
        return self.func.lower, self.func.upper

    @property
    def global_minimum(self) -> tuple[NDArray[np.float64], float]:
        """
        Get the global minimum of the CEC function.

        Returns:
            Tuple of (optimal_solution, optimal_value)
        """
        return self.func.optimal_solution, self.func.optimal_value
