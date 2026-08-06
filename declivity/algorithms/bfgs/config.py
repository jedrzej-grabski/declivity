from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from declivity.core.config_base import BaseConfig

__all__ = [
    "BFGSConfig",
]


@dataclass
class BFGSConfig(BaseConfig):
    """Configuration for the BFGS quasi-Newton method.

    BFGS is a single-point, gradient-based method, so it inherits directly
    from :class:`BaseConfig` (same as L-BFGS-B and Powell).  Termination
    *budget* is not configured here — inject a
    :class:`~declivity.utils.stopping_conditions.StoppingCondition` (this
    replaces SciPy's ``maxiter`` / ``maxfun``).  The fields below are the
    algorithm's *internal* convergence tests and line-search parameters,
    mirroring SciPy's ``method='BFGS'`` options.
    """

    gtol: float = 1e-5
    """Gradient convergence tolerance. The algorithm stops when
    ``vecnorm(projected_gradient, ord=norm) <= gtol``.  The projected gradient
    equals the plain gradient when bounds are infinite, so this reproduces
    SciPy's ``gtol`` test exactly in the unconstrained regime."""

    norm: float = np.inf
    """Order of the norm used for the gradient convergence test.  SciPy's
    default is ``inf`` (max-abs); ``-inf`` is min-abs, ``2`` is Euclidean."""

    xrtol: float = 0.0
    """Relative tolerance on the step. Terminate successfully when
    ``alpha * ||p|| <= xrtol * (xrtol + ||x||)`` — SciPy's ``xrtol`` test.
    The default 0 disables it."""

    c1: float = 1e-4
    """Armijo (sufficient-decrease) parameter for the Wolfe line search.
    Passed as the line search's ``ftol``. Must satisfy ``0 < c1 < c2 < 1``."""

    c2: float = 0.9
    """Curvature parameter for the Wolfe line search. Passed as the line
    search's ``gtol``. Must satisfy ``0 < c1 < c2 < 1``."""

    initial_inverse_hessian: None | float | NDArray[np.float64] = None
    """Initial inverse-Hessian approximation ``H_0`` (SciPy's ``hess_inv0``).
        - None (default): identity ``H_0 = I``
        - float: ``scalar * I``
        - 1D array of length n: diagonal ``diag(array)`` (entries > 0)
        - 2D array (n, n): full symmetric positive-definite matrix (Cholesky-checked)
    A supplied ``initial_geometry`` (curvature ``B_0``) overrides this, seeding
    ``H_0 = B_0^{-1}``."""

    fd_eps: float = 0.0
    """Finite-difference step size used by both the gradient strategy and the
    optimizer's directional-derivative computation.  0 = auto
    (``sqrt(machine_eps)``)."""

    xtol_ls: float = 1e-14
    """Relative tolerance for the line-search interval width (dcsrch ``xtol``).
    Matches SciPy's ``line_search_wolfe1`` default."""

    max_ls_iter: int = 100
    """Maximum number of function evaluations per line-search call.  Matches
    SciPy's ``scalar_search_wolfe1`` ``maxiter``."""

    # Diagnostic flags specific to BFGS
    diag_gradient_norm: bool = False
    """Log the gradient norm (convergence measure) each iteration."""

    diag_step_length: bool = False
    """Log the accepted line-search step length each iteration."""

    diag_curvature: bool = False
    """Log the BFGS curvature ``yk . sk`` each iteration (the update is
    well-conditioned when this is positive)."""

    diag_hessian_condition: bool = False
    """Log the condition number of the inverse-Hessian ``Hk`` each iteration
    (O(n^3) SVD per iteration; enable only to inspect conditioning)."""

    # Derived parameters
    _fd_eps_actual: float = field(init=False, repr=False)
    """Actual finite difference epsilon (computed from fd_eps)."""

    def __post_init__(self) -> None:
        self._recalculate_derived_params()

    def _recalculate_derived_params(self) -> None:
        eps = np.finfo(float).eps
        if self.fd_eps <= 0:
            self._fd_eps_actual = eps**0.5
        else:
            self._fd_eps_actual = self.fd_eps

        self.validate()

    def validate(self) -> None:
        if self.dimensions <= 0:
            raise ValueError("Dimensions must be a positive integer.")
        if self.gtol < 0:
            raise ValueError("gtol must be non-negative.")
        if self.xrtol < 0:
            raise ValueError("xrtol must be non-negative.")
        if not (0.0 < self.c1 < self.c2 < 1.0):
            raise ValueError("Line-search parameters must satisfy 0 < c1 < c2 < 1.")
        if self.max_ls_iter < 1:
            raise ValueError("max_ls_iter must be at least 1.")

    def __setattr__(self, name: str, value) -> None:
        super().__setattr__(name, value)
        if name == "fd_eps" and hasattr(self, "_fd_eps_actual"):
            self._recalculate_derived_params()

    def enable_all_diagnostics(self) -> None:
        super().enable_all_diagnostics()
        self.diag_gradient_norm = True
        self.diag_step_length = True
        self.diag_curvature = True
        self.diag_hessian_condition = True

    def disable_all_diagnostics(self) -> None:
        super().disable_all_diagnostics()
        self.diag_gradient_norm = False
        self.diag_step_length = False
        self.diag_curvature = False
        self.diag_hessian_condition = False

    def __str__(self) -> str:
        return (
            f"BFGSConfig(dimensions={self.dimensions}, "
            f"gtol={self.gtol:.1e}, c1={self.c1:.1e}, c2={self.c2:.1e})"
        )
