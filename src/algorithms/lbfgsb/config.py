from dataclasses import dataclass, field
from typing import Union
import numpy as np
from numpy.typing import NDArray

from src.core.config_base import BaseConfig

__all__ = [
    "LBFGSBConfig",
    "default_budget",
]


def default_budget(dimensions: int) -> int:
    return 10000 * dimensions


@dataclass
class LBFGSBConfig(BaseConfig):

    budget: int = field(default=0)
    """Maximum number of function evaluations (0 = auto: 10000 * dimensions)."""

    population_size: int = field(default=1)
    """Fixed to 1 for L-BFGS-B (single-point optimizer)"""

    initial_hessian: Union[None, float, NDArray[np.float64]] = None
    """Initial Hessian approximation B_0. Controls the scaling of the first iteration
    before L-BFGS corrections are available. Once correction pairs accumulate, the
    Barzilai-Borwein scaling theta = y'y/y's takes over.
        - None (default): identity matrix (B_0 = I)
        - float: scalar * I  (e.g. 0.1 for a gentler first step)
        - 1D array of length n: diagonal matrix diag(array)
        - 2D array (n, n): full symmetric positive-definite Hessian (DENSE mode,
          precomputes Cholesky; subspace minimization extracts the free-variable
          subblock and uses cho_solve). Cost is O(m n^2) per iteration vs O(m n)
          for the diagonal case, so prefer diagonal for large n."""

    persist_initial_hessian: bool = True
    """If True (default), the per-variable relative scaling from initial_hessian is
    preserved throughout the entire optimization: the effective base Hessian is
    theta * diag(initial_hessian) at every iteration. If False, initial_hessian only
    affects the first iteration; afterwards the base Hessian becomes theta * I."""

    m: int = 10
    """Number of L-BFGS correction pairs to store. Controls memory usage and Hessian
    approximation quality. Recommended range: 3-20. Higher values give better Hessian
    approximation but cost more per iteration."""

    factr: float = 1e7
    """Function value convergence tolerance factor. The algorithm stops when:
        (f_old - f_new) / max(|f_old|, |f_new|, 1) <= factr * machine_epsilon
    Typical values: 1e12 (low accuracy), 1e7 (moderate), 1e1 (high accuracy)."""

    pgtol: float = 1e-5
    """Projected gradient convergence tolerance. The algorithm stops when:
        ||projected_gradient||_inf <= pgtol
    where the projected gradient accounts for active bound constraints."""

    ftol: float = 1e-3
    """Sufficient decrease (Armijo) parameter for the line search. Controls how much
    decrease in f is required for a step to be acceptable. In (0, 0.5)."""

    gtol_ls: float = 0.9
    """Curvature condition parameter for the line search (More-Thuente only).
    Controls how close the directional derivative must be to zero. In (ftol, 1)."""

    xtol_ls: float = 0.1
    """Relative tolerance for the line search interval width. The line search terminates
    if the interval of uncertainty is within xtol_ls of the current step."""

    max_ls_iter: int = 20
    """Maximum number of function evaluations per line search call."""

    fd_eps: float = 0.0
    """Finite-difference step size used by both the gradient strategy
    and the optimizer's directional-derivative computation.  0 = auto
    (``sqrt(machine_eps)`` — appropriate for the default ``CentralFD``).
    Override per-strategy via the ``gradient_strategy`` ctor parameter
    if a different step is needed (e.g., for ``ForwardFD``)."""

    # Diagnostic flags specific to L-BFGS-B
    diag_gradient_norm: bool = False
    """Log gradient norm and projected gradient norm each iteration."""

    diag_step_length: bool = False
    """Log line search step length each iteration."""

    diag_theta: bool = False
    """Log L-BFGS scaling factor theta each iteration."""

    diag_num_free: bool = False
    """Log number of free variables (not at bounds) each iteration."""

    diag_line_search_iters: bool = False
    """Log number of line search function evaluations each iteration."""

    # Derived parameters
    maxit: int = field(init=False)
    """Maximum iterations (budget, since 1 eval per function call minimum)."""

    _fd_eps_actual: float = field(init=False, repr=False)
    """Actual finite difference epsilon (computed from fd_eps)."""

    def __post_init__(self) -> None:
        if self.budget <= 0:
            self.budget = default_budget(self.dimensions)
        self.population_size = 1
        self._recalculate_derived_params()

    def _recalculate_derived_params(self) -> None:
        eps = np.finfo(float).eps
        if self.fd_eps <= 0:
            self._fd_eps_actual = eps**0.5
        else:
            self._fd_eps_actual = self.fd_eps

        self.maxit = self.budget
        self.validate()

    def validate(self) -> None:
        if self.dimensions <= 0:
            raise ValueError("Dimensions must be a positive integer.")
        if self.budget <= 0:
            raise ValueError("Budget must be a positive integer.")
        if self.m < 1:
            raise ValueError("m (memory size) must be at least 1.")
        if self.factr < 0:
            raise ValueError("factr must be non-negative.")
        if self.pgtol < 0:
            raise ValueError("pgtol must be non-negative.")

    def __setattr__(self, name: str, value) -> None:
        super().__setattr__(name, value)
        if name in ("budget", "fd_eps") and hasattr(self, "maxit"):
            self._recalculate_derived_params()

    def enable_all_diagnostics(self) -> None:
        super().enable_all_diagnostics()
        self.diag_gradient_norm = True
        self.diag_step_length = True
        self.diag_theta = True
        self.diag_num_free = True
        self.diag_line_search_iters = True

    def __str__(self) -> str:
        return (
            f"LBFGSBConfig(dimensions={self.dimensions}, budget={self.budget}, "
            f"m={self.m}, factr={self.factr:.1e}, pgtol={self.pgtol:.1e})"
        )
