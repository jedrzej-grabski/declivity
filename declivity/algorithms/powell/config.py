from dataclasses import dataclass

from declivity.core.config_base import BaseConfig

__all__ = [
    "PowellConfig",
]


@dataclass
class PowellConfig(BaseConfig):
    """Configuration for Powell's conjugate-direction method.

    Powell is a single-point, derivative-free method, so it inherits
    directly from :class:`BaseConfig` (same as L-BFGS-B).  Termination
    budget is *not* configured here — inject a
    :class:`~declivity.utils.stopping_conditions.StoppingCondition`;
    the ``xtol`` / ``ftol`` fields below are the algorithm's *internal*
    convergence tests, mirroring SciPy's ``method='Powell'`` options.
    """

    xtol: float = 1e-4
    """Line-search tolerance on the step location. Each directional
    minimization runs with tolerance ``100 * xtol`` (relative for the
    unbounded Brent branch, absolute ``xtol`` for the bounded branch) —
    the same convention SciPy's ``_linesearch_powell`` uses."""

    ftol: float = 1e-4
    """Relative function-decrease convergence tolerance. The algorithm
    stops when one full sweep over all directions achieves
    ``2 * (f_before - f_after) <= ftol * (|f_before| + |f_after|) + 1e-20``."""

    ls_maxiter: int = 500
    """Iteration cap for each scalar line-search call (Brent / bounded)."""

    # Diagnostic flags specific to Powell
    diag_delta: bool = False
    """Log the largest single-direction decrease (delta), which direction
    achieved it, and whether the direction set was updated."""

    diag_step_length: bool = False
    """Log the per-iteration displacement norm ||x_k - x_{k-1}||."""

    diag_line_search_iters: bool = False
    """Log the number of function evaluations spent in line searches
    each iteration."""

    diag_direc: bool = False
    """Log condition number and |determinant| of the direction-set matrix
    each iteration (O(n^3) SVD per iteration)."""

    diag_direc_matrix: bool = False
    """Store the full (n, n) direction-set matrix each iteration
    (memory-expensive; enable only to inspect direction evolution)."""

    def validate(self) -> None:
        super().validate()
        if self.xtol <= 0:
            raise ValueError("xtol must be positive.")
        if self.ftol <= 0:
            raise ValueError("ftol must be positive.")
        if self.ls_maxiter < 1:
            raise ValueError("ls_maxiter must be at least 1.")

    def enable_all_diagnostics(self) -> None:
        super().enable_all_diagnostics()
        self.diag_delta = True
        self.diag_step_length = True
        self.diag_line_search_iters = True
        self.diag_direc = True
        self.diag_direc_matrix = True

    def disable_all_diagnostics(self) -> None:
        super().disable_all_diagnostics()
        self.diag_delta = False
        self.diag_step_length = False
        self.diag_line_search_iters = False
        self.diag_direc = False
        self.diag_direc_matrix = False

    def __str__(self) -> str:
        return (
            f"PowellConfig(dimensions={self.dimensions}, "
            f"xtol={self.xtol:.1e}, ftol={self.ftol:.1e})"
        )
