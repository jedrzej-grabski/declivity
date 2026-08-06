import warnings
from typing import Callable, Union, final, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from declivity.algorithms.choices import AlgorithmChoice
from declivity.algorithms.powell.config import PowellConfig
from declivity.core.algorithm_factory import register_optimizer
from declivity.core.base_optimizer import BaseOptimizer, OptimizationResult
from declivity.utils.constraint_handlers import ConstraintHandler
from declivity.utils.line_search import BrentLineSearch, DerivativeFreeLineSearch
from declivity.utils.stopping_conditions import StoppingCondition

if TYPE_CHECKING:
    from declivity.logging.powell_logger import PowellLogData
    from declivity.utils.initial_geometry import InitialGeometry


@final
@register_optimizer(AlgorithmChoice.POWELL, PowellConfig)
class PowellOptimizer(BaseOptimizer["PowellLogData", PowellConfig]):
    """Powell's method — derivative-free conjugate-direction minimization.

    Single-point method: inherits :class:`BaseOptimizer` directly, like
    L-BFGS-B.  The direction set starts as the identity (coordinate
    descent) unless ``initial_directions`` overrides it, and evolves via
    Powell's replacement heuristic.
    """

    def __init__(
        self,
        func: Callable[[NDArray[np.float64]], float],
        initial_point: NDArray[np.float64],
        config: PowellConfig | None = None,
        constraint_handler: ConstraintHandler | None = None,
        stopping_condition: StoppingCondition | None = None,
        lower_bounds: Union[float, NDArray[np.float64], list[float]] = -100.0,
        upper_bounds: Union[float, NDArray[np.float64], list[float]] = 100.0,
        seed: int | np.random.Generator | None = None,
        line_search: DerivativeFreeLineSearch | None = None,
        initial_directions: NDArray[np.float64] | None = None,
        initial_geometry: "InitialGeometry | None" = None,
    ) -> None:
        if config is None:
            config = PowellConfig(dimensions=len(initial_point))

        super().__init__(
            func=func,
            initial_point=initial_point,
            config=config,
            algorithm=AlgorithmChoice.POWELL,
            constraint_handler=constraint_handler,
            stopping_condition=stopping_condition,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            seed=seed,
        )

        self._line_search: DerivativeFreeLineSearch = line_search or BrentLineSearch()

        if initial_geometry is not None:
            if initial_directions is not None:
                raise ValueError(
                    "Pass either initial_directions or initial_geometry, not both."
                )
            # principal_directions() returns eigenvectors as COLUMNS; Powell
            # stores one search direction per ROW, so transpose. Seeding the
            # direction set with the CMA-ES covariance eigenvectors un-rotates
            # coordinate descent onto the landscape's principal axes.
            initial_directions = initial_geometry.principal_directions().T

        if initial_directions is None:
            self._initial_directions = np.eye(self.dimensions)
        else:
            self._initial_directions = np.array(initial_directions, dtype=float)
            if self._initial_directions.shape != (self.dimensions, self.dimensions):
                raise ValueError(
                    "initial_directions must be an (n, n) matrix of row "
                    "direction vectors."
                )
            if np.linalg.matrix_rank(self._initial_directions) != self.dimensions:
                raise ValueError(
                    "initial_directions is rank-deficient — some parameters "
                    "would never be optimized."
                )

    # Feasible step interval

    def _feasible_step_interval(
        self, x: NDArray[np.float64], direction: NDArray[np.float64]
    ) -> tuple[float, float]:

        (nonzero,) = direction.nonzero()
        lower = self.lower_bounds[nonzero]
        upper = self.upper_bounds[nonzero]
        x_nz = x[nonzero]
        d_nz = direction[nonzero]
        low = (lower - x_nz) / d_nz
        high = (upper - x_nz) / d_nz

        pos = d_nz > 0
        lmin = float(np.max(np.where(pos, low, 0) + np.where(pos, 0, high)))
        lmax = float(np.min(np.where(pos, high, 0) + np.where(pos, 0, low)))

        return (lmin, lmax) if lmax >= lmin else (0.0, 0.0)

    # Directional minimization

    def _search_along(
        self,
        x: NDArray[np.float64],
        direction: NDArray[np.float64],
        fval: float,
    ) -> tuple[float, NDArray[np.float64], NDArray[np.float64]]:
        """Minimize the objective along ``direction`` from ``x``.

        Returns ``(f_min, x_new, step_vector)`` where
        ``step_vector = alpha * direction`` — the same contract as
        SciPy's ``_linesearch_powell``.  A zero direction is a no-op
        that reuses ``fval`` without spending evaluations.
        """
        if not np.any(direction):
            return fval, x, direction

        interval = self._feasible_step_interval(x, direction)

        def phi(alpha: float, _x=x, _d=direction) -> float:
            return self.evaluate(_x + alpha * _d)

        result = self._line_search.search(
            phi,
            alpha_bounds=interval,
            tol=self.config.xtol * 100.0,
            maxiter=self.config.ls_maxiter,
        )

        step_vector = result.alpha * direction
        return result.f_min, x + step_vector, step_vector

    # Main loop

    def optimize(self) -> OptimizationResult["PowellLogData"]:
        self.evaluations = 0
        self._begin_run()
        config = self.config
        n = self.dimensions

        x = self.initial_point.copy()
        # SciPy clips an out-of-bounds initial guess into the box (with a
        # warning) before the first evaluation; feasible starts are
        # untouched, keeping trajectories byte-identical.
        if np.any(x < self.lower_bounds) or np.any(x > self.upper_bounds):
            warnings.warn(
                "Initial guess is not within the specified bounds",
                stacklevel=2,
            )
            x = np.clip(x, self.lower_bounds, self.upper_bounds)
        direc = self._initial_directions.copy()

        fval = self.evaluate(x)
        best_fitness = fval
        best_solution = x.copy()
        x_prev_iter = x.copy()

        iteration = 0
        termination_message = None
        # Whether the direction set changed at the end of the *previous*
        # iteration
        direction_replaced = False

        while not self.should_stop(iteration, best_fitness):
            fx = fval
            bigind = 0
            delta = 0.0
            evals_at_start = self.evaluations

            # Sweep: minimize along every direction in the current set.
            for i in range(n):
                fx2 = fval
                fval, x, _ = self._search_along(x, direc[i], fval)
                if (fx2 - fval) > delta:
                    delta = fx2 - fval
                    bigind = i

            iteration += 1

            if fval < best_fitness:
                best_fitness = fval
                best_solution = x.copy()

            if config.diag_direc:
                direc_condition = float(np.linalg.cond(direc))
                direc_determinant = float(abs(np.linalg.det(direc)))
            else:
                direc_condition = 0.0
                direc_determinant = 0.0

            self.logger.log_iteration(
                iteration=iteration,
                evaluations=self.evaluations,
                best_fitness=best_fitness,
                function_value=fval,
                delta=delta,
                big_direction_index=bigind,
                direction_replaced=direction_replaced,
                step_norm=float(np.linalg.norm(x - x_prev_iter)),
                line_search_evals=self.evaluations - evals_at_start,
                direc_condition_number=direc_condition,
                direc_determinant=direc_determinant,
                direction_set=direc,
                best_solution=best_solution,
                current_point=x,
            )
            direction_replaced = False

            # Internal convergence: relative decrease of one full sweep.
            decrease_bound = config.ftol * (abs(fx) + abs(fval)) + 1e-20
            if 2.0 * (fx - fval) <= decrease_bound:
                relative_decrease = fx - fval
                termination_message = (
                    f"Converged: sweep decrease {relative_decrease:.2e} within "
                    f"ftol bound {decrease_bound:.2e}"
                )
                break
            # Budget check placed exactly where SciPy tests maxfun/maxiter:
            # after the ftol test, before the extrapolated point spends
            # another evaluation.  ``termination_message`` stays None so the
            # final-message logic falls through to ``self.stop_message``,
            # identical to the top-of-loop exit.
            if self.should_stop(iteration, best_fitness):
                break
            if np.isnan(fx) and np.isnan(fval):
                termination_message = "NaN region encountered"
                break

            # Extrapolate along the aggregate displacement of the sweep.
            direc1 = x - x_prev_iter
            x_prev_iter = x.copy()
            if not np.any(direc1):
                # No net movement (only possible through exact
                # cancellation) — nothing to extrapolate or replace.
                continue

            if np.all(np.isneginf(self.lower_bounds)) and np.all(
                np.isposinf(self.upper_bounds)
            ):
                lmax = 1.0
            else:
                _, lmax = self._feasible_step_interval(x, direc1)
            x2 = x + min(lmax, 1.0) * direc1
            fx2 = self.evaluate(x2)
            if fx2 < best_fitness:
                best_fitness = fx2
                best_solution = x2.copy()

            # Powell's replacement test: drop the direction of largest
            # decrease and adopt the aggregate displacement if that is
            # profitable
            if fx > fx2:
                t = 2.0 * (fx + fx2 - 2.0 * fval)
                temp = fx - fval - delta
                t *= temp * temp
                temp = fx - fx2
                t -= delta * temp * temp
                if t < 0.0:
                    fval, x, direc1 = self._search_along(x, direc1, fval)
                    if fval < best_fitness:
                        best_fitness = fval
                        best_solution = x.copy()
                    if np.any(direc1):
                        direc[bigind] = direc[-1]
                        direc[-1] = direc1
                        direction_replaced = True

        if termination_message is None:
            termination_message = self.stop_message

        return OptimizationResult(
            best_solution=best_solution,
            best_fitness=best_fitness,
            evaluations=self.evaluations,
            message=termination_message,
            diagnostic=self.get_logs(),
            algorithm=AlgorithmChoice.POWELL,
        )
