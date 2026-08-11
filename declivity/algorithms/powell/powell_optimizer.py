import warnings
from typing import TYPE_CHECKING, Callable, Union, final

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

    The direction set starts as the identity unless ``initial_directions``
    overrides it, and evolves via Powell's replacement heuristic.

    Constraints come from the injected
    :class:`~declivity.utils.constraint_handlers.ConstraintHandler`.  When it
    bounds a ray via ``feasible_step_interval`` the line search is confined to
    that interval; when it returns ``None`` the search runs unconstrained and
    every point is passed through ``repair``.  ``penalty`` applies to every
    evaluation in either case.
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
            # principal_directions() returns eigenvectors as columns; Powell
            # stores one direction per row.
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

        self._final_directions: NDArray[np.float64] | None = None

    @property
    def final_directions(self) -> NDArray[np.float64] | None:
        """The direction set (rows) at the end of the last ``optimize()`` call
        (defensive copy); ``None`` before any run."""
        if self._final_directions is None:
            return None
        return self._final_directions.copy()

    # Directional minimization

    def _search_along(
        self,
        x: NDArray[np.float64],
        direction: NDArray[np.float64],
        fval: float,
    ) -> tuple[float, NDArray[np.float64], NDArray[np.float64]]:
        """Minimize the objective along ``direction`` from ``x``.

        Returns ``(f_min, x_new, step_vector)`` where
        ``step_vector = alpha * direction``.  A zero direction is a no-op that
        reuses ``fval`` without spending evaluations.

        When the handler bounds the ray the search is confined to that
        interval and nothing is repaired; when it returns ``None`` the search
        runs unconstrained and every point is passed through ``repair``.
        """
        if not np.any(direction):
            return fval, x, direction

        bounded_interval = self.constraint_handler.feasible_step_interval(x, direction)
        repair_needed = bounded_interval is None
        interval = (-np.inf, np.inf) if repair_needed else bounded_interval

        if repair_needed:

            def phi(alpha: float, _x=x, _d=direction) -> float:
                return self.evaluate(self.constraint_handler.repair(_x + alpha * _d))

        else:

            def phi(alpha: float, _x=x, _d=direction) -> float:
                return self.evaluate(_x + alpha * _d)

        result = self._line_search.search(
            phi,
            alpha_bounds=interval,
            tol=self.config.xtol * 100.0,
            maxiter=self.config.ls_maxiter,
        )

        if repair_needed:
            x_new = self.constraint_handler.repair(x + result.alpha * direction)
            return result.f_min, x_new, x_new - x

        step_vector = result.alpha * direction
        return result.f_min, x + step_vector, step_vector

    # Main loop

    def optimize(self) -> OptimizationResult["PowellLogData"]:
        self.evaluations = 0
        self._begin_run()
        config = self.config
        n = self.dimensions

        x = self.initial_point.copy()
        # An out-of-bounds initial guess is projected (with a warning) before
        # the first evaluation; feasible starts are untouched.
        if not self.constraint_handler.is_feasible(x):
            warnings.warn(
                "Initial guess is not within the specified bounds",
                stacklevel=2,
            )
            x = self.constraint_handler.repair(x)
        direc = self._initial_directions.copy()

        fval = self.evaluate(x)
        best_fitness = fval
        best_solution = x.copy()
        x_prev_iter = x.copy()

        iteration = 0
        termination_message = None
        # Whether the direction set changed at the end of the previous
        # iteration.
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
            # Budget check after the ftol test, before the extrapolated point
            # spends another evaluation.
            if self.should_stop(iteration, best_fitness):
                break
            if np.isnan(fx) and np.isnan(fval):
                termination_message = "NaN region encountered"
                break

            # Extrapolate along the aggregate displacement of the sweep.
            direc1 = x - x_prev_iter
            x_prev_iter = x.copy()
            if not np.any(direc1):
                continue

            # The extrapolated point obeys the same handler interval as the
            # line searches; an unbounded ray gives lmax = inf.
            extrapolation_interval = self.constraint_handler.feasible_step_interval(
                x, direc1
            )
            lmax = (
                np.inf if extrapolation_interval is None else extrapolation_interval[1]
            )
            x2 = x + min(lmax, 1.0) * direc1
            if extrapolation_interval is None:
                x2 = self.constraint_handler.repair(x2)
            fx2 = self.evaluate(x2)
            if fx2 < best_fitness:
                best_fitness = fx2
                best_solution = x2.copy()

            # Replacement test: drop the direction of largest decrease and
            # adopt the aggregate displacement if that is profitable.
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

        self._final_directions = direc
        return OptimizationResult(
            best_solution=best_solution,
            best_fitness=best_fitness,
            evaluations=self.evaluations,
            message=termination_message,
            diagnostic=self.get_logs(),
            algorithm=AlgorithmChoice.POWELL,
        )
