"""
Pure Python implementation of the BFGS algorithm.

A dense quasi-Newton method that maintains an approximation ``Hk`` of the
*inverse* Hessian and takes Newton-like steps ``pk = -Hk @ grad`` along a
Wolfe line search.  This is a faithful port of SciPy's ``_minimize_bfgs``
(Broyden–Fletcher–Goldfarb–Shanno) into declivity's injected-strategy
interface: the line search, gradient strategy, constraint handler, and
stopping condition are all pluggable framework components.

SciPy's ``method='BFGS'`` is unconstrained; here the default bounds are
±inf (identical behaviour).  Every boundary decision is delegated to the
injected :class:`~declivity.utils.constraint_handlers.ConstraintHandler`:
it projects the search direction at active constraints, caps the
line-search step (``max_feasible_step``), repairs accepted points, and
supplies the projected gradient used as the convergence measure.  All four
reduce to their unconstrained forms under the default ±inf handler, so
parity with SciPy is unaffected.

References:
    R. Fletcher, "Practical Methods of Optimization", 2nd ed., Wiley (1987).
    J. Nocedal and S.J. Wright, "Numerical Optimization", 2nd ed.,
    Springer (2006), Ch. 6.
"""

from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING, final

import numpy as np
from numpy.typing import NDArray

from declivity.algorithms.bfgs.config import BFGSConfig
from declivity.algorithms.choices import AlgorithmChoice
from declivity.core.algorithm_factory import register_optimizer
from declivity.core.base_optimizer import BaseOptimizer, OptimizationResult
from declivity.utils.constraint_handlers import ConstraintHandler
from declivity.utils.gradient_strategies import (
    CentralFD,
    GradientStrategy,
    directional_derivative,
)
from declivity.utils.initial_geometry import InitialGeometry
from declivity.utils.line_search import LineSearchStrategy, MoreThuenteLineSearch
from declivity.utils.stopping_conditions import StoppingCondition

if TYPE_CHECKING:
    from declivity.logging.bfgs_logger import BFGSLogData


@final
@register_optimizer(AlgorithmChoice.BFGS, BFGSConfig)
class BFGSOptimizer(BaseOptimizer["BFGSLogData", BFGSConfig]):
    """BFGS quasi-Newton optimizer for smooth minimization.

    Maintains a dense inverse-Hessian approximation ``Hk`` updated by the
    Sherman–Morrison BFGS formula.  Single-point method: inherits
    :class:`BaseOptimizer` directly, like L-BFGS-B and Powell.
    """

    def __init__(
        self,
        func: Callable[[NDArray[np.float64]], float],
        initial_point: NDArray[np.float64],
        config: BFGSConfig | None = None,
        constraint_handler: ConstraintHandler | None = None,
        stopping_condition: StoppingCondition | None = None,
        lower_bounds: float | NDArray[np.float64] | list[float] = -np.inf,
        upper_bounds: float | NDArray[np.float64] | list[float] = np.inf,
        seed: int | np.random.Generator | None = None,
        gradient_fn: Callable[[NDArray[np.float64]], NDArray[np.float64]] | None = None,
        gradient_strategy: GradientStrategy | None = None,
        line_search: LineSearchStrategy | None = None,
        initial_geometry: "InitialGeometry | None" = None,
    ) -> None:
        if config is None:
            config = BFGSConfig(dimensions=len(initial_point))

        super().__init__(
            func=func,
            initial_point=initial_point,
            config=config,
            algorithm=AlgorithmChoice.BFGS,
            constraint_handler=constraint_handler,
            stopping_condition=stopping_condition,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            seed=seed,
        )

        self._gradient_fn = gradient_fn
        self._finite_diff_epsilon = config._fd_eps_actual
        self._gradient_strategy = gradient_strategy or CentralFD()
        self._line_search = line_search or MoreThuenteLineSearch()
        self._machine_epsilon = np.finfo(float).eps

        # A supplied InitialGeometry stores the curvature B_0; BFGS tracks the
        # inverse Hessian H_0 = B_0^{-1}, so seed it via ``solve`` (this also
        # gives the CMA-ES covariance shape back for a from_covariance handoff,
        # symmetric with the L-BFGS-B B_0 handoff).  It and
        # ``config.initial_inverse_hessian`` are mutually exclusive seams —
        # same contract as L-BFGS-B, Powell, and Nelder-Mead.
        if initial_geometry is not None and config.initial_inverse_hessian is not None:
            raise ValueError(
                "Pass either config.initial_inverse_hessian or initial_geometry, "
                "not both."
            )
        self._initial_geometry = initial_geometry
        self._cached_gradient: NDArray[np.float64] | None = None

    # Gradient computation (mirrors L-BFGS-B)

    def _evaluate_function_and_gradient(
        self, x: NDArray[np.float64]
    ) -> tuple[float, NDArray[np.float64]]:
        function_value = self.evaluate(x)
        gradient = self._compute_gradient(x, function_value)
        return function_value, gradient

    def _compute_gradient(
        self, x: NDArray[np.float64], function_value_at_x: float | None = None
    ) -> NDArray[np.float64]:
        if self._gradient_fn is not None:
            return np.asarray(self._gradient_fn(x), dtype=float)
        return self._gradient_strategy.compute(
            f=self.evaluate,
            x=x,
            eps=self._finite_diff_epsilon,
            f_at_x=function_value_at_x,
            constraint_handler=self.constraint_handler,
        )

    def _compute_directional_derivative(
        self, x: NDArray[np.float64], direction: NDArray[np.float64], alpha: float
    ) -> tuple[float, float]:
        """Evaluate phi(alpha) = f(x + alpha*d) and phi'(alpha) = grad f . d.

        When an analytical gradient is available, the full gradient is cached
        for reuse after the line search completes.  Otherwise the derivative
        comes from the *injected* :class:`GradientStrategy` applied to the 1-D
        restriction along the ray, with probe feasibility delegated to the
        constraint handler — identical to L-BFGS-B, and a no-op under the
        default unbounded handler (where no probe is ever infeasible), so
        SciPy parity is unaffected.
        """
        x_trial = x + alpha * direction
        f_trial = self.evaluate(x_trial)

        if self._gradient_fn is not None:
            gradient_at_trial = self._gradient_fn(x_trial)
            self._cached_gradient = np.asarray(gradient_at_trial, dtype=float)
            return f_trial, float(np.dot(self._cached_gradient, direction))

        return f_trial, directional_derivative(
            self._gradient_strategy,
            f=self.evaluate,
            x=x_trial,
            direction=direction,
            eps=self._finite_diff_epsilon,
            f_at_x=f_trial,
            constraint_handler=self.constraint_handler,
        )

    # Convergence measure

    def _gradient_norm(
        self, x: NDArray[np.float64], gradient: NDArray[np.float64]
    ) -> float:
        """Norm of the handler's projected gradient under ``config.norm``.

        Equals ``vecnorm(gradient, ord=norm)`` when the handler is
        unconstrained (its projection clips nothing), so it reproduces SciPy's
        ``vecnorm(gfk, ord=norm)`` test in the unconstrained regime while
        remaining a valid KKT measure at active constraints.
        """
        pg = self.constraint_handler.projected_gradient(x, gradient)
        if len(pg) == 0:
            return 0.0
        return float(np.linalg.norm(pg, ord=self.config.norm))

    # Initial inverse Hessian

    def _build_initial_inverse_hessian(self) -> NDArray[np.float64]:
        """Construct the dense inverse-Hessian seed ``H_0`` (n x n, SPD)."""
        n = self.dimensions

        if self._initial_geometry is not None:
            # A supplied geometry stores the *curvature* B_0; BFGS tracks the
            # inverse Hessian, so seed H_0 = B_0^{-1}.
            inverse_hessian = self._initial_geometry.solve(np.eye(n))
            return 0.5 * (inverse_hessian + inverse_hessian.T)

        # ``initial_inverse_hessian`` (None / scalar / 1D / 2D) is H_0 directly.
        # Reuse InitialGeometry's shared dispatch and its positivity / shape /
        # SPD validation rather than re-rolling it here; ``scale_columns(I)``
        # materializes the dense matrix (diag(h) in diagonal mode, the
        # symmetrized SPD-checked matrix in dense mode).
        geometry = InitialGeometry(self.config.initial_inverse_hessian, n)
        return np.asarray(geometry.scale_columns(np.eye(n)), dtype=np.float64)

    # Main loop

    def optimize(self) -> OptimizationResult["BFGSLogData"]:
        self.evaluations = 0
        self._begin_run()
        config = self.config
        n = self.dimensions
        identity = np.eye(n)

        # Annotated explicitly: ``x`` is reassigned from values that flow
        # through the line-search closures, and those closures capture ``x`` as
        # a default argument — without an annotation the inference is circular.
        x: NDArray[np.float64] = self.constraint_handler.repair(
            self.initial_point.copy()
        )
        self._cached_gradient = None

        function_value, gradient = self._evaluate_function_and_gradient(x)
        best_fitness = function_value
        best_solution = x.copy()
        termination_message = None

        inverse_hessian = self._build_initial_inverse_hessian()

        grad_norm = self._gradient_norm(x, gradient)
        if grad_norm <= config.gtol:
            return OptimizationResult(
                best_solution=best_solution,
                best_fitness=best_fitness,
                evaluations=self.evaluations,
                message=(
                    f"Converged: gradient norm {grad_norm:.2e} <= {config.gtol:.2e}"
                ),
                diagnostic=self.get_logs(),
                algorithm=AlgorithmChoice.BFGS,
            )

        # SciPy seeds the previous function value so the first line-search step
        # guess ``alpha1 = min(1, 1.01*2*(f - f_prev)/(g.p))`` evaluates to ~1.
        previous_function_value = function_value + np.linalg.norm(gradient) / 2.0

        iteration = 0
        while not self.should_stop(iteration, best_fitness):
            iteration += 1

            direction = np.asarray(-(inverse_hessian @ gradient), dtype=np.float64)

            # Constraint projection: a component pushing outward at an
            # already active constraint cannot move, and leaving it in makes
            # the largest feasible step exactly zero — which would end the run
            # even though every other coordinate is still free.  The handler
            # decides which components those are; with an unconstrained
            # handler (the SciPy-parity regime) nothing is ever active and the
            # direction passes through untouched.
            direction = self.constraint_handler.project_direction(x, direction)
            directional_derivative = float(np.dot(gradient, direction))

            # Ascent-direction guard: if bound clamping (or a degenerate Hk)
            # made ``direction`` non-descent, fall back to projected steepest
            # descent with active-bound components zeroed (as in L-BFGS-B).
            if directional_derivative >= 0:
                direction = self.constraint_handler.project_direction(x, -gradient)
                directional_derivative = float(np.dot(gradient, direction))
                if directional_derivative >= 0:
                    termination_message = "Cannot find descent direction"
                    break

            max_feas_step = self.constraint_handler.max_feasible_step(x, direction)
            if max_feas_step <= 0:
                termination_message = "Maximum feasible step is zero"
                break

            # SciPy's scalar_search_wolfe1 initial-step heuristic.
            if directional_derivative != 0:
                initial_step = min(
                    1.0,
                    1.01
                    * 2.0
                    * (function_value - previous_function_value)
                    / directional_derivative,
                )
                if initial_step < 0:
                    initial_step = 1.0
            else:
                initial_step = 1.0
            initial_step = min(initial_step, max_feas_step)

            self._cached_gradient = None

            # phi(alpha), phi'(alpha) along the current point/direction — bound
            # eagerly (x/direction do not change during the line search).
            phi_and_dphi = partial(self._compute_directional_derivative, x, direction)

            def phi_only(alpha: float, _x=x, _d=direction) -> float:
                return self.evaluate(_x + alpha * _d)

            line_search_result = self._line_search.search(
                phi_dphi=phi_and_dphi,
                stp0=initial_step,
                phi0=function_value,
                dphi0=directional_derivative,
                stpmax=max_feas_step,
                ftol=config.c1,
                gtol=config.c2,
                xtol=config.xtol_ls,
                maxiter=config.max_ls_iter,
                # Value-only fast path: a derivative-free search (Armijo) takes
                # one evaluation per trial step instead of three.  L-BFGS-B
                # supplies this too; omitting it here made the injected
                # line_search= seam cost 3x more in BFGS than in L-BFGS-B for
                # the same strategy, which is exactly the kind of asymmetry a
                # shared-budget comparison must not have.
                phi=phi_only,
            )

            accepted_step = line_search_result.step
            if accepted_step <= 0 or not line_search_result.converged:
                # Mirrors SciPy's warnflag=2: the line search could not find a
                # point satisfying the Wolfe conditions (precision loss).
                termination_message = (
                    "Line search failed: desired error not necessarily achieved "
                    "due to precision loss"
                )
                break

            step_vector = accepted_step * direction
            x_line_search = x + step_vector
            x_new = self.constraint_handler.repair(x_line_search)
            step_vector = x_new - x

            if not np.array_equal(x_new, x_line_search):
                # The line search's f / gradient describe the unrepaired trial
                # point; a handler that moved it invalidates both.
                function_value_new, gradient_new = self._evaluate_function_and_gradient(
                    x_new
                )
            elif self._cached_gradient is not None:
                gradient_new = self._cached_gradient
                function_value_new = line_search_result.f_new
            else:
                function_value_new, gradient_new = self._evaluate_function_and_gradient(
                    x_new
                )

            gradient_difference = gradient_new - gradient
            curvature = float(np.dot(gradient_difference, step_vector))

            if function_value_new < best_fitness:
                best_fitness = function_value_new
                best_solution = x_new.copy()

            grad_norm = self._gradient_norm(x_new, gradient_new)

            if config.diag_hessian_condition:
                hessian_condition = float(np.linalg.cond(inverse_hessian))
            else:
                hessian_condition = 0.0

            self.logger.log_iteration(
                iteration=iteration,
                evaluations=self.evaluations,
                best_fitness=best_fitness,
                function_value=function_value_new,
                gradient_norm=grad_norm,
                step_length=accepted_step,
                curvature=curvature,
                hessian_condition=hessian_condition,
                best_solution=best_solution,
            )

            if grad_norm <= config.gtol:
                termination_message = (
                    f"Converged: gradient norm {grad_norm:.2e} <= {config.gtol:.2e}"
                )
                break

            # SciPy's xrtol step-size test.
            if accepted_step * float(np.linalg.norm(direction)) <= config.xrtol * (
                config.xrtol + float(np.linalg.norm(x_new))
            ):
                termination_message = (
                    f"Converged: step size below xrtol = {config.xrtol:.2e}"
                )
                break

            if not np.isfinite(function_value_new):
                termination_message = (
                    "Desired error not necessarily achieved due to precision loss"
                )
                break

            # BFGS inverse-Hessian update (Sherman–Morrison), SciPy port.
            # rho = 1 / (yk . sk), guarded when the curvature vanishes.
            rho = 1000.0 if curvature == 0.0 else 1.0 / curvature

            left = identity - rho * np.outer(step_vector, gradient_difference)
            right = identity - rho * np.outer(gradient_difference, step_vector)
            inverse_hessian = left @ inverse_hessian @ right + rho * np.outer(
                step_vector, step_vector
            )

            previous_function_value = function_value
            x = x_new
            function_value = function_value_new
            gradient = gradient_new

        if termination_message is None:
            termination_message = self.stop_message

        return OptimizationResult(
            best_solution=best_solution,
            best_fitness=best_fitness,
            evaluations=self.evaluations,
            message=termination_message,
            diagnostic=self.get_logs(),
            algorithm=AlgorithmChoice.BFGS,
        )
