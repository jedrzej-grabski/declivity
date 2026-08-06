"""
Pure Python implementation of the L-BFGS-B algorithm.

A limited-memory quasi-Newton method for bound-constrained optimization.
Reimplemented from the Fortran L-BFGS-B v3.0 (Byrd, Lu, Nocedal, Zhu 1995)
with the subspace minimization correction from Morales and Nocedal (2011).

Extended to accept a diagonal or dense initial Hessian approximation B_0.
The compact representation becomes B = theta * B_0 - W * M * W', with B_0
threaded consistently through the Cauchy point, subspace minimization,
W matrix construction, and middle matrix operations.

References:
    R.H. Byrd, P. Lu, J. Nocedal, C. Zhu, "A Limited Memory Algorithm for
    Bound Constrained Optimization", SIAM J. Scientific Computing 16 (1995).

    J.L. Morales, J. Nocedal, "Remark on Algorithm 778: L-BFGS-B",
    ACM Trans. Math. Software 38 (2011).
"""

from typing import TYPE_CHECKING, Callable, Union, final

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import cho_factor, cho_solve

from declivity.algorithms.choices import AlgorithmChoice
from declivity.algorithms.lbfgsb.config import LBFGSBConfig
from declivity.core.algorithm_factory import register_optimizer
from declivity.core.base_optimizer import BaseOptimizer, OptimizationResult
from declivity.utils.constraint_handlers import ConstraintHandler
from declivity.utils.gradient_strategies import CentralFD, ForwardFD, GradientStrategy
from declivity.utils.initial_geometry import InitialHessian, InitialHessianMode
from declivity.utils.line_search import (
    LineSearchStrategy,
    MoreThuenteLineSearch,
    max_feasible_step,
)
from declivity.utils.optimality import projected_gradient_inf_norm
from declivity.utils.stopping_conditions import StoppingCondition

if TYPE_CHECKING:
    from declivity.logging.lbfgsb_logger import LBFGSBLogData


@final
@register_optimizer(AlgorithmChoice.LBFGSB, LBFGSBConfig)
class LBFGSBOptimizer(BaseOptimizer["LBFGSBLogData", LBFGSBConfig]):
    """L-BFGS-B optimizer for bound-constrained minimization.

    Uses a compact L-BFGS Hessian approximation B = theta * B_0 - W * M * W'
    with generalized Cauchy point computation for active set identification
    and subspace minimization for second-order refinement.

    B_0 can be the identity (default), a diagonal, or a full matrix.
    """

    def __init__(
        self,
        func: Callable[[NDArray[np.float64]], float],
        initial_point: NDArray[np.float64],
        config: LBFGSBConfig | None = None,
        constraint_handler: ConstraintHandler | None = None,
        stopping_condition: StoppingCondition | None = None,
        lower_bounds: Union[float, NDArray[np.float64], list[float]] = -100.0,
        upper_bounds: Union[float, NDArray[np.float64], list[float]] = 100.0,
        seed: int | np.random.Generator | None = None,
        gradient_fn: Callable[[NDArray[np.float64]], NDArray[np.float64]] | None = None,
        gradient_strategy: GradientStrategy | None = None,
        line_search: LineSearchStrategy | None = None,
        initial_geometry: "InitialHessian | None" = None,
    ) -> None:
        if config is None:
            config = LBFGSBConfig(dimensions=len(initial_point))

        super().__init__(
            func=func,
            initial_point=initial_point,
            config=config,
            algorithm=AlgorithmChoice.LBFGSB,
            constraint_handler=constraint_handler,
            stopping_condition=stopping_condition,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            seed=seed,
        )

        self._gradient_fn = gradient_fn
        self._gradient_strategy = gradient_strategy or CentralFD()
        self._line_search = line_search or MoreThuenteLineSearch()
        self._memory_size = config.m
        self._machine_epsilon = np.finfo(float).eps

        # Resolve the finite-difference step: explicit config values pass
        # through untouched; the 0 = auto sentinel picks the step that
        # balances truncation against rounding error for the strategy in
        # use — eps**(1/3) for central differences, sqrt(eps) for forward.
        if config.fd_eps > 0:
            self._finite_diff_epsilon = config.fd_eps
        elif isinstance(self._gradient_strategy, ForwardFD):
            self._finite_diff_epsilon = float(self._machine_epsilon**0.5)
        else:
            self._finite_diff_epsilon = float(self._machine_epsilon ** (1.0 / 3.0))

        # A supplied InitialGeometry (e.g. a CMA-ES-derived B_0 from a
        # handoff) and ``config.initial_hessian`` are mutually exclusive
        # seams; otherwise build one from the config's raw curvature as
        # before.
        if initial_geometry is not None:
            if config.initial_hessian is not None:
                raise ValueError(
                    "Pass either config.initial_hessian or initial_geometry, not both."
                )
            self._initial_hessian = initial_geometry
        else:
            self._initial_hessian = InitialHessian(
                config.initial_hessian, len(initial_point)
            )

        self._step_vectors: list[NDArray[np.float64]] = []
        self._gradient_diff_vectors: list[NDArray[np.float64]] = []
        self._theta: float = 1.0
        self._num_corrections: int = 0

        self._steps_dot_grad_diffs: NDArray[np.float64] = np.empty((0, 0))
        self._steps_dot_steps: NDArray[np.float64] = np.empty((0, 0))
        self._steps_B0_steps: NDArray[np.float64] = np.empty((0, 0))
        self._cholesky_factor_of_T: tuple | None = None

    # Gradient computation

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
        # The strategy routes its evaluations through ``self.evaluate`` so
        # the evaluation budget is incremented exactly as the inline
        # FD loops used to do.
        return self._gradient_strategy.compute(
            f=self.evaluate,
            x=x,
            eps=self._finite_diff_epsilon,
            f_at_x=function_value_at_x,
            lower_bounds=self.lower_bounds,
            upper_bounds=self.upper_bounds,
        )

    def _compute_directional_derivative(
        self, x: NDArray[np.float64], direction: NDArray[np.float64], alpha: float
    ) -> tuple[float, float]:
        """Evaluate phi(alpha) = f(x + alpha*d) and phi'(alpha) = grad f . d.

        When an analytical gradient is available, the full gradient is cached
        for reuse after the line search completes.
        """
        x_trial = x + alpha * direction
        f_trial = self.evaluate(x_trial)

        if self._gradient_fn is not None:
            gradient_at_trial = self._gradient_fn(x_trial)
            self._cached_gradient = np.asarray(gradient_at_trial, dtype=float)
            return f_trial, float(np.dot(self._cached_gradient, direction))
        else:
            epsilon = self._finite_diff_epsilon
            forward_point = x_trial + epsilon * direction
            backward_point = x_trial - epsilon * direction
            forward_feasible = bool(
                np.all(forward_point >= self.lower_bounds)
                and np.all(forward_point <= self.upper_bounds)
            )
            backward_feasible = bool(
                np.all(backward_point >= self.lower_bounds)
                and np.all(backward_point <= self.upper_bounds)
            )
            if forward_feasible == backward_feasible:
                # Both probes feasible: central difference.  (Also the
                # degenerate fallback when neither probe fits the box.)
                f_forward = self.evaluate(forward_point)
                f_backward = self.evaluate(backward_point)
                return f_trial, (f_forward - f_backward) / (2.0 * epsilon)
            if forward_feasible:
                # Backward probe would exit the box (x_trial sits on a
                # bound): one-sided difference toward the feasible side,
                # anchored on the already-evaluated f_trial.
                f_forward = self.evaluate(forward_point)
                return f_trial, (f_forward - f_trial) / epsilon
            f_backward = self.evaluate(backward_point)
            return f_trial, (f_trial - f_backward) / epsilon

    # Projected gradient

    def _compute_projected_gradient_inf_norm(
        self, x: NDArray[np.float64], gradient: NDArray[np.float64]
    ) -> float:
        """Infinity norm of the projected gradient (KKT optimality measure)."""
        return projected_gradient_inf_norm(
            x, gradient, self.lower_bounds, self.upper_bounds
        )

    # L-BFGS compact representation

    def _update_correction_pairs(
        self,
        step_vector: NDArray[np.float64],
        gradient_difference: NDArray[np.float64],
    ) -> bool:
        """Store a new (s, y) correction pair and rebuild cached matrices.

        The pair is accepted only if s'y > eps * y'y (curvature condition).
        """
        step_dot_grad_diff = float(np.dot(step_vector, gradient_difference))
        grad_diff_dot_grad_diff = float(
            np.dot(gradient_difference, gradient_difference)
        )

        if step_dot_grad_diff <= self._machine_epsilon * grad_diff_dot_grad_diff:
            return False
        if not (
            np.all(np.isfinite(step_vector))
            and np.all(np.isfinite(gradient_difference))
        ):
            return False

        self._step_vectors.append(step_vector.copy())
        self._gradient_diff_vectors.append(gradient_difference.copy())
        if len(self._step_vectors) > self._memory_size:
            self._step_vectors.pop(0)
            self._gradient_diff_vectors.pop(0)

        self._num_corrections = len(self._step_vectors)
        self._theta = grad_diff_dot_grad_diff / step_dot_grad_diff

        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            step_matrix = np.column_stack(self._step_vectors)
            grad_diff_matrix = np.column_stack(self._gradient_diff_vectors)
            self._steps_dot_grad_diffs = step_matrix.T @ grad_diff_matrix
            self._steps_dot_steps = step_matrix.T @ step_matrix
            # S' B_0 S — used in the M^{-1} and T matrices
            self._steps_B0_steps = self._initial_hessian.quadratic_form(step_matrix)

        if not (
            np.all(np.isfinite(self._steps_dot_grad_diffs))
            and np.all(np.isfinite(self._steps_dot_steps))
            and np.all(np.isfinite(self._steps_B0_steps))
        ):
            self._reset_correction_memory()
            return False

        self._factorize_middle_matrix()
        return True

    def _factorize_middle_matrix(self) -> None:
        """Form and Cholesky-factorize T = theta * S' B_0 S + L D^{-1} L'."""
        num_corrections = self._num_corrections
        if num_corrections == 0:
            self._cholesky_factor_of_T = None
            return

        curvature_diagonal = np.diag(self._steps_dot_grad_diffs).copy()
        strict_lower_triangle = np.tril(self._steps_dot_grad_diffs, -1)
        safe_curvature_diagonal = np.maximum(curvature_diagonal, self._machine_epsilon)

        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            T = (
                self._theta * self._steps_B0_steps
                + strict_lower_triangle
                @ np.diag(1.0 / safe_curvature_diagonal)
                @ strict_lower_triangle.T
            )

        T = 0.5 * (T + T.T)

        if not np.all(np.isfinite(T)):
            self._reset_correction_memory()
            return

        try:
            self._cholesky_factor_of_T = cho_factor(T)
        except (np.linalg.LinAlgError, ValueError):
            self._reset_correction_memory()

    def _middle_matrix_multiply(
        self, vector: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Compute p = M * v where M is the middle matrix.

        M^{-1} = [-D,              L'           ]
                 [ L,   theta * S' B_0 S        ]

        Solved via block elimination with the Cholesky factor of T.
        """
        num_corrections = self._num_corrections
        if num_corrections == 0 or self._cholesky_factor_of_T is None:
            return np.zeros_like(vector)

        gradient_diff_part = vector[:num_corrections]
        step_part = vector[num_corrections : 2 * num_corrections]

        curvature_diagonal = np.diag(self._steps_dot_grad_diffs).copy()
        strict_lower_triangle = np.tril(self._steps_dot_grad_diffs, -1)
        safe_curvature_diagonal = np.maximum(curvature_diagonal, self._machine_epsilon)

        right_hand_side = step_part + strict_lower_triangle @ (
            gradient_diff_part / safe_curvature_diagonal
        )
        solution_step_part = cho_solve(self._cholesky_factor_of_T, right_hand_side)

        solution_gradient_diff_part = (
            strict_lower_triangle.T @ solution_step_part - gradient_diff_part
        ) / safe_curvature_diagonal

        result = np.zeros(2 * num_corrections)
        result[:num_corrections] = solution_gradient_diff_part
        result[num_corrections : 2 * num_corrections] = solution_step_part
        return result

    def _reset_correction_memory(self) -> None:
        self._step_vectors.clear()
        self._gradient_diff_vectors.clear()
        self._num_corrections = 0
        self._theta = 1.0
        self._cholesky_factor_of_T = None

    def _build_w_projection(self, vectors: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute W' * v where W = [Y | theta * B_0 * S].

        Returns a 2*col vector: [Y' v, theta * (B_0 S)' v].
        """
        num_corrections = self._num_corrections
        result = np.zeros(2 * num_corrections)
        if num_corrections == 0:
            return result

        step_matrix = np.column_stack(self._step_vectors)
        grad_diff_matrix = np.column_stack(self._gradient_diff_vectors)
        B0_times_step_matrix = self._initial_hessian.scale_columns(step_matrix)

        result[:num_corrections] = grad_diff_matrix.T @ vectors
        result[num_corrections:] = self._theta * (B0_times_step_matrix.T @ vectors)
        return result

    def _get_w_row(self, variable_index: int) -> NDArray[np.float64]:
        """Extract row i of W = [Y | theta * B_0 * S].

        For diagonal B_0, the row is [Y[i,:], theta * h_i * S[i,:]].
        For dense B_0, the row is [Y[i,:], theta * (B_0 S)[i,:]].
        """
        num_corrections = self._num_corrections
        row = np.zeros(2 * num_corrections)

        for k in range(num_corrections):
            row[k] = self._gradient_diff_vectors[k][variable_index]

        step_matrix = np.column_stack(self._step_vectors)
        B0_times_step_matrix = self._initial_hessian.scale_columns(step_matrix)
        for k in range(num_corrections):
            row[num_corrections + k] = (
                self._theta * B0_times_step_matrix[variable_index, k]
            )

        return row

    # Generalized Cauchy Point

    def _compute_cauchy_point(
        self, x: NDArray[np.float64], gradient: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int32]]:
        """Compute the Generalized Cauchy Point (GCP).

        The GCP is the first local minimizer of the quadratic model along
        the piecewise-linear projected steepest descent path. It determines
        which variables are active (at bounds) and which are free.

        Returns:
            cauchy_point, w_displacement (= W'(xcp - x)), variable_status.

        Variable status codes:
             3  permanently fixed (lower == upper)
             2  fixed at upper bound
             1  fixed at lower bound
             0  free, moved during search
            -3  free, zero gradient
        """
        num_vars = self.dimensions
        num_corrections = self._num_corrections
        lower = self.lower_bounds
        upper = self.upper_bounds
        B0 = self._initial_hessian

        cauchy_direction = np.zeros(num_vars)
        variable_status = np.zeros(num_vars, dtype=np.int32)

        # Classify variables and compute breakpoints
        breakpoint_list = []

        for i in range(num_vars):
            if lower[i] == upper[i]:
                variable_status[i] = 3
                continue

            negative_gradient_i = -gradient[i]

            if x[i] <= lower[i] and negative_gradient_i <= 0:
                variable_status[i] = 1
                continue
            if x[i] >= upper[i] and negative_gradient_i >= 0:
                variable_status[i] = 2
                continue

            if negative_gradient_i == 0:
                variable_status[i] = -3
                continue

            variable_status[i] = 0
            cauchy_direction[i] = negative_gradient_i

            if negative_gradient_i < 0:
                time_to_bound = (x[i] - lower[i]) / (-negative_gradient_i)
            else:
                time_to_bound = (upper[i] - x[i]) / negative_gradient_i

            if time_to_bound > 0:
                breakpoint_list.append((time_to_bound, i))
            else:
                cauchy_direction[i] = 0.0
                variable_status[i] = 1 if negative_gradient_i < 0 else 2

        breakpoint_list.sort()
        num_breakpoints = len(breakpoint_list)

        # Project the Cauchy direction onto the W basis: p = W' d
        # W = [Y | theta * B_0 * S]
        projected_direction = self._build_w_projection(cauchy_direction)

        # Effective base Hessian: theta * B_0 (or just B_0 when theta = 1 initially).
        # When persist is False and corrections exist, fall back to theta * I.
        use_initial_hessian = (
            self._num_corrections == 0 or self.config.persist_initial_hessian
        )

        # Initialize quadratic model derivatives
        first_derivative = float(np.dot(gradient, cauchy_direction))

        if use_initial_hessian:
            base_curvature = self._theta * B0.weighted_dot(cauchy_direction)
        else:
            base_curvature = self._theta * float(
                np.dot(cauchy_direction, cauchy_direction)
            )

        if num_corrections > 0:
            middle_times_projected = self._middle_matrix_multiply(projected_direction)
            second_derivative = base_curvature - float(
                np.dot(projected_direction, middle_times_projected)
            )
        else:
            second_derivative = base_curvature
            middle_times_projected = np.zeros(2 * num_corrections)

        if second_derivative > self._machine_epsilon:
            unconstrained_minimizer_step = -first_derivative / second_derivative
        else:
            unconstrained_minimizer_step = 0.0

        # Walk along breakpoints
        total_time = 0.0
        w_displacement = np.zeros(2 * num_corrections)

        for breakpoint_index in range(num_breakpoints):
            breakpoint_time, var_index = breakpoint_list[breakpoint_index]
            segment_length = breakpoint_time - total_time

            if unconstrained_minimizer_step < segment_length:
                total_time += unconstrained_minimizer_step
                break

            total_time = breakpoint_time

            direction_component = cauchy_direction[var_index]
            if direction_component > 0:
                distance_to_bound = upper[var_index] - x[var_index]
                variable_status[var_index] = 2
            else:
                distance_to_bound = lower[var_index] - x[var_index]
                variable_status[var_index] = 1

            # Breakpoint derivative updates use the diagonal of B_0 (or theta
            # for the non-persist case). For dense B_0, this is an approximation
            # at the per-variable level; the off-diagonal contributions are
            # captured by the L-BFGS correction terms.
            if use_initial_hessian:
                h_i = self._theta * B0.diagonal_element(var_index)
            else:
                h_i = self._theta

            first_derivative = (
                first_derivative
                + segment_length * second_derivative
                + direction_component * direction_component
                - h_i * direction_component * distance_to_bound
            )
            second_derivative = (
                second_derivative - h_i * direction_component * direction_component
            )

            if num_corrections > 0:
                w_displacement += segment_length * projected_direction

                w_row = self._get_w_row(var_index)
                middle_times_w_row = self._middle_matrix_multiply(w_row)

                first_derivative += direction_component * float(
                    np.dot(w_displacement, middle_times_w_row)
                )
                second_derivative += 2.0 * direction_component * float(
                    np.dot(w_row, middle_times_projected)
                ) - direction_component * direction_component * float(
                    np.dot(w_row, middle_times_w_row)
                )

                projected_direction -= direction_component * w_row
                middle_times_projected = self._middle_matrix_multiply(
                    projected_direction
                )

            cauchy_direction[var_index] = 0.0

            if second_derivative > self._machine_epsilon:
                unconstrained_minimizer_step = -first_derivative / second_derivative
            else:
                unconstrained_minimizer_step = 0.0
        else:
            total_time += max(unconstrained_minimizer_step, 0.0)

        # Reconstruct the Cauchy point
        cauchy_point = x.copy()
        for i in range(num_vars):
            if variable_status[i] == 0:
                cauchy_point[i] = x[i] + total_time * (-gradient[i])
            elif variable_status[i] == 1:
                cauchy_point[i] = lower[i]
            elif variable_status[i] == 2:
                cauchy_point[i] = upper[i]

        cauchy_point = np.clip(cauchy_point, lower, upper)

        # Recompute w_displacement = W'(cauchy_point - x) at the final GCP
        w_displacement = self._build_w_projection(cauchy_point - x)

        return cauchy_point, w_displacement, variable_status

    def _identify_free_variables(
        self, variable_status: NDArray[np.int32]
    ) -> tuple[list[int], int]:
        free_variable_indices = [
            i for i in range(self.dimensions) if variable_status[i] <= 0
        ]
        return free_variable_indices, len(free_variable_indices)

    # Subspace minimization

    def _perform_subspace_minimization(
        self,
        x: NDArray[np.float64],
        cauchy_point: NDArray[np.float64],
        gradient: NDArray[np.float64],
        w_displacement: NDArray[np.float64],
        free_variable_indices: list[int],
        num_free: int,
    ) -> NDArray[np.float64]:
        """Minimize the quadratic model in the subspace of free variables.

        Uses the Woodbury identity with B_0 threaded through consistently:
            (Z'BZ)^{-1} = (1/theta) B_0_Z^{-1}
                + (1/theta^2) B_0_Z^{-1} A K^{-1} A' B_0_Z^{-1}
        where B_0_Z is B_0 restricted to the free variables, and
        K = M^{-1} - (1/theta) A' B_0_Z^{-1} A.
        """
        num_corrections = self._num_corrections
        B0 = self._initial_hessian

        if num_free == 0 or num_corrections == 0:
            return cauchy_point.copy()

        # Reduced gradient: r = -Z'(g + B(xcp - x))
        # B(xcp-x) = theta * B_0 * (xcp-x) - W * M * W'(xcp-x)
        middle_times_displacement = self._middle_matrix_multiply(w_displacement)

        cauchy_displacement = cauchy_point - x
        B0_times_displacement = B0.multiply(cauchy_displacement)

        reduced_gradient = np.zeros(num_free)
        for j in range(num_free):
            idx = free_variable_indices[j]
            # -(g + theta * B_0 * (xcp - x))
            reduced_gradient[j] = -(
                gradient[idx] + self._theta * B0_times_displacement[idx]
            )
            # + W[idx,:] * M * W'(xcp - x)  (the correction contribution)
            w_row_j = self._get_w_row(idx)
            reduced_gradient[j] += float(np.dot(w_row_j, middle_times_displacement))

        # A = Z'W  (the L-BFGS basis restricted to free variables)
        restricted_basis = np.zeros((num_free, 2 * num_corrections))
        for j in range(num_free):
            restricted_basis[j, :] = self._get_w_row(free_variable_indices[j])

        # B_0^{-1} restricted to free variables, applied to the reduced gradient
        # For diagonal B_0: element-wise division
        # For dense B_0: extract the free-variable subblock and solve
        if B0.mode == InitialHessianMode.DENSE:
            # Extract the free-variable subblock of B_0
            assert B0._matrix is not None
            free_idx = np.array(free_variable_indices)
            B0_free = np.zeros((num_free, num_free))
            for j1 in range(num_free):
                for j2 in range(num_free):
                    B0_free[j1, j2] = B0._matrix[free_idx[j1], free_idx[j2]]
            B0_free_cholesky = cho_factor(0.5 * (B0_free + B0_free.T))
            B0_inv_reduced_gradient = cho_solve(B0_free_cholesky, reduced_gradient)

            # B_0_Z^{-1} * A  for the Woodbury K matrix
            B0_inv_A = cho_solve(B0_free_cholesky, restricted_basis)
        else:
            # Diagonal: B_0^{-1} is element-wise inverse
            B0_inv_diag_free = np.array(
                [
                    1.0 / B0.diagonal_element(free_variable_indices[j])
                    for j in range(num_free)
                ]
            )
            B0_inv_reduced_gradient = B0_inv_diag_free * reduced_gradient
            B0_inv_A = B0_inv_diag_free[:, np.newaxis] * restricted_basis

        projected_reduced_gradient = restricted_basis.T @ B0_inv_reduced_gradient

        # K = M^{-1} - (1/theta) * A' * B_0_Z^{-1} * A
        curvature_diagonal = np.diag(self._steps_dot_grad_diffs).copy()
        strict_lower_triangle = np.tril(self._steps_dot_grad_diffs, -1)
        middle_matrix_inverse = np.zeros((2 * num_corrections, 2 * num_corrections))
        middle_matrix_inverse[:num_corrections, :num_corrections] = -np.diag(
            curvature_diagonal
        )
        middle_matrix_inverse[:num_corrections, num_corrections:] = (
            strict_lower_triangle.T
        )
        middle_matrix_inverse[num_corrections:, :num_corrections] = (
            strict_lower_triangle
        )
        middle_matrix_inverse[num_corrections:, num_corrections:] = (
            self._theta * self._steps_B0_steps
        )

        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            woodbury_matrix = middle_matrix_inverse - (1.0 / self._theta) * (
                restricted_basis.T @ B0_inv_A
            )

        if not np.all(np.isfinite(woodbury_matrix)):
            return cauchy_point.copy()

        try:
            woodbury_solution = np.linalg.solve(
                woodbury_matrix, projected_reduced_gradient
            )
        except (np.linalg.LinAlgError, ValueError):
            return cauchy_point.copy()

        if not np.all(np.isfinite(woodbury_solution)):
            return cauchy_point.copy()

        # Newton direction: d = (1/theta)*B_0^{-1}*r + (1/theta^2)*B_0^{-1}*A*K^{-1}*A'*B_0^{-1}*r
        reduced_newton_direction = (1.0 / self._theta) * B0_inv_reduced_gradient + (
            1.0 / self._theta**2
        ) * (B0_inv_A @ woodbury_solution)

        # Lift back to full space
        candidate = cauchy_point.copy()
        for j in range(num_free):
            candidate[free_variable_indices[j]] = (
                cauchy_point[free_variable_indices[j]] + reduced_newton_direction[j]
            )

        # Morales-Nocedal 2011 safeguard
        projected_candidate = np.clip(candidate, self.lower_bounds, self.upper_bounds)
        directional_derivative = float(np.dot(gradient, projected_candidate - x))

        if directional_derivative <= 0:
            return projected_candidate

        # Backtracking fallback
        max_feasible_alpha = 1.0
        for j in range(num_free):
            idx = free_variable_indices[j]
            if reduced_newton_direction[j] > self._machine_epsilon:
                max_feasible_alpha = min(
                    max_feasible_alpha,
                    (self.upper_bounds[idx] - cauchy_point[idx])
                    / reduced_newton_direction[j],
                )
            elif reduced_newton_direction[j] < -self._machine_epsilon:
                max_feasible_alpha = min(
                    max_feasible_alpha,
                    (self.lower_bounds[idx] - cauchy_point[idx])
                    / reduced_newton_direction[j],
                )

        max_feasible_alpha = max(0.0, min(1.0, max_feasible_alpha))
        fallback = cauchy_point.copy()
        for j in range(num_free):
            fallback[free_variable_indices[j]] = (
                cauchy_point[free_variable_indices[j]]
                + max_feasible_alpha * reduced_newton_direction[j]
            )
        return np.clip(fallback, self.lower_bounds, self.upper_bounds)

    # Main loop

    def optimize(self) -> OptimizationResult["LBFGSBLogData"]:
        self.evaluations = 0
        self._begin_run()
        num_vars = self.dimensions
        config = self.config

        self._reset_correction_memory()
        self._cached_gradient: NDArray[np.float64] | None = None

        x = self.constraint_handler.repair(self.initial_point.copy())

        function_value, gradient = self._evaluate_function_and_gradient(x)
        best_fitness = function_value
        best_solution = x.copy()
        termination_message = None

        projected_gradient_norm = self._compute_projected_gradient_inf_norm(x, gradient)
        if projected_gradient_norm <= config.pgtol:
            termination_message = (
                f"Converged: projected gradient norm {projected_gradient_norm:.2e} "
                f"<= {config.pgtol:.2e}"
            )
            return OptimizationResult(
                best_solution=best_solution,
                best_fitness=best_fitness,
                evaluations=self.evaluations,
                message=termination_message,
                diagnostic=self.get_logs(),
                algorithm=AlgorithmChoice.LBFGSB,
            )

        iteration = 0
        consecutive_resets = 0
        max_consecutive_resets = 20

        while not self.should_stop(iteration, best_fitness):
            iteration += 1

            cauchy_point, w_displacement, variable_status = self._compute_cauchy_point(
                x, gradient
            )

            free_variable_indices, num_free = self._identify_free_variables(
                variable_status
            )

            if num_free > 0 and self._num_corrections > 0:
                search_target = self._perform_subspace_minimization(
                    x,
                    cauchy_point,
                    gradient,
                    w_displacement,
                    free_variable_indices,
                    num_free,
                )
            else:
                search_target = cauchy_point

            direction = search_target - x
            direction_norm = float(np.linalg.norm(direction))

            if direction_norm < self._machine_epsilon:
                projected_gradient_norm = self._compute_projected_gradient_inf_norm(
                    x, gradient
                )
                if projected_gradient_norm <= config.pgtol:
                    termination_message = (
                        f"Converged: projected gradient norm "
                        f"{projected_gradient_norm:.2e}"
                    )
                    break
                consecutive_resets += 1
                if consecutive_resets >= max_consecutive_resets:
                    termination_message = "Stalled: repeated memory resets"
                    break
                self._reset_correction_memory()
                continue

            directional_derivative = float(np.dot(gradient, direction))
            if directional_derivative >= 0:
                direction = -gradient.copy()
                for i in range(num_vars):
                    if x[i] <= self.lower_bounds[i] and direction[i] < 0:
                        direction[i] = 0
                    if x[i] >= self.upper_bounds[i] and direction[i] > 0:
                        direction[i] = 0
                directional_derivative = float(np.dot(gradient, direction))
                direction_norm = float(np.linalg.norm(direction))
                if (
                    directional_derivative >= 0
                    or direction_norm < self._machine_epsilon
                ):
                    termination_message = "Cannot find descent direction"
                    break

            # L-BFGS-B convention (Byrd-Lu-Nocedal-Zhu 1995): always try the
            # full Newton step alpha=1 on the first iteration, regardless of
            # box geometry.
            max_feas_step = (
                1.0
                if iteration == 1
                else max_feasible_step(
                    x, direction, self.lower_bounds, self.upper_bounds
                )
            )
            if max_feas_step <= 0:
                termination_message = "Maximum feasible step is zero"
                break

            # Initial step guess matches the canonical L-BFGS-B convention
            # (Byrd–Lu–Nocedal–Zhu 1995, scipy/Fortran v3.0): try the full
            # Newton step ``alpha = 1`` on every iteration, capped by the
            # largest feasible step ``alpha_max`` that keeps ``x + alpha d``
            # inside the box.  An earlier ``min(1/||d||, alpha_max)`` heuristic
            # on the first iteration shrank the very first step to a tiny
            # ``alpha`` whenever ``||d||`` was large at ``x_0``, which the
            # line search would then accept (Wolfe conditions are trivially
            # satisfied at small alpha), wasting the first iteration.
            initial_step = min(1.0, max_feas_step)

            self._cached_gradient = None

            def phi_and_dphi(alpha: float, _x=x, _d=direction) -> tuple[float, float]:
                return self._compute_directional_derivative(_x, _d, alpha)

            def phi_only(alpha: float, _x=x, _d=direction) -> float:
                return self.evaluate(_x + alpha * _d)

            line_search_result = self._line_search.search(
                phi_dphi=phi_and_dphi,
                stp0=initial_step,
                phi0=function_value,
                dphi0=directional_derivative,
                stpmax=max_feas_step,
                ftol=config.ftol,
                gtol=config.gtol_ls,
                xtol=config.xtol_ls,
                maxiter=config.max_ls_iter,
                phi=phi_only,
            )

            accepted_step = line_search_result.step

            if (
                not line_search_result.converged
                and line_search_result.step > 0
                and line_search_result.f_new < best_fitness
            ):
                # Even a failed search evaluated real trial points; keep a
                # strictly better feasible one for reporting without
                # accepting the step into the algorithm state.
                best_fitness = line_search_result.f_new
                best_solution = (x + line_search_result.step * direction).copy()

            if accepted_step <= 0 or (
                not line_search_result.converged and self._num_corrections == 0
            ):
                termination_message = "Line search failed"
                break

            if not line_search_result.converged and self._num_corrections > 0:
                consecutive_resets += 1
                if consecutive_resets >= max_consecutive_resets:
                    termination_message = "Stalled: repeated line search failures"
                    break
                self._reset_correction_memory()
                continue

            consecutive_resets = 0
            step_vector = accepted_step * direction
            x_line_search = x + step_vector
            x_new = self.constraint_handler.repair(x_line_search)
            repair_moved_x = not np.array_equal(x_new, x_line_search)
            step_vector = x_new - x

            if repair_moved_x:
                # The line search's f/gradient describe x_line_search, not
                # the repaired point — re-evaluate both.
                function_value_new, gradient_new = self._evaluate_function_and_gradient(
                    x_new
                )
            elif self._cached_gradient is not None:
                gradient_new = self._cached_gradient
                function_value_new = line_search_result.f_new
            else:
                # The line search already evaluated f at exactly x_new; only
                # the gradient is missing.
                function_value_new = line_search_result.f_new
                gradient_new = self._compute_gradient(x_new, function_value_new)

            gradient_difference = gradient_new - gradient

            if function_value_new < best_fitness:
                best_fitness = function_value_new
                best_solution = x_new.copy()

            projected_gradient_norm = self._compute_projected_gradient_inf_norm(
                x_new, gradient_new
            )

            self.logger.log_iteration(
                iteration=iteration,
                evaluations=self.evaluations,
                best_fitness=best_fitness,
                function_value=function_value_new,
                gradient_norm=float(np.linalg.norm(gradient_new)),
                projected_gradient_norm=projected_gradient_norm,
                step_length=accepted_step,
                theta=self._theta,
                num_free=num_free,
                num_corrections=self._num_corrections,
                line_search_iters=line_search_result.num_evals,
                best_solution=best_solution,
            )

            if projected_gradient_norm <= config.pgtol:
                termination_message = (
                    f"Converged: projected gradient norm "
                    f"{projected_gradient_norm:.2e} <= {config.pgtol:.2e}"
                )
                break

            denominator = max(abs(function_value), abs(function_value_new), 1.0)
            relative_decrease = (function_value - function_value_new) / denominator
            if relative_decrease <= config.factr * self._machine_epsilon:
                termination_message = (
                    f"Converged: relative function decrease {relative_decrease:.2e} "
                    f"<= factr*eps = {config.factr * self._machine_epsilon:.2e}"
                )
                break

            self._update_correction_pairs(step_vector, gradient_difference)

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
            algorithm=AlgorithmChoice.LBFGSB,
        )
