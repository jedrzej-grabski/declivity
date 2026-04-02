"""
Pure Python implementation of the L-BFGS-B algorithm.

A limited-memory quasi-Newton method for bound-constrained optimization.
Reimplemented from the Fortran L-BFGS-B v3.0 (Byrd, Lu, Nocedal, Zhu 1995)
with the subspace minimization correction from Morales and Nocedal (2011).

References:
    R.H. Byrd, P. Lu, J. Nocedal, C. Zhu, "A Limited Memory Algorithm for
    Bound Constrained Optimization", SIAM J. Scientific Computing 16 (1995).

    J.L. Morales, J. Nocedal, "Remark on Algorithm 778: L-BFGS-B",
    ACM Trans. Math. Software 38 (2011).
"""

from typing import Callable, Union, final, TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray
from scipy.linalg import cho_factor, cho_solve

from src.algorithms.choices import AlgorithmChoice
from src.algorithms.lbfgsb.config import LBFGSBConfig
from src.algorithms.lbfgsb.line_search import perform_line_search
from src.utils.boundary_handlers import BoundaryHandler, BoundaryHandlerType
from src.core.base_optimizer import BaseOptimizer, OptimizationResult
from src.logging.lbfgsb_logger import LBFGSBLogger

if TYPE_CHECKING:
    from src.logging.lbfgsb_logger import LBFGSBLogData


@final
class LBFGSBOptimizer(BaseOptimizer["LBFGSBLogData", LBFGSBConfig]):
    """L-BFGS-B optimizer for bound-constrained minimization.

    Uses a compact L-BFGS Hessian approximation with generalized Cauchy point
    computation for active set identification and subspace minimization for
    second-order refinement within the free variable space.
    """

    def __init__(
        self,
        func: Callable[[NDArray[np.float64]], float],
        initial_point: NDArray[np.float64],
        config: LBFGSBConfig | None = None,
        boundary_handler: BoundaryHandler | None = None,
        boundary_strategy: BoundaryHandlerType | None = None,
        lower_bounds: Union[float, NDArray[np.float64], list[float]] = -100.0,
        upper_bounds: Union[float, NDArray[np.float64], list[float]] = 100.0,
        seed: int | np.random.Generator | None = None,
        gradient_fn: Callable[[NDArray[np.float64]], NDArray[np.float64]] | None = None,
    ) -> None:
        if config is None:
            config = LBFGSBConfig(dimensions=len(initial_point))

        super().__init__(
            func=func,
            initial_point=initial_point,
            config=config,
            algorithm=AlgorithmChoice.LBFGSB,
            boundary_handler=boundary_handler,
            boundary_strategy=boundary_strategy,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            seed=seed,
        )

        self.logger = LBFGSBLogger(config=self.config)
        self._gradient_fn = gradient_fn
        self._finite_diff_epsilon = config._fd_eps_actual
        self._finite_diff_method = config.fd_method
        self._memory_size = config.m
        self._machine_epsilon = np.finfo(float).eps

        # Initial Hessian diagonal B_0 = diag(initial_hessian_diagonal)
        num_dimensions = len(initial_point)
        if config.initial_hessian is None:
            self._initial_hessian_diagonal = np.ones(num_dimensions)
        elif np.isscalar(config.initial_hessian):
            self._initial_hessian_diagonal = np.full(
                num_dimensions, float(config.initial_hessian)
            )
        else:
            hessian_diag = np.asarray(config.initial_hessian, dtype=float)
            if hessian_diag.shape != (num_dimensions,):
                raise ValueError(
                    f"initial_hessian array must have length {num_dimensions}, "
                    f"got {hessian_diag.shape}"
                )
            if np.any(hessian_diag <= 0):
                raise ValueError("initial_hessian diagonal entries must be positive")
            self._initial_hessian_diagonal = hessian_diag

        # Correction pair storage (most recent pair last)
        self._step_vectors: list[NDArray[np.float64]] = []
        self._gradient_diff_vectors: list[NDArray[np.float64]] = []
        self._theta: float = 1.0
        self._num_corrections: int = 0

        # Cached Gram matrices, rebuilt after each correction pair update
        self._steps_dot_grad_diffs: NDArray[np.float64] = np.empty((0, 0))
        self._steps_dot_steps: NDArray[np.float64] = np.empty((0, 0))
        self._cholesky_factor_of_T: tuple | None = None

    def _evaluate_function_and_gradient(
        self, x: NDArray[np.float64]
    ) -> tuple[float, NDArray[np.float64]]:
        """Evaluate the objective function and its gradient at x."""
        function_value = self.evaluate(x)
        gradient = self._compute_gradient(x, function_value)
        return function_value, gradient

    def _compute_gradient(
        self, x: NDArray[np.float64], function_value_at_x: float | None = None
    ) -> NDArray[np.float64]:
        """Compute gradient via analytical function or finite differences."""
        if self._gradient_fn is not None:
            return np.asarray(self._gradient_fn(x), dtype=float)

        num_vars = len(x)
        gradient = np.zeros(num_vars)
        epsilon = self._finite_diff_epsilon

        if self._finite_diff_method == "central":
            for i in range(num_vars):
                x_forward = x.copy()
                x_backward = x.copy()
                x_forward[i] += epsilon
                x_backward[i] -= epsilon
                gradient[i] = (
                    self.evaluate(x_forward) - self.evaluate(x_backward)
                ) / (2.0 * epsilon)
        else:
            if function_value_at_x is None:
                function_value_at_x = self.evaluate(x)
            for i in range(num_vars):
                x_forward = x.copy()
                x_forward[i] += epsilon
                gradient[i] = (
                    self.evaluate(x_forward) - function_value_at_x
                ) / epsilon

        return gradient

    def _compute_directional_derivative(
        self, x: NDArray[np.float64], direction: NDArray[np.float64], alpha: float
    ) -> tuple[float, float]:
        """Evaluate phi(alpha) = f(x + alpha*d) and phi'(alpha) = grad f . d.

        When an analytical gradient is available, the full gradient is cached
        for reuse after the line search completes.
        When using finite differences, the directional derivative is computed
        directly with two evaluations rather than a full gradient (2n evals).
        """
        x_trial = x + alpha * direction
        f_trial = self.evaluate(x_trial)

        if self._gradient_fn is not None:
            gradient_at_trial = self._gradient_fn(x_trial)
            self._cached_gradient = np.asarray(gradient_at_trial, dtype=float)
            return f_trial, float(np.dot(self._cached_gradient, direction))
        else:
            epsilon = self._finite_diff_epsilon
            f_forward = self.evaluate(x_trial + epsilon * direction)
            f_backward = self.evaluate(x_trial - epsilon * direction)
            return f_trial, (f_forward - f_backward) / (2.0 * epsilon)

    def _compute_projected_gradient_inf_norm(
        self, x: NDArray[np.float64], gradient: NDArray[np.float64]
    ) -> float:
        """Compute the infinity norm of the projected gradient.

        The projected gradient zeros out components where a variable sits at
        its bound and the gradient points into the bound. This is the standard
        KKT optimality measure for bound-constrained problems.
        """
        projected_gradient = gradient.copy()
        negative_mask = gradient < 0
        positive_mask = gradient > 0
        projected_gradient[negative_mask] = np.maximum(
            x[negative_mask] - self.upper_bounds[negative_mask],
            gradient[negative_mask],
        )
        projected_gradient[positive_mask] = np.minimum(
            x[positive_mask] - self.lower_bounds[positive_mask],
            gradient[positive_mask],
        )
        if len(projected_gradient) == 0:
            return 0.0
        return float(np.max(np.abs(projected_gradient)))

    # L-BFGS compact representation operations

    def _update_correction_pairs(
        self,
        step_vector: NDArray[np.float64],
        gradient_difference: NDArray[np.float64],
    ) -> bool:
        """Store a new (s, y) correction pair and rebuild cached matrices.

        The pair is accepted only if it satisfies the curvature condition
        s'y > eps * y'y, which ensures the Hessian approximation remains
        positive definite. Returns True if accepted.
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

        if not (
            np.all(np.isfinite(self._steps_dot_grad_diffs))
            and np.all(np.isfinite(self._steps_dot_steps))
        ):
            self._reset_correction_memory()
            return False

        self._factorize_middle_matrix()
        return True

    def _factorize_middle_matrix(self) -> None:
        """Form and Cholesky-factorize T = theta * S'S + L * D^{-1} * L'.

        T appears in the block elimination used by the middle matrix multiply.
        If T is not positive definite due to numerical issues, the correction
        memory is reset as a safe fallback.
        """
        num_corrections = self._num_corrections
        if num_corrections == 0:
            self._cholesky_factor_of_T = None
            return

        curvature_diagonal = np.diag(self._steps_dot_grad_diffs).copy()
        strict_lower_triangle = np.tril(self._steps_dot_grad_diffs, -1)
        safe_curvature_diagonal = np.maximum(
            curvature_diagonal, self._machine_epsilon
        )

        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            T = (
                self._theta * self._steps_dot_steps
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
        """Compute p = M * v, where M is the middle matrix in B = theta*I - W*M*W'.

        The inverse of M has block structure:
            M^{-1} = [-D    L']
                     [ L    theta * S'S]
        where D = diag(s_i' y_i) and L is the strict lower triangle of S'Y.
        The system M^{-1} * p = v is solved via block elimination using
        the Cholesky factorization of T.
        """
        num_corrections = self._num_corrections
        if num_corrections == 0 or self._cholesky_factor_of_T is None:
            return np.zeros_like(vector)

        gradient_diff_part = vector[:num_corrections]
        step_part = vector[num_corrections : 2 * num_corrections]

        curvature_diagonal = np.diag(self._steps_dot_grad_diffs).copy()
        strict_lower_triangle = np.tril(self._steps_dot_grad_diffs, -1)
        safe_curvature_diagonal = np.maximum(
            curvature_diagonal, self._machine_epsilon
        )

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
        """Clear all stored correction pairs and reset to B_0 = theta * I."""
        self._step_vectors.clear()
        self._gradient_diff_vectors.clear()
        self._num_corrections = 0
        self._theta = 1.0
        self._cholesky_factor_of_T = None

    # Generalized Cauchy Point computation

    def _compute_cauchy_point(
        self, x: NDArray[np.float64], gradient: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int32]]:
        """Compute the Generalized Cauchy Point (GCP).

        The GCP is the first local minimizer of the quadratic model Q along the
        piecewise-linear projected steepest descent path P(x - t*g, l, u). As t
        increases, variables that hit their bounds are fixed, creating breakpoints
        where the path changes direction.

        Returns:
            cauchy_point: the GCP vector.
            w_displacement: W'(cauchy_point - x), used in subspace minimization.
            variable_status: classification of each variable (see below).

        Variable status codes:
             3  permanently fixed (lower bound equals upper bound)
             2  fixed at upper bound
             1  fixed at lower bound
             0  free, moved during Cauchy search
            -3  free, but zero gradient component
        """
        num_vars = self.dimensions
        num_corrections = self._num_corrections
        lower = self.lower_bounds
        upper = self.upper_bounds

        cauchy_direction = np.zeros(num_vars)
        variable_status = np.zeros(num_vars, dtype=np.int32)

        # Classify variables and compute breakpoints along the projected path
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

        # Project the Cauchy direction onto the L-BFGS basis: p = W' d
        projected_direction = np.zeros(2 * num_corrections)
        if num_corrections > 0:
            step_matrix = np.column_stack(self._step_vectors)
            grad_diff_matrix = np.column_stack(self._gradient_diff_vectors)
            projected_direction[:num_corrections] = (
                grad_diff_matrix.T @ cauchy_direction
            )
            projected_direction[num_corrections : 2 * num_corrections] = (
                self._theta * (step_matrix.T @ cauchy_direction)
            )

        # Effective base Hessian diagonal for the quadratic model.
        # When persist_initial_hessian is True, the user-supplied per-variable
        # scaling is multiplied by the adaptive theta at every iteration.
        # When False, the scaling is only used before any corrections exist.
        if self._num_corrections > 0 and not self.config.persist_initial_hessian:
            hessian_diagonal = np.full(num_vars, self._theta)
        else:
            hessian_diagonal = self._theta * self._initial_hessian_diagonal

        # Initialize the quadratic model derivatives along the Cauchy path.
        # first_derivative = dQ/dt, second_derivative = d^2 Q / dt^2.
        first_derivative = float(np.dot(gradient, cauchy_direction))

        if num_corrections > 0:
            middle_times_projected = self._middle_matrix_multiply(projected_direction)
            second_derivative = float(
                np.dot(hessian_diagonal * cauchy_direction, cauchy_direction)
            ) - float(np.dot(projected_direction, middle_times_projected))
        else:
            second_derivative = float(
                np.dot(hessian_diagonal * cauchy_direction, cauchy_direction)
            )
            middle_times_projected = np.zeros(2 * num_corrections)

        if second_derivative > self._machine_epsilon:
            unconstrained_minimizer_step = -first_derivative / second_derivative
        else:
            unconstrained_minimizer_step = 0.0

        # Walk along breakpoints, updating the quadratic model at each one.
        # At each breakpoint a variable hits its bound and is fixed, changing
        # the search direction and the model derivatives.
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

            # Update model derivatives for removing this variable
            first_derivative = (
                first_derivative
                + segment_length * second_derivative
                + direction_component * direction_component
                - hessian_diagonal[var_index]
                * direction_component
                * distance_to_bound
            )
            second_derivative = (
                second_derivative
                - hessian_diagonal[var_index]
                * direction_component
                * direction_component
            )

            if num_corrections > 0:
                # Advance the W-space displacement to the breakpoint
                w_displacement += segment_length * projected_direction

                # Extract the W-matrix row for the variable being fixed
                w_row_for_variable = np.zeros(2 * num_corrections)
                for k in range(num_corrections):
                    w_row_for_variable[k] = self._gradient_diff_vectors[k][var_index]
                    w_row_for_variable[num_corrections + k] = (
                        self._theta * self._step_vectors[k][var_index]
                    )

                middle_times_w_row = self._middle_matrix_multiply(w_row_for_variable)

                # L-BFGS correction terms for the derivative updates
                first_derivative += direction_component * float(
                    np.dot(w_displacement, middle_times_w_row)
                )
                second_derivative += 2.0 * direction_component * float(
                    np.dot(w_row_for_variable, middle_times_projected)
                ) - direction_component * direction_component * float(
                    np.dot(w_row_for_variable, middle_times_w_row)
                )

                # Update projected direction for the new search direction
                projected_direction -= direction_component * w_row_for_variable
                middle_times_projected = self._middle_matrix_multiply(
                    projected_direction
                )

            cauchy_direction[var_index] = 0.0

            if second_derivative > self._machine_epsilon:
                unconstrained_minimizer_step = (
                    -first_derivative / second_derivative
                )
            else:
                unconstrained_minimizer_step = 0.0
        else:
            total_time += max(unconstrained_minimizer_step, 0.0)

        # Reconstruct the Cauchy point from variable statuses and total time
        cauchy_point = x.copy()
        for i in range(num_vars):
            if variable_status[i] == 0:
                cauchy_point[i] = x[i] + total_time * (-gradient[i])
            elif variable_status[i] == 1:
                cauchy_point[i] = lower[i]
            elif variable_status[i] == 2:
                cauchy_point[i] = upper[i]

        cauchy_point = np.clip(cauchy_point, lower, upper)

        # Compute w_displacement = W'(cauchy_point - x) at the final GCP
        if num_corrections > 0:
            displacement = cauchy_point - x
            step_matrix = np.column_stack(self._step_vectors)
            grad_diff_matrix = np.column_stack(self._gradient_diff_vectors)
            w_displacement[:num_corrections] = grad_diff_matrix.T @ displacement
            w_displacement[num_corrections : 2 * num_corrections] = (
                self._theta * (step_matrix.T @ displacement)
            )

        return cauchy_point, w_displacement, variable_status

    def _identify_free_variables(
        self, variable_status: NDArray[np.int32]
    ) -> tuple[list[int], int]:
        """Return indices of free variables (status <= 0) at the Cauchy point."""
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

        Solves the reduced Newton system (Z'BZ)d = r where Z projects onto the
        free variables, using the Woodbury identity to exploit the compact L-BFGS
        structure. The cost is O(n * m^2) rather than O(n^3).

        Includes the Morales-Nocedal 2011 correction: after computing the Newton
        direction, project onto bounds and verify descent. If projection breaks
        the descent property, fall back to a feasible backtracking step.
        """
        num_corrections = self._num_corrections

        if num_free == 0 or num_corrections == 0:
            return cauchy_point.copy()

        # Reduced gradient: r = -Z'(g + B(xcp - x))
        middle_times_displacement = self._middle_matrix_multiply(w_displacement)

        reduced_gradient = np.zeros(num_free)
        for j in range(num_free):
            idx = free_variable_indices[j]
            reduced_gradient[j] = -(
                gradient[idx] + self._theta * (cauchy_point[idx] - x[idx])
            )
            for k in range(num_corrections):
                reduced_gradient[j] += (
                    self._gradient_diff_vectors[k][idx]
                    * middle_times_displacement[k]
                )
                reduced_gradient[j] += (
                    self._theta
                    * self._step_vectors[k][idx]
                    * middle_times_displacement[num_corrections + k]
                )

        # A = Z'W, the L-BFGS basis restricted to free variables
        restricted_basis = np.zeros((num_free, 2 * num_corrections))
        for k in range(num_corrections):
            for j in range(num_free):
                idx = free_variable_indices[j]
                restricted_basis[j, k] = self._gradient_diff_vectors[k][idx]
                restricted_basis[j, num_corrections + k] = (
                    self._theta * self._step_vectors[k][idx]
                )

        projected_reduced_gradient = restricted_basis.T @ reduced_gradient

        # K = M^{-1} - (1/theta) * A'A, the Woodbury auxiliary matrix
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
            self._theta * self._steps_dot_steps
        )

        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            woodbury_matrix = middle_matrix_inverse - (1.0 / self._theta) * (
                restricted_basis.T @ restricted_basis
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

        # Newton direction: d = (1/theta)*r + (1/theta^2) * A * K^{-1} * A' * r
        reduced_newton_direction = (1.0 / self._theta) * reduced_gradient + (
            1.0 / self._theta**2
        ) * (restricted_basis @ woodbury_solution)

        # Lift back to full space
        candidate = cauchy_point.copy()
        for j in range(num_free):
            candidate[free_variable_indices[j]] = (
                cauchy_point[free_variable_indices[j]] + reduced_newton_direction[j]
            )

        # Morales-Nocedal 2011 safeguard: project and verify descent
        projected_candidate = np.clip(
            candidate, self.lower_bounds, self.upper_bounds
        )
        directional_derivative = float(
            np.dot(gradient, projected_candidate - x)
        )

        if directional_derivative <= 0:
            return projected_candidate

        # Projection broke descent; fall back to maximum feasible step from xcp
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

    def _compute_maximum_feasible_step(
        self,
        x: NDArray[np.float64],
        direction: NDArray[np.float64],
        is_first_iteration: bool,
    ) -> float:
        """Find the largest step along direction that keeps all variables in bounds."""
        if is_first_iteration:
            return 1.0

        max_step = 1e10
        for i in range(self.dimensions):
            if direction[i] < -self._machine_epsilon:
                distance_to_lower = self.lower_bounds[i] - x[i]
                if distance_to_lower >= 0:
                    max_step = 0.0
                    break
                max_step = min(max_step, distance_to_lower / direction[i])
            elif direction[i] > self._machine_epsilon:
                distance_to_upper = self.upper_bounds[i] - x[i]
                if distance_to_upper <= 0:
                    max_step = 0.0
                    break
                max_step = min(max_step, distance_to_upper / direction[i])

        return max(max_step, 0.0)

    # Main optimization loop

    def optimize(self) -> OptimizationResult["LBFGSBLogData"]:
        """Run the L-BFGS-B optimization algorithm."""
        self.evaluations = 0
        num_vars = self.dimensions
        config = self.config

        self._reset_correction_memory()
        self._cached_gradient: NDArray[np.float64] | None = None

        x = np.clip(self.initial_point.copy(), self.lower_bounds, self.upper_bounds)

        function_value, gradient = self._evaluate_function_and_gradient(x)
        best_fitness = function_value
        best_solution = x.copy()
        termination_message = None

        projected_gradient_norm = self._compute_projected_gradient_inf_norm(
            x, gradient
        )
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

        while self.evaluations < config.budget:
            iteration += 1

            # Generalized Cauchy Point: identify active set
            cauchy_point, w_displacement, variable_status = (
                self._compute_cauchy_point(x, gradient)
            )

            free_variable_indices, num_free = self._identify_free_variables(
                variable_status
            )

            # Subspace minimization: refine within the free variable space
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

            # Construct search direction
            direction = search_target - x
            direction_norm = float(np.linalg.norm(direction))

            if direction_norm < self._machine_epsilon:
                projected_gradient_norm = (
                    self._compute_projected_gradient_inf_norm(x, gradient)
                )
                if projected_gradient_norm <= config.pgtol:
                    termination_message = (
                        f"Converged: projected gradient norm "
                        f"{projected_gradient_norm:.2e}"
                    )
                    break
                self._reset_correction_memory()
                continue

            directional_derivative = float(np.dot(gradient, direction))
            if directional_derivative >= 0:
                # Fall back to projected steepest descent
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

            # Line search
            max_feasible_step = self._compute_maximum_feasible_step(
                x, direction, iteration == 1
            )
            if max_feasible_step <= 0:
                termination_message = "Maximum feasible step is zero"
                break

            if iteration == 1:
                initial_step = min(1.0 / direction_norm, max_feasible_step)
            else:
                initial_step = min(1.0, max_feasible_step)

            self._cached_gradient = None

            def phi_and_dphi(
                alpha: float, _x=x, _d=direction
            ) -> tuple[float, float]:
                return self._compute_directional_derivative(_x, _d, alpha)

            line_search_result = perform_line_search(
                method=config.line_search,
                phi_dphi=phi_and_dphi,
                stp0=initial_step,
                phi0=function_value,
                dphi0=directional_derivative,
                stpmax=max_feasible_step,
                ftol=config.ftol,
                gtol=config.gtol_ls,
                xtol=config.xtol_ls,
                maxiter=config.max_ls_iter,
            )

            accepted_step = line_search_result.step

            if accepted_step <= 0 or (
                not line_search_result.converged and self._num_corrections == 0
            ):
                termination_message = "Line search failed"
                break

            if not line_search_result.converged and self._num_corrections > 0:
                self._reset_correction_memory()
                continue

            # Accept the step
            step_vector = accepted_step * direction
            x_new = np.clip(
                x + step_vector, self.lower_bounds, self.upper_bounds
            )
            step_vector = x_new - x

            if self._cached_gradient is not None:
                gradient_new = self._cached_gradient
                function_value_new = line_search_result.f_new
            else:
                function_value_new, gradient_new = (
                    self._evaluate_function_and_gradient(x_new)
                )

            gradient_difference = gradient_new - gradient

            if function_value_new < best_fitness:
                best_fitness = function_value_new
                best_solution = x_new.copy()

            # Convergence tests
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

            # Store the correction pair and update the Hessian approximation
            self._update_correction_pairs(step_vector, gradient_difference)

            x = x_new
            function_value = function_value_new
            gradient = gradient_new

            if self.evaluations >= config.budget:
                termination_message = "Maximum function evaluations reached"
                break

        if termination_message is None:
            termination_message = "Maximum function evaluations reached"

        return OptimizationResult(
            best_solution=best_solution,
            best_fitness=best_fitness,
            evaluations=self.evaluations,
            message=termination_message,
            diagnostic=self.get_logs(),
            algorithm=AlgorithmChoice.LBFGSB,
        )
