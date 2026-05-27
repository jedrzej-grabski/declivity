from typing import Callable, final, Union, TYPE_CHECKING
import numpy as np

from numpy.typing import NDArray

from src.algorithms.choices import AlgorithmChoice
from src.algorithms.cmaes.config import CMAESConfig
from src.algorithms.cmaes.cmaes_reference import CMA

from src.utils.constraint_handlers import ConstraintHandler
from src.utils.repair_strategies import RepairStrategy, IdentityRepair
from src.utils.population_initializers import PopulationInitializer, IdentityPopulationInitializer

from src.core.base_optimizer import OptimizationResult
from src.core.population_optimizer import PopulationOptimizer
from src.core.algorithm_factory import register_optimizer

if TYPE_CHECKING:
    from src.logging.cmaes_logger import CMAESLogData
    from src.utils.covariance import CovarianceMatrix


@final
@register_optimizer(AlgorithmChoice.CMAES, CMAESConfig)
class CMAESOptimizer(PopulationOptimizer["CMAESLogData", CMAESConfig]):
    """CMA-ES optimizer wrapper around reference implementation with proper logging."""

    def __init__(
        self,
        func: Callable[[NDArray[np.float64]], float],
        initial_point: NDArray[np.float64],
        config: CMAESConfig | None = None,
        repair_strategy: RepairStrategy | None = None,
        population_initializer: PopulationInitializer | None = None,
        constraint_handler: ConstraintHandler | None = None,
        lower_bounds: Union[float, NDArray[np.float64], list[float]] = -100.0,
        upper_bounds: Union[float, NDArray[np.float64], list[float]] = 100.0,
        seed: int | np.random.Generator | None = None,
    ) -> None:
        """Initialize the CMA-ES optimizer."""

        if config is None:
            config = CMAESConfig(dimensions=len(initial_point))

        super().__init__(
            func=func,
            initial_point=initial_point,
            config=config,
            repair_strategy=repair_strategy or IdentityRepair(),
            population_initializer=population_initializer or IdentityPopulationInitializer(),
            algorithm=AlgorithmChoice.CMAES,
            constraint_handler=constraint_handler,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            seed=seed,
        )

        # Auto-calculate sigma if not set
        if self.config.sigma == 0.0:
            bounds_range = self.upper_bounds - self.lower_bounds
            self.config.sigma = float(np.mean(bounds_range) / 5.0)

        # Prepare bounds in the format expected by reference implementation
        # bounds should be (n_dim, 2) where bounds[:, 0] is lower, bounds[:, 1] is upper
        bounds_array = np.column_stack(
            (self.lower_bounds, self.upper_bounds)
        )

        # Initialize the reference CMA-ES implementation, sharing the rng
        self._cma = CMA(
            mean=self.initial_point.copy(),
            sigma=self.config.sigma,
            bounds=bounds_array,
            population_size=self.config.population_size,
            seed=self.rng,
        )

    def optimize(self) -> OptimizationResult["CMAESLogData"]:
        """Run the CMA-ES optimization algorithm using reference implementation."""

        self.evaluations = 0
        best_fitness = float("inf")
        best_solution = self.initial_point.copy()
        worst_fitness = None
        message = None

        # Main optimization loop
        while self.evaluations < self.config.budget:
            generation = self._cma.generation + 1

            # Ask for new solutions
            solutions: list[tuple[NDArray[np.float64], float]] = []
            population = []
            fitness_values = []

            for _ in range(self._cma.population_size):
                x = self._cma.ask()

                fitness = self.evaluate(x)

                solutions.append((x, fitness))
                population.append(x)
                fitness_values.append(fitness)

                # Track best
                if fitness < best_fitness:
                    best_fitness = fitness
                    best_solution = x.copy()

            # ``tell`` requires a full population; if the budget elapses mid
            # population, the inner loop above always completes (a few extra
            # evaluations vs corrupted internal state). The outer ``while``
            # will exit on the next iteration.
            self._cma.tell(solutions)

            # Calculate statistics for logging
            fitness_array = np.array(fitness_values)
            population_array = np.array(population)

            if worst_fitness is None or np.max(fitness_array) > worst_fitness:
                worst_fitness = float(np.max(fitness_array))

            median_fitness = float(np.median(fitness_array))

            # Evaluate mean
            mean_repaired = self.constraint_handler.repair(self._cma.mean)
            mean_fitness_value = self.evaluate(mean_repaired)

            # Get internal state for logging
            # Perform eigendecomposition to get B and D
            B, D = self._cma._eigen_decomposition()
            eigenvalues_sorted = np.sort(D**2) if self.config.diag_eigen else None

            # Log iteration
            self.logger.log_iteration(
                iteration=generation,
                evaluations=self.evaluations,
                sigma=self._cma._sigma,
                fitness=fitness_array,
                population=population_array if self.config.diag_pop else None,
                best_fitness=best_fitness,
                worst_fitness=(
                    worst_fitness if worst_fitness is not None else float("inf")
                ),
                best_solution=best_solution,
                mean_fitness=mean_fitness_value,
                median_fitness=median_fitness,
                pc=self._cma._pc,
                ps=self._cma._p_sigma,
                mean_vector=self._cma._mean,
                eigenvalues=eigenvalues_sorted,
                covariance_matrix=self._cma._C if self.config.diag_eigen else None,
            )

            # Check termination
            if self._cma.should_stop():
                message = self._get_termination_message()
                break

            # Check budget
            if self.evaluations >= self.config.budget:
                break

        if message is None:
            message = "Maximum function evaluations reached."

        result: OptimizationResult["CMAESLogData"] = OptimizationResult(
            best_solution=best_solution,
            best_fitness=best_fitness,
            evaluations=self.evaluations,
            message=message,
            diagnostic=self.get_logs(),
            algorithm=AlgorithmChoice.CMAES,
        )

        return result

    def get_learned_covariance(self) -> "CovarianceMatrix":
        """Return the current covariance matrix as a CovarianceMatrix object.

        Wraps the internal CMA-ES covariance C in the framework's
        CovarianceMatrix dataclass, providing the eigendecomposition
        and utility methods (sqrt, inv_sqrt, condition number).
        """
        from src.utils.covariance import _decompose

        C = self._cma._C.copy()
        mean = self._cma._mean.copy()
        rank = min(C.shape[0], C.shape[1])
        return _decompose(C, mean, rank)

    def get_eigendecomposition(
        self,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return the cached eigendecomposition of the covariance matrix.

        Returns ``(B, D)`` where ``B`` are the eigenvectors as columns and
        ``D`` are the square roots of the eigenvalues, so that
        ``C = B @ diag(D**2) @ B.T``. The reference implementation already
        maintains this decomposition; reusing it avoids re-running an
        ``eigh`` or ``inv`` call from the outside (e.g. for handoff).
        """
        B, D = self._cma._eigen_decomposition()
        return B.copy(), D.copy()

    @property
    def sigma(self) -> float:
        """Current step-size parameter sigma."""
        return float(self._cma._sigma)

    @property
    def mean(self) -> NDArray[np.float64]:
        """Current distribution mean."""
        return self._cma._mean.copy()

    def _get_termination_message(self) -> str:
        """Get the reason for termination from the reference implementation."""
        B, D = self._cma._eigen_decomposition()
        dC = np.diag(self._cma._C)

        # Check each termination criterion
        if (
            self._cma.generation > self._cma._funhist_term
            and np.max(self._cma._funhist_values) - np.min(self._cma._funhist_values)
            < self._cma._tolfun
        ):
            return "Function value range below tolerance."

        if np.all(self._cma._sigma * dC < self._cma._tolx) and np.all(
            self._cma._sigma * self._cma._pc < self._cma._tolx
        ):
            return "All standard deviations smaller than tolerance."

        if self._cma._sigma * np.max(D) > self._cma._tolxup:
            return "Step size diverged (too large)."

        if np.any(
            self._cma._mean == self._cma._mean + (0.2 * self._cma._sigma * np.sqrt(dC))
        ):
            return "No effect in coordinate update."

        i = self._cma.generation % self._cma.dim
        if np.all(
            self._cma._mean
            == self._cma._mean + (0.1 * self._cma._sigma * D[i] * B[:, i])
        ):
            return "No effect in axis update."

        condition_cov = np.max(D) / np.min(D)
        if condition_cov > self._cma._tolconditioncov:
            return "Condition number of covariance matrix exceeded tolerance."

        return "CMA-ES internal termination criterion met."
