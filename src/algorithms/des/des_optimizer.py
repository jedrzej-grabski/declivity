from typing import Callable, final, Union, TYPE_CHECKING
import numpy as np
import math
from numpy.typing import NDArray
from scipy.special import gamma
from src.algorithms.choices import AlgorithmChoice
from src.algorithms.des.config import DESConfig
from src.logging.des_logger import DESLogData
from src.utils.constraint_handlers import ConstraintHandler
from src.utils.ring_buffer import RingBuffer
from src.utils.helpers import delete_inf_nan, calculate_ft
from src.utils.repair_strategies import RepairStrategy, LamarckianRepair
from src.utils.population_initializers import PopulationInitializer, NormalPopulationInitializer

from src.core.base_optimizer import OptimizationResult
from src.core.population_optimizer import PopulationOptimizer
from src.core.algorithm_factory import register_optimizer


@final
@register_optimizer(AlgorithmChoice.DES, DESConfig)
class DESOptimizer(PopulationOptimizer[DESLogData, DESConfig]):
    """Differential Evolution Strategy optimizer with proper typing."""

    def __init__(
        self,
        func: Callable[[NDArray[np.float64]], float],
        initial_point: NDArray[np.float64],
        config: DESConfig | None = None,
        repair_strategy: RepairStrategy | None = None,
        population_initializer: PopulationInitializer | None = None,
        constraint_handler: ConstraintHandler | None = None,
        lower_bounds: Union[float, NDArray[np.float64], list[float]] = -100.0,
        upper_bounds: Union[float, NDArray[np.float64], list[float]] = 100.0,
        seed: int | np.random.Generator | None = None,
    ) -> None:
        """Initialize the DES optimizer."""

        if config is None:
            config = DESConfig(dimensions=len(initial_point))

        super().__init__(
            func=func,
            initial_point=initial_point,
            config=config,
            repair_strategy=repair_strategy or LamarckianRepair(),
            population_initializer=population_initializer or NormalPopulationInitializer(),
            algorithm=AlgorithmChoice.DES,
            constraint_handler=constraint_handler,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            seed=seed,
        )

    def optimize(self) -> OptimizationResult[DESLogData]:
        """Run the DES optimization algorithm."""

        N = self.dimensions
        budget = self.config.budget
        lambda_ = self.config.population_size
        path_length = self.config.path_length
        init_ft = self.config.init_ft
        hist_size = self.config.history
        c_ft = self.config.c_ft
        cp = self.config.cp
        max_iter = self.config.maxit
        lamarckian = self.config.lamarckian
        weights = self.config.weights
        mu = self.config.mu
        mu_eff = self.config.mu_eff
        c_cum = self.config.c_cum
        path_ratio = self.config.path_ratio

        self.evaluations = 0
        best_fitness = float("inf")
        best_solution = self.initial_point.copy()
        worst_fitness = None
        message = None
        iter_count = 0

        hist_head = 0
        history: list[NDArray[np.float64]] = []
        ft = init_ft

        # Create first population
        population = self.population_initializer.generate_population(
            rng=self.rng,
            x0=self.initial_point,
            pop_size=lambda_,
            lower_bounds=self.lower_bounds,
            upper_bounds=self.upper_bounds,
        )

        cumulative_mean = (self.upper_bounds + self.lower_bounds) / 2

        population_repaired = self.repair_strategy.repair_population(population, self.constraint_handler)

        if lamarckian:
            population = population_repaired

        # Evaluate initial population
        fitness = self.evaluate_population(
            population if lamarckian else population_repaired
        )

        old_mean = np.zeros(N)
        new_mean = cumulative_mean.copy()
        worst_fitness = np.max(fitness)

        # Store population and selection means
        pop_mean = np.mean(population, axis=0)
        mu_mean = new_mean

        # Initialize matrices for creating diffs
        diffs = np.zeros((N, lambda_))

        # Calculate chi_N
        chi_N = np.sqrt(2) * gamma((N + 1) / 2) / gamma(N / 2)
        hist_norm = 1 / np.sqrt(2)
        counter_repaired = 0

        # Allocate buffers
        steps = RingBuffer(path_length * N)
        d_mean = np.zeros((N, hist_size))
        ft_history = np.zeros(hist_size)
        pc = np.zeros((N, hist_size))

        # Main optimization loop
        while self.evaluations < budget:
            iter_count += 1
            hist_head = (hist_head + 1) % hist_size

            mu = math.floor(lambda_ / 2)
            weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
            weights = weights / np.sum(weights)

            self.logger.log_iteration(
                iteration=iter_count,
                evaluations=self.evaluations,
                ft=ft,
                fitness=fitness,
                population=population if self.config.diag_pop else None,
                best_fitness=best_fitness,
                worst_fitness=(
                    worst_fitness if worst_fitness is not None else float("inf")
                ),
                best_solution=best_solution,
                mean_fitness=float(np.mean(fitness)),
                eigenvalues=(
                    np.sort(np.linalg.eigvals(np.cov(population.T)))[::-1]
                    if self.config.diag_eigen
                    else None
                ),
            )

            # Select best mu individuals
            selection = np.argsort(fitness)[:mu]
            selected_points = population[selection]

            # Save selected population in history buffer
            if len(history) < hist_size:
                history.append(selected_points.T * hist_norm / ft)
            else:
                history[hist_head] = selected_points.T * hist_norm / ft

            # Calculate weighted mean of selected points
            old_mean = new_mean.copy()
            new_mean = np.sum(selected_points * weights.reshape(-1, 1), axis=0)

            # Write to buffers
            mu_mean = new_mean
            d_mean[:, hist_head] = (mu_mean - pop_mean) / ft

            step = (new_mean - old_mean) / ft

            # Update buffer of steps
            steps.push_all(step)

            # Update ft
            ft_history[hist_head] = ft
            if (
                iter_count > path_length - 1
                and not np.any(step == 0)
                and counter_repaired < 0.1 * lambda_
            ):
                ft = calculate_ft(
                    steps.peek(),
                    N,
                    lambda_,
                    path_length,
                    ft,
                    c_ft,
                    path_ratio,
                    chi_N,
                    mu_eff,
                )

            # Update parameters
            if hist_head == 0:
                pc[:, hist_head] = np.sqrt(mu) * step
            else:
                pc[:, hist_head] = (1 - cp) * pc[:, hist_head - 1] + np.sqrt(
                    mu * cp * (2 - cp)
                ) * step

            # Sample from history
            limit = min(iter_count, hist_size)
            history_sample = self.rng.choice(limit, lambda_, replace=True)
            history_sample2 = self.rng.choice(limit, lambda_, replace=True)

            x1_sample = np.zeros(lambda_, dtype=int)
            x2_sample = np.zeros(lambda_, dtype=int)

            for i in range(lambda_):
                hist_idx = history_sample[i]
                x1_sample[i] = self.rng.integers(0, history[hist_idx].shape[1])
                x2_sample[i] = self.rng.integers(0, history[hist_idx].shape[1])

            # Make diffs
            for i in range(lambda_):
                hist_idx = history_sample[i]
                x1 = history[hist_idx][:, x1_sample[i]]
                x2 = history[hist_idx][:, x2_sample[i]]

                diffs[:, i] = (
                    np.sqrt(c_cum) * (x1 - x2 + self.rng.standard_normal() * d_mean[:, hist_idx])
                    + np.sqrt(1 - c_cum) * self.rng.standard_normal() * pc[:, history_sample2[i]]
                )

            # Generate new population
            population = (
                new_mean.reshape(1, -1)
                + ft * diffs.T
                + (1 - 2 / N**2) ** (iter_count / 2)
                * self.rng.standard_normal(size=(lambda_, N))
                / chi_N
            )

            population = delete_inf_nan(population)

            # Check constraints violations and repair if necessary
            population_repaired = self.repair_strategy.repair_population(population, self.constraint_handler)

            # Count repaired individuals
            counter_repaired = int(np.any(population != population_repaired, axis=1).sum())

            if lamarckian:
                population = population_repaired

            pop_mean = np.mean(population, axis=0)

            # Evaluate population
            fitness = self.evaluate_population(
                population if lamarckian else population_repaired
            )

            # Check for best fitness
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < best_fitness:
                best_fitness = fitness[best_idx]
                best_solution = (
                    population_repaired[best_idx]
                    if not lamarckian
                    else population[best_idx]
                )

            # Check worst fitness
            worst_idx = np.argmax(fitness)
            if (
                fitness[worst_idx] > worst_fitness
                if worst_fitness is not None
                else float("-inf")
            ):
                worst_fitness = fitness[worst_idx]

            # Check if the mean point is better
            cumulative_mean = 0.8 * cumulative_mean + 0.2 * new_mean
            cumulative_mean_repaired = self.constraint_handler.repair(cumulative_mean)
            mean_fitness = self.evaluate(cumulative_mean_repaired)

            if mean_fitness < best_fitness:
                best_fitness = mean_fitness
                best_solution = cumulative_mean_repaired

        result: OptimizationResult[DESLogData] = OptimizationResult(
            best_solution=best_solution,
            best_fitness=best_fitness,
            evaluations=self.evaluations,
            message=message if message else "Maximum function evaluations reached.",
            diagnostic=self.get_logs(),
            algorithm=AlgorithmChoice.DES,
        )

        return result
