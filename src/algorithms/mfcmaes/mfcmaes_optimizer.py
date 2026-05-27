from typing import Callable, final, Union, TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray

from src.algorithms.choices import AlgorithmChoice
from src.algorithms.mfcmaes.mfcmaes_config import MFCMAESConfig
from src.utils.constraint_handlers import ConstraintHandler
from src.utils.repair_strategies import RepairStrategy, ClampRepair
from src.utils.population_initializers import PopulationInitializer, MeanSigmaPopulationInitializer
from src.core.base_optimizer import OptimizationResult
from src.core.population_optimizer import PopulationOptimizer
from src.core.algorithm_factory import register_optimizer

if TYPE_CHECKING:
    from src.logging.mfcmaes_logger import MFCMAESLogData


@final
@register_optimizer(AlgorithmChoice.MFCMAES, MFCMAESConfig)
class MFCMAESOptimizer(PopulationOptimizer["MFCMAESLogData", MFCMAESConfig]):
    """Matrix-Free CMA-ES optimizer implementation."""

    def __init__(
        self,
        func: Callable[[NDArray[np.float64]], float],
        initial_point: NDArray[np.float64],
        config: MFCMAESConfig | None = None,
        repair_strategy: RepairStrategy | None = None,
        population_initializer: PopulationInitializer | None = None,
        constraint_handler: ConstraintHandler | None = None,
        lower_bounds: Union[float, NDArray[np.float64], list[float]] = -100.0,
        upper_bounds: Union[float, NDArray[np.float64], list[float]] = 100.0,
        seed: int | np.random.Generator | None = None,
    ) -> None:

        if config is None:
            config = MFCMAESConfig(dimensions=len(initial_point))

        super().__init__(
            func=func,
            initial_point=initial_point,
            config=config,
            repair_strategy=repair_strategy or ClampRepair(),
            population_initializer=population_initializer or MeanSigmaPopulationInitializer(sigma=config.sigma),
            algorithm=AlgorithmChoice.MFCMAES,
            constraint_handler=constraint_handler,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            seed=seed,
        )

        self.mean = self.initial_point.copy()
        self.sigma = self.config.sigma
        self.pc = np.zeros(self.config.dimensions)

        # p_history: evolution paths (dimensions x window)
        self.p_history = np.zeros((self.config.dimensions, self.config.window))

        # d_history: selected difference vectors (dimensions x window*mu)
        self.d_history = np.zeros(
            (self.config.dimensions, self.config.window * self.config.mu)
        )

        self.prev_midpoint_fitness = np.inf
        self.current_midpoint_fitness = np.inf
        self.p_succ = 0.0

        self._precompute_decay_table()

        self.constraint_violations = 0

    def _precompute_decay_table(self) -> None:
        """Precompute the decay factors (1 - c_cov)^((t-τ)/2) for the archive."""
        t = np.arange(1, self.config.maxit + 1)
        self.decay_table = (1 - self.config.c_cov) ** ((t - 1) / 2)

    def _p_index(self, t: int) -> int:
        """Get the index in p_history for generation t (circular buffer)."""
        return (t - 1) % self.config.window

    def _d_range(self, t: int) -> tuple[int, int]:
        """Get the slice range in d_history for generation t."""
        start = self._p_index(t) * self.config.mu
        end = start + self.config.mu
        return start, end

    def _shift_array(self, arr: NDArray[np.float64], n: int) -> NDArray[np.float64]:
        """Circular shift array by n positions."""
        n = n % len(arr)
        return np.concatenate([arr[-n:], arr[:-n]])

    def _generate_population(self, generation: int) -> NDArray[np.float64]:
        # Get decay factors for current generation
        window_size = min(generation, self.config.window)
        decay = self.decay_table[self.config.window - 1 :: -1][:window_size]
        decay = self._shift_array(decay, generation - 1)

        decay_rep = np.repeat(decay, self.config.mu)[: generation * self.config.mu]

        w = np.tile(np.sqrt(self.config.weights), window_size)[: len(decay_rep)]

        r_mu = self.rng.standard_normal((len(decay_rep), self.config.population_size))

        relevant_d_size = min(generation * self.config.mu, self.d_history.shape[1])
        d_relevant = self.d_history[:, :relevant_d_size]

        weighted_d = d_relevant * (decay_rep[:relevant_d_size] * w[:relevant_d_size])
        rank_mu = np.sqrt(self.config.c_mu) * (weighted_d @ r_mu[:relevant_d_size, :])

        r_1 = self.rng.standard_normal((window_size, self.config.population_size))
        p_relevant = self.p_history[:, :window_size]
        rank_1 = np.sqrt(self.config.c_1) * (
            p_relevant @ (r_1 * decay[:window_size, np.newaxis])
        )

        if generation <= self.config.window:
            last_decay = self.decay_table[generation - 1]
        else:
            last_decay = np.sqrt(
                (1 - self.config.c_cov) ** (generation - 1)
                + ((1 - self.config.c_cov) ** self.config.window)
                * (1 - self.config.c_cov) ** (generation - self.config.window - 1)
            )

        r_last = self.rng.standard_normal((self.config.dimensions, self.config.population_size))
        last_term = last_decay * r_last

        # Combine all terms to get difference vectors
        d = rank_mu + rank_1 + last_term

        return d

    def optimize(self) -> OptimizationResult["MFCMAESLogData"]:
        """Run the MF-CMA-ES optimization algorithm."""

        self.evaluations = 0
        best_fitness = float("inf")
        best_solution = self.initial_point.copy()
        worst_fitness = float("inf")
        message = None
        generation = 0

        self.midpoint_fitness = np.inf
        self.prev_midpoint_fitness = np.inf

        initial_pop = self.population_initializer.generate_population(
            rng=self.rng,
            x0=self.mean,
            pop_size=self.config.population_size,
            lower_bounds=self.lower_bounds,
            upper_bounds=self.upper_bounds,
        )
        initial_arx = initial_pop.T  # (dim, pop_size)
        initial_d = (initial_arx - self.mean[:, np.newaxis]) / self.sigma
        initial_vx = self.repair_strategy.repair_population(initial_arx.T, self.constraint_handler).T

        initial_fitness = np.array(
            [self.evaluate(initial_vx[:, i]) for i in range(initial_vx.shape[1])]
        )

        arindex = np.argsort(initial_fitness)
        aripop = arindex[: self.config.mu]
        initial_seld = initial_d[:, aripop]

        d_start, d_end = self._d_range(1)
        self.d_history[:, d_start:d_end] = initial_seld

        dmean = initial_seld @ self.config.weights
        self.mean = self.mean + dmean
        self.pc = (
            np.sqrt(self.config.cc * (2 - self.config.cc) * self.config.mu_eff) * dmean
        )

        p_idx = self._p_index(1)
        self.p_history[:, p_idx] = self.pc

        best_idx = np.argmin(initial_fitness)
        if initial_fitness[best_idx] < best_fitness:
            best_fitness = initial_fitness[best_idx]
            best_solution = initial_vx[:, best_idx].copy()

        self._update_sigma_ppmf_first(initial_vx, initial_fitness)

        generation = 1
        while self.evaluations < self.config.budget:
            generation += 1

            d = self._generate_population(generation)
            arx = self.mean[:, np.newaxis] + self.sigma * d

            vx = self.repair_strategy.repair_population(arx.T, self.constraint_handler).T

            self.constraint_violations = int(np.sum(np.any(vx != arx, axis=0)))

            fitness_values = np.array([self.evaluate(vx[:, i]) for i in range(vx.shape[1])])

            best_idx = np.argmin(fitness_values)
            if fitness_values[best_idx] < best_fitness:
                best_fitness = fitness_values[best_idx]
                best_solution = vx[:, best_idx].copy()

            worst_fitness = max(worst_fitness, float(np.max(fitness_values)))

            arindex = np.argsort(fitness_values)
            aripop = arindex[: self.config.mu]
            seld = d[:, aripop]

            dmean = seld @ self.config.weights
            self.mean = self.mean + dmean

            self.pc = (1 - self.config.cc) * self.pc + np.sqrt(
                self.config.cc * (2 - self.config.cc) * self.config.mu_eff
            ) * dmean

            d_start, d_end = self._d_range(generation)
            self.d_history[:, d_start:d_end] = seld

            p_idx = self._p_index(generation)
            self.p_history[:, p_idx] = self.pc

            self._update_sigma_ppmf(vx, fitness_values)

            self.logger.log_iteration(
                iteration=generation,
                evaluations=self.evaluations,
                best_fitness=best_fitness,
                worst_fitness=worst_fitness,
                mean_fitness=float(np.mean(fitness_values)),
                sigma=self.sigma,
                p_succ=self.p_succ,
                midpoint_fitness=self.midpoint_fitness,
                constraint_violations=self.constraint_violations,
                fitness=fitness_values,
                population=vx.T if self.config.diag_pop else None,
                best_solution=best_solution,
                pc=self.pc,
                mean_vector=self.mean,
            )

            if fitness_values[0] <= self.config.tolfun:
                message = "Target fitness reached."
                break

            if self.evaluations >= self.config.budget:
                message = "Maximum function evaluations reached."
                break

            if self.sigma > self.config.tolxup:
                message = f"Step size diverged (too large): sigma={self.sigma:.2e}"
                break

            if self.sigma < self.config.tolx:
                message = f"Step size too small: sigma={self.sigma:.2e}"
                break

        if message is None:
            message = "Maximum function evaluations reached."

        result: OptimizationResult["MFCMAESLogData"] = OptimizationResult(
            best_solution=best_solution,
            best_fitness=best_fitness,
            evaluations=self.evaluations,
            message=message,
            diagnostic=self.get_logs(),
            algorithm=AlgorithmChoice.MFCMAES,
        )

        return result

    def _update_sigma_ppmf_first(
        self, vx: NDArray[np.float64], fitness_values: NDArray[np.float64]
    ) -> None:
        """
        First call to PPMF - just initialize midpoint_fitness.
        Don't update sigma yet.
        """
        if not self.config.use_ppmf:
            self.midpoint_fitness = np.inf
            self.prev_midpoint_fitness = np.inf
            self.p_succ = 0.0
            return

        self.prev_midpoint_fitness = self.midpoint_fitness

        population_midpoint = np.mean(vx, axis=1)
        self.midpoint_fitness = self.evaluate(population_midpoint)

        num_successes = np.sum(fitness_values < self.prev_midpoint_fitness)
        self.p_succ = num_successes / self.config.population_size

    def _update_sigma_ppmf(
        self, vx: NDArray[np.float64], fitness_values: NDArray[np.float64]
    ) -> None:
        """
        Update step size using PPMF rule.
        This matches the R code exactly.
        If use_ppmf is False, sigma remains constant.
        """
        if not self.config.use_ppmf:
            # Keep sigma constant, just calculate p_succ for logging
            population_midpoint = np.mean(vx, axis=1)
            self.midpoint_fitness = self.evaluate(population_midpoint)

            num_successes = np.sum(fitness_values < self.prev_midpoint_fitness)
            self.p_succ = num_successes / self.config.population_size

            self.prev_midpoint_fitness = self.midpoint_fitness
            return

        self.prev_midpoint_fitness = self.midpoint_fitness

        population_midpoint = np.mean(vx, axis=1)

        self.midpoint_fitness = self.evaluate(population_midpoint)

        num_successes = np.sum(fitness_values < self.prev_midpoint_fitness)
        self.p_succ = num_successes / self.config.population_size

        self.sigma = self.sigma * np.exp(
            (1.0 / self.config.damps_ppmf)
            * (self.p_succ - self.config.p_target_ppmf)
            / (1.0 - self.config.p_target_ppmf)
        )
