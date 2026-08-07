import math
from typing import TYPE_CHECKING, Callable, Union, final

import numpy as np
from numpy.typing import NDArray

from declivity.algorithms.choices import AlgorithmChoice
from declivity.algorithms.mfcmaes.mfcmaes_config import MFCMAESConfig
from declivity.core.algorithm_factory import register_optimizer
from declivity.core.base_optimizer import OptimizationResult
from declivity.core.population_optimizer import PopulationOptimizer
from declivity.utils.constraint_handlers import ConstraintHandler
from declivity.utils.population_initializers import (
    MeanSigmaPopulationInitializer,
    PopulationInitializer,
)
from declivity.utils.repair_strategies import LamarckianRepair, RepairStrategy
from declivity.utils.stopping_conditions import StoppingCondition

if TYPE_CHECKING:
    from declivity.logging.mfcmaes_logger import MFCMAESLogData


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
        stopping_condition: StoppingCondition | None = None,
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
            repair_strategy=repair_strategy or LamarckianRepair(),
            population_initializer=population_initializer
            or MeanSigmaPopulationInitializer(sigma=config.sigma),
            algorithm=AlgorithmChoice.MFCMAES,
            constraint_handler=constraint_handler,
            stopping_condition=stopping_condition,
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
        """Precompute the decay factors (1 - c_cov)^((t-τ)/2) for the archive.

        Sized by the archive ``window`` (not the total budget): the table is
        only ever read through its first ``window`` entries — see
        :meth:`_generate_population`, which reverses the full table
        (``decay_table[::-1]``) — so ``window`` factors suffice
        regardless of how many generations the run lasts."""
        t = np.arange(1, self.config.window + 1)
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
        # Get decay factors for current generation.  Match the reference
        # (R lines 136–138): always reverse the FULL window-length table
        # and cyclically shift by ``generation - 1``; before the archive
        # fills, its zero columns absorb the meaningless decay entries.
        decay = self.decay_table[::-1].copy()  # length = window
        decay = self._shift_array(decay, generation - 1)

        # decay index of archive column c is ``c // mu`` (its generation
        # slot); weight index is ``c % mu`` (its selection rank) — the
        # d_history layout is slot-major (see :meth:`_d_range`), so
        # ``np.tile`` (not the reference's ``np.repeat``, which pairs the
        # weight by ``c // window``) gives each column sqrt(weight of its
        # rank).  Identical numerically under the uniform weights both
        # implementations use.
        decay_rep = np.repeat(decay, self.config.mu)  # length = window * mu
        w = np.tile(np.sqrt(self.config.weights), self.config.window)

        # Late-iter decay-table underflow makes the matmul multiply
        # denormalised numbers — numpy reports spurious "divide by
        # zero" / "invalid" / "overflow" warnings even though the
        # accumulated result is mathematically well-defined.
        with np.errstate(
            divide="ignore", invalid="ignore", over="ignore", under="ignore"
        ):
            r_mu = self.rng.standard_normal(
                (self.config.window * self.config.mu, self.config.population_size)
            )
            rank_mu = np.sqrt(self.config.c_mu) * (
                self.d_history @ (r_mu * decay_rep[:, np.newaxis] * w[:, np.newaxis])
            )

            r_1 = self.rng.standard_normal(
                (self.config.window, self.config.population_size)
            )
            rank_1 = np.sqrt(self.config.c_1) * (
                self.p_history @ (r_1 * decay[:, np.newaxis])
            )

        if generation <= self.config.window:
            last_decay = self.decay_table[generation - 1]
        else:
            last_decay = np.sqrt(
                (1 - self.config.c_cov) ** (generation - 1)
                + ((1 - self.config.c_cov) ** self.config.window)
                * (1 - self.config.c_cov) ** (generation - self.config.window - 1)
            )

        r_last = self.rng.standard_normal(
            (self.config.dimensions, self.config.population_size)
        )
        last_term = last_decay * r_last

        # Combine all terms to get difference vectors
        d = rank_mu + rank_1 + last_term

        return d

    def optimize(self) -> OptimizationResult["MFCMAESLogData"]:
        """Run the MF-CMA-ES optimization algorithm."""

        self.evaluations = 0
        self._begin_run()
        best_fitness = float("inf")
        best_solution = self.initial_point.copy()
        message = None
        generation = 0

        self.midpoint_fitness = np.inf
        self.prev_midpoint_fitness = np.inf

        initial_pop = self.population_initializer.generate_population(
            rng=self.rng,
            x0=self.mean,
            pop_size=self.config.population_size,
            constraint_handler=self.constraint_handler,
        )
        initial_arx = initial_pop.T  # (dim, pop_size)
        initial_d = (initial_arx - self.mean[:, np.newaxis]) / self.sigma
        initial_vx = self.repair_strategy.repair_population(
            initial_arx.T, self.constraint_handler
        ).T

        initial_pen = 1.0 + np.sum((initial_arx - initial_vx) ** 2, axis=0)
        initial_pen = np.where(
            np.isfinite(initial_pen), initial_pen, np.finfo(np.float64).max / 2.0
        )
        self.constraint_violations = int(np.sum(initial_pen > 1.0))
        initial_raw = np.array(
            [self.evaluate(initial_vx[:, i]) for i in range(initial_vx.shape[1])]
        )
        # ``arfitness = raw * pen`` (R lines 163, 179).  Penalty is the
        # quadratic distance from the unclamped sample to its repaired
        # image, which is zero for in-bounds points and grows
        # quadratically with the violation otherwise.
        initial_fitness = initial_raw * initial_pen

        arindex = np.argsort(initial_fitness)
        aripop = arindex[: self.config.mu]
        initial_seld = initial_d[:, aripop]

        d_start, d_end = self._d_range(1)
        self.d_history[:, d_start:d_end] = initial_seld

        # R: ``xmean <- drop(selx %*% weights)`` where ``selx = arx[, aripop]``.
        # ``arx = xmean + sigma * d``, so ``selx @ weights = xmean + sigma * dmean``.
        # The path cumulation update uses the un-scaled ``dmean`` directly.
        dmean = initial_seld @ self.config.weights
        self.mean = self.mean + self.sigma * dmean
        self.pc = (
            np.sqrt(self.config.cc * (2 - self.config.cc) * self.config.mu_eff) * dmean
        )

        p_idx = self._p_index(1)
        self.p_history[:, p_idx] = self.pc

        # Track the best raw-fitness individual among in-bounds samples
        # only — R lines 181–186.  Penalised fitness wins selection, but
        # the reported ``best_fit`` is the un-penalised value among the
        # feasible subset.
        valid_mask = initial_pen <= 1.0
        if np.any(valid_mask):
            valid_raw = initial_raw[valid_mask]
            valid_vx = initial_vx[:, valid_mask]
            wb = int(np.argmin(valid_raw))
            if valid_raw[wb] < best_fitness:
                best_fitness = float(valid_raw[wb])
                best_solution = valid_vx[:, wb].copy()

        self._update_sigma_ppmf_first(initial_vx, initial_fitness)

        self.logger.log_iteration(
            iteration=1,
            evaluations=self.evaluations,
            best_fitness=best_fitness,
            worst_fitness=float(np.max(initial_fitness)),
            mean_fitness=float(np.mean(initial_fitness)),
            sigma=self.sigma,
            p_succ=self.p_succ,
            midpoint_fitness=self.midpoint_fitness,
            constraint_violations=self.constraint_violations,
            fitness=initial_fitness,
            population=initial_vx.T if self.config.diag_pop else None,
            best_solution=best_solution,
            pc=self.pc,
            mean_vector=self.mean,
        )

        # Same algorithm-internal ``tolfun`` convergence test as the main
        # loop (see below) — the reference applies it on iteration 1 too.
        if float(initial_fitness[arindex[0]]) <= self.config.tolfun:
            message = "Target fitness reached."

        generation = 1
        while message is None and not self.should_stop(generation, best_fitness):
            generation += 1

            d = self._generate_population(generation)
            arx = self.mean[:, np.newaxis] + self.sigma * d

            vx = self.repair_strategy.repair_population(
                arx.T, self.constraint_handler
            ).T

            pen = 1.0 + np.sum((arx - vx) ** 2, axis=0)
            pen = np.where(np.isfinite(pen), pen, np.finfo(np.float64).max / 2.0)
            self.constraint_violations = int(np.sum(pen > 1.0))

            raw_fitness = np.array(
                [self.evaluate(vx[:, i]) for i in range(vx.shape[1])]
            )
            fitness_values = raw_fitness * pen

            valid_mask = pen <= 1.0
            if np.any(valid_mask):
                valid_raw = raw_fitness[valid_mask]
                valid_vx = vx[:, valid_mask]
                wb = int(np.argmin(valid_raw))
                if valid_raw[wb] < best_fitness:
                    best_fitness = float(valid_raw[wb])
                    best_solution = valid_vx[:, wb].copy()

            arindex = np.argsort(fitness_values)
            aripop = arindex[: self.config.mu]
            seld = d[:, aripop]

            # R: ``xmean <- selx %*% weights`` ≡ ``xmean_old + sigma * dmean``;
            # ``pc`` uses the un-scaled ``dmean`` (see R lines 194–205).
            dmean = seld @ self.config.weights
            self.mean = self.mean + self.sigma * dmean

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
                worst_fitness=float(np.max(fitness_values)),
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

            # Match R: ``terminate.stopfitness`` checks the best penalised
            # fitness from this generation against ``stopfitness``.  The
            # shared evaluation / time / target budget is owned by the
            # injected stopping condition (the main-loop guard above); this
            # is the algorithm-internal ``tolfun`` convergence test only.
            if float(fitness_values[arindex[0]]) <= self.config.tolfun:
                message = "Target fitness reached."
                break

            # R-DES-style "flatland escape": when the best and the
            # ⌊λ/2⌋-th individual tie, the population has collapsed onto
            # a flat patch — bump sigma to escape (R lines 232–238).
            if self.config.do_flatland_escape:
                cmp_idx = min(
                    1 + self.config.population_size // 2,
                    2 + math.ceil(self.config.population_size / 4),
                )
                fitness_sorted = fitness_values[arindex]
                if (
                    cmp_idx <= self.config.population_size
                    and fitness_sorted[0] == fitness_sorted[cmp_idx - 1]
                ):
                    self.sigma = self.sigma * math.exp(
                        0.2 + self.config.cs / self.config.damps
                    )

        if message is None:
            message = self.stop_message

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
        """First PPMF call — matches what R does on iteration 1.

        In R, the very first call to ``sigma_updater(sigma)`` sees
        ``prev_midpoint_fitness = +Inf``, which makes ``p_succ = 1`` and
        bumps sigma by ``exp(damps_ppmf * (1 - p_target) / (1 - p_target))``
        — typically a single multiplicative jump of ``exp(damps_ppmf)``.
        Doing nothing on the first call (the previous framework
        behaviour) leaves sigma a factor of ``exp(damps_ppmf)`` below
        R for the rest of the run.
        """
        if not self.config.use_ppmf:
            self.midpoint_fitness = float("nan")
            self.prev_midpoint_fitness = np.inf
            self.p_succ = float("nan")
            return

        self._update_sigma_ppmf(vx, fitness_values)

    def _update_sigma_ppmf(
        self, vx: NDArray[np.float64], fitness_values: NDArray[np.float64]
    ) -> None:
        """
        Update step size using PPMF rule.
        This matches the R code exactly.
        If use_ppmf is False, sigma remains constant.
        """
        if not self.config.use_ppmf:
            # Keep sigma constant.  The reference's ``sigma_updater =
            # "identity"`` makes no midpoint call, so spend no evaluation
            # here either — log NaN for the midpoint-derived fields to
            # keep the diagnostic series aligned.
            self.midpoint_fitness = float("nan")
            self.p_succ = float("nan")
            return

        self.prev_midpoint_fitness = self.midpoint_fitness

        population_midpoint = np.mean(vx, axis=1)

        self.midpoint_fitness = self.evaluate(population_midpoint)

        num_successes = np.sum(fitness_values < self.prev_midpoint_fitness)
        self.p_succ = num_successes / self.config.population_size

        # PPMF update — matches sigma_updaters.R lines 54–67.  The R
        # source multiplies the exponent by ``damps_ppmf``; an earlier
        # version of this file divided by it, which damped the update
        # by a factor of ``damps_ppmf**2`` per iteration and shifted the
        # whole sigma trajectory.
        self.sigma = self.sigma * np.exp(
            self.config.damps_ppmf
            * (self.p_succ - self.config.p_target_ppmf)
            / (1.0 - self.config.p_target_ppmf)
        )
