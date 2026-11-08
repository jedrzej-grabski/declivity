"""Framework-native CMA-ES optimizer.

Mean, sigma, covariance, evolution paths and eigendecomposition all live on
the optimizer instance and step forward inside ``optimize()`` using the
framework's primitives:

* :class:`~declivity.utils.constraint_handlers.ConstraintHandler` —
  feasibility test and per-point repair.
* :class:`~declivity.utils.repair_strategies.RepairStrategy` —
  population-level repair applied to every generation's λ candidates.
* :class:`~declivity.utils.population_initializers.PopulationInitializer` —
  seeds the iteration-0 population from ``N(m, σ²I)``.
* ``BaseOptimizer.evaluate`` plus a caller-owned ``rng``.

``cmaes_reference.CMA`` is retained as the oracle in
``experiments/cross_validation/cmaes_vs_reference.py``.  See
``docs/cmaes_framework_integration.md``.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Union, final

import numpy as np
from numpy.typing import NDArray

from declivity.algorithms.choices import AlgorithmChoice
from declivity.algorithms.cmaes.config import CMAESConfig
from declivity.core.algorithm_factory import register_optimizer
from declivity.core.base_optimizer import BaseOptimizer, OptimizationResult
from declivity.core.population_optimizer import PopulationOptimizer
from declivity.utils.constraint_handlers import (
    BoxConstraintHandler,
    BoxStrategy,
    ConstraintHandler,
)
from declivity.utils.initial_geometry import covariance_invertible_for_handoff
from declivity.utils.population_initializers import (
    MeanSigmaPopulationInitializer,
    PopulationInitializer,
)
from declivity.utils.repair_strategies import LamarckianRepair, RepairStrategy
from declivity.utils.stopping_conditions import StoppingCondition

if TYPE_CHECKING:
    from declivity.logging.cmaes_logger import CMAESLogData
    from declivity.utils.covariance import CovarianceMatrix


_EPS = 1e-8
_MEAN_MAX = 1e32
_SIGMA_MAX = 1e32


@dataclass(frozen=True)
class CMAESState:
    """Immutable snapshot of a CMA-ES optimizer's evolvable state.

    Holds the sampling distribution (``mean``, ``sigma``, ``covariance``),
    both evolution paths, the generation counter, and the function-value
    history that drives the ``tolfun`` termination test.

    Obtain one with :meth:`CMAESOptimizer.get_state`; restart from it with
    ``CMAESOptimizer(..., initial_state=state)``.  Resuming with the same RNG
    (pass the same ``np.random.Generator`` as ``seed``) reproduces a single
    continuous run bit-for-bit.

    Config-derived constants (weights, learning rates, ...) are not part of
    the state; they are recomputed from the dimension and population size,
    which must match between the snapshot and the optimizer restoring it.
    """

    mean: NDArray[np.float64]
    sigma: float
    covariance: NDArray[np.float64]
    evolution_path_c: NDArray[np.float64]
    evolution_path_sigma: NDArray[np.float64]
    generation: int
    funhist_values: NDArray[np.float64]

    # Cached eigendecomposition of ``covariance`` (``C = B diag(D**2) B.T``),
    # so a resumed run reuses the same ``(B, D)`` rather than re-running
    # ``eigh``, which differs in the last bits.  ``None`` before the first
    # decomposition.
    eigenvectors: NDArray[np.float64] | None = None
    eigenvalues_sqrt: NDArray[np.float64] | None = None


@final
@register_optimizer(AlgorithmChoice.CMAES, CMAESConfig)
class CMAESOptimizer(PopulationOptimizer["CMAESLogData", CMAESConfig]):
    """Hansen-style active CMA-ES.

    Constraint handling is delegated to the injected
    :class:`~declivity.utils.repair_strategies.RepairStrategy` (default
    :class:`~declivity.utils.repair_strategies.LamarckianRepair`).  The
    iteration-0 population comes from the injected
    :class:`~declivity.utils.population_initializers.PopulationInitializer`
    (default
    :class:`~declivity.utils.population_initializers.MeanSigmaPopulationInitializer`
    with the resolved initial sigma, the canonical ``N(m, σ²I)`` start).
    """

    def __init__(
        self,
        func: Callable[[NDArray[np.float64]], float],
        initial_point: NDArray[np.float64],
        config: CMAESConfig | None = None,
        repair_strategy: RepairStrategy | None = None,
        population_initializer: PopulationInitializer | None = None,
        constraint_handler: ConstraintHandler | None = None,
        stopping_condition: StoppingCondition | None = None,
        lower_bounds: Union[float, NDArray[np.float64], list[float]] = -100.0,
        upper_bounds: Union[float, NDArray[np.float64], list[float]] = 100.0,
        seed: int | np.random.Generator | None = None,
        initial_state: CMAESState | None = None,
    ) -> None:
        if config is None:
            config = CMAESConfig(dimensions=len(initial_point))

        # The default sigma is derived from the search range, so the handler
        # must exist before ``super().__init__`` rather than being built
        # inside it.
        if constraint_handler is None:
            constraint_handler = BoxConstraintHandler(
                BoxStrategy.CLAMP,
                BaseOptimizer._process_bounds(lower_bounds, len(initial_point)),
                BaseOptimizer._process_bounds(upper_bounds, len(initial_point)),
            )

        # Resolve auto-sigma (config.sigma == 0.0) up-front so the default
        # population_initializer can be constructed with the final value.  The
        # caller's config is not mutated.
        initial_sigma = float(config.sigma)
        if initial_sigma == 0.0:
            lb_array, ub_array = constraint_handler.bounding_box(len(initial_point))
            span = ub_array - lb_array
            # An unbounded region gives an infinite span, so derive the scale
            # from the finite dimensions when there are any, else from the
            # magnitude of the starting point.
            finite_span = span[np.isfinite(span)]
            if finite_span.size > 0:
                initial_sigma = float(np.mean(finite_span) / 5.0)
            else:
                initial_sigma = max(1.0, float(np.max(np.abs(initial_point))) / 5.0)

        super().__init__(
            func=func,
            initial_point=initial_point,
            config=config,
            repair_strategy=repair_strategy or LamarckianRepair(),
            population_initializer=(
                population_initializer
                or MeanSigmaPopulationInitializer(sigma=initial_sigma)
            ),
            algorithm=AlgorithmChoice.CMAES,
            constraint_handler=constraint_handler,
            stopping_condition=stopping_condition,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            seed=seed,
        )

        n = self.dimensions
        self._mean: NDArray[np.float64] = self.initial_point.copy()
        self._initial_sigma: float = initial_sigma
        self._sigma: float = initial_sigma
        # config.tolx is derived from config.sigma, so with auto-sigma
        # (config.sigma == 0.0 → config.tolx == 0.0) re-derive it here.
        self._tolx: float = (
            self.config.tolx if self.config.sigma != 0.0 else 1e-12 * initial_sigma
        )
        self._C: NDArray[np.float64] = np.eye(n)
        self._pc: NDArray[np.float64] = np.zeros(n)
        self._p_sigma: NDArray[np.float64] = np.zeros(n)
        self._B: NDArray[np.float64] | None = None
        self._D: NDArray[np.float64] | None = None

        self._generation = 0
        self._funhist_values = np.full(self.config.funhist_term * 2, np.inf)

        # With a restored state, generation-0 sampling honours the restored
        # sigma/covariance instead of the PopulationInitializer.
        self._restored_from_state = initial_state is not None
        if initial_state is not None:
            self._apply_state(initial_state)

    # Public state, used by the handoff runners and for pausing / resuming.

    def get_state(self) -> CMAESState:
        """Snapshot the evolvable state for a later :class:`CMAESState` resume."""
        return CMAESState(
            mean=self._mean.copy(),
            sigma=float(self._sigma),
            covariance=self._C.copy(),
            evolution_path_c=self._pc.copy(),
            evolution_path_sigma=self._p_sigma.copy(),
            generation=int(self._generation),
            funhist_values=self._funhist_values.copy(),
            eigenvectors=None if self._B is None else self._B.copy(),
            eigenvalues_sqrt=None if self._D is None else self._D.copy(),
        )

    def _apply_state(self, state: CMAESState) -> None:
        """Restore a snapshot produced by :meth:`get_state` onto this instance."""
        if state.mean.shape != (self.dimensions,):
            raise ValueError(
                f"initial_state.mean has shape {state.mean.shape}, expected "
                f"({self.dimensions},)."
            )
        if state.covariance.shape != (self.dimensions, self.dimensions):
            raise ValueError(
                f"initial_state.covariance has shape {state.covariance.shape}, "
                f"expected ({self.dimensions}, {self.dimensions})."
            )
        self._mean = state.mean.astype(float, copy=True)
        self._sigma = float(state.sigma)
        self._C = state.covariance.astype(float, copy=True)
        self._pc = state.evolution_path_c.astype(float, copy=True)
        self._p_sigma = state.evolution_path_sigma.astype(float, copy=True)
        self._generation = int(state.generation)
        self._funhist_values = state.funhist_values.astype(float, copy=True)
        # Restore the cached eigendecomposition when present; otherwise it is
        # rebuilt lazily from C.
        self._B = (
            None
            if state.eigenvectors is None
            else state.eigenvectors.astype(float, copy=True)
        )
        self._D = (
            None
            if state.eigenvalues_sqrt is None
            else state.eigenvalues_sqrt.astype(float, copy=True)
        )

    @property
    def sigma(self) -> float:
        """Current step-size σ."""
        return self._sigma

    @property
    def mean(self) -> NDArray[np.float64]:
        """Current distribution mean (defensive copy)."""
        return self._mean.copy()

    def get_eigendecomposition(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return ``(B, D)`` with ``C = B @ diag(D**2) @ B.T``."""
        B, D = self._eigen_decomposition()
        return B.copy(), D.copy()

    def get_learned_covariance(self) -> "CovarianceMatrix":
        """Return the current covariance wrapped in a :class:`CovarianceMatrix`."""
        from declivity.utils.covariance import _decompose

        C = self._C.copy()
        return _decompose(C, self._mean.copy(), min(C.shape[0], C.shape[1]))

    # Core algorithm.

    def optimize(self) -> OptimizationResult["CMAESLogData"]:
        self.evaluations = 0
        self._begin_run()
        best_fitness = float("inf")
        best_solution = self._mean.copy()
        message: str | None = None

        lambda_ = self.config.population_size

        while not self.should_stop(self._generation, best_fitness):
            # Generate the λ-candidate population.
            population = self._generate_population(lambda_)
            population = self.repair_strategy.repair_population(
                population, self.constraint_handler
            )

            # Divergence guard: a distribution past the representable range
            # cannot recover, so terminate instead of raising.
            if not np.all(np.abs(population) < _MEAN_MAX):
                message = "Diverged: sampled parameters exceeded the safe range."
                break

            # Evaluate.
            fitness_values = np.empty(lambda_)
            for k in range(lambda_):
                fitness_values[k] = self.evaluate(population[k])
                if fitness_values[k] < best_fitness:
                    best_fitness = float(fitness_values[k])
                    best_solution = population[k].copy()

            # Bookkeeping for the iteration log.
            worst_fitness = float(np.max(fitness_values))
            median_fitness = float(np.median(fitness_values))
            repaired_mean = self.constraint_handler.repair(self._mean)
            mean_fitness_value = self.evaluate(repaired_mean)
            if mean_fitness_value < best_fitness:
                best_fitness = float(mean_fitness_value)
                best_solution = repaired_mean.copy()

            # Update distribution.
            self._tell(population, fitness_values)

            # Logging.
            B, D = self._eigen_decomposition()
            eigenvalues_sorted = np.sort(D**2) if self.config.diag_eigen else None

            self.logger.log_iteration(
                iteration=self._generation,
                evaluations=self.evaluations,
                sigma=self._sigma,
                fitness=fitness_values,
                population=population if self.config.diag_pop else None,
                best_fitness=best_fitness,
                worst_fitness=worst_fitness,
                best_solution=best_solution,
                mean_fitness=mean_fitness_value,
                median_fitness=median_fitness,
                pc=self._pc,
                ps=self._p_sigma,
                mean_vector=self._mean,
                eigenvalues=eigenvalues_sorted,
                covariance_matrix=self._C if self.config.diag_eigen else None,
            )

            stop_reason = self._termination_reason()
            if stop_reason is not None:
                message = stop_reason
                break

        if message is None:
            message = self.stop_message

        return OptimizationResult(
            best_solution=best_solution,
            best_fitness=best_fitness,
            evaluations=self.evaluations,
            message=message,
            diagnostic=self.get_logs(),
            algorithm=AlgorithmChoice.CMAES,
        )

    # Sampling.

    def _generate_population(self, lambda_: int) -> NDArray[np.float64]:
        """Produce the λ candidates for the current generation.

        Iteration 0 of a fresh run routes through the injected
        :class:`PopulationInitializer`.  Later iterations, and every iteration
        of a run restored from a :class:`CMAESState`, sample from ``N(m, σ²C)``
        using the current eigendecomposition.
        """
        if self._generation == 0 and not self._restored_from_state:
            return self.population_initializer.generate_population(
                rng=self.rng,
                x0=self._mean,
                pop_size=lambda_,
                constraint_handler=self.constraint_handler,
            )

        population = np.empty((lambda_, self.dimensions))
        for k in range(lambda_):
            population[k] = self._sample_solution()
        return population

    def _eigen_decomposition(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return cached ``(B, D)`` with ``C = B @ diag(D**2) @ B.T``."""
        if self._B is not None and self._D is not None:
            return self._B, self._D

        self._C = (self._C + self._C.T) / 2.0
        D2, B = np.linalg.eigh(self._C)
        D = np.sqrt(np.where(D2 < 0, _EPS, D2))
        # Re-symmetrise from the eigendecomposition to suppress drift.
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            self._C = np.dot(np.dot(B, np.diag(D**2)), B.T)
        self._B, self._D = B, D
        return B, D

    def _sample_solution(self) -> NDArray[np.float64]:
        B, D = self._eigen_decomposition()
        z = self.rng.standard_normal(self.dimensions)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            y = np.dot(np.dot(B, np.diag(D)), z)
        return self._mean + self._sigma * y  # ~ N(m, σ² C)

    def _tell(
        self,
        population: NDArray[np.float64],
        fitness_values: NDArray[np.float64],
    ) -> None:
        assert population.shape == (self.config.population_size, self.dimensions)

        self._generation += 1

        # Stable sort matches the reference's tie-breaking.
        order = np.argsort(fitness_values, kind="stable")
        sorted_pop = population[order]
        sorted_fit = fitness_values[order]

        # Function-value history (used by tolfun termination).
        funhist_idx = 2 * (self._generation % self.config.funhist_term)
        self._funhist_values[funhist_idx] = sorted_fit[0]
        self._funhist_values[funhist_idx + 1] = sorted_fit[-1]

        # Eigendecomposition prior to the C update.
        B, D = self._eigen_decomposition()
        self._B, self._D = None, None  # C is about to change

        weights = self.config.weights
        mu = self.config.mu
        mu_eff = self.config.mu_eff
        cc = self.config.cc
        c1 = self.config.c1
        cmu = self.config.cmu
        c_sigma = self.config.c_sigma
        d_sigma = self.config.d_sigma
        chi_n = self.config.chi_n
        n = self.dimensions

        y_k = (sorted_pop - self._mean) / self._sigma  # ~ N(0, C)
        y_w = (y_k[:mu].T * weights[:mu]).sum(axis=1)  # eq. 41

        # Mean update (eq. 42 with cm).
        self._mean = self._mean + self.config.cm * self._sigma * y_w

        # C^(-1/2) = B · diag(1/D) · Bᵀ.
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            C_invsqrt = np.dot(np.dot(B, np.diag(1.0 / D)), B.T)

            # Step-size evolution path (eq. 43) and σ update.
            self._p_sigma = (1.0 - c_sigma) * self._p_sigma + math.sqrt(
                c_sigma * (2.0 - c_sigma) * mu_eff
            ) * C_invsqrt.dot(y_w)

        norm_p_sigma = float(np.linalg.norm(self._p_sigma))
        self._sigma = min(
            self._sigma * math.exp((c_sigma / d_sigma) * (norm_p_sigma / chi_n - 1.0)),
            _SIGMA_MAX,
        )

        # Heaviside h_sigma for the rank-one path (Hansen 2016, p. 28).
        h_sigma_left = norm_p_sigma / math.sqrt(
            1.0 - (1.0 - c_sigma) ** (2 * (self._generation + 1))
        )
        h_sigma_right = (1.4 + 2.0 / (n + 1.0)) * chi_n
        h_sigma = 1.0 if h_sigma_left < h_sigma_right else 0.0

        # Rank-one path (eq. 45).
        self._pc = (1.0 - cc) * self._pc + h_sigma * math.sqrt(
            cc * (2.0 - cc) * mu_eff
        ) * y_w

        # Active-CMA weight rescaling for negative weights (eq. 46).
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            norms_sq = np.linalg.norm(C_invsqrt.dot(y_k.T), axis=0) ** 2
        w_io = weights * np.where(weights >= 0.0, 1.0, n / (norms_sq + _EPS))

        delta_h_sigma = (1.0 - h_sigma) * cc * (2.0 - cc)
        assert delta_h_sigma <= 1.0

        rank_one = np.outer(self._pc, self._pc)
        rank_mu = np.sum(
            np.array([w * np.outer(y, y) for w, y in zip(w_io, y_k)]), axis=0
        )

        self._C = (
            (1.0 + c1 * delta_h_sigma - c1 - cmu * float(np.sum(weights))) * self._C
            + c1 * rank_one
            + cmu * rank_mu
        )

    # Termination.

    def _termination_reason(self) -> str | None:
        """Return a CMA-ES-internal convergence message (``tolfun`` / ``tolx``
        / conditioning), or ``None``.  The evaluation / time / target budget
        is the injected :class:`StoppingCondition`'s job."""
        B, D = self._eigen_decomposition()
        dC = np.diag(self._C)
        sigma = self._sigma
        cfg = self.config

        # Checked first: local-optimizer handoffs invert the covariance into
        # a dense matrix (covariance_to_hessian_matrix) and Cholesky-factor
        # it, which can lose enough precision to come back non-PD once the
        # covariance has degenerated. Rather than guess a condition-number
        # threshold, attempt that exact reconstruction here and stop if it
        # actually fails, before any looser heuristic below gets a chance to
        # fire first on a generation that's already past this point. A non-PD
        # matrix supplied up front is still a real error (InitialGeometry's
        # constructor keeps raising for that); this is only about a
        # covariance going bad mid-run.
        if not covariance_invertible_for_handoff(B, D):
            warnings.warn(
                "CMA-ES covariance degenerated: no longer invertible to a "
                "positive-definite matrix, which a local-optimizer handoff "
                "would need. Terminating instead of failing downstream.",
                stacklevel=2,
            )
            return (
                "Covariance no longer invertible to a positive-definite handoff matrix."
            )

        if (
            self._generation > cfg.funhist_term
            and np.max(self._funhist_values) - np.min(self._funhist_values) < cfg.tolfun
        ):
            return "Function value range below tolerance."

        if np.all(sigma * dC < self._tolx) and np.all(sigma * self._pc < self._tolx):
            return "All standard deviations smaller than tolerance."

        if sigma * float(np.max(D)) > cfg.tolxup:
            return "Step size diverged (too large)."

        if np.any(self._mean == self._mean + (0.2 * sigma * np.sqrt(dC))):
            return "No effect in coordinate update."

        i = self._generation % self.dimensions
        if np.all(self._mean == self._mean + (0.1 * sigma * D[i] * B[:, i])):
            return "No effect in axis update."

        # tolconditioncov bounds the covariance's own eigenvalue condition
        # number, which is (max(D)/min(D))**2 since D holds sqrt-eigenvalues
        # (pycma: condition_number property, evolution_strategy.py:3582-3593).
        # Comparing the unsquared ratio against tolconditioncov directly, as
        # this used to, silently raised the real cutoff to tolconditioncov**2.
        if (float(np.max(D)) / float(np.min(D))) ** 2 > cfg.tolconditioncov:
            return "Condition number of covariance matrix exceeded tolerance."

        return None
