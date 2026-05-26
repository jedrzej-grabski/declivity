"""Algorithm specifications for benchmarking.

Each :class:`AlgorithmRun` knows how to run itself on a :class:`Problem`
given an ``x0`` and a seed, and reports a :class:`RunTrace`. The
:class:`AlgorithmRun` protocol is the only contract the framework needs;
the rest is convenience.

Pre-built runners:

- :class:`SingleAlgorithm`     — any one optimizer registered with
                                 :class:`~src.core.algorithm_factory.AlgorithmFactory`.
- :class:`CMAESLBFGSBHandoff`  — CMA-ES warm-up + L-BFGS-B refinement
                                 with a covariance-derived initial Hessian.

Base class for custom two-phase handoffs:

- :class:`HandoffAlgorithm`    — implement :py:meth:`HandoffAlgorithm.run_phases`
                                 and the base class handles trace
                                 stitching, fitness clamping, and
                                 handoff metadata.

For anything more elaborate (N-phase pipelines, restarts, conditional
branches), implement :class:`AlgorithmRun` directly — it's just three
attributes.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from typing import Callable, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from src.algorithms.choices import AlgorithmChoice
from src.algorithms.cmaes.config import CMAESConfig
from src.algorithms.lbfgsb.config import LBFGSBConfig
from src.benchmarking.problem import Problem
from src.benchmarking.run_trace import RunTrace
from src.core.algorithm_factory import AlgorithmFactory
from src.core.base_optimizer import OptimizationResult
from src.core.config_base import BaseConfig


_EIGENVALUE_FLOOR = 1e-30


class HandoffTransform(StrEnum):
    """How to turn the CMA-ES covariance into an L-BFGS-B initial Hessian.

    Used by :class:`CMAESLBFGSBHandoff`.
    """

    INVERSE = "inverse"
    """Use ``C^{-1}`` directly. The L-BFGS-B model becomes a true quadratic
    approximation of the CMA-ES posterior around the warm-up mean."""

    SIGMA_INVERSE = "sigma_inverse"
    """Use ``(sigma^2 C)^{-1}`` — accounts for the CMA-ES global step-size
    scaling. Sometimes more conservative than the bare inverse."""

    IDENTITY = "identity"
    """Drop the covariance and use the L-BFGS-B default (B_0 = I). Mainly a
    control experiment: isolates the value of *passing covariance information*
    from the value of *sharing a starting point* with CMA-ES."""


@runtime_checkable
class AlgorithmRun(Protocol):
    """An algorithm spec: name, color, and a runner.

    Anything matching this shape — a frozen dataclass, a plain class, even
    a SimpleNamespace — slots into :class:`~src.benchmarking.Benchmark`
    and the plotter. The framework never looks for more than these three
    attributes.
    """

    name: str
    color: str

    def run(
        self,
        problem: Problem,
        x0: NDArray[np.float64],
        seed: int,
    ) -> RunTrace: ...


def _enable_diagnostics(config: BaseConfig) -> None:
    """Make sure best-fitness logging is on (it usually is by default)."""
    if hasattr(config, "diag_bestVal"):
        config.diag_bestVal = True


@dataclass
class SingleAlgorithm:
    """A single optimizer registered in the :class:`AlgorithmFactory`."""

    name: str
    color: str
    algorithm: AlgorithmChoice
    config_factory: Callable[[int], BaseConfig]
    """Builds a fresh config given the problem's dimensions."""

    extra_diagnostics: tuple[str, ...] = ()
    """Names of additional diag_* flags to enable on the config."""

    def run(
        self,
        problem: Problem,
        x0: NDArray[np.float64],
        seed: int,
    ) -> RunTrace:
        config = self.config_factory(problem.dimensions)
        _enable_diagnostics(config)
        for flag in self.extra_diagnostics:
            if hasattr(config, flag):
                setattr(config, flag, True)

        kwargs: dict = {}
        if problem.gradient is not None and self.algorithm == AlgorithmChoice.LBFGSB:
            kwargs["gradient_fn"] = problem.gradient

        optimizer = AlgorithmFactory.create_optimizer(
            self.algorithm,
            problem.function,
            x0,
            config,
            lower_bounds=problem.lower_bound,
            upper_bounds=problem.upper_bound,
            seed=seed,
            **kwargs,
        )
        result = optimizer.optimize()

        return RunTrace(
            algorithm=self.name,
            problem=problem.name,
            seed=seed,
            evaluations=list(result.diagnostic.evaluations),
            best_fitness=list(result.diagnostic.best_fitness),
            final_evaluations=result.evaluations,
            final_fitness=float(result.best_fitness),
        )


class HandoffAlgorithm(ABC):
    """Base class for two-phase (warm-up -> refinement) handoff algorithms.

    Saves the trace-stitching boilerplate that every handoff would
    otherwise re-implement. Subclasses provide ``name`` + ``color``
    (typically via :py:func:`dataclasses.dataclass`) and override
    :py:meth:`run_phases` to return the two phase results. The base class
    fills in a :class:`RunTrace` with:

    - eval counts in the refinement segment offset by the warm-up's
      total evaluations, so the convergence trace is continuous;
    - the refinement segment's fitness clamped to never exceed the
      warm-up's best (the refinement logger reports its own best, which
      starts at ``f(x0_refinement)`` — that may be slightly worse than
      the warm-up's running best);
    - ``handoff_eval`` and ``handoff_iter`` set from the warm-up totals,
      which the plotter uses to draw vertical handoff markers.

    Minimal usage::

        @dataclass
        class MyHandoff(HandoffAlgorithm):
            name: str
            color: str
            # ... whatever config fields you need ...

            def run_phases(self, problem, x0, seed):
                warmup_result   = ...  # run phase 1
                refinement_result = ...  # run phase 2, using warmup state
                return warmup_result, refinement_result

    Pass state from warm-up to refinement through ordinary local
    variables in :py:meth:`run_phases` — both phases live in the same
    method so any state from one is in scope for the other.

    For more elaborate runners (3+ phases, restarts, conditional
    branches), implement :class:`AlgorithmRun` directly instead.
    """

    name: str
    color: str

    @abstractmethod
    def run_phases(
        self,
        problem: Problem,
        x0: NDArray[np.float64],
        seed: int,
    ) -> tuple[OptimizationResult, OptimizationResult]:
        """Run the two phases. Return ``(warmup_result, refinement_result)``.

        The framework only sees a :class:`RunTrace` — this is the only
        method most custom handoffs need to write.
        """
        ...

    def run(
        self,
        problem: Problem,
        x0: NDArray[np.float64],
        seed: int,
    ) -> RunTrace:
        warmup, refinement = self.run_phases(problem, x0, seed)
        return self._stitch_traces(problem, seed, warmup, refinement)

    def _stitch_traces(
        self,
        problem: Problem,
        seed: int,
        warmup: OptimizationResult,
        refinement: OptimizationResult,
    ) -> RunTrace:
        """Concatenate the two phases into a single continuous trace."""
        warmup_evals = warmup.evaluations
        warmup_iters = len(warmup.diagnostic.iteration)
        warmup_best = float(warmup.best_fitness)

        warmup_eval_list = list(warmup.diagnostic.evaluations)
        warmup_fitness_list = list(warmup.diagnostic.best_fitness)
        # Offset refinement eval counts so the trace is continuous.
        refinement_eval_list = [
            evaluation + warmup_evals
            for evaluation in refinement.diagnostic.evaluations
        ]
        # Clamp refinement fitness so it never reports worse than the
        # warm-up's best — its first point is f(x0_refinement) which can
        # be slightly worse than the warm-up's running minimum.
        refinement_fitness_list = [
            min(value, warmup_best) for value in refinement.diagnostic.best_fitness
        ]

        return RunTrace(
            algorithm=self.name,
            problem=problem.name,
            seed=seed,
            evaluations=warmup_eval_list + refinement_eval_list,
            best_fitness=warmup_fitness_list + refinement_fitness_list,
            final_evaluations=warmup_evals + refinement.evaluations,
            final_fitness=min(warmup_best, float(refinement.best_fitness)),
            handoff_eval=warmup_evals,
            handoff_iter=warmup_iters,
        )


@dataclass
class CMAESLBFGSBHandoff(HandoffAlgorithm):
    """CMA-ES warm-up followed by L-BFGS-B with a covariance-derived ``B_0``.

    Steps:

    1. Run CMA-ES on the problem until ``cmaes_config.budget`` evaluations
       are consumed. Same seed and same ``x0`` as a standalone CMA-ES run
       will produce an identical CMA-ES prefix in the convergence trace.
    2. Read the cached eigendecomposition ``(B, D)`` from CMA-ES.
    3. Compose the initial Hessian for L-BFGS-B according to
       :attr:`transform` (see :class:`HandoffTransform`).
    4. Run L-BFGS-B from the CMA-ES mean with that ``B_0`` and a separate
       budget.

    Trace stitching, fitness clamping, and handoff metadata are inherited
    from :class:`HandoffAlgorithm`; this class only owns the
    CMA-ES-specific covariance transformation.
    """

    name: str
    color: str
    cmaes_config_factory: Callable[[int], CMAESConfig]
    lbfgsb_config_factory: Callable[[int], LBFGSBConfig]
    transform: HandoffTransform | str = HandoffTransform.INVERSE

    cmaes_extra_diagnostics: tuple[str, ...] = ("diag_eigen",)
    lbfgsb_extra_diagnostics: tuple[str, ...] = ()

    def _initial_hessian_from_cmaes(
        self,
        eigenvectors: NDArray[np.float64],
        eigenvalues_sqrt: NDArray[np.float64],
        sigma: float,
    ) -> NDArray[np.float64] | None:
        transform = str(self.transform)
        if transform == HandoffTransform.IDENTITY:
            return None

        eigenvalues = np.maximum(eigenvalues_sqrt**2, _EIGENVALUE_FLOOR)

        # Floored 1/eigenvalues can be huge (up to 1e30); the matmul values
        # are still valid but numpy raises spurious divide/overflow warnings
        # on the intermediates.
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            if transform == HandoffTransform.INVERSE:
                return (eigenvectors * (1.0 / eigenvalues)) @ eigenvectors.T
            if transform == HandoffTransform.SIGMA_INVERSE:
                inverse = (eigenvectors * (1.0 / eigenvalues)) @ eigenvectors.T
                return inverse / (sigma * sigma)

        valid = ", ".join(repr(value.value) for value in HandoffTransform)
        raise ValueError(
            f"Unknown handoff transform: {self.transform!r}. Use one of {valid}."
        )

    def run_phases(
        self,
        problem: Problem,
        x0: NDArray[np.float64],
        seed: int,
    ) -> tuple[OptimizationResult, OptimizationResult]:
        # Phase 1: CMA-ES warm-up.
        cmaes_config = self.cmaes_config_factory(problem.dimensions)
        _enable_diagnostics(cmaes_config)
        for flag in self.cmaes_extra_diagnostics:
            if hasattr(cmaes_config, flag):
                setattr(cmaes_config, flag, True)

        cmaes_optimizer = AlgorithmFactory.create_optimizer(
            AlgorithmChoice.CMAES,
            problem.function,
            x0,
            cmaes_config,
            lower_bounds=problem.lower_bound,
            upper_bounds=problem.upper_bound,
            seed=seed,
        )
        cmaes_result = cmaes_optimizer.optimize()

        # Pull CMA-ES internal state for the handoff: the eigendecomposition
        # of the covariance and the current mean become L-BFGS-B's B_0
        # and starting point respectively.
        eigenvectors, eigenvalues_sqrt = cmaes_optimizer.get_eigendecomposition()
        initial_hessian = self._initial_hessian_from_cmaes(
            eigenvectors, eigenvalues_sqrt, cmaes_optimizer.sigma,
        )

        # Phase 2: L-BFGS-B from the CMA-ES mean with the derived B_0.
        lbfgsb_config = self.lbfgsb_config_factory(problem.dimensions)
        lbfgsb_config.initial_hessian = initial_hessian
        _enable_diagnostics(lbfgsb_config)
        for flag in self.lbfgsb_extra_diagnostics:
            if hasattr(lbfgsb_config, flag):
                setattr(lbfgsb_config, flag, True)

        lbfgsb_kwargs: dict = {}
        if problem.gradient is not None:
            lbfgsb_kwargs["gradient_fn"] = problem.gradient

        lbfgsb_result = AlgorithmFactory.create_optimizer(
            AlgorithmChoice.LBFGSB,
            problem.function,
            cmaes_optimizer.mean,
            lbfgsb_config,
            lower_bounds=problem.lower_bound,
            upper_bounds=problem.upper_bound,
            **lbfgsb_kwargs,
        ).optimize()

        return cmaes_result, lbfgsb_result
