"""Algorithm specifications for benchmarking.

The framework only needs three things from anything you put inside a
:class:`~src.benchmarking.Benchmark`:

- ``name``  — appears in legends, summary tables, and persisted traces
- ``color`` — hex string the plotter uses for this algorithm's line
- ``run(problem, x0, seed)`` — returns a :class:`RunTrace`

How you provide them determines which base class to pick:

==============================  ===========================================
Your runner is...               Inherit from
==============================  ===========================================
one optimizer from the          :class:`SingleAlgorithm` (concrete, just
factory, no special wrapping    instantiate it).
warmup -> refinement, two       :class:`HandoffAlgorithm`. Implement
phases that share state via     ``run_phases()``; the base class stitches
local variables                 traces, clamps fitness, and fills in
                                handoff metadata for you.
anything else (3+ phases,       :class:`BenchmarkAlgorithm`. Implement
restarts, wrappers, custom      ``run()``; use ``trace_from_result()`` to
schedules)                      package an :class:`OptimizationResult`
                                into a :class:`RunTrace`.
already a class, can't          conform to the :class:`AlgorithmRun`
inherit                         :class:`Protocol` — just expose ``name``,
                                ``color``, and ``run()``. No inheritance
                                required.
==============================  ===========================================
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
    """Duck-typed interface for anything :class:`~src.benchmarking.Benchmark` consumes.

    For most use cases prefer subclassing :class:`BenchmarkAlgorithm` —
    same three-attribute contract, plus you get the
    :py:meth:`BenchmarkAlgorithm.trace_from_result` helper. The Protocol
    is here for the case where you can't inherit (e.g. you're adapting
    an existing class) and just want to conform structurally.
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


class BenchmarkAlgorithm(ABC):
    """Base class for anything you can plug into a :class:`Benchmark`.

    Subclasses provide:

    - ``name`` and ``color`` (typically as :py:func:`dataclasses.dataclass` fields).
    - :py:meth:`run`, which executes the algorithm on one
      ``(problem, x0, seed)`` triple and returns a :class:`RunTrace`.

    The :py:meth:`trace_from_result` helper packages a single
    :class:`OptimizationResult` into a :class:`RunTrace` for the common
    one-optimizer case. Multi-phase runners with their own stitching
    rules (handoffs, restarts) skip the helper and assemble the trace
    by hand, or use :class:`HandoffAlgorithm` for the standard two-phase
    pattern.

    Minimal subclass::

        @dataclass
        class MyAlgorithm(BenchmarkAlgorithm):
            name: str
            color: str
            # ... whatever config fields you need ...

            def run(self, problem, x0, seed) -> RunTrace:
                result = ...  # run an optimizer
                return self.trace_from_result(problem, seed, result)
    """

    name: str
    color: str

    @abstractmethod
    def run(
        self,
        problem: Problem,
        x0: NDArray[np.float64],
        seed: int,
    ) -> RunTrace:
        """Run the algorithm on one (problem, x0, seed) triple."""
        ...

    def trace_from_result(
        self,
        problem: Problem,
        seed: int,
        result: OptimizationResult,
    ) -> RunTrace:
        """Package a single :class:`OptimizationResult` into a :class:`RunTrace`.

        Handles the common case where one optimizer call produces the
        whole convergence trace. ``result.diagnostic`` is expected to
        carry ``evaluations`` and ``best_fitness`` (every built-in
        LogData does).

        For multi-phase runners with custom stitching rules, build the
        :class:`RunTrace` by hand instead — or use
        :class:`HandoffAlgorithm`, which handles the two-phase case.
        """
        return RunTrace(
            algorithm=self.name,
            problem=problem.name,
            seed=seed,
            evaluations=list(result.diagnostic.evaluations),
            best_fitness=list(result.diagnostic.best_fitness),
            final_evaluations=result.evaluations,
            final_fitness=float(result.best_fitness),
        )


@dataclass
class SingleAlgorithm(BenchmarkAlgorithm):
    """A single optimizer registered in the :class:`AlgorithmFactory`.

    Concrete — instantiate directly with the optimizer choice and a
    config factory; no subclass needed unless you want to override
    :py:meth:`run`.
    """

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

        result = AlgorithmFactory.create_optimizer(
            self.algorithm,
            problem.function,
            x0,
            config,
            lower_bounds=problem.lower_bound,
            upper_bounds=problem.upper_bound,
            seed=seed,
            **kwargs,
        ).optimize()

        return self.trace_from_result(problem, seed, result)


class HandoffAlgorithm(BenchmarkAlgorithm):
    """Two-phase (warm-up -> refinement) handoff base class.

    Saves the trace-stitching boilerplate every handoff would otherwise
    re-implement. Subclasses provide ``name`` + ``color`` (typically via
    :py:func:`dataclasses.dataclass`) and override :py:meth:`run_phases`
    to return the two phase results. The base class fills in a
    :class:`RunTrace` with:

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
    branches), subclass :class:`BenchmarkAlgorithm` directly instead.
    """

    @abstractmethod
    def run_phases(
        self,
        problem: Problem,
        x0: NDArray[np.float64],
        seed: int,
    ) -> tuple[OptimizationResult, OptimizationResult]:
        """Run the two phases. Return ``(warmup_result, refinement_result)``."""
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
