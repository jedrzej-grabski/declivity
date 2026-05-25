"""Algorithm specifications for benchmarking.

Each AlgorithmRun knows how to run itself on a Problem given an x0 and a
seed, and reports a RunTrace. Concrete classes:

- SingleAlgorithm: any registered AlgorithmFactory algorithm.
- CMAESLBFGSBHandoff: warm-up CMA-ES, transform its covariance into B_0
  for L-BFGS-B, then continue from the CMA-ES mean.
"""

from dataclasses import dataclass, field
from typing import Callable, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from src.algorithms.choices import AlgorithmChoice
from src.algorithms.cmaes.config import CMAESConfig
from src.algorithms.lbfgsb.config import LBFGSBConfig
from src.benchmarking.problem import Problem
from src.benchmarking.run_trace import RunTrace
from src.core.algorithm_factory import AlgorithmFactory
from src.core.config_base import BaseConfig


_EIGENVALUE_FLOOR = 1e-30


@runtime_checkable
class AlgorithmRun(Protocol):
    """An algorithm spec: name, color, and a runner."""

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
    """A single optimizer registered in the AlgorithmFactory."""

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


@dataclass
class CMAESLBFGSBHandoff:
    """CMA-ES warm-up followed by L-BFGS-B with a covariance-derived B_0.

    Steps:

    1. Run CMA-ES on the problem until ``cmaes_config.budget`` evaluations
       are consumed. Same seed and same x0 as a standalone CMA-ES run will
       produce an identical CMA-ES prefix in the convergence trace.
    2. Read the cached eigendecomposition (B, D) from CMA-ES.
    3. Compose the initial Hessian for L-BFGS-B according to ``transform``:
       - "inverse" (default) -> C^{-1}
       - "sigma_inverse"      -> (sigma^2 C)^{-1}
       - "identity"           -> no Hessian info (L-BFGS-B default B_0 = I);
                                 isolates the value of just sharing the
                                 starting point with CMA-ES.
    4. Run L-BFGS-B from the CMA-ES mean with that B_0 and a separate
       budget.

    The convergence trace is the concatenation of the two phases. The
    L-BFGS-B segment is clamped so it never reports a fitness worse than
    the CMA-ES handoff value (handles cases where L-BFGS-B's first step
    happens to land slightly worse than the warm-up's best).
    """

    name: str
    color: str
    cmaes_config_factory: Callable[[int], CMAESConfig]
    lbfgsb_config_factory: Callable[[int], LBFGSBConfig]
    transform: str = "inverse"

    cmaes_extra_diagnostics: tuple[str, ...] = ("diag_eigen",)
    lbfgsb_extra_diagnostics: tuple[str, ...] = ()

    def _initial_hessian_from_cmaes(
        self,
        eigenvectors: NDArray[np.float64],
        eigenvalues_sqrt: NDArray[np.float64],
        sigma: float,
    ) -> NDArray[np.float64] | None:
        if self.transform == "identity":
            return None

        eigenvalues = np.maximum(eigenvalues_sqrt**2, _EIGENVALUE_FLOOR)

        # Floored 1/eigenvalues can be huge (up to 1e30); the matmul values
        # are still valid but numpy raises spurious divide/overflow warnings
        # on the intermediates.
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            if self.transform == "inverse":
                return (eigenvectors * (1.0 / eigenvalues)) @ eigenvectors.T
            if self.transform == "sigma_inverse":
                inverse = (eigenvectors * (1.0 / eigenvalues)) @ eigenvectors.T
                return inverse / (sigma * sigma)

        raise ValueError(
            f"Unknown handoff transform: {self.transform!r}. "
            f"Use 'inverse', 'sigma_inverse', or 'identity'."
        )

    def run(
        self,
        problem: Problem,
        x0: NDArray[np.float64],
        seed: int,
    ) -> RunTrace:
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
        warmup_evals = cmaes_result.evaluations
        warmup_iters = len(cmaes_result.diagnostic.iteration)

        eigenvectors, eigenvalues_sqrt = cmaes_optimizer.get_eigendecomposition()
        sigma = cmaes_optimizer.sigma
        starting_point = cmaes_optimizer.mean
        cmaes_best = float(cmaes_result.best_fitness)

        # Phase 2: L-BFGS-B with a covariance-derived B_0.
        initial_hessian = self._initial_hessian_from_cmaes(
            eigenvectors, eigenvalues_sqrt, sigma
        )

        lbfgsb_config = self.lbfgsb_config_factory(problem.dimensions)
        lbfgsb_config.initial_hessian = initial_hessian
        _enable_diagnostics(lbfgsb_config)
        for flag in self.lbfgsb_extra_diagnostics:
            if hasattr(lbfgsb_config, flag):
                setattr(lbfgsb_config, flag, True)

        kwargs: dict = {}
        if problem.gradient is not None:
            kwargs["gradient_fn"] = problem.gradient

        lbfgsb_optimizer = AlgorithmFactory.create_optimizer(
            AlgorithmChoice.LBFGSB,
            problem.function,
            starting_point,
            lbfgsb_config,
            lower_bounds=problem.lower_bound,
            upper_bounds=problem.upper_bound,
            **kwargs,
        )
        lbfgsb_result = lbfgsb_optimizer.optimize()

        # Concatenate traces. Offset L-BFGS-B evals so the trace is continuous,
        # and clamp its fitness to never exceed the handoff value (the L-BFGS-B
        # logger reports its own best, which starts at f(x0_lbfgsb), not f_best).
        cmaes_evals = list(cmaes_result.diagnostic.evaluations)
        cmaes_fits = list(cmaes_result.diagnostic.best_fitness)
        lbfgsb_evals = [
            evaluation + warmup_evals
            for evaluation in lbfgsb_result.diagnostic.evaluations
        ]
        lbfgsb_fits = [min(value, cmaes_best) for value in lbfgsb_result.diagnostic.best_fitness]

        final_fitness = min(cmaes_best, float(lbfgsb_result.best_fitness))

        return RunTrace(
            algorithm=self.name,
            problem=problem.name,
            seed=seed,
            evaluations=cmaes_evals + lbfgsb_evals,
            best_fitness=cmaes_fits + lbfgsb_fits,
            final_evaluations=warmup_evals + lbfgsb_result.evaluations,
            final_fitness=final_fitness,
            handoff_eval=warmup_evals,
            handoff_iter=warmup_iters,
        )
