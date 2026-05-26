"""Building a custom handoff algorithm from scratch.

This file demonstrates how to plug a brand-new algorithm into the
benchmarking framework when the pre-built :class:`CMAESLBFGSBHandoff`
doesn't fit. We build a DES -> L-BFGS-B handoff (no such class ships)
that:

1. Runs DES for ``warmup_budget`` evaluations to find a basin.
2. Hands the best-so-far point to L-BFGS-B for local refinement.

The whole thing is ~50 lines of glue and slots into ``Benchmark`` next to
``SingleAlgorithm`` and ``CMAESLBFGSBHandoff`` exactly as if it were
pre-built. The plotter doesn't know or care that it's custom — it asks
the trace for ``evaluations`` / ``best_fitness`` / ``handoff_eval`` and
that's all the framework needs.

The benchmark at the bottom compares DES alone, L-BFGS-B alone, the
custom DES -> L-BFGS-B, and the pre-built CMA-ES -> L-BFGS-B on a 10D
Rosenbrock so you can see the new algorithm side-by-side with the
existing ones.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from src.algorithms.choices import AlgorithmChoice
from src.algorithms.cmaes.config import CMAESConfig
from src.algorithms.des.config import DESConfig
from src.algorithms.lbfgsb.config import LBFGSBConfig
from src.benchmarking import (
    Benchmark,
    CMAESLBFGSBHandoff,
    HandoffTransform,
    Problem,
    RunTrace,
    SingleAlgorithm,
)
from src.core.algorithm_factory import AlgorithmFactory
from src.plotting import plot_benchmark_boxplot, plot_benchmark_convergence
from src.utils.benchmark_functions import Rosenbrock


plt.ioff()
plt.switch_backend("Agg")


OUTPUT_DIR = Path("plots/basic/custom_handoff")


# ---------------------------------------------------------------------------
# The custom algorithm.
#
# ``AlgorithmRun`` is a runtime-checkable Protocol with just three slots:
#   - ``name``: shows up in legends and traces
#   - ``color``: hex string the plotter uses for this algo's line
#   - ``run(problem, x0, seed) -> RunTrace``
#
# Anything matching that shape works. A frozen-ish ``@dataclass`` keeps
# us honest about state — the per-run state lives on the ``RunTrace``
# we return.
# ---------------------------------------------------------------------------

@dataclass
class DESLBFGSBHandoff:
    """DES warm-up followed by L-BFGS-B refinement.

    No covariance is passed — DES doesn't carry one externally — so
    L-BFGS-B starts with its default B_0 = I and rebuilds curvature
    from line-search information. The interesting variable is whether
    the DES warm-up has found a basin worth refining.
    """

    name: str
    color: str
    des_config_factory: Callable[[int], DESConfig]
    lbfgsb_config_factory: Callable[[int], LBFGSBConfig]

    def run(
        self,
        problem: Problem,
        x0: NDArray[np.float64],
        seed: int,
    ) -> RunTrace:
        # Phase 1: DES warm-up
        des_config = self.des_config_factory(problem.dimensions)
        des = AlgorithmFactory.create_optimizer(
            AlgorithmChoice.DES,
            problem.function,
            x0,
            des_config,
            lower_bounds=problem.lower_bound,
            upper_bounds=problem.upper_bound,
            seed=seed,
        )
        des_result = des.optimize()

        warmup_evals = des_result.evaluations
        warmup_iters = len(des_result.diagnostic.iteration)
        des_best = float(des_result.best_fitness)
        starting_point = des_result.best_solution.copy()

        # Phase 2: L-BFGS-B refinement from DES's best point
        lbfgsb_config = self.lbfgsb_config_factory(problem.dimensions)
        lbfgsb_kwargs: dict = {}
        if problem.gradient is not None:
            lbfgsb_kwargs["gradient_fn"] = problem.gradient
        lbfgsb = AlgorithmFactory.create_optimizer(
            AlgorithmChoice.LBFGSB,
            problem.function,
            starting_point,
            lbfgsb_config,
            lower_bounds=problem.lower_bound,
            upper_bounds=problem.upper_bound,
            **lbfgsb_kwargs,
        )
        lbfgsb_result = lbfgsb.optimize()

        # Stitch the two convergence traces together. L-BFGS-B's eval
        # counts are local to itself, so offset them by the DES warm-up
        # count. Clamp the L-BFGS-B segment so it never reports a fitness
        # *worse* than the handoff value — its first reported point is
        # f(x0_lbfgsb), which might be slightly worse than the DES best.
        des_evals = list(des_result.diagnostic.evaluations)
        des_fits = list(des_result.diagnostic.best_fitness)
        lbfgsb_evals = [e + warmup_evals for e in lbfgsb_result.diagnostic.evaluations]
        lbfgsb_fits = [min(v, des_best) for v in lbfgsb_result.diagnostic.best_fitness]

        return RunTrace(
            algorithm=self.name,
            problem=problem.name,
            seed=seed,
            evaluations=des_evals + lbfgsb_evals,
            best_fitness=des_fits + lbfgsb_fits,
            final_evaluations=warmup_evals + lbfgsb_result.evaluations,
            final_fitness=min(des_best, float(lbfgsb_result.best_fitness)),
            handoff_eval=warmup_evals,
            handoff_iter=warmup_iters,
        )


# ---------------------------------------------------------------------------
# Benchmark wiring — the custom class slots in beside SingleAlgorithm
# and CMAESLBFGSBHandoff with no special handling.
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dimensions = 10
    total_budget = 4000
    warmup_budget = 1500

    problems = [
        Problem.from_benchmark("Rosenbrock", Rosenbrock(dimensions=dimensions)),
    ]

    algorithms = [
        SingleAlgorithm(
            name="DES",
            color="#f39c12",
            algorithm=AlgorithmChoice.DES,
            config_factory=lambda d: DESConfig(dimensions=d, budget=total_budget),
        ),
        SingleAlgorithm(
            name="L-BFGS-B",
            color="#3498db",
            algorithm=AlgorithmChoice.LBFGSB,
            config_factory=lambda d: LBFGSBConfig(dimensions=d, budget=total_budget),
        ),
        # The custom one.
        DESLBFGSBHandoff(
            name="DES -> L-BFGS-B",
            color="#9b59b6",
            des_config_factory=lambda d: DESConfig(dimensions=d, budget=warmup_budget),
            lbfgsb_config_factory=lambda d: LBFGSBConfig(
                dimensions=d, budget=total_budget - warmup_budget,
            ),
        ),
        # Pre-built handoff for side-by-side comparison.
        CMAESLBFGSBHandoff(
            name="CMA-ES -> L-BFGS-B",
            color="#2ecc71",
            cmaes_config_factory=lambda d: CMAESConfig(
                dimensions=d, budget=warmup_budget,
            ),
            lbfgsb_config_factory=lambda d: LBFGSBConfig(
                dimensions=d, budget=total_budget - warmup_budget,
            ),
            transform=HandoffTransform.INVERSE,
        ),
    ]

    bench = Benchmark(
        problems=problems,
        algorithms=algorithms,
        seeds=list(range(5)),
        output_dir=OUTPUT_DIR / "_bench",
        save_artifacts=False,
    )
    print("Running 1 problem x 4 algorithms x 5 seeds...")
    bench.run(verbose=True)
    bench.print_summary()

    # Same plotting calls as for any pre-built algorithm — the
    # framework doesn't know DESLBFGSBHandoff is custom.
    plot_benchmark_convergence(
        bench.traces,
        problems=problems,
        algorithms=algorithms,
        title=f"Custom DES -> L-BFGS-B vs baselines on {dimensions}D Rosenbrock",
        save_path=OUTPUT_DIR / "convergence.png",
    )
    plot_benchmark_boxplot(
        bench.traces,
        problems=problems,
        algorithms=algorithms,
        title=f"Final fitness distribution ({dimensions}D, 5 seeds)",
        save_path=OUTPUT_DIR / "final_fitness.png",
    )
    print(f"\nOutput written to: {OUTPUT_DIR.absolute()}")


if __name__ == "__main__":
    main()
