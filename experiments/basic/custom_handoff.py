"""Building a custom handoff algorithm from scratch.

This file demonstrates how to plug a brand-new algorithm into the
benchmarking framework when the pre-built :class:`CMAESLBFGSBHandoff`
doesn't fit. We build a DES -> L-BFGS-B handoff (no such class ships)
that:

1. Runs DES for ``warmup_budget`` evaluations to find a basin.
2. Hands the best-so-far point to L-BFGS-B for local refinement.

The whole thing is **one method** (``run_phases``) on a dataclass that
subclasses :class:`HandoffAlgorithm`. The base class handles the
trace-stitching boilerplate that every handoff would otherwise have to
re-write: eval-count offsets, fitness clamping, and the
``handoff_eval`` / ``handoff_iter`` metadata the plotter uses to draw
vertical markers.

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

from declivity.algorithms.choices import AlgorithmChoice
from declivity.algorithms.cmaes.config import CMAESConfig
from declivity.algorithms.des.config import DESConfig
from declivity.algorithms.lbfgsb.config import LBFGSBConfig
from declivity.benchmarking import (
    Benchmark,
    CMAESLBFGSBHandoff,
    HandoffAlgorithm,
    HandoffTransform,
    Problem,
    SingleAlgorithm,
)
from declivity.core.algorithm_factory import AlgorithmFactory
from declivity.core.base_optimizer import OptimizationResult
from declivity.plotting import plot_benchmark_boxplot, plot_benchmark_convergence
from declivity.utils.benchmark_functions import Rosenbrock


plt.ioff()
plt.switch_backend("Agg")


OUTPUT_DIR = Path("plots/basic/custom_handoff")


# ---------------------------------------------------------------------------
# The custom algorithm.
#
# Inherit from HandoffAlgorithm, declare your config fields with
# @dataclass, and implement ``run_phases``. The base class handles
# everything else.
# ---------------------------------------------------------------------------

@dataclass
class DESLBFGSBHandoff(HandoffAlgorithm):
    """DES warm-up followed by L-BFGS-B refinement.

    No covariance is passed — DES doesn't carry one externally — so
    L-BFGS-B starts with its default ``B_0 = I`` and rebuilds curvature
    from line-search information. The interesting variable is whether
    the DES warm-up has found a basin worth refining.
    """

    name: str
    color: str
    des_config_factory: Callable[[int], DESConfig]
    lbfgsb_config_factory: Callable[[int], LBFGSBConfig]

    def run_phases(
        self,
        problem: Problem,
        x0: NDArray[np.float64],
        seed: int,
    ) -> tuple[OptimizationResult, OptimizationResult]:
        # Phase 1: DES warm-up
        des_result = AlgorithmFactory.create_optimizer(
            AlgorithmChoice.DES,
            problem.function,
            x0,
            self.des_config_factory(problem.dimensions),
            lower_bounds=problem.lower_bound,
            upper_bounds=problem.upper_bound,
            seed=seed,
        ).optimize()

        # Phase 2: L-BFGS-B refinement from DES's best point
        lbfgsb_result = AlgorithmFactory.create_optimizer(
            AlgorithmChoice.LBFGSB,
            problem.function,
            des_result.best_solution,
            self.lbfgsb_config_factory(problem.dimensions),
            lower_bounds=problem.lower_bound,
            upper_bounds=problem.upper_bound,
        ).optimize()

        return des_result, lbfgsb_result


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
        # The custom handoff.
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
