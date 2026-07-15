"""Building a custom benchmark algorithm from scratch.

This file demonstrates the most generic extension point in the
benchmarking framework: subclassing :class:`BenchmarkAlgorithm`. The
example here is a **multi-start CMA-ES** that the framework doesn't ship
out of the box — it would not fit :class:`SingleAlgorithm` (more than
one optimizer call) or :class:`HandoffAlgorithm` (no warmup/refinement
structure, just repeated independent restarts with the best one kept).

The minimal contract is the docstring on :class:`BenchmarkAlgorithm`:

    1. Subclass it as a :py:func:`dataclass` declaring ``name``,
       ``color``, and whatever config fields you want.
    2. Implement ``run(problem, x0, seed)`` returning a :class:`RunTrace`.
    3. Use :py:meth:`BenchmarkAlgorithm.trace_from_result` to package a
       single :class:`OptimizationResult` into a :class:`RunTrace`,
       or assemble one by hand for multi-phase runners.

The benchmark at the bottom puts the custom algorithm alongside vanilla
CMA-ES on Rastrigin (which is multimodal — restarts help) so you can see
the lift.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from declivity.algorithms.choices import AlgorithmChoice
from declivity.algorithms.cmaes.config import CMAESConfig
from declivity.benchmarking import (
    Benchmark,
    BenchmarkAlgorithm,
    Problem,
    RunTrace,
    SingleAlgorithm,
)
from declivity.core.algorithm_factory import AlgorithmFactory
from declivity.plotting import plot_benchmark_boxplot, plot_benchmark_convergence
from declivity.utils.benchmark_functions import Rastrigin
from declivity.utils.stopping_conditions import MaxEvaluations, StoppingCondition


plt.ioff()
plt.switch_backend("Agg")


OUTPUT_DIR = Path("plots/basic/custom_algorithm")


# ---------------------------------------------------------------------------
# The custom algorithm.
#
# Multi-start CMA-ES doesn't fit SingleAlgorithm (multiple calls) or
# HandoffAlgorithm (not two-phase, just N independent restarts). Subclass
# BenchmarkAlgorithm and implement run() — that's the most general
# extension point.
# ---------------------------------------------------------------------------

@dataclass
class MultiStartCMAES(BenchmarkAlgorithm):
    """Run CMA-ES ``num_restarts`` times with different x0 per restart.

    Each restart gets its own slice of the total budget. The starting
    point of each restart after the first is drawn uniformly from the
    problem bounds using a per-restart sub-seed; that way the whole
    run is still deterministic given ``seed``.

    The trace concatenates every restart's convergence sequence with
    cumulative eval offsets, so the plotted curve shows the best fitness
    found across all restarts as a function of total budget.
    """

    name: str
    color: str
    config_factory: Callable[[int], CMAESConfig]
    num_restarts: int = 5
    stopping_condition: StoppingCondition | None = None
    """Per-restart stopping condition (default: the optimizer's own
    ``MaxEvaluations(10_000 * d)``). Each restart gets a fresh copy of
    this budget, so the total is ``num_restarts`` times its cap."""

    def run(
        self,
        problem: Problem,
        x0: NDArray[np.float64],
        seed: int,
    ) -> RunTrace:
        # Per-restart sub-seeds so the run is reproducible given the
        # top-level seed but each restart explores a different basin.
        rng = np.random.default_rng(seed)
        restart_seeds = rng.integers(0, 2**31, size=self.num_restarts)

        all_evaluations: list[int] = []
        all_best_fitness: list[float] = []
        running_evals = 0
        running_best = float("inf")

        for restart_index, restart_seed in enumerate(restart_seeds):
            # First restart starts from the framework-provided x0 so the
            # comparison with vanilla CMA-ES is fair on seed 0. Later
            # restarts draw a fresh x0 from a per-restart rng.
            if restart_index == 0:
                restart_x0 = x0
            else:
                restart_rng = np.random.default_rng(int(restart_seed))
                restart_x0 = restart_rng.uniform(
                    problem.lower_bound,
                    problem.upper_bound,
                    size=problem.dimensions,
                )

            result = AlgorithmFactory.create_optimizer(
                AlgorithmChoice.CMAES,
                problem.function,
                restart_x0,
                self.config_factory(problem.dimensions),
                stopping_condition=self.stopping_condition,
                lower_bounds=problem.lower_bound,
                upper_bounds=problem.upper_bound,
                seed=int(restart_seed),
            ).optimize()

            # Concatenate this restart's logged sequence with offsets and
            # a running-min over restarts so the curve is monotone
            # non-increasing across the full budget.
            for evaluation, fitness in zip(
                result.diagnostic.evaluations, result.diagnostic.best_fitness
            ):
                all_evaluations.append(int(evaluation) + running_evals)
                running_best = min(running_best, float(fitness))
                all_best_fitness.append(running_best)

            running_evals += result.evaluations
            running_best = min(running_best, float(result.best_fitness))

        return RunTrace(
            algorithm=self.name,
            problem=problem.name,
            seed=seed,
            evaluations=all_evaluations,
            best_fitness=all_best_fitness,
            final_evaluations=running_evals,
            final_fitness=running_best,
        )


# ---------------------------------------------------------------------------
# Benchmark wiring — the custom class slots in next to SingleAlgorithm
# with no special handling.
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dimensions = 10
    total_budget = 5000
    num_restarts = 5
    per_restart_budget = total_budget // num_restarts

    problems = [
        Problem.from_benchmark("Rastrigin", Rastrigin(dimensions=dimensions)),
    ]

    algorithms = [
        SingleAlgorithm(
            name="CMA-ES (no restarts)",
            color="#e74c3c",
            algorithm=AlgorithmChoice.CMAES,
            config_factory=lambda d: CMAESConfig(dimensions=d),
            stopping_condition=MaxEvaluations(total_budget),
        ),
        MultiStartCMAES(
            name=f"CMA-ES ({num_restarts}x restart)",
            color="#9b59b6",
            config_factory=lambda d: CMAESConfig(dimensions=d),
            stopping_condition=MaxEvaluations(per_restart_budget),
            num_restarts=num_restarts,
        ),
    ]

    bench = Benchmark(
        problems=problems,
        algorithms=algorithms,
        seeds=list(range(10)),
        output_dir=OUTPUT_DIR / "_bench",
        save_artifacts=False,
    )
    print(f"Running 1 problem x 2 algorithms x 10 seeds...")
    bench.run(verbose=True)
    bench.print_summary()

    plot_benchmark_convergence(
        bench.traces,
        problems=problems,
        algorithms=algorithms,
        title=f"Multi-start CMA-ES on {dimensions}D Rastrigin",
        save_path=OUTPUT_DIR / "convergence.png",
    )
    plot_benchmark_boxplot(
        bench.traces,
        problems=problems,
        algorithms=algorithms,
        title=f"Final fitness distribution ({dimensions}D, 10 seeds)",
        save_path=OUTPUT_DIR / "final_fitness.png",
    )
    print(f"\nOutput written to: {OUTPUT_DIR.absolute()}")


if __name__ == "__main__":
    main()
