"""Benchmark the framework CMA-ES under different component injections.

This experiment exercises the two seams that were wired into
:class:`~src.algorithms.cmaes.CMAESOptimizer` during the framework
integration milestone:

* :class:`~src.utils.repair_strategies.RepairStrategy` — vectorised
  ``ClampRepair`` (default) vs per-individual ``LamarckianRepair``.
* :class:`~src.utils.population_initializers.PopulationInitializer` —
  ``MeanSigmaPopulationInitializer`` (default, samples ``N(m, σ²I)``)
  vs ``NormalPopulationInitializer`` (DES-style ``rng.normal`` around
  ``x0`` with bounds-derived scale).

Three variants are run through the standard
:class:`~src.benchmarking.Benchmark` harness across five seeds and
three problems (Sphere, Rosenbrock, Rastrigin — 10D each).  Equivalent
convergence curves across variants confirm that:

1. The default behaviour is preserved (no regression vs the previous
   resampling-based ``_ask`` loop on bound-feasible problems).
2. The injected components actually drive the optimiser — swapping
   them changes the trajectory in the expected direction (Lamarckian
   repair adds the ``_remove_inf_nan`` pass; the DES-style initializer
   widens the iteration-0 distribution).

This script also exercises the
``repair_strategy`` / ``population_initializer`` fields on
:class:`~src.benchmarking.SingleAlgorithm` — the canonical way to swap
evolutionary components through the standard benchmarking surface
without writing a custom :class:`BenchmarkAlgorithm`.

Output goes to ``plots/cross_validation/cmaes_components/``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from src.algorithms.choices import AlgorithmChoice
from src.algorithms.cmaes.config import CMAESConfig
from src.benchmarking import AlgorithmRun, Benchmark, Problem, SingleAlgorithm
from src.plotting import plot_benchmark_boxplot, plot_benchmark_convergence
from src.utils.benchmark_functions import Rastrigin, Rosenbrock, Sphere
from src.utils.population_initializers import NormalPopulationInitializer
from src.utils.repair_strategies import LamarckianRepair


plt.ioff()
plt.switch_backend("Agg")


OUTPUT_DIR = Path("plots/cross_validation/cmaes_components")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dimensions = 10
    budget = 4000

    problems = [
        Problem.from_benchmark("Sphere",     Sphere(dimensions=dimensions)),
        Problem.from_benchmark("Rosenbrock", Rosenbrock(dimensions=dimensions)),
        Problem.from_benchmark("Rastrigin",  Rastrigin(dimensions=dimensions)),
    ]

    config_factory = lambda d: CMAESConfig(dimensions=d, budget=budget)

    algorithms: list[AlgorithmRun] = [
        SingleAlgorithm(
            name="default (ClampRepair + MeanSigma)",
            color="#e74c3c",
            algorithm=AlgorithmChoice.CMAES,
            config_factory=config_factory,
        ),
        SingleAlgorithm(
            name="LamarckianRepair",
            color="#3498db",
            algorithm=AlgorithmChoice.CMAES,
            config_factory=config_factory,
            repair_strategy=LamarckianRepair(),
        ),
        SingleAlgorithm(
            name="NormalPopulationInitializer",
            color="#2ecc71",
            algorithm=AlgorithmChoice.CMAES,
            config_factory=config_factory,
            population_initializer=NormalPopulationInitializer(),
        ),
    ]

    bench = Benchmark(
        problems=problems,
        algorithms=algorithms,
        seeds=list(range(5)),
        output_dir=OUTPUT_DIR / "_bench",
        num_workers=1,
        save_artifacts=False,
    )
    print(
        f"Running {len(problems)} problems x {len(algorithms)} variants x 5 seeds "
        f"(budget={budget}, d={dimensions})..."
    )
    bench.run(verbose=True)
    bench.print_summary()

    plot_benchmark_convergence(
        bench.traces,
        problems=problems,
        algorithms=algorithms,
        title=(
            f"CMA-ES component variants — {dimensions}D, 5 seeds, budget={budget}"
        ),
        save_path=OUTPUT_DIR / "convergence.png",
    )
    print(f"\nSaved: {OUTPUT_DIR / 'convergence.png'}")

    plot_benchmark_boxplot(
        bench.traces,
        problems=problems,
        algorithms=algorithms,
        title=f"Final-fitness distribution ({dimensions}D, 5 seeds)",
        save_path=OUTPUT_DIR / "final_fitness.png",
    )
    print(f"Saved: {OUTPUT_DIR / 'final_fitness.png'}")

    print(f"\nOutput written to: {OUTPUT_DIR.absolute()}")


if __name__ == "__main__":
    main()
