"""End-to-end demo of the multi-seed declarative plotting functions.

Runs a small (problem x algorithm x seed) benchmark and exercises:

1. ``plot_benchmark_convergence`` — one panel per problem, median + IQR
   band across seeds, multiple algorithms overlaid.
2. ``plot_benchmark_boxplot`` — final-fitness distribution per algorithm,
   one panel per problem.
3. Handoff annotation — the CMA-ES -> L-BFGS-B trace marks where the
   warm-up ends with a vertical line.

Output goes to ``plots/basic/declarative_benchmark/``.
"""

from pathlib import Path

import matplotlib.pyplot as plt

from declivity.algorithms.choices import AlgorithmChoice
from declivity.algorithms.cmaes.config import CMAESConfig
from declivity.algorithms.lbfgsb.config import LBFGSBConfig
from declivity.benchmarking import (
    Benchmark,
    CMAESLBFGSBHandoff,
    Problem,
    SingleAlgorithm,
)
from declivity.plotting import plot_benchmark_boxplot, plot_benchmark_convergence
from declivity.utils.benchmark_functions import Rosenbrock, Sphere
from declivity.utils.stopping_conditions import MaxEvaluations


plt.ioff()
plt.switch_backend("Agg")


OUTPUT_DIR = Path("plots/basic/declarative_benchmark")

COLORS = {
    "CMA-ES":             "#e74c3c",
    "L-BFGS-B":           "#3498db",
    "CMA-ES -> L-BFGS-B": "#2ecc71",
}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Two problems: smooth (Sphere) and ill-conditioned (Rosenbrock). Both
    # 10D, so a single shared budget is fair.
    dimensions = 10
    total_budget = 4000

    problems = [
        Problem.from_benchmark("Sphere",     Sphere(dimensions=dimensions)),
        Problem.from_benchmark("Rosenbrock", Rosenbrock(dimensions=dimensions)),
    ]

    algorithms = [
        SingleAlgorithm(
            name="CMA-ES",
            color=COLORS["CMA-ES"],
            algorithm=AlgorithmChoice.CMAES,
            config_factory=lambda d: CMAESConfig(dimensions=d),
            stopping_condition=MaxEvaluations(total_budget),
        ),
        SingleAlgorithm(
            name="L-BFGS-B",
            color=COLORS["L-BFGS-B"],
            algorithm=AlgorithmChoice.LBFGSB,
            config_factory=lambda d: LBFGSBConfig(dimensions=d),
            stopping_condition=MaxEvaluations(total_budget),
        ),
        CMAESLBFGSBHandoff(
            name="CMA-ES -> L-BFGS-B",
            color=COLORS["CMA-ES -> L-BFGS-B"],
            cmaes_config_factory=lambda d: CMAESConfig(dimensions=d),
            lbfgsb_config_factory=lambda d: LBFGSBConfig(dimensions=d),
            cmaes_stopping_condition=MaxEvaluations(1500),
            lbfgsb_stopping_condition=MaxEvaluations(total_budget - 1500),
            transform="inverse",
        ),
    ]

    bench = Benchmark(
        problems=problems,
        algorithms=algorithms,
        seeds=list(range(5)),
        output_dir=OUTPUT_DIR / "_bench",
        num_workers=1,
        save_artifacts=False,  # demo only; skip the traces.json/csv dump
    )
    print("Running 2 problems x 3 algorithms x 5 seeds...")
    bench.run(verbose=True)
    bench.print_summary()

    # Multi-seed convergence (panel-per-problem, algos overlaid with band).
    plot_benchmark_convergence(
        bench.traces,
        problems=problems,
        algorithms=algorithms,
        title=f"Convergence on {dimensions}D problems (5 seeds, budget={total_budget})",
        save_path=OUTPUT_DIR / "convergence.png",
    )
    print(f"\nSaved: {OUTPUT_DIR / 'convergence.png'}")

    # Final-fitness distribution boxplot.
    plot_benchmark_boxplot(
        bench.traces,
        problems=problems,
        algorithms=algorithms,
        title=f"Final fitness distribution ({dimensions}D, 5 seeds)",
        save_path=OUTPUT_DIR / "final_fitness.png",
    )
    print(f"Saved: {OUTPUT_DIR / 'final_fitness.png'}")

    print(f"\nOutput written to: {OUTPUT_DIR.absolute()}")


if __name__ == "__main__":
    main()
