"""Showcase: the declarative plotter's modularity, in three one-liners.

Every figure this script writes is produced by a **single** plotting call.
There is no per-algorithm plotting code anywhere below: each function reads
the algorithm off the result/trace, and the panel registry decides what
every curve *means*. Swapping algorithms or problems changes nothing about
the plotting calls.

The three one-liners:

1. ``plot_comparison(results, panels=["convergence"])`` — two algorithms
   from two different families (evolutionary CMA-ES, quasi-Newton L-BFGS-B)
   overlaid on a single convergence axes. The semantic key ``convergence``
   resolves to ``best_fitness`` for both, so one call draws both curves.

2. ``plot_benchmark_convergence(bench.traces, ...)`` — the multi-seed
   "average": a median line plus a 25/75 IQR band per algorithm, one panel
   per problem. The benchmark grid does the runs; one line draws the
   summary.

3. ``plot_metrics(result, panels=PanelSet.ALL)`` — every registered
   diagnostic panel for one optimizer (CMA-ES has 12), laid out
   automatically. Adding a 13th panel is one line in ``standard_panels.py``
   and it would appear here for free.

Output: ``plots/basic/plotter_showcase/``.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src import AlgorithmFactory
from src.algorithms.choices import AlgorithmChoice
from src.algorithms.cmaes.config import CMAESConfig
from src.algorithms.lbfgsb.config import LBFGSBConfig
from src.benchmarking import (
    Benchmark,
    CMAESLBFGSBHandoff,
    Problem,
    SingleAlgorithm,
)
from src.core.base_optimizer import OptimizationResult
from src.plotting import (
    PanelSet,
    plot_benchmark_convergence,
    plot_comparison,
    plot_metrics,
)
from src.utils.benchmark_functions import (
    BenchmarkFunction,
    Ellipsoid,
    Rosenbrock,
    Sphere,
)


plt.ioff()
plt.switch_backend("Agg")


OUTPUT_DIR = Path("plots/basic/plotter_showcase")
DIMENSIONS = 10
NUM_SEEDS = 11

# One palette, used for both the comparison overlay (keyed by label) and
# the benchmark algorithms (carried on each AlgorithmRun.color).
COLORS = {
    "CMA-ES":             "#e74c3c",
    "L-BFGS-B":           "#3498db",
    "CMA-ES -> L-BFGS-B": "#2ecc71",
}


def make_x0(func: BenchmarkFunction, seed: int) -> np.ndarray:
    """Random uniform start inside the function's box (avoids the
    symmetric-x0 alignment pitfall noted in the project docs)."""
    lower, upper = func.bounds
    return np.random.default_rng(seed).uniform(lower, upper)


def run_single(
    algorithm: AlgorithmChoice,
    func: BenchmarkFunction,
    x0: np.ndarray,
    *,
    budget: int,
    all_diagnostics: bool = False,
    seed: int = 0,
) -> OptimizationResult:
    """Build a config, optionally turn on every diagnostic, and optimize."""
    if algorithm is AlgorithmChoice.CMAES:
        config: CMAESConfig | LBFGSBConfig = CMAESConfig(
            dimensions=len(x0), budget=budget
        )
    elif algorithm is AlgorithmChoice.LBFGSB:
        config = LBFGSBConfig(dimensions=len(x0), budget=budget)
    else:
        raise ValueError(algorithm)

    if all_diagnostics:
        config.enable_all_diagnostics()

    lower, upper = func.bounds
    optimizer = AlgorithmFactory.create_optimizer(
        algorithm=algorithm,
        func=func,
        initial_point=x0,
        config=config,
        lower_bounds=lower,
        upper_bounds=upper,
        seed=seed,
    )
    return optimizer.optimize()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # =====================================================================
    # ONE-LINER 1 — two convergences, two algorithm families, one axes.
    # =====================================================================
    # Same starting point for a fair fight; Rosenbrock separates the two
    # nicely (L-BFGS-B plunges down the valley, CMA-ES descends steadily).
    rosenbrock = Rosenbrock(dimensions=DIMENSIONS)
    x0 = make_x0(rosenbrock, seed=0)
    cmaes_rb = run_single(AlgorithmChoice.CMAES, rosenbrock, x0, budget=3000)
    lbfgsb_rb = run_single(AlgorithmChoice.LBFGSB, rosenbrock, x0, budget=3000)
    print(
        f"[1] Rosenbrock  CMA-ES={cmaes_rb.best_fitness:.2e}  "
        f"L-BFGS-B={lbfgsb_rb.best_fitness:.2e}"
    )

    plot_comparison(
        {"CMA-ES": cmaes_rb, "L-BFGS-B": lbfgsb_rb},
        panels=["convergence"],
        colors=COLORS,
        title="Two convergences, one graph — CMA-ES vs L-BFGS-B (10-D Rosenbrock)",
        save_path=OUTPUT_DIR / "01_two_convergences.png",
    )

    # =====================================================================
    # ONE-LINER 2 — the multi-seed "average" (median + IQR band).
    # =====================================================================
    # The Benchmark grid runs every (problem x algorithm x seed) triple;
    # plot_benchmark_convergence summarises it in one call.
    problems = [
        Problem.from_benchmark("Sphere", Sphere(dimensions=DIMENSIONS)),
        Problem.from_benchmark("Rosenbrock", Rosenbrock(dimensions=DIMENSIONS)),
    ]
    algorithms = [
        SingleAlgorithm(
            name="CMA-ES",
            color=COLORS["CMA-ES"],
            algorithm=AlgorithmChoice.CMAES,
            config_factory=lambda d: CMAESConfig(dimensions=d, budget=3000),
        ),
        SingleAlgorithm(
            name="L-BFGS-B",
            color=COLORS["L-BFGS-B"],
            algorithm=AlgorithmChoice.LBFGSB,
            config_factory=lambda d: LBFGSBConfig(dimensions=d, budget=3000),
        ),
        CMAESLBFGSBHandoff(
            name="CMA-ES -> L-BFGS-B",
            color=COLORS["CMA-ES -> L-BFGS-B"],
            cmaes_config_factory=lambda d: CMAESConfig(dimensions=d, budget=1200),
            lbfgsb_config_factory=lambda d: LBFGSBConfig(dimensions=d, budget=1800),
            transform="inverse",
        ),
    ]
    bench = Benchmark(
        problems=problems,
        algorithms=algorithms,
        seeds=list(range(NUM_SEEDS)),
        output_dir=OUTPUT_DIR / "_bench",
        num_workers=4,
        save_artifacts=False,  # showcase only; skip the traces.json/csv dump
    )
    print(f"[2] Running {len(problems)}x{len(algorithms)}x{NUM_SEEDS} benchmark...")
    bench.run(verbose=False)

    plot_benchmark_convergence(
        bench.traces,
        problems=problems,
        algorithms=algorithms,
        title=f"Benchmark average — median + IQR over {NUM_SEEDS} seeds (10-D)",
        save_path=OUTPUT_DIR / "02_benchmark_average.png",
    )

    # =====================================================================
    # ONE-LINER 3 — every registered diagnostic panel for one optimizer.
    # =====================================================================
    # Ill-conditioned Ellipsoid (cond 1e6) so the geometry panels
    # (condition number, eigenvalues, det C) actually move. all-diagnostics
    # makes sure every panel has data to draw.
    ellipsoid = Ellipsoid(dimensions=DIMENSIONS)
    cmaes_ell = run_single(
        AlgorithmChoice.CMAES,
        ellipsoid,
        make_x0(ellipsoid, seed=0),
        budget=5000,
        all_diagnostics=True,
    )
    print(f"[3] Ellipsoid  CMA-ES={cmaes_ell.best_fitness:.2e}")

    plot_metrics(
        cmaes_ell,
        panels=PanelSet.ALL,
        ncols=3,
        title="All diagnostics in one call — CMA-ES on 10-D Ellipsoid (cond 1e6)",
        save_path=OUTPUT_DIR / "03_all_diagnostics.png",
    )

    print(f"\nOutput written to: {OUTPUT_DIR.absolute()}")


if __name__ == "__main__":
    main()
