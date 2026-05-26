"""L-BFGS-B benchmark on Sphere function.

Compares L-BFGS-B (More-Thuente and Armijo line search) against CMA-ES on
a 10D Sphere. Produces:

- A convergence overlay across all three runs.
- A diagnostic-panel view for the More-Thuente L-BFGS-B run.
- A diagnostic-panel view for the CMA-ES run.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src import AlgorithmFactory
from src.algorithms.choices import AlgorithmChoice
from src.algorithms.lbfgsb.config import LBFGSBConfig, LineSearchMethod
from src.plotting import plot_comparison, plot_metrics
from src.plotting.multi_algorithm_plotter import MultiAlgorithmPlotter
from src.utils.benchmark_functions import Sphere


plt.ioff()
plt.switch_backend("Agg")


PRODUCTION_DIR = Path("plots/lbfgsb/sphere_benchmark")
COMPARE_DIR = Path("plots/_migration_compare/sphere_benchmark")

COLORS = {
    "L-BFGS-B (More-Thuente)": "#3498db",
    "L-BFGS-B (Armijo)":       "#1abc9c",
    "CMA-ES":                  "#e74c3c",
}


def run_lbfgsb_benchmark() -> None:
    dimensions = 10
    objective = Sphere(dimensions=dimensions)
    lower_bounds = -50.0
    upper_bounds = 50.0
    seed = 42

    rng = np.random.default_rng(seed)
    initial_point = rng.uniform(lower_bounds, upper_bounds, size=dimensions)

    print(f"Sphere function benchmark (d={dimensions})")
    print(f"Initial point f(x) = {objective(initial_point):.6f}")
    print(f"Bounds: [{lower_bounds}, {upper_bounds}]\n")

    # L-BFGS-B (More-Thuente)
    config_mt = LBFGSBConfig(
        dimensions=dimensions, m=10, line_search=LineSearchMethod.MORE_THUENTE,
    )
    config_mt.enable_all_diagnostics()
    optimizer_mt = AlgorithmFactory.create_optimizer(
        algorithm=AlgorithmChoice.LBFGSB,
        func=objective,
        initial_point=initial_point,
        config=config_mt,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        seed=seed,
    )
    result_mt = optimizer_mt.optimize()
    print(f"L-BFGS-B (More-Thuente):")
    print(f"  Best fitness:  {result_mt.best_fitness:.12e}")
    print(f"  Evaluations:   {result_mt.evaluations}")
    print(f"  Message:       {result_mt.message}\n")

    # L-BFGS-B (Armijo)
    config_armijo = LBFGSBConfig(
        dimensions=dimensions, m=10, line_search=LineSearchMethod.ARMIJO,
    )
    config_armijo.enable_all_diagnostics()
    optimizer_armijo = AlgorithmFactory.create_optimizer(
        algorithm=AlgorithmChoice.LBFGSB,
        func=objective,
        initial_point=initial_point,
        config=config_armijo,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        seed=seed,
    )
    result_armijo = optimizer_armijo.optimize()
    print(f"L-BFGS-B (Armijo):")
    print(f"  Best fitness:  {result_armijo.best_fitness:.12e}")
    print(f"  Evaluations:   {result_armijo.evaluations}")
    print(f"  Message:       {result_armijo.message}\n")

    # CMA-ES baseline
    config_cmaes = AlgorithmFactory.create_config(
        AlgorithmChoice.CMAES, dimensions=dimensions,
    )
    config_cmaes.enable_all_diagnostics()
    optimizer_cmaes = AlgorithmFactory.create_optimizer(
        algorithm=AlgorithmChoice.CMAES,
        func=objective,
        initial_point=initial_point,
        config=config_cmaes,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        seed=seed,
    )
    result_cmaes = optimizer_cmaes.optimize()
    print(f"CMA-ES:")
    print(f"  Best fitness:  {result_cmaes.best_fitness:.12e}")
    print(f"  Evaluations:   {result_cmaes.evaluations}")
    print(f"  Message:       {result_cmaes.message}\n")

    PRODUCTION_DIR.mkdir(parents=True, exist_ok=True)
    COMPARE_DIR.mkdir(parents=True, exist_ok=True)

    convergence_results = {
        "L-BFGS-B (More-Thuente)": result_mt,
        "L-BFGS-B (Armijo)":       result_armijo,
        "CMA-ES":                  result_cmaes,
    }
    # plot_comparison defaults to the common keys, which on (LBFGSB, CMAES)
    # is just convergence + step_size — exactly what the old call produced.

    # Production output (new plotter).
    plot_comparison(
        convergence_results,
        colors=COLORS,
        title="L-BFGS-B vs CMA-ES on Sphere (10D)",
        save_path=PRODUCTION_DIR / "lbfgsb_convergence_comparison.png",
    )
    plot_metrics(
        result_mt,
        title="L-BFGS-B (More-Thuente) on Sphere (10D)",
        save_path=PRODUCTION_DIR / "lbfgsb_metrics.png",
    )
    plot_metrics(
        result_cmaes,
        title="CMA-ES on Sphere (10D)",
        save_path=PRODUCTION_DIR / "cmaes_sphere_metrics.png",
    )

    # MIGRATION COMPARE — duplicate output for side-by-side review; remove
    # once approved.
    plot_comparison(
        convergence_results,
        colors=COLORS,
        title="L-BFGS-B vs CMA-ES on Sphere (10D) — new plotter",
        save_path=COMPARE_DIR / "new__convergence_comparison.png",
    )
    plot_metrics(
        result_mt,
        title="L-BFGS-B (MT) on Sphere — new plotter",
        save_path=COMPARE_DIR / "new__lbfgsb_metrics.png",
    )
    plot_metrics(
        result_cmaes,
        title="CMA-ES on Sphere — new plotter",
        save_path=COMPARE_DIR / "new__cmaes_metrics.png",
    )
    legacy = MultiAlgorithmPlotter()
    # The old plot_convergence_comparison expected AlgorithmChoice keys.
    legacy_results = {
        AlgorithmChoice.LBFGSB: result_mt,
        AlgorithmChoice.CMAES:  result_cmaes,
    }
    legacy.plot_convergence_comparison(
        legacy_results,
        save_path=COMPARE_DIR / "old__convergence_comparison.png",
        title="L-BFGS-B vs CMA-ES on Sphere (10D)",
    )
    legacy.plot_algorithm_specific_metrics(
        result_mt, AlgorithmChoice.LBFGSB,
        save_path=COMPARE_DIR / "old__lbfgsb_metrics.png",
    )
    legacy.plot_algorithm_specific_metrics(
        result_cmaes, AlgorithmChoice.CMAES,
        save_path=COMPARE_DIR / "old__cmaes_metrics.png",
    )

    print(f"Production plots: {PRODUCTION_DIR.absolute()}")
    print(f"Compare plots:    {COMPARE_DIR.absolute()}")


if __name__ == "__main__":
    run_lbfgsb_benchmark()
