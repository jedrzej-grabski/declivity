"""
L-BFGS-B benchmark on Sphere function.

Compares L-BFGS-B (More-Thuente and Armijo line search) against CMA-ES.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from src import AlgorithmFactory
from src.algorithms.choices import AlgorithmChoice
from src.algorithms.lbfgsb.config import LBFGSBConfig, LineSearchMethod
from src.utils.benchmark_functions import Sphere
from src.plotting.multi_algorithm_plotter import MultiAlgorithmPlotter

plt.ioff()
plt.switch_backend("Agg")


def run_lbfgsb_benchmark():
    dimensions = 10
    opt_func = Sphere(dimensions=dimensions)
    lower_bounds = -50.0
    upper_bounds = 50.0
    seed = 42

    rng = np.random.default_rng(seed)
    initial_point = rng.uniform(lower_bounds, upper_bounds, size=dimensions)

    print(f"Sphere function benchmark (d={dimensions})")
    print(f"Initial point f(x) = {opt_func(initial_point):.6f}")
    print(f"Bounds: [{lower_bounds}, {upper_bounds}]")
    print()

    convergence_results = {}

    config_mt = LBFGSBConfig(
        dimensions=dimensions, m=10, line_search=LineSearchMethod.MORE_THUENTE,
    )
    config_mt.enable_all_diagnostics()

    optimizer_mt = AlgorithmFactory.create_optimizer(
        algorithm=AlgorithmChoice.LBFGSB, func=opt_func,
        initial_point=initial_point, config=config_mt,
        lower_bounds=lower_bounds, upper_bounds=upper_bounds, seed=seed,
    )
    result_mt = optimizer_mt.optimize()
    convergence_results[AlgorithmChoice.LBFGSB] = result_mt

    print(f"L-BFGS-B (More-Thuente):")
    print(f"  Best fitness:  {result_mt.best_fitness:.12e}")
    print(f"  Evaluations:   {result_mt.evaluations}")
    print(f"  Message:       {result_mt.message}")
    print()

    # L-BFGS-B with Armijo line search
    config_armijo = LBFGSBConfig(
        dimensions=dimensions, m=10, line_search=LineSearchMethod.ARMIJO,
    )
    config_armijo.enable_all_diagnostics()

    optimizer_armijo = AlgorithmFactory.create_optimizer(
        algorithm=AlgorithmChoice.LBFGSB, func=opt_func,
        initial_point=initial_point, config=config_armijo,
        lower_bounds=lower_bounds, upper_bounds=upper_bounds, seed=seed,
    )
    result_armijo = optimizer_armijo.optimize()

    print(f"L-BFGS-B (Armijo):")
    print(f"  Best fitness:  {result_armijo.best_fitness:.12e}")
    print(f"  Evaluations:   {result_armijo.evaluations}")
    print(f"  Message:       {result_armijo.message}")
    print()

    # CMA-ES for comparison
    config_cmaes = AlgorithmFactory.create_config(
        AlgorithmChoice.CMAES, dimensions=dimensions,
    )
    config_cmaes.enable_all_diagnostics()

    optimizer_cmaes = AlgorithmFactory.create_optimizer(
        algorithm=AlgorithmChoice.CMAES, func=opt_func,
        initial_point=initial_point, config=config_cmaes,
        lower_bounds=lower_bounds, upper_bounds=upper_bounds, seed=seed,
    )
    result_cmaes = optimizer_cmaes.optimize()
    convergence_results[AlgorithmChoice.CMAES] = result_cmaes

    print(f"CMA-ES:")
    print(f"  Best fitness:  {result_cmaes.best_fitness:.12e}")
    print(f"  Evaluations:   {result_cmaes.evaluations}")
    print(f"  Message:       {result_cmaes.message}")
    print()

    # Plotting
    output_dir = Path("plots/lbfgsb")
    output_dir.mkdir(parents=True, exist_ok=True)
    plotter = MultiAlgorithmPlotter()

    plotter.plot_convergence_comparison(
        convergence_results,
        save_path=output_dir / "lbfgsb_convergence_comparison.png",
        title="L-BFGS-B vs CMA-ES on Sphere (10D)",
    )
    plotter.plot_algorithm_specific_metrics(
        result_mt, AlgorithmChoice.LBFGSB,
        save_path=output_dir / "lbfgsb_metrics.png",
    )
    plotter.plot_algorithm_specific_metrics(
        result_cmaes, AlgorithmChoice.CMAES,
        save_path=output_dir / "cmaes_sphere_metrics.png",
    )

    print(f"Plots saved to: {output_dir.absolute()}")


if __name__ == "__main__":
    run_lbfgsb_benchmark()
