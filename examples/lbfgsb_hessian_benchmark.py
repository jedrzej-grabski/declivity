"""
Benchmark: Impact of initial Hessian approximation on L-BFGS-B convergence.

Compares different initial Hessian choices on the Ellipsoid function (condition
number 10^6), and demonstrates the effect of persist_initial_hessian.
"""

import pprint

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from src import AlgorithmFactory
from src.algorithms.choices import AlgorithmChoice
from src.algorithms.lbfgsb.config import LBFGSBConfig
from src.utils.benchmark_functions import Ellipsoid
from src.plotting.multi_algorithm_plotter import MultiAlgorithmPlotter

plt.ioff()
plt.switch_backend("Agg")


def run_single(func, x0, config, grad):
    config.diag_bestVal = True
    config.diag_gradient_norm = True
    config.diag_step_length = True
    optimizer = AlgorithmFactory.create_optimizer(
        AlgorithmChoice.LBFGSB,
        func,
        x0,
        config,
        lower_bounds=-50.0,
        upper_bounds=50.0,
        gradient_fn=grad,
    )
    return optimizer.optimize()


def run_hessian_benchmark():
    dimensions = 10
    func = Ellipsoid(dimensions=dimensions)
    x0 = np.full(dimensions, 10.0)
    budget = 15000

    scales = 10.0 ** (6.0 * np.arange(dimensions) / (dimensions - 1))
    exact_diag = 2.0 * scales

    inverse_diag = 1 / exact_diag

    def ellipsoid_grad(x):
        return 2.0 * scales * x

    print(f"Ellipsoid benchmark (d={dimensions}, condition number = 1e6)")
    print(f"Hessian diagonal ranges from {exact_diag[0]:.0f} to {exact_diag[-1]:.0e}")
    print()

    output_dir = Path("plots/lbfgsb/hessian_study")
    output_dir.mkdir(parents=True, exist_ok=True)
    plotter = MultiAlgorithmPlotter()

    # Benchmark 1: different initial Hessian choices
    print("Benchmark 1: Initial Hessian choices")

    configs_1 = {
        "Identity (default)": LBFGSBConfig(
            dimensions=dimensions, pgtol=1e-12, factr=0, budget=budget
        ),
        "Exact Hessian diagonal": LBFGSBConfig(
            dimensions=dimensions,
            initial_hessian=exact_diag,
            pgtol=1e-12,
            factr=0,
            budget=budget,
        ),
        "Inverse Hessian": LBFGSBConfig(
            dimensions=dimensions,
            initial_hessian=inverse_diag,
            pgtol=1e-12,
            factr=0,
            budget=budget,
        ),
    }
    print(exact_diag)
    pprint.pprint(float(np.mean(exact_diag)))
    results_1 = {}
    for label, config in configs_1.items():
        result = run_single(func, x0, config, ellipsoid_grad)
        results_1[label] = result
        print(
            f"  {label:30s}: f={result.best_fitness:.4e}  evals={result.evaluations:>6d}"
        )

    colors_1 = {
        "Identity (default)": "#e74c3c",
        "Exact Hessian diagonal": "#2ecc71",
        "Scalar (mean curvature)": "#3498db",
    }

    plotter.plot_labeled_convergence_comparison(
        results_1,
        colors_1,
        title=f"Impact of Initial Hessian on L-BFGS-B Convergence\n"
        f"Ellipsoid ({dimensions}D, condition number $10^6$)",
        save_path=output_dir / "lbfgsb_initial_hessian_benchmark.png",
    )
    plotter.plot_evaluation_bar_chart(
        results_1,
        colors_1,
        title="L-BFGS-B on Ellipsoid (10D): Evaluations by Initial Hessian Choice",
        save_path=output_dir / "lbfgsb_initial_hessian_bar.png",
    )

    # Benchmark 2: persist_initial_hessian True vs False
    print()
    print("Benchmark 2: persist_initial_hessian flag")

    configs_2 = {
        "Exact diag, persist=True": LBFGSBConfig(
            dimensions=dimensions,
            initial_hessian=exact_diag,
            persist_initial_hessian=True,
            pgtol=1e-12,
            factr=0,
            budget=budget,
        ),
        "Exact diag, persist=False": LBFGSBConfig(
            dimensions=dimensions,
            initial_hessian=exact_diag,
            persist_initial_hessian=False,
            pgtol=1e-12,
            factr=0,
            budget=budget,
        ),
        "Identity (no Hessian info)": LBFGSBConfig(
            dimensions=dimensions, pgtol=1e-12, factr=0, budget=budget
        ),
    }

    results_2 = {}
    for label, config in configs_2.items():
        result = run_single(func, x0, config, ellipsoid_grad)
        results_2[label] = result
        print(
            f"  {label:30s}: f={result.best_fitness:.4e}  evals={result.evaluations:>6d}"
        )

    colors_2 = {
        "Exact diag, persist=True": "#2ecc71",
        "Exact diag, persist=False": "#f39c12",
        "Identity (no Hessian info)": "#e74c3c",
    }

    plotter.plot_labeled_convergence_comparison(
        results_2,
        colors_2,
        title=f"persist_initial_hessian: True vs False\n"
        f"Ellipsoid ({dimensions}D, exact diagonal provided)",
        save_path=output_dir / "lbfgsb_persist_hessian_benchmark.png",
    )
    plotter.plot_evaluation_bar_chart(
        results_2,
        colors_2,
        title="persist_initial_hessian: Impact on Convergence (Ellipsoid 10D)",
        save_path=output_dir / "lbfgsb_persist_hessian_bar.png",
    )


if __name__ == "__main__":
    run_hessian_benchmark()
