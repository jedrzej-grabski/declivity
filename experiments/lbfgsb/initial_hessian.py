"""Benchmark: impact of initial Hessian approximation on L-BFGS-B convergence.

Compares different initial Hessian choices on the Ellipsoid function
(condition number 10^6), then a second sweep over
``persist_initial_hessian`` True vs False.

Produces, per sub-benchmark, a 4-panel convergence comparison
(convergence by evals, by iteration, projected gradient norm, line
search step length) and a horizontal bar chart of total evaluations.
"""

import pprint
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from declivity import AlgorithmFactory
from declivity.algorithms.choices import AlgorithmChoice
from declivity.algorithms.lbfgsb.config import LBFGSBConfig
from declivity.plotting import (
    PanelKey,
    plot_comparison,
    plot_evaluation_bars,
)
from declivity.utils.benchmark_functions import Ellipsoid
from declivity.utils.stopping_conditions import MaxEvaluations

plt.ioff()
plt.switch_backend("Agg")


OUTPUT_DIR = Path("plots/lbfgsb/initial_hessian")


# The 4-panel layout that the legacy plot_labeled_convergence_comparison
# produced: convergence by evals, by iteration, projected gradient, step
# length. Reusable across both sub-benchmarks below.
LBFGSB_COMPARISON_PANELS = [
    PanelKey.CONVERGENCE,
    PanelKey.CONVERGENCE_BY_ITER,
    PanelKey.PROJECTED_GRADIENT,
    PanelKey.STEP_SIZE_BY_ITER,
]


def run_single(func, x0, config, grad, stopping_condition):
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
        stopping_condition=stopping_condition,
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
    print(f"Hessian diagonal ranges from {exact_diag[0]:.0f} to {exact_diag[-1]:.0e}\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Benchmark 1: different initial Hessian choices ----------------------
    print("Benchmark 1: Initial Hessian choices")

    configs_1 = {
        "Identity (default)": LBFGSBConfig(
            dimensions=dimensions,
            pgtol=1e-12,
            factr=0,
        ),
        "Exact Hessian diagonal": LBFGSBConfig(
            dimensions=dimensions,
            initial_hessian=exact_diag,
            pgtol=1e-12,
            factr=0,
        ),
        "Inverse Hessian": LBFGSBConfig(
            dimensions=dimensions,
            initial_hessian=inverse_diag,
            pgtol=1e-12,
            factr=0,
        ),
    }
    print(exact_diag)
    pprint.pprint(float(np.mean(exact_diag)))
    results_1 = {}
    for label, config in configs_1.items():
        result = run_single(func, x0, config, ellipsoid_grad, MaxEvaluations(budget))
        results_1[label] = result
        print(
            f"  {label:30s}: f={result.best_fitness:.4e}  evals={result.evaluations:>6d}"
        )

    colors_1 = {
        "Identity (default)": "#e74c3c",
        "Exact Hessian diagonal": "#2ecc71",
        "Inverse Hessian": "#3498db",
    }

    plot_comparison(
        results_1,
        panels=LBFGSB_COMPARISON_PANELS,
        colors=colors_1,
        title=(
            f"Impact of Initial Hessian on L-BFGS-B Convergence\n"
            f"Ellipsoid ({dimensions}D, condition number $10^6$)"
        ),
        save_path=OUTPUT_DIR / "lbfgsb_initial_hessian_benchmark.png",
    )
    plot_evaluation_bars(
        results_1,
        colors=colors_1,
        title="L-BFGS-B on Ellipsoid (10D): Evaluations by Initial Hessian Choice",
        save_path=OUTPUT_DIR / "lbfgsb_initial_hessian_bar.png",
    )

    # --- Benchmark 2: persist_initial_hessian True vs False ------------------
    print()
    print("Benchmark 2: persist_initial_hessian flag")

    configs_2 = {
        "Exact diag, persist=True": LBFGSBConfig(
            dimensions=dimensions,
            initial_hessian=exact_diag,
            persist_initial_hessian=True,
            pgtol=1e-12,
            factr=0,
        ),
        "Exact diag, persist=False": LBFGSBConfig(
            dimensions=dimensions,
            initial_hessian=exact_diag,
            persist_initial_hessian=False,
            pgtol=1e-12,
            factr=0,
        ),
        "Identity (no Hessian info)": LBFGSBConfig(
            dimensions=dimensions,
            pgtol=1e-12,
            factr=0,
        ),
    }

    results_2 = {}
    for label, config in configs_2.items():
        result = run_single(func, x0, config, ellipsoid_grad, MaxEvaluations(budget))
        results_2[label] = result
        print(
            f"  {label:30s}: f={result.best_fitness:.4e}  evals={result.evaluations:>6d}"
        )

    colors_2 = {
        "Exact diag, persist=True": "#2ecc71",
        "Exact diag, persist=False": "#f39c12",
        "Identity (no Hessian info)": "#e74c3c",
    }

    plot_comparison(
        results_2,
        panels=LBFGSB_COMPARISON_PANELS,
        colors=colors_2,
        title=(
            f"persist_initial_hessian: True vs False\n"
            f"Ellipsoid ({dimensions}D, exact diagonal provided)"
        ),
        save_path=OUTPUT_DIR / "lbfgsb_persist_hessian_benchmark.png",
    )
    plot_evaluation_bars(
        results_2,
        colors=colors_2,
        title="persist_initial_hessian: Impact on Convergence (Ellipsoid 10D)",
        save_path=OUTPUT_DIR / "lbfgsb_persist_hessian_bar.png",
    )

    print(f"\nPlots saved to: {OUTPUT_DIR.absolute()}")


if __name__ == "__main__":
    run_hessian_benchmark()
