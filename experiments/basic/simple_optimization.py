"""Single-algorithm diagnostic plot demo on a 10D Sphere.

Runs each algorithm and dumps every registered diagnostic panel.
Production output goes to ``plots/basic/simple_optimization/``.

During the plotter migration, this script also writes ``old__`` and
``new__`` copies of each metrics plot into ``plots/_migration_compare/
simple_optimization/`` for side-by-side review. The legacy plotter call
(marked ``# MIGRATION COMPARE``) goes away once the new output is
approved.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src import AlgorithmFactory
from src.algorithms.choices import AlgorithmChoice
from src.plotting import plot_metrics
from src.plotting.multi_algorithm_plotter import MultiAlgorithmPlotter
from src.utils.benchmark_functions import Sphere
from src.utils.boundary_handlers import BoundaryHandlerType


plt.ioff()
plt.switch_backend("Agg")


PRODUCTION_DIR = Path("plots/basic/simple_optimization")
COMPARE_DIR = Path("plots/_migration_compare/simple_optimization")
ALGORITHMS = (
    AlgorithmChoice.DES,
    AlgorithmChoice.CMAES,
    AlgorithmChoice.MFCMAES,
    AlgorithmChoice.LBFGSB,
)


def run_one(algorithm: AlgorithmChoice) -> None:
    """Run one algorithm on 10D Sphere and save every diagnostic panel."""
    dimensions = 10
    objective = Sphere(dimensions=dimensions)

    rng = np.random.default_rng(0)
    initial_point = rng.uniform(-50.0, 50.0, size=dimensions)

    config = AlgorithmFactory.create_config(algorithm, dimensions=dimensions)
    config.enable_all_diagnostics()

    print(f"\nStarting {algorithm.value}...")
    print(f"  Budget: {config.budget}")
    print(f"  Initial f(x0): {objective(initial_point):.4e}")

    optimizer = AlgorithmFactory.create_optimizer(
        algorithm=algorithm,
        func=objective,
        initial_point=initial_point,
        config=config,
        lower_bounds=-50.12,
        upper_bounds=50.12,
        boundary_strategy=BoundaryHandlerType.CLAMP,
        seed=42,
    )
    result = optimizer.optimize()

    print(f"  Final f: {result.best_fitness:.4e} after {result.evaluations} evals")

    filename = f"{algorithm.value.lower()}_metrics.png"

    # Production output (new plotter).
    plot_metrics(
        result,
        title=f"{algorithm.value} on 10D Sphere",
        save_path=PRODUCTION_DIR / filename,
    )

    # MIGRATION COMPARE — remove once approved.
    plot_metrics(
        result,
        title=f"{algorithm.value} on 10D Sphere (new plotter)",
        save_path=COMPARE_DIR / f"new__{filename}",
    )
    MultiAlgorithmPlotter().plot_algorithm_specific_metrics(
        result, algorithm, save_path=COMPARE_DIR / f"old__{filename}"
    )


def main() -> None:
    PRODUCTION_DIR.mkdir(parents=True, exist_ok=True)
    COMPARE_DIR.mkdir(parents=True, exist_ok=True)
    for algorithm in ALGORITHMS:
        run_one(algorithm)
    print(f"\nProduction plots: {PRODUCTION_DIR.absolute()}")
    print(f"Compare plots:    {COMPARE_DIR.absolute()}")


if __name__ == "__main__":
    main()
