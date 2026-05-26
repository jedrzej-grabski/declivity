"""Single-algorithm diagnostic plot demo on a 10D Sphere.

Runs each of the four algorithms (DES, CMA-ES, MF-CMA-ES, L-BFGS-B)
and dumps every default diagnostic panel registered for it.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src import AlgorithmFactory
from src.algorithms.choices import AlgorithmChoice
from src.plotting import plot_metrics
from src.utils.benchmark_functions import Sphere
from src.utils.boundary_handlers import BoundaryHandlerType


plt.ioff()
plt.switch_backend("Agg")


OUTPUT_DIR = Path("plots/basic/simple_optimization")
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

    plot_metrics(
        result,
        title=f"{algorithm.value} on 10D Sphere",
        save_path=OUTPUT_DIR / f"{algorithm.value.lower()}_metrics.png",
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for algorithm in ALGORITHMS:
        run_one(algorithm)
    print(f"\nOutput: {OUTPUT_DIR.absolute()}")


if __name__ == "__main__":
    main()
