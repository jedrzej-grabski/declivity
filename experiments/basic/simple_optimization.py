from typing import cast
import matplotlib.pyplot as plt
from pathlib import Path

from src import AlgorithmFactory

from src.algorithms.choices import AlgorithmChoice
from src.algorithms.cmaes.config import CMAESConfig
from src.algorithms.des.config import DESConfig
from src.algorithms.mfcmaes.mfcmaes_config import MFCMAESConfig
from src.utils.boundary_handlers import BoundaryHandlerType
from src.utils.benchmark_functions import CEC17Function, Sphere
from src.utils.initial_point_generator import (
    InitialPointGenerator,
    InitialPointGeneratorType,
)
from src.plotting.multi_algorithm_plotter import MultiAlgorithmPlotter

import warnings

warnings.filterwarnings("ignore", category=SyntaxWarning, module="opfunu")

plt.ioff()
plt.switch_backend("Agg")


def run_optimization_example(algorithm: AlgorithmChoice):
    """Run a simple optimization example using the new architecture."""

    dimensions = 10
    # opt_func = CEC17Function(dimensions=dimensions, function_id=1)
    opt_func = Sphere(dimensions=dimensions)

    lower_bounds = -50.12
    upper_bounds = 50.12

    initial_point_generator = InitialPointGenerator(
        strategy=InitialPointGeneratorType.UNIFORM,
        dimensions=dimensions,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
    )

    initial_point = initial_point_generator.generate()

    config = AlgorithmFactory.create_config(algorithm, dimensions=dimensions)

    config = cast(MFCMAESConfig, config)

    # Disable PPMF to test with constant sigma
    config.use_ppmf = True

    config.enable_all_diagnostics()

    print(f"Starting {algorithm.value} optimization...")
    print(f"Dimensions: {dimensions}")
    print(f"Budget: {config.budget}")
    print(f"Population size: {config.population_size}")
    print(f"PPMF enabled: {config.use_ppmf}")
    print(f"Initial point value: {opt_func(initial_point):.20f}")
    print(f"Configuration: {config}")

    optimizer = AlgorithmFactory.create_optimizer(
        algorithm=algorithm,
        func=opt_func,
        initial_point=initial_point,
        config=config,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        boundary_strategy=BoundaryHandlerType.CLAMP,
    )

    result = optimizer.optimize()

    print("\nOptimization completed:")
    print(f"Best fitness: {result.best_fitness:.20f}")
    print(f"Function evaluations: {result.evaluations}")
    print(f"Message: {result.message}")
    print(f"Algorithm: {result.algorithm}")

    output_dir = Path("plots/basic/simple_optimization")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving plots to: {output_dir.absolute()}")

    plotter = MultiAlgorithmPlotter()

    metrics_path = output_dir / f"{algorithm.value.lower()}_metrics.png"
    _ = plotter.plot_algorithm_specific_metrics(
        result, algorithm, save_path=metrics_path
    )

    print(f"Saved {algorithm.value} metrics plot to: {metrics_path}")


if __name__ == "__main__":
    run_optimization_example(AlgorithmChoice.DES)
