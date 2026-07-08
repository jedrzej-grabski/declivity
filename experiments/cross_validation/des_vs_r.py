import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

from declivity.algorithms.choices import AlgorithmChoice
from declivity.algorithms.des.config import DESConfig
from declivity.core.algorithm_factory import AlgorithmFactory
from declivity.utils.constraint_handlers import BoxConstraintHandler, BoxStrategy
from declivity.utils.benchmark_functions import CEC17Function

import warnings

warnings.filterwarnings("ignore", category=SyntaxWarning, module="opfunu")


def run_python_des_cec2017():
    """Run Python DES on CEC2017 function matching R setup."""

    np.random.seed(42)

    function_id = 10  # Changed from 1 to 10
    dimensions = 10
    runs = 10
    budget = 10000 * dimensions

    print("=== Python DES on CEC2017 ===")
    print(f"Function: F{function_id}")
    print(f"Dimensions: {dimensions}")
    print(f"Runs: {runs}")
    print(f"Budget per run: {budget}")
    print("==============================\n")

    results = []

    for run in range(runs):
        print(f"Run {run+1}/{runs}")

        # Reset seed for each run to match R behavior
        np.random.seed(42 + run)

        opt_func = CEC17Function(dimensions=dimensions, function_id=function_id)

        # Use rep(50, dimensions) to match R's initial point
        initial_point = np.full(dimensions, 50.0)

        config = DESConfig(dimensions=dimensions)
        config.budget = budget
        config.population_size = 4 * dimensions
        config.enable_all_diagnostics()

        optimizer = AlgorithmFactory.create_optimizer(
            algorithm=AlgorithmChoice.DES,
            func=opt_func,
            initial_point=initial_point,
            config=config,
            lower_bounds=-100,
            upper_bounds=100,
            constraint_handler=BoxConstraintHandler(
                BoxStrategy.BOUNCE_BACK,
                np.full(dimensions, -100.0),
                np.full(dimensions, 100.0),
            ),
        )

        import time

        start_time = time.time()
        result = optimizer.optimize()
        end_time = time.time()

        results.append(
            {
                "run": run + 1,
                "best_fitness": result.best_fitness,
                "evaluations": result.evaluations,
                "runtime": end_time - start_time,
                "convergence_history": result.diagnostic.best_fitness,
                "Ft_history": (
                    result.diagnostic.ft if hasattr(result.diagnostic, "ft") else None
                ),
            }
        )

    save_python_results(results, function_id, dimensions)

    final_fitness = [r["best_fitness"] for r in results]
    print(f"\n=== PYTHON RESULTS SUMMARY ===")
    print(
        f"Best Fitness - Mean: {np.mean(final_fitness):.6e}, Median: {np.median(final_fitness):.6e}"
    )
    print(f"Best Fitness - Std: {np.std(final_fitness):.6e}")
    print(
        f"Best Fitness - Min: {np.min(final_fitness):.6e}, Max: {np.max(final_fitness):.6e}"
    )

    return results


def save_python_results(results, function_id, dimensions):

    output_dir = Path("reference/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    max_length = max(len(r["convergence_history"]) for r in results)
    convergence_matrix = np.full((max_length, len(results)), np.nan)

    for i, result in enumerate(results):
        history = result["convergence_history"]
        convergence_matrix[: len(history), i] = history

    convergence_df = pd.DataFrame(convergence_matrix)
    convergence_df.columns = [f"run_{i+1}" for i in range(len(results))]
    convergence_path = output_dir / f"python_convergence_f{function_id}_d{dimensions}.csv"
    convergence_df.to_csv(convergence_path, index=False)

    summary_data = {
        "run": [r["run"] for r in results],
        "final_fitness": [r["best_fitness"] for r in results],
        "evaluations": [r["evaluations"] for r in results],
        "runtime": [r["runtime"] for r in results],
    }
    summary_df = pd.DataFrame(summary_data)
    summary_path = output_dir / f"python_summary_f{function_id}_d{dimensions}.csv"
    summary_df.to_csv(summary_path, index=False)

    print(f"Saved Python results to: {convergence_path}, {summary_path}")


if __name__ == "__main__":
    run_python_des_cec2017()
