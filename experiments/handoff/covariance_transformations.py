"""
Benchmark: CMA-ES to L-BFGS-B handoff with different covariance transformations.

After running CMA-ES, the learned covariance matrix C and step-size sigma are
extracted from the diagnostic logs and transformed into an initial Hessian for
L-BFGS-B. Compares five transformations across all four rotation levels.

All CMA-ES state is obtained through the diagnostic logger and public API,
not through private attribute access.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from declivity import AlgorithmFactory
from declivity.algorithms.choices import AlgorithmChoice
from declivity.algorithms.cmaes.config import CMAESConfig
from declivity.algorithms.lbfgsb.config import LBFGSBConfig
from declivity.plotting import (
    PanelKey,
    plot_comparison,
    plot_evaluation_bars,
    plot_matrix_diagonal_comparison,
)
from declivity.utils.benchmark_functions import Ellipsoid, RotatedEllipsoid
from declivity.utils.stopping_conditions import MaxEvaluations

# 4-panel L-BFGS-B comparison layout — the equivalent of the legacy
# plot_labeled_convergence_comparison: convergence by evals, by
# iteration, projected gradient norm, line search step length.
LBFGSB_COMPARISON_PANELS = [
    PanelKey.CONVERGENCE,
    PanelKey.CONVERGENCE_BY_ITER,
    PanelKey.PROJECTED_GRADIENT,
    PanelKey.STEP_SIZE_BY_ITER,
]

plt.ioff()
plt.switch_backend("Agg")


def build_transformations(eigenvectors, eigenvalues_sqrt, sigma, dimensions):
    """Build different Hessian matrices from the CMA-ES eigendecomposition.

    Reuses the eigenvectors B and the sqrt-eigenvalues D that CMA-ES already
    keeps internally (C = B diag(D^2) B^T). Building C and its inverse from
    them avoids an extra ``np.linalg.inv`` call on the outside.
    """
    eigenvalues = np.maximum(eigenvalues_sqrt**2, 1e-30)
    inv_eigenvalues = 1.0 / eigenvalues

    covariance_matrix = (eigenvectors * eigenvalues) @ eigenvectors.T
    covariance_inverse = (eigenvectors * inv_eigenvalues) @ eigenvectors.T

    trace_C = float(np.sum(eigenvalues))
    normalization_factor = dimensions / trace_C
    normalized_covariance = covariance_matrix * normalization_factor
    normalized_inverse = covariance_inverse / normalization_factor

    sigma_sq_inverse = covariance_inverse / (sigma * sigma)

    return {
        "Identity (no CMA-ES info)": None,
        "Raw covariance C": covariance_matrix,
        "Inverse C^{-1}": covariance_inverse,
        "Inverse scaled (s^2 C)^{-1}": sigma_sq_inverse,
        "Normalized C / tr(C) * n": normalized_covariance,
        "Inv. normalized (C/tr*n)^{-1}": normalized_inverse,
    }


TRANSFORMATION_COLORS = {
    "Identity (no CMA-ES info)": "#95a5a6",
    "Raw covariance C": "#e74c3c",
    "Inverse C^{-1}": "#3498db",
    "Inverse scaled (s^2 C)^{-1}": "#2ecc71",
    "Normalized C / tr(C) * n": "#9b59b6",
    "Inv. normalized (C/tr*n)^{-1}": "#e67e22",
}


def run_handoff_study(
    dimensions: int = 50,
    memory_size: int = 5,
    cmaes_generations: int = 300,
):
    output_dir = Path("plots/handoff/covariance_transformations")
    output_dir.mkdir(parents=True, exist_ok=True)

    rotations = [
        ("none", "No rotation"),
        ("uniform_45", "Uniform 45-degree chain"),
        ("golden", "Golden angle chain"),
        ("random", "Random orthogonal"),
    ]

    for rotation_mode, rotation_desc in rotations:
        # Build the test function
        if rotation_mode == "none":
            func = Ellipsoid(dimensions)
            scales = 10.0 ** (6.0 * np.arange(dimensions) / max(dimensions - 1, 1))

            def ellipsoid_gradient(
                x: NDArray[np.float64], s: NDArray[np.float64] = scales
            ) -> NDArray[np.float64]:
                return 2.0 * s * x

            gradient_fn = ellipsoid_gradient
            true_hessian = np.diag(2.0 * scales)
        else:
            func = RotatedEllipsoid(dimensions, rotation=rotation_mode, seed=42)
            gradient_fn = func.gradient
            true_hessian = func.hessian

        lower_bounds = -100.0
        upper_bounds = 100.0
        seed = 42
        rng = np.random.default_rng(seed)
        initial_point = rng.uniform(lower_bounds, upper_bounds, size=dimensions)

        print(f"{'=' * 60}")
        print(f"{rotation_desc} ({dimensions}D, m={memory_size})")
        print(f"Initial f(x) = {func(initial_point):.2e}")

        # Phase 1: CMA-ES warm-up
        cmaes_pop_config = CMAESConfig(dimensions=dimensions)
        evals_per_generation = cmaes_pop_config.population_size + 1
        cmaes_budget = cmaes_generations * evals_per_generation

        cmaes_config = CMAESConfig(
            dimensions=dimensions,
            sigma=10.0,
        )
        cmaes_config.diag_eigen = True

        optimizer_cmaes = AlgorithmFactory.create_optimizer(
            algorithm=AlgorithmChoice.CMAES,
            func=func,
            initial_point=initial_point,
            config=cmaes_config,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            seed=seed,
            stopping_condition=MaxEvaluations(cmaes_budget),
        )
        result_cmaes = optimizer_cmaes.optimize()

        # Extract CMA-ES state directly from the cached eigendecomposition.
        # CMA-ES already maintains B, D; reusing them avoids an outside inv().
        eigenvectors, eigenvalues_sqrt = optimizer_cmaes.get_eigendecomposition()
        sigma = optimizer_cmaes.sigma
        starting_point = optimizer_cmaes.mean
        warmup_evals = result_cmaes.evaluations

        print(
            f"CMA-ES warm-up: {warmup_evals} evals, "
            f"best={result_cmaes.best_fitness:.4e}, sigma={sigma:.4f}"
        )
        print()

        # Phase 2: L-BFGS-B with each transformation
        lbfgsb_budget = 5000
        transformations = build_transformations(
            eigenvectors, eigenvalues_sqrt, sigma, dimensions
        )

        cmaes_iters = len(result_cmaes.diagnostic.iteration)

        results = {}
        for label, hessian_matrix in transformations.items():
            config = LBFGSBConfig(
                dimensions=dimensions,
                initial_hessian=hessian_matrix,
                m=memory_size,
                pgtol=1e-8,
                factr=1e7,
            )
            config.diag_gradient_norm = True
            config.diag_step_length = True

            optimizer = AlgorithmFactory.create_optimizer(
                AlgorithmChoice.LBFGSB,
                func,
                starting_point,
                config,
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
                gradient_fn=gradient_fn,
                stopping_condition=MaxEvaluations(lbfgsb_budget),
            )
            result = optimizer.optimize()

            # Prepend full CMA-ES convergence history for continuous plot
            cmaes_log = result_cmaes.diagnostic
            for i in range(len(result.diagnostic.evaluations)):
                result.diagnostic.evaluations[i] += warmup_evals
            for i in range(len(result.diagnostic.iteration)):
                result.diagnostic.iteration[i] += cmaes_iters
            result.diagnostic.evaluations = (
                list(cmaes_log.evaluations) + result.diagnostic.evaluations
            )
            result.diagnostic.best_fitness = (
                list(cmaes_log.best_fitness) + result.diagnostic.best_fitness
            )
            result.diagnostic.iteration = (
                list(cmaes_log.iteration) + result.diagnostic.iteration
            )

            results[label] = result
            total_evals = warmup_evals + result.evaluations

            print(
                f"  {label:35s}: f={result.best_fitness:.4e}  "
                f"lbfgsb={result.evaluations:>5d}  total={total_evals:>6d}"
            )

        print()

        # Plotting
        safe_mode = rotation_mode.replace(" ", "_")

        plot_comparison(
            results,
            panels=LBFGSB_COMPARISON_PANELS,
            colors=TRANSFORMATION_COLORS,
            title=(
                f"CMA-ES -> L-BFGS-B: Covariance Transformations\n"
                f"{rotation_desc} ({dimensions}D, m={memory_size}, "
                f"{cmaes_generations} CMA-ES gen)"
            ),
            save_path=output_dir / f"{safe_mode}_{dimensions}d_convergence.png",
            handoff_eval=warmup_evals,
            handoff_iter=cmaes_iters,
        )
        plot_evaluation_bars(
            results,
            colors=TRANSFORMATION_COLORS,
            title=(f"Total Evaluations: {rotation_desc} ({dimensions}D)"),
            save_path=output_dir / f"{safe_mode}_{dimensions}d_bar.png",
        )

        # Diagonal profile comparison against true Hessian
        non_identity = {
            label: matrix
            for label, matrix in transformations.items()
            if matrix is not None
        }
        plot_matrix_diagonal_comparison(
            non_identity,
            reference=true_hessian,
            reference_label="True Hessian",
            title=(
                f"Diagonal Profile: True Hessian vs Passed Matrix\n"
                f"{rotation_desc} ({dimensions}D)"
            ),
            save_path=output_dir / f"{safe_mode}_{dimensions}d_diag_compare.png",
        )

        plt.close("all")

    print(f"All plots saved to: {output_dir.absolute()}")


if __name__ == "__main__":
    import sys

    dims = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    mem = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    gens = int(sys.argv[3]) if len(sys.argv) > 3 else 300
    run_handoff_study(dimensions=dims, memory_size=mem, cmaes_generations=gens)
