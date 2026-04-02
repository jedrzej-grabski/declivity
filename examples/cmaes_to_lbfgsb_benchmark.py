import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from src import AlgorithmFactory
from src.algorithms.choices import AlgorithmChoice
from src.algorithms.cmaes.config import CMAESConfig
from src.algorithms.lbfgsb.config import LBFGSBConfig
from src.utils.benchmark_functions import Ellipsoid
from src.plotting.multi_algorithm_plotter import MultiAlgorithmPlotter

plt.ioff()
plt.switch_backend("Agg")


def run_benchmark(dimensions: int = 10, cmaes_generations: int = 200):
    func = Ellipsoid(dimensions=dimensions)
    lower_bounds = -50.0
    upper_bounds = 50.0
    seed = 42

    rng = np.random.default_rng(seed)
    initial_point = rng.uniform(lower_bounds, upper_bounds, size=dimensions)

    scales = 10.0 ** (6.0 * np.arange(dimensions) / (dimensions - 1))

    def ellipsoid_grad(x):
        return 2.0 * scales * x

    cmaes_config = CMAESConfig(dimensions=dimensions)
    # Each CMA-ES generation costs population_size + 1 evaluations
    # (the +1 is the mean fitness evaluation used for logging)
    evals_per_generation = cmaes_config.population_size + 1
    cmaes_budget = cmaes_generations * evals_per_generation

    print(f"Ellipsoid (d={dimensions}, condition number = 1e6)")
    print(
        f"CMA-ES warm-up: {cmaes_generations} generations "
        f"({cmaes_budget} evaluations, pop_size={cmaes_config.population_size})"
    )
    print(f"Initial f(x) = {func(initial_point):.2f}")
    print()

    cmaes_config_warmup = CMAESConfig(
        dimensions=dimensions,
        budget=cmaes_budget,
        sigma=10.0,
    )
    cmaes_config_warmup.diag_bestVal = True

    optimizer_cmaes_warmup = AlgorithmFactory.create_optimizer(
        algorithm=AlgorithmChoice.CMAES,
        func=func,
        initial_point=initial_point,
        config=cmaes_config_warmup,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        seed=seed,
    )
    result_cmaes_warmup = optimizer_cmaes_warmup.optimize()

    # Extract the learned covariance matrix from CMA-ES
    cma_internal = optimizer_cmaes_warmup._cma
    covariance_matrix = cma_internal._C
    sigma = cma_internal._sigma
    cmaes_mean = cma_internal._mean

    # The CMA-ES distribution is N(mean, sigma^2 * C).
    # The covariance diagonal scaled by sigma^2 reflects the learned variance
    # per variable. The Hessian is roughly proportional to the inverse of the
    # covariance: high variance = low curvature and vice versa.
    # We use 1 / (sigma^2 * diag(C)) as the Hessian diagonal estimate.
    covariance_diagonal = sigma**2 * np.diag(covariance_matrix)
    hessian_from_cmaes = 1.0 / covariance_diagonal

    print(f"CMA-ES warm-up complete:")
    print(f"  Best fitness after warm-up: {result_cmaes_warmup.best_fitness:.4e}")
    print(f"  Evaluations used: {result_cmaes_warmup.evaluations}")
    print(f"  Sigma: {sigma:.4f}")
    print(
        f"  Learned Hessian diagonal range: "
        f"[{hessian_from_cmaes.min():.2e}, {hessian_from_cmaes.max():.2e}]"
    )
    print(f"  True Hessian diagonal range: [{2*scales[0]:.2e}, {2*scales[-1]:.2e}]")
    print()

    # Phase 2: L-BFGS-B warm-started from CMA-ES covariance
    lbfgsb_budget = 15000
    starting_point = cmaes_mean.copy()
    norm = np.linalg.norm(covariance_diagonal)
    new_diag = covariance_diagonal / norm
    print(new_diag)
    print(hessian_from_cmaes)
    config_warm = LBFGSBConfig(
        dimensions=dimensions,
        initial_hessian=hessian_from_cmaes,
        persist_initial_hessian=True,
        pgtol=1e-12,
        factr=0,
        budget=lbfgsb_budget,
    )
    config_warm.diag_bestVal = True
    config_warm.diag_gradient_norm = True
    config_warm.diag_step_length = True

    optimizer_warm = AlgorithmFactory.create_optimizer(
        AlgorithmChoice.LBFGSB,
        func,
        starting_point,
        config_warm,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        gradient_fn=ellipsoid_grad,
    )
    result_warm = optimizer_warm.optimize()

    # Prepend the CMA-ES warm-up evaluations to the L-BFGS-B logs so the
    # convergence plot shows the total cost
    warmup_evals = result_cmaes_warmup.evaluations
    for i in range(len(result_warm.diagnostic.evaluations)):
        result_warm.diagnostic.evaluations[i] += warmup_evals
    # Insert the CMA-ES best fitness as the starting point of the L-BFGS-B curve
    result_warm.diagnostic.evaluations.insert(0, warmup_evals)
    result_warm.diagnostic.best_fitness.insert(0, result_cmaes_warmup.best_fitness)
    result_warm.diagnostic.iteration.insert(0, 0)

    total_warm_evals = warmup_evals + result_warm.evaluations

    print(f"L-BFGS-B (warm-started from CMA-ES covariance):")
    print(f"  Best fitness: {result_warm.best_fitness:.4e}")
    print(f"  L-BFGS-B evaluations: {result_warm.evaluations}")
    print(f"  Total evaluations (CMA-ES + L-BFGS-B): {total_warm_evals}")
    print()

    # Phase 3: Cold-start L-BFGS-B from the same CMA-ES mean (no Hessian info)
    config_cold = LBFGSBConfig(
        dimensions=dimensions,
        pgtol=1e-12,
        factr=0,
        budget=lbfgsb_budget,
    )
    config_cold.diag_bestVal = True
    config_cold.diag_gradient_norm = True
    config_cold.diag_step_length = True

    optimizer_cold = AlgorithmFactory.create_optimizer(
        AlgorithmChoice.LBFGSB,
        func,
        starting_point,
        config_cold,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        gradient_fn=ellipsoid_grad,
    )
    result_cold = optimizer_cold.optimize()

    # Offset cold-start evaluations by the same warm-up cost for fair comparison
    for i in range(len(result_cold.diagnostic.evaluations)):
        result_cold.diagnostic.evaluations[i] += warmup_evals
    result_cold.diagnostic.evaluations.insert(0, warmup_evals)
    result_cold.diagnostic.best_fitness.insert(0, result_cmaes_warmup.best_fitness)
    result_cold.diagnostic.iteration.insert(0, 0)

    total_cold_evals = warmup_evals + result_cold.evaluations

    print(f"L-BFGS-B (cold-start, identity Hessian, same starting point):")
    print(f"  Best fitness: {result_cold.best_fitness:.4e}")
    print(f"  L-BFGS-B evaluations: {result_cold.evaluations}")
    print(f"  Total evaluations (CMA-ES + L-BFGS-B): {total_cold_evals}")
    print()

    # Phase 4: Standalone CMA-ES running to full convergence
    cmaes_full_budget = total_cold_evals + 5000
    config_cmaes_full = CMAESConfig(
        dimensions=dimensions,
        budget=cmaes_full_budget,
        sigma=10.0,
    )
    config_cmaes_full.diag_bestVal = True

    optimizer_cmaes_full = AlgorithmFactory.create_optimizer(
        algorithm=AlgorithmChoice.CMAES,
        func=func,
        initial_point=initial_point,
        config=config_cmaes_full,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        seed=seed,
    )
    result_cmaes_full = optimizer_cmaes_full.optimize()

    print(f"CMA-ES (standalone, full run):")
    print(f"  Best fitness: {result_cmaes_full.best_fitness:.4e}")
    print(f"  Evaluations: {result_cmaes_full.evaluations}")
    print()

    # Plotting
    output_dir = Path("plots")
    output_dir.mkdir(exist_ok=True)
    plotter = MultiAlgorithmPlotter()

    results = {
        f"CMA-ES ({cmaes_generations}gen) + L-BFGS-B (warm)": result_warm,
        f"CMA-ES ({cmaes_generations}gen) + L-BFGS-B (cold)": result_cold,
        "CMA-ES standalone": result_cmaes_full,
    }
    colors = {
        f"CMA-ES ({cmaes_generations}gen) + L-BFGS-B (warm)": "#2ecc71",
        f"CMA-ES ({cmaes_generations}gen) + L-BFGS-B (cold)": "#f39c12",
        "CMA-ES standalone": "#e74c3c",
    }

    plotter.plot_labeled_convergence_comparison(
        results,
        colors,
        title=f"CMA-ES Warm-Starting L-BFGS-B\n"
        f"Ellipsoid ({dimensions}D, condition number $10^6$)",
        save_path=output_dir / "cmaes_to_lbfgsb_convergence.png",
    )
    plotter.plot_evaluation_bar_chart(
        {
            f"CMA-ES + L-BFGS-B (warm)": result_warm,
            f"CMA-ES + L-BFGS-B (cold)": result_cold,
            "CMA-ES standalone": result_cmaes_full,
        },
        {
            f"CMA-ES + L-BFGS-B (warm)": "#2ecc71",
            f"CMA-ES + L-BFGS-B (cold)": "#f39c12",
            "CMA-ES standalone": "#e74c3c",
        },
        title=f"Total Evaluations: CMA-ES Warm-Start vs Standalone ({dimensions}D Ellipsoid)",
        save_path=output_dir / "cmaes_to_lbfgsb_bar.png",
    )

    # Hessian comparison: what CMA-ES learned vs the truth
    fig, ax = plt.subplots(figsize=(10, 6))
    true_hessian_diag = 2.0 * scales
    variable_indices = np.arange(1, dimensions + 1)

    ax.semilogy(
        variable_indices,
        true_hessian_diag,
        "k-o",
        linewidth=2,
        markersize=8,
        label="True Hessian diagonal",
        zorder=3,
    )
    ax.semilogy(
        variable_indices,
        hessian_from_cmaes,
        "g--s",
        linewidth=2,
        markersize=8,
        label=f"CMA-ES estimate ({cmaes_generations} gen)",
        zorder=3,
    )

    ax.set_xlabel("Variable index", fontsize=13)
    ax.set_ylabel("Hessian diagonal value (log scale)", fontsize=13)
    ax.set_title(
        f"Hessian Diagonal: True vs CMA-ES Estimate\n"
        f"Ellipsoid ({dimensions}D, {cmaes_generations} CMA-ES generations)",
        fontsize=14,
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(variable_indices)
    plt.tight_layout()
    fig.savefig(
        output_dir / "cmaes_to_lbfgsb_hessian_estimate.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(f"Plots saved to: {output_dir.absolute()}")


if __name__ == "__main__":
    import sys

    dims = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    gens = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    run_benchmark(dimensions=dims, cmaes_generations=gens)
