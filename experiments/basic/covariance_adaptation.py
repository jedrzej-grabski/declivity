"""
Visualize how the empirical covariance matrix of a population
adapts over the course of optimization.

On Sphere the covariance stays roughly spherical (condition ~ 1).
On Ellipsoid the covariance should elongate to compensate for the
different per-dimension scaling (condition ~ 10^6).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

from declivity import AlgorithmFactory
from declivity.algorithms.choices import AlgorithmChoice
from declivity.utils.benchmark_functions import BenchmarkFunction, Ellipsoid
from declivity.utils.covariance import CovarianceMatrix, empirical_covariance

plt.ioff()
plt.switch_backend("Agg")


def run(
    algorithm: AlgorithmChoice = AlgorithmChoice.CMAES,
    func: BenchmarkFunction | None = None,
    dimensions: int = 10,
):
    if func is None:
        func = Ellipsoid(dimensions=dimensions)

    func_name = type(func).__name__
    initial_point = np.full(dimensions, 50.0)

    config = AlgorithmFactory.create_config(
        algorithm, dimensions=dimensions, population_size=40
    )
    config.enable_all_diagnostics()

    optimizer = AlgorithmFactory.create_optimizer(
        algorithm=algorithm,
        func=func,
        initial_point=initial_point,
        config=config,
        lower_bounds=-100,
        upper_bounds=100,
        seed=42,
    )

    result = optimizer.optimize()
    logs = result.diagnostic

    print(f"Algorithm:    {algorithm.value}")
    print(f"Function:     {func_name}")
    print(f"Best fitness: {result.best_fitness:.6e}")
    print(f"Evaluations:  {result.evaluations}")
    print(f"Message:      {result.message}")
    print(f"Generations logged: {len(logs.population)}")

    # Compute covariance at each logged generation
    covariances: list[CovarianceMatrix] = []
    for pop in logs.population:
        covariances.append(empirical_covariance(pop))

    label = f"{algorithm.value.lower()}_{func_name.lower()}"
    title_prefix = f"{algorithm.value} on {func_name}"
    plot_eigenvalue_evolution(covariances, logs.evaluations, label, title_prefix)
    plot_condition_number(covariances, logs.evaluations, label, title_prefix)
    plot_ellipse_snapshots(
        covariances, logs.evaluations, logs.population, label, title_prefix
    )


def plot_eigenvalue_evolution(
    covariances: list[CovarianceMatrix],
    evaluations: list[int],
    label: str,
    title_prefix: str,
):
    """Show how individual eigenvalues evolve across generations."""
    max_rank = max(c.effective_rank for c in covariances)
    n_show = min(max_rank, 10)

    eigenvalue_traces = np.array(
        [c.significant_eigenvalues[:n_show] for c in covariances]
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(n_show):
        style = "-" if i < n_show - 1 else "--"
        ax.semilogy(
            evaluations,
            eigenvalue_traces[:, i],
            linewidth=2,
            linestyle=style,
            label=f"λ_{i + 1}" + (" (smallest)" if i == n_show - 1 else ""),
        )

    ax.set_xlabel("Function Evaluations")
    ax.set_ylabel("Eigenvalue (log scale)")
    ax.set_title(f"Empirical Covariance Eigenvalue Spectrum: {title_prefix}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out = Path("plots/basic/covariance_adaptation")
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{label}_eigenvalue_evolution.png", dpi=300, bbox_inches="tight")
    print(f"Saved: {out / f'{label}_eigenvalue_evolution.png'}")


def plot_condition_number(
    covariances: list[CovarianceMatrix],
    evaluations: list[int],
    label: str,
    title_prefix: str,
):
    """Condition number (from significant eigenvalues only)."""
    conditions = [c.condition_number for c in covariances]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.semilogy(evaluations, conditions, linewidth=2, color="tab:red")
    ax.set_xlabel("Function Evaluations")
    ax.set_ylabel("Condition Number (log scale)")
    ax.set_title(f"Empirical Covariance Condition Number: {title_prefix}")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out = Path("plots/basic/covariance_adaptation")
    fig.savefig(out / f"{label}_condition_sigma.png", dpi=300, bbox_inches="tight")
    print(f"Saved: {out / f'{label}_condition_sigma.png'}")


def plot_ellipse_snapshots(
    covariances: list[CovarianceMatrix],
    evaluations: list[int],
    populations: list,
    label: str,
    title_prefix: str,
):
    """
    Show 2-D projections (dims 0 and 1) of the population cloud
    and the covariance ellipse at a few snapshots in time.
    """
    n_snapshots = min(6, len(covariances))
    indices = np.linspace(0, len(covariances) - 1, n_snapshots, dtype=int)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for ax_idx, gen_idx in enumerate(indices):
        ax = axes[ax_idx]
        cov = covariances[gen_idx]
        pop = populations[gen_idx]  # (pop_size, dimensions)

        x = pop[:, 8]
        y = pop[:, 9]
        ax.scatter(x, y, s=15, alpha=0.6, zorder=2)

        sub_cov = cov.matrix[8:10, 8:10]
        _draw_ellipse(ax, cov.mean[8:10], sub_cov, n_std=2.0)

        ax.set_title(f"Eval {evaluations[gen_idx]}")
        ax.set_xlabel("x₁")
        ax.set_ylabel("x₂")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Population Cloud & Covariance Ellipse (dims 0–1): {title_prefix}",
        fontsize=14,
    )
    plt.tight_layout()

    out = Path("plots/basic/covariance_adaptation")
    fig.savefig(out / f"{label}_ellipse_snapshots.png", dpi=300, bbox_inches="tight")
    print(f"Saved: {out / f'{label}_ellipse_snapshots.png'}")


def _draw_ellipse(
    ax,
    mean: np.ndarray,
    cov_2d: np.ndarray,
    n_std: float = 2.0,
):
    eigenvalues, eigenvectors = np.linalg.eigh(cov_2d)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    width = 2 * n_std * np.sqrt(max(eigenvalues[0], 0))
    height = 2 * n_std * np.sqrt(max(eigenvalues[1], 0))

    ellipse = Ellipse(
        xy=mean,
        width=width,
        height=height,
        angle=angle,
        edgecolor="red",
        facecolor="red",
        alpha=0.15,
        linewidth=2,
        zorder=1,
    )
    ax.add_patch(ellipse)


if __name__ == "__main__":
    run()
    run(func=Ellipse(dimensions=10))
