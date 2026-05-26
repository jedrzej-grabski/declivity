"""
Benchmark: Full Hessian matrix vs diagonal vs identity on rotated Ellipsoids.

Tests how off-diagonal curvature information affects L-BFGS-B convergence
across four rotation levels (none, uniform 45-degree, golden angle, random)
and two dimensionality regimes (n=10 with m=10, and n=50 with m=5).

The n=50, m=5 regime is the decisive case: the L-BFGS corrections can only
span 10 of 50 directions, so the remaining 40 rely on B_0.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from src import AlgorithmFactory
from src.algorithms.choices import AlgorithmChoice
from src.algorithms.lbfgsb.config import LBFGSBConfig
from src.utils.benchmark_functions import Ellipsoid, RotatedEllipsoid
from src.plotting import (
    PanelKey,
    plot_comparison,
    plot_evaluation_bars,
    plot_function_landscape,
    plot_function_landscape_grid,
)


# 4-panel L-BFGS-B comparison layout (convergence by evals, by iteration,
# projected gradient, line search step).
LBFGSB_COMPARISON_PANELS = [
    PanelKey.CONVERGENCE,
    PanelKey.CONVERGENCE_BY_ITER,
    PanelKey.PROJECTED_GRADIENT,
    PanelKey.STEP_SIZE_BY_ITER,
]

plt.ioff()
plt.switch_backend("Agg")


def run_single(func, x0, config, gradient_fn, lower_bounds, upper_bounds):
    config.diag_gradient_norm = True
    config.diag_step_length = True
    optimizer = AlgorithmFactory.create_optimizer(
        AlgorithmChoice.LBFGSB, func, x0, config,
        lower_bounds=lower_bounds, upper_bounds=upper_bounds,
        gradient_fn=gradient_fn,
    )
    return optimizer.optimize()


def run_rotation_study():
    output_dir = Path("plots/lbfgsb/rotation_study")
    output_dir.mkdir(parents=True, exist_ok=True)

    rotations = [
        ("none",       "No rotation (axis-aligned)"),
        ("uniform_45", "Uniform 45-degree chain"),
        ("golden",     "Golden angle chain"),
        ("random",     "Random orthogonal"),
    ]

    # Landscape plots: show the contour shape for each rotation (using n=10 for clarity)
    print("Generating function landscape plots...")
    landscape_n = 10
    landscape_functions = {}
    landscape_hessians = {}

    for rotation_mode, rotation_desc in rotations:
        if rotation_mode == "none":
            func = Ellipsoid(landscape_n)
            scales = 10.0 ** (6.0 * np.arange(landscape_n) / (landscape_n - 1))
            landscape_hessians[rotation_desc] = np.diag(2.0 * scales)
        else:
            func = RotatedEllipsoid(landscape_n, rotation=rotation_mode, seed=42)
            landscape_hessians[rotation_desc] = func.hessian
        landscape_functions[rotation_desc] = func

    plot_function_landscape_grid(
        landscape_functions,
        hessians=landscape_hessians,
        extent=8.0,
        resolution=150,
        suptitle="Ellipsoid Landscapes: Effect of Rotation on Curvature Alignment",
        save_path=output_dir / "landscapes.png",
    )
    print(f"  Saved: {output_dir / 'landscapes.png'}")

    # Individual landscape with eigenvectors for each rotation
    for rotation_mode, rotation_desc in rotations:
        func = landscape_functions[rotation_desc]
        hessian = landscape_hessians[rotation_desc]
        safe_mode = rotation_mode.replace(" ", "_")
        plot_function_landscape(
            func,
            title=f"{rotation_desc}\n(arrows = principal curvature directions)",
            extent=8.0,
            resolution=150,
            show_eigenvectors=True,
            hessian=hessian,
            save_path=output_dir / f"landscape_{safe_mode}.png",
        )
        plt.close("all")

    print()

    # Two regimes: n=m (corrections span full space) and n>>m (they can't)
    regimes = [
        {"n": 10, "m": 10, "budget": 15000, "label": "10D_m10"},
        {"n": 50, "m": 5,  "budget": 10000, "label": "50D_m5"},
    ]

    hessian_colors = {
        "Identity":         "#e74c3c",
        "Hessian diagonal": "#f39c12",
        "Full Hessian":     "#2ecc71",
    }

    for regime in regimes:
        n = regime["n"]
        m = regime["m"]
        budget = regime["budget"]
        regime_label = regime["label"]

        print(f"{'='*60}")
        print(f"Regime: n={n}, m={m} (corrections cover {2*m} of {n} directions)")
        print(f"{'='*60}")
        print()

        # Collect results for the summary bar chart
        summary_data = {}

        for rotation_mode, rotation_desc in rotations:
            # Build the function
            if rotation_mode == "none":
                func = Ellipsoid(n)
                scales = 10.0 ** (6.0 * np.arange(n) / max(n - 1, 1))
                gradient_fn = lambda x, s=scales: 2.0 * s * x
                hessian_diag = 2.0 * scales
                full_hessian = np.diag(hessian_diag)
                diag_fraction = 1.0
            else:
                func = RotatedEllipsoid(n, rotation=rotation_mode, seed=42)
                gradient_fn = func.gradient
                hessian_diag = func.hessian_diagonal
                full_hessian = func.hessian
                diag_fraction = (
                    np.sum(np.diag(func.hessian) ** 2) / np.sum(func.hessian ** 2)
                )

            rng = np.random.default_rng(42)
            x0 = rng.uniform(-100.0, 100.0, size=n)

            print(f"  {rotation_desc} (diagonal fraction: {diag_fraction:.2f})")

            results = {}
            hessian_choices = [
                ("Identity",         None),
                ("Hessian diagonal", hessian_diag),
                ("Full Hessian",     full_hessian),
            ]

            for label, h in hessian_choices:
                config = LBFGSBConfig(
                    dimensions=n, initial_hessian=h, m=m,
                    pgtol=1e-8, factr=1e7, budget=budget,
                )
                result = run_single(
                    func, x0, config, gradient_fn,
                    lower_bounds=-100.0, upper_bounds=100.0,
                )
                results[label] = result
                print(
                    f"    {label:20s}: f={result.best_fitness:.4e}  "
                    f"evals={result.evaluations:>6d}"
                )

            summary_data[rotation_mode] = {
                label: r.evaluations for label, r in results.items()
            }

            # 4-panel convergence comparison per rotation
            safe_mode = rotation_mode.replace(" ", "_")
            plot_comparison(
                results,
                panels=LBFGSB_COMPARISON_PANELS,
                colors=hessian_colors,
                title=(
                    f"L-BFGS-B Initial Hessian Comparison\n"
                    f"{rotation_desc} ({n}D, m={m})"
                ),
                save_path=output_dir / f"{regime_label}_{safe_mode}_convergence.png",
            )
            plot_evaluation_bars(
                results,
                colors=hessian_colors,
                title=f"{rotation_desc} ({n}D, m={m})",
                save_path=output_dir / f"{regime_label}_{safe_mode}_bar.png",
            )

            plt.close("all")
            print()

        # Summary grouped bar chart: all rotations side by side
        _plot_rotation_summary(
            summary_data, rotations, hessian_colors, n, m,
            save_path=output_dir / f"{regime_label}_summary.png",
        )

    print(f"All plots saved to: {output_dir.absolute()}")


def _plot_rotation_summary(summary_data, rotations, colors, n, m, save_path):
    """Grouped bar chart comparing all rotation modes and Hessian choices."""
    fig, ax = plt.subplots(figsize=(14, 6))

    rotation_labels = [desc for _, desc in rotations]
    hessian_labels = ["Identity", "Hessian diagonal", "Full Hessian"]
    num_rotations = len(rotation_labels)
    num_hessians = len(hessian_labels)
    bar_width = 0.25
    x_positions = np.arange(num_rotations)

    for i, hessian_label in enumerate(hessian_labels):
        values = []
        for rotation_mode, _ in rotations:
            evals = summary_data[rotation_mode].get(hessian_label, 0)
            values.append(evals)
        bars = ax.bar(
            x_positions + i * bar_width, values, bar_width,
            label=hessian_label, color=colors[hessian_label],
            edgecolor="white", linewidth=0.5,
        )
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(values) * 0.01,
                    f"{val:,}", ha="center", va="bottom",
                    fontsize=8, fontweight="bold",
                )

    ax.set_xticks(x_positions + bar_width)
    ax.set_xticklabels(rotation_labels, fontsize=10)
    ax.set_ylabel("Function Evaluations", fontsize=12)
    ax.set_title(
        f"L-BFGS-B Evaluations by Rotation and Initial Hessian ({n}D, m={m})",
        fontsize=14,
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.2, axis="y")
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"  Saved summary: {save_path}")
    plt.close(fig)


if __name__ == "__main__":
    run_rotation_study()
