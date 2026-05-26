"""Objective-function landscape plots.

These don't fit the :class:`Panel` model — they take a callable and a
region, not a LogData. They live in their own module so the panel system
stays focused on time-series diagnostics.

Two entry points:

- :py:func:`plot_function_landscape` — one contour plot, optionally with
  Hessian eigenvector arrows.
- :py:func:`plot_function_landscape_grid` — several functions side-by-side.
"""

from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray


def _save_if_path(fig: Figure, save_path: Path | str | None) -> None:
    if save_path is None:
        return
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")


def _evaluate_grid(
    func: Callable[[NDArray[np.float64]], float],
    center: NDArray[np.float64],
    dim1: int,
    dim2: int,
    extent: float,
    resolution: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Compute Z[i,j] = f(point) on a (dim1, dim2) slice through center."""
    x_range = np.linspace(center[dim1] - extent, center[dim1] + extent, resolution)
    y_range = np.linspace(center[dim2] - extent, center[dim2] + extent, resolution)
    X, Y = np.meshgrid(x_range, y_range)
    Z = np.zeros_like(X)
    for i in range(resolution):
        for j in range(resolution):
            point = center.copy()
            point[dim1] = X[i, j]
            point[dim2] = Y[i, j]
            Z[i, j] = func(point)
    return X, Y, Z


def _draw_eigenvectors(
    ax,
    hessian: NDArray[np.float64],
    center: NDArray[np.float64],
    dim1: int,
    dim2: int,
    arrow_scale: float,
    with_legend: bool,
) -> None:
    """Overlay the two eigenvector directions of the dim1/dim2 Hessian sub-block."""
    sub_hessian = np.array([
        [hessian[dim1, dim1], hessian[dim1, dim2]],
        [hessian[dim2, dim1], hessian[dim2, dim2]],
    ])
    eigenvalues, eigenvectors = np.linalg.eigh(sub_hessian)

    for k in range(2):
        eigenvector = eigenvectors[:, k]
        color = "#ff6b6b" if k == 0 else "#4ecdc4"
        # Draw both directions of each eigenvector.
        for sign in (+1, -1):
            ax.annotate(
                "",
                xy=(
                    center[dim1] + sign * arrow_scale * eigenvector[0],
                    center[dim2] + sign * arrow_scale * eigenvector[1],
                ),
                xytext=(center[dim1], center[dim2]),
                arrowprops=dict(arrowstyle="->", color=color, lw=2.5),
            )
        if with_legend:
            ax.plot(
                [], [],
                color=color, lw=2.5,
                label=rf"$\lambda_{k+1}$ = {eigenvalues[k]:.1e}",
            )
    if with_legend:
        ax.legend(loc="upper right", fontsize=10)


def plot_function_landscape(
    func: Callable[[NDArray[np.float64]], float],
    *,
    title: str = "Function Landscape",
    dim1: int = 0,
    dim2: int = 1,
    center: NDArray[np.float64] | None = None,
    extent: float = 10.0,
    resolution: int = 200,
    log_scale: bool = True,
    hessian: NDArray[np.float64] | None = None,
    show_eigenvectors: bool = False,
    figsize: tuple[float, float] = (8.0, 7.0),
    save_path: Path | str | None = None,
) -> Figure:
    """Contour plot of a 2D slice through a benchmark function.

    Fixes every variable except ``dim1`` and ``dim2`` at ``center``, then
    evaluates the function on a grid. Optionally overlays the Hessian's
    eigenvector directions at the center to show principal curvature.

    Args:
        func: Callable taking an n-dimensional vector.
        title: Plot title.
        dim1, dim2: Coordinate indices to slice on.
        center: Point where the other coordinates are pinned (default
            ``np.zeros(n)``, where ``n`` is read from ``func.dimensions``
            if present).
        extent: Half-width of the plot region in each direction.
        resolution: Grid points per axis.
        log_scale: Use log10 of the function value for contour levels.
        hessian: Full ``n x n`` Hessian matrix; only needed when
            ``show_eigenvectors=True``.
        show_eigenvectors: Draw arrow overlays from the Hessian.
        figsize, save_path: Standard.

    Returns:
        The matplotlib :class:`Figure`.
    """
    if center is None:
        n = getattr(func, "dimensions", None) or 2
        center = np.zeros(n)

    X, Y, Z = _evaluate_grid(func, center, dim1, dim2, extent, resolution)

    fig, ax = plt.subplots(figsize=figsize)

    if log_scale:
        Z_plot = np.log10(np.maximum(Z, 1e-30))
        cbar_label = r"$\log_{10} f$"
    else:
        Z_plot = Z
        cbar_label = r"$f(x)$"

    filled = ax.contourf(X, Y, Z_plot, levels=30, cmap="viridis")
    ax.contour(X, Y, Z_plot, levels=15, colors="white", linewidths=0.3, alpha=0.5)
    cbar = plt.colorbar(filled, ax=ax)
    cbar.set_label(cbar_label, fontsize=11)

    if show_eigenvectors and hessian is not None:
        _draw_eigenvectors(
            ax, hessian, center, dim1, dim2,
            arrow_scale=extent * 0.4, with_legend=True,
        )

    ax.set_xlabel(rf"$x_{{{dim1 + 1}}}$", fontsize=13)
    ax.set_ylabel(rf"$x_{{{dim2 + 1}}}$", fontsize=13)
    ax.set_title(title, fontsize=14)
    ax.set_aspect("equal")
    fig.tight_layout()

    _save_if_path(fig, save_path)
    return fig


def plot_function_landscape_grid(
    functions: dict[str, Callable[[NDArray[np.float64]], float]],
    *,
    hessians: dict[str, NDArray[np.float64]] | None = None,
    dim1: int = 0,
    dim2: int = 1,
    extent: float = 10.0,
    resolution: int = 200,
    cols: int = 4,
    figsize_per_panel: tuple[float, float] = (6.0, 5.5),
    suptitle: str = "Function Landscapes",
    save_path: Path | str | None = None,
) -> Figure:
    """Grid of contour plots for multiple functions, one per panel.

    Args:
        functions: ``{label: callable}`` — each callable takes an
            n-dimensional vector and returns a scalar. The label sits
            above the panel.
        hessians: Optional ``{label: matrix}`` — when an entry matches a
            function label, eigenvector arrows are overlaid on that panel.
        dim1, dim2: Coordinate indices to slice on.
        extent: Half-width per axis.
        resolution: Grid points per axis.
        cols: Maximum columns; rows are inferred.
        figsize_per_panel: ``(width, height)`` per panel.
        suptitle: Figure-level title.
        save_path: If set, save here.

    Returns:
        The matplotlib :class:`Figure`.
    """
    num_funcs = len(functions)
    if num_funcs == 0:
        raise ValueError("functions must contain at least one entry")

    actual_cols = min(num_funcs, cols)
    rows = (num_funcs + actual_cols - 1) // actual_cols
    fig, axes = plt.subplots(
        rows,
        actual_cols,
        figsize=(figsize_per_panel[0] * actual_cols, figsize_per_panel[1] * rows),
        squeeze=False,
    )
    flat_axes = axes.flatten()

    for idx, (label, func) in enumerate(functions.items()):
        ax = flat_axes[idx]
        n = getattr(func, "dimensions", None) or 2
        center = np.zeros(n)

        X, Y, Z = _evaluate_grid(func, center, dim1, dim2, extent, resolution)
        Z_plot = np.log10(np.maximum(Z, 1e-30))
        ax.contourf(X, Y, Z_plot, levels=30, cmap="viridis")
        ax.contour(X, Y, Z_plot, levels=15, colors="white", linewidths=0.3, alpha=0.5)

        if hessians and label in hessians:
            _draw_eigenvectors(
                ax, hessians[label], center, dim1, dim2,
                arrow_scale=extent * 0.35, with_legend=False,
            )

        ax.set_xlabel(rf"$x_{{{dim1 + 1}}}$", fontsize=11)
        ax.set_ylabel(rf"$x_{{{dim2 + 1}}}$", fontsize=11)
        ax.set_title(label, fontsize=12)
        ax.set_aspect("equal")

    for idx in range(num_funcs, len(flat_axes)):
        flat_axes[idx].set_visible(False)

    fig.suptitle(suptitle, fontsize=16, y=1.02)
    fig.tight_layout()

    _save_if_path(fig, save_path)
    return fig
