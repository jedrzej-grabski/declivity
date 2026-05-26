"""Specialized one-off diagnostic plots that don't fit the panel model.

The :class:`Panel` system is built around per-iteration time series.
Anything else — matrix comparisons, eigenvalue spectra, contour plots —
lives in a sibling module so the panel API stays focused.

Currently provides:

- :py:func:`plot_matrix_diagonal_comparison` — sorted absolute diagonals
  of one or more matrices against a reference. Used by the covariance
  transformation study to ask "does the matrix we built capture the
  per-variable curvature of the true Hessian?"
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray


def _save_if_path(fig: Figure, save_path: Path | str | None) -> None:
    if save_path is None:
        return
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")


def plot_matrix_diagonal_comparison(
    matrices: dict[str, NDArray[np.float64]],
    reference: NDArray[np.float64],
    *,
    reference_label: str = "True Hessian",
    title: str = "Diagonal Profile Comparison",
    cols: int = 3,
    figsize_per_panel: tuple[float, float] = (6.0, 4.5),
    save_path: Path | str | None = None,
) -> Figure:
    """Sorted absolute diagonals against a reference, one panel per matrix.

    Each panel plots two curves: the sorted absolute diagonal of one
    candidate matrix and the sorted absolute diagonal of the reference,
    both normalized so the largest entry is 1. Use this to ask whether
    a constructed matrix (e.g. ``C^{-1}`` from a CMA-ES handoff)
    matches the per-variable curvature ranking of the true Hessian.

    Args:
        matrices: ``{label: n x n matrix}`` — one panel per entry.
        reference: ``n x n`` matrix used as the ground-truth comparison.
        reference_label: Legend label for the reference curve.
        title: Figure-level suptitle.
        cols: Maximum columns; rows inferred.
        figsize_per_panel: ``(width, height)`` per panel.
        save_path: If set, save here.

    Returns:
        The matplotlib :class:`Figure`.
    """
    num_panels = len(matrices)
    if num_panels == 0:
        raise ValueError("matrices must contain at least one entry")

    actual_cols = min(num_panels, cols)
    rows = (num_panels + actual_cols - 1) // actual_cols
    fig, axes = plt.subplots(
        rows,
        actual_cols,
        figsize=(figsize_per_panel[0] * actual_cols, figsize_per_panel[1] * rows),
        squeeze=False,
    )
    flat_axes = axes.flatten()

    reference_diag = np.sort(np.abs(np.diag(reference)))
    reference_normalized = reference_diag / (np.max(reference_diag) + 1e-30)
    variable_indices = np.arange(1, len(reference_diag) + 1)

    for idx, (label, matrix) in enumerate(matrices.items()):
        ax = flat_axes[idx]
        matrix_diag = np.sort(np.abs(np.diag(matrix)))
        matrix_normalized = matrix_diag / (np.max(matrix_diag) + 1e-30)

        ax.semilogy(
            variable_indices, reference_normalized,
            "k-o", markersize=3, linewidth=1.5, label=reference_label,
        )
        ax.semilogy(
            variable_indices, matrix_normalized,
            "g--s", markersize=3, linewidth=1.5, label="Passed matrix",
        )
        ax.set_xlabel("Variable index (sorted)", fontsize=10)
        ax.set_ylabel("Normalized diagonal (log)", fontsize=10)
        ax.set_title(label, fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    for idx in range(num_panels, len(flat_axes)):
        flat_axes[idx].set_visible(False)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()

    _save_if_path(fig, save_path)
    return fig
