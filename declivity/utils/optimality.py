"""
First-order optimality measures shared by the gradient-based local optimizers.

The **projected gradient** is the KKT optimality measure for box-constrained
minimization: it zeroes out the components of the gradient that push *into* an
active bound (where no further feasible descent is possible), leaving the
components along which the point could still improve.  Its infinity norm is the
standard convergence gauge used by L-BFGS-B (Byrd–Lu–Nocedal–Zhu 1995).

A key property the callers rely on: when every bound is infinite the projected
gradient is *identically* the plain gradient (nothing is ever clipped), so an
optimizer that tests ``‖proj_grad‖`` reproduces a plain ``‖grad‖`` test exactly
in the unconstrained regime — while still recognising KKT points at active
bounds, where a raw-gradient test would never converge.
"""

import numpy as np
from numpy.typing import NDArray


def projected_gradient(
    x: NDArray[np.float64],
    gradient: NDArray[np.float64],
    lower_bounds: NDArray[np.float64],
    upper_bounds: NDArray[np.float64],
) -> NDArray[np.float64]:
    """KKT projected gradient at *x*.

    For each coordinate the gradient is clipped by the distance to the bound it
    points toward:

    - where ``g_i < 0`` (pointing up), by ``x_i - upper_i``;
    - where ``g_i > 0`` (pointing down), by ``x_i - lower_i``.

    Returns a new array; equals *gradient* when the relevant bounds are
    infinite.
    """
    projected = gradient.copy()
    negative_mask = gradient < 0
    positive_mask = gradient > 0
    projected[negative_mask] = np.maximum(
        x[negative_mask] - upper_bounds[negative_mask],
        gradient[negative_mask],
    )
    projected[positive_mask] = np.minimum(
        x[positive_mask] - lower_bounds[positive_mask],
        gradient[positive_mask],
    )
    return projected


def projected_gradient_inf_norm(
    x: NDArray[np.float64],
    gradient: NDArray[np.float64],
    lower_bounds: NDArray[np.float64],
    upper_bounds: NDArray[np.float64],
) -> float:
    """Infinity norm of :func:`projected_gradient` (0.0 for an empty vector)."""
    projected = projected_gradient(x, gradient, lower_bounds, upper_bounds)
    if len(projected) == 0:
        return 0.0
    return float(np.max(np.abs(projected)))
