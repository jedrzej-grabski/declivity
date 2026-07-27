"""
Compatibility shim — the initial-geometry object moved to
:mod:`declivity.utils.initial_geometry` when it was generalized from the
L-BFGS-B initial Hessian ``B_0`` into the shared object that also seeds Powell
(initial search directions).

Import from ``declivity.utils.initial_geometry`` in new code. ``InitialHessian``
and ``InitialHessianMode`` remain valid aliases for backwards compatibility
(mirrors ``declivity.algorithms.lbfgsb.line_search``).
"""

from declivity.utils.initial_geometry import (
    GeometryMode,
    HandoffTransform,
    InitialGeometry,
    InitialHessian,
    InitialHessianMode,
    covariance_to_hessian_matrix,
)

__all__ = [
    "GeometryMode",
    "HandoffTransform",
    "InitialGeometry",
    "InitialHessian",
    "InitialHessianMode",
    "covariance_to_hessian_matrix",
]
