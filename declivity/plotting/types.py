"""Enums for the declarative plotting API.

Every enum here subclasses :class:`~enum.StrEnum`, so an enum value
*is* its string representation::

    PanelKey.CONVERGENCE == "convergence"        # True
    str(PanelKey.CONVERGENCE) == "convergence"   # True
    list(PanelKey) == [...]                       # iterable for introspection

That means existing call sites that pass raw strings (``panels=["convergence"]``)
keep working unchanged, while new code can use the enum (``panels=[PanelKey.CONVERGENCE]``)
to get IDE autocomplete and refactor safety.

The enums cover the *fixed vocabulary* the framework exposes — panel
semantics, axis directions, scale modes, line styles. Free-form things
(LogData field names, plot titles, hex colors) remain plain strings
because the framework can't enumerate them up front.
"""

from enum import StrEnum


class PanelKey(StrEnum):
    """Standard semantic keys for diagnostic panels.

    These are the keys :py:mod:`declivity.plotting.standard_panels` registers
    out of the box. Custom panels can use any string for their key — the
    enum is a convenience for the built-in vocabulary, not a hard
    constraint on the registry.
    """

    # Shared fitness panels
    CONVERGENCE = "convergence"
    MEAN_FITNESS = "mean_fitness"
    WORST_FITNESS = "worst_fitness"
    STD_FITNESS = "std_fitness"
    MIDPOINT_FITNESS = "midpoint_fitness"
    MEDIAN_FITNESS = "median_fitness"
    FUNCTION_VALUE = "function_value"

    # Step-size / search-scale family
    STEP_SIZE = "step_size"

    # Covariance / distribution geometry (CMA-ES, MF-CMA-ES, DES)
    CONDITION_NUMBER = "condition_number"
    EIGENVALUE_MAX = "eigenvalue_max"
    EIGENVALUE_MIN = "eigenvalue_min"
    DET_COVARIANCE = "det_covariance"
    EVOLUTION_PATHS = "evolution_paths"
    EVOLUTION_PATH_C = "evolution_path_c"
    MEAN_NORM = "mean_norm"

    # L-BFGS-B specific
    GRADIENT_NORM = "gradient_norm"
    GRADIENT_NORMS = "gradient_norms"
    PROJECTED_GRADIENT = "projected_gradient"
    THETA = "theta"
    NUM_FREE = "num_free"
    NUM_CORRECTIONS = "num_corrections"
    LINE_SEARCH_ITERS = "line_search_iters"
    CONVERGENCE_BY_ITER = "convergence_by_iter"
    STEP_SIZE_BY_ITER = "step_size_by_iter"

    # BFGS specific
    CURVATURE = "curvature"
    HESSIAN_CONDITION = "hessian_condition"

    # MF-CMA-ES specific
    SUCCESS_PROBABILITY = "success_probability"
    CONSTRAINT_VIOLATIONS = "constraint_violations"

    # Powell specific
    DELTA = "delta"
    DIRECTION_SET_DET = "direction_set_det"

    # Nelder-Mead specific
    FITNESS_SPREAD = "fitness_spread"
    SIMPLEX_VOLUME = "simplex_volume"
    SIMPLEX_OPERATION = "simplex_operation"


class YScale(StrEnum):
    """Y-axis scaling mode for a :class:`Panel`."""

    LINEAR = "linear"
    LOG = "log"


class XAxis(StrEnum):
    """Common x-axis sources. Panels may also use any other LogData attribute name."""

    EVALUATIONS = "evaluations"
    """Cumulative function evaluation count — the budget unit shared across algorithms."""

    ITERATION = "iteration"
    """Algorithm iteration index. Not directly comparable across algorithms with
    different population sizes."""


class LineStyle(StrEnum):
    """Matplotlib line styles for :class:`Series`."""

    SOLID = "-"
    DASHED = "--"
    DOTTED = ":"
    DASH_DOT = "-."


class PanelSet(StrEnum):
    """Sentinel selectors for the ``panels=`` argument of :py:func:`plot_metrics`."""

    DEFAULT = "default"
    """Panels marked ``default=True`` for the algorithm. Same as ``panels=None``."""

    ALL = "all"
    """Every panel registered for the algorithm, including non-default ones."""
