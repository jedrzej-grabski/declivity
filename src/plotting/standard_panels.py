"""Standard panel registrations for the four built-in algorithms.

Adding a new panel is one line. To attach ``"my_metric"`` to CMA-ES::

    PanelRegistry.register(
        AlgorithmChoice.CMAES,
        Panel("my_metric", "My Metric", "units", field="my_metric_on_logdata"),
    )

If the same semantic key (e.g. ``"step_size"``) is registered for multiple
algorithms, ``plot_comparison`` will overlay them automatically — the
``field`` may differ per algorithm (CMA-ES ``sigma``, DES ``Ft``,
L-BFGS-B ``step_length``), but the panel speaks one language.

Importing this module triggers all registrations as a side effect.
``src/plotting/__init__.py`` imports it for that reason.
"""

from src.algorithms.choices import AlgorithmChoice
from src.plotting.panel import Panel, PanelRegistry


# Floor used for log-scale fitness plots — runs that converge to (effectively)
# zero would otherwise blow up on log axes.
_FITNESS_FLOOR = 1e-30


# ---------------------------------------------------------------------------
# Cross-algorithm semantic keys
#
# These are the panels available on multiple algorithms. The same ``key``
# registered against different ``field``s is intentional — that's what
# makes ``PanelRegistry.common`` produce useful comparisons.
# ---------------------------------------------------------------------------

_CONVERGENCE = Panel(
    key="convergence",
    title="Convergence",
    ylabel="Best Fitness",
    field="best_fitness",
    yscale="log",
    floor=_FITNESS_FLOOR,
)

_MEAN_FITNESS_POP = Panel(
    key="mean_fitness",
    title="Mean Fitness (population)",
    ylabel="Mean Fitness",
    field="mean_fitness",
    yscale="log",
    floor=_FITNESS_FLOOR,
)

_WORST_FITNESS = Panel(
    key="worst_fitness",
    title="Worst Fitness",
    ylabel="Worst Fitness",
    field="worst_fitness",
    yscale="log",
    floor=_FITNESS_FLOOR,
)

_STD_FITNESS = Panel(
    key="std_fitness",
    title="Fitness Std Dev",
    ylabel="Std Dev",
    field="std_fitness",
    yscale="log",
)


# ---------------------------------------------------------------------------
# DES
# ---------------------------------------------------------------------------

PanelRegistry.register(
    AlgorithmChoice.DES,
    _CONVERGENCE,
    _MEAN_FITNESS_POP,
    _WORST_FITNESS,
    _STD_FITNESS,
    Panel(
        key="step_size",
        title="Step Size",
        ylabel="Ft",
        field="Ft",
        yscale="log",
    ),
    Panel(
        key="condition_number",
        title="Condition Number",
        ylabel="kappa",
        field="condition_number",
        yscale="log",
    ),
)


# ---------------------------------------------------------------------------
# CMA-ES
#
# Note: CMA-ES populates ``BaseLogData.mean_fitness`` with f(mean) — the
# fitness *evaluated at the distribution mean*, not the population
# average. We register it under ``"midpoint_fitness"`` to keep semantics
# clean; the population mean isn't available on CMA-ES so it is not
# registered under ``"mean_fitness"``.
# ---------------------------------------------------------------------------

PanelRegistry.register(
    AlgorithmChoice.CMAES,
    _CONVERGENCE,
    _WORST_FITNESS,
    _STD_FITNESS,
    Panel(
        key="midpoint_fitness",
        title="Midpoint Fitness f(m)",
        ylabel="f(m)",
        field="mean_fitness",
        yscale="log",
        floor=_FITNESS_FLOOR,
    ),
    Panel(
        key="median_fitness",
        title="Median Fitness",
        ylabel="Median",
        field="median_fitness",
        yscale="log",
        floor=_FITNESS_FLOOR,
    ),
    Panel(
        key="step_size",
        title="Step Size",
        ylabel="sigma",
        field="sigma",
        yscale="log",
    ),
    Panel(
        key="condition_number",
        title="Condition Number",
        ylabel="kappa(C)",
        field="condition_number",
        yscale="log",
    ),
    Panel(
        key="eigenvalue_max",
        title="Max Eigenvalue",
        ylabel="lambda_max",
        field="max_eigenvalue",
        yscale="log",
    ),
    Panel(
        key="eigenvalue_min",
        title="Min Eigenvalue",
        ylabel="lambda_min",
        field="min_eigenvalue",
        yscale="log",
    ),
    Panel(
        key="det_covariance",
        title="Search Volume det(C)",
        ylabel="det(C)",
        field="covariance_determinant",
        yscale="log",
    ),
    Panel(
        key="evolution_path_c",
        title="Evolution Path (covariance)",
        ylabel="||p_c||",
        field="pc_norm",
        yscale="linear",
    ),
    Panel(
        key="evolution_path_s",
        title="Evolution Path (step-size)",
        ylabel="||p_sigma||",
        field="ps_norm",
        yscale="linear",
    ),
    Panel(
        key="mean_norm",
        title="Mean Vector Norm",
        ylabel="||m||",
        field="mean_vector_norm",
        yscale="log",
    ),
)


# ---------------------------------------------------------------------------
# MF-CMA-ES
# ---------------------------------------------------------------------------

PanelRegistry.register(
    AlgorithmChoice.MFCMAES,
    _CONVERGENCE,
    _MEAN_FITNESS_POP,
    _WORST_FITNESS,
    _STD_FITNESS,
    Panel(
        key="midpoint_fitness",
        title="Midpoint Fitness f(m)",
        ylabel="f(m)",
        field="midpoint_fitness",
        yscale="log",
        floor=_FITNESS_FLOOR,
    ),
    Panel(
        key="step_size",
        title="Step Size",
        ylabel="sigma",
        field="sigma",
        yscale="log",
    ),
    Panel(
        key="success_probability",
        title="PPMF Success Probability",
        ylabel="p_succ",
        field="p_succ",
        yscale="linear",
    ),
    Panel(
        key="evolution_path_c",
        title="Evolution Path (covariance)",
        ylabel="||p_c||",
        field="pc_norm",
        yscale="linear",
    ),
    Panel(
        key="mean_norm",
        title="Mean Vector Norm",
        ylabel="||m||",
        field="mean_vector_norm",
        yscale="log",
    ),
    Panel(
        key="constraint_violations",
        title="Constraint Violations",
        ylabel="count",
        field="constraint_violations",
        yscale="linear",
    ),
)


# ---------------------------------------------------------------------------
# L-BFGS-B
#
# L-BFGS-B is single-point — no population, so no mean/worst/std fitness.
# Its diagnostic flags gate most logging (gradient_norm, theta, etc.) so
# empty panels are expected when those flags are off; ``plot_metrics``
# draws an empty axes in that case (clear feedback that the diag flag is
# off, not a silent skip).
# ---------------------------------------------------------------------------

PanelRegistry.register(
    AlgorithmChoice.LBFGSB,
    _CONVERGENCE,
    Panel(
        key="function_value",
        title="Function Value",
        ylabel="f(x)",
        field="function_value",
        yscale="log",
        floor=_FITNESS_FLOOR,
    ),
    Panel(
        key="step_size",
        title="Line Search Step",
        ylabel="alpha",
        field="step_length",
        yscale="log",
    ),
    Panel(
        key="gradient_norm",
        title="Gradient Norm",
        ylabel="||grad f||_2",
        field="gradient_norm",
        yscale="log",
    ),
    Panel(
        key="projected_gradient",
        title="Projected Gradient",
        ylabel="||proj grad f||_inf",
        field="projected_gradient_norm",
        yscale="log",
    ),
    Panel(
        key="theta",
        title="L-BFGS Scaling theta",
        ylabel="theta",
        field="theta",
        yscale="log",
    ),
    Panel(
        key="num_free",
        title="Free Variables",
        ylabel="count",
        field="num_free_vars",
        yscale="linear",
    ),
    Panel(
        key="num_corrections",
        title="L-BFGS Corrections Stored",
        ylabel="count",
        field="num_corrections",
        yscale="linear",
    ),
    Panel(
        key="line_search_iters",
        title="Line Search Iterations",
        ylabel="evals/iter",
        field="line_search_iters",
        yscale="linear",
    ),
)
