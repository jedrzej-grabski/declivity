"""Standard panel registrations for the four built-in algorithms.

Adding a new panel is one line. To attach ``"my_metric"`` to CMA-ES::

    PanelRegistry.register(
        AlgorithmChoice.CMAES,
        Panel("my_metric", "My Metric", "units", field="my_metric_on_logdata"),
    )

Multi-series panels overlay several fields on one axes — useful for the
classic "best + mean + median" convergence view::

    Panel(
        "convergence",
        "Convergence",
        "Fitness",
        series=(
            Series("best_fitness",   "Best"),
            Series("mean_fitness",   "Mean", linestyle="--"),
            Series("median_fitness", "Median", linestyle=":"),
        ),
    )

Each panel has a ``default`` flag (True by default). ``plot_metrics(result)``
with no explicit ``panels=`` only renders the default subset, which is
how we keep the headline figure focused. Pass ``panels="all"`` for every
registered panel.

If the same semantic key (e.g. ``"step_size"``) is registered for multiple
algorithms, ``plot_comparison`` will overlay them automatically — the
``field`` may differ per algorithm (CMA-ES ``sigma``, DES ``Ft``,
L-BFGS-B ``step_length``), but the panel speaks one language. For
multi-series panels, only the first series participates in the overlay.

Importing this module triggers all registrations as a side effect.
``src/plotting/__init__.py`` imports it for that reason.
"""

from src.algorithms.choices import AlgorithmChoice
from src.plotting.panel import Panel, PanelRegistry, Series


# Floor used for log-scale fitness plots — runs that converge to (effectively)
# zero would otherwise blow up on log axes.
_FITNESS_FLOOR = 1e-30


# ===========================================================================
# DES
# ===========================================================================

PanelRegistry.register(
    AlgorithmChoice.DES,
    Panel(
        key="convergence",
        title="Convergence",
        ylabel="Fitness (log)",
        series=(
            Series("best_fitness", "Best", color="tab:blue"),
            Series("mean_fitness", "Mean", linestyle="--", color="tab:green"),
        ),
        yscale="log",
        floor=_FITNESS_FLOOR,
    ),
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
    # Non-default panels — available, but excluded from the headline view.
    Panel(
        key="worst_fitness",
        title="Worst Fitness",
        ylabel="Worst Fitness",
        field="worst_fitness",
        yscale="log",
        floor=_FITNESS_FLOOR,
        default=False,
    ),
    Panel(
        key="std_fitness",
        title="Fitness Std Dev",
        ylabel="Std Dev",
        field="std_fitness",
        yscale="log",
        default=False,
    ),
)


# ===========================================================================
# CMA-ES
#
# Note: CMA-ES populates ``BaseLogData.mean_fitness`` with f(mean) — the
# fitness *evaluated at the distribution mean*, not the population
# average. The Series label calls this out as "Mean f(m)" so the overlay
# stays honest.
# ===========================================================================

PanelRegistry.register(
    AlgorithmChoice.CMAES,
    Panel(
        key="convergence",
        title="Convergence",
        ylabel="Fitness (log)",
        series=(
            Series("best_fitness",   "Best",      color="tab:blue"),
            Series("mean_fitness",   "Mean f(m)", linestyle="--", color="tab:green"),
            Series("median_fitness", "Median",    linestyle=":",  color="tab:red"),
        ),
        yscale="log",
        floor=_FITNESS_FLOOR,
    ),
    Panel(
        key="std_fitness",
        title="Fitness Std Dev",
        ylabel="Std Dev",
        field="std_fitness",
        yscale="log",
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
        key="det_covariance",
        title="Search Volume det(C)",
        ylabel="det(C)",
        field="covariance_determinant",
        yscale="log",
    ),
    Panel(
        key="evolution_paths",
        title="Evolution Paths",
        ylabel="Path Norm",
        series=(
            Series("pc_norm", "||p_c||",     color="tab:blue"),
            Series("ps_norm", "||p_sigma||", linestyle="--", color="tab:red"),
        ),
        yscale="linear",
    ),
    Panel(
        key="mean_norm",
        title="Mean Vector Norm",
        ylabel="||m||",
        field="mean_vector_norm",
        yscale="log",
    ),
    # Non-default — available via plot_metrics(panels=["..."]) or "all".
    Panel(
        key="worst_fitness",
        title="Worst Fitness",
        ylabel="Worst Fitness",
        field="worst_fitness",
        yscale="log",
        floor=_FITNESS_FLOOR,
        default=False,
    ),
    Panel(
        key="eigenvalue_max",
        title="Max Eigenvalue",
        ylabel="lambda_max",
        field="max_eigenvalue",
        yscale="log",
        default=False,
    ),
    Panel(
        key="eigenvalue_min",
        title="Min Eigenvalue",
        ylabel="lambda_min",
        field="min_eigenvalue",
        yscale="log",
        default=False,
    ),
    Panel(
        key="midpoint_fitness",
        title="Midpoint Fitness f(m)",
        ylabel="f(m)",
        field="mean_fitness",
        yscale="log",
        floor=_FITNESS_FLOOR,
        default=False,
    ),
    Panel(
        key="median_fitness",
        title="Median Fitness",
        ylabel="Median",
        field="median_fitness",
        yscale="log",
        floor=_FITNESS_FLOOR,
        default=False,
    ),
)


# ===========================================================================
# MF-CMA-ES
# ===========================================================================

PanelRegistry.register(
    AlgorithmChoice.MFCMAES,
    Panel(
        key="convergence",
        title="Convergence",
        ylabel="Fitness (log)",
        series=(
            Series("best_fitness",     "Best",          color="tab:blue"),
            Series("midpoint_fitness", "Midpoint f(m)", linestyle="--", color="tab:green"),
            Series("mean_fitness",     "Mean",          linestyle=":",  color="tab:red"),
        ),
        yscale="log",
        floor=_FITNESS_FLOOR,
    ),
    Panel(
        key="std_fitness",
        title="Fitness Std Dev",
        ylabel="Std Dev",
        field="std_fitness",
        yscale="log",
    ),
    Panel(
        key="step_size",
        title="Step Size (PPMF)",
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
        key="constraint_violations",
        title="Constraint Violations",
        ylabel="count",
        field="constraint_violations",
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
    # Non-default — subsumed by the overlay or rarely useful headline data.
    Panel(
        key="worst_fitness",
        title="Worst Fitness",
        ylabel="Worst Fitness",
        field="worst_fitness",
        yscale="log",
        floor=_FITNESS_FLOOR,
        default=False,
    ),
    Panel(
        key="midpoint_fitness",
        title="Midpoint Fitness f(m)",
        ylabel="f(m)",
        field="midpoint_fitness",
        yscale="log",
        floor=_FITNESS_FLOOR,
        default=False,
    ),
)


# ===========================================================================
# L-BFGS-B
#
# L-BFGS-B is single-point — no population, so no mean/worst/std fitness.
# Diagnostic flags gate most logging (gradient_norm, theta, etc.) so empty
# panels are expected when those flags are off; the plotter draws an empty
# axes in that case (clear feedback that the diag flag is off, not a
# silent skip).
# ===========================================================================

PanelRegistry.register(
    AlgorithmChoice.LBFGSB,
    Panel(
        key="convergence",
        title="Convergence",
        ylabel="Fitness (log)",
        series=(
            Series("best_fitness",   "Best", color="tab:blue"),
            Series("function_value", "f(x)", linestyle="--", color="tab:green"),
        ),
        yscale="log",
        floor=_FITNESS_FLOOR,
    ),
    Panel(
        key="gradient_norms",
        title="Gradient Norms",
        ylabel="Norm (log)",
        series=(
            Series("projected_gradient_norm", "||proj g||_inf", color="tab:red"),
            Series("gradient_norm",           "||g||_2",        linestyle="--", color="tab:blue"),
        ),
        yscale="log",
    ),
    Panel(
        key="step_size",
        title="Line Search Step",
        ylabel="alpha",
        field="step_length",
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
    # Non-default — function_value and the individual gradient norms are
    # subsumed by the multi-series convergence / gradient_norms panels.
    Panel(
        key="function_value",
        title="Function Value",
        ylabel="f(x)",
        field="function_value",
        yscale="log",
        floor=_FITNESS_FLOOR,
        default=False,
    ),
    Panel(
        key="gradient_norm",
        title="Gradient Norm",
        ylabel="||grad f||_2",
        field="gradient_norm",
        yscale="log",
        default=False,
    ),
    Panel(
        key="projected_gradient",
        title="Projected Gradient",
        ylabel="||proj grad f||_inf",
        field="projected_gradient_norm",
        yscale="log",
        default=False,
    ),
)
