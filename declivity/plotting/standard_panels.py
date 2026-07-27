"""Standard panel registrations for the four built-in algorithms.

Adding a new panel is one line. To attach ``"my_metric"`` to CMA-ES::

    PanelRegistry.register(
        AlgorithmChoice.CMAES,
        Panel(
            key=PanelKey.MY_METRIC,           # or just "my_metric"
            title="My Metric",
            ylabel="units",
            field="my_metric_on_logdata",
        ),
    )

Multi-series panels overlay several fields on one axes — useful for the
classic "best + mean + median" convergence view::

    Panel(
        key=PanelKey.CONVERGENCE,
        title="Convergence",
        ylabel="Fitness",
        series=(
            Series("best_fitness",   "Best"),
            Series("mean_fitness",   "Mean",   linestyle=LineStyle.DASHED),
            Series("median_fitness", "Median", linestyle=LineStyle.DOTTED),
        ),
    )

Each panel has a ``default`` flag (True by default). ``plot_metrics(result)``
with no explicit ``panels=`` only renders the default subset, which is
how we keep the headline figure focused. Pass ``panels=PanelSet.ALL``
for every registered panel.

If the same semantic key (e.g. :attr:`PanelKey.STEP_SIZE`) is registered
for multiple algorithms, ``plot_comparison`` will overlay them
automatically — the ``field`` may differ per algorithm (CMA-ES ``sigma``,
DES ``Ft``, L-BFGS-B ``step_length``), but the panel speaks one
language. For multi-series panels, only the first series participates
in the overlay.

Importing this module triggers all registrations as a side effect.
``src/plotting/__init__.py`` imports it for that reason.
"""

from declivity.algorithms.choices import AlgorithmChoice
from declivity.plotting.panel import Panel, PanelRegistry, Series
from declivity.plotting.types import LineStyle, PanelKey, XAxis, YScale


# Floor used for log-scale fitness plots — runs that converge to (effectively)
# zero would otherwise blow up on log axes.
_FITNESS_FLOOR = 1e-30


# ===========================================================================
# DES
# ===========================================================================

PanelRegistry.register(
    AlgorithmChoice.DES,
    Panel(
        key=PanelKey.CONVERGENCE,
        title="Convergence",
        ylabel="Fitness (log)",
        series=(
            Series("best_fitness", "Best", color="tab:blue"),
            Series("mean_fitness", "Mean", linestyle=LineStyle.DASHED, color="tab:green"),
        ),
        yscale=YScale.LOG,
        floor=_FITNESS_FLOOR,
    ),
    Panel(
        key=PanelKey.STEP_SIZE,
        title="Step Size",
        ylabel="Ft",
        field="ft",
        yscale=YScale.LOG,
    ),
    Panel(
        key=PanelKey.CONDITION_NUMBER,
        title="Condition Number",
        ylabel="kappa",
        field="condition_number",
        yscale=YScale.LOG,
    ),
    # Non-default — available, but excluded from the headline view.
    Panel(
        key=PanelKey.WORST_FITNESS,
        title="Worst Fitness",
        ylabel="Worst Fitness",
        field="worst_fitness",
        yscale=YScale.LOG,
        floor=_FITNESS_FLOOR,
        default=False,
    ),
    Panel(
        key=PanelKey.STD_FITNESS,
        title="Fitness Std Dev",
        ylabel="Std Dev",
        field="std_fitness",
        yscale=YScale.LOG,
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
        key=PanelKey.CONVERGENCE,
        title="Convergence",
        ylabel="Fitness (log)",
        series=(
            Series("best_fitness",   "Best",      color="tab:blue"),
            Series("mean_fitness",   "Mean f(m)", linestyle=LineStyle.DASHED, color="tab:green"),
            Series("median_fitness", "Median",    linestyle=LineStyle.DOTTED, color="tab:red"),
        ),
        yscale=YScale.LOG,
        floor=_FITNESS_FLOOR,
    ),
    Panel(
        key=PanelKey.STD_FITNESS,
        title="Fitness Std Dev",
        ylabel="Std Dev",
        field="std_fitness",
        yscale=YScale.LOG,
    ),
    Panel(
        key=PanelKey.STEP_SIZE,
        title="Step Size",
        ylabel="sigma",
        field="sigma",
        yscale=YScale.LOG,
    ),
    Panel(
        key=PanelKey.CONDITION_NUMBER,
        title="Condition Number",
        ylabel="kappa(C)",
        field="condition_number",
        yscale=YScale.LOG,
    ),
    Panel(
        key=PanelKey.DET_COVARIANCE,
        title="Search Volume det(C)",
        ylabel="det(C)",
        field="covariance_determinant",
        yscale=YScale.LOG,
    ),
    Panel(
        key=PanelKey.EVOLUTION_PATHS,
        title="Evolution Paths",
        ylabel="Path Norm",
        series=(
            Series("pc_norm", "||p_c||",     color="tab:blue"),
            Series("ps_norm", "||p_sigma||", linestyle=LineStyle.DASHED, color="tab:red"),
        ),
        yscale=YScale.LINEAR,
    ),
    Panel(
        key=PanelKey.MEAN_NORM,
        title="Mean Vector Norm",
        ylabel="||m||",
        field="mean_vector_norm",
        yscale=YScale.LOG,
    ),
    # Non-default — available via plot_metrics(panels=[...]) or PanelSet.ALL.
    Panel(
        key=PanelKey.WORST_FITNESS,
        title="Worst Fitness",
        ylabel="Worst Fitness",
        field="worst_fitness",
        yscale=YScale.LOG,
        floor=_FITNESS_FLOOR,
        default=False,
    ),
    Panel(
        key=PanelKey.EIGENVALUE_MAX,
        title="Max Eigenvalue",
        ylabel="lambda_max",
        field="max_eigenvalue",
        yscale=YScale.LOG,
        default=False,
    ),
    Panel(
        key=PanelKey.EIGENVALUE_MIN,
        title="Min Eigenvalue",
        ylabel="lambda_min",
        field="min_eigenvalue",
        yscale=YScale.LOG,
        default=False,
    ),
    Panel(
        key=PanelKey.MIDPOINT_FITNESS,
        title="Midpoint Fitness f(m)",
        ylabel="f(m)",
        field="mean_fitness",
        yscale=YScale.LOG,
        floor=_FITNESS_FLOOR,
        default=False,
    ),
    Panel(
        key=PanelKey.MEDIAN_FITNESS,
        title="Median Fitness",
        ylabel="Median",
        field="median_fitness",
        yscale=YScale.LOG,
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
        key=PanelKey.CONVERGENCE,
        title="Convergence",
        ylabel="Fitness (log)",
        series=(
            Series("best_fitness",     "Best",          color="tab:blue"),
            Series("midpoint_fitness", "Midpoint f(m)", linestyle=LineStyle.DASHED, color="tab:green"),
            Series("mean_fitness",     "Mean",          linestyle=LineStyle.DOTTED, color="tab:red"),
        ),
        yscale=YScale.LOG,
        floor=_FITNESS_FLOOR,
    ),
    Panel(
        key=PanelKey.STD_FITNESS,
        title="Fitness Std Dev",
        ylabel="Std Dev",
        field="std_fitness",
        yscale=YScale.LOG,
    ),
    Panel(
        key=PanelKey.STEP_SIZE,
        title="Step Size (PPMF)",
        ylabel="sigma",
        field="sigma",
        yscale=YScale.LOG,
    ),
    Panel(
        key=PanelKey.SUCCESS_PROBABILITY,
        title="PPMF Success Probability",
        ylabel="p_succ",
        field="p_succ",
        yscale=YScale.LINEAR,
    ),
    Panel(
        key=PanelKey.CONSTRAINT_VIOLATIONS,
        title="Constraint Violations",
        ylabel="count",
        field="constraint_violations",
        yscale=YScale.LINEAR,
    ),
    Panel(
        key=PanelKey.EVOLUTION_PATH_C,
        title="Evolution Path (covariance)",
        ylabel="||p_c||",
        field="pc_norm",
        yscale=YScale.LINEAR,
    ),
    Panel(
        key=PanelKey.MEAN_NORM,
        title="Mean Vector Norm",
        ylabel="||m||",
        field="mean_vector_norm",
        yscale=YScale.LOG,
    ),
    # Non-default — subsumed by the overlay or rarely useful headline data.
    Panel(
        key=PanelKey.WORST_FITNESS,
        title="Worst Fitness",
        ylabel="Worst Fitness",
        field="worst_fitness",
        yscale=YScale.LOG,
        floor=_FITNESS_FLOOR,
        default=False,
    ),
    Panel(
        key=PanelKey.MIDPOINT_FITNESS,
        title="Midpoint Fitness f(m)",
        ylabel="f(m)",
        field="midpoint_fitness",
        yscale=YScale.LOG,
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
        key=PanelKey.CONVERGENCE,
        title="Convergence",
        ylabel="Fitness (log)",
        series=(
            Series("best_fitness",   "Best", color="tab:blue"),
            Series("function_value", "f(x)", linestyle=LineStyle.DASHED, color="tab:green"),
        ),
        yscale=YScale.LOG,
        floor=_FITNESS_FLOOR,
    ),
    Panel(
        key=PanelKey.GRADIENT_NORMS,
        title="Gradient Norms",
        ylabel="Norm (log)",
        series=(
            Series("projected_gradient_norm", "||proj g||_inf", color="tab:red"),
            Series("gradient_norm",           "||g||_2",        linestyle=LineStyle.DASHED, color="tab:blue"),
        ),
        yscale=YScale.LOG,
    ),
    Panel(
        key=PanelKey.STEP_SIZE,
        title="Line Search Step",
        ylabel="alpha",
        field="step_length",
        yscale=YScale.LOG,
    ),
    Panel(
        key=PanelKey.THETA,
        title="L-BFGS Scaling theta",
        ylabel="theta",
        field="theta",
        yscale=YScale.LOG,
    ),
    Panel(
        key=PanelKey.NUM_FREE,
        title="Free Variables",
        ylabel="count",
        field="num_free_vars",
        yscale=YScale.LINEAR,
    ),
    Panel(
        key=PanelKey.NUM_CORRECTIONS,
        title="L-BFGS Corrections Stored",
        ylabel="count",
        field="num_corrections",
        yscale=YScale.LINEAR,
    ),
    Panel(
        key=PanelKey.LINE_SEARCH_ITERS,
        title="Line Search Iterations",
        ylabel="evals/iter",
        field="line_search_iters",
        yscale=YScale.LINEAR,
    ),
    # Non-default — function_value and the individual gradient norms are
    # subsumed by the multi-series convergence / gradient_norms panels.
    Panel(
        key=PanelKey.FUNCTION_VALUE,
        title="Function Value",
        ylabel="f(x)",
        field="function_value",
        yscale=YScale.LOG,
        floor=_FITNESS_FLOOR,
        default=False,
    ),
    Panel(
        key=PanelKey.GRADIENT_NORM,
        title="Gradient Norm",
        ylabel="||grad f||_2",
        field="gradient_norm",
        yscale=YScale.LOG,
        default=False,
    ),
    Panel(
        key=PanelKey.PROJECTED_GRADIENT,
        title="Projected Gradient",
        ylabel="||proj grad f||_inf",
        field="projected_gradient_norm",
        yscale=YScale.LOG,
        default=False,
    ),
    # Iteration-axis variants — useful for L-BFGS-B variant studies that
    # want to compare e.g. different line searches on a per-iteration
    # basis (an evals axis would penalize the variant that does more
    # evals per iter on the line search). Non-default; opt in via
    # panels=[PanelKey.CONVERGENCE_BY_ITER, ...].
    Panel(
        key=PanelKey.CONVERGENCE_BY_ITER,
        title="Convergence (by iteration)",
        ylabel="Best Fitness (log)",
        field="best_fitness",
        x_field=XAxis.ITERATION,
        yscale=YScale.LOG,
        floor=_FITNESS_FLOOR,
        default=False,
    ),
    Panel(
        key=PanelKey.STEP_SIZE_BY_ITER,
        title="Line Search Step (by iteration)",
        ylabel="alpha",
        field="step_length",
        x_field=XAxis.ITERATION,
        yscale=YScale.LOG,
        default=False,
    ),
)
# ===========================================================================
# Powell
#
# Single-point and derivative-free — no population, no gradients. One
# record per outer iteration (one sweep over the direction set), logged
# at the same boundary as scipy's per-iteration callback. Diagnostic
# flags gate most fields (delta, step_norm, direction-set geometry), so
# empty panels signal an off flag, as with L-BFGS-B.
# ===========================================================================

PanelRegistry.register(
    AlgorithmChoice.POWELL,
    Panel(
        key=PanelKey.CONVERGENCE,
        title="Convergence",
        ylabel="Fitness (log)",
        series=(
            Series("best_fitness",   "Best", color="tab:blue"),
            Series("function_value", "f(x)", linestyle=LineStyle.DASHED, color="tab:green"),
        ),
        yscale=YScale.LOG,
        floor=_FITNESS_FLOOR,
    ),
    Panel(
        key=PanelKey.STEP_SIZE,
        title="Sweep Displacement",
        ylabel="||x_k - x_{k-1}||",
        field="step_norm",
        yscale=YScale.LOG,
    ),
    Panel(
        key=PanelKey.DELTA,
        title="Largest Single-Direction Decrease",
        ylabel="delta",
        field="delta",
        yscale=YScale.LOG,
    ),
    Panel(
        key=PanelKey.CONDITION_NUMBER,
        title="Direction-Set Condition Number",
        ylabel="kappa(D)",
        field="direc_condition_number",
        yscale=YScale.LOG,
    ),
    Panel(
        key=PanelKey.DIRECTION_SET_DET,
        title="Direction-Set |det|",
        ylabel="|det D|",
        field="direc_determinant",
        yscale=YScale.LOG,
    ),
    Panel(
        key=PanelKey.LINE_SEARCH_ITERS,
        title="Line Search Evaluations",
        ylabel="evals/iter",
        field="line_search_evals",
        yscale=YScale.LINEAR,
    ),
    # Non-default — function_value is subsumed by the convergence panel.
    Panel(
        key=PanelKey.FUNCTION_VALUE,
        title="Function Value",
        ylabel="f(x)",
        field="function_value",
        yscale=YScale.LOG,
        floor=_FITNESS_FLOOR,
        default=False,
    ),
    Panel(
        key=PanelKey.CONVERGENCE_BY_ITER,
        title="Convergence (by iteration)",
        ylabel="Best Fitness (log)",
        field="best_fitness",
        x_field=XAxis.ITERATION,
        yscale=YScale.LOG,
        floor=_FITNESS_FLOOR,
        default=False,
    ),
)

