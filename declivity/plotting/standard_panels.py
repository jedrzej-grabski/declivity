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

Multi-series panels overlay several fields on one axes, for the
best/mean/median convergence view::

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
``declivity/plotting/__init__.py`` imports it for that reason.
"""

from declivity.algorithms.choices import AlgorithmChoice
from declivity.plotting.panel import Panel, PanelRegistry, Series
from declivity.plotting.types import LineStyle, PanelKey, XAxis, YScale

# Floor for log-scale fitness plots; runs that converge to zero would
# otherwise blow up on a log axis.
_FITNESS_FLOOR = 1e-30


# DES.

PanelRegistry.register(
    AlgorithmChoice.DES,
    Panel(
        key=PanelKey.CONVERGENCE,
        title="Convergence",
        ylabel="Fitness (log)",
        series=(
            Series("best_fitness", "Best", color="tab:blue"),
            Series(
                "mean_fitness", "Mean", linestyle=LineStyle.DASHED, color="tab:green"
            ),
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
    # Non-default: available, but excluded from the headline view.
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


# CMA-ES.
#
# CMA-ES populates ``BaseLogData.mean_fitness`` with f(mean), the fitness at
# the distribution mean rather than the population average, so the Series
# label reads "Mean f(m)".

PanelRegistry.register(
    AlgorithmChoice.CMAES,
    Panel(
        key=PanelKey.CONVERGENCE,
        title="Convergence",
        ylabel="Fitness (log)",
        series=(
            Series("best_fitness", "Best", color="tab:blue"),
            Series(
                "mean_fitness",
                "Mean f(m)",
                linestyle=LineStyle.DASHED,
                color="tab:green",
            ),
            Series(
                "median_fitness", "Median", linestyle=LineStyle.DOTTED, color="tab:red"
            ),
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
            Series("pc_norm", "||p_c||", color="tab:blue"),
            Series(
                "ps_norm", "||p_sigma||", linestyle=LineStyle.DASHED, color="tab:red"
            ),
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
    # Non-default: available via plot_metrics(panels=[...]) or PanelSet.ALL.
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


# MF-CMA-ES.

PanelRegistry.register(
    AlgorithmChoice.MFCMAES,
    Panel(
        key=PanelKey.CONVERGENCE,
        title="Convergence",
        ylabel="Fitness (log)",
        series=(
            Series("best_fitness", "Best", color="tab:blue"),
            Series(
                "midpoint_fitness",
                "Midpoint f(m)",
                linestyle=LineStyle.DASHED,
                color="tab:green",
            ),
            Series("mean_fitness", "Mean", linestyle=LineStyle.DOTTED, color="tab:red"),
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
    # Non-default: subsumed by the overlay, or rarely useful.
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


# L-BFGS-B.
#
# Single-point, so no mean/worst/std fitness.  Diagnostic flags gate most
# logging (gradient_norm, theta, ...); with a flag off the plotter draws an
# empty axes rather than skipping the panel.

PanelRegistry.register(
    AlgorithmChoice.LBFGSB,
    Panel(
        key=PanelKey.CONVERGENCE,
        title="Convergence",
        ylabel="Fitness (log)",
        series=(
            Series("best_fitness", "Best", color="tab:blue"),
            Series(
                "function_value", "f(x)", linestyle=LineStyle.DASHED, color="tab:green"
            ),
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
            Series(
                "gradient_norm", "||g||_2", linestyle=LineStyle.DASHED, color="tab:blue"
            ),
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
    # Non-default: subsumed by the multi-series convergence / gradient_norms
    # panels.
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
    # Iteration-axis variants, for comparing e.g. different line searches
    # per iteration, where an evals axis would penalize the variant that
    # spends more evaluations per iteration.  Non-default; opt in via
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


# BFGS.
#
# Single-point and gradient-based like L-BFGS-B, so it registers the same
# semantic keys (CONVERGENCE, STEP_SIZE, GRADIENT_NORM) and
# ``plot_comparison({...})`` overlays it against the other local methods
# without an explicit ``panels=``.  Everything except CONVERGENCE is gated
# by a diag_* flag.

PanelRegistry.register(
    AlgorithmChoice.BFGS,
    Panel(
        key=PanelKey.CONVERGENCE,
        title="Convergence",
        ylabel="Fitness (log)",
        series=(
            Series("best_fitness", "Best", color="tab:blue"),
            Series(
                "function_value", "f(x)", linestyle=LineStyle.DASHED, color="tab:green"
            ),
        ),
        yscale=YScale.LOG,
        floor=_FITNESS_FLOOR,
    ),
    Panel(
        key=PanelKey.GRADIENT_NORM,
        title="Gradient Norm",
        ylabel="||proj g||",
        field="gradient_norm",
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
        key=PanelKey.CURVATURE,
        title="BFGS Curvature y.s",
        ylabel="y . s",
        field="curvature",
        yscale=YScale.LOG,
    ),
    Panel(
        key=PanelKey.HESSIAN_CONDITION,
        title="Inverse-Hessian Condition Number",
        ylabel="kappa(H_k)",
        field="hessian_condition",
        yscale=YScale.LOG,
    ),
    # Non-default: subsumed by the convergence overlay, or by an
    # iteration-axis variant for line-search comparisons.
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


# Powell.
#
# Single-point and derivative-free: no population, no gradients.  One record
# per outer iteration (one sweep over the direction set), logged at the same
# boundary as scipy's per-iteration callback.  Diagnostic flags gate most
# fields (delta, step_norm, direction-set geometry).

PanelRegistry.register(
    AlgorithmChoice.POWELL,
    Panel(
        key=PanelKey.CONVERGENCE,
        title="Convergence",
        ylabel="Fitness (log)",
        series=(
            Series("best_fitness", "Best", color="tab:blue"),
            Series(
                "function_value", "f(x)", linestyle=LineStyle.DASHED, color="tab:green"
            ),
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
    # Non-default: subsumed by the convergence panel.
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


# Nelder-Mead.
#
# The simplex is the population, so the population panels (worst/mean/std
# fitness) apply.  STEP_SIZE maps to the simplex extent, the quantity tested
# against xatol.  CONDITION_NUMBER reads the vertex-covariance spectrum
# (diag_eigen).

PanelRegistry.register(
    AlgorithmChoice.NELDERMEAD,
    Panel(
        key=PanelKey.CONVERGENCE,
        title="Convergence",
        ylabel="Fitness (log)",
        series=(
            Series("best_fitness", "Best", color="tab:blue"),
            Series(
                "mean_fitness", "Mean", linestyle=LineStyle.DASHED, color="tab:green"
            ),
            Series(
                "worst_fitness", "Worst", linestyle=LineStyle.DOTTED, color="tab:red"
            ),
        ),
        yscale=YScale.LOG,
        floor=_FITNESS_FLOOR,
    ),
    Panel(
        key=PanelKey.STEP_SIZE,
        title="Simplex Extent",
        ylabel="max |sim - sim_0|",
        field="simplex_diameter",
        yscale=YScale.LOG,
    ),
    Panel(
        key=PanelKey.FITNESS_SPREAD,
        title="Fitness Spread",
        ylabel="max |f_0 - f_i|",
        field="fitness_spread",
        yscale=YScale.LOG,
    ),
    Panel(
        key=PanelKey.STD_FITNESS,
        title="Fitness Std Dev",
        ylabel="Std Dev",
        field="std_fitness",
        yscale=YScale.LOG,
    ),
    Panel(
        key=PanelKey.CONDITION_NUMBER,
        title="Simplex Condition Number",
        ylabel="kappa(cov)",
        field="condition_number",
        yscale=YScale.LOG,
    ),
    Panel(
        key=PanelKey.SIMPLEX_VOLUME,
        title="Simplex Volume",
        ylabel="vol (log)",
        field="simplex_volume",
        yscale=YScale.LOG,
    ),
    # Non-default: the operation timeline is a dissection tool, and worst
    # fitness is already in the convergence overlay.
    Panel(
        key=PanelKey.SIMPLEX_OPERATION,
        title="Simplex Operation",
        ylabel="op code",
        field="operation",
        yscale=YScale.LINEAR,
        default=False,
    ),
    Panel(
        key=PanelKey.WORST_FITNESS,
        title="Worst Fitness",
        ylabel="Worst Fitness",
        field="worst_fitness",
        yscale=YScale.LOG,
        floor=_FITNESS_FLOOR,
        default=False,
    ),
)


# Hessian-completed Nelder-Mead
#
# Every classic Nelder-Mead panel still applies -- the simplex is still the
# population and the same convergence tests still fire -- so those are
# re-registered under the same semantic keys, which is what lets
# ``PanelRegistry.common([NELDERMEAD, NELDERMEAD_HC])`` overlay the two
# optimizers panel for panel.  The model-step panels are the addition.

PanelRegistry.register(
    AlgorithmChoice.NELDERMEAD_HC,
    Panel(
        key=PanelKey.CONVERGENCE,
        title="Convergence",
        ylabel="Fitness (log)",
        series=(
            Series("best_fitness", "Best", color="tab:blue"),
            Series(
                "mean_fitness", "Mean", linestyle=LineStyle.DASHED, color="tab:green"
            ),
            Series(
                "worst_fitness", "Worst", linestyle=LineStyle.DOTTED, color="tab:red"
            ),
        ),
        yscale=YScale.LOG,
        floor=_FITNESS_FLOOR,
    ),
    Panel(
        key=PanelKey.STEP_SIZE,
        title="Simplex Extent",
        ylabel="max |sim - sim_0|",
        field="simplex_diameter",
        yscale=YScale.LOG,
    ),
    Panel(
        key=PanelKey.FITNESS_SPREAD,
        title="Fitness Spread",
        ylabel="max |f_0 - f_i|",
        field="fitness_spread",
        yscale=YScale.LOG,
    ),
    Panel(
        key=PanelKey.MODEL_STEPS,
        title="Model Steps",
        ylabel="cumulative count",
        series=(
            Series("model_attempts", "Attempted", color="tab:gray"),
            Series("model_accepted", "Accepted", color="tab:blue"),
            Series(
                "model_improvements",
                "Improved best",
                linestyle=LineStyle.DASHED,
                color="tab:green",
            ),
        ),
        yscale=YScale.LINEAR,
    ),
    # The Hadamard ratio, not the raw volume: it is scale-invariant, so it
    # separates a degenerate simplex from one that has simply converged.
    Panel(
        key=PanelKey.SIMPLEX_SHAPE_QUALITY,
        title="Simplex Shape Quality",
        ylabel="|det D| / prod |d_i|",
        field="simplex_shape_quality",
        yscale=YScale.LOG,
    ),
    Panel(
        key=PanelKey.TRUST_FACTOR,
        title="Trust-Region Factor",
        ylabel="radius / simplex extent",
        field="trust_factor",
        yscale=YScale.LOG,
    ),
    # Non-default: dissection tools for a single run.
    Panel(
        key=PanelKey.MODEL_RATIO,
        title="Trust-Region Ratio",
        ylabel="actual / predicted",
        field="model_ratio",
        yscale=YScale.LINEAR,
        default=False,
    ),
    Panel(
        key=PanelKey.CURVATURE_SCALE,
        title="Fitted Curvature Magnitude",
        ylabel="alpha",
        field="curvature_scale",
        yscale=YScale.LOG,
        default=False,
    ),
    Panel(
        key=PanelKey.CONDITION_NUMBER,
        title="Simplex Condition Number",
        ylabel="kappa(cov)",
        field="condition_number",
        yscale=YScale.LOG,
        default=False,
    ),
    Panel(
        key=PanelKey.SIMPLEX_VOLUME,
        title="Simplex Volume",
        ylabel="vol (log)",
        field="simplex_volume",
        yscale=YScale.LOG,
        default=False,
    ),
    Panel(
        key=PanelKey.SIMPLEX_OPERATION,
        title="Simplex Operation",
        ylabel="op code",
        field="operation",
        yscale=YScale.LINEAR,
        default=False,
    ),
)
