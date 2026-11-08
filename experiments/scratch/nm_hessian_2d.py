#!/usr/bin/env python3
"""Throwaway 2-D visualisation of Nelder-Mead seeded with a Hessian.

This is the ``experiments/conditioning`` exp1 cell (see
``experiments/conditioning/run_grid_local.sh``) shrunk to ``d = 2`` so the
whole thing can be *watched*: same rotated ill-conditioned Ellipsoid, same
shared ``x0``, same "deep" Nelder-Mead config, same conditioners -- but with
one dashboard frame per simplex iteration instead of an aggregate convergence
overlay.

The headline arm is the exp1 best case: the simplex shaped by the **inverse
Hessian** at ``x0`` (central-difference Hessian, SPD-regularised, exactly as
``run_hessian_stage`` computes it), against the isotropic ``I`` control.
Optional ``cK`` arms use the CMA-ES covariance after ``K * d`` generations,
the third conditioner family of exp1.

Nothing here is part of the framework: it imports declivity, it does not
extend it. Delete the file when the picture has served its purpose.

Run::

    PYTHONPATH=. uv run python experiments/scratch/nm_hessian_2d.py
    PYTHONPATH=. uv run python experiments/scratch/nm_hessian_2d.py --live
    PYTHONPATH=. uv run python experiments/scratch/nm_hessian_2d.py \
        --conditioners hessian,c4,identity --condition 1e4 --max-frames 200

The default variant is ``unbounded``.  ``--variant bounded`` (the other half
of the exp1 grid) is worth watching too, and at ``d = 2`` it is a different
picture entirely: the ``H^-1`` simplex is ~1000x longer along the flat axis
than the steep one, so its first reflections march along the valley, hit the
box wall, and every vertex gets clamped onto the same face.  The simplex
loses that dimension for good and NM converges onto the wall (``f ~ 1e4``)
while the isotropic ``I`` control walks to ``1e-16``.

Output lands in ``plots/nm2d/<tag>/`` (gitignored):

    frames/step_XXXX.png   one dashboard per recorded iteration
    final.png              the last frame
    progress.gif           the frames stitched together (--no-gif to skip)
    summary.txt            config + per-arm outcome
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, Rectangle
from matplotlib.ticker import MaxNLocator, ScalarFormatter
from numpy.typing import NDArray

from declivity.algorithms.cmaes.config import CMAESConfig
from declivity.algorithms.neldermead.config import NelderMeadConfig
from declivity.algorithms.neldermead.neldermead_optimizer import (
    NelderMeadOptimizer,
    SimplexOperation,
)
from declivity.benchmarking.cmaes_path import record_cmaes_path
from declivity.benchmarking.conditioning import snapshot_geometry
from declivity.benchmarking.problem import Problem
from declivity.utils.benchmark_functions import BenchmarkFunction
from declivity.utils.hessian import numerical_hessian, spd_regularize
from declivity.utils.initial_geometry import HessianScaling, InitialGeometry
from declivity.utils.initial_point_generator import UniformBoxInitialPointGenerator
from declivity.utils.population_initializers import CovarianceSimplexInitializer
from declivity.utils.repair_strategies import LamarckianRepair
from declivity.utils.stopping_conditions import MaxEvaluations

# Same sampling box / budgets as the exp1 defaults.
SAMPLING_LOWER = -100.0
SAMPLING_UPPER = 100.0
SAMPLING_SPAN = SAMPLING_UPPER - SAMPLING_LOWER
LOCAL_BUDGET_PER_DIM = 500
CMAES_EVALUATIONS_PER_DIM = 10_000
SIGMA0 = SAMPLING_SPAN / 5.0
SIMPLEX_BASE_SIZE = 0.1 * SAMPLING_SPAN

HESSIAN_COLOR = "#ffffff"
IDENTITY_COLOR = "#f5a623"
NEWTON_COLOR = "#63e6be"
CMAES_COLORS = ("#4dabf7", "#51cf66", "#e599f7", "#ff8787")

GAP_FLOOR = 1e-16
DIM = 2


# ---------------------------------------------------------------- objective


class Ellipsoid2D(BenchmarkFunction):
    """``f(x) = z_0^2 + condition * z_1^2`` with ``z = R x``.

    ``condition=1e6`` with ``R = I`` is bit-for-bit
    :class:`declivity.utils.benchmark_functions.Ellipsoid` at ``d = 2``; the
    knob exists only so the contour plot can be made readable. ``batch`` is
    the vectorised path the contour grids use.
    """

    def __init__(
        self,
        condition: float = 1e6,
        rotation: NDArray[np.float64] | None = None,
        lower: float = SAMPLING_LOWER,
        upper: float = SAMPLING_UPPER,
    ) -> None:
        super().__init__(DIM)
        self.condition = float(condition)
        self.scales = np.array([1.0, float(condition)])
        self.rotation = np.eye(DIM) if rotation is None else np.asarray(rotation, float)
        self._lower = lower
        self._upper = upper

    def __call__(self, x: NDArray[np.float64]) -> float:
        z = self.rotation @ np.asarray(x, dtype=float)
        return float(np.sum(self.scales * z**2))

    def batch(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        z = np.asarray(points, dtype=float) @ self.rotation.T
        return z**2 @ self.scales

    def gradient(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        z = self.rotation @ np.asarray(x, dtype=float)
        return self.rotation.T @ (2.0 * self.scales * z)

    @property
    def bounds(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return (
            self._lower * np.ones(DIM),
            self._upper * np.ones(DIM),
        )

    @property
    def global_minimum(self) -> tuple[NDArray[np.float64], float]:
        return np.zeros(DIM), 0.0


def rotation_matrix(seed: int) -> NDArray[np.float64]:
    """The exp1 rotation generator (``common.rotation_matrix``) at ``d = 2``."""
    rng = np.random.default_rng([911, DIM, seed])
    q, r = np.linalg.qr(rng.standard_normal((DIM, DIM)))
    return q * np.sign(np.diag(r))


def build_problem(condition: float, rotate: bool, bounded: bool, seed: int) -> Problem:
    function = Ellipsoid2D(
        condition=condition,
        rotation=rotation_matrix(seed) if rotate else None,
    )
    return Problem(
        name=f"Ellipsoid2D(cond={condition:g}{', rotated' if rotate else ''})",
        function=function,
        dimensions=DIM,
        lower_bound=SAMPLING_LOWER if bounded else -np.inf,
        upper_bound=SAMPLING_UPPER if bounded else np.inf,
        gradient=None,
        initial_point_generator=UniformBoxInitialPointGenerator(
            SAMPLING_LOWER, SAMPLING_UPPER
        ),
    )


# -------------------------------------------------------------- conditioners


def hessian_geometry(
    problem: Problem, x0: NDArray[np.float64], scaling: str
) -> InitialGeometry:
    """exp1's ``H^-1`` arm: FD Hessian at x0, SPD-regularised, as curvature."""
    matrix = spd_regularize(numerical_hessian(problem.function, x0))
    return InitialGeometry.from_curvature(matrix, DIM, scaling=scaling)


def newton_scale(
    problem: Problem, x0: NDArray[np.float64], geometry: InitialGeometry
) -> float:
    """``sqrt(g' H^-1 g)`` at x0 -- the Newton step measured in the H metric.

    For a quadratic this is *exactly* the distance from x0 to the optimum in
    whitened coordinates ``y = H^(1/2) x``, i.e. the only length scale the
    simplex should be sized by. Costs ``2 * d`` extra evaluations; equal to
    ``sqrt(2 * (f(x0) - f*))`` when ``f*`` is known, for zero evaluations.
    """
    eps = 1e-5
    gradient = np.empty(DIM)
    for i in range(DIM):
        step = np.zeros(DIM)
        step[i] = eps * max(1.0, abs(float(x0[i])))
        gradient[i] = (
            float(problem.function(x0 + step)) - float(problem.function(x0 - step))
        ) / (2.0 * step[i])
    return float(np.sqrt(max(gradient @ geometry.solve(gradient), 0.0)))


def cmaes_geometry(
    problem: Problem,
    x0: NDArray[np.float64],
    seed: int,
    k: int,
    scaling: str,
) -> InitialGeometry:
    """exp1's ``C_{k*d}`` arm: the CMA-ES covariance after ``k * d`` generations."""
    path = record_cmaes_path(
        problem,
        x0,
        seed,
        lambda d: CMAESConfig(dimensions=d, sigma=SIGMA0),
        snapshot_interval=max(1, k * DIM),
        max_evaluations=CMAES_EVALUATIONS_PER_DIM * DIM,
    )
    snapshot = path.snapshot_at_or_before(k * DIM)
    if snapshot is None:
        raise RuntimeError(
            f"CMA-ES recorded no snapshot at or before iteration {k * DIM}."
        )
    return snapshot_geometry(snapshot, "inverse", scaling)


# ---------------------------------------------------------------------- runs


@dataclass
class Arm:
    """One conditioner's Nelder-Mead run, unrolled per iteration."""

    key: str
    label: str
    color: str
    simplices: list[NDArray[np.float64]] = field(default_factory=list)
    evaluations: list[int] = field(default_factory=list)
    gaps: list[float] = field(default_factory=list)
    diameters: list[float] = field(default_factory=list)
    operations: list[str] = field(default_factory=list)
    message: str = ""

    @property
    def steps(self) -> int:
        return len(self.simplices)

    def at(self, index: int) -> int:
        """Frame index clamped into this arm's (possibly shorter) timeline."""
        return min(index, self.steps - 1)

    @property
    def best_path(self) -> NDArray[np.float64]:
        return np.array([simplex[0] for simplex in self.simplices])


def make_initializer(
    geometry: InitialGeometry,
    config: NelderMeadConfig,
    base_size: float,
    whitened: bool,
) -> CovarianceSimplexInitializer:
    """The initial-simplex builder for one arm.

    ``whitened=False`` is exactly what ``NelderMeadOptimizer(initial_geometry=)``
    builds internally (exp1's arms): relative anisotropy, longest edge
    ``base_size``.  ``whitened=True`` (``normalize=False``) makes edge ``k``
    ``base_size * D_k``, i.e. a simplex that is *regular in the H metric* with
    whitened side ``base_size`` -- same shape, size now expressed in the metric
    the curvature defines.
    """
    return CovarianceSimplexInitializer(
        geometry,
        base_size=base_size,
        normalize=not whitened,
        min_step=100.0 * config.xatol,
    )


def run_arm(
    key: str,
    label: str,
    color: str,
    problem: Problem,
    x0: NDArray[np.float64],
    initializer: CovarianceSimplexInitializer,
    config: NelderMeadConfig,
    budget: int,
    seed: int,
    optimum: float,
) -> Arm:
    handler = problem.resolved_constraint_handler()

    # The optimizer builds its initial simplex inside optimize() and only logs
    # from iteration 1, so rebuild it here (same initializer instance) purely
    # to render frame 0 -- the conditioner's footprint before any move.
    simplex0 = LamarckianRepair().repair_population(
        initializer.generate_population(
            np.random.default_rng(seed), x0, DIM + 1, handler
        ),
        handler,
    )
    f0 = np.array([float(problem.function(vertex)) for vertex in simplex0])
    simplex0 = simplex0[np.argsort(f0)]
    f0 = np.sort(f0)

    optimizer = NelderMeadOptimizer(
        func=problem.function,
        initial_point=x0,
        config=config,
        constraint_handler=handler,
        stopping_condition=MaxEvaluations(budget),
        lower_bounds=problem.lower_bound,
        upper_bounds=problem.upper_bound,
        seed=seed,
        population_initializer=initializer,
    )
    result = optimizer.optimize()
    logs = result.diagnostic

    arm = Arm(key=key, label=label, color=color, message=result.message)
    arm.simplices.append(np.asarray(simplex0, dtype=float))
    arm.evaluations.append(DIM + 1)
    arm.gaps.append(max(float(f0[0]) - optimum, GAP_FLOOR))
    arm.diameters.append(float(np.max(np.abs(simplex0[1:] - simplex0[0]))))
    arm.operations.append("initial simplex")

    for index, simplex in enumerate(logs.population):
        arm.simplices.append(np.asarray(simplex, dtype=float))
        arm.evaluations.append(int(logs.evaluations[index]))
        arm.gaps.append(max(float(logs.best_fitness[index]) - optimum, GAP_FLOOR))
        arm.diameters.append(max(float(logs.simplex_diameter[index]), 1e-300))
        arm.operations.append(SimplexOperation(logs.operation[index]).name.lower())

    return arm


# ------------------------------------------------------------------ plotting


def apply_dark_style() -> None:
    plt.style.use("dark_background")
    plt.rcParams.update(
        {
            "figure.facecolor": "#111111",
            "axes.facecolor": "#111111",
            "savefig.facecolor": "#111111",
            "grid.alpha": 0.25,
            "legend.framealpha": 0.35,
            "font.size": 9,
        }
    )


def whitening(function: Ellipsoid2D) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """``(H^(1/2), H^(-1/2))`` for the objective's (constant) Hessian.

    ``y = H^(1/2) x`` is the metric the curvature defines: the ellipsoid's
    contours become circles and a correctly conditioned simplex becomes
    regular.  Nelder-Mead is affine invariant, so its whole run is *the same
    run* seen in either frame -- which is why this panel is the honest place
    to judge whether a simplex is well shaped.
    """
    hessian = 2.0 * function.rotation.T @ np.diag(function.scales) @ function.rotation
    eigenvalues, eigenvectors = np.linalg.eigh(hessian)
    eigenvalues = np.maximum(eigenvalues, 1e-300)
    root = (eigenvectors * np.sqrt(eigenvalues)) @ eigenvectors.T
    root_inverse = (eigenvectors / np.sqrt(eigenvalues)) @ eigenvectors.T
    return root, root_inverse


def grid_values(
    function: Ellipsoid2D,
    x_lo: float,
    x_hi: float,
    y_lo: float,
    y_hi: float,
    resolution: int,
    to_x: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    xs = np.linspace(x_lo, x_hi, resolution)
    ys = np.linspace(y_lo, y_hi, resolution)
    mesh_x, mesh_y = np.meshgrid(xs, ys)
    points = np.column_stack([mesh_x.ravel(), mesh_y.ravel()])
    if to_x is not None:
        points = points @ to_x.T
    return xs, ys, function.batch(points).reshape(mesh_x.shape)


def draw_landscape(
    ax,
    function: Ellipsoid2D,
    optimum: float,
    window: tuple[float, float, float, float],
    resolution: int,
    to_x: NDArray[np.float64] | None = None,
) -> None:
    """Filled ``log10(f - f*)`` contours: readable at any condition number.

    ``to_x`` maps the plotted coordinates back to the objective's own, so the
    same routine draws both the x-space and the whitened view.
    """
    x_lo, x_hi, y_lo, y_hi = window
    xs, ys, values = grid_values(function, x_lo, x_hi, y_lo, y_hi, resolution, to_x)
    logged = np.log10(np.maximum(values - optimum, GAP_FLOOR))
    ax.contourf(xs, ys, logged, levels=40, cmap="magma")
    ax.contour(xs, ys, logged, levels=12, colors="white", alpha=0.18, linewidths=0.5)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel("$x_0$")
    ax.set_ylabel("$x_1$")


def draw_simplex(ax, simplex: NDArray[np.float64], color: str, alpha: float) -> None:
    ax.add_patch(
        Polygon(
            simplex,
            closed=True,
            facecolor=color,
            edgecolor=color,
            alpha=alpha,
            linewidth=1.6,
            zorder=5,
        )
    )
    # Every vertex marked: at a 1000:1 aspect ratio the triangle is a
    # sub-pixel sliver and only the dots show it is really a simplex.
    ax.plot(
        simplex[:, 0], simplex[:, 1], "o", color=color, markersize=3.0, zorder=6, ls=""
    )
    ax.plot(simplex[0, 0], simplex[0, 1], "o", color=color, markersize=5.5, zorder=6)


def simplex_aspect(simplex: NDArray[np.float64]) -> float:
    """Longest / shortest singular value of the edge matrix."""
    edges = simplex[1:] - simplex[0]
    values = np.linalg.svd(edges, compute_uv=False)
    smallest = float(values.min())
    return float(values.max()) / smallest if smallest > 0 else float("inf")


def render_frame(
    fig,
    axes: dict[str, object],
    arms: list[Arm],
    frame: int,
    problem: Problem,
    function: Ellipsoid2D,
    x0: NDArray[np.float64],
    optimum: float,
    window: tuple[float, float, float, float],
    resolution: int,
    bounded: bool,
    whiten: tuple[NDArray[np.float64], NDArray[np.float64]],
    focus: int = 0,
) -> None:
    for ax in axes.values():
        ax.clear()  # pyright: ignore[reportAttributeAccessIssue]

    land = axes["land"]
    zoom = axes["zoom"]
    conv = axes["conv"]
    diam = axes["diam"]
    shape = axes["shape"]
    root, root_inverse = whiten

    # --- full landscape + trajectories -------------------------------------
    draw_landscape(land, function, optimum, window, resolution)
    if bounded:
        land.add_patch(  # pyright: ignore[reportAttributeAccessIssue]
            Rectangle(
                (SAMPLING_LOWER, SAMPLING_LOWER),
                SAMPLING_SPAN,
                SAMPLING_SPAN,
                fill=False,
                edgecolor="#66d9e8",
                linestyle=":",
                linewidth=1.0,
                zorder=3,
            )
        )
    for arm in arms:
        index = arm.at(frame)
        path = arm.best_path[: index + 1]
        land.plot(  # pyright: ignore[reportAttributeAccessIssue]
            path[:, 0], path[:, 1], "-", color=arm.color, lw=1.3, alpha=0.9, zorder=4
        )
        # Low fill: a newton-sized simplex is far larger than the plotted box
        # early on, and a solid patch would wash the landscape out.
        draw_simplex(land, arm.simplices[0], arm.color, 0.08)
        draw_simplex(land, arm.simplices[index], arm.color, 0.20)
    land.plot(  # pyright: ignore[reportAttributeAccessIssue]
        x0[0], x0[1], "*", color="#ffffff", markersize=11, zorder=7, label="$x_0$"
    )
    land.plot(  # pyright: ignore[reportAttributeAccessIssue]
        0.0, 0.0, "P", color="#69db7c", markersize=9, zorder=7, label="$x^*$"
    )
    land.set_title(  # pyright: ignore[reportAttributeAccessIssue]
        f"{problem.name} — $\\log_{{10}}(f - f^*)$", fontsize=10
    )
    land.legend(loc="upper right", fontsize=8)  # pyright: ignore[reportAttributeAccessIssue]

    # --- zoom, following the focus arm's simplex ---------------------------
    primary = arms[focus]
    p_index = primary.at(frame)
    simplex = primary.simplices[p_index]
    center = simplex.mean(axis=0)
    # Widen the window to at least a fraction of the distance still to travel,
    # so a collapsed simplex is shown *in context* rather than as a flat
    # single-colour patch at 1e-12 across.
    half = max(
        2.5 * float(np.max(np.abs(simplex - center))),
        0.03 * float(np.linalg.norm(center)),
        1e-12,
    )
    zoom_window = (
        float(center[0] - half),
        float(center[0] + half),
        float(center[1] - half),
        float(center[1] + half),
    )
    draw_landscape(zoom, function, optimum, zoom_window, max(80, resolution // 2))
    for arm in arms:
        index = arm.at(frame)
        path = arm.best_path[: index + 1]
        zoom.plot(  # pyright: ignore[reportAttributeAccessIssue]
            path[:, 0], path[:, 1], "-", color=arm.color, lw=1.2, alpha=0.9, zorder=4
        )
        draw_simplex(zoom, arm.simplices[index], arm.color, 0.45)
    zoom.plot(0.0, 0.0, "P", color="#69db7c", markersize=9, zorder=7)  # pyright: ignore[reportAttributeAccessIssue]
    for axis in (zoom.xaxis, zoom.yaxis):  # pyright: ignore[reportAttributeAccessIssue]
        axis.set_major_formatter(ScalarFormatter(useOffset=False))
        axis.set_major_locator(MaxNLocator(nbins=5))
    zoom.tick_params(axis="x", labelrotation=20)  # pyright: ignore[reportAttributeAccessIssue]
    zoom.set_title(  # pyright: ignore[reportAttributeAccessIssue]
        f"zoom on {primary.label} — half-width {half:.2e}", fontsize=10
    )
    status = "\n".join(
        f"{arm.label}  {arm.operations[arm.at(frame)]}"
        f"{'  [done]' if frame >= arm.steps - 1 else ''}"
        for arm in arms
    )
    zoom.text(  # pyright: ignore[reportAttributeAccessIssue]
        0.02,
        0.02,
        status,
        transform=zoom.transAxes,  # pyright: ignore[reportAttributeAccessIssue]
        fontsize=8,
        va="bottom",
        ha="left",
        color="#ffffff",
        bbox={"facecolor": "#111111", "alpha": 0.55, "edgecolor": "none", "pad": 3},
        zorder=8,
    )

    # --- whitened view: y = H^(1/2) x --------------------------------------
    # Here the objective is a circle and a *correctly shaped* simplex is
    # regular.  This is where the "why is it a line?" question is answered:
    # H^-1's sliver in x-space is the regular one here, and I's tidy triangle
    # is the sliver.
    w_simplices = {arm.key: arm.simplices[arm.at(frame)] @ root.T for arm in arms}
    w_center = w_simplices[primary.key].mean(axis=0)
    w_half = max(
        2.5 * float(np.max(np.abs(w_simplices[primary.key] - w_center))),
        0.03 * float(np.linalg.norm(w_center)),
        1e-12,
    )
    draw_landscape(
        shape,
        function,
        optimum,
        (
            float(w_center[0] - w_half),
            float(w_center[0] + w_half),
            float(w_center[1] - w_half),
            float(w_center[1] + w_half),
        ),
        max(80, resolution // 2),
        to_x=root_inverse,
    )
    for arm in arms:
        index = arm.at(frame)
        w_path = arm.best_path[: index + 1] @ root.T
        shape.plot(  # pyright: ignore[reportAttributeAccessIssue]
            w_path[:, 0],
            w_path[:, 1],
            "-",
            color=arm.color,
            lw=1.2,
            alpha=0.9,
            zorder=4,
        )
        draw_simplex(shape, w_simplices[arm.key], arm.color, 0.45)
    shape.plot(0.0, 0.0, "P", color="#69db7c", markersize=9, zorder=7)  # pyright: ignore[reportAttributeAccessIssue]
    for axis in (shape.xaxis, shape.yaxis):  # pyright: ignore[reportAttributeAccessIssue]
        axis.set_major_formatter(ScalarFormatter(useOffset=False))
        axis.set_major_locator(MaxNLocator(nbins=5))
    shape.tick_params(axis="x", labelrotation=20)  # pyright: ignore[reportAttributeAccessIssue]
    shape.set_xlabel("$y_0$")  # pyright: ignore[reportAttributeAccessIssue]
    shape.set_ylabel("$y_1$")  # pyright: ignore[reportAttributeAccessIssue]
    shape.set_title(  # pyright: ignore[reportAttributeAccessIssue]
        "whitened  $y = H^{1/2}x$  (contours are circles)", fontsize=10
    )
    aspects = "\n".join(
        f"{arm.label}  aspect  x:{simplex_aspect(arm.simplices[arm.at(frame)]):.3g}"
        f"   y:{simplex_aspect(w_simplices[arm.key]):.3g}"
        for arm in arms
    )
    shape.text(  # pyright: ignore[reportAttributeAccessIssue]
        0.02,
        0.02,
        aspects,
        transform=shape.transAxes,  # pyright: ignore[reportAttributeAccessIssue]
        fontsize=8,
        va="bottom",
        ha="left",
        color="#ffffff",
        bbox={"facecolor": "#111111", "alpha": 0.55, "edgecolor": "none", "pad": 3},
        zorder=8,
    )

    # --- convergence -------------------------------------------------------
    for arm in arms:
        index = arm.at(frame)
        conv.plot(  # pyright: ignore[reportAttributeAccessIssue]
            arm.evaluations, arm.gaps, "-", color=arm.color, lw=1.0, alpha=0.25
        )
        conv.plot(  # pyright: ignore[reportAttributeAccessIssue]
            arm.evaluations[: index + 1],
            arm.gaps[: index + 1],
            "-",
            color=arm.color,
            lw=1.8,
            label=f"{arm.label}: {arm.gaps[index]:.2e}",
        )
        conv.plot(  # pyright: ignore[reportAttributeAccessIssue]
            arm.evaluations[index], arm.gaps[index], "o", color=arm.color, markersize=5
        )
    conv.set_yscale("log")  # pyright: ignore[reportAttributeAccessIssue]
    conv.set_xlabel("evaluations")  # pyright: ignore[reportAttributeAccessIssue]
    conv.set_ylabel("$f(x_{best}) - f^*$")  # pyright: ignore[reportAttributeAccessIssue]
    conv.grid(True, alpha=0.25)  # pyright: ignore[reportAttributeAccessIssue]
    conv.legend(loc="upper right", fontsize=8)  # pyright: ignore[reportAttributeAccessIssue]

    # --- simplex extent ----------------------------------------------------
    for arm in arms:
        index = arm.at(frame)
        steps = np.arange(arm.steps)
        diam.plot(  # pyright: ignore[reportAttributeAccessIssue]
            steps, arm.diameters, "-", color=arm.color, lw=1.0, alpha=0.25
        )
        diam.plot(  # pyright: ignore[reportAttributeAccessIssue]
            steps[: index + 1],
            arm.diameters[: index + 1],
            "-",
            color=arm.color,
            lw=1.8,
        )
        diam.plot(  # pyright: ignore[reportAttributeAccessIssue]
            index, arm.diameters[index], "o", color=arm.color, markersize=5
        )
    diam.set_yscale("log")  # pyright: ignore[reportAttributeAccessIssue]
    diam.set_xlabel("iteration")  # pyright: ignore[reportAttributeAccessIssue]
    diam.set_ylabel("simplex extent")  # pyright: ignore[reportAttributeAccessIssue]
    diam.grid(True, alpha=0.25)  # pyright: ignore[reportAttributeAccessIssue]

    fig.suptitle(
        f"Nelder-Mead in 2D — iteration {frame} / {max(arm.steps for arm in arms) - 1}"
        f"   ({primary.label}: {primary.evaluations[p_index]} evaluations, "
        f"gap {primary.gaps[p_index]:.3e})",
        fontsize=12,
    )


# ---------------------------------------------------------------------- main


def parse_conditioners(spec: str) -> list[str]:
    keys = [token.strip() for token in spec.split(",") if token.strip()]
    for key in keys:
        if key in ("hessian", "newton", "identity"):
            continue
        if key.startswith("c") and key[1:].isdigit():
            continue
        raise ValueError(
            f"Unknown conditioner {key!r}; expected 'hessian', 'newton', "
            f"'identity', or 'cK'."
        )
    if not keys:
        raise ValueError("Need at least one conditioner.")
    return keys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--conditioners",
        default="hessian,newton,identity",
        help=(
            "comma-separated: hessian (exp1's arm), newton (same shape, sized by "
            "the Newton step), identity, cK (CMA-ES covariance after K*d gens)"
        ),
    )
    parser.add_argument(
        "--newton-factor",
        type=float,
        default=1.0,
        help="multiplier on the newton arm's whitened simplex side",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--condition",
        type=float,
        default=1e6,
        help="Ellipsoid condition number (1e6 = the exp1 objective at d=2)",
    )
    parser.add_argument(
        "--no-rotate", action="store_true", help="axis-aligned objective"
    )
    parser.add_argument(
        "--variant",
        default="unbounded",
        choices=("unbounded", "bounded"),
        help="exp1 variant; 'bounded' clamps to [-100, 100]^2 (see the module docstring)",
    )
    parser.add_argument(
        "--focus",
        default=None,
        help="conditioner the zoom panel follows (default: the first one)",
    )
    parser.add_argument(
        "--scaling",
        default=str(HessianScaling.NONE),
        choices=[str(value) for value in HessianScaling],
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=LOCAL_BUDGET_PER_DIM * DIM,
        help="evaluation budget per arm",
    )
    parser.add_argument(
        "--view",
        type=float,
        default=SAMPLING_UPPER,
        help="half-width of the fixed landscape panel (default: the sampling box)",
    )
    parser.add_argument("--max-frames", type=int, default=120)
    parser.add_argument("--resolution", type=int, default=220)
    parser.add_argument("--dpi", type=int, default=110)
    parser.add_argument("--fps", type=float, default=6.0, help="live / GIF frame rate")
    parser.add_argument("--live", action="store_true", help="also animate on screen")
    parser.add_argument("--no-gif", action="store_true")
    parser.add_argument("--out-root", type=Path, default=Path("plots/nm2d"))
    args = parser.parse_args()

    keys = parse_conditioners(args.conditioners)
    rotate = not args.no_rotate
    bounded = args.variant == "bounded"
    if args.focus is not None and args.focus not in keys:
        parser.error(f"--focus {args.focus!r} is not among the conditioners {keys}.")

    apply_dark_style()
    if not args.live:
        plt.switch_backend("Agg")

    problem = build_problem(args.condition, rotate, bounded, args.seed)
    function = problem.function
    assert isinstance(function, Ellipsoid2D)
    whiten = whitening(function)
    x0 = problem.starting_point(args.seed)
    optimum = 0.0
    config = NelderMeadConfig(
        dimensions=DIM,
        xatol=1e-10,
        fatol=1e-12,
        adaptive=True,
        diag_pop=True,
        diag_operations=True,
    )

    print(f"problem : {problem.name}, {'bounded' if bounded else 'unbounded'}")
    print(f"x0      : {np.array2string(x0, precision=4)}  f(x0) = {function(x0):.6e}")
    print(f"budget  : {args.budget} evaluations per arm, scaling={args.scaling}")

    arms: list[Arm] = []
    cmaes_index = 0
    for key in keys:
        whitened = False
        base_size = SIMPLEX_BASE_SIZE
        if key == "hessian":
            geometry = hessian_geometry(problem, x0, args.scaling)
            label, color = "$H^{-1}$", HESSIAN_COLOR
        elif key == "newton":
            # Same H^-1 shape, sized in the metric H defines instead of in
            # x-space: the fix for exp1's arm (see the module docstring).
            geometry = hessian_geometry(problem, x0, args.scaling)
            base_size = args.newton_factor * newton_scale(problem, x0, geometry)
            whitened = True
            label, color = "$H^{-1}$ newton-sized", NEWTON_COLOR
        elif key == "identity":
            geometry = InitialGeometry.identity(DIM, scaling=args.scaling)
            label, color = "$I$", IDENTITY_COLOR
        else:
            k = int(key[1:])
            geometry = cmaes_geometry(problem, x0, args.seed, k, args.scaling)
            label = f"$C_{{{k * DIM}}}$"
            color = CMAES_COLORS[cmaes_index % len(CMAES_COLORS)]
            cmaes_index += 1

        arm = run_arm(
            key,
            label,
            color,
            problem,
            x0,
            make_initializer(geometry, config, base_size, whitened),
            config,
            args.budget,
            args.seed,
            optimum,
        )
        arms.append(arm)
        print(
            f"  {key:<9} {arm.steps - 1:>5} iterations, "
            f"final gap {arm.gaps[-1]:.3e} after {arm.evaluations[-1]} evals "
            f"({arm.message})"
        )

    # Frame plan: one dashboard per iteration, thinned to --max-frames.
    total = max(arm.steps for arm in arms)
    count = min(args.max_frames, total)
    frames = sorted(set(np.linspace(0, total - 1, count).round().astype(int).tolist()))

    # The landscape panel stays pinned to the sampling box so the objective's
    # shape is always readable; unbounded excursions simply leave the frame
    # (the zoom panel follows them).
    view = args.view * 1.05
    window = (-view, view, -view, view)

    tag_parts = [
        f"cond{args.condition:g}",
        f"rot{int(rotate)}",
        args.variant,
        f"seed{args.seed:02d}",
        args.scaling,
        "-".join(keys),
    ]
    out_dir = args.out_root / "_".join(tag_parts)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplot_mosaic(
        [["land", "zoom", "shape"], ["conv", "conv", "diam"]],
        figsize=(16.5, 8.5),
        height_ratios=[1.4, 1.0],
    )
    live = args.live
    if live:
        try:
            plt.ion()
            fig.show()
        except Exception as error:
            print(f"  (no interactive display: {error}; frames are still written)")
            live = False

    written: list[Path] = []
    for number, frame in enumerate(frames):
        render_frame(
            fig,
            axes,  # pyright: ignore[reportArgumentType]
            arms,
            frame,
            problem,
            function,
            x0,
            optimum,
            window,
            args.resolution,
            bounded,
            whiten,
            focus=keys.index(args.focus) if args.focus else 0,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        path = frames_dir / f"step_{number:04d}.png"
        fig.savefig(path, dpi=args.dpi)
        written.append(path)
        if live:
            plt.pause(max(1.0 / args.fps, 1e-3))
        if number % 10 == 0 or number == len(frames) - 1:
            print(f"  frame {number + 1}/{len(frames)} (iteration {frame})")

    fig.savefig(out_dir / "final.png", dpi=args.dpi)
    plt.close(fig)

    if not args.no_gif and written:
        try:
            from PIL import Image

            images = [Image.open(path) for path in written]
            images[0].save(
                out_dir / "progress.gif",
                save_all=True,
                append_images=images[1:],
                duration=int(1000 / args.fps),
                loop=0,
            )
        except Exception as error:
            print(f"  (no GIF: {error})")

    summary = [
        f"problem            : {problem.name}",
        f"variant            : {'bounded' if bounded else 'unbounded'}",
        f"seed               : {args.seed}",
        f"x0                 : {np.array2string(x0, precision=6)}",
        f"f(x0)              : {function(x0):.6e}",
        f"budget             : {args.budget}",
        f"scaling            : {args.scaling}",
        f"simplex base size  : {SIMPLEX_BASE_SIZE}",
        f"config             : {config}",
        "",
    ]
    for arm in arms:
        summary.append(
            f"{arm.key:<9} label={arm.label:<10} iterations={arm.steps - 1:<5} "
            f"evals={arm.evaluations[-1]:<6} final_gap={arm.gaps[-1]:.6e}  "
            f"({arm.message})"
        )
    (out_dir / "summary.txt").write_text("\n".join(summary) + "\n")

    print(f"\n{len(written)} frames -> {frames_dir}")
    print(f"final    -> {out_dir / 'final.png'}")
    if not args.no_gif:
        print(f"animation-> {out_dir / 'progress.gif'}")
    print(f"summary  -> {out_dir / 'summary.txt'}")


if __name__ == "__main__":
    main()
