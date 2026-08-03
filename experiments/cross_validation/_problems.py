"""Shared problem specifications for the cross-validation harnesses.

Each :class:`ProblemSpec` bundles the benchmark function, initial point,
bounds, initial step size, and known global optimum into one place so
that :mod:`des_vs_reference` and :mod:`mfcmaes_vs_reference` can iterate
over the same battery of problems with a single source of truth.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from declivity.cec import CECEdition, CECProblem
from declivity.utils.benchmark_functions import (
    Ackley,
    BenchmarkFunction,
    Ellipsoid,
    Rastrigin,
    Rosenbrock,
    Sphere,
)


@dataclass
class ProblemSpec:
    name: str
    """Display name used in plot titles and file tags (PascalCase OK)."""

    fn_factory: Callable[[int], BenchmarkFunction]
    """Build the benchmark function for the spec's dimension."""

    x0_factory: Callable[[int], NDArray[np.float64]]
    """Build the fixed initial point ``x0``."""

    lower: float
    upper: float
    sigma: float
    """Initial step size — only used by the CMA-ES family.  DES ignores it."""

    f_star: float
    """Known (or known-approximate) value of the global optimum."""

    dim: int = 10
    floor: float = 1e-12
    """Log-y plotting floor.  Values below this are clipped on log
    plots so the y-axis does not blow up to ``-∞``."""


PROBLEMS: dict[str, ProblemSpec] = {
    "ellipsoid_d10": ProblemSpec(
        name="Ellipsoid",
        dim=10,
        fn_factory=lambda d: Ellipsoid(d),
        x0_factory=lambda d: np.ones(d),
        lower=-5.0,
        upper=5.0,
        sigma=0.5,
        f_star=0.0,
    ),
    "rosenbrock_d10": ProblemSpec(
        name="Rosenbrock",
        dim=10,
        fn_factory=lambda d: Rosenbrock(d),
        x0_factory=lambda d: np.full(d, 0.5),
        lower=-5.0,
        upper=5.0,
        sigma=0.5,
        f_star=0.0,
    ),
    "rastrigin_d10": ProblemSpec(
        name="Rastrigin",
        dim=10,
        fn_factory=lambda d: Rastrigin(d),
        x0_factory=lambda d: np.full(d, 1.5),
        lower=-5.12,
        upper=5.12,
        sigma=0.5,
        f_star=0.0,
    ),
    "ackley_d10": ProblemSpec(
        name="Ackley",
        dim=10,
        fn_factory=lambda d: Ackley(d),
        x0_factory=lambda d: np.full(d, 1.5),
        lower=-32.0,
        upper=32.0,
        sigma=0.5,
        f_star=0.0,
    ),
    "sphere_d10": ProblemSpec(
        name="Sphere",
        dim=10,
        fn_factory=lambda d: Sphere(d),
        x0_factory=lambda d: np.ones(d),
        lower=-5.0,
        upper=5.0,
        sigma=0.5,
        f_star=0.0,
    ),
    "cec17_F10_d10": ProblemSpec(
        name="CEC17_F10",
        dim=10,
        fn_factory=lambda d: CECProblem(CECEdition.CEC2017, 10, d),
        x0_factory=lambda d: np.full(d, 50.0),
        lower=-100.0,
        upper=100.0,
        sigma=20.0,
        f_star=1e3,
        floor=1.0,
    ),
    "cec17_F3_d10": ProblemSpec(
        name="CEC17_F3",
        dim=10,
        fn_factory=lambda d: CECProblem(CECEdition.CEC2017, 3, d),
        x0_factory=lambda d: np.full(d, 50.0),
        lower=-100.0,
        upper=100.0,
        sigma=20.0,
        f_star=3e2,
        floor=1.0,
    ),
    "cec17_F5_d10": ProblemSpec(
        name="CEC17_F5",
        dim=10,
        fn_factory=lambda d: CECProblem(CECEdition.CEC2017, 5, d),
        x0_factory=lambda d: np.full(d, 50.0),
        lower=-100.0,
        upper=100.0,
        sigma=20.0,
        f_star=5e2,
        floor=1.0,
    ),
}
