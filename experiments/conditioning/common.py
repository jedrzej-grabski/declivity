"""Shared machinery for the conditioning experiment suite.

Both experiments are staged pipelines over one persisted artifact store, so a
study can be re-entered at any point (locally or on a remote worker) without
recomputing what already exists:

    setup/    per (dim, seed): x0 + rotation matrix        (cheap, deterministic)
    cmaes/    per (variant, dim[, function], seed): CMAESPath (expensive, cached)
    hessian/  per (dim, seed): FD Hessian at x0            (exp1 only)
    local/    per (scaling, dim, function): meta.yaml + one seedNN.parquet
              per seed, batching every optimizer/snapshot/alone run for that
              seed into one file                            (exp2 only)
    benchmarks/  Benchmark traces.parquet per stage           (what --replot reads)

Every stage checks for existing artifacts and skips them unless forced.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml
from cecpy.benchmark import CECEdition
from matplotlib.colors import to_hex
from matplotlib.pyplot import get_cmap
from numpy.typing import NDArray

from declivity.algorithms.bfgs.config import BFGSConfig
from declivity.algorithms.choices import AlgorithmChoice
from declivity.algorithms.cmaes.config import CMAESConfig, default_population_size
from declivity.algorithms.lbfgsb.config import LBFGSBConfig
from declivity.algorithms.neldermead.config import NelderMeadConfig
from declivity.algorithms.neldermead_hc.config import NelderMeadHCConfig
from declivity.algorithms.powell.config import PowellConfig
from declivity.benchmarking import (
    CMAESPath,
    Problem,
    ProblemFamily,
    RunTrace,
    load_cmaes_path,
    record_cmaes_path,
    save_cmaes_path,
)
from declivity.cec.problem import CEC_LOWER_BOUND, CEC_UPPER_BOUND, CECProblem
from declivity.core.config_base import BaseConfig
from declivity.utils.benchmark_functions import Ellipsoid, RotatedFunction
from declivity.utils.initial_point_generator import UniformBoxInitialPointGenerator

BOUNDED = "bounded"
UNBOUNDED = "unbounded"
VARIANTS = (BOUNDED, UNBOUNDED)

CEC_OBJECTIVE = "cec"
ELLIPSOID_OBJECTIVE = "ellipsoid"
OBJECTIVES = (CEC_OBJECTIVE, ELLIPSOID_OBJECTIVE)

SAMPLING_LOWER = CEC_LOWER_BOUND
SAMPLING_UPPER = CEC_UPPER_BOUND
SAMPLING_SPAN = SAMPLING_UPPER - SAMPLING_LOWER

# The Nelder-Mead arms form a 2x2 over *how* the conditioner reaches the run,
# which is the only way to attribute a difference to one mechanism:
#
#                       no model step          model step
#   isotropic simplex   neldermead_control     neldermead_hc
#   shaped simplex      neldermead             neldermead_hc_shaped
#
# ``neldermead_control`` receives the conditioner and ignores it, so its curves
# must coincide across every conditioner -- that redundancy is a built-in check
# that the arms really are otherwise identical.
LOCAL_CHOICES: dict[str, AlgorithmChoice] = {
    "lbfgsb": AlgorithmChoice.LBFGSB,
    "bfgs": AlgorithmChoice.BFGS,
    "powell": AlgorithmChoice.POWELL,
    "neldermead": AlgorithmChoice.NELDERMEAD,
    "neldermead_hc": AlgorithmChoice.NELDERMEAD_HC,
    "neldermead_control": AlgorithmChoice.NELDERMEAD_HC,
    "neldermead_hc_shaped": AlgorithmChoice.NELDERMEAD_HC,
}

LOCAL_VARIANTS: dict[str, dict[str, Any]] = {
    "neldermead_control": {"model_step": False, "shape_initial_simplex": False},
    "neldermead_hc": {"model_step": True, "shape_initial_simplex": False},
    "neldermead_hc_shaped": {"model_step": True, "shape_initial_simplex": True},
}
"""Config overrides that distinguish arms sharing one ``AlgorithmChoice``.

``neldermead_control`` with ``model_step=False`` is bit-identical to the
``neldermead`` arm run with an identity conditioner, which is what makes it a
legitimate baseline rather than a second implementation of one."""
LOCAL_LABELS: dict[str, str] = {
    "lbfgsb": "L-BFGS-B",
    "bfgs": "BFGS",
    "powell": "Powell",
    "neldermead": "Nelder-Mead",
    "neldermead_hc": "Nelder-Mead HC",
    "neldermead_control": "Nelder-Mead (control)",
    "neldermead_hc_shaped": "Nelder-Mead HC + simplex",
}

CMAES_COLOR = "#e5484d"
LOCAL_ALONE_COLOR = "#9aa0a6"
# Reference conditioners sit outside the viridis ramp so they read as
# baselines rather than as another step of the k sweep.
IDENTITY_COLOR = "#f5a623"
HESSIAN_COLOR = "#ffffff"

EDITIONS: dict[str, CECEdition] = {
    "cec2013": CECEdition.CEC2013,
    "cec2014": CECEdition.CEC2014,
    "cec2017": CECEdition.CEC2017,
}


def apply_dark_style() -> None:
    plt.style.use("dark_background")
    plt.rcParams.update(
        {
            "figure.facecolor": "#111111",
            "axes.facecolor": "#111111",
            "savefig.facecolor": "#111111",
            "grid.alpha": 0.25,
            "legend.framealpha": 0.35,
        }
    )


def ramp_colors(count: int) -> list[str]:
    """Distinct qualitative series colors for the conditioner sweep.

    A perceptually uniform ramp (e.g. viridis) orders the ``k`` steps but
    leaves adjacent curves nearly the same hue, so overlaid conditioners blur
    together. This mirrors the reference manuscript's ``tab20`` line-colour
    cycle instead: saturated members first, pale variants after, so a small
    sweep draws from maximally distinct hues before any tone repeats -- and
    every hue stays legible on the dark surface."""
    cmap = get_cmap("tab20")
    palette = [to_hex(cmap(i)) for i in range(cmap.N)]
    ordered = palette[0::2] + palette[1::2]
    return [ordered[i % len(ordered)] for i in range(count)]


def dump_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def atomic_save_npy(path: Path, array: NDArray[np.float64]) -> None:
    """``np.save`` guarded against concurrent-writer torn files.

    Writes to a sibling temp file and ``os.replace``s it into place, which is
    atomic on POSIX: a concurrent reader sees either the old (absent) state
    or the complete new file, never a partial one. Needed because
    variant-independent artifacts (setup, hessian) are written by every
    variant's SLURM array task for the same dimension.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_stem = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    # np.save appends ".npy" since tmp_stem's name doesn't already end in it.
    np.save(tmp_stem, array)
    os.replace(tmp_stem.with_name(tmp_stem.name + ".npy"), path)


def atomic_dump_yaml(path: Path, payload: dict[str, Any]) -> None:
    """``dump_yaml`` guarded against concurrent-writer torn files (see
    :func:`atomic_save_npy`)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp_path.write_text(yaml.safe_dump(payload, sort_keys=False))
    os.replace(tmp_path, path)


# Setup store: per (dim, seed) starting point + rotation matrix.


def rotation_matrix(dim: int, seed: int) -> NDArray[np.float64]:
    rng = np.random.default_rng([911, dim, seed])
    q, r = np.linalg.qr(rng.standard_normal((dim, dim)))
    return q * np.sign(np.diag(r))


def starting_point(dim: int, seed: int) -> NDArray[np.float64]:
    """Must match ``Problem.starting_point(seed)`` under the suite's generator."""
    rng = np.random.default_rng(seed)
    return rng.uniform(SAMPLING_LOWER, SAMPLING_UPPER, size=dim)


def setup_dir(setup_root: Path, dim: int, seed: int) -> Path:
    return setup_root / f"d{dim:03d}" / f"seed{seed:02d}"


def ensure_setup(
    setup_root: Path,
    dim: int,
    seed: int,
    rotate: bool,
    write: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.float64] | None]:
    directory = setup_dir(setup_root, dim, seed)
    x0_path = directory / "x0.npy"
    rotation_path = directory / "rotation.npy"

    x0 = np.load(x0_path) if x0_path.exists() else starting_point(dim, seed)
    rotation = None
    if rotate:
        rotation = (
            np.load(rotation_path)
            if rotation_path.exists()
            else rotation_matrix(dim, seed)
        )

    if write and not (directory / "meta.yaml").exists():
        directory.mkdir(parents=True, exist_ok=True)
        # Multiple (dim, variant) array tasks share this dim/seed cell and may
        # race to populate it concurrently; write-then-rename is atomic on
        # POSIX so a concurrent reader never observes a torn file, and the
        # "meta.yaml exists" guard above keeps re-entrant callers cheap.
        atomic_save_npy(x0_path, x0)
        if rotation is not None:
            atomic_save_npy(rotation_path, rotation)
        atomic_dump_yaml(
            directory / "meta.yaml",
            {
                "dimensions": dim,
                "seed": seed,
                "x0_file": "x0.npy",
                "x0_generator": (
                    f"default_rng(seed).uniform({SAMPLING_LOWER:g}, "
                    f"{SAMPLING_UPPER:g}, d)"
                ),
                "rotation_file": "rotation.npy" if rotation is not None else None,
                "rotation_generator": (
                    "QR of default_rng([911, dim, seed]).standard_normal((d, d))"
                    if rotation is not None
                    else None
                ),
            },
        )
    return x0, rotation


# Problems.


def build_family(
    edition: CECEdition | None,
    function_number: int | None,
    dim: int,
    variant: str,
    rotate: bool,
    setup_root: Path,
    objective: str = CEC_OBJECTIVE,
) -> ProblemFamily:
    """Build a ``ProblemFamily`` for either the CEC suite or the Ellipsoid
    objective, sharing the same rotation/bounds/sampling plumbing.

    ``edition``/``function_number`` select the CEC problem and are only
    meaningful when ``objective == "cec"``; passing them alongside
    ``objective == "ellipsoid"`` is rejected rather than silently ignored,
    since a caller supplying them almost certainly expects them to matter.
    """
    if objective == ELLIPSOID_OBJECTIVE:
        if edition is not None or function_number is not None:
            raise ValueError(
                "edition/function_number are not applicable when "
                "objective='ellipsoid'; pass edition=None, function_number=None."
            )
        name = f"Ellipsoid-d{dim}"

        def make_base() -> CECProblem | Ellipsoid:
            return Ellipsoid(dim, lower=SAMPLING_LOWER, upper=SAMPLING_UPPER)

    elif objective == CEC_OBJECTIVE:
        if edition is None or function_number is None:
            raise ValueError(
                "edition/function_number are required when objective='cec'."
            )
        name = f"{edition.name}-F{function_number}"

        def make_base() -> CECProblem | Ellipsoid:
            return CECProblem(edition, function_number, dim)

    else:
        raise ValueError(
            f"Unknown objective {objective!r}; expected one of {OBJECTIVES}."
        )

    def factory(seed: int) -> Problem:
        function = make_base()
        objective_fn = function
        if rotate:
            _, rotation = ensure_setup(setup_root, dim, seed, rotate=True, write=False)
            assert rotation is not None
            objective_fn = RotatedFunction(
                function, rotation=rotation, name_suffix=f"rot-seed{seed}"
            )
        bounded = variant == BOUNDED
        return Problem(
            name=name,
            function=objective_fn,
            dimensions=dim,
            lower_bound=SAMPLING_LOWER if bounded else -np.inf,
            upper_bound=SAMPLING_UPPER if bounded else np.inf,
            gradient=None,
            initial_point_generator=UniformBoxInitialPointGenerator(
                SAMPLING_LOWER, SAMPLING_UPPER
            ),
        )

    return ProblemFamily(name, factory)


# Configs.


def cmaes_config_factory(
    population_size: int, sigma: float
) -> Callable[[int], CMAESConfig]:
    def factory(dim: int) -> CMAESConfig:
        return CMAESConfig(dimensions=dim, population_size=population_size, sigma=sigma)

    return factory


def resolve_population_size(dim: int, factor: float | None) -> int:
    """``factor`` gives lambda = round(factor * d); ``None`` keeps the default."""
    if factor is None:
        return default_population_size(dim)
    return max(4, round(factor * dim))


def local_config_for(optimizer_key: str, dim: int, profile: str = "deep") -> BaseConfig:
    """The config for one *arm*, applying any :data:`LOCAL_VARIANTS` overrides.

    Several arms can share an ``AlgorithmChoice`` and differ only by config -- the
    Nelder-Mead 2x2 above -- so a study must resolve its config from the arm key,
    not from the choice, or every arm in a group collapses onto one setting.
    """
    config = local_config(LOCAL_CHOICES[optimizer_key], dim, profile)
    for field_name, value in LOCAL_VARIANTS.get(optimizer_key, {}).items():
        setattr(config, field_name, value)
    config.validate()
    return config


def local_config(
    choice: AlgorithmChoice, dim: int, profile: str = "deep"
) -> BaseConfig:
    """Local-optimizer configs: ``deep`` grinds until the injected budget or a
    true stall; ``probe`` stops once progress stops being rapid."""
    deep = profile == "deep"
    if choice is AlgorithmChoice.LBFGSB:
        return LBFGSBConfig(
            dimensions=dim,
            factr=0.0 if deep else 10.0,
            pgtol=1e-12 if deep else 1e-10,
        )
    if choice is AlgorithmChoice.BFGS:
        return BFGSConfig(dimensions=dim, gtol=1e-12 if deep else 1e-8)
    if choice is AlgorithmChoice.POWELL:
        return PowellConfig(
            dimensions=dim,
            xtol=1e-10 if deep else 1e-8,
            ftol=1e-12 if deep else 1e-8,
        )
    if choice is AlgorithmChoice.NELDERMEAD:
        return NelderMeadConfig(
            dimensions=dim,
            xatol=1e-10 if deep else 1e-8,
            fatol=1e-12 if deep else 1e-8,
            adaptive=True,
        )
    if choice is AlgorithmChoice.NELDERMEAD_HC:
        # Same tolerances and coefficients as the Nelder-Mead arm above, so the
        # only difference between the two contenders is the model step.  The
        # initial simplex stays isotropic (``shape_initial_simplex=False``):
        # shaping it is precisely the mechanism the NELDERMEAD arm already
        # tests, and mixing both would confound which one is responsible.
        return NelderMeadHCConfig(
            dimensions=dim,
            xatol=1e-10 if deep else 1e-8,
            fatol=1e-12 if deep else 1e-8,
            adaptive=True,
            diag_volume=True,
        )
    raise ValueError(f"Not a local optimizer: {choice!r}")


# CMA-ES path stage.


def cmaes_dir(
    root: Path,
    variant: str,
    dim: int,
    seed: int,
    function_number: int | None = None,
) -> Path:
    directory = root / "cmaes" / variant / f"d{dim:03d}"
    if function_number is not None:
        directory = directory / f"f{function_number:02d}"
    return directory / f"seed{seed:02d}"


def cached_cmaes_path(
    directory: Path, interval: int, max_evaluations: int
) -> CMAESPath | None:
    if not (directory / "meta.json").exists():
        return None
    path_record = load_cmaes_path(directory)
    meta = path_record.meta
    if (
        meta.get("snapshot_interval") == interval
        and meta.get("max_evaluations") == max_evaluations
    ):
        return path_record
    return None


def ensure_cmaes_path(
    directory: Path,
    family: ProblemFamily,
    seed: int,
    config_factory: Callable[[int], CMAESConfig],
    interval: int,
    max_evaluations: int,
    run_allowed: bool,
    force: bool,
    config_payload: dict[str, Any],
) -> CMAESPath:
    if not force:
        cached = cached_cmaes_path(directory, interval, max_evaluations)
        if cached is not None:
            return cached
    if not run_allowed:
        raise FileNotFoundError(
            f"No cached CMA-ES path in {directory} and the CMA-ES stage is "
            f"disabled (--skip-cmaes). Run once without --skip-cmaes."
        )

    problem = family.instance(seed)
    x0 = problem.starting_point(seed)
    path_record = record_cmaes_path(
        problem,
        x0,
        seed,
        config_factory,
        snapshot_interval=interval,
        max_evaluations=max_evaluations,
    )
    save_cmaes_path(directory, path_record)
    dump_yaml(directory / "config.yaml", config_payload)
    return path_record


@dataclass
class CurveSpec:
    """Name + color carrier for plotting composed traces."""

    name: str
    color: str

    def run(self, problem: Problem, x0: NDArray[np.float64], seed: int) -> RunTrace:
        raise NotImplementedError("CurveSpec only labels persisted traces.")


def filter_seed(
    traces: dict[tuple[str, str], list[RunTrace]], seed: int
) -> dict[tuple[str, str], list[RunTrace]]:
    return {
        key: [trace for trace in values if trace.seed == seed]
        for key, values in traces.items()
    }


def gap_traces(
    traces: dict[tuple[str, str], list[RunTrace]],
    optimum: float,
    floor: float,
) -> dict[tuple[str, str], list[RunTrace]]:
    """Traces re-expressed as the gap to the known optimum, floored for log axes."""
    from dataclasses import replace

    return {
        key: [
            replace(
                trace,
                best_fitness=[max(f - optimum, floor) for f in trace.best_fitness],
                final_fitness=max(trace.final_fitness - optimum, floor),
            )
            for trace in values
        ]
        for key, values in traces.items()
    }


def problem_optimum(problem: Problem) -> float:
    minimum = getattr(problem.function, "global_minimum", None)
    return float(minimum[1]) if minimum is not None else 0.0


def anchor_trace(trace: RunTrace, f0: float) -> RunTrace:
    """Prepend the shared starting point (1 evaluation, ``f(x0)``), so every
    contender's curve visibly emanates from the same point."""
    from dataclasses import replace

    if trace.evaluations and trace.evaluations[0] <= 1:
        return trace
    return replace(
        trace,
        evaluations=[1, *trace.evaluations],
        best_fitness=[f0, *trace.best_fitness],
    )


def anchor_traces(
    traces: dict[tuple[str, str], list[RunTrace]],
    f0_by_seed: dict[int, float],
) -> dict[tuple[str, str], list[RunTrace]]:
    return {
        key: [anchor_trace(trace, f0_by_seed[trace.seed]) for trace in values]
        for key, values in traces.items()
    }


def last_improvement_eval(trace: RunTrace) -> int:
    """The evaluation count of the trace's last strict improvement."""
    best = trace.best_fitness
    for index in range(len(best) - 1, 0, -1):
        if best[index] < best[index - 1]:
            return int(trace.evaluations[index])
    return int(trace.evaluations[0]) if trace.evaluations else 1


def plot_xmax(traces: list[RunTrace], headroom: float = 1.08) -> float | None:
    """Where to cut the evaluation axis: just past the slowest contender's last
    improvement, so long flat converged tails do not squash the descent."""
    if not traces:
        return None
    cutoff = max(last_improvement_eval(trace) for trace in traces)
    return float(cutoff) * headroom if cutoff > 1 else None
