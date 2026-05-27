# declivity — Documentation

A unified benchmarking framework with modern Python implementations of
DES, CMA-ES, MF-CMA-ES, and L-BFGS-B.

## Table of contents

- [Architecture](#architecture)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Algorithms](#algorithms)
  - [DES](#des)
  - [CMA-ES](#cma-es)
  - [MF-CMA-ES](#mf-cma-es)
  - [L-BFGS-B](#l-bfgs-b)
- [Configuration](#configuration)
- [Benchmark functions](#benchmark-functions)
- [Boundary handling](#boundary-handling)
- [Logging and diagnostics](#logging-and-diagnostics)
- [Plotting (declarative panel system)](#plotting-declarative-panel-system)
- [Benchmarking framework](#benchmarking-framework)
- [CMA-ES → L-BFGS-B handoff](#cma-es--l-bfgs-b-handoff)
- [API reference](#api-reference)

## Architecture

```
src/
├── core/                  BaseOptimizer (ABC, generic), BaseConfig, AlgorithmFactory
├── algorithms/
│   ├── choices.py         AlgorithmChoice enum (DES, CMAES, MFCMAES, LBFGSB)
│   ├── des/               DESOptimizer + DESConfig
│   ├── cmaes/             CMAESOptimizer (+ reference) + CMAESConfig
│   ├── mfcmaes/           MFCMAESOptimizer + MFCMAESConfig
│   └── lbfgsb/            LBFGSBOptimizer + LBFGSBConfig + InitialHessian + line_search
├── utils/                 benchmark_functions, constraint_handlers, helpers,
│                          ring_buffer, initial_point_generator,
│                          population_initializers, repair_strategies,
│                          covariance
├── logging/               BaseLogger + per-algorithm loggers + LoggerFactory
│                          BaseLogData (minimal) + PopulationLogData (extends)
├── plotting/              Declarative panel system:
│                            panel.py            — Panel, Series, PanelRegistry
│                            types.py            — PanelKey, YScale, XAxis,
│                                                  LineStyle, PanelSet (StrEnum)
│                            declarative.py      — plot_metrics, plot_comparison,
│                                                  plot_evaluation_bars
│                            benchmark.py        — plot_benchmark_convergence,
│                                                  plot_benchmark_boxplot
│                            landscape.py        — plot_function_landscape(_grid)
│                            diagnostics.py      — plot_matrix_diagonal_comparison
│                            standard_panels.py  — 30+ panel registrations
└── benchmarking/          Problem, Benchmark, RunTrace, persistence
                           BenchmarkAlgorithm (ABC)
                             ├── SingleAlgorithm
                             └── HandoffAlgorithm (ABC)
                                   └── CMAESLBFGSBHandoff
                           AlgorithmRun (Protocol) + HandoffTransform (StrEnum)
```

Five high-level concepts:

1. **Factory** — `AlgorithmFactory.create_optimizer(algorithm, func, x0, config, ...)`
   returns a typed `BaseOptimizer[LogData, Config]` instance. Wraps construction
   so callers don't need to import per-algorithm classes.
2. **Generic base** — `BaseOptimizer[LogDataType, ConfigType]` enforces a typed
   `optimize() -> OptimizationResult[LogDataType]` contract per algorithm.
3. **LogData hierarchy** — `BaseLogData` carries the universal fields every
   algorithm logs (iteration, evaluations, best_fitness, best_solution);
   `PopulationLogData` extends it for evolutionary algorithms with
   worst/mean/std/population/eigenvalues. Single-point methods (L-BFGS-B)
   inherit `BaseLogData` directly.
4. **Declarative plotting** — `Panel` specs registered against algorithms
   describe what to plot; rendering functions consume the registry.
   Adding a new panel is one `Panel(...)` line. Cross-algorithm semantic
   keys (`PanelKey.STEP_SIZE` → `sigma`/`Ft`/`step_length`) make
   `plot_comparison` produce meaningful overlays automatically.
5. **Benchmark extension hierarchy** — `BenchmarkAlgorithm` ABC, with
   `HandoffAlgorithm` as a two-phase specialization and `AlgorithmRun`
   Protocol as the no-inheritance escape hatch.

## Installation

```bash
pdm install         # PDM (recommended; uses pdm.lock)
# or
pip install -e .
```

Requirements: Python 3.12, NumPy 2.x, SciPy 1.15+, Matplotlib 3.10+,
opfunu (CEC2017), seaborn, joblib.

## Quick start

```python
import numpy as np
from src import AlgorithmFactory
from src.algorithms.choices import AlgorithmChoice
from src.plotting import plot_metrics
from src.utils.benchmark_functions import Sphere

func = Sphere(dimensions=10)
x0 = np.random.uniform(-50, 50, 10)

optimizer = AlgorithmFactory.create_optimizer(
    algorithm=AlgorithmChoice.CMAES,
    func=func,
    initial_point=x0,
    lower_bounds=-100,
    upper_bounds=100,
)

result = optimizer.optimize()
print(f"f* = {result.best_fitness:.4e}")
print(f"x* = {result.best_solution}")
print(f"evals = {result.evaluations}")

plot_metrics(result, save_path="cmaes.png")
```

The result carries `result.diagnostic` (the per-algorithm LogData) and
`result.algorithm` (`AlgorithmChoice`).

## Algorithms

| Algorithm | Default pop. size      | Default budget | Key mechanism                                              |
|-----------|------------------------|----------------|------------------------------------------------------------|
| DES       | `4·d`                  | `10000·d`      | Adaptive Ft from evolution-path history (ring buffer)      |
| CMA-ES    | `4 + ⌊3 ln d⌋`         | `10000·d`      | Full covariance, rank-1 + rank-μ updates, eigendecomp     |
| MF-CMA-ES | `4 + ⌊3 ln d⌋`         | `10000·d`      | Matrix-free CMA-ES (circular buffers), optional PPMF      |
| L-BFGS-B  | 1 (single-point)       | `10000·d`      | Compact L-BFGS, generalized Cauchy point, More-Thuente LS |

### DES

Differential Evolution Strategy. Adaptive scaling factor `Ft` updated from a
ring buffer of successful steps.

```python
from src.algorithms.des.config import DESConfig

config = DESConfig(dimensions=10)
config.Ft = 1.0            # initial scaling factor of difference vectors
config.pathLength = 6      # evolution-path history length
config.Lamarckism = False  # if True, repair coordinates inherit into individuals
```

Algorithm-specific diagnostic flag: `diag_Ft` logs the per-iteration Ft trajectory.

### CMA-ES

Hansen's CMA-ES with rank-1 + rank-μ updates and an explicit
eigendecomposition cache. `cmaes_optimizer.py` is the framework adapter;
`cmaes_reference.py` is the underlying port (kept as a verification reference).

```python
from src.algorithms.cmaes.config import CMAESConfig

config = CMAESConfig(dimensions=10)
config.sigma = 0.0               # 0 ⇒ auto-derive from bounds
config.population_size = 0       # 0 ⇒ default 4 + ⌊3 ln d⌋
config.tolfun = 1e-12
config.tolxup = 1e4
config.tolconditioncov = 1e14
```

Public hooks for handoff scenarios:

- `optimizer.get_learned_covariance()` → `CovarianceMatrix`
- `optimizer.get_eigendecomposition()` → `(B, D)` where `C = B @ diag(D²) @ Bᵀ`
- `optimizer.sigma`, `optimizer.mean` — properties on the optimizer

Algorithm-specific diagnostic flags: `diag_sigma`, `diag_covariance_matrix`.
(Eigenvalues are gated by the base `diag_eigen` flag.)

### MF-CMA-ES

Arabas's matrix-free CMA-ES. Avoids storing the full covariance by holding
circular buffers of past steps. Optional **PPMF** (Population-based
Precision Modification Framework) step-size adaptation can be enabled via
`config.use_ppmf = True`.

MFCMAES logs `sigma`, `p_succ`, `midpoint_fitness`, `constraint_violations`,
`pc_norm`, and `mean_vector_norm` unconditionally — no algorithm-specific
diag flags.

### L-BFGS-B

Pure-Python port of L-BFGS-B v3.0 (Morales–Nocedal 2011), integrated as
`AlgorithmChoice.LBFGSB`. Includes the generalized Cauchy point, subspace
minimization via the Woodbury identity, More-Thuente or Armijo line
search, and a projected-Newton safeguard.

```python
from src.algorithms.lbfgsb.config import LBFGSBConfig, LineSearchMethod

config = LBFGSBConfig(
    dimensions=10,
    m=10,                                       # number of correction pairs
    pgtol=1e-8,                                 # projected-gradient tolerance
    factr=1e7,                                  # function-value tolerance factor
    line_search=LineSearchMethod.MORE_THUENTE,  # or ARMIJO
)
```

**Initial Hessian** (`config.initial_hessian`) accepts `None | float | NDArray`:

| Value           | Meaning                              | Mode    | Cost per iteration |
|-----------------|--------------------------------------|---------|--------------------|
| `None`          | Identity (`θ·I`, θ adapts)           | scalar  | O(m·n)             |
| `float c`       | Uniform diagonal (`c·I`)             | scalar  | O(m·n)             |
| 1D `NDArray`    | Arbitrary diagonal                   | diagonal| O(m·n)             |
| 2D `NDArray`    | Full dense Hessian + cached Cholesky | DENSE   | O(m·n²)            |

`persist_initial_hessian=True` (default) locks the relative per-variable
scaling for the entire run (the effective base Hessian is
`θ·diag(initial_hessian)`); `False` lets it decay to `θ·I` after the first
iteration.

Gradient — pass an analytic gradient via the `gradient_fn` keyword on
`AlgorithmFactory.create_optimizer`; otherwise the optimizer falls back to
central or forward finite differences (`config.fd_method`).

Algorithm-specific diagnostic flags: `diag_gradient_norm`, `diag_step_length`,
`diag_theta`, `diag_num_free`, `diag_line_search_iters`.

See [`docs/lbfgsb_lecture.md`](docs/lbfgsb_lecture.md) and
[`docs/lbfgsb_initial_hessian_design.md`](docs/lbfgsb_initial_hessian_design.md)
for internals.

## Configuration

Every algorithm has a `*Config` dataclass extending `BaseConfig`:

```python
from src.core.config_base import BaseConfig

@dataclass
class BaseConfig:
    dimensions: int
    budget: int = 0          # 0 ⇒ algorithm-specific default
    population_size: int = 0 # 0 ⇒ algorithm-specific default

    # Diagnostic flags that actually gate logging behavior.
    diag_pop: bool = False    # store the population each iter (memory-expensive)
    diag_eigen: bool = False  # compute + log eigenvalues / condition number
```

Diagnostic flags are kept lean: only flags that an optimizer or logger
actually gates on appear. Algorithm-specific flags live on subclass configs
(e.g. `diag_sigma` on `CMAESConfig`, `diag_Ft` on `DESConfig`).

Helper methods on every config:

- `config.enable_all_diagnostics()` — turns every `diag_*` flag on (the
  algorithm's own subclass methods chain via `super()`).
- `config.disable_all_diagnostics()` — all off
- `config.to_dict()` — for serialization

Configs can also be built via the factory if you don't want to import the
per-algorithm class:

```python
config = AlgorithmFactory.create_config(
    AlgorithmChoice.LBFGSB, dimensions=10, m=15, pgtol=1e-9,
)
```

## Benchmark functions

Built-in classes live in `src/utils/benchmark_functions.py`:

| Class                | Description                                                  |
|----------------------|--------------------------------------------------------------|
| `Sphere`             | `f(x) = Σ xᵢ²`                                              |
| `Ellipsoid`          | `f(x) = Σ 10^(6i/(d-1)) xᵢ²`, condition 10⁶                |
| `Rosenbrock`         | Standard valley function                                     |
| `Rastrigin`          | Highly multimodal                                            |
| `Ackley`             | Multimodal w/ exponential well                              |
| `Schwefel`           | Multimodal w/ deceptive global optimum                       |
| `Griewank`           | Multimodal w/ product term                                   |
| `RotatedEllipsoid`   | Ellipsoid with Givens / random orthogonal rotation           |
| `RotatedFunction`    | Generic rotation wrapper for any `BenchmarkFunction`         |
| `RippledEllipsoid`   | Anisotropic + multimodal: `Σ sᵢxᵢ² + a·Σ(1 - cos(2πxᵢ))`     |
| `CEC17Function`      | opfunu wrapper for CEC2017 F1–F30                           |

Every benchmark function exposes:

```python
func.bounds              # (lower, upper)  — np.float64 arrays of length d
func.global_minimum      # (x*, f*)
func(x)                  # the objective value
func.gradient(x)         # analytic gradient (where defined)
func.hessian             # property (where defined)
```

Rotated functions live entirely in problem space — they construct a fixed
rotation matrix `R` at init time and evaluate `f(Rᵀ x)`. Four rotation
modes for `RotatedEllipsoid` / `RotatedFunction`:

| `rotation=`   | Description                                              |
|---------------|----------------------------------------------------------|
| `"none"`      | Identity (use `Ellipsoid` directly)                      |
| `"uniform_45"`| Givens chain in adjacent planes at 45°                  |
| `"golden"`    | Givens chain at `(k+1)·137.5°` per plane (aperiodic)    |
| `"random"`    | QR factor of an `N(0,1)` matrix (full random orthogonal)|

## Constraint handling

Constraints are injected as `ConstraintHandler` instances. The default
when none is supplied is `BoxConstraintHandler(BoxStrategy.CLAMP, ...)`
constructed from the optimizer's `lower_bounds` / `upper_bounds`.

```python
from src.utils.constraint_handlers import (
    BoxConstraintHandler,
    BoxStrategy,
    ConstraintHandlerType,
)

# Explicit instance construction:
handler = BoxConstraintHandler(
    BoxStrategy.BOUNCE_BACK, lower_bounds, upper_bounds
)

# Or via the discoverability enum:
handler = ConstraintHandlerType.BOX_BOUNCE_BACK.build(
    lower_bounds, upper_bounds
)

optimizer = AlgorithmFactory.create_optimizer(
    algorithm=AlgorithmChoice.CMAES,
    ...,
    constraint_handler=handler,
)
```

Custom handlers (e.g. inequality constraints, penalty methods) subclass
`ConstraintHandler` directly — implement `is_feasible` and
`feasibility_distance`, then override `repair` and/or `penalty`. See
`src/utils/constraint_handlers.py` for the ABC and
`experiments/basic/constrained_rosenbrock.py` for a worked example.

## Logging and diagnostics

### LogData hierarchy

```
BaseLogData                  — universal fields every algorithm logs
  └── PopulationLogData      — adds worst / mean / std / population /
                                eigenvalues / condition_number
```

`BaseLogData` carries `iteration`, `evaluations`, `best_fitness`, and
`best_solution`. `PopulationLogData` extends it for evolutionary algorithms.
Algorithm-specific LogData subclasses add their own fields on top.

| LogData          | Inherits from         | Algorithm-specific fields                        |
|------------------|-----------------------|--------------------------------------------------|
| `DESLogData`     | `PopulationLogData`   | `Ft`, `evolution_path`, `step_size`              |
| `CMAESLogData`   | `PopulationLogData`   | `sigma`, `pc`, `ps`, `mean_vector`, eigen-stats  |
| `MFCMAESLogData` | `PopulationLogData`   | `sigma`, `p_succ`, `midpoint_fitness`, …         |
| `LBFGSBLogData`  | `BaseLogData`         | `gradient_norm`, `step_length`, `theta`, …       |

L-BFGS-B is single-point, so `LBFGSBLogData` inherits `BaseLogData`
directly without the population fields.

### Reading logs

```python
result = optimizer.optimize()
log = result.diagnostic           # the per-algorithm LogData
log.best_fitness                  # list[float] — per-iteration best fitness
log.evaluations                   # list[int]   — eval counter per iteration
log.iteration                     # list[int]   — iteration index
log.to_dict()                     # all fields as a dict (for serialization)
```

Algorithm-specific fields are only populated when the matching `diag_*`
flag is on (e.g. `condition_number` requires `diag_eigen=True`).

## Plotting (declarative panel system)

The framework ships a declarative plotter built around `Panel` specs:
**you describe what a panel is**, the registry knows what each algorithm
exposes, and the rendering functions read both.

### Eight entry points

All in `src.plotting`:

| Function | Use when |
|---|---|
| `plot_metrics(result)` | one optimization run, multi-panel deep dive |
| `plot_comparison(results)` | several runs, semantic-key overlay per panel |
| `plot_evaluation_bars(results)` | horizontal bar chart of total evaluations |
| `plot_benchmark_convergence(traces, problems, algorithms)` | multi-seed median + IQR per algorithm |
| `plot_benchmark_boxplot(traces, problems, algorithms)` | final-fitness distribution per algorithm |
| `plot_function_landscape(func)` | 2D contour of an objective with optional Hessian arrows |
| `plot_function_landscape_grid(funcs)` | several landscapes side-by-side |
| `plot_matrix_diagonal_comparison(matrices, reference)` | sorted-diagonal vs reference matrix |

### Single-run deep dive

```python
from src.plotting import plot_metrics, PanelKey, PanelSet

# Default: every panel marked `default=True` for the algorithm.
plot_metrics(result, save_path="cmaes.png")

# Curated subset.
plot_metrics(
    result,
    panels=[PanelKey.CONVERGENCE, PanelKey.STEP_SIZE, PanelKey.CONDITION_NUMBER],
    save_path="cmaes_focused.png",
)

# Everything registered, including non-default panels.
plot_metrics(result, panels=PanelSet.ALL, save_path="cmaes_all.png")
```

### Cross-algorithm comparison

```python
from src.plotting import plot_comparison

plot_comparison(
    {"CMA-ES": cmaes_result, "L-BFGS-B": lbfgsb_result},
    colors={"CMA-ES": "#e74c3c", "L-BFGS-B": "#3498db"},
    save_path="comparison.png",
)
# panels=None defaults to PanelRegistry.common([algos]) — the intersection
# of registered semantic keys. CMA-ES `sigma` and L-BFGS-B `step_length`
# both register under PanelKey.STEP_SIZE, so they overlay on one axis.
```

For an L-BFGS-B family comparison with handoff annotation:

```python
plot_comparison(
    results,
    panels=[
        PanelKey.CONVERGENCE,
        PanelKey.CONVERGENCE_BY_ITER,
        PanelKey.PROJECTED_GRADIENT,
        PanelKey.STEP_SIZE_BY_ITER,
    ],
    handoff_eval=2500,    # vertical dashed line on evaluations panels
    handoff_iter=42,      # vertical dashed line on iteration panels
)
```

### Multi-seed benchmark

```python
from src.plotting import plot_benchmark_convergence, plot_benchmark_boxplot

plot_benchmark_convergence(
    bench.traces, problems=problems, algorithms=algorithms,
    show_iqr=True,                # median + 25/75 percentile band
    annotate_handoff=True,        # auto vertical lines from RunTrace.handoff_eval
    save_path="convergence.png",
)
plot_benchmark_boxplot(
    bench.traces, problems=problems, algorithms=algorithms,
    save_path="final_fitness.png",
)
```

### Adding a new panel

One line in [`src/plotting/standard_panels.py`](src/plotting/standard_panels.py):

```python
from src.plotting import Panel, PanelKey, PanelRegistry, YScale

PanelRegistry.register(
    AlgorithmChoice.CMAES,
    Panel(
        key=PanelKey.STEP_SIZE,    # or any string for custom keys
        title="Step Size",
        ylabel="σ",
        field="sigma",             # attribute on the LogData
        yscale=YScale.LOG,
        floor=1e-30,               # clip below this for log scale
        default=True,              # included in plot_metrics(panels=None)
    ),
)
```

Multi-series panels (one panel, several lines on the same axes — like
the classic best/mean/median view) use `series=` instead of `field=`:

```python
from src.plotting import Series, LineStyle

Panel(
    key=PanelKey.CONVERGENCE,
    title="Convergence",
    ylabel="Fitness (log)",
    series=(
        Series("best_fitness",   "Best",      color="tab:blue"),
        Series("mean_fitness",   "Mean f(m)", linestyle=LineStyle.DASHED),
        Series("median_fitness", "Median",    linestyle=LineStyle.DOTTED),
    ),
    yscale=YScale.LOG,
    floor=1e-30,
)
```

### StrEnum vocabulary

The user-facing API uses `StrEnum` types so call sites get autocomplete
and refactor safety. The enums compare equal to their string values —
existing call sites passing raw strings keep working.

| Enum | Values |
|---|---|
| `PanelKey` | All standard semantic keys (CONVERGENCE, STEP_SIZE, CONDITION_NUMBER, …) |
| `YScale` | LOG, LINEAR |
| `XAxis` | EVALUATIONS, ITERATION |
| `LineStyle` | SOLID, DASHED, DOTTED, DASH_DOT |
| `PanelSet` | DEFAULT, ALL (sentinels for `panels=`) |

```python
# These are equivalent:
plot_metrics(result, panels=["convergence", "step_size"])
plot_metrics(result, panels=[PanelKey.CONVERGENCE, PanelKey.STEP_SIZE])
```

### Specialized plots

Outside the panel system (they don't fit the per-iteration time-series model):

```python
from src.plotting import (
    plot_function_landscape, plot_function_landscape_grid,
    plot_matrix_diagonal_comparison,
)

# 2D contour with Hessian eigenvector arrows
plot_function_landscape(
    func, title="Rotated Ellipsoid",
    extent=10.0, resolution=200,
    show_eigenvectors=True, hessian=func.hessian,
    save_path="landscape.png",
)

# Several functions side-by-side
plot_function_landscape_grid(
    {"None": ellipsoid, "Random rotation": rotated_ellipsoid},
    extent=10.0, save_path="landscapes.png",
)

# Diagonal-profile comparison against a reference matrix
plot_matrix_diagonal_comparison(
    {"C^-1": inverse_cov, "Normalized": normalized},
    reference=true_hessian,
    save_path="diag_compare.png",
)
```

## Benchmarking framework

`src.benchmarking` is the orchestrator for fair multi-seed comparisons.

### Extension hierarchy

Anything that runs inside a `Benchmark` provides three things: `name`,
`color`, and `run(problem, x0, seed) -> RunTrace`. Pick the narrowest
base class that fits:

| Your runner is... | Inherit from |
|---|---|
| one optimizer from the factory, no special wrapping | `SingleAlgorithm` (concrete) |
| warmup → refinement, two phases sharing state | `HandoffAlgorithm` (ABC, implement `run_phases()`) |
| anything else (restarts, multi-phase, wrappers) | `BenchmarkAlgorithm` (ABC, implement `run()`) |
| already a class, can't inherit | conform to `AlgorithmRun` Protocol |

`HandoffAlgorithm` handles trace-stitching (eval-count offsets, fitness
clamping, handoff metadata) so subclasses only describe the two phases.
`BenchmarkAlgorithm` provides `trace_from_result()` for the common
single-result → `RunTrace` packaging.

### Standard usage

```python
from src.benchmarking import (
    Benchmark, Problem,
    SingleAlgorithm, CMAESLBFGSBHandoff, HandoffTransform,
)
from src.plotting import plot_benchmark_convergence, plot_benchmark_boxplot
from src.utils.benchmark_functions import Rastrigin
from src.algorithms.choices import AlgorithmChoice
from src.algorithms.cmaes.config import CMAESConfig
from src.algorithms.lbfgsb.config import LBFGSBConfig

problem = Problem.from_benchmark("Rastrigin-10D", Rastrigin(10))

algorithms = [
    SingleAlgorithm(
        name="CMA-ES", color="#e74c3c",
        algorithm=AlgorithmChoice.CMAES,
        config_factory=lambda d: CMAESConfig(dimensions=d, budget=10000, sigma=2.0),
    ),
    SingleAlgorithm(
        name="L-BFGS-B", color="#3498db",
        algorithm=AlgorithmChoice.LBFGSB,
        config_factory=lambda d: LBFGSBConfig(dimensions=d, budget=10000),
    ),
    CMAESLBFGSBHandoff(
        name="CMA-ES + L-BFGS-B", color="#2ecc71",
        cmaes_config_factory=lambda d: CMAESConfig(dimensions=d, budget=2500, sigma=2.0),
        lbfgsb_config_factory=lambda d: LBFGSBConfig(dimensions=d, budget=7500),
        transform=HandoffTransform.INVERSE,    # or "inverse" (StrEnum)
    ),
]

bench = Benchmark(
    problems=[problem],
    algorithms=algorithms,
    seeds=list(range(25)),
    output_dir="plots/handoff/rastrigin_demo",
    num_workers=4,             # >1 ⇒ joblib loky parallel execution
)
bench.run(verbose=True)
bench.print_summary()

plot_benchmark_convergence(
    bench.traces, problems=[problem], algorithms=algorithms,
    save_path="plots/handoff/rastrigin_demo/convergence.png",
)
plot_benchmark_boxplot(
    bench.traces, problems=[problem], algorithms=algorithms,
    save_path="plots/handoff/rastrigin_demo/final_fitness.png",
)
```

### Custom runners

Two worked examples in `experiments/basic/`:

**`custom_handoff.py`** — DES → L-BFGS-B via `HandoffAlgorithm`:

```python
@dataclass
class DESLBFGSBHandoff(HandoffAlgorithm):
    name: str
    color: str
    des_config_factory:    Callable[[int], DESConfig]
    lbfgsb_config_factory: Callable[[int], LBFGSBConfig]

    def run_phases(self, problem, x0, seed):
        des_result = AlgorithmFactory.create_optimizer(
            AlgorithmChoice.DES, problem.function, x0,
            self.des_config_factory(problem.dimensions),
            lower_bounds=problem.lower_bound, upper_bounds=problem.upper_bound,
            seed=seed,
        ).optimize()

        lbfgsb_result = AlgorithmFactory.create_optimizer(
            AlgorithmChoice.LBFGSB, problem.function, des_result.best_solution,
            self.lbfgsb_config_factory(problem.dimensions),
            lower_bounds=problem.lower_bound, upper_bounds=problem.upper_bound,
        ).optimize()

        return des_result, lbfgsb_result
```

**`custom_algorithm.py`** — multi-start CMA-ES via `BenchmarkAlgorithm`:
implements `run()` and stitches a custom multi-restart `RunTrace` by hand.

### Invariants

- **Same seed ⇒ same starting point across all algorithms.** `Problem`
  generates `x0` deterministically from the seed.
- **Same seed for `CMAESLBFGSBHandoff` ⇒ same CMA-ES RNG as standalone
  CMA-ES**, so the warmup prefix of the handoff exactly matches a
  standalone run up to the handoff point.
- **Persistence**: every `Benchmark.run()` auto-dumps `traces.json`,
  `runs.csv`, `summary.csv` (toggle with `save_artifacts=False`).
  Restore with `src.benchmarking.persistence.load_traces_json(...)`.

## CMA-ES → L-BFGS-B handoff

`CMAESLBFGSBHandoff` runs CMA-ES, then hands its learned covariance to
L-BFGS-B as the initial Hessian `B₀`. The `transform` parameter accepts
a `HandoffTransform` StrEnum or the equivalent raw string:

| `HandoffTransform.` | `B₀` becomes        | Use when                                       |
|---------------------|---------------------|------------------------------------------------|
| `IDENTITY`          | `I` (no covariance) | Isolating "warmup x0" from "warmup covariance" |
| `INVERSE`           | `C⁻¹`               | Direction-corrected handoff (default)          |
| `SIGMA_INVERSE`     | `(σ²·C)⁻¹`          | Direction + scale, partially cancels σ collapse |

Note: L-BFGS-B (bounded) maintains `B` (the Hessian), not `B⁻¹`. CMA-ES's
covariance `C` is proportional to `B⁻¹`. Passing `C` directly is *wrong*
(it tells L-BFGS-B that steep directions are flat). Empirical findings
under [`docs/covariance_handoff_when_it_matters.md`](docs/covariance_handoff_when_it_matters.md).

## API reference

### `AlgorithmFactory`

```python
class AlgorithmFactory:
    @classmethod
    def create_optimizer(
        cls,
        algorithm: AlgorithmChoice,
        func: Callable[[NDArray[np.float64]], float],
        initial_point: NDArray[np.float64],
        config: BaseConfig | None = None,
        constraint_handler: ConstraintHandler | None = None,
        lower_bounds: float | NDArray[np.float64] | list[float] = -100.0,
        upper_bounds: float | NDArray[np.float64] | list[float] = 100.0,
        seed: int | None = None,           # CMA-ES / handoff repeatability
        gradient_fn: Callable | None = None, # L-BFGS-B analytic gradient
        **kwargs,
    ) -> BaseOptimizer

    @classmethod
    def create_config(
        cls, algorithm: AlgorithmChoice, dimensions: int, **kwargs,
    ) -> BaseConfig

    @classmethod
    def get_available_algorithms(cls) -> list[AlgorithmChoice]

    @classmethod
    def register_algorithm(
        cls, name: AlgorithmChoice,
        optimizer_class: type[BaseOptimizer],
        config_class: type[BaseConfig],
    ) -> None
```

### `BaseOptimizer`

```python
class BaseOptimizer(ABC, Generic[LogDataType, ConfigType]):
    def evaluate(self, x: NDArray[np.float64]) -> float
    def evaluate_population(self, pop: NDArray[np.float64]) -> NDArray[np.float64]
    def get_logs(self) -> LogDataType
    @abstractmethod
    def optimize(self) -> OptimizationResult[LogDataType]
```

### `OptimizationResult`

```python
@dataclass
class OptimizationResult(Generic[LogDataType]):
    best_solution: NDArray[np.float64]
    best_fitness: float
    evaluations: int
    message: str
    diagnostic: LogDataType
    algorithm: AlgorithmChoice
```

### `BenchmarkAlgorithm`

```python
class BenchmarkAlgorithm(ABC):
    name: str
    color: str

    @abstractmethod
    def run(self, problem: Problem, x0: NDArray[np.float64], seed: int) -> RunTrace: ...

    def trace_from_result(
        self, problem: Problem, seed: int, result: OptimizationResult,
    ) -> RunTrace
```

### `HandoffAlgorithm`

```python
class HandoffAlgorithm(BenchmarkAlgorithm):
    @abstractmethod
    def run_phases(
        self, problem: Problem, x0: NDArray[np.float64], seed: int,
    ) -> tuple[OptimizationResult, OptimizationResult]: ...

    # run() and _stitch_traces() are inherited — eval-count offsets,
    # fitness clamping, and handoff metadata are all handled by the base.
```

### `Panel` / `Series` / `PanelRegistry`

```python
@dataclass(frozen=True)
class Series:
    field: str
    label: str = ""
    linestyle: LineStyle | str = LineStyle.SOLID
    color: str | None = None

@dataclass(frozen=True)
class Panel:
    key: PanelKey | str
    title: str
    ylabel: str
    field: str | None = None
    series: tuple[Series, ...] | None = None
    x_field: str = "evaluations"
    yscale: YScale | str = YScale.LOG
    floor: float | None = None
    default: bool = True

class PanelRegistry:
    @classmethod def register(cls, algorithm: AlgorithmChoice, *panels: Panel) -> None
    @classmethod def get(cls, algorithm: AlgorithmChoice, key: PanelKey | str) -> Panel
    @classmethod def available(cls, algorithm: AlgorithmChoice) -> list[str]
    @classmethod def default(cls, algorithm: AlgorithmChoice) -> list[str]
    @classmethod def common(cls, algorithms: Sequence[AlgorithmChoice]) -> list[str]
    @classmethod def all_registered(cls) -> dict[AlgorithmChoice, list[str]]
```

### Enumerations

```python
class AlgorithmChoice(Enum):
    Unknown = "Unknown"
    DES = "DES"
    CMAES = "CMAES"
    MFCMAES = "MFCMAES"
    LBFGSB = "LBFGSB"


class BoxStrategy(Enum):
    CLAMP = "clamp"
    BOUNCE_BACK = "bounce_back"


class ConstraintHandlerType(Enum):
    BOX_CLAMP = "box_clamp"
    BOX_BOUNCE_BACK = "box_bounce_back"


class LineSearchMethod(Enum):
    MORE_THUENTE = "more_thuente"
    ARMIJO = "armijo"


class InitialHessianMode(Enum):
    DIAGONAL = "diagonal"
    DENSE = "dense"


class HandoffTransform(StrEnum):
    INVERSE = "inverse"
    SIGMA_INVERSE = "sigma_inverse"
    IDENTITY = "identity"


# Plotting StrEnums
class PanelKey(StrEnum): ...  # See src/plotting/types.py for all values
class YScale(StrEnum): LINEAR = "linear"; LOG = "log"
class XAxis(StrEnum): EVALUATIONS = "evaluations"; ITERATION = "iteration"
class LineStyle(StrEnum): SOLID = "-"; DASHED = "--"; DOTTED = ":"; DASH_DOT = "-."
class PanelSet(StrEnum): DEFAULT = "default"; ALL = "all"
```

## License

MIT — see [LICENSE](LICENSE).

## References

1. Hansen & Ostermeier (2001). *Completely Derandomized Self-Adaptation in Evolution Strategies*. Evol. Comput. 9(2).
2. Loshchilov (2014). *A Computationally Efficient Limited Memory CMA-ES for Large Scale Optimization*. GECCO.
3. Morales & Nocedal (2011). *Remark on Algorithm 778: L-BFGS-B (v3.0)*. ACM TOMS 38(1).
4. Arabas, J. — Matrix-Free CMA-ES (publication forthcoming; reference R implementation in `reference/mf_cmaes.r`).
5. CEC2017 benchmark suite — Congress on Evolutionary Computation.
