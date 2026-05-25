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
- [Visualization](#visualization)
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
├── utils/                 benchmark_functions, boundary_handlers, helpers,
│                          ring_buffer, initial_point_generator, repair_strategy,
│                          covariance
├── logging/               BaseLogger + per-algorithm loggers + LoggerFactory
├── plotting/              MultiAlgorithmPlotter (single-run multi-panel views)
└── benchmarking/          Problem, AlgorithmRun, Benchmark, BenchmarkPlotter,
                           RunTrace, persistence (multi-seed comparison framework)
```

Three high-level concepts:

1. **Factory** — `AlgorithmFactory.create_optimizer(algorithm, func, x0, config, ...)`
   returns a typed `BaseOptimizer[LogData, Config]` instance. Wraps construction
   so callers don't need to import per-algorithm classes.
2. **Generic base** — `BaseOptimizer[LogDataType, ConfigType]` enforces a typed
   `optimize() -> OptimizationResult[LogDataType]` contract per algorithm.
3. **Diagnostics are opt-in** — each config has `diag_*` flags. `diag_bestVal`
   is on by default; everything else costs memory or time so you enable it
   explicitly. `config.enable_all_diagnostics()` flips them all on.

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
print(f"message = {result.message}")
```

The result also carries `result.diagnostic` (the per-algorithm log dataclass)
and `result.algorithm` (`AlgorithmChoice`).

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

Diagnostics: `diag_Ft` logs the per-iteration Ft trajectory.

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
config.tolx = 1e-12 * config.sigma
config.tolxup = 1e4
config.tolconditioncov = 1e14
```

Public hooks for handoff scenarios:

- `optimizer.get_learned_covariance()` → `CovarianceMatrix`
- `optimizer.get_eigendecomposition()` → `(B, D)` where `C = B @ diag(D²) @ Bᵀ`
- `optimizer.sigma`, `optimizer.mean` — properties on the optimizer

Diagnostics: `diag_sigma`, `diag_cond`, `diag_eigen`, `diag_covariance_matrix`.

### MF-CMA-ES

Arabas's matrix-free CMA-ES. Avoids storing the full covariance by holding
circular buffers of past steps. Optional **PPMF** (Population-based
Precision Modification Framework) step-size adaptation can be enabled via
`config.use_ppmf = True`.

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

Diagnostics: `diag_gradient_norm`, `diag_step_length`, `diag_theta`,
`diag_num_free`, `diag_line_search_iters`. See
[`docs/lbfgsb_lecture.md`](docs/lbfgsb_lecture.md) and
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

    # Diagnostic flags (off by default except diag_bestVal)
    diag_enabled: bool = False
    diag_value: bool = False
    diag_mean: bool = False
    diag_meanCords: bool = False
    diag_pop: bool = False
    diag_bestVal: bool = True
    diag_worstVal: bool = False
    diag_eigen: bool = False
```

Helper methods on every config:

- `config.enable_all_diagnostics()` — turns every `diag_*` flag on
- `config.disable_all_diagnostics()` — all off
- `config.with_convergence_diagnostics()` — minimal convergence-only logging
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

## Boundary handling

```python
from src.utils.boundary_handlers import BoundaryHandlerType

optimizer = AlgorithmFactory.create_optimizer(
    algorithm=AlgorithmChoice.CMAES,
    ...,
    boundary_strategy=BoundaryHandlerType.CLAMP,        # default
    # or BoundaryHandlerType.BOUNCE_BACK
)
```

You can also pass `boundary_handler=<your BoundaryHandler instance>` for
custom logic — see `src/utils/boundary_handlers.py`.

## Logging and diagnostics

```python
result = optimizer.optimize()
log = result.diagnostic           # the per-algorithm log dataclass
log.best_fitness                  # list[float] — per-iteration best fitness
log.evaluations                   # list[int]   — eval counter per iteration
log.iteration                     # list[int]   — iteration index
```

Algorithm-specific fields (only populated when the matching `diag_*` flag
is on):

- **DES**: `Ft` history (`diag_Ft`)
- **CMA-ES / MF-CMA-ES**: `sigma`, `condition_number`, `eigenvalues`,
  `mean_vector`, `pc`, `ps`, `covariance_matrix`
- **L-BFGS-B**: `gradient_norm`, `projected_gradient_norm`, `step_length`,
  `theta`, `num_free_vars`, `num_corrections`, `line_search_iters`

## Visualization

Two complementary plotters:

1. **`src.plotting.MultiAlgorithmPlotter`** — single-run multi-panel
   diagnostic views. Use after an individual `optimizer.optimize()` call.

   ```python
   from src.plotting.multi_algorithm_plotter import MultiAlgorithmPlotter

   plotter = MultiAlgorithmPlotter()
   plotter.plot_algorithm_specific_metrics(
       result, AlgorithmChoice.CMAES, save_path="cmaes_metrics.png",
   )
   plotter.plot_labeled_convergence_comparison(
       {"CMA-ES": result_cmaes, "L-BFGS-B": result_lbfgs},
       colors={"CMA-ES": "#e74c3c", "L-BFGS-B": "#3498db"},
       title="...",
       save_path="convergence.png",
       handoff_eval=2500,    # optional: dashed vertical at handoff point
   )
   plotter.plot_function_landscape(func, ...)
   plotter.plot_function_landscape_grid(funcs_dict, ...)
   plotter.plot_matrix_diagonal_comparison(matrices, reference=...)
   plotter.plot_evaluation_bar_chart(results, colors, ...)
   ```

2. **`src.benchmarking.BenchmarkPlotter`** — multi-seed statistical views
   (median + IQR convergence, final-fitness boxplot). Driven by the
   benchmarking framework; consumes `RunTrace`s, not raw `OptimizationResult`s.

The two plotters are deliberately separate today; unifying them is a
deferred refactor.

## Benchmarking framework

`src.benchmarking` is the orchestrator for fair multi-seed comparisons.

```python
from src.benchmarking import (
    Benchmark, BenchmarkPlotter, Problem,
    SingleAlgorithm, CMAESLBFGSBHandoff,
)
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
        transform="inverse",   # see [Handoff](#cma-es--l-bfgs-b-handoff)
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

plotter = BenchmarkPlotter(
    problems=[problem], algorithms=algorithms, traces=bench.traces,
    output_dir="plots/handoff/rastrigin_demo",
)
plotter.plot_convergence_grid(save_path="...")
plotter.plot_final_fitness_boxplot(save_path="...")
```

Key invariants:

- **Same seed ⇒ same starting point across all algorithms.** `Problem`
  generates `x0` deterministically from the seed.
- **Same seed for `CMAESLBFGSBHandoff` ⇒ same CMA-ES RNG as standalone
  CMA-ES**, so the warmup prefix of the handoff exactly matches a
  standalone run up to the handoff point.
- **Persistence**: every `Benchmark.run()` auto-dumps `traces.json`,
  `runs.csv`, `summary.csv` (toggleable with `save_artifacts=False`).
  Restore with `src.benchmarking.persistence.load_traces_json(...)`.

## CMA-ES → L-BFGS-B handoff

`CMAESLBFGSBHandoff` runs CMA-ES, then hands its learned covariance to
L-BFGS-B as the initial Hessian `B₀`. Available transforms:

| `transform=`     | `B₀` becomes        | Use when                                 |
|------------------|---------------------|------------------------------------------|
| `"identity"`     | `I` (no covariance) | Isolating "warmup x0" from "warmup covariance" |
| `"inverse"`      | `C⁻¹`               | Direction-corrected handoff (default)    |
| `"sigma_inverse"`| `(σ²·C)⁻¹`          | Direction + scale, partially cancels σ collapse |

Note: L-BFGS-B (bounded) maintains `B` (the Hessian), not `B⁻¹`. CMA-ES's
covariance `C` is proportional to `B⁻¹`. Passing `C` directly is *wrong*
(it tells L-BFGS-B that steep directions are flat). Empirical findings
under `docs/covariance_handoff_when_it_matters.md`.

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
        boundary_handler: BoundaryHandler | None = None,
        boundary_strategy: BoundaryHandlerType | None = None,
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

### Enumerations

```python
class AlgorithmChoice(Enum):
    Unknown = "Unknown"
    DES = "DES"
    CMAES = "CMAES"
    MFCMAES = "MFCMAES"
    LBFGSB = "LBFGSB"


class BoundaryHandlerType(Enum):
    BOUNCE_BACK = "bounce_back"
    CLAMP = "clamp"


class LineSearchMethod(Enum):
    MORE_THUENTE = "more_thuente"
    ARMIJO = "armijo"


class InitialHessianMode(Enum):
    DIAGONAL = "diagonal"
    DENSE = "dense"


class InitialPointGeneratorType(Enum):
    UNIFORM = "uniform"
    NORMAL = "normal"
```

## License

MIT — see [LICENSE](LICENSE).

## References

1. Hansen & Ostermeier (2001). *Completely Derandomized Self-Adaptation in Evolution Strategies*. Evol. Comput. 9(2).
2. Loshchilov (2014). *A Computationally Efficient Limited Memory CMA-ES for Large Scale Optimization*. GECCO.
3. Morales & Nocedal (2011). *Remark on Algorithm 778: L-BFGS-B (v3.0)*. ACM TOMS 38(1).
4. Arabas, J. — Matrix-Free CMA-ES (publication forthcoming; reference R implementation in `reference/mf_cmaes.r`).
5. CEC2017 benchmark suite — Congress on Evolutionary Computation.
