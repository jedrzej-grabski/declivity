# declivity

A unified benchmarking framework and modern Python port for evolutionary and
quasi-Newton optimization algorithms. Thesis project by Jedrzej Grabski.

## What this is

Two things, glued together:

1. A **unified framework** for fairly comparing optimizers under identical
   conditions (shared budgets, shared seeds, shared problem definitions).
2. **Modern Python implementations** of algorithms that previously only
   existed as inconsistent academic ports — CMA-ES variants from R,
   L-BFGS-B from Fortran 77 — rewritten as clean, type-checked Python
   that plugs into the framework.

The current algorithm roster:

| Algorithm  | Family             | Notes                                                   |
|------------|--------------------|---------------------------------------------------------|
| DES        | Evolutionary       | Adaptive Ft via evolution-path history (ring buffer)    |
| CMA-ES     | Evolutionary       | Full covariance matrix, rank-1 + rank-μ updates         |
| MF-CMA-ES  | Evolutionary       | Matrix-free CMA-ES (Arabas), optional PPMF              |
| L-BFGS-B   | Quasi-Newton       | Pure-Python port of Fortran v3.0, configurable B₀       |

L-BFGS-B is the newest addition and ships with the more involved studies
(rotated Ellipsoid, CMA-ES → L-BFGS-B covariance handoff).

## Quick start

```bash
pdm install
pdm run run-example      # Sphere demo (DES on 10D Sphere)
pdm run run-r            # CEC2017 F10, 10 seeds, writes CSVs for R cross-check
```

Or directly:

```bash
PYTHONPATH=. pdm run python -m experiments.basic.simple_optimization
PYTHONPATH=. pdm run python experiments/handoff/multimodal.py --num-seeds 25
```

## Minimal example

```python
import numpy as np
from src import AlgorithmFactory
from src.algorithms.choices import AlgorithmChoice
from src.utils.benchmark_functions import Sphere

func = Sphere(dimensions=10)
initial_point = np.random.uniform(-50, 50, 10)

optimizer = AlgorithmFactory.create_optimizer(
    algorithm=AlgorithmChoice.CMAES,
    func=func,
    initial_point=initial_point,
    lower_bounds=-100,
    upper_bounds=100,
)

result = optimizer.optimize()
print(f"f* = {result.best_fitness:.4e} after {result.evaluations} evals")
```

See [`DOCUMENTATION.md`](DOCUMENTATION.md) for the full API.

## Project layout

```
declivity/
├── src/                          Library code (algorithms, framework)
│   ├── core/                     BaseOptimizer, AlgorithmFactory, BaseConfig
│   ├── algorithms/               DES, CMA-ES, MF-CMA-ES, L-BFGS-B
│   ├── benchmarking/             Problem, AlgorithmRun, Benchmark, BenchmarkPlotter
│   ├── utils/                    Benchmark functions, boundary handlers, helpers
│   ├── logging/                  Per-algorithm diagnostic loggers
│   └── plotting/                 Single-run multi-panel plots
│
├── experiments/                  Runnable studies (one script per study)
│   ├── basic/                    Tutorial demos and sanity checks
│   ├── cross_validation/         Cross-checks against R reference impls
│   ├── lbfgsb/                   L-BFGS-B feature studies
│   ├── handoff/                  CMA-ES → L-BFGS-B handoff studies
│   └── report/                   Plot regeneration utilities
│
├── plots/                        Experiment outputs (gitignored, mirrors experiments/)
├── docs/                         Algorithm lectures + design notes + reports
├── reference/                    Reference R implementations and their outputs
└── notes/                        Local scratch notes (gitignored)
```

## Where to look next

- **API reference**: [`DOCUMENTATION.md`](DOCUMENTATION.md)
- **Experiment index**: [`experiments/README.md`](experiments/README.md)
- **L-BFGS-B internals**: [`docs/lbfgsb_lecture.md`](docs/lbfgsb_lecture.md)
- **Initial-Hessian design**: [`docs/lbfgsb_initial_hessian_design.md`](docs/lbfgsb_initial_hessian_design.md)
- **CMA-ES → L-BFGS-B handoff study**: [`docs/covariance_handoff_when_it_matters.md`](docs/covariance_handoff_when_it_matters.md)
- **CMA-ES diagnostic plot legend**: [`docs/cmaes_diagnostic_plots.md`](docs/cmaes_diagnostic_plots.md)

## Tech stack

- Python 3.12 (strict), managed with PDM
- NumPy 2.x, SciPy 1.15+, Matplotlib 3.10+
- `opfunu` for CEC2017 benchmark functions
- `joblib` for optional parallel benchmark execution

## Status

Thesis-in-progress. There is no formal test suite — correctness is
validated by cross-comparison against the R reference implementations in
[`reference/`](reference/) and by the supervisor-report experiments under
[`plots/report/`](plots/report/).

## License

MIT — see [LICENSE](LICENSE).
