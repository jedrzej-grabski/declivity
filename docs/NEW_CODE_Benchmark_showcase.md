# NEW CODE — Benchmark modularity showcase

The companion to [`NEW_CODE_Plotter_showcase.md`](NEW_CODE_Plotter_showcase.md).
That document showed the *plotter* needs no per-algorithm code. This one
makes the same point one layer down — the **runner harness** doesn't either.

A `Benchmark` is a grid over every `(problem × algorithm × seed)` triple.
The only thing it asks of an "algorithm" is three things: a `name`, a
`color`, and `run(problem, x0, seed) -> RunTrace`. So four *structurally
unrelated* runners — two plain optimizers from different families, a
two-phase handoff, and a multi-start scheme written from scratch — drop into
**one algorithm list** and the grid runs them all identically. Nothing in
`Benchmark`, in persistence, or in the plotter knows the difference.

For the design rationale (why a thin contract, why an extension *hierarchy*
rather than one base class) see
[`framework_design.md`](framework_design.md) §"Premise 2"; the worked
single-runner examples live in
[`experiments/basic/custom_handoff.py`](../experiments/basic/custom_handoff.py)
and
[`experiments/basic/custom_algorithm.py`](../experiments/basic/custom_algorithm.py).

- **Script:** [`experiments/basic/benchmark_showcase.py`](../experiments/basic/benchmark_showcase.py)
- **Run:** `PYTHONPATH=. uv run python experiments/basic/benchmark_showcase.py`
- **Output:** `plots/basic/benchmark_showcase/`

### The four runners — one per rung of the hierarchy

| Runner | Base class | Rung it exercises |
|---|---|---|
| `CMA-ES` | `SingleAlgorithm` (concrete) | one factory optimizer, no wrapping |
| `L-BFGS-B` | `SingleAlgorithm` (concrete) | a *different family* (quasi-Newton), the **same** wrapper |
| `CMA-ES -> L-BFGS-B` | `HandoffAlgorithm` | the pre-built two-phase warm-up → refine |
| `Multi-start CMA-ES` | `BenchmarkAlgorithm` (custom, in-file) | the open-ended point: *N* restarts, defined in the script |

Each row is the **narrowest base class that fits**, exactly the choice
[`framework_design.md`](framework_design.md) asks you to make. The harness is
closed for modification (you never touch `Benchmark`) and open for extension
(you subclass at whichever rung matches your runner's shape).

---

## 1. One grid, four runner types, one convergence plot

```python
from src.benchmarking import (
    Benchmark,
    Problem,
    SingleAlgorithm,
    CMAESLBFGSBHandoff,
)
from src.plotting import plot_benchmark_convergence

algorithms = [
    # Rungs 1 & 2 — concrete SingleAlgorithm, two different families.
    SingleAlgorithm(
        "CMA-ES",
        "#e74c3c",
        AlgorithmChoice.CMAES,
        config_factory=lambda d: CMAESConfig(d, budget=4000),
    ),
    SingleAlgorithm(
        "L-BFGS-B",
        "#3498db",
        AlgorithmChoice.LBFGSB,
        config_factory=lambda d: LBFGSBConfig(d, budget=4000),
    ),
    # Rung 3 — the pre-built two-phase handoff.
    CMAESLBFGSBHandoff(
        "CMA-ES -> L-BFGS-B",
        "#2ecc71",
        cmaes_config_factory=lambda d: CMAESConfig(d, budget=1600),
        lbfgsb_config_factory=lambda d: LBFGSBConfig(d, budget=2400),
        transform="inverse",
    ),
    # Rung 4 — a custom BenchmarkAlgorithm defined in the same file.
    MultiStartCMAES(
        "Multi-start CMA-ES",
        "#9b59b6",
        config_factory=lambda d: CMAESConfig(d, budget=1000),
        num_restarts=4,
    ),
]

bench = Benchmark(
    problems, algorithms, seeds=range(15), output_dir="plots/basic/benchmark_showcase"
)
bench.run()  # 2 x 4 x 15 = 120 runs, then traces.json/csv

plot_benchmark_convergence(  # one call — heterogeneity is invisible here
    bench.traces,
    problems=problems,
    algorithms=algorithms,
    save_path="01_heterogeneous_convergence.png",
)
```

![Multi-seed convergence — median + IQR per runner, per problem](../plots/basic/benchmark_showcase/01_heterogeneous_convergence.png)

[`plots/basic/benchmark_showcase/01_heterogeneous_convergence.png`](../plots/basic/benchmark_showcase/01_heterogeneous_convergence.png)

**Why this is modular.** `MultiStartCMAES` is not a framework class — it is a
compact `BenchmarkAlgorithm` subclass (a `@dataclass`) living in the
experiment script
([definition here](../experiments/basic/benchmark_showcase.py)). It makes
*four* optimizer calls per run, so it fits neither `SingleAlgorithm` (one
call) nor `HandoffAlgorithm` (two phases). Yet it sits in the same list as
the three built-ins, and `Benchmark.run()`, the seed→`x0` machinery,
`traces.json`, the summary table, and `plot_benchmark_convergence` all consume
it with **zero** special-casing — they only ever touch `name`, `color`, and
`run()`. Add a fifth runner and nothing above changes.

**Fair by construction.** `Problem.starting_point(seed)` is deterministic, so
on a given seed *every* runner starts from the *same* `x0` — the only
variable across a column of the grid is the algorithm, never the initial
conditions or the budget. That is the property the unified-benchmarking goal
rests on ([`framework_design.md`](framework_design.md) §"Premise 7"), and it
is enforced in one place ([`problem.py`](../src/benchmarking/problem.py))
rather than per-runner.

**Same figure type, a different point.** The plotter showcase (§2) also drew
a convergence grid — but it was making a *plotting* point with three
pre-built runners. The point here is the *harness*: four runners spanning
every rung of the hierarchy, one of them (`Multi-start CMA-ES`) written from
scratch in the experiment file, and a single `plot_benchmark_convergence`
call renders all four from one `bench.traces` dict. The custom runner's
restart *staircase*, the handoff's vertical marker, and the per-seed IQR
bands are all derived from the traces — there is no runner-aware branch
anywhere in the call.

**Validated.** 15 seeds, two 10-D problems, and the headline result is that
**the ranking inverts between them** — which is precisely why one benchmarks
across problem characters rather than trusting a single number:

| Problem | CMA-ES | L-BFGS-B | CMA-ES → L-BFGS-B | Multi-start |
|---|---:|---:|---:|---:|
| **Rastrigin** (multimodal) | 1.49e+01 | 8.16e+01 | 1.39e+01 | **1.29e+01** |
| **Rosenbrock** (smooth valley) | 1.40e+00 | **5.22e-11** | 1.07e-06 | 7.90e+00 |

*(median final fitness; full table — best/worst/median-evals — is the
`print_summary()` output and the persisted `summary.csv`.)*

- **Rastrigin** rewards global search. L-BFGS-B's blue curve flat-lines
  almost immediately — it dives into the nearest local minimum and quits
  after a **median of 22 evaluations** of its 4000 budget, then sits an order
  of magnitude above everything else. Multi-start CMA-ES wins: its purple
  curve descends in visible *steps*, each restart sampling a fresh basin and
  ratcheting the running best down, with the handoff and plain CMA-ES close
  behind.
- **Rosenbrock** rewards sustained refinement, and the curves flip. L-BFGS-B
  is now a near-vertical *cliff*, plunging down the valley to ~5e-11 (its
  best seed hits 2.2e-12, a hair above the plotter's 1e-12 log-floor, so
  every run is still drawn honestly). The same restart fragmentation that won
  Rastrigin leaves Multi-start CMA-ES *worst* here — its curve stays flat near
  the top, since no single 1000-eval restart refines the valley.
- The handoff and the IQR bands read straight off the traces. The dashed
  marker (median **~1613 evals**) is auto-drawn from `handoff_eval`; on
  Rosenbrock the green `CMA-ES -> L-BFGS-B` curve tracks plain CMA-ES up to it,
  then peels away and dives to ~1e-6 as L-BFGS-B refinement takes over. The
  shaded bands carry the per-seed spread a median line hides: L-BFGS-B's band
  on Rosenbrock fans from ~1e-11 up to ~4, because a few seeds stall in a hard
  region — the same bimodal story, shown as a band.

---

## 2. Run once, plot many — persistence

`bench.run()` above wrote `traces.json`, `runs.csv`, and `summary.csv` into
`output_dir` (it is `save_artifacts=True` by default). Re-plotting or any
post-hoc analysis then reloads the lean `RunTrace`s and **never re-runs an
optimizer** — the 120-run grid is paid for once:

```python
from src.benchmarking import load_traces_json
from src.plotting import plot_benchmark_convergence, plot_benchmark_boxplot

# Days later, no optimizers involved:
traces = load_traces_json("plots/basic/benchmark_showcase/traces.json")
plot_benchmark_convergence(traces, problems=problems, algorithms=algorithms, ...)
plot_benchmark_boxplot(traces, problems=problems, algorithms=algorithms, ...)  # or any other view
```

`load_traces_json` returns exactly the `{(problem, algorithm): [RunTrace]}`
dict that `Benchmark.run()` produces, so it is a drop-in source for every
benchmark plotter. The showcase script asserts the round-trip: all 120 traces
reload from disk and the reconstructed grid is identical to the in-memory one.

This is the division of labour the framework is built around — the expensive,
non-deterministic part (running the grid) happens once and is frozen to disk;
the cheap, iterative part (plotting, summarising, comparing) reads that frozen
record as often as you like. The `RunTrace` is deliberately lean (the
convergence trace plus any retained scalar series, final scalars, and optional
handoff metadata — a *trimmed* `LogData`, not the full population history),
which is what keeps `traces.json` portable and the reload trivial — the
persistence boundary the design draws at `RunTrace`
([`framework_design.md`](framework_design.md) §"Premise 9").
