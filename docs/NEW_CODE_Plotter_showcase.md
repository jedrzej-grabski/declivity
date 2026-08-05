# NEW CODE — Plotter modularity showcase

Three figures, each produced by a **single plotting call**, demonstrating
that the declarative plotter needs no per-algorithm plotting code. The
framework reads the algorithm off each result/trace and the panel registry
decides what every curve *means* — so swapping algorithms or problems
changes nothing about the calls below.

For the design rationale (why declarative, why semantic keys, why a curated
default set) see [`framework_design.md`](framework_design.md) §"Premise 1".

- **Script:** [`experiments/basic/plotter_showcase.py`](../experiments/basic/plotter_showcase.py)
- **Run:** `PYTHONPATH=. uv run python experiments/basic/plotter_showcase.py`
- **Output:** `plots/basic/plotter_showcase/`

| # | One-liner | Entry point | What it proves |
|---|---|---|---|
| 1 | two algorithms' convergence on one axes | `plot_comparison(..., panels=["convergence"])` | cross-family overlay from one semantic key |
| 2 | multi-seed average + spread | `plot_benchmark_convergence(bench.traces, ...)` | median + IQR band per algorithm, per problem |
| 3 | every diagnostic of one optimizer | `plot_metrics(result, panels=PanelSet.ALL)` | the whole registered vocabulary, auto-laid-out |

---

## 1. Two convergences, one graph, two algorithm families

```python
from src.plotting import plot_comparison

plot_comparison(
    {"CMA-ES": cmaes_result, "L-BFGS-B": lbfgsb_result},
    panels=["convergence"],  # one panel → both algorithms overlaid on it
    colors=COLORS,
    save_path="01_two_convergences.png",
)
```

![Two convergences on one graph](../plots/basic/plotter_showcase/01_two_convergences.png)

[`plots/basic/plotter_showcase/01_two_convergences.png`](../plots/basic/plotter_showcase/01_two_convergences.png)

**Why this is modular.** `CMA-ES` is a population evolutionary method and
`L-BFGS-B` is a single-point quasi-Newton method — completely different
internals, different `LogData` types. The semantic key `convergence`
resolves to `best_fitness` for *both*, so a single overlay call draws both
curves with no algorithm-specific branching. (For multi-series panels like
`convergence`, the overlay uses each algorithm's *primary* series, so the
comparison stays a clean one-curve-per-algorithm fitness plot.)

Drop the `panels=` argument entirely and `plot_comparison` defaults to
`PanelRegistry.common([...])` — the intersection of everything registered
for both algorithms. Here it is narrowed to the single `convergence` axes
the brief asked for.

**Validated:** the two curves are cleanly separated on a 10-D Rosenbrock
(both started from the same `x0`). L-BFGS-B plunges down the valley to
≈8e-12 in ~1600 evaluations; CMA-ES descends steadily to ≈3e-1 over the
full 3000-evaluation budget — the textbook contrast between a gradient
refiner and a global searcher, on one graph.

---

## 2. Benchmark average (median + IQR) in one call

```python
from src.plotting import plot_benchmark_convergence

# bench.run() has already populated bench.traces:
#   {(problem.name, algorithm.name): [RunTrace per seed]}
plot_benchmark_convergence(
    bench.traces,
    problems=problems,
    algorithms=algorithms,
    save_path="02_benchmark_average.png",
)
```

![Benchmark median + IQR](../plots/basic/plotter_showcase/02_benchmark_average.png)

[`plots/basic/plotter_showcase/02_benchmark_average.png`](../plots/basic/plotter_showcase/02_benchmark_average.png)

**Why this is modular.** The `Benchmark` grid runs every
(problem × algorithm × seed) triple and persists lean `RunTrace`s; the
plot is one call over `bench.traces`. The "average" is the **median**
convergence line, with a 25/75 **IQR band** shaded beneath it — drawn per
algorithm, one panel per problem, with the algorithm `color`/`name` read
straight off each `AlgorithmRun`. Adding a fourth algorithm or a third
problem requires zero plotting changes.

The CMA-ES → L-BFGS-B handoff runner contributes a `handoff_eval` on its
traces, so the plotter draws the dashed handoff marker automatically (the
vertical line at ≈1200 evals) — another piece of metadata the call reads
rather than is told.

**Validated:** 11 seeds, two 10-D problems. On the smooth **Sphere** the
bands are tight (all three algorithms agree); on **Rosenbrock** the bands
fan out, exposing the genuine per-seed variance — L-BFGS-B's wide IQR
(some seeds stall in the valley, median 5e-11) versus CMA-ES's narrower,
higher band (median 3e0). That spread is exactly what a multi-seed
"average" plot exists to show.

---

## 3. Every diagnostic of an optimizer in one call

```python
from src.plotting import plot_metrics, PanelSet

plot_metrics(
    cmaes_result,
    panels=PanelSet.ALL,  # every registered panel (12 for CMA-ES)
    ncols=3,
    save_path="03_all_diagnostics.png",
)
```

![All CMA-ES diagnostics](../plots/basic/plotter_showcase/03_all_diagnostics.png)

[`plots/basic/plotter_showcase/03_all_diagnostics.png`](../plots/basic/plotter_showcase/03_all_diagnostics.png)

**Why this is modular.** `PanelSet.ALL` renders every panel registered for
the result's algorithm — 12 for CMA-ES — and the grid is laid out
automatically. Each panel is **one line** in
[`standard_panels.py`](../src/plotting/standard_panels.py); register a 13th
and it appears here for free, with no change to this call. (`panels=None`
would render only the curated `default=True` headline subset; `"all"` is
the string-equivalent of `PanelSet.ALL`, since these are `StrEnum`s.)

**Validated:** CMA-ES on a 10-D **Ellipsoid** (condition number 1e6) with
`config.enable_all_diagnostics()`, so every panel has data. The headline
diagnostic — **Condition Number** — climbs toward 1e6, i.e. CMA-ES is
correctly *learning* the problem's true ill-conditioning; `det(C)` shrinks
monotonically as the search volume collapses onto the optimum; the
multi-series **Convergence** panel overlays best/mean/median together.
`PanelSet.ALL` also pulls in the non-default panels — note **Worst
Fitness**, which is near-constant on CMA-ES and is exactly why it is
excluded from the default headline view (see `framework_design.md`
§"Curated defaults").

---

## Framework fix this surfaced

The benchmark one-liner (§2) initially crashed with
`NotImplementedError: Subclass does not implement gradient()`. Root cause —
a pre-existing bug, **not** in the new code:
`Problem.from_benchmark(...)` set the problem gradient via
`getattr(function, "gradient", None)`. Because `BenchmarkFunction` defines
`gradient` as a stub that *raises*, that `getattr` returned the stub (never
`None`) for any function that doesn't override it (`Sphere`, `Rosenbrock`,
`Ellipsoid`, …). L-BFGS-B then called the stub and blew up instead of
falling back to finite differences.

Fix in [`src/benchmarking/problem.py`](../src/benchmarking/problem.py): only
advertise an analytic gradient when the concrete class actually overrides
the base method —

```python
overrides_gradient = (
    getattr(type(function), "gradient", None) is not BenchmarkFunction.gradient
)
... gradient=function.gradient if overrides_gradient else None ...
```

This only changes behavior for functions that previously *crashed* (now
they use finite differences, the documented fallback), and leaves functions
with a real analytic gradient (`Rastrigin`, `Griewank`, `RotatedEllipsoid`,
…) untouched. As a side effect it un-breaks the existing
[`experiments/basic/declarative_benchmark.py`](../experiments/basic/declarative_benchmark.py)
demo, which hit the same path.
