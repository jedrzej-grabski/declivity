# NEW CODE — Unifying single-run and benchmark plotting

A run and a benchmark are now the *same shape at a different seed count*, and
one panel-driven plotter serves both. Pass a single `OptimizationResult` to a
panel and you get a line; pass a 25-seed benchmark to the *same* panel and you
get an aggregated median + IQR band. No second plotting stack.

This supersedes the previous framing (see [§ The transition](#the-transition))
in which single-run diagnostics and multi-seed benchmarks lived on two
unrelated plotting layers that "could not be unified."

---

## 1. The idea

Before, the framework had two plotting worlds:

| Layer | Consumed | Driven by | Functions |
|---|---|---|---|
| single-run | `OptimizationResult` (full `LogData`) | `PanelRegistry` | `plot_metrics`, `plot_comparison` |
| multi-seed | `RunTrace` lists | hand-written, **not** panels | `plot_benchmark_convergence`, … |

A `Panel` (`field="sigma"`, `x_field="evaluations"`, log-scale, …) could only
describe a field on a single run's `LogData`; the benchmark layer was hardwired
to `best_fitness` and couldn't see the registry. "Register a new panel" had no
effect on benchmark plots.

The unification rests on one observation: **`RunTrace` is already a subset of
`LogData`** — both expose `evaluations` and `best_fitness` as parallel arrays.
So they aren't two different things; they're the same shape with different
field coverage. Make that explicit and one plotter falls out:

- **a run** exposes named per-step arrays via `get_series(field)` — both a
  live `LogData` and a persisted `RunTrace` implement it (the `RunSeries`
  contract);
- **a result** is a `RunGroup` — 1..N runs of one algorithm. `N=1` → a line;
  `N=25` → a median + IQR band;
- **one renderer** (`draw_groups`) draws a line-or-band per group for a panel's
  field — and *every* entry point renders through it.

A single run is just a benchmark with one seed.

---

## 2. The new data model

### `RunSeries` — the contract both sources satisfy

```python
# declivity/plotting/unified.py
@runtime_checkable
class RunSeries(Protocol):
    def get_series(self, name: str) -> list[float] | None: ...
```

- `BaseLogData.get_series(name)` → `getattr(self, name, None)` — a live run
  exposes every field it logged (`sigma`, `condition_number`, …).
- `RunTrace.get_series(name)` → `evaluations` / `best_fitness` from its
  first-class lists, everything else from its `series` dict (see §4).

Missing fields return `None`, which the plotter renders as an empty series —
never an error.

### `RunTrace` is now a trimmed `LogData` + metadata

```python
@dataclass
class RunTrace:
    algorithm: str
    problem: str
    seed: int
    evaluations: list[int]
    best_fitness: list[float]
    final_evaluations: int
    final_fitness: float
    handoff_eval: int | None = None
    handoff_iter: int | None = None
    series: dict[str, list[float]] = field(
        default_factory=dict
    )  # ← NEW: retained scalars
```

`series` holds the *additional* cheap scalar-per-step diagnostics kept from the
full `LogData` at run time (`sigma`, `mean_fitness`, `condition_number`, …).
That is what lets the panel that drew `sigma` on one run also band `sigma`
across seeds.

### `RunGroup` — 1..N runs of one algorithm

```python
@dataclass
class RunGroup:
    label: str
    runs: list[RunSeries]  # len 1 → line; len N → band
    color: str | None = None
    algorithm: AlgorithmChoice | None = None  # registry bucket for semantic keys


RunGroup.from_result(result)  # a single run
RunGroup.from_runs(
    "CMA-ES", traces, color=..., algorithm=AlgorithmChoice.CMAES
)  # a benchmark
```

---

## 3. The single plotter

`draw_groups(ax, groups, *, field, …)` is the shared atomic renderer — **the
unification point**. For each group: one run → a raw line; many runs →
`series_grid` + `stack_runs_on_grid` + `percentile_band` → a median curve with
a 25/75 band. `plot_metrics`, `plot_comparison`, `plot_benchmark_convergence`,
and `plot_convergence_overlay` all now render through it.

`plot_panels(data, panels=…)` is the unified front door:

```python
from src.plotting import plot_panels, RunGroup

# Single run → lines (this is exactly plot_metrics):
plot_panels(result, panels=["convergence"])

# Benchmark of one algorithm → the SAME panel, now a median + IQR band:
group = RunGroup.from_runs(
    "CMA-ES",
    bench.traces[("Rastrigin", "CMA-ES")],
    color="#c0392b",
    algorithm=AlgorithmChoice.CMAES,
)
plot_panels(group, panels=["convergence"])  # convergence band
plot_panels(
    group, panels=["step_size"]
)  # σ band — a NON-best_fitness metric, aggregated
plot_panels(group)  # every default panel, as bands

# A dict overlays several algorithms (single runs or benchmarks):
plot_panels({"CMA-ES": cmaes_result, "L-BFGS-B": lbfgsb_result}, panels=["convergence"])
```

`plot_panels` dispatches on what it's handed: a lone run with one seed →
multi-series lines (best/mean/median); anything with `>1` run → the primary
series aggregated into a band. Semantic keys still resolve per algorithm
(`step_size` → `sigma` here, `Ft` there), so a heterogeneous overlay works too.

---

## 4. How `sigma` bands "just work" — the storage policy

When `BenchmarkAlgorithm.trace_from_result` trims a `LogData` into a
`RunTrace`, it now also captures the cheap scalar-per-step fields:

```python
series = capture_scalar_series(result.diagnostic, retain=self.retain_series)
```

`capture_scalar_series` keeps every field that is a list of scalars the same
length as `best_fitness` — and **drops** the heavy vector/matrix fields
(`population`, `eigenvalues`, `best_solution`) that can't persist at scale or be
aggregated into a band. The length filter doubles as a diag-flag gate: a field
only logged when its `diag_*` flag is on (e.g. `sigma` under `diag_sigma`,
`condition_number` under `diag_eigen`) has length 0 when the flag is off, so it
is silently skipped.

So a CMA-ES benchmark run with `diag_sigma` on automatically retains `sigma`,
and `plot_panels(group, panels=["step_size"])` produces a σ band with **zero**
extra configuration. The modularity knob is `BenchmarkAlgorithm.retain_series`:
`None` (default) = auto-retain every cheap scalar; a tuple = exactly those; `()`
= only `best_fitness` (the old lean behavior). Heavy fields are never retained,
so `traces.json` grows by only a few KB of scalar arrays per run.

> **Premise 9, reframed.** The persistence boundary still exists — but it's a
> *storage policy* (which fields to trim), not a *type divide*. `RunTrace` is a
> trimmed `LogData`; the plotter targets the shared `get_series` shape. The
> heavy per-iteration matrices remain single-run-only, because a population
> matrix genuinely can't become a cross-seed band — that's an honest physical
> limit, not an architectural one.

---

## 5. What changed, file by file

| File | Change |
|---|---|
| `declivity/benchmarking/run_trace.py` | `RunTrace.series` dict + `get_series()`; `capture_scalar_series()` + `HEAVY_LOGDATA_FIELDS`. |
| `declivity/logging/base_logger.py` | `BaseLogData.get_series()` (the `RunSeries` contract for live runs). |
| `declivity/benchmarking/persistence.py` | round-trips `series` (omitted when empty → old `traces.json` still load). |
| `declivity/benchmarking/algorithm_run.py` | `trace_from_result` auto-captures scalar series; `BenchmarkAlgorithm.retain_series` knob. |
| `declivity/benchmarking/aggregation.py` | field-agnostic `series_grid` / `stack_runs_on_grid`; old `common_evaluation_grid` / `stack_traces_on_grid` kept as wrappers. |
| `declivity/plotting/unified.py` | **new** — `RunSeries`, `RunGroup`, `draw_groups`, `draw_single_run`, `plot_panels`, shared primitives. |
| `declivity/plotting/declarative.py` | `plot_metrics` → thin wrapper over `plot_panels`; `plot_comparison` → renders via `draw_groups`. |
| `declivity/plotting/benchmark.py` | `plot_benchmark_convergence` / `plot_convergence_overlay` render via `draw_groups`; boxplot unchanged (final-scalar). |
| `declivity/plotting/__init__.py` | export `plot_panels`, `RunGroup`, `RunSeries`, `draw_groups`. |

---

## 6. What unifies, what stays separate

| Plot | Unified? |
|---|---|
| `plot_metrics`, `plot_comparison` | ✅ render through `draw_groups` / `draw_single_run` |
| `plot_benchmark_convergence`, `plot_convergence_overlay` | ✅ render through `draw_groups` (1 seed → line, N → band) |
| `plot_panels` (single run **or** benchmark, any retained metric) | ✅ the new front door |
| `plot_benchmark_boxplot` | ⚠️ shares the `RunTrace` record + colours, but it's a final-*scalar* distribution, a different glyph |
| `plot_interleaved_convergence` (staircase) | ❌ still bespoke — multiple non-aligned x-series + a variable number of disconnected burst segments; no "named time series" abstraction covers it |

The staircase staying bespoke isn't a failure of the unification — it's a
figure whose data was never a single per-step series in the first place. Its
secondary-axis sibling (`plot_convergence_overlay`) *did* fold in, because its
data is exactly `best_fitness` per run.

---

## 7. Backward compatibility

- **Public signatures unchanged.** `plot_metrics`, `plot_comparison`,
  `plot_benchmark_convergence`, `plot_benchmark_boxplot`, and
  `plot_convergence_overlay` keep their exact arguments — every existing
  experiment calls them as before.
- **`traces.json` is forward- and backward-compatible.** `series` is written
  only when non-empty and read with a default, so old files load and new files
  read in older checkouts (they just ignore the extra key).
- **Verified.** A smoke test exercises single-run → `plot_panels` lines,
  benchmark → `plot_panels` σ/convergence bands, the persistence round-trip of
  retained series, and every re-pointed function. The repo experiments
  `plotter_showcase.py` (uses `plot_metrics(PanelSet.ALL)` + `plot_comparison`
  + `plot_benchmark_convergence`), `declarative_benchmark.py` (convergence +
  boxplot), and `handoff/interleaved.py` (staircase + benchmark plots) all run
  unchanged.

---

## 8. The headline figure

A CMA-ES benchmark (6 seeds, `diag_sigma` on), aggregated on `step_size` — i.e.
a median + IQR **σ band**, produced by `plot_panels(group, panels=["step_size"])`,
the *same* `Panel` that draws σ as a single line for one run:

```python
plot_panels(
    RunGroup.from_runs(
        "CMA-ES", traces, color="#c0392b", algorithm=AlgorithmChoice.CMAES
    ),
    panels=["step_size"],
    annotate_final="median",
    save_path="sigma_band.png",
)
```

Before this work, that figure was impossible without a hand-written plotter —
`sigma` never survived into the benchmark layer. Now it's one call, and so is
the same band for `condition_number`, `mean_fitness`, or any other retained
scalar.

---

## The transition

The earlier design notes — `framework_design.md` § "Premise 9" and the Panel
discussion in `NEW_CODE_handoff_experiment_walkthrough.md` § 6 — argued that
the single-run and benchmark layers *could not* be unified because Panels read
`OptimizationResult` and benchmark plots read `RunTrace`. That reasoning was
correct about the code as it stood, but it took the type split as fundamental.
It wasn't: the split was a storage-cost decision, and once `RunTrace` is modeled
as a trimmed `LogData` exposing the same `get_series` shape, the two layers
collapse onto one renderer. Those two sections have been annotated to point
here.
