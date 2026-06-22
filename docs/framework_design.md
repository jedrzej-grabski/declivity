# Framework design

The premises behind how the declivity framework is structured. Read this
before adding a new algorithm, a new plotter primitive, a new extension
point, or anything that changes the public API.

The framework has two surface areas the user actually touches:

1. **Optimizing** — running one or more optimizers on a problem.
2. **Diagnosing** — looking at what they did.

Most of the design effort goes into making (2) cheap. Optimization
results that nobody can read aren't useful.

---

## Premise 1: Declarative > imperative for diagnostic plots

A plotter that hardcodes "show CMA-ES like this, show DES like that"
duplicates layout code per algorithm. Adding a new metric means editing
the algorithm's bespoke plotting method. Comparing two algorithms means
re-stating what's comparable each time.

The framework's plotter avoids this by being **declarative**: you say
what a panel is, the registry says what each algorithm exposes, and
the rendering function reads both.

### What a `Panel` describes

```python
Panel(
    key=PanelKey.STEP_SIZE,   # semantic identifier
    title="Step Size",
    ylabel="σ",
    field="sigma",            # attribute on the LogData
    yscale=YScale.LOG,
    floor=1e-30,              # log-scale safety
    default=True,             # included in headline plot
)
```

That's the whole vocabulary. No layout code, no subplot setup, no
gridspec arithmetic. Adding a new panel for any algorithm is **one
line in `src/plotting/standard_panels.py`**.

### Why semantic keys are the load-bearing piece

The same `PanelKey.STEP_SIZE` is registered with:

- `field="sigma"` on CMA-ES
- `field="Ft"` on DES
- `field="step_length"` on L-BFGS-B

`PanelRegistry.common([CMAES, DES, LBFGSB])` returns the keys
registered for all three (`convergence`, `step_size`). `plot_comparison`
defaults to that intersection: you ask to compare algorithms, the
framework figures out what's comparable.

Algorithm-specific fields keep their own keys (CMA-ES has
`eigenvalue_max`, L-BFGS-B has `theta`). They're available via
explicit selection, just not in the auto-intersection default.

### Curated defaults vs. complete vocabulary

`plot_metrics(result)` with no arguments renders panels marked
`default=True`. That's the **headline** view — the panels that
typically carry signal for that algorithm.

Non-default panels stay registered (queryable via
`PanelRegistry.available(...)`) but aren't in the default render.
Users opt in with `panels=[PanelKey.WORST_FITNESS, ...]` or
`panels=PanelSet.ALL`.

Why this matters: registering everything means the registry is the
single source of truth for "what diagnostics exist." Curated defaults
mean the headline plot stays focused. Without the curation, CMA-ES's
`plot_metrics` was producing 13-panel plots dominated by panels that
never move (worst_fitness on CMA-ES is a near-constant).

---

## Premise 2: Progressive disclosure of extension points

A user who needs to plug something into the benchmarking framework
should encounter as little ceremony as possible. The framework
provides four levels of structure, from concrete-and-easy to abstract-
and-flexible:

```
BenchmarkAlgorithm (ABC)              — implement run() yourself
├── SingleAlgorithm (concrete)        — wraps one optimizer
└── HandoffAlgorithm (ABC)            — implement run_phases()
    └── CMAESLBFGSBHandoff (concrete) — pre-built handoff
```

And, for the case where inheritance isn't viable, the
`AlgorithmRun` Protocol — duck-typed, three attributes (`name`,
`color`, `run()`).

### How to pick which level

| Your runner is... | Inherit from | What you write |
|---|---|---|
| one optimizer | `SingleAlgorithm` | nothing — just instantiate it |
| warmup → refinement | `HandoffAlgorithm` | `run_phases()` returning `(warmup, refinement)` |
| anything else | `BenchmarkAlgorithm` | `run()` returning a `RunTrace` |
| can't inherit | `AlgorithmRun` Protocol | conform structurally |

### Why subclass instead of compose

Each level adds **shared boilerplate**. `BenchmarkAlgorithm` provides
`trace_from_result()` that packages a single `OptimizationResult` into a
`RunTrace`. `HandoffAlgorithm` provides `_stitch_traces()` that handles
eval-count offsets, fitness clamping, and handoff metadata. Subclasses
get those for free and only describe what's algorithm-specific.

When `CMAESLBFGSBHandoff` was refactored to subclass `HandoffAlgorithm`,
~40 lines of trace-stitching boilerplate disappeared from the class.
The remaining ~50 lines are *only* about the CMA-ES covariance
transformation — which is the actually-interesting bit.

### Validation via existing code

A good abstraction makes existing code shorter. The
`HandoffAlgorithm` extraction was validated by re-fitting
`CMAESLBFGSBHandoff` onto it: if the existing pre-built class doesn't
fit, the abstraction is wrong.

---

## Premise 3: Backwards-compatible vocabulary via StrEnum

Public-facing strings (`"inverse"` for handoff transform,
`"convergence"` for panel keys, `"log"` for yscale) get autocomplete
and refactor safety as `StrEnum` types. A `StrEnum` member compares
equal to its value, so:

```python
plot_metrics(result, panels=["convergence", "step_size"])      # works
plot_metrics(result, panels=[PanelKey.CONVERGENCE, PanelKey.STEP_SIZE])  # also works
```

The framework normalizes via `str(...)` internally so the registry
sees the same key regardless of how it was passed.

Why this matters: when the framework added enums, the entire existing
test/experiment base kept working without a single line change. A
breaking enum migration would have been the kind of change that
silently rots in unmaintained corners of a thesis project.

---

## Premise 4: Layered LogData by capability

The framework's optimizers don't all produce the same diagnostic
fields. L-BFGS-B is a single-point method — there's no "population
mean fitness" to log. Putting all fields on a single `BaseLogData`
forced LBFGSB to inherit empty lists for fields it never used.

The fix:

```
BaseLogData                  — iteration, evaluations, best_fitness, best_solution
└── PopulationLogData        — adds worst/mean/std/population/eigenvalues
```

Single-point algorithms inherit `BaseLogData` directly. Population-
based algorithms (DES, CMA-ES, MF-CMA-ES) inherit `PopulationLogData`.

The plotter uses `getattr(log_data, field, None)` so missing fields
just produce empty plots — no exceptions, no special-casing per
algorithm. That tolerance + the layered hierarchy means a user who
writes a new single-point algorithm only inherits fields they'll
actually populate.

---

## Premise 5: Trace stitching is the base class's job

A handoff algorithm runs two optimizers and concatenates their results
into a single convergence trace. That concatenation has subtle bits:

- The refinement's evaluation counter starts at 0; it needs to be
  offset by the warmup's total.
- The refinement's logged "best fitness" starts at `f(x0_refinement)`,
  which might be slightly worse than the warmup's running best. Without
  clamping, the convergence trace goes *up* at the handoff point.
- The plotter wants a `handoff_eval` annotation, and an optional
  `handoff_iter` annotation for iteration-axis plots.

`HandoffAlgorithm._stitch_traces()` handles all three. Subclasses
return `(warmup_result, refinement_result)` and never touch the
arithmetic. That keeps the interesting algorithmic work (e.g. the
covariance transformation in `CMAESLBFGSBHandoff`) un-tangled from the
bookkeeping.

---

## Premise 6: Diagnostics opt-in, not opt-out

Every algorithm always logs the universal trace fields (iteration,
evaluations, best_fitness, best_solution). Population, eigenvalues,
detailed gradient norms, etc. are gated by `diag_*` flags on the
config because they're memory- or compute-expensive.

The flag set is intentionally **lean**: a flag exists only when an
optimizer or logger actually gates behavior on it. During the
refactor, six dead flags were removed from `BaseConfig` (`diag_enabled`,
`diag_value`, `diag_mean`, `diag_meanCords`, `diag_worstVal`,
`diag_bestVal`) plus algorithm-specific dead flags
(`CMAESConfig.diag_cond`, `MFCMAESConfig.diag_archive`).

The principle: **a config field that toggles nothing is misleading,
not flexible**. Users will spend time setting it expecting an effect.
Remove it.

---

## Premise 7: Same seed ⇒ same starting point

The framework guarantees that running multiple algorithms with the
same seed gives them the same initial point. `Problem.starting_point(seed)`
is deterministic. This means:

- A standalone CMA-ES run and a `CMAESLBFGSBHandoff` run with the same
  seed share the entire CMA-ES warmup trace exactly.
- A side-by-side comparison plot reflects *only* the algorithm
  differences, not random variation in starting points.

The `RunTrace.seed` field is the load-bearing piece: persistence,
re-plotting, and statistical aggregation all hinge on it.

---

## Premise 8: Swappable components, structurally enforced

The four moving parts inside every evolutionary optimizer — feasibility,
population repair, single-point initialization, population initialization
— are pluggable via four ABCs:

```
ConstraintHandler         — single-point feasibility, repair, penalty
RepairStrategy            — population-level repair policy (evolutionary-only)
InitialPointGenerator     — where the run starts
PopulationInitializer     — how the initial population matrix is sampled
```

Each ABC ships a discoverability `*Type` enum with a `.build(...)`
factory method. Two equivalent ways to construct a component:

```python
handler = BoxConstraintHandler(BoxStrategy.BOUNCE_BACK, lb, ub)
handler = ConstraintHandlerType.BOX_BOUNCE_BACK.build(lb, ub)
```

The instance form covers parametrized cases that an enum alone can't
express (user-supplied inequality callables, penalty coefficients).
The enum form preserves the discoverability that the previous
`BoundaryHandlerType` enum-as-API provided.

### Why a `PopulationOptimizer` base class

`ConstraintHandler` is universal — every optimizer takes one. The other
three only make sense for population-based algorithms. The split is
expressed by the inheritance hierarchy:

```
BaseOptimizer[LogDataType, ConfigType]      — single-point methods (L-BFGS-B)
└── PopulationOptimizer[LogDataType, …]     — evolutionary algorithms
        __init__ requires repair_strategy and population_initializer
```

`PopulationOptimizer.__init__` takes `repair_strategy` and
`population_initializer` as **required** parameters with no defaults.
Pyright/mypy reject a subclass whose `__init__` does not forward
concrete instances — type-system enforcement instead of runtime
"forgot to set self.repair_strategy" bugs.

This mirrors the existing `BaseLogData` / `PopulationLogData` split
at the logging layer. The two splits are independent (a single-point
optimizer pairs `BaseOptimizer` with `BaseLogData`; an evolutionary
optimizer pairs `PopulationOptimizer` with `PopulationLogData`), but
the parallel structure makes the inheritance choices obvious.

### Why `ConstraintHandler` and `RepairStrategy` are separate

`ConstraintHandler` decides what is feasible and how to repair a
*single point*. `RepairStrategy` decides how to apply that repair
across an *entire population* (or whether to skip it). The split lets
L-BFGS-B (single-point) carry a `ConstraintHandler` without paying for
a `RepairStrategy` it would never use, and lets evolutionary
algorithms swap population-level policy (Lamarckian vs. non-Lamarckian
vs. clamp-only) without touching feasibility semantics.

Folding the two into a single ABC would force CMA-ES to declare a
no-op population-repair method just to satisfy the interface — a
classic "interface segregation" violation.

---

## Premise 9: `RunTrace` as the persistence boundary

Multi-seed benchmarks would produce gigabytes of `LogData` if every seed kept
the full population history. So the persisted per-seed record is a **trimmed
`LogData`**, not a separate type:

- **`OptimizationResult` (in-memory)** — the full `LogData` for a single run.
- **`RunTrace` (persisted)** — the convergence trace + handoff metadata +
  retained cheap scalar-per-step series (`sigma`, `condition_number`, …) in
  `RunTrace.series`. Heavy per-iteration matrices (population, eigenvalues, the
  per-step solution) are dropped.

Both expose the same `get_series(field)` shape, so **one** panel-driven plotter
(`plot_panels`) renders a single run as lines and a benchmark as median + IQR
bands — the *same* `Panel` drives both. `draw_groups` is the shared renderer
(1 run → line, N runs → band); `plot_metrics` / `plot_comparison` /
`plot_benchmark_convergence` are thin wrappers over it.

The boundary is therefore a **storage policy** (which fields to trim), not a
type divide. What's retained is the knob (`BenchmarkAlgorithm.retain_series`,
auto by default); the only inherent limit is that heavy matrices stay
single-run-only, because a population matrix can't become a cross-seed band.
Full write-up: [`NEW_CODE_unified_plotting.md`](NEW_CODE_unified_plotting.md).

---

## How the design got here (iteration log)

The framework wasn't designed top-down. The current shape emerged by
iterating on review feedback:

1. **Initial review.** Audit found the old `MultiAlgorithmPlotter`
   had a 100-line `_plot_<algo>_metrics` method for each algorithm,
   duplicating layout boilerplate four times. The benchmarking layer
   had a separate `BenchmarkPlotter` that didn't share any code with
   it. Adding a panel meant editing the algorithm's method.

2. **Declarative panel system.** First pass: `Panel` + `PanelRegistry`
   + `plot_metrics` + `plot_comparison`. Cross-algorithm semantic keys
   from the start. Multi-seed left for later.

3. **Multi-seed declarative functions.** `plot_benchmark_convergence`
   and `plot_benchmark_boxplot` as module functions, reusing the
   existing aggregation helpers. Verified pixel-for-pixel match against
   the legacy `BenchmarkPlotter` on the multimodal benchmark.

4. **Migration of two flagship experiments.** Sanity check: do the
   new plots produce the same insights as the old?

5. **Side-by-side review of `_plot_cmaes_metrics`.** Three regressions
   noticed:
   - 13 panels vs. legacy 8 (too noisy).
   - Multi-series convergence (best+mean+median on one axes) gone.
   - "Worst Fitness on CMA-ES" panel rendering essentially flat noise.

   Fix: added `Series` + `default=bool` to `Panel`. Curated default
   subsets per algorithm. Multi-series for the convergence panel.

6. **StrEnum vocabulary.** Realized every public string had a finite
   set of values. `PanelKey`, `YScale`, `XAxis`, `LineStyle`, `PanelSet`.
   Backwards-compatible because `StrEnum` members are strings.

7. **`HandoffAlgorithm` extraction.** The custom-handoff demo
   (DES → L-BFGS-B) duplicated ~40 lines of trace stitching from the
   pre-built `CMAESLBFGSBHandoff`. Extracted the boilerplate into a
   `HandoffAlgorithm` ABC. Validated by re-fitting `CMAESLBFGSBHandoff`
   onto it — the refactor dropped ~40 lines from the pre-built class.

8. **`BenchmarkAlgorithm` extraction.** Same pattern one level up.
   `SingleAlgorithm` and `HandoffAlgorithm` both had `trace_from_result`-
   shaped helpers; extracted into a common `BenchmarkAlgorithm` ABC.
   Now any custom benchmark runner picks the level of structure it
   wants.

9. **`BaseLogData` split + diag flag prune.** Cleanup pass: removed
   the dead diag flags (six on `BaseConfig`, two algorithm-specific),
   split `BaseLogData` so single-point methods don't inherit empty
   population fields.

10. **All experiments migrated, legacy plotters deleted.** −1409 lines
    of legacy plotting code removed.

The pattern through all of this: **extract abstractions only after the
duplication makes them obvious**. Each extraction (`Panel`,
`HandoffAlgorithm`, `BenchmarkAlgorithm`) was preceded by writing the
same boilerplate twice (or finding it already written four times in
the old code).

---

## What we *didn't* do

A few things were deliberately left as deferred work:

- **Extending `RunTrace`** with per-seed diagnostic fields (so
  multi-seed plots could show e.g. step-size bands across algorithms).
  Doable, but expands persistence scope significantly.
- **Generalizing `HandoffAlgorithm` to N phases.** Two-phase covers
  every current use case; three-phase would be over-engineering until
  a real example shows up.
- **`MultiPhaseAlgorithm` as an explicit base class.** A three-phase
  algorithm today implements `BenchmarkAlgorithm.run()` directly. If
  three-phase becomes common, extract a base class then, not now.
- **Removing the `cmaes_reference.py` port.** It is kept solely as the
  oracle in `experiments/cross_validation/cmaes_vs_reference.py`. The
  framework CMA-ES (`cmaes_optimizer.py`) is now a clean Hansen-2016
  implementation that uses `RepairStrategy` + `PopulationInitializer`
  through the standard `PopulationOptimizer` ABC, so the two
  implementations are no longer bit-equivalent — see
  `docs/cmaes_framework_integration.md` for the convergence-equivalence
  evidence.

The point of leaving them: **abstractions extracted before you have
the duplication are usually wrong**.
