# NEW CODE — Building a handoff experiment from the bare framework

A developer walkthrough: starting from the framework's public API, what steps
do you actually take to build a study like the interleaved CMA-ES ⇆ L-BFGS-B
experiment (`experiments/handoff/interleaved.py`) and its CMABFGS replication
(`experiments/handoff/cmabfgs_replication.py`)?

This is the hands-on, code-first companion to:

- [`framework_design.md`](framework_design.md) — *why* the framework is
  shaped this way (the premises cited below by number).
- [`NEW_CODE_interleaved_handoff.md`](NEW_CODE_interleaved_handoff.md) —
  *what* this experiment is and the results it produced.
- [`NEW_CODE_building_an_experiment.md`](NEW_CODE_building_an_experiment.md)
  — the terser conventions + anti-pattern reference.

Everything below is grounded in the actual classes you call, in the order a
developer hits the decisions.

> **Scope assumptions for this walkthrough.** We treat CMA-ES's pause/resume
> machinery (`CMAESState` + `get_state()` + `initial_state=`) as *already
> present* in the framework — it is a pre-existing capability the interleaved
> scheme simply *uses*. The genuinely new code this study writes is: one new
> `BenchmarkFunction` (the SDP base), the interleaving scheme, and two bespoke
> plots. Everything else (the `ShiftedFunction` wrapper, the handoff base
> classes, `Benchmark`, the stock plotters) is reused.

---

## The mental model

> **An experiment is thin orchestration. Anything reusable — a function, a
> scheme, a plot type — lives in `src/`, not in the experiment file.**

The experiment script's whole job is to *wire together* four things:

```
Problem(s)          what you optimize        src/benchmarking/problem.py
   │                                         + src/utils/benchmark_functions.py
   ▼
Algorithm specs     what you compare         src/benchmarking/algorithm_run.py
   │   name · color · run(problem, x0, seed) -> RunTrace
   ▼
Benchmark           runs every (problem × algo × seed), persists traces
   │   bench.traces : {(problem.name, algorithm.name): [RunTrace]}
   ▼
Declarative plots   read traces, render      src/plotting/*
```

Each step below is one of those arrows.

---

## Step 1 — Frame the comparison as `problems × contenders × seeds`

The interleaved study asks: *on an ill-conditioned problem whose optimum sits
in a corner of the feasible box, how does alternating CMA-ES with short
L-BFGS-B probes compare to each algorithm alone and to a single one-shot
handoff?* That decomposes into:

- **Problem:** Shifted Different Powers (`SDP`) — an ill-conditioned bowl with
  its optimum pushed near a corner of an asymmetric `[-180, 20]^d` box.
- **Contenders:** CMA-ES, L-BFGS-B, one-shot `CMA-ES → L-BFGS-B`, interleaved
  `CMA-ES ⇆ L-BFGS-B`.
- **Seeds:** a multi-seed grid for medians, plus one "headline" seed for a
  single-run dissection.

Everything else is expressing that grid in framework terms.

---

## Step 2 — Build the objective: a new `BenchmarkFunction` + an existing wrapper

A `Problem` bundles the objective, dimensions, bounds, an optional analytic
gradient, and a **deterministic** `starting_point(seed)`. That determinism is
the fairness guarantee (Premise 7): every contender run with the same seed
starts from the *same* `x0`.

The SDP problem is two pieces — and only one of them is new code.

### 2a. The `BenchmarkFunction` contract

A new objective subclasses `BenchmarkFunction`. The base class (in
`src/utils/benchmark_functions.py`) defines exactly four things a subclass
fills in — three required, one optional:

```python
class BenchmarkFunction:
    def __init__(self, dimensions: int):
        self.dimensions = dimensions

    def __call__(self, x: NDArray) -> float: ...           # the objective       (required)

    @property
    def bounds(self) -> tuple[NDArray, NDArray]: ...        # (lower, upper) arrays (required)

    @property
    def global_minimum(self) -> tuple[NDArray, float]: ...  # (x*, f*)             (required)

    def gradient(self, x: NDArray) -> NDArray:              # analytic gradient    (optional)
        raise NotImplementedError                            # base raises → FD fallback
```

`gradient` is special: the base method *raises*, and `Problem.from_benchmark`
only advertises an analytic gradient when the concrete class **overrides** it
(it checks `type(function).gradient is not BenchmarkFunction.gradient`). So if
you implement `gradient`, L-BFGS-B uses it; if you don't, L-BFGS-B silently
falls back to finite differences. You get to choose, per function.

### 2b. Write SDP from scratch

The SDP base — Different Powers, `f(x) = Σ_i |x_i|^(2 + 4 i/(d-1))`, exponents
ramping 2→6 — is genuinely new, so it lands in `benchmark_functions.py` as a
full `BenchmarkFunction`. We implement `gradient` because it's cheap and
closed-form, which lets L-BFGS-B use exact derivatives:

```python
# src/utils/benchmark_functions.py
class DifferentPowers(BenchmarkFunction):
    """f(x) = Σ_i |x_i|^(2 + 4 i/(d-1))  — a smooth, single-basin, strongly
    ill-conditioned bowl. Global minimum f(0) = 0."""

    def __init__(self, dimensions: int, lower: float = -100.0, upper: float = 100.0):
        super().__init__(dimensions)
        self._lower, self._upper = lower, upper
        # per-coordinate exponent, 2 .. 6
        self._exponents = 2.0 + 4.0 * np.arange(dimensions) / max(dimensions - 1, 1)

    def __call__(self, x):
        return float(np.sum(np.abs(x) ** self._exponents))

    def gradient(self, x):                       # d/dx_i |x_i|^p = p|x_i|^(p-1) sign(x_i)
        p = self._exponents
        return p * np.abs(x) ** (p - 1.0) * np.sign(x)

    @property
    def bounds(self):                            # constructor-configurable box
        n = self.dimensions
        return self._lower * np.ones(n), self._upper * np.ones(n)

    @property
    def global_minimum(self):
        return np.zeros(self.dimensions), 0.0
```

Note `lower` / `upper` are constructor knobs: the optimum is at the origin,
but the *box* can be made asymmetric (`[-180, 20]`) so the origin is no longer
its centre — which sets up the corner shift next.

### 2c. Apply the *existing* `ShiftedFunction` wrapper (reuse, don't rewrite)

We do **not** write a new "shifted SDP" class. The translation is a generic
concern the framework already owns: `ShiftedFunction` wraps any base function
as `f_shifted(x) = f_base(x − shift)`, and its `near_corner(...)` constructor
drops the optimum a `fraction` of the way from the box centre toward a corner.
We just compose it:

```python
from src.benchmarking import Problem
from src.utils.benchmark_functions import DifferentPowers, ShiftedFunction

base = DifferentPowers(100, lower=-180.0, upper=20.0)            # NEW code (2b)
func = ShiftedFunction.near_corner(base, fraction=0.9,           # REUSED wrapper
                                   name_suffix="SDP-corner")
problem = Problem.from_benchmark("SDP", func)                    # bounds + gradient picked up
```

Two things the *existing* wrapper gives us for free:

- **Bounds stay in x-space.** `ShiftedFunction.bounds` returns the base
  bounds unchanged, so the feasible box does not move with the optimum —
  that's precisely what makes the problem bound-active and gives L-BFGS-B's
  projected-gradient / Cauchy-point machinery work to do.
- **The gradient chains through.** `ShiftedFunction.gradient(x)` returns
  `base.gradient(x − shift)`, so because we implemented `DifferentPowers.gradient`,
  the composed problem still advertises an analytic gradient to L-BFGS-B.

> **Why `fraction=0.9`, not `1.0`.** Exactly on the corner a *bounded*
> L-BFGS-B solves it in one step — the first Cauchy point projects every
> coordinate onto the active bound, which *is* the optimum. Keeping the
> optimum slightly interior forces a real ill-conditioned descent. A modelling
> subtlety — captured by a constructor argument on the reused wrapper, not by
> new code.

The lesson: **new base function in `src/`, generic transform reused.** If
you find yourself writing a `Shifted<Anything>` class in an experiment, stop —
compose `ShiftedFunction` instead.

---

## Step 3 — Express each contender as an `AlgorithmRun`

Everything inside a `Benchmark` provides three things: `name`, `color`, and
`run(problem, x0, seed) -> RunTrace`. *How* you provide them is a
progressive-disclosure choice (Premise 2) — pick the **narrowest** base class
that fits:

| Your runner is… | Inherit from | You write |
|---|---|---|
| one factory optimizer | `SingleAlgorithm` (concrete) | nothing — just instantiate |
| warm-up → refinement (2 phases) | `HandoffAlgorithm` | `run_phases() -> (warmup, refinement)` |
| anything else (3+ phases, restarts, interleaving) | `BenchmarkAlgorithm` | `run() -> RunTrace` |
| already a class, can't inherit | `AlgorithmRun` Protocol | conform structurally |

Carry `name` + `color` as dataclass fields and take a
**`config_factory: Callable[[int], Config]`** so the config adapts to the
problem dimension at run time.

The three contenders that need no new code:

```python
from src.algorithms.choices import AlgorithmChoice
from src.algorithms.cmaes.config import CMAESConfig
from src.algorithms.lbfgsb.config import LBFGSBConfig
from src.benchmarking import SingleAlgorithm, CMAESLBFGSBHandoff

TOTAL = 1_000_000

cmaes = SingleAlgorithm(
    name="CMA-ES", color=COLORS["CMA-ES"],
    algorithm=AlgorithmChoice.CMAES,
    config_factory=lambda d: CMAESConfig(dimensions=d, budget=TOTAL),
)
lbfgsb = SingleAlgorithm(
    name="L-BFGS-B", color=COLORS["L-BFGS-B"],
    algorithm=AlgorithmChoice.LBFGSB,
    config_factory=lambda d: LBFGSBConfig(dimensions=d, budget=TOTAL, pgtol=1e-10, factr=0),
)
one_shot = CMAESLBFGSBHandoff(                 # pre-built two-phase rung
    name="CMA-ES -> L-BFGS-B (one-shot)", color=COLORS["one-shot"],
    cmaes_config_factory=lambda d: CMAESConfig(dimensions=d, budget=2000),
    lbfgsb_config_factory=lambda d: LBFGSBConfig(dimensions=d, budget=TOTAL - 2000),
    transform="inverse",                       # B₀ = C⁻¹
)
```

The fourth contender — the interleaved scheme — is **multi-phase** (CMA-ES
slice → probe → repeat). `HandoffAlgorithm` is strictly two-phase and can't
express it, so it's the textbook case for subclassing `BenchmarkAlgorithm`
directly. That's Step 4.

---

## Step 4 — Write the multi-phase scheme in `src/benchmarking/`

The interleaving scheme is reusable code, so it's an
`InterleavedCMAESLBFGSB(BenchmarkAlgorithm)` in
`src/benchmarking/algorithm_run.py` — **not** logic in the experiment. Its
`run()` returns a standard `RunTrace` (the overall-best staircase), so it drops
into `Benchmark` and the stock plotters like any other contender.

It leans on two capabilities the framework already provides:

- **CMA-ES pause/resume** — `cmaes.get_state()` snapshots the evolvable state;
  `CMAESOptimizer(..., initial_state=state, seed=rng)` resumes it. With a
  shared RNG this reproduces a standalone run bit-for-bit, so the CMA-ES
  *backbone* is a true reference curve. (Pre-existing; we just call it.)
- **The covariance→Hessian transform** — `initial_hessian_from_cmaes(...)`,
  already shared with `CMAESLBFGSBHandoff`, turns the cached eigendecomposition
  `(B, D)` into the probe's `B₀`. We reuse it rather than re-derive the math.

```python
# src/benchmarking/algorithm_run.py
@dataclass
class InterleavedCMAESLBFGSB(BenchmarkAlgorithm):
    name: str
    color: str
    cmaes_config_factory: Callable[[int], CMAESConfig]
    lbfgsb_config_factory: Callable[[int], LBFGSBConfig]
    cmaes_interval: int = 20            # N: CMA-ES generations between probes
    total_budget: int = 0
    transform: HandoffTransform | str = HandoffTransform.INVERSE
    probe_factr: float = 1e7           # burst "stops advancing rapidly" stop
    probe_pgtol: float = 1e-8
    probe_max_evals: int = 1000        # hard cap per burst

    def run(self, problem, x0, seed) -> RunTrace:
        return self.run_with_detail(problem, x0, seed).trace      # framework entry point

    def run_with_detail(self, problem, x0, seed) -> "InterleaveResult":
        rng = np.random.default_rng(seed)
        state, cumulative, overall_best = None, 0, float("inf")
        while cumulative < total_budget:
            # 1. advance CMA-ES one slice, RESUMED from its own state (shared RNG)
            cmaes = CMAESOptimizer(problem.function, x0, cfg, ..., seed=rng,
                                   initial_state=state)
            cmaes_result = cmaes.optimize()
            state = cmaes.get_state()                              # snapshot for next slice

            # 2. fire an L-BFGS-B side-probe from the CMA-ES mean, B₀ = C⁻¹
            B, Dsqrt = cmaes.get_eigendecomposition()
            probe_cfg.initial_hessian = initial_hessian_from_cmaes(
                self.transform, B, Dsqrt, cmaes.sigma)
            probe = LBFGSBOptimizer(problem.function, cmaes.mean, probe_cfg, ...)
            probe_result = probe.optimize()

            # 3. fold the probe's improvement into the tracked OVERALL BEST,
            #    then loop with CMA-ES untouched (the probe never feeds back).
            ...
        return InterleaveResult(trace=RunTrace(...), ...)   # standard trace + rich detail
```

`run()` returning a plain `RunTrace` is the only contract `Benchmark` cares
about. The *richer* `InterleaveResult` returned by `run_with_detail()` is for
the bespoke staircase plot — see Step 6.

Now the experiment can instantiate the fourth spec with no new code:

```python
from src.benchmarking import InterleavedCMAESLBFGSB

interleaved = InterleavedCMAESLBFGSB(
    name="Interleaved CMA-ES + L-BFGS-B", color=COLORS["interleaved"],
    cmaes_config_factory=lambda d: CMAESConfig(dimensions=d, budget=TOTAL),
    lbfgsb_config_factory=lambda d: LBFGSBConfig(dimensions=d, budget=TOTAL, pgtol=1e-10, factr=0),
    cmaes_interval=20, total_budget=TOTAL, transform="inverse",
    probe_max_evals=80,
)
```

---

## Step 5 — Run the grid with `Benchmark` (never hand-roll the loop)

```python
from src.benchmarking import Benchmark

bench = Benchmark(
    problems=[problem],
    algorithms=[cmaes, lbfgsb, one_shot, interleaved],
    seeds=list(range(15)),
    output_dir="plots/handoff/interleaved/sdp",
    num_workers=1,
)
bench.run(verbose=True)        # runs every (problem × algo × seed)
bench.print_summary()
```

`Benchmark.run()` gives you, from the code in `benchmark.py`:

- the `(problem × algorithm × seed)` loop with progress printing;
- **same-seed `x0`** shared across algorithms (`Problem.starting_point(seed)` —
  the fairness guarantee);
- optional parallelism (`num_workers > 1`, joblib/loky — the
  `config_factory` lambdas survive because loky uses cloudpickle);
- **auto-persistence**: `traces.json`, `runs.csv`, `summary.csv` in
  `output_dir`;
- `bench.traces`, keyed `(problem.name, algorithm.name)`.

A **single-seed** study is still a `Benchmark` — just `seeds=[0]` (the CMABFGS
replication does exactly this and still gets persistence + the keyed dict).

> ⚠️ **Anti-pattern:** a hand-rolled `for algo in algos: algo.run(problem, x0,
> seed)` loop that assembles its own `{label: RunTrace}` dict and calls
> `save_traces_json` by hand. It reimplements `Benchmark` and loses the
> fairness / persistence / parallelism plumbing. Use `Benchmark` even for one
> seed.

---

## Step 6 — Plot: stock plotters, and the limits of the Panel registry

This study wants three figures: a multi-seed median+IQR convergence grid, a
multi-seed final-fitness boxplot, and (for CMABFGS) a single-panel overlay of
every `k`-variant with a secondary "CMA-ES iterations" axis — plus the
headline **staircase** dissecting one interleaved run.

Before reaching for bespoke code, the right question is the one you asked:
**can any of these be a registered `Panel` instead?** The answer comes
straight out of the plotting code, and it splits the framework's plots into
two layers.

> **Update — this section has been partly superseded.** The single-run and
> benchmark layers were since unified: `RunTrace` is now a trimmed `LogData`
> exposing the same `get_series` shape, and one panel-driven plotter
> (`plot_panels`) renders both — so the convergence band *and* any retained
> scalar (`sigma`, `condition_number`, …) are now reachable from a registered
> `Panel`, not just `best_fitness`. The CMABFGS overlay's secondary axis and
> the staircase remain bespoke for the reasons below (multiple non-aligned
> x-series, variable burst segments). See
> [`NEW_CODE_unified_plotting.md`](NEW_CODE_unified_plotting.md). The layer
> analysis below explains *why* the split existed and why the staircase still
> can't be a panel.

### What the `Panel` registry actually governs

A `Panel` (`src/plotting/panel.py`) is a declarative spec read by
`plot_metrics(result)` and `plot_comparison(results)` — and **those two
functions consume `OptimizationResult` objects** (single runs, with a full
`LogData`/`diagnostic`). A panel says: *read field `f` (or several `Series`)
off this run's LogData, against a shared `x_field` (default `"evaluations"`),
on a log/linear y-axis.* Concretely, each `Series` becomes one
`ax.plot(getattr(log_data, x_field), getattr(log_data, field))`.

So the Panel registry's reach is: **per-iteration diagnostic fields of a
single optimizer run.** That is the layer where "register a new panel" is the
right move (e.g. adding a `gradient_norm` panel to L-BFGS-B is one
`PanelRegistry.register(...)` line).

The multi-seed plotters live on a **different layer**:
`plot_benchmark_convergence`, `plot_benchmark_boxplot`, and
`plot_convergence_overlay` consume **`RunTrace` lists**, not
`OptimizationResult`s, and they are *not* panel-driven at all — they read
`trace.evaluations` / `trace.best_fitness` / `trace.final_fitness` directly and
aggregate across seeds. `RunTrace` is the lean persistence record — it carries
the convergence trace plus any *retained* scalar series. (Originally it kept
only `best_fitness`, so the Panel registry couldn't reach the benchmark layer
at all; the unified-plotting work since closed that gap — `RunTrace` is now a
trimmed `LogData` and `plot_panels` bands any retained metric. See the note at
the top of this section.)

### Can the CMABFGS overlay be a Panel?

*Partly — but not the version we need.* `plot_comparison({...},
panels=["convergence"])` already overlays one best-fitness line per algorithm
on one axes, which is the overlay's core shape. But:

- it consumes single-run `OptimizationResult`s, **not** the multi-seed
  `RunTrace` aggregation (median + IQR) the figure shows; and
- the secondary "iterations = evals/(λ+1)" axis is not part of the Panel
  vocabulary — `Panel` has `x_field`, `yscale`, `floor`, `default`, and
  nothing for a second transformed axis.

So the overlay belongs on the `RunTrace` benchmark layer, as a sibling of
`plot_benchmark_convergence` — which is exactly what `plot_convergence_overlay`
is. Not a Panel.

### Can the staircase be a Panel?

**No — this one is genuinely beyond the Panel model**, for four independent
reasons, each visible in the data the figure needs:

1. **No `OptimizationResult` to attach to.** The interleaved runner yields a
   `RunTrace` + `InterleaveResult`, never the single-run `LogData` that
   `plot_metrics`/`plot_comparison` read. A Panel has nothing to bind to here.
2. **Multiple, non-aligned x-series.** A panel's `Series` all share one
   `x_field`. The staircase overlays the overall-best curve (on
   `overall_evaluations`) *and* the CMA-ES backbone (on `cmaes_evaluations`) —
   two different x-arrays on one axes. A Panel can't do that.
3. **A variable number of disconnected segments.** The L-BFGS-B bursts are *N*
   separate polylines that must not be joined to each other. `Series` is a
   fixed tuple, declared at registration time, each field → one contiguous
   line. There's no way to express "one segment per burst, count known only at
   run time."
4. **Non-line decorations.** The per-burst `axvline` markers aren't a `Series`
   concept at all.

You could imagine cramming the backbone/burst arrays into a fake LogData and
registering a multi-series panel, but reasons 2–4 still make it impossible —
and reason 1 means you'd be fighting the entire `OptimizationResult` →
`plot_metrics` pipeline. **Verdict: too far-fetched. A bespoke function is the
correct tool**, and it's the *blessed* one for exactly this case (Premise 9:
"if you need richer multi-seed/per-run diagnostics, return a richer dataclass
and plot it" — not "force it through the panel registry").

### So: stock plotters for the comparison, bespoke for the rest

The two multi-seed comparison figures are stock `RunTrace` plotters — colours
and names flow from the algorithm objects (Premise 1), never re-specified:

```python
from src.plotting import plot_benchmark_convergence, plot_benchmark_boxplot

plot_benchmark_convergence(bench.traces, problems=[problem], algorithms=algorithms,
                           save_path=out / "convergence.png")
plot_benchmark_boxplot(bench.traces, problems=[problem], algorithms=algorithms,
                       save_path=out / "final_fitness.png")
```

The CMABFGS overlay is the single-panel `RunTrace`-layer sibling, with its
secondary axis:

```python
from src.plotting import plot_convergence_overlay

plot_convergence_overlay(
    traces, problem, algorithms,                 # colours/names read off `algorithms`
    secondary_iter_lambda=effective_lambda,      # adds iterations = evals/(λ+1) axis
    secondary_label=f"CMA-ES iterations (λ={effective_lambda})",
    save_path=save_path,
)
```

The staircase is the bespoke function, following the three rules for a
sanctioned custom plot: **(1)** it lives in `src/plotting/interleaved.py` and
is exported; **(2)** it takes colours from arguments, never re-derives them
from labels; **(3)** it consumes the richer dataclass the runner returns. So
the runner exposes:

```python
@dataclass
class InterleaveResult:
    trace: RunTrace                 # the standard staircase Benchmark consumes
    overall_evaluations: list[int]; overall_best: list[float]
    cmaes_evaluations: list[int];   cmaes_best: list[float]     # the backbone
    burst_segments: list[tuple[list[int], list[float]]]         # each L-BFGS-B drop
    burst_starts: list[int]; cmaes_generations: int; num_bursts: int
```

…and the experiment runs one headline seed directly (the multi-seed comparison
still goes through `Benchmark`) to feed it:

```python
from src.plotting import plot_interleaved_convergence

x0 = problem.starting_point(headline_seed)
detail = interleaved.run_with_detail(problem, x0, headline_seed)   # -> InterleaveResult
baseline = cmaes.run(problem, x0, headline_seed)                   # standalone reference
plot_interleaved_convergence(detail, baseline_trace=baseline,
                             save_path=out / "staircase.png")
```

The trade-off is explicit: this figure is *not* re-plottable from
`traces.json`, because the backbone/burst detail isn't persisted — which is the
same reason it can't be a Panel.

---

## Step 7 — Persist once, re-plot forever

`Benchmark` already wrote `traces.json`. To iterate on a figure without
re-running the optimizers, reload and re-plot:

```python
from src.benchmarking import load_traces_json

traces = load_traces_json(out / "traces.json")
plot_convergence_overlay(traces, problem, algorithms, ...)   # instant
```

The CMABFGS replication wires this in as a `--replot-from` flag: it rebuilds
the (cheap) problem + algorithm specs for colours/names and skips the
expensive run. The bespoke staircase is the lone exception — it re-runs the
headline seed, since its detail isn't in `traces.json`.

---

## Step 8 — Document and index

- Add a row to [`experiments/README.md`](../experiments/README.md)
  (script · what it does · output dir); the `plots/` tree mirrors
  `experiments/`.
- Write a `docs/NEW_CODE_<topic>.md` summary if the study introduces new code
  or a new result.
- If it's a milestone, leave a one-line pointer in auto-memory.

---

## The finished experiment is *thin*

After Steps 2–4 push the new pieces into `src/`, the experiment file itself is
just orchestration — `build_problem`, `build_algorithms`, a `Benchmark`, and a
handful of `src.plotting` calls. The only module-level boilerplate is the
headless-rendering setup (so a saved-figure run never tries to open a window):

```python
import matplotlib.pyplot as plt
plt.ioff()
plt.switch_backend("Agg")          # render to files, no display

COLORS = {"CMA-ES": "#c0392b", "L-BFGS-B": "#2980b9", ...}   # palette fed into the specs
```

The interleaving scheme, the `DifferentPowers` base function, the bespoke
plots — every genuinely reusable piece — live in `src/`. That's the whole
game.

---

## Conventions checklist

- [ ] Problem built with `Problem.from_benchmark`; a new objective is a
      `BenchmarkFunction` (implement `__call__` / `bounds` / `global_minimum`,
      and `gradient` if you want exact derivatives). Generic transforms
      (`RotatedFunction`, `ShiftedFunction`) are **composed, not rewritten**.
- [ ] Each contender is `name` + `color` + `run() -> RunTrace` via the
      **narrowest** base class; configs come from a dimension-keyed
      `config_factory`.
- [ ] A new multi-phase scheme is a `BenchmarkAlgorithm` subclass in
      `src/benchmarking/`, returning a standard `RunTrace`; reuse existing
      optimizer capabilities (`CMAESState`, `initial_hessian_from_cmaes`)
      rather than re-deriving them.
- [ ] Runs go through `Benchmark` (even single-seed).
- [ ] Per-run, per-iteration diagnostics → a registered `Panel`
      (`plot_metrics` / `plot_comparison`). Multi-seed / RunTrace figures →
      the benchmark plotters. Don't try to force a `RunTrace`-layer figure
      through the Panel registry — it can't reach that data.
- [ ] Any custom plot lives in `src/plotting/`, consumes
      `RunTrace` / a returned dataclass, and takes colours from the algorithms.
- [ ] `experiments/README.md` row + `docs/NEW_CODE_*.md` write-up.

## Anti-patterns (and the conforming fix)

| Anti-pattern | Why it's wrong | Fix |
|---|---|---|
| A `Shifted<X>` / `Rotated<X>` class in the experiment | re-implements a generic transform | compose `ShiftedFunction` / `RotatedFunction` |
| Hand-rolled `(algo × seed)` loop + manual `save_traces_json` | loses fairness / persistence / parallelism | `Benchmark([...], seeds=[...]).run()` |
| `matplotlib` figure-building inside the experiment | un-reusable, duplicates layout | a function in `src/plotting/`, exported |
| Forcing a multi-seed / staircase figure into a `Panel` | Panels read single-run LogData on one shared x-axis; can't reach `RunTrace`/`InterleaveResult` | a `RunTrace`-layer plotter, or a bespoke function consuming a returned dataclass |
| Colours re-derived from string labels | symptom of bypassing the `algorithms` list | pass the algorithm objects; read `.color` |

**Legitimate custom plots** (still in `src/plotting`, still reusable): a
secondary-axis overlay (`plot_convergence_overlay`) and a single-run
dissection that needs richer-than-`RunTrace` detail
(`plot_interleaved_convergence` + `InterleaveResult`). Both are real gaps the
Panel registry structurally cannot fill, which is exactly why they're
functions, not panels.
