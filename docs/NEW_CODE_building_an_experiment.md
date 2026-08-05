# NEW CODE — Authoring a benchmark experiment, step by step

A practical recipe for adding a new study under `experiments/`, using the
interleaved CMA-ES ⇆ L-BFGS-B work (`experiments/handoff/interleaved.py` and
`experiments/handoff/cmabfgs_replication.py`) as the running example. It
states the framework conventions, the order to apply them, and — explicitly —
where those two experiments stepped *off* the conventions and what the
conforming alternative is.

Companion reading: [`framework_design.md`](framework_design.md) (the *why*),
[`NEW_CODE_interleaved_handoff.md`](NEW_CODE_interleaved_handoff.md) (the
*what* this example builds), and
[`NEW_CODE_handoff_experiment_walkthrough.md`](NEW_CODE_handoff_experiment_walkthrough.md)
(the same recipe as a code-first, step-by-step developer walkthrough). This
document is the terser conventions + anti-pattern reference.

---

## The golden rule

> **An experiment is thin orchestration. Anything reusable — a function, an
> algorithm, a plot type, an optimizer capability — lives in `declivity/`, not in
> the experiment file.**

A good `experiments/<group>/<name>.py` only: builds `Problem`s, builds a list
of algorithm specs, hands them to `Benchmark`, and calls plotting entry
points. If you find yourself writing a `for` loop over seeds, raw
`matplotlib`, or a persistence call, stop — the framework already owns that.

## Anatomy of an experiment

```
Problem(s)            what you optimize         (declivity/benchmarking/problem.py)
   │                                            + declivity/utils/benchmark_functions.py
   ▼
Algorithm specs       what you compare          (declivity/benchmarking/algorithm_run.py)
   │   name · color · run(problem, x0, seed) -> RunTrace
   ▼
Benchmark             runs every (problem×algo×seed), persists traces
   │   bench.traces : {(problem, algorithm): [RunTrace]}
   ▼
Declarative plots     read traces, render        (declivity/plotting/*)
```

---

## Step 1 — Define the problem

**Stock function** → one line:

```python
from src.benchmarking import Problem
from src.utils.benchmark_functions import Rastrigin

problem = Problem.from_benchmark(
    "Rastrigin", Rastrigin(10)
)  # picks up bounds + gradient
```

**Custom objective** → add it to `declivity/utils/benchmark_functions.py` as a
`BenchmarkFunction` (implement `__call__`, `bounds`, `global_minimum`, and
`gradient` if you want analytic gradients for L-BFGS-B). Compose with the
wrappers instead of writing a new class when you can:

- `RotatedFunction(base, rotation=...)` — rotate the input (off-diagonal curvature).
- `ShiftedFunction(base, shift)` / `ShiftedFunction.near_corner(base, fraction)` —
  translate the optimum (e.g. onto a box corner). Composes with `RotatedFunction`.

Example from the replication (`f = SDP`):

```python
base = DifferentPowers(100, lower=-180.0, upper=20.0)  # new BenchmarkFunction
func = ShiftedFunction.near_corner(base, fraction=0.9)  # optimum near the corner
problem = Problem.from_benchmark("SDP", func)
```

Why this matters: `Problem.starting_point(seed)` is deterministic, so every
algorithm run with the same seed starts from the same `x0` — the property the
fair-comparison guarantee rests on. Get the objective into a `Problem` and you
inherit that for free.

---

## Step 2 — Express each contender as an `AlgorithmRun`

Everything inside a `Benchmark` provides three things: `name`, `color`, and
`run(problem, x0, seed) -> RunTrace`. Pick the **narrowest** base class:

| Your runner is… | Inherit from | You write |
|---|---|---|
| one factory optimizer | `SingleAlgorithm` (concrete) | nothing — just instantiate |
| warm-up → refinement (2 phases) | `HandoffAlgorithm` | `run_phases() -> (warmup, refinement)` |
| anything else (3+ phases, restarts, interleaving) | `BenchmarkAlgorithm` | `run() -> RunTrace` |
| already a class, can't inherit | `AlgorithmRun` Protocol | conform structurally |

Use a **`config_factory: Callable[[int], Config]`** so the config adapts to the
problem dimension. Carry `name` + `color` as dataclass fields.

The interleaved study uses three rungs at once:

```python
SingleAlgorithm(name="CMA-ES", color=..., algorithm=AlgorithmChoice.CMAES,
                config_factory=lambda d: CMAESConfig(dimensions=d, budget=B))
CMAESLBFGSBHandoff(name="CMA-ES -> L-BFGS-B", ...)        # the 2-phase rung
InterleavedCMAESLBFGSB(name="Interleaved ...", ...)      # a BenchmarkAlgorithm
```

`InterleavedCMAESLBFGSB` is the textbook case for `BenchmarkAlgorithm`: it is
multi-phase (CMA-ES slice → probe → repeat), which `HandoffAlgorithm`
(strictly two-phase) can't express — exactly the case `framework_design.md`
defers to a direct `run()`.

---

## Step 3 — Need a capability the optimizer lacks? Extend it *additively*

Sometimes the runner needs something the optimizer doesn't expose. The
interleaved scheme needed to **pause and resume** CMA-ES. The conforming move:

- Add the capability to the optimizer as a **small, additive, documented API**
  with a default that preserves existing behaviour (`CMAESState` +
  `initial_state=None` + `get_state()`). Existing callers are untouched.
- **Verify it's non-regressing.** There's no test suite, so prove it: the
  resume is byte-identical to a continuous run (same best/mean/C) — that check
  *is* the regression test.
- Prefer the existing **component seams** (`RepairStrategy`,
  `PopulationInitializer`, `line_search`, `gradient_strategy`,
  `ConstraintHandler`) over hardcoding behaviour.

Do **not** fork the optimizer or reach into private state from the experiment.
If the experiment wants behaviour the optimizer can't provide, that's a signal
to extend `declivity/`, not to hack around it in `experiments/`.

---

## Step 4 — Run with `Benchmark` (do not hand-roll the loop)

```python
bench = Benchmark(
    problems=[problem],
    algorithms=algorithms,
    seeds=list(range(num_seeds)),
    output_dir=out,
    num_workers=1,
)
bench.run(verbose=True)  # runs every (problem × algo × seed)
bench.print_summary()
```

`Benchmark.run()` gives you, for free:

- the `(problem × algorithm × seed)` loop, with progress printing;
- **same-seed `x0`** shared across algorithms (fairness);
- optional parallelism (`num_workers > 1`, joblib/loky — lambdas in
  `config_factory` survive because loky uses cloudpickle);
- **auto-persistence**: `traces.json`, `runs.csv`, `summary.csv` in `output_dir`;
- `bench.traces` keyed `(problem.name, algorithm.name)`.

A **single-seed** study is just `seeds=[0]` — you still get persistence and the
keyed trace dict. Re-plot later without re-running via `load_traces_json`.

> ⚠️ **Anti-pattern:** a manual `for algo in algos: algo.run(problem, x0,
> seed)` loop that assembles its own `{label: RunTrace}` dict and calls
> `save_traces_json` by hand. It reimplements `Benchmark` and loses the
> fairness/persistence/parallelism plumbing. Use `Benchmark` even for one
> seed. *(An early draft of `cmabfgs_replication.py` did exactly this; it was
> since refactored onto a single-seed `Benchmark` — the conforming form.)*

---

## Step 5 — Plot with the declarative entry points

The declarative entry points in `src.plotting` — pick by the question you're
answering:

| Function | Use when |
|---|---|
| `plot_panels(data)` | unified front door: a single run → lines, a benchmark (`RunGroup`/dict) → median+IQR bands, same panels |
| `plot_metrics(result)` | one run, multi-panel deep dive |
| `plot_comparison(results)` | a few runs, semantic-key overlay |
| `plot_evaluation_bars(results)` | total-evaluation bar chart |
| `plot_benchmark_convergence(traces, problems, algorithms)` | multi-seed median + IQR per algorithm |
| `plot_benchmark_boxplot(traces, problems, algorithms)` | final-fitness distribution |
| `plot_function_landscape[_grid]` | 2-D contour of an objective |
| `plot_matrix_diagonal_comparison` | sorted-diagonal vs a reference matrix |

```python
plot_benchmark_convergence(
    bench.traces,
    problems=[problem],
    algorithms=algorithms,
    save_path=out / "convergence.png",
)
plot_benchmark_boxplot(
    bench.traces,
    problems=[problem],
    algorithms=algorithms,
    save_path=out / "final_fitness.png",
)
```

Colours and names flow from the **algorithm objects** — you pass the
`algorithms` list and the plotter reads `.color`/`.name`. You never re-specify
them at plot time.

---

## Step 6 — When the stock plots don't fit, go custom *the framework way*

Some figures genuinely aren't covered: the interleaved **staircase**
(needs the CMA-ES backbone + per-burst segments, which `RunTrace` doesn't
carry) and the CMABFGS **single-seed overlay with a secondary "CMA-ES
iterations" axis**. Going custom is fine — but follow three rules:

1. **The plot function lives in `declivity/plotting/`**, not in the experiment —
   alongside the other non-panel specialized plots (`landscape.py`,
   `diagnostics.py`, `interleaved.py`). Export it from `declivity/plotting/__init__.py`.
2. **It consumes `RunTrace` (or a richer dataclass you return) and takes the
   colours from the algorithm objects** — it does not re-derive colour from a
   string label.
3. **If you need richer-than-`RunTrace` per-run detail, return a dataclass from
   your runner** (e.g. `InterleaveResult` from `run_with_detail`) and feed it to
   the custom plot. Be aware this figure steps off the `Benchmark`/persistence
   path — that's the documented trade-off ("`RunTrace` is the persistence
   boundary; extend it if you need richer per-seed diagnostics").

Both custom figures in this study do it right, and both now live in
`declivity/plotting/`:

- `plot_interleaved_convergence` (in `declivity/plotting/interleaved.py`) takes an
  `InterleaveResult` and draws the staircase.
- `plot_convergence_overlay` (in `declivity/plotting/benchmark.py`) draws one
  semilogy line per algorithm on a single panel, colours read from
  `algorithms`, with an optional secondary "iterations" axis:

```python
plot_convergence_overlay(traces, problem, algorithms, *,
                         secondary_iter_lambda=None, ...)   # reusable single-seed overlay
```

`cmabfgs_replication.py` calls `plot_convergence_overlay(bench.traces, ...)` —
the conforming form. *(An early draft instead had a `plot_replication`
function building raw matplotlib inside the experiment and re-deriving colours
from labels; extracting it into `declivity/plotting/` is what produced the reusable
`plot_convergence_overlay`.)*

---

## Step 7 — Persist and re-plot

`Benchmark` already wrote `traces.json`. To iterate on a figure without
re-running the optimizers:

```python
from src.benchmarking import load_traces_json
traces = load_traces_json(out / "traces.json")
plot_benchmark_convergence(traces, problems=[problem], algorithms=algorithms, ...)
```

(For a bespoke single-run figure, persist your richer dataclass yourself, or
re-run — but a multi-seed `Benchmark` study should always re-plot from JSON.)

---

## Step 8 — Document and index

- Add a row to [`experiments/README.md`](../experiments/README.md) (script ·
  what it does · output dir). The `plots/` tree mirrors `experiments/`.
- Write a `docs/NEW_CODE_<topic>.md` summary (problem, method, key findings,
  how to run) if the study introduces new code or a new result.
- If it's a milestone, leave a one-line pointer in auto-memory.

---

## Conventions checklist

- [ ] Problem built with `Problem.from_benchmark`; any custom objective is a
      `BenchmarkFunction` in `declivity/utils/benchmark_functions.py` (compose
      `RotatedFunction`/`ShiftedFunction` where possible).
- [ ] Each contender is `name` + `color` + `run()->RunTrace` via the narrowest
      base class; configs come from a dimension-keyed `config_factory`.
- [ ] New optimizer capability = additive, defaulted API on the optimizer in
      `declivity/`, proven non-regressing — never a fork or private-state hack.
- [ ] Runs go through `Benchmark` (even single-seed); rely on its
      auto-persistence and same-seed `x0`.
- [ ] Figures use `src.plotting` entry points; the `algorithms` list supplies
      colours/names.
- [ ] Any custom plot lives in `declivity/plotting/`, consumes `RunTrace`/a returned
      dataclass, and takes colours from the algorithms — not from labels.
- [ ] `experiments/README.md` row + `docs/NEW_CODE_*.md` write-up.

## Anti-patterns (and the conforming fix)

| Anti-pattern | Why it's wrong | Fix |
|---|---|---|
| Hand-rolled `(algo × seed)` loop | loses fairness/persistence/parallelism plumbing | `Benchmark([...], seeds=[...]).run()` |
| `matplotlib` inside the experiment | un-reusable, duplicates layout per study | a function in `declivity/plotting/`, exported |
| Colours re-derived from string labels | symptom of bypassing the `algorithms` list | pass the algorithm objects; read `.color` |
| Manual `save_traces_json` | `Benchmark.run()` already persists | drop it; let `Benchmark` persist |
| Reaching into optimizer private state from an experiment | brittle, un-typed | add an additive API on the optimizer |

**Legitimate exceptions** (a custom *function*, still in `declivity/plotting`): a
secondary axis, a single-seed overlay, or richer-than-`RunTrace` per-run detail
are real gaps in the stock plotters. The fix is a small reusable plot + (if
needed) a small dataclass returned by your runner — not inline experiment code.
