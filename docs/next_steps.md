# Next steps

A roadmap derived from the 2026-05-26 honest-assessment review. Captures
what the framework still needs to fully deliver on its stated premise
(modular, simple to benchmark/write/experiment/replace parts of), in
priority order and with effort estimates.

For the rationale behind these items — what each one fixes and why it
matters — see the assessment that produced this list (in the chat
history that accompanied this commit).

---

## Status after milestone M001 (2026-05-27)

Milestone **M001 — Component modularization** closed pass on 2026-05-27,
landing on `main` via merge `1adf3e5`. See `M001-VALIDATION.md` and the
project-wide `REQUIREMENTS.md` (R001–R010 validated). M001 closed the
following items from this roadmap:

- [x] **#1 Delete dead code** — `repair_strategy.py` reborn as the new
      `RepairStrategy` ABC; `register_all_algorithms()` duplicate
      removed; `try/except ImportError` blocks replaced with bare
      imports + decorator registration.
- [x] **#3 Snake-case DES** — `ft`, `init_ft`, `path_length`, `c_ft`,
      `lamarckian`, `mu_eff`, `c_cum`, `path_ratio`, `ft_scale`, `diag_ft`
      all renamed; `optimize()` locals follow suit; vectorized
      duplicate-detection; fixed 1-based `hist_head`.
- [x] **#4 Auto-register via decorator** — `@register_optimizer` and
      `@register_logger` ship; touch-points for adding an algorithm
      dropped from 9 to ~6.
- [x] **#5 Wire `InitialPointGenerator`** — `InitialPointGenerator` now
      lives on `Problem` (single-point) and is paired with a separate
      `PopulationInitializer` ABC on `PopulationOptimizer` (population-
      level). Both ship discoverability `*Type` enums.
- [x] **#6 Constraint-handling subsystem** (PoC scope) — `ConstraintHandler`
      ABC with `BoxConstraintHandler` + `BoxStrategy` (CLAMP, BOUNCE_BACK);
      `ConstraintHandlerType` discoverability enum; `BoundaryHandlerType`
      removed (no backcompat alias — explicit instance injection only);
      `experiments/basic/constrained_rosenbrock.py` demos custom handler
      subclassing.

Still open from this list:

- [ ] **#2 Automated cross-validation tests** — deferred (R012).
- [ ] **#6 Full constraint subsystem** — inequality / composite handlers
      remain deferred (R011).
- [ ] **#7 Algorithm-internal modularity (Lego blocks)** — deferred
      (R013); thesis writeup still scopes the claim to "internal
      modularity at the component-ABC level," not full compositional
      algorithm construction.
- [ ] **#8 Formal test suite** — deferred (R012, R028 merged).

New follow-ups recorded during M001 (see `REQUIREMENTS.md`):

- [ ] **R014** — CMA-ES reference-port migration onto the new ABCs.
- [ ] **R015** — L-BFGS-B native inequality-constraint handling.

---

## Maturity scorecard

| Stated goal | Current grade | Limiting factor |
|---|---|---|
| Unified benchmarking | A | — |
| Modeling cross-algorithm experiments | A | — |
| Modular plotting | A | — |
| Simple to write experiments | A | — |
| Modern Python ports | B+ | DES still half-transliterated from R |
| Simple to add a new algorithm | C+ | 9 registration touch points |
| Simple to modify algorithm internals | C | Monolithic `optimize()` methods, no hook points |
| **Simple to replace constraint handling** | **D** | **`BoundaryHandler` is box-only; `RepairStrategy` is dead code** |

Strongest claims for thesis defense: **unified benchmarking + L-BFGS-B
pure-Python port + CMA-ES → L-BFGS-B handoff study**. These don't depend
on the rest of the framework being perfect.

---

## Quick wins (low effort, high payoff)

Pick these off in order. Cumulative ~8 hours of work, materially
improves the framework's "framework-ness".

### 1. Delete dead code (~30 min)

- [ ] Remove `src/utils/repair_strategy.py` — `RepairStrategy` is
      defined but **never imported anywhere**. Verified via
      `grep -rn "RepairStrategy" src/ experiments/`. DES inlines its
      own Lamarckism logic at `des_optimizer.py:249`.
- [ ] Remove the duplicate `register_all_algorithms()` in
      `src/algorithms/__init__.py` — it only registers DES and is
      shadowed by the comprehensive registration in `src/__init__.py`.
- [ ] Replace the four `try/except ImportError: pass` blocks in
      `src/__init__.py` with bare imports, or at least re-raise after
      logging. Bare-except hides real syntax/import errors until
      use-site, which is a debugging trap.

### 2. Automated cross-validation tests (~2 h)

- [ ] Add `tests/test_cross_validation.py` that runs DES on CEC2017
      F10 for 3 fixed seeds and asserts the convergence trace stays
      within tolerance of the saved R reference output in
      `reference/outputs/`. Same for the CMA-ES variants.
- [ ] Wire into `pdm test` or a Makefile target.

  Why: every refactor risks subtle drift in the optimizer numerics.
  The supervisor experiments catch catastrophic breaks but not 1e-4
  drift. A ~50-line test would gate every future change.

### 3. Snake-case DES (~3 h)

- [ ] Rename `DESConfig` fields:
      `Ft → ft`, `initFt → init_ft`, `c_Ft → c_ft`,
      `Lamarckism → lamarckian`, `pathLength → path_length`,
      `pathRatio → path_ratio`, `mueff → mu_eff`, `ccum → c_cum`,
      `Ft_scale → ft_scale`, `diag_Ft → diag_ft`.
- [ ] Update `des_optimizer.py` references.
- [ ] Update `experiments/` that touch these fields (likely just
      `simple_optimization.py`).
- [ ] Vectorize the `np.array_equal(population[i], …)` loop at
      `des_optimizer.py:244-246` (it's quadratic in dimension for no
      reason).
- [ ] Fix the 1-based `hist_head` indexing at `des_optimizer.py:127` —
      stop pretending it's R.

  Why: makes the "modern Python ports" thesis claim honest for DES.
  Right now CMA-ES delivers on it; DES reads like a stalled port.

### 4. Auto-register optimizers via decorator (~2 h)

- [ ] Replace the four `try/except` registration blocks in
      `src/__init__.py` with a `@register(AlgorithmChoice.X)` decorator
      on each optimizer class.
- [ ] Same for `LoggerFactory` — `@register_logger(AlgorithmChoice.X)`
      on each logger class.

  Drops the registration touch points from 9 to ~6 when adding a new
  algorithm, and removes a class of "I forgot to add it to the
  factory" bugs.

### 5. Wire or delete `InitialPointGenerator` (~30 min)

- [ ] Either delete `src/utils/initial_point_generator.py` (currently
      not used by `Benchmark.run` — `Problem.starting_point` is
      hardcoded to uniform), or
- [ ] Plug it into `Problem` so experiments can pick a strategy.

  Right now it's a public utility that's not part of the pipeline,
  which is misleading.

---

## Larger gaps

These are real work but directly address the stated premise.

### 6. Constraint-handling subsystem (~1–2 weeks) — **the headline gap**

Currently `BoundaryHandler` only supports box constraints (clamp /
bounce-back). For a framework whose premise includes "simple to replace
constraint handling," this is the weakest area.

Suggested shape:

```python
class ConstraintHandler(ABC):
    """Algorithm-agnostic handling of constraints during optimization."""

    @abstractmethod
    def is_feasible(self, x: NDArray) -> bool: ...

    @abstractmethod
    def feasibility_distance(self, x: NDArray) -> float:
        """0 if feasible; positive distance otherwise."""

    def repair(self, x: NDArray) -> NDArray:
        """Default: return x unchanged. Override for repair strategies."""
        return x

    def penalty(self, x: NDArray, f_x: float) -> float:
        """Default: no penalty. Override for penalty strategies."""
        return f_x
```

Concrete handlers to ship:

- [ ] `BoxConstraintHandler` (subsumes current `BoundaryHandler` —
      clamp and bounce-back become this with `repair()` overridden).
- [ ] `InequalityConstraintHandler` — accepts a list of
      `Callable[[NDArray], float]` (each `g_i(x) ≤ 0`) and a strategy
      enum: `{DEATH_PENALTY, STATIC_PENALTY(c), DYNAMIC_PENALTY,
      RESAMPLE, REPAIR_TO_NEAREST_FEASIBLE}`.
- [ ] `CompositeConstraintHandler` — chain box + inequality + custom.

Migration:
- [ ] Replace `BaseOptimizer.boundary_handler` with
      `BaseOptimizer.constraint_handler: ConstraintHandler`.
- [ ] Keep the `BoundaryHandlerType` enum as a backwards-compat alias
      that constructs a `BoxConstraintHandler`.
- [ ] Add a constrained benchmark to demo this:
      `experiments/basic/constrained_rosenbrock.py` showing the same
      problem solved with three different handler strategies.

Why: this is the single change that would most clearly upgrade the
framework's "framework-ness". A thesis reviewer asking "show me a
constrained problem" today has no good answer.

### 7. Algorithm-internal modularity (~2–4 weeks) — **probably out of scope**

The framework currently treats each optimizer as a black box with
one `optimize()` method. There's one good counter-example:
`LineSearchMethod` enum lets you swap line searches inside L-BFGS-B.

Generalizing this is genuinely research-scale work. Sketch:

```python
class CMAESOptimizer(BaseOptimizer):
    sampler: Sampler            # default: NormalSampler(mean, cov)
    recombinator: Recombinator  # default: WeightedRecombinator(weights)
    selector: Selector          # default: TopMuSelector(mu)
    sigma_adapter: SigmaAdapter # default: PathLengthControl
    ...
```

Users compose new variants by swapping components, not subclassing.
This is the "Lego-block optimizers" claim done properly.

**Recommendation: do not attempt for the thesis.** State this as future
work in the thesis writeup and explicitly scope the current claim to
"unified benchmarking" rather than "compositional algorithm
construction." Reviewers will respect the honest scoping more than an
overpromise.

### 8. Formal test suite (~1 day for the basics)

Currently zero tests. Documented honestly in CLAUDE.md but still a
weakness. Suggested minimum:

- [ ] `tests/test_smoke.py` — import every algorithm, run 50 evals on
      Sphere, assert convergence trace is monotone.
- [ ] `tests/test_benchmark_algorithm.py` — round-trip test for the
      `BenchmarkAlgorithm` / `HandoffAlgorithm` ABCs (subclass each,
      run, assert `RunTrace` shape).
- [ ] `tests/test_panel_registry.py` — every algorithm's panels
      resolve, no broken field names against the LogData.
- [ ] `tests/test_cross_validation.py` — covered in quick win #2.

About 200 lines of pytest. Catches ~90% of refactor regressions.

---

## Out of scope (acknowledge in the thesis writeup)

These would help the framework but aren't worth doing for the thesis
defense. Worth a "Future work" paragraph each.

- **CLI tooling.** Each experiment is a one-off `argparse` script.
  A unified `declivity bench ...` CLI would replace 11 ad-hoc parsers.
  Polish, not premise.
- **Pluggable benchmark functions.** A `BenchmarkFunctionRegistry`
  mirroring `PanelRegistry` for custom functions.
- **Multi-seed diagnostic plots.** Extending `RunTrace` with per-iter
  diagnostic fields so cross-seed plots of step-size, gradient norm,
  etc. work. Currently `plot_benchmark_*` only does convergence.
- **More algorithms.** xNES, sNES, IPOP-CMA-ES, BIPOP-CMA-ES, SLSQP.
  Each one would test the framework's extensibility, but you've
  already proven that with the existing four.

---

## Honest framing for the thesis writeup

If a reviewer asks "is this a *framework*, or four optimizers in a
shared directory?":

- **Framework, for benchmarking and plotting.** The `Benchmark` /
  `Problem` / `RunTrace` triple, the `BenchmarkAlgorithm` ABC hierarchy,
  the declarative panel system — these are reusable infrastructure that
  works across all four current algorithms *and* a hypothetical fifth
  one. The `custom_handoff.py` and `custom_algorithm.py` examples prove
  a third-party can plug in without modifying the framework.

- **Library of four optimizers, for the algorithms themselves.** The
  optimizers are well-documented monolithic classes. They share a
  Factory and a typed contract but not internal *mechanisms*. You
  can't swap CMA-ES's sampler from outside without forking the class.

The L-BFGS-B port + CMA-ES → L-BFGS-B handoff are genuine research
contributions that don't depend on the framework's extensibility claim.
The thesis can stand on those even if the framework claims get softened.

---

## Priorities if you only have 1 week

1. Constraint-handling subsystem (#6) — closes the biggest premise gap.
2. Snake-case DES + cross-validation tests (#3 + #2) — makes the
   "modern Python ports" claim honest and locks in regression protection.
3. Delete dead code + auto-register (#1 + #4) — polish that pays back
   in every future change.

Defer everything else, scope the thesis claims accordingly.
