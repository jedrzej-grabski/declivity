# NEW CODE — Interleaved CMA-ES ⇆ L-BFGS-B handoff

A new experiment and the framework pieces it needed. This extends the
one-shot `CMAESLBFGSBHandoff` into a scheme that *alternates* between the
global searcher and the local refiner for the whole budget, producing the
characteristic "staircase" convergence curve.

This document is the summary the supervisor asked for. For the broader
design rationale see [`framework_design.md`](framework_design.md); for the
covariance-transform background see
[`covariance_handoff_when_it_matters.md`](covariance_handoff_when_it_matters.md).

---

## 1. The idea

The existing handoff runs CMA-ES, cuts over **once** to L-BFGS-B, and stops.
The interleaved scheme keeps cycling:

1. **Advance CMA-ES** for `cmaes_interval` generations (the handoff interval
   *N*).
2. **Fire an L-BFGS-B side-probe** from the current CMA-ES mean, with its
   initial Hessian `B₀ = C⁻¹` derived from the CMA-ES covariance (covariance
   information only — the same transform the one-shot handoff uses by
   default). The probe runs until it *stops advancing rapidly* — the
   L-BFGS-B `factr` relative-decrease test — capped by `probe_max_evals`.
3. **Fold the probe's improvement into the tracked OVERALL BEST**, then
   return to step 1 with **CMA-ES untouched**. The probe is a pure
   refinement of the running best; it never feeds back into the CMA-ES
   distribution.

Tracking the overall best across both algorithms gives the staircase: a
gently-descending CMA-ES *backbone* with sharp L-BFGS-B *drops*, each one
deeper as CMA-ES's covariance becomes a better Hessian model.

### Why CMA-ES is a pure "side-probe" target (not re-seeded)

This was a deliberate design choice (confirmed with the supervisor). The
probe does **not** move CMA-ES's mean or reset its covariance. Consequences:

- The CMA-ES *backbone* is identical to a standalone CMA-ES run with the
  same seed — it is a true, un-perturbed reference curve. (See §3 on the
  byte-identical resume.)
- The method is *robust*: CMA-ES keeps exploring globally regardless of
  where the probes wander, so a probe that descends into a bad local basin
  costs only its own (capped) budget and never derails the global search.
- The two alternatives (memetic re-seeding of the mean; full restart) were
  considered and left as future knobs — they would collapse the backbone
  onto the staircase and change the algorithm's character.

---

## 2. What was added

| File | Addition |
|---|---|
| `src/utils/benchmark_functions.py` | **`ShiftedFunction`** — translation wrapper (the counterpart of `RotatedFunction`), with a `near_corner(...)` constructor. |
| `src/algorithms/cmaes/cmaes_optimizer.py` | **`CMAESState`** dataclass + `initial_state=` ctor param + `get_state()` — explicit pause/resume for CMA-ES. |
| `src/algorithms/cmaes/__init__.py` | Export `CMAESOptimizer`, `CMAESState`. |
| `src/benchmarking/algorithm_run.py` | **`InterleavedCMAESLBFGSB`** (the scheme) + **`InterleaveResult`** (detailed record) + module-level **`initial_hessian_from_cmaes`** helper (de-duplicated out of `CMAESLBFGSBHandoff`). |
| `src/benchmarking/__init__.py` | Export the three new symbols. |
| `src/plotting/interleaved.py` | **`plot_interleaved_convergence`** — the staircase figure. |
| `src/plotting/__init__.py` | Export `plot_interleaved_convergence`. |
| `experiments/handoff/interleaved.py` | The runnable experiment. |

### `ShiftedFunction` — move the optimum anywhere

```python
f(x) = f_base(x - shift)            # gradient and global_minimum chain through
```

Mirrors `RotatedFunction` (which does `f_base(R x)`); the two **compose**, so
`ShiftedFunction(RotatedFunction(Ellipsoid(d)), shift)` is a rotated,
off-centre ellipsoid. Bounds are inherited from the base in *x-space* (the
feasible box does **not** move with the optimum). The `near_corner(base,
fraction)` constructor drops the optimum a `fraction` of the way from the box
centre toward a corner — `fraction=0.9` puts it near a corner, `1.0` exactly
on it (fully bound-active). This is what makes the test problem genuinely
bound-constrained, so L-BFGS-B's projected-gradient / Cauchy-point machinery
has work to do — the classic "optimum in the corner of the feasible region"
benchmark.

### `CMAESState` — explicit, reusable resume

`get_state()` snapshots the full evolvable state (mean, σ, covariance, both
evolution paths, generation counter, function-value history, **and** the
cached eigendecomposition). `CMAESOptimizer(..., initial_state=state)`
restores it. The interleaved scheme creates a fresh optimizer each cycle,
resumed from the previous cycle's snapshot, sharing one RNG. This is the
clean, first-class version of "pause CMA-ES, do something, continue" — no
reliance on implementation details of `optimize()`.

### `InterleavedCMAESLBFGSB` — the scheme

Implemented directly on `BenchmarkAlgorithm` (not `HandoffAlgorithm`, which
is strictly two-phase) — exactly the multi-phase case `framework_design.md`
defers to a direct `run()`. `run()` returns a standard `RunTrace` (the
overall-best staircase) so it drops straight into `Benchmark` /
`plot_benchmark_convergence`; `run_with_detail()` additionally returns the
`InterleaveResult` (backbone, per-burst segments, burst start positions) for
the staircase plot.

Key parameters: `cmaes_interval` (N, generations between probes),
`total_budget`, `transform` (default `INVERSE` = C⁻¹), `probe_factr` /
`probe_pgtol` / `probe_max_evals` (the burst stop), `lbfgsb_line_search`.

---

## 3. Correctness: the resume is byte-identical

Because the probe never touches CMA-ES, and each CMA-ES slice is resumed from
a complete `CMAESState` while sharing one RNG, the interleaved run's CMA-ES
backbone reproduces a standalone CMA-ES run **bit-for-bit**:

```
continuous best: 70780.25069402353
sliced     best: 70780.25069402353   (3 × 2-generation slices, shared RNG)
EXACT best match: True   EXACT mean match: True   EXACT C match: True
```

Caching the eigendecomposition `(B, D)` inside `CMAESState` was necessary for
this: re-running `eigh` on a reconstructed `C` returns eigenvectors that
differ in the last bits, which otherwise drifted the run by ~1e-10. With the
cache restored, the match is exact. This makes the backbone a trustworthy
reference line on every staircase plot.

`ShiftedFunction`'s analytic gradient matches finite differences to ~5e-11.

---

## 4. Test problem and results

**Default problem:** `ShiftedFunction.near_corner(RotatedEllipsoid(10,
"random"), fraction=0.9)` — a randomly-rotated, ill-conditioned (cond 10⁶)
ellipsoid whose minimum sits near a corner of the `[-100, 100]¹⁰` box.

**Settings:** total budget 8000, `cmaes_interval = 20` generations,
`probe_max_evals = 80`, `transform = C⁻¹`.

### Multi-seed medians (15 seeds)

| Problem | CMA-ES | L-BFGS-B | One-shot (C⁻¹) | **Interleaved** |
|---|---:|---:|---:|---:|
| Shifted-rotated Ellipsoid (corner) | 6.8e+04 | **4.2e-18** | 9.7e-18 | 1.0e+00 |
| Shifted Rastrigin (corner)         | 1.9e+01 | 2.6e+02 | 1.9e+01 | **1.7e+01** |

- **Convex ellipsoid:** the quasi-Newton methods (L-BFGS-B, one-shot) win, as
  they should on a smooth quadratic. The interleaved scheme is far better
  than CMA-ES alone and individual seeds reach machine precision (best
  1.4e-16), but its median lags because — by design — it never lets a single
  burst run to full convergence; it trades peak speed for robustness.
- **Multimodal Rastrigin:** L-BFGS-B alone collapses into the nearest local
  minimum after ~22 evaluations (median 262). The interleaved scheme is the
  **best** of the four — its periodic probes opportunistically find better
  local minima than CMA-ES or a single handoff, while CMA-ES keeps it from
  getting stuck.

### The staircase (headline figure)

`plots/handoff/interleaved/ellipsoid/staircase.png` (seed 0) shows the three
curves cleanly separated: the **green** CMA-ES backbone descending gently and
staying above; the **orange** overall-best staircase; the **red** L-BFGS-B
burst drops. It reaches 1.9e-16, matched against the **gray** standalone-CMA-ES
reference. This is the curve the experiment was designed to produce.

---

## 5. Key empirical finding — short, frequent bursts win

The per-burst cap is the decisive hyperparameter, and it is *not* "bigger is
better":

| `cmaes_interval` | `probe_max_evals` | bursts | final f | sharp drops |
|---:|---:|---:|---:|---:|
| 20 | **80**  | 29 | **1.9e-16** | ~16 |
| 20 | 250 | 18 | 2.6e-06 | ~2 |
| 10 | 60  | 51 | 6.4e-13 | ~14 |
| 10 | 250 | 22 | 2.7e-01 | ~3 |

A **large** cap lets the first burst spend its whole budget descending with a
*stale, isotropic* early covariance, stall at ~1e-3, and then the run
plateaus. **Short, frequent** bursts instead ride CMA-ES's *continuously
improving* `C⁻¹`: each one makes incremental progress with a fresher Hessian
model, producing both the clean multi-step staircase and orders-of-magnitude
better final fitness. The lesson: in a side-probe scheme, refresh the
covariance model often rather than over-committing to one probe.

---

## 6. How to run

```bash
# Headline staircase + multi-seed comparison on the near-corner ellipsoid
PYTHONPATH=. pdm run python experiments/handoff/interleaved.py

# Multimodal variant
PYTHONPATH=. pdm run python experiments/handoff/interleaved.py --problem rastrigin

# Tunables
PYTHONPATH=. pdm run python experiments/handoff/interleaved.py \
    --dimensions 20 --num-seeds 25 --cmaes-interval 15 --probe-max-evals 60 \
    --corner-fraction 1.0          # optimum exactly on the corner
```

Outputs land in `plots/handoff/interleaved/<problem>/`:
`staircase.png` (single-run dissection), `convergence.png` (multi-seed median
+ IQR), `final_fitness.png` (boxplot), plus the usual `traces.json` /
`runs.csv` / `summary.csv`.

---

## 7. CMABFGS comparison — our CMA-ES ⇆ L-BFGS-B variant vs Maksym's CMA-ES ⇆ BFGS

`experiments/handoff/cmabfgs_replication.py` reproduces the **CMABFGS** figure
from **Maksym's thesis** (`notes/test_bfgs.png`) — a CMA-ES ⇆ **BFGS** study —
with our interleaved scheme as a **variant that swaps BFGS for L-BFGS-B**
(CMA-ES ⇆ L-BFGS-B), so the two can be set side by side. Figure locations:

- **Reference:** `notes/test_bfgs.png` — Maksym's figure (local; `notes/` is
  gitignored).
- **Ours:** `plots/handoff/cmabfgs_replication/cmabfgs_pop4d.png` and
  `…/cmabfgs_popdefault.png` (gitignored — regenerate with the run command at
  the end of this section).

Setup, matched to the reference figure:

- **Problem:** `f = SDP` = Shifted Different Powers, added as
  `DifferentPowers` (`f(x) = Σ |x_i|^(2 + 4 i/(d-1))`, exponents 2→6, analytic
  gradient) shifted onto a corner of the `[-180, 20]^d` box; `d = 100`,
  budget `1e6`.
- **Sweep:** handoff interval `N = k·d` for `k ∈ {0.5, 1, 2, 4, 8}`, plus
  standalone CMA-ES. Standalone BFGS (L-BFGS-B) is **omitted by default**:
  on this smooth bowl it converges in ~12k evals and visually dwarfs the
  CMA-ES-vs-CMABFGS comparison the figure is about (`--show-bfgs` adds it back).
- **Plot:** best fitness (log) vs evaluations, with the reference's secondary
  "Iteracje CMA-ES" axis (`iterations = evals/(λ+1)`), matching colours/legend.
- **Population:** run with both `λ = 4·d = 400` (matches the reference's
  ~2500-iteration axis) and the framework default (`λ ≈ 17`).

Outputs: `plots/handoff/cmabfgs_replication/cmabfgs_pop4d.png` and
`…/cmabfgs_popdefault.png`.

### Two corner subtleties (both worth knowing)

1. **Exact-corner optima trivialise a *bounded* solver.** If the optimum sits
   exactly on the corner (`--corner-fraction 1.0`), L-BFGS-B solves it in a
   single step — the first Cauchy point projects every coordinate onto the
   active bound, which *is* the optimum. The reference's *unbounded* BFGS has
   no such shortcut. To reproduce the reference's shape we place the optimum
   slightly interior (`fraction 0.9`), forcing a real ill-conditioned descent.
2. **Burst stop vs. burst depth.** With the interleaved default
   ("stops advancing rapidly", `factr=1e7`/`pgtol=1e-8`) the bursts halt on
   the very flat `|x|⁶` valley floor at ~1e-7. To match the reference (BFGS
   run to convergence each handoff) the replication runs bursts to full
   accuracy (`--probe-factr 10 --probe-pgtol 1e-12`, generous cap), so the
   drops complete to the floor and stay.

### Result (λ = 4·d = 400)

The shape replicates: slow smooth CMA-ES descent (converges at ~6.9e5 evals
≈ 1730 iterations, matching the reference's ~0.68e6) and the CMABFGS variants
tracking CMA-ES then dropping in **k-order** (k=0.5 first at ~2e4 evals, larger
k progressively later, each to the ~1e-11..1e-9 floor).

The headline difference from the reference — and the answer to "compare
CMA-ES⇆L-BFGS-B with CMA-ES⇆BFGS" — is **timing**: our CMA-ES⇆L-BFGS-B drops
*far earlier* than the reference's CMA-ES⇆BFGS (which only plunges once CMA-ES
is nearly converged). The bounded, covariance-warm-started (`C⁻¹`) L-BFGS-B
refines effectively from an *early, rough* CMA-ES state, so even the first
handoff already reaches the basin floor. The smaller `k` (more frequent
handoffs) reach the floor soonest. (Standalone L-BFGS-B is so dominant here —
~12k evals to full accuracy — that it is left off the figure; the comparison
of interest is CMA-ES vs the CMABFGS variants.)

Each run also writes `traces_<tag>.json`; re-render the figure (e.g. to toggle
BFGS) without re-running via `--replot-from`.

Run::

    PYTHONPATH=. pdm run python experiments/handoff/cmabfgs_replication.py --popsize 400 --tag pop4d
    PYTHONPATH=. pdm run python experiments/handoff/cmabfgs_replication.py --popsize 0   --tag popdefault
    # re-render from saved traces (instant), e.g. add the BFGS baseline back:
    PYTHONPATH=. pdm run python experiments/handoff/cmabfgs_replication.py \
        --replot-from plots/handoff/cmabfgs_replication/traces_pop4d.json --show-bfgs --tag pop4d
