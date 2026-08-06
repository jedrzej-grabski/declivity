# CEC benchmark provider: opfunu to cecxx

The CEC benchmark suite is now evaluated by [`cecpy`][cecpy], the Python
binding of Eryk Warchulski's `cecxx`, instead of `opfunu`. This is a
correctness fix, not a swap of equivalent providers: the two disagree on
almost the whole CEC2017 suite, and the disagreement is opfunu's.

Anything derived from opfunu CEC values has to be regenerated. The
committed baselines under `reference/outputs/` already have been.

[cecpy]: https://pypi.org/project/cecpy/

## What changed numerically

CEC2017, d=10. Of the 29 functions opfunu defines, **only F1 agrees** with
cecxx; the other 28 differ materially, and opfunu has no F30 at all.
Sampled at `x = 50·1`, `x = 0`, `x = linspace(-30, 30, 10)` and `x = -80·1`,
with agreement taken as a 1e-6 relative match on every point.

At `x = 50·1`:

| Function | opfunu       | cecxx        |
|----------|--------------|--------------|
| F3       | 3.93407e+04  | 3.95368e+10  |
| F5       | 5.00080e+02  | 8.00666e+02  |
| F10      | 2.96084e+09  | 6.26853e+03  |

F10 is the load-bearing one — it is the function `experiments/cross_validation/des_vs_r.py`
cross-checks against the R reference.

## Why cecxx is the correct one

Not a data mismatch. Both providers agree on where the optimum is and what
it is worth: feeding cecxx **opfunu's own** `f_shift` vector returns exactly
the nominal bias, to ten decimal places.

| Function | cecxx at opfunu's shift vector | nominal bias `100·i` |
|----------|--------------------------------|----------------------|
| F3       | 300.0000000000                 | 300                  |
| F5       | 500.0000000000                 | 500                  |
| F10      | 1000.0000000000                | 1000                 |

So the shift and rotation tables are shared; what differs is the
transformation applied to them. Reconstructing the canonical CEC2017
composition by hand settles which is right:

```
z = M @ ((x - o) * sh_rate)
f = f_base(z) + 100 * i
```

using opfunu's own `f_shift` and `f_matrix` as `o` and `M`:

| Case            | hand-computed | cecxx        | opfunu       |
|-----------------|---------------|--------------|--------------|
| F3, `x = 50·1`  | 3.953676906e10| 3.953676906e10 | 3.934063776e04 |
| F3, `x = 0`     | 1343217.04    | 1343217.04   | 2687.159916  |
| F3, `x = -30..30` | 8210134.023 | 8210134.023  | 3581.620793  |
| F5, `x = 50·1`  | 800.6659851   | 800.6659851  | 500.0796654  |
| F5, `x = 0`     | 726.7145613   | 726.7145613  | 500.0518371  |
| F5, `x = -30..30` | 757.0750476 | 757.0750476  | 500.2350002  |

F3 is Zakharov with `sh_rate = 1`, F5 is Rastrigin with `sh_rate = 5.12/100`.
The hand-computed column matches cecxx to 1e-9 relative in all six cases and
misses opfunu by up to six orders of magnitude. Since the reconstruction
uses opfunu's *own* tables, opfunu has the right data and the wrong
transformation.

An independent sanity check on F5: opfunu returns 500.08 at `x = 50·1`, i.e.
a Rastrigin value of 0.08 at a point roughly 100 units from the optimum in
every coordinate, which that function cannot produce.

## Why this strengthens the R cross-validation

`cecxx` ships Python and R bindings over one C++ core (`src/cpp11.cpp` in
the same sdist is the R binding). The R reference DES implementation this
project cross-checks against was run against that core, so both sides of
`des_vs_r.py` now evaluate the same compiled functions rather than two
independent reimplementations. The comparison got tighter, not looser.

## Other behaviour changes

- **Supported dimensions narrowed.** cecxx CEC2017 accepts d ∈ {10, 30, 50,
  100}; opfunu also accepted {2, 20}. Every CEC spec in this repository uses
  d=10, and `CECProblem.__init__` validates up front. CEC2013 accepts
  {2, 5, 10, 20, ..., 100}.
- **Function numbering is 1-based** in both, and the additive bias is `100·i`
  in both. `benchmark_problem_num(CEC2017)` is 30.
- **Shift and rotation tables are bundled** in the wheel (701 data files
  under `cecpy/data`); nothing is downloaded at runtime and no state
  directory is needed.
- **Roughly an order of magnitude faster** — the C++ core evaluates ~870k
  points/s against ~71k for opfunu on this machine.
- **No wheels on PyPI.** `cecpy` 0.1.7 publishes an sdist only, so
  `uv sync` compiles it. That needs a compiler with full C++23 library
  support; cmake and ninja come in via scikit-build-core.
- **Dropped the setuptools cap.** It existed because opfunu imports
  `pkg_resources` without declaring setuptools. Both are gone.
- One cecxx quirk, noted for completeness: F9 returns 901.44 at its shift
  vector rather than the nominal 900. This is the known CEC2017 Levy
  ambiguity. Only F3, F5 and F10 are used here, so it does not bite.

## Reproducing the comparison

`opfunu` is no longer a dependency. To re-derive the table above, install
both providers into a throwaway environment:

```bash
uv venv /tmp/ceccmp --python 3.12
VIRTUAL_ENV=/tmp/ceccmp uv pip install "opfunu>=1.0.1" "setuptools<82" numpy cecpy
```

then evaluate `opfunu.cec_based.F<n>2017(ndim=10).evaluate(x)` against
`cecpy.benchmark.CECEvaluator(CECEdition.CEC2017, [10])(n, x.reshape(-1, 1))`.
