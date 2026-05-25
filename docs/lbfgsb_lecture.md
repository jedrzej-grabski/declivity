# L-BFGS-B: Algorithm Theory and Implementation Guide

*A complete walkthrough of the Limited-memory Broyden-Fletcher-Goldfarb-Shanno
algorithm for Bound-constrained optimization, and its pure Python implementation
in the declivity framework.*

---

## Table of Contents

1. [Where L-BFGS-B comes from](#1-where-l-bfgs-b-comes-from)
2. [The big picture: what does one iteration look like?](#2-the-big-picture)
3. [The compact L-BFGS Hessian approximation](#3-the-compact-l-bfgs-hessian-approximation)
4. [The Generalized Cauchy Point (GCP)](#4-the-generalized-cauchy-point)
5. [Subspace minimization](#5-subspace-minimization)
6. [The line search](#6-the-line-search)
7. [Convergence criteria](#7-convergence-criteria)
8. [Configuration parameters](#8-configuration-parameters)
9. [Implementation walkthrough](#9-implementation-walkthrough)
10. [Diagnostics and plotting](#10-diagnostics-and-plotting)
11. [How it compares to evolutionary algorithms](#11-how-it-compares-to-evolutionary-algorithms)

---

## 1. Where L-BFGS-B comes from

### The family tree

To understand L-BFGS-B, we trace through a line of optimization ideas:

**Newton's method** finds the minimizer of a quadratic model of `f` at each
step. The quadratic model is:

    Q(x + d) = f(x) + g'd + (1/2) d' H d

where `g = nabla f(x)` is the gradient and `H = nabla^2 f(x)` is the Hessian
(matrix of second derivatives). The Newton direction is `d = -H^{-1} g`. This
converges quadratically near a minimum, but requires computing and inverting
the full `n x n` Hessian --- which is `O(n^2)` storage and `O(n^3)` per solve.

**Quasi-Newton methods** (BFGS, DFP) avoid computing the true Hessian. Instead,
they build an approximation `B_k approx H` from the observed pairs:

    s_k = x_{k+1} - x_k          (the step taken)
    y_k = g_{k+1} - g_k          (how the gradient changed)

The secant condition `B_{k+1} s_k = y_k` constrains the update. BFGS chooses
the rank-2 update closest to `B_k` (in a weighted Frobenius norm) that satisfies
the secant condition. This gives superlinear convergence without second
derivatives --- but still requires `O(n^2)` storage for the dense `B` matrix.

**L-BFGS** (Limited-memory BFGS) drops the explicit matrix entirely. Instead of
storing `B`, it stores only the `m` most recent `(s, y)` pairs and reconstructs
matrix-vector products `B*v` on the fly via the compact representation. Storage
drops from `O(n^2)` to `O(mn)`. For `m = 10` and `n = 10000`, that is 100x
less memory.

**L-BFGS-B** extends L-BFGS to handle **bound constraints** `l <= x <= u`.
This was published by Byrd, Lu, Nocedal, and Zhu in 1995 (SIAM J. Sci.
Computing), with corrections by Morales and Nocedal in 2011. The "B" stands
for "Bounds". The original implementation was 3,950 lines of Fortran 77 with
BLAS/LINPACK dependencies. Our implementation is a faithful reimplementation
in pure Python/NumPy.

### The two papers

**Byrd et al. 1995** introduced the algorithm: the compact representation, the
generalized Cauchy point, and the subspace minimization strategy.

**Morales & Nocedal 2011** (Algorithm 778 remark) fixed a subtle bug in the
subspace minimization step. The original code would sometimes compute a Newton
direction that, when projected onto bounds, was no longer a descent direction.
The v3.0 fix adds a **project-then-check** safeguard: project the Newton step
onto bounds, check if `(x_projected - x_current) . g < 0`, and fall back to
a backtracking strategy if not. Our implementation includes this v3.0 fix.

---

## 2. The big picture

Each iteration of L-BFGS-B performs these steps:

```
1.  CAUCHY POINT      Walk along the projected steepest-descent path until
                      the first local minimizer of the quadratic model.
                      This determines which variables are "active" (pinned
                      at a bound) and which are "free".

2.  SUBSPACE MIN      In the space of free variables, solve a reduced Newton
                      system using the compact L-BFGS Hessian. This refines
                      the Cauchy point using second-order information.

3.  LINE SEARCH       Starting from x, search along d = z - x (where z is
                      the result of steps 1-2) for a step length alpha
                      satisfying the Wolfe conditions.

4.  UPDATE            Accept x_new = x + alpha*d. Compute the new gradient.
                      Store the new (s, y) correction pair.
                      Update the compact Hessian representation.

5.  CONVERGENCE       Check if ||projected_gradient||_inf <= pgtol
                      or if the relative function decrease is below factr*eps.
```

The Cauchy point handles the bound constraints. The subspace minimization
exploits curvature. The line search ensures global convergence. Together, they
give a method that is both globally convergent and locally superlinear.

---

## 3. The compact L-BFGS Hessian approximation

### The core formula

The Hessian approximation is stored in compact form:

    B_k = theta_k * I  -  W_k * M_k * W_k'

where:

- `theta_k = y_{k-1}' y_{k-1} / (y_{k-1}' s_{k-1})` --- the Barzilai-Borwein
  scaling. This makes `B_0 = theta * I` an approximation of the Hessian's
  overall scale. It's the ratio of how much the gradient changed squared to
  how much curvature we observed along the step.

- `W_k = [Y_k | theta_k * S_k]` --- an `n x 2m` matrix formed by stacking
  the gradient-difference vectors `Y` and the scaled step vectors `theta*S`.

- `M_k` is a `2m x 2m` middle matrix whose inverse has a known block structure:

      M^{-1} = [ -D    L' ]
               [  L    theta * S'S ]

  where `D = diag(s_i' y_i)`, `L` = strict lower triangle of `S'Y`, and
  `S'S` is the Gram matrix of the step vectors.

### In our code (`lbfgsb_optimizer.py`)

The correction pairs are stored as Python lists:

```python
self._S: list[NDArray]   # step vectors s = x_{k+1} - x_k
self._Y: list[NDArray]   # gradient diffs y = g_{k+1} - g_k
self._theta: float       # scaling factor
self._col: int           # number of stored pairs (up to m)
```

After each update, we rebuild the inner-product matrices:

```python
self._SY = S_mat.T @ Y_mat   # S'Y (col x col)
self._SS = S_mat.T @ S_mat   # S'S (col x col)
```

This is `O(m^2 * n)` but `m` is typically 3-20, so this is fast.

### The T matrix and its Cholesky factor

The algorithm frequently needs to solve systems involving the middle matrix `M`.
This is done via `_bmv()` (B-matrix-vector multiply), which requires the
Cholesky factorization of:

    T = theta * S'S + L * D^{-1} * L'

This is formed in `_form_t()` and stored as `self._wt_factor` (a scipy
`cho_factor` tuple). The bmv operation then reduces to block elimination:

1. Solve `T * p2 = v2 + L * (v1 / D)` using the Cholesky factor
2. Compute `p1 = (L' * p2 - v1) / D`

This gives `p = M * v` by solving `M^{-1} * p = v`.

### When things go wrong numerically

If `T` fails to be positive definite (e.g., due to floating-point issues near
convergence), the algorithm **resets** the L-BFGS memory:

```python
self._S.clear()
self._Y.clear()
self._col = 0
self._theta = 1.0
```

This is equivalent to restarting with `B = I` (steepest descent). The Fortran
code does the same thing. It's a safe fallback that allows the algorithm to
recover and continue.

---

## 4. The Generalized Cauchy Point

### The idea

Consider the path you'd follow if you did projected gradient descent:

    x(t) = P(x - t*g,  l, u)

where `P` is projection onto the box `[l, u]` (component-wise clamping).
As `t` increases from 0, each variable `x_i` either moves freely (in the
direction `-g_i`) or is clamped at its bound `l_i` or `u_i`. Each time a
variable hits its bound, the path "bends" --- the direction changes because
that variable is now fixed.

The **breakpoint** for variable `i` is the value of `t` where `x_i` hits its
bound:

    If -g_i < 0 (moving toward lower bound):  t_i = (x_i - l_i) / g_i
    If -g_i > 0 (moving toward upper bound):  t_i = (u_i - x_i) / (-g_i)

The Generalized Cauchy Point (GCP) is the **first local minimizer** of the
quadratic model `Q(x(t))` along this piecewise-linear path.

### The algorithm

The GCP computation walks through the breakpoints in order:

1. **Classify variables**: For each variable, determine if it's free (will
   move), fixed at a bound (gradient pointing into the bound), or permanently
   fixed (l_i = u_i).

2. **Sort breakpoints**: Order the breakpoints by increasing `t`.

3. **Walk segments**: On each segment between breakpoints, the quadratic
   model is `Q(t) = const + f1*dt + (1/2)*f2*dt^2` where `f1` and `f2`
   are the first and second derivatives. The unconstrained minimizer is at
   `dt = -f1/f2`. If this falls within the current segment, stop. Otherwise,
   advance to the next breakpoint and update `f1`, `f2`.

### The derivative updates at each breakpoint

When variable `i` hits its bound, we need to update `f1` and `f2` to account
for removing that variable from the active search direction. With `d_i` being
the direction component and `z_b = bound_i - x_i`:

    f1 = f1 + dt*f2 + d_i^2 - theta*d_i*z_b    (advance + remove variable)
    f2 = f2 - theta*d_i^2                        (remove from curvature)

If we have L-BFGS correction pairs (`col > 0`), there are additional
correction terms involving the middle matrix `M`:

    f1 += d_i * (c . M*w_i)
    f2 += 2*d_i*(w_i . M*p) - d_i^2*(w_i . M*w_i)

where `w_i` is row `i` of the `W` matrix, `p = W'd` is updated as variables
are fixed, and `c = W'(x(t) - x)` accumulates the displacement.

### In our code

`_cauchy_point()` at `lbfgsb_optimizer.py:297-484` implements this. The four
phases are clearly marked:

- **Phase 1** (lines 318-358): Classify variables into the `iwhere` array and
  compute breakpoints.
- **Phase 2** (lines 361-367): Compute `p = W'd`, the initial projection of
  the search direction onto the L-BFGS basis.
- **Phase 3** (lines 369-381): Initialize the quadratic model derivatives `f1`
  and `f2`.
- **Phase 4** (lines 383-483): Walk along breakpoints, update derivatives,
  stop at the first minimizer.

### The variable classification (`iwhere`)

| Value | Meaning |
|-------|---------|
| `3`   | Permanently fixed (`l_i = u_i`) |
| `2`   | Fixed at upper bound (gradient pushes inward) |
| `1`   | Fixed at lower bound (gradient pushes inward) |
| `0`   | Free variable (moving, has a breakpoint) |
| `-3`  | Free variable with zero gradient (stationary) |

After the GCP is computed, variables with `iwhere <= 0` are "free" and
variables with `iwhere > 0` are "active" (at their bounds).

---

## 5. Subspace minimization

### The idea

The Cauchy point is a first-order solution --- it only uses gradient information
to determine the active set. Once we know which variables are free (the "Z"
subspace), we can do a second-order Newton step within that subspace.

We want to solve:

    min_d  Q(xcp + Z*d)  =  r'd + (1/2) d' (Z'BZ) d

where `Z` projects onto the free variables, `r = -Z'(g + B(xcp - x))` is the
reduced gradient, and `Z'BZ` is the reduced Hessian.

### The Woodbury formula trick

With `B = theta*I - W*M*W'`, the reduced Hessian is:

    Z'BZ = theta*I - A*M*A'     where A = Z'W  (nfree x 2*col)

The Newton direction is `(Z'BZ)^{-1} r`. Using the Woodbury matrix identity:

    (Z'BZ)^{-1} = (1/theta)*I + (1/theta^2) * A * K^{-1} * A'

where:

    K = M^{-1} - (1/theta) * A'A

This `K` is a `2*col x 2*col` matrix (at most 40x40 for `m=20`), so solving
`K*v = w` is negligible. The Fortran code uses a specialized `LEL^T`
factorization; we simply use `np.linalg.solve` since the system is tiny.

### The v3.0 improvement

After computing the Newton direction and lifting it to full space, we
**project onto bounds** and then check descent:

```python
z_proj = np.clip(z, self.lower_bounds, self.upper_bounds)
dd = z_proj - x
gd = float(np.dot(g, dd))

if gd <= 0:
    return z_proj      # Good: projected step is still descent
else:
    # Bad: projection ruined descent. Fall back to backtracking.
    ...
```

This was the key fix in Morales & Nocedal 2011. Without it, the algorithm
could accept non-descent directions after projection, causing failure.

### In our code

`_subspace_minimization()` at `lbfgsb_optimizer.py:505-604`. The key steps:

1. **Reduced gradient** (lines 527-538): Compute `r = -Z'(g + B(xcp - x))`.
2. **Form A = Z'W** (lines 540-546): Restrict W to free variables.
3. **Form K and solve** (lines 548-576): Build `K = M^{-1} - (1/theta)*A'A`,
   solve `K*v = A'*r`, compute `d = (1/theta)*r + (1/theta^2)*A*v`.
4. **v3.0 safeguard** (lines 583-604): Project, check descent, backtrack if
   needed.

---

## 6. The line search

### Why a line search?

The Cauchy point + subspace minimization gives us a direction `d = z - x`.
But `Q(x + d)` is only a quadratic **model** --- it may not accurately reflect
`f(x + d)`. A line search finds a step length `alpha` such that `x + alpha*d`
gives sufficient actual decrease in `f`, ensuring global convergence.

### The strong Wolfe conditions

The More-Thuente line search finds `alpha` satisfying:

    f(x + alpha*d)  <=  f(x) + ftol * alpha * g'd     (sufficient decrease)
    |g(x + alpha*d) . d|  <=  gtol * |g(x) . d|       (curvature condition)

The first condition (Armijo) ensures the function actually decreased enough.
The second condition ensures we don't stop too early --- the slope should have
flattened out, indicating we're near a minimizer along the ray.

### More-Thuente algorithm (`line_search.py:82-206`)

This is a faithful port of the Fortran `dcsrch`/`dcstep` subroutines. The
algorithm maintains a **bracket** --- an interval `[stx, sty]` known to
contain a point satisfying the Wolfe conditions.

**Two stages:**
- **Stage 1**: Uses a modified function `psi(alpha) = f(alpha) - f(0) -
  ftol*alpha*f'(0)`. Transitions to stage 2 when `psi <= 0` and `f' >= 0`.
- **Stage 2**: Works with `f` directly to find the minimizer.

**The `_dcstep` function** (`line_search.py:209-356`) computes each trial step
using safeguarded cubic/quadratic interpolation. It handles four cases:

| Case | Condition | Meaning |
|------|-----------|---------|
| 1 | `f_trial > f_best` | Higher value: minimum is bracketed |
| 2 | `f_trial < f_best`, opposite derivatives | Minimum is bracketed (sign change) |
| 3 | `f_trial < f_best`, same sign, `|f'_trial| < |f'_best|` | Making progress |
| 4 | `f_trial < f_best`, same sign, `|f'_trial| >= |f'_best|` | Derivative not improving |

Each case uses cubic interpolation `theta, gamma` (from 3 data points) and/or
quadratic interpolation (secant step) to propose the next trial point, with
safeguards to prevent the step from leaving the bracket.

**Bisection safeguard**: If the bracket hasn't shrunk sufficiently (less than
66% of the previous width), force a midpoint step:

```python
if abs(sty - stx) >= p66 * width1:
    stp = stx + 0.5 * (sty - stx)
```

### Armijo backtracking (`line_search.py:364-404`)

A much simpler alternative: start with a step, check Armijo condition, halve
the step if not satisfied. No curvature condition. Cheaper per trial but may
need more outer iterations (the Hessian updates are less informative without
the curvature condition).

### Maximum feasible step

Before calling the line search, the optimizer computes the largest step that
keeps all variables in bounds (`_compute_max_step`, line 610):

```python
for each variable i:
    if d[i] < 0:  stpmax = min(stpmax, (l[i] - x[i]) / d[i])
    if d[i] > 0:  stpmax = min(stpmax, (u[i] - x[i]) / d[i])
```

This `stpmax` is passed to the line search to prevent out-of-bounds steps.

### Directional derivative shortcut

During the line search, each trial point needs `phi(alpha) = f(x + alpha*d)`
and `dphi(alpha) = g(x + alpha*d) . d`. Computing the full gradient at each
trial point would cost `2n` evaluations (central FD).

Instead, `_directional_derivative()` (line 132) computes the directional
derivative directly:

```python
dphi = (f(x + alpha*d + eps*d) - f(x + alpha*d - eps*d)) / (2*eps)
```

This costs only **2 extra evaluations** regardless of dimension, versus `2n`
for the full gradient. The full gradient is only computed once, at the accepted
point, for the L-BFGS update.

If an analytical gradient function is provided, it's used instead, and the
full gradient is cached:

```python
if self._gradient_fn is not None:
    ga = self._gradient_fn(xa)
    self._cached_gradient = ga
    return fa, np.dot(ga, d)
```

---

## 7. Convergence criteria

### Test 1: Projected gradient norm (`pgtol`)

    ||projected_gradient||_inf  <=  pgtol

The projected gradient accounts for active bounds. For each variable:

- If `g_i < 0` (wants to increase `x_i`): clip by distance to upper bound
  `-> pg_i = max(x_i - u_i, g_i)`
- If `g_i > 0` (wants to decrease `x_i`): clip by distance to lower bound
  `-> pg_i = min(x_i - l_i, g_i)`

At a KKT point, `pg_i = 0` for all `i`. This is the primary convergence test.

Implementation at `lbfgsb_optimizer.py:157-170`:

```python
pg[mask_neg] = np.maximum(x[mask_neg] - self.upper_bounds[mask_neg], g[mask_neg])
pg[mask_pos] = np.minimum(x[mask_pos] - self.lower_bounds[mask_pos], g[mask_pos])
return float(np.max(np.abs(pg)))
```

### Test 2: Relative function decrease (`factr`)

    (f_old - f_new) / max(|f_old|, |f_new|, 1)  <=  factr * machine_epsilon

This detects when the function is barely changing between iterations. The
`factr` parameter controls precision:

| `factr` | Meaning | Effective tolerance |
|---------|---------|---------------------|
| `1e12`  | Low accuracy | ~2.2e-4 |
| `1e7`   | Moderate (default) | ~2.2e-9 |
| `1e1`   | High accuracy | ~2.2e-15 |
| `0`     | Disabled | Only `pgtol` matters |

Implementation at `lbfgsb_optimizer.py:820-828`.

---

## 8. Configuration parameters

### `LBFGSBConfig` --- every parameter explained

| Parameter | Default | What it controls |
|-----------|---------|------------------|
| **`initial_hessian`** | `None` | Initial Hessian approximation `B_0`. `None` = identity, a `float` = scalar * I, a 1D `ndarray` = diagonal matrix. This controls the scaling of the very first iteration (before L-BFGS corrections kick in). After the first `(s,y)` update, the Barzilai-Borwein scaling `theta = y'y/y's` takes over. If you know the Hessian is roughly `c*I`, setting `initial_hessian=c` makes the first step better scaled. For Sphere, the exact Hessian is `2*I`, so `initial_hessian=2.0` is optimal. |
| **`m`** | `10` | Number of `(s,y)` correction pairs to store. This is the "L" in L-BFGS --- the limited memory. More pairs = better Hessian approximation but more work per iteration (`O(mn)` for matrix-vector products). The sweet spot is 5-20 for most problems. |
| **`factr`** | `1e7` | Function decrease tolerance factor (see Test 2 above). |
| **`pgtol`** | `1e-5` | Projected gradient tolerance (see Test 1 above). |
| **`line_search`** | `MORE_THUENTE` | Which line search to use. `MORE_THUENTE` finds strong Wolfe steps via cubic interpolation. `ARMIJO` does simple backtracking. |
| **`ftol`** | `1e-3` | Sufficient decrease parameter (Armijo constant `c_1`). The Wolfe condition requires `f(x + alpha*d) <= f(x) + ftol * alpha * g'd`. |
| **`gtol_ls`** | `0.9` | Curvature condition parameter (`c_2`). Only used by More-Thuente. Requires `|g(x + alpha*d) . d| <= gtol * |g(x) . d|`. Values close to 1.0 are lenient; smaller values force the line search to find a better minimum along the ray. |
| **`xtol_ls`** | `0.1` | Interval tolerance for the line search bracket. The search stops if the bracket width is within `xtol` of the current step. |
| **`max_ls_iter`** | `20` | Maximum function evaluations per line search call. If the line search can't find an acceptable step in this many tries, it signals failure and the algorithm resets the L-BFGS memory. |
| **`fd_method`** | `"central"` | Finite difference method when no analytical gradient is provided. `"central"` uses `(f(x+h) - f(x-h)) / 2h` (2n evals, `O(h^2)` error). `"forward"` uses `(f(x+h) - f(x)) / h` (n evals, `O(h)` error). |
| **`fd_eps`** | `0` (auto) | Step size for finite differences. Auto-computed as `sqrt(eps)` for central (~1.49e-8) or `eps^(1/3)` for forward (~6.06e-6), where `eps` is machine epsilon (~2.22e-16). |
| **`budget`** | `10000*n` | Hard cap on total function evaluations. The algorithm typically converges long before this via `factr`/`pgtol`. |

### The `gradient_fn` parameter

Not part of `LBFGSBConfig` (since it's a callable). Passed to the optimizer
constructor:

```python
optimizer = AlgorithmFactory.create_optimizer(
    algorithm=AlgorithmChoice.LBFGSB,
    func=func,
    initial_point=x0,
    gradient_fn=lambda x: 2*x,   # <-- analytical gradient
    ...
)
```

When provided, the gradient is free (0 evaluations). When absent, each
gradient costs 2n evaluations (central FD). For the Sphere function in 10D,
that's the difference between 5 evaluations and 99 evaluations to converge.

---

## 9. Implementation walkthrough

### File structure

```
src/algorithms/lbfgsb/
    __init__.py             Exports: LBFGSBConfig, LBFGSBOptimizer, LineSearchMethod
    config.py               LBFGSBConfig dataclass + LineSearchMethod enum
    line_search.py          More-Thuente and Armijo line searches
    lbfgsb_optimizer.py     The optimizer (all algorithm logic)

src/logging/
    lbfgsb_logger.py        LBFGSBLogData + LBFGSBLogger
```

### The optimize() main loop (`lbfgsb_optimizer.py:640-852`)

Here is the complete flow, step by step:

**Initialization** (lines 642-674):
```python
x = np.clip(self.initial_point.copy(), self.lower_bounds, self.upper_bounds)
f, g = self._eval_fg(x)
```
Project onto bounds, evaluate `f` and `g`. Check initial convergence.

**Step 1 --- Cauchy Point** (line 682):
```python
xcp, c, iwhere = self._cauchy_point(x, g)
```
Returns the GCP `xcp`, the displacement in W-space `c = W'(xcp - x)`,
and the variable classification `iwhere`.

**Step 2 --- Free variables** (line 685):
```python
free_indices, nfree = self._determine_free_variables(iwhere)
```

**Step 3 --- Subspace minimization** (lines 688-691):
```python
if nfree > 0 and self._col > 0:
    z = self._subspace_minimization(x, xcp, g, c, free_indices, nfree)
else:
    z = xcp
```
If we have no L-BFGS history (`col == 0`), skip subspace minimization and
just use the Cauchy point. This happens on the first iteration and after
memory resets.

**Step 4 --- Search direction** (lines 694-725):
```python
d = z - x
gd = np.dot(g, d)
```
Compute direction and verify it's descent. If not (shouldn't happen normally),
fall back to the projected negative gradient.

**Step 5 --- Line search** (lines 727-773):
```python
stpmax = self._compute_max_step(x, d, iteration == 1)
stp0 = min(1.0 / d_norm, stpmax) if iteration == 1 else min(1.0, stpmax)

ls_result = perform_line_search(
    method=config.line_search,
    phi_dphi=phi_dphi,
    stp0=stp0,
    ...
)
```
Initial step is `1/||d||` on the first iteration (since we don't know the
scale yet) and `1.0` afterwards (quasi-Newton methods are designed so that
unit step is asymptotically optimal).

If the line search fails with L-BFGS info, reset memory and retry.

**Step 6 --- Update** (lines 775-787):
```python
x_new = np.clip(x + stp * d, self.lower_bounds, self.upper_bounds)
s = x_new - x
y = g_new - g
```

**Step 7 --- Convergence** (lines 794-828): Check both criteria.

**Step 8 --- L-BFGS update** (line 831):
```python
self._update_lbfgs(s, y)
```
This stores the new `(s, y)` pair (discarding the oldest if at capacity `m`),
recomputes `theta = y'y / y's`, rebuilds the inner-product matrices `S'Y` and
`S'S`, and Cholesky-factorizes `T`.

### The curvature check

Before accepting an `(s, y)` pair, we check:

```python
if sy <= self._machine_eps * yy:
    return False   # skip this update
```

This ensures `s'y > 0` (positive curvature). Without this, the Hessian
approximation could become indefinite. The Fortran code uses the same check.

---

## 10. Diagnostics and plotting

### LBFGSBLogData fields

| Field | Meaning | When logged |
|-------|---------|-------------|
| `best_fitness` | Running best `f(x)` | Always |
| `function_value` | Current `f(x)` at this iteration | Always |
| `evaluations` | Total function evaluations so far | Always |
| `gradient_norm` | `||g||_2` (L2 norm) | `diag_gradient_norm` |
| `projected_gradient_norm` | `||proj g||_inf` (convergence measure) | `diag_gradient_norm` |
| `step_length` | Line search `alpha` accepted | `diag_step_length` |
| `theta` | L-BFGS scaling `y'y / y's` | `diag_theta` |
| `num_free_vars` | Variables not at bounds | `diag_num_free` |
| `num_corrections` | `(s,y)` pairs stored | `diag_num_free` |
| `line_search_iters` | Evals used in line search | `diag_line_search_iters` |

### The 8-panel diagnostic plot

The `_plot_lbfgsb_metrics()` in `multi_algorithm_plotter.py` produces:

**Row 1 --- Convergence:**
- **(1,1) Convergence**: `best_fitness` and `function_value` on log scale.
  Shows the actual optimization progress. These should decrease monotonically.
- **(1,2) Gradient Norms**: `||proj g||_inf` (red, convergence measure) and
  `||g||_2` (blue). The projected gradient norm is what `pgtol` tests against.
  Watching both reveals whether bounds are affecting convergence.

**Row 2 --- Adaptation:**
- **(2,1) Step Length**: Line search `alpha` over iterations. On well-conditioned
  problems, this should stabilize near 1.0 as the L-BFGS Hessian becomes
  accurate. Early iterations may use smaller steps.
- **(2,2) Theta**: The L-BFGS scaling factor `y'y / y's`. This adapts the
  initial Hessian `B_0 = theta * I` to the local curvature. Large theta means
  high curvature (small steps); small theta means gentle curvature (large steps).

**Row 3 --- Constraint dynamics:**
- **(3,1) Free Variables**: How many of the `n` variables are free (not at
  bounds). On unconstrained problems, this equals `n` always. On constrained
  problems, watching this reveals how the active set changes.
- **(3,2) Corrections Stored**: Number of `(s,y)` pairs, ramping from 0 to
  `m` and staying there. If it drops to 0, the memory was reset (numerical
  issue).

**Row 4 --- Line search:**
- **(4,1) Line Search Evals**: Function evaluations per line search. Should
  typically be 1-3 for More-Thuente on smooth problems. Spikes indicate
  difficult terrain.
- **(4,2) Function Value**: `f(x)` on log scale per iteration. Similar to
  convergence plot but shows every iteration's value, not just the running
  best.

---

## 11. How it compares to evolutionary algorithms

### Fundamental differences

| Aspect | L-BFGS-B | CMA-ES / DES / MF-CMA-ES |
|--------|----------|--------------------------|
| **Type** | Gradient-based, deterministic | Population-based, stochastic |
| **Population** | 1 point | lambda individuals (e.g., 4+3*ln(n)) |
| **Information used** | Gradient (first-order) + Hessian approx (second-order) | Fitness ranking only |
| **Gradient required** | Yes (analytical or finite differences) | No |
| **Local/global** | Local optimizer (finds nearest local minimum) | More global exploration |
| **Convergence rate** | Superlinear (near optimum) | Linear |
| **Per-iteration cost** | `O(mn)` + gradient cost | `O(n^2)` to `O(n^3)` (covariance update) |
| **Smoothness required** | Yes (needs smooth gradient) | No (works on noisy, non-smooth) |
| **Bound handling** | Exact (GCP + active sets) | Repair strategies (clamp, bounce-back) |

### When to use which

**L-BFGS-B wins** when:
- The function is smooth and differentiable
- An analytical gradient is available (or `n` is small enough for FD)
- You want the local minimum nearest to your starting point
- You need high precision (1e-10 or better)
- Speed matters (convergence in 10-100 iterations vs 1000+)

**Evolutionary algorithms win** when:
- The function is noisy, non-smooth, or has many local minima
- No gradient information is available or meaningful
- You want global exploration, not just the nearest local minimum
- The function has discrete or mixed-integer variables

### Benchmark illustration

On Sphere(10D) from our benchmark:

    L-BFGS-B:  7.94e-22 in 99 evaluations     (3 iterations)
    CMA-ES:    5.73e-16 in 3146 evaluations    (hundreds of generations)

L-BFGS-B is ~32x faster in evaluations and achieves 6 orders of magnitude
better precision. This is expected --- Sphere is smooth and convex, the ideal
case for quasi-Newton methods.

On harder problems (Rosenbrock, Ellipsoid), L-BFGS-B still converges in
hundreds to low thousands of evaluations, while evolutionary methods may
need tens of thousands.

The thesis value of including L-BFGS-B is that the **unified benchmarking
framework** can now fairly compare gradient-based and gradient-free methods
under identical conditions (same function, same bounds, same evaluation
counting, same logging).

---

## Appendix: Key formulas at a glance

**Compact L-BFGS:**
    B = theta*I - W*M*W',  where W = [Y | theta*S],  theta = y'y / y's

**Middle matrix inverse:**
    M^{-1} = [-D, L'; L, theta*S'S]

**Cholesky target:**
    T = theta*S'S + L*D^{-1}*L'

**bmv solve (M*v):**
    T*p2 = v2 + L*(v1/D),  then  p1 = (L'*p2 - v1)/D

**Cauchy breakpoint:**
    t_i = distance_to_bound / |g_i|

**Cauchy derivative update at breakpoint:**
    f1 += dt*f2 + d_i^2 - theta*d_i*z_b  (+ L-BFGS corrections)
    f2 -= theta*d_i^2                      (+ L-BFGS corrections)

**Subspace Newton via Woodbury:**
    d = (1/theta)*r + (1/theta^2)*A*K^{-1}*A'*r
    where K = M^{-1} - (1/theta)*A'*A, A = Z'W

**Strong Wolfe conditions:**
    f(x + alpha*d)  <=  f(x) + c1*alpha*g'd        (sufficient decrease)
    |g(x + alpha*d).d|  <=  c2*|g(x).d|            (curvature)

**Projected gradient (KKT measure):**
    pg_i = max(x_i - u_i, g_i)   if g_i < 0
    pg_i = min(x_i - l_i, g_i)   if g_i > 0

**Convergence test 2:**
    (f_old - f_new) / max(|f_old|, |f_new|, 1)  <=  factr * eps_mach
