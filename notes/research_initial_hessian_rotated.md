# Research: Initial Hessian on Rotated Problems

## Experiment setup

Function: RotatedEllipsoid (10D, condition number 10^6, uniform_45 rotation).
Analytical gradient provided. Three initial Hessian choices: identity, diagonal
of the rotated Hessian, full rotated Hessian matrix.

## Key finding: diagonal is worse than identity on rotated problems

With wide bounds ([-100, 100], x0=[10,...,10]), no variables hit bounds.
The Cauchy point active set identification is irrelevant. Results:

    Identity            : f=3.63e-23  evals=     4
    Hessian diagonal    : f=6.93e-02  evals=  3173
    Full Hessian        : f=5.68e-15  evals=    19

The diagonal case is catastrophic: 3173 evals to reach only 6.9e-2.
Identity reaches 3.6e-23 in 4 evals. Full Hessian reaches 5.7e-15 in 19.

## Why the diagonal fails on rotated problems

The Ellipsoid has eigenvalues from 2 to 2e6. After a 45-degree chain rotation,
the principal curvature directions no longer align with the coordinate axes.
The diagonal of the rotated Hessian captures per-coordinate curvature, which
is a mixture of many eigenvalues projected onto each axis.

This diagonal tells the algorithm "variable 1 has curvature X" but the actual
curvature in the variable-1 direction is a weighted combination of all
eigenvalues. The Cauchy point uses this wrong per-variable scaling to determine
step sizes, sending the algorithm in poorly scaled directions. The L-BFGS
corrections then spend thousands of iterations trying to undo the damage from
the wrong prior.

The diagonal energy fraction for uniform_45 rotation is 0.49 — only half the
Hessian energy is on the diagonal. The other half is off-diagonal cross-terms
that the diagonal approximation discards.

## Why identity works well here

Identity (B_0 = I) is unbiased. It doesn't prefer any coordinate direction
over any other. On a pure quadratic with analytical gradient:

1. The first step is scaled steepest descent with an exact line search
   (the More-Thuente search finds the 1D minimum along the ray).
2. This single step makes enormous progress (25+ orders of magnitude on
   the quadratic).
3. After 1-2 steps, the L-BFGS corrections (rank-2 each) already capture
   significant curvature information.
4. The algorithm converges in 3-4 total iterations.

Identity doesn't help, but it doesn't hurt either. The L-BFGS corrections
do all the curvature learning, starting from a clean slate.

## Why full Hessian is good but not better than identity here

The full Hessian provides exact curvature information. However:

1. With persist_initial_hessian=True, the effective base is theta * H.
   The theta scalar adapts every iteration (theta = y'y / y's). On a
   quadratic, theta can overshoot or undershoot the correct scale,
   distorting the otherwise perfect H.
2. With no active bounds, the Cauchy point is essentially irrelevant —
   no variables hit bounds regardless of the Hessian scaling. The benefit
   of a good Hessian for active set identification is wasted.
3. On a pure quadratic with analytical gradient, L-BFGS converges so fast
   that the initial Hessian barely matters — the corrections dominate
   after 2-3 iterations.

## When would the full Hessian shine?

The full Hessian advantage would appear when:

1. **Bounds are active**: tight bounds force variables to their limits. The
   Cauchy point uses B_0 to determine which variables should be fixed at
   bounds. A well-scaled B_0 identifies the correct active set faster.

2. **Non-quadratic functions**: on quadratics, one-dimensional line search
   is exact and L-BFGS converges in ~n iterations regardless of B_0. On
   non-quadratic functions, the quadratic model is only approximate, and
   the quality of B_0 affects how many iterations are needed to reach the
   basin of convergence.

3. **Many dimensions with few corrections**: when n >> m (e.g. n=1000,
   m=10), the L-BFGS corrections can only capture 2m=20 directions of
   curvature. A full B_0 provides the remaining n-2m directions that
   corrections cannot cover.

## Conclusion

Wrong curvature information is worse than no information. On rotated problems
where the Hessian is non-diagonal, passing only the diagonal is actively
harmful because it encodes incorrect per-variable scaling that the L-BFGS
corrections must fight against.

The full Hessian is the correct input for rotated problems, but its advantage
only materializes when bounds are active or the function is non-quadratic.
On unconstrained quadratics with analytical gradient, the corrections learn
so fast that the initial Hessian is nearly irrelevant.

## Experiment: tight bounds with origin excluded

Setup: lower_bounds=1.0, upper_bounds=10.0. All variables at their lower
bound at the constrained optimum. Finite differences.

    Identity            : f=5.77e+06  evals=    72
    Hessian diagonal    : f=5.77e+06  evals=    72
    Full Hessian        : f=5.82e+06  evals=   402 (line search failed)

Uninformative: the constrained optimum has all variables at the lower bound.
The Cauchy point identifies this immediately regardless of B_0. The initial
Hessian has no opportunity to influence the active set.

## Experiment: high dimension, few corrections (n=50, m=5)

This is the decisive experiment. With n=50 and m=5, the L-BFGS corrections
span at most 2m=10 directions out of 50. The remaining 40 directions are
determined entirely by B_0.

Setup: RotatedEllipsoid (uniform_45, condition number 10^6), n=50, m=5,
analytical gradient, wide bounds ([-100, 100]), pgtol=1e-8, factr=1e7.

    Identity            : f=3.12e-06  evals=  3508
    Hessian diagonal    : f=2.28e+05  evals=  5000 (budget exhausted)
    Full Hessian        : f=1.74e-13  evals=    22

Results:
- Full Hessian converges in 22 evaluations to 1.7e-13.
- Identity needs 3508 evaluations to reach only 3.1e-6 (7 orders worse).
- Diagonal fails entirely: stuck at 2.3e5 after exhausting the budget.

The full Hessian is 160x faster than identity and reaches 7 orders of
magnitude better precision. The diagonal is catastrophic.

Analysis: with 2m=10 correction directions and n=50 problem dimensions,
the corrections can only capture curvature along 10 directions. The
remaining 40 directions rely entirely on B_0:

- Full Hessian: all 50 directions are correctly scaled from the start.
  The corrections refine the 10 most active directions.
- Identity: 40 directions are scaled by theta (a single scalar) which
  cannot distinguish between eigenvalues spanning 6 orders of magnitude.
  The algorithm slowly converges by cycling through directions.
- Diagonal: 40 directions are scaled by the diagonal of the rotated
  Hessian, which encodes wrong per-variable curvature. The corrections
  fight against this misleading prior and make almost no progress.

This confirms the central thesis: when n >> 2m and the Hessian has
significant off-diagonal structure (rotation), the full initial Hessian
matrix provides information that the limited-memory corrections cannot
discover on their own.

## Summary of findings

| Scenario                          | Identity | Diagonal | Full Hessian |
|-----------------------------------|----------|----------|--------------|
| Ellipsoid 10D, axis-aligned       | 14109    | 881      | N/A          |
| Rotated Ellipsoid 10D, n=m        | 4        | 3173     | 19           |
| Rotated Ellipsoid 50D, n>>m       | 3508     | >5000    | 22           |

The full matrix wins decisively when:
1. The Hessian has off-diagonal structure (rotation)
2. The correction memory cannot span the full space (n >> 2m)

The diagonal wins when the Hessian is actually diagonal (axis-aligned Ellipsoid).
The diagonal is catastrophic when the Hessian is non-diagonal (rotated problems).
