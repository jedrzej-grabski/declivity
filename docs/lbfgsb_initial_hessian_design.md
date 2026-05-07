# L-BFGS-B Initial Hessian: Design and Cost Analysis

## The inconsistency in the original implementation

The standard L-BFGS-B compact representation assumes B_0 = theta * I everywhere.
Our first extension allowed passing a diagonal initial Hessian, but only threaded
it through the Cauchy point computation. The subspace minimization still assumed
B_0 = theta * I:

- Cauchy point: used `theta * diag(h)` for the quadratic model curvature
- Subspace minimization: used `(1/theta) * I` in the Woodbury formula
- W matrix: built as `[Y | theta * S]` (assumes B_0 = theta * I)
- M^{-1} matrix: block `theta * S'S` (assumes B_0 = theta * I)
- T matrix: `theta * S'S + L D^{-1} L'` (assumes B_0 = theta * I)

This meant the initial Hessian helped with active set identification (which
variables hit bounds in the Cauchy point) but did not improve the Newton step
in the subspace minimization.

## Where B_0 appears in the algorithm

The compact L-BFGS representation is B = theta * B_0 - W * M * W', where:

    W = [Y | theta * B_0 * S]
    M^{-1} = [-D, L'; L, theta * S' * B_0 * S]
    T = theta * S' * B_0 * S + L * D^{-1} * L'

The Woodbury formula for the subspace Newton step:

    d = B_0^{-1} * r  +  (1/theta) * B_0^{-1} * A * K^{-1} * A' * r

where A = Z' W and K = M^{-1} - (1/theta) * A' * B_0^{-1} * A.

Every instance of I in the original algorithm is replaced by B_0, and every
instance of 1/theta (as a scalar inverse) is replaced by B_0^{-1} / theta.

## Cost per iteration

### B_0 = I (original L-BFGS-B)

| Operation              | Cost      |
|------------------------|-----------|
| B_0 * v               | free      |
| B_0^{-1} * v          | free      |
| B_0 * S (m vectors)   | free      |
| S' B_0 S (m x m)      | O(m^2 n)  |
| Total per iteration    | O(m n)    |
| Storage                | 0         |

### B_0 = diag(h)

| Operation              | Cost       |
|------------------------|------------|
| B_0 * v               | O(n)       |
| B_0^{-1} * v          | O(n)       |
| B_0 * S (m vectors)   | O(m n)     |
| S' B_0 S (m x m)      | O(m^2 n)   |
| Total per iteration    | O(m n)     |
| Storage                | O(n)       |

The diagonal case has the same asymptotic cost as the identity case because
element-wise multiplication is O(n), same as a dot product.

### B_0 = full matrix

| Operation              | Cost          |
|------------------------|---------------|
| B_0 * v               | O(n^2)        |
| B_0^{-1} * v          | O(n^2) *      |
| B_0 * S (m vectors)   | O(m n^2)      |
| S' B_0 S (m x m)      | O(m n^2) **   |
| Total per iteration    | O(m n^2)      |
| Storage                | O(n^2)        |
| One-time setup         | O(n^3) ***    |

 *  Using precomputed Cholesky factorization, the solve is O(n^2).
 ** Precompute B_0 S first (O(m n^2)), then S' times that (O(m^2 n)).
*** Cholesky factorization of B_0, computed once at initialization.

## When the full matrix cost is acceptable

For n = 10:   m n^2 = 10 * 100 = 1,000 operations per iteration. Trivial.
For n = 100:  m n^2 = 10 * 10,000 = 100,000 per iteration. Still fast.
For n = 1000: m n^2 = 10 * 1,000,000 = 10^7 per iteration. Noticeable.
For n = 10000: m n^2 = 10^9 per iteration. Defeats the purpose of L-BFGS.

For thesis-scale problems (n up to ~100), the full matrix is entirely practical.

## Implementation strategy

The implementation branches on the type of B_0 provided:

- None or scalar: B_0 is stored as a diagonal vector (all entries equal).
  Uses element-wise operations throughout. No Cholesky needed.

- 1D array (length n): B_0 is stored as a diagonal vector.
  Uses element-wise operations throughout. No Cholesky needed.

- 2D array (n x n): B_0 is stored as a dense matrix alongside its
  precomputed Cholesky factorization. Uses matrix-vector products and
  triangular solves throughout.

The branching is encapsulated in a helper object that provides:
  - multiply(v): returns B_0 * v
  - solve(v): returns B_0^{-1} * v
  - quadratic_form(S): returns S' * B_0 * S
  - scale_columns(S): returns B_0 * S (column-wise)

The optimizer calls these methods uniformly. The helper dispatches to either
element-wise or dense operations based on the type stored at initialization.
This keeps the algorithm code clean with no type-checking branches in the
hot path.
