from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Callable, final, override

import numpy as np

__all__ = [
    "ScalarSearchResult",
    "DerivativeFreeLineSearch",
    "BrentLineSearch",
    "GoldenSectionLineSearch",
    "DerivativeFreeLineSearchType",
]


@dataclass
class ScalarSearchResult:
    """Result of a derivative-free line search.

    ``alpha`` is the accepted step along the direction (may be negative:
    a derivative-free search explores both orientations of the ray),
    ``f_min`` the objective value at that step.
    """

    alpha: float
    f_min: float
    num_evals: int
    converged: bool


class DerivativeFreeLineSearch(ABC):
    """Abstract base class for derivative-free line-search strategies.

    Finds ``alpha`` minimizing ``phi(alpha) = f(x + alpha * d)`` using
    function values only.  Unlike the gradient-based branch there is no
    sufficient-decrease test — the search is a genuine 1-D minimization,
    so the natural tolerance is on the *location* of the minimizer.
    """

    @abstractmethod
    def search(
        self,
        phi: Callable[[float], float],
        alpha_bounds: tuple[float, float] | None = None,
        tol: float = 1e-2,
        maxiter: int = 500,
    ) -> ScalarSearchResult:
        """Minimize ``phi`` over the step interval.

        Parameters
        ----------
        phi:
            1-D restriction of the objective, ``phi(alpha) = f(x + alpha*d)``.
        alpha_bounds:
            Feasible step interval ``(alpha_min, alpha_max)``.  ``None``
            or ``(-inf, +inf)`` means unbounded; either end may be
            infinite.  Powell derives this interval from the box bounds
            so that every trial point stays feasible.
        tol:
            Line-search tolerance, matching SciPy's
            ``_linesearch_powell(tol=...)`` convention: the unbounded
            Brent branch uses it as a *relative* tolerance on ``alpha``
            and the bounded branch uses ``tol / 100`` as an *absolute*
            tolerance (SciPy applies the same split).  Powell passes
            ``100 * xtol``.
        maxiter:
            Iteration cap for the scalar minimizer.
        """
        ...


@final
class BrentLineSearch(DerivativeFreeLineSearch):
    """SciPy-faithful derivative-free search (Powell's default).

    Dispatches exactly like ``scipy.optimize._optimize._linesearch_powell``:

    - no usable bounds → downhill :func:`bracket_minimum` from
      ``(0, 1)`` followed by :func:`brent_minimize`;
    - finite ``(alpha_min, alpha_max)`` → :func:`bounded_minimize`
      (golden section + parabolic interpolation on a fixed interval);
    - one finite end → arctan change of variables onto a finite
      interval, then :func:`bounded_minimize`.
    """

    def __init__(self, grow_limit: float = 110.0) -> None:
        self.grow_limit = grow_limit
        """Maximum bracket growth factor for the downhill search."""

    @override
    def search(
        self,
        phi: Callable[[float], float],
        alpha_bounds: tuple[float, float] | None = None,
        tol: float = 1e-2,
        maxiter: int = 500,
    ) -> ScalarSearchResult:
        if alpha_bounds is not None:
            lower, upper = alpha_bounds
            if np.isneginf(lower) and np.isposinf(upper):
                alpha_bounds = None

        if alpha_bounds is None:
            return brent_minimize(
                phi, xtol=tol, maxiter=maxiter, grow_limit=self.grow_limit
            )

        lower, upper = alpha_bounds
        if np.isfinite(lower) and np.isfinite(upper):
            return bounded_minimize(
                phi, lower, upper, xatol=tol / 100.0, maxiter=maxiter
            )

        # Half-infinite interval: map onto a finite one with tan/arctan.
        # The bounded region is a subinterval of (-pi/2, pi/2).
        t_lower, t_upper = np.arctan(lower), np.arctan(upper)
        result = bounded_minimize(
            lambda t: phi(float(np.tan(t))),
            t_lower,
            t_upper,
            xatol=tol / 100.0,
            maxiter=maxiter,
        )
        return ScalarSearchResult(
            alpha=float(np.tan(result.alpha)),
            f_min=result.f_min,
            num_evals=result.num_evals,
            converged=result.converged,
        )


@final
class GoldenSectionLineSearch(DerivativeFreeLineSearch):
    """Pure golden-section search — no parabolic acceleration.

    Unbounded intervals are first bracketed downhill from ``(0, 1)``
    (same :func:`bracket_minimum` as Brent); the golden-section loop
    then shrinks the bracket at the fixed rate 0.618 per evaluation.
    Linear convergence versus Brent's superlinear, but each step is
    unconditionally safe — a useful robustness baseline on noisy or
    kinked ``phi``.
    """

    def __init__(self, grow_limit: float = 110.0) -> None:
        self.grow_limit = grow_limit

    @override
    def search(
        self,
        phi: Callable[[float], float],
        alpha_bounds: tuple[float, float] | None = None,
        tol: float = 1e-2,
        maxiter: int = 500,
    ) -> ScalarSearchResult:
        if alpha_bounds is not None:
            lower, upper = alpha_bounds
            if np.isneginf(lower) and np.isposinf(upper):
                alpha_bounds = None

        if alpha_bounds is None:
            try:
                xa, xb, xc, fa, fb, fc, num_evals = bracket_minimum(
                    phi, 0.0, 1.0, grow_limit=self.grow_limit
                )
            except BracketError as exc:
                return _best_of_failed_bracket(exc)
            lower, upper = min(xa, xc), max(xa, xc)
            return _golden_section(
                phi,
                lower,
                upper,
                xatol=tol / 100.0,
                maxiter=maxiter,
                num_evals=num_evals,
            )

        lower, upper = alpha_bounds
        if not (np.isfinite(lower) and np.isfinite(upper)):
            t_lower, t_upper = np.arctan(lower), np.arctan(upper)
            result = _golden_section(
                lambda t: phi(float(np.tan(t))),
                t_lower,
                t_upper,
                xatol=tol / 100.0,
                maxiter=maxiter,
            )
            return ScalarSearchResult(
                alpha=float(np.tan(result.alpha)),
                f_min=result.f_min,
                num_evals=result.num_evals,
                converged=result.converged,
            )

        return _golden_section(phi, lower, upper, xatol=tol / 100.0, maxiter=maxiter)


class DerivativeFreeLineSearchType(Enum):
    """
    Discoverability enum listing all built-in derivative-free line searches.

    Call ``.build()`` to obtain a ready-to-use
    ``DerivativeFreeLineSearch`` instance without importing concrete
    classes directly.

    Members
    -------
    BRENT
        SciPy-faithful Brent search: automatic downhill bracketing on
        unbounded intervals, golden section + parabolic interpolation on
        bounded ones.  Powell's default.
    GOLDEN_SECTION
        Pure golden-section shrinking — no parabolic acceleration.
        Linear convergence, but unconditionally safe on noisy or kinked
        objectives.
    """

    BRENT = "brent"
    GOLDEN_SECTION = "golden_section"

    def build(self) -> DerivativeFreeLineSearch:
        """
        Construct and return a fresh ``DerivativeFreeLineSearch`` instance.

        Returns
        -------
        DerivativeFreeLineSearch
            A concrete line search for this enum member.
        """
        if self is DerivativeFreeLineSearchType.BRENT:
            return BrentLineSearch()
        elif self is DerivativeFreeLineSearchType.GOLDEN_SECTION:
            return GoldenSectionLineSearch()
        # Exhaustive match — new members must extend this method.
        raise NotImplementedError(f"No build() implementation for {self!r}")


# ---------------------------------------------------------------------------
# Scalar-minimizer primitives (ports of scipy.optimize._optimize internals).
# ---------------------------------------------------------------------------


class BracketError(RuntimeError):
    """Downhill bracketing failed to isolate a minimum.

    Carries ``data = (xa, xb, xc, fa, fb, fc, num_evals)`` — the best
    information gathered before failure — so callers can degrade
    gracefully instead of losing the evaluations.
    """

    data: tuple[float, float, float, float, float, float, int]


def bracket_minimum(
    phi: Callable[[float], float],
    xa: float = 0.0,
    xb: float = 1.0,
    grow_limit: float = 110.0,
    maxiter: int = 1000,
) -> tuple[float, float, float, float, float, float, int]:
    """Bracket a minimum of ``phi`` downhill from two initial points.

    Port of :func:`scipy.optimize.bracket`.  Returns ``(xa, xb, xc,
    fa, fb, fc, num_evals)`` with ``xa < xb < xc`` (or reversed) and
    ``phi(xb)`` below both endpoints.  Raises :class:`BracketError`
    (with partial data attached) if no valid bracket exists — e.g. when
    ``phi`` is monotonically decreasing to infinity.
    """
    _gold = 1.618034  # golden ratio: (1.0 + sqrt(5.0)) / 2.0
    _verysmall_num = 1e-21

    fa = phi(xa)
    fb = phi(xb)
    if fa < fb:  # Switch so fa > fb
        xa, xb = xb, xa
        fa, fb = fb, fa
    xc = xb + _gold * (xb - xa)
    fc = phi(xc)
    num_evals = 3
    iteration = 0

    while fc < fb:
        tmp1 = (xb - xa) * (fb - fc)
        tmp2 = (xb - xc) * (fb - fa)
        val = tmp2 - tmp1
        if np.abs(val) < _verysmall_num:
            denom = 2.0 * _verysmall_num
        else:
            denom = 2.0 * val
        w = xb - ((xb - xc) * tmp2 - (xb - xa) * tmp1) / denom
        wlim = xb + grow_limit * (xc - xb)
        if iteration > maxiter:
            raise RuntimeError(
                "No valid bracket found before the iteration limit; try "
                "different initial points or a larger maxiter."
            )
        iteration += 1
        if (w - xc) * (xb - w) > 0.0:
            fw = phi(w)
            num_evals += 1
            if fw < fc:
                xa, xb = xb, w
                fa, fb = fb, fw
                break
            elif fw > fb:
                xc, fc = w, fw
                break
            w = xc + _gold * (xc - xb)
            fw = phi(w)
            num_evals += 1
        elif (w - wlim) * (wlim - xc) >= 0.0:
            w = wlim
            fw = phi(w)
            num_evals += 1
        elif (w - wlim) * (xc - w) > 0.0:
            fw = phi(w)
            num_evals += 1
            if fw < fc:
                xb, xc = xc, w
                w = xc + _gold * (xc - xb)
                fb, fc = fc, fw
                fw = phi(w)
                num_evals += 1
        else:
            w = xc + _gold * (xc - xb)
            fw = phi(w)
            num_evals += 1
        xa, xb, xc = xb, xc, w
        fa, fb, fc = fb, fc, fw

    # Three conditions for a valid bracket.
    cond1 = (fb < fc and fb <= fa) or (fb < fa and fb <= fc)
    cond2 = xa < xb < xc or xc < xb < xa
    cond3 = np.isfinite(xa) and np.isfinite(xb) and np.isfinite(xc)
    if not (cond1 and cond2 and cond3):
        error = BracketError("Bracketing terminated without isolating a minimum.")
        error.data = (xa, xb, xc, fa, fb, fc, num_evals)
        raise error

    return xa, xb, xc, fa, fb, fc, num_evals


def _best_of_failed_bracket(exc: BracketError) -> ScalarSearchResult:
    """Degrade gracefully when bracketing fails (scipy gh-14858 behavior):
    return the best of the three probe points instead of raising."""
    xa, xb, xc, fa, fb, fc, num_evals = exc.data
    xs, fs = [xa, xb, xc], [fa, fb, fc]
    if np.any(np.isnan(xs)) or np.any(np.isnan(fs)):
        return ScalarSearchResult(
            alpha=float("nan"),
            f_min=float("nan"),
            num_evals=num_evals,
            converged=False,
        )
    imin = int(np.argmin(fs))
    return ScalarSearchResult(
        alpha=float(xs[imin]),
        f_min=float(fs[imin]),
        num_evals=num_evals,
        converged=False,
    )


def brent_minimize(
    phi: Callable[[float], float],
    xtol: float = 1.48e-8,
    maxiter: int = 500,
    grow_limit: float = 110.0,
) -> ScalarSearchResult:
    """Brent's method with automatic downhill bracketing from ``(0, 1)``.

    Port of ``scipy.optimize._optimize.Brent`` (golden-section steps
    accelerated by parabolic interpolation), fed by
    :func:`bracket_minimum`.  ``xtol`` is a *relative* tolerance on the
    minimizer location.
    """
    _mintol = 1.0e-11
    _cg = 0.3819660

    try:
        xa, xb, xc, fa, fb, fc, num_evals = bracket_minimum(
            phi, 0.0, 1.0, grow_limit=grow_limit
        )
    except BracketError as exc:
        return _best_of_failed_bracket(exc)

    x = w = v = xb
    fw = fv = fx = fb
    if xa < xc:
        a, b = xa, xc
    else:
        a, b = xc, xa
    deltax = 0.0
    rat = 0.0
    iteration = 0

    while iteration < maxiter:
        tol1 = xtol * np.abs(x) + _mintol
        tol2 = 2.0 * tol1
        xmid = 0.5 * (a + b)
        # check for convergence
        if np.abs(x - xmid) < (tol2 - 0.5 * (b - a)):
            break
        if np.abs(deltax) <= tol1:
            if x >= xmid:
                deltax = a - x  # do a golden section step
            else:
                deltax = b - x
            rat = _cg * deltax
        else:  # do a parabolic step
            tmp1 = (x - w) * (fx - fv)
            tmp2 = (x - v) * (fx - fw)
            p = (x - v) * tmp2 - (x - w) * tmp1
            tmp2 = 2.0 * (tmp2 - tmp1)
            if tmp2 > 0.0:
                p = -p
            tmp2 = np.abs(tmp2)
            dx_temp = deltax
            deltax = rat
            # check parabolic fit
            if (
                (p > tmp2 * (a - x))
                and (p < tmp2 * (b - x))
                and (np.abs(p) < np.abs(0.5 * tmp2 * dx_temp))
            ):
                rat = p * 1.0 / tmp2  # if parabolic step is useful
                u = x + rat
                if (u - a) < tol2 or (b - u) < tol2:
                    if xmid - x >= 0:
                        rat = tol1
                    else:
                        rat = -tol1
            else:
                if x >= xmid:
                    deltax = a - x  # if it's not do a golden section step
                else:
                    deltax = b - x
                rat = _cg * deltax

        if np.abs(rat) < tol1:  # update by at least tol1
            if rat >= 0:
                u = x + tol1
            else:
                u = x - tol1
        else:
            u = x + rat
        fu = phi(u)
        num_evals += 1

        if fu > fx:  # if it's bigger than current
            if u < x:
                a = u
            else:
                b = u
            if (fu <= fw) or (w == x):
                v, w = w, u
                fv, fw = fw, fu
            elif (fu <= fv) or (v == x) or (v == w):
                v = u
                fv = fu
        else:
            if u >= x:
                a = x
            else:
                b = x
            v, w, x = w, x, u
            fv, fw, fx = fw, fx, fu

        iteration += 1

    converged = iteration < maxiter and not (np.isnan(x) or np.isnan(fx))
    return ScalarSearchResult(
        alpha=float(x), f_min=float(fx), num_evals=num_evals, converged=converged
    )


def bounded_minimize(
    phi: Callable[[float], float],
    lower: float,
    upper: float,
    xatol: float = 1e-5,
    maxiter: int = 500,
) -> ScalarSearchResult:
    """Golden-section / parabolic minimization on a fixed finite interval.

    Port of ``scipy.optimize._optimize._minimize_scalar_bounded`` (the
    ``fminbound`` core).  ``xatol`` is an *absolute* tolerance on the
    minimizer location.  ``maxiter`` caps function evaluations (matching
    the SciPy port, where ``maxfun = maxiter``).
    """
    if lower > upper:
        raise ValueError("The lower bound exceeds the upper bound.")

    sqrt_eps = np.sqrt(2.2e-16)
    golden_mean = 0.5 * (3.0 - np.sqrt(5.0))
    a, b = lower, upper
    fulc = a + golden_mean * (b - a)
    nfc, xf = fulc, fulc
    rat = e = 0.0
    x = xf
    fx = phi(x)
    num_evals = 1
    fu = np.inf

    ffulc = fnfc = fx
    xm = 0.5 * (a + b)
    tol1 = sqrt_eps * np.abs(xf) + xatol / 3.0
    tol2 = 2.0 * tol1

    converged = True
    while np.abs(xf - xm) > (tol2 - 0.5 * (b - a)):
        golden = 1
        # Check for parabolic fit
        if np.abs(e) > tol1:
            golden = 0
            r = (xf - nfc) * (fx - ffulc)
            q = (xf - fulc) * (fx - fnfc)
            p = (xf - fulc) * q - (xf - nfc) * r
            q = 2.0 * (q - r)
            if q > 0.0:
                p = -p
            q = np.abs(q)
            r = e
            e = rat

            # Check for acceptability of parabola
            if (
                (np.abs(p) < np.abs(0.5 * q * r))
                and (p > q * (a - xf))
                and (p < q * (b - xf))
            ):
                rat = (p + 0.0) / q
                x = xf + rat

                if ((x - a) < tol2) or ((b - x) < tol2):
                    si = np.sign(xm - xf) + ((xm - xf) == 0)
                    rat = tol1 * si
            else:  # do a golden-section step
                golden = 1

        if golden:  # do a golden-section step
            if xf >= xm:
                e = a - xf
            else:
                e = b - xf
            rat = golden_mean * e

        si = np.sign(rat) + (rat == 0)
        x = xf + si * np.maximum(np.abs(rat), tol1)
        fu = phi(x)
        num_evals += 1

        if fu <= fx:
            if x >= xf:
                a = xf
            else:
                b = xf
            fulc, ffulc = nfc, fnfc
            nfc, fnfc = xf, fx
            xf, fx = x, fu
        else:
            if x < xf:
                a = x
            else:
                b = x
            if (fu <= fnfc) or (nfc == xf):
                fulc, ffulc = nfc, fnfc
                nfc, fnfc = x, fu
            elif (fu <= ffulc) or (fulc == xf) or (fulc == nfc):
                fulc, ffulc = x, fu

        xm = 0.5 * (a + b)
        tol1 = sqrt_eps * np.abs(xf) + xatol / 3.0
        tol2 = 2.0 * tol1

        if num_evals >= maxiter:
            converged = False
            break

    if np.isnan(xf) or np.isnan(fx) or np.isnan(fu):
        converged = False

    return ScalarSearchResult(
        alpha=float(xf), f_min=float(fx), num_evals=num_evals, converged=converged
    )


def _golden_section(
    phi: Callable[[float], float],
    lower: float,
    upper: float,
    xatol: float = 1e-5,
    maxiter: int = 500,
    num_evals: int = 0,
) -> ScalarSearchResult:
    """Plain golden-section shrinking on ``[lower, upper]``.

    ``num_evals`` seeds the evaluation counter so callers that spent
    evaluations bracketing can report a faithful total.
    """
    invphi = (np.sqrt(5.0) - 1.0) / 2.0  # 1/phi ≈ 0.618

    a, b = lower, upper
    c = b - invphi * (b - a)
    d = a + invphi * (b - a)
    fc = phi(c)
    fd = phi(d)
    num_evals += 2

    iteration = 0
    while (b - a) > xatol and iteration < maxiter:
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - invphi * (b - a)
            fc = phi(c)
        else:
            a, c, fc = c, d, fd
            d = a + invphi * (b - a)
            fd = phi(d)
        num_evals += 1
        iteration += 1

    if fc < fd:
        x, fx = c, fc
    else:
        x, fx = d, fd

    converged = iteration < maxiter and not (np.isnan(x) or np.isnan(fx))
    return ScalarSearchResult(
        alpha=float(x), f_min=float(fx), num_evals=num_evals, converged=converged
    )
