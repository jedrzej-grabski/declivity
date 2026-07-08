"""Literal Python port of ``nm-cma-es-vectorized.R``.

The R source ships as
``thesis-paper/nm-cma-es-vectorized.R`` (a re-export of
``declivity/reference/mf_cmaes.r``).  This module mirrors that function
line-by-line so it can be audited against the R original by a reader
holding both files side-by-side.

Like :mod:`des_reference`, the port is framework-free — no
:class:`PopulationOptimizer` base, no logger, no constraint-handler
injection.  Variable names follow R (``xmean``, ``arx``, ``vx``,
``d_history``, ``p_history``, ``mueff``, ``cs``…) instead of PEP-8.

Parity with the R source is **distributional**: NumPy's RNG cannot be
threaded into R's, so two instances of this port started with the same
NumPy seed produce identical traces, and that is the parity claim used
by :mod:`mfcmaes_vs_reference`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np
from numpy.typing import NDArray


SigmaUpdater = Literal["identity", "ppmf"]


@dataclass
class MFCMAESReferenceResult:
    best_par: NDArray[np.float64]
    best_fit: float
    counteval: int
    iter_count: int
    iter_best: list[float] = field(default_factory=list)
    iter_evals: list[int] = field(default_factory=list)
    sigma_history: list[float] = field(default_factory=list)
    xmean_norm: list[float] = field(default_factory=list)
    pc_norm: list[float] = field(default_factory=list)
    p_succ_history: list[float] = field(default_factory=list)
    midpoint_history: list[float] = field(default_factory=list)
    message: str = ""


def nm_cma_es_vectorized(
    par: NDArray[np.float64],
    fn: Callable[[NDArray[np.float64]], float],
    *,
    lower: NDArray[np.float64] | float = -100.0,
    upper: NDArray[np.float64] | float = 100.0,
    budget: int | None = None,
    sigma: float = 1.0,
    stopfitness: float = 1e-8,
    keep_best: bool = True,
    terminate_stopfitness: bool = True,
    terminate_maxiter: bool = True,
    do_flatland_escape: bool = True,
    lambda_: int | None = None,
    mu: int | None = None,
    window: int | None = None,
    sigma_updater: SigmaUpdater = "ppmf",
    damps_ppmf: float = 2.0,
    p_target_ppmf: float = 0.1,
    fnscale: float = 1.0,
    seed: int | np.random.Generator | None = None,
) -> MFCMAESReferenceResult:
    """Port of ``nm_cma_es_vectorized`` from
    ``nm-cma-es-vectorized.R`` (lines 19–269).

    The signature exposes the same ``control`` parameters R supports as
    keyword-only arguments.  Defaults match R's ``controlParam`` fallbacks.
    """
    par = np.asarray(par, dtype=np.float64).copy()
    xmean = par.copy()
    N = xmean.size

    lower_v = np.full(N, lower, dtype=np.float64) if np.isscalar(lower) else np.asarray(lower, dtype=np.float64)
    upper_v = np.full(N, upper, dtype=np.float64) if np.isscalar(upper) else np.asarray(upper, dtype=np.float64)

    if budget is None:
        budget = 10_000 * N
    # Strategy parameters (R lines 48–69).
    if lambda_ is None:
        lambda_ = 4 * N
    maxiter = int(round(budget / lambda_))
    if mu is None:
        mu = lambda_ // 2

    # Uniform weights (R line 51): ``weights = rep(1, mu)``, then normalised.
    weights = np.ones(mu) / mu
    mueff = (weights.sum() ** 2) / (weights ** 2).sum()

    cc = 4.0 / (N + 4.0)
    cs = (mueff + 2.0) / (N + mueff + 3.0)
    mucov = mueff
    ccov = (
        (1.0 / mucov) * 2.0 / (N + 1.4) ** 2
        + (1.0 - 1.0 / mucov)
        * ((2.0 * mucov - 1.0) / ((N + 2.0) ** 2 + 2.0 * mucov))
    )

    c_mu = ccov * (1.0 - 1.0 / mucov)
    c_1 = ccov - c_mu

    damps = 1.0 + 2.0 * max(0.0, math.sqrt((mueff - 1.0) / (N + 1.0)) - 1.0) + cs
    if window is None:
        window = int(math.floor(1.4 * N)) + 20

    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)

    # Bookkeeping (R lines 82–104).
    best_fit = math.inf
    best_par: NDArray[np.float64] = xmean.copy()
    pc = np.zeros(N)
    iter_count = 0
    counteval = 0
    cviol = 0
    msg = ""

    # Precompute decay table (R lines 106–107).
    t_axis = np.arange(1, maxiter + 1)
    decay_table = (1.0 - ccov) ** ((t_axis - 1) / 2.0)

    # Archives (R lines 109–123).
    p_history = np.zeros((N, window))
    d_history = np.zeros((N, window * mu))

    def p_index(t: int) -> int:
        return (t - 1) % window

    def d_range(t: int) -> tuple[int, int]:
        start = p_index(t) * mu
        return start, start + mu

    def shift_arr(v: NDArray[np.float64], n: int) -> NDArray[np.float64]:
        n = n % len(v)
        if n == 0:
            return v.copy()
        return np.concatenate([v[-n:], v[:-n]])

    # PPMF state (R lines 126–127): ``sign(stopfitness) * Inf`` resolves to
    # +Inf for the default stopfitness=1e-8.
    sign = 1.0 if stopfitness >= 0 else -1.0
    prev_midpoint_fitness = sign * math.inf
    midpoint_fitness = sign * math.inf

    result = MFCMAESReferenceResult(best_par=best_par, best_fit=best_fit, counteval=0, iter_count=0)

    while counteval < budget:
        iter_count += 1
        t = iter_count

        # Per-iteration decay vector (R lines 136–138).  ``decay_table`` is
        # length maxiter; we take the first ``window`` entries reversed,
        # then cyclically shift by ``t - 1``.
        decay = decay_table[:window][::-1].copy()  # length=window
        decay = shift_arr(decay, t - 1)
        decay_rep = np.repeat(decay, mu)  # length = window * mu
        w_repeat = np.repeat(np.sqrt(weights), window)  # length = mu * window
        # R: ``rep(sqrt(weights), each = window)`` — each weight repeated
        # ``window`` times.  numpy ``np.repeat`` with default axis matches.

        # Late-iter underflow in the decay table makes the rank-1 and
        # rank-mu matmuls multiply denormalised numbers; numpy reports a
        # spurious "divide by zero" RuntimeWarning even though the
        # result is mathematically well-defined (zero).  Suppress just
        # for these accumulations.
        with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
            # rank-mu term (R lines 141–143).  Note R lays out r_mu as
            # (window*mu, lambda) and right-multiplies the (N, window*mu)
            # archive — we keep that exact layout.
            r_mu = rng.standard_normal((window * mu, lambda_))
            inner_sum = d_history @ (r_mu * decay_rep[:, None] * w_repeat[:, None])
            rank_mu = math.sqrt(c_mu) * inner_sum  # (N, lambda)

            # rank-1 term (R lines 145–146).
            r_1 = rng.standard_normal((window, lambda_))
            rank_1 = math.sqrt(c_1) * (p_history @ (r_1 * decay[:, None]))

        outer_sum = rank_mu + rank_1

        # Residual decay term (R lines 149–155).
        if t <= window:
            last_decay = float(decay_table[t - 1])
        else:
            last_decay = math.sqrt(
                (1.0 - ccov) ** (t - 1)
                + ((1.0 - ccov) ** window) * (1.0 - ccov) ** (t - window - 1)
            )
        r_last = rng.standard_normal((N, lambda_))
        last_term = last_decay * r_last
        d = outer_sum + last_term
        arx = xmean[:, None] + sigma * d  # (N, lambda)

        # Clamp repair + quadratic penalty (R lines 160–165).
        vx = np.where(arx > lower_v[:, None], np.where(arx < upper_v[:, None], arx, upper_v[:, None]), lower_v[:, None])
        pen = 1.0 + np.sum((arx - vx) ** 2, axis=0)
        pen = np.where(np.isfinite(pen), pen, np.finfo(np.float64).max / 2.0)
        cviol += int(np.sum(pen > 1.0))

        # Evaluate (R lines 167–172).
        y = np.array([fn(vx[:, k]) for k in range(lambda_)]) * fnscale
        counteval += lambda_

        if not keep_best:
            best_fit = math.inf

        arfitness = y * pen
        valid_mask = pen <= 1.0
        if np.any(valid_mask):
            valid_y = y[valid_mask]
            wb = int(np.argmin(valid_y))
            if valid_y[wb] < best_fit:
                best_fit = float(valid_y[wb])
                valid_arx = arx[:, valid_mask]
                best_par = valid_arx[:, wb].copy()

        # Selection (R lines 190–196).
        arindex = np.argsort(arfitness)
        arfitness_sorted = arfitness[arindex]
        aripop = arindex[:mu]
        selx = arx[:, aripop]
        xmean = selx @ weights
        seld = d[:, aripop]
        dmean = seld @ weights

        # Path update (R line 205).
        pc = (1.0 - cc) * pc + math.sqrt(cc * (2.0 - cc) * mueff) * dmean

        # Archive update (R lines 207–208).
        ds, de = d_range(t)
        d_history[:, ds:de] = d[:, aripop]
        p_history[:, p_index(t)] = pc

        # Sigma update (R line 211).
        p_succ_iter = 0.0
        if sigma_updater == "identity":
            sigma_new = sigma
        elif sigma_updater == "ppmf":
            # PPMF policy from sigma_updaters.R lines 54–67.
            prev_midpoint_fitness = midpoint_fitness
            mean_point = vx.mean(axis=1)  # rowMeans(vx) = midpoint of repaired pop
            midpoint_fitness = float(fn(mean_point))
            counteval += 1
            p_succ_iter = float(np.sum(arfitness < prev_midpoint_fitness)) / lambda_
            sigma_new = sigma * math.exp(
                damps_ppmf * (p_succ_iter - p_target_ppmf) / (1.0 - p_target_ppmf)
            )
        else:
            raise ValueError(f"Unknown sigma_updater: {sigma_updater}")
        sigma = sigma_new

        # Record trajectory.
        result.iter_best.append(best_fit)
        result.iter_evals.append(counteval)
        result.sigma_history.append(sigma)
        result.xmean_norm.append(float(np.linalg.norm(xmean)))
        result.pc_norm.append(float(np.linalg.norm(pc)))
        result.p_succ_history.append(p_succ_iter)
        result.midpoint_history.append(midpoint_fitness)

        # Termination (R lines 222–230).
        if terminate_stopfitness and arfitness_sorted[0] <= stopfitness * fnscale:
            msg = "Stop fitness reached."
            break
        if terminate_maxiter and iter_count >= maxiter:
            msg = "Exceeded maximal number of iterations."
            break

        # Flatland escape (R lines 232–238).
        if do_flatland_escape:
            cmp_idx = min(1 + lambda_ // 2, 2 + math.ceil(lambda_ / 4))
            # R uses 1-indexed; arfitness_sorted is sorted ascending, so
            # arfitness[1] == arfitness[cmp_idx] (R 1-indexed) means the
            # lambda/2-th and best are tied — a flat patch.
            if cmp_idx <= lambda_ and arfitness_sorted[0] == arfitness_sorted[cmp_idx - 1]:
                sigma = sigma * math.exp(0.2 + cs / damps)

    if not msg:
        msg = "Budget exhausted."

    result.best_par = best_par
    result.best_fit = best_fit
    result.counteval = counteval
    result.iter_count = iter_count
    result.message = msg
    return result
