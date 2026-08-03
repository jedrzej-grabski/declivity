"""Literal Python port of ``thesis-experiments/des_comparison/DES.R``.

This module is the cross-validation oracle for
:class:`~src.algorithms.des.des_optimizer.DESOptimizer`.  It is *not*
plugged into the framework — no :class:`PopulationOptimizer` base, no
logger, no constraint-handler injection.  Each block of code below
corresponds line-for-line with a block in ``DES.R`` so that the port can
be audited against the R source by a reader holding both files
side-by-side.

Key conventions kept from R for direct correspondence:

- Arrays are kept in ``(N, lambda)`` column-major layout (column = individual).
- Variable names match R (``Ft``, ``cp``, ``cc``, ``histHead``, ``muMean``…)
  rather than PEP-8.
- RNG draw order matches R: one ``rng.standard_normal()`` per
  ``dMean`` term and one per ``pc`` term, per individual, in the same order.
- Budget accounting matches R's ``fn_``: out-of-bounds points return a
  huge sentinel without consuming budget.

The mirror is **distributional**, not bit-identical: R's RNG state cannot
be threaded into NumPy.  Two instances of this port started with the
same NumPy seed produce identical traces; that is the parity claim used
by the cross-validation harness.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.special import gamma as gamma_fn

_DBL_MAX = float(np.finfo(np.float64).max)


@dataclass
class DESReferenceResult:
    best_par: NDArray[np.float64]
    best_fit: float
    counteval: int
    iter_count: int
    # Trajectories logged once per inner-loop iteration (after the iter
    # counter is bumped, before the new population is generated).
    iter_best: list[float] = field(default_factory=list)
    iter_evals: list[int] = field(default_factory=list)
    ft_history: list[float] = field(default_factory=list)
    new_mean_norm: list[float] = field(default_factory=list)
    pc_norm: list[float] = field(default_factory=list)
    pop_mean_norm: list[float] = field(default_factory=list)


def _bounce_back_boundary2(
    x: NDArray[np.float64],
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Mirror of ``bounceBackBoundary2`` in DES.R lines 32–43."""
    x = np.asarray(x, dtype=np.float64).copy()
    if np.all(x >= lower) and np.all(x <= upper):
        return x
    below = x < lower
    if np.any(below):
        for i in np.where(below)[0]:
            x[i] = lower[i] + abs(lower[i] - x[i]) % (upper[i] - lower[i])
    else:
        above = x > upper
        if np.any(above):
            for i in np.where(above)[0]:
                x[i] = upper[i] - abs(upper[i] - x[i]) % (upper[i] - lower[i])
    x = _delete_inf_nan(x)
    if np.all(x >= lower) and np.all(x <= upper):
        return x
    return _bounce_back_boundary2(x, lower, upper)


def _delete_inf_nan(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Mirror of ``deleteInfsNaNs`` (DES.R lines 413–417)."""
    x = np.asarray(x, dtype=np.float64).copy()
    x[~np.isfinite(x)] = _DBL_MAX
    return x


def des_reference(
    par: NDArray[np.float64],
    fn: Callable[[NDArray[np.float64]], float],
    *,
    lower: NDArray[np.float64] | float = -100.0,
    upper: NDArray[np.float64] | float = 100.0,
    budget: int | None = None,
    initFt: float = 1.0,
    lambda_: int | None = None,
    path_length: int = 6,
    cp: float | None = None,
    c_ft: float = 0.0,
    history_size: int | None = None,
    tol: float = 1e-12,
    stopfitness: float = -math.inf,
    lamarckian: bool = False,
    seed: int | np.random.Generator | None = None,
) -> DESReferenceResult:
    """Port of the ``DES`` function in ``DES.R`` (lines 1–404).

    The argument names follow the R signature; defaults match R's
    ``controlParam`` fallbacks.
    """
    par = np.asarray(par, dtype=np.float64).copy()
    N = par.size

    lower = (
        np.full(N, lower, dtype=np.float64)
        if np.isscalar(lower)
        else np.asarray(lower, dtype=np.float64)
    )
    upper = (
        np.full(N, upper, dtype=np.float64)
        if np.isscalar(upper)
        else np.asarray(upper, dtype=np.float64)
    )

    if budget is None:
        budget = 10_000 * N
    if lambda_ is None:
        lambda_ = 4 * N
    if cp is None:
        cp = 1.0 / math.sqrt(N)
    if history_size is None:
        history_size = math.ceil(6 + math.ceil(3 * math.sqrt(N)))

    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)

    # Derived params identical to DES.R lines 56–69 (block 'Algorithm parameters').
    mu = lambda_ // 2
    weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
    weights = weights / weights.sum()
    weightsPop = np.log(lambda_ + 1) - np.log(np.arange(1, lambda_ + 1))
    weightsPop = weightsPop / weightsPop.sum()
    mu_eff = weights.sum() ** 2 / (weights**2).sum()
    cc = mu / (mu + 2)  # DES.R line 61: 'ccum'
    sqrt_N = math.sqrt(N)

    counteval = 0
    best_fit = math.inf
    best_par: NDArray[np.float64] = par.copy()

    # Wrap fn to enforce bounds-feasibility and budget bookkeeping
    # (DES.R lines 91–98).
    def fn_(x: NDArray[np.float64]) -> float:
        nonlocal counteval
        if np.all(x >= lower) and np.all(x <= upper):
            counteval += 1
            return float(fn(x))
        return _DBL_MAX

    # Lamarckian fitness wrapper for matrices (DES.R lines 101–121).
    def fn_l(P: NDArray[np.float64]) -> NDArray[np.float64]:
        if P.ndim == 1:
            return np.array([fn_(P) if counteval < budget else _DBL_MAX])
        n_cols = P.shape[1]
        out = np.empty(n_cols, dtype=np.float64)
        for k in range(n_cols):
            if counteval >= budget:
                out[k] = _DBL_MAX
            else:
                out[k] = fn_(P[:, k])
        return out

    # Non-Lamarckian penalty (DES.R lines 124–144).  Only invoked when
    # ``lamarckian`` is False.
    def fn_d(
        P: NDArray[np.float64],
        P_repaired: NDArray[np.float64],
        fitness: NDArray[np.float64],
        worst_fit: float,
    ) -> NDArray[np.float64]:
        P = _delete_inf_nan(P)
        P_repaired = _delete_inf_nan(P_repaired)
        if P.ndim == 2:
            # R uses ``apply(P != P_repaired, 2, all)`` — TRUE iff every
            # coordinate differs.  Faithful to the R bug/feature.
            repaired_mask = np.all(P != P_repaired, axis=0)
            vecDist = np.sum((P - P_repaired) ** 2, axis=0)
            out = fitness.copy()
            out[repaired_mask] = worst_fit + vecDist[repaired_mask]
            return _delete_inf_nan(out)
        if np.any(P != P_repaired):
            return _delete_inf_nan(
                np.array([worst_fit + np.sum((P - P_repaired) ** 2)])
            )
        return fitness.copy()

    # Outer restart loop (DES.R line 188).  ``stoptol`` is never set in
    # the current R source (the trigger is commented out), so the outer
    # loop executes exactly once until the budget is exhausted.  Kept
    # here for literal correspondence.
    result = DESReferenceResult(
        best_par=best_par, best_fit=best_fit, counteval=0, iter_count=0
    )
    while counteval < budget:
        Ft = initFt

        # First population: R draws uniform per-individual in 80% of bounds
        # (DES.R line 202).  Note: ``par`` is NOT used for the first pop;
        # it shows up only in ``newMean = par`` below.
        population = rng.uniform(
            low=0.8 * lower[:, None],
            high=0.8 * upper[:, None],
            size=(N, lambda_),
        )
        cumMean = (upper + lower) / 2.0

        # Repair (column-wise bounce-back, DES.R line 205).
        populationRepaired = np.column_stack(
            [
                _bounce_back_boundary2(population[:, k], lower, upper)
                for k in range(lambda_)
            ]
        )
        if lamarckian:
            population = populationRepaired

        fitness = fn_l(population if lamarckian else population)  # DES.R line 213
        oldMean = np.zeros(N)
        newMean = par.copy()
        worst_fit = float(np.max(fitness))

        popMean = population @ weightsPop  # DES.R line 220
        muMean = newMean.copy()

        diffs = np.zeros((N, lambda_))
        chiN = math.sqrt(2.0) * float(gamma_fn((N + 1) / 2)) / float(gamma_fn(N / 2))
        histNorm = 1.0 / math.sqrt(2.0)
        counterRepaired = 0

        history: list[NDArray[np.float64]] = []
        FtHistory = np.zeros(history_size)
        pc = np.zeros((N, history_size))
        dMean = np.zeros((N, history_size))

        histHead = -1  # R uses 1-indexed cycling 1..histSize; we use 0..histSize-1.
        iter_count = 0
        stoptol = False

        # Inner loop (DES.R line 233).
        while counteval < budget and not stoptol:
            iter_count += 1
            histHead = (histHead + 1) % history_size

            # mu/weights recompute is a no-op given fixed lambda, but
            # mirrored for fidelity (DES.R lines 237–239).
            mu = lambda_ // 2
            weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
            weights = weights / weights.sum()

            # Select best mu individuals (DES.R lines 251–252).
            selection = np.argsort(fitness)[:mu]
            selectedPoints = population[:, selection]  # (N, mu)

            # Push into history (DES.R lines 255–256).
            entry = selectedPoints * (histNorm / Ft)
            if len(history) <= histHead:
                history.append(entry)
            else:
                history[histHead] = entry

            oldMean = newMean.copy()
            newMean = selectedPoints @ weights  # (N,)
            muMean = newMean.copy()
            dMean[:, histHead] = (muMean - popMean) / Ft
            step = (newMean - oldMean) / Ft

            FtHistory[histHead] = Ft
            # Ft adaptation block (DES.R lines 271–273) is commented out
            # in R — Ft stays at initFt for the entire run.  Not ported.

            # Evolution path update (DES.R lines 276–279).
            if histHead == 0:
                pc[:, histHead] = (1 - cp) * np.zeros(N) / sqrt_N + math.sqrt(
                    mu * cp * (2 - cp)
                ) * step
            else:
                pc[:, histHead] = (1 - cp) * pc[:, histHead - 1] + math.sqrt(
                    mu * cp * (2 - cp)
                ) * step

            # Sample from history (DES.R lines 282–287).
            limit = histHead + 1 if iter_count < history_size else history_size
            # R: ``historySample <- sample(1:limit, lambda, T)``.  Python: rng.integers.
            historySample = rng.integers(0, limit, size=lambda_)
            historySample2 = rng.integers(0, limit, size=lambda_)
            # ``sampleFromHistory`` (DES.R lines 406–411): for each i,
            # sample(1:ncol(history[[historySample[i]]]), 1).  ncol == mu.
            x1sample = np.array(
                [
                    rng.integers(0, history[historySample[i]].shape[1])
                    for i in range(lambda_)
                ]
            )
            # Note: R uses ``historySample`` (not ``historySample2``) as
            # the slot index for the second draw too (DES.R line 287).
            x2sample = np.array(
                [
                    rng.integers(0, history[historySample[i]].shape[1])
                    for i in range(lambda_)
                ]
            )

            # Diff construction (DES.R lines 290–296).
            for i in range(lambda_):
                slot1 = historySample[i]
                slot2 = historySample2[i]
                x1 = history[slot1][:, x1sample[i]]
                x2 = history[slot1][:, x2sample[i]]
                diffs[:, i] = (
                    math.sqrt(cc)
                    * ((x1 - x2) + rng.standard_normal() * dMean[:, slot1])
                    + math.sqrt(1 - cc) * rng.standard_normal() * pc[:, slot2]
                )

            # New population (DES.R line 299).
            noise = rng.standard_normal(size=(N, lambda_))
            population = (
                newMean[:, None]
                + Ft * diffs
                + tol * (1 - 2 / (N**2)) ** (iter_count / 2) * noise / chiN
            )
            population = _delete_inf_nan(population)

            # Repair + count (DES.R lines 304–311).
            populationTemp = population
            populationRepaired = np.column_stack(
                [
                    _bounce_back_boundary2(population[:, k], lower, upper)
                    for k in range(lambda_)
                ]
            )
            counterRepaired = int(
                np.sum(
                    [
                        np.any(populationTemp[:, t] != populationRepaired[:, t])
                        for t in range(lambda_)
                    ]
                )
            )

            if lamarckian:
                population = populationRepaired

            popMean = population @ weightsPop  # DES.R line 317

            fitness = fn_l(
                population
            )  # DES.R line 320; on the original pop in nonLamarckian
            if not lamarckian:
                fitnessNonLamarckian = fn_d(
                    population, populationRepaired, fitness, worst_fit
                )
            else:
                fitnessNonLamarckian = fitness

            # Track best (DES.R lines 327–333).
            wb = int(np.argmin(fitness))
            if fitness[wb] < best_fit:
                best_fit = float(fitness[wb])
                best_par = (
                    population[:, wb] if lamarckian else populationRepaired[:, wb]
                ).copy()

            # Track worst (DES.R lines 336–339).
            ww = int(np.argmax(fitness))
            if fitness[ww] > worst_fit:
                worst_fit = float(fitness[ww])

            if not lamarckian:
                fitness = fitnessNonLamarckian

            # Check the cumMean (DES.R lines 348–355).
            cumMean = 0.8 * cumMean + 0.2 * newMean
            cumMeanRepaired = _bounce_back_boundary2(cumMean, lower, upper)
            fn_cum = float(fn_l(cumMeanRepaired)[0])
            if fn_cum < best_fit:
                best_fit = fn_cum
                best_par = cumMeanRepaired.copy()

            # Record trajectory.
            result.iter_best.append(best_fit)
            result.iter_evals.append(counteval)
            result.ft_history.append(Ft)
            result.new_mean_norm.append(float(np.linalg.norm(newMean)))
            result.pc_norm.append(float(np.linalg.norm(pc[:, histHead])))
            result.pop_mean_norm.append(float(np.linalg.norm(popMean)))

            if fitness[0] <= stopfitness:
                break

    result.best_par = best_par
    result.best_fit = best_fit
    result.counteval = counteval
    result.iter_count = iter_count
    return result
