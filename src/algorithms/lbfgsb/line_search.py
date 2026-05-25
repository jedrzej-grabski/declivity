"""
Line search implementations for L-BFGS-B.

Provides the More-Thuente line search (port of the Fortran dcsrch/dcstep
subroutines by More and Thuente, 1994) and a simple Armijo backtracking search.

References:
    J.J. More and D.J. Thuente, "Line Search Algorithms with Guaranteed
    Sufficient Decrease", ACM Trans. Math. Software 20 (1994), pp. 286-307.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np

from src.algorithms.lbfgsb.config import LineSearchMethod


@dataclass
class LineSearchResult:
    """Result of a line search."""

    step: float
    f_new: float
    dphi_new: float
    num_evals: int
    converged: bool


def perform_line_search(
    method: LineSearchMethod,
    phi_dphi: Callable[[float], tuple[float, float]],
    stp0: float,
    phi0: float,
    dphi0: float,
    stpmax: float,
    ftol: float = 1e-3,
    gtol: float = 0.9,
    xtol: float = 0.1,
    maxiter: int = 20,
) -> LineSearchResult:
    """Dispatch to the selected line search method.

    Args:
        method: Line search algorithm to use.
        phi_dphi: Callable returning (phi(alpha), phi'(alpha)) at a given step.
        stp0: Initial step length.
        phi0: Function value at alpha = 0.
        dphi0: Directional derivative at alpha = 0 (must be negative).
        stpmax: Maximum step length for bound feasibility.
        ftol: Sufficient decrease parameter.
        gtol: Curvature condition parameter (More-Thuente only).
        xtol: Interval width tolerance (More-Thuente only).
        maxiter: Maximum trial evaluations.
    """
    if method == LineSearchMethod.MORE_THUENTE:
        return more_thuente_search(
            phi_dphi, stp0, phi0, dphi0, stpmax, ftol, gtol, xtol, maxiter
        )
    elif method == LineSearchMethod.ARMIJO:
        return armijo_search(phi_dphi, stp0, phi0, dphi0, stpmax, ftol, maxiter)
    else:
        raise ValueError(f"Unknown line search method: {method}")


def more_thuente_search(
    phi_dphi: Callable[[float], tuple[float, float]],
    step: float,
    phi0: float,
    dphi0: float,
    step_max: float,
    ftol: float = 1e-3,
    gtol: float = 0.9,
    xtol: float = 0.1,
    maxiter: int = 20,
) -> LineSearchResult:
    """More-Thuente line search for the strong Wolfe conditions.

    Finds a step alpha satisfying:
        f(x + alpha*d) <= f(x) + ftol * alpha * f'(x).d   (sufficient decrease)
        |f'(x + alpha*d).d| <= gtol * |f'(x).d|           (curvature condition)

    Uses a two-stage safeguarded interpolation scheme. Stage 1 operates on the
    modified function psi = f - f0 - ftol*alpha*f'0 to locate a region where
    the sufficient decrease condition holds with a non-negative derivative.
    Stage 2 directly minimizes f within the bracket.
    """
    step_min = 0.0
    extrapolation_lower = 1.1
    extrapolation_upper = 4.0
    bisection_threshold = 0.66

    if dphi0 >= 0:
        return LineSearchResult(
            step=0.0, f_new=phi0, dphi_new=dphi0, num_evals=0, converged=False
        )

    is_bracketed = False
    stage = 1
    interval_width = step_max - step_min
    previous_interval_width = 2.0 * interval_width

    # Interval of uncertainty endpoints
    best_step, best_f, best_deriv = 0.0, phi0, dphi0
    other_step, other_f, other_deriv = 0.0, phi0, dphi0

    num_evals = 0

    for _ in range(maxiter):
        if is_bracketed:
            trial_lower = min(best_step, other_step)
            trial_upper = max(best_step, other_step)
        else:
            trial_lower = best_step
            trial_upper = step + extrapolation_upper * (step - best_step)

        step = np.clip(step, step_min, step_max)

        if is_bracketed and (step <= trial_lower or step >= trial_upper):
            step = best_step + 0.5 * (other_step - best_step)
        if is_bracketed and (trial_upper - trial_lower <= xtol * trial_upper):
            step = best_step + 0.5 * (other_step - best_step)

        trial_f, trial_deriv = phi_dphi(step)
        num_evals += 1

        sufficient_decrease_threshold = phi0 + step * ftol * dphi0

        # Strong Wolfe convergence test
        if (
            trial_f <= sufficient_decrease_threshold
            and abs(trial_deriv) <= gtol * abs(dphi0)
        ):
            return LineSearchResult(
                step=step, f_new=trial_f, dphi_new=trial_deriv,
                num_evals=num_evals, converged=True,
            )

        if is_bracketed and trial_upper - trial_lower <= xtol * trial_upper:
            return LineSearchResult(
                step=step, f_new=trial_f, dphi_new=trial_deriv,
                num_evals=num_evals, converged=False,
            )
        if step == step_max and trial_f <= sufficient_decrease_threshold and trial_deriv <= dphi0:
            return LineSearchResult(
                step=step, f_new=trial_f, dphi_new=trial_deriv,
                num_evals=num_evals, converged=False,
            )
        if step == step_min and (trial_f > sufficient_decrease_threshold or trial_deriv >= dphi0):
            return LineSearchResult(
                step=step, f_new=trial_f, dphi_new=trial_deriv,
                num_evals=num_evals, converged=False,
            )

        # Stage transition: switch from modified function to direct minimization
        if (
            stage == 1
            and trial_f <= sufficient_decrease_threshold
            and trial_deriv >= min(ftol, gtol) * dphi0
        ):
            stage = 2

        if stage == 1:
            # Operate on the modified function psi = f - f0 - ftol*alpha*f'0
            modified_best_f = best_f - best_step * ftol * dphi0
            modified_other_f = other_f - other_step * ftol * dphi0
            modified_trial_f = trial_f - step * ftol * dphi0
            modified_best_deriv = best_deriv - ftol * dphi0
            modified_other_deriv = other_deriv - ftol * dphi0
            modified_trial_deriv = trial_deriv - ftol * dphi0

            (
                best_step, modified_best_f, modified_best_deriv,
                other_step, modified_other_f, modified_other_deriv,
                step, is_bracketed,
            ) = _safeguarded_step(
                best_step, modified_best_f, modified_best_deriv,
                other_step, modified_other_f, modified_other_deriv,
                step, modified_trial_f, modified_trial_deriv,
                is_bracketed, trial_lower, trial_upper,
            )

            best_f = modified_best_f + best_step * ftol * dphi0
            other_f = modified_other_f + other_step * ftol * dphi0
            best_deriv = modified_best_deriv + ftol * dphi0
            other_deriv = modified_other_deriv + ftol * dphi0
        else:
            (
                best_step, best_f, best_deriv,
                other_step, other_f, other_deriv,
                step, is_bracketed,
            ) = _safeguarded_step(
                best_step, best_f, best_deriv,
                other_step, other_f, other_deriv,
                step, trial_f, trial_deriv,
                is_bracketed, trial_lower, trial_upper,
            )

        # Bisection safeguard: force contraction if interval stalls
        if is_bracketed:
            if abs(other_step - best_step) >= bisection_threshold * previous_interval_width:
                step = best_step + 0.5 * (other_step - best_step)
            previous_interval_width = interval_width
            interval_width = abs(other_step - best_step)

    return LineSearchResult(
        step=step, f_new=trial_f, dphi_new=trial_deriv,
        num_evals=num_evals, converged=False,
    )


def _safeguarded_step(
    best_step: float, best_f: float, best_deriv: float,
    other_step: float, other_f: float, other_deriv: float,
    trial_step: float, trial_f: float, trial_deriv: float,
    is_bracketed: bool, step_lower: float, step_upper: float,
) -> tuple[float, float, float, float, float, float, float, bool]:
    """Compute a safeguarded trial step and update the interval of uncertainty.

    Port of the dcstep subroutine. Uses cubic and quadratic interpolation
    with safeguards to compute the next trial step length while maintaining
    a valid bracketing interval.

    Four cases based on function values and derivative signs at the trial point
    relative to the best point.
    """
    signed_derivative = trial_deriv * (best_deriv / abs(best_deriv)) if best_deriv != 0 else trial_deriv

    if trial_f > best_f:
        # Case 1: trial has higher function value, minimum is bracketed
        theta = 3.0 * (best_f - trial_f) / (trial_step - best_step) + best_deriv + trial_deriv
        scale = max(abs(theta), abs(best_deriv), abs(trial_deriv))
        gamma_squared = (theta / scale) ** 2 - (best_deriv / scale) * (trial_deriv / scale)
        gamma = scale * np.sqrt(max(0.0, gamma_squared))
        if trial_step < best_step:
            gamma = -gamma

        cubic_step = best_step + ((gamma - best_deriv) + theta) / (((gamma - best_deriv) + gamma) + trial_deriv) * (trial_step - best_step)
        quadratic_step = best_step + ((best_deriv / ((best_f - trial_f) / (trial_step - best_step) + best_deriv)) / 2.0) * (trial_step - best_step)

        if abs(cubic_step - best_step) < abs(quadratic_step - best_step):
            new_step = cubic_step
        else:
            new_step = cubic_step + (quadratic_step - cubic_step) / 2.0

        is_bracketed = True
        new_other_step, new_other_f, new_other_deriv = trial_step, trial_f, trial_deriv
        new_best_step, new_best_f, new_best_deriv = best_step, best_f, best_deriv

    elif signed_derivative < 0:
        # Case 2: lower function value, opposite derivative sign, minimum is bracketed
        theta = 3.0 * (best_f - trial_f) / (trial_step - best_step) + best_deriv + trial_deriv
        scale = max(abs(theta), abs(best_deriv), abs(trial_deriv))
        gamma_squared = (theta / scale) ** 2 - (best_deriv / scale) * (trial_deriv / scale)
        gamma = scale * np.sqrt(max(0.0, gamma_squared))
        if trial_step > best_step:
            gamma = -gamma

        cubic_step = trial_step + ((gamma - trial_deriv) + theta) / (((gamma - trial_deriv) + gamma) + best_deriv) * (best_step - trial_step)
        secant_step = trial_step + (trial_deriv / (trial_deriv - best_deriv)) * (best_step - trial_step)

        if abs(cubic_step - trial_step) > abs(secant_step - trial_step):
            new_step = cubic_step
        else:
            new_step = secant_step

        is_bracketed = True
        new_other_step, new_other_f, new_other_deriv = best_step, best_f, best_deriv
        new_best_step, new_best_f, new_best_deriv = trial_step, trial_f, trial_deriv

    elif abs(trial_deriv) < abs(best_deriv):
        # Case 3: lower function value, same derivative sign, magnitude decreased
        theta = 3.0 * (best_f - trial_f) / (trial_step - best_step) + best_deriv + trial_deriv
        scale = max(abs(theta), abs(best_deriv), abs(trial_deriv))
        gamma_squared = (theta / scale) ** 2 - (best_deriv / scale) * (trial_deriv / scale)
        gamma = scale * np.sqrt(max(0.0, gamma_squared))
        if trial_step > best_step:
            gamma = -gamma

        p = (gamma - trial_deriv) + theta
        q = (gamma + (best_deriv - trial_deriv)) + gamma
        r = p / q

        if r < 0 and gamma != 0:
            cubic_step = trial_step + r * (best_step - trial_step)
        elif trial_step > best_step:
            cubic_step = step_upper
        else:
            cubic_step = step_lower

        secant_step = trial_step + (trial_deriv / (trial_deriv - best_deriv)) * (best_step - trial_step)

        if is_bracketed:
            if abs(cubic_step - trial_step) < abs(secant_step - trial_step):
                new_step = cubic_step
            else:
                new_step = secant_step
            if trial_step > best_step:
                new_step = min(trial_step + 0.66 * (other_step - trial_step), new_step)
            else:
                new_step = max(trial_step + 0.66 * (other_step - trial_step), new_step)
        else:
            if abs(cubic_step - trial_step) > abs(secant_step - trial_step):
                new_step = cubic_step
            else:
                new_step = secant_step
            new_step = min(step_upper, new_step)
            new_step = max(step_lower, new_step)

        new_other_step, new_other_f, new_other_deriv = other_step, other_f, other_deriv
        new_best_step, new_best_f, new_best_deriv = trial_step, trial_f, trial_deriv

    else:
        # Case 4: lower function value, same derivative sign, magnitude not decreased
        if is_bracketed:
            theta = 3.0 * (trial_f - other_f) / (other_step - trial_step) + other_deriv + trial_deriv
            scale = max(abs(theta), abs(other_deriv), abs(trial_deriv))
            gamma_squared = (theta / scale) ** 2 - (other_deriv / scale) * (trial_deriv / scale)
            gamma = scale * np.sqrt(max(0.0, gamma_squared))
            if trial_step > other_step:
                gamma = -gamma
            new_step = trial_step + ((gamma - trial_deriv) + theta) / (((gamma - trial_deriv) + gamma) + other_deriv) * (other_step - trial_step)
        elif trial_step > best_step:
            new_step = step_upper
        else:
            new_step = step_lower

        new_other_step, new_other_f, new_other_deriv = other_step, other_f, other_deriv
        new_best_step, new_best_f, new_best_deriv = trial_step, trial_f, trial_deriv

    return (
        new_best_step, new_best_f, new_best_deriv,
        new_other_step, new_other_f, new_other_deriv,
        new_step, is_bracketed,
    )


def armijo_search(
    phi_dphi: Callable[[float], tuple[float, float]],
    step: float,
    phi0: float,
    dphi0: float,
    step_max: float,
    ftol: float = 1e-3,
    maxiter: int = 20,
    contraction_factor: float = 0.5,
) -> LineSearchResult:
    """Armijo backtracking line search.

    Finds a step alpha satisfying the sufficient decrease condition:
        f(x + alpha*d) <= f(x) + ftol * alpha * f'(x).d

    Each rejected trial is contracted by the given factor.
    """
    step = min(step, step_max)
    num_evals = 0

    for _ in range(maxiter):
        trial_f, trial_deriv = phi_dphi(step)
        num_evals += 1

        if trial_f <= phi0 + ftol * step * dphi0:
            return LineSearchResult(
                step=step, f_new=trial_f, dphi_new=trial_deriv,
                num_evals=num_evals, converged=True,
            )

        step *= contraction_factor

        if step < 1e-20:
            return LineSearchResult(
                step=step, f_new=trial_f, dphi_new=trial_deriv,
                num_evals=num_evals, converged=False,
            )

    return LineSearchResult(
        step=step, f_new=trial_f, dphi_new=trial_deriv,
        num_evals=num_evals, converged=False,
    )
