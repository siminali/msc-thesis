#!/usr/bin/env python3
"""
Diebold–Mariano (DM) test utilities.

Implements the Diebold–Mariano test for equal predictive accuracy with
Newey–West heteroskedasticity- and autocorrelation-consistent variance
estimation and the small-sample correction of Harvey–Leybourne–Newbold (HLN).

Usage:
    from utils.dm import diebold_mariano

Notes:
- Deterministic: no randomness is used.
- Supports alternatives: "two-sided", "less", "greater".
"""

from typing import Tuple

import numpy as np
from scipy.stats import t as student_t


def _newey_west_variance(diff: np.ndarray, lag: int) -> float:
    """
    Newey–West HAC variance estimator for the mean of a series.

    Var(\bar{d}) ≈ (1/T) * (\gamma_0 + 2 * sum_{k=1..lag} w_k * \gamma_k)
    where w_k are Bartlett weights: w_k = 1 - k/(lag+1).
    """
    T = diff.shape[0]
    if T <= 1:
        return np.nan

    d = diff - np.mean(diff)
    gamma0 = np.dot(d, d) / T

    if lag <= 0:
        return gamma0 / T

    var = gamma0
    for k in range(1, min(lag, T - 1) + 1):
        w = 1.0 - k / (lag + 1.0)
        cov = np.dot(d[k:], d[:-k]) / T
        var += 2.0 * w * cov
    return float(var / T)


def _quantile_loss(y: np.ndarray, q: np.ndarray, alpha: float) -> np.ndarray:
    """Check (pinball) loss series for quantile level alpha."""
    u = y - q
    return u * (alpha - (u < 0).astype(float))


def diebold_mariano(
    loss1: np.ndarray,
    loss2: np.ndarray,
    h: int = 1,
    alternative: str = "two-sided",
    small_sample: bool = True,
) -> Tuple[float, float]:
    """
    Diebold–Mariano (DM) test for equal predictive accuracy.

    Args:
        loss1: Loss series from model 1 (shape [T]).
        loss2: Loss series from model 2 (shape [T]).
        h: Forecast horizon (>=1). Sets Newey–West lag to max(h-1, 0).
        alternative: "two-sided" (default), "less", or "greater" for E[d] relation,
                     where d_t = loss1_t - loss2_t.
        small_sample: If True, applies Harvey–Leybourne–Newbold small-sample correction.

    Returns:
        (dm_statistic, p_value)
    """
    if loss1.ndim != 1 or loss2.ndim != 1:
        raise ValueError("loss1 and loss2 must be 1-D arrays")
    if loss1.shape[0] != loss2.shape[0]:
        raise ValueError("loss1 and loss2 must have the same length")
    if h < 1:
        raise ValueError("h must be >= 1")

    T = loss1.shape[0]
    if T < 5:
        return np.nan, np.nan

    d = loss1 - loss2
    d_bar = float(np.mean(d))

    # Newey–West variance with lag = h-1
    nw_lag = max(h - 1, 0)
    var_dbar = _newey_west_variance(d, nw_lag)
    if not np.isfinite(var_dbar) or var_dbar <= 0:
        return np.nan, np.nan

    dm = d_bar / np.sqrt(var_dbar)

    # HLN small-sample correction (Harvey, Leybourne, Newbold, 1997)
    if small_sample:
        # Commonly-used finite-sample scaling factor
        c = np.sqrt((T + 1 - 2 * h + (h * (h - 1)) / T) / T)
        if np.isfinite(c) and c > 0:
            dm = dm * c

    # p-value depending on alternative
    # Use t_{T-1} approximation (robust) rather than standard normal
    df = max(T - 1, 1)
    if alternative == "two-sided":
        p = 2.0 * (1.0 - student_t.cdf(abs(dm), df=df))
    elif alternative == "less":
        p = student_t.cdf(dm, df=df)
    elif alternative == "greater":
        p = 1.0 - student_t.cdf(dm, df=df)
    else:
        raise ValueError("alternative must be one of {'two-sided','less','greater'}")

    return float(dm), float(p)


__all__ = ["diebold_mariano", "_quantile_loss"]


