#!/usr/bin/env python3
"""
Bootstrap confidence intervals for Expected Shortfall (ES).

Provides IID percentile bootstrap and Moving-Block Bootstrap (MBB).
Deterministic: uses a local RandomState seeded by `random_state`.
"""

from typing import Optional, Tuple

import numpy as np


def _moving_block_bootstrap_indices(T: int, block_size: int, B: int, rng: np.random.RandomState) -> np.ndarray:
    """Generate indices for Moving-Block Bootstrap (overlapping blocks)."""
    if block_size <= 0:
        raise ValueError("block_size must be positive for moving-block bootstrap")
    n_blocks = int(np.ceil(T / block_size))
    starts = rng.randint(0, T - block_size + 1, size=(B, n_blocks))
    idx = np.concatenate([s[:, None] + np.arange(block_size)[None, :] for s in starts], axis=1)
    # Wrap indices to length T
    return idx[:, :T]


def bootstrap_es_ci(
    x: np.ndarray,
    level: float,
    B: int = 1000,
    block_size: Optional[int] = None,
    ci: float = 0.95,
    random_state: int = 42,
) -> Tuple[float, float, float]:
    """
    Bootstrap ES point estimate and percentile CI.

    Args:
        x: 1-D array of returns (percent or decimal, consistent with ES definition).
        level: Confidence level for VaR/ES (e.g., 0.95, 0.99). ES computed over x[x <= VaR(level)].
        B: Number of bootstrap replicates (default 1000).
        block_size: If None, IID bootstrap; otherwise Moving-Block Bootstrap length.
        ci: Confidence level for CI (default 0.95 percentile interval).
        random_state: Seed for deterministic resampling.

    Returns:
        (es_point, ci_low, ci_high)
    """
    if x.ndim != 1:
        raise ValueError("x must be a 1-D array")
    if not (0.0 < level < 1.0):
        raise ValueError("level must be in (0,1)")
    if not (0.0 < ci < 1.0):
        raise ValueError("ci must be in (0,1)")
    if B <= 0:
        raise ValueError("B must be positive")

    x = np.asarray(x)
    T = x.shape[0]

    # ES point estimate
    var = np.percentile(x, (1 - level) * 100.0)
    tail = x[x <= var]
    es_point = float(np.mean(tail)) if tail.size > 0 else np.nan

    # Bootstrap replicates
    rng = np.random.RandomState(random_state)
    es_boot = np.empty(B, dtype=float)

    if block_size is None:
        # IID percentile bootstrap
        for b in range(B):
            sample = x[rng.randint(0, T, size=T)]
            v = np.percentile(sample, (1 - level) * 100.0)
            t = sample[sample <= v]
            es_boot[b] = np.mean(t) if t.size > 0 else np.nan
    else:
        idx = _moving_block_bootstrap_indices(T, int(block_size), B, rng)
        for b in range(B):
            sample = x[idx[b]]
            v = np.percentile(sample, (1 - level) * 100.0)
            t = sample[sample <= v]
            es_boot[b] = np.mean(t) if t.size > 0 else np.nan

    # Percentile CI with NaN-safe handling
    valid = es_boot[np.isfinite(es_boot)]
    if valid.size == 0:
        return es_point, np.nan, np.nan

    alpha = (1.0 - ci) * 100.0
    lo, hi = np.percentile(valid, [alpha / 2.0, 100.0 - alpha / 2.0])
    return es_point, float(lo), float(hi)


__all__ = ["bootstrap_es_ci"]


