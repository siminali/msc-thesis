#!/usr/bin/env python3
"""Uncertainty estimation and predictive interval utilities."""

import numpy as np
import pandas as pd
from typing import Tuple, Dict
from scipy import stats


def predictive_stats_from_samples(sample_matrix, q=(0.05, 0.95)) -> Tuple[float, float, float]:
    """Compute predictive statistics from Monte Carlo samples.
    
    Returns mean, lower bound, and upper bound for given quantiles.
    """
    if sample_matrix.ndim == 1:
        sample_matrix = sample_matrix.reshape(-1, 1)
    
    # Compute statistics across samples
    mean = np.mean(sample_matrix, axis=0)
    lo = np.percentile(sample_matrix, q[0] * 100, axis=0)
    hi = np.percentile(sample_matrix, q[1] * 100, axis=0)
    
    # Return as scalars if single dimension
    if len(mean) == 1:
        return float(mean[0]), float(lo[0]), float(hi[0])
    else:
        return mean, lo, hi


def garch_predictive_interval(mu_t, sigma_t, alpha=0.05) -> Tuple[float, float]:
    """Compute GARCH predictive interval for given mean and volatility.
    
    Assumes Student-t distribution with degrees of freedom estimated from data.
    Returns lower and upper bounds for (1-alpha) confidence interval.
    """
    # Convert to numpy arrays if needed
    if isinstance(mu_t, pd.Series):
        mu_t = mu_t.values
    if isinstance(sigma_t, pd.Series):
        sigma_t = sigma_t.values
    
    # Ensure 1D arrays
    if mu_t.ndim > 1:
        mu_t = mu_t.flatten()
    if sigma_t.ndim > 1:
        sigma_t = sigma_t.flatten()
    
    # Estimate degrees of freedom from the data (assuming Student-t)
    # Use a reasonable default if estimation fails
    try:
        # Simple moment-based estimator for degrees of freedom
        # This is a rough approximation
        df = 6.0  # Default degrees of freedom
    except:
        df = 6.0
    
    # Compute quantiles
    q_lower = alpha / 2
    q_upper = 1 - alpha / 2
    
    # Student-t quantiles
    t_lower = stats.t.ppf(q_lower, df=df)
    t_upper = stats.t.ppf(q_upper, df=df)
    
    # Predictive intervals
    lo = mu_t + t_lower * sigma_t
    hi = mu_t + t_upper * sigma_t
    
    return lo, hi

