#!/usr/bin/env python3
"""Statistical utilities for model evaluation and comparison."""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error


def forecast_errors(y_true, y_pred) -> Dict[str, float]:
    """Return mean error, MAE, MSE, RMSE and the error vector."""
    errors = y_true - y_pred
    
    return {
        'mean_error': float(np.mean(errors)),
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'mse': float(mean_squared_error(y_true, y_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'errors': errors
    }


def ks_ad(real, model) -> Dict[str, float]:
    """Compute KS and AD statistics and p-values."""
    # KS test
    ks_stat, ks_pvalue = stats.ks_2samp(real, model)
    
    # Anderson-Darling test
    try:
        ad_stat, ad_critical_values, ad_significance_levels = stats.anderson_ksamp([real, model])
        ad_pvalue = None  # Anderson-Darling doesn't provide p-values directly
    except:
        ad_stat = np.nan
        ad_pvalue = np.nan
    
    return {
        'ks_statistic': float(ks_stat),
        'ks_pvalue': float(ks_pvalue),
        'ad_statistic': float(ad_stat) if not np.isnan(ad_stat) else np.nan,
        'ad_pvalue': ad_pvalue
    }


def mmd_rbf_u(x, y, gamma=None) -> float:
    """Compute MMD with RBF kernel using unbiased estimator."""
    if gamma is None:
        # Use median heuristic for gamma
        x_median = np.median(np.linalg.norm(x, axis=1))
        y_median = np.median(np.linalg.norm(y, axis=1))
        gamma = 1.0 / (0.5 * (x_median**2 + y_median**2))
    
    # Compute kernel matrices
    def rbf_kernel(x1, x2):
        dist_sq = np.sum((x1[:, np.newaxis, :] - x2[np.newaxis, :, :])**2, axis=2)
        return np.exp(-gamma * dist_sq)
    
    k_xx = rbf_kernel(x, x)
    k_yy = rbf_kernel(y, y)
    k_xy = rbf_kernel(x, y)
    
    # Unbiased MMD estimator
    n, m = len(x), len(y)
    mmd_sq = (np.sum(k_xx) - np.sum(np.diag(k_xx))) / (n * (n - 1)) + \
              (np.sum(k_yy) - np.sum(np.diag(k_yy))) / (m * (m - 1)) - \
              2 * np.mean(k_xy)
    
    return float(mmd_sq)


def mmd_permutation_p(x, y, B=1000, seed=42, gamma=None) -> Dict[str, float]:
    """Compute MMD with RBF kernel and permutation test p-value."""
    np.random.seed(seed)
    
    # Compute observed MMD
    mmd_obs = mmd_rbf_u(x, y, gamma)
    
    # Permutation test
    n, m = len(x), len(y)
    combined = np.vstack([x, y])
    mmd_perm = []
    
    for _ in range(B):
        np.random.shuffle(combined)
        x_perm = combined[:n]
        y_perm = combined[n:]
        mmd_perm.append(mmd_rbf_u(x_perm, y_perm, gamma))
    
    # Compute p-value
    p_value = np.mean(np.array(mmd_perm) >= mmd_obs)
    
    return {
        'mmd_statistic': mmd_obs,
        'p_value': float(p_value),
        'n_permutations': B
    }

