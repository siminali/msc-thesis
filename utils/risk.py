#!/usr/bin/env python3
"""Risk metrics and VaR backtesting utilities."""

import numpy as np
import pandas as pd
from typing import Tuple, Dict
from scipy import stats


def var_es(x, alpha=0.01) -> Tuple[float, float]:
    """Return Value at Risk and Expected Shortfall at confidence level alpha."""
    var = np.percentile(x, alpha * 100)
    es = np.mean(x[x <= var])
    return float(var), float(es)


def kupiec_pvalue(n, n_viol, alpha) -> float:
    """Compute Kupiec POF test p-value."""
    if n_viol == 0:
        return 1.0
    
    # Likelihood ratio test statistic
    if n_viol == n:
        lr_stat = 2 * (n * np.log(alpha) + n * np.log(1 - alpha) - n * np.log(n_viol/n) - n * np.log(1 - n_viol/n))
    else:
        lr_stat = 2 * (n * np.log(alpha) + n * np.log(1 - alpha) - n_viol * np.log(n_viol/n) - (n - n_viol) * np.log(1 - n_viol/n))
    
    # Chi-square test with 1 degree of freedom
    p_value = 1 - stats.chi2.cdf(lr_stat, 1)
    return float(p_value)


def christoffersen_independence_pvalue(violations: np.ndarray) -> float:
    """Compute Christoffersen independence test p-value."""
    if len(violations) < 2:
        return 1.0
    
    # Count transitions
    n00 = n01 = n10 = n11 = 0
    
    for i in range(len(violations) - 1):
        if violations[i] == 0 and violations[i + 1] == 0:
            n00 += 1
        elif violations[i] == 0 and violations[i + 1] == 1:
            n01 += 1
        elif violations[i] == 1 and violations[i + 1] == 0:
            n10 += 1
        elif violations[i] == 1 and violations[i + 1] == 1:
            n11 += 1
    
    # Compute transition probabilities
    if n00 + n01 > 0:
        p01 = n01 / (n00 + n01)
    else:
        p01 = 0
    
    if n10 + n11 > 0:
        p11 = n11 / (n10 + n11)
    else:
        p11 = 0
    
    # Likelihood ratio test statistic
    if p01 == p11:
        lr_stat = 0
    else:
        lr_stat = 2 * (n01 * np.log(1 - p01) + n11 * np.log(p11) - 
                       n01 * np.log(1 - (n01 + n11) / (n00 + n01 + n10 + n11)) - 
                       n11 * np.log((n01 + n11) / (n00 + n01 + n10 + n11)))
    
    # Chi-square test with 1 degree of freedom
    p_value = 1 - stats.chi2.cdf(lr_stat, 1)
    return float(p_value)


def var_backtest(returns, var_level) -> Dict[str, float]:
    """Perform VaR backtest and return test statistics."""
    # Identify violations
    violations = (returns < var_level).astype(int)
    n = len(returns)
    n_viol = np.sum(violations)
    
    # Compute hit rate
    hit_rate = n_viol / n
    
    # Kupiec POF test
    kupiec_p = kupiec_pvalue(n, n_viol, 0.05)  # Assuming 5% VaR
    
    # Christoffersen independence test
    christoffersen_p = christoffersen_independence_pvalue(violations)
    
    # Conditional coverage test (combines both)
    cc_p = 1 - stats.chi2.cdf(-2 * (np.log(kupiec_p) + np.log(christoffersen_p)), 2)
    
    return {
        'n_observations': n,
        'n_violations': int(n_viol),
        'hit_rate': float(hit_rate),
        'expected_hit_rate': 0.05,  # Assuming 5% VaR
        'kupiec_pvalue': kupiec_p,
        'christoffersen_pvalue': christoffersen_p,
        'conditional_coverage_pvalue': float(cc_p)
    }

