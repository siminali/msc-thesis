#!/usr/bin/env python3
"""Unit conversion utilities for financial data."""

import pandas as pd
import numpy as np


def ensure_units(series, target_units: str) -> pd.Series:
    """Ensure returns are in either percent or decimal form.
    
    If target is percent and median absolute value < 0.02, multiply by 100.
    If target is decimal and median absolute value > 0.5, divide by 100.
    Else return unchanged.
    """
    if target_units not in ['percent', 'decimal']:
        raise ValueError("target_units must be 'percent' or 'decimal'")
    
    median_abs = np.median(np.abs(series))
    
    if target_units == 'percent' and median_abs < 0.02:
        return series * 100
    elif target_units == 'decimal' and median_abs > 0.5:
        return series / 100
    else:
        return series

