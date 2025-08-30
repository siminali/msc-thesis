"""
Single source of truth for inverse scaling and data validation.
Ensures all data flows through proper inverse-scaling pipeline with consistent units.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Union, Tuple, Any, Optional
from functools import wraps
import warnings


@dataclass
class ReturnsBundle:
    """Container for properly inverse-scaled returns with metadata."""
    returns: np.ndarray
    mean: float
    std: float
    min: float
    max: float
    kurtosis: float
    used_scaler_name: str
    output_kind: str  # "returns", "volatility", etc.
    annualise_mode: str  # "none", "sqrt252"
    provenance: str = "inverse_scaled"
    
    def __post_init__(self):
        """Validate bundle consistency."""
        if not isinstance(self.returns, np.ndarray):
            self.returns = np.array(self.returns)
        
        # Ensure 1D
        if self.returns.ndim > 1:
            self.returns = self.returns.flatten()
        
        # Recompute stats to ensure consistency
        self.mean = float(np.mean(self.returns))
        self.std = float(np.std(self.returns, ddof=1))
        self.min = float(np.min(self.returns))
        self.max = float(np.max(self.returns))
        
        # Compute kurtosis (scipy's definition: excess kurtosis + 3)
        from scipy.stats import kurtosis
        self.kurtosis = float(kurtosis(self.returns, fisher=False, bias=False))


def detect_scaler(data: np.ndarray, verbose: bool = True) -> str:
    """
    Detect scaling of financial returns data.
    
    Returns:
        str: "percent", "decimal", "log", "prices", "unknown"
    """
    data = np.array(data).flatten()
    data_clean = data[~np.isnan(data)]
    
    if len(data_clean) == 0:
        return "unknown"
    
    mean_abs = np.mean(np.abs(data_clean))
    std_val = np.std(data_clean, ddof=1)
    data_range = np.max(data_clean) - np.min(data_clean)
    
    if verbose:
        print(f"Scaler detection: mean_abs={mean_abs:.6f}, std={std_val:.6f}, range={data_range:.6f}")
    
    # Decimal returns: typical daily stock returns
    if 0.005 <= std_val <= 0.05 and mean_abs < 0.01:
        return "decimal"
    
    # Percent returns: 10x larger than decimal
    elif 0.5 <= std_val <= 5.0 and mean_abs < 1.0:
        return "percent"
    
    # Log returns: similar to decimal but different mean structure
    elif 0.005 <= std_val <= 0.05 and -0.01 <= np.mean(data_clean) <= 0.01:
        return "log"
    
    # Prices: large absolute values, small relative changes
    elif mean_abs > 10 and std_val / mean_abs < 0.1:
        return "prices"
    
    else:
        return "unknown"


def assert_fitted(scaler: Any) -> None:
    """Assert that a scaler has been fitted."""
    if hasattr(scaler, 'mean_') and scaler.mean_ is None:
        raise ValueError("Scaler not fitted")
    if hasattr(scaler, 'fitted_') and not scaler.fitted_:
        raise ValueError("Scaler not fitted")


def inverse_returns(data: np.ndarray, scaler: Any = None, 
                   scaler_name: str = "Identity", 
                   force_decimal: bool = True) -> np.ndarray:
    """
    Apply inverse transformation to get decimal returns.
    
    Args:
        data: Raw model outputs or scaled data
        scaler: Fitted scaler object (optional)
        scaler_name: Name of scaler for metadata
        force_decimal: Convert percent to decimal if detected
        
    Returns:
        np.ndarray: Decimal returns
    """
    data = np.array(data).flatten()
    
    # Apply scaler inverse if provided
    if scaler is not None:
        assert_fitted(scaler)
        if hasattr(scaler, 'inverse_transform'):
            data = scaler.inverse_transform(data.reshape(-1, 1)).flatten()
        elif hasattr(scaler, 'inverse'):
            data = scaler.inverse(data)
    
    # Detect current scaling and convert to decimal if needed
    detected_scale = detect_scaler(data, verbose=False)
    
    if force_decimal and detected_scale == "percent":
        print(f"Converting percent returns to decimal (dividing by 100)")
        data = data / 100.0
    elif detected_scale == "prices":
        # Convert prices to returns
        print(f"Converting prices to decimal returns")
        data = np.diff(np.log(data))
    
    return data


def inverse_volatility_or_variance(data: np.ndarray, scaler: Any = None,
                                  output_type: str = "volatility",
                                  annualise: str = "none") -> np.ndarray:
    """
    Inverse transform volatility or variance data.
    
    Args:
        data: Scaled volatility/variance data
        scaler: Fitted scaler
        output_type: "volatility" or "variance"
        annualise: "none", "sqrt252" for volatility, "252" for variance
        
    Returns:
        np.ndarray: Decimal volatility/variance
    """
    data = np.array(data).flatten()
    
    # Apply scaler inverse
    if scaler is not None:
        assert_fitted(scaler)
        if hasattr(scaler, 'inverse_transform'):
            data = scaler.inverse_transform(data.reshape(-1, 1)).flatten()
    
    # Convert variance to volatility if needed
    if output_type == "volatility" and "var" in str(type(scaler)).lower():
        data = np.sqrt(np.maximum(data, 0))
    
    # Annualise if requested
    if annualise == "sqrt252" and output_type == "volatility":
        data = data * np.sqrt(252)
    elif annualise == "252" and output_type == "variance":
        data = data * 252
    
    return data


def ensure_same_units(*bundles: ReturnsBundle) -> bool:
    """
    Check if all ReturnsBundle objects have consistent units.
    
    Returns:
        bool: True if all bundles have same scaler and output kind
    """
    if len(bundles) <= 1:
        return True
    
    reference = bundles[0]
    for bundle in bundles[1:]:
        if (bundle.used_scaler_name != reference.used_scaler_name or
            bundle.output_kind != reference.output_kind or
            bundle.annualise_mode != reference.annualise_mode):
            return False
    
    return True


def create_real_bundle(returns: np.ndarray, annualise_mode: str = "none") -> ReturnsBundle:
    """Create a ReturnsBundle from real returns data."""
    returns = np.array(returns).flatten()
    
    # Detect and validate scaling
    detected_scale = detect_scaler(returns)
    if detected_scale == "percent":
        print("Converting real data from percent to decimal returns")
        returns = returns / 100.0
        scaler_name = "PercentToDecimal"
    elif detected_scale == "prices":
        print("Converting real data from prices to decimal returns")  
        returns = np.diff(np.log(returns))
        scaler_name = "PricesToReturns"
    else:
        scaler_name = "Identity"
    
    return ReturnsBundle(
        returns=returns,
        mean=0.0,  # Will be computed in __post_init__
        std=0.0,   # Will be computed in __post_init__
        min=0.0,   # Will be computed in __post_init__
        max=0.0,   # Will be computed in __post_init__
        kurtosis=0.0,  # Will be computed in __post_init__
        used_scaler_name=scaler_name,
        output_kind="returns",
        annualise_mode=annualise_mode,
        provenance="real_data"
    )


def create_model_bundle(returns: np.ndarray, scaler: Any, model_name: str,
                       annualise_mode: str = "none") -> ReturnsBundle:
    """Create a ReturnsBundle from model-generated data."""
    # Apply inverse scaling
    inverse_returns_data = inverse_returns(returns, scaler, 
                                         scaler_name=str(type(scaler).__name__))
    
    return ReturnsBundle(
        returns=inverse_returns_data,
        mean=0.0,  # Will be computed in __post_init__
        std=0.0,   # Will be computed in __post_init__
        min=0.0,   # Will be computed in __post_init__
        max=0.0,   # Will be computed in __post_init__
        kurtosis=0.0,  # Will be computed in __post_init__
        used_scaler_name=str(type(scaler).__name__) if scaler else "Identity",
        output_kind="returns",
        annualise_mode=annualise_mode,
        provenance=f"model_{model_name}"
    )


def require_inverse_scaled_data(func):
    """
    Decorator to ensure functions only accept properly inverse-scaled ReturnsBundle data.
    Rejects raw numpy arrays to enforce proper data flow.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Check all positional args
        for i, arg in enumerate(args):
            if isinstance(arg, np.ndarray):
                raise TypeError(
                    f"Function {func.__name__} received raw numpy array at position {i}. "
                    f"Please use ReturnsBundle with proper inverse scaling."
                )
            elif isinstance(arg, (list, tuple)) and len(arg) > 0:
                if isinstance(arg[0], np.ndarray):
                    raise TypeError(
                        f"Function {func.__name__} received container of raw arrays at position {i}. "
                        f"Please use ReturnsBundle objects with proper inverse scaling."
                    )
        
        # Check keyword args
        for key, value in kwargs.items():
            if isinstance(value, np.ndarray):
                raise TypeError(
                    f"Function {func.__name__} received raw numpy array for '{key}'. "
                    f"Please use ReturnsBundle with proper inverse scaling."
                )
        
        return func(*args, **kwargs)
    
    return wrapper


def compute_rolling_vol(returns: np.ndarray, window: int = 20, ddof: int = 1,
                       demean: bool = False, annualise: str = 'none') -> np.ndarray:
    """Compute rolling volatility from returns array."""
    returns_series = pd.Series(returns) if not isinstance(returns, pd.Series) else returns

    if demean:
        returns_series = returns_series - returns_series.rolling(window).mean()

    rolling_std = returns_series.rolling(window, min_periods=1).std(ddof=ddof)

    if annualise == 'sqrt252':
        rolling_std = rolling_std * np.sqrt(252)

    return rolling_std.values