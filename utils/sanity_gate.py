"""
Sanity gate for validating inverse-scaled financial returns data.
Ensures data falls within realistic bounds for daily financial returns.
"""

import numpy as np
from dataclasses import dataclass
from typing import Callable, Optional, Union
from utils.scaling_guard import ReturnsBundle


@dataclass
class SanityThresholds:
    """Configurable thresholds for sanity gate validation."""
    std_bounds: tuple  # (min_std, max_std) for daily returns
    absmax: float      # Maximum absolute return magnitude
    
    def __post_init__(self):
        """Validate threshold values."""
        if len(self.std_bounds) != 2:
            raise ValueError("std_bounds must be a tuple of (min_std, max_std)")
        
        min_std, max_std = self.std_bounds
        if min_std >= max_std:
            raise ValueError("min_std must be less than max_std")
        
        if min_std <= 0:
            raise ValueError("min_std must be positive")
        
        if self.absmax <= 0:
            raise ValueError("absmax must be positive")


class SanityGateError(ValueError):
    """Exception raised when data fails sanity gate validation."""
    pass


class SanityGate:
    """
    Validator for inverse-scaled financial returns data.
    Ensures data represents realistic daily returns in decimal units.
    """
    
    @staticmethod
    def validate(bundle: ReturnsBundle, 
                model_name: str,
                window_name: str,
                thresholds: SanityThresholds,
                allow_bypass: bool = False,
                logger: Optional[Callable[[str], None]] = None) -> str:
        """
        Validate that a ReturnsBundle contains realistic daily returns.
        
        Args:
            bundle: ReturnsBundle to validate
            model_name: Name of model (for error messages)
            window_name: Name of window/scenario (for error messages)
            thresholds: SanityThresholds configuration
            allow_bypass: If True, log warning instead of raising error
            logger: Optional logging function (default: print)
            
        Returns:
            str: Status tag ("OK" or "SUSPECT SCALE (...)")
            
        Raises:
            SanityGateError: If validation fails and allow_bypass=False
        """
        if logger is None:
            logger = print
        
        # Extract values for checking
        std_val = bundle.std
        absmax_val = max(abs(bundle.min), abs(bundle.max))
        mean_val = bundle.mean
        kurtosis_val = bundle.kurtosis
        
        # Check standard deviation bounds
        min_std, max_std = thresholds.std_bounds
        std_ok = min_std <= std_val <= max_std
        
        # Check absolute maximum bounds
        absmax_ok = absmax_val <= thresholds.absmax
        
        # Overall validation
        validation_passed = std_ok and absmax_ok
        
        if validation_passed:
            return "OK"
        
        # Create detailed error message
        error_details = (
            f"mean={mean_val:.6f}, std={std_val:.6f}, min={bundle.min:.3f}, "
            f"max={bundle.max:.3f}, kurtosis={kurtosis_val:.2f}, "
            f"scaler={bundle.used_scaler_name}, kind={bundle.output_kind}, "
            f"annualise={bundle.annualise_mode}; "
            f"thresholds std∈[{min_std},{max_std}], absmax≤{thresholds.absmax}. "
            f"Likely causes: missing inverse_transform, wrong units (percent), "
            f"or using prices instead of returns."
        )
        
        error_message = (
            f"SanityGate FAIL for {model_name}/{window_name}: {error_details}"
        )
        
        if allow_bypass:
            # Create suspect scale tag
            suspect_tag = f"SUSPECT SCALE (std={std_val:.6f}, max|r|={absmax_val:.3f})"
            logger(f"[WARNING] {error_message}")
            return suspect_tag
        else:
            raise SanityGateError(error_message)
    
    @staticmethod
    def create_default_thresholds() -> SanityThresholds:
        """Create default sanity thresholds for daily stock returns."""
        return SanityThresholds(
            std_bounds=(0.005, 0.05),  # 0.5% to 5% daily volatility
            absmax=0.5                  # 50% maximum single-day move
        )
    
    @staticmethod
    def check_multiple_bundles(bundles: dict, 
                              window_name: str,
                              thresholds: SanityThresholds,
                              allow_bypass: bool = False,
                              logger: Optional[Callable[[str], None]] = None) -> dict:
        """
        Validate multiple ReturnsBundle objects.
        
        Args:
            bundles: Dict of {model_name: ReturnsBundle}
            window_name: Window/scenario name
            thresholds: SanityThresholds configuration
            allow_bypass: If True, log warnings instead of raising errors
            logger: Optional logging function
            
        Returns:
            dict: {model_name: status_tag}
        """
        results = {}
        
        for model_name, bundle in bundles.items():
            try:
                status = SanityGate.validate(
                    bundle, model_name, window_name, thresholds, 
                    allow_bypass, logger
                )
                results[model_name] = status
            except SanityGateError:
                # Re-raise to fail fast if bypass not allowed
                raise
        
        return results
    
    @staticmethod
    def create_sanity_summary_table(bundles: dict, 
                                   sanity_results: dict,
                                   window_name: str) -> str:
        """
        Create a compact summary table of sanity check results.
        
        Args:
            bundles: Dict of {model_name: ReturnsBundle}
            sanity_results: Dict of {model_name: status_tag}
            window_name: Window/scenario name
            
        Returns:
            str: Formatted table string
        """
        lines = [f"Sanity Check Summary - {window_name}"]
        lines.append("=" * 50)
        lines.append(f"{'Model':<12} {'Mean':<8} {'Std':<8} {'Status':<20}")
        lines.append("-" * 50)
        
        for model_name, bundle in bundles.items():
            status = sanity_results.get(model_name, "UNKNOWN")
            lines.append(
                f"{model_name:<12} {bundle.mean:<8.4f} {bundle.std:<8.4f} {status:<20}"
            )
        
        return "\n".join(lines)


def add_suspect_scale_tag(title: str, suspect_tag: str) -> str:
    """
    Add suspect scale warning to plot/table titles.
    
    Args:
        title: Original title
        suspect_tag: Suspect scale tag (if any)
        
    Returns:
        str: Modified title with warning if applicable
    """
    if suspect_tag and suspect_tag != "OK":
        return f"{title}\n⚠️ {suspect_tag}"
    return title


def check_data_consistency(real_bundle: ReturnsBundle, 
                          model_bundles: dict) -> list:
    """
    Check consistency across real and model data bundles.
    
    Args:
        real_bundle: Real data bundle
        model_bundles: Dict of model bundles
        
    Returns:
        list: List of consistency warnings
    """
    warnings = []
    
    # Check if all bundles use same scaling/units
    all_bundles = [real_bundle] + list(model_bundles.values())
    scaler_names = [b.used_scaler_name for b in all_bundles]
    output_kinds = [b.output_kind for b in all_bundles]
    annualise_modes = [b.annualise_mode for b in all_bundles]
    
    if len(set(scaler_names)) > 1:
        warnings.append(f"Inconsistent scalers: {set(scaler_names)}")
    
    if len(set(output_kinds)) > 1:
        warnings.append(f"Inconsistent output kinds: {set(output_kinds)}")
    
    if len(set(annualise_modes)) > 1:
        warnings.append(f"Inconsistent annualisation: {set(annualise_modes)}")
    
    # Check for extreme differences in volatility
    real_std = real_bundle.std
    for model_name, model_bundle in model_bundles.items():
        model_std = model_bundle.std
        ratio = model_std / real_std if real_std > 0 else float('inf')
        
        if ratio > 10 or ratio < 0.1:
            warnings.append(
                f"Model '{model_name}' volatility {ratio:.1f}x real data - "
                f"possible scaling issue"
            )
    
    return warnings