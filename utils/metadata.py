#!/usr/bin/env python3
"""Metadata and utility functions for logging and system information."""

import torch
import numpy as np
import json
import time
import pandas as pd
from typing import Optional, Dict, Any


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_params(model) -> Optional[int]:
    """Return number of parameters if model has parameters else None."""
    if hasattr(model, 'parameters'):
        return sum(p.numel() for p in model.parameters())
    return None


def gpu_info() -> Dict[str, Any]:
    """Return dict with CUDA availability, device name, and total VRAM in GB."""
    info = {
        'cuda_available': torch.cuda.is_available(),
        'device_name': None,
        'total_vram_gb': None
    }
    
    if torch.cuda.is_available():
        info['device_name'] = torch.cuda.get_device_name(0)
        info['total_vram_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    return info


def timer():
    """Return a closure to measure elapsed time."""
    start_time = time.time()
    
    def elapsed():
        return time.time() - start_time
    
    return elapsed


def dataset_summary(series: pd.Series, name: str, units: str) -> Dict[str, Any]:
    """Generate summary statistics for a dataset series."""
    return {
        'name': name,
        'units': units,
        'n_observations': len(series),
        'start_date': series.index[0] if len(series) > 0 else None,
        'end_date': series.index[-1] if len(series) > 0 else None,
        'mean': float(series.mean()),
        'std': float(series.std()),
        'min': float(series.min()),
        'max': float(series.max()),
        'skewness': float(series.skew()),
        'kurtosis': float(series.kurtosis())
    }


def save_json(obj, path: str):
    """Write object to JSON file."""
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2, default=str)

