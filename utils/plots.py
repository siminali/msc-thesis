#!/usr/bin/env python3
"""Plotting utilities for financial data visualization."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List, Tuple


# Set matplotlib default background to white
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'


def hist_line_logy(data_list, labels, bins=60, title="", xlabel="Log Returns (%)"):
    """Draw line histograms with log-y scale."""
    plt.figure(figsize=(10, 6))
    
    for data, label in zip(data_list, labels):
        hist, bin_edges = np.histogram(data, bins=bins, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        plt.semilogy(bin_centers, hist, label=label, linewidth=2)
    
    plt.xlabel(xlabel)
    plt.ylabel('Density (log scale)')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


def qq_vs_normal(x, title="QQ Plot vs Normal"):
    """Create QQ plot comparing data to normal distribution."""
    plt.figure(figsize=(8, 6))
    
    from scipy import stats
    stats.probplot(x, dist="norm", plot=plt)
    
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


def acf_stem(x, nlags=20, title="Autocorrelation"):
    """Create stem plot of autocorrelation function."""
    plt.figure(figsize=(10, 6))
    
    from statsmodels.tsa.stattools import acf
    acf_values = acf(x, nlags=nlags, fft=False)
    lags = np.arange(nlags + 1)
    
    plt.stem(lags, acf_values)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.axhline(y=1.96/np.sqrt(len(x)), color='red', linestyle='--', alpha=0.7, label='95% CI')
    plt.axhline(y=-1.96/np.sqrt(len(x)), color='red', linestyle='--', alpha=0.7)
    
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


def rolling_std(x, window=20) -> pd.Series:
    """Compute rolling standard deviation."""
    return x.rolling(window=window).std()


def rolling_vol_panel(real, models: List[Tuple[np.ndarray, str]], window=20, title="Rolling Volatility"):
    """Create panel plot comparing rolling volatility across models."""
    n_models = len(models)
    fig, axes = plt.subplots(n_models + 1, 1, figsize=(12, 3 * (n_models + 1)))
    
    if n_models == 0:
        axes = [axes]
    
    # Plot real data volatility
    real_vol = rolling_std(real, window=window)
    axes[0].plot(real_vol.index, real_vol.values, 'b-', linewidth=2, label='Real Data')
    axes[0].set_title(f'{title} - Real Data')
    axes[0].set_ylabel('Volatility')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot model volatilities
    for i, (model_data, model_name) in enumerate(models):
        if isinstance(model_data, pd.Series):
            model_vol = rolling_std(model_data, window=window)
        else:
            model_vol = rolling_std(pd.Series(model_data), window=window)
        
        axes[i + 1].plot(model_vol.index, model_vol.values, 'r-', linewidth=2, label=model_name)
        axes[i + 1].set_title(f'{title} - {model_name}')
        axes[i + 1].set_ylabel('Volatility')
        axes[i + 1].legend()
        axes[i + 1].grid(True, alpha=0.3)
    
    # Set x-axis label for bottom plot only
    axes[-1].set_xlabel('Time')
    
    plt.tight_layout()
    return fig, axes

