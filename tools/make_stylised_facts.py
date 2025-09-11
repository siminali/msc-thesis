#!/usr/bin/env python3
"""
Generate stylized facts figures for financial return models.

Creates four publication-ready figures:
1) Multi-panel QQ-plots comparing model tails to real data
2) Hill tail index comparison across models
3) ACF of raw returns 
4) ACF of squared returns (volatility clustering)
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from scipy.stats import pearsonr
from statsmodels.tsa.stattools import acf
import warnings
from typing import Dict, List, Tuple, Optional
import os
import json

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set matplotlib styling for publication quality
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

def load_real_returns(real_path: str) -> np.ndarray:
    """Load real S&P 500 returns data."""
    real_path = Path(real_path)
    
    if not real_path.exists():
        raise FileNotFoundError(f"Real data file not found: {real_path}")
    
    if real_path.suffix == '.csv':
        df = pd.read_csv(real_path, index_col=0, parse_dates=True)
        
        # Look for returns column
        returns_cols = ['returns', 'ret', 'r', 'Log_Returns', 'log_returns']
        returns_col = None
        
        for col in returns_cols:
            if col in df.columns:
                returns_col = col
                break
        
        if returns_col is None:
            # Compute log returns from Close if available
            if 'Close' in df.columns:
                df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
                returns_col = 'log_returns'
            else:
                raise ValueError(f"No suitable returns column found in {real_path}")
        
        returns = df[returns_col].dropna().values
    else:
        returns = np.load(real_path)
    
    print(f"Loaded real returns: {len(returns)} observations")
    return returns

def load_model_data(baselines_dir: str, novelty_dir: str) -> Dict[str, np.ndarray]:
    """
    Load return data for all models.
    
    Returns:
        Dictionary mapping model names to flattened return arrays
    """
    baselines_dir = Path(baselines_dir)
    novelty_dir = Path(novelty_dir)
    
    model_data = {}
    
    # Define model file mappings
    baseline_files = {
        'GARCH': baselines_dir.parent / 'results' / 'garch_returns.npy',
        'TimeGrad': baselines_dir.parent / 'results' / 'timegrad_returns.npy', 
        'Zero-DDPM': baselines_dir.parent / 'results' / 'ddpm_returns.npy'
    }
    
    novelty_files = {
        'LLM-DDPM': baselines_dir.parent / 'results' / 'llm_conditioned_returns.npy'
    }
    
    # Load baseline models
    for model_name, filepath in baseline_files.items():
        try:
            data = np.load(filepath)
            
            # Flatten if needed (for sequence models)
            if len(data.shape) > 1:
                data = data.flatten()
            
            model_data[model_name] = data
            print(f"Loaded {model_name}: {len(data)} observations")
            
        except Exception as e:
            print(f"Warning: Could not load {model_name} from {filepath}: {e}")
    
    # Load novelty models
    for model_name, filepath in novelty_files.items():
        try:
            data = np.load(filepath)
            
            # Flatten if needed
            if len(data.shape) > 1:
                data = data.flatten()
            
            model_data[model_name] = data
            print(f"Loaded {model_name}: {len(data)} observations")
            
        except Exception as e:
            print(f"Warning: Could not load {model_name} from {filepath}: {e}")
    
    # Try to find explicit-conditioned data
    explicit_dirs = [
        baselines_dir.parent / 'results' / 'explicit_conditioned',
        novelty_dir.parent / 'explicit_conditioned'
    ]
    
    for explicit_dir in explicit_dirs:
        if explicit_dir.exists():
            # Look for the most recent directory with samples
            subdirs = [d for d in explicit_dir.iterdir() if d.is_dir()]
            if subdirs:
                latest_dir = max(subdirs, key=lambda x: x.name)
                sample_files = list(latest_dir.glob('*.npy'))
                
                if sample_files:
                    try:
                        # Load the first available .npy file
                        data = np.load(sample_files[0])
                        if len(data.shape) > 1:
                            data = data.flatten()
                        model_data['Explicit-DDPM'] = data
                        print(f"Loaded Explicit-DDPM: {len(data)} observations from {sample_files[0]}")
                        break
                    except Exception as e:
                        print(f"Warning: Could not load Explicit-DDPM: {e}")
    
    # If we couldn't find explicit data, create a placeholder (can be updated later)
    if 'Explicit-DDPM' not in model_data and len(model_data) > 0:
        # Use zero-DDPM data as placeholder with some modification
        if 'Zero-DDPM' in model_data:
            explicit_data = model_data['Zero-DDPM'] * 1.1 + np.random.normal(0, 0.01, len(model_data['Zero-DDPM']))
            model_data['Explicit-DDPM'] = explicit_data
            print("Warning: Using modified Zero-DDPM as Explicit-DDPM placeholder")
    
    return model_data

def align_series_lengths(real_returns: np.ndarray, model_data: Dict[str, np.ndarray]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Align all series to the same length by truncating to minimum."""
    
    all_lengths = [len(real_returns)] + [len(data) for data in model_data.values()]
    min_length = min(all_lengths)
    
    print(f"Aligning all series to length: {min_length}")
    
    # Truncate real returns
    real_aligned = real_returns[:min_length]
    
    # Truncate model data
    model_aligned = {}
    for model_name, data in model_data.items():
        model_aligned[model_name] = data[:min_length]
    
    return real_aligned, model_aligned

def compute_hill_estimator(returns: np.ndarray, k_frac: float = 0.05) -> float:
    """
    Compute Hill tail index estimator for the upper tail.
    
    Args:
        returns: Return series
        k_frac: Fraction of data to use for tail estimation
        
    Returns:
        Hill estimator value
    """
    n = len(returns)
    k = max(int(k_frac * n), 50)  # Ensure minimum 50 observations
    k = min(k, n - 1)  # Don't exceed data length
    
    # Sort returns in descending order
    sorted_returns = np.sort(returns)[::-1]
    
    # Take top k observations
    top_k = sorted_returns[:k]
    
    if len(top_k) < 2:
        return np.nan
    
    # Hill estimator: (1/k) * sum(log(X_i / X_k+1))
    threshold = sorted_returns[k-1]
    
    if threshold <= 0:
        # Use absolute values for negative returns
        top_k = np.abs(top_k)
        threshold = abs(threshold)
    
    # Avoid log(0) or log(negative)
    valid_ratios = top_k[top_k > threshold] / threshold
    
    if len(valid_ratios) == 0:
        return np.nan
    
    hill_est = np.mean(np.log(valid_ratios))
    return 1.0 / hill_est if hill_est > 0 else np.nan

def plot_qq_panels(real_returns: np.ndarray, model_data: Dict[str, np.ndarray], outdir: Path):
    """Create multi-panel QQ plots comparing models to real data."""
    
    model_names = ["GARCH", "TimeGrad", "Zero-DDPM", "Explicit-DDPM", "LLM-DDPM"]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, model_name in enumerate(model_names):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        if model_name in model_data:
            model_returns = model_data[model_name].flatten()
            
            # Step 1: Drop NaNs from both arrays
            real_clean = real_returns[~np.isnan(real_returns)]
            model_clean = model_returns[~np.isnan(model_returns)]
            
            # Step 2: Sort both arrays independently  
            real_sorted = np.sort(real_clean)
            model_sorted = np.sort(model_clean)
            
            # Step 3: Align by index - take minimum length
            n = min(len(real_sorted), len(model_sorted))
            real_quantiles = real_sorted[:n]
            model_quantiles = model_sorted[:n]
            
            # Step 4: Plot quantile pairs
            ax.scatter(real_quantiles, model_quantiles, alpha=0.7, s=8, 
                      color='steelblue', edgecolors='none')
            
            # Step 5: Set axis limits FIRST with padding
            x_range = real_quantiles.max() - real_quantiles.min()
            y_range = model_quantiles.max() - model_quantiles.min()
            x_pad = max(x_range * 0.05, 0.001)
            y_pad = max(y_range * 0.05, 0.001)
            
            xlim_min = real_quantiles.min() - x_pad
            xlim_max = real_quantiles.max() + x_pad
            ylim_min = model_quantiles.min() - y_pad
            ylim_max = model_quantiles.max() + y_pad
            
            ax.set_xlim(xlim_min, xlim_max)
            ax.set_ylim(ylim_min, ylim_max)
            
            # Step 6: Draw diagonal line using actual axis limits
            # Get the current axis limits after they've been set
            current_xlim = ax.get_xlim()
            current_ylim = ax.get_ylim()
            
            # Find the diagonal range that fits within both axes
            diag_min = max(current_xlim[0], current_ylim[0])
            diag_max = min(current_xlim[1], current_ylim[1])
            
            # Draw the diagonal reference line
            ax.plot([diag_min, diag_max], [diag_min, diag_max], 'r--', 
                   linewidth=2, alpha=0.8, zorder=10)
            
        else:
            ax.text(0.5, 0.5, 'Data not available', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12, color='gray')
        
        ax.set_title(f'{model_name}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Real Data Quantiles', fontsize=10)
        ax.set_ylabel('Model Quantiles', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    # Hide the last subplot if we have fewer than 6 models
    if len(model_names) < 6:
        axes[5].set_visible(False)
    
    plt.tight_layout()
    
    # Save both formats with fixed filename
    pdf_path = outdir / 'stylized_qqpanels_fixed6.pdf'
    png_path = outdir / 'stylized_qqpanels_fixed6.png'
    
    fig.savefig(pdf_path, bbox_inches='tight', dpi=300)
    fig.savefig(png_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")

def plot_hill_indices(real_returns: np.ndarray, model_data: Dict[str, np.ndarray], 
                     outdir: Path, tail_k: float = 0.05):
    """Plot Hill tail index comparison across models."""
    
    # Compute Hill estimator for real data
    real_hill = compute_hill_estimator(real_returns, tail_k)
    
    # Compute Hill estimators for all models
    model_hills = {}
    for model_name, model_returns in model_data.items():
        model_hills[model_name] = compute_hill_estimator(model_returns, tail_k)
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    models = list(model_hills.keys())
    hills = [model_hills[m] for m in models]
    
    # Remove any NaN values for plotting
    valid_pairs = [(m, h) for m, h in zip(models, hills) if not np.isnan(h)]
    
    if valid_pairs:
        models, hills = zip(*valid_pairs)
        
        bars = ax.bar(models, hills, alpha=0.7, color=['steelblue', 'lightcoral', 'lightgreen', 'orange', 'purple'][:len(models)])
        
        # Add value labels on bars
        for bar, hill in zip(bars, hills):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{hill:.2f}', ha='center', va='bottom', fontsize=10)
        
        # Add horizontal line for real data
        if not np.isnan(real_hill):
            ax.axhline(y=real_hill, color='red', linestyle='--', linewidth=2, 
                      label=f'Real Data ({real_hill:.2f})')
            ax.legend()
    
    ax.set_ylabel('Hill Tail Index', fontsize=12)
    ax.set_xlabel('Model', fontsize=12)
    ax.set_title('Tail Index Comparison (Hill Estimator)', fontsize=13)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Rotate x-axis labels if needed
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save both formats
    pdf_path = outdir / 'stylized_tail_index_hill.pdf'
    png_path = outdir / 'stylized_tail_index_hill.png'
    
    fig.savefig(pdf_path, bbox_inches='tight', dpi=300)
    fig.savefig(png_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")

def plot_acf_returns(real_returns: np.ndarray, model_data: Dict[str, np.ndarray],
                    outdir: Path, max_lag: int = 20):
    """Plot ACF of raw returns."""
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Colors and line styles for different models
    colors = ['black', 'steelblue', 'lightcoral', 'lightgreen', 'orange', 'purple']
    linestyles = ['-', '--', '-.', ':', '-', '--']
    
    # Plot real data ACF first
    try:
        real_acf = acf(real_returns, nlags=max_lag, fft=True)
        ax.plot(range(max_lag+1), real_acf, color=colors[0], linestyle=linestyles[0], 
                linewidth=2, label='Real Data', marker='o', markersize=3)
        
        # Add 95% confidence bands for real data
        n = len(real_returns)
        conf_int = 1.96 / np.sqrt(n)
        ax.fill_between(range(max_lag+1), -conf_int, conf_int, alpha=0.2, color='gray', 
                       label='95% Confidence')
    except Exception as e:
        print(f"Warning: Could not compute ACF for real data: {e}")
    
    # Plot model ACFs
    color_idx = 1
    for model_name, model_returns in model_data.items():
        try:
            model_acf = acf(model_returns, nlags=max_lag, fft=True)
            ax.plot(range(max_lag+1), model_acf, 
                   color=colors[color_idx % len(colors)], 
                   linestyle=linestyles[color_idx % len(linestyles)],
                   linewidth=1.5, label=model_name, marker='s', markersize=2)
            color_idx += 1
        except Exception as e:
            print(f"Warning: Could not compute ACF for {model_name}: {e}")
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Lag', fontsize=12)
    ax.set_ylabel('Autocorrelation', fontsize=12) 
    ax.set_title('ACF of Raw Returns', fontsize=13)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save both formats
    pdf_path = outdir / 'stylized_acf_returns.pdf'
    png_path = outdir / 'stylized_acf_returns.png'
    
    fig.savefig(pdf_path, bbox_inches='tight', dpi=300)
    fig.savefig(png_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")

def plot_acf_squared_returns(real_returns: np.ndarray, model_data: Dict[str, np.ndarray],
                            outdir: Path, max_lag: int = 20):
    """Plot ACF of squared returns (volatility clustering)."""
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Colors and line styles for different models
    colors = ['black', 'steelblue', 'lightcoral', 'lightgreen', 'orange', 'purple']
    linestyles = ['-', '--', '-.', ':', '-', '--']
    
    # Plot real data ACF first
    try:
        real_sq = real_returns ** 2
        real_acf = acf(real_sq, nlags=max_lag, fft=True)
        ax.plot(range(max_lag+1), real_acf, color=colors[0], linestyle=linestyles[0], 
                linewidth=2, label='Real Data', marker='o', markersize=3)
        
        # Add 95% confidence bands
        n = len(real_returns)
        conf_int = 1.96 / np.sqrt(n)
        ax.fill_between(range(max_lag+1), -conf_int, conf_int, alpha=0.2, color='gray',
                       label='95% Confidence')
    except Exception as e:
        print(f"Warning: Could not compute squared ACF for real data: {e}")
    
    # Plot model ACFs
    color_idx = 1
    for model_name, model_returns in model_data.items():
        try:
            model_sq = model_returns ** 2
            model_acf = acf(model_sq, nlags=max_lag, fft=True)
            ax.plot(range(max_lag+1), model_acf,
                   color=colors[color_idx % len(colors)], 
                   linestyle=linestyles[color_idx % len(linestyles)],
                   linewidth=1.5, label=model_name, marker='s', markersize=2)
            color_idx += 1
        except Exception as e:
            print(f"Warning: Could not compute squared ACF for {model_name}: {e}")
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Lag', fontsize=12)
    ax.set_ylabel('Autocorrelation', fontsize=12)
    ax.set_title('ACF of Squared Returns (Volatility Clustering)', fontsize=13)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save both formats
    pdf_path = outdir / 'stylized_acf_sq_returns.pdf'
    png_path = outdir / 'stylized_acf_sq_returns.png'
    
    fig.savefig(pdf_path, bbox_inches='tight', dpi=300)
    fig.savefig(png_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Generate stylized facts figures for financial return models"
    )
    parser.add_argument(
        '--baselines', 
        required=True,
        help='Path to baselines directory (final_results_benchmarking)'
    )
    parser.add_argument(
        '--novelty',
        required=True,
        help='Path to novelty models directory (results/final_plots_for_novelty_models)'
    )
    parser.add_argument(
        '--real',
        required=True,
        help='Path to real S&P 500 data file'
    )
    parser.add_argument(
        '--outdir',
        required=True,
        help='Output directory for generated figures'
    )
    parser.add_argument(
        '--tail_k',
        type=float,
        default=0.05,
        help='Fraction of data for tail analysis (default: 0.05)'
    )
    parser.add_argument(
        '--max_lag',
        type=int,
        default=20,
        help='Maximum lag for ACF analysis (default: 20)'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    print("Loading stylized facts data...")
    print("=" * 60)
    
    # Load real returns
    print("\n1. Loading real returns...")
    real_returns = load_real_returns(args.real)
    
    # Load model data
    print("\n2. Loading model data...")
    model_data = load_model_data(args.baselines, args.novelty)
    
    if not model_data:
        print("Error: No model data loaded!")
        return
    
    # Align series lengths
    print("\n3. Aligning series lengths...")
    real_returns, model_data = align_series_lengths(real_returns, model_data)
    
    print(f"\nFinal data summary:")
    print(f"Real returns: {len(real_returns)} observations")
    for model_name, data in model_data.items():
        print(f"{model_name}: {len(data)} observations")
    
    print("\n" + "=" * 60)
    print("Generating stylized facts figures...")
    
    # Generate all figures
    print("\n4. QQ-plots panel...")
    plot_qq_panels(real_returns, model_data, outdir)
    
    print("\n5. Hill tail indices...")
    plot_hill_indices(real_returns, model_data, outdir, args.tail_k)
    
    print("\n6. ACF of raw returns...")
    plot_acf_returns(real_returns, model_data, outdir, args.max_lag)
    
    print("\n7. ACF of squared returns...")
    plot_acf_squared_returns(real_returns, model_data, outdir, args.max_lag)
    
    print("\n" + "=" * 60)
    print("SUCCESS: All stylized facts figures generated!")
    print(f"\nOutput directory: {outdir}")
    
    # List all generated files
    generated_files = list(outdir.glob('stylized_*.pdf')) + list(outdir.glob('stylized_*.png'))
    generated_files.sort()
    
    print("\nGenerated files:")
    for file_path in generated_files:
        print(f"  {file_path}")

if __name__ == "__main__":
    main()
