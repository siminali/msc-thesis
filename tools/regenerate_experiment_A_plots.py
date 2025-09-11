#!/usr/bin/env python3
"""
Regenerate Experiment A plots: ECDF overlay and rolling volatility comparison.
Uses data from results/addons/period_slices/A_v15/covid_crash/
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from scipy.stats import gaussian_kde
from statsmodels.tsa import stattools

def load_real_data_covid_period() -> np.ndarray:
    """Load real S&P 500 data for COVID crash period (2020-02-23 to 2020-04-01)."""
    
    # Load S&P 500 data
    sp500_file = Path("/Users/siminali/Desktop/Thesis Coding/data/sp500_data.csv")
    if not sp500_file.exists():
        raise FileNotFoundError(f"S&P 500 data file not found: {sp500_file}")
    
    df = pd.read_csv(sp500_file)
    
    # Parse dates if needed
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
    elif df.index.dtype == 'object':
        df.index = pd.to_datetime(df.index)
    
    # Filter for COVID crash period (2020-02-23 to 2020-04-01)
    start_date = '2020-02-23'
    end_date = '2020-04-01'
    
    covid_df = df.loc[start_date:end_date]
    
    if len(covid_df) == 0:
        print(f"Warning: No data found for period {start_date} to {end_date}")
        # Fallback: get COVID period data around March 2020
        covid_df = df.loc['2020-02-01':'2020-04-30']
    
    # Calculate returns
    if 'Close' in covid_df.columns:
        prices = covid_df['Close'].values
    elif 'close' in covid_df.columns:
        prices = covid_df['close'].values
    else:
        # Assume first numeric column is price
        prices = covid_df.iloc[:, 0].values
    
    # Log returns in percentage
    returns = np.diff(np.log(prices)) * 100
    
    print(f"Loaded real COVID crash returns: {len(returns)} observations")
    print(f"Date range: {covid_df.index[0]} to {covid_df.index[-1]}")
    
    return returns

def load_model_samples(base_dir: Path) -> Dict[str, np.ndarray]:
    """Load synthetic samples from all models."""
    
    models = ['zero', 'explicit', 'llm']
    model_samples = {}
    
    for model in models:
        model_dir = base_dir / model
        samples_file = model_dir / "samples.npy"
        
        if samples_file.exists():
            samples = np.load(samples_file)  # Shape: [1000, 60]
            # Flatten to get all returns
            returns = samples.flatten()
            model_samples[model] = returns
            print(f"Loaded {model} samples: {samples.shape} -> {len(returns)} returns")
        else:
            print(f"Warning: {model} samples not found at {samples_file}")
    
    return model_samples

def calculate_rolling_volatility(returns: np.ndarray, window: int = 5) -> np.ndarray:
    """Calculate rolling volatility with specified window."""
    
    if len(returns) < window:
        return np.full(len(returns), np.std(returns))
    
    rolling_vol = []
    for i in range(len(returns)):
        start_idx = max(0, i - window + 1)
        end_idx = i + 1
        window_returns = returns[start_idx:end_idx]
        vol = np.std(window_returns)
        rolling_vol.append(vol)
    
    return np.array(rolling_vol)

def create_ecdf_overlay(real_returns: np.ndarray, model_samples: Dict[str, np.ndarray], 
                       outdir: Path):
    """Create ECDF overlay plot comparing Real vs Zero, Explicit, LLM."""
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Enhanced colors and line styles for better distinction
    model_styles = {
        'Real': {'color': 'black', 'linewidth': 3.0, 'linestyle': '-', 'alpha': 1.0, 'zorder': 10},
        'zero': {'color': 'green', 'linewidth': 2.5, 'linestyle': '--', 'alpha': 0.9, 'zorder': 7},
        'explicit': {'color': 'red', 'linewidth': 2.5, 'linestyle': ':', 'alpha': 0.9, 'zorder': 8}, 
        'llm': {'color': 'blue', 'linewidth': 2.5, 'linestyle': '-.', 'alpha': 0.9, 'zorder': 9}
    }
    
    labels = {'Real': 'Real S&P 500', 'zero': 'Zero-DDPM', 'explicit': 'Explicit-DDPM', 'llm': 'LLM-DDPM'}
    
    # Plot real returns ECDF first
    real_sorted = np.sort(real_returns)
    real_ecdf = np.arange(1, len(real_sorted) + 1) / len(real_sorted)
    ax.plot(real_sorted, real_ecdf, 
           color=model_styles['Real']['color'],
           linewidth=model_styles['Real']['linewidth'],
           linestyle=model_styles['Real']['linestyle'],
           alpha=model_styles['Real']['alpha'],
           zorder=model_styles['Real']['zorder'],
           label=labels['Real'])
    
    # Plot model ECDFs with distinct styles
    plot_order = ['zero', 'explicit', 'llm']  # Explicit order to ensure visibility
    for model in plot_order:
        if model in model_samples and len(model_samples[model]) > 0:
            returns = model_samples[model]
            model_sorted = np.sort(returns)
            model_ecdf = np.arange(1, len(model_sorted) + 1) / len(model_sorted)
            
            style = model_styles[model]
            ax.plot(model_sorted, model_ecdf,
                   color=style['color'],
                   linewidth=style['linewidth'], 
                   linestyle=style['linestyle'],
                   alpha=style['alpha'],
                   zorder=style['zorder'],
                   label=labels[model])
            
            print(f"Plotted {model}: mean={np.mean(returns):.3f}, std={np.std(returns):.3f}")
    
    # Styling
    ax.set_xlabel('Returns (%)', fontsize=12)
    ax.set_ylabel('Cumulative Probability', fontsize=12)
    ax.set_title('ECDF Overlay: COVID Crash Period Returns\n(Real vs Synthetic Models)', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='lower right')
    
    # Add summary statistics with better formatting
    real_mean = np.mean(real_returns)
    real_std = np.std(real_returns)
    
    stats_text = f"Real: μ={real_mean:.2f}%, σ={real_std:.2f}%\n"
    for model in plot_order:
        if model in model_samples and len(model_samples[model]) > 0:
            returns = model_samples[model]
            model_mean = np.mean(returns)
            model_std = np.std(returns)
            stats_text += f"{labels[model]}: μ={model_mean:.2f}%, σ={model_std:.2f}%\n"
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    
    # Save plots
    pdf_path = outdir / 'experiment_A_ecdf_overlay.pdf'
    png_path = outdir / 'experiment_A_ecdf_overlay.png'
    fig.savefig(pdf_path, bbox_inches='tight', dpi=300)
    fig.savefig(png_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    
    print(f"Saved ECDF overlay: {pdf_path}")
    print(f"Saved ECDF overlay: {png_path}")

def create_rolling_volatility_comparison(real_returns: np.ndarray, model_samples: Dict[str, np.ndarray],
                                       outdir: Path, window: int = 5):
    """Create rolling volatility comparison plots."""
    
    # Calculate rolling volatilities
    real_vol = calculate_rolling_volatility(real_returns, window)
    
    # For synthetic data, we need to reshape to get time series
    # Each model has samples of shape [1000, 60], we'll take mean across paths
    model_vols = {}
    for model, samples in model_samples.items():
        if len(samples) > 0:
            # Reshape back to [1000, 60] if flattened
            if len(samples) == 60000:  # 1000 * 60
                samples_reshaped = samples.reshape(1000, 60)
                # Calculate volatility for each path, then average
                path_vols = []
                for path in samples_reshaped:
                    path_vol = calculate_rolling_volatility(path, window)
                    path_vols.append(path_vol)
                model_vols[model] = np.mean(path_vols, axis=0)
            else:
                # Calculate rolling volatility directly
                model_vols[model] = calculate_rolling_volatility(samples, window)
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Enhanced colors and line styles for better distinction
    model_styles = {
        'Real': {'color': 'black', 'linewidth': 3.0, 'linestyle': '-', 'alpha': 1.0},
        'zero': {'color': 'green', 'linewidth': 2.5, 'linestyle': '--', 'alpha': 0.9},
        'explicit': {'color': 'red', 'linewidth': 2.5, 'linestyle': ':', 'alpha': 0.9},
        'llm': {'color': 'blue', 'linewidth': 2.5, 'linestyle': '-.', 'alpha': 0.9}
    }
    
    labels = {'Real': 'Real S&P 500', 'zero': 'Zero-DDPM', 'explicit': 'Explicit-DDPM', 'llm': 'LLM-DDPM'}
    
    # Top panel: Time series comparison
    time_idx_real = np.arange(len(real_vol))
    real_style = model_styles['Real']
    ax1.plot(time_idx_real, real_vol, 
            color=real_style['color'], 
            linewidth=real_style['linewidth'], 
            linestyle=real_style['linestyle'],
            alpha=real_style['alpha'],
            label=labels['Real'])
    
    plot_order = ['zero', 'explicit', 'llm']
    for model in plot_order:
        if model in model_vols:
            vol = model_vols[model]
            time_idx_model = np.arange(len(vol))
            style = model_styles[model]
            ax1.plot(time_idx_model, vol,
                    color=style['color'],
                    linewidth=style['linewidth'], 
                    linestyle=style['linestyle'],
                    alpha=style['alpha'],
                    label=labels[model])
            print(f"Volatility plotted for {model}: mean={np.mean(vol):.3f}, std={np.std(vol):.3f}")
    
    ax1.set_ylabel(f'Rolling Volatility (window={window})', fontsize=12)
    ax1.set_title('Rolling Volatility: COVID Crash Period\n(Realized vs Synthetic)', 
                 fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10, loc='upper left')
    
    # Bottom panel: Scatter plot comparison with distinct markers
    markers = {'zero': 'o', 'explicit': 's', 'llm': '^'}
    
    for model in plot_order:
        if model in model_vols:
            vol = model_vols[model]
            # Align lengths for scatter plot
            min_len = min(len(real_vol), len(vol))
            real_aligned = real_vol[:min_len]
            model_aligned = vol[:min_len]
            
            style = model_styles[model]
            ax2.scatter(real_aligned, model_aligned, 
                       color=style['color'],
                       alpha=0.7, s=30, 
                       marker=markers[model],
                       label=labels[model],
                       edgecolors='black', linewidth=0.5)
    
    # Add 45-degree line
    if len(real_vol) > 0:
        min_val = min(real_vol.min(), min(vol.min() for vol in model_vols.values() if len(vol) > 0))
        max_val = max(real_vol.max(), max(vol.max() for vol in model_vols.values() if len(vol) > 0))
        ax2.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, linewidth=1)
    
    ax2.set_xlabel('Realized Volatility', fontsize=12)
    ax2.set_ylabel('Synthetic Volatility', fontsize=12)
    ax2.set_title('Realized vs Synthetic Volatility Correlation', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    
    # Save plots
    pdf_path = outdir / 'experiment_A_rolling_volatility.pdf'
    png_path = outdir / 'experiment_A_rolling_volatility.png'
    fig.savefig(pdf_path, bbox_inches='tight', dpi=300)
    fig.savefig(png_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    
    print(f"Saved rolling volatility: {pdf_path}")
    print(f"Saved rolling volatility: {png_path}")
    
    # Print correlation statistics
    print("\nVolatility Correlation Analysis:")
    for model, vol in model_vols.items():
        min_len = min(len(real_vol), len(vol))
        if min_len > 1:
            real_aligned = real_vol[:min_len]
            model_aligned = vol[:min_len]
            correlation = np.corrcoef(real_aligned, model_aligned)[0, 1]
            rmse = np.sqrt(np.mean((real_aligned - model_aligned) ** 2))
            print(f"  {labels[model]}: corr={correlation:.3f}, RMSE={rmse:.3f}")

def main():
    parser = argparse.ArgumentParser(description='Regenerate Experiment A plots from A_v15 data')
    parser.add_argument('--data_dir', 
                       default='/Users/siminali/Desktop/Thesis Coding/results/addons/period_slices/A_v15/covid_crash',
                       help='Path to A_v15/covid_crash directory')
    parser.add_argument('--outdir', 
                       default='/Users/siminali/Desktop/Thesis Coding/final_results_thesis/experiment_A',
                       help='Output directory')
    parser.add_argument('--window', type=int, default=5, help='Rolling volatility window')
    
    args = parser.parse_args()
    
    # Setup paths
    data_dir = Path(args.data_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    print("Experiment A Plot Regeneration")
    print("=" * 40)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {outdir}")
    print(f"Rolling window: {args.window}")
    print()
    
    try:
        # Load data
        print("1. Loading real COVID crash data...")
        real_returns = load_real_data_covid_period()
        
        print("2. Loading synthetic model samples...")
        model_samples = load_model_samples(data_dir)
        
        if len(model_samples) == 0:
            print("Error: No model samples loaded!")
            return
        
        print("3. Creating ECDF overlay plot...")
        create_ecdf_overlay(real_returns, model_samples, outdir)
        
        print("4. Creating rolling volatility comparison...")
        create_rolling_volatility_comparison(real_returns, model_samples, outdir, args.window)
        
        print()
        print("✅ Experiment A plots regenerated successfully!")
        print(f"📁 Files saved in: {outdir}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
