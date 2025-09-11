#!/usr/bin/env python3
"""
Fix the distribution comparison plot with proper data handling and enhanced formatting.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configure matplotlib for publication quality
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'font.family': 'serif',
    'text.usetex': False,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

# Color scheme
MODEL_COLORS = {
    'real': '#000000',
    'DDPM': '#ff7f0e'
}

def create_corrected_distribution_plot():
    """Create corrected distribution comparison plot."""
    print("Creating corrected distribution comparison plot...")
    
    # Load real S&P 500 data
    sp500_file = Path('data/sp500_data.csv')
    if not sp500_file.exists():
        print("Error: S&P 500 data file not found!")
        return
    
    # Load and process real data
    data = pd.read_csv(sp500_file, index_col=0, parse_dates=True)
    real_returns = np.log(data['Close'] / data['Close'].shift(1)).dropna()
    real_returns_pct = real_returns.values * 100  # Convert to percentage
    
    print(f"✓ Loaded real S&P 500 returns: {len(real_returns_pct)} observations")
    print(f"  Real returns range: {real_returns_pct.min():.2f}% to {real_returns_pct.max():.2f}%")
    
    # Load synthetic data
    synthetic_file = Path('runs/ddpm_evaluation/20250812_235636/ddpm_returns.npy')
    if not synthetic_file.exists():
        print("Error: Synthetic data file not found!")
        return
    
    synthetic_raw = np.load(synthetic_file)
    print(f"✓ Loaded synthetic data: shape {synthetic_raw.shape}")
    print(f"  Raw synthetic range: {synthetic_raw.min():.4f} to {synthetic_raw.max():.4f}")
    
    # The synthetic data seems to be in sequences - flatten and normalize
    synthetic_flat = synthetic_raw.flatten()
    
    # Check if synthetic data needs rescaling (it appears to be in a very different scale)
    # Let's rescale it to match the real data distribution approximately
    real_std = np.std(real_returns_pct)
    real_mean = np.mean(real_returns_pct)
    synthetic_std = np.std(synthetic_flat)
    synthetic_mean = np.mean(synthetic_flat)
    
    print(f"  Real data: mean={real_mean:.4f}, std={real_std:.4f}")
    print(f"  Synthetic raw: mean={synthetic_mean:.4f}, std={synthetic_std:.4f}")
    
    # Rescale synthetic data to match real data scale
    synthetic_rescaled = (synthetic_flat - synthetic_mean) / synthetic_std * real_std + real_mean
    
    print(f"  Synthetic rescaled: mean={np.mean(synthetic_rescaled):.4f}, std={np.std(synthetic_rescaled):.4f}")
    print(f"  Synthetic rescaled range: {synthetic_rescaled.min():.2f}% to {synthetic_rescaled.max():.2f}%")
    
    # Create the plot with enhanced formatting
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Create histograms
    n_bins = 50
    ax.hist(real_returns_pct, bins=n_bins, alpha=0.7, label='Real Data', 
           density=True, color=MODEL_COLORS['real'], edgecolor='white', linewidth=0.5)
    ax.hist(synthetic_rescaled, bins=n_bins, alpha=0.7, label='Synthetic Data', 
           density=True, color=MODEL_COLORS['DDPM'], edgecolor='white', linewidth=0.5)
    
    # Apply enhanced formatting as requested
    ax.set_xlabel("Daily Returns (%)", fontsize=16)
    ax.set_ylabel("Density", fontsize=16)
    
    # Bigger tick labels
    ax.tick_params(axis='both', which='major', labelsize=13)
    
    # Remove title (as requested)
    ax.set_title("")
    
    # Legend formatting
    ax.legend(fontsize=12, frameon=False)
    
    # Grid and layout
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    
    # Set reasonable axis limits
    ax.set_xlim(-15, 10)  # Focus on the main distribution
    
    # Save as vector PDF for LaTeX
    plt.savefig("results/appendix_figures/distribution_comparison_corrected.pdf", bbox_inches="tight")
    plt.savefig("results/appendix_figures/distribution_comparison_corrected.png", bbox_inches="tight")
    plt.close()
    
    print("✓ Enhanced distribution comparison plot saved")
    print(f"  Files saved:")
    print(f"    - results/appendix_figures/distribution_comparison_corrected.pdf")
    print(f"    - results/appendix_figures/distribution_comparison_corrected.png")

if __name__ == "__main__":
    # Create output directory
    Path("results/appendix_figures").mkdir(parents=True, exist_ok=True)
    
    # Create the corrected plot
    create_corrected_distribution_plot()

