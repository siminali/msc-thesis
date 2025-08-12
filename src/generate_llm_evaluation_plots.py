#!/usr/bin/env python3
"""
Generate Missing Plots and Tables for LLM-Conditioned Model Evaluation
This script creates the plots and tables that are missing from the llm_conditioned_evaluation folder

Author: Simin Ali
Supervisor: Dr Mikael Mieskolainen
Institution: Imperial College London
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style for professional plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def create_llm_evaluation_directories():
    """Create the necessary directories for LLM-conditioned evaluation."""
    base_dir = "results/llm_conditioned_evaluation"
    
    # Create plots and tables directories
    os.makedirs(f"{base_dir}/plots", exist_ok=True)
    os.makedirs(f"{base_dir}/tables", exist_ok=True)
    
    print(f"✅ Created directories: {base_dir}/plots and {base_dir}/tables")

def load_llm_data():
    """Load LLM-conditioned model data and real data for comparison."""
    try:
        # Load LLM-conditioned synthetic data
        llm_data = np.load("results/llm_conditioned_returns.npy")
        
        # Load real data (using GARCH returns as reference)
        real_data = np.load("results/garch_returns.npy")
        
        # Load LLM-conditioned stats
        with open("results/llm_conditioned_evaluation/llm_conditioned_stats.json", 'r') as f:
            llm_stats = json.load(f)
        
        return llm_data, real_data, llm_stats
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None, None, None

def create_distribution_comparison_plot(llm_data, real_data, save_dir):
    """Create distribution comparison plot."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Flatten data for histogram
    llm_flat = llm_data.flatten()
    real_flat = real_data.flatten()
    
    # Left plot: Individual histograms
    ax1.hist(real_flat, bins=50, alpha=0.7, label='Real Data', density=True, color='blue', edgecolor='blue', linewidth=0.5)
    ax1.hist(llm_flat, bins=50, alpha=0.7, label='LLM-Conditioned', density=True, color='purple', edgecolor='purple', linewidth=0.5)
    ax1.set_xlabel('Log Returns (%)')
    ax1.set_ylabel('Density')
    ax1.set_title('Distribution Comparison: Real vs LLM-Conditioned')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Right plot: Overlayed histograms
    ax2.hist(real_flat, bins=50, alpha=0.6, label='Real Data', density=True, color='blue', edgecolor='blue', linewidth=0.5)
    ax2.hist(llm_flat, bins=50, alpha=0.6, label='LLM-Conditioned', density=True, color='purple', edgecolor='purple', linewidth=0.5)
    ax2.set_xlabel('Log Returns (%)')
    ax2.set_ylabel('Density')
    ax2.set_title('Distribution Overlay: Real vs LLM-Conditioned')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = f"{save_dir}/llm_distribution_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Created distribution comparison plot: {plot_path}")
    return plot_path

def create_sequence_comparison_plot(llm_data, real_data, save_dir):
    """Create sequence comparison plot."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Plot first 200 points of real data
    real_sequence = real_data.flatten()[:200]
    llm_sequence = llm_data.flatten()[:200]
    
    x = range(len(real_sequence))
    
    # Top plot: Real data
    ax1.plot(x, real_sequence, label='Real Data', linewidth=2, color='blue')
    ax1.set_xlabel('Time Steps')
    ax1.set_ylabel('Log Returns (%)')
    ax1.set_title('Real Data Sequence (First 200 Observations)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Bottom plot: LLM-Conditioned data
    ax2.plot(x, llm_sequence, label='LLM-Conditioned', linewidth=2, color='purple')
    ax2.set_xlabel('Time Steps')
    ax2.set_ylabel('Log Returns (%)')
    ax2.set_title('LLM-Conditioned Sequence (First 200 Observations)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = f"{save_dir}/llm_sequence_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Created sequence comparison plot: {plot_path}")
    return plot_path

def create_volatility_analysis_plot(llm_data, real_data, save_dir):
    """Create volatility clustering analysis plot."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Flatten data
    llm_flat = llm_data.flatten()
    real_flat = real_data.flatten()
    
    # Compute rolling volatility (20-day window)
    window = 20
    real_vol = pd.Series(real_flat).rolling(window=window).std().dropna()
    llm_vol = pd.Series(llm_flat).rolling(window=window).std().dropna()
    
    # Plot first 300 points
    x_real = range(len(real_vol[:300]))
    x_llm = range(len(llm_vol[:300]))
    
    # Top plot: Real data volatility
    ax1.plot(x_real, real_vol[:300], label='Real Data Volatility', linewidth=2, color='blue')
    ax1.set_xlabel('Time Steps')
    ax1.set_ylabel('Rolling Volatility (20-day window)')
    ax1.set_title('Real Data: Volatility Clustering Analysis')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Bottom plot: LLM-Conditioned volatility
    ax2.plot(x_llm, llm_vol[:300], label='LLM-Conditioned Volatility', linewidth=2, color='purple')
    ax2.set_xlabel('Time Steps')
    ax2.set_ylabel('Rolling Volatility (20-day window)')
    ax2.set_title('LLM-Conditioned: Volatility Clustering Analysis')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = f"{save_dir}/llm_volatility_analysis.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Created volatility analysis plot: {plot_path}")
    return plot_path

def create_qq_comparison_plot(llm_data, real_data, save_dir):
    """Create Q-Q plot comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Flatten data
    llm_flat = llm_data.flatten()
    real_flat = real_data.flatten()
    
    # Left plot: Real data Q-Q
    stats.probplot(real_flat, dist="norm", plot=ax1)
    ax1.set_title('Real Data vs Normal Distribution')
    ax1.grid(True, alpha=0.3)
    
    # Right plot: LLM-Conditioned Q-Q
    stats.probplot(llm_flat, dist="norm", plot=ax2)
    ax2.set_title('LLM-Conditioned vs Normal Distribution')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = f"{save_dir}/llm_qq_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Created Q-Q comparison plot: {plot_path}")
    return plot_path

def create_autocorrelation_plot(llm_data, real_data, save_dir):
    """Create autocorrelation function plot."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Flatten data
    llm_flat = llm_data.flatten()
    real_flat = real_data.flatten()
    
    # Compute autocorrelation of squared returns
    max_lag = 20
    
    # Real data autocorrelation
    real_acf = [1.0]  # lag 0
    for lag in range(1, max_lag + 1):
        if lag < len(real_flat):
            corr = np.corrcoef(real_flat[:-lag], real_flat[lag:])[0,1]
            real_acf.append(corr if not np.isnan(corr) else 0)
    
    # LLM-Conditioned autocorrelation
    llm_acf = [1.0]
    for lag in range(1, max_lag + 1):
        if lag < len(llm_flat):
            corr = np.corrcoef(llm_flat[:-lag], llm_flat[lag:])[0,1]
            llm_acf.append(corr if not np.isnan(corr) else 0)
    
    lags = range(max_lag + 1)
    
    # Top plot: Real data ACF
    ax1.plot(lags, real_acf, 'o-', label='Real Data', linewidth=2, markersize=6, color='blue')
    ax1.set_xlabel('Lag')
    ax1.set_ylabel('Autocorrelation')
    ax1.set_title('Real Data: Autocorrelation Function')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Bottom plot: LLM-Conditioned ACF
    ax2.plot(lags, llm_acf, 'o-', label='LLM-Conditioned', linewidth=2, markersize=6, color='purple')
    ax2.set_xlabel('Lag')
    ax2.set_ylabel('Autocorrelation')
    ax2.set_title('LLM-Conditioned: Autocorrelation Function')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = f"{save_dir}/llm_autocorrelation.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Created autocorrelation plot: {plot_path}")
    return plot_path

def create_performance_summary_table(llm_stats, save_dir):
    """Create performance summary table."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    
    # Title
    ax.text(0.5, 0.95, 'LLM-Conditioned Model Performance Summary', 
            fontsize=18, fontweight='bold', ha='center', va='top', transform=ax.transAxes)
    
    # Create table data
    data = [
        ['Metric', 'Value'],
        ['Mean', f"{llm_stats['Mean']:.4f}"],
        ['Standard Deviation', f"{llm_stats['Std Dev']:.4f}"],
        ['Skewness', f"{llm_stats['Skewness']:.4f}"],
        ['Kurtosis', f"{llm_stats['Kurtosis']:.4f}"],
        ['Minimum', f"{llm_stats['Min']:.4f}"],
        ['Maximum', f"{llm_stats['Max']:.4f}"]
    ]
    
    # Create table
    table = ax.table(cellText=data, cellLoc='center', loc='center',
                    bbox=[0.1, 0.2, 0.8, 0.6])
    
    # Style table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)
    
    # Color header row
    for i in range(2):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color data rows
    for i in range(1, len(data)):
        for j in range(2):
            table[(i, j)].set_facecolor('#E3F2FD')
    
    # Save as PNG
    plot_path = f"{save_dir}/llm_performance_summary.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Created performance summary table: {plot_path}")
    return plot_path

def main():
    """Generate all missing plots and tables for LLM-conditioned evaluation."""
    print("🎯 Generating Missing LLM-Conditioned Evaluation Plots and Tables...")
    
    # Create directories
    create_llm_evaluation_directories()
    
    # Load data
    llm_data, real_data, llm_stats = load_llm_data()
    if llm_data is None:
        print("❌ Failed to load data")
        return
    
    # Define save directory
    save_dir = "results/llm_conditioned_evaluation/plots"
    
    # Generate all plots
    print("\n📊 Generating plots...")
    
    # Distribution comparison
    create_distribution_comparison_plot(llm_data, real_data, save_dir)
    
    # Sequence comparison
    create_sequence_comparison_plot(llm_data, real_data, save_dir)
    
    # Volatility analysis
    create_volatility_analysis_plot(llm_data, real_data, save_dir)
    
    # Q-Q comparison
    create_qq_comparison_plot(llm_data, real_data, save_dir)
    
    # Autocorrelation
    create_autocorrelation_plot(llm_data, real_data, save_dir)
    
    # Performance summary table
    create_performance_summary_table(llm_stats, save_dir)
    
    print("\n✅ All LLM-conditioned evaluation plots and tables generated!")
    print(f"📁 Location: results/llm_conditioned_evaluation/")
    print("\n📊 Generated:")
    print("   - Distribution comparison plot")
    print("   - Sequence comparison plot")
    print("   - Volatility analysis plot")
    print("   - Q-Q comparison plot")
    print("   - Autocorrelation plot")
    print("   - Performance summary table")

if __name__ == "__main__":
    main()

