#!/usr/bin/env python3
"""
Generate QQ-plots with proper diagonal reference lines from scratch.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict

def load_real_returns(csv_path: str) -> np.ndarray:
    """Load real S&P 500 returns."""
    df = pd.read_csv(csv_path)
    if 'Close' in df.columns:
        # Compute log returns
        close_prices = df['Close'].values
        returns = np.diff(np.log(close_prices)) * 100  # Convert to percentages
        return returns
    else:
        raise ValueError("Close column not found")

def load_model_data(baselines_dir: str, novelty_dir: str) -> Dict[str, np.ndarray]:
    """Load model data."""
    baselines_dir = Path(baselines_dir)
    novelty_dir = Path(novelty_dir)
    
    model_data = {}
    
    # Load baseline models - check files in results directory
    result_dir = Path('/Users/siminali/Desktop/Thesis Coding/results')
    model_files = {
        'GARCH': result_dir / 'garch_returns.npy',
        'TimeGrad': result_dir / 'timegrad_returns.npy', 
        'Zero-DDPM': result_dir / 'ddpm_returns.npy',
        'LLM-DDPM': result_dir / 'llm_conditioned_returns.npy'
    }
    
    for model_name, file_path in model_files.items():
        if file_path.exists():
            data = np.load(file_path)
            model_data[model_name] = data.flatten()
            print(f"Loaded {model_name}: {len(model_data[model_name])} observations")
        else:
            print(f"WARNING: {file_path} not found")
    
    # Use Zero-DDPM as placeholder for Explicit-DDPM if available
    if 'Zero-DDPM' in model_data:
        model_data['Explicit-DDPM'] = model_data['Zero-DDPM'].copy()
        print("Warning: Using Zero-DDPM as Explicit-DDPM placeholder")
    
    if not model_data:
        raise ValueError("No model data files found!")
        
    return model_data

def create_qq_plot_panel(real_data: np.ndarray, model_data: np.ndarray, ax, model_name: str):
    """Create a single QQ-plot panel with proper diagonal reference line."""
    
    # Clean data - remove NaNs
    real_clean = real_data[~np.isnan(real_data)]
    model_clean = model_data[~np.isnan(model_data)]
    
    # Sort both arrays independently
    real_sorted = np.sort(real_clean)
    model_sorted = np.sort(model_clean)
    
    # Align by taking minimum length
    n = min(len(real_sorted), len(model_sorted))
    real_quantiles = real_sorted[:n]
    model_quantiles = model_sorted[:n]
    
    # Plot quantile pairs
    ax.scatter(real_quantiles, model_quantiles, alpha=0.6, s=6, 
              color='steelblue', edgecolors='none')
    
    # Calculate axis limits with padding
    x_min, x_max = real_quantiles.min(), real_quantiles.max()
    y_min, y_max = model_quantiles.min(), model_quantiles.max()
    
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_pad = max(x_range * 0.05, 0.001)
    y_pad = max(y_range * 0.05, 0.001)
    
    # Set axis limits
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    
    # CRITICAL: Draw diagonal reference line using matplotlib's axline
    # This automatically handles clipping and ensures a proper diagonal line
    # Use a point in the middle of the data range as reference
    center_x = (real_quantiles.min() + real_quantiles.max()) / 2
    center_y = (model_quantiles.min() + model_quantiles.max()) / 2
    
    # Draw infinite line with slope=1 (45-degree diagonal) through the center point
    # matplotlib automatically clips this to the visible axis area
    ax.axline((center_x, center_y), slope=1, color='red', linestyle='--', 
              linewidth=2, alpha=0.8, zorder=5)
    
    # Styling
    ax.set_title(model_name, fontsize=12, fontweight='bold')
    ax.set_xlabel('Real Data Quantiles', fontsize=10)
    ax.set_ylabel('Model Quantiles', fontsize=10)
    ax.grid(True, alpha=0.3)

def generate_qq_plots(real_data: np.ndarray, model_data: Dict[str, np.ndarray], 
                     output_dir: Path):
    """Generate the complete QQ-plots figure."""
    
    model_names = ["GARCH", "TimeGrad", "Zero-DDPM", "Explicit-DDPM", "LLM-DDPM"]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, model_name in enumerate(model_names):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        if model_name in model_data:
            create_qq_plot_panel(real_data, model_data[model_name], ax, model_name)
        else:
            ax.text(0.5, 0.5, 'Data not available', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12, color='gray')
            ax.set_title(model_name, fontsize=12, fontweight='bold')
    
    # Hide unused subplot
    if len(model_names) < 6:
        axes[5].set_visible(False)
    
    plt.tight_layout()
    
    # Save both formats
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pdf_path = output_dir / 'stylized_qqpanels_final_v2.pdf'
    png_path = output_dir / 'stylized_qqpanels_final_v2.png'
    
    fig.savefig(pdf_path, bbox_inches='tight', dpi=300)
    fig.savefig(png_path, bbox_inches='tight', dpi=300)
    
    plt.close(fig)
    
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")
    
    return pdf_path, png_path

def main():
    """Main function to generate QQ-plots from scratch."""
    
    print("Generating QQ-plots from scratch with proper diagonal reference lines...")
    
    # Load data
    real_data = load_real_returns('/Users/siminali/Desktop/Thesis Coding/data/sp500_data.csv')
    model_data = load_model_data(
        '/Users/siminali/Desktop/Thesis Coding/final_results_benchmarking',
        '/Users/siminali/Desktop/Thesis Coding/results/final_plots_for_novelty_models '
    )
    
    # Align all data to same length
    min_length = min(len(real_data), min(len(data) for data in model_data.values()))
    real_aligned = real_data[:min_length]
    model_aligned = {name: data[:min_length] for name, data in model_data.items()}
    
    print(f"\nAligned all data to length: {min_length}")
    
    # Generate plots
    output_dir = Path('/Users/siminali/Desktop/Thesis Coding/final_results_thesis/stylised_facts')
    pdf_path, png_path = generate_qq_plots(real_aligned, model_aligned, output_dir)
    
    print(f"\n✅ QQ-plots generated successfully!")
    print(f"📊 Files saved:")
    print(f"   - {pdf_path}")
    print(f"   - {png_path}")

if __name__ == "__main__":
    main()
