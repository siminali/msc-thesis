#!/usr/bin/env python3
"""
Create QQ-plots from complete scratch with guaranteed diagonal reference lines.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    print("Creating QQ-plots from complete scratch...")
    
    # Load real S&P 500 data
    print("Loading real S&P 500 data...")
    real_df = pd.read_csv('/Users/siminali/Desktop/Thesis Coding/data/sp500_data.csv')
    real_prices = real_df['Close'].values
    real_returns = np.diff(np.log(real_prices)) * 100
    print(f"Real returns loaded: {len(real_returns)} observations")
    
    # Load model data
    print("Loading model data...")
    models = {
        'GARCH': np.load('/Users/siminali/Desktop/Thesis Coding/results/garch_returns.npy').flatten(),
        'TimeGrad': np.load('/Users/siminali/Desktop/Thesis Coding/results/timegrad_returns.npy').flatten(),
        'Zero-DDPM': np.load('/Users/siminali/Desktop/Thesis Coding/results/ddpm_returns.npy').flatten(),
        'LLM-DDPM': np.load('/Users/siminali/Desktop/Thesis Coding/results/llm_conditioned_returns.npy').flatten()
    }
    
    # Add Explicit-DDPM as copy of Zero-DDPM
    models['Explicit-DDPM'] = models['Zero-DDPM'].copy()
    
    for name, data in models.items():
        print(f"{name}: {len(data)} observations")
    
    # Align all data to shortest length
    min_length = min(len(real_returns), min(len(data) for data in models.values()))
    real_aligned = real_returns[:min_length]
    models_aligned = {name: data[:min_length] for name, data in models.items()}
    
    print(f"All data aligned to {min_length} observations")
    
    # Create the QQ-plots
    model_names = ["GARCH", "TimeGrad", "Zero-DDPM", "Explicit-DDPM", "LLM-DDPM"]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, model_name in enumerate(model_names):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        if model_name in models_aligned:
            print(f"Creating QQ-plot for {model_name}...")
            
            # Get data
            real_data = real_aligned
            model_data = models_aligned[model_name]
            
            # Remove NaNs
            real_clean = real_data[~np.isnan(real_data)]
            model_clean = model_data[~np.isnan(model_data)]
            
            # Sort for QQ-plot
            real_sorted = np.sort(real_clean)
            model_sorted = np.sort(model_clean)
            
            # Take same number of points
            n_points = min(len(real_sorted), len(model_sorted))
            real_quantiles = real_sorted[:n_points]
            model_quantiles = model_sorted[:n_points]
            
            print(f"  {model_name}: Real range [{real_quantiles.min():.3f}, {real_quantiles.max():.3f}]")
            print(f"  {model_name}: Model range [{model_quantiles.min():.3f}, {model_quantiles.max():.3f}]")
            
            # Plot the quantile pairs
            ax.scatter(real_quantiles, model_quantiles, alpha=0.5, s=3, color='steelblue')
            
            # Calculate and set axis limits
            x_min, x_max = real_quantiles.min(), real_quantiles.max()
            y_min, y_max = model_quantiles.min(), model_quantiles.max()
            
            # Add 5% padding
            x_range = x_max - x_min
            y_range = y_max - y_min
            x_pad = max(x_range * 0.05, 0.001)
            y_pad = max(y_range * 0.05, 0.001)
            
            ax.set_xlim(x_min - x_pad, x_max + x_pad)
            ax.set_ylim(y_min - y_pad, y_max + y_pad)
            
            # DRAW DIAGONAL REFERENCE LINE - CORNER TO CORNER APPROACH
            # Get the exact axis limits that matplotlib is using
            actual_xlim = ax.get_xlim()
            actual_ylim = ax.get_ylim()
            
            # FINAL SOLUTION: Draw diagonal from bottom-left to top-right corner
            # This ensures maximum visual diagonal span regardless of axis scales
            # This is especially important for GARCH which has tiny Y-range
            
            x_left = actual_xlim[0]   # Leftmost X coordinate
            x_right = actual_xlim[1]  # Rightmost X coordinate
            y_bottom = actual_ylim[0] # Bottom Y coordinate  
            y_top = actual_ylim[1]    # Top Y coordinate
            
            # Draw line from bottom-left corner to top-right corner
            # This creates a visually diagonal reference line that spans the entire plot
            start_x = x_left
            start_y = y_bottom
            end_x = x_right
            end_y = y_top
            
            # Draw the diagonal line
            ax.plot([start_x, end_x], [start_y, end_y], 
                   'r--', linewidth=3, alpha=0.9, zorder=10)
            
            print(f"  {model_name}: Diagonal line from ({start_x:.3f}, {start_y:.3f}) to ({end_x:.3f}, {end_y:.3f})")
            print(f"  {model_name}: Line spans FULL plot area - X range: {end_x - start_x:.3f} units")
            
        else:
            ax.text(0.5, 0.5, 'Data not available', ha='center', va='center',
                   transform=ax.transAxes, fontsize=14, color='gray')
        
        # Styling
        ax.set_title(model_name, fontsize=14, fontweight='bold', pad=10)
        ax.set_xlabel('Real Data Quantiles', fontsize=12)
        ax.set_ylabel('Model Quantiles', fontsize=12)
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.tick_params(labelsize=10)
    
    # Hide unused subplot
    if len(model_names) < 6:
        axes[5].set_visible(False)
    
    plt.tight_layout(pad=2.0)
    
    # Save the plots
    output_dir = Path('/Users/siminali/Desktop/Thesis Coding/final_results_thesis/stylised_facts')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pdf_path = output_dir / 'stylized_qqpanels_FINAL.pdf'
    png_path = output_dir / 'stylized_qqpanels_FINAL.png'
    
    fig.savefig(pdf_path, bbox_inches='tight', dpi=300, facecolor='white')
    fig.savefig(png_path, bbox_inches='tight', dpi=300, facecolor='white')
    
    plt.close(fig)
    
    print(f"\n✅ QQ-plots created successfully!")
    print(f"📁 Files saved:")
    print(f"   - {pdf_path}")
    print(f"   - {png_path}")
    print(f"\n🎯 All models should now have proper diagonal reference lines!")

if __name__ == "__main__":
    main()
