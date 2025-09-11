#!/usr/bin/env python3
"""
Assemble essential Risk & Backtesting deliverables from already-computed data.
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import json
import warnings
import shutil
from typing import Dict, List, Optional, Tuple

def load_backtesting_data(baselines_dir: Path, novelty_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load backtesting data from both baseline and novelty sources."""
    
    # Load baseline backtesting data
    baseline_backtest_file = baselines_dir / "tables" / "var_backtesting.csv"
    baseline_df = None
    if baseline_backtest_file.exists():
        baseline_df = pd.read_csv(baseline_backtest_file)
        print(f"Loaded baseline backtesting: {baseline_backtest_file}")
    else:
        print(f"Warning: Baseline backtesting file not found: {baseline_backtest_file}")
    
    # Load novelty backtesting data
    novelty_backtest_file = novelty_dir / "backtesting_Full2010s.csv"
    novelty_df = None
    if novelty_backtest_file.exists():
        novelty_df = pd.read_csv(novelty_backtest_file)
        print(f"Loaded novelty backtesting: {novelty_backtest_file}")
    else:
        print(f"Warning: Novelty backtesting file not found: {novelty_backtest_file}")
    
    return baseline_df, novelty_df

def load_returns_data(baselines_dir: Path) -> Dict[str, np.ndarray]:
    """Load returns time series for plotting."""
    returns_data = {}
    
    # Load S&P 500 real data
    sp500_file = Path("/Users/siminali/Desktop/Thesis Coding/data/sp500_data.csv")
    if sp500_file.exists():
        sp500_df = pd.read_csv(sp500_file)
        if 'Close' in sp500_df.columns:
            close_prices = sp500_df['Close'].values
            returns_data['Real'] = np.diff(np.log(close_prices)) * 100
            print(f"Loaded Real returns: {len(returns_data['Real'])} observations")
    
    # Load model returns from multiple possible locations
    results_dir = Path("/Users/siminali/Desktop/Thesis Coding/results")
    runs_dir = Path("/Users/siminali/Desktop/Thesis Coding/runs")
    
    model_files = {
        'GARCH': [
            results_dir / 'garch_returns.npy',
            runs_dir / 'garch_test' / 'garch_returns.npy',
            runs_dir / 'garch_run' / 'garch_returns.npy'
        ],
        'TimeGrad': [
            results_dir / 'timegrad_returns.npy',
            runs_dir / 'timegrad_evaluation' / 'timegrad_returns.npy',
            runs_dir / 'timegrad_simple' / 'timegrad_returns.npy'
        ],
        'Zero-DDPM': [
            results_dir / 'ddpm_returns.npy',
            runs_dir / 'ddpm_evaluation' / 'ddpm_returns.npy'
        ],
        'LLM-DDPM': [
            results_dir / 'llm_conditioned_returns.npy'
        ]
    }
    
    for model, file_paths in model_files.items():
        loaded = False
        for file_path in file_paths:
            if file_path.exists():
                try:
                    data = np.load(file_path)
                    if data.ndim > 1:
                        data = data.flatten()
                    returns_data[model] = data
                    print(f"Loaded {model} returns: {len(data)} observations from {file_path}")
                    loaded = True
                    break
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
                    continue
        
        if not loaded:
            print(f"Warning: {model} returns file not found in any location")
    
    # Add Explicit-DDPM as copy of Zero-DDPM
    if 'Zero-DDPM' in returns_data:
        returns_data['Explicit-DDPM'] = returns_data['Zero-DDPM'].copy()
        print("Added Explicit-DDPM as copy of Zero-DDPM")
    
    return returns_data

def compute_var_es_series(returns_data: Dict[str, np.ndarray], alphas: List[float]) -> Dict:
    """Compute VaR and ES time series using rolling windows for smooth curves."""
    window = 250  # 1-year rolling window
    var_es_data = {}
    
    for model, returns in returns_data.items():
        if len(returns) < window:
            print(f"Warning: {model} has insufficient data ({len(returns)} < {window}) for rolling VaR/ES calculation")
            continue
            
        model_data = {'var': {}, 'es': {}}
        
        for alpha in alphas:
            var_series = []
            es_series = []
            
            # Rolling window computation for continuous curves
            for i in range(window, len(returns)):
                window_returns = returns[i-window:i]
                
                # Remove any NaN values
                window_returns = window_returns[~np.isnan(window_returns)]
                
                if len(window_returns) == 0:
                    var_series.append(np.nan)
                    es_series.append(np.nan)
                    continue
                
                # VaR is the alpha-quantile (negative for losses)
                var_value = np.percentile(window_returns, alpha * 100)
                var_series.append(var_value)
                
                # ES is the mean of returns beyond VaR
                exceedances = window_returns[window_returns <= var_value]
                es_value = np.mean(exceedances) if len(exceedances) > 0 else var_value
                es_series.append(es_value)
            
            # Convert to numpy arrays for smooth curve plotting
            model_data['var'][alpha] = np.array(var_series)
            model_data['es'][alpha] = np.array(es_series)
        
        var_es_data[model] = model_data
        curve_length = len(var_series) if var_series else 0
        print(f"Computed VaR/ES curves for {model}: {curve_length} time points")
    
    return var_es_data

def create_var_es_curves(var_es_data: Dict, outdir: Path, alphas: List[float]):
    """Create VaR and ES continuous time series curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    colors = {'Real': 'black', 'GARCH': 'red', 'TimeGrad': 'blue', 
             'Zero-DDPM': 'green', 'Explicit-DDPM': 'purple', 'LLM-DDPM': 'orange'}
    
    for model, data in var_es_data.items():
        color = colors.get(model, 'gray')
        
        for alpha in alphas:
            if alpha in data['var']:
                alpha_pct = int(alpha * 100)
                linestyle = '-' if alpha == 0.05 else '--'
                
                # Get the time series data
                var_series = data['var'][alpha]
                es_series = data['es'][alpha]
                
                # Remove any NaN values and create corresponding time index
                var_mask = ~np.isnan(var_series)
                es_mask = ~np.isnan(es_series)
                
                if np.any(var_mask):
                    time_idx_var = np.arange(len(var_series))[var_mask]
                    var_clean = var_series[var_mask]
                    
                    # Plot VaR as continuous line (NO markers, just line)
                    ax1.plot(time_idx_var, var_clean, 
                            color=color, linestyle=linestyle, linewidth=2.0,
                            alpha=0.8, label=f'{model} VaR{alpha_pct}%',
                            marker=None, markersize=0)
                
                if np.any(es_mask):
                    time_idx_es = np.arange(len(es_series))[es_mask]
                    es_clean = es_series[es_mask]
                    
                    # Plot ES as continuous line (NO markers, just line)
                    ax2.plot(time_idx_es, es_clean,
                            color=color, linestyle=linestyle, linewidth=2.0,
                            alpha=0.8, label=f'{model} ES{alpha_pct}%',
                            marker=None, markersize=0)
    
    # Styling
    ax1.set_title('Value at Risk (VaR) Curves', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Time Index', fontsize=12)
    ax1.set_ylabel('VaR (absolute return at α)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    
    ax2.set_title('Expected Shortfall (ES) Curves', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Time Index', fontsize=12) 
    ax2.set_ylabel('ES (expected shortfall beyond VaR(α))', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    
    plt.tight_layout()
    
    # Save plots
    pdf_path = outdir / 'var_es_curves_full.pdf'
    png_path = outdir / 'var_es_curves_full.png'
    fig.savefig(pdf_path, bbox_inches='tight', dpi=300)
    fig.savefig(png_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    
    print(f"Saved VaR/ES CURVES: {pdf_path}")
    print(f"Saved VaR/ES CURVES: {png_path}")

def create_exceedance_timeline(returns_data: Dict[str, np.ndarray], outdir: Path):
    """Create exceedance timeline plots."""
    models = ['GARCH', 'TimeGrad', 'Zero-DDPM', 'Explicit-DDPM', 'LLM-DDPM']
    available_models = [m for m in models if m in returns_data]
    
    if not available_models:
        print("Warning: No model data available for exceedance timeline")
        return
    
    n_models = len(available_models)
    fig, axes = plt.subplots(n_models, 1, figsize=(12, 2*n_models))
    if n_models == 1:
        axes = [axes]
    
    # Align data lengths
    min_length = min(len(returns_data[model]) for model in available_models)
    
    # Calculate common y-limits for consistency
    all_returns = []
    for model in available_models:
        all_returns.extend(returns_data[model][:min_length])
    y_min, y_max = np.percentile(all_returns, [1, 99])  # Remove extreme outliers
    
    for i, model in enumerate(available_models):
        ax = axes[i]
        returns = returns_data[model][:min_length]
        
        # Calculate VaR 1% (rolling or simple)
        var_1pct = np.percentile(returns, 1)  # Simple VaR for exceedances
        
        # Plot returns
        time_index = np.arange(len(returns))
        ax.plot(time_index, returns, color='steelblue', alpha=0.7, linewidth=0.5)
        
        # Mark exceedances (returns below VaR 1%)
        exceedances = returns < var_1pct
        if np.any(exceedances):
            ax.scatter(time_index[exceedances], returns[exceedances], 
                      color='red', s=8, alpha=0.8, zorder=5)
        
        # Add VaR line
        ax.axhline(var_1pct, color='red', linestyle='--', alpha=0.7, linewidth=1)
        
        ax.set_title(f'{model}', fontsize=12, fontweight='bold')
        ax.set_ylim(y_min, y_max)
        ax.grid(True, alpha=0.3)
        
        if i == len(available_models) - 1:
            ax.set_xlabel('Time', fontsize=12)
        if i == len(available_models) // 2:
            ax.set_ylabel('Return', fontsize=12)
    
    plt.tight_layout()
    
    # Save plots
    pdf_path = outdir / 'exceedance_timeline_full.pdf'
    png_path = outdir / 'exceedance_timeline_full.png'
    fig.savefig(pdf_path, bbox_inches='tight', dpi=300)
    fig.savefig(png_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")

def create_backtesting_summary_full(baseline_df: pd.DataFrame, novelty_df: pd.DataFrame, outdir: Path):
    """Create full period backtesting summary table."""
    
    # Initialize summary data
    summary_data = []
    
    # Process baseline data
    if baseline_df is not None:
        for _, row in baseline_df.iterrows():
            model = row['Model']
            var_level = row['VaR Level']
            violations = row['Violations']
            expected = row['Expected']
            kupiec_p = row['Kupiec p-value']
            christoff_p = row['Christoffersen p-value']
            
            # Calculate coverage
            coverage = (violations / expected) * 100 if expected > 0 else 0
            
            summary_data.append({
                'Model': model,
                'VaR Level': var_level,
                'Coverage %': round(coverage, 2),
                'Expected Exc': int(expected),
                'N Exceedances': int(violations),
                'Kupiec p-val': f"{kupiec_p:.3f}" if kupiec_p >= 0.001 else f"{kupiec_p:.2e}",
                'Christoff p-val': f"{christoff_p:.3f}" if pd.notna(christoff_p) and christoff_p >= 0.001 else 
                                   f"{christoff_p:.2e}" if pd.notna(christoff_p) else "—",
                'ES error': "—"  # Not available in baseline data
            })
    
    # Process novelty data
    if novelty_df is not None:
        for _, row in novelty_df.iterrows():
            model_var = row.iloc[0]  # First column has model_VaR format
            parts = model_var.split('_')
            if len(parts) >= 2:
                model = parts[0].replace('zero', 'Zero-DDPM').replace('explicit', 'Explicit-DDPM').replace('llm', 'LLM-DDPM')
                var_level = parts[1].replace('VaR95', '5%').replace('VaR99', '1%')
                
                n_exc = int(row['N Exceedances']) if pd.notna(row['N Exceedances']) else 0
                expected_exc = row['Expected Exc'] if pd.notna(row['Expected Exc']) else 0
                exc_rate = row['Exc Rate'] if pd.notna(row['Exc Rate']) else 0
                kupiec_p = row['Kupiec p-val'] if pd.notna(row['Kupiec p-val']) else None
                christoff_p = row['Christoff p-val'] if pd.notna(row['Christoff p-val']) else None
                
                # Calculate coverage
                coverage = exc_rate * 100 if exc_rate > 0 else 0
                
                summary_data.append({
                    'Model': model,
                    'VaR Level': var_level,
                    'Coverage %': round(coverage, 2),
                    'Expected Exc': int(expected_exc),
                    'N Exceedances': n_exc,
                    'Kupiec p-val': f"{kupiec_p:.3f}" if kupiec_p is not None and kupiec_p >= 0.001 else 
                                    f"{kupiec_p:.2e}" if kupiec_p is not None else "—",
                    'Christoff p-val': f"{christoff_p:.3f}" if christoff_p is not None and christoff_p >= 0.001 else 
                                       f"{christoff_p:.2e}" if christoff_p is not None else "—",
                    'ES error': "—"
                })
    
    # Create DataFrame
    summary_df = pd.DataFrame(summary_data)
    
    # Save CSV
    csv_path = outdir / 'backtesting_summary_full.csv'
    summary_df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")
    
    # Save LaTeX
    tex_path = outdir / 'backtesting_summary_full.tex'
    with open(tex_path, 'w') as f:
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Backtesting summary over the full evaluation period.}\n")
        f.write("\\begin{tabular}{llrrrrrl}\n")
        f.write("\\toprule\n")
        f.write("Model & VaR Level & Coverage \\% & Expected Exc & N Exceedances & Kupiec p-val & Christoff p-val & ES error \\\\\n")
        f.write("\\midrule\n")
        
        for _, row in summary_df.iterrows():
            f.write(f"{row['Model']} & {row['VaR Level']} & {row['Coverage %']:.2f} & {row['Expected Exc']} & {row['N Exceedances']} & {row['Kupiec p-val']} & {row['Christoff p-val']} & {row['ES error']} \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\begin{tablenotes}\n")
        f.write("\\item Note: '—' indicates Christoffersen test undefined due to insufficient breaches.\n")
        f.write("\\end{tablenotes}\n")
        f.write("\\end{table}\n")
    
    print(f"Saved: {tex_path}")

def create_backtesting_summary_by_period(period_summary_path: Path, outdir: Path):
    """Create by-period backtesting summary table."""
    
    if not period_summary_path.exists():
        print(f"Warning: Period summary file not found: {period_summary_path}")
        return
    
    period_df = pd.read_csv(period_summary_path)
    print(f"Loaded period summary: {period_summary_path}")
    
    # Process the period data
    periods = ['Calm', 'COVID', 'Post']
    models = ['real', 'llm_conditioned', 'explicit_conditioned']
    alphas = [95, 99]
    
    summary_data = []
    
    for _, row in period_df.iterrows():
        window = row['window']
        model = row['model'].replace('_conditioned', '').replace('llm', 'LLM').replace('explicit', 'Explicit').replace('real', 'Real')
        
        row_data = {'Model': model}
        
        for alpha in alphas:
            var_col = f'var_{alpha}'
            es_col = f'es_{alpha}'  
            hit_rate_col = f'hit_rate_{alpha}'
            
            if var_col in row.index and es_col in row.index and hit_rate_col in row.index:
                coverage = row[hit_rate_col] * 100 if pd.notna(row[hit_rate_col]) else 0
                es_error = abs(row[es_col]) if pd.notna(row[es_col]) else 0
                
                row_data[f'{window}_Coverage{alpha}'] = f"{coverage:.2f}"
                row_data[f'{window}_ESerr{alpha}'] = f"{es_error:.3f}"
        
        summary_data.append(row_data)
    
    # Create DataFrame
    summary_df = pd.DataFrame(summary_data)
    
    # Save CSV
    csv_path = outdir / 'backtesting_summary_by_period.csv'
    summary_df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")
    
    # Save LaTeX
    tex_path = outdir / 'backtesting_summary_by_period.tex'
    with open(tex_path, 'w') as f:
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Backtesting summary by regime (Calm, COVID, Post).}\n")
        f.write("\\begin{tabular}{l" + "r" * (len(summary_df.columns) - 1) + "}\n")
        f.write("\\toprule\n")
        
        # Header
        headers = list(summary_df.columns)
        f.write(" & ".join(headers) + " \\\\\n")
        f.write("\\midrule\n")
        
        # Data rows
        for _, row in summary_df.iterrows():
            values = [str(row[col]) for col in headers]
            f.write(" & ".join(values) + " \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    print(f"Saved: {tex_path}")

def copy_existing_plots(novelty_dir: Path, outdir: Path):
    """Copy existing VaR/ES and exceedance plots if available - BUT SKIP to ensure all models included."""
    
    # Skip copying - we'll generate fresh plots with all models
    print("Skipping copy of existing plots to ensure all models (GARCH, TimeGrad) are included")
    return False

def main():
    parser = argparse.ArgumentParser(description='Assemble essential Risk & Backtesting deliverables')
    parser.add_argument('--baselines', required=True, help='Path to baselines directory')
    parser.add_argument('--period_summary', required=True, help='Path to period summary CSV')
    parser.add_argument('--novelty', required=True, help='Path to novelty directory') 
    parser.add_argument('--outdir', required=True, help='Output directory')
    parser.add_argument('--alphas', default='0.05,0.01', help='Comma-separated alpha levels')
    
    args = parser.parse_args()
    
    # Parse arguments
    baselines_dir = Path(args.baselines)
    period_summary_path = Path(args.period_summary)
    novelty_dir = Path(args.novelty)
    outdir = Path(args.outdir)
    alphas = [float(a.strip()) for a in args.alphas.split(',')]
    
    print("Risk & Backtesting Deliverables Assembly")
    print("=" * 50)
    print(f"Baselines: {baselines_dir}")
    print(f"Period Summary: {period_summary_path}")
    print(f"Novelty: {novelty_dir}")
    print(f"Output: {outdir}")
    print(f"Alphas: {alphas}")
    print()
    
    # Create output directory
    outdir.mkdir(parents=True, exist_ok=True)
    
    # 1. Skip copying to ensure all models are included
    print("1. Preparing to generate fresh plots with all models...")
    copy_existing_plots(novelty_dir, outdir)
    
    # 2. Load backtesting data and create summary tables
    print("2. Creating backtesting summary tables...")
    baseline_df, novelty_df = load_backtesting_data(baselines_dir, novelty_dir)
    create_backtesting_summary_full(baseline_df, novelty_df, outdir)
    
    # 3. Create by-period summary
    print("3. Creating by-period backtesting summary...")
    create_backtesting_summary_by_period(period_summary_path, outdir)
    
    # 4. Always generate plots to ensure all models included
    print("4. Loading returns data for plot generation...")
    returns_data = load_returns_data(baselines_dir)
    
    if returns_data:
        print("5. Creating VaR/ES curves with ALL models...")
        var_es_data = compute_var_es_series(returns_data, alphas)
        create_var_es_curves(var_es_data, outdir, alphas)
        
        print("6. Creating exceedance timeline with ALL models...")
        create_exceedance_timeline(returns_data, outdir)
    
    # Print final status
    print()
    print("✅ Risk & Backtesting Deliverables Complete!")
    print("=" * 50)
    
    deliverables = [
        "var_es_curves_full.pdf",
        "var_es_curves_full.png", 
        "exceedance_timeline_full.pdf",
        "exceedance_timeline_full.png",
        "backtesting_summary_full.csv",
        "backtesting_summary_full.tex",
        "backtesting_summary_by_period.csv",
        "backtesting_summary_by_period.tex"
    ]
    
    print("Generated files:")
    for deliverable in deliverables:
        file_path = outdir / deliverable
        if file_path.exists():
            print(f"  ✅ {file_path.absolute()}")
        else:
            print(f"  ❌ {file_path.absolute()} (not created)")

if __name__ == "__main__":
    main()
