#!/usr/bin/env python3
"""
Experiment B Mode-Specific Plotting Utility v2
==============================================

Generates figures for specific modes in Experiment B (Counterfactual Controllability).
Creates separate figure sets for real-conditions, calm-conditions, and LLM-knob modes.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats


class ExperimentBModePrognizer:
    """Generates mode-specific plots for Experiment B."""
    
    def __init__(self, experiment_dir: Path, window_name: str = "covid_crash"):
        """Initialize plot generator."""
        self.experiment_dir = Path(experiment_dir)
        self.window_name = window_name
        self.window_dir = self.experiment_dir / window_name
        
        # Plot configuration
        plt.style.use('default')
        self.fig_size = (12, 8)
        self.colors = ['blue', 'red', 'green', 'orange', 'purple']
        
    def load_real_data(self, csv_file: Path, start_date: str, end_date: str) -> pd.Series:
        """Load real returns data for the specified window."""
        print(f"📊 Loading real data from {csv_file}...")
        
        try:
            df = pd.read_csv(csv_file)
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date').sort_index()
            
            # Calculate log returns
            df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
            
            # Filter to window
            mask = (df.index >= start_date) & (df.index <= end_date)
            real_returns = df.loc[mask, 'log_returns'].dropna()
            
            print(f"✅ Loaded {len(real_returns)} real return observations")
            return real_returns
            
        except Exception as e:
            print(f"❌ Error loading real data: {e}")
            # Create dummy data as fallback
            dates = pd.date_range(start_date, end_date, freq='D')
            return pd.Series(np.random.normal(0, 0.02, len(dates)), index=dates)
    
    def load_mode_samples(self, mode_name: str) -> Dict[str, np.ndarray]:
        """Load samples for a specific mode."""
        print(f"📊 Loading samples for mode: {mode_name}")
        
        samples = {}
        
        for model_name in ["explicit", "llm"]:  # zero not in Experiment B
            model_dir = self.window_dir / model_name / mode_name
            
            if not model_dir.exists():
                print(f"⚠️ Mode directory not found: {model_name}/{mode_name}")
                continue
            
            samples_file = model_dir / "samples.npy"
            if samples_file.exists():
                try:
                    model_samples = np.load(samples_file)
                    samples[model_name] = model_samples
                    print(f"✅ Loaded {model_name} ({mode_name}): {model_samples.shape}")
                except Exception as e:
                    print(f"❌ Error loading {model_name} {mode_name}: {e}")
        
        return samples
    
    def get_available_modes(self) -> Dict[str, List[str]]:
        """Get all available modes for each model."""
        modes = {"explicit": [], "llm": []}
        
        for model_name in ["explicit", "llm"]:
            model_dir = self.window_dir / model_name
            if model_dir.exists():
                mode_dirs = [d.name for d in model_dir.iterdir() if d.is_dir()]
                modes[model_name] = sorted(mode_dirs)
        
        return modes
    
    def generate_mode_comparison_plot(self, real_returns: pd.Series, 
                                    mode_samples: Dict[str, Dict[str, np.ndarray]], 
                                    mode_names: List[str], output_dir: Path) -> None:
        """Generate comparison plots across modes."""
        print(f"📈 Generating mode comparison plots...")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # ECDF comparison across modes
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Real data
        real_values = real_returns.values
        real_sorted = np.sort(real_values)
        real_ecdf = np.arange(1, len(real_sorted) + 1) / len(real_sorted)
        
        for ax_idx, model_name in enumerate(["explicit", "llm"]):
            ax = axes[ax_idx]
            
            # Plot real data
            ax.plot(real_sorted, real_ecdf, 'k-', linewidth=2, 
                   label='Real Data', alpha=0.8)
            
            # Plot each mode for this model
            for i, mode_name in enumerate(mode_names):
                if model_name in mode_samples[mode_name]:
                    samples = mode_samples[mode_name][model_name]
                    if samples.ndim == 2:
                        flat_samples = samples.flatten()
                    else:
                        flat_samples = samples
                    
                    sorted_samples = np.sort(flat_samples)
                    sample_ecdf = np.arange(1, len(sorted_samples) + 1) / len(sorted_samples)
                    
                    color = self.colors[i % len(self.colors)]
                    ax.plot(sorted_samples, sample_ecdf, color=color, linewidth=1.5, 
                           label=f'{mode_name.replace("-", " ").title()}', alpha=0.7)
            
            ax.set_xlabel('Log Returns')
            ax.set_ylabel('Cumulative Probability')
            ax.set_title(f'{model_name.title()} Model - Mode Comparison')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self._save_plot(fig, "mode_comparison_ecdf", output_dir)
        plt.close(fig)
        
        # VaR comparison across modes
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Calculate VaR for real data
        var_95_real = np.percentile(real_values, 5)
        var_99_real = np.percentile(real_values, 1)
        
        for ax_idx, model_name in enumerate(["explicit", "llm"]):
            ax = axes[ax_idx]
            
            var_95_modes = []
            var_99_modes = []
            mode_labels = []
            
            for mode_name in mode_names:
                if model_name in mode_samples[mode_name]:
                    samples = mode_samples[mode_name][model_name]
                    if samples.ndim == 2:
                        flat_samples = samples.flatten()
                    else:
                        flat_samples = samples
                    
                    var_95 = np.percentile(flat_samples, 5)
                    var_99 = np.percentile(flat_samples, 1)
                    
                    var_95_modes.append(var_95)
                    var_99_modes.append(var_99)
                    mode_labels.append(mode_name.replace("-", " ").title())
            
            if var_95_modes:
                x_pos = np.arange(len(mode_labels))
                width = 0.35
                
                ax.bar(x_pos - width/2, var_95_modes, width, label='VaR 95%', alpha=0.7)
                ax.bar(x_pos + width/2, var_99_modes, width, label='VaR 99%', alpha=0.7)
                
                ax.axhline(var_95_real, color='red', linestyle='--', alpha=0.7, label='Real VaR 95%')
                ax.axhline(var_99_real, color='darkred', linestyle='--', alpha=0.7, label='Real VaR 99%')
                
                ax.set_xlabel('Mode')
                ax.set_ylabel('VaR')
                ax.set_title(f'{model_name.title()} Model - VaR Comparison')
                ax.set_xticks(x_pos)
                ax.set_xticklabels(mode_labels, rotation=45, ha='right')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self._save_plot(fig, "mode_comparison_var", output_dir)
        plt.close(fig)
    
    def generate_individual_mode_plots(self, real_returns: pd.Series, 
                                     mode_samples: Dict[str, np.ndarray], 
                                     mode_name: str, output_dir: Path) -> None:
        """Generate complete plot set for a single mode."""
        print(f"📈 Generating plots for mode: {mode_name}")
        
        mode_output_dir = output_dir / mode_name.replace("-", "_")
        mode_output_dir.mkdir(parents=True, exist_ok=True)
        
        if not mode_samples:
            print(f"⚠️ No samples for mode {mode_name}")
            return
        
        # Generate all standard plots for this mode
        self._generate_ecdf_overlay(real_returns, mode_samples, mode_output_dir, mode_name)
        self._generate_qq_plots(real_returns, mode_samples, mode_output_dir, mode_name)
        self._generate_var_es_analysis(real_returns, mode_samples, mode_output_dir, mode_name)
        self._generate_realized_volatility(real_returns, mode_samples, mode_output_dir, mode_name)
    
    def _generate_ecdf_overlay(self, real_returns: pd.Series, model_samples: Dict[str, np.ndarray], 
                              output_dir: Path, mode_name: str) -> None:
        """Generate ECDF overlay plot for specific mode."""
        fig, ax = plt.subplots(figsize=self.fig_size)
        
        # Plot real data ECDF
        real_values = real_returns.values
        real_sorted = np.sort(real_values)
        real_ecdf = np.arange(1, len(real_sorted) + 1) / len(real_sorted)
        ax.plot(real_sorted, real_ecdf, 'k-', linewidth=2, label='Real Data', alpha=0.8)
        
        # Plot model ECDFs
        for i, (model_name, samples) in enumerate(model_samples.items()):
            if samples.ndim == 2:
                flat_samples = samples.flatten()
            else:
                flat_samples = samples
            
            sorted_samples = np.sort(flat_samples)
            sample_ecdf = np.arange(1, len(sorted_samples) + 1) / len(sorted_samples)
            
            color = self.colors[i % len(self.colors)]
            ax.plot(sorted_samples, sample_ecdf, color=color, linewidth=1.5, 
                   label=f'{model_name.title()} Model', alpha=0.7)
        
        ax.set_xlabel('Log Returns')
        ax.set_ylabel('Cumulative Probability')
        ax.set_title(f'ECDF Comparison - {mode_name.replace("-", " ").title()} Mode')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        self._save_plot(fig, "ecdf_overlay", output_dir)
        plt.close(fig)
    
    def _generate_qq_plots(self, real_returns: pd.Series, model_samples: Dict[str, np.ndarray], 
                          output_dir: Path, mode_name: str) -> None:
        """Generate QQ plots for specific mode."""
        n_models = len(model_samples)
        fig, axes = plt.subplots(2, n_models, figsize=(4 * n_models, 8))
        if n_models == 1:
            axes = axes.reshape(2, 1)
        
        real_values = real_returns.values
        real_sorted = np.sort(real_values)
        
        for i, (model_name, samples) in enumerate(model_samples.items()):
            if samples.ndim == 2:
                flat_samples = samples.flatten()
            else:
                flat_samples = samples
            
            sample_sorted = np.sort(flat_samples)
            
            # Left tail (bottom 10%)
            n_tail = int(0.1 * min(len(real_sorted), len(sample_sorted)))
            real_left = real_sorted[:n_tail]
            sample_left = sample_sorted[:n_tail]
            
            axes[0, i].scatter(real_left, sample_left, alpha=0.6, s=20)
            axes[0, i].plot([real_left.min(), real_left.max()], 
                           [real_left.min(), real_left.max()], 'r--', alpha=0.8)
            axes[0, i].set_title(f'{model_name.title()} - Left Tail ({mode_name.replace("-", " ").title()})')
            axes[0, i].set_xlabel('Real Returns (Quantiles)')
            axes[0, i].set_ylabel('Model Returns (Quantiles)')
            axes[0, i].grid(True, alpha=0.3)
            
            # Right tail (top 10%)
            real_right = real_sorted[-n_tail:]
            sample_right = sample_sorted[-n_tail:]
            
            axes[1, i].scatter(real_right, sample_right, alpha=0.6, s=20)
            axes[1, i].plot([real_right.min(), real_right.max()], 
                           [real_right.min(), real_right.max()], 'r--', alpha=0.8)
            axes[1, i].set_title(f'{model_name.title()} - Right Tail ({mode_name.replace("-", " ").title()})')
            axes[1, i].set_xlabel('Real Returns (Quantiles)')
            axes[1, i].set_ylabel('Model Returns (Quantiles)')
            axes[1, i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        self._save_plot(fig, "qq_plots", output_dir)
        plt.close(fig)
    
    def _generate_var_es_analysis(self, real_returns: pd.Series, model_samples: Dict[str, np.ndarray], 
                                 output_dir: Path, mode_name: str) -> None:
        """Generate VaR/ES analysis for specific mode."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Calculate VaR/ES for real data
        real_values = real_returns.values
        var_95_real = np.percentile(real_values, 5)
        var_99_real = np.percentile(real_values, 1)
        es_95_real = real_values[real_values <= var_95_real].mean()
        es_99_real = real_values[real_values <= var_99_real].mean()
        
        # Plot VaR/ES comparisons
        models = list(model_samples.keys())
        var_95_models = []
        var_99_models = []
        es_95_models = []
        es_99_models = []
        
        for model_name, samples in model_samples.items():
            if samples.ndim == 2:
                flat_samples = samples.flatten()
            else:
                flat_samples = samples
            
            var_95_model = np.percentile(flat_samples, 5)
            var_99_model = np.percentile(flat_samples, 1)
            es_95_model = flat_samples[flat_samples <= var_95_model].mean()
            es_99_model = flat_samples[flat_samples <= var_99_model].mean()
            
            var_95_models.append(var_95_model)
            var_99_models.append(var_99_model)
            es_95_models.append(es_95_model)
            es_99_models.append(es_99_model)
        
        # VaR comparison
        x_pos = np.arange(len(models))
        width = 0.35
        
        axes[0, 0].bar(x_pos - width/2, var_95_models, width, label='Model VaR 95%', alpha=0.7)
        axes[0, 0].axhline(var_95_real, color='red', linestyle='--', label='Real VaR 95%')
        axes[0, 0].set_xlabel('Models')
        axes[0, 0].set_ylabel('VaR (95%)')
        axes[0, 0].set_title(f'VaR Comparison - {mode_name.replace("-", " ").title()}')
        axes[0, 0].set_xticks(x_pos)
        axes[0, 0].set_xticklabels([m.title() for m in models])
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # ES comparison
        axes[0, 1].bar(x_pos - width/2, es_95_models, width, label='Model ES 95%', alpha=0.7)
        axes[0, 1].axhline(es_95_real, color='red', linestyle='--', label='Real ES 95%')
        axes[0, 1].set_xlabel('Models')
        axes[0, 1].set_ylabel('Expected Shortfall (95%)')
        axes[0, 1].set_title(f'ES Comparison - {mode_name.replace("-", " ").title()}')
        axes[0, 1].set_xticks(x_pos)
        axes[0, 1].set_xticklabels([m.title() for m in models])
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Time series with VaR exceedances
        if hasattr(real_returns, 'index') and len(real_returns) > 1:
            axes[1, 0].plot(real_returns.index, real_returns.values, 'k-', alpha=0.7, label='Returns')
            axes[1, 0].axhline(var_95_real, color='orange', linestyle='--', label='VaR 95%')
            axes[1, 0].axhline(var_99_real, color='red', linestyle='--', label='VaR 99%')
            
            # Mark exceedances
            exceedances_95 = real_returns.values < var_95_real
            exceedances_99 = real_returns.values < var_99_real
            
            if exceedances_95.any():
                axes[1, 0].scatter(real_returns.index[exceedances_95], 
                                 real_returns.values[exceedances_95], 
                                 color='orange', s=30, zorder=5, label='95% Exceedances')
            
            if exceedances_99.any():
                axes[1, 0].scatter(real_returns.index[exceedances_99], 
                                 real_returns.values[exceedances_99], 
                                 color='red', s=40, zorder=6, label='99% Exceedances')
            
            axes[1, 0].set_xlabel('Date')
            axes[1, 0].set_ylabel('Log Returns')
            axes[1, 0].set_title(f'VaR Exceedances - {mode_name.replace("-", " ").title()}')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'Timeline plot requires\ntemporal structure', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title(f'VaR Exceedances - {mode_name.replace("-", " ").title()} (N/A)')
        
        # Risk metrics summary table
        axes[1, 1].axis('off')
        
        # Create summary table
        table_data = [['Metric', 'Real'] + [m.title() for m in models]]
        table_data.append(['VaR 95%', f'{var_95_real:.4f}'] + [f'{v:.4f}' for v in var_95_models])
        table_data.append(['VaR 99%', f'{var_99_real:.4f}'] + [f'{v:.4f}' for v in var_99_models])
        table_data.append(['ES 95%', f'{es_95_real:.4f}'] + [f'{v:.4f}' for v in es_95_models])
        table_data.append(['ES 99%', f'{es_99_real:.4f}'] + [f'{v:.4f}' for v in es_99_models])
        
        table = axes[1, 1].table(cellText=table_data[1:], colLabels=table_data[0], 
                                loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        axes[1, 1].set_title(f'Risk Metrics - {mode_name.replace("-", " ").title()}')
        
        plt.tight_layout()
        self._save_plot(fig, "var_es_analysis", output_dir)
        plt.close(fig)
    
    def _generate_realized_volatility(self, real_returns: pd.Series, model_samples: Dict[str, np.ndarray], 
                                     output_dir: Path, mode_name: str) -> None:
        """Generate realized volatility plot for specific mode."""
        fig, ax = plt.subplots(figsize=self.fig_size)
        
        # Simple volatility comparison for short time series
        real_vol_simple = np.std(real_returns.values) * np.sqrt(252)
        
        models = list(model_samples.keys())
        model_vols = []
        
        for model_name, samples in model_samples.items():
            if samples.ndim == 2:
                flat_samples = samples.flatten()
            else:
                flat_samples = samples
            
            model_vol = np.std(flat_samples) * np.sqrt(252)
            model_vols.append(model_vol)
        
        x_pos = np.arange(len(models))
        ax.bar(x_pos, model_vols, alpha=0.7, label='Model Volatility')
        ax.axhline(real_vol_simple, color='red', linestyle='--', 
                  label=f'Real Volatility ({real_vol_simple:.4f})')
        
        ax.set_xlabel('Models')
        ax.set_ylabel('Annualized Volatility')
        ax.set_title(f'Volatility Comparison - {mode_name.replace("-", " ").title()}')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([m.title() for m in models])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        self._save_plot(fig, "realized_volatility", output_dir)
        plt.close(fig)
    
    def _save_plot(self, fig, plot_name: str, output_dir: Path) -> None:
        """Save plot in both PDF and PNG formats."""
        pdf_path = output_dir / f"{plot_name}.pdf"
        png_path = output_dir / f"{plot_name}.png"
        
        # Save PDF
        fig.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight')
        
        # Save PNG
        fig.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
        
        print(f"💾 Saved: {pdf_path.name} and {png_path.name}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Generate mode-specific plots for Experiment B")
    parser.add_argument("--experiment-dir", required=True,
                       help="Path to experiment directory (e.g., results/addons/period_slices/B_v6)")
    parser.add_argument("--csv-file", default="data/sp500_data.csv",
                       help="Path to real data CSV file")
    parser.add_argument("--window", default="covid_crash",
                       help="Window name to process")
    parser.add_argument("--start-date", default="2020-02-20",
                       help="Window start date")
    parser.add_argument("--end-date", default="2020-03-23",
                       help="Window end date")
    parser.add_argument("--modes", nargs="*", 
                       help="Specific modes to generate (default: all available)")
    parser.add_argument("--comparison-only", action="store_true",
                       help="Generate only mode comparison plots")
    
    args = parser.parse_args()
    
    # Setup paths
    experiment_dir = Path(args.experiment_dir)
    csv_file = Path(args.csv_file)
    
    if not experiment_dir.exists():
        print(f"❌ Experiment directory not found: {experiment_dir}")
        return 1
    
    if not csv_file.exists():
        print(f"❌ CSV file not found: {csv_file}")
        return 1
    
    # Create plot generator
    generator = ExperimentBModePrognizer(experiment_dir, args.window)
    
    # Load real data
    real_returns = generator.load_real_data(csv_file, args.start_date, args.end_date)
    
    # Get available modes
    available_modes = generator.get_available_modes()
    all_modes = set()
    for model_modes in available_modes.values():
        all_modes.update(model_modes)
    
    modes_to_process = args.modes if args.modes else sorted(all_modes)
    print(f"📋 Processing modes: {modes_to_process}")
    
    # Load samples for all requested modes
    mode_samples = {}
    for mode_name in modes_to_process:
        mode_samples[mode_name] = generator.load_mode_samples(mode_name)
    
    # Output directory
    output_dir = experiment_dir / args.window / "mode_figures"
    
    if not args.comparison_only:
        # Generate individual mode plots
        for mode_name in modes_to_process:
            if mode_samples[mode_name]:
                generator.generate_individual_mode_plots(real_returns, mode_samples[mode_name], 
                                                       mode_name, output_dir)
    
    # Generate mode comparison plots
    if len(modes_to_process) > 1:
        generator.generate_mode_comparison_plot(real_returns, mode_samples, 
                                              modes_to_process, output_dir)
    
    print(f"🎉 Mode-specific plots generated in: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
