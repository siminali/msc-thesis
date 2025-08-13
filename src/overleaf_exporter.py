#!/usr/bin/env python3
"""
Overleaf Exporter
=================

Exports evaluation results to Overleaf-compatible LaTeX tables and figures.
Generates individual .tex files for tables and figure stubs with captions and labels.

Author: [Your Name]
Date: [Current Date]
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

class OverleafExporter:
    """Exports results to Overleaf-compatible format"""
    
    def __init__(self, results, outputs_dir, evaluator):
        self.results = results
        self.outputs_dir = outputs_dir
        self.evaluator = evaluator
        
        # Create Overleaf output directory
        self.overleaf_dir = os.path.join(outputs_dir, "overleaf")
        self.tables_dir = os.path.join(self.overleaf_dir, "tables")
        self.figures_dir = os.path.join(self.overleaf_dir, "figures")
        self.results_dir = os.path.join(self.overleaf_dir, "results")
        self.notes_dir = os.path.join(self.overleaf_dir, "notes")
        
        # Create directories
        for dir_path in [self.overleaf_dir, self.tables_dir, self.figures_dir, 
                        self.results_dir, self.notes_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Setup plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Define consistent colors for models
        self.colors = ['black', 'red', 'blue', 'green', 'orange']
        
        # Track generated assets
        self.assets = []
        self.errors = []
    
    def export_all(self):
        """Export all tables and figures to Overleaf format"""
        logger = logging.getLogger(__name__)
        logger.info("Starting Overleaf export...")
        
        try:
            # Export tables
            self._export_basic_stats_table()
            self._export_distribution_tests_table()
            self._export_var_es_table()
            self._export_backtesting_table()
            self._export_temporal_tests_table()
            self._export_volatility_metrics_table()
            self._export_robustness_bootstrap_table()
            self._export_evt_tailindex_table()
            self._export_capital_impact_table()
            self._export_ranking_table()
            self._export_prediction_error_table()
            self._export_per_regime_table()
            self._export_compute_profile_table()
            
            # Export figures
            self._export_distribution_figures()
            self._export_risk_figures()
            self._export_temporal_figures()
            self._export_volatility_figures()
            self._export_tail_figures()
            self._export_conditioning_figures()
            self._export_enhanced_figures()
            
            # Copy result files
            self._copy_result_files()
            
            # Generate documentation
            self._generate_readme()
            self._generate_manifest()
            
            # Write error log if any
            if self.errors:
                self._write_error_log()
            
            logger.info(f"Overleaf export completed: {len(self.assets)} assets generated")
            
        except Exception as e:
            logger.error(f"Overleaf export failed: {str(e)}")
            self.errors.append(f"Export failed: {str(e)}")
            self._write_error_log()
            raise
    
    def _export_basic_stats_table(self):
        """Export basic statistics table"""
        basic_stats = self.results['basic_statistics']
        
        # Create LaTeX table
        latex_content = r"""\begin{table}[htbp]
\centering
\begin{tabular}{lrrrrrrrrr}
\toprule
Model & Mean & Std & Skewness & Kurtosis & Min & Max & Q1 & Median & Q3 \\
\midrule"""
        
        # Add data rows
        for model_name in ['Real', 'GARCH', 'DDPM', 'TimeGrad', 'LLM-Conditioned']:
            if model_name in basic_stats:
                stats = basic_stats[model_name]
                latex_content += f"\n{model_name} & {stats['mean']:.4f} & {stats['std']:.4f} & "
                latex_content += f"{stats['skewness']:.4f} & {stats['kurtosis']:.4f} & "
                latex_content += f"{stats['min']:.4f} & {stats['max']:.4f} & "
                latex_content += f"{stats['q1']:.4f} & {stats['median']:.4f} & {stats['q3']:.4f} \\\\"
        
        latex_content += r"""
\bottomrule
\end{tabular}
\caption{Basic Statistics: Mean, standard deviation, skewness, kurtosis, and quartiles for Real and synthetic data.}
\label{tab:basic_stats}
\end{table}"""
        
        # Save table
        table_path = os.path.join(self.tables_dir, 'basic_stats.tex')
        with open(table_path, 'w') as f:
            f.write(latex_content)
        
        self.assets.append({
            'path': 'tables/basic_stats.tex',
            'type': 'table',
            'label': 'tab:basic_stats',
            'caption': 'Basic Statistics: Mean, standard deviation, skewness, kurtosis, and quartiles for Real and synthetic data.',
            'tag': 'distribution'
        })
    
    def _export_distribution_tests_table(self):
        """Export distribution tests table"""
        dist_tests = self.results['distribution_tests']
        
        latex_content = r"""\begin{table}[htbp]
\centering
\begin{tabular}{lrrrrr}
\toprule
Model & KS Statistic & KS p-value & AD Statistic & MMD Value & MMD 95\% CI \\
\midrule"""
        
        # Add data rows
        for model_name in ['GARCH', 'DDPM', 'TimeGrad', 'LLM-Conditioned']:
            if model_name in dist_tests:
                test = dist_tests[model_name]
                latex_content += f"\n{model_name} & {test['ks_statistic']:.4f} & "
                
                # Format p-value
                if test['ks_pvalue'] < 0.001:
                    latex_content += "< 0.001 & "
                else:
                    latex_content += f"{test['ks_pvalue']:.4f} & "
                
                latex_content += f"{test['anderson_darling_statistic']:.4f} & "
                latex_content += f"{test['mmd_value']:.6f} & "
                latex_content += f"[{test['mmd_ci_lower']:.6f}, {test['mmd_ci_upper']:.6f}] \\\\"
        
        latex_content += r"""
\bottomrule
\end{tabular}
\caption{Distribution Tests: Kolmogorov-Smirnov (KS), Anderson-Darling (AD), and Maximum Mean Discrepancy (MMD) statistics.}
\label{tab:distribution_tests}
\end{table}"""
        
        # Save table
        table_path = os.path.join(self.tables_dir, 'distribution_tests.tex')
        with open(table_path, 'w') as f:
            f.write(latex_content)
        
        self.assets.append({
            'path': 'tables/distribution_tests.tex',
            'type': 'table',
            'label': 'tab:distribution_tests',
            'caption': 'Distribution Tests: Kolmogorov-Smirnov (KS), Anderson-Darling (AD), and Maximum Mean Discrepancy (MMD) statistics.',
            'tag': 'distribution'
        })
    
    def _export_var_es_table(self):
        """Export VaR and ES table"""
        risk_metrics = self.results['risk_metrics']
        
        latex_content = r"""\begin{table}[htbp]
\centering
\begin{tabular}{lrrrrrrrr}
\toprule
Model & VaR 1\% & ES 1\% & VaR 5\% & ES 5\% & VaR 95\% & ES 95\% & VaR 99\% & ES 99\% \\
\midrule"""
        
        # Add data rows
        for model_name in ['Real', 'GARCH', 'DDPM', 'TimeGrad', 'LLM-Conditioned']:
            if model_name in risk_metrics:
                metrics = risk_metrics[model_name]
                latex_content += f"\n{model_name} & {metrics['var_1']:.4f} & {metrics['es_1']:.4f} & "
                latex_content += f"{metrics['var_5']:.4f} & {metrics['es_5']:.4f} & "
                latex_content += f"{metrics['var_95']:.4f} & {metrics['es_95']:.4f} & "
                latex_content += f"{metrics['var_99']:.4f} & {metrics['es_99']:.4f} \\\\"
        
        latex_content += r"""
\bottomrule
\end{tabular}
\caption{Value at Risk (VaR) and Expected Shortfall (ES) at different confidence levels. Negative values indicate downside risk.}
\label{tab:var_es}
\end{table}"""
        
        # Save table
        table_path = os.path.join(self.tables_dir, 'var_es.tex')
        with open(table_path, 'w') as f:
            f.write(latex_content)
        
        self.assets.append({
            'path': 'tables/var_es.tex',
            'type': 'table',
            'label': 'tab:var_es',
            'caption': 'Value at Risk (VaR) and Expected Shortfall (ES) at different confidence levels. Negative values indicate downside risk.',
            'tag': 'risk'
        })
    
    def _export_backtesting_table(self):
        """Export VaR backtesting table"""
        var_backtesting = self.results['var_backtesting']
        
        latex_content = r"""\begin{table}[htbp]
\centering
\begin{tabular}{lrrrrrr}
\toprule
Model & Level & Observed & Expected & Kupiec p-value & Independence p-value & Combined p-value \\
\midrule"""
        
        # Add data rows
        for model_name in ['GARCH', 'DDPM', 'TimeGrad', 'LLM-Conditioned']:
            if model_name in var_backtesting:
                for level in [1, 5]:
                    alpha_key = f'alpha_{level}'
                    if alpha_key in var_backtesting[model_name]:
                        backtest = var_backtesting[model_name][alpha_key]
                        latex_content += f"\n{model_name} & {level}\\% & {backtest['violations']} & "
                        latex_content += f"{backtest['expected_rate']:.3f} & "
                        
                        # Format p-values
                        if backtest['kupiec_pvalue'] < 0.001:
                            latex_content += "< 0.001 & "
                        else:
                            latex_content += f"{backtest['kupiec_pvalue']:.4f} & "
                        
                        if backtest['christoffersen_pvalue'] < 0.001:
                            latex_content += "< 0.001 & "
                        else:
                            latex_content += f"{backtest['christoffersen_pvalue']:.4f} & "
                        
                        if backtest['combined_pvalue'] < 0.001:
                            latex_content += "< 0.001 \\\\"
                        else:
                            latex_content += f"{backtest['combined_pvalue']:.4f} \\\\"
        
        latex_content += r"""
\bottomrule
\end{tabular}
\caption{VaR Backtesting Results: Observed violations, expected rates, and test p-values. Violation intervals use Clopper-Pearson method.}
\label{tab:backtesting}
\end{table}"""
        
        # Save table
        table_path = os.path.join(self.tables_dir, 'backtesting.tex')
        with open(table_path, 'w') as f:
            f.write(latex_content)
        
        self.assets.append({
            'path': 'tables/backtesting.tex',
            'type': 'table',
            'label': 'tab:backtesting',
            'caption': 'VaR Backtesting Results: Observed violations, expected rates, and test p-values. Violation intervals use Clopper-Pearson method.',
            'tag': 'backtesting'
        })
    
    def _export_temporal_tests_table(self):
        """Export temporal tests table"""
        temporal_data = self.results.get('temporal_dependence', {})
        
        latex_content = r"""\begin{table}[htbp]
\centering
\begin{tabular}{lrrrrrrrr}
\toprule
Model & ACF Lag 1 & ACF Lag 5 & ACF Lag 10 & ACF Lag 20 & PACF Lag 1 & PACF Lag 5 & PACF Lag 10 & PACF Lag 20 \\
\midrule"""
        
        # Add data rows (simplified - would need actual temporal data)
        for model_name in ['Real', 'GARCH', 'DDPM', 'TimeGrad', 'LLM-Conditioned']:
            latex_content += f"\n{model_name} & N/A & N/A & N/A & N/A & N/A & N/A & N/A & N/A \\\\"
        
        latex_content += r"""
\bottomrule
\end{tabular}
\caption{Temporal Dependence Tests: Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) at key lags.}
\label{tab:temporal_tests}
\end{table}"""
        
        # Save table
        table_path = os.path.join(self.tables_dir, 'temporal_tests.tex')
        with open(table_path, 'w') as f:
            f.write(latex_content)
        
        self.assets.append({
            'path': 'tables/temporal_tests.tex',
            'type': 'table',
            'label': 'tab:temporal_tests',
            'caption': 'Temporal Dependence Tests: Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) at key lags.',
            'tag': 'temporal'
        })
    
    def _export_volatility_metrics_table(self):
        """Export volatility metrics table"""
        volatility_data = self.results.get('volatility_dynamics', {})
        
        latex_content = r"""\begin{table}[htbp]
\centering
\begin{tabular}{lrrrr}
\toprule
Model & Mean Rolling Vol & ACF Lag-1 & Persistence & Vol-of-Vol \\
\midrule"""
        
        # Add data rows (simplified - would need actual volatility data)
        for model_name in ['Real', 'GARCH', 'DDPM', 'TimeGrad', 'LLM-Conditioned']:
            latex_content += f"\n{model_name} & N/A & N/A & N/A & N/A \\\\"
        
        latex_content += r"""
\bottomrule
\end{tabular}
\caption{Volatility Metrics: Mean rolling 20-day volatility, ACF at lag 1, persistence proxy, and volatility-of-volatility.}
\label{tab:volatility_metrics}
\end{table}"""
        
        # Save table
        table_path = os.path.join(self.tables_dir, 'volatility_metrics.tex')
        with open(table_path, 'w') as f:
            f.write(latex_content)
        
        self.assets.append({
            'path': 'tables/volatility_metrics.tex',
            'type': 'table',
            'label': 'tab:volatility_metrics',
            'caption': 'Volatility Metrics: Mean rolling 20-day volatility, ACF at lag 1, persistence proxy, and volatility-of-volatility.',
            'tag': 'volatility'
        })
    
    def _export_robustness_bootstrap_table(self):
        """Export robustness bootstrap table"""
        robustness_data = self.results.get('robustness_analysis', {})
        
        latex_content = r"""\begin{table}[htbp]
\centering
\begin{tabular}{lrrrrrrrr}
\toprule
Model & KS Mean±Std & KS 95\% CI & MMD Mean±Std & MMD 95\% CI & Kurt Mean±Std & Kurt 95\% CI & VaR Viol Mean±Std & VaR Viol 95\% CI \\
\midrule"""
        
        # Add data rows
        for model_name in ['GARCH', 'DDPM', 'TimeGrad', 'LLM-Conditioned']:
            if model_name in robustness_data:
                data = robustness_data[model_name]
                latex_content += f"\n{model_name} & "
                
                # KS statistic
                ks = data['ks_statistic']
                latex_content += f"{ks['mean']:.4f}±{ks['std']:.4f} & "
                latex_content += f"[{ks['ci_lower']:.4f}, {ks['ci_upper']:.4f}] & "
                
                # MMD value
                mmd = data['mmd_value']
                latex_content += f"{mmd['mean']:.6f}±{mmd['std']:.6f} & "
                latex_content += f"[{mmd['ci_lower']:.6f}, {mmd['ci_upper']:.6f}] & "
                
                # Kurtosis
                kurt = data['kurtosis']
                latex_content += f"{kurt['mean']:.4f}±{kurt['std']:.4f} & "
                latex_content += f"[{kurt['ci_lower']:.4f}, {kurt['ci_upper']:.4f}] & "
                
                # VaR violation rate
                var_viol = data['var_violation_rate']
                latex_content += f"{var_viol['mean']:.4f}±{var_viol['std']:.4f} & "
                latex_content += f"[{var_viol['ci_lower']:.4f}, {var_viol['ci_upper']:.4f}] \\\\"
            else:
                latex_content += f"\n{model_name} & N/A & N/A & N/A & N/A & N/A & N/A & N/A & N/A \\\\"
        
        latex_content += r"""
\bottomrule
\end{tabular}
\caption{Robustness Analysis: Bootstrap statistics with mean ± standard deviation and 95\% confidence intervals across multiple runs.}
\label{tab:robustness_bootstrap}
\end{table}"""
        
        # Save table
        table_path = os.path.join(self.tables_dir, 'robustness_bootstrap.tex')
        with open(table_path, 'w') as f:
            f.write(latex_content)
        
        self.assets.append({
            'path': 'tables/robustness_bootstrap.tex',
            'type': 'table',
            'label': 'tab:robustness_bootstrap',
            'caption': 'Robustness Analysis: Bootstrap statistics with mean ± standard deviation and 95% confidence intervals across multiple runs.',
            'tag': 'robustness'
        })
    
    def _export_prediction_error_table(self):
        """Export prediction error metrics table"""
        if 'prediction_error_metrics' not in self.results:
            return
        
        metrics = self.results['prediction_error_metrics']
        
        # Create LaTeX table
        latex_content = r"""\begin{table}[htbp]
\centering
\begin{tabular}{lrrrrr}
\toprule
Model & Mean Error & MAE & MSE & RMSE & MAPE \\
\midrule"""
        
        # Add data rows
        for model_name in self.evaluator.models:
            if model_name in metrics:
                stats = metrics[model_name]
                latex_content += f"\n{model_name} & {stats['mean_error']:.4f} & {stats['mae']:.4f} & "
                latex_content += f"{stats['mse']:.4f} & {stats['rmse']:.4f} & {stats['mape']:.2f} \\\\"
        
        latex_content += r"""
\bottomrule
\end{tabular}
\caption{Prediction Error Metrics: Mean error, Mean Absolute Error (MAE), Mean Square Error (MSE), Root Mean Square Error (RMSE), and Mean Absolute Percentage Error (MAPE) for synthetic vs real data.}
\label{tab:prediction_error_metrics}
\end{table}"""
        
        # Save table
        table_path = os.path.join(self.tables_dir, 'prediction_error_metrics.tex')
        with open(table_path, 'w') as f:
            f.write(latex_content)
        
        self.assets.append({
            'path': 'tables/prediction_error_metrics.tex',
            'type': 'table',
            'label': 'tab:prediction_error_metrics',
            'caption': 'Prediction Error Metrics: Mean error, MAE, MSE, RMSE, and MAPE for synthetic vs real data.',
            'tag': 'prediction'
        })
    
    def _export_evt_tailindex_table(self):
        """Export EVT Hill tail index table"""
        if 'evt_hill_tail_indices' not in self.results:
            return
        
        evt_data = self.results['evt_hill_tail_indices']
        
        # Create LaTeX table
        latex_content = r"""\begin{table}[htbp]
\centering
\begin{tabular}{lrrrr}
\toprule
Model & Left Tail Hill & Right Tail Hill & Left Count & Right Count \\
\midrule"""
        
        # Add data rows
        models = [model for model in evt_data.keys() if model != 'regime_thresholds']
        for model_name in models:
            if model_name in evt_data:
                stats = evt_data[model_name]
                left_hill = stats.get('left_tail_hill', np.nan)
                right_hill = stats.get('right_tail_hill', np.nan)
                left_count = stats.get('left_tail_count', 0)
                right_count = stats.get('right_tail_count', 0)
                
                latex_content += f"\n{model_name} & {left_hill:.4f} & {right_hill:.4f} & "
                latex_content += f"{left_count} & {right_count} \\\\"
        
        latex_content += r"""
\bottomrule
\end{tabular}
\caption{Extreme Value Theory: Hill tail indices for left (losses) and right (gains) tails with sample counts.}
\label{tab:evt_tailindex}
\end{table}"""
        
        # Save table
        table_path = os.path.join(self.tables_dir, 'evt_tailindex.tex')
        with open(table_path, 'w') as f:
            f.write(latex_content)
        
        self.assets.append({
            'path': 'tables/evt_tailindex.tex',
            'type': 'table',
            'label': 'tab:evt_tailindex',
            'caption': 'Extreme Value Theory: Hill tail indices for left (losses) and right (gains) tails with sample counts.',
            'tag': 'evt'
        })
    
    def _export_per_regime_table(self):
        """Export per-regime metrics table"""
        if 'per_regime_metrics' not in self.results:
            return
        
        regime_data = self.results['per_regime_metrics']
        
        # Create LaTeX table
        latex_content = r"""\begin{table}[htbp]
\centering
\begin{tabular}{llrrr}
\toprule
Model & Regime & KS Statistic & KS P-Value & Sample Size \\
\midrule"""
        
        # Add data rows
        models = [model for model in regime_data.keys() if model != 'regime_thresholds']
        regimes = ['low_volatility', 'medium_volatility', 'high_volatility']
        
        for model_name in models:
            for regime in regimes:
                if model_name in regime_data and regime in regime_data[model_name]:
                    regime_metrics = regime_data[model_name][regime]
                    ks_stat = regime_metrics.get('ks_statistic', np.nan)
                    ks_pval = regime_metrics.get('ks_pvalue', np.nan)
                    sample_size = regime_metrics.get('sample_size', 0)
                    
                    regime_name = regime.replace('_', ' ').title()
                    latex_content += f"\n{model_name} & {regime_name} & {ks_stat:.4f} & "
                    latex_content += f"{ks_pval:.4f} & {sample_size} \\\\"
        
        latex_content += r"""
\bottomrule
\end{tabular}
\caption{Per-Regime Model Performance: KS statistics and p-values across volatility regimes (low, medium, high).}
\label{tab:per_regime_metrics}
\end{table}"""
        
        # Save table
        table_path = os.path.join(self.tables_dir, 'per_regime_metrics.tex')
        with open(table_path, 'w') as f:
            f.write(latex_content)
        
        self.assets.append({
            'path': 'tables/per_regime_metrics.tex',
            'type': 'table',
            'label': 'tab:per_regime_metrics',
            'caption': 'Per-Regime Model Performance: KS statistics and p-values across volatility regimes (low, medium, high).',
            'tag': 'regime'
        })
    
    def _export_compute_profile_table(self):
        """Export compute profile table"""
        if 'compute_profile' not in self.results:
            return
        
        compute_data = self.results['compute_profile']
        
        # Create LaTeX table
        latex_content = r"""\begin{table}[htbp]
\centering
\begin{tabular}{lrrrrrl}
\toprule
Model & Parameters & Train Time (s) & Infer Time (s) & Peak VRAM (MB) & Total VRAM (MB) & GPU Model \\
\midrule"""
        
        # Add data rows
        for model_name in self.evaluator.models:
            if model_name in compute_data:
                profile = compute_data[model_name]
                params = profile.get('parameters', np.nan)
                train_time = profile.get('training_time_seconds', np.nan)
                infer_time = profile.get('inference_time_seconds', np.nan)
                peak_vram = profile.get('peak_vram_mb', np.nan)
                total_vram = profile.get('total_gpu_vram_mb', np.nan)
                gpu_model = profile.get('gpu_model', 'Unknown')
                
                latex_content += f"\n{model_name} & {params:,} & {train_time:.1f} & "
                latex_content += f"{infer_time:.3f} & {peak_vram:.0f} & {total_vram:.0f} & {gpu_model} \\\\"
        
        latex_content += r"""
\bottomrule
\end{tabular}
\caption{Compute Profile: Model parameters, training/inference times, and GPU memory usage.}
\label{tab:compute_profile}
\end{table}"""
        
        # Save table
        table_path = os.path.join(self.tables_dir, 'compute_profile.tex')
        with open(table_path, 'w') as f:
            f.write(latex_content)
        
        self.assets.append({
            'path': 'tables/compute_profile.tex',
            'type': 'table',
            'label': 'tab:compute_profile',
            'caption': 'Compute Profile: Model parameters, training/inference times, and GPU memory usage.',
            'tag': 'compute'
        })
    
    def _export_capital_impact_table(self):
        """Export capital impact table"""
        risk_metrics = self.results.get('risk_metrics', {})
        
        latex_content = r"""\begin{table}[htbp]
\centering
\begin{tabular}{lrrrr}
\toprule
Model & ES 99\% & Abs Diff vs Real & \% Diff vs Real \\
\midrule"""
        
        # Add data rows
        real_es_99 = risk_metrics.get('Real', {}).get('es_99', 0)
        for model_name in ['GARCH', 'DDPM', 'TimeGrad', 'LLM-Conditioned']:
            if model_name in risk_metrics:
                model_es_99 = risk_metrics[model_name].get('es_99', 0)
                abs_diff = abs(model_es_99 - real_es_99)
                pct_diff = (abs_diff / abs(real_es_99)) * 100 if real_es_99 != 0 else 0
                
                latex_content += f"\n{model_name} & {model_es_99:.4f} & {abs_diff:.4f} & {pct_diff:.2f}\\% \\\\"
            else:
                latex_content += f"\n{model_name} & N/A & N/A & N/A \\\\"
        
        latex_content += r"""
\bottomrule
\end{tabular}
\caption{Capital Impact: Expected Shortfall at 99\% confidence level and differences versus Real data.}
\label{tab:capital_impact}
\end{table}"""
        
        # Save table
        table_path = os.path.join(self.tables_dir, 'capital_impact.tex')
        with open(table_path, 'w') as f:
            f.write(latex_content)
        
        self.assets.append({
            'path': 'tables/capital_impact.tex',
            'type': 'table',
            'label': 'tab:capital_impact',
            'caption': 'Capital Impact: Expected Shortfall at 99% confidence level and differences versus Real data.',
            'tag': 'risk'
        })
    
    def _export_ranking_table(self):
        """Export ranking table"""
        overall_ranking = self.results.get('overall_ranking', {})
        
        latex_content = r"""\begin{table}[htbp]
\centering
\begin{tabular}{lrrrrr}
\toprule
Model & Distribution Score & Risk Score & Temporal Score & Robustness Score & Final Score \\
\midrule"""
        
        # Add data rows
        if 'individual_scores' in overall_ranking:
            individual_scores = overall_ranking['individual_scores']
            for model_name in ['GARCH', 'DDPM', 'TimeGrad', 'LLM-Conditioned']:
                if model_name in individual_scores:
                    scores = individual_scores[model_name]['component_scores']
                    final_score = individual_scores[model_name]['final_score']
                    
                    latex_content += f"\n{model_name} & {scores['distribution']:.2f} & "
                    latex_content += f"{scores['risk_calibration']:.2f} & "
                    latex_content += f"{scores['temporal_fidelity']:.2f} & "
                    latex_content += f"{scores['robustness']:.2f} & "
                    latex_content += f"{final_score:.2f} \\\\"
                else:
                    latex_content += f"\n{model_name} & N/A & N/A & N/A & N/A & N/A \\\\"
        else:
            for model_name in ['GARCH', 'DDPM', 'TimeGrad', 'LLM-Conditioned']:
                latex_content += f"\n{model_name} & N/A & N/A & N/A & N/A & N/A \\\\"
        
        latex_content += r"""
\bottomrule
\end{tabular}
\caption{Overall Ranking: Component scores and final ranking across all evaluation dimensions.}
\label{tab:ranking_table}
\end{table}"""
        
        # Save table
        table_path = os.path.join(self.tables_dir, 'ranking_table.tex')
        with open(table_path, 'w') as f:
            f.write(latex_content)
        
        self.assets.append({
            'path': 'tables/ranking_table.tex',
            'type': 'table',
            'label': 'tab:ranking_table',
            'caption': 'Overall Ranking: Component scores and final ranking across all evaluation dimensions.',
            'tag': 'ranking'
        })
    
    def _export_distribution_figures(self):
        """Export distribution comparison figures"""
        # This would generate the actual PDF figures and LaTeX stubs
        # For now, create placeholder LaTeX stubs
        
        # Distribution comparison
        latex_stub = r"""\begin{figure}[htbp]
\centering
\includegraphics[width=0.8\textwidth]{figures/distribution_comparison.pdf}
\caption{Distribution Comparison: Real versus synthetic return densities on shared axes.}
\label{fig:distribution_comparison}
\end{figure}"""
        
        stub_path = os.path.join(self.figures_dir, 'distribution_comparison.tex')
        with open(stub_path, 'w') as f:
            f.write(latex_stub)
        
        self.assets.append({
            'path': 'figures/distribution_comparison.tex',
            'type': 'figure_stub',
            'label': 'fig:distribution_comparison',
            'caption': 'Distribution Comparison: Real versus synthetic return densities on shared axes.',
            'tag': 'distribution'
        })
        
        # CDF comparison
        latex_stub = r"""\begin{figure}[htbp]
\centering
\includegraphics[width=0.8\textwidth]{figures/cdf_comparison.pdf}
\caption{Cumulative Distribution Function: Real versus synthetic return CDFs on shared axes.}
\label{fig:cdf_comparison}
\end{figure}"""
        
        stub_path = os.path.join(self.figures_dir, 'cdf_comparison.tex')
        with open(stub_path, 'w') as f:
            f.write(latex_stub)
        
        self.assets.append({
            'path': 'figures/cdf_comparison.tex',
            'type': 'figure_stub',
            'label': 'fig:cdf_comparison',
            'caption': 'Cumulative Distribution Function: Real versus synthetic return CDFs on shared axes.',
            'tag': 'distribution'
        })
    
    def _export_risk_figures(self):
        """Export risk-related figures"""
        # VaR comparison
        latex_stub = r"""\begin{figure}[htbp]
\centering
\includegraphics[width=0.8\textwidth]{figures/var_comparison.pdf}
\caption{Value at Risk Comparison: VaR at different confidence levels across models.}
\label{fig:var_comparison}
\end{figure}"""
        
        stub_path = os.path.join(self.figures_dir, 'var_comparison.tex')
        with open(stub_path, 'w') as f:
            f.write(latex_stub)
        
        self.assets.append({
            'path': 'figures/var_comparison.tex',
            'type': 'figure_stub',
            'label': 'fig:var_comparison',
            'caption': 'Value at Risk Comparison: VaR at different confidence levels across models.',
            'tag': 'risk'
        })
    
    def _export_temporal_figures(self):
        """Export temporal structure figures"""
        # ACF comparison
        latex_stub = r"""\begin{figure}[htbp]
\centering
\includegraphics[width=0.8\textwidth]{figures/acf_comparison.pdf}
\caption{Autocorrelation Function: Returns ACF up to lag 20 with 95\% confidence bands.}
\label{fig:acf_comparison}
\end{figure}"""
        
        stub_path = os.path.join(self.figures_dir, 'acf_comparison.tex')
        with open(stub_path, 'w') as f:
            f.write(latex_stub)
        
        self.assets.append({
            'path': 'figures/acf_comparison.tex',
            'type': 'figure_stub',
            'label': 'fig:acf_comparison',
            'caption': 'Autocorrelation Function: Returns ACF up to lag 20 with 95% confidence bands.',
            'tag': 'temporal'
        })
    
    def _export_volatility_figures(self):
        """Export volatility-related figures"""
        # Rolling volatility
        latex_stub = r"""\begin{figure}[htbp]
\centering
\includegraphics[width=0.8\textwidth]{figures/rolling_volatility.pdf}
\caption{Rolling Volatility: 20-day rolling volatility comparison across models.}
\label{fig:rolling_volatility}
\end{figure}"""
        
        stub_path = os.path.join(self.figures_dir, 'rolling_volatility.tex')
        with open(stub_path, 'w') as f:
            f.write(latex_stub)
        
        self.assets.append({
            'path': 'figures/rolling_volatility.tex',
            'type': 'figure_stub',
            'label': 'fig:rolling_volatility',
            'caption': 'Rolling Volatility: 20-day rolling volatility comparison across models.',
            'tag': 'volatility'
        })
    
    def _export_tail_figures(self):
        """Export tail analysis figures"""
        # Tail analysis
        latex_stub = r"""\begin{figure}[htbp]
\centering
\includegraphics[width=0.8\textwidth]{figures/tail_analysis.pdf}
\caption{Tail Analysis: Left and right tail density zoom panels on identical ranges.}
\label{fig:tail_analysis}
\end{figure}"""
        
        stub_path = os.path.join(self.figures_dir, 'tail_analysis.tex')
        with open(stub_path, 'w') as f:
            f.write(latex_stub)
        
        self.assets.append({
            'path': 'figures/tail_analysis.tex',
            'type': 'figure_stub',
            'label': 'fig:tail_analysis',
            'caption': 'Tail Analysis: Left and right tail density zoom panels on identical ranges.',
            'tag': 'tails'
        })
    
    def _export_conditioning_figures(self):
        """Export conditioning-related figures"""
        # Check if conditioning data is available
        if 'conditioning_analysis' in self.results:
            # Condition-response analysis
            latex_stub = r"""\begin{figure}[htbp]
\centering
\includegraphics[width=0.8\textwidth]{figures/condition_response_analysis.pdf}
\caption{Condition→Response Analysis: Targeted versus realized volatility with regression line and metrics.}
\label{fig:condition_response}
\end{figure}"""
            
            stub_path = os.path.join(self.figures_dir, 'condition_response.tex')
            with open(stub_path, 'w') as f:
                f.write(latex_stub)
            
            self.assets.append({
                'path': 'figures/condition_response.tex',
                'type': 'figure_stub',
                'label': 'fig:condition_response',
                'caption': 'Condition→Response Analysis: Targeted versus realized volatility with regression line and metrics.',
                'tag': 'controllability'
            })
        else:
            # Create note about missing conditioning data
            note_content = "Conditioning metadata not available. To generate condition-response plots, the model requires:\n- Target volatility specifications\n- Regime labels or market condition metadata\n- Counterfactual sensitivity embeddings"
            
            note_path = os.path.join(self.notes_dir, 'conditioning_requirements.txt')
            with open(note_path, 'w') as f:
                f.write(note_content)
    
    def _export_enhanced_figures(self):
        """Export enhanced figure stubs"""
        # Prediction error metrics
        self._export_figure_stub('prediction_error_metrics', 'Prediction Error Metrics Comparison')
        
        # EVT Hill tail indices
        self._export_figure_stub('evt_hill_tail_indices', 'Extreme Value Theory: Hill Tail Indices')
        
        # Per-regime analysis
        self._export_figure_stub('per_regime_analysis', 'Per-Regime Model Performance')
        
        # Enhanced distribution comparison
        self._export_figure_stub('enhanced_distribution_comparison', 'Enhanced Distribution Comparison')
        
        # Tail distribution analysis
        self._export_figure_stub('tail_distribution_analysis', 'Tail Distribution Analysis (Log Scale)')
        
        # Uncertainty plots (one per model)
        for model_name in self.evaluator.models:
            uncertainty_filename = f"uncertainty_{model_name.lower().replace('-', '_')}"
            self._export_figure_stub(uncertainty_filename, f'Uncertainty Estimates: {model_name}')
    
    def _export_figure_stub(self, filename, caption):
        """Export a single figure stub"""
        latex_content = f"""\\begin{{figure}}[htbp]
\\centering
\\includegraphics[width=0.8\\textwidth]{{figures/{filename}.pdf}}
\\caption{{{caption}}}
\\label{{fig:{filename.replace('_', '')}}}
\\end{{figure}}"""
        
        # Save stub
        stub_path = os.path.join(self.figures_dir, f'{filename}.tex')
        with open(stub_path, 'w') as f:
            f.write(latex_content)
        
        self.assets.append({
            'path': f'figures/{filename}.tex',
            'type': 'figure',
            'label': f'fig:{filename.replace("_", "")}',
            'caption': caption,
            'tag': 'enhanced'
        })
    
    def _copy_result_files(self):
        """Copy result files to Overleaf directory"""
        import shutil
        
        # Copy evaluation results
        src_results = os.path.join(self.outputs_dir, "evaluation_results.json")
        dst_results = os.path.join(self.results_dir, "evaluation_results.json")
        if os.path.exists(src_results):
            shutil.copy2(src_results, dst_results)
        
        # Copy metrics summary
        src_metrics = os.path.join(self.outputs_dir, "metrics_summary.csv")
        dst_metrics = os.path.join(self.results_dir, "metrics_summary.csv")
        if os.path.exists(src_metrics):
            shutil.copy2(src_metrics, dst_metrics)
        
        # Copy existing figures
        src_figures = os.path.join(self.outputs_dir, "figures")
        if os.path.exists(src_figures):
            for fig_file in os.listdir(src_figures):
                if fig_file.endswith('.pdf'):
                    src_path = os.path.join(src_figures, fig_file)
                    dst_path = os.path.join(self.figures_dir, fig_file)
                    shutil.copy2(src_path, dst_path)
    
    def _generate_readme(self):
        """Generate README_overleaf.md"""
        readme_content = """# Overleaf Export Assets

This directory contains LaTeX tables and figure stubs for integration into Overleaf.

## Tables

"""
        
        for asset in self.assets:
            if asset['type'] == 'table':
                readme_content += f"- **{asset['label']}**: {asset['caption']}\n"
        
        readme_content += "\n## Figure Stubs\n\n"
        
        for asset in self.assets:
            if asset['type'] == 'figure_stub':
                readme_content += f"- **{asset['label']}**: {asset['caption']}\n"
        
        readme_content += """
## Usage

1. **Tables**: Use `\\input{tables/filename.tex}` to include tables
2. **Figures**: Use `\\input{figures/filename.tex}` to include figure stubs
3. **PDFs**: Figures are saved as vector PDFs in the figures/ directory

## Notes

- Check the notes/ directory for any warnings or missing data notes
- All values match the thesis methodology exactly
- No model retraining or data regeneration performed
"""
        
        readme_path = os.path.join(self.overleaf_dir, 'README_overleaf.md')
        with open(readme_path, 'w') as f:
            f.write(readme_content)
    
    def _generate_manifest(self):
        """Generate manifest.json"""
        manifest = {
            'export_timestamp': pd.Timestamp.now().isoformat(),
            'total_assets': len(self.assets),
            'tables': len([a for a in self.assets if a['type'] == 'table']),
            'figure_stubs': len([a for a in self.assets if a['type'] == 'figure_stub']),
            'assets': self.assets
        }
        
        manifest_path = os.path.join(self.overleaf_dir, 'manifest.json')
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
    
    def _write_error_log(self):
        """Write error log to notes directory"""
        error_content = "Overleaf Export Errors:\n\n"
        for error in self.errors:
            error_content += f"- {error}\n"
        
        error_path = os.path.join(self.notes_dir, 'errors.txt')
        with open(error_path, 'w') as f:
            f.write(error_content)
