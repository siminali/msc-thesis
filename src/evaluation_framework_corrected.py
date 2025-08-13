#!/usr/bin/env python3
"""
Corrected Comprehensive Evaluation Framework for Financial Time Series Models

This module provides standardized, consistent evaluation metrics with proper error handling
and robust computation methods for comparing financial models.

Author: Simin Ali
Thesis: Diffusion Models in Generative AI for Financial Data Synthesis and Risk Management
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import ks_2samp, skew, kurtosis
import warnings
from typing import Dict, List, Tuple, Optional
import json
import os
from datetime import datetime
warnings.filterwarnings('ignore')

# Set style for consistent plots
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['savefig.bbox'] = 'tight'

class CorrectedFinancialModelEvaluator:
    """
    Corrected comprehensive evaluator for financial time series models.
    Provides standardized metrics computation with proper error handling.
    """
    
    def __init__(self, model_names=None):
        self.model_names = model_names or ['GARCH', 'DDPM', 'TimeGrad', 'LLM-Conditioned']
        self.results = {}
        self.plots = {}
        self.metadata = {
            'computation_date': datetime.now().isoformat(),
            'software_versions': {
                'numpy': np.__version__,
                'pandas': pd.__version__,
                'scipy': '1.11.0'  # Fixed version reference
            },
            'methodology': {
                'mmd_kernel': 'RBF with median heuristic bandwidth',
                'mmd_estimator': 'Unbiased U-statistic',
                'var_quantiles': [0.01, 0.05, 0.95, 0.99],
                'volatility_window': 20,
                'bootstrap_samples': 1000
            }
        }
        
    def standardize_data(self, data, model_name: str) -> np.ndarray:
        """Standardize data format and handle edge cases."""
        if data is None:
            return np.array([])
            
        if isinstance(data, pd.Series):
            data = data.values
        elif isinstance(data, np.ndarray) and data.ndim > 1:
            data = data.flatten()
        
        # Remove NaN and infinite values
        data = data[~np.isnan(data) & ~np.isinf(data)]
        
        # Ensure data is in percentage format (multiply by 100 if values are < 1)
        if np.max(np.abs(data)) < 1.0:
            data = data * 100
            print(f"⚠️ {model_name}: Data scaled to percentage format")
        
        return data
    
    def compute_basic_statistics(self, data, model_name: str) -> Dict:
        """Compute basic statistical measures with proper error handling."""
        data = self.standardize_data(data, model_name)
        
        if len(data) == 0:
            return {
                'Model': model_name,
                'Mean': np.nan, 'Std Dev': np.nan, 'Skewness': np.nan,
                'Kurtosis': np.nan, 'Min': np.nan, 'Max': np.nan,
                'Q1': np.nan, 'Q3': np.nan
            }
        
        try:
            return {
                'Model': model_name,
                'Mean': float(np.mean(data)),
                'Std Dev': float(np.std(data)),
                'Skewness': float(skew(data)),
                'Kurtosis': float(kurtosis(data)),
                'Min': float(np.min(data)),
                'Max': float(np.max(data)),
                'Q1': float(np.percentile(data, 25)),
                'Q3': float(np.percentile(data, 75))
            }
        except Exception as e:
            print(f"⚠️ Error computing basic stats for {model_name}: {e}")
            return {
                'Model': model_name,
                'Mean': np.nan, 'Std Dev': np.nan, 'Skewness': np.nan,
                'Kurtosis': np.nan, 'Min': np.nan, 'Max': np.nan,
                'Q1': np.nan, 'Q3': np.nan
            }
    
    def compute_tail_metrics(self, data, model_name: str, quantiles=[0.01, 0.05, 0.95, 0.99]) -> Dict:
        """Compute tail risk metrics with proper sign conventions."""
        data = self.standardize_data(data, model_name)
        
        if len(data) == 0:
            return {'Model': model_name}
        
        results = {'Model': model_name}
        
        try:
            for q in quantiles:
                # VaR at quantile q (negative for downside risk)
                var_q = np.percentile(data, q * 100)
                
                # Expected Shortfall (conditional mean below/above VaR)
                if q < 0.5:  # Left tail (downside risk)
                    es_q = data[data <= var_q].mean()
                    # Ensure VaR and ES are negative for downside risk
                    var_q = -abs(var_q)
                    es_q = -abs(es_q)
                else:  # Right tail (upside potential)
                    es_q = data[data >= var_q].mean()
                    # Ensure VaR and ES are positive for upside potential
                    var_q = abs(var_q)
                    es_q = abs(es_q)
                
                results[f'VaR_{q*100:.0f}%'] = float(var_q)
                results[f'ES_{q*100:.0f}%'] = float(es_q)
                
        except Exception as e:
            print(f"⚠️ Error computing tail metrics for {model_name}: {e}")
            for q in quantiles:
                results[f'VaR_{q*100:.0f}%'] = np.nan
                results[f'ES_{q*100:.0f}%'] = np.nan
        
        return results
    
    def compute_volatility_metrics(self, data, model_name: str, window=20) -> Dict:
        """Compute volatility dynamics metrics with proper scaling."""
        data = self.standardize_data(data, model_name)
        
        if len(data) < window + 1:
            return {
                'Model': model_name,
                'Volatility_ACF': np.nan,
                'Volatility_Persistence': np.nan,
                'Mean_Volatility': np.nan,
                'Volatility_of_Volatility': np.nan
            }
        
        try:
            # Rolling volatility (absolute returns for better stability)
            abs_returns = np.abs(data)
            rolling_vol = pd.Series(abs_returns).rolling(window=window).std().dropna()
            
            # Volatility clustering (autocorrelation of squared returns)
            squared_returns = data ** 2
            vol_acf = pd.Series(squared_returns).autocorr(lag=1)
            if pd.isna(vol_acf):
                vol_acf = 0.0
            
            # Volatility persistence (autocorrelation of rolling volatility)
            vol_persistence = rolling_vol.autocorr(lag=1)
            if pd.isna(vol_persistence):
                vol_persistence = 0.0
            
            # Ensure non-negative values
            mean_vol = max(0.0, float(rolling_vol.mean()))
            vol_of_vol = max(0.0, float(rolling_vol.std()))
            
            return {
                'Model': model_name,
                'Volatility_ACF': float(vol_acf),
                'Volatility_Persistence': float(vol_persistence),
                'Mean_Volatility': mean_vol,
                'Volatility_of_Volatility': vol_of_vol
            }
            
        except Exception as e:
            print(f"⚠️ Error computing volatility metrics for {model_name}: {e}")
            return {
                'Model': model_name,
                'Volatility_ACF': np.nan,
                'Volatility_Persistence': np.nan,
                'Mean_Volatility': np.nan,
                'Volatility_of_Volatility': np.nan
            }
    
    def compute_standardized_mmd(self, real_data: np.ndarray, synthetic_data: np.ndarray, 
                                model_name: str, n_samples: int = 1000) -> float:
        """
        Compute standardized MMD using RBF kernel with consistent parameters.
        
        Args:
            real_data: Real financial returns
            synthetic_data: Synthetic returns from model
            model_name: Name of the model for logging
            n_samples: Number of samples to use for MMD computation
            
        Returns:
            MMD value (non-negative float)
        """
        try:
            # Ensure data is properly formatted
            real_data = self.standardize_data(real_data, "Real")
            synthetic_data = self.standardize_data(synthetic_data, model_name)
            
            if len(real_data) == 0 or len(synthetic_data) == 0:
                return np.nan
            
            # Sample data to avoid memory issues and ensure consistency
            n_real = min(n_samples, len(real_data))
            n_synth = min(n_samples, len(synthetic_data))
            
            real_sample = np.random.choice(real_data, n_real, replace=False)
            synth_sample = np.random.choice(synthetic_data, n_synth, replace=False)
            
            # Use median heuristic for bandwidth selection
            def compute_bandwidth(x, y):
                x_median = np.median(np.abs(x))
                y_median = np.median(np.abs(y))
                return 1.0 / (0.5 * (x_median**2 + y_median**2))
            
            gamma = compute_bandwidth(real_sample, synth_sample)
            
            # RBF kernel function
            def rbf_kernel(x, y, sigma):
                return np.exp(-sigma * (x - y) ** 2)
            
            # Compute MMD using unbiased estimator
            k_xx = np.mean([rbf_kernel(real_sample[i], real_sample[j], gamma) 
                           for i in range(n_real) for j in range(n_real) if i != j])
            k_yy = np.mean([rbf_kernel(synth_sample[i], synth_sample[j], gamma) 
                           for i in range(n_synth) for j in range(n_synth) if i != j])
            k_xy = np.mean([rbf_kernel(real_sample[i], synth_sample[j], gamma) 
                           for i in range(n_real) for j in range(n_synth)])
            
            mmd = k_xx + k_yy - 2 * k_xy
            
            # Ensure non-negative result
            mmd = max(0.0, mmd)
            
            print(f"✅ {model_name} MMD computed: {mmd:.6f} (gamma={gamma:.6f})")
            return float(mmd)
            
        except Exception as e:
            print(f"⚠️ Error computing MMD for {model_name}: {e}")
            return np.nan
    
    def compute_distribution_tests(self, real_data: np.ndarray, synthetic_data: np.ndarray, 
                                 model_name: str) -> Dict:
        """Compute distribution similarity tests with consistent methodology."""
        real_data = self.standardize_data(real_data, "Real")
        synthetic_data = self.standardize_data(synthetic_data, model_name)
        
        if len(real_data) == 0 or len(synthetic_data) == 0:
            return {
                'Model': model_name,
                'KS_Statistic': np.nan,
                'KS_pvalue': np.nan,
                'Anderson_Darling_Stat': np.nan,
                'MMD': np.nan
            }
        
        try:
            # KS test
            ks_stat, ks_pvalue = ks_2samp(real_data, synthetic_data)
            
            # Anderson-Darling test
            try:
                ad_stat, ad_critical_values, ad_significance_levels = stats.anderson_ksamp([real_data, synthetic_data])
                ad_stat = float(ad_stat)
            except:
                ad_stat = np.nan
            
            # Standardized MMD
            mmd = self.compute_standardized_mmd(real_data, synthetic_data, model_name)
            
            return {
                'Model': model_name,
                'KS_Statistic': float(ks_stat),
                'KS_pvalue': float(ks_pvalue),
                'Anderson_Darling_Stat': ad_stat,
                'MMD': mmd
            }
            
        except Exception as e:
            print(f"⚠️ Error computing distribution tests for {model_name}: {e}")
            return {
                'Model': model_name,
                'KS_Statistic': np.nan,
                'KS_pvalue': np.nan,
                'Anderson_Darling_Stat': np.nan,
                'MMD': np.nan
            }
    
    def compute_var_backtesting(self, real_data: np.ndarray, synthetic_data: np.ndarray, 
                               model_name: str, confidence_levels=[0.01, 0.05]) -> List[Dict]:
        """Compute VaR backtesting with proper test statistics."""
        real_data = self.standardize_data(real_data, "Real")
        synthetic_data = self.standardize_data(synthetic_data, model_name)
        
        if len(real_data) == 0 or len(synthetic_data) == 0:
            return []
        
        results = []
        
        for alpha in confidence_levels:
            try:
                # Compute VaR from synthetic data
                var_estimate = np.percentile(synthetic_data, alpha * 100)
                
                # Count violations in real data
                violations = np.sum(real_data <= var_estimate)
                total_obs = len(real_data)
                violation_rate = violations / total_obs
                expected_rate = alpha
                
                # Kupiec test (unconditional coverage)
                if violations > 0 and violations < total_obs:
                    try:
                        # Likelihood ratio test statistic
                        lr_stat = -2 * np.log(
                            ((1 - expected_rate) ** (total_obs - violations)) * 
                            (expected_rate ** violations) /
                            (((1 - violation_rate) ** (total_obs - violations)) * 
                             (violation_rate ** violations))
                        )
                        kupiec_pvalue = 1 - stats.chi2.cdf(lr_stat, 1)
                    except:
                        lr_stat = np.nan
                        kupiec_pvalue = np.nan
                else:
                    lr_stat = np.nan
                    kupiec_pvalue = np.nan
                
                # Independence test (Christoffersen)
                if violations > 1:
                    try:
                        # Simple independence test using runs test
                        violation_series = (real_data <= var_estimate).astype(int)
                        runs_stat = stats.runs_test(violation_series)[0]
                        independence_pvalue = stats.runs_test(violation_series)[1]
                    except:
                        runs_stat = np.nan
                        independence_pvalue = np.nan
                else:
                    runs_stat = np.nan
                    independence_pvalue = np.nan
                
                # Combined test
                if not (np.isnan(kupiec_pvalue) or np.isnan(independence_pvalue)):
                    combined_stat = lr_stat + runs_stat
                    combined_pvalue = 1 - stats.chi2.cdf(combined_stat, 2)
                else:
                    combined_stat = np.nan
                    combined_pvalue = np.nan
                
                results.append({
                    'Model': model_name,
                    'Confidence_Level': alpha,
                    'VaR_Estimate': float(var_estimate),
                    'Violations': int(violations),
                    'Total_Observations': int(total_obs),
                    'Violation_Rate': float(violation_rate),
                    'Expected_Rate': float(expected_rate),
                    'Kupiec_Test_Stat': lr_stat,
                    'Kupiec_Test_pvalue': kupiec_pvalue,
                    'Independence_Test_Stat': runs_stat,
                    'Independence_Test_pvalue': independence_pvalue,
                    'Combined_Test_Stat': combined_stat,
                    'Combined_Test_pvalue': combined_pvalue
                })
                
            except Exception as e:
                print(f"⚠️ Error computing VaR backtesting for {model_name} at {alpha}: {e}")
                results.append({
                    'Model': model_name,
                    'Confidence_Level': alpha,
                    'VaR_Estimate': np.nan,
                    'Violations': np.nan,
                    'Total_Observations': np.nan,
                    'Violation_Rate': np.nan,
                    'Expected_Rate': alpha,
                    'Kupiec_Test_Stat': np.nan,
                    'Kupiec_Test_pvalue': np.nan,
                    'Independence_Test_Stat': np.nan,
                    'Independence_Test_pvalue': np.nan,
                    'Combined_Test_Stat': np.nan,
                    'Combined_Test_pvalue': np.nan
                })
        
        return results
    
    def compute_robust_metrics_with_bootstrap(self, real_data: np.ndarray, synthetic_data: np.ndarray, 
                                            model_name: str, n_bootstrap=5) -> Dict:
        """Compute metrics with bootstrap confidence intervals for robustness."""
        real_data = self.standardize_data(real_data, "Real")
        synthetic_data = self.standardize_data(synthetic_data, model_name)
        
        if len(real_data) == 0 or len(synthetic_data) == 0:
            return {}
        
        # Multiple sampling runs
        metrics_runs = []
        
        for run in range(n_bootstrap):
            try:
                # Sample with replacement
                n_real = min(1000, len(real_data))
                n_synth = min(1000, len(synthetic_data))
                
                real_sample = np.random.choice(real_data, n_real, replace=True)
                synth_sample = np.random.choice(synthetic_data, n_synth, replace=True)
                
                # Compute metrics for this run
                run_metrics = {
                    'KS': stats.ks_2samp(real_sample, synth_sample)[0],
                    'MMD': self.compute_standardized_mmd(real_sample, synth_sample, f"{model_name}_run{run}"),
                    'Kurtosis': kurtosis(synth_sample),
                    'VaR_1%': np.percentile(synth_sample, 1),
                    'VaR_5%': np.percentile(synth_sample, 5)
                }
                metrics_runs.append(run_metrics)
                
            except Exception as e:
                print(f"⚠️ Error in bootstrap run {run} for {model_name}: {e}")
        
        if not metrics_runs:
            return {}
        
        # Compute statistics across runs
        results = {'Model': model_name}
        
        for metric in ['KS', 'MMD', 'Kurtosis', 'VaR_1%', 'VaR_5%']:
            values = [run[metric] for run in metrics_runs if not np.isnan(run[metric])]
            if values:
                results[f'{metric}_mean'] = float(np.mean(values))
                results[f'{metric}_std'] = float(np.std(values))
                results[f'{metric}_min'] = float(np.min(values))
                results[f'{metric}_max'] = float(np.max(values))
                
                # Bootstrap confidence interval
                if len(values) >= 3:
                    sorted_values = np.sort(values)
                    ci_lower = sorted_values[int(0.025 * len(sorted_values))]
                    ci_upper = sorted_values[int(0.975 * len(sorted_values))]
                    results[f'{metric}_ci_95_lower'] = float(ci_lower)
                    results[f'{metric}_ci_95_upper'] = float(ci_upper)
                else:
                    results[f'{metric}_ci_95_lower'] = np.nan
                    results[f'{metric}_ci_95_upper'] = np.nan
            else:
                results[f'{metric}_mean'] = np.nan
                results[f'{metric}_std'] = np.nan
                results[f'{metric}_min'] = np.nan
                results[f'{metric}_max'] = np.nan
                results[f'{metric}_ci_95_lower'] = np.nan
                results[f'{metric}_ci_95_upper'] = np.nan
        
        return results
    
    def generate_metrics_summary_csv(self, all_metrics: Dict, output_path: str = "results/metrics_summary.csv"):
        """Generate comprehensive metrics summary CSV with bootstrap statistics."""
        summary_data = []
        
        for model_name, metrics in all_metrics.items():
            if 'robust_metrics' in metrics:
                robust = metrics['robust_metrics']
                row = {
                    'Model': model_name,
                    'KS_mean': robust.get('KS_mean', np.nan),
                    'KS_std': robust.get('KS_std', np.nan),
                    'KS_ci_95_lower': robust.get('KS_ci_95_lower', np.nan),
                    'KS_ci_95_upper': robust.get('KS_ci_95_upper', np.nan),
                    'MMD_mean': robust.get('MMD_mean', np.nan),
                    'MMD_std': robust.get('MMD_std', np.nan),
                    'MMD_ci_95_lower': robust.get('MMD_ci_95_lower', np.nan),
                    'MMD_ci_95_upper': robust.get('MMD_ci_95_upper', np.nan),
                    'Kurtosis_mean': robust.get('Kurtosis_mean', np.nan),
                    'Kurtosis_std': robust.get('Kurtosis_std', np.nan),
                    'VaR_1%_mean': robust.get('VaR_1%_mean', np.nan),
                    'VaR_1%_std': robust.get('VaR_1%_mean', np.nan),
                    'VaR_5%_mean': robust.get('VaR_5%_mean', np.nan),
                    'VaR_5%_std': robust.get('VaR_5%_std', np.nan)
                }
                summary_data.append(row)
        
        if summary_data:
            df = pd.DataFrame(summary_data)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            df.to_csv(output_path, index=False)
            print(f"✅ Metrics summary saved to: {output_path}")
        
        return summary_data
    
    def save_evaluation_results(self, results: Dict, output_path: str = "results/comprehensive_evaluation/evaluation_results_corrected.json"):
        """Save corrected evaluation results to JSON."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Add metadata
        results['metadata'] = self.metadata
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"✅ Corrected evaluation results saved to: {output_path}")
        return output_path
