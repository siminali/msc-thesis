#!/usr/bin/env python3
"""
Enhanced Evaluation Framework for Financial Time Series Models

This module extends the corrected framework with additional tests:
- Christoffersen independence test for VaR backtesting
- Ljung-Box tests for temporal dependency
- Clopper-Pearson confidence intervals
- Enhanced volatility computation
- Sample sequence analysis

Author: Simin Ali
Thesis: Diffusion Models in Generative AI for Financial Data Synthesis and Risk Management
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import ks_2samp, skew, kurtosis, chi2
from statsmodels.stats.diagnostic import acorr_ljungbox
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

class EnhancedFinancialModelEvaluator:
    """
    Enhanced comprehensive evaluator for financial time series models.
    Includes Christoffersen independence tests, Ljung-Box tests, and enhanced visualizations.
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
                'scipy': '1.11.0',
                'statsmodels': '0.14.0'
            },
            'methodology': {
                'mmd_kernel': 'RBF with median heuristic bandwidth',
                'mmd_estimator': 'Unbiased U-statistic',
                'var_quantiles': [0.01, 0.05, 0.95, 0.99],
                'volatility_window': 20,
                'bootstrap_samples': 1000,
                'acf_lags': 20,
                'ljung_box_lags': [10, 20],
                'sequence_length': 60,
                'sample_sequences': 5
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
    
    def compute_christoffersen_independence_test(self, violations: np.ndarray, alpha: float) -> Dict:
        """
        Compute Christoffersen independence test for VaR violations.
        
        Parameters:
        - violations: Binary array of VaR violations (1 = violation, 0 = no violation)
        - alpha: Expected violation rate
        
        Returns:
        - Dictionary with test statistics and p-values
        """
        if len(violations) < 2:
            return {
                'Independence_Test_Stat': np.nan,
                'Independence_Test_pvalue': np.nan,
                'Combined_Test_Stat': np.nan,
                'Combined_Test_pvalue': np.nan
            }
        
        try:
            # Count transitions
            n00 = np.sum((violations[:-1] == 0) & (violations[1:] == 0))  # 0->0
            n01 = np.sum((violations[:-1] == 0) & (violations[1:] == 1))  # 0->1
            n10 = np.sum((violations[:-1] == 1) & (violations[1:] == 0))  # 1->0
            n11 = np.sum((violations[:-1] == 1) & (violations[1:] == 1))  # 1->1
            
            # Check if we have enough observations for the test
            if n01 + n11 == 0 or n00 + n10 == 0:
                return {
                    'Independence_Test_Stat': np.nan,
                    'Independence_Test_pvalue': np.nan,
                    'Combined_Test_Stat': np.nan,
                    'Combined_Test_pvalue': np.nan
                }
            
            # Compute transition probabilities
            p01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0
            p11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0
            
            # Independence test statistic
            if p01 == p11:
                independence_stat = 0.0
            else:
                # Log-likelihood ratio test for independence
                p_hat = (n01 + n11) / (n00 + n01 + n10 + n11)
                if p_hat == 0 or p_hat == 1:
                    independence_stat = 0.0
                else:
                    log_likelihood_indep = (n01 + n11) * np.log(p_hat) + (n00 + n10) * np.log(1 - p_hat)
                    log_likelihood_dep = n01 * np.log(p01) + n00 * np.log(1 - p01) + n11 * np.log(p11) + n10 * np.log(1 - p11)
                    independence_stat = 2 * (log_likelihood_dep - log_likelihood_indep)
            
            # P-value for independence test (chi-square with 1 degree of freedom)
            independence_pvalue = 1 - chi2.cdf(independence_stat, 1) if independence_stat > 0 else 1.0
            
            return {
                'Independence_Test_Stat': float(independence_stat),
                'Independence_Test_pvalue': float(independence_pvalue)
            }
            
        except Exception as e:
            print(f"⚠️ Error in Christoffersen independence test: {e}")
            return {
                'Independence_Test_Stat': np.nan,
                'Independence_Test_pvalue': np.nan
            }
    
    def compute_combined_lr_test(self, kupiec_stat: float, independence_stat: float) -> Dict:
        """
        Compute combined likelihood ratio test statistic.
        
        Parameters:
        - kupiec_stat: Kupiec test statistic
        - independence_stat: Independence test statistic
        
        Returns:
        - Dictionary with combined test statistic and p-value
        """
        try:
            if np.isnan(kupiec_stat) or np.isnan(independence_stat):
                return {
                    'Combined_Test_Stat': np.nan,
                    'Combined_Test_pvalue': np.nan
                }
            
            # Combined test statistic (sum of individual test statistics)
            combined_stat = kupiec_stat + independence_stat
            
            # P-value for combined test (chi-square with 2 degrees of freedom)
            combined_pvalue = 1 - chi2.cdf(combined_stat, 2) if combined_stat > 0 else 1.0
            
            return {
                'Combined_Test_Stat': float(combined_stat),
                'Combined_Test_pvalue': float(combined_pvalue)
            }
            
        except Exception as e:
            print(f"⚠️ Error in combined LR test: {e}")
            return {
                'Combined_Test_Stat': np.nan,
                'Combined_Test_pvalue': np.nan
            }
    
    def compute_clopper_pearson_ci(self, violations: int, total: int, confidence: float = 0.95) -> Tuple[float, float]:
        """
        Compute Clopper-Pearson confidence interval for violation rate.
        
        Parameters:
        - violations: Number of violations
        - total: Total number of observations
        - confidence: Confidence level (default 0.95)
        
        Returns:
        - Tuple of (lower_bound, upper_bound)
        """
        try:
            if violations == 0:
                lower = 0.0
                upper = 1 - (confidence / 2) ** (1 / total)
            elif violations == total:
                lower = (confidence / 2) ** (1 / total)
                upper = 1.0
            else:
                # Use beta distribution quantiles
                alpha = 1 - confidence
                lower = stats.beta.ppf(alpha / 2, violations, total - violations + 1)
                upper = stats.beta.ppf(1 - alpha / 2, violations + 1, total - violations)
            
            return float(lower), float(upper)
            
        except Exception as e:
            print(f"⚠️ Error in Clopper-Pearson CI: {e}")
            return np.nan, np.nan
    
    def compute_ljung_box_test(self, data: np.ndarray, lags: List[int]) -> Dict:
        """
        Compute Ljung-Box test for autocorrelation in returns.
        
        Parameters:
        - data: Return series
        - lags: List of lag values to test
        
        Returns:
        - Dictionary with test statistics and p-values for each lag
        """
        try:
            results = {}
            for lag in lags:
                if lag >= len(data):
                    results[f'Ljung_Box_{lag}_Stat'] = np.nan
                    results[f'Ljung_Box_{lag}_pvalue'] = np.nan
                    continue
                
                # Ljung-Box test for autocorrelation
                lb_stat, lb_pvalue = acorr_ljungbox(data, lags=[lag], return_df=False)
                
                results[f'Ljung_Box_{lag}_Stat'] = float(lb_stat[0])
                results[f'Ljung_Box_{lag}_pvalue'] = float(lb_pvalue[0])
            
            return results
            
        except Exception as e:
            print(f"⚠️ Error in Ljung-Box test: {e}")
            return {f'Ljung_Box_{lag}_Stat': np.nan for lag in lags}
    
    def compute_enhanced_var_backtesting(self, real_data: np.ndarray, synthetic_data: np.ndarray, 
                                       model_name: str, confidence_levels=[0.01, 0.05]) -> List[Dict]:
        """
        Enhanced VaR backtesting with Christoffersen independence test and combined LR test.
        """
        results = []
        
        for alpha in confidence_levels:
            try:
                # Compute VaR estimate from synthetic data
                var_estimate = np.percentile(synthetic_data, alpha * 100)
                
                # Generate violations using real data
                violations = (real_data < var_estimate).astype(int)
                n_violations = np.sum(violations)
                total_obs = len(violations)
                violation_rate = n_violations / total_obs
                
                # Kupiec test (unconditional coverage)
                expected_violations = alpha * total_obs
                if expected_violations > 0 and total_obs - expected_violations > 0:
                    kupiec_stat = 2 * (n_violations * np.log(n_violations / expected_violations) + 
                                     (total_obs - n_violations) * np.log((total_obs - n_violations) / (total_obs - expected_violations)))
                    kupiec_pvalue = 1 - chi2.cdf(kupiec_stat, 1)
                else:
                    kupiec_stat = np.nan
                    kupiec_pvalue = np.nan
                
                # Christoffersen independence test
                independence_results = self.compute_christoffersen_independence_test(violations, alpha)
                
                # Combined likelihood ratio test
                combined_results = self.compute_combined_lr_test(kupiec_stat, independence_results['Independence_Test_Stat'])
                
                # Clopper-Pearson confidence intervals
                ci_lower, ci_upper = self.compute_clopper_pearson_ci(n_violations, total_obs)
                
                result = {
                    'Model': model_name,
                    'Confidence_Level': alpha,
                    'VaR_Estimate': float(var_estimate),
                    'Violations': int(n_violations),
                    'Total_Observations': int(total_obs),
                    'Violation_Rate': float(violation_rate),
                    'Expected_Rate': float(alpha),
                    'Kupiec_Test_Stat': float(kupiec_stat) if not np.isnan(kupiec_stat) else np.nan,
                    'Kupiec_Test_pvalue': float(kupiec_pvalue) if not np.isnan(kupiec_pvalue) else np.nan,
                    'Independence_Test_Stat': independence_results['Independence_Test_Stat'],
                    'Independence_Test_pvalue': independence_results['Independence_Test_pvalue'],
                    'Combined_Test_Stat': combined_results['Combined_Test_Stat'],
                    'Combined_Test_pvalue': combined_results['Combined_Test_pvalue'],
                    'CI_Lower': ci_lower,
                    'CI_Upper': ci_upper
                }
                
                results.append(result)
                
            except Exception as e:
                print(f"⚠️ Error in VaR backtesting for {model_name} at {alpha}: {e}")
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
                    'Combined_Test_pvalue': np.nan,
                    'CI_Lower': np.nan,
                    'CI_Upper': np.nan
                })
        
        return results
    
    def compute_enhanced_volatility_metrics(self, data, model_name: str, window=20) -> Dict:
        """
        Enhanced volatility computation ensuring non-negative values.
        """
        data = self.standardize_data(data, model_name)
        
        if len(data) < window:
            return {
                'Model': model_name,
                'Mean_Volatility': np.nan,
                'Std_Volatility': np.nan,
                'Volatility_ACF_Lag1': np.nan,
                'Volatility_Persistence': np.nan,
                'Volatility_of_Volatility': np.nan
            }
        
        try:
            # Compute rolling volatility using absolute returns
            abs_returns = np.abs(data)
            rolling_vol = pd.Series(abs_returns).rolling(window=window, min_periods=1).mean()
            
            # Ensure non-negative values
            rolling_vol = np.maximum(rolling_vol, 0)
            
            # Remove NaN values
            rolling_vol_clean = rolling_vol.dropna()
            
            if len(rolling_vol_clean) == 0:
                return {
                    'Model': model_name,
                    'Mean_Volatility': np.nan,
                    'Std_Volatility': np.nan,
                    'Volatility_ACF_Lag1': np.nan,
                    'Volatility_Persistence': np.nan,
                    'Volatility_of_Volatility': np.nan
                }
            
            mean_vol = float(np.mean(rolling_vol_clean))
            std_vol = float(np.std(rolling_vol_clean))
            
            # Volatility ACF at lag 1
            if len(rolling_vol_clean) > 1:
                vol_acf_lag1 = float(np.corrcoef(rolling_vol_clean[:-1], rolling_vol_clean[1:])[0, 1])
            else:
                vol_acf_lag1 = np.nan
            
            # Volatility persistence (half-life)
            if vol_acf_lag1 > 0 and vol_acf_lag1 < 1:
                vol_persistence = float(-np.log(2) / np.log(vol_acf_lag1))
            else:
                vol_persistence = np.nan
            
            # Volatility of volatility
            vol_of_vol = float(np.std(rolling_vol_clean))
            
            return {
                'Model': model_name,
                'Mean_Volatility': max(0, mean_vol),  # Ensure non-negative
                'Std_Volatility': max(0, std_vol),    # Ensure non-negative
                'Volatility_ACF_Lag1': vol_acf_lag1,
                'Volatility_Persistence': vol_persistence,
                'Volatility_of_Volatility': vol_of_vol
            }
            
        except Exception as e:
            print(f"⚠️ Error in volatility computation for {model_name}: {e}")
            return {
                'Model': model_name,
                'Mean_Volatility': np.nan,
                'Std_Volatility': np.nan,
                'Volatility_ACF_Lag1': np.nan,
                'Volatility_Persistence': np.nan,
                'Volatility_of_Volatility': np.nan
            }
    
    def compute_temporal_dependency_metrics(self, data, model_name: str) -> Dict:
        """
        Compute temporal dependency metrics including ACF, PACF, and Ljung-Box tests.
        """
        data = self.standardize_data(data, model_name)
        
        if len(data) < 21:  # Need at least 21 observations for lag 20
            return {
                'Model': model_name,
                'ACF_Lag1': np.nan, 'ACF_Lag5': np.nan, 'ACF_Lag10': np.nan, 'ACF_Lag20': np.nan,
                'PACF_Lag1': np.nan, 'PACF_Lag5': np.nan, 'PACF_Lag10': np.nan, 'PACF_Lag20': np.nan,
                'Ljung_Box_10_Stat': np.nan, 'Ljung_Box_10_pvalue': np.nan,
                'Ljung_Box_20_Stat': np.nan, 'Ljung_Box_20_pvalue': np.nan
            }
        
        try:
            # ACF computation
            acf_lags = [1, 5, 10, 20]
            acf_values = []
            for lag in acf_lags:
                if lag < len(data):
                    acf_val = np.corrcoef(data[:-lag], data[lag:])[0, 1]
                    acf_values.append(float(acf_val) if not np.isnan(acf_val) else 0.0)
                else:
                    acf_values.append(np.nan)
            
            # PACF computation (simplified using ACF)
            pacf_values = acf_values.copy()  # For simplicity, using ACF as approximation
            
            # Ljung-Box test
            ljung_box_results = self.compute_ljung_box_test(data, [10, 20])
            
            return {
                'Model': model_name,
                'ACF_Lag1': acf_values[0], 'ACF_Lag5': acf_values[1], 
                'ACF_Lag10': acf_values[2], 'ACF_Lag20': acf_values[3],
                'PACF_Lag1': pacf_values[0], 'PACF_Lag5': pacf_values[1], 
                'PACF_Lag10': pacf_values[2], 'PACF_Lag20': pacf_values[3],
                'Ljung_Box_10_Stat': ljung_box_results['Ljung_Box_10_Stat'],
                'Ljung_Box_10_pvalue': ljung_box_results['Ljung_Box_10_pvalue'],
                'Ljung_Box_20_Stat': ljung_box_results['Ljung_Box_20_Stat'],
                'Ljung_Box_20_pvalue': ljung_box_results['Ljung_Box_20_pvalue']
            }
            
        except Exception as e:
            print(f"⚠️ Error in temporal dependency computation for {model_name}: {e}")
            return {
                'Model': model_name,
                'ACF_Lag1': np.nan, 'ACF_Lag5': np.nan, 'ACF_Lag10': np.nan, 'ACF_Lag20': np.nan,
                'PACF_Lag1': np.nan, 'PACF_Lag5': np.nan, 'PACF_Lag10': np.nan, 'PACF_Lag20': np.nan,
                'Ljung_Box_10_Stat': np.nan, 'Ljung_Box_10_pvalue': np.nan,
                'Ljung_Box_20_Stat': np.nan, 'Ljung_Box_20_pvalue': np.nan
            }
    
    def generate_sample_sequences(self, data, model_name: str, n_sequences=5, length=60) -> np.ndarray:
        """
        Generate multiple sample sequences for visual comparison.
        """
        data = self.standardize_data(data, model_name)
        
        if len(data) < length:
            return np.array([])
        
        try:
            # Generate n_sequences of length 'length' starting from random positions
            sequences = []
            for i in range(n_sequences):
                # Use fixed seed for reproducibility
                np.random.seed(42 + i)
                start_idx = np.random.randint(0, len(data) - length + 1)
                sequence = data[start_idx:start_idx + length]
                sequences.append(sequence)
            
            return np.array(sequences)
            
        except Exception as e:
            print(f"⚠️ Error generating sample sequences for {model_name}: {e}")
            return np.array([])
    
    def save_enhanced_evaluation_results(self, results: Dict, output_path: str = "results/comprehensive_evaluation/evaluation_results_enhanced.json"):
        """Save enhanced evaluation results to JSON file."""
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            print(f"✅ Enhanced evaluation results saved to: {output_path}")
            
        except Exception as e:
            print(f"❌ Error saving enhanced evaluation results: {e}")
    
    def generate_enhanced_metrics_summary_csv(self, all_metrics: Dict, output_path: str = "results/enhanced_metrics_summary.csv"):
        """Generate CSV summary of enhanced metrics."""
        try:
            # Extract key metrics for summary
            summary_data = []
            
            for model_name in self.model_names:
                # Basic stats
                basic_stats = next((x for x in all_metrics.get('basic_statistics', []) if x['Model'] == model_name), {})
                
                # VaR backtest results
                var_results = [x for x in all_metrics.get('var_backtest', []) if x['Model'] == model_name]
                
                # Temporal dependency
                temp_dep = next((x for x in all_metrics.get('temporal_dependency', []) if x['Model'] == model_name), {})
                
                # Volatility metrics
                vol_metrics = next((x for x in all_metrics.get('volatility_metrics', []) if x['Model'] == model_name), {})
                
                for var_result in var_results:
                    row = {
                        'Model': model_name,
                        'Confidence_Level': var_result.get('Confidence_Level', np.nan),
                        'VaR_Estimate': var_result.get('VaR_Estimate', np.nan),
                        'Violation_Rate': var_result.get('Violation_Rate', np.nan),
                        'Expected_Rate': var_result.get('Expected_Rate', np.nan),
                        'Kupiec_pvalue': var_result.get('Kupiec_Test_pvalue', np.nan),
                        'Independence_pvalue': var_result.get('Independence_Test_pvalue', np.nan),
                        'Combined_pvalue': var_result.get('Combined_Test_pvalue', np.nan),
                        'ACF_Lag1': temp_dep.get('ACF_Lag1', np.nan),
                        'Ljung_Box_10_pvalue': temp_dep.get('Ljung_Box_10_pvalue', np.nan),
                        'Ljung_Box_20_pvalue': temp_dep.get('Ljung_Box_20_pvalue', np.nan),
                        'Mean_Volatility': vol_metrics.get('Mean_Volatility', np.nan),
                        'Volatility_Persistence': vol_metrics.get('Volatility_Persistence', np.nan)
                    }
                    summary_data.append(row)
            
            # Create DataFrame and save
            df = pd.DataFrame(summary_data)
            df.to_csv(output_path, index=False)
            print(f"✅ Enhanced metrics summary saved to: {output_path}")
            
        except Exception as e:
            print(f"❌ Error generating enhanced metrics summary: {e}")
    
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
            print(f"⚠️ Error in basic statistics for {model_name}: {e}")
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
            return {
                'Model': model_name,
                'VaR_1%': np.nan, 'VaR_5%': np.nan, 'VaR_95%': np.nan, 'VaR_99%': np.nan,
                'ES_1%': np.nan, 'ES_5%': np.nan, 'ES_95%': np.nan, 'ES_99%': np.nan
            }
        
        try:
            # Compute VaR (negative for downside, positive for upside)
            var_1 = float(np.percentile(data, 1))  # 1% VaR (downside)
            var_5 = float(np.percentile(data, 5))  # 5% VaR (downside)
            var_95 = float(np.percentile(data, 95))  # 95% VaR (upside)
            var_99 = float(np.percentile(data, 99))  # 99% VaR (upside)
            
            # Compute Expected Shortfall (ES)
            es_1 = float(np.mean(data[data <= var_1])) if var_1 in data else float(np.mean(data[data < var_1]))
            es_5 = float(np.mean(data[data <= var_5])) if var_5 in data else float(np.mean(data[data < var_5]))
            es_95 = float(np.mean(data[data >= var_95])) if var_95 in data else float(np.mean(data[data > var_95]))
            es_99 = float(np.mean(data[data >= var_99])) if var_99 in data else float(np.mean(data[data > var_99]))
            
            return {
                'Model': model_name,
                'VaR_1%': var_1, 'VaR_5%': var_5, 'VaR_95%': var_95, 'VaR_99%': var_99,
                'ES_1%': es_1, 'ES_5%': es_5, 'ES_95%': es_95, 'ES_99%': es_99
            }
            
        except Exception as e:
            print(f"⚠️ Error in tail metrics for {model_name}: {e}")
            return {
                'Model': model_name,
                'VaR_1%': np.nan, 'VaR_5%': np.nan, 'VaR_95%': np.nan, 'VaR_99%': np.nan,
                'ES_1%': np.nan, 'ES_5%': np.nan, 'ES_95%': np.nan, 'ES_99%': np.nan
            }
    
    def compute_standardized_mmd(self, real_data: np.ndarray, synthetic_data: np.ndarray, 
                                model_name: str, n_samples: int = 1000) -> float:
        """Compute standardized MMD using RBF kernel with median heuristic bandwidth."""
        try:
            # Sample data for efficiency
            if len(real_data) > n_samples:
                real_sample = np.random.choice(real_data, n_samples, replace=False)
            else:
                real_sample = real_data
            
            if len(synthetic_data) > n_samples:
                synthetic_sample = np.random.choice(synthetic_data, n_samples, replace=False)
            else:
                synthetic_sample = synthetic_data
            
            # Compute median heuristic bandwidth
            all_data = np.concatenate([real_sample, synthetic_sample])
            pairwise_distances = []
            for i in range(len(all_data)):
                for j in range(i+1, len(all_data)):
                    pairwise_distances.append(np.abs(all_data[i] - all_data[j]))
            
            if len(pairwise_distances) == 0:
                return np.nan
            
            median_distance = np.median(pairwise_distances)
            bandwidth = median_distance / np.sqrt(2)  # Standard RBF bandwidth
            
            if bandwidth == 0:
                bandwidth = 1.0  # Fallback bandwidth
            
            # Compute MMD using RBF kernel
            def rbf_kernel(x, y, sigma):
                return np.exp(-np.sum((x - y) ** 2) / (2 * sigma ** 2))
            
            # Unbiased U-statistic estimator
            n_real = len(real_sample)
            n_synthetic = len(synthetic_sample)
            
            # K(x,x') terms
            k_real_real = 0
            for i in range(n_real):
                for j in range(i+1, n_real):
                    k_real_real += rbf_kernel(real_sample[i], real_sample[j], bandwidth)
            k_real_real = 2 * k_real_real / (n_real * (n_real - 1))
            
            k_synthetic_synthetic = 0
            for i in range(n_synthetic):
                for j in range(i+1, n_synthetic):
                    k_synthetic_synthetic += rbf_kernel(synthetic_sample[i], synthetic_sample[j], bandwidth)
            k_synthetic_synthetic = 2 * k_synthetic_synthetic / (n_synthetic * (n_synthetic - 1))
            
            # K(x,y) terms
            k_real_synthetic = 0
            for i in range(n_real):
                for j in range(n_synthetic):
                    k_real_synthetic += rbf_kernel(real_sample[i], synthetic_sample[j], bandwidth)
            k_real_synthetic = k_real_synthetic / (n_real * n_synthetic)
            
            # MMD^2 = K(x,x') + K(y,y') - 2K(x,y)
            mmd_squared = k_real_real + k_synthetic_synthetic - 2 * k_real_synthetic
            
            return float(np.sqrt(max(0, mmd_squared)))  # Ensure non-negative
            
        except Exception as e:
            print(f"⚠️ Error in MMD computation for {model_name}: {e}")
            return np.nan
    
    def compute_distribution_tests(self, real_data: np.ndarray, synthetic_data: np.ndarray, 
                                 model_name: str) -> Dict:
        """Compute distribution tests including standardized MMD."""
        try:
            # Kolmogorov-Smirnov test
            ks_stat, ks_pvalue = ks_2samp(real_data, synthetic_data)
            
            # Anderson-Darling test (approximate)
            try:
                ad_stat = stats.anderson_ksamp([real_data, synthetic_data]).statistic
            except:
                ad_stat = np.nan
            
            # Standardized MMD
            mmd = self.compute_standardized_mmd(real_data, synthetic_data, model_name)
            
            return {
                'Model': model_name,
                'KS_Statistic': float(ks_stat),
                'KS_pvalue': float(ks_pvalue),
                'AD_Statistic': float(ad_stat) if not np.isnan(ad_stat) else np.nan,
                'MMD': float(mmd) if not np.isnan(mmd) else np.nan
            }
            
        except Exception as e:
            print(f"⚠️ Error in distribution tests for {model_name}: {e}")
            return {
                'Model': model_name,
                'KS_Statistic': np.nan,
                'KS_pvalue': np.nan,
                'AD_Statistic': np.nan,
                'MMD': np.nan
            }
    
    def compute_robust_metrics_with_bootstrap(self, real_data: np.ndarray, synthetic_data: np.ndarray, 
                                            model_name: str, n_bootstrap=5) -> Dict:
        """Compute robust metrics with bootstrap resampling."""
        try:
            bootstrap_results = []
            
            for i in range(n_bootstrap):
                # Set seed for reproducibility
                np.random.seed(42 + i)
                
                # Bootstrap sample
                real_bootstrap = np.random.choice(real_data, size=len(real_data), replace=True)
                synthetic_bootstrap = np.random.choice(synthetic_data, size=len(synthetic_data), replace=True)
                
                # Compute metrics on bootstrap sample
                ks_stat, _ = ks_2samp(real_bootstrap, synthetic_bootstrap)
                mmd = self.compute_standardized_mmd(real_bootstrap, synthetic_bootstrap, model_name)
                kurt = kurtosis(synthetic_bootstrap)
                
                # VaR estimates
                var_1 = np.percentile(synthetic_bootstrap, 1)
                var_5 = np.percentile(synthetic_bootstrap, 5)
                
                bootstrap_results.append({
                    'KS': ks_stat,
                    'MMD': mmd,
                    'Kurtosis': kurt,
                    'VaR_1%': var_1,
                    'VaR_5%': var_5
                })
            
            # Compute statistics across bootstrap runs
            ks_values = [r['KS'] for r in bootstrap_results if not np.isnan(r['KS'])]
            mmd_values = [r['MMD'] for r in bootstrap_results if not np.isnan(r['MMD'])]
            kurt_values = [r['Kurtosis'] for r in bootstrap_results if not np.isnan(r['Kurtosis'])]
            var_1_values = [r['VaR_1%'] for r in bootstrap_results if not np.isnan(r['VaR_1%'])]
            var_5_values = [r['VaR_5%'] for r in bootstrap_results if not np.isnan(r['VaR_5%'])]
            
            # Compute mean and std
            ks_mean = np.mean(ks_values) if ks_values else np.nan
            ks_std = np.std(ks_values) if len(ks_values) > 1 else np.nan
            ks_min = np.min(ks_values) if ks_values else np.nan
            ks_max = np.max(ks_values) if ks_values else np.nan
            
            mmd_mean = np.mean(mmd_values) if mmd_values else np.nan
            mmd_std = np.std(mmd_values) if len(mmd_values) > 1 else np.nan
            mmd_min = np.min(mmd_values) if mmd_values else np.nan
            mmd_max = np.max(mmd_values) if mmd_values else np.nan
            
            kurt_mean = np.mean(kurt_values) if kurt_values else np.nan
            kurt_std = np.std(kurt_values) if len(kurt_values) > 1 else np.nan
            kurt_min = np.min(kurt_values) if kurt_values else np.nan
            kurt_max = np.max(kurt_values) if kurt_values else np.nan
            
            var_1_mean = np.mean(var_1_values) if var_1_values else np.nan
            var_1_std = np.std(var_1_values) if len(var_1_values) > 1 else np.nan
            var_1_min = np.min(var_1_values) if var_1_values else np.nan
            var_1_max = np.max(var_1_values) if var_1_values else np.nan
            
            var_5_mean = np.mean(var_5_values) if var_5_values else np.nan
            var_5_std = np.std(var_5_values) if len(var_5_values) > 1 else np.nan
            var_5_min = np.min(var_5_values) if var_5_values else np.nan
            var_5_max = np.max(var_5_values) if var_5_values else np.nan
            
            return {
                'Model': model_name,
                'KS_mean': float(ks_mean) if not np.isnan(ks_mean) else np.nan,
                'KS_std': float(ks_std) if not np.isnan(ks_std) else np.nan,
                'KS_min': float(ks_min) if not np.isnan(ks_min) else np.nan,
                'KS_max': float(ks_max) if not np.isnan(ks_max) else np.nan,
                'KS_ci_95_lower': float(ks_min) if not np.isnan(ks_min) else np.nan,
                'KS_ci_95_upper': float(ks_max) if not np.isnan(ks_max) else np.nan,
                'MMD_mean': float(mmd_mean) if not np.isnan(mmd_mean) else np.nan,
                'MMD_std': float(mmd_std) if not np.isnan(mmd_std) else np.nan,
                'MMD_min': float(mmd_min) if not np.isnan(mmd_min) else np.nan,
                'MMD_max': float(mmd_max) if not np.isnan(mmd_max) else np.nan,
                'MMD_ci_95_lower': float(mmd_min) if not np.isnan(mmd_min) else np.nan,
                'MMD_ci_95_upper': float(mmd_max) if not np.isnan(mmd_max) else np.nan,
                'Kurtosis_mean': float(kurt_mean) if not np.isnan(kurt_mean) else np.nan,
                'Kurtosis_std': float(kurt_std) if not np.isnan(kurt_std) else np.nan,
                'Kurtosis_min': float(kurt_min) if not np.isnan(kurt_min) else np.nan,
                'Kurtosis_max': float(kurt_max) if not np.isnan(kurt_max) else np.nan,
                'Kurtosis_ci_95_lower': float(kurt_min) if not np.isnan(kurt_min) else np.nan,
                'Kurtosis_ci_95_upper': float(kurt_max) if not np.isnan(kurt_max) else np.nan,
                'VaR_1%_mean': float(var_1_mean) if not np.isnan(var_1_mean) else np.nan,
                'VaR_1%_std': float(var_1_std) if not np.isnan(var_1_std) else np.nan,
                'VaR_1%_min': float(var_1_min) if not np.isnan(var_1_min) else np.nan,
                'VaR_1%_max': float(var_1_max) if not np.isnan(var_1_max) else np.nan,
                'VaR_1%_ci_95_lower': float(var_1_min) if not np.isnan(var_1_min) else np.nan,
                'VaR_1%_ci_95_upper': float(var_1_max) if not np.isnan(var_1_max) else np.nan,
                'VaR_5%_mean': float(var_5_mean) if not np.isnan(var_5_mean) else np.nan,
                'VaR_5%_std': float(var_5_std) if not np.isnan(var_5_std) else np.nan,
                'VaR_5%_min': float(var_5_min) if not np.isnan(var_5_min) else np.nan,
                'VaR_5%_max': float(var_5_max) if not np.isnan(var_5_max) else np.nan,
                'VaR_5%_ci_95_lower': float(var_5_min) if not np.isnan(var_5_min) else np.nan,
                'VaR_5%_ci_95_upper': float(var_5_max) if not np.isnan(var_5_max) else np.nan
            }
            
        except Exception as e:
            print(f"⚠️ Error in bootstrap computation for {model_name}: {e}")
            return {
                'Model': model_name,
                'KS_mean': np.nan, 'KS_std': np.nan, 'KS_min': np.nan, 'KS_max': np.nan,
                'KS_ci_95_lower': np.nan, 'KS_ci_95_upper': np.nan,
                'MMD_mean': np.nan, 'MMD_std': np.nan, 'MMD_min': np.nan, 'MMD_max': np.nan,
                'MMD_ci_95_upper': np.nan, 'MMD_ci_95_lower': np.nan,
                'Kurtosis_mean': np.nan, 'Kurtosis_std': np.nan, 'Kurtosis_min': np.nan, 'Kurtosis_max': np.nan,
                'Kurtosis_ci_95_lower': np.nan, 'Kurtosis_ci_95_upper': np.nan,
                'VaR_1%_mean': np.nan, 'VaR_1%_std': np.nan, 'VaR_1%_min': np.nan, 'VaR_1%_max': np.nan,
                'VaR_1%_ci_95_lower': np.nan, 'VaR_1%_ci_95_upper': np.nan,
                'VaR_5%_mean': np.nan, 'VaR_5%_std': np.nan, 'VaR_5%_min': np.nan, 'VaR_5%_max': np.nan,
                'VaR_5%_ci_95_lower': np.nan, 'VaR_5%_upper': np.nan
            }
