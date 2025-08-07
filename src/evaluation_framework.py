"""
Comprehensive Evaluation Framework for Financial Time Series Models

This module provides automated evaluation metrics, plotting functions, and LaTeX table generation
for comparing GARCH, DDPM, and TimeGrad models as requested by the supervisor.

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
warnings.filterwarnings('ignore')

# Set style for consistent plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class FinancialModelEvaluator:
    """
    Comprehensive evaluator for financial time series models.
    Generates automated metrics, plots, and LaTeX tables for thesis reporting.
    """
    
    def __init__(self, model_names=None):
        self.model_names = model_names or ['GARCH', 'DDPM', 'TimeGrad']
        self.results = {}
        self.plots = {}
        
    def compute_basic_statistics(self, data, model_name):
        """Compute basic statistical measures for a dataset."""
        if isinstance(data, pd.Series):
            data = data.values
        elif isinstance(data, np.ndarray) and data.ndim > 1:
            data = data.flatten()
            
        return {
            'Mean': np.mean(data),
            'Std Dev': np.std(data),
            'Skewness': skew(data),
            'Kurtosis': kurtosis(data),
            'Min': np.min(data),
            'Max': np.max(data),
            'Q1': np.percentile(data, 25),
            'Q3': np.percentile(data, 75),
            'Model': model_name
        }
    
    def compute_tail_metrics(self, data, model_name, quantiles=[0.01, 0.05, 0.95, 0.99]):
        """Compute tail risk metrics including VaR and Expected Shortfall."""
        if isinstance(data, pd.Series):
            data = data.values
        elif isinstance(data, np.ndarray) and data.ndim > 1:
            data = data.flatten()
            
        results = {'Model': model_name}
        
        for q in quantiles:
            var_q = np.percentile(data, q * 100)
            es_q = data[data <= var_q].mean()
            results[f'VaR_{q*100:.0f}%'] = var_q
            results[f'ES_{q*100:.0f}%'] = es_q
            
        return results
    
    def compute_volatility_metrics(self, data, model_name, window=20):
        """Compute volatility clustering and persistence metrics."""
        if isinstance(data, pd.Series):
            data = data.values
        elif isinstance(data, np.ndarray) and data.ndim > 1:
            data = data.flatten()
            
        # Rolling volatility
        rolling_vol = pd.Series(data).rolling(window=window).std().dropna()
        
        # Volatility clustering (autocorrelation of squared returns)
        squared_returns = data ** 2
        vol_acf = pd.Series(squared_returns).autocorr(lag=1)
        
        # Volatility persistence (GARCH-like measure)
        vol_persistence = rolling_vol.autocorr(lag=1)
        
        return {
            'Model': model_name,
            'Volatility_ACF': vol_acf,
            'Volatility_Persistence': vol_persistence,
            'Mean_Volatility': rolling_vol.mean(),
            'Volatility_of_Volatility': rolling_vol.std()
        }
    
    def compute_distribution_tests(self, real_data, synthetic_data, model_name):
        """Compute distribution similarity tests."""
        if isinstance(real_data, pd.Series):
            real_data = real_data.values
        if isinstance(synthetic_data, pd.Series):
            synthetic_data = synthetic_data.values
        if isinstance(real_data, np.ndarray) and real_data.ndim > 1:
            real_data = real_data.flatten()
        if isinstance(synthetic_data, np.ndarray) and synthetic_data.ndim > 1:
            synthetic_data = synthetic_data.flatten()
            
        # Kolmogorov-Smirnov test
        ks_stat, ks_pvalue = ks_2samp(real_data, synthetic_data)
        
        # Anderson-Darling test
        try:
            ad_stat, ad_critical, ad_significance = stats.anderson_ksamp([real_data, synthetic_data])
        except:
            ad_stat, ad_significance = np.nan, np.nan
            
        # Maximum Mean Discrepancy (simplified version)
        mmd = self._compute_mmd(real_data, synthetic_data)
        
        return {
            'Model': model_name,
            'KS_Statistic': ks_stat,
            'KS_pvalue': ks_pvalue,
            'Anderson_Darling_Stat': ad_stat,
            'MMD': mmd
        }
    
    def _compute_mmd(self, x, y, kernel='rbf'):
        """Compute Maximum Mean Discrepancy between two samples."""
        # Simplified MMD computation
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        x_var = np.var(x)
        y_var = np.var(y)
        
        # MMD approximation using means and variances
        mmd = (x_mean - y_mean)**2 + (np.sqrt(x_var) - np.sqrt(y_var))**2
        return mmd
    
    def compute_var_backtest(self, returns, var_forecasts, model_name, confidence_level=0.01):
        """Compute VaR backtesting metrics including Kupiec and Christoffersen tests."""
        if isinstance(returns, pd.Series):
            returns = returns.values
        if isinstance(var_forecasts, pd.Series):
            var_forecasts = var_forecasts.values
            
        # Identify violations
        violations = returns < var_forecasts
        n_violations = np.sum(violations)
        n_observations = len(returns)
        violation_rate = n_violations / n_observations
        
        # Kupiec test
        kupiec_stat, kupiec_pvalue = self._kupiec_test(n_violations, n_observations, confidence_level)
        
        # Christoffersen test (independence)
        christoffersen_stat, christoffersen_pvalue = self._christoffersen_test(violations, confidence_level)
        
        return {
            'Model': model_name,
            'Total_Observations': n_observations,
            'Violations': n_violations,
            'Violation_Rate': violation_rate,
            'Expected_Rate': confidence_level,
            'Kupiec_Statistic': kupiec_stat,
            'Kupiec_pvalue': kupiec_pvalue,
            'Christoffersen_Statistic': christoffersen_stat,
            'Christoffersen_pvalue': christoffersen_pvalue
        }
    
    def _kupiec_test(self, n_violations, n_observations, confidence_level):
        """Perform Kupiec test for VaR backtesting."""
        if n_violations == 0:
            return np.nan, np.nan
            
        p_hat = n_violations / n_observations
        p_0 = confidence_level
        
        # Likelihood ratio test statistic
        if p_hat > 0 and p_hat < 1:
            lr_stat = 2 * (n_violations * np.log(p_hat / p_0) + 
                          (n_observations - n_violations) * np.log((1 - p_hat) / (1 - p_0)))
        else:
            lr_stat = np.nan
            
        # p-value (chi-square with 1 degree of freedom)
        p_value = 1 - stats.chi2.cdf(lr_stat, 1) if not np.isnan(lr_stat) else np.nan
        
        return lr_stat, p_value
    
    def _christoffersen_test(self, violations, confidence_level):
        """Perform Christoffersen test for independence of VaR violations."""
        if len(violations) < 2:
            return np.nan, np.nan
            
        # Create transition matrix
        n_00 = n_01 = n_10 = n_11 = 0
        
        for i in range(len(violations) - 1):
            if violations[i] == 0 and violations[i+1] == 0:
                n_00 += 1
            elif violations[i] == 0 and violations[i+1] == 1:
                n_01 += 1
            elif violations[i] == 1 and violations[i+1] == 0:
                n_10 += 1
            elif violations[i] == 1 and violations[i+1] == 1:
                n_11 += 1
        
        # Compute transition probabilities
        p_01 = n_01 / (n_00 + n_01) if (n_00 + n_01) > 0 else 0
        p_11 = n_11 / (n_10 + n_11) if (n_10 + n_11) > 0 else 0
        
        # Independence test statistic
        if p_01 > 0 and p_11 > 0:
            lr_ind = 2 * (n_01 * np.log(p_01) + n_11 * np.log(p_11) - 
                         (n_01 + n_11) * np.log(confidence_level))
        else:
            lr_ind = np.nan
            
        # p-value
        p_value = 1 - stats.chi2.cdf(lr_ind, 1) if not np.isnan(lr_ind) else np.nan
        
        return lr_ind, p_value
    
    def generate_comparison_plots(self, real_data, synthetic_data_dict, save_path="plots/"):
        """Generate comprehensive comparison plots."""
        import os
        os.makedirs(save_path, exist_ok=True)
        
        # 1. Distribution comparison
        self._plot_distribution_comparison(real_data, synthetic_data_dict, save_path)
        
        # 2. Time series comparison
        self._plot_time_series_comparison(real_data, synthetic_data_dict, save_path)
        
        # 3. Volatility clustering comparison
        self._plot_volatility_clustering(real_data, synthetic_data_dict, save_path)
        
        # 4. QQ plots
        self._plot_qq_comparison(real_data, synthetic_data_dict, save_path)
        
        # 5. Autocorrelation comparison
        self._plot_autocorrelation_comparison(real_data, synthetic_data_dict, save_path)
    
    def _plot_distribution_comparison(self, real_data, synthetic_data_dict, save_path):
        """Plot histogram comparison of distributions."""
        plt.figure(figsize=(12, 8))
        
        # Plot real data
        plt.hist(real_data.flatten(), bins=50, alpha=0.7, density=True, 
                label='Real Data', color='blue', edgecolor='black')
        
        # Plot synthetic data for each model
        colors = ['red', 'green', 'orange']
        for i, (model_name, synthetic_data) in enumerate(synthetic_data_dict.items()):
            plt.hist(synthetic_data.flatten(), bins=50, alpha=0.5, density=True,
                    label=f'{model_name} Synthetic', color=colors[i], edgecolor='black')
        
        plt.title('Distribution Comparison: Real vs Synthetic Data', fontsize=14, fontweight='bold')
        plt.xlabel('Returns', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{save_path}distribution_comparison.pdf", dpi=300, bbox_inches='tight')
        plt.savefig(f"{save_path}distribution_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_time_series_comparison(self, real_data, synthetic_data_dict, save_path):
        """Plot time series comparison."""
        fig, axes = plt.subplots(len(synthetic_data_dict) + 1, 1, figsize=(12, 3 * (len(synthetic_data_dict) + 1)))
        
        # Plot real data
        axes[0].plot(real_data[:100], label='Real Data', color='blue', linewidth=1)
        axes[0].set_title('Real Data Sample', fontweight='bold')
        axes[0].set_ylabel('Returns')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot synthetic data for each model
        colors = ['red', 'green', 'orange']
        for i, (model_name, synthetic_data) in enumerate(synthetic_data_dict.items()):
            axes[i+1].plot(synthetic_data[:100], label=f'{model_name} Synthetic', 
                          color=colors[i], linewidth=1)
            axes[i+1].set_title(f'{model_name} Synthetic Sample', fontweight='bold')
            axes[i+1].set_ylabel('Returns')
            axes[i+1].legend()
            axes[i+1].grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Time Steps')
        plt.tight_layout()
        plt.savefig(f"{save_path}time_series_comparison.pdf", dpi=300, bbox_inches='tight')
        plt.savefig(f"{save_path}time_series_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_volatility_clustering(self, real_data, synthetic_data_dict, save_path):
        """Plot volatility clustering comparison."""
        fig, axes = plt.subplots(len(synthetic_data_dict) + 1, 1, figsize=(12, 3 * (len(synthetic_data_dict) + 1)))
        
        # Compute rolling volatility for real data
        real_vol = pd.Series(real_data.flatten()).rolling(20).std()
        axes[0].plot(real_vol, label='Real Data', color='blue', linewidth=1)
        axes[0].set_title('Real Data Volatility Clustering', fontweight='bold')
        axes[0].set_ylabel('Rolling Std Dev')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot synthetic data volatility for each model
        colors = ['red', 'green', 'orange']
        for i, (model_name, synthetic_data) in enumerate(synthetic_data_dict.items()):
            synthetic_vol = pd.Series(synthetic_data.flatten()).rolling(20).std()
            axes[i+1].plot(synthetic_vol, label=f'{model_name} Synthetic', 
                          color=colors[i], linewidth=1)
            axes[i+1].set_title(f'{model_name} Synthetic Volatility Clustering', fontweight='bold')
            axes[i+1].set_ylabel('Rolling Std Dev')
            axes[i+1].legend()
            axes[i+1].grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Time Steps')
        plt.tight_layout()
        plt.savefig(f"{save_path}volatility_clustering.pdf", dpi=300, bbox_inches='tight')
        plt.savefig(f"{save_path}volatility_clustering.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_qq_comparison(self, real_data, synthetic_data_dict, save_path):
        """Plot QQ plots for comparison."""
        n_models = len(synthetic_data_dict)
        fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5))
        
        if n_models == 1:
            axes = [axes]
        
        colors = ['red', 'green', 'orange']
        for i, (model_name, synthetic_data) in enumerate(synthetic_data_dict.items()):
            stats.probplot(real_data.flatten(), dist="norm", plot=axes[i])
            axes[i].set_title(f'QQ Plot: Real vs {model_name}', fontweight='bold')
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{save_path}qq_comparison.pdf", dpi=300, bbox_inches='tight')
        plt.savefig(f"{save_path}qq_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_autocorrelation_comparison(self, real_data, synthetic_data_dict, save_path):
        """Plot autocorrelation comparison."""
        from statsmodels.graphics.tsaplots import plot_acf
        
        n_models = len(synthetic_data_dict)
        fig, axes = plt.subplots(1, n_models + 1, figsize=(5 * (n_models + 1), 5))
        
        # Plot real data ACF
        plot_acf(real_data.flatten(), ax=axes[0], lags=20)
        axes[0].set_title('Real Data ACF', fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Plot synthetic data ACF for each model
        colors = ['red', 'green', 'orange']
        for i, (model_name, synthetic_data) in enumerate(synthetic_data_dict.items()):
            plot_acf(synthetic_data.flatten(), ax=axes[i+1], lags=20)
            axes[i+1].set_title(f'{model_name} Synthetic ACF', fontweight='bold')
            axes[i+1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{save_path}autocorrelation_comparison.pdf", dpi=300, bbox_inches='tight')
        plt.savefig(f"{save_path}autocorrelation_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_latex_tables(self, results_dict, save_path="tables/"):
        """Generate LaTeX tables for thesis reporting."""
        import os
        os.makedirs(save_path, exist_ok=True)
        
        # 1. Basic statistics table
        self._generate_basic_stats_table(results_dict, save_path)
        
        # 2. Tail risk metrics table
        self._generate_tail_risk_table(results_dict, save_path)
        
        # 3. Volatility metrics table
        self._generate_volatility_table(results_dict, save_path)
        
        # 4. Distribution tests table
        self._generate_distribution_tests_table(results_dict, save_path)
        
        # 5. VaR backtesting table
        self._generate_var_backtest_table(results_dict, save_path)
        
        # 6. Summary comparison table
        self._generate_summary_table(results_dict, save_path)
    
    def _generate_basic_stats_table(self, results_dict, save_path):
        """Generate LaTeX table for basic statistics."""
        if 'basic_stats' not in results_dict:
            return
            
        df = pd.DataFrame(results_dict['basic_stats'])
        df = df.round(4)
        
        latex_code = df.to_latex(index=False, 
                                caption="Basic Statistical Measures Comparison",
                                label="tab:basic_stats",
                                float_format="%.4f")
        
        with open(f"{save_path}basic_statistics.tex", 'w') as f:
            f.write(latex_code)
        
        print("Basic statistics table saved to basic_statistics.tex")
    
    def _generate_tail_risk_table(self, results_dict, save_path):
        """Generate LaTeX table for tail risk metrics."""
        if 'tail_metrics' not in results_dict:
            return
            
        df = pd.DataFrame(results_dict['tail_metrics'])
        df = df.round(4)
        
        latex_code = df.to_latex(index=False,
                                caption="Tail Risk Metrics Comparison",
                                label="tab:tail_risk",
                                float_format="%.4f")
        
        with open(f"{save_path}tail_risk_metrics.tex", 'w') as f:
            f.write(latex_code)
        
        print("Tail risk metrics table saved to tail_risk_metrics.tex")
    
    def _generate_volatility_table(self, results_dict, save_path):
        """Generate LaTeX table for volatility metrics."""
        if 'volatility_metrics' not in results_dict:
            return
            
        df = pd.DataFrame(results_dict['volatility_metrics'])
        df = df.round(4)
        
        latex_code = df.to_latex(index=False,
                                caption="Volatility Metrics Comparison",
                                label="tab:volatility_metrics",
                                float_format="%.4f")
        
        with open(f"{save_path}volatility_metrics.tex", 'w') as f:
            f.write(latex_code)
        
        print("Volatility metrics table saved to volatility_metrics.tex")
    
    def _generate_distribution_tests_table(self, results_dict, save_path):
        """Generate LaTeX table for distribution tests."""
        if 'distribution_tests' not in results_dict:
            return
            
        df = pd.DataFrame(results_dict['distribution_tests'])
        df = df.round(4)
        
        latex_code = df.to_latex(index=False,
                                caption="Distribution Similarity Tests",
                                label="tab:distribution_tests",
                                float_format="%.4f")
        
        with open(f"{save_path}distribution_tests.tex", 'w') as f:
            f.write(latex_code)
        
        print("Distribution tests table saved to distribution_tests.tex")
    
    def _generate_var_backtest_table(self, results_dict, save_path):
        """Generate LaTeX table for VaR backtesting results."""
        if 'var_backtest' not in results_dict:
            return
            
        df = pd.DataFrame(results_dict['var_backtest'])
        df = df.round(4)
        
        latex_code = df.to_latex(index=False,
                                caption="VaR Backtesting Results",
                                label="tab:var_backtest",
                                float_format="%.4f")
        
        with open(f"{save_path}var_backtest.tex", 'w') as f:
            f.write(latex_code)
        
        print("VaR backtesting table saved to var_backtest.tex")
    
    def _generate_summary_table(self, results_dict, save_path):
        """Generate summary comparison table."""
        summary_data = []
        
        for model_name in self.model_names:
            summary_row = {'Model': model_name}
            
            # Add key metrics from different tables
            if 'basic_stats' in results_dict:
                model_stats = next((row for row in results_dict['basic_stats'] if row['Model'] == model_name), {})
                summary_row.update({
                    'Mean': model_stats.get('Mean', np.nan),
                    'Std Dev': model_stats.get('Std Dev', np.nan),
                    'Skewness': model_stats.get('Skewness', np.nan),
                    'Kurtosis': model_stats.get('Kurtosis', np.nan)
                })
            
            if 'distribution_tests' in results_dict:
                model_tests = next((row for row in results_dict['distribution_tests'] if row['Model'] == model_name), {})
                summary_row.update({
                    'KS_Stat': model_tests.get('KS_Statistic', np.nan),
                    'MMD': model_tests.get('MMD', np.nan)
                })
            
            summary_data.append(summary_row)
        
        df = pd.DataFrame(summary_data)
        df = df.round(4)
        
        latex_code = df.to_latex(index=False,
                                caption="Summary Comparison of All Models",
                                label="tab:summary_comparison",
                                float_format="%.4f")
        
        with open(f"{save_path}summary_comparison.tex", 'w') as f:
            f.write(latex_code)
        
        print("Summary comparison table saved to summary_comparison.tex")
    
    def run_comprehensive_evaluation(self, real_data, synthetic_data_dict, var_forecasts_dict=None, save_path="results/"):
        """
        Run comprehensive evaluation and generate all outputs.
        
        Parameters:
        -----------
        real_data : array-like
            Real financial time series data
        synthetic_data_dict : dict
            Dictionary with model names as keys and synthetic data as values
        var_forecasts_dict : dict, optional
            Dictionary with model names as keys and VaR forecasts as values
        save_path : str
            Path to save results
        """
        import os
        os.makedirs(save_path, exist_ok=True)
        
        print("Starting comprehensive evaluation...")
        
        # Initialize results dictionary
        results = {
            'basic_stats': [],
            'tail_metrics': [],
            'volatility_metrics': [],
            'distribution_tests': [],
            'var_backtest': []
        }
        
        # Add real data statistics
        results['basic_stats'].append(self.compute_basic_statistics(real_data, 'Real Data'))
        results['tail_metrics'].append(self.compute_tail_metrics(real_data, 'Real Data'))
        results['volatility_metrics'].append(self.compute_volatility_metrics(real_data, 'Real Data'))
        
        # Evaluate each model
        for model_name, synthetic_data in synthetic_data_dict.items():
            print(f"Evaluating {model_name}...")
            
            # Basic statistics
            results['basic_stats'].append(self.compute_basic_statistics(synthetic_data, model_name))
            
            # Tail metrics
            results['tail_metrics'].append(self.compute_tail_metrics(synthetic_data, model_name))
            
            # Volatility metrics
            results['volatility_metrics'].append(self.compute_volatility_metrics(synthetic_data, model_name))
            
            # Distribution tests
            results['distribution_tests'].append(
                self.compute_distribution_tests(real_data, synthetic_data, model_name)
            )
            
            # VaR backtesting (if forecasts provided)
            if var_forecasts_dict and model_name in var_forecasts_dict:
                results['var_backtest'].append(
                    self.compute_var_backtest(real_data, var_forecasts_dict[model_name], model_name)
                )
        
        # Generate plots
        print("Generating comparison plots...")
        self.generate_comparison_plots(real_data, synthetic_data_dict, f"{save_path}plots/")
        
        # Generate LaTeX tables
        print("Generating LaTeX tables...")
        self.generate_latex_tables(results, f"{save_path}tables/")
        
        # Save results as JSON for reproducibility
        import json
        with open(f"{save_path}evaluation_results.json", 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            json_results = {}
            for key, value in results.items():
                json_results[key] = []
                for item in value:
                    json_item = {}
                    for k, v in item.items():
                        if isinstance(v, (np.integer, np.floating)):
                            json_item[k] = float(v)
                        else:
                            json_item[k] = v
                    json_results[key].append(json_item)
            
            json.dump(json_results, f, indent=2)
        
        print(f"Comprehensive evaluation completed. Results saved to {save_path}")
        return results
