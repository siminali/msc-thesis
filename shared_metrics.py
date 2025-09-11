#!/usr/bin/env python3
"""
Shared Metrics & Tests Module

Comprehensive evaluation metrics for financial time series models including:
- Risk backtests (VaR/ES, Kupiec POF, Christoffersen independence)
- Quantile loss evaluation
- Diebold-Mariano tests with HAC and small-sample corrections
- Distribution analysis (ECDFs, QQ plots, moments)
- Realized volatility tracking

Author: Simin Ali
Thesis: Diffusion Models in Generative AI for Financial Data Synthesis and Risk Management
"""

import numpy as np
import pandas as pd
import json
import os
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
from scipy import stats
from scipy.stats import jarque_bera, normaltest, kstest
import logging

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RiskMetrics:
    """Risk backtesting and evaluation metrics."""
    
    @staticmethod
    def value_at_risk(returns: np.ndarray, alpha: float = 0.05) -> float:
        """Calculate Value at Risk."""
        return np.percentile(returns, alpha * 100)
    
    @staticmethod
    def expected_shortfall(returns: np.ndarray, alpha: float = 0.05) -> float:
        """Calculate Expected Shortfall (Conditional VaR)."""
        var = RiskMetrics.value_at_risk(returns, alpha)
        return returns[returns <= var].mean()
    
    @staticmethod
    def kupiec_pof_test(returns: np.ndarray, var_forecast: np.ndarray, alpha: float = 0.05) -> Dict[str, float]:
        """
        Kupiec Proportion of Failures test for VaR backtesting.
        
        Args:
            returns: Actual returns
            var_forecast: VaR forecasts
            alpha: VaR confidence level
            
        Returns:
            Dictionary with test statistics and p-value
        """
        # Count violations
        violations = (returns <= var_forecast).astype(int)
        n_violations = violations.sum()
        n_total = len(returns)
        
        # Expected number of violations
        expected_violations = alpha * n_total
        
        # Kupiec test statistic
        if n_violations == 0:
            lr_stat = -2 * n_total * np.log(1 - alpha)
        elif n_violations == n_total:
            lr_stat = -2 * n_total * np.log(alpha)
        else:
            p_hat = n_violations / n_total
            lr_stat = -2 * (n_total * np.log(1 - alpha) + 
                           n_violations * np.log(alpha / p_hat) + 
                           (n_total - n_violations) * np.log((1 - alpha) / (1 - p_hat)))
        
        # P-value from chi-squared distribution with 1 df
        p_value = 1 - stats.chi2.cdf(lr_stat, df=1)
        
        return {
            'n_violations': int(n_violations),
            'n_total': int(n_total),
            'violation_rate': float(n_violations / n_total),
            'expected_rate': float(alpha),
            'lr_statistic': float(lr_stat),
            'p_value': float(p_value),
            'reject_h0': bool(p_value < 0.05)
        }
    
    @staticmethod
    def christoffersen_independence_test(violations: np.ndarray) -> Dict[str, float]:
        """
        Christoffersen independence test for VaR violations.
        
        Args:
            violations: Binary array of VaR violations (1=violation, 0=no violation)
            
        Returns:
            Dictionary with test statistics and p-value
        """
        # Count transitions
        n00 = np.sum((violations[:-1] == 0) & (violations[1:] == 0))
        n01 = np.sum((violations[:-1] == 0) & (violations[1:] == 1))
        n10 = np.sum((violations[:-1] == 1) & (violations[1:] == 0))
        n11 = np.sum((violations[:-1] == 1) & (violations[1:] == 1))
        
        # Calculate transition probabilities
        n0 = n00 + n01
        n1 = n10 + n11
        
        if n0 == 0 or n1 == 0:
            return {
                'lr_independence': 0.0,
                'p_value': 1.0,
                'reject_h0': False,
                'transitions': {'n00': int(n00), 'n01': int(n01), 'n10': int(n10), 'n11': int(n11)}
            }
        
        pi_01 = n01 / n0 if n0 > 0 else 0
        pi_11 = n11 / n1 if n1 > 0 else 0
        pi = (n01 + n11) / (n0 + n1)
        
        # Likelihood ratio test statistic
        if pi_01 == 0 or pi_11 == 0 or pi == 0 or pi == 1:
            lr_stat = 0.0
        else:
            lr_stat = -2 * (n01 * np.log(pi / pi_01) + n11 * np.log(pi / pi_11) + 
                           n00 * np.log((1 - pi) / (1 - pi_01)) + n10 * np.log((1 - pi) / (1 - pi_11)))
        
        # P-value from chi-squared distribution with 1 df
        p_value = 1 - stats.chi2.cdf(lr_stat, df=1)
        
        return {
            'lr_independence': float(lr_stat),
            'p_value': float(p_value),
            'reject_h0': bool(p_value < 0.05),
            'transitions': {'n00': int(n00), 'n01': int(n01), 'n10': int(n10), 'n11': int(n11)},
            'transition_probs': {'pi_01': float(pi_01), 'pi_11': float(pi_11), 'pi': float(pi)}
        }
    
    @staticmethod
    def es_bootstrap_ci(returns: np.ndarray, alpha: float = 0.05, n_bootstrap: int = 1000, 
                       confidence: float = 0.95) -> Dict[str, float]:
        """
        Bootstrap confidence intervals for Expected Shortfall.
        
        Args:
            returns: Return series
            alpha: VaR/ES confidence level
            n_bootstrap: Number of bootstrap samples
            confidence: Confidence level for CI
            
        Returns:
            Dictionary with ES estimate and confidence intervals
        """
        # Original ES estimate
        original_es = RiskMetrics.expected_shortfall(returns, alpha)
        
        # Bootstrap samples
        bootstrap_es = []
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(returns, size=len(returns), replace=True)
            bootstrap_es.append(RiskMetrics.expected_shortfall(bootstrap_sample, alpha))
        
        bootstrap_es = np.array(bootstrap_es)
        
        # Confidence intervals
        ci_lower = (1 - confidence) / 2
        ci_upper = 1 - ci_lower
        
        return {
            'es_estimate': float(original_es),
            'es_mean_bootstrap': float(bootstrap_es.mean()),
            'es_std_bootstrap': float(bootstrap_es.std()),
            'ci_lower': float(np.percentile(bootstrap_es, ci_lower * 100)),
            'ci_upper': float(np.percentile(bootstrap_es, ci_upper * 100)),
            'n_bootstrap': int(n_bootstrap),
            'confidence_level': float(confidence)
        }

class QuantileLoss:
    """Quantile loss evaluation metrics."""
    
    @staticmethod
    def quantile_loss(y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> float:
        """
        Calculate quantile loss.
        
        Args:
            y_true: True values
            y_pred: Predicted quantiles
            alpha: Quantile level
            
        Returns:
            Quantile loss value
        """
        residual = y_true - y_pred
        loss = np.where(residual >= 0, alpha * residual, (alpha - 1) * residual)
        return loss.mean()
    
    @staticmethod
    def evaluate_quantile_forecasts(y_true: np.ndarray, samples: np.ndarray, 
                                  alphas: List[float] = [0.01, 0.05]) -> Dict[str, float]:
        """
        Evaluate quantile forecasts from sample paths.
        
        Args:
            y_true: True return series
            samples: Sample paths [n_paths, n_periods]
            alphas: Quantile levels to evaluate
            
        Returns:
            Dictionary with quantile losses
        """
        results = {}
        
        for alpha in alphas:
            # Calculate empirical quantiles from samples
            quantile_forecasts = np.percentile(samples, alpha * 100, axis=0)
            
            # Calculate quantile loss
            ql = QuantileLoss.quantile_loss(y_true, quantile_forecasts, alpha)
            
            results[f'quantile_loss_{alpha:.3f}'] = float(ql)
        
        return results

class DieboldMarianoTest:
    """Diebold-Mariano test for forecast comparison."""
    
    @staticmethod
    def loss_function(y_true: np.ndarray, y_pred: np.ndarray, loss_type: str = 'mse') -> np.ndarray:
        """Calculate loss series for DM test."""
        if loss_type == 'mse':
            return (y_true - y_pred) ** 2
        elif loss_type == 'mae':
            return np.abs(y_true - y_pred)
        elif loss_type == 'qlike':
            # Quasi-likelihood loss for volatility forecasting
            return y_pred - y_true * np.log(y_pred)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    @staticmethod
    def newey_west_variance(d: np.ndarray, max_lag: int) -> float:
        """
        Calculate Newey-West HAC variance estimator.
        
        Args:
            d: Loss differential series
            max_lag: Maximum lag for HAC estimation
            
        Returns:
            HAC variance estimate
        """
        n = len(d)
        gamma_0 = np.var(d, ddof=1)
        
        gamma_sum = 0
        for j in range(1, max_lag + 1):
            if j < n:
                gamma_j = np.cov(d[:-j], d[j:], ddof=1)[0, 1]
                weight = 1 - j / (max_lag + 1)  # Bartlett kernel
                gamma_sum += 2 * weight * gamma_j
        
        return gamma_0 + gamma_sum
    
    @staticmethod
    def harvey_leybourne_newbold_correction(dm_stat: float, n: int, h: int) -> float:
        """
        Harvey, Leybourne, and Newbold small-sample correction.
        
        Args:
            dm_stat: Original DM statistic
            n: Sample size
            h: Forecast horizon
            
        Returns:
            Corrected test statistic
        """
        correction_factor = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
        return dm_stat * correction_factor
    
    @staticmethod
    def diebold_mariano_test(y_true: np.ndarray, forecast1: np.ndarray, forecast2: np.ndarray,
                           h: int = 1, loss_type: str = 'mse', use_hln_correction: bool = True) -> Dict[str, float]:
        """
        Diebold-Mariano test for forecast comparison.
        
        Args:
            y_true: True values
            forecast1: Forecasts from model 1
            forecast2: Forecasts from model 2
            h: Forecast horizon
            loss_type: Loss function type
            use_hln_correction: Whether to apply HLN small-sample correction
            
        Returns:
            Dictionary with test results
        """
        # Calculate loss differentials
        loss1 = DieboldMarianoTest.loss_function(y_true, forecast1, loss_type)
        loss2 = DieboldMarianoTest.loss_function(y_true, forecast2, loss_type)
        d = loss1 - loss2
        
        n = len(d)
        d_mean = d.mean()
        
        # HAC variance with lag h-1
        max_lag = max(1, h - 1)
        d_var = DieboldMarianoTest.newey_west_variance(d, max_lag)
        
        # DM test statistic
        if d_var <= 0:
            dm_stat = 0.0
        else:
            dm_stat = d_mean / np.sqrt(d_var / n)
        
        # Apply HLN correction
        if use_hln_correction:
            dm_stat_corrected = DieboldMarianoTest.harvey_leybourne_newbold_correction(dm_stat, n, h)
        else:
            dm_stat_corrected = dm_stat
        
        # P-value (two-tailed test)
        p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat_corrected)))
        
        return {
            'dm_statistic': float(dm_stat),
            'dm_statistic_hln': float(dm_stat_corrected),
            'p_value': float(p_value),
            'loss_differential_mean': float(d_mean),
            'loss_differential_var': float(d_var),
            'n_observations': int(n),
            'forecast_horizon': int(h),
            'loss_type': loss_type,
            'hln_correction': use_hln_correction,
            'reject_h0_5pct': bool(p_value < 0.05),
            'reject_h0_10pct': bool(p_value < 0.10)
        }

class DistributionAnalysis:
    """Distribution and structural analysis metrics."""
    
    @staticmethod
    def empirical_cdf_comparison(real_data: np.ndarray, generated_samples: np.ndarray) -> Dict[str, float]:
        """Compare empirical CDFs using Kolmogorov-Smirnov test."""
        # Flatten samples if needed
        if len(generated_samples.shape) > 1:
            generated_flat = generated_samples.flatten()
        else:
            generated_flat = generated_samples
        
        # KS test
        ks_stat, ks_pvalue = stats.ks_2samp(real_data, generated_flat)
        
        return {
            'ks_statistic': float(ks_stat),
            'ks_pvalue': float(ks_pvalue),
            'reject_equal_distributions': bool(ks_pvalue < 0.05)
        }
    
    @staticmethod
    def qq_tail_analysis(real_data: np.ndarray, generated_samples: np.ndarray, 
                        tail_quantiles: List[float] = [0.01, 0.05, 0.95, 0.99]) -> Dict[str, Dict[str, float]]:
        """Analyze tail behavior using Q-Q plots."""
        if len(generated_samples.shape) > 1:
            generated_flat = generated_samples.flatten()
        else:
            generated_flat = generated_samples
        
        results = {}
        
        for q in tail_quantiles:
            real_quantile = np.percentile(real_data, q * 100)
            gen_quantile = np.percentile(generated_flat, q * 100)
            
            results[f'quantile_{q:.3f}'] = {
                'real': float(real_quantile),
                'generated': float(gen_quantile),
                'difference': float(gen_quantile - real_quantile),
                'relative_error': float((gen_quantile - real_quantile) / abs(real_quantile)) if real_quantile != 0 else float('inf')
            }
        
        return results
    
    @staticmethod
    def moment_analysis(real_data: np.ndarray, generated_samples: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Compare statistical moments."""
        if len(generated_samples.shape) > 1:
            generated_flat = generated_samples.flatten()
        else:
            generated_flat = generated_samples
        
        moments = ['mean', 'std', 'skewness', 'kurtosis']
        results = {}
        
        # Calculate moments
        real_mean = real_data.mean()
        real_std = real_data.std()
        real_skew = stats.skew(real_data)
        real_kurt = stats.kurtosis(real_data)
        
        gen_mean = generated_flat.mean()
        gen_std = generated_flat.std()
        gen_skew = stats.skew(generated_flat)
        gen_kurt = stats.kurtosis(generated_flat)
        
        real_moments = [real_mean, real_std, real_skew, real_kurt]
        gen_moments = [gen_mean, gen_std, gen_skew, gen_kurt]
        
        for i, moment in enumerate(moments):
            real_val = real_moments[i]
            gen_val = gen_moments[i]
            
            results[moment] = {
                'real': float(real_val),
                'generated': float(gen_val),
                'difference': float(gen_val - real_val),
                'relative_error': float((gen_val - real_val) / abs(real_val)) if real_val != 0 else float('inf')
            }
        
        return results
    
    @staticmethod
    def realized_volatility_tracking(real_returns: np.ndarray, generated_samples: np.ndarray, 
                                   window: int = 20) -> Dict[str, float]:
        """
        Track realized volatility patterns.
        
        Args:
            real_returns: Real return series
            generated_samples: Generated samples [n_paths, n_periods]
            window: Rolling window for volatility calculation
            
        Returns:
            Dictionary with volatility tracking metrics
        """
        # Calculate rolling realized volatility for real data
        real_vol = pd.Series(real_returns).rolling(window=window).std().dropna()
        
        # Calculate for each generated path and average
        gen_vols = []
        for i in range(generated_samples.shape[0]):
            path_vol = pd.Series(generated_samples[i]).rolling(window=window).std().dropna()
            gen_vols.append(path_vol.values)
        
        # Average across paths
        gen_vol_mean = np.mean(gen_vols, axis=0)
        
        # Ensure same length
        min_len = min(len(real_vol), len(gen_vol_mean))
        real_vol = real_vol.iloc[:min_len]
        gen_vol_mean = gen_vol_mean[:min_len]
        
        if min_len == 0:
            return {
                'rmse': float('nan'),
                'mape': float('nan'),
                'correlation': float('nan'),
                'n_observations': 0
            }
        
        # Calculate tracking metrics
        rmse = np.sqrt(np.mean((real_vol - gen_vol_mean) ** 2))
        mape = np.mean(np.abs((real_vol - gen_vol_mean) / real_vol)) * 100
        correlation = np.corrcoef(real_vol, gen_vol_mean)[0, 1] if min_len > 1 else 0.0
        
        return {
            'rmse': float(rmse),
            'mape': float(mape),
            'correlation': float(correlation),
            'n_observations': int(min_len),
            'window_size': int(window)
        }

class MetricsCalculator:
    """Main metrics calculation class."""
    
    def __init__(self):
        self.results = {}
        self.metadata = {
            'calculated_at': datetime.now().isoformat(),
            'version': '1.0',
            'description': 'Comprehensive financial time series evaluation metrics'
        }
    
    def calculate_all_metrics(self, real_data: np.ndarray, model_samples: Dict[str, np.ndarray],
                            window_id: str = 'default') -> Dict[str, Any]:
        """
        Calculate all metrics for a set of model samples.
        
        Args:
            real_data: Real return series
            model_samples: Dictionary of {model_name: samples_array}
            window_id: Identifier for the evaluation window
            
        Returns:
            Complete metrics dictionary
        """
        logger.info(f"Calculating metrics for window: {window_id}")
        
        results = {
            'window_id': window_id,
            'metadata': self.metadata.copy(),
            'real_data_stats': self._calculate_basic_stats(real_data),
            'models': {},
            'pairwise_comparisons': {}
        }
        
        # Individual model metrics
        for model_name, samples in model_samples.items():
            logger.info(f"Processing model: {model_name}")
            
            try:
                model_results = self._calculate_model_metrics(real_data, samples, model_name)
                results['models'][model_name] = model_results
            except Exception as e:
                logger.error(f"Error calculating metrics for {model_name}: {e}")
                results['models'][model_name] = {
                    'status': 'error',
                    'error_message': str(e)
                }
        
        # Pairwise comparisons
        model_names = list(model_samples.keys())
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                model1, model2 = model_names[i], model_names[j]
                
                if (results['models'][model1].get('status') != 'error' and 
                    results['models'][model2].get('status') != 'error'):
                    
                    try:
                        comparison_key = f"{model1}_vs_{model2}"
                        comparison_results = self._calculate_pairwise_comparison(
                            real_data, model_samples[model1], model_samples[model2], model1, model2
                        )
                        results['pairwise_comparisons'][comparison_key] = comparison_results
                    except Exception as e:
                        logger.error(f"Error in pairwise comparison {model1} vs {model2}: {e}")
                        results['pairwise_comparisons'][f"{model1}_vs_{model2}"] = {
                            'status': 'error',
                            'error_message': str(e)
                        }
        
        return results
    
    def _calculate_basic_stats(self, data: np.ndarray) -> Dict[str, float]:
        """Calculate basic statistics for real data."""
        return {
            'n_observations': int(len(data)),
            'mean': float(data.mean()),
            'std': float(data.std()),
            'min': float(data.min()),
            'max': float(data.max()),
            'skewness': float(stats.skew(data)),
            'kurtosis': float(stats.kurtosis(data)),
            'jarque_bera_stat': float(jarque_bera(data)[0]),
            'jarque_bera_pvalue': float(jarque_bera(data)[1])
        }
    
    def _calculate_model_metrics(self, real_data: np.ndarray, samples: np.ndarray, 
                               model_name: str) -> Dict[str, Any]:
        """Calculate comprehensive metrics for a single model."""
        
        # Calculate path-level returns (sum across time dimension)
        if len(samples.shape) == 2:
            path_returns = samples.sum(axis=1)
        else:
            path_returns = samples
        
        # Basic statistics
        basic_stats = {
            'n_paths': int(samples.shape[0]) if len(samples.shape) > 1 else 1,
            'n_periods': int(samples.shape[1]) if len(samples.shape) > 1 else len(samples),
            'mean': float(path_returns.mean()),
            'std': float(path_returns.std()),
            'min': float(path_returns.min()),
            'max': float(path_returns.max())
        }
        
        # Risk metrics
        risk_metrics = {}
        for alpha in [0.01, 0.05]:
            var = RiskMetrics.value_at_risk(path_returns, alpha)
            es = RiskMetrics.expected_shortfall(path_returns, alpha)
            es_bootstrap = RiskMetrics.es_bootstrap_ci(path_returns, alpha)
            
            risk_metrics[f'var_{alpha:.3f}'] = float(var)
            risk_metrics[f'es_{alpha:.3f}'] = float(es)
            risk_metrics[f'es_bootstrap_{alpha:.3f}'] = es_bootstrap
        
        # Quantile loss (if we have time series)
        quantile_metrics = {}
        if len(samples.shape) == 2 and len(real_data) >= samples.shape[1]:
            quantile_metrics = QuantileLoss.evaluate_quantile_forecasts(
                real_data[:samples.shape[1]], samples
            )
        
        # Distribution analysis
        distribution_metrics = {
            'ecdf_comparison': DistributionAnalysis.empirical_cdf_comparison(real_data, samples),
            'qq_tail_analysis': DistributionAnalysis.qq_tail_analysis(real_data, samples),
            'moment_analysis': DistributionAnalysis.moment_analysis(real_data, samples)
        }
        
        # Realized volatility tracking
        volatility_metrics = {}
        if len(samples.shape) == 2 and len(real_data) >= samples.shape[1]:
            volatility_metrics = DistributionAnalysis.realized_volatility_tracking(
                real_data[:samples.shape[1]], samples
            )
        
        return {
            'status': 'success',
            'model_name': model_name,
            'basic_stats': basic_stats,
            'risk_metrics': risk_metrics,
            'quantile_metrics': quantile_metrics,
            'distribution_metrics': distribution_metrics,
            'volatility_metrics': volatility_metrics
        }
    
    def _calculate_pairwise_comparison(self, real_data: np.ndarray, samples1: np.ndarray, 
                                     samples2: np.ndarray, model1_name: str, 
                                     model2_name: str) -> Dict[str, Any]:
        """Calculate pairwise comparison metrics."""
        
        # Calculate forecasts (mean of samples for each time period)
        if len(samples1.shape) == 2:
            forecast1 = samples1.mean(axis=0)
            forecast2 = samples2.mean(axis=0)
        else:
            forecast1 = samples1
            forecast2 = samples2
        
        # Ensure same length as real data
        min_len = min(len(real_data), len(forecast1), len(forecast2))
        real_truncated = real_data[:min_len]
        forecast1_truncated = forecast1[:min_len]
        forecast2_truncated = forecast2[:min_len]
        
        # Diebold-Mariano tests for different loss functions
        dm_results = {}
        for loss_type in ['mse', 'mae']:
            dm_test = DieboldMarianoTest.diebold_mariano_test(
                real_truncated, forecast1_truncated, forecast2_truncated,
                h=1, loss_type=loss_type, use_hln_correction=True
            )
            dm_results[f'dm_{loss_type}'] = dm_test
        
        # Calculate individual losses for interpretation
        mse1 = np.mean((real_truncated - forecast1_truncated) ** 2)
        mse2 = np.mean((real_truncated - forecast2_truncated) ** 2)
        mae1 = np.mean(np.abs(real_truncated - forecast1_truncated))
        mae2 = np.mean(np.abs(real_truncated - forecast2_truncated))
        
        return {
            'status': 'success',
            'model1_name': model1_name,
            'model2_name': model2_name,
            'n_observations': int(min_len),
            'diebold_mariano_tests': dm_results,
            'individual_losses': {
                'model1_mse': float(mse1),
                'model2_mse': float(mse2),
                'model1_mae': float(mae1),
                'model2_mae': float(mae2),
                'mse_ratio': float(mse2 / mse1) if mse1 != 0 else float('inf'),
                'mae_ratio': float(mae2 / mae1) if mae1 != 0 else float('inf')
            }
        }
    
    def save_results(self, results: Dict[str, Any], output_dir: Path):
        """Save results to JSON and CSV files."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save complete results as JSON
        json_file = output_dir / 'metrics.json'
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved metrics to: {json_file}")
        
        # Create tables directory
        tables_dir = output_dir / 'tables'
        tables_dir.mkdir(exist_ok=True)
        
        # Save summary tables as CSV
        self._save_summary_tables(results, tables_dir)
    
    def _save_summary_tables(self, results: Dict[str, Any], tables_dir: Path):
        """Save summary tables as CSV files."""
        
        # Model comparison table
        if 'models' in results:
            model_data = []
            for model_name, model_results in results['models'].items():
                if model_results.get('status') == 'success':
                    basic_stats = model_results.get('basic_stats', {})
                    risk_metrics = model_results.get('risk_metrics', {})
                    
                    row = {
                        'model': model_name,
                        'n_paths': basic_stats.get('n_paths', 0),
                        'mean_return': basic_stats.get('mean', 0),
                        'volatility': basic_stats.get('std', 0),
                        'var_5pct': risk_metrics.get('var_0.050', 0),
                        'es_5pct': risk_metrics.get('es_0.050', 0),
                        'var_1pct': risk_metrics.get('var_0.010', 0),
                        'es_1pct': risk_metrics.get('es_0.010', 0)
                    }
                    model_data.append(row)
            
            if model_data:
                df_models = pd.DataFrame(model_data)
                df_models.to_csv(tables_dir / 'model_comparison.csv', index=False)
                logger.info(f"Saved model comparison table")
        
        # Pairwise comparison table
        if 'pairwise_comparisons' in results:
            comparison_data = []
            for comparison_name, comparison_results in results['pairwise_comparisons'].items():
                if comparison_results.get('status') == 'success':
                    dm_tests = comparison_results.get('diebold_mariano_tests', {})
                    losses = comparison_results.get('individual_losses', {})
                    
                    row = {
                        'comparison': comparison_name,
                        'model1': comparison_results.get('model1_name', ''),
                        'model2': comparison_results.get('model2_name', ''),
                        'dm_mse_stat': dm_tests.get('dm_mse', {}).get('dm_statistic_hln', 0),
                        'dm_mse_pvalue': dm_tests.get('dm_mse', {}).get('p_value', 1),
                        'dm_mae_stat': dm_tests.get('dm_mae', {}).get('dm_statistic_hln', 0),
                        'dm_mae_pvalue': dm_tests.get('dm_mae', {}).get('p_value', 1),
                        'mse_ratio': losses.get('mse_ratio', 1),
                        'mae_ratio': losses.get('mae_ratio', 1)
                    }
                    comparison_data.append(row)
            
            if comparison_data:
                df_comparisons = pd.DataFrame(comparison_data)
                df_comparisons.to_csv(tables_dir / 'pairwise_comparisons.csv', index=False)
                logger.info(f"Saved pairwise comparison table")
