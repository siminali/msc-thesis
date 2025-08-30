"""
Comprehensive metrics computation for fresh evaluation pipeline.
Computes all financial statistics, risk metrics, and backtesting results.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import kstest, normaltest, jarque_bera
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import warnings

from utils.scaling_guard import ReturnsBundle, require_inverse_scaled_data, compute_rolling_vol
from utils.sanity_gate import SanityGate


@require_inverse_scaled_data
def compute_comprehensive_metrics(real_bundle: ReturnsBundle,
                                 model_bundles: Dict[str, ReturnsBundle],
                                 window_name: str) -> Dict:
    """
    Compute comprehensive metrics for all models in a window.
    
    Args:
        real_bundle: Real data bundle
        model_bundles: Dict of model bundles
        window_name: Window/scenario name
        
    Returns:
        Dict: Comprehensive metrics dictionary
    """
    all_bundles = {'Real': real_bundle, **model_bundles}
    metrics = {'window': window_name, 'models': {}}
    
    for name, bundle in all_bundles.items():
        model_metrics = {}
        
        # Basic statistics
        model_metrics.update(compute_basic_statistics(bundle))
        
        # Residual statistics (vs mean)
        model_metrics.update(compute_residual_statistics(bundle))
        
        # Normality tests
        model_metrics.update(compute_normality_tests(bundle))
        
        # Tail risk metrics
        model_metrics.update(compute_tail_risk_metrics(bundle))
        
        # Rolling volatility statistics
        model_metrics.update(compute_rolling_volatility_stats(bundle))
        
        # Stylized facts
        model_metrics.update(compute_stylized_facts(bundle))
        
        metrics['models'][name] = model_metrics
    
    # Cross-model comparisons
    if len(model_bundles) > 1:
        metrics['comparisons'] = compute_model_comparisons(real_bundle, model_bundles)
    
    # Backtesting analysis
    metrics['backtesting'] = compute_backtesting_metrics(real_bundle, model_bundles)
    
    return metrics


def compute_basic_statistics(bundle: ReturnsBundle) -> Dict:
    """Compute basic statistical measures."""
    returns = bundle.returns
    
    return {
        'mean': float(np.mean(returns)),
        'std': float(np.std(returns, ddof=1)),
        'variance': float(np.var(returns, ddof=1)),
        'min': float(np.min(returns)),
        'max': float(np.max(returns)),
        'median': float(np.median(returns)),
        'skewness': float(stats.skew(returns, bias=False)),
        'kurtosis': float(stats.kurtosis(returns, fisher=False, bias=False)),  # Regular kurtosis
        'excess_kurtosis': float(stats.kurtosis(returns, fisher=True, bias=False)),  # Excess kurtosis
        'n_observations': len(returns),
        'annualized_mean': float(np.mean(returns) * 252) if bundle.annualise_mode == 'none' else float(np.mean(returns)),
        'annualized_std': float(np.std(returns, ddof=1) * np.sqrt(252)) if bundle.annualise_mode == 'none' else float(np.std(returns, ddof=1)),
    }


def compute_residual_statistics(bundle: ReturnsBundle) -> Dict:
    """Compute residual statistics (deviations from mean)."""
    returns = bundle.returns
    residuals = returns - np.mean(returns)
    
    return {
        'residual_me': float(np.mean(residuals)),  # Mean Error (should be ~0)
        'residual_mae': float(np.mean(np.abs(residuals))),  # Mean Absolute Error
        'residual_mse': float(np.mean(residuals**2)),  # Mean Squared Error
        'residual_rmse': float(np.sqrt(np.mean(residuals**2))),  # Root Mean Squared Error
        'residual_std': float(np.std(residuals, ddof=1)),  # Residual Standard Deviation
    }


def compute_normality_tests(bundle: ReturnsBundle) -> Dict:
    """Compute normality tests for returns and standardized residuals."""
    returns = bundle.returns
    
    # Standardized residuals
    residuals = returns - np.mean(returns)
    std_residuals = residuals / np.std(residuals, ddof=1)
    
    # Kolmogorov-Smirnov test
    ks_stat, ks_pval = kstest(std_residuals, 'norm')
    
    # Anderson-Darling test
    try:
        ad_stat, ad_crit, ad_sig = stats.anderson(std_residuals, dist='norm')
        # Convert to p-value approximation
        ad_pval = np.interp(ad_stat, ad_crit[::-1], (1 - ad_sig/100)[::-1])
        ad_pval = max(0.001, min(0.999, ad_pval))  # Clamp to reasonable range
    except Exception:
        ad_stat, ad_pval = np.nan, np.nan
    
    # Jarque-Bera test
    try:
        jb_stat, jb_pval = jarque_bera(returns)
    except Exception:
        jb_stat, jb_pval = np.nan, np.nan
    
    # Shapiro-Wilk test (if sample size is reasonable)
    if len(returns) <= 5000:
        try:
            sw_stat, sw_pval = stats.shapiro(std_residuals[:5000])  # Limit sample size
        except Exception:
            sw_stat, sw_pval = np.nan, np.nan
    else:
        sw_stat, sw_pval = np.nan, np.nan
    
    return {
        'ks_statistic': float(ks_stat),
        'ks_pvalue': float(ks_pval),
        'ad_statistic': float(ad_stat) if not np.isnan(ad_stat) else None,
        'ad_pvalue': float(ad_pval) if not np.isnan(ad_pval) else None,
        'jb_statistic': float(jb_stat) if not np.isnan(jb_stat) else None,
        'jb_pvalue': float(jb_pval) if not np.isnan(jb_pval) else None,
        'sw_statistic': float(sw_stat) if not np.isnan(sw_stat) else None,
        'sw_pvalue': float(sw_pval) if not np.isnan(sw_pval) else None,
    }


def compute_tail_risk_metrics(bundle: ReturnsBundle) -> Dict:
    """Compute VaR and ES at various confidence levels."""
    returns = bundle.returns
    losses = -returns  # Convert to losses for VaR/ES calculation
    
    confidence_levels = [0.90, 0.95, 0.99]
    metrics = {}
    
    for alpha in confidence_levels:
        alpha_str = f"{int(alpha*100)}"
        
        # Value at Risk
        var_val = np.percentile(losses, alpha * 100)
        metrics[f'var_{alpha_str}'] = float(var_val)
        
        # Expected Shortfall (Conditional VaR)
        tail_losses = losses[losses >= var_val]
        if len(tail_losses) > 0:
            es_val = np.mean(tail_losses)
        else:
            es_val = var_val
        metrics[f'es_{alpha_str}'] = float(es_val)
        
        # Lower tail (negative VaR for positive returns)
        var_lower = np.percentile(-losses, alpha * 100)  # Positive extreme returns
        metrics[f'var_lower_{alpha_str}'] = float(var_lower)
    
    return metrics


def compute_rolling_volatility_stats(bundle: ReturnsBundle, window: int = 20) -> Dict:
    """Compute rolling volatility statistics."""
    returns = bundle.returns
    rolling_vol = compute_rolling_vol(returns, window=window, 
                                     annualise=bundle.annualise_mode)
    
    return {
        'rolling_vol_mean': float(np.mean(rolling_vol)),
        'rolling_vol_median': float(np.median(rolling_vol)),
        'rolling_vol_std': float(np.std(rolling_vol, ddof=1)),
        'rolling_vol_min': float(np.min(rolling_vol)),
        'rolling_vol_max': float(np.max(rolling_vol)),
        'rolling_vol_p90': float(np.percentile(rolling_vol, 90)),
        'rolling_vol_p95': float(np.percentile(rolling_vol, 95)),
    }


def compute_stylized_facts(bundle: ReturnsBundle) -> Dict:
    """Compute stylized facts of financial returns."""
    returns = bundle.returns
    
    # Leverage effect (correlation between returns and future volatility)
    try:
        rolling_vol = compute_rolling_vol(returns, window=20)
        if len(rolling_vol) > 21:
            # Correlation between returns and next-period volatility
            leverage_corr = np.corrcoef(returns[:-20], rolling_vol[20:])[0, 1]
        else:
            leverage_corr = np.nan
    except Exception:
        leverage_corr = np.nan
    
    # Volatility clustering (ACF of squared returns)
    returns_sq = returns ** 2
    try:
        from statsmodels.tsa.stattools import acf
        acf_sq = acf(returns_sq, nlags=10, fft=True)
        vol_clustering = float(np.mean(acf_sq[1:6]))  # Average of first 5 lags
    except Exception:
        vol_clustering = np.nan
    
    # Fat tails test
    excess_kurtosis = stats.kurtosis(returns, fisher=True, bias=False)
    fat_tails_flag = excess_kurtosis > 0.5
    
    # Volatility clustering test
    vol_clustering_flag = vol_clustering > 0.1 if not np.isnan(vol_clustering) else False
    
    # Leverage effect test
    leverage_flag = leverage_corr < -0.1 if not np.isnan(leverage_corr) else False
    
    return {
        'leverage_correlation': float(leverage_corr) if not np.isnan(leverage_corr) else None,
        'volatility_clustering': float(vol_clustering) if not np.isnan(vol_clustering) else None,
        'fat_tails_flag': bool(fat_tails_flag),
        'vol_clustering_flag': bool(vol_clustering_flag),
        'leverage_flag': bool(leverage_flag),
    }


def compute_model_comparisons(real_bundle: ReturnsBundle, 
                             model_bundles: Dict[str, ReturnsBundle]) -> Dict:
    """Compute pairwise model comparisons."""
    comparisons = {}
    
    # Rolling volatility correlations with real data
    real_vol = compute_rolling_vol(real_bundle.returns, window=20)
    
    for name, bundle in model_bundles.items():
        model_vol = compute_rolling_vol(bundle.returns, window=20)
        
        # Trim to same length
        min_len = min(len(real_vol), len(model_vol))
        vol_corr = np.corrcoef(real_vol[:min_len], model_vol[:min_len])[0, 1]
        
        # Volatility ratio statistics
        vol_ratio = model_vol[:min_len] / np.maximum(real_vol[:min_len], 1e-8)
        
        comparisons[name] = {
            'vol_correlation_with_real': float(vol_corr) if not np.isnan(vol_corr) else None,
            'vol_ratio_mean': float(np.mean(vol_ratio)),
            'vol_ratio_median': float(np.median(vol_ratio)),
            'vol_ratio_std': float(np.std(vol_ratio, ddof=1)),
        }
    
    return comparisons


def compute_backtesting_metrics(real_bundle: ReturnsBundle,
                               model_bundles: Dict[str, ReturnsBundle]) -> Dict:
    """Compute VaR backtesting metrics (Kupiec, Christoffersen tests)."""
    backtesting = {}
    
    confidence_levels = [0.95, 0.99]
    real_returns = real_bundle.returns
    n_obs = len(real_returns)
    
    for name, bundle in model_bundles.items():
        model_results = {}
        model_returns = bundle.returns
        
        # Align lengths
        min_len = min(len(real_returns), len(model_returns))
        real_aligned = real_returns[:min_len]
        
        for alpha in confidence_levels:
            alpha_str = f"{int(alpha*100)}"
            
            # Use model's VaR as the forecast
            var_threshold = np.percentile(-model_returns, alpha * 100)  # VaR in loss terms
            
            # Exceedances (when actual loss > VaR)
            actual_losses = -real_aligned
            exceedances = actual_losses > var_threshold
            
            n_exceedances = np.sum(exceedances)
            expected_exceedances = (1 - alpha) * min_len
            
            # Kupiec Likelihood Ratio test (unconditional coverage)
            lr_uc_stat, lr_uc_pval = kupiec_test(n_exceedances, min_len, 1 - alpha)
            
            # Christoffersen Likelihood Ratio test (conditional coverage)
            lr_cc_stat, lr_cc_pval = christoffersen_test(exceedances, 1 - alpha)
            
            model_results[f'alpha_{alpha_str}'] = {
                'n_exceedances': int(n_exceedances),
                'expected_exceedances': float(expected_exceedances),
                'exceedance_rate': float(n_exceedances / min_len),
                'expected_rate': float(1 - alpha),
                'kupiec_lr_stat': float(lr_uc_stat),
                'kupiec_lr_pval': float(lr_uc_pval),
                'christoffersen_lr_stat': float(lr_cc_stat),
                'christoffersen_lr_pval': float(lr_cc_pval),
            }
        
        backtesting[name] = model_results
    
    return backtesting


def kupiec_test(n_exceedances: int, n_obs: int, alpha: float) -> Tuple[float, float]:
    """
    Kupiec likelihood ratio test for unconditional coverage.
    H0: Exceedance rate = alpha
    """
    if n_exceedances == 0 or n_exceedances == n_obs:
        return np.nan, np.nan
    
    # Observed rate
    p_hat = n_exceedances / n_obs
    
    # Log-likelihood under H0
    ll_h0 = n_exceedances * np.log(alpha) + (n_obs - n_exceedances) * np.log(1 - alpha)
    
    # Log-likelihood under H1 (unrestricted)
    ll_h1 = n_exceedances * np.log(p_hat) + (n_obs - n_exceedances) * np.log(1 - p_hat)
    
    # Likelihood ratio statistic
    lr_stat = -2 * (ll_h0 - ll_h1)
    
    # P-value (chi-squared with 1 df)
    p_value = 1 - stats.chi2.cdf(lr_stat, df=1)
    
    return lr_stat, p_value


def christoffersen_test(exceedances: np.ndarray, alpha: float) -> Tuple[float, float]:
    """
    Christoffersen likelihood ratio test for conditional coverage.
    H0: Exceedances are independent and have correct unconditional coverage
    """
    # Count transitions
    n00 = np.sum((exceedances[:-1] == 0) & (exceedances[1:] == 0))
    n01 = np.sum((exceedances[:-1] == 0) & (exceedances[1:] == 1))
    n10 = np.sum((exceedances[:-1] == 1) & (exceedances[1:] == 0))
    n11 = np.sum((exceedances[:-1] == 1) & (exceedances[1:] == 1))
    
    # Avoid division by zero
    if n01 + n00 == 0 or n10 + n11 == 0:
        return np.nan, np.nan
    
    # Transition probabilities
    p01 = n01 / (n01 + n00)  # P(exceedance | no previous exceedance)
    p11 = n11 / (n10 + n11)  # P(exceedance | previous exceedance)
    
    # Overall exceedance rate
    n_exc = np.sum(exceedances)
    p = n_exc / len(exceedances)
    
    if p <= 0 or p >= 1 or p01 <= 0 or p01 >= 1 or p11 <= 0 or p11 >= 1:
        return np.nan, np.nan
    
    # Log-likelihood under H0 (independence)
    ll_h0 = (n00 + n10) * np.log(1 - p) + (n01 + n11) * np.log(p)
    
    # Log-likelihood under H1 (dependence)
    ll_h1 = (n00 * np.log(1 - p01) + n01 * np.log(p01) + 
             n10 * np.log(1 - p11) + n11 * np.log(p11))
    
    # Likelihood ratio statistic
    lr_stat = -2 * (ll_h0 - ll_h1)
    
    # P-value (chi-squared with 1 df)
    p_value = 1 - stats.chi2.cdf(lr_stat, df=1)
    
    return lr_stat, p_value


def save_metrics_tables(metrics: Dict, output_dir: Path) -> Dict[str, Tuple[Path, Path]]:
    """
    Save metrics as CSV and LaTeX tables.
    
    Args:
        metrics: Comprehensive metrics dictionary
        output_dir: Output directory
        
    Returns:
        Dict: Paths to saved tables
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    window_name = metrics['window']
    saved_paths = {}
    
    # 1. Basic statistics table
    basic_stats_df = create_basic_stats_table(metrics)
    csv_path = output_dir / f"basic_stats_{window_name}.csv"
    tex_path = output_dir / f"basic_stats_{window_name}.tex"
    basic_stats_df.to_csv(csv_path, index=True)
    basic_stats_df.to_latex(tex_path, float_format='%.4f')
    saved_paths['basic_stats'] = (csv_path, tex_path)
    
    # 2. Tail risk metrics table
    tail_metrics_df = create_tail_metrics_table(metrics)
    csv_path = output_dir / f"tail_metrics_{window_name}.csv"
    tex_path = output_dir / f"tail_metrics_{window_name}.tex"
    tail_metrics_df.to_csv(csv_path, index=True)
    tail_metrics_df.to_latex(tex_path, float_format='%.4f')
    saved_paths['tail_metrics'] = (csv_path, tex_path)
    
    # 3. Stylized facts table
    stylized_df = create_stylized_facts_table(metrics)
    csv_path = output_dir / f"stylized_facts_{window_name}.csv"
    tex_path = output_dir / f"stylized_facts_{window_name}.tex"
    stylized_df.to_csv(csv_path, index=True)
    stylized_df.to_latex(tex_path, float_format='%.4f')
    saved_paths['stylized_facts'] = (csv_path, tex_path)
    
    # 4. Backtesting table
    if 'backtesting' in metrics:
        backtest_df = create_backtesting_table(metrics)
        csv_path = output_dir / f"backtesting_{window_name}.csv"
        tex_path = output_dir / f"backtesting_{window_name}.tex"
        backtest_df.to_csv(csv_path, index=True)
        backtest_df.to_latex(tex_path, float_format='%.4f')
        saved_paths['backtesting'] = (csv_path, tex_path)
    
    # 5. Model comparisons table
    if 'comparisons' in metrics:
        comp_df = create_comparisons_table(metrics)
        csv_path = output_dir / f"model_comparisons_{window_name}.csv"
        tex_path = output_dir / f"model_comparisons_{window_name}.tex"
        comp_df.to_csv(csv_path, index=True)
        comp_df.to_latex(tex_path, float_format='%.4f')
        saved_paths['comparisons'] = (csv_path, tex_path)
    
    return saved_paths


def create_basic_stats_table(metrics: Dict) -> pd.DataFrame:
    """Create basic statistics table."""
    data = {}
    for model_name, model_metrics in metrics['models'].items():
        data[model_name] = {
            'Mean': model_metrics.get('mean', np.nan),
            'Std': model_metrics.get('std', np.nan),
            'Skewness': model_metrics.get('skewness', np.nan),
            'Kurtosis': model_metrics.get('kurtosis', np.nan),
            'Excess Kurt': model_metrics.get('excess_kurtosis', np.nan),
            'Min': model_metrics.get('min', np.nan),
            'Max': model_metrics.get('max', np.nan),
            'N Obs': model_metrics.get('n_observations', np.nan),
        }
    
    return pd.DataFrame(data).T


def create_tail_metrics_table(metrics: Dict) -> pd.DataFrame:
    """Create tail risk metrics table."""
    data = {}
    for model_name, model_metrics in metrics['models'].items():
        data[model_name] = {
            'VaR95%': model_metrics.get('var_95', np.nan),
            'VaR99%': model_metrics.get('var_99', np.nan),
            'ES95%': model_metrics.get('es_95', np.nan),
            'ES99%': model_metrics.get('es_99', np.nan),
        }
    
    return pd.DataFrame(data).T


def create_stylized_facts_table(metrics: Dict) -> pd.DataFrame:
    """Create stylized facts table."""
    data = {}
    for model_name, model_metrics in metrics['models'].items():
        data[model_name] = {
            'Fat Tails': model_metrics.get('fat_tails_flag', False),
            'Vol Clustering': model_metrics.get('vol_clustering_flag', False),
            'Leverage Effect': model_metrics.get('leverage_flag', False),
            'Leverage Corr': model_metrics.get('leverage_correlation', np.nan),
            'Vol Clust Coeff': model_metrics.get('volatility_clustering', np.nan),
        }
    
    return pd.DataFrame(data).T


def create_backtesting_table(metrics: Dict) -> pd.DataFrame:
    """Create backtesting results table."""
    data = {}
    
    for model_name, backtest_results in metrics['backtesting'].items():
        for alpha_key, alpha_results in backtest_results.items():
            alpha_str = alpha_key.replace('alpha_', '')
            
            data[f'{model_name}_VaR{alpha_str}'] = {
                'N Exceedances': alpha_results.get('n_exceedances', np.nan),
                'Expected Exc': alpha_results.get('expected_exceedances', np.nan),
                'Exc Rate': alpha_results.get('exceedance_rate', np.nan),
                'Kupiec LR Stat': alpha_results.get('kupiec_lr_stat', np.nan),
                'Kupiec p-val': alpha_results.get('kupiec_lr_pval', np.nan),
                'Christoff LR Stat': alpha_results.get('christoffersen_lr_stat', np.nan),
                'Christoff p-val': alpha_results.get('christoffersen_lr_pval', np.nan),
            }
    
    return pd.DataFrame(data).T


def create_comparisons_table(metrics: Dict) -> pd.DataFrame:
    """Create model comparisons table."""
    data = {}
    
    for model_name, comp_results in metrics['comparisons'].items():
        if isinstance(comp_results, dict) and 'vol_correlation_with_real' in comp_results:
            data[model_name] = {
                'Vol Corr w/ Real': comp_results.get('vol_correlation_with_real', np.nan),
                'Vol Ratio Mean': comp_results.get('vol_ratio_mean', np.nan),
                'Vol Ratio Median': comp_results.get('vol_ratio_median', np.nan),
                'Vol Ratio Std': comp_results.get('vol_ratio_std', np.nan),
            }
    
    return pd.DataFrame(data).T