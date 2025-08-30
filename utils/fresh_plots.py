"""
Essential plotting functions for fresh evaluation pipeline.
All functions accept ReturnsBundle objects and produce publication-quality figures.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from scipy import stats
from scipy.stats import norm, kstest
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import warnings

from utils.scaling_guard import ReturnsBundle, require_inverse_scaled_data, compute_rolling_vol
from utils.sanity_gate import SanityGate, add_suspect_scale_tag

# Set consistent style
plt.style.use('default')
sns.set_palette("husl")

# Global plot settings
FIGSIZE_SINGLE = (10, 8)
FIGSIZE_MULTI = (15, 10)
DPI = 150
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']


@require_inverse_scaled_data
def create_histogram_plot(real_bundle: ReturnsBundle, 
                         model_bundles: Dict[str, ReturnsBundle],
                         window_name: str,
                         output_path: Path,
                         suspect_tags: Dict[str, str] = None) -> None:
    """
    Create histogram plots with log-y axis and Gaussian overlays.
    
    Args:
        real_bundle: Real data bundle
        model_bundles: Dict of model bundles
        window_name: Window/scenario name
        output_path: Output path (without extension)
        suspect_tags: Dict of suspect scale tags per model
    """
    suspect_tags = suspect_tags or {}
    
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_MULTI)
    axes = axes.flatten()
    
    all_bundles = {'Real': real_bundle, **model_bundles}
    
    for idx, (name, bundle) in enumerate(all_bundles.items()):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        returns = bundle.returns
        
        # Create histogram
        bins = np.linspace(np.percentile(returns, 1), np.percentile(returns, 99), 50)
        counts, bin_edges, _ = ax.hist(returns, bins=bins, alpha=0.7, 
                                      density=True, label=f'{name} Data',
                                      color=COLORS[idx % len(COLORS)])
        
        # Overlay Gaussian fit
        mu, sigma = np.mean(returns), np.std(returns, ddof=1)
        x = np.linspace(bin_edges[0], bin_edges[-1], 200)
        gaussian_fit = norm.pdf(x, mu, sigma)
        ax.plot(x, gaussian_fit, 'r-', linewidth=2, label=f'N({mu:.4f}, {sigma:.4f}²)')
        
        # Set log scale
        ax.set_yscale('log')
        
        # Add statistics
        kurtosis_val = bundle.kurtosis
        excess_kurtosis = kurtosis_val - 3.0
        stats_text = f'Kurt: {kurtosis_val:.2f}\nExcess Kurt: {excess_kurtosis:.2f}'
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))
        
        # Title with suspect tag if applicable
        title = f'{name} Returns Distribution'
        suspect_tag = suspect_tags.get(name, "OK")
        title = add_suspect_scale_tag(title, suspect_tag)
        ax.set_title(title)
        
        ax.set_xlabel('Returns')
        ax.set_ylabel('Density (log scale)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(len(all_bundles), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle(f'Return Distributions - {window_name}', fontsize=16)
    plt.tight_layout()
    
    # Save both PDF and PNG
    plt.savefig(f"{output_path}.pdf", dpi=DPI, bbox_inches='tight')
    plt.savefig(f"{output_path}.png", dpi=DPI, bbox_inches='tight')
    plt.close()


@require_inverse_scaled_data
def create_qq_plots(real_bundle: ReturnsBundle,
                   model_bundles: Dict[str, ReturnsBundle],
                   window_name: str,
                   output_path: Path,
                   suspect_tags: Dict[str, str] = None) -> None:
    """
    Create Q-Q plots for left and right tails with identical axes.
    
    Args:
        real_bundle: Real data bundle
        model_bundles: Dict of model bundles  
        window_name: Window/scenario name
        output_path: Output path (without extension)
        suspect_tags: Dict of suspect scale tags per model
    """
    suspect_tags = suspect_tags or {}
    
    all_bundles = {'Real': real_bundle, **model_bundles}
    n_models = len(all_bundles)
    
    fig, axes = plt.subplots(2, n_models, figsize=(4*n_models, 8))
    if n_models == 1:
        axes = axes.reshape(2, 1)
    
    # Determine shared axis limits
    all_data = np.concatenate([bundle.returns for bundle in all_bundles.values()])
    q_min, q_max = np.percentile(all_data, [5, 95])
    
    for col, (name, bundle) in enumerate(all_bundles.items()):
        returns = bundle.returns
        
        # Left tail Q-Q plot (lower 10%)
        ax_left = axes[0, col]
        left_tail = returns[returns <= np.percentile(returns, 10)]
        if len(left_tail) > 5:
            stats.probplot(left_tail, dist="norm", plot=ax_left)
            ax_left.set_title(f'{name} - Left Tail (≤10%)')
        
        # Right tail Q-Q plot (upper 10%)  
        ax_right = axes[1, col]
        right_tail = returns[returns >= np.percentile(returns, 90)]
        if len(right_tail) > 5:
            stats.probplot(right_tail, dist="norm", plot=ax_right)
            ax_right.set_title(f'{name} - Right Tail (≥90%)')
        
        # Set identical axis limits
        for ax in [ax_left, ax_right]:
            ax.set_xlim(q_min, q_max)
            ax.set_ylim(q_min, q_max)
            ax.grid(True, alpha=0.3)
            
            # Add suspect tag if applicable
            suspect_tag = suspect_tags.get(name, "OK")
            if suspect_tag != "OK":
                ax.text(0.05, 0.95, f"⚠️ {suspect_tag}", transform=ax.transAxes,
                       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    plt.suptitle(f'Q-Q Plots - {window_name}', fontsize=16)
    plt.tight_layout()
    
    plt.savefig(f"{output_path}.pdf", dpi=DPI, bbox_inches='tight')
    plt.savefig(f"{output_path}.png", dpi=DPI, bbox_inches='tight')
    plt.close()


@require_inverse_scaled_data  
def create_acf_pacf_plots(real_bundle: ReturnsBundle,
                         model_bundles: Dict[str, ReturnsBundle],
                         window_name: str,
                         output_path: Path,
                         max_lags: int = 20,
                         suspect_tags: Dict[str, str] = None) -> None:
    """
    Create ACF/PACF plots for returns and squared returns.
    
    Args:
        real_bundle: Real data bundle
        model_bundles: Dict of model bundles
        window_name: Window/scenario name
        output_path: Output path (without extension)
        max_lags: Maximum number of lags
        suspect_tags: Dict of suspect scale tags per model
    """
    from statsmodels.tsa.stattools import acf, pacf
    
    suspect_tags = suspect_tags or {}
    all_bundles = {'Real': real_bundle, **model_bundles}
    n_models = len(all_bundles)
    
    fig, axes = plt.subplots(4, n_models, figsize=(4*n_models, 16))
    if n_models == 1:
        axes = axes.reshape(4, 1)
    
    for col, (name, bundle) in enumerate(all_bundles.items()):
        returns = bundle.returns
        returns_sq = returns ** 2
        
        # ACF of returns
        ax_acf = axes[0, col]
        try:
            acf_vals = acf(returns, nlags=max_lags, fft=True)
            lags = np.arange(len(acf_vals))
            ax_acf.stem(lags, acf_vals, basefmt=' ')
            ax_acf.axhline(0, color='black', linestyle='-', alpha=0.5)
            
            # Add 95% confidence bands
            n = len(returns)
            conf_int = 1.96 / np.sqrt(n)
            ax_acf.axhline(conf_int, color='red', linestyle='--', alpha=0.7)
            ax_acf.axhline(-conf_int, color='red', linestyle='--', alpha=0.7)
            
            ax_acf.set_title(f'{name} - ACF Returns')
            ax_acf.set_ylabel('ACF')
        except Exception as e:
            ax_acf.text(0.5, 0.5, f'ACF Error: {str(e)[:30]}...', 
                       transform=ax_acf.transAxes, ha='center')
        
        # PACF of returns
        ax_pacf = axes[1, col]
        try:
            pacf_vals = pacf(returns, nlags=max_lags)
            lags = np.arange(len(pacf_vals))
            ax_pacf.stem(lags, pacf_vals, basefmt=' ')
            ax_pacf.axhline(0, color='black', linestyle='-', alpha=0.5)
            ax_pacf.axhline(conf_int, color='red', linestyle='--', alpha=0.7)
            ax_pacf.axhline(-conf_int, color='red', linestyle='--', alpha=0.7)
            
            ax_pacf.set_title(f'{name} - PACF Returns')
            ax_pacf.set_ylabel('PACF')
        except Exception as e:
            ax_pacf.text(0.5, 0.5, f'PACF Error: {str(e)[:30]}...', 
                        transform=ax_pacf.transAxes, ha='center')
        
        # ACF of squared returns (volatility clustering)
        ax_acf_sq = axes[2, col]
        try:
            acf_sq_vals = acf(returns_sq, nlags=max_lags, fft=True)
            lags = np.arange(len(acf_sq_vals))
            ax_acf_sq.stem(lags, acf_sq_vals, basefmt=' ')
            ax_acf_sq.axhline(0, color='black', linestyle='-', alpha=0.5)
            ax_acf_sq.axhline(conf_int, color='red', linestyle='--', alpha=0.7)
            ax_acf_sq.axhline(-conf_int, color='red', linestyle='--', alpha=0.7)
            
            ax_acf_sq.set_title(f'{name} - ACF Squared Returns')
            ax_acf_sq.set_ylabel('ACF')
        except Exception as e:
            ax_acf_sq.text(0.5, 0.5, f'ACF Sq Error: {str(e)[:30]}...', 
                          transform=ax_acf_sq.transAxes, ha='center')
        
        # PACF of squared returns
        ax_pacf_sq = axes[3, col]
        try:
            pacf_sq_vals = pacf(returns_sq, nlags=max_lags)
            lags = np.arange(len(pacf_sq_vals))
            ax_pacf_sq.stem(lags, pacf_sq_vals, basefmt=' ')
            ax_pacf_sq.axhline(0, color='black', linestyle='-', alpha=0.5)
            ax_pacf_sq.axhline(conf_int, color='red', linestyle='--', alpha=0.7)
            ax_pacf_sq.axhline(-conf_int, color='red', linestyle='--', alpha=0.7)
            
            ax_pacf_sq.set_title(f'{name} - PACF Squared Returns')
            ax_pacf_sq.set_ylabel('PACF')
            ax_pacf_sq.set_xlabel('Lag')
        except Exception as e:
            ax_pacf_sq.text(0.5, 0.5, f'PACF Sq Error: {str(e)[:30]}...', 
                           transform=ax_pacf_sq.transAxes, ha='center')
        
        # Add suspect tag to first subplot if applicable
        suspect_tag = suspect_tags.get(name, "OK")
        if suspect_tag != "OK":
            axes[0, col].text(0.95, 0.95, f"⚠️ {suspect_tag}", 
                             transform=axes[0, col].transAxes, ha='right',
                             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    plt.suptitle(f'Autocorrelation Analysis - {window_name}', fontsize=16)
    plt.tight_layout()
    
    plt.savefig(f"{output_path}.pdf", dpi=DPI, bbox_inches='tight')
    plt.savefig(f"{output_path}.png", dpi=DPI, bbox_inches='tight')
    plt.close()


@require_inverse_scaled_data
def create_standardized_residuals_plot(real_bundle: ReturnsBundle,
                                     model_bundles: Dict[str, ReturnsBundle],
                                     window_name: str,
                                     output_path: Path,
                                     suspect_tags: Dict[str, str] = None) -> None:
    """
    Create standardized residuals histogram with N(0,1) overlay and stats.
    
    Args:
        real_bundle: Real data bundle
        model_bundles: Dict of model bundles
        window_name: Window/scenario name
        output_path: Output path (without extension)
        suspect_tags: Dict of suspect scale tags per model
    """
    suspect_tags = suspect_tags or {}
    
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_MULTI)
    axes = axes.flatten()
    
    all_bundles = {'Real': real_bundle, **model_bundles}
    
    for idx, (name, bundle) in enumerate(all_bundles.items()):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        returns = bundle.returns
        
        # Compute residuals (deviations from mean)
        residuals = returns - np.mean(returns)
        
        # Standardize residuals
        std_residuals = residuals / np.std(residuals, ddof=1)
        
        # Create histogram
        bins = np.linspace(-4, 4, 40)
        counts, bin_edges, _ = ax.hist(std_residuals, bins=bins, alpha=0.7,
                                      density=True, label='Standardized Residuals',
                                      color=COLORS[idx % len(COLORS)])
        
        # Overlay N(0,1)
        x = np.linspace(-4, 4, 200)
        normal_pdf = norm.pdf(x, 0, 1)
        ax.plot(x, normal_pdf, 'r-', linewidth=2, label='N(0,1)')
        
        # Compute residual statistics
        me = np.mean(residuals)  # Mean Error
        mae = np.mean(np.abs(residuals))  # Mean Absolute Error
        mse = np.mean(residuals**2)  # Mean Squared Error
        rmse = np.sqrt(mse)  # Root Mean Squared Error
        resid_std = np.std(residuals, ddof=1)
        
        # KS test for normality of standardized residuals
        ks_stat, ks_pval = kstest(std_residuals, 'norm')
        
        # Verify standardization (should be ≈ 1.0)
        std_residuals_std = np.std(std_residuals, ddof=1)
        
        # Format p-value with scientific notation for very small values
        if ks_pval < 0.001:
            ks_pval_str = f'{ks_pval:.2e}'
        else:
            ks_pval_str = f'{ks_pval:.3f}'
        
        # Add statistics text
        stats_text = (f'ME: {me:.5f}\nMAE: {mae:.5f}\n'
                     f'MSE: {mse:.5f}\nRMSE: {rmse:.5f}\n'
                     f'Resid Std: {resid_std:.5f}\n'
                     f'Std(Std.Resid): {std_residuals_std:.3f}\n'
                     f'KS p-val: {ks_pval_str}')
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               verticalalignment='top', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Title with suspect tag
        title = f'{name} Standardized Residuals'
        suspect_tag = suspect_tags.get(name, "OK")
        title = add_suspect_scale_tag(title, suspect_tag)
        ax.set_title(title)
        
        ax.set_xlabel('Standardized Residuals')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-4, 4)
    
    # Hide unused subplots
    for idx in range(len(all_bundles), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle(f'Standardized Residuals Analysis - {window_name}', fontsize=16)
    plt.tight_layout()
    
    plt.savefig(f"{output_path}.pdf", dpi=DPI, bbox_inches='tight')
    plt.savefig(f"{output_path}.png", dpi=DPI, bbox_inches='tight')
    plt.close()


@require_inverse_scaled_data
def create_rolling_volatility_plots(real_bundle: ReturnsBundle,
                                   model_bundles: Dict[str, ReturnsBundle],
                                   window_name: str,
                                   output_path: Path,
                                   window_size: int = 20,
                                   suspect_tags: Dict[str, str] = None) -> None:
    """
    Create distributional rolling volatility comparison plots (no timestamp alignment).
    
    Args:
        real_bundle: Real data bundle
        model_bundles: Dict of model bundles
        window_name: Window/scenario name
        output_path: Output path (without extension)
        window_size: Rolling window size
        suspect_tags: Dict of suspect scale tags per model
    """
    suspect_tags = suspect_tags or {}
    
    # Create figure with histogram/KDE and ECDF panels
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Compute rolling volatilities for all series (no alignment required)
    volatility_data = {}
    vol_stats = {}
    gof_results = {}
    
    # Real data volatility
    real_vol = compute_rolling_vol(real_bundle.returns, window=window_size, 
                                  ddof=1, demean=False, annualise='none')
    real_vol_clean = real_vol[np.isfinite(real_vol)]
    volatility_data['Real'] = real_vol_clean
    
    # Model volatilities (use all available data, no truncation)
    for model_name, bundle in model_bundles.items():
        model_vol = compute_rolling_vol(bundle.returns, window=window_size,
                                       ddof=1, demean=False, annualise='none')
        model_vol_clean = model_vol[np.isfinite(model_vol)]
        volatility_data[model_name] = model_vol_clean
    
    # Compute distributional statistics
    for series_name, vol_data in volatility_data.items():
        if len(vol_data) > 0:
            vol_stats[series_name] = {
                'mean': np.mean(vol_data),
                'median': np.median(vol_data),
                'p90': np.percentile(vol_data, 90),
                'p95': np.percentile(vol_data, 95),
                'n': len(vol_data)
            }
        else:
            vol_stats[series_name] = {'mean': 0, 'median': 0, 'p90': 0, 'p95': 0, 'n': 0}
    
    # Define consistent colors
    colors = {'Real': 'black', 'zero': '#1f77b4', 'explicit': '#ff7f0e', 'llm': '#2ca02c'}
    
    # Panel 1: Histogram with KDE overlay
    for series_name, vol_data in volatility_data.items():
        if len(vol_data) > 10:
            ax1.hist(vol_data, bins=30, density=True, alpha=0.6,
                    color=colors.get(series_name, '#666666'), label=f'{series_name}')
            
            # Add KDE if we have enough points
            try:
                from scipy.stats import gaussian_kde
                kde = gaussian_kde(vol_data)
                x_range = np.linspace(vol_data.min(), vol_data.max(), 100)
                ax1.plot(x_range, kde(x_range), color=colors.get(series_name, '#666666'),
                        linewidth=2, linestyle='--')
            except Exception:
                pass  # Skip KDE if scipy not available or other issues
    
    # Panel 2: ECDF
    for series_name, vol_data in volatility_data.items():
        if len(vol_data) > 0:
            sorted_vol = np.sort(vol_data)
            ecdf_y = np.arange(1, len(sorted_vol) + 1) / len(sorted_vol)
            ax2.plot(sorted_vol, ecdf_y, color=colors.get(series_name, '#666666'),
                    linewidth=2, label=f'{series_name}')
    
    # Compute goodness-of-fit tests vs real data
    if len(real_vol_clean) > 0:
        for series_name, vol_data in volatility_data.items():
            if series_name != 'Real' and len(vol_data) > 0:
                try:
                    from scipy import stats as scipy_stats
                    ks_stat, ks_pvalue = scipy_stats.ks_2samp(real_vol_clean, vol_data)
                    gof_results[series_name] = {'ks_stat': ks_stat, 'ks_pvalue': ks_pvalue}
                except Exception:
                    gof_results[series_name] = {'ks_stat': np.nan, 'ks_pvalue': np.nan}
    
    # Configure panel 1 (Histogram/KDE)
    ax1.set_xlabel('Volatility σ$_{w}$ (decimal)')
    ax1.set_ylabel('Density')
    ax1.set_title(f'σ$_{{w}}$ Distribution Comparison (window={window_size}) - {window_name}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Configure panel 2 (ECDF)
    ax2.set_xlabel('Volatility σ$_{w}$ (decimal)')
    ax2.set_ylabel('Cumulative Probability')
    ax2.set_title('σ$_{w}$ Empirical CDFs')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add summary statistics as text
    stats_text = []
    for series_name, stats in vol_stats.items():
        stats_text.append(f"{series_name}: μ={stats['mean']:.6f}, med={stats['median']:.6f}, p90={stats['p90']:.6f}, n={stats['n']}")
    
    # Add GOF tests as footnote
    gof_text = []
    for series_name, gof in gof_results.items():
        if not np.isnan(gof['ks_pvalue']):
            if gof['ks_pvalue'] < 0.001:
                gof_text.append(f"{series_name}: KS p={gof['ks_pvalue']:.2e}")
            else:
                gof_text.append(f"{series_name}: KS p={gof['ks_pvalue']:.4f}")
    
    # Add sample size note
    real_n = len(real_vol_clean)
    model_ns = [len(vol_data) for series_name, vol_data in volatility_data.items() if series_name != 'Real']
    unique_ns = set(model_ns + [real_n])
    if len(unique_ns) > 1:
        sample_size_note = f"Sample sizes: Real n={real_n}, Models n={model_ns[0] if model_ns else 'N/A'}"
    else:
        sample_size_note = f"Sample size: n={real_n} (all series)"
    
    # Add text box with statistics
    if stats_text:
        stats_str = '\n'.join(stats_text)
        if gof_text:
            stats_str += '\n\nGOF vs Real:\n' + ', '.join(gof_text)
        stats_str += f'\n\n{sample_size_note}'
        fig.text(0.02, 0.02, stats_str, fontsize=8, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Add suspect scale tags if present
    add_suspect_scale_tag(fig, suspect_tags)
    
    # Save both formats
    plt.savefig(f"{output_path}.pdf", dpi=DPI, bbox_inches='tight')
    plt.savefig(f"{output_path}.png", dpi=DPI, bbox_inches='tight')
    plt.close()


@require_inverse_scaled_data
def create_var_es_curves(real_bundle: ReturnsBundle,
                        model_bundles: Dict[str, ReturnsBundle],
                        window_name: str,
                        output_path: Path,
                        suspect_tags: Dict[str, str] = None) -> None:
    """
    Create VaR and ES curves as functions of confidence level.
    
    Args:
        real_bundle: Real data bundle
        model_bundles: Dict of model bundles
        window_name: Window/scenario name
        output_path: Output path (without extension)
        suspect_tags: Dict of suspect scale tags per model
    """
    suspect_tags = suspect_tags or {}
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE_MULTI)
    
    # Confidence levels
    alphas = np.linspace(0.90, 0.99, 50)
    
    all_bundles = {'Real': real_bundle, **model_bundles}
    
    for idx, (name, bundle) in enumerate(all_bundles.items()):
        returns = bundle.returns
        
        # Convert to losses (negative returns)
        losses = -returns
        
        var_values = []
        es_values = []
        
        for alpha in alphas:
            # VaR at confidence level alpha
            var_val = np.percentile(losses, alpha * 100)
            var_values.append(var_val)
            
            # ES (Expected Shortfall) at confidence level alpha
            tail_losses = losses[losses >= var_val]
            if len(tail_losses) > 0:
                es_val = np.mean(tail_losses)
            else:
                es_val = var_val
            es_values.append(es_val)
        
        color = COLORS[idx % len(COLORS)]
        
        # Plot VaR curve
        ax1.plot(alphas, var_values, label=f'{name}', color=color, linewidth=2)
        
        # Plot ES curve
        ax2.plot(alphas, es_values, label=f'{name}', color=color, linewidth=2)
        
        # Mark specific confidence levels
        for alpha_mark in [0.95, 0.99]:
            var_mark = np.percentile(losses, alpha_mark * 100)
            es_mark = np.mean(losses[losses >= var_mark]) if len(losses[losses >= var_mark]) > 0 else var_mark
            
            ax1.scatter([alpha_mark], [var_mark], color=color, s=50, zorder=5)
            ax2.scatter([alpha_mark], [es_mark], color=color, s=50, zorder=5)
    
    # Format VaR plot
    ax1.set_title(f'Value at Risk (VaR) Curves - {window_name}')
    ax1.set_xlabel('Confidence Level α')
    ax1.set_ylabel('VaR (Loss magnitude)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0.90, 0.99)
    
    # Format ES plot
    ax2.set_title(f'Expected Shortfall (ES) Curves - {window_name}')
    ax2.set_xlabel('Confidence Level α')
    ax2.set_ylabel('ES (Expected loss beyond VaR)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0.90, 0.99)
    
    # Add suspect tags if any
    suspect_models = [name for name, tag in suspect_tags.items() if tag != "OK"]
    if suspect_models:
        fig.text(0.02, 0.02, f"⚠️ Suspect scale: {', '.join(suspect_models)}", 
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    
    plt.savefig(f"{output_path}.pdf", dpi=DPI, bbox_inches='tight')
    plt.savefig(f"{output_path}.png", dpi=DPI, bbox_inches='tight')
    plt.close()


@require_inverse_scaled_data
def create_exceedance_timeline(real_bundle: ReturnsBundle,
                              model_bundles: Dict[str, ReturnsBundle],
                              window_name: str,
                              output_path: Path,
                              suspect_tags: Dict[str, str] = None) -> None:
    """
    Create exceedance timeline for VaR breaches.
    
    Args:
        real_bundle: Real data bundle
        model_bundles: Dict of model bundles
        window_name: Window/scenario name
        output_path: Output path (without extension)
        suspect_tags: Dict of suspect scale tags per model
    """
    suspect_tags = suspect_tags or {}
    
    all_bundles = {'Real': real_bundle, **model_bundles}
    n_models = len(all_bundles)
    
    fig, axes = plt.subplots(n_models, 1, figsize=(12, 3*n_models))
    if n_models == 1:
        axes = [axes]
    
    confidence_levels = [0.95, 0.99]
    colors_conf = ['orange', 'red']
    
    # Use real data length as reference for all series
    real_length = len(real_bundle.returns)
    real_time_axis = np.arange(real_length)
    
    for idx, (name, bundle) in enumerate(all_bundles.items()):
        ax = axes[idx]
        returns = bundle.returns
        
        # Align all series to real data length for comparable breach analysis
        if name == 'Real':
            aligned_returns = returns
            time_axis = real_time_axis
        else:
            # Trim or pad model data to match real data length
            if len(returns) >= real_length:
                aligned_returns = returns[:real_length]
            else:
                # If model has fewer samples, pad with NaN (should not happen in practice)
                aligned_returns = np.full(real_length, np.nan)
                aligned_returns[:len(returns)] = returns
            time_axis = real_time_axis
        
        losses = -aligned_returns  # Convert to losses
        
        # Plot returns time series
        ax.plot(time_axis, aligned_returns, color='blue', alpha=0.6, linewidth=1, label='Returns')
        
        breach_counts = {}
        
        for conf_idx, alpha in enumerate(confidence_levels):
            # Calculate VaR threshold
            var_threshold = np.percentile(losses, alpha * 100)
            
            # Find breaches (returns below -VaR)
            breaches = aligned_returns < -var_threshold
            breach_times = time_axis[breaches]
            breach_values = aligned_returns[breaches]
            
            # Calculate expected breaches for this series length
            expected_breaches = len(aligned_returns) * (1 - alpha)
            
            # Plot breach markers
            if len(breach_times) > 0:
                ax.scatter(breach_times, breach_values, 
                          color=colors_conf[conf_idx], s=30, alpha=0.8,
                          label=f'VaR@{int(alpha*100)}% breaches ({len(breach_times)}/{expected_breaches:.1f})',
                          zorder=5)
            
            # Add horizontal line for -VaR threshold
            ax.axhline(-var_threshold, color=colors_conf[conf_idx], 
                      linestyle='--', alpha=0.7, linewidth=1)
            
            breach_counts[f'VaR{int(alpha*100)}'] = len(breach_times)
        
        # Format subplot
        title = f'{name} VaR Exceedances'
        suspect_tag = suspect_tags.get(name, "OK")
        title = add_suspect_scale_tag(title, suspect_tag)
        ax.set_title(title)
        
        ax.set_ylabel('Returns')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        if idx == len(all_bundles) - 1:
            ax.set_xlabel('Time')
    
    plt.suptitle(f'VaR Exceedance Timeline - {window_name}', fontsize=16)
    plt.tight_layout()
    
    plt.savefig(f"{output_path}.pdf", dpi=DPI, bbox_inches='tight')
    plt.savefig(f"{output_path}.png", dpi=DPI, bbox_inches='tight')
    plt.close()


@require_inverse_scaled_data
def create_density_ecdf_plots(real_bundle: ReturnsBundle,
                             model_bundles: Dict[str, ReturnsBundle],
                             window_name: str,
                             output_path: Path,
                             suspect_tags: Dict[str, str] = None) -> None:
    """
    Create comparative density and ECDF overlay plots.
    
    Args:
        real_bundle: Real data bundle
        model_bundles: Dict of model bundles
        window_name: Window/scenario name
        output_path: Output path (without extension)
        suspect_tags: Dict of suspect scale tags per model
    """
    suspect_tags = suspect_tags or {}
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE_MULTI)
    
    all_bundles = {'Real': real_bundle, **model_bundles}
    
    # Determine symmetric x-limits based on real data
    real_std = real_bundle.std
    x_limit = 8 * real_std
    x_range = np.linspace(-x_limit, x_limit, 1000)
    
    for idx, (name, bundle) in enumerate(all_bundles.items()):
        returns = bundle.returns
        color = COLORS[idx % len(COLORS)]
        
        # Density plot (KDE)
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(returns)
            density = kde(x_range)
            ax1.plot(x_range, density, label=name, color=color, linewidth=2)
        except Exception as e:
            # Fallback to histogram
            ax1.hist(returns, bins=50, alpha=0.3, density=True, 
                    label=f'{name} (hist)', color=color)
        
        # ECDF plot
        sorted_returns = np.sort(returns)
        ecdf_y = np.arange(1, len(sorted_returns) + 1) / len(sorted_returns)
        ax2.plot(sorted_returns, ecdf_y, label=name, color=color, linewidth=2)
    
    # Format density plot
    ax1.set_title('Probability Density Comparison')
    ax1.set_xlabel('Returns')
    ax1.set_ylabel('Density')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-x_limit, x_limit)
    
    # Format ECDF plot
    ax2.set_title('Empirical CDF Comparison')
    ax2.set_xlabel('Returns')
    ax2.set_ylabel('Cumulative Probability')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-x_limit, x_limit)
    ax2.set_ylim(0, 1)
    
    # Add sanity table
    create_sanity_table(ax1, all_bundles, suspect_tags)
    
    plt.suptitle(f'Distribution Comparison - {window_name}', fontsize=16)
    plt.tight_layout()
    
    plt.savefig(f"{output_path}.pdf", dpi=DPI, bbox_inches='tight')
    plt.savefig(f"{output_path}.png", dpi=DPI, bbox_inches='tight')
    plt.close()


def create_sanity_table(ax, bundles: Dict[str, ReturnsBundle], 
                       suspect_tags: Dict[str, str] = None) -> None:
    """
    Add a small sanity table to a plot.
    
    Args:
        ax: Matplotlib axis
        bundles: Dict of ReturnsBundle objects
        suspect_tags: Dict of suspect scale tags
    """
    suspect_tags = suspect_tags or {}
    
    # Create table data
    table_data = []
    for name, bundle in bundles.items():
        status = "✓" if suspect_tags.get(name, "OK") == "OK" else "⚠️"
        row = [
            name[:8],  # Truncate long names
            f"{bundle.mean:.4f}",
            f"{bundle.std:.4f}",
            f"{bundle.min:.3f}",
            f"{bundle.max:.3f}",
            f"{bundle.kurtosis:.1f}",
            status
        ]
        table_data.append(row)
    
    headers = ['Model', 'Mean', 'Std', 'Min', 'Max', 'Kurt', 'OK']
    
    # Position table in upper right corner
    table_text = "Sanity Check:\n" + "\n".join([
        " ".join(f"{h:>6}" for h in headers)
    ] + [
        " ".join(f"{cell:>6}" for cell in row) for row in table_data
    ])
    
    ax.text(0.98, 0.98, table_text, transform=ax.transAxes,
           verticalalignment='top', horizontalalignment='right',
           fontfamily='monospace', fontsize=8,
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))