#!/usr/bin/env python3
"""
GARCH(1,1) Model Implementation and Evaluation

This script implements a GARCH(1,1) model for volatility modeling and evaluation.
It serves as a classical baseline for comparison with modern generative models.

Author: Simin Ali
Thesis: Diffusion Models in Generative AI for Financial Data Synthesis and Risk Management
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import warnings
import os
import json
import argparse
import sys
from datetime import datetime
from arch import arch_model
from statsmodels.stats.diagnostic import acorr_ljungbox

# Add utils to path and import
import sys
import os
# Get the project root directory
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
utils_path = os.path.join(project_root, 'utils')
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

# Try to import utils modules directly
try:
    import metadata
    import units
    import plots
    import stats as stats_utils
    import risk
    import uncertainty
except ImportError as e:
    print(f"Error importing utils modules: {e}")
    print(f"Current sys.path: {sys.path}")
    print(f"Utils path: {utils_path}")
    print(f"Utils exists: {os.path.exists(utils_path)}")
    raise

warnings.filterwarnings('ignore')

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='GARCH(1,1) Model Evaluation')
    parser.add_argument('--units', choices=['percent', 'decimal'], 
                       help='Target units for returns (default: auto-detect)')
    parser.add_argument('--write-metadata', action='store_true',
                       help='Write metadata JSON to output directory')
    parser.add_argument('--outdir', type=str, default=None,
                       help='Output directory (default: ./runs/garch_simple/<timestamp>)')
    
    return parser.parse_args()

def setup_output_directory():
    """Setup output directory with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_outdir = f"./runs/garch_simple/{timestamp}"
    
    args = parse_arguments()
    outdir = args.outdir if args.outdir else default_outdir
    
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(os.path.join(outdir, "plots"), exist_ok=True)
    
    return outdir

def load_and_prepare_data():
    """Load and prepare S&P 500 data with robust parsing."""
    print("Loading S&P 500 data...")
    
    # Robust data file path handling
    data_path = os.getenv('SP500_DATA_PATH', "../data/sp500_data.csv")
    fallback_paths = [data_path, "data/sp500_data.csv", "../data/sp500_data.csv"]
    
    data = None
    for path in fallback_paths:
        if os.path.exists(path):
            try:
                data = pd.read_csv(path, index_col=0, parse_dates=True)
                print(f"Data loaded from: {path}")
                break
            except Exception as e:
                print(f"Failed to load from {path}: {e}")
                continue
    
    if data is None:
        raise FileNotFoundError(f"Could not find sp500_data.csv in paths: {fallback_paths}")
    
    # Strictly select Close column and coerce to numeric
    if 'Close' not in data.columns:
        raise ValueError("Close column not found in data")
    
    data = data[['Close']].copy()
    data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
    data = data.dropna(subset=['Close'])
    
    # Ensure time order is preserved
    data = data.sort_index()
    
    # Compute log returns (keep in decimal form for internal calculations)
    data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
    data = data.dropna()
    
    # Convert to percentage for numerical stability and consistency
    data['Returns_Pct'] = data['Log_Returns'] * 100
    
    # Apply unit standardization if requested
    args = parse_arguments()
    if args.units:
        data['Returns_Pct'] = units.ensure_units(data['Returns_Pct'], args.units)
        print(f"Returns standardized to {args.units} units")
    
    print(f"Data loaded: {len(data)} observations")
    print(f"Date range: {data.index.min()} to {data.index.max()}")
    print(f"Returns stats - Mean: {data['Returns_Pct'].mean():.4f}%, Std: {data['Returns_Pct'].std():.4f}%")
    
    return data

def split_data(data):
    """Split data chronologically into training and testing sets (80/20) without shuffling."""
    returns_pct = data['Returns_Pct'].dropna()
    
    # Split 80/20 without shuffling to preserve time order
    split_idx = int(len(returns_pct) * 0.8)
    train_returns = returns_pct[:split_idx]
    test_returns = returns_pct[split_idx:]
    
    print(f"Data split:")
    print(f"  Training: {len(train_returns)} observations")
    print(f"  Testing: {len(test_returns)} observations")
    print(f"  Training period: {train_returns.index[0]} to {train_returns.index[-1]}")
    print(f"  Test period: {test_returns.index[0]} to {test_returns.index[-1]}")
    
    return train_returns, test_returns

def fit_garch_model(train_returns, full_returns):
    """Fit GARCH(1,1) model using MLE with Student-t innovations, constrained to training window."""
    print("Fitting GARCH(1,1) model with Student-t innovations...")
    
    # Fit GARCH(1,1) model using arch library
    # Constant mean, Student-t innovations
    # Pass full_returns to establish timeline while constraining estimation to training window
    model = arch_model(
        full_returns, 
        vol='GARCH', 
        p=1, q=1, 
        dist='t',
        mean='Constant'
    )
    
    # Fit the model with last_obs constraint to training window
    fitted_model = model.fit(disp='off', last_obs=train_returns.index[-1])
    
    # Extract fitted parameters
    params = {
        'omega': fitted_model.params['omega'],
        'alpha': fitted_model.params['alpha[1]'],
        'beta': fitted_model.params['beta[1]'],
        'nu': fitted_model.params['nu'],
        'mu': fitted_model.params['mu'],
        'persistence': fitted_model.params['alpha[1]'] + fitted_model.params['beta[1]']
    }
    
    # Print full model summary
    print("\nGARCH(1,1) Model Summary:")
    print(fitted_model.summary())
    
    # Print concise parameter summary
    print(f"\nGARCH(1,1) parameters estimated:")
    print(f"  ω (constant): {params['omega']:.8f}")
    print(f"  α (ARCH): {params['alpha']:.6f}")
    print(f"  β (GARCH): {params['beta']:.6f}")
    print(f"  ν (degrees of freedom): {params['nu']:.2f}")
    print(f"  μ (constant mean): {params['mu']:.6f}")
    print(f"  Persistence (α+β): {params['persistence']:.6f}")
    print(f"  Model is {'stationary' if params['persistence'] < 1.0 else 'non-stationary'} (persistence {'<' if params['persistence'] < 1.0 else '>='} 1.0)")
    
    return fitted_model, params

def generate_garch_forecasts(model, test_returns, full_returns, params):
    """Generate GARCH forecasts aligned by date for the test window."""
    print("Generating GARCH forecasts...")
    
    # Generate 1-step-ahead forecasts over the entire series
    fcast = model.forecast(horizon=1, reindex=True)
    
    # Extract variance forecasts and align to test window dates
    # Use h.1 for 1-step-ahead variance forecasts
    variance_forecasts = fcast.variance['h.1']
    
    # Align forecasts to test window dates
    sigma_pct = np.sqrt(variance_forecasts.loc[test_returns.index])
    
    # Compute VaR threshold using fitted Student-t distribution and model mean
    # 5% left-tail VaR: mu + t_quantile * sigma
    var_threshold_pct = params['mu'] + stats.t.ppf(0.05, df=params['nu']) * sigma_pct
    
    print(f"Generated {len(sigma_pct)} volatility and VaR forecasts")
    print(f"Forecast period: {test_returns.index[0]} to {test_returns.index[-1]}")
    print(f"VaR threshold range: {var_threshold_pct.min():.4f}% to {var_threshold_pct.max():.4f}%")
    
    return sigma_pct, var_threshold_pct

def evaluate_garch_performance(test_returns, var_threshold_pct, sigma_pct, model):
    """Evaluate GARCH performance using only the test set."""
    print("Evaluating GARCH performance...")
    
    # Identify VaR violations in test set only
    violations = (test_returns < var_threshold_pct)
    hits = violations.to_numpy(dtype=int)
    
    # Basic statistics
    n_obs = len(test_returns)
    n_violations = np.sum(hits)
    violation_rate = n_violations / n_obs
    hit_rate = 1 - violation_rate
    
    # Residual diagnostics for test set only
    std_resid_test = (test_returns - model.params['mu']) / sigma_pct
    
    # Ljung-Box test for standardized residuals at reasonable lags
    print("Residual diagnostic - Ljung-Box test for autocorrelation:")
    for lag in [10, 20]:
        try:
            lb_df = acorr_ljungbox(std_resid_test, lags=lag, return_df=True)
            lb_stat = lb_df.iloc[-1]['lb_stat']
            lb_pvalue = lb_df.iloc[-1]['lb_pvalue']
            print(f"  Lag {lag}: statistic={lb_stat:.4f}, p-value={lb_pvalue:.4f}")
        except Exception as e:
            print(f"  Lag {lag}: Error computing Ljung-Box test: {e}")
    
    # VaR backtesting using only test set
    backtest_results = risk.var_backtest(test_returns.values, var_threshold_pct.values)
    
    # Compile evaluation results
    stats_dict = {
        'Total_Observations': n_obs,
        'Violations': n_violations,
        'Violation_Rate': violation_rate,
        'Hit_Rate': hit_rate,
        'Kupiec_PValue': backtest_results['kupiec_pvalue'],
        'Christoffersen_PValue': backtest_results['christoffersen_pvalue'],
        'ConditionalCoverage_PValue': backtest_results['conditional_coverage_pvalue']
    }
    
    print(f"GARCH Evaluation Results:")
    print(f"  Violations: {n_violations}/{n_obs} ({violation_rate:.4f})")
    print(f"  Expected rate: 0.050")
    print(f"  Hit rate: {hit_rate:.4f}")
    print(f"  Kupiec POF test: LR={backtest_results['kupiec_pvalue']:.4f}, p-value={backtest_results['kupiec_pvalue']:.4f}")
    print(f"  Christoffersen independence: LR={backtest_results['christoffersen_pvalue']:.4f}, p-value={backtest_results['christoffersen_pvalue']:.4f}")
    print(f"  Conditional coverage: LR={backtest_results['conditional_coverage_pvalue']:.4f}, p-value={backtest_results['conditional_coverage_pvalue']:.4f}")
    
    return stats_dict, hits

def save_results(test_returns, var_threshold_pct, sigma_pct, stats_dict, outdir):
    """Save results for comprehensive evaluation."""
    print("Saving GARCH results...")
    
    # Ensure all directories exist
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(os.path.join(outdir, "checkpoints"), exist_ok=True)
    
    # Save returns and forecasts (in percentage) - convert Series to numpy arrays for saving
    np.save(os.path.join(outdir, "garch_returns.npy"), test_returns.values)
    np.save(os.path.join(outdir, "garch_var_forecasts.npy"), var_threshold_pct.values)
    np.save(os.path.join(outdir, "garch_volatility_forecasts.npy"), sigma_pct.values)
    
    # Save statistics as JSON with native Python types
    metadata.save_json(stats_dict, os.path.join(outdir, "garch_stats.json"))
    
    print(f"GARCH results saved to: {outdir}")

def create_plots(test_returns, var_threshold_pct, sigma_pct, hits, model, train_returns, outdir):
    """Create evaluation plots using only test set data."""
    print("Creating GARCH evaluation plots...")
    
    # Ensure plot directory exists
    plots_dir = os.path.join(outdir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot 1: VaR backtest chart over test window
    plt.figure(figsize=(14, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(test_returns.index, test_returns.values, label='Returns (%)', alpha=0.7, linewidth=0.8)
    plt.plot(var_threshold_pct.index, var_threshold_pct.values, label='VaR Threshold (5%)', color='red', linewidth=1.5)
    # Use hits for scatter mask with positional indexing
    plt.scatter(test_returns.index[hits == 1], test_returns.values[hits == 1], 
               color='red', s=30, label='VaR Violations', zorder=5, alpha=0.8)
    plt.title('GARCH(1,1) VaR Backtest - Test Period Only')
    plt.ylabel('Returns (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    plt.plot(sigma_pct.index, sigma_pct.values, label='Conditional Volatility (%)', color='blue', linewidth=1.5)
    plt.title('GARCH(1,1) Conditional Volatility Forecasts - Test Period Only')
    plt.xlabel('Date')
    plt.ylabel('Volatility (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "garch_var_forecasts.pdf"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(plots_dir, "garch_var_forecasts.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: In-sample estimated conditional volatility (training period only)
    plt.figure(figsize=(12, 6))
    
    # Get in-sample conditional volatility from fitted model directly
    # This is already in standard deviation units, no need for square root or multiplication
    in_sample_vol_pct = model.conditional_volatility
    
    # Use only the training period for in-sample volatility
    train_vol = in_sample_vol_pct[:len(train_returns)]
    
    plt.plot(train_returns.index, train_vol, label='In-Sample Conditional Volatility (%)', 
             color='green', linewidth=1.5)
    plt.title('GARCH(1,1) In-Sample Conditional Volatility Estimates - Training Period Only')
    plt.xlabel('Date')
    plt.ylabel('Volatility (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "garch_in_sample_volatility.pdf"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(plots_dir, "garch_in_sample_volatility.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: KDE of test returns vs fitted Student-t distribution (diagnostic)
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    # KDE of test returns only
    from scipy.stats import gaussian_kde
    returns_data = test_returns.values
    kde = gaussian_kde(returns_data)
    x_range = np.linspace(returns_data.min(), returns_data.max(), 200)
    kde_curve = kde(x_range)
    
    plt.plot(x_range, kde_curve, 'b-', linewidth=2, label='KDE of Test Returns')
    
    # Fitted Student-t distribution with updated legend
    nu = model.params['nu']
    mu = model.params['mu']
    sigma = np.std(returns_data)
    
    t_dist = stats.t.pdf((x_range - mu) / sigma, df=nu) / sigma
    plt.plot(x_range, t_dist, 'r--', linewidth=2, label=f'Student-t (μ, ν) with sample σ (marginal approx)')
    
    plt.title('Test Returns Distribution: KDE vs Fitted Student-t')
    plt.xlabel('Returns (%)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # QQ-plot of standardized test residuals against fitted t distribution
    plt.subplot(1, 2, 2)
    # Calculate standardized residuals using time-varying volatility for test set only
    std_resid_test = (test_returns - model.params['mu']) / sigma_pct
    
    # Use string distribution name and shape parameters tuple for correct Student-t behavior
    stats.probplot(std_resid_test.to_numpy(), dist="t", sparams=(nu,), plot=plt)
    plt.title(f'QQ-Plot: Test Set Standardized Residuals vs Student-t(ν={nu:.1f})')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "garch_returns_distribution.pdf"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(plots_dir, "garch_returns_distribution.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Ensure all figures are released and no memory is held
    plt.close('all')
    
    print(f"GARCH plots saved to: {plots_dir}")

def main():
    """Main function to run GARCH evaluation."""
    print("GARCH(1,1) Model Evaluation")
    print("=" * 50)
    
    # Setup output directory and arguments
    outdir = setup_output_directory()
    args = parse_arguments()
    
    # Start timer for metadata
    timer_func = metadata.timer()
    
    try:
        # Load and prepare data
        data = load_and_prepare_data()
        
        # Split data chronologically
        train_returns, test_returns = split_data(data)
        
        # Get full returns for timeline establishment
        full_returns = data['Returns_Pct'].dropna()
        
        # Fit GARCH model with training window constraint
        model, params = fit_garch_model(train_returns, full_returns)
        
        # Save model checkpoint
        checkpoint_dir = os.path.join(outdir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        # Note: ARCH models don't have a standard save method, but we can save parameters
        metadata.save_json(params, os.path.join(checkpoint_dir, "garch_params.json"))
        print(f"Model parameters saved to: {checkpoint_dir}")
        
        # Generate forecasts with hardened indexing
        sigma_pct, var_threshold_pct = generate_garch_forecasts(model, test_returns, full_returns, params)
        
        # Compute predictive intervals using utils
        try:
            lo_interval, hi_interval = uncertainty.garch_predictive_interval(
                np.full_like(sigma_pct.values, params['mu']), 
                sigma_pct.values, 
                alpha=0.05
            )
            predictive_intervals = {
                'lower_5pct': lo_interval,
                'upper_95pct': hi_interval
            }
        except Exception as e:
            print(f"Warning: Could not compute predictive intervals: {e}")
            predictive_intervals = None
        
        # Evaluate performance with residual diagnostics using only test set
        stats_dict, hits = evaluate_garch_performance(test_returns, var_threshold_pct, sigma_pct, model)
        
        # Add enhanced metrics using utils (test set only)
        # VaR and ES at 1% and 5% for test set only
        var_1pct, es_1pct = risk.var_es(test_returns.values, alpha=0.01)
        var_5pct, es_5pct = risk.var_es(test_returns.values, alpha=0.05)
        
        # VaR backtest for test set only
        backtest_results = risk.var_backtest(test_returns.values, var_threshold_pct.values)
        
        # Enhanced statistics
        enhanced_stats = {
            'var_1pct': var_1pct,
            'es_1pct': es_1pct,
            'var_5pct': var_5pct,
            'es_5pct': es_5pct,
            'backtest_results': backtest_results
        }
        
        # Merge fitted parameters into stats_dict before saving
        stats_dict.update({
            'omega': params['omega'],
            'alpha': params['alpha'],
            'beta': params['beta'],
            'nu': params['nu'],
            'mu': params['mu'],
            'persistence': params['persistence']
        })
        
        # Add Train_Period and Test_Period keys for reproducibility
        stats_dict.update({
            'Train_Period': f"{train_returns.index[0]} to {train_returns.index[-1]}",
            'Test_Period': f"{test_returns.index[0]} to {test_returns.index[-1]}"
        })
        
        # Add enhanced stats
        stats_dict.update(enhanced_stats)
        
        # Save results
        save_results(test_returns, var_threshold_pct, sigma_pct, stats_dict, outdir)
        
        # Create plots using only test set data
        create_plots(test_returns, var_threshold_pct, sigma_pct, hits, model, train_returns, outdir)
        
        # Create enhanced plots using utils if requested
        if args.write_metadata:
            # Create QQ plot vs normal for test set only
            plots.qq_vs_normal(test_returns.values, title="GARCH Test Returns vs Normal")
            plt.savefig(os.path.join(outdir, "plots", "qq_vs_normal.pdf"), dpi=300, bbox_inches='tight')
            plt.savefig(os.path.join(outdir, "plots", "qq_vs_normal.png"), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Create ACF plots for test set only
            plots.acf_stem(test_returns.values, nlags=20, title="GARCH Test Returns ACF")
            plt.savefig(os.path.join(outdir, "plots", "acf_returns.pdf"), dpi=300, bbox_inches='tight')
            plt.savefig(os.path.join(outdir, "plots", "acf_returns.png"), dpi=300, bbox_inches='tight')
            plt.close()
            
            plots.acf_stem(np.abs(test_returns.values), nlags=20, title="GARCH Test Absolute Returns ACF")
            plt.savefig(os.path.join(outdir, "plots", "acf_abs_returns.pdf"), dpi=300, bbox_inches='tight')
            plt.savefig(os.path.join(outdir, "plots", "acf_abs_returns.png"), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Rolling volatility panel for test set only
            rolling_vol_fig, rolling_vol_axes = plots.rolling_vol_panel(
                test_returns, 
                [(sigma_pct.values, 'GARCH Volatility')], 
                window=20, 
                title="GARCH Rolling Volatility - Test Set Only"
            )
            rolling_vol_fig.savefig(os.path.join(outdir, "plots", "rolling_volatility.pdf"), dpi=300, bbox_inches='tight')
            rolling_vol_fig.savefig(os.path.join(outdir, "plots", "rolling_volatility.png"), dpi=300, bbox_inches='tight')
            plt.close(rolling_vol_fig)
        
        # Save enhanced metrics to CSV
        if args.write_metadata:
            # Save tail risk metrics for test set only
            tail_metrics_df = pd.DataFrame({
                'metric': ['VaR_1pct', 'ES_1pct', 'VaR_5pct', 'ES_5pct'],
                'value': [var_1pct, es_1pct, var_5pct, es_5pct]
            })
            tail_metrics_df.to_csv(os.path.join(outdir, "tail_metrics.csv"), index=False)
            
            # Save backtest results for test set only
            backtest_df = pd.DataFrame([backtest_results])
            backtest_df.to_csv(os.path.join(outdir, "backtests.csv"), index=False)
            
            # Save predictive intervals if available
            if predictive_intervals:
                intervals_df = pd.DataFrame({
                    'date': test_returns.index,
                    'lower_5pct': predictive_intervals['lower_5pct'],
                    'upper_95pct': predictive_intervals['upper_95pct']
                })
                intervals_df.to_csv(os.path.join(outdir, "predictive_intervals.csv"), index=False)
        
        # Collect and save metadata if requested
        if args.write_metadata:
            # Determine units used
            units_used = args.units if args.units else 'auto-detected'
            
            # Create metadata dictionary
            metadata_dict = {
                'timestamp': datetime.now().isoformat(),
                'script_name': 'garch_simple.py',
                'seed': 42,  # Default seed
                'units_used': units_used,
                'dataset_summary': metadata.dataset_summary(full_returns, 'S&P 500 Returns', units_used),
                'python_version': sys.version,
                'gpu_info': metadata.gpu_info(),
                'training_time_seconds': timer_func(),
                'model_parameters': {
                    'omega': params['omega'],
                    'alpha': params['alpha'],
                    'beta': params['beta'],
                    'nu': params['nu'],
                    'mu': params['mu'],
                    'persistence': params['persistence']
                },
                'train_test_split': {
                    'train_start': str(train_returns.index[0]),
                    'train_end': str(train_returns.index[-1]),
                    'test_start': str(test_returns.index[0]),
                    'test_end': str(test_returns.index[-1]),
                    'train_obs': len(train_returns),
                    'test_obs': len(test_returns)
                }
            }
            
            # Save metadata
            metadata.save_json(metadata_dict, os.path.join(outdir, "metadata.json"))
            
            # Write report notes
            report_notes = """# GARCH Model Evaluation Report Notes

## Train/Test Split
- **Training Period**: 80% of data (chronological, no shuffling)
- **Test Period**: 20% of data (chronological, no shuffling)
- **Validation**: None (GARCH uses MLE on training data only)

## Observables
- **Returns**: Log returns computed from S&P 500 closing prices
- **Volatility**: Conditional volatility from GARCH(1,1) model
- **VaR**: Value at Risk at 5% confidence level

## Rolling Volatility Definition
Rolling volatility is computed using a 20-day window standard deviation of returns, providing a smoothed view of volatility evolution over time.

## QQ Plot Interpretation
QQ plots compare the empirical distribution of returns against theoretical normal distribution. Points following the diagonal line indicate normality, while deviations suggest non-normal behavior.

## ACF Interpretation
Autocorrelation Function (ACF) measures the correlation between returns at different time lags. Significant correlations at lag 1 suggest momentum effects, while correlations in absolute returns indicate volatility clustering.

## Statistical Tests

### Kolmogorov-Smirnov (KS) Test
Tests whether two samples come from the same distribution. Low p-values indicate significant differences.

### Anderson-Darling (AD) Test
More sensitive to differences in the tails than KS test. Higher values indicate greater deviation from the reference distribution.

### Maximum Mean Discrepancy (MMD)
Measures the distance between two probability distributions using kernel methods. Lower values indicate more similar distributions.

### Value at Risk (VaR)
The maximum expected loss at a given confidence level. 5% VaR means 95% of the time, losses won't exceed this threshold.

### Expected Shortfall (ES)
The average loss when VaR is exceeded, providing a measure of tail risk beyond VaR.

### Kupiec Test
Tests whether the observed VaR violation rate matches the expected rate. Low p-values suggest VaR underestimation.

### Christoffersen Test
Tests whether VaR violations are independent over time. Low p-values suggest clustering of violations.

## Important Notes
- All evaluation metrics (VaR, ES, backtests) are computed using ONLY the test set
- Model is fitted on training data only to prevent data leakage
- Forecasts are generated for the test period only
- In-sample volatility plots show training period only
"""
            
            with open(os.path.join(outdir, "report_notes.md"), 'w') as f:
                f.write(report_notes)
        
        # Print final summary with all requested metrics
        n_obs = stats_dict['Total_Observations']
        n_violations = stats_dict['Violations']
        hit_rate = 1 - stats_dict['Violation_Rate']
        persistence = params['persistence']
        kupiec_p = stats_dict['Kupiec_PValue']
        christoff_p = stats_dict['Christoffersen_PValue']
        lrcc_p = stats_dict['ConditionalCoverage_PValue']
        
        print(f"\nFinal Summary: N={n_obs}, violations={n_violations}, hit_rate={hit_rate:.4f}, persistence={persistence:.4f}")
        print(f"Test p-values - Kupiec: {kupiec_p:.4f}, Christoffersen: {christoff_p:.4f}, Conditional Coverage: {lrcc_p:.4f}")
        
        if args.write_metadata:
            print(f"Enhanced evaluation results saved to: {outdir}")
        else:
            print("GARCH evaluation completed successfully!")
            print(f"Results saved in: {outdir}")
        
    except Exception as e:
        print(f"Error in GARCH evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
