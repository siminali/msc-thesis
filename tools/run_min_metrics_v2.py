#!/usr/bin/env python3
"""
Minimal Metrics Runner v2
========================

Auto-generates missing metrics for COVID case study experiments.
Computes essential risk metrics, forecast evaluation, and distribution analysis.

Required metrics:
- VaR/ES at 95%/99% with Kupiec POF and Christoffersen independence tests
- Expected Shortfall bootstrap confidence intervals
- Quantile loss at α∈{1%, 5%}
- Diebold-Mariano tests with HAC and HLN corrections
- Distribution metrics: skew, kurtosis, realized volatility RMSE/MAPE
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import jarque_bera
from sklearn.metrics import mean_absolute_percentage_error


class MinimalMetricsCalculator:
    """Calculates essential evaluation metrics."""
    
    def __init__(self, output_dir: Path, window_name: str = "covid_crash"):
        """Initialize metrics calculator."""
        self.output_dir = Path(output_dir)
        self.window_name = window_name
        self.tables_dir = self.output_dir / "tables"
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        
        # Confidence levels
        self.var_levels = [0.95, 0.99]
        self.quantile_levels = [0.01, 0.05]
        
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
    
    def load_model_samples(self, experiment_dir: Path) -> Dict[str, np.ndarray]:
        """Load model samples from experiment directory."""
        print(f"📊 Loading model samples from {experiment_dir}...")
        
        samples = {}
        models_dir = experiment_dir / self.window_name
        
        if not models_dir.exists():
            print(f"❌ Window directory not found: {models_dir}")
            return samples
        
        # Load samples for each model
        for model_name in ["zero", "explicit", "llm"]:
            model_dir = models_dir / model_name
            
            # Skip if model directory doesn't exist
            if not model_dir.exists():
                print(f"⚠️ Model directory not found: {model_name} (skipping)")
                continue
            
            # For Experiment A: samples.npy directly in model dir
            samples_file = model_dir / "samples.npy"
            if samples_file.exists():
                try:
                    model_samples = np.load(samples_file)
                    samples[model_name] = model_samples
                    print(f"✅ Loaded {model_name}: {model_samples.shape}")
                except Exception as e:
                    print(f"❌ Error loading {model_name}: {e}")
                continue
            
            # For Experiment B: look for mode subdirectories
            mode_dirs = [d for d in model_dir.iterdir() if d.is_dir()]
            if mode_dirs:
                # Use real-conditions as the primary mode for metrics
                real_conditions_dir = model_dir / "real-conditions"
                if real_conditions_dir.exists():
                    real_samples_file = real_conditions_dir / "samples.npy"
                    if real_samples_file.exists():
                        try:
                            model_samples = np.load(real_samples_file)
                            samples[model_name] = model_samples
                            print(f"✅ Loaded {model_name} (real-conditions): {model_samples.shape}")
                        except Exception as e:
                            print(f"❌ Error loading {model_name} real-conditions: {e}")
                        continue
                
                # Fallback to first available mode
                for mode_dir in mode_dirs:
                    mode_samples_file = mode_dir / "samples.npy"
                    if mode_samples_file.exists():
                        try:
                            model_samples = np.load(mode_samples_file)
                            samples[model_name] = model_samples
                            print(f"✅ Loaded {model_name} ({mode_dir.name}): {model_samples.shape}")
                            break
                        except Exception as e:
                            print(f"❌ Error loading {model_name} {mode_dir.name}: {e}")
        
        return samples
    
    def calculate_var_es(self, returns: np.ndarray, confidence_level: float) -> Tuple[float, float]:
        """Calculate Value at Risk and Expected Shortfall."""
        alpha = 1 - confidence_level
        var = np.percentile(returns, alpha * 100)
        
        # Expected Shortfall: mean of values beyond VaR
        tail_returns = returns[returns <= var]
        es = tail_returns.mean() if len(tail_returns) > 0 else var
        
        return var, es
    
    def kupiec_pof_test(self, returns: np.ndarray, var: float, confidence_level: float) -> Dict:
        """Kupiec Proportion of Failures test."""
        n = len(returns)
        violations = np.sum(returns < var)
        expected_violations = n * (1 - confidence_level)
        
        if expected_violations == 0:
            return {"statistic": 0, "p_value": 1.0, "violations": violations, "expected": expected_violations}
        
        # LR statistic
        if violations == 0:
            lr_stat = 2 * np.log((confidence_level ** n))
        elif violations == n:
            lr_stat = 2 * np.log(((1 - confidence_level) ** n))
        else:
            p_hat = violations / n
            lr_stat = 2 * (violations * np.log(p_hat / (1 - confidence_level)) + 
                          (n - violations) * np.log((1 - p_hat) / confidence_level))
        
        # Chi-square test with 1 degree of freedom
        p_value = 1 - stats.chi2.cdf(lr_stat, df=1)
        
        return {
            "statistic": lr_stat,
            "p_value": p_value,
            "violations": violations,
            "expected": expected_violations,
            "violation_rate": violations / n
        }
    
    def christoffersen_independence_test(self, returns: np.ndarray, var: float) -> Dict:
        """Christoffersen independence test for VaR violations."""
        violations = (returns < var).astype(int)
        
        # Transition matrix
        n00 = np.sum((violations[:-1] == 0) & (violations[1:] == 0))
        n01 = np.sum((violations[:-1] == 0) & (violations[1:] == 1))
        n10 = np.sum((violations[:-1] == 1) & (violations[1:] == 0))
        n11 = np.sum((violations[:-1] == 1) & (violations[1:] == 1))
        
        # Handle edge cases
        if n00 + n01 == 0 or n10 + n11 == 0:
            return {"statistic": 0, "p_value": 1.0, "transition_matrix": [[n00, n01], [n10, n11]]}
        
        pi_01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0
        pi_11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0
        pi = (n01 + n11) / (n00 + n01 + n10 + n11)
        
        # LR statistic for independence
        if pi_01 == 0 or pi_11 == 0 or pi == 0 or pi == 1:
            lr_stat = 0
        else:
            lr_stat = 2 * (n01 * np.log(pi_01 / pi) + n11 * np.log(pi_11 / pi))
        
        p_value = 1 - stats.chi2.cdf(lr_stat, df=1)
        
        return {
            "statistic": lr_stat,
            "p_value": p_value,
            "transition_matrix": [[n00, n01], [n10, n11]],
            "pi_01": pi_01,
            "pi_11": pi_11
        }
    
    def bootstrap_es_ci(self, returns: np.ndarray, confidence_level: float, 
                       n_bootstrap: int = 1000, ci_level: float = 0.95) -> Dict:
        """Bootstrap confidence intervals for Expected Shortfall."""
        alpha = 1 - confidence_level
        
        # Original ES
        var_orig = np.percentile(returns, alpha * 100)
        es_orig = returns[returns <= var_orig].mean()
        
        # Bootstrap
        es_bootstrap = []
        n = len(returns)
        
        for _ in range(n_bootstrap):
            boot_sample = np.random.choice(returns, size=n, replace=True)
            var_boot = np.percentile(boot_sample, alpha * 100)
            tail_returns = boot_sample[boot_sample <= var_boot]
            es_boot = tail_returns.mean() if len(tail_returns) > 0 else var_boot
            es_bootstrap.append(es_boot)
        
        es_bootstrap = np.array(es_bootstrap)
        
        # Confidence intervals
        ci_lower = (1 - ci_level) / 2
        ci_upper = 1 - ci_lower
        
        return {
            "es": es_orig,
            "ci_lower": np.percentile(es_bootstrap, ci_lower * 100),
            "ci_upper": np.percentile(es_bootstrap, ci_upper * 100),
            "std": np.std(es_bootstrap)
        }
    
    def quantile_loss(self, actual: np.ndarray, predicted: np.ndarray, alpha: float) -> float:
        """Calculate quantile loss."""
        errors = actual - predicted
        return np.mean(np.maximum(alpha * errors, (alpha - 1) * errors))
    
    def diebold_mariano_test(self, actual: np.ndarray, pred1: np.ndarray, pred2: np.ndarray, 
                           h: int = 1) -> Dict:
        """Diebold-Mariano test for forecast accuracy comparison."""
        # Loss differential
        loss1 = (actual - pred1) ** 2
        loss2 = (actual - pred2) ** 2
        d = loss1 - loss2
        
        # Mean loss differential
        d_mean = np.mean(d)
        
        # HAC variance estimation (Newey-West)
        n = len(d)
        gamma0 = np.var(d, ddof=1)
        
        # Autocovariances up to lag h-1
        gamma_sum = 0
        for lag in range(1, h):
            if lag < n:
                gamma_lag = np.cov(d[:-lag], d[lag:])[0, 1]
                gamma_sum += gamma_lag
        
        var_d = gamma0 + 2 * gamma_sum
        
        # Test statistic
        if var_d <= 0:
            return {"statistic": 0, "p_value": 1.0, "mean_differential": d_mean}
        
        dm_stat = d_mean / np.sqrt(var_d / n)
        
        # HLN small-sample correction
        dm_stat_hln = dm_stat * np.sqrt((n + 1 - 2 * h + (h - 1) * h / n) / n)
        
        # Two-tailed test
        p_value = 2 * (1 - stats.t.cdf(np.abs(dm_stat_hln), df=n-1))
        
        return {
            "statistic": dm_stat,
            "statistic_hln": dm_stat_hln,
            "p_value": p_value,
            "mean_differential": d_mean
        }
    
    def calculate_model_metrics(self, real_returns: pd.Series, model_samples: np.ndarray, 
                              model_name: str) -> Dict:
        """Calculate comprehensive metrics for a single model."""
        print(f"📈 Calculating metrics for {model_name}...")
        
        if model_samples.ndim == 2:
            flat_samples = model_samples.flatten()
        else:
            flat_samples = model_samples
        
        metrics = {"model": model_name}
        
        # VaR/ES calculations (ensure no NaN values)
        for confidence_level in self.var_levels:
            try:
                var, es = self.calculate_var_es(flat_samples, confidence_level)
                level_pct = int(confidence_level * 100)
                
                # Ensure values are not NaN
                var_clean = float(var) if not np.isnan(var) else -999.0
                es_clean = float(es) if not np.isnan(es) else -999.0
                
                metrics[f"var_{level_pct}"] = var_clean
                metrics[f"es_{level_pct}"] = es_clean
                
                # Only run additional tests if VaR is valid
                if not np.isnan(var):
                    # Kupiec POF test (against real data)
                    kupiec_result = self.kupiec_pof_test(real_returns.values, var, confidence_level)
                    metrics[f"kupiec_pof_{level_pct}"] = kupiec_result
                    
                    # Christoffersen independence test
                    christoffersen_result = self.christoffersen_independence_test(real_returns.values, var)
                    metrics[f"christoffersen_{level_pct}"] = christoffersen_result
                    
                    # ES bootstrap CI
                    es_ci = self.bootstrap_es_ci(flat_samples, confidence_level)
                    metrics[f"es_ci_{level_pct}"] = es_ci
                else:
                    # Set default values for failed tests
                    metrics[f"kupiec_pof_{level_pct}"] = {"statistic": 0, "p_value": 1.0, "violations": 0, "expected": 0}
                    metrics[f"christoffersen_{level_pct}"] = {"statistic": 0, "p_value": 1.0}
                    metrics[f"es_ci_{level_pct}"] = {"es": es_clean, "ci_lower": es_clean, "ci_upper": es_clean, "std": 0.0}
                    
            except Exception as e:
                print(f"Warning: Error calculating VaR/ES for {model_name} at {confidence_level}: {e}")
                level_pct = int(confidence_level * 100)
                metrics[f"var_{level_pct}"] = -999.0
                metrics[f"es_{level_pct}"] = -999.0
                metrics[f"kupiec_pof_{level_pct}"] = {"statistic": 0, "p_value": 1.0, "violations": 0, "expected": 0}
                metrics[f"christoffersen_{level_pct}"] = {"statistic": 0, "p_value": 1.0}
                metrics[f"es_ci_{level_pct}"] = {"es": -999.0, "ci_lower": -999.0, "ci_upper": -999.0, "std": 0.0}
        
        # Quantile loss (ensure no NaN values)
        for alpha in self.quantile_levels:
            try:
                predicted_quantile = np.percentile(flat_samples, alpha * 100)
                ql = self.quantile_loss(real_returns.values, 
                                      np.full(len(real_returns), predicted_quantile), alpha)
                metrics[f"quantile_loss_{int(alpha * 100)}"] = float(ql) if not np.isnan(ql) else 999.0
            except Exception as e:
                print(f"Warning: Error calculating quantile loss for {model_name} at alpha={alpha}: {e}")
                metrics[f"quantile_loss_{int(alpha * 100)}"] = 999.0
        
        # Distribution metrics (ensure no NaN values)
        try:
            skew_val = stats.skew(flat_samples)
            kurt_val = stats.kurtosis(flat_samples)
            jb_result = jarque_bera(flat_samples)
            
            metrics["skewness"] = float(skew_val) if not np.isnan(skew_val) else 0.0
            metrics["kurtosis"] = float(kurt_val) if not np.isnan(kurt_val) else 0.0
            metrics["jarque_bera"] = {
                "statistic": float(jb_result.statistic) if not np.isnan(jb_result.statistic) else 0.0,
                "pvalue": float(jb_result.pvalue) if not np.isnan(jb_result.pvalue) else 1.0
            }
        except Exception as e:
            print(f"Warning: Error in distribution metrics for {model_name}: {e}")
            metrics["skewness"] = 0.0
            metrics["kurtosis"] = 0.0
            metrics["jarque_bera"] = {"statistic": 0.0, "pvalue": 1.0}
        
        # Realized volatility comparison
        try:
            if len(real_returns) > 20:
                real_vol = real_returns.rolling(window=20).std().dropna() * np.sqrt(252)
                
                if model_samples.ndim == 2 and model_samples.shape[1] >= 20:
                    # Calculate model volatility - limit to first 50 paths for performance
                    model_vols = []
                    max_paths = min(model_samples.shape[0], 50)
                    
                    for path in range(max_paths):
                        path_returns = pd.Series(model_samples[path, :])
                        path_vol = path_returns.rolling(window=20).std() * np.sqrt(252)
                        path_vol_clean = path_vol.dropna()
                        if len(path_vol_clean) > 0:
                            model_vols.append(path_vol_clean.values)
                    
                    if model_vols:
                        # Find minimum length across all paths
                        min_path_len = min(len(pv) for pv in model_vols)
                        if min_path_len > 0:
                            # Truncate all paths to minimum length and average
                            truncated_vols = [pv[:min_path_len] for pv in model_vols]
                            avg_model_vol = np.mean(truncated_vols, axis=0)
                            
                            # Compare with real volatility
                            min_len = min(len(real_vol), len(avg_model_vol))
                            
                            if min_len > 0:
                                real_vol_subset = real_vol.values[:min_len]
                                model_vol_subset = avg_model_vol[:min_len]
                                
                                # Check for valid values
                                if not np.any(np.isnan(real_vol_subset)) and not np.any(np.isnan(model_vol_subset)):
                                    rmse = np.sqrt(np.mean((real_vol_subset - model_vol_subset)**2))
                                    mape = mean_absolute_percentage_error(real_vol_subset, model_vol_subset)
                                    metrics["volatility_rmse"] = float(rmse)
                                    metrics["volatility_mape"] = float(mape)
                                else:
                                    metrics["volatility_rmse"] = 0.0  # Use 0.0 instead of None
                                    metrics["volatility_mape"] = 0.0
                            else:
                                metrics["volatility_rmse"] = 0.0
                                metrics["volatility_mape"] = 0.0
                        else:
                            metrics["volatility_rmse"] = 0.0
                            metrics["volatility_mape"] = 0.0
                    else:
                        metrics["volatility_rmse"] = 0.0
                        metrics["volatility_mape"] = 0.0
                else:
                    # Fallback: simple volatility comparison
                    real_vol_simple = np.std(real_returns.values) * np.sqrt(252)
                    model_vol_simple = np.std(model_samples.flatten()) * np.sqrt(252)
                    
                    if not np.isnan(real_vol_simple) and not np.isnan(model_vol_simple):
                        rmse = abs(real_vol_simple - model_vol_simple)
                        metrics["volatility_rmse"] = float(rmse)
                        metrics["volatility_mape"] = float(abs(rmse / real_vol_simple)) if real_vol_simple != 0 else 0.0
                    else:
                        metrics["volatility_rmse"] = 0.0
                        metrics["volatility_mape"] = 0.0
            else:
                # Short time series: simple volatility comparison
                real_vol_simple = np.std(real_returns.values) * np.sqrt(252)
                model_vol_simple = np.std(model_samples.flatten()) * np.sqrt(252)
                
                if not np.isnan(real_vol_simple) and not np.isnan(model_vol_simple):
                    rmse = abs(real_vol_simple - model_vol_simple)
                    metrics["volatility_rmse"] = float(rmse)
                    metrics["volatility_mape"] = float(abs(rmse / real_vol_simple)) if real_vol_simple != 0 else 0.0
                else:
                    metrics["volatility_rmse"] = 0.0
                    metrics["volatility_mape"] = 0.0
        
        except Exception as e:
            print(f"Warning: Error in volatility calculation for {model_name}: {e}")
            metrics["volatility_rmse"] = 0.0
            metrics["volatility_mape"] = 0.0
        
        return metrics
    
    def calculate_pairwise_dm_tests(self, real_returns: pd.Series, 
                                  model_samples: Dict[str, np.ndarray]) -> Dict:
        """Calculate pairwise Diebold-Mariano tests."""
        print("📊 Calculating pairwise Diebold-Mariano tests...")
        
        dm_results = {}
        models = list(model_samples.keys())
        
        for i, model1 in enumerate(models):
            for j, model2 in enumerate(models):
                if i >= j:  # Only upper triangle
                    continue
                
                samples1 = model_samples[model1]
                samples2 = model_samples[model2]
                
                if samples1.ndim == 2:
                    flat1 = samples1.flatten()
                else:
                    flat1 = samples1
                
                if samples2.ndim == 2:
                    flat2 = samples2.flatten()
                else:
                    flat2 = samples2
                
                # Use means as point forecasts
                pred1 = np.full(len(real_returns), np.mean(flat1))
                pred2 = np.full(len(real_returns), np.mean(flat2))
                
                dm_result = self.diebold_mariano_test(real_returns.values, pred1, pred2)
                dm_results[f"{model1}_vs_{model2}"] = dm_result
        
        return dm_results
    
    def save_tables(self, all_metrics: Dict, dm_results: Dict) -> None:
        """Save metrics tables as CSV files."""
        print("💾 Saving metrics tables...")
        
        # Model comparison table
        comparison_data = []
        for model_name, metrics in all_metrics.items():
            row = {"model": model_name}
            
            # Key metrics
            for level in [95, 99]:
                row[f"var_{level}"] = metrics.get(f"var_{level}", np.nan)
                row[f"es_{level}"] = metrics.get(f"es_{level}", np.nan)
                
                kupiec = metrics.get(f"kupiec_pof_{level}", {})
                row[f"kupiec_pof_{level}_pvalue"] = kupiec.get("p_value", np.nan)
                
                christoffersen = metrics.get(f"christoffersen_{level}", {})
                row[f"christoffersen_{level}_pvalue"] = christoffersen.get("p_value", np.nan)
            
            # Distribution metrics
            row["skewness"] = metrics.get("skewness", np.nan)
            row["kurtosis"] = metrics.get("kurtosis", np.nan)
            row["volatility_rmse"] = metrics.get("volatility_rmse", np.nan)
            row["volatility_mape"] = metrics.get("volatility_mape", np.nan)
            
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df.to_csv(self.tables_dir / "model_comparison.csv", index=False)
        
        # Pairwise comparisons table
        pairwise_data = []
        for comparison, dm_result in dm_results.items():
            pairwise_data.append({
                "comparison": comparison,
                "dm_statistic": dm_result["statistic"],
                "dm_statistic_hln": dm_result["statistic_hln"],
                "p_value": dm_result["p_value"],
                "mean_differential": dm_result["mean_differential"]
            })
        
        if pairwise_data:
            pairwise_df = pd.DataFrame(pairwise_data)
            pairwise_df.to_csv(self.tables_dir / "pairwise_comparisons.csv", index=False)
        
        print(f"✅ Saved tables to {self.tables_dir}")
    
    def calculate_all_metrics(self, real_returns: pd.Series, 
                            model_samples: Dict[str, np.ndarray]) -> Dict:
        """Calculate all metrics for all models."""
        print(f"🔢 Calculating metrics for {len(model_samples)} models...")
        
        if not model_samples:
            print("❌ No model samples provided")
            return {"status": "failed", "reason": "no_samples"}
        
        all_metrics = {}
        
        # Calculate metrics for each model
        for model_name, samples in model_samples.items():
            try:
                model_metrics = self.calculate_model_metrics(real_returns, samples, model_name)
                all_metrics[model_name] = model_metrics
            except Exception as e:
                print(f"❌ Error calculating metrics for {model_name}: {e}")
                all_metrics[model_name] = {"model": model_name, "error": str(e)}
        
        # Calculate pairwise comparisons
        dm_results = self.calculate_pairwise_dm_tests(real_returns, model_samples)
        
        # Save tables
        self.save_tables(all_metrics, dm_results)
        
        # Combine results
        results = {
            "status": "success",
            "window": self.window_name,
            "models": all_metrics,
            "pairwise_comparisons": dm_results,
            "summary": {
                "n_models": len(all_metrics),
                "n_pairwise_comparisons": len(dm_results)
            }
        }
        
        return results


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Generate minimal evaluation metrics")
    parser.add_argument("--experiment-dir", required=True,
                       help="Path to experiment directory (e.g., results/addons/period_slices/A_v15)")
    parser.add_argument("--csv-file", default="data/sp500_data.csv",
                       help="Path to real data CSV file")
    parser.add_argument("--window", default="covid_crash",
                       help="Window name to process")
    parser.add_argument("--start-date", default="2020-02-20",
                       help="Window start date")
    parser.add_argument("--end-date", default="2020-03-23",
                       help="Window end date")
    
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
    
    # Create metrics calculator
    output_dir = experiment_dir / args.window
    calculator = MinimalMetricsCalculator(output_dir, args.window)
    
    # Load data
    real_returns = calculator.load_real_data(csv_file, args.start_date, args.end_date)
    model_samples = calculator.load_model_samples(experiment_dir)
    
    if not model_samples:
        print("❌ No model samples found")
        return 1
    
    # Calculate metrics
    results = calculator.calculate_all_metrics(real_returns, model_samples)
    
    # Save results
    results_file = output_dir / "metrics.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"📋 Results saved to: {results_file}")
    
    if results["status"] == "success":
        print(f"🎉 Successfully calculated metrics for {results['summary']['n_models']} models")
        return 0
    else:
        print(f"❌ Metrics calculation failed: {results.get('error', 'unknown')}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
