#!/usr/bin/env python3
"""
Complete LLM-Conditioned Model Evaluation
Calculate missing metrics: Tail Risk, Volatility Dynamics, Anderson-Darling, MMD, VaR Backtesting

Author: Simin Ali
Supervisor: Dr Mikael Mieskolainen
Institution: Imperial College London
"""

import numpy as np
import pandas as pd
import json
from scipy import stats
from scipy.stats import anderson
import warnings
warnings.filterwarnings('ignore')

def calculate_tail_risk_metrics(data):
    """Calculate VaR and Expected Shortfall at different confidence levels."""
    # Sort data for percentile calculations
    sorted_data = np.sort(data)
    n = len(sorted_data)
    
    # Calculate VaR and ES at different confidence levels
    var_1 = np.percentile(data, 1)
    var_5 = np.percentile(data, 5)
    var_95 = np.percentile(data, 95)
    var_99 = np.percentile(data, 99)
    
    # Calculate Expected Shortfall (average of tail beyond VaR)
    es_1 = np.mean(sorted_data[:int(0.01 * n)])
    es_5 = np.mean(sorted_data[:int(0.05 * n)])
    es_95 = np.mean(sorted_data[int(0.95 * n):])
    es_99 = np.mean(sorted_data[int(0.99 * n):])
    
    return {
        'VaR_1%': float(var_1),
        'ES_1%': float(es_1),
        'VaR_5%': float(var_5),
        'ES_5%': float(es_5),
        'VaR_95%': float(var_95),
        'ES_95%': float(es_95),
        'VaR_99%': float(var_99),
        'ES_99%': float(es_99)
    }

def calculate_volatility_metrics(data):
    """Calculate volatility dynamics metrics."""
    # Convert to pandas series for rolling calculations
    series = pd.Series(data)
    
    # Rolling volatility (20-day window)
    rolling_vol = series.rolling(window=20).std().dropna()
    
    # Volatility ACF (autocorrelation of squared returns)
    squared_returns = series ** 2
    vol_acf = squared_returns.autocorr(lag=1)
    
    # Volatility persistence (AR(1) coefficient)
    if len(rolling_vol) > 1:
        vol_persistence = rolling_vol.autocorr(lag=1)
    else:
        vol_persistence = 0.0
    
    # Mean volatility
    mean_vol = float(rolling_vol.mean())
    
    # Volatility of volatility
    vol_of_vol = float(rolling_vol.std())
    
    return {
        'Volatility_ACF': float(vol_acf) if not np.isnan(vol_acf) else 0.0,
        'Volatility_Persistence': float(vol_persistence) if not np.isnan(vol_persistence) else 0.0,
        'Mean_Volatility': mean_vol,
        'Volatility_of_Volatility': vol_of_vol
    }

def calculate_anderson_darling(data):
    """Calculate Anderson-Darling test statistic."""
    try:
        # Anderson-Darling test for normality
        result = anderson(data)
        return float(result.statistic)
    except:
        return 0.0

def calculate_mmd(data1, data2):
    """Calculate Maximum Mean Discrepancy between two datasets."""
    try:
        # Simple MMD approximation using kernel trick
        # Using RBF kernel with sigma = 1
        def rbf_kernel(x, y, sigma=1.0):
            return np.exp(-np.sum((x - y) ** 2) / (2 * sigma ** 2))
        
        # Sample points for MMD calculation (to avoid memory issues)
        n_samples = min(1000, len(data1), len(data2))
        sample1 = np.random.choice(data1, n_samples, replace=False)
        sample2 = np.random.choice(data2, n_samples, replace=False)
        
        # Calculate MMD
        mmd = 0.0
        for i in range(n_samples):
            for j in range(n_samples):
                if i != j:
                    mmd += rbf_kernel(sample1[i], sample1[j]) + rbf_kernel(sample2[i], sample2[j]) - 2 * rbf_kernel(sample1[i], sample2[j])
        
        mmd = mmd / (n_samples * (n_samples - 1))
        return float(max(0, mmd))  # Ensure non-negative
    except:
        return 0.0

def calculate_var_backtest(real_data, synthetic_data, confidence_levels=[0.01, 0.05]):
    """Calculate VaR backtesting metrics for multiple confidence levels."""
    backtest_results = []
    
    for alpha in confidence_levels:
        # Calculate VaR from synthetic data
        var_alpha = np.percentile(synthetic_data, alpha * 100)
        
        # Count violations in real data
        violations = np.sum(real_data < var_alpha)
        total_observations = len(real_data)
        violation_rate = violations / total_observations
        
        # Expected violation rate
        expected_rate = alpha
        
        # Kupiec test (unconditional coverage)
        if violations > 0:
            # Likelihood ratio test statistic
            lr_stat = -2 * (np.log(((1 - expected_rate) ** (total_observations - violations)) * 
                                   (expected_rate ** violations)) - 
                           np.log(((1 - violation_rate) ** (total_observations - violations)) * 
                                   (violation_rate ** violations)))
            lr_pvalue = 1 - stats.chi2.cdf(lr_stat, 1)
        else:
            lr_stat = float('inf')
            lr_pvalue = 0.0
        
        # Christoffersen test (independence of violations)
        # Create violation sequence
        violation_sequence = (real_data < var_alpha).astype(int)
        
        # Count transitions
        n00 = n01 = n10 = n11 = 0
        for i in range(len(violation_sequence) - 1):
            if violation_sequence[i] == 0 and violation_sequence[i+1] == 0:
                n00 += 1
            elif violation_sequence[i] == 0 and violation_sequence[i+1] == 1:
                n01 += 1
            elif violation_sequence[i] == 1 and violation_sequence[i+1] == 0:
                n10 += 1
            elif violation_sequence[i] == 1 and violation_sequence[i+1] == 1:
                n11 += 1
        
        # Independence test statistic
        if n00 + n01 > 0 and n10 + n11 > 0:
            p01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0
            p11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0
            
            # Likelihood ratio for independence
            if p01 > 0 and p11 > 0 and p01 < 1 and p11 < 1:
                lr_indep = -2 * (np.log(((1 - violation_rate) ** (n00 + n10)) * 
                                        (violation_rate ** (n01 + n11))) - 
                                np.log(((1 - p01) ** n00) * (p01 ** n01) * 
                                       ((1 - p11) ** n10) * (p11 ** n11)))
                lr_indep_pvalue = 1 - stats.chi2.cdf(lr_indep, 1)
            else:
                lr_indep = float('inf')
                lr_indep_pvalue = 0.0
        else:
            lr_indep = float('inf')
            lr_indep_pvalue = 0.0
        
        # Combined test (Christoffersen)
        lr_combined = lr_stat + lr_indep
        lr_combined_pvalue = 1 - stats.chi2.cdf(lr_combined, 2)
        
        backtest_results.append({
            'Confidence_Level': alpha,
            'VaR_Estimate': float(var_alpha),
            'Violations': int(violations),
            'Total_Observations': int(total_observations),
            'Violation_Rate': float(violation_rate),
            'Expected_Rate': float(expected_rate),
            'Kupiec_Test_Stat': float(lr_stat),
            'Kupiec_Test_pvalue': float(lr_pvalue),
            'Independence_Test_Stat': float(lr_indep),
            'Independence_Test_pvalue': float(lr_indep_pvalue),
            'Combined_Test_Stat': float(lr_combined),
            'Combined_Test_pvalue': float(lr_combined_pvalue)
        })
    
    return backtest_results

def main():
    """Complete the LLM-conditioned model evaluation."""
    print("🔍 Completing LLM-Conditioned Model Evaluation...")
    
    try:
        # Load real data for comparison
        print("📊 Loading real data...")
        data = pd.read_csv("data/sp500_data.csv", index_col=0, parse_dates=True)
        if data.index[0] == 'Ticker':
            data = data.iloc[1:]
        data.index = pd.to_datetime(data.index)
        returns = np.log(data['Close'] / data['Close'].shift(1)).dropna().values * 100
        
        # Load LLM-conditioned synthetic data
        print("📊 Loading LLM-conditioned synthetic data...")
        llm_data = np.load("results/llm_conditioned_returns.npy").flatten()
        
        print(f"✅ Real data: {len(returns)} observations")
        print(f"✅ LLM-conditioned data: {len(llm_data)} observations")
        
        # Calculate missing metrics
        print("🧮 Calculating tail risk metrics...")
        tail_metrics = calculate_tail_risk_metrics(llm_data)
        
        print("📊 Calculating volatility metrics...")
        volatility_metrics = calculate_volatility_metrics(llm_data)
        
        print("🧪 Calculating Anderson-Darling statistic...")
        anderson_stat = calculate_anderson_darling(llm_data)
        
        print("📏 Calculating MMD...")
        mmd_value = calculate_mmd(returns, llm_data)
        
        print("⚠️ Calculating VaR backtesting...")
        var_backtest_results = calculate_var_backtest(returns, llm_data)
        
        # Update the comprehensive evaluation results
        print("💾 Updating evaluation results...")
        with open("results/comprehensive_evaluation/evaluation_results.json", 'r') as f:
            data = json.load(f)
        
        # Add LLM-conditioned to tail metrics
        llm_tail = {"Model": "LLM-Conditioned", **tail_metrics}
        data['tail_metrics'].append(llm_tail)
        
        # Add LLM-conditioned to volatility metrics
        llm_vol = {"Model": "LLM-Conditioned", **volatility_metrics}
        data['volatility_metrics'].append(llm_vol)
        
        # Update distribution tests with proper values
        for test in data['distribution_tests']:
            if test['Model'] == 'LLM-Conditioned':
                test['Anderson_Darling_Stat'] = anderson_stat
                test['MMD'] = mmd_value
                break
        
        # Add VaR backtesting results
        if 'var_backtest' not in data:
            data['var_backtest'] = []
        
        # Add LLM-conditioned backtest results
        for backtest in var_backtest_results:
            llm_backtest = {"Model": "LLM-Conditioned", **backtest}
            data['var_backtest'].append(llm_backtest)
        
        # Save updated results
        with open("results/comprehensive_evaluation/evaluation_results.json", 'w') as f:
            json.dump(data, f, indent=2)
        
        print("✅ LLM-conditioned evaluation completed!")
        print("\n📊 New Metrics Added:")
        print(f"   Tail Risk: VaR 1% = {tail_metrics['VaR_1%']:.4f}")
        print(f"   Volatility ACF: {volatility_metrics['Volatility_ACF']:.4f}")
        print(f"   Anderson-Darling: {anderson_stat:.4f}")
        print(f"   MMD: {mmd_value:.4f}")
        print(f"   VaR Backtesting: {len(var_backtest_results)} confidence levels")
        
        # Show VaR backtesting summary
        print("\n⚠️ VaR Backtesting Results:")
        for backtest in var_backtest_results:
            alpha = backtest['Confidence_Level']
            var_est = backtest['VaR_Estimate']
            violations = backtest['Violations']
            rate = backtest['Violation_Rate']
            expected = backtest['Expected_Rate']
            kupiec_p = backtest['Kupiec_Test_pvalue']
            combined_p = backtest['Combined_Test_pvalue']
            
            print(f"   {alpha*100}% VaR: {var_est:.4f}")
            print(f"     Violations: {violations}/{backtest['Total_Observations']} ({rate:.4f})")
            print(f"     Expected: {expected:.4f}")
            print(f"     Kupiec p-value: {kupiec_p:.4f}")
            print(f"     Combined p-value: {combined_p:.4f}")
        
        # Show complete comparison
        print("\n🏆 Complete Model Performance Ranking:")
        tests = data['distribution_tests']
        sorted_models = sorted(tests, key=lambda x: x['KS_Statistic'])
        for i, model in enumerate(sorted_models):
            print(f"{i+1}. {model['Model']}: KS={model['KS_Statistic']:.4f} (p-value={model['KS_pvalue']:.4f})")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
