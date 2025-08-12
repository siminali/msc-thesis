#!/usr/bin/env python3
"""
Add VaR Backtesting for Baseline Models
GARCH, DDPM, and TimeGrad

Author: Simin Ali
Supervisor: Dr Mikael Mieskolainen
Institution: Imperial College London
"""

import numpy as np
import pandas as pd
import json
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def calculate_var_backtest(real_data, synthetic_data, model_name, confidence_levels=[0.01, 0.05]):
    """Calculate VaR backtesting metrics for baseline models."""
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
            'Model': model_name,
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
    """Add VaR backtesting for all baseline models."""
    print("⚠️ Adding VaR Backtesting for Baseline Models...")
    
    try:
        # Load real data
        print("📊 Loading real data...")
        data = pd.read_csv("data/sp500_data.csv", index_col=0, parse_dates=True)
        if data.index[0] == 'Ticker':
            data = data.iloc[1:]
        data.index = pd.to_datetime(data.index)
        returns = np.log(data['Close'] / data['Close'].shift(1)).dropna().values * 100
        
        print(f"✅ Real data: {len(returns)} observations")
        
        # Load comprehensive evaluation results
        with open("results/comprehensive_evaluation/evaluation_results.json", 'r') as f:
            eval_data = json.load(f)
        
        # Initialize var_backtest if it doesn't exist
        if 'var_backtest' not in eval_data:
            eval_data['var_backtest'] = []
        
        # Add VaR backtesting for each baseline model
        baseline_models = ['GARCH', 'DDPM', 'TimeGrad']
        
        for model_name in baseline_models:
            print(f"🧮 Calculating VaR backtesting for {model_name}...")
            
            try:
                # Load synthetic data for the model
                if model_name == 'GARCH':
                    synthetic_data = np.load("results/garch_returns.npy").flatten()
                elif model_name == 'DDPM':
                    synthetic_data = np.load("results/ddpm_returns.npy").flatten()
                elif model_name == 'TimeGrad':
                    synthetic_data = np.load("results/timegrad_returns.npy").flatten()
                
                # Calculate VaR backtesting
                backtest_results = calculate_var_backtest(returns, synthetic_data, model_name)
                
                # Add to evaluation data
                for backtest in backtest_results:
                    eval_data['var_backtest'].append(backtest)
                
                print(f"✅ {model_name} VaR backtesting completed")
                
                # Show summary
                for backtest in backtest_results:
                    alpha = backtest['Confidence_Level']
                    var_est = backtest['VaR_Estimate']
                    violations = backtest['Violations']
                    rate = backtest['Violation_Rate']
                    expected = backtest['Expected_Rate']
                    kupiec_p = backtest['Kupiec_Test_pvalue']
                    
                    print(f"   {alpha*100}% VaR: {var_est:.4f}")
                    print(f"     Violations: {violations}/{backtest['Total_Observations']} ({rate:.4f})")
                    print(f"     Expected: {expected:.4f}")
                    print(f"     Kupiec p-value: {kupiec_p:.4f}")
                
            except Exception as e:
                print(f"❌ Error with {model_name}: {e}")
        
        # Save updated results
        print("💾 Saving updated evaluation results...")
        with open("results/comprehensive_evaluation/evaluation_results.json", 'w') as f:
            json.dump(eval_data, f, indent=2)
        
        print("✅ VaR backtesting for all baseline models completed!")
        
        # Show complete comparison
        print("\n🏆 Complete Model Performance Ranking with VaR Backtesting:")
        tests = eval_data['distribution_tests']
        sorted_models = sorted(tests, key=lambda x: x['KS_Statistic'])
        for i, model in enumerate(sorted_models):
            print(f"{i+1}. {model['Model']}: KS={model['KS_Statistic']:.4f} (p-value={model['KS_pvalue']:.4f})")
        
        # Show VaR backtesting summary for all models
        print("\n⚠️ VaR Backtesting Summary for All Models:")
        var_backtests = eval_data['var_backtest']
        models_with_backtest = list(set([b['Model'] for b in var_backtests]))
        
        for model in models_with_backtest:
            model_backtests = [b for b in var_backtests if b['Model'] == model]
            print(f"\n{model}:")
            for backtest in model_backtests:
                alpha = backtest['Confidence_Level']
                var_est = backtest['VaR_Estimate']
                violations = backtest['Violations']
                rate = backtest['Violation_Rate']
                expected = backtest['Expected_Rate']
                print(f"  {alpha*100}% VaR: {var_est:.4f}, Violations: {violations}/{backtest['Total_Observations']} ({rate:.4f}), Expected: {expected:.4f}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

