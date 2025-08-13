#!/usr/bin/env python3
"""
Run Enhanced Evaluation Framework

This script runs the enhanced evaluation framework to compute:
- Christoffersen independence tests for VaR backtesting
- Ljung-Box tests for temporal dependency
- Enhanced volatility metrics (corrected for negative values)
- Sample sequence generation
- Clopper-Pearson confidence intervals

Author: Simin Ali
"""

import numpy as np
import pandas as pd
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from enhanced_evaluation_framework import EnhancedFinancialModelEvaluator

def load_real_data():
    """Load real S&P 500 data for comparison."""
    try:
        data = pd.read_csv("data/sp500_data.csv", index_col=0, parse_dates=True)
        data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
        returns = np.log(data['Close'] / data['Close'].shift(1)).dropna() * 100
        print("✅ Real data loaded")
        return returns.values
    except Exception as e:
        print(f"⚠️ Real data not found: {e}")
        return None

def main():
    """Run enhanced evaluation for all models."""
    print("🚀 Running Enhanced Evaluation Framework...")
    
    # Initialize enhanced evaluator
    evaluator = EnhancedFinancialModelEvaluator()
    
    # Load real S&P 500 data
    print("📊 Loading real S&P 500 data...")
    real_data = load_real_data()
    
    if real_data is None:
        print("❌ Real S&P 500 data not found. Exiting.")
        return
    
    # Load model returns
    print("📊 Loading model returns...")
    model_returns = {}
    model_files = {
        'GARCH': 'results/garch_returns.npy',
        'DDPM': 'results/ddpm_returns.npy',
        'TimeGrad': 'results/timegrad_returns.npy',
        'LLM-Conditioned': 'results/llm_conditioned_returns.npy'
    }
    
    for model_name, file_path in model_files.items():
        try:
            if os.path.exists(file_path):
                model_returns[model_name] = np.load(file_path)
                print(f"✅ {model_name}: {len(model_returns[model_name])} observations")
            else:
                print(f"⚠️ {model_name}: File not found at {file_path}")
        except Exception as e:
            print(f"❌ Error loading {model_name}: {e}")
    
    if not model_returns:
        print("❌ No model returns loaded. Exiting.")
        return
    
    # Initialize results dictionary
    all_metrics = {
        'basic_statistics': [],
        'tail_risk_metrics': [],
        'volatility_metrics': [],
        'distribution_tests': [],
        'var_backtest': [],
        'temporal_dependency': [],
        'robust_metrics': [],
        'metadata': evaluator.metadata
    }
    
    # Compute metrics for each model
    for model_name, synthetic_data in model_returns.items():
        print(f"\n🔍 Computing metrics for {model_name}...")
        
        try:
            # Basic statistics
            print(f"  📊 Computing basic statistics...")
            basic_stats = evaluator.compute_basic_statistics(synthetic_data, model_name)
            all_metrics['basic_statistics'].append(basic_stats)
            
            # Tail risk metrics
            print(f"  📊 Computing tail risk metrics...")
            tail_metrics = evaluator.compute_tail_metrics(synthetic_data, model_name)
            all_metrics['tail_risk_metrics'].append(tail_metrics)
            
            # Enhanced volatility metrics (corrected for negative values)
            print(f"  📊 Computing enhanced volatility metrics...")
            vol_metrics = evaluator.compute_enhanced_volatility_metrics(synthetic_data, model_name)
            all_metrics['volatility_metrics'].append(vol_metrics)
            
            # Distribution tests
            print(f"  📊 Computing distribution tests...")
            dist_tests = evaluator.compute_distribution_tests(real_data, synthetic_data, model_name)
            all_metrics['distribution_tests'].append(dist_tests)
            
            # Enhanced VaR backtesting with Christoffersen tests
            print(f"  📊 Computing enhanced VaR backtesting...")
            var_backtest = evaluator.compute_enhanced_var_backtesting(real_data, synthetic_data, model_name)
            all_metrics['var_backtest'].extend(var_backtest)
            
            # Temporal dependency metrics (ACF, PACF, Ljung-Box)
            print(f"  📊 Computing temporal dependency metrics...")
            temp_dep = evaluator.compute_temporal_dependency_metrics(synthetic_data, model_name)
            all_metrics['temporal_dependency'].append(temp_dep)
            
            # Robust metrics with bootstrap
            print(f"  📊 Computing robust metrics with bootstrap...")
            robust_metrics = evaluator.compute_robust_metrics_with_bootstrap(real_data, synthetic_data, model_name)
            all_metrics['robust_metrics'].append(robust_metrics)
            
            print(f"✅ {model_name} metrics computed successfully")
            
        except Exception as e:
            print(f"❌ Error computing metrics for {model_name}: {e}")
            continue
    
    # Save enhanced evaluation results
    print("\n💾 Saving enhanced evaluation results...")
    evaluator.save_enhanced_evaluation_results(all_metrics)
    
    # Generate enhanced metrics summary CSV
    print("📊 Generating enhanced metrics summary...")
    evaluator.generate_enhanced_metrics_summary_csv(all_metrics)
    
    # Print summary of results
    print("\n📋 Enhanced Evaluation Summary:")
    print(f"  • Models evaluated: {len(model_returns)}")
    print(f"  • Basic statistics: {len(all_metrics['basic_statistics'])}")
    print(f"  • VaR backtests: {len(all_metrics['var_backtest'])}")
    print(f"  • Temporal dependency tests: {len(all_metrics['temporal_dependency'])}")
    print(f"  • Robust metrics: {len(all_metrics['robust_metrics'])}")
    
    # Check for any NaN values in critical metrics
    print("\n🔍 Checking for data quality issues...")
    for model_name in model_returns.keys():
        # Check volatility metrics
        vol_metrics = next((x for x in all_metrics['volatility_metrics'] if x['Model'] == model_name), {})
        mean_vol = vol_metrics.get('Mean_Volatility', np.nan)
        if mean_vol is not None and mean_vol < 0:
            print(f"  ⚠️ {model_name}: Negative mean volatility detected: {mean_vol}")
        
        # Check VaR backtest results
        var_results = [x for x in all_metrics['var_backtest'] if x['Model'] == model_name]
        for var_result in var_results:
            if var_result.get('Independence_Test_pvalue') is None:
                print(f"  ⚠️ {model_name} ({var_result.get('Confidence_Level')}): Missing independence test p-value")
    
    print("\n✅ Enhanced evaluation completed successfully!")
    print("📁 Results saved to:")
    print("  • results/comprehensive_evaluation/evaluation_results_enhanced.json")
    print("  • results/enhanced_metrics_summary.csv")

if __name__ == "__main__":
    main()
