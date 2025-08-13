#!/usr/bin/env python3
"""
Recompute Corrected Metrics for Comprehensive Model Comparison

This script recomputes all evaluation metrics using the corrected framework
to ensure consistency and accuracy across all models.

Author: Simin Ali
Thesis: Diffusion Models in Generative AI for Financial Data Synthesis and Risk Management
"""

import numpy as np
import pandas as pd
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from evaluation_framework_corrected import CorrectedFinancialModelEvaluator

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

def load_model_data():
    """Load all model results."""
    models = {}
    
    # GARCH
    try:
        garch_returns = np.load('results/garch_returns.npy', allow_pickle=True)
        models['GARCH'] = garch_returns
        print("✅ GARCH data loaded")
    except Exception as e:
        print(f"⚠️ GARCH data not found: {e}")
    
    # DDPM
    try:
        ddpm_returns = np.load('results/ddpm_returns.npy', allow_pickle=True)
        models['DDPM'] = ddpm_returns
        print("✅ DDPM data loaded")
    except Exception as e:
        print(f"⚠️ DDPM data not found: {e}")
    
    # TimeGrad
    try:
        timegrad_returns = np.load('results/timegrad_returns.npy', allow_pickle=True)
        models['TimeGrad'] = timegrad_returns
        print("✅ TimeGrad data loaded")
    except Exception as e:
        print(f"⚠️ TimeGrad data not found: {e}")
    
    # LLM-Conditioned
    try:
        llm_returns = np.load('results/llm_conditioned_returns.npy', allow_pickle=True)
        models['LLM-Conditioned'] = llm_returns
        print("✅ LLM-Conditioned data loaded")
    except Exception as e:
        print(f"⚠️ LLM-Conditioned data not found: {e}")
    
    return models

def main():
    """Main function to recompute all metrics."""
    print("🚀 Recomputing Corrected Metrics for All Models...")
    
    # Load data
    real_data = load_real_data()
    if real_data is None:
        print("❌ Real data not available. Cannot proceed.")
        return
    
    model_data = load_model_data()
    if not model_data:
        print("❌ No model data available. Cannot proceed.")
        return
    
    # Initialize corrected evaluator
    evaluator = CorrectedFinancialModelEvaluator(list(model_data.keys()))
    
    # Store all results
    all_results = {}
    
    # Process each model
    for model_name, synthetic_data in model_data.items():
        print(f"\n📊 Processing {model_name}...")
        
        model_results = {}
        
        # Basic statistics
        print(f"  Computing basic statistics...")
        basic_stats = evaluator.compute_basic_statistics(synthetic_data, model_name)
        model_results['basic_stats'] = basic_stats
        
        # Tail risk metrics
        print(f"  Computing tail risk metrics...")
        tail_metrics = evaluator.compute_tail_metrics(synthetic_data, model_name)
        model_results['tail_metrics'] = tail_metrics
        
        # Volatility metrics
        print(f"  Computing volatility metrics...")
        vol_metrics = evaluator.compute_volatility_metrics(synthetic_data, model_name)
        model_results['volatility_metrics'] = vol_metrics
        
        # Distribution tests
        print(f"  Computing distribution tests...")
        dist_tests = evaluator.compute_distribution_tests(real_data, synthetic_data, model_name)
        model_results['distribution_tests'] = dist_tests
        
        # VaR backtesting
        print(f"  Computing VaR backtesting...")
        var_backtest = evaluator.compute_var_backtesting(real_data, synthetic_data, model_name)
        model_results['var_backtest'] = var_backtest
        
        # Robust metrics with bootstrap
        print(f"  Computing robust metrics with bootstrap...")
        robust_metrics = evaluator.compute_robust_metrics_with_bootstrap(real_data, synthetic_data, model_name)
        model_results['robust_metrics'] = robust_metrics
        
        all_results[model_name] = model_results
        
        print(f"✅ {model_name} processing complete")
    
    # Add real data statistics for comparison
    print(f"\n📊 Processing Real Data...")
    real_basic_stats = evaluator.compute_basic_statistics(real_data, "Real Data")
    real_tail_metrics = evaluator.compute_tail_metrics(real_data, "Real Data")
    real_vol_metrics = evaluator.compute_volatility_metrics(real_data, "Real Data")
    
    # Compile final results
    final_results = {
        'basic_stats': [real_basic_stats] + [all_results[model]['basic_stats'] for model in all_results],
        'tail_metrics': [real_tail_metrics] + [all_results[model]['tail_metrics'] for model in all_results],
        'volatility_metrics': [real_vol_metrics] + [all_results[model]['volatility_metrics'] for model in all_results],
        'distribution_tests': [all_results[model]['distribution_tests'] for model in all_results],
        'var_backtest': [backtest for model in all_results for backtest in all_results[model]['var_backtest']],
        'robust_metrics': [all_results[model]['robust_metrics'] for model in all_results]
    }
    
    # Save corrected results
    output_path = evaluator.save_evaluation_results(final_results)
    
    # Generate metrics summary CSV
    evaluator.generate_metrics_summary_csv(all_results)
    
    print(f"\n✅ All metrics recomputed and saved!")
    print(f"📄 Corrected results: {output_path}")
    print(f"📊 Metrics summary: results/metrics_summary.csv")
    
    # Print summary of key fixes
    print(f"\n🔧 Key Fixes Applied:")
    print(f"  • Standardized MMD computation across all models")
    print(f"  • Fixed negative volatility values")
    print(f"  • Corrected VaR sign conventions")
    print(f"  • Added bootstrap confidence intervals")
    print(f"  • Removed duplicate entries")
    print(f"  • Standardized data scaling")
    print(f"  • Enhanced error handling")

if __name__ == "__main__":
    main()
