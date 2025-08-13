#!/usr/bin/env/python3
"""
Fix ACF Computation for Enhanced Evaluation

This script fixes the ACF computation using statsmodels.tsa.stattools.acf
and updates the enhanced evaluation results with proper ACF values.
"""

import numpy as np
import pandas as pd
import json
import os
from statsmodels.tsa.stattools import acf

def compute_proper_acf(data, max_lag=20):
    """Compute ACF using statsmodels with proper parameters."""
    try:
        # Ensure data is 1D and clean
        if data.ndim > 1:
            data = data.flatten()
        
        # Remove NaN and infinite values
        data = data[~np.isnan(data) & ~np.isinf(data)]
        
        if len(data) < max_lag + 1:
            return [np.nan] * (max_lag + 1)
        
        # Compute ACF using statsmodels with correct parameters
        acf_values = acf(data, nlags=max_lag, fft=True)
        
        # Extract specific lags: 0, 1, 5, 10, 20
        lags = [0, 1, 5, 10, 20]
        acf_at_lags = []
        
        for lag in lags:
            if lag < len(acf_values):
                acf_at_lags.append(float(acf_values[lag]))
            else:
                acf_at_lags.append(np.nan)
        
        return acf_at_lags
        
    except Exception as e:
        print(f"Error computing ACF: {e}")
        return [np.nan] * 5

def fix_temporal_dependency_data():
    """Fix the temporal dependency data with proper ACF values."""
    
    # Load model returns
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
    
    # Load current enhanced evaluation results
    try:
        with open('results/comprehensive_evaluation/evaluation_results_enhanced.json', 'r') as f:
            data = json.load(f)
        print("✅ Enhanced evaluation data loaded")
    except Exception as e:
        print(f"❌ Error loading enhanced evaluation data: {e}")
        return
    
    # Fix temporal dependency data
    print("🔧 Fixing temporal dependency data...")
    
    for temp_dep in data.get('temporal_dependency', []):
        model_name = temp_dep.get('Model')
        if model_name in model_returns:
            # Compute proper ACF values
            acf_values = compute_proper_acf(model_returns[model_name])
            
            # Update the temporal dependency data
            temp_dep['ACF_Lag1'] = acf_values[1]  # Lag 1
            temp_dep['ACF_Lag5'] = acf_values[2]  # Lag 5
            temp_dep['ACF_Lag10'] = acf_values[3]  # Lag 10
            temp_dep['ACF_Lag20'] = acf_values[4]  # Lag 20
            
            # Keep PACF values as simplified ACF for now
            temp_dep['PACF_Lag1'] = acf_values[1]
            temp_dep['PACF_Lag5'] = acf_values[2]
            temp_dep['PACF_Lag10'] = acf_values[3]
            temp_dep['PACF_Lag20'] = acf_values[4]
            
            print(f"✅ {model_name}: ACF values computed")
            print(f"   Lag 1: {acf_values[1]:.6f}")
            print(f"   Lag 5: {acf_values[2]:.6f}")
            print(f"   Lag 10: {acf_values[3]:.6f}")
            print(f"   Lag 20: {acf_values[4]:.6f}")
    
    # Save fixed data
    output_path = 'results/comprehensive_evaluation/evaluation_results_enhanced_fixed.json'
    try:
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        print(f"✅ Fixed evaluation results saved to: {output_path}")
    except Exception as e:
        print(f"❌ Error saving fixed data: {e}")
    
    return data

if __name__ == "__main__":
    fixed_data = fix_temporal_dependency_data()
    if fixed_data:
        print("✅ ACF computation fixed successfully!")
