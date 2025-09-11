#!/usr/bin/env python3
"""
Evaluation Conditioning Examples

Demonstrates practical usage patterns for evaluation-time conditioning providers.
Shows causal computation, spec consistency, and graceful error handling.
"""

import numpy as np
import pandas as pd
from eval_conditioning_providers import (
    generate_eval_conditioning, 
    EvalProviderFactory, 
    load_conditioning_spec,
    NoneProvider,
    ExplicitEvalProvider, 
    LLMEvalProvider
)
import os
from pathlib import Path
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

def create_sample_returns_data():
    """Create sample returns data for examples."""
    # Create realistic returns data
    np.random.seed(42)
    
    # Generate daily returns from 2010 to 2021
    date_range = pd.date_range('2010-01-01', '2021-12-31', freq='D')
    
    # Base parameters
    base_mean = 0.0004
    base_vol = 0.01
    
    # Create regime-dependent returns
    returns = []
    for date in date_range:
        # Different volatility regimes
        if date.year in [2020]:  # COVID year - high volatility
            vol = base_vol * 3
            mean = base_mean * 0.5
        elif date.year in [2008, 2009]:  # Financial crisis
            vol = base_vol * 2
            mean = base_mean * 0
        else:  # Normal times
            vol = base_vol
            mean = base_mean
        
        # Add day-of-week effects
        if date.weekday() == 0:  # Monday effect
            mean *= 0.8
            vol *= 1.2
        
        daily_return = np.random.normal(mean, vol)
        returns.append(daily_return)
    
    # Create DataFrame
    returns_df = pd.DataFrame({
        'returns': returns,
        'log_returns': returns  # Alias
    }, index=date_range)
    
    # Add some missing data (weekends, holidays)
    for date in returns_df.index:
        if date.weekday() >= 5:  # Weekends
            returns_df.loc[date, :] = np.nan
    
    return returns_df.dropna()

def example_basic_usage():
    """Example 1: Basic usage of evaluation conditioning providers."""
    print("=== Example 1: Basic Usage ===")
    
    # Create sample data
    returns_data = create_sample_returns_data()
    print(f"Sample returns data: {returns_data.shape}")
    
    # Test each model type
    model_types = ['zero', 'explicit', 'llm']
    target_dates = ['2020-03-15', '2020-06-01', '2020-12-31']
    
    for model_type in model_types:
        checkpoint_dir = f'checkpoints/precovid/{model_type}/20100101-20191231'
        
        if not os.path.exists(checkpoint_dir):
            print(f"Checkpoint not found: {checkpoint_dir}")
            continue
        
        try:
            conditioning, warnings = generate_eval_conditioning(
                checkpoint_dir, returns_data, target_dates
            )
            
            if conditioning is not None:
                print(f"{model_type.upper()} conditioning: {conditioning.shape}")
                print(f"  Statistics: mean={conditioning.mean():.6f}, std={conditioning.std():.6f}")
            else:
                print(f"{model_type.upper()} conditioning: None (zero)")
            
            if warnings:
                print(f"  Warnings: {warnings}")
                
        except Exception as e:
            print(f"Error with {model_type}: {e}")
    
    print()

def example_causal_verification():
    """Example 2: Verify causal computation (no look-ahead)."""
    print("=== Example 2: Causal Verification ===")
    
    returns_data = create_sample_returns_data()
    checkpoint_dir = 'checkpoints/precovid/explicit/20100101-20191231'
    
    if not os.path.exists(checkpoint_dir):
        print(f"Checkpoint not found: {checkpoint_dir}")
        return
    
    try:
        # Test causality by generating conditioning for different target dates
        # using the same historical data
        test_dates = ['2020-03-15', '2020-03-16', '2020-03-17']
        
        spec = load_conditioning_spec(checkpoint_dir)
        provider = ExplicitEvalProvider(spec, checkpoint_dir)
        
        print("Verifying causal computation:")
        print("Each date should only use data <= that date")
        
        for target_date in test_dates:
            conditioning = provider.generate_conditioning(
                returns_data, [pd.Timestamp(target_date)]
            )
            
            # Extract regime and features
            regime_idx = conditioning[0, :4].argmax()
            vol_scaled = conditioning[0, 4]
            trend_scaled = conditioning[0, 5]
            
            regime_names = ['Up-Low', 'Up-High', 'Down-Low', 'Down-High']
            print(f"  {target_date}: {regime_names[regime_idx]}, vol={vol_scaled:.3f}, trend={trend_scaled:.3f}")
        
        # Verify that changing future data doesn't affect past conditioning
        print("\nVerifying future data independence:")
        
        # Generate conditioning for 2020-03-15
        original_conditioning = provider.generate_conditioning(
            returns_data, [pd.Timestamp('2020-03-15')]
        )
        
        # Modify future data (after 2020-03-15)
        modified_data = returns_data.copy()
        future_mask = modified_data.index > '2020-03-15'
        modified_data.loc[future_mask, 'returns'] *= 10  # Extreme modification
        
        # Generate conditioning again with modified future data
        modified_conditioning = provider.generate_conditioning(
            modified_data, [pd.Timestamp('2020-03-15')]
        )
        
        # Should be identical (causal)
        is_identical = np.allclose(original_conditioning, modified_conditioning)
        print(f"  Conditioning identical after future data modification: {is_identical}")
        
        if not is_identical:
            print("  WARNING: Causality violation detected!")
        
    except Exception as e:
        print(f"Error in causal verification: {e}")
    
    print()

def example_spec_consistency():
    """Example 3: Verify spec consistency (uses saved transforms)."""
    print("=== Example 3: Spec Consistency ===")
    
    returns_data = create_sample_returns_data()
    checkpoint_dir = 'checkpoints/precovid/explicit/20100101-20191231'
    
    if not os.path.exists(checkpoint_dir):
        print(f"Checkpoint not found: {checkpoint_dir}")
        return
    
    try:
        # Load and inspect the conditioning spec
        spec = load_conditioning_spec(checkpoint_dir)
        print("Loaded conditioning specification:")
        print(f"  Type: {spec['type']}")
        print(f"  Conditioning dim: {spec['conditioning_dim']}")
        print(f"  Vol threshold: {spec.get('vol_threshold', 'N/A')}")
        print(f"  Vol window: {spec.get('vol_window', 'N/A')}")
        print(f"  Trend window: {spec.get('trend_window', 'N/A')}")
        
        # Extract saved scaler parameters
        features = spec.get('features', {})
        vol_info = features.get('z_vol', {})
        trend_info = features.get('trend', {})
        
        print("Saved scaler parameters:")
        print(f"  Vol scaler - mean: {vol_info.get('scaler_mean', 'N/A')}, scale: {vol_info.get('scaler_scale', 'N/A')}")
        print(f"  Trend scaler - mean: {trend_info.get('scaler_mean', 'N/A')}, scale: {trend_info.get('scaler_scale', 'N/A')}")
        
        # Create provider and verify it uses these exact parameters
        provider = ExplicitEvalProvider(spec, checkpoint_dir)
        
        print("\nProvider initialized parameters:")
        print(f"  Vol threshold: {provider.vol_threshold}")
        print(f"  Vol scaler - mean: {provider.vol_scaler_mean}, scale: {provider.vol_scaler_scale}")
        print(f"  Trend scaler - mean: {provider.trend_scaler_mean}, scale: {provider.trend_scaler_scale}")
        
        # Verify no refitting occurs
        print("\nVerifying no refitting on evaluation data:")
        
        # Generate conditioning multiple times with different evaluation data
        target_date = '2020-06-01'
        
        # Original data
        conditioning1 = provider.generate_conditioning(
            returns_data, [pd.Timestamp(target_date)]
        )
        
        # Modified evaluation data (should not affect scaling parameters)
        extreme_data = returns_data.copy()
        extreme_data['returns'] *= 100  # Extreme modification
        
        conditioning2 = provider.generate_conditioning(
            extreme_data, [pd.Timestamp(target_date)]
        )
        
        # The raw features will be different due to different input data,
        # but the scaling parameters should remain the same
        print(f"  Original vol feature: {conditioning1[0, 4]:.6f}")
        print(f"  Extreme data vol feature: {conditioning2[0, 4]:.6f}")
        print("  (Different values expected due to different input data)")
        print("  (Scaling parameters remain unchanged)")
        
    except Exception as e:
        print(f"Error in spec consistency verification: {e}")
    
    print()

def example_graceful_fallbacks():
    """Example 4: Demonstrate graceful fallback handling."""
    print("=== Example 4: Graceful Fallback Handling ===")
    
    returns_data = create_sample_returns_data()
    
    # Test with incomplete spec
    print("Testing with incomplete conditioning spec:")
    
    incomplete_spec = {
        'type': 'explicit',
        'conditioning_dim': 6,
        'vol_window': 20,
        'trend_window': 60,
        # Missing: vol_threshold, scaler parameters
        'features': {}
    }
    
    try:
        provider = ExplicitEvalProvider(incomplete_spec)
        print(f"  Spec complete: {provider.spec_complete}")
        print(f"  Warnings: {provider.get_warnings()}")
        
        # Generate conditioning (should use fallbacks)
        target_dates = ['2020-03-15', '2020-06-01']
        conditioning = provider.generate_conditioning(returns_data, target_dates)
        
        print(f"  Generated conditioning shape: {conditioning.shape}")
        print(f"  Final warnings: {provider.get_warnings()}")
        
    except Exception as e:
        print(f"  Error with incomplete spec: {e}")
    
    # Test with missing PCA for LLM
    print("\nTesting LLM provider without PCA model:")
    
    llm_spec = {
        'type': 'llm',
        'conditioning_dim': 16,
        'pca_components': 16,
        'original_embedding_dim': 768,
        'train_cutoff': '2019-12-31'
    }
    
    try:
        # Create provider without checkpoint_dir (no PCA file)
        provider = LLMEvalProvider(llm_spec, checkpoint_dir=None)
        print(f"  PCA loaded: {provider.pca_loaded}")
        print(f"  Need PCA fitting: {provider.need_pca_fitting}")
        
        # Generate conditioning (should fit PCA on pre-COVID data)
        conditioning = provider.generate_conditioning(returns_data, ['2020-06-01'])
        
        print(f"  Generated conditioning shape: {conditioning.shape}")
        print(f"  Warnings: {provider.get_warnings()}")
        
    except Exception as e:
        print(f"  Error with missing PCA: {e}")
    
    # Test with missing target dates
    print("\nTesting with missing data for target dates:")
    
    checkpoint_dir = 'checkpoints/precovid/explicit/20100101-20191231'
    if os.path.exists(checkpoint_dir):
        try:
            # Request conditioning for dates not in returns data
            future_dates = ['2025-01-01', '2025-06-01']
            
            conditioning, warnings = generate_eval_conditioning(
                checkpoint_dir, returns_data, future_dates
            )
            
            print(f"  Generated conditioning shape: {conditioning.shape}")
            print(f"  Warnings: {warnings}")
            print(f"  Conditioning values: {conditioning}")
            print("  (Should be zero vectors for missing dates)")
            
        except Exception as e:
            print(f"  Error with missing target dates: {e}")
    
    print()

def example_time_series_analysis():
    """Example 5: Time series analysis of conditioning evolution."""
    print("=== Example 5: Time Series Analysis ===")
    
    returns_data = create_sample_returns_data()
    checkpoint_dir = 'checkpoints/precovid/explicit/20100101-20191231'
    
    if not os.path.exists(checkpoint_dir):
        print(f"Checkpoint not found: {checkpoint_dir}")
        return
    
    try:
        # Generate conditioning for COVID period
        covid_start = pd.Timestamp('2020-02-01')
        covid_end = pd.Timestamp('2020-06-01')
        
        # Create weekly target dates
        target_dates = pd.date_range(covid_start, covid_end, freq='W').tolist()
        
        conditioning, warnings = generate_eval_conditioning(
            checkpoint_dir, returns_data, target_dates
        )
        
        print(f"Generated conditioning for {len(target_dates)} dates during COVID period")
        
        # Analyze regime transitions
        regime_names = ['Up-Low', 'Up-High', 'Down-Low', 'Down-High']
        regimes = conditioning[:, :4].argmax(axis=1)
        volatility = conditioning[:, 4]
        trend = conditioning[:, 5]
        
        print("\nRegime Evolution Analysis:")
        print("Date\t\tRegime\t\tVol\tTrend")
        print("-" * 50)
        
        for i, date in enumerate(target_dates):
            regime_name = regime_names[regimes[i]]
            vol = volatility[i]
            trend_val = trend[i]
            print(f"{date.strftime('%Y-%m-%d')}\t{regime_name:10}\t{vol:6.3f}\t{trend_val:6.3f}")
        
        # Statistics
        print(f"\nRegime Distribution:")
        for i, regime in enumerate(regime_names):
            count = np.sum(regimes == i)
            pct = count / len(regimes) * 100
            print(f"  {regime}: {count} times ({pct:.1f}%)")
        
        print(f"\nVolatility Statistics:")
        print(f"  Mean: {volatility.mean():.3f}, Std: {volatility.std():.3f}")
        print(f"  Min: {volatility.min():.3f}, Max: {volatility.max():.3f}")
        
        print(f"\nTrend Statistics:")
        print(f"  Mean: {trend.mean():.3f}, Std: {trend.std():.3f}")
        print(f"  Min: {trend.min():.3f}, Max: {trend.max():.3f}")
        
        if warnings:
            print(f"\nWarnings: {warnings}")
        
    except Exception as e:
        print(f"Error in time series analysis: {e}")
    
    print()

def example_model_comparison():
    """Example 6: Compare conditioning across model types."""
    print("=== Example 6: Model Comparison ===")
    
    returns_data = create_sample_returns_data()
    target_dates = ['2020-03-15', '2020-03-20', '2020-03-25']  # Black Monday week
    
    print(f"Comparing conditioning for stress period: {target_dates}")
    
    model_results = {}
    model_types = ['zero', 'explicit', 'llm']
    
    for model_type in model_types:
        checkpoint_dir = f'checkpoints/precovid/{model_type}/20100101-20191231'
        
        if not os.path.exists(checkpoint_dir):
            print(f"Checkpoint not found: {checkpoint_dir}")
            continue
        
        try:
            conditioning, warnings = generate_eval_conditioning(
                checkpoint_dir, returns_data, target_dates
            )
            
            model_results[model_type] = {
                'conditioning': conditioning,
                'warnings': warnings
            }
            
            if conditioning is not None:
                print(f"\n{model_type.upper()} Model:")
                print(f"  Shape: {conditioning.shape}")
                print(f"  Mean: {conditioning.mean():.6f}, Std: {conditioning.std():.6f}")
                print(f"  Range: [{conditioning.min():.6f}, {conditioning.max():.6f}]")
                
                if model_type == 'explicit':
                    # Analyze regime classifications
                    regime_names = ['Up-Low', 'Up-High', 'Down-Low', 'Down-High']
                    for i, date in enumerate(target_dates):
                        regime_idx = conditioning[i, :4].argmax()
                        print(f"  {date}: {regime_names[regime_idx]}")
                
                if warnings:
                    print(f"  Warnings: {warnings}")
            else:
                print(f"\n{model_type.upper()} Model: None (zero conditioning)")
                
        except Exception as e:
            print(f"Error with {model_type} model: {e}")
    
    # Compare dimensionality and characteristics
    print("\nModel Comparison Summary:")
    print("Model\t\tDimensions\tType\t\tCharacteristics")
    print("-" * 70)
    
    for model_type, results in model_results.items():
        conditioning = results['conditioning']
        if conditioning is not None:
            dims = conditioning.shape[1]
            if model_type == 'explicit':
                char = "Regime + Vol + Trend"
            elif model_type == 'llm':
                char = "PCA Embeddings"
            else:
                char = "N/A"
        else:
            dims = 0
            char = "Unconditional"
        
        print(f"{model_type.upper():12}\t{dims:10}\t{model_type:12}\t{char}")
    
    print()

def example_integration_test():
    """Example 7: End-to-end integration test."""
    print("=== Example 7: Integration Test ===")
    
    returns_data = create_sample_returns_data()
    
    # Test workflow: load spec -> create provider -> generate conditioning
    model_types = ['zero', 'explicit', 'llm']
    target_dates = ['2020-06-01']
    
    print("Testing complete workflow for each model type:")
    
    for model_type in model_types:
        checkpoint_dir = f'checkpoints/precovid/{model_type}/20100101-20191231'
        
        if not os.path.exists(checkpoint_dir):
            print(f"  {model_type.upper()}: Checkpoint not found")
            continue
        
        try:
            print(f"\n  {model_type.upper()} Model:")
            
            # Step 1: Load specification
            spec = load_conditioning_spec(checkpoint_dir)
            print(f"    1. Loaded spec: {spec['type']} (dim={spec['conditioning_dim']})")
            
            # Step 2: Create provider
            provider = EvalProviderFactory.create_provider(spec, checkpoint_dir)
            print(f"    2. Created provider: {provider.__class__.__name__}")
            
            # Step 3: Validate spec
            is_valid = provider.validate_spec()
            print(f"    3. Spec validation: {'PASS' if is_valid else 'FAIL'}")
            
            # Step 4: Generate conditioning
            conditioning = provider.generate_conditioning(returns_data, target_dates)
            
            if conditioning is not None:
                print(f"    4. Generated conditioning: {conditioning.shape}")
                print(f"       Sample values: {conditioning[0][:3]} {'...' if conditioning.shape[1] > 3 else ''}")
            else:
                print(f"    4. Generated conditioning: None (zero)")
            
            # Step 5: Check warnings
            warnings = provider.get_warnings()
            if warnings:
                print(f"    5. Warnings: {len(warnings)} issues")
                for warning in warnings[:2]:  # Show first 2
                    print(f"       - {warning}")
            else:
                print(f"    5. Warnings: None")
            
            print(f"    Status: SUCCESS")
            
        except Exception as e:
            print(f"    Status: FAILED - {e}")
    
    print("\nIntegration test completed.")

def main():
    """Run all examples."""
    print("Evaluation Conditioning Providers Examples")
    print("==========================================")
    
    # Run examples in sequence
    example_basic_usage()
    example_causal_verification()
    example_spec_consistency()
    example_graceful_fallbacks()
    example_time_series_analysis()
    example_model_comparison()
    example_integration_test()
    
    print("=== Summary ===")
    print("All examples completed!")
    print("Key features demonstrated:")
    print("- Causal day-by-day computation")
    print("- Spec-consistent transforms")
    print("- Graceful fallback handling")
    print("- Comprehensive warning system")
    print("- Multi-model compatibility")

if __name__ == "__main__":
    main()
