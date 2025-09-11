#!/usr/bin/env python3
"""
Checkpoint Loader Examples

Demonstrates various usage patterns for the checkpoint loader and sampler utility.
Run this script to see practical examples of loading and sampling from trained models.
"""

import numpy as np
import pandas as pd
from checkpoint_loader_sampler import CheckpointSampler, load_and_sample
import os
from pathlib import Path

def example_basic_usage():
    """Example 1: Basic checkpoint loading and sampling."""
    print("=== Example 1: Basic Usage ===")
    
    checkpoint_dir = "checkpoints/precovid/zero/20100101-20191231"
    
    if not os.path.exists(checkpoint_dir):
        print(f"Checkpoint not found: {checkpoint_dir}")
        return
    
    try:
        # Simple one-line usage
        samples = load_and_sample(
            checkpoint_dir=checkpoint_dir,
            dates=['2020-03-01', '2020-06-01'],
            num_paths=100,
            seq_len=60
        )
        
        print(f"Generated samples shape: {samples.shape}")
        print(f"Sample statistics: mean={samples.mean():.6f}, std={samples.std():.6f}")
        
    except Exception as e:
        print(f"Error in basic usage: {e}")

def example_reusable_sampler():
    """Example 2: Reusable sampler for multiple generations."""
    print("\n=== Example 2: Reusable Sampler ===")
    
    checkpoint_dir = "checkpoints/precovid/explicit/20100101-20191231"
    
    if not os.path.exists(checkpoint_dir):
        print(f"Checkpoint not found: {checkpoint_dir}")
        return
    
    try:
        # Load once, use multiple times
        sampler = CheckpointSampler(checkpoint_dir)
        
        # Generate samples for different scenarios
        pre_covid = sampler.generate_samples(
            dates=['2019-01-01', '2019-06-01'],
            num_paths=200,
            output_dir="examples/pre_covid"
        )
        
        covid_start = sampler.generate_samples(
            dates=['2020-03-01', '2020-04-01'],
            num_paths=200,
            output_dir="examples/covid_start"
        )
        
        print(f"Pre-COVID samples: {pre_covid.shape}")
        print(f"COVID-start samples: {covid_start.shape}")
        
        # Compare volatilities
        pre_vol = pre_covid.std(axis=1).mean()
        covid_vol = covid_start.std(axis=1).mean()
        
        print(f"Pre-COVID avg volatility: {pre_vol:.6f}")
        print(f"COVID-start avg volatility: {covid_vol:.6f}")
        print(f"Volatility ratio: {covid_vol/pre_vol:.2f}x")
        
    except Exception as e:
        print(f"Error in reusable sampler: {e}")

def example_all_model_types():
    """Example 3: Compare all model types."""
    print("\n=== Example 3: All Model Types ===")
    
    model_types = ['zero', 'explicit', 'llm']
    results = {}
    
    for model_type in model_types:
        checkpoint_dir = f"checkpoints/precovid/{model_type}/20100101-20191231"
        
        if not os.path.exists(checkpoint_dir):
            print(f"Checkpoint not found: {checkpoint_dir}")
            continue
        
        try:
            samples = load_and_sample(
                checkpoint_dir=checkpoint_dir,
                dates=['2020-06-01'],  # Single date for comparison
                num_paths=500,
                output_dir=f"examples/{model_type}_comparison"
            )
            
            # Calculate statistics
            results[model_type] = {
                'mean': samples.mean(),
                'std': samples.std(),
                'var_95': np.percentile(samples.sum(axis=1), 5),
                'max_drawdown': np.percentile(samples.min(axis=1), 5)
            }
            
            print(f"{model_type.upper()} model:")
            print(f"  Mean return: {results[model_type]['mean']:.6f}")
            print(f"  Volatility: {results[model_type]['std']:.6f}")
            print(f"  95% VaR: {results[model_type]['var_95']:.6f}")
            
        except Exception as e:
            print(f"Error with {model_type} model: {e}")
    
    return results

def example_risk_analysis():
    """Example 4: Risk analysis with generated samples."""
    print("\n=== Example 4: Risk Analysis ===")
    
    checkpoint_dir = "checkpoints/precovid/zero/20100101-20191231"
    
    if not os.path.exists(checkpoint_dir):
        print(f"Checkpoint not found: {checkpoint_dir}")
        return
    
    try:
        # Generate large sample for risk analysis
        samples = load_and_sample(
            checkpoint_dir=checkpoint_dir,
            dates=['2020-03-01', '2020-06-01', '2020-12-31'],
            num_paths=5000,
            output_dir="examples/risk_analysis"
        )
        
        # Calculate cumulative returns for each path
        cumulative_returns = samples.cumsum(axis=1)
        final_returns = cumulative_returns[:, -1]
        
        # Risk metrics
        var_levels = [1, 5, 10]
        print("Value at Risk (VaR) Analysis:")
        for level in var_levels:
            var = np.percentile(final_returns, level)
            print(f"  {level}% VaR: {var:.6f}")
        
        # Conditional VaR (Expected Shortfall)
        var_5 = np.percentile(final_returns, 5)
        cvar_5 = final_returns[final_returns <= var_5].mean()
        print(f"  5% CVaR: {cvar_5:.6f}")
        
        # Maximum drawdown analysis
        running_max = np.maximum.accumulate(cumulative_returns, axis=1)
        drawdowns = cumulative_returns - running_max
        max_drawdowns = drawdowns.min(axis=1)
        
        print("\nDrawdown Analysis:")
        print(f"  Mean max drawdown: {max_drawdowns.mean():.6f}")
        print(f"  95% worst drawdown: {np.percentile(max_drawdowns, 5):.6f}")
        
        # Volatility clustering
        returns_volatility = np.abs(samples)
        vol_autocorr = np.corrcoef(returns_volatility[:, :-1].flatten(), 
                                   returns_volatility[:, 1:].flatten())[0, 1]
        print(f"  Volatility autocorrelation: {vol_autocorr:.3f}")
        
    except Exception as e:
        print(f"Error in risk analysis: {e}")

def example_custom_dates():
    """Example 5: Custom date ranges and scenarios."""
    print("\n=== Example 5: Custom Scenarios ===")
    
    checkpoint_dir = "checkpoints/precovid/llm/20100101-20191231"
    
    if not os.path.exists(checkpoint_dir):
        print(f"Checkpoint not found: {checkpoint_dir}")
        return
    
    try:
        # Define custom scenarios
        scenarios = {
            'market_crash': ['2008-09-15', '2008-10-15', '2008-11-15'],  # Lehman Brothers
            'covid_onset': ['2020-02-15', '2020-03-15', '2020-04-15'],   # COVID market crash
            'recovery': ['2020-06-01', '2020-09-01', '2020-12-01'],      # Recovery period
            'normal_times': ['2015-01-01', '2016-01-01', '2017-01-01']   # Normal market
        }
        
        sampler = CheckpointSampler(checkpoint_dir)
        scenario_results = {}
        
        for scenario_name, dates in scenarios.items():
            samples = sampler.generate_samples(
                dates=dates,
                num_paths=1000,
                output_dir=f"examples/{scenario_name}"
            )
            
            # Calculate scenario statistics
            scenario_results[scenario_name] = {
                'mean_return': samples.mean(),
                'volatility': samples.std(),
                'skewness': ((samples - samples.mean()) ** 3).mean() / (samples.std() ** 3),
                'kurtosis': ((samples - samples.mean()) ** 4).mean() / (samples.std() ** 4) - 3
            }
            
            print(f"{scenario_name.upper().replace('_', ' ')} Scenario:")
            print(f"  Mean return: {scenario_results[scenario_name]['mean_return']:.6f}")
            print(f"  Volatility: {scenario_results[scenario_name]['volatility']:.6f}")
            print(f"  Skewness: {scenario_results[scenario_name]['skewness']:.3f}")
            print(f"  Excess kurtosis: {scenario_results[scenario_name]['kurtosis']:.3f}")
        
        return scenario_results
        
    except Exception as e:
        print(f"Error in custom scenarios: {e}")

def example_batch_processing():
    """Example 6: Batch processing multiple checkpoints."""
    print("\n=== Example 6: Batch Processing ===")
    
    # Find all available checkpoints
    checkpoint_base = Path("checkpoints/precovid")
    
    if not checkpoint_base.exists():
        print("No pre-COVID checkpoints found")
        return
    
    results = {}
    
    for model_dir in checkpoint_base.iterdir():
        if not model_dir.is_dir():
            continue
            
        model_type = model_dir.name
        checkpoint_dir = model_dir / "20100101-20191231"
        
        if not checkpoint_dir.exists():
            continue
        
        try:
            print(f"Processing {model_type} model...")
            
            samples = load_and_sample(
                checkpoint_dir=str(checkpoint_dir),
                dates=['2020-03-01'],  # Single stress test date
                num_paths=100,
                output_dir=f"examples/batch_{model_type}"
            )
            
            results[model_type] = {
                'samples_shape': samples.shape,
                'mean': samples.mean(),
                'std': samples.std(),
                'min': samples.min(),
                'max': samples.max()
            }
            
        except Exception as e:
            print(f"Error processing {model_type}: {e}")
            results[model_type] = {'error': str(e)}
    
    print("\nBatch Processing Results:")
    for model_type, stats in results.items():
        if 'error' in stats:
            print(f"  {model_type}: ERROR - {stats['error']}")
        else:
            print(f"  {model_type}: {stats['samples_shape']} | "
                  f"μ={stats['mean']:.6f}, σ={stats['std']:.6f}")
    
    return results

def main():
    """Run all examples."""
    print("Checkpoint Loader & Sampler Examples")
    print("=====================================")
    
    # Create examples directory
    os.makedirs("examples", exist_ok=True)
    
    # Run examples
    example_basic_usage()
    example_reusable_sampler()
    model_comparison = example_all_model_types()
    example_risk_analysis()
    scenario_analysis = example_custom_dates()
    batch_results = example_batch_processing()
    
    print("\n=== Summary ===")
    print("All examples completed! Check the 'examples/' directory for generated samples.")
    print("Each example demonstrates different aspects of the checkpoint loader utility.")

if __name__ == "__main__":
    main()
