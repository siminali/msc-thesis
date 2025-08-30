#!/usr/bin/env python3
"""
Debug script to examine raw model outputs and compare with real data scales.
"""

import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path
sys.path.append("src/novelty models")

# Load real data for comparison
def load_real_data():
    data = pd.read_csv("data/sp500_data.csv", index_col=0, parse_dates=True)
    log_returns = np.log(data['Close'] / data['Close'].shift(1)).dropna()
    simple_returns = (data['Close'] / data['Close'].shift(1) - 1).dropna()
    return log_returns, simple_returns

# Load a model and generate raw samples
def test_model_outputs(model_path, model_type="zero"):
    print(f"\n=== Testing {model_type} model ===")
    print(f"Loading from: {model_path}")
    
    try:
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint if isinstance(checkpoint, dict) and 'model_state_dict' not in checkpoint else checkpoint.get('model_state_dict', checkpoint)
        
        # Load model
        from explicit_cond_ddpm import ExplicitConditioningDDPM, ExplicitConditioningTrainer
        
        model = ExplicitConditioningDDPM(sequence_length=60, conditioning_dim=5)
        model.load_state_dict(state_dict)
        trainer = ExplicitConditioningTrainer(model, num_timesteps=1000)
        
        # Create dummy conditioning
        conditioning = torch.zeros(1, 5)  # Neutral conditioning
        
        # Generate samples
        print("Generating samples...")
        with torch.no_grad():
            raw_samples = trainer.sample(conditioning, num_samples=1, sampler="ddim", sample_steps=50)
            raw_samples = raw_samples.cpu().numpy().flatten()
        
        # Analyze raw outputs
        print(f"Raw model outputs:")
        print(f"  Mean: {np.mean(raw_samples):.6f}")
        print(f"  Std:  {np.std(raw_samples):.6f}")
        print(f"  Min:  {np.min(raw_samples):.6f}")
        print(f"  Max:  {np.max(raw_samples):.6f}")
        print(f"  P5:   {np.percentile(raw_samples, 5):.6f}")
        print(f"  P95:  {np.percentile(raw_samples, 95):.6f}")
        
        # Try different scaling approaches
        print(f"\nTesting different scaling approaches:")
        
        # 1. As is (assuming they're already simple returns)
        as_simple = raw_samples
        print(f"1. As simple returns:     std={np.std(as_simple):.6f}")
        
        # 2. Convert from log to simple (current approach)
        clipped_log = np.clip(raw_samples, -10, 10)
        log_to_simple = np.exp(clipped_log) - 1.0
        print(f"2. Log to simple (exp-1): std={np.std(log_to_simple):.6f}")
        
        # 3. Scale down by factor
        for factor in [100, 1000, 10000]:
            scaled = raw_samples / factor
            print(f"3. Divide by {factor}:        std={np.std(scaled):.6f}")
        
        # 4. If they're percent returns, divide by 100
        if np.std(raw_samples) > 0.1:  # Likely percent units
            percent_to_decimal = raw_samples / 100
            print(f"4. Percent to decimal:    std={np.std(percent_to_decimal):.6f}")
        
        return raw_samples
        
    except Exception as e:
        print(f"Error loading {model_type} model: {e}")
        return None

def main():
    print("Loading real data for comparison...")
    log_returns, simple_returns = load_real_data()
    
    print(f"Real data statistics:")
    print(f"  Log returns    - Mean: {np.mean(log_returns):.6f}, Std: {np.std(log_returns):.6f}")
    print(f"  Simple returns - Mean: {np.mean(simple_returns):.6f}, Std: {np.std(simple_returns):.6f}")
    
    # Test models
    model_paths = {
        "zero": "results/zero_conditioned/20250816_194604/checkpoints/final_model.pth",
        "explicit": "results/explicit_conditioned/20250816_194604/checkpoints/final_model.pth"
    }
    
    for model_type, model_path in model_paths.items():
        if Path(model_path).exists():
            test_model_outputs(model_path, model_type)
        else:
            print(f"Model path not found: {model_path}")

if __name__ == "__main__":
    main()
