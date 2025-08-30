#!/usr/bin/env python3
"""
Debug script to examine LLM model outputs specifically.
"""

import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path
sys.path.append("src/novelty models")

def test_llm_model():
    print("=== Testing LLM model ===")
    model_path = "results/llm_conditioned/20250816_194604/checkpoints/final_model.pth"
    print(f"Loading from: {model_path}")
    
    try:
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint if isinstance(checkpoint, dict) and 'model_state_dict' not in checkpoint else checkpoint.get('model_state_dict', checkpoint)
        
        # Load model
        from llm_conditioned_diffusion_refactored import LLMConditionedDiffusion, LLMDiffusionTrainer
        
        model = LLMConditionedDiffusion(sequence_length=60, conditioning_dim=64)
        model.load_state_dict(state_dict, strict=False)
        trainer = LLMDiffusionTrainer(model, num_timesteps=1000)
        
        # Create dummy conditioning
        conditioning = torch.zeros(1, 64)  # LLM conditioning dimension
        
        # Generate samples
        print("Generating samples...")
        with torch.no_grad():
            raw_samples = trainer.sample(conditioning, num_samples=1, sample_steps=50)
            raw_samples = raw_samples.cpu().numpy().flatten()
        
        # Analyze raw outputs
        print(f"Raw LLM model outputs:")
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
        
        # 2. If they're percent returns, divide by 100
        if np.std(raw_samples) > 0.1:  # Likely percent units
            percent_to_decimal = raw_samples / 100
            print(f"2. Percent to decimal:    std={np.std(percent_to_decimal):.6f}")
        
        # 3. Scale down by factor
        for factor in [10, 100, 1000]:
            scaled = raw_samples / factor
            print(f"3. Divide by {factor}:        std={np.std(scaled):.6f}")
            
        return raw_samples
        
    except Exception as e:
        print(f"Error loading LLM model: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    # Load real data for comparison
    data = pd.read_csv("data/sp500_data.csv", index_col=0, parse_dates=True)
    simple_returns = (data['Close'] / data['Close'].shift(1) - 1).dropna()
    
    print(f"Real simple returns - Mean: {np.mean(simple_returns):.6f}, Std: {np.std(simple_returns):.6f}")
    
    test_llm_model()

if __name__ == "__main__":
    main()
