#!/usr/bin/env python3
"""
Deep investigation of explicit model's training vs LLM model to find root cause.
"""
import torch
import numpy as np
import json
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.append('src/novelty models')
from explicit_cond_ddpm import ExplicitConditioningDDPM, ExplicitConditioningTrainer

def compare_model_checkpoints():
    """Compare explicit and LLM model training details."""
    print("="*70)
    print("INVESTIGATING MODEL TRAINING DIFFERENCES")
    print("="*70)
    
    # Load metadata for both models
    explicit_meta_path = Path("checkpoints/precovid/explicit/20100101-20191231/meta.json")
    llm_meta_path = Path("checkpoints/precovid/llm/20100101-20191231/meta.json")
    
    with open(explicit_meta_path) as f:
        explicit_meta = json.load(f)
    
    with open(llm_meta_path) as f:
        llm_meta = json.load(f)
    
    print("TRAINING DATA COMPARISON:")
    print(f"Explicit - mean: {explicit_meta['data_info']['train_stats']['mean']:.8f}")
    print(f"LLM      - mean: {llm_meta['data_info']['train_stats']['mean']:.8f}")
    print(f"Explicit - std:  {explicit_meta['data_info']['train_stats']['std']:.8f}")
    print(f"LLM      - std:  {llm_meta['data_info']['train_stats']['std']:.8f}")
    
    print("\nTRAINING PERFORMANCE:")
    print(f"Explicit - train loss: {explicit_meta['training_info']['train_loss']:.6f}")
    print(f"LLM      - train loss: {llm_meta['training_info']['train_loss']:.6f}")
    print(f"Explicit - val loss:   {explicit_meta['training_info']['val_loss']:.6f}")
    print(f"LLM      - val loss:   {llm_meta['training_info']['val_loss']:.6f}")
    
    # Check if explicit model had training issues
    if explicit_meta['training_info']['train_loss'] > 0.01:
        print("🚨 EXPLICIT MODEL: High training loss may indicate poor training!")
    
    if explicit_meta['training_info']['val_loss'] < explicit_meta['training_info']['train_loss']:
        print("⚠️  EXPLICIT MODEL: Val loss lower than train loss (unusual)")
    
    return explicit_meta, llm_meta

def check_explicit_model_internals():
    """Load and inspect explicit model's internal behavior."""
    print("\n" + "="*70)
    print("INSPECTING EXPLICIT MODEL INTERNALS")
    print("="*70)
    
    # Load the model
    checkpoint_path = "checkpoints/precovid/explicit/20100101-20191231/best.pt"
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model = ExplicitConditioningDDPM(60, 6, 128)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Check if model weights are reasonable
    print("MODEL WEIGHT ANALYSIS:")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    weight_stats = []
    for name, param in model.named_parameters():
        if 'weight' in name:
            mean_val = param.mean().item()
            std_val = param.std().item()
            max_val = param.abs().max().item()
            
            weight_stats.append({
                'layer': name,
                'mean': mean_val,
                'std': std_val,
                'max_abs': max_val
            })
    
    # Look for problematic layers
    print("\nWEIGHT STATISTICS (first 10 layers):")
    for stat in weight_stats[:10]:
        print(f"{stat['layer'][:40]:<40} mean={stat['mean']:8.4f} std={stat['std']:8.4f} max={stat['max_abs']:8.2f}")
        
        if stat['max_abs'] > 100:
            print(f"  🚨 {stat['layer']} has very large weights!")
        
        if abs(stat['mean']) > 10:
            print(f"  ⚠️  {stat['layer']} has large mean bias!")
    
    # Test with zero input to see base behavior
    print("\nTESTING MODEL WITH ZERO INPUT:")
    batch_size = 4
    x_zero = torch.zeros(batch_size, 1, 60)
    t_zero = torch.zeros(batch_size, 1)
    cond_zero = torch.zeros(batch_size, 6)
    cond_zero[:, 0] = 1  # Up-Low regime
    
    with torch.no_grad():
        output_zero = model(x_zero, t_zero, cond_zero)
        
    print(f"Zero input -> Model output: mean={output_zero.mean():.6f}, std={output_zero.std():.6f}")
    print(f"Zero input -> Output range: [{output_zero.min():.6f}, {output_zero.max():.6f}]")
    
    if abs(output_zero.mean()) > 1:
        print("🚨 CRITICAL: Model has large output bias even with zero input!")
        print("This suggests the model learned to predict non-zero values systematically")
        return False
    
    return True

def test_data_preprocessing_match():
    """Check if explicit model was trained on the same data preprocessing as expected."""
    print("\n" + "="*70)
    print("CHECKING DATA PREPROCESSING CONSISTENCY")
    print("="*70)
    
    # Load raw data and recreate preprocessing
    data = pd.read_csv("data/sp500_data.csv", index_col=0, parse_dates=True)
    data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
    
    # Calculate returns
    returns = np.log(data['Close'] / data['Close'].shift(1)).dropna()
    precovid_returns = returns['2010-01-01':'2019-12-31']
    
    print(f"Expected training data:")
    print(f"  Mean: {precovid_returns.mean():.8f}")
    print(f"  Std:  {precovid_returns.std():.8f}")
    print(f"  Min:  {precovid_returns.min():.8f}")
    print(f"  Max:  {precovid_returns.max():.8f}")
    
    # Compare with saved metadata
    with open("checkpoints/precovid/explicit/20100101-20191231/meta.json") as f:
        meta = json.load(f)
    
    saved_stats = meta['data_info']['train_stats']
    print(f"\nSaved in checkpoint:")
    print(f"  Mean: {saved_stats['mean']:.8f}")
    print(f"  Std:  {saved_stats['std']:.8f}")
    print(f"  Min:  {saved_stats['min']:.8f}")
    print(f"  Max:  {saved_stats['max']:.8f}")
    
    # Check for exact match
    mean_diff = abs(precovid_returns.mean() - saved_stats['mean'])
    std_diff = abs(precovid_returns.std() - saved_stats['std'])
    
    if mean_diff > 1e-6 or std_diff > 1e-6:
        print(f"🚨 MISMATCH: Training data preprocessing differs!")
        print(f"  Mean diff: {mean_diff:.10f}")
        print(f"  Std diff:  {std_diff:.10f}")
        return False
    else:
        print("✅ Training data preprocessing matches expected")
        return True

def test_explicit_vs_llm_forward_pass():
    """Compare explicit model forward pass with LLM model for same scale input."""
    print("\n" + "="*70)
    print("COMPARING FORWARD PASS: EXPLICIT VS LLM")
    print("="*70)
    
    # Load both models
    explicit_checkpoint = torch.load("checkpoints/precovid/explicit/20100101-20191231/best.pt", map_location='cpu')
    explicit_model = ExplicitConditioningDDPM(60, 6, 128)
    explicit_model.load_state_dict(explicit_checkpoint['model_state_dict'])
    explicit_model.eval()
    
    # Load LLM model for comparison
    sys.path.append('src')
    from llm_conditioned_diffusion import ConditionedDiffusionModel
    
    llm_checkpoint = torch.load("checkpoints/precovid/llm/20100101-20191231/best.pt", map_location='cpu')
    llm_model = ConditionedDiffusionModel(60, 32, 128)
    llm_model.load_state_dict(llm_checkpoint['model_state_dict'])
    llm_model.eval()
    
    # Test with same scale input (training data scale)
    batch_size = 4
    seq_len = 60
    
    # Training scale input
    training_std = 0.009317
    x_input = torch.randn(batch_size, 60) * training_std
    t_input = torch.ones(batch_size, 1) * 0.5
    
    # Explicit conditioning  
    explicit_cond = torch.zeros(batch_size, 6)
    explicit_cond[:, 0] = 1  # Up-Low
    
    # LLM conditioning (dummy)
    llm_cond = torch.randn(batch_size, 32) * 0.1
    
    with torch.no_grad():
        # Explicit model
        x_explicit = x_input.unsqueeze(1)  # [B, 1, T]
        explicit_output = explicit_model(x_explicit, t_input, explicit_cond)
        
        # LLM model  
        x_llm = x_input.unsqueeze(-1)  # [B, T, 1]
        llm_output = llm_model(x_llm, t_input, llm_cond)
    
    print(f"Input scale: mean={x_input.mean():.6f}, std={x_input.std():.6f}")
    print(f"Explicit output: mean={explicit_output.mean():.6f}, std={explicit_output.std():.6f}")
    print(f"LLM output:      mean={llm_output.mean():.6f}, std={llm_output.std():.6f}")
    
    # Check for scale explosion
    input_scale = x_input.std()
    explicit_scale = explicit_output.std()
    llm_scale = llm_output.std()
    
    explicit_ratio = explicit_scale / input_scale
    llm_ratio = llm_scale / input_scale
    
    print(f"\nScale amplification:")
    print(f"Explicit: {explicit_ratio:.2f}x input scale")
    print(f"LLM:      {llm_ratio:.2f}x input scale")
    
    if explicit_ratio > 100:
        print("🚨 EXPLICIT MODEL: Massive scale amplification in forward pass!")
        return False
    elif explicit_ratio > 10:
        print("⚠️  EXPLICIT MODEL: Large scale amplification")
        return False
    else:
        print("✅ Both models have reasonable scale behavior")
        return True

if __name__ == "__main__":
    print("DEEP INVESTIGATION: EXPLICIT MODEL TRAINING ISSUES")
    print("=" * 70)
    
    # Run all diagnostic tests
    explicit_meta, llm_meta = compare_model_checkpoints()
    model_ok = check_explicit_model_internals()
    data_ok = test_data_preprocessing_match()
    forward_ok = test_explicit_vs_llm_forward_pass()
    
    print("\n" + "="*70)
    print("DIAGNOSIS SUMMARY")
    print("="*70)
    
    issues_found = []
    if not model_ok:
        issues_found.append("Model weights/bias issues")
    if not data_ok:
        issues_found.append("Training data preprocessing mismatch")
    if not forward_ok:
        issues_found.append("Forward pass scale amplification")
    
    if issues_found:
        print("🚨 ISSUES IDENTIFIED:")
        for issue in issues_found:
            print(f"   • {issue}")
        print("\nRECOMMENDATION: Re-train explicit model from scratch")
    else:
        print("⚠️  No obvious training issues found")
        print("The scale problem may be in the sampling loop implementation itself")
    
    print("="*70)
