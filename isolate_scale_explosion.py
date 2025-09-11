#!/usr/bin/env python3
"""
Isolate exactly where the scale explosion occurs in DDPM sampling.
"""
import torch
import numpy as np
import sys
import json
from pathlib import Path

# Add src to path
sys.path.append('src/novelty models')
from explicit_cond_ddpm import ExplicitConditioningDDPM, ExplicitConditioningTrainer

# Add current directory to path
sys.path.append('.')

def trace_ddpm_sampling_step_by_step():
    """Trace through our DDPM sampling implementation step by step to find scale explosion."""
    print("="*70)
    print("TRACING DDPM SAMPLING STEP BY STEP")
    print("="*70)
    
    # Load the explicit model
    checkpoint_path = "checkpoints/precovid/explicit/20100101-20191231/best.pt"
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model = ExplicitConditioningDDPM(60, 6, 128)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Get training data scale
    with open("checkpoints/precovid/explicit/20100101-20191231/meta.json") as f:
        meta = json.load(f)
    training_std = meta['data_info']['train_stats']['std']
    
    print(f"Training data scale: {training_std:.6f}")
    
    # Set up initial state
    batch_size = 4
    seq_len = 60
    num_timesteps = 1000
    
    # Initialize conditioning
    conditioning = torch.zeros(batch_size, 6)
    conditioning[:, 0] = 1  # Up-Low regime
    
    # Initialize noise at training scale (our fix)
    x = torch.randn(batch_size, 1, seq_len) * training_std
    print(f"Initial noise: mean={x.mean():.6f}, std={x.std():.6f}, range=[{x.min():.6f}, {x.max():.6f}]")
    
    # Create exact same beta schedule as our CheckpointSampler
    def cosine_beta_schedule(timesteps):
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + 0.008) / (1 + 0.008) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, 0.0001, 0.9999)

    betas = cosine_beta_schedule(num_timesteps)
    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    
    print(f"Beta schedule: min={betas.min():.6f}, max={betas.max():.6f}")
    
    # Test just a few steps to see where explosion happens
    test_timesteps = [999, 950, 900, 850, 800, 750, 500, 250, 100, 50, 10, 0]
    
    for i, t_idx in enumerate(test_timesteps):
        print(f"\n--- TIMESTEP {t_idx} (step {i+1}/{len(test_timesteps)}) ---")
        
        t = t_idx
        alpha_bar_t = alphas_cumprod[t]
        alpha_bar_tm1 = alphas_cumprod[t-1] if t > 0 else torch.ones(1)
        
        print(f"alpha_bar_t: {alpha_bar_t:.8f}")
        print(f"alpha_bar_tm1: {alpha_bar_tm1:.8f}")
        
        # Get model prediction
        t_normalized = (t / num_timesteps) * torch.ones(batch_size, 1)
        
        with torch.no_grad():
            predicted_noise = model(x, t_normalized, conditioning)
        
        print(f"x before step: mean={x.mean():.6f}, std={x.std():.6f}")
        print(f"predicted_noise: mean={predicted_noise.mean():.6f}, std={predicted_noise.std():.6f}")
        
        # Apply DDIM step (same as our implementation)
        x_denoised = x / torch.sqrt(alpha_bar_t).view(1, 1, 1) - torch.sqrt(1/alpha_bar_t - 1).view(1, 1, 1) * predicted_noise
        x_new = torch.sqrt(alpha_bar_tm1).view(1, 1, 1) * x_denoised + torch.sqrt(1 - alpha_bar_tm1).view(1, 1, 1) * predicted_noise
        
        print(f"x_denoised: mean={x_denoised.mean():.6f}, std={x_denoised.std():.6f}")
        print(f"x_new: mean={x_new.mean():.6f}, std={x_new.std():.6f}")
        
        # Check for explosion
        scale_change = x_new.std() / x.std()
        print(f"Scale change: {scale_change:.2f}x")
        
        if scale_change > 10:
            print(f"🚨 SCALE EXPLOSION at timestep {t_idx}!")
            print(f"   1/sqrt(alpha_bar_t) = {1/torch.sqrt(alpha_bar_t):.2f}")
            print(f"   sqrt(1/alpha_bar_t - 1) = {torch.sqrt(1/alpha_bar_t - 1):.2f}")
            break
        
        if abs(x_new.mean()) > 1000:
            print(f"🚨 MEAN EXPLOSION at timestep {t_idx}!")
            break
            
        x = x_new
        
        # Stop if we've gone too far 
        if i >= 5 and abs(x.mean()) > 100:
            print("⚠️ Stopping early due to scale growth")
            break
    
    final_samples = x.squeeze(1).numpy()
    print(f"\nFINAL SAMPLES:")
    print(f"Shape: {final_samples.shape}")
    print(f"Mean: {final_samples.mean():.6f}")
    print(f"Std: {final_samples.std():.6f}")
    print(f"Range: [{final_samples.min():.6f}, {final_samples.max():.6f}]")

def compare_with_working_trainer():
    """Compare our implementation with the working ExplicitConditioningTrainer."""
    print("\n" + "="*70)
    print("COMPARING WITH WORKING TRAINER IMPLEMENTATION")
    print("="*70)
    
    # Load model
    checkpoint_path = "checkpoints/precovid/explicit/20100101-20191231/best.pt"
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model = ExplicitConditioningDDPM(60, 6, 128)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    trainer = ExplicitConditioningTrainer(model, device="cpu")
    
    # Test with trainer (known to work)
    conditioning = torch.zeros(4, 6)
    conditioning[:, 0] = 1
    
    print("TRAINER IMPLEMENTATION:")
    with torch.no_grad():
        trainer_samples = trainer.sample(
            conditioning, 
            num_samples=4, 
            sampler="ddim",
            sample_steps=20  # Reduced for comparison
        )
    
    trainer_np = trainer_samples.squeeze().cpu().numpy()
    print(f"Trainer samples: mean={trainer_np.mean():.6f}, std={trainer_np.std():.6f}")
    print(f"Trainer range: [{trainer_np.min():.6f}, {trainer_np.max():.6f}]")
    
    # Check what the trainer does differently
    print("\nTRAINER CONFIGURATION:")
    print(f"Number of timesteps: {trainer.num_timesteps}")
    print(f"Beta schedule: {getattr(trainer, 'beta_schedule', 'unknown')}")
    
    # Check if trainer uses different noise initialization
    print("\nThe key difference might be in noise initialization or sampling schedule")

if __name__ == "__main__":
    print("ISOLATING SCALE EXPLOSION IN DDPM SAMPLING")
    print("=" * 70)
    
    # Run step-by-step tracing
    trace_ddpm_sampling_step_by_step()
    
    # Compare with working trainer
    compare_with_working_trainer()
    
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    print("If scale explosion occurs at high timesteps (t=999, t=950), the issue is")
    print("likely in the 1/sqrt(alpha_bar_t) term which becomes very large for t→1000.")
    print("If it occurs at low timesteps, the issue is elsewhere.")
    print("="*70)
