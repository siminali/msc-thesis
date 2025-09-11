#!/usr/bin/env python3
"""
Debug what timesteps the working ExplicitConditioningTrainer actually uses.
"""
import torch
import sys
from pathlib import Path

# Add src to path  
sys.path.append('src/novelty models')
from explicit_cond_ddpm import ExplicitConditioningDDPM, ExplicitConditioningTrainer

def investigate_trainer_sampling():
    """Investigate the exact sampling schedule used by ExplicitConditioningTrainer."""
    print("="*70)
    print("INVESTIGATING TRAINER SAMPLING SCHEDULE")
    print("="*70)
    
    # Load model
    checkpoint_path = "checkpoints/precovid/explicit/20100101-20191231/best.pt"
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model = ExplicitConditioningDDPM(60, 6, 128)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    trainer = ExplicitConditioningTrainer(model, device="cpu")
    
    # Check trainer's default parameters
    print(f"Trainer num_timesteps: {trainer.num_timesteps}")
    print(f"Trainer beta schedule: {getattr(trainer, 'beta_schedule', 'cosine (default)')}")
    
    # Check what sample_steps the trainer uses by default
    conditioning = torch.zeros(1, 6)
    conditioning[:, 0] = 1
    
    # Let's look at the trainer's sample method to see what it does
    print("\nINVESTIGATING TRAINER'S SAMPLE METHOD:")
    
    # Check the default sample_steps parameter
    import inspect
    sample_signature = inspect.signature(trainer.sample)
    print(f"Sample method signature: {sample_signature}")
    
    # Let's see what happens with default parameters
    with torch.no_grad():
        # Try with very conservative sampling
        samples_10 = trainer.sample(conditioning, num_samples=1, sampler="ddim", sample_steps=10)
        samples_5 = trainer.sample(conditioning, num_samples=1, sampler="ddim", sample_steps=5)
        samples_2 = trainer.sample(conditioning, num_samples=1, sampler="ddim", sample_steps=2)
        
    print(f"\n10 steps: mean={samples_10.mean():.6f}, std={samples_10.std():.6f}")
    print(f"5 steps:  mean={samples_5.mean():.6f}, std={samples_5.std():.6f}")
    print(f"2 steps:  mean={samples_2.mean():.6f}, std={samples_2.std():.6f}")
    
    return trainer

def test_ultra_conservative_sampling():
    """Test with extremely conservative sampling (only a few steps)."""
    print("\n" + "="*70)
    print("TESTING ULTRA-CONSERVATIVE SAMPLING")
    print("="*70)
    
    # Modify our CheckpointSampler to use only 5 steps
    sys.path.append('.')
    from checkpoint_loader_sampler import CheckpointSampler
    
    # Load sampler
    sampler = CheckpointSampler("checkpoints/precovid/explicit/20100101-20191231")
    
    # Create minimal conditioning
    import numpy as np
    conditioning = np.zeros((4, 6))
    conditioning[:, 0] = 1  # Up-Low
    
    # We need to temporarily modify the sampling to use very few steps
    # Let's test what happens with just 2-3 denoising steps
    print("Testing ultra-conservative sampling (this will require code modification)")
    
    # For now, let's see what our current 20-step sampling produces
    samples = sampler.generator.generate_ddpm_samples(conditioning, 4, 60)
    print(f"Current 20-step sampling: mean={samples.mean():.6f}, std={samples.std():.6f}")
    
def analyze_coefficient_explosion():
    """Analyze which specific timesteps cause coefficient explosion."""
    print("\n" + "="*70)
    print("ANALYZING COEFFICIENT EXPLOSION BY TIMESTEP")
    print("="*70)
    
    num_timesteps = 50
    
    # Create beta schedule
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
    
    # Check coefficients for our current 20-step sampling schedule
    sampling_timesteps = torch.linspace(num_timesteps - 1, 0, 20, dtype=torch.long)
    
    print("Checking coefficients for 20-step schedule:")
    problematic_steps = []
    
    for i, t_idx in enumerate(sampling_timesteps):
        t = int(t_idx.item())
        alpha_bar_t = alphas_cumprod[t]
        
        coeff1 = 1 / torch.sqrt(alpha_bar_t)
        coeff2 = torch.sqrt(1/alpha_bar_t - 1)
        
        print(f"Step {i+1:2d}, t={t:2d}: alpha_bar={alpha_bar_t:.8f}, 1/sqrt(alpha)={coeff1:.2f}, sqrt(1/alpha-1)={coeff2:.2f}")
        
        if coeff1 > 100 or coeff2 > 100:
            problematic_steps.append((i+1, t, coeff1.item(), coeff2.item()))
    
    if problematic_steps:
        print(f"\n🚨 FOUND {len(problematic_steps)} PROBLEMATIC STEPS:")
        for step, t, c1, c2 in problematic_steps:
            print(f"   Step {step}, t={t}: coefficients {c1:.0f}x, {c2:.0f}x")
        print("\nSOLUTION: Use even fewer steps or skip high timesteps entirely")
    else:
        print("\n✅ No problematic coefficients found")

if __name__ == "__main__":
    print("DEBUGGING TRAINER TIMESTEPS AND COEFFICIENTS")
    print("=" * 70)
    
    trainer = investigate_trainer_sampling()
    test_ultra_conservative_sampling()
    analyze_coefficient_explosion()
    
    print("\n" + "="*70)
    print("CONCLUSION & NEXT STEPS")
    print("="*70)
    print("1. Check if trainer uses fewer steps than we assumed")
    print("2. Identify specific problematic timesteps in our schedule")
    print("3. Use ultra-conservative sampling (2-5 steps max)")
    print("4. Or avoid high timesteps entirely (start from t=20 instead of t=49)")
    print("="*70)
