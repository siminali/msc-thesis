#!/usr/bin/env python3
"""
DDPM (Denoising Diffusion Probabilistic Model) Implementation and Evaluation

This script implements a DDPM for synthetic financial return generation.
It serves as a modern generative baseline for comparison with classical models.

Author: Simin Ali
Thesis: Diffusion Models in Generative AI for Financial Data Synthesis and Risk Management
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """Load and prepare S&P 500 data."""
    print("📊 Loading S&P 500 data for DDPM...")
    
    # Load data from data folder
    data = pd.read_csv("../data/sp500_data.csv", index_col=0, parse_dates=True)
    # Remove the header row if it exists
    if data.index[0] == 'Ticker':
        data = data.iloc[1:]
    data.index = pd.to_datetime(data.index)
    data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
    data = data[['Close']]
    
    # Compute log returns
    returns = np.log(data['Close'] / data['Close'].shift(1)).dropna() * 100
    
    print(f"✅ Data loaded: {len(returns)} observations")
    print(f"   Date range: {data.index.min()} to {data.index.max()}")
    
    return returns

def create_sequences(returns, sequence_length=60):
    """Create sequences for DDPM training."""
    print("🔄 Creating training sequences...")
    
    # Create sequences of specified length
    X = np.array([returns[i:i+sequence_length].values for i in range(len(returns) - sequence_length)])
    X = X[..., np.newaxis]  # Add channel dimension
    
    print(f"✅ Created {X.shape[0]} sequences of length {sequence_length}")
    
    return X

def define_ddpm_model(seq_len):
    """Define the DDPM denoising model."""
    print("🏗️  Defining DDPM model...")
    
    class DenoiseMLP(nn.Module):
        def __init__(self, seq_len):
            super().__init__()
            self.model = nn.Sequential(
                nn.Flatten(),
                nn.Linear(seq_len, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, seq_len)
            )

        def forward(self, x):
            return self.model(x).unsqueeze(-1)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DenoiseMLP(seq_len).to(device)
    
    print(f"✅ DDPM model defined (device: {device})")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, device

def train_ddpm_model(model, X, device, epochs=100):
    """Train the DDPM model."""
    print("🎯 Training DDPM model...")
    
    # Convert to torch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    dataset = TensorDataset(X_tensor)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # Define diffusion schedule
    timesteps = 50  # Reduced for faster training
    betas = torch.linspace(1e-4, 0.02, timesteps).to(device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    
    # Forward process function
    def q_sample(x_start, t, noise):
        sqrt_alpha_bar = torch.sqrt(alpha_bars[t])[:, None, None]
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bars[t])[:, None, None]
        return sqrt_alpha_bar * x_start + sqrt_one_minus_alpha_bar * noise
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        for batch in loader:
            x = batch[0].to(device)
            
            # Sample random timesteps
            t = torch.randint(0, timesteps, (x.size(0),), device=device).long()
            
            # Sample noise and generate x_t
            noise = torch.randn_like(x)
            x_t = q_sample(x, t, noise)
            
            # Predict noise
            predicted_noise = model(x_t)
            loss = loss_fn(predicted_noise, noise)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        if epoch % 20 == 0:
            print(f"   Epoch {epoch}: Loss = {epoch_loss/len(loader):.6f}")
    
    print("✅ DDPM training completed!")
    return model, (betas, alphas, alpha_bars)

def generate_synthetic_data(model, device, betas, alphas, alpha_bars, n_samples=1000):
    """Generate synthetic data using the trained DDPM."""
    print("🎨 Generating synthetic data...")
    
    @torch.no_grad()
    def sample_ddpm(model, n_samples, seq_len=60):
        model.eval()
        x = torch.randn(n_samples, seq_len, 1).to(device)
        
        for t in reversed(range(len(betas))):
            beta_t = betas[t]
            alpha_t = alphas[t]
            alpha_bar_t = alpha_bars[t]
            
            predicted_noise = model(x)
            
            coef1 = 1 / torch.sqrt(alpha_t)
            coef2 = (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)
            mean = coef1 * (x - coef2 * predicted_noise)
            
            if t > 0:
                noise = torch.randn_like(x)
                sigma = torch.sqrt(beta_t)
                x = mean + sigma * noise
            else:
                x = mean
        
        return x.cpu()
    
    # Generate samples
    samples = sample_ddpm(model, n_samples)
    synthetic_data = samples.numpy().squeeze()
    
    print(f"✅ Generated {n_samples} synthetic sequences")
    
    return synthetic_data

def evaluate_ddpm_performance(real_data, synthetic_data):
    """Evaluate DDPM performance."""
    print("📈 Evaluating DDPM performance...")
    
    # Flatten data for evaluation
    real_flat = real_data.flatten()
    synthetic_flat = synthetic_data.flatten()
    
    # Basic statistics
    stats_dict = {
        'Real_Mean': np.mean(real_flat),
        'Synthetic_Mean': np.mean(synthetic_flat),
        'Real_Std': np.std(real_flat),
        'Synthetic_Std': np.std(synthetic_flat),
        'Real_Skewness': stats.skew(real_flat),
        'Synthetic_Skewness': stats.skew(synthetic_flat),
        'Real_Kurtosis': stats.kurtosis(real_flat),
        'Synthetic_Kurtosis': stats.kurtosis(synthetic_flat)
    }
    
    # KS test
    ks_stat, ks_pvalue = stats.ks_2samp(real_flat, synthetic_flat)
    stats_dict['KS_Statistic'] = ks_stat
    stats_dict['KS_pvalue'] = ks_pvalue
    
    print(f"✅ DDPM Evaluation Results:")
    print(f"   KS Statistic: {ks_stat:.4f} (p-value: {ks_pvalue:.4f})")
    print(f"   Real vs Synthetic Mean: {stats_dict['Real_Mean']:.4f} vs {stats_dict['Synthetic_Mean']:.4f}")
    print(f"   Real vs Synthetic Std: {stats_dict['Real_Std']:.4f} vs {stats_dict['Synthetic_Std']:.4f}")
    
    return stats_dict

def save_results(synthetic_data, stats_dict):
    """Save DDPM results."""
    print("💾 Saving DDPM results...")
    
    # Save synthetic data
    np.save("../results/ddpm_returns.npy", synthetic_data)
    
    # Save statistics
    import json
    # Convert numpy types to native Python types for JSON serialization
    json_stats = {}
    for key, value in stats_dict.items():
        if isinstance(value, (np.integer, np.floating)):
            json_stats[key] = float(value)
        else:
            json_stats[key] = value
    
    with open("../results/ddpm_evaluation/ddpm_stats.json", 'w') as f:
        json.dump(json_stats, f, indent=2)
    
    print("✅ DDPM results saved to results/ directory")

def create_plots(real_data, synthetic_data):
    """Create DDPM evaluation plots."""
    print("📊 Creating DDPM evaluation plots...")
    
    # Create results directory
    import os
    os.makedirs("../results/ddpm_evaluation/plots", exist_ok=True)
    
    # Plot 1: Distribution comparison
    plt.figure(figsize=(12, 6))
    plt.hist(real_data.flatten(), bins=50, alpha=0.7, density=True, label='Real Data')
    plt.hist(synthetic_data.flatten(), bins=50, alpha=0.7, density=True, label='DDPM Synthetic')
    plt.title('DDPM: Real vs Synthetic Data Distribution')
    plt.xlabel('Returns')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("../results/ddpm_evaluation/plots/ddpm_distribution_comparison.pdf", dpi=300, bbox_inches='tight')
    plt.savefig("../results/ddpm_evaluation/plots/ddpm_distribution_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Sample sequences
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for i in range(4):
        row, col = i // 2, i % 2
        if i < 2:
            axes[row, col].plot(real_data[i], label='Real')
            axes[row, col].set_title(f'Real Sequence {i+1}')
        else:
            axes[row, col].plot(synthetic_data[i-2], label='Synthetic')
            axes[row, col].set_title(f'Synthetic Sequence {i-1}')
        axes[row, col].legend()
        axes[row, col].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("../results/ddpm_evaluation/plots/ddpm_sequence_comparison.pdf", dpi=300, bbox_inches='tight')
    plt.savefig("../results/ddpm_evaluation/plots/ddpm_sequence_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ DDPM plots saved to results/ddpm_evaluation/plots/")

def main():
    """Main function to run DDPM evaluation."""
    print("🎓 DDPM Model Evaluation")
    print("=" * 50)
    
    try:
        # Load and prepare data
        returns = load_and_prepare_data()
        
        # Create sequences
        X = create_sequences(returns)
        
        # Define model
        model, device = define_ddpm_model(X.shape[1])
        
        # Train model
        model, diffusion_params = train_ddpm_model(model, X, device, epochs=50)
        
        # Generate synthetic data
        synthetic_data = generate_synthetic_data(model, device, *diffusion_params)
        
        # Evaluate performance
        stats_dict = evaluate_ddpm_performance(X, synthetic_data)
        
        # Save results
        save_results(synthetic_data, stats_dict)
        
        # Create plots
        create_plots(X, synthetic_data)
        
        print("\n🎉 DDPM evaluation completed successfully!")
        print("📁 Results saved in: results/ddpm_evaluation/")
        
    except Exception as e:
        print(f"❌ Error in DDPM evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
