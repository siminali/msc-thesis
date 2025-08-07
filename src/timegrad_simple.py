#!/usr/bin/env python3
"""
TimeGrad Model Implementation and Evaluation

This script implements a TimeGrad model for autoregressive diffusion-based forecasting.
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
    print("📊 Loading S&P 500 data for TimeGrad...")
    
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
    """Create sequences for TimeGrad training."""
    print("🔄 Creating training sequences...")
    
    # Create sequences of specified length
    X = np.array([returns[i:i+sequence_length].values for i in range(len(returns) - sequence_length)])
    X = X[..., np.newaxis]  # Add channel dimension
    
    print(f"✅ Created {X.shape[0]} sequences of length {sequence_length}")
    
    return X

def define_timegrad_model(input_dim=1, hidden_dim=64):
    """Define the TimeGrad model."""
    print("🏗️  Defining TimeGrad model...")
    
    class TimeGrad(nn.Module):
        def __init__(self, input_dim=1, hidden_dim=64, num_layers=1):
            super(TimeGrad, self).__init__()
            self.embedding = nn.Linear(input_dim, hidden_dim)
            self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True)
            self.denoise_head = nn.Linear(hidden_dim, input_dim)

        def forward(self, x):
            x_embed = self.embedding(x)
            out, _ = self.gru(x_embed)
            return self.denoise_head(out)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TimeGrad(input_dim, hidden_dim).to(device)
    
    print(f"✅ TimeGrad model defined (device: {device})")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, device

def train_timegrad_model(model, X, device, epochs=50):
    """Train the TimeGrad model."""
    print("🎯 Training TimeGrad model...")
    
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
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        
        for (x_0,) in loader:
            x_0 = x_0.to(device)
            B, L, D = x_0.shape
            
            # Sample random timesteps
            t = torch.randint(0, timesteps, (B,), dtype=torch.long).to(device)
            alpha_bar = alpha_bars[t].view(B, 1, 1).expand(B, L, 1)
            
            # Add noise
            noise = torch.randn_like(x_0)
            one_minus_ab = 1.0 - alpha_bar
            x_t = torch.sqrt(alpha_bar) * x_0 + torch.sqrt(one_minus_ab) * noise
            
            # Predict noise
            predicted_noise = model(x_t)
            loss = loss_fn(predicted_noise, noise)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if epoch % 10 == 0:
            print(f"   Epoch {epoch}: Loss = {total_loss/len(loader):.6f}")
    
    print("✅ TimeGrad training completed!")
    return model, (betas, alphas, alpha_bars)

def generate_synthetic_data(model, device, betas, alphas, alpha_bars, n_samples=1000):
    """Generate synthetic data using the trained TimeGrad model."""
    print("🎨 Generating synthetic data...")
    
    @torch.no_grad()
    def sample_timegrad(model, n_samples, seq_len=60):
        model.eval()
        x_t = torch.randn(n_samples, seq_len, 1).to(device)
        
        for t in reversed(range(len(betas))):
            alpha = alphas[t]
            beta = betas[t]
            alpha_bar = alpha_bars[t]
            
            z = torch.randn_like(x_t) if t > 0 else 0
            predicted_noise = model(x_t)
            
            x_t = (1 / torch.sqrt(alpha)) * (
                x_t - ((1 - alpha) / torch.sqrt(1 - alpha_bar)) * predicted_noise
            ) + torch.sqrt(beta) * z
        
        return x_t.cpu().squeeze(-1)
    
    # Generate samples
    samples = sample_timegrad(model, n_samples)
    synthetic_data = samples.numpy()
    
    print(f"✅ Generated {n_samples} synthetic sequences")
    
    return synthetic_data

def evaluate_timegrad_performance(real_data, synthetic_data):
    """Evaluate TimeGrad performance."""
    print("📈 Evaluating TimeGrad performance...")
    
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
    
    print(f"✅ TimeGrad Evaluation Results:")
    print(f"   KS Statistic: {ks_stat:.4f} (p-value: {ks_pvalue:.4f})")
    print(f"   Real vs Synthetic Mean: {stats_dict['Real_Mean']:.4f} vs {stats_dict['Synthetic_Mean']:.4f}")
    print(f"   Real vs Synthetic Std: {stats_dict['Real_Std']:.4f} vs {stats_dict['Synthetic_Std']:.4f}")
    
    return stats_dict

def save_results(synthetic_data, stats_dict):
    """Save TimeGrad results."""
    print("💾 Saving TimeGrad results...")
    
    # Save synthetic data
    np.save("../results/timegrad_returns.npy", synthetic_data)
    
    # Save statistics
    import json
    # Convert numpy types to native Python types for JSON serialization
    json_stats = {}
    for key, value in stats_dict.items():
        if isinstance(value, (np.integer, np.floating)):
            json_stats[key] = float(value)
        else:
            json_stats[key] = value
    
    with open("../results/timegrad_evaluation/timegrad_stats.json", 'w') as f:
        json.dump(json_stats, f, indent=2)
    
    print("✅ TimeGrad results saved to results/ directory")

def create_plots(real_data, synthetic_data):
    """Create TimeGrad evaluation plots."""
    print("📊 Creating TimeGrad evaluation plots...")
    
    # Create results directory
    import os
    os.makedirs("../results/timegrad_evaluation/plots", exist_ok=True)
    
    # Plot 1: Distribution comparison
    plt.figure(figsize=(12, 6))
    plt.hist(real_data.flatten(), bins=50, alpha=0.7, density=True, label='Real Data')
    plt.hist(synthetic_data.flatten(), bins=50, alpha=0.7, density=True, label='TimeGrad Synthetic')
    plt.title('TimeGrad: Real vs Synthetic Data Distribution')
    plt.xlabel('Returns')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("../results/timegrad_evaluation/plots/timegrad_distribution_comparison.pdf", dpi=300, bbox_inches='tight')
    plt.savefig("../results/timegrad_evaluation/plots/timegrad_distribution_comparison.png", dpi=300, bbox_inches='tight')
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
    plt.savefig("../results/timegrad_evaluation/plots/timegrad_sequence_comparison.pdf", dpi=300, bbox_inches='tight')
    plt.savefig("../results/timegrad_evaluation/plots/timegrad_sequence_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ TimeGrad plots saved to results/timegrad_evaluation/plots/")

def main():
    """Main function to run TimeGrad evaluation."""
    print("🎓 TimeGrad Model Evaluation")
    print("=" * 50)
    
    try:
        # Load and prepare data
        returns = load_and_prepare_data()
        
        # Create sequences
        X = create_sequences(returns)
        
        # Define model
        model, device = define_timegrad_model()
        
        # Train model
        model, diffusion_params = train_timegrad_model(model, X, device, epochs=30)
        
        # Generate synthetic data
        synthetic_data = generate_synthetic_data(model, device, *diffusion_params)
        
        # Evaluate performance
        stats_dict = evaluate_timegrad_performance(X, synthetic_data)
        
        # Save results
        save_results(synthetic_data, stats_dict)
        
        # Create plots
        create_plots(X, synthetic_data)
        
        print("\n🎉 TimeGrad evaluation completed successfully!")
        print("📁 Results saved in: results/timegrad_evaluation/")
        
    except Exception as e:
        print(f"❌ Error in TimeGrad evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
