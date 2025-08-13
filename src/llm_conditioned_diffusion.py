#!/usr/bin/env python3
"""
LLM-Conditioned Diffusion Model for Financial Data Synthesis
Based on supervisor feedback: Using LLM embeddings from internet data as conditioning vectors

Author: Simin Ali
Thesis: Diffusion Models in Generative AI for Financial Data Synthesis and Risk Management
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel
from datetime import datetime, timedelta
import json
import os
from tqdm import tqdm
import warnings
import scipy.stats
warnings.filterwarnings('ignore')

# Global constants for consistency
SEQ_LEN = 60
EMBEDDING_DIM = 768  # DistilBERT native dimension

class LLMConditioningModule:
    """
    Module to generate conditioning vectors from internet data using LLMs.
    Implements the approach suggested by supervisor: feed internet data into LLM,
    get latent embeddings, use as conditioning for diffusion model.
    """
    
    def __init__(self, model_name="distilbert-base-uncased", device="cpu"):
        """
        Initialize LLM for text embedding generation.
        
        Args:
            model_name: Hugging Face model name for text processing
            device: Device to run tokenizer and model on
        """
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model = self.model.to(device)
        self.model.eval()
        
        # Freeze LLM parameters (we only use it for embeddings)
        for param in self.model.parameters():
            param.requires_grad = False
            
        print(f"LLM Conditioning Module initialized with {model_name} on {device}")
    
    def get_market_sentiment_data(self, start_date, end_date):
        """
        Generate synthetic market sentiment data for demonstration.
        In practice, this would fetch real market news/sentiment data.
        
        Args:
            start_date: Start date for sentiment data
            end_date: End date for sentiment data
            
        Returns:
            List of sentiment strings indexed by date
        """
        sentiment_data = []
        current_date = start_date
        
        # Base market sentiments
        base_sentiments = [
            "Market shows strong bullish momentum with positive earnings reports",
            "Volatility increases as uncertainty grows in global markets",
            "Central bank policy decisions drive market sentiment",
            "Tech sector leads market gains on innovation news",
            "Economic indicators suggest stable growth trajectory"
        ]
        
        while current_date <= end_date:
            # Randomly select and modify sentiment based on date
            sentiment = np.random.choice(base_sentiments)
            
            # Add date-specific context
            sentiment += f" on {current_date.strftime('%Y-%m-%d')}"
            
            sentiment_data.append(sentiment)
            current_date += timedelta(days=1)
        
        return sentiment_data
    
    def generate_embeddings(self, text_data, max_length=512, batch_size=32):
        """
        Generate embeddings from text data using the LLM.
        
        Args:
            text_data: List of text strings
            max_length: Maximum sequence length for tokenization
            batch_size: Batch size for processing
            
        Returns:
            numpy array of embeddings
        """
        embeddings = []
        
        with torch.no_grad():
            # Process in batches
            for i in tqdm(range(0, len(text_data), batch_size), desc="Generating LLM embeddings"):
                batch_texts = text_data[i:i + batch_size]
                
                # Tokenize batch
                inputs = self.tokenizer(
                    batch_texts, 
                    return_tensors="pt", 
                    max_length=max_length, 
                    truncation=True, 
                    padding=True
                )
                
                # Move inputs to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Generate embeddings
                outputs = self.model(**inputs)
                
                # Use mean pooling over sequence length
                batch_embeddings = outputs.last_hidden_state.mean(dim=1)  # [batch_size, hidden_size]
                embeddings.append(batch_embeddings.cpu().numpy())
        
        return np.vstack(embeddings)
    
    def create_conditioning_vectors(self, returns_index, seq_len=SEQ_LEN, embedding_dim=EMBEDDING_DIM):
        """
        Create conditioning vectors for the diffusion model.
        
        Args:
            returns_index: Index of returns data (trading days)
            seq_len: Length of training sequences
            embedding_dim: Dimension of LLM embeddings (should be 768 for DistilBERT)
            
        Returns:
            numpy array of conditioning vectors with shape (num_sequences, embed_dim)
        """
        # Get market sentiment data for the full date range
        start_date = returns_index[0]
        end_date = returns_index[-1]
        sentiment_data = self.get_market_sentiment_data(start_date, end_date)
        
        # Generate LLM embeddings
        embeddings = self.generate_embeddings(sentiment_data)
        
        # Create DataFrame of daily embeddings indexed by trading days
        embedding_df = pd.DataFrame(embeddings, index=pd.date_range(start_date, end_date))
        
        # Align with trading days using forward-fill and backfill
        aligned_embeddings = embedding_df.reindex(returns_index, method='ffill').fillna(method='bfill')
        
        # Aggregate embeddings per training window
        conditioning_vectors = []
        for i in range(len(returns_index) - seq_len + 1):
            window_embeddings = aligned_embeddings.iloc[i:i+seq_len].values
            # Aggregate using mean (configurable method)
            window_conditioning = window_embeddings.mean(axis=0)
            conditioning_vectors.append(window_conditioning)
        
        conditioning_vectors = np.array(conditioning_vectors)
        
        print(f"Generated {len(conditioning_vectors)} conditioning vectors of dimension {conditioning_vectors.shape[1]}")
        return conditioning_vectors

class ConditionedDiffusionModel(nn.Module):
    """
    Conditional Diffusion Model that uses LLM embeddings as conditioning.
    Based on DDPM architecture but with conditioning capabilities.
    """
    
    def __init__(self, sequence_length=SEQ_LEN, conditioning_dim=EMBEDDING_DIM, hidden_dim=128):
        super(ConditionedDiffusionModel, self).__init__()
        
        self.sequence_length = sequence_length
        self.conditioning_dim = conditioning_dim
        self.hidden_dim = hidden_dim
        
        # Conditioning projection
        self.conditioning_projection = nn.Linear(conditioning_dim, hidden_dim)
        
        # Time embedding
        self.time_embedding = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Main denoising network
        self.denoising_network = nn.Sequential(
            nn.Linear(1 + hidden_dim + hidden_dim, hidden_dim),  # input + time + conditioning
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        print(f"Conditioned Diffusion Model initialized")
        print(f"   - Sequence length: {sequence_length}")
        print(f"   - Conditioning dimension: {conditioning_dim}")
        print(f"   - Hidden dimension: {hidden_dim}")
    
    def forward(self, x, t, conditioning):
        """
        Forward pass of the conditioned diffusion model.
        
        Args:
            x: Input sequence [batch_size, sequence_length, 1]
            t: Time step [batch_size, 1]
            conditioning: Conditioning vector [batch_size, conditioning_dim]
            
        Returns:
            Predicted noise [batch_size, sequence_length, 1]
        """
        batch_size, seq_len, _ = x.shape
        
        # Project conditioning to hidden dimension
        conditioning_proj = self.conditioning_projection(conditioning)  # [batch_size, hidden_dim]
        conditioning_proj = conditioning_proj.unsqueeze(1).expand(-1, seq_len, -1)  # [batch_size, seq_len, hidden_dim]
        
        # Time embedding
        t_embed = self.time_embedding(t)  # [batch_size, hidden_dim]
        t_embed = t_embed.unsqueeze(1).expand(-1, seq_len, -1)  # [batch_size, seq_len, hidden_dim]
        
        # Concatenate input, time, and conditioning
        combined = torch.cat([x, t_embed, conditioning_proj], dim=-1)  # [batch_size, seq_len, 1+hidden_dim+hidden_dim]
        
        # Process through denoising network
        output = self.denoising_network(combined)  # [batch_size, seq_len, 1]
        
        return output

class ConditionedDiffusionTrainer:
    """
    Trainer for the LLM-conditioned diffusion model.
    Implements the training loop with proper conditioning and device safety.
    """
    
    def __init__(self, model, num_timesteps=1000, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.model = model
        self.num_timesteps = num_timesteps
        self.device = device
        
        # Move model to device
        self.model = self.model.to(device)
        
        # Linear noise schedule on device
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # Precompute values for sampling
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)
        
        print(f"Conditioned Diffusion Trainer initialized")
        print(f"   - Number of timesteps: {num_timesteps}")
        print(f"   - Beta schedule: {beta_start:.6f} to {beta_end:.6f}")
        print(f"   - Device: {device}")
    
    def add_noise(self, x_start, t):
        """
        Add noise to data according to diffusion schedule.
        
        Args:
            x_start: Original data [batch_size, sequence_length, 1]
            t: Time steps [batch_size]
            
        Returns:
            Noisy data and noise
        """
        noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        
        x_noisy = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
        return x_noisy, noise
    
    def train_step(self, x, conditioning, optimizer):
        """
        Single training step.
        
        Args:
            x: Input sequences [batch_size, sequence_length, 1]
            conditioning: Conditioning vectors [batch_size, conditioning_dim]
            optimizer: PyTorch optimizer
            
        Returns:
            Loss value
        """
        batch_size = x.shape[0]
        
        # Sample random timesteps
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=self.device)
        
        # Add noise
        x_noisy, noise = self.add_noise(x, t)
        
        # Normalize timesteps to [0, 1]
        t_normalized = t.float() / self.num_timesteps
        
        # Predict noise
        predicted_noise = self.model(x_noisy, t_normalized.unsqueeze(-1), conditioning)
        
        # Compute loss
        loss = F.mse_loss(predicted_noise, noise)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    def p_sample(self, x, t, conditioning, sampler="ddpm"):
        """
        Sample from the posterior distribution.
        
        Args:
            x: Current noisy sample [batch_size, sequence_length, 1]
            t: Current timestep (Python integer)
            conditioning: Conditioning vectors [batch_size, conditioning_dim]
            sampler: Sampling method ("ddpm" or "ddim")
            
        Returns:
            Denoised sample [batch_size, sequence_length, 1]
        """
        batch_size = x.shape[0]
        
        # Compute all timestep-dependent scalars as batch-shaped tensors [B,1,1]
        alpha_t = self.alphas[t].view(1, 1, 1)
        beta_t = self.betas[t].view(1, 1, 1)
        alpha_bar_t = self.alphas_cumprod[t].view(1, 1, 1)
        alpha_bar_tm1 = self.alphas_cumprod[t-1].view(1, 1, 1) if t > 0 else torch.ones(1, 1, 1, device=self.device)
        
        if sampler == "ddpm":
            # DDPM posterior sampling
            # Compute x0_hat (predicted x0) using the predicted noise
            predicted_noise = self.model(x, (t / self.num_timesteps) * torch.ones(batch_size, 1, device=self.device), conditioning)
            x0_hat = (x - torch.sqrt(1 - alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_bar_t)
            
            # Posterior mean: mu_t = (1/sqrt(alpha_t)) * (x_t - (beta_t/sqrt(1-alpha_bar_t)) * epsilon_theta)
            mean = (1 / torch.sqrt(alpha_t)) * (x - (beta_t / torch.sqrt(1 - alpha_bar_t)) * predicted_noise)
            
            if t > 0:
                # Posterior variance: tilde_beta_t = (1 - alpha_bar_{t-1})/(1 - alpha_bar_t) * beta_t
                tilde_beta_t = (1 - alpha_bar_tm1) / (1 - alpha_bar_t) * beta_t
                
                # Add noise for stochastic sampling
                noise = torch.randn_like(x)
                x = mean + torch.sqrt(tilde_beta_t) * noise
            else:
                # For t == 0, return the posterior mean without noise (same formula as t > 0)
                x = mean
        
        elif sampler == "ddim":
            # DDIM deterministic sampling
            predicted_noise = self.model(x, (t / self.num_timesteps) * torch.ones(batch_size, 1, device=self.device), conditioning)
            
            # DDIM update: x_{t-1} = sqrt(alpha_bar_{t-1}) * (x_t/sqrt(alpha_bar_t) - sqrt(1/alpha_bar_t - 1) * epsilon_theta) + sqrt(1 - alpha_bar_{t-1}) * epsilon_theta
            x = torch.sqrt(alpha_bar_tm1) * (x / torch.sqrt(alpha_bar_t) - torch.sqrt(1/alpha_bar_t - 1) * predicted_noise) + torch.sqrt(1 - alpha_bar_tm1) * predicted_noise
        
        return x
    
    def sample(self, conditioning, num_samples=1, sampler="ddpm"):
        """
        Generate samples using the trained model.
        
        Args:
            conditioning: Conditioning vectors [num_samples, conditioning_dim]
            num_samples: Number of samples to generate
            sampler: Sampling method ("ddpm" or "ddim")
            
        Returns:
            Generated sequences [num_samples, sequence_length, 1]
        """
        self.model.eval()
        
        with torch.no_grad():
            # Start from pure noise
            x = torch.randn(num_samples, self.model.sequence_length, 1, device=self.device)
            
            # Reverse diffusion process
            for t in tqdm(reversed(range(self.num_timesteps)), desc="Generating samples"):
                x = self.p_sample(x, t, conditioning, sampler)
        
        return x

def load_and_prepare_data():
    """Load and prepare financial returns data."""
    print("Loading financial data...")
    
    # Robust data file path handling with environment variable support
    data_path = os.getenv('SP500_DATA_PATH', "../data/sp500_data.csv")
    
    # Try multiple fallback paths
    fallback_paths = [
        data_path,
        "data/sp500_data.csv",
        "../data/sp500_data.csv",
        "../../data/sp500_data.csv"
    ]
    
    data = None
    for path in fallback_paths:
        if os.path.exists(path):
            try:
                data = pd.read_csv(path, index_col=0, parse_dates=True)
                print(f"Data loaded from: {path}")
                break
            except Exception as e:
                print(f"Failed to load from {path}: {e}")
                continue
    
    if data is None:
        raise FileNotFoundError(
            f"Could not find sp500_data.csv in any of the following paths: {fallback_paths}. "
            "Please set SP500_DATA_PATH environment variable or ensure the file exists."
        )
    
    # Ensure index is datetime
    data.index = pd.to_datetime(data.index)
    
    # Calculate log returns for more stable statistical properties in diffusion training
    # Log returns are additive and have better statistical properties for time series modeling
    returns = np.log(data['Close'] / data['Close'].shift(1)).dropna()
    
    print(f"Loaded {len(returns)} days of return data")
    print(f"Date range: {returns.index[0]} to {returns.index[-1]}")
    return returns

def create_sequences(returns, seq_len=SEQ_LEN):
    """Create sequences for training."""
    print(f"Creating sequences of length {seq_len}...")
    
    sequences = []
    for i in range(len(returns) - seq_len + 1):
        seq = returns.iloc[i:i+seq_len].values
        sequences.append(seq)
    
    X = np.array(sequences)
    X = X[..., np.newaxis]  # Add channel dimension
    print(f"Created {len(X)} sequences")
    return X

def train_conditioned_diffusion_model(X, conditioning_vectors, num_epochs=50, batch_size=32, device="cpu"):
    """Train the LLM-conditioned diffusion model."""
    print("Training LLM-conditioned diffusion model...")
    
    # Initialize model and trainer
    model = ConditionedDiffusionModel(
        sequence_length=SEQ_LEN,
        conditioning_dim=conditioning_vectors.shape[1],
        hidden_dim=128
    )
    
    trainer = ConditionedDiffusionTrainer(model, num_timesteps=1000, device=device)
    
    # Prepare data
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(conditioning_vectors, dtype=torch.float32)
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Training loop
    losses = []
    for epoch in range(num_epochs):
        epoch_losses = []
        
        for batch_x, batch_conditioning in dataloader:
            # Move to device
            batch_x = batch_x.to(device)
            batch_conditioning = batch_conditioning.to(device)
            
            loss = trainer.train_step(batch_x, batch_conditioning, optimizer)
            epoch_losses.append(loss)
        
        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        scheduler.step()
        
        if epoch % 10 == 0:
            print(f"   Epoch {epoch}: Loss = {avg_loss:.6f}")
    
    print("LLM-conditioned diffusion training completed!")
    return model, trainer, losses

def generate_conditioned_samples(model, trainer, conditioning_vectors, num_samples=1000, sampler="ddpm"):
    """Generate samples using the trained model."""
    print("Generating conditioned synthetic data...")
    
    # Ensure conditioning vectors are on the same device as the trainer
    device = trainer.device
    conditioning_tensor = torch.tensor(conditioning_vectors[:num_samples], dtype=torch.float32, device=device)
    
    # Generate samples
    samples = trainer.sample(conditioning_tensor, num_samples=num_samples, sampler=sampler)
    samples = samples.squeeze(-1).cpu().numpy()  # Remove channel dimension and move to CPU
    
    print(f"Generated {len(samples)} conditioned synthetic sequences")
    return samples

def evaluate_conditioned_performance(real_returns, synthetic_data, conditioning_vectors):
    """Evaluate the performance of the conditioned model."""
    print("Evaluating conditioned model performance...")
    
    # Basic statistics
    real_stats = {
        'mean': np.mean(real_returns),
        'std': np.std(real_returns),
        'skew': scipy.stats.skew(real_returns),
        'kurtosis': scipy.stats.kurtosis(real_returns)
    }
    
    synthetic_stats = {
        'mean': np.mean(synthetic_data),
        'std': np.std(synthetic_data),
        'skew': scipy.stats.skew(synthetic_data.flatten()),
        'kurtosis': scipy.stats.kurtosis(synthetic_data.flatten())
    }
    
    # KS test
    ks_stat, ks_pvalue = scipy.stats.ks_2samp(real_returns, synthetic_data.flatten())
    
    # Controllability check
    print("\nControllability Check:")
    
    # Use L2 norm of embeddings as volatility proxy (placeholder - can be replaced with realized volatility)
    # TODO: Replace with realized volatility over the input window if that data becomes available
    embedding_norms = np.linalg.norm(conditioning_vectors, axis=1)
    
    # Split into top and bottom volatility buckets
    split_point = np.median(embedding_norms)
    low_vol_indices = embedding_norms <= split_point
    high_vol_indices = embedding_norms > split_point
    
    if np.sum(low_vol_indices) > 0 and np.sum(high_vol_indices) > 0:
        low_vol_samples = synthetic_data[low_vol_indices]
        high_vol_samples = synthetic_data[high_vol_indices]
        
        low_vol_std = np.std(low_vol_samples)
        high_vol_std = np.std(high_vol_samples)
        
        print(f"   Low volatility bucket (n={np.sum(low_vol_indices)}): std = {low_vol_std:.6f}")
        print(f"   High volatility bucket (n={np.sum(high_vol_indices)}): std = {high_vol_std:.6f}")
        print(f"   Volatility ratio (high/low): {high_vol_std/low_vol_std:.3f}")
    
    # Print KS test results
    print(f"\nKS Test Results:")
    print(f"   KS statistic: {ks_stat:.6f}")
    print(f"   p-value: {ks_pvalue:.6f}")
    
    return real_stats, synthetic_stats, ks_stat, ks_pvalue

def save_conditioned_results(synthetic_data, stats, losses, model, trainer):
    """Save results and training artifacts."""
    print("Saving results...")
    
    # Create results directory
    results_dir = "results/llm_conditioned_evaluation"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save synthetic data
    np.save(f"{results_dir}/llm_conditioned_returns.npy", synthetic_data)
    
    # Save training losses
    np.save(f"{results_dir}/losses.npy", losses)
    
    # Save evaluation stats
    with open(f"{results_dir}/evaluation_stats.txt", "w") as f:
        f.write("LLM-Conditioned Diffusion Model Evaluation Results\n")
        f.write("=" * 50 + "\n")
        f.write(f"Real data statistics: {stats[0]}\n")
        f.write(f"Synthetic data statistics: {stats[1]}\n")
        f.write(f"KS statistic: {stats[2]:.6f}\n")
        f.write(f"KS p-value: {stats[3]:.6f}\n")
    
    # Plot loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(f"{results_dir}/loss_curve.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot sample sequences with clean grid layout
    num_samples = min(5, len(synthetic_data))
    fig, axes = plt.subplots(num_samples, 1, figsize=(12, 2*num_samples))
    
    if num_samples == 1:
        axes = [axes]
    
    for i in range(num_samples):
        axes[i].plot(synthetic_data[i], linewidth=1.5)
        axes[i].set_title(f"Sample Sequence {i+1}", fontsize=12)
        axes[i].set_ylabel("Returns", fontsize=10)
        axes[i].grid(True, alpha=0.3)
        axes[i].set_xlim(0, len(synthetic_data[i])-1)
    
    # Only show x-label for bottom subplot
    axes[-1].set_xlabel("Time Step", fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f"{results_dir}/sample_sequences.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Results saved successfully!")

def main():
    """Main function to run the LLM-conditioned diffusion model."""
    print("LLM-Conditioned Diffusion Model for Financial Data Synthesis")
    print("Based on supervisor feedback: Using LLM embeddings as conditioning vectors")
    print("=" * 80)
    
    # Set reproducibility seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load and prepare data
    returns = load_and_prepare_data()
    
    # Compute statistics for standardization
    mu = returns.mean()
    sigma = returns.std()
    print(f"Returns statistics - Mean: {mu:.6f}, Std: {sigma:.6f}")
    
    # Standardize returns
    returns_standardized = (returns - mu) / sigma
    
    # Create sequences
    X = create_sequences(returns_standardized)
    
    # Initialize LLM conditioning module with device parameter
    device = "cuda" if torch.cuda.is_available() else "cpu"
    llm_conditioner = LLMConditioningModule(device=device)
    
    # Generate conditioning vectors
    conditioning_vectors = llm_conditioner.create_conditioning_vectors(returns.index)
    
    # Ensure conditioning vectors match data length
    assert len(conditioning_vectors) == len(X), f"Conditioning vectors length {len(conditioning_vectors)} must match sequences length {len(X)}"
    
    # Train the conditioned model
    model, trainer, losses = train_conditioned_diffusion_model(X, conditioning_vectors, device=device)
    
    # Generate synthetic data using the same subset of conditioning vectors for consistency
    num_samples = min(1000, len(conditioning_vectors))
    synthetic_data_standardized = generate_conditioned_samples(model, trainer, conditioning_vectors[:num_samples], num_samples=num_samples, sampler="ddpm")
    
    # De-standardize samples
    synthetic_data = synthetic_data_standardized * sigma + mu
    
    # Evaluate performance using the same subset
    stats = evaluate_conditioned_performance(returns.values, synthetic_data, conditioning_vectors[:num_samples])
    
    # Save results
    save_conditioned_results(synthetic_data, stats, losses, model, trainer)
    
    print("\nLLM-conditioned diffusion model completed successfully!")
    print("Results saved in: results/llm_conditioned_evaluation/")
    
    return model, trainer, synthetic_data, stats

if __name__ == "__main__":
    main()
