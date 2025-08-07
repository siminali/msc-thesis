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
import requests
from datetime import datetime, timedelta
import json
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class LLMConditioningModule:
    """
    Module to generate conditioning vectors from internet data using LLMs.
    Implements the approach suggested by supervisor: feed internet data into LLM,
    get latent embeddings, use as conditioning for diffusion model.
    """
    
    def __init__(self, model_name="distilbert-base-uncased"):
        """
        Initialize LLM for text embedding generation.
        
        Args:
            model_name: Hugging Face model name for text processing
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        
        # Freeze LLM parameters (we only use it for embeddings)
        for param in self.model.parameters():
            param.requires_grad = False
            
        print(f"✅ LLM Conditioning Module initialized with {model_name}")
    
    def get_market_sentiment_data(self, start_date, end_date):
        """
        Fetch market sentiment data from various sources.
        This simulates the internet data that would be fed into the LLM.
        
        Args:
            start_date: Start date for data collection
            end_date: End date for data collection
            
        Returns:
            List of text data representing market sentiment
        """
        # Simulate market sentiment data from different sources
        sentiment_data = []
        
        # Generate synthetic market sentiment text based on S&P 500 movements
        # In practice, this would come from news APIs, social media, etc.
        
        # Example sentiment texts based on market conditions
        base_sentiments = [
            "Market shows strong bullish momentum with increasing trading volumes",
            "Volatility spikes as investors react to economic uncertainty",
            "Risk appetite increases as market participants seek higher returns",
            "Market sentiment turns cautious amid geopolitical tensions",
            "Trading activity remains subdued with low volatility environment",
            "Institutional investors show confidence in market fundamentals",
            "Retail investors drive market momentum with increased participation",
            "Market volatility normalizes after recent turbulence",
            "Risk-off sentiment prevails as safe-haven assets gain",
            "Market participants remain optimistic about economic recovery"
        ]
        
        # Generate daily sentiment based on market conditions
        current_date = start_date
        while current_date <= end_date:
            # Randomly select and modify sentiment based on date
            sentiment = np.random.choice(base_sentiments)
            
            # Add date-specific context
            sentiment += f" on {current_date.strftime('%Y-%m-%d')}"
            
            sentiment_data.append(sentiment)
            current_date += timedelta(days=1)
        
        return sentiment_data
    
    def generate_embeddings(self, text_data, max_length=512):
        """
        Generate embeddings from text data using the LLM.
        
        Args:
            text_data: List of text strings
            max_length: Maximum sequence length for tokenization
            
        Returns:
            numpy array of embeddings
        """
        embeddings = []
        
        with torch.no_grad():
            for text in tqdm(text_data, desc="Generating LLM embeddings"):
                # Tokenize text
                inputs = self.tokenizer(
                    text, 
                    return_tensors="pt", 
                    max_length=max_length, 
                    truncation=True, 
                    padding=True
                )
                
                # Generate embeddings
                outputs = self.model(**inputs)
                
                # Use mean pooling over sequence length
                embedding = outputs.last_hidden_state.mean(dim=1)  # [1, hidden_size]
                embeddings.append(embedding.squeeze().numpy())
        
        return np.array(embeddings)
    
    def create_conditioning_vectors(self, start_date, end_date, embedding_dim=768):
        """
        Create conditioning vectors for the diffusion model.
        
        Args:
            start_date: Start date for conditioning data
            end_date: End date for conditioning data
            embedding_dim: Dimension of LLM embeddings
            
        Returns:
            numpy array of conditioning vectors
        """
        # Get market sentiment data
        sentiment_data = self.get_market_sentiment_data(start_date, end_date)
        
        # Generate LLM embeddings
        embeddings = self.generate_embeddings(sentiment_data)
        
        # Project to desired conditioning dimension if needed
        if embeddings.shape[1] != embedding_dim:
            # Simple linear projection (could be learned)
            projection = np.random.randn(embeddings.shape[1], embedding_dim) * 0.1
            embeddings = embeddings @ projection
        
        print(f"✅ Generated {len(embeddings)} conditioning vectors of dimension {embeddings.shape[1]}")
        return embeddings

class ConditionedDiffusionModel(nn.Module):
    """
    Conditional Diffusion Model that uses LLM embeddings as conditioning.
    Based on DDPM architecture but with conditioning capabilities.
    """
    
    def __init__(self, sequence_length=60, conditioning_dim=768, hidden_dim=128):
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
        
        print(f"✅ Conditioned Diffusion Model initialized")
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
    Implements the training loop with proper conditioning.
    """
    
    def __init__(self, model, num_timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.model = model
        self.num_timesteps = num_timesteps
        
        # Linear noise schedule
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        print(f"✅ Conditioned Diffusion Trainer initialized")
        print(f"   - Number of timesteps: {num_timesteps}")
        print(f"   - Beta schedule: {beta_start:.6f} to {beta_end:.6f}")
    
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
        sqrt_alphas_cumprod_t = self.alphas_cumprod[t].sqrt().view(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = (1 - self.alphas_cumprod[t]).sqrt().view(-1, 1, 1)
        
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
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=x.device)
        
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
    
    def sample(self, conditioning, num_samples=1, device='cpu'):
        """
        Generate samples using the trained model.
        
        Args:
            conditioning: Conditioning vectors [num_samples, conditioning_dim]
            num_samples: Number of samples to generate
            device: Device to run on
            
        Returns:
            Generated sequences [num_samples, sequence_length, 1]
        """
        self.model.eval()
        
        with torch.no_grad():
            # Start from pure noise
            x = torch.randn(num_samples, self.model.sequence_length, 1, device=device)
            
            # Reverse diffusion process
            for t in tqdm(reversed(range(self.num_timesteps)), desc="Generating samples"):
                t_normalized = torch.full((num_samples,), t / self.num_timesteps, device=device)
                
                # Predict noise
                predicted_noise = self.model(x, t_normalized.unsqueeze(-1), conditioning)
                
                # Remove predicted noise
                alpha_t = self.alphas[t]
                alpha_cumprod_t = self.alphas_cumprod[t]
                beta_t = self.betas[t]
                
                if t > 0:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                
                x = (1 / alpha_t.sqrt()) * (x - (beta_t / (1 - alpha_cumprod_t).sqrt()) * predicted_noise) + beta_t.sqrt() * noise
        
        return x

def load_and_prepare_data():
    """Load and prepare S&P 500 data for training."""
    print("📊 Loading S&P 500 data...")
    
    # Load data
    data = pd.read_csv("../data/sp500_data.csv", index_col=0, parse_dates=True)
    data.index = pd.to_datetime(data.index)
    
    # Handle potential header row
    if data.index[0] == 'Ticker':
        data = data.iloc[1:]
        data.index = pd.to_datetime(data.index)
    
    data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
    data = data[['Close']]
    
    # Compute log returns
    returns = np.log(data['Close'] / data['Close'].shift(1)).dropna() * 100
    
    print(f"✅ Data loaded: {len(returns)} observations")
    print(f"   Date range: {returns.index[0]} to {returns.index[-1]}")
    
    return returns

def create_sequences(returns, sequence_length=60):
    """Create sequences for training."""
    print("🔄 Creating training sequences...")
    
    # Create sequences
    X = []
    for i in range(len(returns) - sequence_length):
        X.append(returns.iloc[i:i+sequence_length].values)
    
    X = np.array(X)
    X = X[..., np.newaxis]  # Add channel dimension
    
    print(f"✅ Created {len(X)} sequences of length {sequence_length}")
    return X

def train_conditioned_diffusion_model(X, conditioning_vectors, num_epochs=50, batch_size=32):
    """Train the LLM-conditioned diffusion model."""
    print("🎯 Training LLM-conditioned diffusion model...")
    
    # Initialize model and trainer
    model = ConditionedDiffusionModel(
        sequence_length=60,
        conditioning_dim=conditioning_vectors.shape[1],
        hidden_dim=128
    )
    
    trainer = ConditionedDiffusionTrainer(model, num_timesteps=1000)
    
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
            loss = trainer.train_step(batch_x, batch_conditioning, optimizer)
            epoch_losses.append(loss)
        
        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        scheduler.step()
        
        if epoch % 10 == 0:
            print(f"   Epoch {epoch}: Loss = {avg_loss:.6f}")
    
    print("✅ LLM-conditioned diffusion training completed!")
    return model, trainer, losses

def generate_conditioned_samples(model, trainer, conditioning_vectors, num_samples=1000):
    """Generate samples using the trained model."""
    print("🎨 Generating conditioned synthetic data...")
    
    # Convert conditioning vectors to tensor
    conditioning_tensor = torch.tensor(conditioning_vectors[:num_samples], dtype=torch.float32)
    
    # Generate samples
    samples = trainer.sample(conditioning_tensor, num_samples=num_samples)
    samples = samples.squeeze(-1).numpy()  # Remove channel dimension
    
    print(f"✅ Generated {len(samples)} conditioned synthetic sequences")
    return samples

def evaluate_conditioned_performance(real_data, synthetic_data, model_name="LLM-Conditioned"):
    """Evaluate the performance of the conditioned model."""
    print(f"📈 Evaluating {model_name} performance...")
    
    # Import scipy stats
    from scipy import stats as scipy_stats
    
    # Flatten data for evaluation
    real_flat = real_data.flatten()
    synthetic_flat = synthetic_data.flatten()
    
    # Basic statistics
    stats_dict = {
        'Model': model_name,
        'Mean': np.mean(synthetic_flat),
        'Std Dev': np.std(synthetic_flat),
        'Skewness': scipy_stats.skew(synthetic_flat),
        'Kurtosis': scipy_stats.kurtosis(synthetic_flat),
        'Min': np.min(synthetic_flat),
        'Max': np.max(synthetic_flat)
    }
    
    # Distribution similarity (KS test)
    ks_stat, ks_pvalue = scipy_stats.ks_2samp(real_flat, synthetic_flat)
    
    print(f"✅ {model_name} Evaluation Results:")
    print(f"   KS Statistic: {ks_stat:.4f} (p-value: {ks_pvalue:.4f})")
    print(f"   Real vs Synthetic Mean: {np.mean(real_flat):.4f} vs {stats_dict['Mean']:.4f}")
    print(f"   Real vs Synthetic Std: {np.std(real_flat):.4f} vs {stats_dict['Std Dev']:.4f}")
    
    return stats_dict, ks_stat, ks_pvalue

def save_conditioned_results(synthetic_data, stats, model_name="llm_conditioned"):
    """Save results for the conditioned model."""
    print("💾 Saving conditioned results...")
    
    # Create results directory
    os.makedirs(f"results/{model_name}_evaluation", exist_ok=True)
    
    # Save synthetic data
    np.save(f"results/{model_name}_returns.npy", synthetic_data)
    
    # Convert numpy types to native Python types for JSON serialization
    json_stats = {}
    for key, value in stats.items():
        if isinstance(value, (np.integer, np.floating)):
            json_stats[key] = float(value)
        else:
            json_stats[key] = value
    
    # Save statistics
    import json
    with open(f"results/{model_name}_evaluation/{model_name}_stats.json", 'w') as f:
        json.dump(json_stats, f, indent=2)
    
    print(f"✅ {model_name} results saved to results/ directory")

def main():
    """Main function to run the LLM-conditioned diffusion model."""
    print("🚀 LLM-Conditioned Diffusion Model for Financial Data Synthesis")
    print("Based on supervisor feedback: Using LLM embeddings as conditioning vectors")
    print("=" * 80)
    
    # Load and prepare data
    returns = load_and_prepare_data()
    X = create_sequences(returns)
    
    # Initialize LLM conditioning module
    llm_conditioner = LLMConditioningModule()
    
    # Generate conditioning vectors
    start_date = returns.index[0]
    end_date = returns.index[-1]
    conditioning_vectors = llm_conditioner.create_conditioning_vectors(start_date, end_date)
    
    # Ensure conditioning vectors match data length
    if len(conditioning_vectors) > len(X):
        conditioning_vectors = conditioning_vectors[:len(X)]
    elif len(conditioning_vectors) < len(X):
        # Pad with last conditioning vector
        last_conditioning = conditioning_vectors[-1:]
        padding_needed = len(X) - len(conditioning_vectors)
        padding = np.tile(last_conditioning, (padding_needed, 1))
        conditioning_vectors = np.vstack([conditioning_vectors, padding])
    
    # Train the conditioned model
    model, trainer, losses = train_conditioned_diffusion_model(X, conditioning_vectors)
    
    # Generate synthetic data
    synthetic_data = generate_conditioned_samples(model, trainer, conditioning_vectors)
    
    # Evaluate performance
    stats, ks_stat, ks_pvalue = evaluate_conditioned_performance(returns.values, synthetic_data)
    
    # Save results
    save_conditioned_results(synthetic_data, stats)
    
    print("\n🎉 LLM-conditioned diffusion model completed successfully!")
    print("📁 Results saved in: results/llm_conditioned_evaluation/")
    
    return model, trainer, synthetic_data, stats

if __name__ == "__main__":
    main()
