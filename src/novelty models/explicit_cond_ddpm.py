#!/usr/bin/env python3
"""
Explicit Conditioning DDPM for Financial Data Synthesis
Uses regime classification (Up/Down × Low/High) + target volatility scalar

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
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from datetime import datetime
import json
import os
import argparse
import warnings
from tqdm import tqdm
import scipy.stats
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
warnings.filterwarnings('ignore')

# Global constants
REGIME_ORDER = ['Up-Low', 'Up-High', 'Down-Low', 'Down-High']

class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation layer for conditioning injection."""
    
    def __init__(self, hidden_dim, conditioning_dim):
        super().__init__()
        self.scale_proj = nn.Linear(conditioning_dim, hidden_dim)
        self.shift_proj = nn.Linear(conditioning_dim, hidden_dim)
        
        # Initialize weights and biases to zero for identity mapping
        nn.init.zeros_(self.scale_proj.weight)
        nn.init.zeros_(self.scale_proj.bias)
        nn.init.zeros_(self.shift_proj.weight)
        nn.init.zeros_(self.shift_proj.bias)
        
    def forward(self, x, conditioning):
        # conditioning: [B, 5] -> scale/shift: [B, H] -> [B, H, 1] for broadcasting
        scale = self.scale_proj(conditioning).unsqueeze(-1)  # [B, H, 1]
        shift = self.shift_proj(conditioning).unsqueeze(-1)  # [B, H, 1]
        return x * (1 + scale) + shift

class DilatedResidualBlock(nn.Module):
    """1D dilated convolutional residual block with FiLM conditioning."""
    
    def __init__(self, hidden_dim, dilation, conditioning_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, 3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, 3, padding=dilation, dilation=dilation)
        self.norm1 = nn.GroupNorm(8, hidden_dim)
        self.norm2 = nn.GroupNorm(8, hidden_dim)
        self.film = FiLMLayer(hidden_dim, conditioning_dim)
        self.activation = nn.SiLU()
        
    def forward(self, x, conditioning):
        # x: [B, H, T], conditioning: [B, 5]
        residual = x
        
        # First conv + norm + activation
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)
        
        # FiLM conditioning with original 5-dimensional vector
        x = self.film(x, conditioning)
        
        # Second conv + norm
        x = self.conv2(x)
        x = self.norm2(x)
        
        # Residual connection
        x = x + residual
        x = self.activation(x)
        
        return x

class TemporalDenoiser(nn.Module):
    """Lightweight temporal denoiser with dilated convolutions and FiLM conditioning."""
    
    def __init__(self, sequence_length, conditioning_dim, hidden_dim=128):
        super().__init__()
        self.sequence_length = sequence_length
        self.conditioning_dim = conditioning_dim
        self.hidden_dim = hidden_dim
        
        # Safety checks for hidden_dim
        if hidden_dim % 8 != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by 8 for GroupNorm compatibility")
        if hidden_dim % 2 != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be even for sinusoidal time embedding")
        
        # Input projection
        self.input_proj = nn.Conv1d(1, hidden_dim, 1)
        
        # Time embedding with sinusoidal encoding
        self.time_embedding = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Sinusoidal time embedding parameters
        self.max_freq_log2 = np.log2(hidden_dim // 2)
        self.num_freq_bands = hidden_dim // 2
        
        # Conditioning projection
        self.conditioning_proj = nn.Linear(conditioning_dim, hidden_dim)
        
        # Initialize conditioning projection to zero for identity mapping
        nn.init.zeros_(self.conditioning_proj.weight)
        nn.init.zeros_(self.conditioning_proj.bias)
        
        # Dilated residual blocks with exponentially increasing dilations
        dilations = [1, 2, 4, 8, 16, 32]
        self.residual_blocks = nn.ModuleList([
            DilatedResidualBlock(hidden_dim, dilation, conditioning_dim)
            for dilation in dilations
        ])
        
        # Output projection
        self.output_proj = nn.Conv1d(hidden_dim, 1, 1)
        
        print(f"Temporal Denoiser initialized:")
        print(f"   - Sequence length: {sequence_length}")
        print(f"   - Conditioning dimension: {conditioning_dim}")
        print(f"   - Hidden dimension: {hidden_dim}")
        print(f"   - Number of residual blocks: {len(self.residual_blocks)}")
    
    def sinusoidal_time_embedding(self, t):
        """Generate sinusoidal time embeddings."""
        # t: [B, 1] -> [B, H]
        batch_size = t.shape[0]
        
        # Create frequency bands
        freq_bands = 2.0 ** torch.linspace(0, self.max_freq_log2, self.num_freq_bands, device=t.device)
        freq_bands = freq_bands.unsqueeze(0).expand(batch_size, -1)  # [B, H//2]
        
        # Compute sinusoidal embeddings with 2π phase
        t_expanded = t.expand(-1, self.num_freq_bands)  # [B, H//2]
        sin_emb = torch.sin(2 * np.pi * freq_bands * t_expanded)
        cos_emb = torch.cos(2 * np.pi * freq_bands * t_expanded)
        
        # Concatenate sin and cos embeddings
        time_emb = torch.cat([sin_emb, cos_emb], dim=1)  # [B, H]
        
        # Pass through the time embedding MLP
        time_emb = self.time_embedding(time_emb)
        
        return time_emb
    
    def forward(self, x, t, conditioning):
        """
        Forward pass.
        
        Args:
            x: Input [B, 1, T]
            t: Time steps [B, 1]
            conditioning: Conditioning vector [B, C]
        """
        batch_size = x.shape[0]
        
        # Input projection
        x = self.input_proj(x)  # [B, H, T]
        
        # Time embedding with sinusoidal encoding
        t_embed = self.sinusoidal_time_embedding(t)  # [B, H]
        t_embed = t_embed.unsqueeze(-1).expand(-1, -1, self.sequence_length)  # [B, H, T]
        
        # Conditioning projection
        cond_embed = self.conditioning_proj(conditioning)  # [B, H]
        cond_embed = cond_embed.unsqueeze(-1).expand(-1, -1, self.sequence_length)  # [B, H, T]
        
        # Add time and conditioning embeddings
        x = x + t_embed + cond_embed
        
        # Process through residual blocks
        for block in self.residual_blocks:
            x = block(x, conditioning)  # Pass original 5-dimensional conditioning
        
        # Output projection
        x = self.output_proj(x)  # [B, 1, T]
        
        return x

class ExplicitConditioningDDPM(nn.Module):
    """Explicit conditioning DDPM using regime classification + volatility scalar."""
    
    def __init__(self, sequence_length, conditioning_dim, hidden_dim=128):
        super().__init__()
        self.sequence_length = sequence_length
        self.conditioning_dim = conditioning_dim
        self.hidden_dim = hidden_dim
        
        self.denoiser = TemporalDenoiser(sequence_length, conditioning_dim, hidden_dim)
        
    def forward(self, x, t, conditioning):
        return self.denoiser(x, t, conditioning)

class ExplicitConditioningTrainer:
    """Trainer for explicit conditioning DDPM with EMA and early stopping."""
    
    def __init__(self, model, num_timesteps=1000, beta_schedule="cosine", device="cpu", grad_clip=1.0, cfg_p=0.1, amp=False, compile=False):
        self.model = model
        self.num_timesteps = num_timesteps
        self.device = device
        self.grad_clip = grad_clip
        self.cfg_p = cfg_p  # Probability of conditioning dropout during training
        self.amp = amp
        
        # Move model to device
        self.model = self.model.to(device)
        
        # Apply torch.compile if requested
        if compile and hasattr(torch, 'compile'):
            try:
                self.model = torch.compile(self.model, mode="max-autotune")
                print("✅ Model compiled with torch.compile")
            except Exception as e:
                print(f"⚠️  torch.compile failed, using standard model: {e}")
        
        # Beta schedule
        if beta_schedule == "cosine":
            self.betas = self._cosine_beta_schedule(num_timesteps)
        else:  # linear
            self.betas = torch.linspace(1e-4, 0.02, num_timesteps, device=device)
        
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # Precompute values for sampling
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)
        
        print(f"Explicit Conditioning Trainer initialized:")
        print(f"   - Number of timesteps: {num_timesteps}")
        print(f"   - Beta schedule: {beta_schedule}")
        print(f"   - Device: {device}")
        print(f"   - AMP: {amp}")
        print(f"   - Compiled: {compile}")
    
    def _cosine_beta_schedule(self, num_timesteps):
        """Cosine beta schedule as in Improved DDPM."""
        steps = num_timesteps + 1
        x = torch.linspace(0, num_timesteps, steps, device=self.device)
        alphas_cumprod = torch.cos(((x / num_timesteps) + 0.008) / 1.008 * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, 0.0001, 0.9999)
    
    def add_noise(self, x_start, t):
        """Add noise according to diffusion schedule."""
        noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        
        x_noisy = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
        return x_noisy, noise
    
    def train_step(self, x, conditioning, optimizer, scaler=None):
        """Single training step with classifier-free guidance conditioning dropout."""
        batch_size = x.shape[0]
        
        # Sample random timesteps
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=self.device)
        
        # Add noise
        x_noisy, noise = self.add_noise(x, t)
        
        # Normalize timesteps to [0, 1]
        t_normalized = t.float() / self.num_timesteps
        
        # Apply conditioning dropout for classifier-free guidance
        # Randomly zero out conditioning with probability cfg_p
        dropout_mask = torch.rand(batch_size, device=self.device) > self.cfg_p
        conditioning_dropped = conditioning.clone()
        conditioning_dropped[~dropout_mask] = 0.0  # Zero conditioning for dropout samples
        
        # Predict noise with potentially dropped conditioning
        if self.amp and scaler is not None:
            with torch.cuda.amp.autocast():
                predicted_noise = self.model(x_noisy, t_normalized.unsqueeze(-1), conditioning_dropped)
                loss = F.mse_loss(predicted_noise, noise)
            
            # Backward pass with mixed precision
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            predicted_noise = self.model(x_noisy, t_normalized.unsqueeze(-1), conditioning_dropped)
            loss = F.mse_loss(predicted_noise, noise)
            
            # Standard backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            
            optimizer.step()
        
        return loss.item()
    
    def p_sample(self, x, t, conditioning, sampler="ddim"):
        """Sample from the posterior distribution."""
        batch_size = x.shape[0]
        
        # Compute timestep-dependent scalars
        alpha_t = self.alphas[t].view(1, 1, 1)
        beta_t = self.betas[t].view(1, 1, 1)
        alpha_bar_t = self.alphas_cumprod[t].view(1, 1, 1)
        alpha_bar_tm1 = self.alphas_cumprod[t-1].view(1, 1, 1) if t > 0 else torch.ones(1, 1, 1, device=self.device)
        
        if sampler == "ddpm":
            # DDPM posterior sampling
            predicted_noise = self.model(x, (t / self.num_timesteps) * torch.ones(batch_size, 1, device=self.device), conditioning)
            
            mean = (1 / torch.sqrt(alpha_t)) * (x - (beta_t / torch.sqrt(1 - alpha_bar_t)) * predicted_noise)
            
            if t > 0:
                tilde_beta_t = (1 - alpha_bar_tm1) / (1 - alpha_bar_t) * beta_t
                noise = torch.randn_like(x)
                x = mean + torch.sqrt(tilde_beta_t) * noise
            else:
                x = mean
        
        elif sampler == "ddim":
            # DDIM deterministic sampling
            predicted_noise = self.model(x, (t / self.num_timesteps) * torch.ones(batch_size, 1, device=self.device), conditioning)
            
            x = torch.sqrt(alpha_bar_tm1) * (x / torch.sqrt(alpha_bar_t) - torch.sqrt(1/alpha_bar_t - 1) * predicted_noise) + torch.sqrt(1 - alpha_bar_tm1) * predicted_noise
        
        return x
    
    def get_predicted_noise(self, x, t, conditioning):
        """Get predicted noise from the model."""
        batch_size = x.shape[0]
        t_normalized = (t / self.num_timesteps) * torch.ones(batch_size, 1, device=self.device)
        return self.model(x, t_normalized, conditioning)
    
    def guided_sample_step(self, x, t, conditioning, sampler="ddim", cfg_scale=7.5):
        """Single guided sampling step using classifier-free guidance on predicted noise."""
        batch_size = x.shape[0]
        
        # Get conditional and unconditional noise predictions
        predicted_noise_cond = self.get_predicted_noise(x, t, conditioning)
        zero_conditioning = torch.zeros_like(conditioning)
        predicted_noise_uncond = self.get_predicted_noise(x, t, zero_conditioning)
        
        # Blend noise predictions using classifier-free guidance
        if cfg_scale > 1.0:
            predicted_noise = predicted_noise_uncond + cfg_scale * (predicted_noise_cond - predicted_noise_uncond)
        else:
            predicted_noise = predicted_noise_cond
        
        # Apply DDPM or DDIM step with blended noise
        if sampler == "ddpm":
            # DDPM posterior sampling
            alpha_t = self.alphas[t].view(1, 1, 1)
            beta_t = self.betas[t].view(1, 1, 1)
            alpha_bar_t = self.alphas_cumprod[t].view(1, 1, 1)
            alpha_bar_tm1 = self.alphas_cumprod[t-1].view(1, 1, 1) if t > 0 else torch.ones(1, 1, 1, device=self.device)
            
            mean = (1 / torch.sqrt(alpha_t)) * (x - (beta_t / torch.sqrt(1 - alpha_bar_t)) * predicted_noise)
            
            if t > 0:
                tilde_beta_t = (1 - alpha_bar_tm1) / (1 - alpha_bar_t) * beta_t
                noise = torch.randn_like(x)
                x = mean + torch.sqrt(tilde_beta_t) * noise
            else:
                x = mean
                
        elif sampler == "ddim":
            # DDIM deterministic sampling
            alpha_bar_t = self.alphas_cumprod[t].view(1, 1, 1)
            alpha_bar_tm1 = self.alphas_cumprod[t-1].view(1, 1, 1) if t > 0 else torch.ones(1, 1, 1, device=self.device)
            
            x = torch.sqrt(alpha_bar_tm1) * (x / torch.sqrt(alpha_bar_t) - torch.sqrt(1/alpha_bar_t - 1) * predicted_noise) + torch.sqrt(1 - alpha_bar_tm1) * predicted_noise
        
        return x
    
    def sample(self, conditioning, num_samples=1, sampler="ddim", sample_steps=None, cfg_scale=7.5):
        """Generate samples using the trained model with classifier-free guidance."""
        self.model.eval()
        
        if sample_steps is None:
            sample_steps = self.num_timesteps
        
        # Create timestep indices for reduced sampling (robust to sample_steps < num_timesteps)
        if sample_steps >= self.num_timesteps:
            timesteps = torch.arange(self.num_timesteps, device=self.device)
        else:
            # Use linspace and convert to long, ensuring unique timesteps within valid bounds
            timesteps = torch.linspace(0, self.num_timesteps - 1, sample_steps, device=self.device).long()
            # Ensure unique timesteps and clamp within valid bounds
            timesteps = torch.unique(timesteps)
            timesteps = torch.clamp(timesteps, 0, self.num_timesteps - 1)
            # Ensure the last index corresponds to the final timestep
            if timesteps[-1] != self.num_timesteps - 1:
                timesteps = torch.cat([timesteps, torch.tensor([self.num_timesteps - 1], device=self.device)])
                timesteps = torch.unique(timesteps)
        
        with torch.no_grad():
            # Start from pure noise
            x = torch.randn(num_samples, 1, self.model.sequence_length, device=self.device)
            
            # Reverse diffusion process with classifier-free guidance
            for i, t in enumerate(tqdm(reversed(timesteps), desc="Generating samples")):
                if cfg_scale > 1.0:
                    # Use guided sampling that blends predicted noise rather than full states
                    x = self.guided_sample_step(x, t, conditioning, sampler, cfg_scale)
                else:
                    # No guidance, use standard sampling
                    x = self.p_sample(x, t.item(), conditioning, sampler)
        
        return x

class EMAModel:
    """Exponential Moving Average of model weights."""
    
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Register model parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """Update EMA parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        """Apply EMA parameters to model."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        """Restore original parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

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
    
    # Calculate log returns
    returns = np.log(data['Close'] / data['Close'].shift(1)).dropna()
    
    print(f"Loaded {len(returns)} days of return data")
    print(f"Date range: {returns.index[0]} to {returns.index[-1]}")
    return returns

def create_conditioning_vectors(returns, seq_len, vol_window, val_split=0.2):
    """Create conditioning vectors with regime classification and volatility scalar."""
    print(f"Creating conditioning vectors with vol_window={vol_window}...")
    
    # Calculate rolling volatility
    rolling_vol = returns.rolling(window=vol_window).std()
    
    # Create conditioning vectors for all sequences first
    conditioning_vectors = []
    regime_labels = []
    
    for i in range(len(returns) - seq_len + 1):
        # Get sequence returns
        seq_returns = returns.iloc[i:i+seq_len]
        
        # Compute cumulative return for trend
        cum_return = seq_returns.sum()
        trend = 1 if cum_return > 0 else 0  # 1 for Up, 0 for Down
        
        # Get volatility for the sequence (use last vol_window values)
        seq_vol = rolling_vol.iloc[i:i+seq_len].iloc[-vol_window:].mean()
        
        # For now, use raw volatility (will be scaled after split)
        conditioning_vectors.append([0, 0, 0, 0, seq_vol])  # Placeholder regime, will be set after scaling
        regime_labels.append('placeholder')  # Will be set after scaling
    
    conditioning_vectors = np.array(conditioning_vectors)
    
    # Now split based on number of sequences, not raw returns length
    num_sequences = len(conditioning_vectors)
    train_split_idx = int(num_sequences * (1 - val_split))
    
    # Split sequences for training/validation
    train_sequences = conditioning_vectors[:train_split_idx]
    train_vol_values = train_sequences[:, -1]  # Extract volatility values
    
    # Fit scaler on training sequences
    vol_scaler = StandardScaler()
    train_vol_scaled = vol_scaler.fit_transform(train_vol_values.reshape(-1, 1)).flatten()
    
    # Compute volatility threshold (median of training data)
    vol_threshold = np.median(train_vol_scaled)
    
    # Now properly set the conditioning vectors with scaled values and correct regimes
    for i in range(len(conditioning_vectors)):
        # Get sequence returns
        seq_returns = returns.iloc[i:i+seq_len]
        
        # Compute cumulative return for trend
        cum_return = seq_returns.sum()
        trend = 1 if cum_return > 0 else 0  # 1 for Up, 0 for Down
        
        # Get volatility for the sequence (use last vol_window values)
        seq_vol = rolling_vol.iloc[i:i+seq_len].iloc[-vol_window:].mean()
        
        # Scale volatility using training scaler
        if not pd.isna(seq_vol):
            seq_vol_scaled = vol_scaler.transform([[seq_vol]])[0, 0]
        else:
            seq_vol_scaled = 0.0
        
        # Classify volatility regime
        vol_regime = 1 if seq_vol_scaled > vol_threshold else 0  # 1 for High, 0 for Low
        
        # Create regime one-hot: [Up-Low, Up-High, Down-Low, Down-High]
        regime_idx = (1 - trend) * 2 + vol_regime
        regime_onehot = np.zeros(4)
        regime_onehot[regime_idx] = 1
        
        # Set the conditioning vector
        conditioning_vectors[i] = np.concatenate([regime_onehot, [seq_vol_scaled]])
        regime_labels[i] = REGIME_ORDER[regime_idx]
    
    # Save metadata
    metadata = {
        'vol_window': vol_window,
        'vol_threshold': vol_threshold,
        'vol_scaler_mean': vol_scaler.mean_[0],
        'vol_scaler_scale': vol_scaler.scale_[0],
        'regime_order': REGIME_ORDER,
        'train_split_idx': train_split_idx,
        'val_split': val_split,
        'description': 'sigma_star = mean of rolling std over last vol_window points, z-scored with training scaler'
    }
    
    print(f"Generated {len(conditioning_vectors)} conditioning vectors")
    print(f"Volatility threshold: {vol_threshold:.4f}")
    print(f"Regime distribution: {dict(zip(*np.unique(regime_labels, return_counts=True)))}")
    
    return conditioning_vectors, regime_labels, metadata

def create_sequences(returns, seq_len):
    """Create sequences for training."""
    print(f"Creating sequences of length {seq_len}...")
    
    sequences = []
    for i in range(len(returns) - seq_len + 1):
        seq = returns.iloc[i:i+seq_len].values
        sequences.append(seq)
    
    X = np.array(sequences)
    X = X[:, np.newaxis, :]  # Add channel dimension at axis 1: [N, 1, T]
    print(f"Created {len(X)} sequences")
    return X

def train_model(X, conditioning_vectors, regime_labels, metadata, args):
    """Train the explicit conditioning DDPM."""
    print("Training explicit conditioning DDPM...")
    
    # Split data
    split_idx = metadata['train_split_idx']
    X_train = X[:split_idx]
    X_val = X[split_idx:]
    cond_train = conditioning_vectors[:split_idx]
    cond_val = conditioning_vectors[split_idx:]
    
    print(f"Training set: {len(X_train)} sequences")
    print(f"Validation set: {len(X_val)} sequences")
    
    # Initialize model and trainer
    model = ExplicitConditioningDDPM(
        sequence_length=args.seq_len,
        conditioning_dim=5,  # 4 regime + 1 volatility
        hidden_dim=args.hidden_dim
    )
    
    trainer = ExplicitConditioningTrainer(
        model, 
        num_timesteps=args.num_timesteps, 
        beta_schedule=args.beta_schedule, 
        device=args.device,
        grad_clip=args.grad_clip,
        cfg_p=args.cfg_p
    )
    
    # Initialize EMA (handle both --use-ema and --no-ema flags)
    use_ema = args.use_ema and not args.no_ema
    ema = EMAModel(model, decay=0.999) if use_ema else None
    
    # Prepare data
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(cond_train, dtype=torch.float32)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(cond_val, dtype=torch.float32)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(args.epochs):
        # Training
        model.train()
        epoch_train_losses = []
        
        for batch_x, batch_conditioning in train_loader:
            batch_x = batch_x.to(args.device)
            batch_conditioning = batch_conditioning.to(args.device)
            
            loss = trainer.train_step(batch_x, batch_conditioning, optimizer)
            epoch_train_losses.append(loss)
            
            # Update EMA
            if ema is not None:
                ema.update()
        
        # Validation
        model.eval()
        epoch_val_losses = []
        
        with torch.no_grad():
            for batch_x, batch_conditioning in val_loader:
                batch_x = batch_x.to(args.device)
                batch_conditioning = batch_conditioning.to(args.device)
                
                # Sample random timesteps for validation
                batch_size = batch_x.shape[0]
                t = torch.randint(0, args.num_timesteps, (batch_size,), device=args.device)
                
                # Add noise
                x_noisy, noise = trainer.add_noise(batch_x, t)
                
                # Predict noise
                t_normalized = t.float() / args.num_timesteps
                predicted_noise = model(x_noisy, t_normalized.unsqueeze(-1), batch_conditioning)
                
                # Compute loss
                loss = F.mse_loss(predicted_noise, noise)
                epoch_val_losses.append(loss.item())
        
        # Record losses
        avg_train_loss = np.mean(epoch_train_losses)
        avg_val_loss = np.mean(epoch_val_losses)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        # Learning rate scheduling
        scheduler.step()
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            # Save best model
            os.makedirs(f"{args.results_dir}/checkpoints", exist_ok=True)
            torch.save(model.state_dict(), f"{args.results_dir}/checkpoints/best_model.pth")
            if ema is not None:
                torch.save(ema.shadow, f"{args.results_dir}/checkpoints/best_model_ema.pth")
        else:
            patience_counter += 1
        
        # Log progress
        if epoch % 10 == 0:
            print(f"   Epoch {epoch}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}")
        
        # Early stopping check
        if patience_counter >= args.patience:
            print(f"   Early stopping at epoch {epoch}")
            break
    
    # Save final model
    torch.save(model.state_dict(), f"{args.results_dir}/checkpoints/final_model.pth")
    if ema is not None:
        torch.save(ema.shadow, f"{args.results_dir}/checkpoints/final_model_ema.pth")
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_epoch': epoch - patience_counter
    }
    
    with open(f"{args.results_dir}/training_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save CSV
    df = pd.DataFrame({
        'epoch': range(len(train_losses)),
        'train_loss': train_losses,
        'val_loss': val_losses
    })
    df.to_csv(f"{args.results_dir}/training_history.csv", index=False)
    
    print("Training completed!")
    return model, trainer, ema, history

def evaluate_controllability(model, trainer, ema, conditioning_vectors, args):
    """Evaluate controllability of the model."""
    print("Evaluating controllability...")
    
    # Use EMA model if available
    if ema is not None:
        ema.apply_shadow()
    
    try:
        # Generate samples
        num_samples = min(1000, len(conditioning_vectors))
        device = next(model.parameters()).device
        conditioning_tensor = torch.tensor(conditioning_vectors[:num_samples], dtype=torch.float32, device=device)
        
        samples = trainer.sample(
            conditioning_tensor, 
            num_samples=num_samples, 
            sampler=args.sampler, 
            sample_steps=args.sample_steps,
            cfg_scale=args.cfg_scale
        )
        
        samples = samples.squeeze(1).cpu().numpy()  # Remove channel dimension
        
        # Save samples for compatibility with existing evaluation scripts
        np.save(f"{args.results_dir}/explicit_cond_returns.npy", samples)
        np.save(f"{args.results_dir}/explicit_cond_returns_flattened.npy", samples.flatten())
        
        # Compute realized volatility for generated samples (matching training definition of σ*)
        realized_vols = []
        for sample in samples:
            # Compute rolling standard deviation over vol_window, then take mean of last vol_window values
            rolling_stds = []
            for i in range(len(sample) - args.vol_window + 1):
                rolling_stds.append(np.std(sample[i:i+args.vol_window], ddof=1))
            # Take mean of last vol_window values of the rolling-σ series
            vol = np.mean(rolling_stds[-args.vol_window:])
            realized_vols.append(vol)
        
        realized_vols = np.array(realized_vols)
        
        # Scale realized volatilities using the same training scaler parameters
        realized_vols_scaled = (realized_vols - args.vol_scaler_mean) / args.vol_scaler_scale
        
        # Get target volatility from conditioning (already scaled)
        target_vols = conditioning_vectors[:num_samples, -1]  # Last dimension is volatility
        
        # Compute controllability metrics using scaled values
        mae = mean_absolute_error(target_vols, realized_vols_scaled)
        r2 = r2_score(target_vols, realized_vols_scaled)
        
        # Save metrics
        control_metrics = {
            'mae': mae,
            'r2': r2,
            'num_samples': num_samples
        }
        
        with open(f"{args.results_dir}/control_metrics.json", 'w') as f:
            json.dump(control_metrics, f, indent=2)
        
        # Create controllability plot
        plt.figure(figsize=(10, 8))
        plt.scatter(target_vols, realized_vols_scaled, alpha=0.6, s=20)
        plt.plot([target_vols.min(), target_vols.max()], [target_vols.min(), target_vols.max()], 'r--', lw=2, label='y=x')
        plt.xlabel('Target Volatility (σ*) - Scaled')
        plt.ylabel('Realized Volatility (σ̂) - Scaled')
        plt.title('Controllability: Target vs Realized Volatility (Same Scale)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add metrics text
        plt.text(0.05, 0.95, f'MAE: {mae:.4f}\nR²: {r2:.4f}', 
                 transform=plt.gca().transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f"{args.results_dir}/figures/controllability_scatter.pdf", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create reliability/calibration curve using binned averages
        plt.figure(figsize=(10, 8))
        # Create bins for target volatility
        num_bins = 10
        bin_edges = np.linspace(target_vols.min(), target_vols.max(), num_bins + 1)
        bin_indices = np.digitize(target_vols, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, num_bins - 1)
        
        bin_means_target = []
        bin_means_realized = []
        bin_stds_realized = []
        bin_counts = []
        
        for i in range(num_bins):
            mask = (bin_indices == i)
            if np.sum(mask) > 0:
                bin_means_target.append(np.mean(target_vols[mask]))
                bin_means_realized.append(np.mean(realized_vols_scaled[mask]))
                bin_stds_realized.append(np.std(realized_vols_scaled[mask]))
                bin_counts.append(np.sum(mask))
        
        bin_means_target = np.array(bin_means_target)
        bin_means_realized = np.array(bin_means_realized)
        bin_stds_realized = np.array(bin_stds_realized)
        bin_counts = np.array(bin_counts)
        
        # Plot binned averages with error bars
        plt.errorbar(bin_means_target, bin_means_realized, yerr=bin_stds_realized, 
                    fmt='o-', capsize=5, capthick=2, linewidth=2, markersize=8, 
                    label='Binned Averages ± 1σ')
        plt.plot([target_vols.min(), target_vols.max()], [target_vols.min(), target_vols.max()], 
                'r--', lw=2, label='Perfect Calibration')
        plt.xlabel('Target Volatility (σ*) - Scaled')
        plt.ylabel('Realized Volatility (σ̂) - Scaled')
        plt.title('Controllability: Reliability/Calibration Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{args.results_dir}/figures/controllability_calibration.pdf", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create residuals plot
        residuals = realized_vols_scaled - target_vols
        plt.figure(figsize=(10, 8))
        plt.scatter(target_vols, residuals, alpha=0.6, s=20)
        plt.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Zero Residual')
        plt.xlabel('Target Volatility (σ*) - Scaled')
        plt.ylabel('Residual (Realized - Target)')
        plt.title('Controllability: Residuals Analysis')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{args.results_dir}/figures/controllability_residuals.pdf", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save calibration data to CSV
        calibration_data = pd.DataFrame({
            'bin_target_mean': bin_means_target,
            'bin_realized_mean': bin_means_realized,
            'bin_realized_std': bin_stds_realized,
            'bin_count': bin_counts
        })
        calibration_data.to_csv(f"{args.results_dir}/controllability_calibration.csv", index=False)
        
        # Save residuals data to CSV
        residuals_data = pd.DataFrame({
            'target_volatility': target_vols,
            'realized_volatility': realized_vols_scaled,
            'residual': residuals
        })
        residuals_data.to_csv(f"{args.results_dir}/controllability_residuals.csv", index=False)
        
        # Create LaTeX table
        with open(f"{args.results_dir}/tables/control_metrics.tex", 'w') as f:
            f.write("\\begin{table}[h]\n")
            f.write("\\centering\n")
            f.write("\\begin{tabular}{lc}\n")
            f.write("\\hline\n")
            f.write("Metric & Value \\\\\n")
            f.write("\\hline\n")
            f.write(f"MAE & {mae:.4f} \\\\\n")
            f.write(f"R² & {r2:.4f} \\\\\n")
            f.write(f"Number of Samples & {num_samples} \\\\\n")
            f.write("\\hline\n")
            f.write("\\end{tabular}\n")
            f.write("\\caption{Controllability Metrics}\n")
            f.write("\\label{tab:control_metrics}\n")
            f.write("\\end{table}\n")
        
        print(f"Controllability evaluation completed - MAE: {mae:.4f}, R²: {r2:.4f}")
        return control_metrics
        
    finally:
        # Restore original weights after evaluation
        if ema is not None:
            ema.restore()

def evaluate_regime_accuracy(model, trainer, ema, conditioning_vectors, args):
    """Evaluate regime classification accuracy."""
    print("Evaluating regime accuracy...")
    
    # Use EMA model if available
    if ema is not None:
        ema.apply_shadow()
    
    try:
        # Generate samples for each regime
        num_samples_per_regime = 50
        regime_samples = {}
        regime_accuracies = {}
    
        for regime_idx, regime_name in enumerate(REGIME_ORDER):
            # Create conditioning for this regime
            regime_conditioning = np.zeros((num_samples_per_regime, 5))
            regime_conditioning[:, regime_idx] = 1  # Set regime one-hot
            
            # Set representative volatility (median for Low, 90th percentile for High)
            if 'Low' in regime_name:
                vol_value = np.median(conditioning_vectors[:, -1])
            else:  # High
                vol_value = np.percentile(conditioning_vectors[:, -1], 90)
            
            regime_conditioning[:, -1] = vol_value
            
            # Generate samples
            device = next(model.parameters()).device
            conditioning_tensor = torch.tensor(regime_conditioning, dtype=torch.float32, device=device)
            
            samples = trainer.sample(
                conditioning_tensor, 
                num_samples=num_samples_per_regime, 
                sampler=args.sampler, 
                sample_steps=args.sample_steps,
                cfg_scale=args.cfg_scale
            )
            
            samples = samples.squeeze(1).cpu().numpy()
            regime_samples[regime_name] = samples
            
            # Classify generated samples
            correct_classifications = 0
            for sample in samples:
                # Compute trend (cumulative return sign)
                cum_return = sample.sum()
                trend = 1 if cum_return > 0 else 0
                
                # Compute volatility regime using same σ* definition as controllability
                # Compute rolling standard deviation over vol_window, then take mean of last vol_window values
                rolling_stds = []
                for i in range(len(sample) - args.vol_window + 1):
                    rolling_stds.append(np.std(sample[i:i+args.vol_window], ddof=1))
                # Take mean of last vol_window values of the rolling-σ series
                vol = np.mean(rolling_stds[-args.vol_window:])
                vol_scaled = (vol - args.vol_scaler_mean) / args.vol_scaler_scale
                vol_regime = 1 if vol_scaled > args.vol_threshold else 0
                
                # Determine regime
                predicted_regime_idx = (1 - trend) * 2 + vol_regime
                predicted_regime = REGIME_ORDER[predicted_regime_idx]
                
                if predicted_regime == regime_name:
                    correct_classifications += 1
            
            accuracy = correct_classifications / num_samples_per_regime
            regime_accuracies[regime_name] = accuracy
        
        # Overall accuracy
        overall_accuracy = np.mean(list(regime_accuracies.values()))
        
        # Save metrics
        regime_metrics = {
            'regime_accuracies': regime_accuracies,
            'overall_accuracy': overall_accuracy,
            'num_samples_per_regime': num_samples_per_regime
        }
        
        with open(f"{args.results_dir}/regime_metrics.json", 'w') as f:
            json.dump(regime_metrics, f, indent=2)
        
        # Create regime grid plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, (regime_name, samples) in enumerate(regime_samples.items()):
            ax = axes[i]
            
            # Plot sample paths
            for j in range(min(20, len(samples))):
                ax.plot(samples[j], alpha=0.7, linewidth=0.8)
            
            ax.set_title(f'{regime_name} (Acc: {regime_accuracies[regime_name]:.2%})')
            ax.set_ylabel('Returns')
            ax.grid(True, alpha=0.3)
            
            if i >= 2:  # Bottom row
                ax.set_xlabel('Time Step')
        
        plt.tight_layout()
        plt.savefig(f"{args.results_dir}/figures/regime_grid.pdf", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create confusion matrix heatmap
        # Collect all predictions and true labels for confusion matrix
        all_true_labels = []
        all_predicted_labels = []
        
        for regime_name, samples in regime_samples.items():
            true_regime_idx = REGIME_ORDER.index(regime_name)
            for sample in samples:
                # Compute trend (cumulative return sign)
                cum_return = sample.sum()
                trend = 1 if cum_return > 0 else 0
                
                # Compute volatility regime using same σ* definition
                rolling_stds = []
                for i in range(len(sample) - args.vol_window + 1):
                    rolling_stds.append(np.std(sample[i:i+args.vol_window], ddof=1))
                vol = np.mean(rolling_stds[-args.vol_window:])
                vol_scaled = (vol - args.vol_scaler_mean) / args.vol_scaler_scale
                vol_regime = 1 if vol_scaled > args.vol_threshold else 0
                
                # Determine predicted regime
                predicted_regime_idx = (1 - trend) * 2 + vol_regime
                predicted_regime = REGIME_ORDER[predicted_regime_idx]
                
                all_true_labels.append(regime_name)
                all_predicted_labels.append(predicted_regime)
        
        # Create confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(all_true_labels, all_predicted_labels, labels=REGIME_ORDER)
        
        # Normalize by row (true labels)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized, nan=0.0)
        
        # Plot confusion matrix heatmap
        plt.figure(figsize=(10, 8))
        im = plt.imshow(cm_normalized, cmap='Blues', aspect='auto')
        plt.colorbar(im, fraction=0.046, pad=0.04)
        
        # Add text annotations
        for i in range(len(REGIME_ORDER)):
            for j in range(len(REGIME_ORDER)):
                text = plt.text(j, i, f'{cm_normalized[i, j]:.2f}',
                               ha="center", va="center", color="black" if cm_normalized[i, j] < 0.5 else "white")
        
        plt.xlabel('Predicted Regime')
        plt.ylabel('True Regime')
        plt.title('Regime Classification: Confusion Matrix (Normalized by Row)')
        plt.xticks(range(len(REGIME_ORDER)), REGIME_ORDER, rotation=45)
        plt.yticks(range(len(REGIME_ORDER)), REGIME_ORDER)
        
        plt.tight_layout()
        plt.savefig(f"{args.results_dir}/figures/regime_confusion_matrix.pdf", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save confusion matrix data to CSV
        cm_df = pd.DataFrame(cm_normalized, index=REGIME_ORDER, columns=REGIME_ORDER)
        cm_df.to_csv(f"{args.results_dir}/regime_confusion_matrix.csv")
        
        # Create LaTeX table
        with open(f"{args.results_dir}/tables/regime_accuracy.tex", 'w') as f:
            f.write("\\begin{table}[h]\n")
            f.write("\\centering\n")
            f.write("\\begin{tabular}{lc}\n")
            f.write("\\hline\n")
            f.write("Regime & Accuracy \\\\\n")
            f.write("\\hline\n")
            for regime_name, accuracy in regime_accuracies.items():
                f.write(f"{regime_name} & {accuracy:.2%} \\\\\n")
            f.write("\\hline\n")
            f.write(f"Overall & {overall_accuracy:.2%} \\\\\n")
            f.write("\\hline\n")
            f.write("\\end{tabular}\n")
            f.write("\\caption{Regime Classification Accuracy}\n")
            f.write("\\label{tab:regime_accuracy}\n")
            f.write("\\end{table}\n")
        
        print(f"Regime accuracy evaluation completed - Overall: {overall_accuracy:.2%}")
        return regime_metrics
    
    finally:
        # Restore original weights after evaluation
        if ema is not None:
            ema.restore()

def evaluate_distributional_fidelity(model, trainer, ema, conditioning_vectors, real_returns, args):
    """Evaluate distributional fidelity."""
    print("Evaluating distributional fidelity...")
    
    # Use EMA model if available
    if ema is not None:
        ema.apply_shadow()
    
    try:
        # Generate samples
        num_samples = min(1000, len(conditioning_vectors))
        device = next(model.parameters()).device
        conditioning_tensor = torch.tensor(conditioning_vectors[:num_samples], dtype=torch.float32, device=device)
        
        samples = trainer.sample(
            conditioning_tensor, 
            num_samples=num_samples, 
            sampler=args.sampler, 
            sample_steps=args.sample_steps,
            cfg_scale=args.cfg_scale
        )
        
        samples = samples.squeeze(1).cpu().numpy()
        synthetic_returns = samples.flatten()
    
        # Basic statistics
        real_stats = {
            'mean': np.mean(real_returns),
            'std': np.std(real_returns),
            'skew': scipy.stats.skew(real_returns),
            'kurtosis': scipy.stats.kurtosis(real_returns)
        }
        
        synthetic_stats = {
            'mean': np.mean(synthetic_returns),
            'std': np.std(synthetic_returns),
            'skew': scipy.stats.skew(synthetic_returns),
            'kurtosis': scipy.stats.kurtosis(synthetic_returns)
        }
        
        # KS test
        ks_stat, ks_pvalue = scipy.stats.ks_2samp(real_returns, synthetic_returns)
        
        # Hill tail index (for positive tail) - custom implementation
        def estimate_hill_index(data, quantile=0.1):
            """Estimate Hill tail index from largest values."""
            if len(data) < 10:
                return np.nan
            
            # Get positive values and sort
            positive_data = data[data > 0]
            if len(positive_data) < 10:
                return np.nan
            
            # Sort in descending order and take top quantile
            sorted_data = np.sort(positive_data)[::-1]
            n_tail = max(1, int(len(sorted_data) * quantile))
            tail_data = sorted_data[:n_tail]
            
            # Compute Hill estimator: 1/mean(log(x_i/x_min))
            if len(tail_data) < 2:
                return np.nan
            
            x_min = tail_data[-1]
            if x_min <= 0:
                return np.nan
            
            log_ratios = np.log(tail_data / x_min)
            hill_index = 1.0 / np.mean(log_ratios)
            
            return hill_index
        
        real_hill = estimate_hill_index(real_returns)
        synthetic_hill = estimate_hill_index(synthetic_returns)
        
        # MMD (simplified version using first two moments)
        def mmd_estimate(x, y):
            """Simplified MMD estimate using first two moments."""
            x_mean = np.mean(x)
            y_mean = np.mean(y)
            x_var = np.var(x)
            y_var = np.var(y)
            
            # Simplified MMD using means and variances
            mmd = (x_mean - y_mean)**2 + (x_var - y_var)**2
            return mmd
        
        mmd_stat = mmd_estimate(real_returns, synthetic_returns)
        
        # Save metrics
        dist_metrics = {
            'real_stats': real_stats,
            'synthetic_stats': synthetic_stats,
            'ks_stat': ks_stat,
            'ks_pvalue': ks_pvalue,
            'real_hill': real_hill,
            'synthetic_hill': synthetic_hill,
            'mmd_stat': mmd_stat
        }
        
        with open(f"{args.results_dir}/dist_metrics.json", 'w') as f:
            json.dump(dist_metrics, f, indent=2)
        
        # Create tail distribution plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # ECDF comparison
        real_sorted = np.sort(real_returns)
        synthetic_sorted = np.sort(synthetic_returns)
        
        ax1.plot(real_sorted, np.linspace(0, 1, len(real_sorted)), label='Real', linewidth=2)
        ax1.plot(synthetic_sorted, np.linspace(0, 1, len(synthetic_sorted)), label='Synthetic', linewidth=2)
        ax1.set_xlabel('Returns')
        ax1.set_ylabel('Cumulative Probability')
        ax1.set_title('ECDF Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # PDF comparison with zoomed tail
        ax2.hist(real_returns, bins=50, density=True, alpha=0.7, label='Real', color='blue')
        ax2.hist(synthetic_returns, bins=50, density=True, alpha=0.7, label='Synthetic', color='red')
        ax2.set_xlabel('Returns')
        ax2.set_ylabel('Density')
        ax2.set_title('PDF Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add zoomed inset for tail
        axins = inset_axes(ax2, width="35%", height="35%", loc="upper right")
        tail_threshold = np.percentile(real_returns, 95)
        real_tail = real_returns[real_returns > tail_threshold]
        synthetic_tail = synthetic_returns[synthetic_returns > tail_threshold]
        
        axins.hist(real_tail, bins=20, density=True, alpha=0.7, label='Real', color='blue')
        axins.hist(synthetic_tail, bins=20, density=True, alpha=0.7, label='Synthetic', color='red')
        axins.set_title('Tail Zoom (95th percentile+)')
        axins.legend()
        
        plt.tight_layout()
        plt.savefig(f"{args.results_dir}/figures/tail_distribution_zoom.pdf", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create QQ plots focusing on right and left tails
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Right tail QQ plot (positive returns)
        real_positive = real_returns[real_returns > 0]
        synthetic_positive = synthetic_returns[synthetic_returns > 0]
        
        if len(real_positive) > 0 and len(synthetic_positive) > 0:
            # Sort positive returns
            real_positive_sorted = np.sort(real_positive)
            synthetic_positive_sorted = np.sort(synthetic_positive)
            
            # Create quantiles for comparison
            quantiles = np.linspace(0, 1, min(len(real_positive), len(synthetic_positive)))
            real_quantiles = np.quantile(real_positive_sorted, quantiles)
            synthetic_quantiles = np.quantile(synthetic_positive_sorted, quantiles)
            
            ax1.scatter(real_quantiles, synthetic_quantiles, alpha=0.6, s=20)
            ax1.plot([real_quantiles.min(), real_quantiles.max()], 
                    [synthetic_quantiles.min(), synthetic_quantiles.max()], 'r--', lw=2, label='y=x')
            ax1.set_xlabel('Real Returns (Positive)')
            ax1.set_ylabel('Synthetic Returns (Positive)')
            ax1.set_title('Right Tail QQ Plot')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Save right tail QQ data
            right_tail_data = pd.DataFrame({
                'real_quantiles': real_quantiles,
                'synthetic_quantiles': synthetic_quantiles,
                'quantiles': quantiles
            })
            right_tail_data.to_csv(f"{args.results_dir}/distribution_right_tail_qq.csv", index=False)
        
        # Left tail QQ plot (negative returns)
        real_negative = real_returns[real_returns < 0]
        synthetic_negative = synthetic_returns[synthetic_returns < 0]
        
        if len(real_negative) > 0 and len(synthetic_negative) > 0:
            # Sort negative returns (in descending order for left tail)
            real_negative_sorted = np.sort(real_negative)[::-1]
            synthetic_negative_sorted = np.sort(synthetic_negative)[::-1]
            
            # Create quantiles for comparison
            quantiles = np.linspace(0, 1, min(len(real_negative), len(synthetic_negative)))
            real_quantiles = np.quantile(real_negative_sorted, quantiles)
            synthetic_quantiles = np.quantile(synthetic_negative_sorted, quantiles)
            
            ax2.scatter(real_quantiles, synthetic_quantiles, alpha=0.6, s=20)
            ax2.plot([real_quantiles.min(), real_quantiles.max()], 
                    [synthetic_quantiles.min(), synthetic_quantiles.max()], 'r--', lw=2, label='y=x')
            ax2.set_xlabel('Real Returns (Negative)')
            ax2.set_ylabel('Synthetic Returns (Negative)')
            ax2.set_title('Left Tail QQ Plot')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Save left tail QQ data
            left_tail_data = pd.DataFrame({
                'real_quantiles': real_quantiles,
                'synthetic_quantiles': synthetic_quantiles,
                'quantiles': quantiles
            })
            left_tail_data.to_csv(f"{args.results_dir}/distribution_left_tail_qq.csv", index=False)
        
        plt.tight_layout()
        plt.savefig(f"{args.results_dir}/figures/distribution_qq_plots.pdf", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create LaTeX table
        with open(f"{args.results_dir}/tables/dist_metrics.tex", 'w') as f:
            f.write("\\begin{table}[h]\n")
            f.write("\\centering\n")
            f.write("\\begin{tabular}{lcc}\n")
            f.write("\\hline\n")
            f.write("Metric & Real & Synthetic \\\\\n")
            f.write("\\hline\n")
            f.write(f"Mean & {real_stats['mean']:.6f} & {synthetic_stats['mean']:.6f} \\\\\n")
            f.write(f"Std & {real_stats['std']:.6f} & {synthetic_stats['std']:.6f} \\\\\n")
            f.write(f"Skew & {real_stats['skew']:.6f} & {synthetic_stats['skew']:.6f} \\\\\n")
            f.write(f"Kurtosis & {real_stats['kurtosis']:.6f} & {synthetic_stats['kurtosis']:.6f} \\\\\n")
            f.write("\\hline\n")
            f.write(f"KS Statistic & \\multicolumn{{2}}{{c}}{{{ks_stat:.6f}}} \\\\\n")
            f.write(f"KS p-value & \\multicolumn{{2}}{{c}}{{{ks_pvalue:.6f}}} \\\\\n")
            f.write(f"Real Hill Index & \\multicolumn{{2}}{{c}}{{{real_hill:.6f}}} \\\\\n")
            f.write(f"Synthetic Hill Index & \\multicolumn{{2}}{{c}}{{{synthetic_hill:.6f}}} \\\\\n")
            f.write(f"MMD Statistic & \\multicolumn{{2}}{{c}}{{{mmd_stat:.6f}}} \\\\\n")
            f.write("\\hline\n")
            f.write("\\end{tabular}\n")
            f.write("\\caption{Distributional Fidelity Metrics (Kurtosis values are excess kurtosis)}\n")
            f.write("\\label{tab:dist_metrics}\n")
            f.write("\\end{table}\n")
        
        print("Distributional fidelity evaluation completed")
        return dist_metrics
        
    finally:
        # Restore original weights after evaluation
        if ema is not None:
            ema.restore()

def march2020_case_study(model, trainer, ema, args):
    """March 2020 COVID-19 case study."""
    print("Running March 2020 case study...")
    
    # Use EMA model if available
    if ema is not None:
        ema.apply_shadow()
    
    try:
        # Load data for March 2020 period
        returns = load_and_prepare_data()
        
        # Filter March 2020 period
        march_data = returns['2020-02-01':'2020-04-30']
        print(f"March 2020 data: {len(march_data)} days")
        
        # Compute volatility for this period using same σ* definition as training
        vol_window = args.vol_window
        seq_len = args.seq_len
        
        # Create sequences of seq_len and compute rolling volatility for each
        # Then take mean of last vol_window values of the rolling-σ series
        rolling_vols = []
        for i in range(len(march_data) - seq_len + 1):
            seq_returns = march_data.iloc[i:i+seq_len]
            # Compute rolling std over vol_window within the sequence
            seq_rolling_vol = seq_returns.rolling(window=vol_window).std()
            # Take mean of last vol_window values of the rolling-σ series
            if len(seq_rolling_vol.dropna()) >= vol_window:
                last_vols = seq_rolling_vol.dropna().iloc[-vol_window:]
                rolling_vols.append(last_vols.mean())
        
        # Use the mean of all computed rolling volatilities
        avg_vol = np.mean(rolling_vols) if rolling_vols else march_data.rolling(window=vol_window).std().mean()
        
        # Scale volatility using training parameters
        vol_scaled = (avg_vol - args.vol_scaler_mean) / args.vol_scaler_scale
        
        # Determine regime: explicitly set to Down-High by default
        # Only switch to Down-Low if scaled volatility is at or below the low-volatility threshold
        # This follows the regime ordering: [Up-Low, Up-High, Down-Low, Down-High] where index 3 = Down-High
        regime_idx = 3  # Down-High = 3 (default)
        if vol_scaled <= args.vol_threshold:
            regime_idx = 2  # Down-Low = 2
        
        # Create conditioning for Down-High regime
        conditioning = np.zeros((100, 5))  # Generate 100 paths
        conditioning[:, regime_idx] = 1  # Set regime one-hot
        conditioning[:, -1] = vol_scaled  # Set volatility scalar
        
        # Generate synthetic paths
        device = next(model.parameters()).device
        conditioning_tensor = torch.tensor(conditioning, dtype=torch.float32, device=device)
        
        samples = trainer.sample(
            conditioning_tensor, 
            num_samples=100, 
            sampler=args.sampler, 
            sample_steps=args.sample_steps,
            cfg_scale=args.cfg_scale
        )
        
        samples = samples.squeeze(1).cpu().numpy()
        
        # Create case study plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        # Plot 1: Historical vs Synthetic Returns
        ax1.plot(march_data.index, march_data.values, 'b-', linewidth=2, label='Historical Returns', alpha=0.8)
        
        # Plot synthetic paths
        for i in range(min(20, len(samples))):
            ax1.plot(march_data.index, samples[i], 'r-', alpha=0.3, linewidth=0.8)
        
        ax1.set_title('March 2020 Case Study: Historical vs Synthetic Returns')
        ax1.set_ylabel('Returns')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Portfolio Value (assuming $100 initial investment)
        # Note: Returns are log returns, so use exponential compounding
        initial_value = 100
        historical_portfolio = [initial_value]
        for ret in march_data.values:
            historical_portfolio.append(historical_portfolio[-1] * np.exp(ret))
        
        ax2.plot(march_data.index, historical_portfolio[1:], 'b-', linewidth=2, label='Historical Portfolio', alpha=0.8)
        
        # Compute synthetic portfolio values using exponential compounding for log returns
        synthetic_portfolios = []
        for sample in samples:
            portfolio = [initial_value]
            for ret in sample:
                portfolio.append(portfolio[-1] * np.exp(ret))
            synthetic_portfolios.append(portfolio[1:])
        
        # Plot synthetic portfolios
        for i in range(min(20, len(synthetic_portfolios))):
            ax2.plot(march_data.index, synthetic_portfolios[i], 'r-', alpha=0.3, linewidth=0.8)
        
        ax2.set_title('Portfolio Value Evolution')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Portfolio Value ($)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{args.results_dir}/figures/march2020_case.pdf", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Compute VaR and ES for synthetic scenarios
        final_values = [portfolio[-1] for portfolio in synthetic_portfolios]
        losses = [initial_value - value for value in final_values]
        
        var_95 = np.percentile(losses, 95)
        var_99 = np.percentile(losses, 99)
        es_95 = np.mean([loss for loss in losses if loss >= var_95])
        es_99 = np.mean([loss for loss in losses if loss >= var_99])
    
        # Save case study metrics
        case_metrics = {
            'historical_volatility': float(avg_vol),
            'scaled_volatility': float(vol_scaled),
            'regime': REGIME_ORDER[regime_idx],
            'var_95': float(var_95),
            'var_99': float(var_99),
            'es_95': float(es_95),
            'es_99': float(es_99),
            'num_paths': len(samples)
        }
        
        with open(f"{args.results_dir}/case_metrics.json", 'w') as f:
            json.dump(case_metrics, f, indent=2)
        
        # Create LaTeX table
        with open(f"{args.results_dir}/tables/case_var_es.tex", 'w') as f:
            f.write("\\begin{table}[h]\n")
            f.write("\\centering\n")
            f.write("\\begin{tabular}{lc}\n")
            f.write("\\hline\n")
            f.write("Metric & Value \\\\\n")
            f.write("\\hline\n")
            f.write(f"Historical Volatility & {avg_vol:.6f} \\\\\n")
            f.write(f"Scaled Volatility & {vol_scaled:.4f} \\\\\n")
            f.write(f"Regime & {REGIME_ORDER[regime_idx]} \\\\\n")
            f.write(f"95\\% VaR & \\${var_95:.2f} \\\\\n")
            f.write(f"99\\% VaR & \\${var_99:.2f} \\\\\n")
            f.write(f"95\\% ES & \\${es_95:.2f} \\\\\n")
            f.write(f"99\\% ES & \\${es_99:.2f} \\\\\n")
            f.write(f"Number of Paths & {len(samples)} \\\\\n")
            f.write("\\hline\n")
            f.write("\\end{tabular}\n")
            f.write("\\caption{March 2020 Case Study: Portfolio Risk Metrics}\n")
            f.write("\\label{tab:case_var_es}\n")
            f.write("\\end{table}\n")
        
        print("March 2020 case study completed")
        return case_metrics
        
    finally:
        # Restore original weights after evaluation
        if ema is not None:
            ema.restore()

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Explicit Conditioning DDPM for Financial Data Synthesis')
    
    # Data parameters
    parser.add_argument('--seq-len', type=int, default=60, help='Sequence length for training')
    parser.add_argument('--vol-window', type=int, default=20, help='Volatility rolling window')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Training batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--num-timesteps', type=int, default=1000, help='Number of diffusion timesteps')
    parser.add_argument('--beta-schedule', choices=['cosine', 'linear'], default='cosine', help='Beta schedule')
    parser.add_argument('--sampler', choices=['ddpm', 'ddim'], default='ddim', help='Sampling method')
    parser.add_argument('--sample-steps', type=int, default=50, help='Number of sampling steps')
    parser.add_argument('--val-split', type=float, default=0.2, help='Validation split ratio')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--grad-clip', type=float, default=1.0, help='Gradient clipping value')
    parser.add_argument('--use-ema', action='store_true', default=False, help='Use EMA')
    parser.add_argument('--no-ema', action='store_true', default=False, help='Disable EMA (overrides --use-ema)')
    
    # Classifier-free guidance parameters
    parser.add_argument('--cfg-p', type=float, default=0.1, help='Conditioning dropout probability during training')
    parser.add_argument('--cfg-scale', type=float, default=7.5, help='Classifier-free guidance scale during sampling')
    
    # Model parameters
    parser.add_argument('--hidden-dim', type=int, default=128, help='Hidden dimension')
    
    # Output parameters
    parser.add_argument('--results-dir', type=str, default='results/explicit_cond_ddpm', help='Results directory')
    parser.add_argument('--run-id', type=str, default=None, help='Run ID (default: timestamp)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto', help='Device to use')
    
    args = parser.parse_args()
    
    # Safety check: ensure vol_window <= seq_len
    if args.vol_window > args.seq_len:
        raise ValueError(f"vol_window ({args.vol_window}) cannot be greater than seq_len ({args.seq_len})")
    
    # Set run ID if not provided
    if args.run_id is None:
        args.run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Set device
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Update results directory with run ID
    args.results_dir = f"{args.results_dir}/{args.run_id}"
    
    return args

def main():
    """Main function to run the explicit conditioning DDPM."""
    print("Explicit Conditioning DDPM for Financial Data Synthesis")
    print("=" * 80)
    
    # Parse arguments
    args = parse_args()
    
    # Set reproducibility seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Extra seeding controls for deterministic behaviour on CUDA
    if args.device == 'cuda':
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print(f"Run ID: {args.run_id}")
    print(f"Device: {args.device}")
    print(f"Results directory: {args.results_dir}")
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(f"{args.results_dir}/figures", exist_ok=True)
    os.makedirs(f"{args.results_dir}/tables", exist_ok=True)
    os.makedirs(f"{args.results_dir}/checkpoints", exist_ok=True)
    
    # Save metadata
    metadata = {
        'run_id': args.run_id,
        'timestamp': datetime.now().isoformat(),
        'args': vars(args),
        'seed': args.seed
    }
    
    with open(f"{args.results_dir}/metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Load and prepare data
    returns = load_and_prepare_data()
    
    # Create sequences and conditioning vectors
    X = create_sequences(returns, args.seq_len)
    conditioning_vectors, regime_labels, cond_metadata = create_conditioning_vectors(
        returns, args.seq_len, args.vol_window, args.val_split
    )
    
    # Save conditioning metadata
    with open(f"{args.results_dir}/conditioning_metadata.json", 'w') as f:
        json.dump(cond_metadata, f, indent=2)
    
    # Add conditioning metadata to args for evaluation functions
    args.vol_scaler_mean = cond_metadata['vol_scaler_mean']
    args.vol_scaler_scale = cond_metadata['vol_scaler_scale']
    args.vol_threshold = cond_metadata['vol_threshold']
    
    # Train model
    model, trainer, ema, history = train_model(X, conditioning_vectors, regime_labels, cond_metadata, args)
    
    # Run evaluations
    print("\nRunning evaluations...")
    
    # Controllability evaluation
    control_metrics = evaluate_controllability(model, trainer, ema, conditioning_vectors, args)
    
    # Regime accuracy evaluation
    regime_metrics = evaluate_regime_accuracy(model, trainer, ema, conditioning_vectors, args)
    
    # Distributional fidelity evaluation
    dist_metrics = evaluate_distributional_fidelity(model, trainer, ema, conditioning_vectors, returns.values, args)
    
    # March 2020 case study
    case_metrics = march2020_case_study(model, trainer, ema, args)
    
    # Ablation study: compare conditioned vs unconditional model
    print("\nRunning ablation study...")
    ablation_metrics = run_ablation_study(model, trainer, ema, args)
    
    # Create README
    create_readme(args, cond_metadata, control_metrics, regime_metrics, dist_metrics, case_metrics, ablation_metrics)
    
    print(f"\nExplicit conditioning DDPM completed successfully!")
    print(f"Results saved in: {args.results_dir}")
    
    return model, trainer, ema

def run_ablation_study(model, trainer, ema, args):
    """Run ablation study comparing conditioned vs unconditional model."""
    print("Running ablation study: conditioned vs unconditional...")
    
    # Use EMA model if available
    if ema is not None:
        ema.apply_shadow()
    
    try:
        # Generate samples with conditioning (conditioned version)
        num_samples = 1000
        device = next(model.parameters()).device
        
        # Use representative conditioning (median volatility, balanced regime distribution)
        representative_conditioning = np.zeros((num_samples, 5))
        # Set balanced regime distribution
        for i in range(num_samples):
            regime_idx = i % 4  # Cycle through all regimes
            representative_conditioning[i, regime_idx] = 1
            # Set median volatility
            representative_conditioning[i, -1] = 0.0  # Median in scaled space
        
        conditioning_tensor = torch.tensor(representative_conditioning, dtype=torch.float32, device=device)
        
        # Generate conditioned samples
        conditioned_samples = trainer.sample(
            conditioning_tensor, 
            num_samples=num_samples, 
            sampler=args.sampler, 
            sample_steps=args.sample_steps,
            cfg_scale=args.cfg_scale
        )
        conditioned_samples = conditioned_samples.squeeze(1).cpu().numpy()
        conditioned_returns = conditioned_samples.flatten()
        
        # Generate unconditional samples (zero out conditioning)
        unconditional_conditioning = np.zeros((num_samples, 5))
        unconditional_tensor = torch.tensor(unconditional_conditioning, dtype=torch.float32, device=device)
        
        unconditional_samples = trainer.sample(
            unconditional_tensor, 
            num_samples=num_samples, 
            sampler=args.sampler, 
            sample_steps=args.sample_steps,
            cfg_scale=args.cfg_scale
        )
        unconditional_samples = unconditional_samples.squeeze(1).cpu().numpy()
        unconditional_returns = unconditional_samples.flatten()
        
        # Create histogram comparison plot
        plt.figure(figsize=(12, 8))
        
        # Plot histograms
        plt.hist(conditioned_returns, bins=50, density=True, alpha=0.7, 
                label='Conditioned Model', color='blue')
        plt.hist(unconditional_returns, bins=50, density=True, alpha=0.7, 
                label='Unconditional Model', color='red')
        
        plt.xlabel('Returns')
        plt.ylabel('Density')
        plt.title('Ablation Study: Zero-Conditioning vs Explicit-Conditioning on Conditioned Model')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{args.results_dir}/figures/ablation_histogram_comparison.pdf", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Compute comparison metrics
        conditioned_stats = {
            'mean': np.mean(conditioned_returns),
            'std': np.std(conditioned_returns),
            'skew': scipy.stats.skew(conditioned_returns),
            'kurtosis': scipy.stats.kurtosis(conditioned_returns)
        }
        
        zero_conditioned_stats = {
            'mean': np.mean(unconditional_returns),
            'std': np.std(unconditional_returns),
            'skew': scipy.stats.skew(unconditional_returns),
            'kurtosis': scipy.stats.kurtosis(unconditional_returns)
        }
        
        # KS test between conditioned and zero-conditioned
        ks_stat, ks_pvalue = scipy.stats.ks_2samp(conditioned_returns, unconditional_returns)
        
        ablation_metrics = {
            'explicit_conditioned_stats': conditioned_stats,
            'zero_conditioned_stats': zero_conditioned_stats,
            'ks_stat': ks_stat,
            'ks_pvalue': ks_pvalue,
            'num_samples': num_samples
        }
        
        # Save ablation metrics
        with open(f"{args.results_dir}/ablation_metrics.json", 'w') as f:
            json.dump(ablation_metrics, f, indent=2)
        
        # Save return data to CSV
        ablation_data = pd.DataFrame({
            'explicit_conditioned_returns': conditioned_returns,
            'zero_conditioned_returns': unconditional_returns
        })
        ablation_data.to_csv(f"{args.results_dir}/ablation_returns_comparison.csv", index=False)
        
        # Create LaTeX table
        with open(f"{args.results_dir}/tables/ablation_metrics.tex", 'w') as f:
            f.write("\\begin{table}[h]\n")
            f.write("\\centering\n")
            f.write("\\begin{tabular}{lcc}\n")
            f.write("\\hline\n")
            f.write("Metric & Explicit-Conditioned & Zero-Conditioned \\\\\n")
            f.write("\\hline\n")
            f.write(f"Mean & {conditioned_stats['mean']:.6f} & {zero_conditioned_stats['mean']:.6f} \\\\\n")
            f.write(f"Std & {conditioned_stats['std']:.6f} & {zero_conditioned_stats['std']:.6f} \\\\\n")
            f.write(f"Skew & {conditioned_stats['skew']:.6f} & {zero_conditioned_stats['skew']:.6f} \\\\\n")
            f.write(f"Kurtosis & {conditioned_stats['kurtosis']:.6f} & {zero_conditioned_stats['kurtosis']:.6f} \\\\\n")
            f.write("\\hline\n")
            f.write(f"KS Statistic & \\multicolumn{{2}}{{c}}{{{ks_stat:.6f}}} \\\\\n")
            f.write(f"KS p-value & \\multicolumn{{2}}{{c}}{{{ks_pvalue:.6f}}} \\\\\n")
            f.write("\\hline\n")
            f.write("\\end{tabular}\n")
            f.write("\\caption{Ablation Study: Zero-Conditioning vs Explicit-Conditioning on Conditioned Model (Kurtosis values are excess kurtosis)}\n")
            f.write("\\label{tab:ablation_metrics}\n")
            f.write("\\end{table}\n")
        
        print(f"Ablation study completed - KS p-value: {ks_pvalue:.6f}")
        return ablation_metrics
        
    finally:
        # Restore original weights after evaluation
        if ema is not None:
            ema.restore()

def create_readme(args, cond_metadata, control_metrics, regime_metrics, dist_metrics, case_metrics, ablation_metrics):
    """Create README file for the run."""
    readme_content = f"""# Explicit Conditioning DDPM Run Summary

## Run Information
- **Run ID**: {args.run_id}
- **Timestamp**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Device**: {args.device}

## Data Parameters
- **Data Span**: S&P 500 daily returns
- **Sequence Length**: {args.seq_len}
- **Volatility Window**: {args.vol_window}
- **Validation Split**: {args.val_split}

## Model Parameters
- **Hidden Dimension**: {args.hidden_dim}
- **Number of Timesteps**: {args.num_timesteps}
- **Beta Schedule**: {args.beta_schedule}
- **Sampler**: {args.sampler}
- **Sample Steps**: {args.sample_steps}

## Training Parameters
- **Epochs**: {args.epochs}
- **Batch Size**: {args.batch_size}
- **Learning Rate**: {args.lr}
- **Patience**: {args.patience}
- **Gradient Clipping**: {args.grad_clip}
- **EMA**: {args.use_ema and not args.no_ema}
- **CFG Dropout (p)**: {args.cfg_p}
- **CFG Scale**: {args.cfg_scale}

## Conditioning Parameters
- **Volatility Threshold**: {cond_metadata['vol_threshold']:.4f}
- **Regime Order**: {cond_metadata['regime_order']}

## Key Metrics

### Controllability
- **MAE**: {control_metrics['mae']:.4f}
- **R²**: {control_metrics['r2']:.4f}

### Regime Accuracy
- **Overall Accuracy**: {regime_metrics['overall_accuracy']:.2%}

### Distributional Fidelity
- **KS Statistic**: {dist_metrics['ks_stat']:.6f}
- **KS p-value**: {dist_metrics['ks_pvalue']:.6f}
- **MMD Statistic**: {dist_metrics['mmd_stat']:.6f}

### March 2020 Case Study
- **Regime**: {case_metrics['regime']}
- **95% VaR**: ${case_metrics['var_95']:.2f}
- **99% VaR**: ${case_metrics['var_99']:.2f}

### Ablation Study
- **KS p-value**: {ablation_metrics['ks_pvalue']:.6f}
- **Zero-Conditioning vs Explicit-Conditioning**: Statistical comparison on conditioned model

## Generated Files

### Figures
- `figures/controllability_scatter.pdf` - Controllability scatter plot
- `figures/controllability_calibration.pdf` - Reliability/calibration curve
- `figures/controllability_residuals.pdf` - Residuals analysis
- `figures/regime_grid.pdf` - Regime grid visualization
- `figures/regime_confusion_matrix.pdf` - Confusion matrix heatmap
- `figures/tail_distribution_zoom.pdf` - Distribution comparison with tail zoom
- `figures/distribution_qq_plots.pdf` - QQ plots for tail analysis
- `figures/march2020_case.pdf` - March 2020 case study
- `figures/ablation_histogram_comparison.pdf` - Ablation study comparison

### Tables
- `tables/control_metrics.tex` - Controllability metrics
- `tables/regime_accuracy.tex` - Regime accuracy metrics
- `tables/dist_metrics.tex` - Distributional fidelity metrics
- `tables/case_var_es.tex` - Case study risk metrics
- `tables/ablation_metrics.tex` - Ablation study metrics

### Data
- `checkpoints/` - Model checkpoints
- `training_history.csv` - Training progress
- `metadata.json` - Run metadata
- `conditioning_metadata.json` - Conditioning parameters
- `controllability_calibration.csv` - Controllability calibration data
- `controllability_residuals.csv` - Controllability residuals data
- `regime_confusion_matrix.csv` - Regime confusion matrix data
- `distribution_right_tail_qq.csv` - Right tail QQ plot data
- `distribution_left_tail_qq.csv` - Left tail QQ plot data
- `ablation_returns_comparison.csv` - Ablation study return data

## Reproducibility
- **Seed**: {args.seed}
- **Device**: {args.device}
- **PyTorch Version**: {torch.__version__}

## Notes
This run demonstrates the explicit conditioning DDPM with regime classification and volatility control.
The model successfully generates financial time series conditioned on market regimes and target volatility levels.
"""
    
    with open(f"{args.results_dir}/README_RUN.md", 'w') as f:
        f.write(readme_content)

if __name__ == "__main__":
    main()
