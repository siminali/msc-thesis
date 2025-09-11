#!/usr/bin/env python3
"""
Pre-COVID Training Runner - Simplified Version (No Utils Dependencies)

Standalone training for financial models (2010-2019) without external utils dependencies.
This version includes only the core model classes needed for training.

Author: Simin Ali
Thesis: Diffusion Models in Generative AI for Financial Data Synthesis and Risk Management
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import json
import os
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from tqdm import tqdm
import argparse
import logging
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pickle
import sys

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global constants
SEQ_LEN = 60
EMBEDDING_DIM = 768  # DistilBERT native dimension
DEFAULT_PCA_COMPONENTS = 32  # Default PCA dimensions for LLM embeddings

# =============================================================================
# ZERO CONDITIONING MODEL (Simplified from diffusion_simple.py)
# =============================================================================

class TimeEmbedding(nn.Module):
    """Enhanced time embedding module with sinusoidal positional encoding."""
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
        # Sinusoidal positional encoding
        self.register_buffer('freqs', torch.exp(-torch.arange(0, dim, 2) * (np.log(10000) / dim)))
        
        # Projection layers
        self.proj = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim)
        )
    
    def forward(self, t):
        # t: [batch_size, 1] -> [batch_size, dim]
        # Apply sinusoidal encoding
        t_expanded = t.unsqueeze(-1) * self.freqs.unsqueeze(0)  # [batch_size, dim//2]
        
        # Create full embedding with sin and cos
        emb = torch.cat([torch.sin(t_expanded), torch.cos(t_expanded)], dim=-1)
        
        # Ensure correct dimension
        if emb.shape[-1] != self.dim:
            emb = emb[..., :self.dim]
        
        # Project through MLP and ensure 2D output
        output = self.proj(emb)
        if output.dim() == 3:
            output = output.squeeze(1)  # Remove extra dimension
        
        return output

class DenoiseMLP(nn.Module):
    """Enhanced MLP denoiser with better architecture, normalization, and residual connections."""
    
    def __init__(self, seq_len, hidden_dim=256, time_dim=128, num_layers=6):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.time_dim = time_dim
        self.num_layers = num_layers
        
        # Enhanced time embedding with sinusoidal encoding
        self.time_embedding = TimeEmbedding(time_dim)
        
        # Input projection
        self.input_proj = nn.Linear(seq_len, hidden_dim)
        
        # Main network with residual connections and layer normalization
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = nn.Sequential(
                nn.LayerNorm(hidden_dim + time_dim),
                nn.Linear(hidden_dim + time_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Dropout(0.1)
            )
            self.layers.append(layer)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, seq_len)
        )
        
        # Initialize weights properly
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights for better training."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, x, t):
        """Forward pass."""
        # Time embedding
        t_emb = self.time_embedding(t)  # [batch_size, time_dim]
        
        # Input projection
        h = self.input_proj(x)  # [batch_size, hidden_dim]
        
        # Process through layers with residual connections
        for i, layer in enumerate(self.layers):
            # Concatenate h and time embedding
            combined = torch.cat([h, t_emb], dim=-1)  # [batch_size, hidden_dim + time_dim]
            
            # Apply layer
            residual = h
            h = layer(combined)
            
            # Residual connection (every other layer for stability)
            if i % 2 == 1:
                h = h + residual
        
        # Output projection
        return self.output_proj(h)

# =============================================================================
# EXPLICIT CONDITIONING MODEL (From explicit_cond_ddpm.py)
# =============================================================================

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
        """Forward pass."""
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

# =============================================================================
# LLM CONDITIONING MODEL (Simplified from llm_conditioned_diffusion.py)
# =============================================================================

class ConditionedDiffusionModel(nn.Module):
    """Conditional Diffusion Model that uses LLM embeddings as conditioning."""
    
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
            nn.Linear(1 + hidden_dim + hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x, t, conditioning):
        """Forward pass with conditioning."""
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

# =============================================================================
# TRAINER CLASSES
# =============================================================================

class SimpleTrainer:
    """Simple trainer for zero conditioning model."""
    
    def __init__(self, model, num_timesteps=1000, device='cpu'):
        self.model = model
        self.device = device
        self.num_timesteps = num_timesteps
        
        # Linear noise schedule
        self.betas = torch.linspace(1e-4, 0.02, num_timesteps, device=device)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
    
    def train_step(self, x, optimizer):
        batch_size = x.shape[0]
        
        # Sample random timesteps
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=self.device)
        
        # Add noise
        noise = torch.randn_like(x)
        sqrt_alphas_cumprod_t = self.alphas_cumprod[t].view(-1, 1)
        sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1 - self.alphas_cumprod[t]).view(-1, 1)
        
        x_noisy = sqrt_alphas_cumprod_t * x + sqrt_one_minus_alphas_cumprod_t * noise
        
        # Predict noise
        t_normalized = t.float() / self.num_timesteps
        predicted_noise = self.model(x_noisy, t_normalized.unsqueeze(-1))
        
        # Compute loss
        loss = F.mse_loss(predicted_noise, noise)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        optimizer.step()
        
        return loss.item()

class ExplicitConditioningTrainer:
    """Trainer for explicit conditioning DDPM."""
    
    def __init__(self, model, num_timesteps=1000, beta_schedule="cosine", device="cpu", grad_clip=1.0, cfg_p=0.1):
        self.model = model
        self.num_timesteps = num_timesteps
        self.device = device
        self.grad_clip = grad_clip
        self.cfg_p = cfg_p  # Probability of conditioning dropout during training
        
        # Move model to device
        self.model = self.model.to(device)
        
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
    
    def train_step(self, x, conditioning, optimizer):
        """Single training step with classifier-free guidance conditioning dropout."""
        batch_size = x.shape[0]
        
        # Sample random timesteps
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=self.device)
        
        # Add noise
        x_noisy, noise = self.add_noise(x, t)
        
        # Normalize timesteps to [0, 1]
        t_normalized = t.float() / self.num_timesteps
        
        # Apply conditioning dropout for classifier-free guidance
        dropout_mask = torch.rand(batch_size, device=self.device) > self.cfg_p
        conditioning_dropped = conditioning.clone()
        conditioning_dropped[~dropout_mask] = 0.0  # Zero conditioning for dropout samples
        
        # Predict noise
        predicted_noise = self.model(x_noisy, t_normalized.unsqueeze(-1), conditioning_dropped)
        loss = F.mse_loss(predicted_noise, noise)
        
        # Standard backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Apply gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        
        optimizer.step()
        
        return loss.item()

class ConditionedDiffusionTrainer:
    """Trainer for the LLM-conditioned diffusion model."""
    
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
    
    def add_noise(self, x_start, t):
        """Add noise to data according to diffusion schedule."""
        noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        
        x_noisy = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
        return x_noisy, noise
    
    def train_step(self, x, conditioning, optimizer):
        """Single training step."""
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
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        optimizer.step()
        
        return loss.item()

# =============================================================================
# CORE TRAINING FUNCTIONS
# =============================================================================

def set_deterministic_mode(seed=42):
    """Set deterministic mode for reproducible training."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Set environment variables for additional determinism
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    logger.info(f"Set deterministic mode with seed: {seed}")

def load_and_prepare_data():
    """Load financial data and prepare pre-COVID training split."""
    logger.info("Loading S&P 500 data...")
    
    # Try multiple data paths
    data_paths = [
        '/Users/siminali/Desktop/Thesis Coding/data/sp500_data.csv',
        'data/sp500_data.csv',
        '../data/sp500_data.csv'
    ]
    
    data = None
    for path in data_paths:
        if os.path.exists(path):
            try:
                data = pd.read_csv(path, index_col=0, parse_dates=True)
                logger.info(f"Data loaded from: {path}")
                break
            except Exception as e:
                logger.warning(f"Failed to load from {path}: {e}")
                continue
    
    if data is None:
        raise FileNotFoundError(f"Could not find sp500_data.csv in any of the paths: {data_paths}")
    
    # Ensure index is datetime and sort chronologically
    data.index = pd.to_datetime(data.index)
    data = data.sort_index()
    
    # Calculate log returns
    data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1)).dropna()
    returns = data['Log_Returns'].dropna()
    
    # Define pre-COVID period: 2010-01-01 to 2019-12-31
    train_start = '2010-01-01'
    train_end = '2019-12-31'
    val_start = '2019-07-01'  # Validation on last 6 months of training
    val_end = '2019-12-31'
    
    # Filter data for pre-COVID period
    train_data = returns[train_start:train_end]
    val_data = returns[val_start:val_end]
    
    if len(train_data) == 0:
        raise ValueError(f"No training data found for period {train_start} to {train_end}")
    
    if len(val_data) == 0:
        raise ValueError(f"No validation data found for period {val_start} to {val_end}")
    
    logger.info(f"Training data: {len(train_data)} observations ({train_data.index[0]} to {train_data.index[-1]})")
    logger.info(f"Validation data: {len(val_data)} observations ({val_data.index[0]} to {val_data.index[-1]})")
    logger.info(f"Training stats - Mean: {train_data.mean():.6f}, Std: {train_data.std():.6f}")
    
    return train_data, val_data

def create_sequences(returns, seq_len=SEQ_LEN):
    """Create sequences for training."""
    logger.info(f"Creating sequences of length {seq_len}...")
    
    sequences = []
    indices = []
    
    for i in range(len(returns) - seq_len + 1):
        seq = returns.iloc[i:i+seq_len].values
        sequences.append(seq)
        indices.append(returns.index[i:i+seq_len])
    
    X = np.array(sequences)
    logger.info(f"Created {len(X)} sequences")
    
    return X, indices

def create_zero_conditioning(X):
    """Create zero conditioning (no conditioning)."""
    logger.info("Creating zero conditioning (no conditioning)")
    
    # Return None to indicate no conditioning
    conditioning_spec = {
        'type': 'zero',
        'description': 'No conditioning - basic unconditional DDPM',
        'conditioning_dim': 0
    }
    
    return None, conditioning_spec

def create_explicit_conditioning(returns_sequences, returns_index, vol_window=20, trend_window=60):
    """Create explicit conditioning with regime classification and financial features."""
    logger.info(f"Creating explicit conditioning with vol_window={vol_window}, trend_window={trend_window}")
    
    conditioning_vectors = []
    
    # Process each sequence
    for i, seq_returns in enumerate(returns_sequences):
        # Get the corresponding returns for causality
        seq_index = returns_index[i]
        
        # 1. Calculate causal 20-day volatility (using only past data)
        if i >= vol_window:
            # Use past vol_window days before the sequence
            past_returns = returns_sequences[max(0, i-vol_window):i].flatten()
            z_vol = np.std(past_returns) if len(past_returns) > 1 else 0.0
        else:
            # For early sequences, use available past data
            past_returns = returns_sequences[:i].flatten() if i > 0 else [0.0]
            z_vol = np.std(past_returns) if len(past_returns) > 1 else 0.0
        
        # 2. Calculate trend from 60-day sum (causal)
        if i >= trend_window:
            # Use past trend_window days before the sequence
            past_returns_trend = returns_sequences[max(0, i-trend_window):i].flatten()
            trend = np.sum(past_returns_trend)
        else:
            # For early sequences, use available past data
            past_returns_trend = returns_sequences[:i].flatten() if i > 0 else [0.0]
            trend = np.sum(past_returns_trend)
        
        # 3. Determine regime based on current sequence (this is for the target, not causal)
        # But we use past information to classify the regime state
        seq_cumulative = np.sum(seq_returns)
        
        # Classify Up/Down based on cumulative return
        is_up = seq_cumulative > 0
        
        # Store raw features for later normalization
        conditioning_vectors.append({
            'z_vol': z_vol,
            'trend': trend,
            'is_up': is_up,
            'seq_vol': np.std(seq_returns)  # For regime classification
        })
    
    # Convert to structured format for normalization
    z_vols = np.array([c['z_vol'] for c in conditioning_vectors])
    trends = np.array([c['trend'] for c in conditioning_vectors])
    seq_vols = np.array([c['seq_vol'] for c in conditioning_vectors])
    is_ups = np.array([c['is_up'] for c in conditioning_vectors])
    
    # Fit scalers on training data only (first 80% for safety)
    train_split = int(len(conditioning_vectors) * 0.8)
    
    # Normalize z_vol (volatility)
    vol_scaler = StandardScaler()
    z_vols_train = z_vols[:train_split].reshape(-1, 1)
    vol_scaler.fit(z_vols_train)
    z_vols_norm = vol_scaler.transform(z_vols.reshape(-1, 1)).flatten()
    
    # Normalize trend
    trend_scaler = StandardScaler()
    trends_train = trends[:train_split].reshape(-1, 1)
    trend_scaler.fit(trends_train)
    trends_norm = trend_scaler.transform(trends.reshape(-1, 1)).flatten()
    
    # Determine volatility regime using median split on training data
    vol_threshold = np.median(seq_vols[:train_split])
    
    # Create final conditioning vectors: [Up-Low, Up-High, Down-Low, Down-High, z_vol, trend]
    final_conditioning = []
    
    for i in range(len(conditioning_vectors)):
        is_up = is_ups[i]
        is_high_vol = seq_vols[i] > vol_threshold
        
        # Create one-hot regime encoding
        regime_onehot = np.zeros(4)
        if is_up and not is_high_vol:      # Up-Low
            regime_onehot[0] = 1
        elif is_up and is_high_vol:        # Up-High
            regime_onehot[1] = 1
        elif not is_up and not is_high_vol: # Down-Low
            regime_onehot[2] = 1
        else:                              # Down-High
            regime_onehot[3] = 1
        
        # Combine regime + normalized volatility + normalized trend
        conditioning_vector = np.concatenate([
            regime_onehot,           # 4 dimensions: regime
            [z_vols_norm[i]],       # 1 dimension: normalized volatility
            [trends_norm[i]]        # 1 dimension: normalized trend
        ])
        
        final_conditioning.append(conditioning_vector)
    
    final_conditioning = np.array(final_conditioning)
    
    # Create conditioning specification
    conditioning_spec = {
        'type': 'explicit',
        'description': 'Explicit conditioning with regime classification + volatility + trend',
        'conditioning_dim': 6,  # 4 regime + 1 volatility + 1 trend
        'features': {
            'regime_onehot': {
                'description': '4 one-hot features for Up-Low, Up-High, Down-Low, Down-High',
                'indices': [0, 1, 2, 3]
            },
            'z_vol': {
                'description': 'Normalized causal 20-day volatility',
                'index': 4,
                'scaler_mean': float(vol_scaler.mean_[0]),
                'scaler_scale': float(vol_scaler.scale_[0])
            },
            'trend': {
                'description': 'Normalized causal 60-day cumulative return',
                'index': 5,
                'scaler_mean': float(trend_scaler.mean_[0]),
                'scaler_scale': float(trend_scaler.scale_[0])
            }
        },
        'vol_threshold': float(vol_threshold),
        'vol_window': vol_window,
        'trend_window': trend_window
    }
    
    logger.info(f"Created explicit conditioning: {final_conditioning.shape}")
    logger.info(f"Regime distribution: {np.sum(final_conditioning[:, :4], axis=0)}")
    
    return final_conditioning, conditioning_spec

def load_real_embeddings_for_training(start_date, end_date):
    """Load real embeddings from cache files for training."""
    cache_dir = Path("cache/news_embeddings")
    if not cache_dir.exists():
        logger.warning(f"Embedding cache directory not found: {cache_dir}")
        return None, None
    
    # Discover available embedding files
    embedding_files = []
    for pkl_file in cache_dir.glob("*.pkl"):
        filename = pkl_file.name
        if filename.startswith("embeddings_") and filename.endswith(".pkl"):
            date_part = filename.replace("embeddings_", "").replace(".pkl", "")
            parts = date_part.split("_")
            if len(parts) == 2:
                try:
                    start_str, end_str = parts
                    file_start = pd.Timestamp(start_str)
                    file_end = pd.Timestamp(end_str)
                    
                    embedding_files.append({
                        'path': pkl_file,
                        'start_date': file_start,
                        'end_date': file_end,
                        'filename': filename
                    })
                except Exception as e:
                    logger.warning(f"Could not parse date range from {filename}: {e}")
    
    if not embedding_files:
        logger.warning("No embedding files found in cache")
        return None, None
    
    # Find the best file that covers our date range
    best_file = None
    for file_info in embedding_files:
        if file_info['start_date'] <= start_date and file_info['end_date'] >= end_date:
            if best_file is None or (file_info['end_date'] - file_info['start_date']) < (best_file['end_date'] - best_file['start_date']):
                best_file = file_info
    
    if not best_file:
        # Find file with best partial coverage
        best_overlap = 0
        for file_info in embedding_files:
            overlap_start = max(file_info['start_date'], start_date)
            overlap_end = min(file_info['end_date'], end_date)
            if overlap_start <= overlap_end:
                overlap_days = (overlap_end - overlap_start).days
                if overlap_days > best_overlap:
                    best_overlap = overlap_days
                    best_file = file_info
    
    if not best_file:
        logger.error(f"No embedding files cover training date range {start_date.date()} to {end_date.date()}")
        return None, None
    
    # Load embeddings
    try:
        with open(best_file['path'], 'rb') as f:
            embeddings_dict = pickle.load(f)
        
        logger.info(f"Loaded embeddings from {best_file['filename']}")
        logger.info(f"Contains {len(embeddings_dict)} dates")
        
        # Determine actual embedding dimension
        sample_embedding = next(iter(embeddings_dict.values()))
        actual_embedding_dim = len(sample_embedding)
        
        return embeddings_dict, actual_embedding_dim
        
    except Exception as e:
        logger.error(f"Failed to load embeddings from {best_file['path']}: {e}")
        return None, None

def create_llm_conditioning(returns_index, seq_len=SEQ_LEN, pca_components=DEFAULT_PCA_COMPONENTS, 
                           device='cpu', fallback_on_error=True):
    """Create LLM conditioning with real cached embeddings."""
    logger.info(f"Creating LLM conditioning with real embeddings, PCA components={pca_components}")
    
    try:
        # Get all unique dates
        all_dates = []
        for seq_idx in returns_index:
            all_dates.extend(seq_idx.tolist())
        
        unique_dates = sorted(list(set(all_dates)))
        start_date = unique_dates[0]
        end_date = unique_dates[-1]
        
        logger.info(f"Loading real embeddings for {start_date} to {end_date}")
        
        # Load real embeddings
        embeddings_dict, actual_embedding_dim = load_real_embeddings_for_training(start_date, end_date)
        
        if embeddings_dict is None:
            if fallback_on_error:
                logger.warning("Failed to load real embeddings, falling back to mock embeddings")
                # Create mock embeddings as fallback
                num_dates = len(unique_dates)
                np.random.seed(42)  # For reproducible mock embeddings
                mock_embeddings = np.random.normal(0, 1, (num_dates, EMBEDDING_DIM))
                embeddings_dict = {date: mock_embeddings[i] for i, date in enumerate(unique_dates)}
                actual_embedding_dim = EMBEDDING_DIM
            else:
                raise ValueError("Failed to load real embeddings and fallback disabled")
        
        # Create conditioning vectors for each sequence
        conditioning_vectors = []
        missing_dates = []
        
        for seq_idx in returns_index:
            # Find the embeddings corresponding to this sequence
            seq_embeddings = []
            for date in seq_idx:
                if date in embeddings_dict:
                    seq_embeddings.append(embeddings_dict[date])
                else:
                    missing_dates.append(date)
                    # Use zero vector for missing dates (will be handled causally)
                    seq_embeddings.append(np.zeros(actual_embedding_dim))
            
            # Aggregate embeddings (mean pooling)
            if len(seq_embeddings) > 0:
                seq_conditioning = np.mean(seq_embeddings, axis=0)
            else:
                seq_conditioning = np.zeros(actual_embedding_dim)
            conditioning_vectors.append(seq_conditioning)
        
        conditioning_vectors = np.array(conditioning_vectors)
        
        if missing_dates:
            logger.warning(f"Missing embeddings for {len(set(missing_dates))} unique dates (using zero vectors)")
        
        # Fit PCA only on training data (≤ 2019-12-31)
        train_cutoff = pd.Timestamp('2019-12-31')
        train_mask = np.array([idx[-1] <= train_cutoff for idx in returns_index])
        
        train_embeddings = conditioning_vectors[train_mask]
        
        if len(train_embeddings) == 0:
            raise ValueError("No training embeddings found for PCA fitting")
        
        # Fit PCA on training data
        pca = PCA(n_components=pca_components)
        pca.fit(train_embeddings)
        
        # Transform all conditioning vectors
        conditioning_vectors_pca = pca.transform(conditioning_vectors)
        
        explained_variance_ratio = np.sum(pca.explained_variance_ratio_)
        
        # Create conditioning specification
        conditioning_spec = {
            'type': 'llm',
            'description': f'Real LLM embeddings with PCA reduction to {pca_components} dimensions',
            'conditioning_dim': pca_components,
            'pca_components': pca_components,
            'explained_variance_ratio': float(explained_variance_ratio),
            'original_embedding_dim': actual_embedding_dim,
            'train_cutoff': train_cutoff.isoformat(),
            'date_range': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            }
        }
        
        logger.info(f"Created real LLM conditioning: {conditioning_vectors_pca.shape}")
        logger.info(f"PCA explained variance ratio: {explained_variance_ratio:.4f}")
        
        return conditioning_vectors_pca, conditioning_spec, pca
        
    except Exception as e:
        error_msg = f"LLM conditioning failed: {e}"
        logger.error(error_msg)
        
        if fallback_on_error:
            logger.warning("Falling back to zero conditioning due to LLM error")
            return create_zero_conditioning(None)
        else:
            raise RuntimeError(error_msg)

def save_checkpoint(model, trainer, epoch, train_loss, val_loss, is_best, checkpoint_dir, 
                   conditioning_spec, metadata):
    """Save model checkpoint with comprehensive metadata."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Prepare checkpoint data
    checkpoint_data = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'timestamp': datetime.now().isoformat()
    }
    
    # Save trainer state if available
    if hasattr(trainer, 'betas'):
        checkpoint_data.update({
            'betas': trainer.betas.cpu() if torch.is_tensor(trainer.betas) else trainer.betas,
            'alphas': trainer.alphas.cpu() if torch.is_tensor(trainer.alphas) else trainer.alphas,
            'alphas_cumprod': trainer.alphas_cumprod.cpu() if torch.is_tensor(trainer.alphas_cumprod) else trainer.alphas_cumprod
        })
    
    # Save checkpoints
    if is_best:
        torch.save(checkpoint_data, os.path.join(checkpoint_dir, 'best.pt'))
        logger.info(f"Saved best checkpoint at epoch {epoch}")
    
    torch.save(checkpoint_data, os.path.join(checkpoint_dir, 'last.pt'))
    
    # Save metadata
    meta_data = {
        'model_info': {
            'type': str(conditioning_spec['type']),
            'conditioning_dim': int(conditioning_spec['conditioning_dim']),
            'sequence_length': int(SEQ_LEN),
            'parameter_count': int(sum(p.numel() for p in model.parameters())),
            'trainable_parameters': int(sum(p.numel() for p in model.parameters() if p.requires_grad))
        },
        'training_info': {
            'epoch': int(epoch),
            'train_loss': float(train_loss),
            'val_loss': float(val_loss),
            'is_best': bool(is_best)
        },
        'system_info': metadata['system_info'],
        'data_info': metadata['data_info']
    }
    
    with open(os.path.join(checkpoint_dir, 'meta.json'), 'w') as f:
        json.dump(meta_data, f, indent=2)
    
    # Save conditioning specification
    with open(os.path.join(checkpoint_dir, 'conditioning_spec.json'), 'w') as f:
        json.dump(conditioning_spec, f, indent=2)

def train_zero_model(X_train, X_val, args, metadata):
    """Train zero conditioning model (basic DDPM)."""
    logger.info("Training zero conditioning model...")
    
    # Model setup
    device = torch.device(args.device)
    model = DenoiseMLP(SEQ_LEN, hidden_dim=args.hidden_dim).to(device)
    trainer = SimpleTrainer(model, args.num_timesteps, device)
    
    # Data preparation
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    
    train_dataset = TensorDataset(X_train_tensor)
    val_dataset = TensorDataset(X_val_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    conditioning_spec = {
        'type': 'zero',
        'description': 'No conditioning - basic unconditional DDPM',
        'conditioning_dim': 0
    }
    
    checkpoint_dir = os.path.join(args.checkpoint_dir, 'zero', '20100101-20191231')
    
    for epoch in range(args.epochs):
        # Training
        model.train()
        epoch_train_losses = []
        
        for batch in train_loader:
            batch_x = batch[0].to(device)
            loss = trainer.train_step(batch_x, optimizer)
            epoch_train_losses.append(loss)
        
        # Validation
        model.eval()
        epoch_val_losses = []
        
        with torch.no_grad():
            for batch in val_loader:
                batch_x = batch[0].to(device)
                batch_size = batch_x.shape[0]
                
                # Sample random timesteps for validation
                t = torch.randint(0, args.num_timesteps, (batch_size,), device=device)
                
                # Add noise
                noise = torch.randn_like(batch_x)
                sqrt_alphas_cumprod_t = trainer.alphas_cumprod[t].view(-1, 1)
                sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1 - trainer.alphas_cumprod[t]).view(-1, 1)
                
                x_noisy = sqrt_alphas_cumprod_t * batch_x + sqrt_one_minus_alphas_cumprod_t * noise
                
                # Predict noise
                t_normalized = t.float() / args.num_timesteps
                predicted_noise = model(x_noisy, t_normalized.unsqueeze(-1))
                
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
        
        # Check for best model
        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss
        
        # Save checkpoint
        save_checkpoint(model, trainer, epoch, avg_train_loss, avg_val_loss, 
                       is_best, checkpoint_dir, conditioning_spec, metadata)
        
        # Logging
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            logger.info(f"Zero Model - Epoch {epoch}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}")
    
    logger.info(f"Zero model training completed. Best val loss: {best_val_loss:.6f}")
    return model, trainer

def train_explicit_model(X_train, X_val, conditioning_train, conditioning_val, 
                        conditioning_spec, args, metadata):
    """Train explicit conditioning model."""
    logger.info("Training explicit conditioning model...")
    
    device = torch.device(args.device)
    
    # Model setup
    model = ExplicitConditioningDDPM(
        sequence_length=SEQ_LEN,
        conditioning_dim=conditioning_spec['conditioning_dim'],
        hidden_dim=args.hidden_dim
    ).to(device)
    
    trainer = ExplicitConditioningTrainer(
        model,
        num_timesteps=args.num_timesteps,
        beta_schedule='cosine',
        device=device,
        grad_clip=1.0,
        cfg_p=0.1
    )
    
    # Data preparation
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)  # [N, 1, T]
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1)
    
    cond_train_tensor = torch.tensor(conditioning_train, dtype=torch.float32)
    cond_val_tensor = torch.tensor(conditioning_val, dtype=torch.float32)
    
    train_dataset = TensorDataset(X_train_tensor, cond_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, cond_val_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    checkpoint_dir = os.path.join(args.checkpoint_dir, 'explicit', '20100101-20191231')
    
    for epoch in range(args.epochs):
        # Training
        model.train()
        epoch_train_losses = []
        
        for batch_x, batch_conditioning in train_loader:
            batch_x = batch_x.to(device)
            batch_conditioning = batch_conditioning.to(device)
            
            loss = trainer.train_step(batch_x, batch_conditioning, optimizer)
            epoch_train_losses.append(loss)
        
        # Validation
        model.eval()
        epoch_val_losses = []
        
        with torch.no_grad():
            for batch_x, batch_conditioning in val_loader:
                batch_x = batch_x.to(device)
                batch_conditioning = batch_conditioning.to(device)
                
                # Sample random timesteps for validation
                batch_size = batch_x.shape[0]
                t = torch.randint(0, args.num_timesteps, (batch_size,), device=device)
                
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
        
        # Check for best model
        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss
        
        # Save checkpoint
        save_checkpoint(model, trainer, epoch, avg_train_loss, avg_val_loss,
                       is_best, checkpoint_dir, conditioning_spec, metadata)
        
        # Logging
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            logger.info(f"Explicit Model - Epoch {epoch}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}")
    
    logger.info(f"Explicit model training completed. Best val loss: {best_val_loss:.6f}")
    return model, trainer

def train_llm_model(X_train, X_val, conditioning_train, conditioning_val, 
                   conditioning_spec, args, metadata, pca=None):
    """Train LLM conditioning model."""
    logger.info("Training LLM conditioning model...")
    
    device = torch.device(args.device)
    
    # Model setup
    model = ConditionedDiffusionModel(
        sequence_length=SEQ_LEN,
        conditioning_dim=conditioning_spec['conditioning_dim'],
        hidden_dim=args.hidden_dim
    ).to(device)
    
    trainer = ConditionedDiffusionTrainer(
        model,
        num_timesteps=args.num_timesteps,
        beta_start=1e-4,
        beta_end=0.02,
        device=device
    )
    
    # Data preparation
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)  # [N, T, 1]
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).unsqueeze(-1)
    
    cond_train_tensor = torch.tensor(conditioning_train, dtype=torch.float32)
    cond_val_tensor = torch.tensor(conditioning_val, dtype=torch.float32)
    
    train_dataset = TensorDataset(X_train_tensor, cond_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, cond_val_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    checkpoint_dir = os.path.join(args.checkpoint_dir, 'llm', '20100101-20191231')
    
    # Save PCA model if provided
    if pca is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)
        with open(os.path.join(checkpoint_dir, 'pca_model.pkl'), 'wb') as f:
            pickle.dump(pca, f)
    
    for epoch in range(args.epochs):
        # Training
        model.train()
        epoch_train_losses = []
        
        for batch_x, batch_conditioning in train_loader:
            batch_x = batch_x.to(device)
            batch_conditioning = batch_conditioning.to(device)
            
            loss = trainer.train_step(batch_x, batch_conditioning, optimizer)
            epoch_train_losses.append(loss)
        
        # Validation
        model.eval()
        epoch_val_losses = []
        
        with torch.no_grad():
            for batch_x, batch_conditioning in val_loader:
                batch_x = batch_x.to(device)
                batch_conditioning = batch_conditioning.to(device)
                
                # Sample random timesteps for validation
                batch_size = batch_x.shape[0]
                t = torch.randint(0, args.num_timesteps, (batch_size,), device=device)
                
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
        
        # Check for best model
        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss
        
        # Save checkpoint
        save_checkpoint(model, trainer, epoch, avg_train_loss, avg_val_loss,
                       is_best, checkpoint_dir, conditioning_spec, metadata)
        
        # Logging
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            logger.info(f"LLM Model - Epoch {epoch}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}")
    
    logger.info(f"LLM model training completed. Best val loss: {best_val_loss:.6f}")
    return model, trainer

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Pre-COVID Training Runner for Financial Models (Simplified)'
    )
    
    # Model selection
    parser.add_argument('--models', nargs='+', 
                       choices=['zero', 'explicit', 'llm', 'all'],
                       default=['all'],
                       help='Models to train (default: all)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs (default: 50)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate (default: 1e-3)')
    parser.add_argument('--num-timesteps', type=int, default=1000,
                       help='Number of diffusion timesteps (default: 1000)')
    parser.add_argument('--hidden-dim', type=int, default=128,
                       help='Hidden dimension for models (default: 128)')
    
    # Conditioning parameters
    parser.add_argument('--pca-components', type=int, default=DEFAULT_PCA_COMPONENTS,
                       help=f'PCA components for LLM embeddings (default: {DEFAULT_PCA_COMPONENTS})')
    parser.add_argument('--vol-window', type=int, default=20,
                       help='Volatility window for explicit conditioning (default: 20)')
    parser.add_argument('--trend-window', type=int, default=60,
                       help='Trend window for explicit conditioning (default: 60)')
    
    # System parameters
    parser.add_argument('--device', choices=['cpu', 'cuda', 'auto'], default='auto',
                       help='Device to use (default: auto)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints/precovid',
                       help='Checkpoint directory (default: checkpoints/precovid)')
    
    # Error handling
    parser.add_argument('--skip-on-error', action='store_true',
                       help='Skip models that fail instead of stopping')
    parser.add_argument('--llm-fallback', action='store_true', default=True,
                       help='Fallback to zero conditioning if LLM fails')
    
    args = parser.parse_args()
    
    # Handle "all" model selection
    if 'all' in args.models:
        args.models = ['zero', 'explicit', 'llm']
    
    # Set device
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    return args

def main():
    """Main training function."""
    print("=" * 80)
    print("Pre-COVID Training Runner - Simplified Version (No Utils Dependencies)")
    print("=" * 80)
    
    # Parse arguments
    args = parse_arguments()
    
    # Set deterministic mode
    set_deterministic_mode(args.seed)
    
    # Log system information
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    logger.info(f"Models to train: {args.models}")
    logger.info(f"Training parameters: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}")
    
    try:
        # Load and prepare data
        train_data, val_data = load_and_prepare_data()
        
        # Create sequences
        X_train, train_indices = create_sequences(train_data)
        X_val, val_indices = create_sequences(val_data)
        
        # Prepare metadata
        metadata = {
            'system_info': {
                'device': str(device),
                'torch_version': str(torch.__version__),
                'cuda_available': bool(torch.cuda.is_available()),
                'python_version': str(sys.version),
                'timestamp': datetime.now().isoformat(),
                'seed': int(args.seed)
            },
            'data_info': {
                'train_period': f"{train_data.index[0]} to {train_data.index[-1]}",
                'val_period': f"{val_data.index[0]} to {val_data.index[-1]}",
                'train_sequences': int(len(X_train)),
                'val_sequences': int(len(X_val)),
                'sequence_length': int(SEQ_LEN),
                'train_stats': {
                    'mean': float(train_data.mean()),
                    'std': float(train_data.std()),
                    'min': float(train_data.min()),
                    'max': float(train_data.max())
                }
            }
        }
        
        # Create checkpoint directory
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        
        # Train each model
        trained_models = {}
        
        for model_type in args.models:
            logger.info(f"\n{'='*60}")
            logger.info(f"Training {model_type.upper()} model")
            logger.info(f"{'='*60}")
            
            try:
                if model_type == 'zero':
                    # Zero conditioning
                    model, trainer = train_zero_model(X_train, X_val, args, metadata)
                    trained_models['zero'] = (model, trainer)
                    
                elif model_type == 'explicit':
                    # Create explicit conditioning
                    conditioning_train, conditioning_spec = create_explicit_conditioning(
                        X_train, train_indices, args.vol_window, args.trend_window
                    )
                    conditioning_val, _ = create_explicit_conditioning(
                        X_val, val_indices, args.vol_window, args.trend_window
                    )
                    
                    # Train model
                    model, trainer = train_explicit_model(
                        X_train, X_val, conditioning_train, conditioning_val,
                        conditioning_spec, args, metadata
                    )
                    trained_models['explicit'] = (model, trainer)
                    
                elif model_type == 'llm':
                    # Create LLM conditioning (simplified with mock embeddings)
                    try:
                        all_indices = train_indices + val_indices
                        conditioning_all, conditioning_spec, pca = create_llm_conditioning(
                            all_indices, SEQ_LEN, args.pca_components, args.device, args.llm_fallback
                        )
                        
                        # Split conditioning
                        conditioning_train = conditioning_all[:len(X_train)]
                        conditioning_val = conditioning_all[len(X_train):]
                        
                        # Train model
                        model, trainer = train_llm_model(
                            X_train, X_val, conditioning_train, conditioning_val,
                            conditioning_spec, args, metadata, pca
                        )
                        trained_models['llm'] = (model, trainer)
                        
                    except Exception as e:
                        if args.llm_fallback:
                            logger.warning(f"LLM conditioning failed, falling back to zero: {e}")
                            model, trainer = train_zero_model(X_train, X_val, args, metadata)
                            trained_models['llm_fallback'] = (model, trainer)
                        else:
                            raise
            
            except Exception as e:
                error_msg = f"Failed to train {model_type} model: {e}"
                logger.error(error_msg)
                
                if args.skip_on_error:
                    logger.warning(f"Skipping {model_type} model due to error")
                    continue
                else:
                    raise RuntimeError(error_msg)
        
        # Summary
        logger.info(f"\n{'='*60}")
        logger.info("TRAINING SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Successfully trained {len(trained_models)} models:")
        for model_type in trained_models:
            logger.info(f"  ✓ {model_type}")
        
        logger.info(f"\nCheckpoints saved to: {args.checkpoint_dir}")
        logger.info("Pre-COVID training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()
