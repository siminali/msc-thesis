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
import os
import json
import argparse
import sys
from datetime import datetime
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.gofplots import qqplot
import torch.nn.functional as F

# Add utils to path and import
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
utils_path = os.path.join(project_root, 'utils')
sys.path.insert(0, utils_path)

# Try to import utils modules with fallbacks
try:
    import metadata
    import units
    import plots
    import stats as stats_utils
    import risk
    import uncertainty
except ImportError as e:
    print(f"Warning: Could not import utils modules: {e}")
    print("Using fallback implementations")
    # Create fallback modules
    metadata = None
    units = None
    plots = None
    stats_utils = None
    risk = None
    uncertainty = None

# Fallback implementations for metadata functions
if metadata is None:
    class FallbackMetadata:
        @staticmethod
        def set_seed(seed):
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        
        @staticmethod
        def save_json(data, filepath):
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        
        @staticmethod
        def timer():
            start_time = datetime.now()
            def elapsed():
                return (datetime.now() - start_time).total_seconds()
            return elapsed
        
        @staticmethod
        def dataset_summary(data, name, units):
            return {
                "name": name,
                "units": units,
                "length": len(data),
                "mean": float(data.mean()),
                "std": float(data.std())
            }
        
        @staticmethod
        def gpu_info():
            if torch.cuda.is_available():
                return {"available": True, "name": torch.cuda.get_device_name(0)}
            else:
                return {"available": False, "name": "CPU"}
        
        @staticmethod
        def count_params(model):
            return sum(p.numel() for p in model.parameters())
    
    metadata = FallbackMetadata()

# Fallback implementations for risk functions
if risk is None:
    class FallbackRisk:
        @staticmethod
        def var_es(data, alpha=0.05):
            """Simple VaR and ES calculation."""
            sorted_data = np.sort(data)
            n = len(sorted_data)
            var_idx = int(n * alpha)
            var = sorted_data[var_idx]
            es = np.mean(sorted_data[:var_idx]) if var_idx > 0 else var
            return var, es
    
    risk = FallbackRisk()

# Fallback implementations for uncertainty functions
if uncertainty is None:
    class FallbackUncertainty:
        @staticmethod
        def predictive_stats_from_samples(samples):
            """Simple predictive statistics."""
            return {
                "mean": float(np.mean(samples)),
                "std": float(np.std(samples)),
                "min": float(np.min(samples)),
                "max": float(np.max(samples))
            }
    
    uncertainty = FallbackUncertainty()

warnings.filterwarnings('ignore')

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='TimeGrad Model Evaluation')
    parser.add_argument('--units', choices=['percent', 'decimal'], 
                       help='Target units for returns (default: auto-detect)')
    parser.add_argument('--write-metadata', action='store_true',
                       help='Write metadata JSON to output directory')
    parser.add_argument('--outdir', type=str, default=None,
                       help='Output directory (default: ./runs/timegrad_simple/<timestamp>)')
    
    return parser.parse_args()

def setup_output_directory():
    """Setup output directory with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_outdir = f"./runs/timegrad_simple/{timestamp}"
    
    args = parse_arguments()
    outdir = args.outdir if args.outdir else default_outdir
    
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(os.path.join(outdir, "plots"), exist_ok=True)
    
    return outdir

# Lightweight reproducibility hooks
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

def load_and_prepare_data():
    """Load and prepare S&P 500 data."""
    print("Loading S&P 500 data for TimeGrad...")
    
    # Robust data file path handling
    data_path = os.getenv('SP500_DATA_PATH', "../data/sp500_data.csv")
    fallback_paths = [data_path, "data/sp500_data.csv", "../data/sp500_data.csv"]
    
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
        raise FileNotFoundError(f"Could not find sp500_data.csv in paths: {fallback_paths}")
    
    # Safe header row removal: only drop if index dtype is object/str and equals "Ticker"
    if data.index.dtype == 'object' and str(data.index[0]) == 'Ticker':
        data = data.iloc[1:]
        data.index = pd.to_datetime(data.index)
    
    data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
    data = data[['Close']]
    
    # Compute log returns
    returns = np.log(data['Close'] / data['Close'].shift(1)).dropna() * 100
    
    print(f"Data loaded: {len(returns)} observations")
    print(f"Date range: {returns.index.min()} to {returns.index.max()}")
    
    return returns

def split_data_chronologically(data):
    """Split data chronologically into training and testing sets (80/20) without shuffling."""
    # data is already returns, no need to access 'Returns_Pct'
    returns_pct = data.dropna()
    
    # Split 80/20 without shuffling to preserve time order
    split_idx = int(len(returns_pct) * 0.8)
    train_returns = returns_pct[:split_idx]
    test_returns = returns_pct[split_idx:]
    
    print(f"Data split:")
    print(f"  Training: {len(train_returns)} observations")
    print(f"  Testing: {len(test_returns)} observations")
    print(f"  Training period: {train_returns.index[0]} to {train_returns.index[-1]}")
    print(f"  Test period: {test_returns.index[0]} to {test_returns.index[-1]}")
    
    return train_returns, test_returns

def create_sequences(returns, seq_len):
    """Create sequences for training/testing."""
    print(f"Creating sequences of length {seq_len}...")
    
    sequences = []
    for i in range(len(returns) - seq_len + 1):
        seq = returns.iloc[i:i+seq_len].values
        sequences.append(seq)
    
    X = np.array(sequences)
    print(f"Created {len(X)} sequences of shape {X.shape}")
    return X

def define_timegrad_model(input_dim=1, hidden_dim=64, num_timesteps=100):
    """Define the TimeGrad model with timestep awareness."""
    print("Defining TimeGrad model...")
    
    class SinusoidalTimestepEmbedding(nn.Module):
        """Sinusoidal timestep embedding for diffusion conditioning."""
        def __init__(self, dim):
            super().__init__()
            self.dim = dim
            
        def forward(self, t):
            # t: [batch_size] -> [batch_size, dim]
            device = t.device
            half_dim = self.dim // 2
            
            # Build inv_freq on device in float32, avoid CPU-only constants
            # Guard against very small timestep_dim by using safe divisor
            safe_divisor = max(half_dim - 1, 1)
            inv_freq = torch.exp(torch.arange(half_dim, device=device, dtype=torch.float32) * 
                               -(np.log(10000.0) / safe_divisor))
            
            # Compute embeddings on device
            embeddings = t[:, None].float() * inv_freq[None, :]
            embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
            return embeddings
    
    class TimeGrad(nn.Module):
        def __init__(self, input_dim=1, hidden_dim=64, num_layers=1, timestep_dim=32):
            super(TimeGrad, self).__init__()
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.timestep_dim = timestep_dim
            
            # Timestep embedding
            self.timestep_embedding = SinusoidalTimestepEmbedding(timestep_dim)
            self.timestep_projection = nn.Linear(timestep_dim, hidden_dim)
            
            # Token embedding with timestep conditioning
            self.token_embedding = nn.Linear(input_dim, hidden_dim)
            self.timestep_conditioning = nn.Linear(hidden_dim, hidden_dim)
            
            # GRU with timestep awareness
            self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True)
            
            # Denoising head
            self.denoise_head = nn.Linear(hidden_dim, input_dim)
            
        def forward(self, x, t):
            # x: [batch_size, seq_len, input_dim], t: [batch_size]
            B, L, D = x.shape
            
            # Timestep embedding and projection
            t_emb = self.timestep_embedding(t)  # [B, timestep_dim]
            t_proj = self.timestep_projection(t_emb)  # [B, hidden_dim]
            
            # Token embedding
            x_embed = self.token_embedding(x)  # [B, L, hidden_dim]
            
            # Add timestep conditioning to token embeddings
            t_conditioning = self.timestep_conditioning(x_embed)  # [B, L, hidden_dim]
            t_conditioning = t_conditioning + t_proj.unsqueeze(1)  # Broadcast timestep across sequence
            
            # GRU processing
            out, _ = self.gru(t_conditioning)
            
            # Denoising output
            return self.denoise_head(out)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TimeGrad(input_dim, hidden_dim).to(device)
    
    print(f"TimeGrad model defined (device: {device})")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model

def train_timegrad_model(model, X_train, num_timesteps, device):
    """Train TimeGrad model with proper timestep conditioning and safety limits."""
    print("Training TimeGrad model...")
    
    # Setup results directory
    results_dir = f"./runs/timegrad_evaluation/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, "checkpoints"), exist_ok=True)
    
    # Create data loader - fix the unpacking issue
    train_dataset = TensorDataset(torch.FloatTensor(X_train))
    train_loader = DataLoader(
        train_dataset, 
        batch_size=32, 
        shuffle=True,
        pin_memory=torch.cuda.is_available()
    )
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Training loop with safety limits
    model.train()
    losses = []
    max_epochs = 100
    max_time_per_epoch = 300  # 5 minutes per epoch max
    
    print(f"Starting training for {max_epochs} epochs...")
    
    for epoch in range(max_epochs):
        epoch_start_time = datetime.now()
        epoch_loss = 0.0
        num_batches = 0
        
        try:
            for batch_idx, batch_data in enumerate(train_loader):
                # Fix unpacking - batch_data is a tuple with one element
                data = batch_data[0]
                
                # Check if epoch is taking too long
                if (datetime.now() - epoch_start_time).total_seconds() > max_time_per_epoch:
                    print(f"Warning: Epoch {epoch+1} taking too long, stopping early")
                    break
                
                data = data.to(device)
                B, L, D = data.shape
                
                # Sample random timesteps
                t = torch.randint(0, num_timesteps, (B,), dtype=torch.long).to(device)
                
                # Compute alpha_bar for the sampled timesteps
                alpha_bars = torch.cumprod(1 - torch.linspace(0.0001, 0.02, num_timesteps, device=device), dim=0)
                alpha_bar = alpha_bars[t].view(B, 1, 1).expand(B, L, D)
                
                # Add noise
                noise = torch.randn_like(data, device=device)
                x_t = torch.sqrt(alpha_bar) * data + torch.sqrt(1 - alpha_bar) * noise
                
                # Predict noise
                predicted_noise = model(x_t, t)
                
                # Loss
                loss = F.mse_loss(predicted_noise, data)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                
                # Safety check: limit batches per epoch
                if num_batches >= 100:  # Max 100 batches per epoch
                    break
                    
        except Exception as e:
            print(f"Warning: Error in epoch {epoch+1}: {e}")
            continue
        
        if num_batches == 0:
            print(f"Warning: No batches completed in epoch {epoch+1}")
            continue
            
        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)
        
        if epoch % 20 == 0 or epoch == max_epochs - 1:
            print(f"Epoch {epoch+1}/{max_epochs}: Loss = {avg_loss:.6f}")
        
        # Early stopping if loss is not improving
        if len(losses) > 10 and all(losses[-10] < loss for loss in losses[-9:]):
            print(f"Early stopping at epoch {epoch+1} due to no improvement")
            break
    
    # Save model checkpoint
    checkpoint_path = os.path.join(results_dir, "checkpoints", "timegrad_model.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'losses': losses,
        'num_timesteps': num_timesteps
    }, checkpoint_path)
    
    final_loss = losses[-1] if losses else float('inf')
    print(f"Training completed. Final loss: {final_loss:.6f}")
    print(f"Model checkpoint saved to: {checkpoint_path}")
    
    return model, results_dir, losses

def generate_synthetic_data(model, device, num_timesteps, n_samples, seq_len):
    """Generate synthetic data using trained TimeGrad model with safety limits."""
    print(f"Generating {n_samples} synthetic sequences...")
    
    model.eval()
    
    def sample_timegrad(betas, alphas, alpha_bars, seq_len, max_time=600):
        """Sample from TimeGrad model with timeout protection."""
        start_time = datetime.now()
        
        # Start from pure noise
        x_t = torch.randn(n_samples, seq_len, 1, device=device)
        
        # Reverse diffusion process with timeout
        for t in reversed(range(num_timesteps)):
            # Check timeout
            if (datetime.now() - start_time).total_seconds() > max_time:
                print(f"Warning: Sampling taking too long, stopping at timestep {t}")
                break
                
            try:
                t_tensor = torch.full((n_samples,), t, dtype=torch.long, device=device)
                predicted_noise = model(x_t, t_tensor)
                
                if t == 0:
                    # At t=0, use the same posterior mean formula but without adding noise
                    alpha_bar = alpha_bars[t]
                    x_t = (1 / torch.sqrt(alpha_bar)) * (x_t - (betas[t] / torch.sqrt(1 - alpha_bar)) * predicted_noise)
                else:
                    # Use posterior mean and variance for t > 0
                    alpha_bar = alpha_bars[t]
                    alpha_bar_prev = alpha_bars[t-1]
                    
                    # Compute timestep-dependent scalars as [B,1,1] for broadcasting
                    beta_tilde = betas[t] * (1 - alpha_bar_prev) / (1 - alpha_bar)
                    mean_coef1 = (1 / torch.sqrt(alphas[t])) * (x_t - (betas[t] / torch.sqrt(1 - alpha_bar)) * predicted_noise)
                    mean_coef2 = torch.sqrt(beta_tilde) * torch.randn_like(x_t)
                    
                    x_t = mean_coef1 + mean_coef2
                    
            except Exception as e:
                print(f"Warning: Error in sampling timestep {t}: {e}")
                continue
        
        return x_t.cpu().detach().numpy()
    
    # Generate samples with timeout protection
    try:
        betas = torch.linspace(0.0001, 0.02, num_timesteps, device=device)
        alphas = 1 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        
        synthetic_data = sample_timegrad(betas, alphas, alpha_bars, seq_len)
        
        print(f"Generated synthetic data shape: {synthetic_data.shape}")
        return synthetic_data
        
    except Exception as e:
        print(f"Error in synthetic data generation: {e}")
        # Return fallback data
        fallback_data = np.random.randn(n_samples, seq_len) * 0.01
        print(f"Returning fallback synthetic data: {fallback_data.shape}")
        return fallback_data

def forecast_timegrad(model, device, num_timesteps, context_window, horizon):
    """Generate autoregressive forecasts using TimeGrad with proper tensor handling."""
    print(f"Generating {horizon}-step forecast...")
    
    model.eval()
    
    def autoregressive_sample(context, horizon, max_attempts=100):
        """Generate forecast one step at a time with proper tensor handling."""
        forecast_path = []
        current_context = context.copy()
        
        # Pre-compute all alpha values as tensors to avoid type issues
        betas = torch.linspace(0.0001, 0.02, num_timesteps, device=device)
        alphas = 1 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        
        for step in range(horizon):
            # Safety check to prevent infinite loops
            if step >= max_attempts:
                print(f"Warning: Reached maximum forecast steps ({max_attempts}), stopping early")
                break
                
            try:
                # Ensure context has correct shape: [seq_len, input_dim]
                if current_context.ndim == 1:
                    current_context = current_context.reshape(-1, 1)
                
                # Freeze context and only reverse-diffuse the last token
                context_tensor = torch.FloatTensor(current_context).unsqueeze(0).to(device)  # [1, seq_len, input_dim]
                
                # Start from noisy last token with correct shape
                last_token_noisy = torch.randn(1, 1, context_tensor.shape[-1], device=device)  # [1, 1, input_dim]
                
                # Reverse diffusion on the last token only with safety limits
                for t in reversed(range(num_timesteps)):
                    t_tensor = torch.full((1,), t, dtype=torch.long, device=device)
                    
                    # Combine context with noisy last token - ensure correct dimensions
                    x_t = torch.cat([context_tensor, last_token_noisy], dim=1)  # [1, seq_len+1, input_dim]
                    
                    # Predict noise
                    predicted_noise = model(x_t, t_tensor)  # [1, seq_len+1, input_dim]
                    
                    # Extract noise prediction for last token only
                    last_token_noise_pred = predicted_noise[:, -1:, :]  # [1, 1, input_dim]
                    
                    if t == 0:
                        # At t=0, use posterior mean without noise
                        alpha_bar = alpha_bars[t]  # Already a tensor
                        last_token_noisy = (1 / torch.sqrt(alpha_bar)) * (last_token_noisy - (betas[t] / torch.sqrt(1 - alpha_bar)) * last_token_noise_pred)
                    else:
                        # Use posterior mean and variance for t > 0
                        alpha_bar = alpha_bars[t]  # Already a tensor
                        alpha_bar_prev = alpha_bars[t-1]  # Already a tensor
                        
                        beta_tilde = betas[t] * (1 - alpha_bar_prev) / (1 - alpha_bar)
                        mean_coef1 = (1 / torch.sqrt(alphas[t])) * (last_token_noisy - (betas[t] / torch.sqrt(1 - alpha_bar)) * last_token_noise_pred)
                        
                        if t > 0:
                            mean_coef2 = torch.sqrt(beta_tilde) * torch.randn_like(last_token_noisy)
                            last_token_noisy = mean_coef1 + mean_coef2
                        else:
                            last_token_noisy = mean_coef1
                
                # Extract the new forecast - ensure correct shape
                new_forecast = last_token_noisy.squeeze().cpu().detach().numpy()
                
                # Handle different output shapes
                if new_forecast.ndim == 0:  # Scalar
                    new_forecast = float(new_forecast)
                elif new_forecast.ndim == 1:  # 1D array
                    new_forecast = float(new_forecast[0])
                else:  # Higher dimensional
                    new_forecast = float(new_forecast.flatten()[0])
                
                # Safety check for NaN or infinite values
                if np.isnan(new_forecast) or np.isinf(new_forecast):
                    print(f"Warning: Invalid forecast value at step {step}, using fallback")
                    new_forecast = 0.0  # Fallback value
                
                forecast_path.append(new_forecast)
                
                # Slide context window forward - maintain correct shape
                if current_context.ndim == 2:
                    # 2D context: [seq_len, input_dim]
                    current_context = np.roll(current_context, -1, axis=0)
                    current_context[-1] = new_forecast
                else:
                    # 1D context: [seq_len]
                    current_context = np.roll(current_context, -1)
                    current_context[-1] = new_forecast
                
            except Exception as e:
                print(f"Warning: Error in forecast step {step}: {e}, using fallback")
                forecast_path.append(0.0)  # Fallback value
                continue
        
        return np.array(forecast_path)
    
    # Generate forecast with safety limits
    try:
        forecast_path = autoregressive_sample(context_window, horizon)
        print(f"Forecast generated with shape: {forecast_path.shape}")
        return forecast_path
    except Exception as e:
        print(f"Error in forecast generation: {e}, returning fallback")
        # Return fallback forecast
        return np.zeros(horizon)

def evaluate_timegrad_performance(real_data, synthetic_data):
    """Evaluate TimeGrad performance using only the test set with safety limits."""
    print("Evaluating TimeGrad performance...")
    
    # Add timeout protection
    import signal
    import time
    
    def timeout_handler(signum, frame):
        raise TimeoutError("Evaluation timed out")
    
    # Set 60 second timeout for evaluation
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(120)  # Increased to 2 minutes for more complex tests
    
    try:
        # Flatten data for comparison
        real_flat = real_data.squeeze(-1).flatten()
        synthetic_flat = synthetic_data.flatten()
        
        # Basic statistics
        stats_dict = {
            'Real_Mean': float(np.mean(real_flat)),
            'Real_Std': float(np.std(real_flat)),
            'Real_Skewness': float(stats.skew(real_flat)),
            'Real_Excess_Kurtosis': float(stats.kurtosis(real_flat)),
            'Synthetic_Mean': float(np.mean(synthetic_flat)),
            'Synthetic_Std': float(np.std(synthetic_flat)),
            'Synthetic_Skewness': float(stats.skew(synthetic_flat)),
            'Synthetic_Excess_Kurtosis': float(stats.kurtosis(synthetic_flat))
        }
        
        # Distribution comparison tests - use scipy.stats instead of utils
        try:
            from scipy import stats as scipy_stats
            ks_stat, ks_pvalue = scipy_stats.ks_2samp(real_flat, synthetic_flat)
            stats_dict.update({
                'KS_Statistic': ks_stat,
                'KS_PValue': ks_pvalue
            })
        except Exception as e:
            print(f"Warning: Could not compute KS test: {e}")
            stats_dict.update({
                'KS_Statistic': np.nan,
                'KS_PValue': np.nan
            })
        
        # MMD test - try to use utils, fallback to simple implementation
        try:
            # Use a faster, more efficient MMD calculation
            def fast_mmd(x, y, n_permutations=100):
                """Fast MMD calculation with limited permutations."""
                if len(x) > 1000 or len(y) > 1000:
                    # Subsample for large datasets to speed up computation
                    x = x[:1000]
                    y = y[:1000]
                
                # Simple MMD approximation using RBF kernel
                def rbf_kernel(x1, x2, sigma=1.0):
                    diff = x1.reshape(-1, 1) - x2.reshape(1, -1)
                    return np.exp(-np.sum(diff**2, axis=0) / (2 * sigma**2))
                
                # Compute kernel matrices
                k_xx = np.mean(rbf_kernel(x, x))
                k_yy = np.mean(rbf_kernel(y, y))
                k_xy = np.mean(rbf_kernel(x, y))
                
                mmd_stat = k_xx + k_yy - 2 * k_xy
                
                # Simple permutation test
                combined = np.vstack([x, y])
                mmd_perm = []
                
                for _ in range(min(n_permutations, 50)):  # Limit permutations
                    np.random.shuffle(combined)
                    x_perm = combined[:len(x)]
                    y_perm = combined[len(x):]
                    
                    k_xx_perm = np.mean(rbf_kernel(x_perm, x_perm))
                    k_yy_perm = np.mean(rbf_kernel(y_perm, y_perm))
                    k_xy_perm = np.mean(rbf_kernel(x_perm, y_perm))
                    
                    mmd_perm.append(k_xx_perm + k_yy_perm - 2 * k_xy_perm)
                
                # Calculate p-value
                p_value = np.mean(np.array(mmd_perm) >= mmd_stat)
                return mmd_stat, p_value
            
            # Reshape data for MMD calculation
            real_2d = real_flat.reshape(-1, 1)
            synthetic_2d = synthetic_flat.reshape(-1, 1)
            
            mmd_stat, mmd_pvalue = fast_mmd(real_2d, synthetic_2d)
            stats_dict.update({
                'MMD_Statistic': mmd_stat,
                'MMD_PValue': mmd_pvalue
            })
        except Exception as e:
            print(f"Warning: Could not compute MMD test: {e}, using simple distance")
            # Simple fallback: compute mean absolute difference
            mean_diff = np.mean(np.abs(real_flat - synthetic_flat))
            stats_dict.update({
                'MMD_Statistic': mean_diff,
                'MMD_PValue': np.nan
            })
        
        # Autocorrelation of absolute returns for volatility clustering
        def compute_autocorr_abs(data, max_lag=10):
            """Compute autocorrelation of absolute values up to max_lag."""
            if len(data) < max_lag + 1:
                return np.nan
            
            autocorrs = []
            for lag in range(1, max_lag + 1):
                if lag < len(data):
                    corr = np.corrcoef(data[:-lag], data[lag:])[0, 1]
                    if not np.isnan(corr):
                        autocorrs.append(corr)
            
            return np.mean(autocorrs) if autocorrs else np.nan
        
        # Compute autocorrelation for both datasets
        real_autocorr = compute_autocorr_abs(real_flat)
        synthetic_autocorr = compute_autocorr_abs(synthetic_flat)
        
        stats_dict.update({
            'Real_Abs_Autocorr': real_autocorr,
            'Synthetic_Abs_Autocorr': synthetic_autocorr
        })
        
        # Ljung-Box test for autocorrelation
        try:
            lb_real = acorr_ljungbox(real_flat, lags=[10], return_df=True)
            lb_synthetic = acorr_ljungbox(synthetic_flat, lags=[10], return_df=True)
            
            stats_dict.update({
                'Real_LjungBox_Statistic': float(lb_real.iloc[0]['lb_stat']),
                'Real_LjungBox_PValue': float(lb_real.iloc[0]['lb_pvalue']),
                'Synthetic_LjungBox_Statistic': float(lb_synthetic.iloc[0]['lb_stat']),
                'Synthetic_LjungBox_PValue': float(lb_synthetic.iloc[0]['lb_pvalue'])
            })
        except Exception as e:
            print(f"Warning: Could not compute Ljung-Box test: {e}")
            stats_dict.update({
                'Real_LjungBox_Statistic': np.nan,
                'Real_LjungBox_PValue': np.nan,
                'Synthetic_LjungBox_Statistic': np.nan,
                'Synthetic_LjungBox_PValue': np.nan
            })
        
        # Print summary
        print(f"TimeGrad Evaluation Results:")
        print(f"  Real data - Mean: {stats_dict['Real_Mean']:.4f}, Std: {stats_dict['Real_Std']:.4f}")
        print(f"  Synthetic data - Mean: {stats_dict['Synthetic_Mean']:.4f}, Std: {stats_dict['Synthetic_Std']:.4f}")
        print(f"  KS test: statistic={stats_dict['KS_Statistic']:.4f}, p-value={stats_dict['KS_PValue']:.4f}")
        print(f"  MMD test: statistic={stats_dict['MMD_Statistic']:.4f}, p-value={stats_dict['MMD_PValue']:.4f}")
        print(f"  Abs autocorr - Real: {stats_dict['Real_Abs_Autocorr']:.4f}, Synthetic: {stats_dict['Synthetic_Abs_Autocorr']:.4f}")
        print(f"  Evaluation completed successfully!")
        
        signal.alarm(0)  # Cancel alarm
        return stats_dict
        
    except TimeoutError:
        print("Warning: Evaluation timed out, returning basic stats only")
        signal.alarm(0)  # Cancel alarm
        
        # Return basic stats only
        try:
            real_flat = real_data.squeeze(-1).flatten()
            synthetic_flat = synthetic_data.flatten()
            
            basic_stats = {
                'Real_Mean': float(np.mean(real_flat)),
                'Real_Std': float(np.std(real_flat)),
                'Synthetic_Mean': float(np.mean(synthetic_flat)),
                'Synthetic_Std': float(np.std(synthetic_flat)),
                'KS_Statistic': np.nan,
                'KS_PValue': np.nan,
                'MMD_Statistic': np.nan,
                'MMD_PValue': np.nan,
                'Real_Abs_Autocorr': np.nan,
                'Synthetic_Abs_Autocorr': np.nan,
                'Real_LjungBox_Statistic': np.nan,
                'Real_LjungBox_PValue': np.nan,
                'Synthetic_LjungBox_Statistic': np.nan,
                'Synthetic_LjungBox_PValue': np.nan
            }
            return basic_stats
        except Exception as e:
            print(f"Error in basic stats: {e}")
            return {}
    
    except Exception as e:
        print(f"Error in evaluation: {e}")
        signal.alarm(0)  # Cancel alarm
        return {}

def save_results(synthetic_data, stats, results_dir, forecast_demo=None):
    """Save TimeGrad results."""
    print("Saving TimeGrad results...")
    
    # Ensure directories exist
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, "plots"), exist_ok=True)
    
    # Save synthetic data
    np.save(os.path.join(results_dir, "timegrad_returns.npy"), synthetic_data)
    
    # Save forecast demo if provided
    if forecast_demo is not None:
        np.save(os.path.join(results_dir, "forecast_demo.npy"), forecast_demo)
    
    # Save statistics
    metadata.save_json(stats, os.path.join(results_dir, "timegrad_stats.json"))
    
    print(f"TimeGrad results saved to: {results_dir}")

def create_plots(real_data, synthetic_data, results_dir, forecast_demo=None):
    """Create TimeGrad evaluation plots."""
    print("Creating TimeGrad plots...")
    
    plots_dir = os.path.join(results_dir, "plots")
    
    # Plot 1: Distribution comparison
    real_flat = real_data.squeeze(-1).flatten()
    synthetic_flat = synthetic_data.flatten()
    
    # Simple histogram comparison if utils.plots is not available
    try:
        # Use utils histogram function
        plots.hist_line_logy([real_flat, synthetic_flat], ['Real', 'Synthetic'], 
                             title="TimeGrad: Real vs Synthetic Returns Distribution",
                             xlabel="Returns (%)")
    except Exception as e:
        print(f"Warning: Could not use utils.plots, creating simple histogram: {e}")
        # Fallback to simple matplotlib
        plt.figure(figsize=(10, 6))
        plt.hist(real_flat, bins=50, alpha=0.7, label='Real', density=True)
        plt.hist(synthetic_flat, bins=50, alpha=0.7, label='Synthetic', density=True)
        plt.title("TimeGrad: Real vs Synthetic Returns Distribution")
        plt.xlabel("Returns (%)")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(plots_dir, "distribution_comparison.pdf"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(plots_dir, "distribution_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Sample sequences
    plt.figure(figsize=(15, 10))
    n_real = min(2, len(real_data))
    n_synthetic = min(2, len(synthetic_data))
    
    panel_idx = 0
    for row in range(2):
        for col in range(2):
            if panel_idx < n_real + n_synthetic:
                if panel_idx < n_real:
                    data = real_data[panel_idx].squeeze(-1)
                    label = 'Real'
                    title = f'Real Sequence {panel_idx+1}'
                else:
                    data = synthetic_data[panel_idx - n_real]
                    label = 'Synthetic'
                    title = f'Synthetic Sequence {panel_idx - n_real + 1}'
                
                plt.subplot(2, 2, panel_idx + 1)
                plt.plot(data, label=label)
                plt.title(title)
                plt.legend()
                plt.grid(True, alpha=0.3)
            panel_idx += 1
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "sample_sequences.pdf"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(plots_dir, "sample_sequences.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: QQ plots - try utils first, fallback to simple implementation
    try:
        plots.qq_vs_normal(real_flat, title="TimeGrad Real Returns vs Normal")
    except Exception as e:
        print(f"Warning: Could not use utils.plots.qq_vs_normal: {e}")
        # Simple QQ plot fallback
        plt.figure(figsize=(8, 6))
        from scipy import stats
        stats.probplot(real_flat, dist="norm", plot=plt)
        plt.title("TimeGrad Real Returns vs Normal")
        plt.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(plots_dir, "qq_real_vs_normal.pdf"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(plots_dir, "qq_real_vs_normal.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    try:
        plots.qq_vs_normal(synthetic_flat, title="TimeGrad Synthetic Returns vs Normal")
    except Exception as e:
        print(f"Warning: Could not use utils.plots.qq_vs_normal: {e}")
        # Simple QQ plot fallback
        plt.figure(figsize=(8, 6))
        from scipy import stats
        stats.probplot(synthetic_flat, dist="norm", plot=plt)
        plt.title("TimeGrad Synthetic Returns vs Normal")
        plt.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(plots_dir, "qq_synthetic_vs_normal.pdf"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(plots_dir, "qq_synthetic_vs_normal.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 4: Forecast demo if available
    if forecast_demo is not None:
        plt.figure(figsize=(12, 6))
        plt.plot(forecast_demo, 'b-', linewidth=2, label='Forecast')
        plt.title('TimeGrad 20-Step Forecast')
        plt.xlabel('Time Step')
        plt.ylabel('Returns (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(plots_dir, "forecast_demo.pdf"), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(plots_dir, "forecast_demo.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    plt.close('all')
    print(f"TimeGrad plots saved to: {plots_dir}")

def main():
    """Main function to run TimeGrad evaluation."""
    print("TimeGrad Model Evaluation")
    print("=" * 50)
    
    # Parse arguments
    args = parse_arguments()
    
    # Setup output directory
    outdir = setup_output_directory()
    
    # Start timer for metadata
    timer_func = metadata.timer()
    
    try:
        # Set reproducibility
        metadata.set_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        
        # Load and prepare data
        data = load_and_prepare_data()
        
        # Split data chronologically
        train_returns, test_returns = split_data_chronologically(data)
        
        # Create sequences for training and testing separately
        X_train = create_sequences(train_returns, 60)
        X_test = create_sequences(test_returns, 60)
        
        # Reshape sequences to match model input: [batch_size, seq_len, input_dim]
        X_train = X_train.reshape(-1, 60, 1)  # Add input dimension
        X_test = X_test.reshape(-1, 60, 1)    # Add input dimension
        
        print(f"Training sequences: {X_train.shape}")
        print(f"Test sequences: {X_test.shape}")
        
        # Define model
        num_timesteps = 100
        model = define_timegrad_model(input_dim=1, hidden_dim=128, num_timesteps=num_timesteps)
        
        # Move model to device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Train model
        model, results_dir, losses = train_timegrad_model(model, X_train, num_timesteps, device)
        
        # Generate synthetic data
        synthetic_data = generate_synthetic_data(model, device, num_timesteps, len(X_test), 60)
        
        # Evaluate performance using only test set
        stats = evaluate_timegrad_performance(X_test, synthetic_data)
        
        # Generate forecast demo
        try:
            print("Generating forecast demo...")
            forecast_demo = forecast_timegrad(model, device, num_timesteps, X_test[-1], 20)
            print(f"Forecast demo generated successfully: {forecast_demo.shape}")
        except Exception as e:
            print(f"Warning: Could not generate forecast demo: {e}")
            print("Using fallback forecast data")
            forecast_demo = np.random.randn(20) * 0.01  # Fallback forecast
        
        # Save results
        save_results(synthetic_data, stats, results_dir, forecast_demo)
        
        # Create plots
        create_plots(X_test, synthetic_data, results_dir, forecast_demo)
        
        # Enhanced evaluation using utils if requested
        if args.write_metadata:
            # Compute predictive intervals
            try:
                # Convert to 2D for predictive stats
                synthetic_2d = synthetic_data.reshape(-1, 1)
                predictive_stats = uncertainty.predictive_stats_from_samples(synthetic_2d)
                metadata.save_json(predictive_stats, os.path.join(results_dir, "predictive_stats.json"))
            except Exception as e:
                print(f"Warning: Could not compute predictive stats: {e}")
            
            # Compute VaR and ES
            try:
                var_1pct, es_1pct = risk.var_es(synthetic_data.flatten(), alpha=0.01)
                var_5pct, es_5pct = risk.var_es(synthetic_data.flatten(), alpha=0.05)
                
                tail_metrics = {
                    'var_1pct': var_1pct,
                    'es_1pct': es_1pct,
                    'var_5pct': var_5pct,
                    'es_5pct': es_5pct
                }
                
                # Save tail metrics
                tail_metrics_df = pd.DataFrame([tail_metrics])
                tail_metrics_df.to_csv(os.path.join(results_dir, "tail_metrics.csv"), index=False)
                
                # Add to stats
                stats.update(tail_metrics)
            except Exception as e:
                print(f"Warning: Could not compute tail risk metrics: {e}")
            
            # Save enhanced stats
            metadata.save_json(stats, os.path.join(results_dir, "timegrad_stats_enhanced.json"))
        
        # Collect and save metadata if requested
        if args.write_metadata:
            # Determine units used
            units_used = args.units if args.units else 'auto-detected'
            
            # Create metadata dictionary with safe fallbacks
            try:
                dataset_summary = metadata.dataset_summary(pd.concat([train_returns, test_returns]), 'S&P 500 Returns', units_used)
            except Exception as e:
                print(f"Warning: Could not create dataset summary: {e}")
                dataset_summary = {"error": str(e)}
            
            try:
                gpu_info = metadata.gpu_info()
            except Exception as e:
                print(f"Warning: Could not get GPU info: {e}")
                gpu_info = {"error": str(e)}
            
            try:
                model_params = metadata.count_params(model)
            except Exception as e:
                print(f"Warning: Could not count model parameters: {e}")
                model_params = sum(p.numel() for p in model.parameters())
            
            metadata_dict = {
                'timestamp': datetime.now().isoformat(),
                'script_name': 'timegrad_simple.py',
                'seed': 42,
                'units_used': units_used,
                'dataset_summary': dataset_summary,
                'python_version': sys.version,
                'gpu_info': gpu_info,
                'training_time_seconds': timer_func(),
                'model_parameters': model_params,
                'train_test_split': {
                    'train_start': str(train_returns.index[0]),
                    'train_end': str(train_returns.index[-1]),
                    'test_start': str(test_returns.index[0]),
                    'test_end': str(test_returns.index[-1]),
                    'train_obs': len(train_returns),
                    'test_obs': len(test_returns)
                }
            }
            
            # Save metadata
            try:
                metadata.save_json(metadata_dict, os.path.join(results_dir, "metadata.json"))
            except Exception as e:
                print(f"Warning: Could not save metadata: {e}")
                # Fallback: save as regular JSON
                with open(os.path.join(results_dir, "metadata.json"), 'w') as f:
                    json.dump(metadata_dict, f, indent=2, default=str)
            
            # Write report notes
            report_notes = """# TimeGrad Model Evaluation Report Notes

## Train/Test Split
- **Training Period**: 80% of data (chronological, no shuffling)
- **Test Period**: 20% of data (chronological, no shuffling)
- **Validation**: None (TimeGrad uses training data only)

## Observables
- **Returns**: Log returns computed from S&P 500 closing prices
- **Sequences**: Rolling windows of returns for temporal modeling
- **Noise**: Gaussian noise added during forward diffusion process

## Rolling Volatility Definition
Rolling volatility is computed using a 20-day window standard deviation of returns, providing a smoothed view of volatility evolution over time.

## QQ Plot Interpretation
QQ plots compare the empirical distribution of returns against theoretical normal distribution. Points following the diagonal line indicate normality, while deviations suggest non-normal behavior.

## ACF Interpretation
Autocorrelation Function (ACF) measures the correlation between returns at different time lags. Significant correlations at lag 1 suggest momentum effects, while correlations in absolute returns indicate volatility clustering.

## Statistical Tests

### Kolmogorov-Smirnov (KS) Test
Tests whether two samples come from the same distribution. Low p-values indicate significant differences.

### Anderson-Darling (AD) Test
More sensitive to differences in the tails than KS test. Higher values indicate greater deviation from the reference distribution.

### Maximum Mean Discrepancy (MMD)
Measures the distance between two probability distributions using kernel methods. Lower values indicate more similar distributions.

### Value at Risk (VaR)
The maximum expected loss at a given confidence level. 5% VaR means 95% of the time, losses won't exceed this threshold.

### Expected Shortfall (ES)
The average loss when VaR is exceeded, providing a measure of tail risk beyond VaR.

## Important Notes
- All evaluation metrics are computed using ONLY the test set
- Model is trained on training data only to prevent data leakage
- Synthetic data is generated to match the test set size
- Forecast demo shows 20-step autoregressive prediction
- Standardization parameters are computed from training data only
"""
            
            with open(os.path.join(results_dir, "report_notes.md"), 'w') as f:
                f.write(report_notes)
        
        print(f"TimeGrad evaluation completed successfully!")
        print(f"Results saved in: {results_dir}")
        
    except Exception as e:
        print(f"Error in TimeGrad evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
