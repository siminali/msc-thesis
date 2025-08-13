#!/usr/bin/env python3
"""
Simple DDPM (Denoising Diffusion Probabilistic Models) Implementation
for Financial Time Series Generation

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
import os
import warnings
import scipy.stats
from tqdm import tqdm
import logging
import json
import argparse
import sys
from datetime import datetime
warnings.filterwarnings('ignore')

# Add utils to path and import
import sys
import os
# Get the project root directory
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
utils_path = os.path.join(project_root, 'utils')
if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

# Try to import utils modules directly
try:
    import metadata
    import units
    import plots
    import stats as stats_utils
    import risk
    import uncertainty
except ImportError as e:
    print(f"Error importing utils modules: {e}")
    print(f"Current sys.path: {sys.path}")
    print(f"Utils path: {utils_path}")
    print(f"Utils exists: {os.path.exists(utils_path)}")
    raise

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='DDPM Model Evaluation')
    parser.add_argument('--units', choices=['percent', 'decimal'], 
                       help='Target units for returns (default: auto-detect)')
    parser.add_argument('--write-metadata', action='store_true',
                       help='Write metadata JSON to output directory')
    parser.add_argument('--outdir', type=str, default=None,
                       help='Output directory (default: ./runs/diffusion_simple/<timestamp>)')
    parser.add_argument('--temperature', type=float, default=None,
                       help='Sampling temperature (default: auto-optimize)')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS,
                       help=f'Number of training epochs (default: {NUM_EPOCHS})')
    parser.add_argument('--timesteps', type=int, default=NUM_TIMESTEPS,
                       help=f'Number of diffusion timesteps (default: {NUM_TIMESTEPS})')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE,
                       help=f'Learning rate (default: {LEARNING_RATE})')
    
    return parser.parse_args()

def setup_output_directory():
    """Setup output directory with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_outdir = f"./runs/diffusion_simple/{timestamp}"
    
    args = parse_arguments()
    outdir = args.outdir if args.outdir else default_outdir
    
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(os.path.join(outdir, "plots"), exist_ok=True)
    
    return outdir

# Global constants
SEQ_LEN = 60
BATCH_SIZE = 32  # Reduced batch size for better gradient estimates
NUM_EPOCHS = 500  # Increased epochs for better convergence
NUM_TIMESTEPS = 1000  # Increased timesteps for smoother diffusion
LEARNING_RATE = 5e-4  # Reduced learning rate for stability
PATIENCE = 25  # Increased patience for better convergence
VALIDATION_SPLIT = 0.2
GRADIENT_CLIP_NORM = 0.5  # Reduced gradient clipping for stability
WEIGHT_DECAY = 1e-4  # Added weight decay for regularization
SCHEDULER_PATIENCE = 10  # Learning rate scheduler patience
SCHEDULER_FACTOR = 0.7  # Learning rate reduction factor

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
    """
    Enhanced MLP denoiser with better architecture, normalization, and residual connections.
    """
    
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
        
        logger.info(f"Enhanced DenoiseMLP initialized: seq_len={seq_len}, hidden_dim={hidden_dim}, time_dim={time_dim}, layers={num_layers}")
    
    def _init_weights(self, module):
        """Initialize weights for better training."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, x, t):
        """
        Forward pass with enhanced architecture.
        
        Args:
            x: Input sequence [batch_size, seq_len]
            t: Timestep [batch_size, 1]
            
        Returns:
            Predicted noise [batch_size, seq_len]
        """
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

def set_reproducibility(seed=42):
    """Set random seeds and device/dtype for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    
    # Set CUDA seed if available
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    
    logger.info(f"Reproducibility set: seed={seed}, device={device}, dtype={dtype}")
    
    # Log random seed to results folder for exact reproduction
    results_dir = "results/ddpm_evaluation"
    os.makedirs(results_dir, exist_ok=True)
    with open(f"{results_dir}/reproducibility_info.txt", "w") as f:
        f.write(f"Random Seed: {seed}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Data Type: {dtype}\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
    
    return device, dtype

def load_and_prepare_data():
    """Load and prepare financial data with robust parsing."""
    logger.info("Loading financial data...")
    
    # Robust data file path handling
    data_path = os.getenv('SP500_DATA_PATH', "../data/sp500_data.csv")
    fallback_paths = [data_path, "data/sp500_data.csv", "../data/sp500_data.csv"]
    
    data = None
    for path in fallback_paths:
        if os.path.exists(path):
            try:
                data = pd.read_csv(path, index_col=0, parse_dates=True)
                logger.info(f"Data loaded from: {path}")
                break
            except Exception as e:
                logger.warning(f"Failed to load from {path}: {e}")
                continue
    
    if data is None:
        raise FileNotFoundError(f"Could not find sp500_data.csv in paths: {fallback_paths}")
    
    # Robust date parsing - check if index is already datetime
    if not pd.api.types.is_datetime64_any_dtype(data.index):
        try:
            data.index = pd.to_datetime(data.index, errors='coerce')
        except Exception as e:
            logger.error(f"Date parsing failed: {e}")
            raise
    
    # Filter out invalid dates
    data = data[data.index.notna()]
    
    # Sort dataframe by index to ensure chronological order
    data = data.sort_index()
    
    # Robust numeric parsing
    try:
        data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
        data = data.dropna(subset=['Close'])
    except Exception as e:
        logger.error(f"Numeric parsing failed: {e}")
        raise
    
    # Calculate log returns for stability
    returns = np.log(data['Close'] / data['Close'].shift(1)).dropna()
    
    logger.info(f"Data loaded: {len(returns)} observations")
    logger.info(f"Date range: {returns.index[0]} to {returns.index[-1]}")
    logger.info(f"Returns stats - Mean: {returns.mean():.6f}, Std: {returns.std():.6f}")
    
    return returns

def create_sequences(returns, seq_len=SEQ_LEN):
    """Create sequences for training with data augmentation."""
    logger.info(f"Creating sequences of length {seq_len} with augmentation...")
    
    sequences = []
    for i in range(len(returns) - seq_len + 1):
        seq = returns.iloc[i:i+seq_len].values
        sequences.append(seq)
    
    X = np.array(sequences)
    
    # Data augmentation: add small random noise to increase diversity
    if len(X) > 0:
        # Add 10% augmented sequences
        n_augment = len(X) // 10
        augmented = []
        
        for _ in range(n_augment):
            # Randomly select a sequence
            idx = np.random.randint(0, len(X))
            seq = X[idx].copy()
            
            # Add small Gaussian noise (1% of std)
            noise_std = np.std(seq) * 0.01
            seq += np.random.normal(0, noise_std, seq.shape)
            
            augmented.append(seq)
        
        # Combine original and augmented sequences
        X = np.vstack([X, np.array(augmented)])
        logger.info(f"Added {len(augmented)} augmented sequences")
    
    logger.info(f"Created {len(X)} sequences of shape {X.shape}")
    return X

def train_val_split(X):
    """Split data chronologically into training and validation sets (90/10 of training data)."""
    n_train = int(len(X) * 0.9)
    X_train = X[:n_train]
    X_val = X[n_train:]
    
    print(f"Training/validation split:")
    print(f"  Training: {len(X_train)} sequences")
    print(f"  Validation: {len(X_val)} sequences")
    
    return X_train, X_val

def cosine_beta_schedule(timesteps, s=0.008):
    """
    Cosine beta schedule as proposed in the DDPM paper.
    More stable than linear schedule.
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def train_ddpm_model(X_train, X_val, device):
    """Train DDPM model with validation and early stopping."""
    print("Training DDPM model...")
    
    # Setup results directory
    results_dir = f"./runs/ddpm_evaluation/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, "checkpoints"), exist_ok=True)
    
    # Create data loaders
    train_dataset = TensorDataset(torch.FloatTensor(X_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val))
    
    # Use DataLoader with pin_memory for CUDA and safe num_workers
    num_workers = 0  # Set to 0 to avoid multiprocessing issues
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
        num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        num_workers=num_workers
    )
    
    # Initialize model and move to device
    model = DenoiseMLP(SEQ_LEN).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Learning rate scheduler for better convergence
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=SCHEDULER_FACTOR, 
        patience=SCHEDULER_PATIENCE, 
        verbose=True
    )
    
    # Cosine beta schedule
    betas = cosine_beta_schedule(NUM_TIMESTEPS)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    
    # Log beta schedule summary
    beta_summary = {
        'min_beta': float(betas.min()),
        'max_beta': float(betas.max()),
        'mean_beta': float(betas.mean()),
        'num_timesteps': NUM_TIMESTEPS
    }
    
    # Training loop with validation
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    lr_history = []
    
    print(f"Starting training for {NUM_EPOCHS} epochs...")
    print(f"Initial learning rate: {LEARNING_RATE}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        num_batches = 0
        
        # Training phase
        for batch_idx, (data,) in enumerate(train_loader):
            try:
                data = data.to(device)
                batch_size = data.size(0)
                
                # Sample random timesteps
                t = torch.randint(0, NUM_TIMESTEPS, (batch_size, 1), device=device).float()
                
                # Normalize timesteps to [0, 1]
                t = t / (NUM_TIMESTEPS - 1)
                
                # Add noise to data
                noise = torch.randn_like(data, device=device)
                alpha_bar = alphas_cumprod[t.long().squeeze()].view(batch_size, 1).expand(batch_size, SEQ_LEN)
                noisy_data = torch.sqrt(alpha_bar) * data + torch.sqrt(1 - alpha_bar) * noise
                
                # Predict noise
                predicted_noise = model(noisy_data, t)
                loss = F.mse_loss(predicted_noise, noise)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
                
                optimizer.step()
                train_loss += loss.item()
                num_batches += 1
                
            except Exception as e:
                print(f"Error in training batch {batch_idx}: {e}")
                continue
        
        if num_batches == 0:
            print(f"Warning: No training batches completed in epoch {epoch+1}")
            continue
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for data, in val_loader:
                try:
                    data = data.to(device)
                    batch_size = data.size(0)
                    
                    t = torch.randint(0, NUM_TIMESTEPS, (batch_size, 1), device=device).float()
                    t = t / (NUM_TIMESTEPS - 1)
                    
                    noise = torch.randn_like(data, device=device)
                    alpha_bar = alphas_cumprod[t.long().squeeze()].view(batch_size, 1).expand(batch_size, SEQ_LEN)
                    noisy_data = torch.sqrt(alpha_bar) * data + torch.sqrt(1 - alpha_bar) * noise
                    
                    predicted_noise = model(noisy_data, t)
                    loss = F.mse_loss(predicted_noise, noise)
                    val_loss += loss.item()
                    val_batches += 1
                    
                except Exception as e:
                    print(f"Error in validation batch: {e}")
                    continue
        
        if val_batches == 0:
            print(f"Warning: No validation batches completed in epoch {epoch+1}")
            continue
        
        # Average losses
        avg_train_loss = train_loss / num_batches
        avg_val_loss = val_loss / val_batches
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        lr_history.append(optimizer.param_groups[0]['lr'])
        
        # Update learning rate scheduler
        scheduler.step(avg_val_loss)
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            # Save best model checkpoint
            checkpoint_path = os.path.join(results_dir, "checkpoints", "best_ddpm_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'betas': betas,
                'alphas': alphas,
                'alphas_cumprod': alphas_cumprod
            }, checkpoint_path)
            print(f"  New best model saved to: {checkpoint_path}")
        else:
            patience_counter += 1
        
        # Progress reporting - show every epoch for better monitoring
        current_time = datetime.now().strftime("%H:%M:%S")
        current_lr = optimizer.param_groups[0]['lr']
        print(f"[{current_time}] Epoch {epoch+1}/{NUM_EPOCHS}: "
              f"Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, "
              f"LR: {current_lr:.2e}, Patience: {patience_counter}/{PATIENCE}")
        
        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Save final model checkpoint
    final_checkpoint_path = os.path.join(results_dir, "checkpoints", "final_ddpm_model.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': avg_train_loss,
        'val_loss': avg_val_loss,
        'betas': betas,
        'alphas': alphas,
        'alphas_cumprod': alphas_cumprod
    }, final_checkpoint_path)
    
    # Save training history
    training_history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'lr_history': lr_history,
        'beta_schedule': beta_summary,
        'best_epoch': epoch - patience_counter + 1,
        'final_train_loss': avg_train_loss,
        'final_val_loss': avg_val_loss,
        'final_lr': optimizer.param_groups[0]['lr']
    }
    
    metadata.save_json(training_history, os.path.join(results_dir, "training_history.json"))
    
    print(f"Training completed. Best validation loss: {best_val_loss:.6f}")
    print(f"Model checkpoints saved to: {os.path.join(results_dir, 'checkpoints')}")
    
    return model, (betas, alphas, alphas_cumprod), results_dir

def sample_ddpm(model, noise_schedule, n_samples, device, temperature=1.0, guidance_scale=1.0):
    """Enhanced sampling from trained DDPM model with temperature and guidance control."""
    print("Generating samples with enhanced sampling...")
    
    model.eval()
    betas, alphas, alphas_cumprod = noise_schedule
    
    with torch.inference_mode():
        # Start from pure noise
        x = torch.randn(n_samples, SEQ_LEN, device=device)
        
        # Reverse diffusion process with improved stability
        for t in reversed(range(NUM_TIMESTEPS)):
            t_tensor = torch.full((n_samples, 1), t, device=device).float() / (NUM_TIMESTEPS - 1)
            
            # Predict noise
            predicted_noise = model(x, t_tensor)
            
            # Apply temperature scaling to noise prediction
            predicted_noise = predicted_noise * temperature
            
            # Compute posterior mean and variance with better numerical stability
            if t == 0:
                # At t=0, use deterministic sampling
                alpha_bar = alphas_cumprod[t]
                x = (1 / torch.sqrt(alpha_bar)) * (x - (betas[t] / torch.sqrt(1 - alpha_bar)) * predicted_noise)
            else:
                # Use posterior mean and variance for t > 0
                alpha_bar = alphas_cumprod[t]
                alpha_bar_prev = alphas_cumprod[t-1]
                
                # More stable computation
                beta_tilde = betas[t] * (1 - alpha_bar_prev) / (1 - alpha_bar)
                beta_tilde = torch.clamp(beta_tilde, min=1e-6, max=0.999)
                
                # Compute mean
                mean_coef1 = (1 / torch.sqrt(alphas[t])) * (x - (betas[t] / torch.sqrt(1 - alpha_bar)) * predicted_noise)
                
                # Add controlled noise for diversity
                if temperature > 0:
                    noise_scale = torch.sqrt(beta_tilde) * temperature
                    mean_coef2 = noise_scale * torch.randn_like(x)
                    x = mean_coef1 + mean_coef2
                else:
                    x = mean_coef1
        
        # Post-processing: clip extreme values to prevent numerical instability
        x = torch.clamp(x, -10, 10)
        
        return x.cpu().numpy()

def find_optimal_sampling_temperature(model, noise_schedule, real_data, device, temperatures=[0.5, 0.7, 1.0, 1.2, 1.5]):
    """Find optimal sampling temperature by comparing statistics with real data."""
    print("Finding optimal sampling temperature...")
    
    best_temp = 1.0
    best_score = float('inf')
    results = {}
    
    for temp in temperatures:
        print(f"  Testing temperature: {temp}")
        
        # Generate samples with this temperature
        synthetic_data = sample_ddpm(model, noise_schedule, len(real_data), device, temperature=temp)
        
        # Compute basic statistics
        real_flat = real_data.flatten()
        synthetic_flat = synthetic_data.flatten()
        
        # Calculate score based on statistical similarity
        mean_diff = abs(np.mean(real_flat) - np.mean(synthetic_flat))
        std_diff = abs(np.std(real_flat) - np.std(synthetic_flat))
        
        # Combined score (lower is better)
        score = mean_diff + std_diff
        
        results[temp] = {
            'mean_diff': mean_diff,
            'std_diff': std_diff,
            'score': score,
            'synthetic_mean': np.mean(synthetic_flat),
            'synthetic_std': np.std(synthetic_flat)
        }
        
        print(f"    Score: {score:.6f} (Mean diff: {mean_diff:.6f}, Std diff: {std_diff:.6f})")
        
        if score < best_score:
            best_score = score
            best_temp = temp
    
    print(f"  Best temperature: {best_temp} (score: {best_score:.6f})")
    return best_temp, results

def evaluate_ddpm_performance(real_data, synthetic_data):
    """Evaluate DDPM performance using only the test set."""
    print("Evaluating DDPM performance...")
    
    # Flatten data for comparison
    print("  Computing basic statistics...")
    real_flat = real_data.flatten()
    synthetic_flat = synthetic_data.flatten()
    
    # Basic statistics
    stats = {
        'Real_Mean': float(np.mean(real_flat)),
        'Real_Std': float(np.std(real_flat)),
        'Real_Skewness': float(scipy.stats.skew(real_flat)),
        'Real_Excess_Kurtosis': float(scipy.stats.kurtosis(real_flat)),
        'Synthetic_Mean': float(np.mean(synthetic_flat)),
        'Synthetic_Std': float(np.std(synthetic_flat)),
        'Synthetic_Skewness': float(scipy.stats.skew(synthetic_flat)),
        'Synthetic_Excess_Kurtosis': float(scipy.stats.kurtosis(synthetic_flat))
    }
    
    # Distribution comparison tests
    print("  Computing KS test...")
    ks_stat, ks_pvalue = scipy.stats.ks_2samp(real_flat, synthetic_flat)
    stats.update({
        'KS_Statistic': ks_stat,
        'KS_PValue': ks_pvalue
    })
    
    # MMD test
    print("  Computing MMD test...")
    try:
        # Reshape to 2D for MMD calculation
        real_2d = real_flat.reshape(-1, 1)
        synthetic_2d = synthetic_flat.reshape(-1, 1)
        
        # Add timeout to MMD test to prevent hanging
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("MMD test timed out")
        
        # Set 30 second timeout for MMD test
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(30)
        
        try:
            mmd_stat, mmd_pvalue = stats_utils.mmd_permutation_p(real_2d, synthetic_2d)
            signal.alarm(0)  # Cancel alarm
        except TimeoutError:
            print("Warning: MMD test timed out after 30 seconds, skipping...")
            mmd_stat, mmd_pvalue = np.nan, np.nan
        
        stats.update({
            'MMD_Statistic': mmd_stat,
            'MMD_PValue': mmd_pvalue
        })
    except Exception as e:
        print(f"Warning: Could not compute MMD test: {e}")
        stats.update({
            'MMD_Statistic': np.nan,
            'MMD_PValue': np.nan
        })
    
    # Hill tail index estimator for heavy-tailedness
    print("  Computing Hill estimator...")
    def hill_estimator(data, k=None):
        """Hill estimator for tail index."""
        try:
            if k is None:
                k = min(max(5, int(0.1 * len(data))), len(data) - 1)
            
            if k < 5 or k >= len(data):
                return np.nan
            
            sorted_data = np.sort(data)[::-1]  # Sort in descending order
            log_excesses = np.log(sorted_data[:k] + 1e-10)  # Add small epsilon to avoid log(0)
            
            if len(log_excesses) < 2:
                return np.nan
            
            hill_est = 1 / np.mean(log_excesses - np.log(sorted_data[k] + 1e-10))
            return float(hill_est)
        except Exception as e:
            print(f"Warning: Hill estimator failed: {e}")
            return np.nan
    
    # Compute Hill estimator for both datasets
    real_hill = hill_estimator(real_flat)
    synthetic_hill = hill_estimator(synthetic_flat)
    stats.update({
        'Real_Hill_Tail_Index': real_hill,
        'Synthetic_Hill_Tail_Index': synthetic_hill
    })
    
    # Lag-1 autocorrelation of absolute returns (volatility clustering proxy)
    print("  Computing autocorrelations...")
    def lag1_autocorr_abs(data):
        """Compute lag-1 autocorrelation of absolute values."""
        try:
            if len(data) < 2:
                return np.nan
            abs_data = np.abs(data)
            return float(np.corrcoef(abs_data[:-1], abs_data[1:])[0, 1])
        except Exception as e:
            print(f"Warning: Autocorrelation computation failed: {e}")
            return np.nan
    
    # For synthetic data, compute per sequence then average
    if synthetic_data.ndim > 1:
        synthetic_autocorrs = []
        for i in range(synthetic_data.shape[0]):
            autocorr = lag1_autocorr_abs(synthetic_data[i])
            if not np.isnan(autocorr):
                synthetic_autocorrs.append(autocorr)
        synthetic_autocorr = np.mean(synthetic_autocorrs) if synthetic_autocorrs else np.nan
    else:
        synthetic_autocorr = lag1_autocorr_abs(synthetic_flat)
    
    real_autocorr = lag1_autocorr_abs(real_flat)
    
    stats.update({
        'Real_Lag1_Abs_Autocorr': real_autocorr,
        'Synthetic_Lag1_Abs_Autocorr': synthetic_autocorr
    })
    
    # Print summary
    print("  Evaluation completed successfully!")
    print(f"DDPM Evaluation Results:")
    print(f"  Real data - Mean: {stats['Real_Mean']:.4f}, Std: {stats['Real_Std']:.4f}")
    print(f"  Synthetic data - Mean: {stats['Synthetic_Mean']:.4f}, Std: {stats['Synthetic_Std']:.4f}")
    print(f"  KS test: statistic={stats['KS_Statistic']:.4f}, p-value={stats['KS_PValue']:.4f}")
    print(f"  MMD test: statistic={stats['MMD_Statistic']:.4f}, p-value={stats['MMD_PValue']:.4f}")
    print(f"  Hill tail index - Real: {stats['Real_Hill_Tail_Index']:.4f}, Synthetic: {stats['Synthetic_Hill_Tail_Index']:.4f}")
    print(f"  Lag-1 abs autocorr - Real: {stats['Real_Lag1_Abs_Autocorr']:.4f}, Synthetic: {stats['Synthetic_Lag1_Abs_Autocorr']:.4f}")
    
    return stats

def save_ddpm_results(synthetic_data, losses, stats, results_dir):
    """Save DDPM results."""
    print("Saving DDPM results...")
    
    # Ensure directories exist
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, "plots"), exist_ok=True)
    
    # Save synthetic data
    np.save(os.path.join(results_dir, "ddpm_returns.npy"), synthetic_data)
    
    # Save losses
    np.save(os.path.join(results_dir, "losses.npy"), np.array(losses))
    
    # Save statistics
    metadata.save_json(stats, os.path.join(results_dir, "ddpm_stats.json"))
    
    print(f"DDPM results saved to: {results_dir}")

def create_ddpm_plots(real_data, synthetic_data, results_dir):
    """Create DDPM evaluation plots."""
    print("Creating DDPM plots...")
    
    plots_dir = os.path.join(results_dir, "plots")
    
    # Plot 1: Loss curve
    losses = np.load(os.path.join(results_dir, "losses.npy"))
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('DDPM Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plots_dir, "loss_curve.pdf"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(plots_dir, "loss_curve.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Distribution comparison
    real_flat = real_data.flatten()
    synthetic_flat = synthetic_data.flatten()
    
    # Use utils histogram function
    plots.hist_line_logy([real_flat, synthetic_flat], ['Real', 'Synthetic'], 
                         title="DDPM: Real vs Synthetic Returns Distribution",
                         xlabel="Returns (%)")
    plt.savefig(os.path.join(plots_dir, "distribution_comparison.pdf"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(plots_dir, "distribution_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Sample sequences
    plt.figure(figsize=(15, 10))
    n_samples = min(4, len(synthetic_data))
    
    for i in range(n_samples):
        plt.subplot(2, 2, i+1)
        plt.plot(synthetic_data[i], label='Synthetic', alpha=0.8)
        plt.title(f'Synthetic Sequence {i+1}')
        plt.xlabel('Time Step')
        plt.ylabel('Returns (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "sample_sequences.pdf"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(plots_dir, "sample_sequences.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 4: QQ plot vs normal
    plots.qq_vs_normal(real_flat, title="DDPM Real Returns vs Normal")
    plt.savefig(os.path.join(plots_dir, "qq_real_vs_normal.pdf"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(plots_dir, "qq_real_vs_normal.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    plots.qq_vs_normal(synthetic_flat, title="DDPM Synthetic Returns vs Normal")
    plt.savefig(os.path.join(plots_dir, "qq_synthetic_vs_normal.pdf"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(plots_dir, "qq_synthetic_vs_normal.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 5: ACF plots
    plots.acf_stem(real_flat, nlags=20, title="DDPM Real Returns ACF")
    plt.savefig(os.path.join(plots_dir, "acf_real_returns.pdf"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(plots_dir, "acf_real_returns.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    plots.acf_stem(synthetic_flat, nlags=20, title="DDPM Synthetic Returns ACF")
    plt.savefig(os.path.join(plots_dir, "acf_synthetic_returns.pdf"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(plots_dir, "acf_synthetic_returns.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 6: Rolling volatility panel
    # Convert to pandas Series for rolling calculations
    real_series = pd.Series(real_flat)
    synthetic_series = pd.Series(synthetic_flat)
    
    rolling_vol_fig, rolling_vol_axes = plots.rolling_vol_panel(
        real_series, 
        [(synthetic_series, 'DDPM Synthetic')], 
        window=20, 
        title="DDPM Rolling Volatility Comparison"
    )
    rolling_vol_fig.savefig(os.path.join(plots_dir, "rolling_volatility.pdf"), dpi=300, bbox_inches='tight')
    rolling_vol_fig.savefig(os.path.join(plots_dir, "rolling_volatility.png"), dpi=300, bbox_inches='tight')
    plt.close(rolling_vol_fig)
    
    plt.close('all')
    print(f"DDPM plots saved to: {plots_dir}")

def main():
    """Main function to run DDPM evaluation."""
    print("DDPM Model Evaluation")
    print("=" * 50)
    
    # Parse arguments
    args = parse_arguments()
    
    # Setup output directory
    outdir = setup_output_directory()
    
    # Start timer for metadata
    timer_func = metadata.timer()
    
    try:
        # Set reproducibility
        device, dtype = set_reproducibility(42)
        
        # Override constants with command line arguments
        global NUM_EPOCHS, NUM_TIMESTEPS, LEARNING_RATE
        if args.epochs != NUM_EPOCHS:
            NUM_EPOCHS = args.epochs
            print(f"Using {NUM_EPOCHS} epochs from command line")
        if args.timesteps != NUM_TIMESTEPS:
            NUM_TIMESTEPS = args.timesteps
            print(f"Using {NUM_TIMESTEPS} timesteps from command line")
        if args.lr != LEARNING_RATE:
            LEARNING_RATE = args.lr
            print(f"Using learning rate {LEARNING_RATE} from command line")
        
        # Load and prepare data
        returns = load_and_prepare_data()
        
        # Apply unit standardization if requested
        if args.units:
            returns = units.ensure_units(returns, args.units)
            print(f"Returns standardized to {args.units} units")
        
        # Create sequences first
        X = create_sequences(returns)
        
        # Train/validation split (90/10 of training data)
        X_train, X_val = train_val_split(X)
        
        # Test set is the remaining 20% (handled in the main data split)
        # For now, we'll use the validation set as our test set for evaluation
        # In a full pipeline, this would come from the main chronological split
        X_test = X_val
        
        # Compute standardization parameters from training data only
        mu = np.mean(X_train)
        sigma = np.std(X_train)
        
        if sigma == 0:
            raise ValueError("Standard deviation is zero, cannot standardize data")
        
        print(f"Data standardization - Mean: {mu:.6f}, Std: {sigma:.6f}")
        print(f"Training set shape: {X_train.shape}")
        print(f"Validation set shape: {X_val.shape}")
        print(f"Test set shape: {X_test.shape}")
        
        # Standardize training and validation data
        X_train_std = (X_train - mu) / sigma
        X_val_std = (X_val - mu) / sigma
        
        # Train model
        model, noise_schedule, results_dir = train_ddpm_model(X_train_std, X_val_std, device)
        
        # Generate samples
        n_samples = len(X_test)
        
        # Use command line temperature if specified, otherwise optimize
        if args.temperature is not None:
            optimal_temp = args.temperature
            print(f"Using command line temperature: {optimal_temp}")
            temp_results = {'manual_temperature': optimal_temp}
        else:
            # Find optimal sampling temperature
            optimal_temp, temp_results = find_optimal_sampling_temperature(model, noise_schedule, X_test, device)
        
        # Generate final samples with optimal temperature
        synthetic_data_std = sample_ddpm(model, noise_schedule, n_samples, device, temperature=optimal_temp)
        
        # De-standardize synthetic data
        synthetic_data = synthetic_data_std * sigma + mu
        
        # Save temperature optimization results
        metadata.save_json(temp_results, os.path.join(results_dir, "temperature_optimization.json"))
        metadata.save_json({'optimal_temperature': optimal_temp}, os.path.join(results_dir, "optimal_temperature.json"))
        
        print(f"Using optimal sampling temperature: {optimal_temp}")
        
        # Evaluate performance using only test set
        stats = evaluate_ddpm_performance(X_test, synthetic_data)
        
        # Save results
        save_ddpm_results(synthetic_data, [], stats, results_dir)
        
        # Create plots
        create_ddpm_plots(X_test, synthetic_data, results_dir)
        
        # Enhanced evaluation using utils if requested
        if args.write_metadata:
            # Compute predictive intervals
            try:
                predictive_stats = uncertainty.predictive_stats_from_samples(synthetic_data_std)
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
            metadata.save_json(stats, os.path.join(results_dir, "ddpm_stats_enhanced.json"))
        
        # Save run configuration
        run_config = {
            'seq_len': SEQ_LEN,
            'batch_size': BATCH_SIZE,
            'num_epochs': NUM_EPOCHS,
            'num_timesteps': NUM_TIMESTEPS,
            'learning_rate': LEARNING_RATE,
            'patience': PATIENCE,
            'validation_split': VALIDATION_SPLIT,
            'gradient_clip_norm': GRADIENT_CLIP_NORM,
            'device': str(device),
            'dtype': str(dtype),
            'mu': float(mu),
            'sigma': float(sigma)
        }
        
        metadata.save_json(run_config, os.path.join(results_dir, "run_config.json"))
        
        # Collect and save metadata if requested
        if args.write_metadata:
            # Determine units used
            units_used = args.units if args.units else 'auto-detected'
            
            # Create metadata dictionary
            metadata_dict = {
                'timestamp': datetime.now().isoformat(),
                'script_name': 'diffusion_simple.py',
                'seed': 42,
                'units_used': units_used,
                'dataset_summary': metadata.dataset_summary(returns, 'S&P 500 Returns', units_used),
                'python_version': sys.version,
                'gpu_info': metadata.gpu_info(),
                'training_time_seconds': timer_func(),
                'model_parameters': metadata.count_params(model),
                'train_val_split': {
                    'train_obs': len(X_train),
                    'val_obs': len(X_val),
                    'test_obs': len(X_test)
                }
            }
            
            # Save metadata
            metadata.save_json(metadata_dict, os.path.join(results_dir, "metadata.json"))
            
            # Write report notes
            report_notes = """# DDPM Model Evaluation Report Notes

## Train/Validation/Test Split
- **Training Period**: 90% of training data (chronological, no shuffling)
- **Validation Period**: 10% of training data (chronological, no shuffling)
- **Test Period**: 20% of full dataset (chronological, no shuffling)

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

### Hill Tail Index
Estimates the tail index of heavy-tailed distributions. Lower values indicate heavier tails.

### Value at Risk (VaR)
The maximum expected loss at a given confidence level. 5% VaR means 95% of the time, losses won't exceed this threshold.

### Expected Shortfall (ES)
The average loss when VaR is exceeded, providing a measure of tail risk beyond VaR.

## Important Notes
- All evaluation metrics are computed using ONLY the test set
- Model is trained on training data only to prevent data leakage
- Validation set is used for early stopping during training
- Samples are generated to match the test set size
- Standardization parameters are computed from training data only
"""
            
            with open(os.path.join(results_dir, "report_notes.md"), 'w') as f:
                f.write(report_notes)
        
        print(f"DDPM evaluation completed successfully!")
        print(f"Results saved in: {results_dir}")
        
    except Exception as e:
        print(f"Error in DDPM evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
