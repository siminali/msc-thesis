#!/usr/bin/env python3
"""
Comprehensive Training Pipeline for Three DDPM Models
Trains zero-conditioned, explicit-conditioned, and LLM-conditioned models with shared config

Author: Simin Ali
Thesis: Diffusion Models in Generative AI for Financial Data Synthesis and Risk Management
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.cuda.amp as amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use Agg backend to avoid GUI stalls
import matplotlib.pyplot as plt
import json
import os
import argparse
import subprocess
import hashlib
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import warnings
import pickle
import time
warnings.filterwarnings('ignore')

# Import model implementations
from explicit_cond_ddpm import (
    load_and_prepare_data, 
    create_conditioning_vectors, 
    create_sequences,
    ExplicitConditioningDDPM,
    ExplicitConditioningTrainer,
    EMAModel
)

# Conditional LLM imports
try:
    from llm_conditioned_diffusion_refactored import (
        NewsDataLoader,
        LLMConditionedDiffusion,
        LLMDiffusionTrainer,
        ControllabilityProbe,
        create_time_based_splits
    )
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    print("Warning: LLM model dependencies not available. Install with: pip install sentence-transformers")

# Global constants
DEFAULT_CONFIG = {
    # Data parameters
    'data_path': None,  # Will be auto-detected
    'seq_len': 60,
    'vol_window': 20,
    
    # Training parameters
    'epochs': 100,
    'batch_size': 64,
    'lr': 1e-3,
    'num_timesteps': 1000,
    'beta_schedule': 'cosine',
    'sampler': 'ddim',
    'sample_steps': 50,
    'patience': 10,
    'grad_clip': 1.0,
    
    # CFG parameters (for explicit and LLM models)
    'cfg_p': 0.1,
    'cfg_scale': 7.5,
    
    # LLM-specific parameters
    'encoder_name': 'sentence-transformers/all-MiniLM-L6-v2',
    'target_embedding_dim': 64,
    'cache_dir': 'cache/news_embeddings',
    
    # System parameters
    'seed': 42,
    'device': 'auto',
    'results_dir': 'results',
    
    # Performance optimization parameters
    'amp': False,
    'compile': False,
    'workers': 0,
    'prefetch': 2,
    'pin_memory': False,
    'persistent_workers': False,
    'grad_accum': 1,
    'channels_last': False,
    'fast_dev_run': False,
    'ddp': False,
    'make_plots': True,
    'fast_sampling': False,
    'warmup_epochs': 0,
    'ema_decay': 0.999
}

def get_git_hash():
    """Get current git hash if available."""
    try:
        result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                              capture_output=True, text=True, cwd=os.getcwd())
        if result.returncode == 0:
            return result.stdout.strip()[:8]
    except:
        pass
    return None

def set_determinism(seed, device, deterministic=True):
    """Set full determinism for reproducible training."""
    # Global seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # CUDA-specific settings
    if device == 'cuda' or (device == 'auto' and torch.cuda.is_available()):
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            print(f"CUDA determinism enabled with seed {seed}")
        else:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
            print(f"CUDA benchmark enabled for speed (non-deterministic)")
    
    # Set float32 matmul precision for speed
    if hasattr(torch, 'set_float32_matmul_precision'):
        torch.set_float32_matmul_precision("high")
    
    print(f"Global determinism set with seed {seed}")

def setup_ddp(rank, world_size):
    """Setup distributed training."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_ddp():
    """Cleanup distributed training."""
    torch.distributed.destroy_process_group()

def load_and_prepare_data_shared(config):
    """Load and prepare data with shared preprocessing."""
    print("Loading financial data...")
    
    # Robust data file path handling
    data_path = config.get('data_path')
    if data_path is None:
        fallback_paths = [
            "data/sp500_data.csv",
            "../data/sp500_data.csv",
            "../../data/sp500_data.csv"
        ]
        
        for path in fallback_paths:
            if os.path.exists(path):
                data_path = path
                break
        
        if data_path is None:
            raise FileNotFoundError(f"Could not find sp500_data.csv in fallback paths")
    
    # Load data
    data = pd.read_csv(data_path, index_col=0, parse_dates=True)
    data.index = pd.to_datetime(data.index)
    
    # Calculate log returns
    returns = np.log(data['Close'] / data['Close'].shift(1)).dropna()
    
    print(f"Loaded {len(returns)} days of return data")
    print(f"Date range: {returns.index[0]} to {returns.index[-1]}")
    
    return returns

def create_zero_conditioned_model(config):
    """Create zero-conditioned DDPM (unconditional baseline)."""
    print("Creating zero-conditioned DDPM...")
    
    # Create sequences
    returns = load_and_prepare_data_shared(config)
    X = create_sequences(returns, config['seq_len'])
    
    # Split data
    num_sequences = len(X)
    train_split_idx = int(num_sequences * 0.8)  # 80% train, 20% val
    
    X_train = X[:train_split_idx]
    X_val = X[train_split_idx:]
    
    # Create zero conditioning (no explicit conditioning)
    conditioning_dim = 5  # Same as explicit model for consistency
    zero_conditioning_train = np.zeros((len(X_train), conditioning_dim))
    zero_conditioning_val = np.zeros((len(X_val), conditioning_dim))
    
    # Initialize model
    model = ExplicitConditioningDDPM(
        sequence_length=config['seq_len'],
        conditioning_dim=conditioning_dim,
        hidden_dim=128  # Default hidden dim
    )
    
    # Initialize trainer (no CFG for zero model)
    trainer = ExplicitConditioningTrainer(
        model, 
        num_timesteps=config['num_timesteps'], 
        beta_schedule=config['beta_schedule'], 
        device=config['device'],
        grad_clip=config['grad_clip'],
        cfg_p=0.0,  # No conditioning dropout for zero-conditioned
        amp=config['amp'],
        compile=config['compile']
    )
    
    return model, trainer, X_train, X_val, zero_conditioning_train, zero_conditioning_val, returns

def create_explicit_conditioned_model(config):
    """Create explicitly-conditioned DDPM with regime + volatility."""
    print("Creating explicitly-conditioned DDPM...")
    
    # Load data
    returns = load_and_prepare_data_shared(config)
    
    # Create conditioning vectors
    conditioning_vectors, regime_labels, metadata = create_conditioning_vectors(
        returns, config['seq_len'], config['vol_window'], 0.2
    )
    
    # Create sequences
    X = create_sequences(returns, config['seq_len'])
    
    # Split data
    num_sequences = len(X)
    train_split_idx = int(num_sequences * 0.8)
    
    X_train = X[:train_split_idx]
    X_val = X[train_split_idx:]
    conditioning_train = conditioning_vectors[:train_split_idx]
    conditioning_val = conditioning_vectors[train_split_idx:]
    
    # Initialize model
    model = ExplicitConditioningDDPM(
        sequence_length=config['seq_len'],
        conditioning_dim=conditioning_train.shape[1],
        hidden_dim=128
    )
    
    # Initialize trainer
    trainer = ExplicitConditioningTrainer(
        model, 
        num_timesteps=config['num_timesteps'], 
        beta_schedule=config['beta_schedule'], 
        device=config['device'],
        grad_clip=config['grad_clip'],
        cfg_p=config['cfg_p'],
        amp=config['amp'],
        compile=config['compile']
    )
    
    return model, trainer, X_train, X_val, conditioning_train, conditioning_val, returns, metadata

def create_llm_conditioned_model(config):
    """Create LLM-conditioned DDPM with news embeddings."""
    if not LLM_AVAILABLE:
        raise ImportError("LLM model dependencies not available. Install with: pip install sentence-transformers")
    
    print("Creating LLM-conditioned DDPM...")
    
    # Load data
    returns = load_and_prepare_data_shared(config)
    
    # Create time-based splits (60/20/20)
    X_train, X_val, X_test, train_dates, val_dates, test_dates = create_time_based_splits(
        returns, config['seq_len']
    )
    
    # Initialize news data loader
    news_loader = NewsDataLoader(cache_dir=config['cache_dir'])
    
    # Create conditioning vectors for each split
    print("Creating conditioning vectors for training split...")
    conditioning_train = news_loader.create_conditioning_vectors(
        train_dates, config['seq_len'], config['target_embedding_dim']
    )
    
    print("Creating conditioning vectors for validation split...")
    conditioning_val = news_loader.create_conditioning_vectors(
        val_dates, config['seq_len'], config['target_embedding_dim']
    )
    
    print("Creating conditioning vectors for test split...")
    conditioning_test = news_loader.create_conditioning_vectors(
        test_dates, config['seq_len'], config['target_embedding_dim']
    )
    
    # Initialize model
    model = LLMConditionedDiffusion(
        sequence_length=config['seq_len'],
        conditioning_dim=conditioning_train.shape[1],
        hidden_dim=128
    )
    
    # Initialize trainer
    trainer = LLMDiffusionTrainer(
        model, 
        num_timesteps=config['num_timesteps'], 
        beta_schedule=config['beta_schedule'], 
        device=config['device'],
        grad_clip=config['grad_clip'],
        cfg_p=config['cfg_p'],
        amp=config['amp'],
        compile=config['compile']
    )
    
    # Store additional metadata
    llm_metadata = {
        'encoder_name': config['encoder_name'],
        'target_embedding_dim': config['target_embedding_dim'],
        'train_dates': [train_dates[0].isoformat(), train_dates[-1].isoformat()],
        'val_dates': [val_dates[0].isoformat(), val_dates[-1].isoformat()],
        'test_dates': [test_dates[0].isoformat(), test_dates[-1].isoformat()],
        'leakage_controls': 'Strict temporal alignment - no look-ahead, forward-fill only'
    }
    
    return model, trainer, X_train, X_val, conditioning_train, conditioning_val, returns, llm_metadata

def precompute_conditioning_tensors(X_train, X_val, conditioning_train, conditioning_val, config):
    """Precompute and cache conditioning tensors for speed."""
    cache_dir = Path(config['results_dir']) / 'cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Cache file paths
    train_cache = cache_dir / 'train_conditioning.pt'
    val_cache = cache_dir / 'val_conditioning.pt'
    
    # Check if cache exists
    if train_cache.exists() and val_cache.exists():
        print("Loading cached conditioning tensors...")
        conditioning_train_tensor = torch.load(train_cache, map_location='cpu')
        conditioning_val_tensor = torch.load(val_cache, map_location='cpu')
    else:
        print("Precomputing conditioning tensors...")
        # Convert to tensors and cache
        conditioning_train_tensor = torch.tensor(conditioning_train, dtype=torch.float32)
        conditioning_val_tensor = torch.tensor(conditioning_val, dtype=torch.float32)
        
        torch.save(conditioning_train_tensor, train_cache)
        torch.save(conditioning_val_tensor, val_cache)
    
    return conditioning_train_tensor, conditioning_val_tensor

def train_model(model, trainer, X_train, X_val, conditioning_train, conditioning_val, config, model_name):
    """Train a model with optimized training loop."""
    print(f"Training {model_name}...")
    
    # Precompute conditioning tensors (skip for LLM model)
    if model_name == 'llm_conditioned':
        # LLM model uses its own conditioning tensors directly
        conditioning_train_tensor = torch.tensor(conditioning_train, dtype=torch.float32)
        conditioning_val_tensor = torch.tensor(conditioning_val, dtype=torch.float32)
    else:
        # Other models use the precomputed tensors
        conditioning_train_tensor, conditioning_val_tensor = precompute_conditioning_tensors(
            X_train, X_val, conditioning_train, conditioning_val, config
        )
    
    # Debug tensor shapes
    print(f"Debug - X_train shape: {X_train.shape}, conditioning_train shape: {conditioning_train_tensor.shape}")
    print(f"Debug - X_val shape: {X_val.shape}, conditioning_val shape: {conditioning_val_tensor.shape}")
    
    # Prepare data with optimizations
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        conditioning_train_tensor
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        conditioning_val_tensor
    )
    
    # Optimized DataLoader
    train_loader_kwargs = {
        'batch_size': config['batch_size'],
        'shuffle': True,
        'num_workers': config['workers'],
        'pin_memory': config['pin_memory'] and config['workers'] > 0,
        'drop_last': True
    }
    
    # Add multiprocessing options only when workers > 0
    if config['workers'] > 0:
        train_loader_kwargs.update({
            'persistent_workers': config['persistent_workers'],
            'prefetch_factor': config['prefetch']
        })
    
    train_loader = DataLoader(train_dataset, **train_loader_kwargs)
    
    val_loader_kwargs = {
        'batch_size': config['batch_size'],
        'shuffle': False,
        'num_workers': config['workers'],
        'pin_memory': config['pin_memory'] and config['workers'] > 0
    }
    
    if config['workers'] > 0:
        val_loader_kwargs.update({
            'persistent_workers': config['persistent_workers'],
            'prefetch_factor': config['prefetch']
        })
    
    val_loader = DataLoader(val_dataset, **val_loader_kwargs)
    
    # Training setup with optimizations
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config['lr'], 
        weight_decay=1e-5,
        fused=True if hasattr(optim.AdamW, 'fused') else False
    )
    
    # Learning rate scheduling with warmup
    total_steps = len(train_loader) * config['epochs']
    warmup_steps = int(total_steps * config['warmup_epochs'] / config['epochs'])
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return 0.5 * (1 + np.cos(np.pi * (step - warmup_steps) / (total_steps - warmup_steps)))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Mixed precision setup
    scaler = amp.GradScaler() if config['amp'] else None
    
    # EMA setup
    ema_model = None
    if config.get('ema_decay', 0) > 0:
        ema_model = EMAModel(model, decay=config['ema_decay'])
    
    # Check for existing checkpoint
    checkpoint_dir = Path(config['results_dir']) / model_name / config['run_id'] / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    best_checkpoint_path = checkpoint_dir / 'best_model.pth'
    start_epoch = 0
    
    if best_checkpoint_path.exists():
        print(f"Loading existing checkpoint from {best_checkpoint_path}")
        checkpoint = torch.load(best_checkpoint_path, map_location=config['device'])
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['val_loss']
        print(f"Resuming from epoch {start_epoch} with best val loss: {best_val_loss:.6f}")
    else:
        best_val_loss = float('inf')
    
    # Training loop with optimizations
    train_losses = []
    val_losses = []
    patience_counter = 0
    
    # Fast dev run check
    if config['fast_dev_run']:
        config['epochs'] = 1
        print("Fast dev run: training only 1 epoch")
    
    # Progress tracking with percentages
    total_epochs = config['epochs']
    epoch_pbar = tqdm(range(start_epoch, total_epochs), desc=f"Training {model_name}", 
                      bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    
    for epoch in epoch_pbar:
        epoch_start_time = time.time()
        
        # Training
        model.train()
        epoch_train_losses = []
        total_batches = len(train_loader)
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{total_epochs-1} Training", 
                          leave=False, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        
        for batch_idx, (batch_x, batch_conditioning) in enumerate(train_pbar):
            # Move to device with non_blocking
            batch_x = batch_x.to(config['device'], non_blocking=True)
            batch_conditioning = batch_conditioning.to(config['device'], non_blocking=True)
            
            # Set channels last if requested
            if config['channels_last']:
                batch_x = batch_x.to(memory_format=torch.channels_last)
            
            # Training step with gradient accumulation
            loss = 0
            for acc_step in range(config['grad_accum']):
                # Get batch slice for gradient accumulation
                start_idx = acc_step * (config['batch_size'] // config['grad_accum'])
                end_idx = (acc_step + 1) * (config['batch_size'] // config['grad_accum'])
                
                if start_idx >= batch_x.shape[0]:
                    break
                
                x_slice = batch_x[start_idx:end_idx]
                c_slice = batch_conditioning[start_idx:end_idx]
                
                # Forward pass with mixed precision
                if config['amp']:
                    with amp.autocast():
                        step_loss = trainer.train_step(x_slice, c_slice, optimizer, scaler)
                else:
                    step_loss = trainer.train_step(x_slice, c_slice, optimizer)
                
                loss += step_loss / config['grad_accum']
            
            epoch_train_losses.append(loss)
            
            # Update progress bar
            train_pbar.set_postfix({'loss': f'{loss:.6f}'})
            
            # Fast dev run: limit batches
            if config['fast_dev_run'] and batch_idx >= 2:
                break
        
        # Validation
        model.eval()
        epoch_val_losses = []
        total_val_batches = len(val_loader)
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{total_epochs-1} Validation", 
                           leave=False, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
            
            for batch_x, batch_conditioning in val_pbar:
                batch_x = batch_x.to(config['device'], non_blocking=True)
                batch_conditioning = batch_conditioning.to(config['device'], non_blocking=True)
                
                if config['channels_last']:
                    batch_x = batch_x.to(memory_format=torch.channels_last)
                
                # Sample random timesteps for validation
                batch_size = batch_x.shape[0]
                t = torch.randint(0, config['num_timesteps'], (batch_size,), device=config['device'])
                
                # Add noise
                x_noisy, noise = trainer.add_noise(batch_x, t)
                
                # Predict noise with mixed precision
                t_normalized = t.float() / config['num_timesteps']
                
                if config['amp']:
                    with amp.autocast():
                        predicted_noise = model(x_noisy, t_normalized.unsqueeze(-1), batch_conditioning)
                else:
                    predicted_noise = model(x_noisy, t_normalized.unsqueeze(-1), batch_conditioning)
                
                # Compute loss
                loss = nn.functional.mse_loss(predicted_noise, noise)
                epoch_val_losses.append(loss.item())
                
                val_pbar.set_postfix({'val_loss': f'{loss.item():.6f}'})
                
                # Fast dev run: limit validation batches
                if config['fast_dev_run'] and len(epoch_val_losses) >= 2:
                    break
        
        # Record losses
        avg_train_loss = np.mean(epoch_train_losses)
        avg_val_loss = np.mean(epoch_val_losses)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        # Learning rate scheduling
        scheduler.step()
        
        # Update EMA
        if ema_model:
            ema_model.update()
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            # Save best model
            save_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
            }
            if ema_model and hasattr(ema_model, 'state_dict'):
                save_dict['ema_state_dict'] = ema_model.state_dict()
            
            torch.save(save_dict, best_checkpoint_path)
        else:
            patience_counter += 1
        
        # Update epoch progress bar with percentage
        epoch_time = time.time() - epoch_start_time
        epoch_progress = (epoch + 1) / total_epochs * 100
        epoch_pbar.set_postfix({
            'Progress': f'{epoch_progress:.1f}%',
            'Train Loss': f'{avg_train_loss:.6f}',
            'Val Loss': f'{avg_val_loss:.6f}',
            'Time': f'{epoch_time:.1f}s'
        })
        
        # Print detailed progress every 10 epochs
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}/{total_epochs-1} ({epoch_progress:.1f}%): Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, Time: {epoch_time:.1f}s")
        
        # Early stopping
        if patience_counter >= config['patience']:
            print(f"Early stopping at epoch {epoch}")
            break
        
        # Fast dev run: only one epoch
        if config['fast_dev_run']:
            break
    
    # Save final model
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': avg_val_loss,
    }, checkpoint_dir / 'final_model.pth')
    
    # Save training history
    history = {
        'epochs': list(range(len(train_losses))),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_epoch': epoch - patience_counter
    }
    
    return model, trainer, history, best_val_loss

def save_training_results(config, model_name, history, best_val_loss, metadata=None):
    """Save training results and metadata."""
    results_dir = Path(config['results_dir']) / model_name / config['run_id']
    
    # Create subdirectories
    (results_dir / 'figures').mkdir(parents=True, exist_ok=True)
    (results_dir / 'tables').mkdir(parents=True, exist_ok=True)
    
    # Save training history
    history_df = pd.DataFrame({
        'epoch': history['epochs'],
        'train_loss': history['train_losses'],
        'val_loss': history['val_losses']
    })
    
    history_df.to_csv(results_dir / 'training_history.csv', index=False)
    
    with open(results_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save metadata
    run_metadata = {
        'run_id': config['run_id'],
        'timestamp': datetime.now().isoformat(),
        'model_name': model_name,
        'args': config,
        'seed': config['seed'],
        'device': config['device'],
        'git_hash': get_git_hash(),
        'final_val_loss': best_val_loss,
        'best_epoch': history['best_epoch']
    }
    
    if metadata:
        run_metadata['model_metadata'] = metadata
    
    with open(results_dir / 'metadata.json', 'w') as f:
        json.dump(run_metadata, f, indent=2)
    
    # Save model-specific metadata
    if metadata and model_name == 'explicit_conditioned':
        with open(results_dir / 'conditioning_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
    
    # Create training plots only if requested
    if config['make_plots']:
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(history['epochs'], history['train_losses'], label='Training Loss')
        plt.plot(history['epochs'], history['val_losses'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'{model_name.replace("_", " ").title()} Training Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(history['epochs'], history['val_losses'], 'r-', label='Validation Loss')
        plt.axvline(x=history['best_epoch'], color='g', linestyle='--', label=f'Best Epoch ({history["best_epoch"]})')
        plt.xlabel('Epoch')
        plt.ylabel('Validation Loss')
        plt.title('Validation Loss with Best Epoch')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(results_dir / 'figures' / 'training_curves.pdf', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Create README
    create_run_readme(config, model_name, best_val_loss, results_dir)

def create_run_readme(config, model_name, best_val_loss, results_dir):
    """Create README for the run."""
    model_display = model_name.replace("_", " ").title()
    
    readme_content = f"""# {model_display} Training Run Summary

## Run Information
- **Run ID**: {config['run_id']}
- **Model**: {model_display}
- **Timestamp**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Device**: {config['device']}

## Key Parameters
- **Sequence Length**: {config['seq_len']}
- **Volatility Window**: {config['vol_window']}
- **Epochs**: {config['epochs']}
- **Batch Size**: {config['batch_size']}
- **Learning Rate**: {config['lr']}
- **Patience**: {config['patience']}
- **Seed**: {config['seed']}

## Training Results
- **Best Validation Loss**: {best_val_loss:.6f}
- **Best Epoch**: {config.get('best_epoch', 'N/A')}

## Generated Files
- `checkpoints/` - Model checkpoints
- `figures/` - Training curves and visualizations
- `tables/` - Evaluation tables
- `training_history.csv` - Training progress data
- `metadata.json` - Complete run metadata

## Next Steps
Run the evaluation pipeline using:
```bash
python src/evaluate_model.py --model-name {model_name} --run-id {config['run_id']}
```
"""
    
    with open(results_dir / 'README_RUN.md', 'w') as f:
        f.write(readme_content)

def train_llm_with_probe(config, model, trainer, X_train, X_val, conditioning_train, conditioning_val, vol_window):
    """Train LLM model and fit controllability probe."""
    print("Training LLM model...")
    
    # Train the model first
    model, trainer, history, best_val_loss = train_model(
        model, trainer, X_train, X_val, conditioning_train, conditioning_val, config, 'llm_conditioned'
    )
    
    # Fit controllability probe
    print("Training controllability probe...")
    probe = ControllabilityProbe()
    
    # Compute realized volatilities and trends for training data
    train_volatilities = []
    train_trends = []
    
    for i in range(len(X_train)):
        seq_returns = X_train[i, 0, :]  # Remove channel dimension
        
        # Compute volatility
        rolling_stds = []
        for j in range(len(seq_returns) - vol_window + 1):
            rolling_stds.append(np.std(seq_returns[j:j+vol_window], ddof=1))
        vol = np.mean(rolling_stds[-vol_window:])
        train_volatilities.append(vol)
        
        # Compute trend
        trend = seq_returns.sum()
        train_trends.append(trend)
    
    train_volatilities = np.array(train_volatilities)
    train_trends = np.array(train_trends)
    
    # Train probe
    probe.train(conditioning_train, train_volatilities, train_trends)
    
    # Save probe and scaler
    results_dir = Path(config['results_dir']) / 'llm_conditioned' / config['run_id']
    with open(results_dir / 'controllability_probe.pkl', 'wb') as f:
        pickle.dump(probe, f)
    
    # Save probe diagnostics
    probe_diagnostics = {
        'volatility_model_coef': probe.volatility_model.coef_.tolist(),
        'volatility_model_intercept': float(probe.volatility_model.intercept_),
        'trend_model_coef': probe.trend_model.coef_.tolist(),
        'trend_model_intercept': float(probe.trend_model.intercept_),
        'vol_scaler_mean': float(probe.vol_scaler.mean_),
        'vol_scaler_scale': float(probe.vol_scaler.scale_),
        'num_training_samples': len(train_volatilities)
    }
    
    with open(results_dir / 'probe_diagnostics.json', 'w') as f:
        json.dump(probe_diagnostics, f, indent=2)
    
    print("Controllability probe trained and saved")
    return model, trainer, history, best_val_loss

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train all three DDPM models')
    
    # Model selection
    parser.add_argument('--models', nargs='+', 
                       choices=['zero', 'explicit', 'llm', 'all'],
                       default=['all'], help='Models to train')
    
    # Configuration file
    parser.add_argument('--config', type=str, help='Path to config JSON file')
    
    # CLI overrides
    parser.add_argument('--seq-len', type=int, help='Sequence length')
    parser.add_argument('--vol-window', type=int, help='Volatility window')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, help='Batch size')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], help='Device')
    parser.add_argument('--results-dir', type=str, help='Results directory')
    
    # Performance optimization flags
    parser.add_argument('--amp', action='store_true', help='Enable automatic mixed precision')
    parser.add_argument('--compile', action='store_true', help='Enable torch.compile')
    parser.add_argument('--workers', type=int, default=0, help='Number of DataLoader workers')
    parser.add_argument('--prefetch', type=int, default=2, help='DataLoader prefetch factor')
    parser.add_argument('--pin-memory', action='store_true', help='Enable pin memory')
    parser.add_argument('--persistent-workers', action='store_true', help='Enable persistent workers')
    parser.add_argument('--grad-accum', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--channels-last', action='store_true', help='Use channels last memory format')
    parser.add_argument('--fast-dev-run', action='store_true', help='Fast development run (1 epoch, few batches)')
    parser.add_argument('--ddp', action='store_true', help='Enable distributed training')
    parser.add_argument('--make-plots', action='store_true', default=True, help='Generate training plots')
    parser.add_argument('--no-make-plots', action='store_false', dest='make_plots', help='Disable training plots')
    parser.add_argument('--fast-sampling', action='store_true', help='Use fast sampling (20 steps)')
    parser.add_argument('--warmup-epochs', type=float, default=0, help='Warmup epochs as fraction of total')
    
    args = parser.parse_args()
    
    # Load configuration
    config = DEFAULT_CONFIG.copy()
    
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config.update(json.load(f))
    
    # Apply CLI overrides
    for key, value in vars(args).items():
        if value is not None and key != 'models' and key != 'config':
            config[key] = value
    
    # Set run ID
    config['run_id'] = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Set device
    if config['device'] == 'auto':
        config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Fast sampling override
    if config['fast_sampling']:
        config['sample_steps'] = 20
        print("Fast sampling enabled: using 20 steps")
    
    # Set determinism (respect determinism flag for speed vs reproducibility)
    deterministic = not (config['amp'] or config['compile'] or config['ddp'])
    set_determinism(config['seed'], config['device'], deterministic)
    
    print(f"Training pipeline initialized:")
    print(f"  Models: {args.models}")
    print(f"  Device: {config['device']}")
    print(f"  Results: {config['results_dir']}")
    print(f"  Run ID: {config['run_id']}")
    print(f"  Performance: AMP={config['amp']}, Compile={config['compile']}, Workers={config['workers']}")
    
    # Calculate total models to train
    models_to_train = []
    if 'all' in args.models or 'zero' in args.models:
        models_to_train.append('zero_conditioned')
    if 'all' in args.models or 'explicit' in args.models:
        models_to_train.append('explicit_conditioned')
    if 'all' in args.models or 'llm' in args.models:
        if LLM_AVAILABLE:
            models_to_train.append('llm_conditioned')
    
    total_models = len(models_to_train)
    print(f"\nTotal models to train: {total_models}")
    
    # Train selected models with progress tracking
    for model_idx, model_name in enumerate(models_to_train, 1):
        print(f"\n" + "="*60)
        print(f"TRAINING {model_name.upper().replace('_', ' ')} ({model_idx}/{total_models})")
        print("="*60)
        
        if model_name == 'zero_conditioned':
            model, trainer, X_train, X_val, cond_train, cond_val, returns = create_zero_conditioned_model(config)
            model, trainer, history, best_val_loss = train_model(
                model, trainer, X_train, X_val, cond_train, cond_val, config, 'zero_conditioned'
            )
            save_training_results(config, 'zero_conditioned', history, best_val_loss)
        
        elif model_name == 'explicit_conditioned':
            model, trainer, X_train, X_val, cond_train, cond_val, returns, metadata = create_explicit_conditioned_model(config)
            model, trainer, history, best_val_loss = train_model(
                model, trainer, X_train, X_val, cond_train, cond_val, config, 'explicit_conditioned'
            )
            save_training_results(config, 'explicit_conditioned', history, best_val_loss, metadata)
        
        elif model_name == 'llm_conditioned':
            model, trainer, X_train, X_val, cond_train, cond_val, returns, metadata = create_llm_conditioned_model(config)
            model, trainer, history, best_val_loss = train_llm_with_probe(
                config, model, trainer, X_train, X_val, cond_train, cond_val, config['vol_window']
            )
            save_training_results(config, 'llm_conditioned', history, best_val_loss, metadata)
        
        print(f"✅ {model_name.upper().replace('_', ' ')} completed ({model_idx}/{total_models})")
    
    # Handle case where LLM is skipped
    if 'all' in args.models or 'llm' in args.models:
        if not LLM_AVAILABLE:
            print("\n" + "="*60)
            print("LLM-CONDITIONED MODEL SKIPPED")
            print("="*60)
            print("LLM model dependencies not available.")
            print("Install with: pip install sentence-transformers")
            print("="*60)
    
    print(f"\nTraining completed successfully!")
    print(f"Results saved in: {config['results_dir']}")
    print(f"Run ID: {config['run_id']}")

if __name__ == "__main__":
    main()
