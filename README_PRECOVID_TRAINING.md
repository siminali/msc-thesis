# Pre-COVID Training Runner Documentation

## Overview

This repository contains a standalone training system for financial diffusion models trained exclusively on pre-COVID data (2010-2019). The system reuses existing model classes without modification and implements proper causality, determinism, and comprehensive logging.

## Key Features

- **Causality**: No look-ahead bias - all conditioning uses only past information
- **Determinism**: Fixed seeds and deterministic PyTorch operations
- **Versioning Safety**: Automatic V2 fallback if original file exists
- **Graceful Error Handling**: Missing LLM embeddings fallback to zero conditioning
- **Comprehensive Checkpointing**: Full metadata and conditioning specifications saved

## Models Supported

### 1. Zero Conditioning (`zero`)
- **Description**: Basic unconditional DDPM
- **Conditioning**: None (unconditional generation)
- **Use Case**: Baseline model for comparison

### 2. Explicit Conditioning (`explicit`)
- **Description**: Regime classification + financial features
- **Conditioning**: 
  - 4 one-hot regime features: [Up-Low, Up-High, Down-Low, Down-High]
  - Normalized causal 20-day volatility (z_vol)
  - Normalized causal 60-day trend
- **Features**:
  - Volatility: 20-day rolling standard deviation (causal)
  - Trend: 60-day cumulative return sum (causal)
  - Regime classification based on cumulative return and volatility threshold

### 3. LLM Conditioning (`llm`)
- **Description**: Daily embeddings with PCA reduction
- **Conditioning**:
  - DistilBERT embeddings (768-dim) reduced to 32-dim via PCA
  - PCA fitted only on training data (≤ 2019-12-31)
  - Date-aligned embeddings for each sequence
- **Fallback**: Automatically falls back to zero conditioning if LLM processing fails

## Installation & Setup

### Prerequisites
```bash
pip install torch numpy pandas scikit-learn transformers tqdm
```

### File Structure
```
Thesis Coding/
├── train_precovid_models.py       # Main training script
├── train_precovid_models_v2.py    # Enhanced version (auto-selected)
├── run_precovid_training.sh       # Shell script for easy execution
├── src/                           # Existing model classes
│   ├── benchmarking models/
│   │   └── diffusion_simple.py   # Zero conditioning model
│   └── novelty models/
│       ├── explicit_cond_ddpm.py # Explicit conditioning model
│       └── llm_conditioned_diffusion.py # LLM conditioning model
└── data/
    └── sp500_data.csv            # S&P 500 data
```

## Usage

### Quick Start
```bash
# Run all models with default settings
./run_precovid_training.sh

# Or directly with Python
python train_precovid_models.py --models all
```

### Command Line Options

#### Model Selection
```bash
# Train all models
python train_precovid_models.py --models all

# Train specific models
python train_precovid_models.py --models zero explicit
python train_precovid_models.py --models llm

# Train only zero conditioning (fastest)
python train_precovid_models.py --models zero
```

#### Training Parameters
```bash
# Custom training settings
python train_precovid_models.py \
    --models all \
    --epochs 200 \
    --batch-size 64 \
    --lr 1e-4 \
    --hidden-dim 256

# Quick training for testing
python train_precovid_models.py \
    --models zero \
    --epochs 10 \
    --batch-size 16
```

#### Conditioning Parameters
```bash
# Custom explicit conditioning windows
python train_precovid_models.py \
    --models explicit \
    --vol-window 30 \
    --trend-window 90

# Custom LLM PCA components
python train_precovid_models.py \
    --models llm \
    --pca-components 64
```

#### System & Error Handling
```bash
# Force CPU usage
python train_precovid_models.py --device cpu

# Skip models that fail (continue training others)
python train_precovid_models.py --skip-on-error

# Disable LLM fallback (fail if LLM conditioning fails)
python train_precovid_models.py --no-llm-fallback

# Custom random seed
python train_precovid_models.py --seed 123
```

### Full Parameter Reference
```bash
python train_precovid_models.py \
    --models zero explicit llm \     # Models to train
    --epochs 100 \                   # Training epochs
    --batch-size 32 \                # Batch size
    --lr 1e-3 \                      # Learning rate
    --num-timesteps 1000 \           # Diffusion timesteps
    --hidden-dim 128 \               # Model hidden dimension
    --pca-components 32 \            # LLM PCA components
    --vol-window 20 \                # Volatility window (explicit)
    --trend-window 60 \              # Trend window (explicit)
    --device auto \                  # Device (auto/cpu/cuda)
    --seed 42 \                      # Random seed
    --checkpoint-dir checkpoints/precovid \  # Checkpoint directory
    --skip-on-error \                # Skip failed models
    --llm-fallback                   # Enable LLM fallback
```

## Output Structure

### Checkpoint Directory Structure
```
checkpoints/precovid/
├── zero/
│   └── 20100101-20191231/
│       ├── best.pt                 # Best model checkpoint
│       ├── last.pt                 # Latest model checkpoint
│       ├── meta.json               # Model and training metadata
│       └── conditioning_spec.json  # Conditioning specification
├── explicit/
│   └── 20100101-20191231/
│       ├── best.pt
│       ├── last.pt
│       ├── meta.json
│       └── conditioning_spec.json
└── llm/
    └── 20100101-20191231/
        ├── best.pt
        ├── last.pt
        ├── meta.json
        ├── conditioning_spec.json
        └── pca_model.pkl           # Fitted PCA model
```

### Checkpoint Contents

#### `best.pt` / `last.pt`
```python
{
    'epoch': 95,
    'model_state_dict': {...},      # PyTorch model state
    'train_loss': 0.0234,
    'val_loss': 0.0267,
    'timestamp': '2024-01-15T10:30:00',
    'betas': tensor(...),           # Diffusion schedule (if applicable)
    'alphas': tensor(...),
    'alphas_cumprod': tensor(...)
}
```

#### `meta.json`
```json
{
    "model_info": {
        "type": "explicit",
        "conditioning_dim": 6,
        "sequence_length": 60,
        "parameter_count": 125000,
        "trainable_parameters": 125000
    },
    "training_info": {
        "epoch": 95,
        "train_loss": 0.0234,
        "val_loss": 0.0267,
        "is_best": true
    },
    "system_info": {
        "device": "cuda",
        "torch_version": "2.0.0",
        "seed": 42,
        "timestamp": "2024-01-15T10:30:00"
    },
    "data_info": {
        "train_period": "2010-01-01 to 2019-12-31",
        "val_period": "2019-07-01 to 2019-12-31",
        "train_sequences": 2400,
        "val_sequences": 120
    }
}
```

#### `conditioning_spec.json`
```json
{
    "type": "explicit",
    "description": "Explicit conditioning with regime classification + volatility + trend",
    "conditioning_dim": 6,
    "features": {
        "regime_onehot": {
            "description": "4 one-hot features for Up-Low, Up-High, Down-Low, Down-High",
            "indices": [0, 1, 2, 3]
        },
        "z_vol": {
            "description": "Normalized causal 20-day volatility",
            "index": 4,
            "scaler_mean": 0.0123,
            "scaler_scale": 0.0045
        },
        "trend": {
            "description": "Normalized causal 60-day cumulative return",
            "index": 5,
            "scaler_mean": 0.0067,
            "scaler_scale": 0.0234
        }
    },
    "vol_threshold": 0.0156,
    "vol_window": 20,
    "trend_window": 60
}
```

## Data Requirements

### S&P 500 Data Format
The system expects a CSV file with the following structure:
```csv
Date,Close
2010-01-04,1132.99
2010-01-05,1136.52
...
```

### Data Locations
The system searches for data in the following order:
1. `/Users/siminali/Desktop/Thesis Coding/data/sp500_data.csv`
2. `data/sp500_data.csv`
3. `../data/sp500_data.csv`

## Training Process

### Data Splits
- **Training Period**: 2010-01-01 to 2019-12-31 (full pre-COVID period)
- **Validation Period**: 2019-07-01 to 2019-12-31 (last 6 months for early stopping)
- **Sequence Length**: 60 trading days
- **No Look-ahead**: All conditioning uses only causally available information

### Conditioning Implementation

#### Zero Conditioning
- No conditioning vectors
- Pure unconditional DDPM
- Baseline for comparison

#### Explicit Conditioning
- **Regime Classification**: Based on cumulative return (Up/Down) and volatility level (Low/High)
- **Volatility Feature**: 20-day rolling standard deviation, normalized using training data statistics
- **Trend Feature**: 60-day cumulative return sum, normalized using training data statistics
- **Causality**: All features computed using only past data relative to the sequence

#### LLM Conditioning
- **Text Generation**: Daily market sentiment data using DistilBERT
- **Embedding**: 768-dimensional embeddings per trading day
- **PCA Reduction**: Fitted only on training data (≤ 2019-12-31), reduced to 32 dimensions
- **Alignment**: Date-aligned embeddings for each 60-day sequence
- **Fallback**: Automatic fallback to zero conditioning if LLM processing fails

### Determinism
- Fixed random seeds (default: 42)
- Deterministic CUDA operations when available
- Reproducible data splits and preprocessing
- Consistent initialization across runs

## Troubleshooting

### Common Issues

#### 1. Missing Data File
```
Error: Could not find sp500_data.csv
```
**Solution**: Ensure S&P 500 data is in one of the expected locations.

#### 2. CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```
**Solutions**:
```bash
# Reduce batch size
python train_precovid_models.py --batch-size 16

# Use CPU
python train_precovid_models.py --device cpu

# Reduce hidden dimension
python train_precovid_models.py --hidden-dim 64
```

#### 3. LLM Import Errors
```
ImportError: No module named 'transformers'
```
**Solution**:
```bash
pip install transformers
```

#### 4. Model Import Errors
```
ImportError: No module named 'explicit_cond_ddpm'
```
**Solution**: Ensure all model files are in the correct src/ directories.

### Debug Mode
```bash
# Enable verbose logging
python train_precovid_models.py --models zero --epochs 1 --batch-size 4
```

### Validation
```bash
# Quick test run
python train_precovid_models.py --models zero --epochs 2 --batch-size 8
```

## Loading Trained Models

### Loading Checkpoints
```python
import torch
import json

# Load model checkpoint
checkpoint = torch.load('checkpoints/precovid/explicit/20100101-20191231/best.pt')
model_state = checkpoint['model_state_dict']

# Load metadata
with open('checkpoints/precovid/explicit/20100101-20191231/meta.json', 'r') as f:
    metadata = json.load(f)

# Load conditioning specification
with open('checkpoints/precovid/explicit/20100101-20191231/conditioning_spec.json', 'r') as f:
    conditioning_spec = json.load(f)
```

### Model Reconstruction
```python
# For explicit model
from src.novelty.models.explicit_cond_ddpm import ExplicitConditioningDDPM

model = ExplicitConditioningDDPM(
    sequence_length=metadata['model_info']['sequence_length'],
    conditioning_dim=metadata['model_info']['conditioning_dim'],
    hidden_dim=128  # Or from your training args
)
model.load_state_dict(checkpoint['model_state_dict'])
```

## Performance Notes

### Training Times (Approximate)
- **Zero Model**: ~30 minutes (100 epochs, batch_size=32, GPU)
- **Explicit Model**: ~45 minutes (100 epochs, batch_size=32, GPU)
- **LLM Model**: ~2 hours (100 epochs, batch_size=32, GPU, including embedding generation)

### Memory Usage
- **GPU Memory**: ~2-4 GB (depending on batch size and model)
- **RAM**: ~4-8 GB (for data loading and preprocessing)

### Recommendations
- Use GPU for training if available
- Start with smaller epochs for testing (--epochs 10)
- Use --skip-on-error for robustness
- Enable LLM fallback (default) for reliability

## License and Citation

This code is part of the MSc thesis "Diffusion Models in Generative AI for Financial Data Synthesis and Risk Management" by Simin Ali, Imperial College London.

## Contact

For questions or issues, please refer to the main thesis documentation or contact the author through the appropriate academic channels.
