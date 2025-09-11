# Pre-COVID Training System - Usage Examples

## Quick Start Examples

### 1. Basic Usage - All Models
```bash
# Train all models with default settings (recommended for production)
python train_precovid_simplified.py --models all --epochs 100

# Quick test run with all models (for testing)
python train_precovid_simplified.py --models all --epochs 5 --batch-size 8
```

### 2. Individual Model Training
```bash
# Train only zero conditioning model (fastest)
python train_precovid_simplified.py --models zero --epochs 50

# Train only explicit conditioning model
python train_precovid_simplified.py --models explicit --epochs 50

# Train only LLM conditioning model (takes longest)
python train_precovid_simplified.py --models llm --epochs 50
```

### 3. Production Training Settings
```bash
# High-quality training for research/production
python train_precovid_simplified.py \
    --models all \
    --epochs 200 \
    --batch-size 64 \
    --lr 1e-4 \
    --hidden-dim 256 \
    --num-timesteps 1000

# GPU accelerated training (if CUDA available)
python train_precovid_simplified.py \
    --models all \
    --epochs 100 \
    --batch-size 128 \
    --device cuda
```

### 4. Custom Conditioning Parameters
```bash
# Custom explicit conditioning windows
python train_precovid_simplified.py \
    --models explicit \
    --vol-window 30 \
    --trend-window 90 \
    --epochs 50

# Custom LLM PCA components
python train_precovid_simplified.py \
    --models llm \
    --pca-components 64 \
    --epochs 50

# Reduced PCA for faster training/testing
python train_precovid_simplified.py \
    --models llm \
    --pca-components 16 \
    --epochs 10
```

### 5. Development and Testing
```bash
# Quick development test (minimal training)
python train_precovid_simplified.py \
    --models zero \
    --epochs 2 \
    --batch-size 8

# Error-resistant training (skip models that fail)
python train_precovid_simplified.py \
    --models all \
    --epochs 50 \
    --skip-on-error

# CPU-only training (when GPU unavailable)
python train_precovid_simplified.py \
    --models zero explicit \
    --epochs 20 \
    --device cpu \
    --batch-size 16
```

### 6. Reproducibility and Seeds
```bash
# Reproducible training with custom seed
python train_precovid_simplified.py \
    --models all \
    --epochs 50 \
    --seed 123

# Reproducible comparison runs
python train_precovid_simplified.py --models zero --seed 42 --epochs 50
python train_precovid_simplified.py --models explicit --seed 42 --epochs 50
python train_precovid_simplified.py --models llm --seed 42 --epochs 50
```

## Expected Outputs

### Directory Structure After Training
```
checkpoints/precovid/
├── zero/
│   └── 20100101-20191231/
│       ├── best.pt                 # Best model (lowest validation loss)
│       ├── last.pt                 # Final model
│       ├── meta.json               # Model and training metadata
│       └── conditioning_spec.json  # Zero conditioning specification
├── explicit/
│   └── 20100101-20191231/
│       ├── best.pt
│       ├── last.pt
│       ├── meta.json
│       └── conditioning_spec.json  # Explicit conditioning specification
└── llm/
    └── 20100101-20191231/
        ├── best.pt
        ├── last.pt
        ├── meta.json
        ├── conditioning_spec.json  # LLM conditioning specification
        └── pca_model.pkl           # PCA transformation model
```

### Training Output Example
```
================================================================================
Pre-COVID Training Runner - Simplified Version (No Utils Dependencies)
================================================================================
2025-08-30 17:18:31,297 - INFO - Set deterministic mode with seed: 42
2025-08-30 17:18:31,297 - INFO - Using device: cpu
2025-08-30 17:18:31,297 - INFO - Models to train: ['zero']
2025-08-30 17:18:31,297 - INFO - Training parameters: epochs=50, batch_size=32, lr=0.001
2025-08-30 17:18:31,297 - INFO - Loading S&P 500 data...
2025-08-30 17:18:31,305 - INFO - Data loaded from: /path/to/sp500_data.csv
2025-08-30 17:18:31,308 - INFO - Training data: 2515 observations (2010-01-05 to 2019-12-31)
2025-08-30 17:18:31,308 - INFO - Validation data: 128 observations (2019-07-01 to 2019-12-31)
2025-08-30 17:18:31,309 - INFO - Training stats - Mean: 0.000417, Std: 0.009317
...
============================================================
TRAINING SUMMARY
============================================================
Successfully trained 1 models:
  ✓ zero
Checkpoints saved to: checkpoints/precovid
Pre-COVID training completed successfully!
```

## Performance Expectations

### Training Times (Approximate)
- **Zero Model**: ~5-15 minutes (50 epochs, CPU)
- **Explicit Model**: ~10-25 minutes (50 epochs, CPU)  
- **LLM Model**: ~20-45 minutes (50 epochs, CPU, 32 PCA components)

### Memory Usage
- **RAM**: ~2-4 GB
- **GPU Memory**: ~1-3 GB (if using CUDA)

### Model Sizes
- **Zero Model**: ~2.7 MB checkpoint file
- **Explicit Model**: ~2.8 MB checkpoint file
- **LLM Model**: ~2.9 MB checkpoint file + ~100 KB PCA model

## Loading and Using Trained Models

### Basic Loading Example
```python
import torch
import json
import pickle

# Load zero conditioning model
checkpoint = torch.load('checkpoints/precovid/zero/20100101-20191231/best.pt')
model_state = checkpoint['model_state_dict']

# Load metadata and specs
with open('checkpoints/precovid/zero/20100101-20191231/meta.json', 'r') as f:
    metadata = json.load(f)

with open('checkpoints/precovid/zero/20100101-20191231/conditioning_spec.json', 'r') as f:
    conditioning_spec = json.load(f)

print(f"Model type: {conditioning_spec['type']}")
print(f"Parameters: {metadata['model_info']['parameter_count']}")
print(f"Best validation loss: {metadata['training_info']['val_loss']}")
```

### Advanced Model Reconstruction
```python
from train_precovid_simplified import DenoiseMLP, ExplicitConditioningDDPM, ConditionedDiffusionModel

# Load explicit conditioning model
checkpoint = torch.load('checkpoints/precovid/explicit/20100101-20191231/best.pt')
metadata = ...  # Load metadata as above

# Reconstruct model
model = ExplicitConditioningDDPM(
    sequence_length=metadata['model_info']['sequence_length'],
    conditioning_dim=metadata['model_info']['conditioning_dim'],
    hidden_dim=128  # From your training args
)

# Load trained weights
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()  # Set to evaluation mode

print("Model loaded and ready for inference!")
```

### LLM Model with PCA
```python
# Load LLM model with PCA
checkpoint = torch.load('checkpoints/precovid/llm/20100101-20191231/best.pt')
with open('checkpoints/precovid/llm/20100101-20191231/pca_model.pkl', 'rb') as f:
    pca = pickle.load(f)

# Now you have both the trained model and the fitted PCA transformer
print(f"PCA components: {pca.n_components_}")
print(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")
```

## Troubleshooting Common Issues

### Issue 1: Out of Memory
```bash
# Reduce batch size
python train_precovid_simplified.py --batch-size 8

# Use CPU instead of GPU
python train_precovid_simplified.py --device cpu
```

### Issue 2: Training Too Slow
```bash
# Reduce epochs for testing
python train_precovid_simplified.py --epochs 10

# Reduce PCA components for LLM
python train_precovid_simplified.py --models llm --pca-components 16

# Train only one model at a time
python train_precovid_simplified.py --models zero
```

### Issue 3: Missing Data File
```bash
# Check data paths - the script looks for sp500_data.csv in:
# 1. /Users/siminali/Desktop/Thesis Coding/data/sp500_data.csv
# 2. data/sp500_data.csv  
# 3. ../data/sp500_data.csv

# Ensure the CSV has columns: Date (index), Close
```

### Issue 4: JSON Serialization Errors
The simplified version handles this automatically, but if you see JSON errors:
- All numpy types are converted to Python native types
- Metadata is carefully constructed to be JSON-serializable

## Integration with Existing Workflow

This pre-COVID training system is designed to integrate seamlessly with your existing model evaluation pipeline:

1. **Train models** using this system
2. **Load checkpoints** in your evaluation scripts
3. **Generate samples** using the loaded models
4. **Compare performance** against full-period models

The checkpoint format is compatible with your existing evaluation framework and can be used directly in downstream analysis.

## Next Steps

After training, you can:
1. Load the models and generate synthetic data
2. Evaluate model performance on post-2019 data
3. Compare pre-COVID vs full-period model behavior
4. Use the models for risk assessment and scenario analysis

The pre-COVID training provides a clean baseline for understanding how financial models perform when trained exclusively on "normal" market conditions, excluding the COVID-19 market disruption.
