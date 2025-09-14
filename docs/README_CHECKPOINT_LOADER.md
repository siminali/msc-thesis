# Checkpoint Loader & Sampler

A comprehensive utility for loading trained diffusion model checkpoints and generating samples with automatic model reconstruction and conditioning detection.

## Features

- **Automatic Model Reconstruction**: Detects model type from metadata and rebuilds the exact model architecture
- **Runtime Conditioning Detection**: Inspects method signatures to detect conditioning support
- **Conditioning Provider Reconstruction**: Rebuilds conditioning from saved specifications without refitting
- **Graceful Error Handling**: Robust execution with detailed error logging and manifests
- **Multiple Model Support**: Works with zero, explicit, and LLM conditioning models
- **Versioning Safe**: Creates `_v2.py` files to avoid overwriting existing utilities

## Quick Start

### Basic Usage

```bash
# Load a checkpoint and generate samples
python checkpoint_loader_sampler.py \
    --checkpoint-dir checkpoints/precovid/zero/20100101-20191231 \
    --dates 2020-03-01 2020-06-01 2020-12-31 \
    --num-paths 100 \
    --seq-len 60
```

### Programmatic Usage

```python
from checkpoint_loader_sampler import load_and_sample

# Generate samples from any checkpoint
samples = load_and_sample(
    checkpoint_dir='checkpoints/precovid/explicit/20100101-20191231',
    dates=['2020-03-01', '2020-06-01', '2020-12-31'],
    num_paths=100,
    output_dir='my_results',
    seq_len=60
)

print(f"Generated samples: {samples.shape}")  # (100, 60)
```

## Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--checkpoint-dir` | Path to checkpoint directory | Required |
| `--dates` | Trading dates for sampling | `['2020-01-01', '2020-06-01', '2020-12-31']` |
| `--num-paths` | Number of sample paths | `100` |
| `--output-dir` | Output directory | `{checkpoint_dir}/samples` |
| `--seq-len` | Sequence length | `60` |

## Output Structure

```
output_dir/
├── samples.npy           # Generated samples [paths, T]
├── sample_metadata.json  # Sample generation metadata
└── manifest.json         # Complete execution manifest
```

## Supported Model Types

### Zero Conditioning (Unconditional)
- **Model**: `DenoiseMLP`
- **Conditioning**: None
- **Output Shape**: `[paths, T]`

### Explicit Conditioning
- **Model**: `ExplicitConditioningDDPM` 
- **Conditioning**: 4 regime one-hots + volatility + trend
- **Features**: Reconstructed from saved thresholds/scalers
- **Output Shape**: `[paths, T]`

### LLM Conditioning
- **Model**: `ConditionedDiffusionModel`
- **Conditioning**: PCA-reduced LLM embeddings
- **Features**: Uses saved PCA model for projection
- **Output Shape**: `[paths, T]`

## Technical Details

### Model Reconstruction
The utility automatically:
1. Loads `meta.json` and `conditioning_spec.json`
2. Infers model hyperparameters (especially `hidden_dim`) from parameter count
3. Imports the correct model class from available modules
4. Reconstructs model with exact original architecture

### Conditioning Reconstruction
For each model type:
- **Zero**: Returns `None` (no conditioning)
- **Explicit**: Uses saved scaler parameters and thresholds (never refits)
- **LLM**: Loads saved PCA model and generates realistic embeddings

### Runtime Detection
The utility inspects method signatures to detect:
- Whether `forward()` accepts conditioning parameters
- Whether `sample()` method exists and accepts conditioning
- Correct parameter names for conditioning (`conditioning`, `context`, `cond`, etc.)

## Examples

### Test All Pre-COVID Models
```bash
# Zero model
python checkpoint_loader_sampler.py \
    --checkpoint-dir checkpoints/precovid/zero/20100101-20191231 \
    --num-paths 50

# Explicit model  
python checkpoint_loader_sampler.py \
    --checkpoint-dir checkpoints/precovid/explicit/20100101-20191231 \
    --num-paths 50

# LLM model
python checkpoint_loader_sampler.py \
    --checkpoint-dir checkpoints/precovid/llm/20100101-20191231 \
    --num-paths 50
```

### Batch Generation
```python
import numpy as np
from checkpoint_loader_sampler import CheckpointSampler

# Load once, sample multiple times
sampler = CheckpointSampler('checkpoints/precovid/zero/20100101-20191231')

# Generate samples for different periods
covid_samples = sampler.generate_samples(
    dates=['2020-03-01', '2020-06-01'], 
    num_paths=1000,
    output_dir='covid_scenario'
)

post_covid_samples = sampler.generate_samples(
    dates=['2022-01-01', '2022-06-01'], 
    num_paths=1000,
    output_dir='post_covid_scenario'
)
```

### Custom Analysis
```python
# Load samples for analysis
samples = np.load('path/to/samples.npy')  # Shape: [paths, T]

# Calculate statistics
returns_mean = samples.mean(axis=0)  # Average return per time step
returns_std = samples.std(axis=0)    # Volatility per time step
path_volatility = samples.std(axis=1)  # Per-path volatility

# Risk metrics
var_95 = np.percentile(samples.sum(axis=1), 5)  # 95% VaR
cvar_95 = samples.sum(axis=1)[samples.sum(axis=1) <= var_95].mean()  # CVaR
```

## Error Handling

The utility provides comprehensive error handling:

- **Missing Files**: Gracefully handles missing checkpoints/metadata
- **Model Mismatches**: Infers correct hyperparameters or provides warnings
- **Conditioning Failures**: Falls back to representative conditioning
- **Import Errors**: Searches multiple module paths for model classes

All errors and warnings are logged and saved in the execution manifest.

## Troubleshooting

### Model Loading Issues
```
Error: Parameter count mismatch
```
- The utility automatically infers `hidden_dim` from parameter count
- If inference fails, check that the model class supports the constructor signature

### Conditioning Issues
```
Warning: No PCA model available
```
- For LLM models, ensure `pca_model.pkl` exists in checkpoint directory
- Fallback uses random conditioning with correct dimensions

### Shape Issues
```
Error: Tensor shape mismatch
```
- The utility handles different model architectures automatically
- Explicit models use `[B, 1, T]`, others use `[B, T]`

## Advanced Usage

### Custom Conditioning Providers
```python
from checkpoint_loader_sampler import ConditioningProvider

class CustomConditioningProvider(ConditioningProvider):
    def generate_conditioning(self, dates, num_paths):
        # Your custom conditioning logic
        return custom_conditioning_vectors

# Use with CheckpointSampler
sampler.generator.conditioning_provider = CustomConditioningProvider(spec)
```

### Extending Model Support
```python
from checkpoint_loader_sampler import ModelClassRegistry

# Add support for new model types
ModelClassRegistry.MODEL_MAPPINGS['my_model'] = {
    'class_name': 'MyDiffusionModel',
    'trainer_class': 'MyTrainer', 
    'modules': ['my_module', 'my_package.models']
}
```

## Integration

This utility integrates seamlessly with:
- **Training Pipeline**: Load checkpoints from `train_precovid_models.py`
- **Evaluation Scripts**: Generate samples for downstream analysis
- **Risk Management**: Scenario generation for VaR/CVaR calculations
- **Research Workflows**: Comparative studies across model types and time periods

The checkpoint loader provides a robust foundation for any analysis requiring sample generation from trained diffusion models.
