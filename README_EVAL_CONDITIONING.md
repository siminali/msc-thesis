# Evaluation-Time Conditioning Providers

Causal, day-by-day conditioning providers for evaluation that never refit on evaluation data and use exact transforms from saved specifications.

## Features

- **Strictly Causal**: Day-by-day computation with no look-ahead bias
- **Spec-Consistent**: Uses exact transforms saved in `conditioning_spec.json`
- **Never Refits**: Preserves training-evaluation separation 
- **Graceful Fallbacks**: Handles incomplete specs and missing data
- **Comprehensive Logging**: Detailed warnings and error tracking
- **Versioning Safe**: Creates `_v2.py` files to avoid overwriting

## Quick Start

### Basic Usage

```python
from eval_conditioning_providers import generate_eval_conditioning
import pandas as pd

# Load your returns data
returns_data = pd.read_csv('returns.csv', index_col=0, parse_dates=True)

# Generate conditioning for target dates
conditioning, warnings = generate_eval_conditioning(
    checkpoint_dir='checkpoints/precovid/explicit/20100101-20191231',
    returns_data=returns_data,
    target_dates=['2020-03-01', '2020-06-01', '2020-12-31']
)

print(f"Conditioning shape: {conditioning.shape}")
print(f"Warnings: {warnings}")
```

### Command Line Usage

```bash
# Test zero conditioning
python eval_conditioning_providers.py \
    --checkpoint-dir checkpoints/precovid/zero/20100101-20191231 \
    --target-dates 2020-03-01 2020-06-01

# Test explicit conditioning
python eval_conditioning_providers.py \
    --checkpoint-dir checkpoints/precovid/explicit/20100101-20191231 \
    --target-dates 2020-03-01 2020-06-01 \
    --returns-file sp500_data.csv

# Test LLM conditioning  
python eval_conditioning_providers.py \
    --checkpoint-dir checkpoints/precovid/llm/20100101-20191231 \
    --target-dates 2020-03-01 2020-06-01 \
    --output-file llm_conditioning.npy
```

## Provider Types

### NoneProvider (Zero Conditioning)
- **Purpose**: Unconditional models (no conditioning)
- **Output**: `None`
- **Usage**: Baseline comparisons, unconditional generation

```python
from eval_conditioning_providers import NoneProvider

provider = NoneProvider(conditioning_spec)
conditioning = provider.generate_conditioning(returns_data, target_dates)
# conditioning is None
```

### ExplicitEvalProvider (Financial Features)
- **Purpose**: Regime classification + financial features
- **Output**: `[n_dates, 6]` - 4 regime one-hots + volatility + trend
- **Features**:
  - **Regime Classification**: Up/Down × Low/High volatility (4 one-hots)
  - **Causal Volatility**: 20-day rolling std, scaled by training parameters
  - **Causal Trend**: 60-day rolling sum, scaled by training parameters

```python
from eval_conditioning_providers import ExplicitEvalProvider

provider = ExplicitEvalProvider(conditioning_spec, checkpoint_dir)
conditioning = provider.generate_conditioning(returns_data, target_dates)
# conditioning shape: [n_dates, 6]
# [Up-Low, Up-High, Down-Low, Down-High, z_vol, z_trend]
```

#### Causal Computation Details
- **No Look-Ahead**: Uses only data ≤ target date
- **Expanding Windows**: For early periods with insufficient history
- **Saved Transforms**: Uses training-fitted mean/std and thresholds
- **Fallback Handling**: Derives minimal parameters if spec incomplete

### LLMEvalProvider (Embeddings + PCA)
- **Purpose**: LLM embeddings reduced via PCA
- **Output**: `[n_dates, k]` - k PCA components (default k=16)
- **Features**:
  - **Saved PCA**: Uses PCA model fitted during training
  - **Causal Embeddings**: Uses latest available embedding ≤ target date
  - **Missing Data**: Handles missing embeddings gracefully
  - **Fallback PCA**: Fits on ≤2019-12-31 if PCA model missing (logged as suspect)

```python
from eval_conditioning_providers import LLMEvalProvider

provider = LLMEvalProvider(conditioning_spec, checkpoint_dir)
conditioning = provider.generate_conditioning(returns_data, target_dates)
# conditioning shape: [n_dates, pca_components]
```

## Specification Format

Each provider reads from `conditioning_spec.json` in the checkpoint directory:

### Zero Conditioning
```json
{
    "type": "zero",
    "conditioning_dim": 0,
    "description": "No conditioning - basic unconditional DDPM"
}
```

### Explicit Conditioning
```json
{
    "type": "explicit",
    "conditioning_dim": 6,
    "vol_threshold": 0.007623,
    "vol_window": 20,
    "trend_window": 60,
    "features": {
        "z_vol": {
            "scaler_mean": 0.008538,
            "scaler_scale": 0.003673
        },
        "trend": {
            "scaler_mean": 1.508790,
            "scaler_scale": 2.317966
        }
    }
}
```

### LLM Conditioning
```json
{
    "type": "llm",
    "conditioning_dim": 16,
    "pca_components": 16,
    "original_embedding_dim": 768,
    "explained_variance_ratio": 0.4195,
    "train_cutoff": "2019-12-31"
}
```

## Causal Computation Guarantees

### Strict Causality
- **No Future Data**: Only uses data with timestamps ≤ target date
- **Day-by-Day**: Computes features independently for each target date
- **Expanding Windows**: For early periods, uses all available history

### Training-Evaluation Separation
- **Never Refits**: Uses exact scalers/thresholds from training
- **Preserved Transforms**: Applies training-fitted PCA without modification
- **Spec Consistency**: Reads all parameters from saved specifications

### Fallback Handling
- **Incomplete Specs**: Derives minimal parameters from early evaluation data (logged as "SUSPECT")
- **Missing Data**: Returns zero conditioning for unavailable dates
- **Failed Transforms**: Graceful degradation with comprehensive logging

## Examples

### Example 1: Basic Evaluation Conditioning

```python
import pandas as pd
from eval_conditioning_providers import generate_eval_conditioning

# Load returns data
returns_data = pd.read_csv('sp500_data.csv', index_col=0, parse_dates=True)

# Generate conditioning for COVID period using pre-COVID model
conditioning, warnings = generate_eval_conditioning(
    checkpoint_dir='checkpoints/precovid/explicit/20100101-20191231',
    returns_data=returns_data,
    target_dates=['2020-03-15', '2020-03-20', '2020-03-25']  # Market crash period
)

print(f"Conditioning shape: {conditioning.shape}")  # (3, 6)
print("Regime classifications:")
regimes = ['Up-Low', 'Up-High', 'Down-Low', 'Down-High']
for i, date in enumerate(['2020-03-15', '2020-03-20', '2020-03-25']):
    regime_idx = conditioning[i, :4].argmax()
    vol_scaled = conditioning[i, 4]
    trend_scaled = conditioning[i, 5]
    print(f"  {date}: {regimes[regime_idx]}, vol={vol_scaled:.3f}, trend={trend_scaled:.3f}")
```

### Example 2: Stress Testing with Multiple Models

```python
from eval_conditioning_providers import EvalProviderFactory, load_conditioning_spec

stress_dates = ['2020-03-09', '2020-03-12', '2020-03-16', '2020-03-20']  # Black Monday week
model_types = ['zero', 'explicit', 'llm']

for model_type in model_types:
    checkpoint_dir = f'checkpoints/precovid/{model_type}/20100101-20191231'
    
    # Load spec and create provider
    spec = load_conditioning_spec(checkpoint_dir)
    provider = EvalProviderFactory.create_provider(spec, checkpoint_dir)
    
    # Generate conditioning
    conditioning = provider.generate_conditioning(returns_data, stress_dates)
    
    if conditioning is not None:
        print(f"{model_type.upper()} conditioning: {conditioning.shape}")
        print(f"  Mean: {conditioning.mean():.6f}, Std: {conditioning.std():.6f}")
    else:
        print(f"{model_type.upper()} conditioning: None (zero)")
```

### Example 3: Time Series Analysis

```python
from eval_conditioning_providers import ExplicitEvalProvider
import matplotlib.pyplot as plt

# Generate conditioning for a longer period
covid_period = pd.date_range('2020-01-01', '2020-12-31', freq='D')
covid_dates = [date for date in covid_period if date.weekday() < 5]  # Weekdays only

spec = load_conditioning_spec('checkpoints/precovid/explicit/20100101-20191231')
provider = ExplicitEvalProvider(spec, checkpoint_dir)

conditioning = provider.generate_conditioning(returns_data, covid_dates)

# Extract regime classifications over time
regimes = conditioning[:, :4].argmax(axis=1)
volatility = conditioning[:, 4]
trend = conditioning[:, 5]

# Plot regime transitions
regime_names = ['Up-Low', 'Up-High', 'Down-Low', 'Down-High']
regime_colors = ['green', 'orange', 'blue', 'red']

plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.scatter(covid_dates, regimes, c=[regime_colors[r] for r in regimes], alpha=0.7)
plt.ylabel('Regime')
plt.title('Market Regime Classification During COVID-19')
plt.yticks(range(4), regime_names)

plt.subplot(3, 1, 2)
plt.plot(covid_dates, volatility)
plt.ylabel('Scaled Volatility')
plt.title('Volatility Evolution')

plt.subplot(3, 1, 3)
plt.plot(covid_dates, trend)
plt.ylabel('Scaled Trend')
plt.title('Trend Evolution')
plt.xlabel('Date')

plt.tight_layout()
plt.show()
```

### Example 4: Custom Provider

```python
from eval_conditioning_providers import BaseEvalProvider

class CustomEvalProvider(BaseEvalProvider):
    """Custom provider with additional features."""
    
    def __init__(self, conditioning_spec, checkpoint_dir=None):
        super().__init__(conditioning_spec, checkpoint_dir)
        # Add custom initialization
        
    def generate_conditioning(self, returns_data, target_dates):
        # Custom causal computation
        conditioning_vectors = []
        
        for target_date in target_dates:
            # Ensure causality
            causal_data = returns_data[returns_data.index <= target_date]
            
            # Your custom feature computation
            custom_features = self._compute_custom_features(causal_data)
            conditioning_vectors.append(custom_features)
        
        return np.array(conditioning_vectors)
    
    def _compute_custom_features(self, causal_data):
        # Implement your custom causal feature computation
        return np.array([1.0, 2.0, 3.0])  # Example
```

## Integration with Sampling

The evaluation providers integrate seamlessly with the checkpoint loader for end-to-end evaluation:

```python
from checkpoint_loader_sampler import CheckpointSampler
from eval_conditioning_providers import generate_eval_conditioning

# Load model
sampler = CheckpointSampler('checkpoints/precovid/explicit/20100101-20191231')

# Generate evaluation conditioning
target_dates = ['2020-03-15', '2020-03-20']
conditioning, warnings = generate_eval_conditioning(
    sampler.checkpoint_dir,
    returns_data,
    target_dates
)

# Generate samples with evaluation conditioning
# (This would require extending the sampler to accept external conditioning)
```

## Error Handling and Logging

The providers include comprehensive error handling:

### Warning Categories
- **`spec_missing_pca`**: PCA model not found, fitting on evaluation data
- **`SUSPECT`**: Using evaluation-derived fallback parameters
- **Missing Data**: Embeddings or returns not available for target dates
- **Transform Failures**: PCA transformation or scaling errors

### Logging Levels
- **INFO**: Normal operation, successful conditioning generation
- **WARNING**: Non-critical issues, fallback usage, missing data
- **ERROR**: Critical failures that prevent conditioning generation

### Best Practices
1. **Always Check Warnings**: Review returned warnings for data quality issues
2. **Validate Specs**: Ensure conditioning specs are complete before evaluation
3. **Monitor Causality**: Verify no future data leakage in custom implementations
4. **Test Fallbacks**: Validate behavior with incomplete or missing data

## Performance Considerations

- **Caching**: Providers don't cache computed features (implement if needed)
- **Memory**: Large embedding datasets may require streaming
- **Computation**: Causal volatility/trend computation is O(n²) for n target dates
- **Parallelization**: Each target date is computed independently (parallelizable)

## Advanced Usage

### Batch Processing
```python
# Process multiple checkpoints
checkpoints = [
    'checkpoints/precovid/zero/20100101-20191231',
    'checkpoints/precovid/explicit/20100101-20191231', 
    'checkpoints/precovid/llm/20100101-20191231'
]

results = {}
for checkpoint_dir in checkpoints:
    conditioning, warnings = generate_eval_conditioning(
        checkpoint_dir, returns_data, target_dates
    )
    results[checkpoint_dir] = {'conditioning': conditioning, 'warnings': warnings}
```

### Custom Evaluation Periods
```python
# Evaluate on specific market events
events = {
    'dot_com_crash': pd.date_range('2000-03-01', '2000-04-01'),
    'financial_crisis': pd.date_range('2008-09-01', '2008-10-01'),
    'covid_crash': pd.date_range('2020-02-20', '2020-03-20')
}

for event_name, event_dates in events.items():
    conditioning, warnings = generate_eval_conditioning(
        checkpoint_dir, returns_data, event_dates.tolist()
    )
    # Analyze conditioning for this event
```

The evaluation conditioning providers ensure that your model evaluation maintains the same rigor as training while providing the flexibility to handle real-world evaluation scenarios with missing data and incomplete specifications.
