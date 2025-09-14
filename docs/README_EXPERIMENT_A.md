# Experiment A: Out-of-Sample Stress Testing Evaluator

Comprehensive evaluator for testing pre-COVID trained models on stress periods (COVID crash, recovery, etc.).

## Features

- **Automatic Checkpoint Discovery**: Finds and validates pre-COVID checkpoints
- **Pre-COVID Verification**: Ensures models were trained only on pre-2020 data
- **Stress Testing Windows**: Pre-defined crisis periods for evaluation
- **Causal Conditioning**: Uses evaluation conditioning providers for realistic stress testing
- **Sample Generation Orchestration**: Generates samples for all window × model combinations
- **Comprehensive Metadata**: Detailed execution plans and manifests
- **Versioning Safety**: Creates A_v2, A_v3, etc. to avoid overwrites
- **Robust Error Handling**: Continues execution even if individual models fail

## Quick Start

### Basic Usage

```bash
# Run with default settings (COVID crash, recovery, post-COVID windows)
python experiment_A_evaluator.py --csv-file sp500_data.csv --num-paths 1000

# Quick test with small sample size
python experiment_A_evaluator.py --windows covid_crash --num-paths 50
```

### List Available Windows

```bash
python experiment_A_evaluator.py --list-windows
```

Output:
```
Available Stress Testing Windows:
========================================
covid_crash          2020-02-20 to 2020-04-01
                     Initial COVID-19 market crash period

covid_recovery       2020-04-15 to 2020-06-15
                     Post-crash recovery period

covid_second_wave    2020-10-01 to 2020-12-31
                     COVID second wave and vaccine news

post_covid           2021-06-01 to 2021-12-31
                     Post-COVID normalization period

inflation_2022       2022-01-01 to 2022-06-30
                     High inflation and rate hike concerns
```

## Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--windows` | Stress testing windows to evaluate | `['covid_crash', 'covid_recovery', 'post_covid']` |
| `--checkpoints-dir` | Directory with pre-COVID checkpoints | `checkpoints/precovid` |
| `--csv-file` | CSV file with returns data | `sp500_data.csv` |
| `--seq-len` | Sequence length for samples | `60` |
| `--num-paths` | Number of sample paths | `1000` |
| `--seeds` | Random seeds for reproducibility | `[42]` |
| `--results-base` | Base directory for results | `results/addons/period_slices` |
| `--experiment-name` | Experiment identifier | `A` |

## Output Structure

```
results/addons/period_slices/A/
├── plan.json                     # Execution plan
├── manifest.json                 # Complete execution manifest
├── covid_crash/
│   ├── zero/
│   │   ├── samples.npy           # Generated samples [paths, seq_len]
│   │   ├── sample_metadata.json  # Sample metadata
│   │   ├── manifest.json         # Checkpoint loader manifest
│   │   └── experiment_A_metadata.json  # Experiment-specific metadata
│   ├── explicit/
│   │   └── ... (same structure)
│   └── llm/
│       └── ... (same structure)
├── covid_recovery/
│   └── ... (same structure)
└── post_covid/
    └── ... (same structure)
```

## Pre-COVID Checkpoint Validation

The evaluator automatically validates that checkpoints are truly pre-COVID:

### ✅ Valid Pre-COVID
- Training period ends ≤ 2019-12-31
- Status: `"valid"`

### ⚠️ Suspect for Experiment A
- Training period spans 2019-2020 boundary
- Training period starts after 2019-12-31
- Status: `"suspect_for_A"` (logged but continues)

### Example Validation Output
```
2025-08-30 18:05:31,869 - INFO - Validated checkpoint: zero/20100101-20191231 (pre-COVID)
2025-08-30 18:05:31,869 - INFO - Validated checkpoint: explicit/20100101-20191231 (pre-COVID)
2025-08-30 18:05:31,869 - INFO - Validated checkpoint: llm/20100101-20191231 (pre-COVID)
```

## Example Usage Patterns

### 1. Comprehensive Stress Testing

```bash
# Test all models on major crisis periods
python experiment_A_evaluator.py \
    --windows covid_crash covid_recovery covid_second_wave post_covid \
    --csv-file data/sp500_returns.csv \
    --num-paths 5000 \
    --seq-len 60 \
    --seeds 42 123 456
```

### 2. Focused COVID Analysis

```bash
# Focus on COVID crash period with high sample count
python experiment_A_evaluator.py \
    --windows covid_crash \
    --num-paths 10000 \
    --csv-file sp500_data.csv
```

### 3. Custom Checkpoint Directory

```bash
# Use custom checkpoint location
python experiment_A_evaluator.py \
    --checkpoints-dir my_models/precovid \
    --results-base my_results \
    --csv-file data.csv
```

### 4. Quick Validation Run

```bash
# Small test to validate setup
python experiment_A_evaluator.py \
    --windows covid_crash \
    --num-paths 10 \
    --csv-file test_data.csv
```

## Generated Files

### plan.json
Contains the execution plan before running:
```json
{
    "experiment": "A",
    "description": "Out-of-sample stress testing with pre-COVID checkpoints",
    "parameters": {
        "num_paths": 1000,
        "seq_len": 60,
        "seeds": [42]
    },
    "windows": {
        "covid_crash": {
            "name": "COVID Market Crash",
            "start": "2020-02-20",
            "end": "2020-04-01"
        }
    },
    "execution_matrix": {
        "covid_crash": {
            "zero": {"output_path": "results/.../zero/samples.npy"},
            "explicit": {"output_path": "results/.../explicit/samples.npy"},
            "llm": {"output_path": "results/.../llm/samples.npy"}
        }
    }
}
```

### manifest.json
Contains complete execution results:
```json
{
    "experiment": "A",
    "status": "completed",
    "windows": {...},
    "models": {
        "zero": {
            "checkpoints": 1,
            "primary_checkpoint": "20100101-20191231",
            "status": "valid"
        }
    },
    "results": {
        "covid_crash": {
            "zero": {
                "status": "success",
                "samples_shape": [1000, 60],
                "generated_at": "2025-08-30T18:05:44"
            }
        }
    },
    "errors": [],
    "warnings": []
}
```

### experiment_A_metadata.json (per model)
Detailed metadata for each window-model combination:
```json
{
    "window_id": "covid_crash",
    "model_name": "zero",
    "checkpoint_info": {
        "period": "20100101-20191231",
        "is_precovid": true,
        "status": "valid"
    },
    "generation_info": {
        "target_dates": ["2020-02-23T00:00:00", "2020-03-01T00:00:00", ...],
        "num_paths": 1000,
        "seq_len": 60,
        "samples_shape": [1000, 60]
    },
    "conditioning_info": {
        "type": "zero",
        "conditioning_dim": 0,
        "conditioning_shape": null
    }
}
```

## Integration with Analysis Pipeline

The generated samples can be directly used for downstream analysis:

```python
import numpy as np
import json

# Load samples
covid_crash_zero = np.load('results/addons/period_slices/A/covid_crash/zero/samples.npy')
covid_crash_explicit = np.load('results/addons/period_slices/A/covid_crash/explicit/samples.npy')

# Load metadata
with open('results/addons/period_slices/A/manifest.json', 'r') as f:
    manifest = json.load(f)

print(f"Experiment status: {manifest['status']}")
print(f"Zero model samples: {covid_crash_zero.shape}")
print(f"Explicit model samples: {covid_crash_explicit.shape}")

# Compare model performance
zero_volatility = covid_crash_zero.std(axis=1).mean()
explicit_volatility = covid_crash_explicit.std(axis=1).mean()
print(f"Average volatility - Zero: {zero_volatility:.6f}, Explicit: {explicit_volatility:.6f}")
```

## Model Types and Conditioning

### Zero Model (Unconditional)
- **Conditioning**: None
- **Use Case**: Baseline comparison
- **Expected Behavior**: Consistent generation regardless of stress period

### Explicit Model (Financial Features)
- **Conditioning**: Regime classification + volatility + trend
- **Use Case**: Market-aware generation
- **Expected Behavior**: Should adapt to stress conditions based on market regime

### LLM Model (Embedding-based)
- **Conditioning**: PCA-reduced embeddings
- **Use Case**: News/sentiment-aware generation
- **Expected Behavior**: Should reflect information environment of stress periods

## Error Handling

The evaluator handles various error conditions gracefully:

### Common Issues
1. **Missing Checkpoints**: Warns and continues with available models
2. **Conditioning Failures**: Uses fallback conditioning and logs warnings
3. **Sample Generation Errors**: Logs error and continues with next model
4. **Missing Data**: Handles missing returns or embedding data

### Example Error Output
```
Errors:
  - Failed to generate samples for covid_crash × llm: tensor shape mismatch

Warnings:
  - Missing embeddings for 3 target dates in covid_crash window
  - Checkpoint explicit/20200101-20201231 marked as suspect_for_A
```

## Performance Considerations

### Sample Generation Time
- **Zero Model**: ~1-2 minutes for 1000 paths
- **Explicit Model**: ~3-5 minutes for 1000 paths  
- **LLM Model**: ~2-4 minutes for 1000 paths

### Memory Usage
- **1000 paths × 60 seq_len**: ~240KB per model-window combination
- **10000 paths**: ~2.4MB per combination
- Scales linearly with `num_paths` and `seq_len`

### Disk Space
For a full experiment (3 windows × 3 models × 1000 paths):
- **Sample files**: ~2MB total
- **Metadata files**: ~100KB total
- **Complete experiment**: <5MB

## Troubleshooting

### Checkpoint Discovery Issues
```bash
# Check checkpoint structure
ls -la checkpoints/precovid/
ls -la checkpoints/precovid/zero/20100101-20191231/

# Required files: meta.json, conditioning_spec.json, best.pt
```

### Data Loading Issues
```bash
# Test with synthetic data (removes CSV dependency)
python experiment_A_evaluator.py --csv-file nonexistent.csv --num-paths 10
```

### Memory Issues
```bash
# Reduce sample size for testing
python experiment_A_evaluator.py --num-paths 100 --seq-len 30
```

### Versioning Issues
```bash
# Clean up previous experiments
rm -rf results/addons/period_slices/A*

# Or use custom experiment name
python experiment_A_evaluator.py --experiment-name test
```

## Best Practices

1. **Start Small**: Use `--num-paths 50` for initial validation
2. **Check Logs**: Review warnings for data quality issues
3. **Validate Checkpoints**: Ensure all models are truly pre-COVID
4. **Monitor Resources**: Large `num_paths` values require more memory/time
5. **Save Results**: Archive experiment directories for reproducibility
6. **Document Runs**: Use descriptive experiment names for organization

## Integration Example

```python
# Complete workflow example
import subprocess
import numpy as np
import json
from pathlib import Path

# Run experiment
result = subprocess.run([
    'python', 'experiment_A_evaluator.py',
    '--windows', 'covid_crash', 'covid_recovery',
    '--num-paths', '1000',
    '--csv-file', 'sp500_data.csv'
], capture_output=True, text=True)

if result.returncode == 0:
    print("Experiment completed successfully")
    
    # Load and analyze results
    manifest_path = Path('results/addons/period_slices/A/manifest.json')
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    # Extract sample paths for analysis
    for window, models in manifest['results'].items():
        for model, info in models.items():
            if info['status'] == 'success':
                samples_path = f"results/addons/period_slices/A/{window}/{model}/samples.npy"
                samples = np.load(samples_path)
                print(f"{window} × {model}: {samples.shape}")
else:
    print(f"Experiment failed: {result.stderr}")
```

The Experiment A evaluator provides a robust foundation for systematic out-of-sample stress testing, enabling comprehensive evaluation of how pre-COVID models perform during various crisis periods.
