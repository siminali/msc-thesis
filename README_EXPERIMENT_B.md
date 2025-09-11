# Experiment B: Counterfactual Controllability Testing

Tests the controllability of pre-COVID models by manipulating conditioning inputs while keeping model weights fixed. Evaluates how different conditioning scenarios affect generation patterns and risk metrics.

## Overview

Experiment B answers the key question: **"How much can we control model outputs by changing only the conditioning inputs?"**

This is crucial for understanding:
- Model sensitivity to different market conditions
- Robustness of conditioning mechanisms  
- Practical controllability for scenario generation
- Counterfactual "what-if" analysis capabilities

## Features

- **Real-conditions**: Use actual COVID-era market conditioning
- **Calm-conditions**: Use pre-COVID calm period conditioning on COVID dates
- **LLM-knob**: Systematically shift PCA components to test controllability
- **Risk Metric Analysis**: VaR, ES, tail mass, volatility comparisons
- **Comparative Analysis**: Quantified controllability scores
- **Versioning Safety**: Creates B_v2, B_v3, etc. to avoid overwrites

## Quick Start

### Basic Usage

```bash
# Test controllability during COVID crash
python experiment_B_evaluator.py --window covid_crash --num-paths 1000

# Test different window with custom settings
python experiment_B_evaluator.py \
    --window covid_recovery \
    --num-paths 2000 \
    --csv-file sp500_data.csv
```

## Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--window` | COVID-era window for testing | `covid_crash` |
| `--checkpoints-dir` | Directory with pre-COVID checkpoints | `checkpoints/precovid` |
| `--csv-file` | CSV file with returns data | `sp500_data.csv` |
| `--num-paths` | Number of sample paths | `1000` |
| `--seq-len` | Sequence length | `60` |
| `--seeds` | Random seeds for reproducibility | `[42]` |
| `--results-base` | Base directory for results | `results/addons/period_slices` |
| `--experiment-name` | Experiment identifier | `B` |

## Testing Modes

### 1. Real-Conditions
Uses actual market conditioning from the COVID period.

**Purpose**: Baseline behavior under real stress conditions
**Implementation**: Direct conditioning from evaluation providers using actual COVID-era data

### 2. Calm-Conditions  
Uses conditioning from equivalent calm periods (2019) applied to COVID dates.

**Purpose**: Counterfactual "what if COVID had calm market conditions?"
**Implementation**: 
- Maps each COVID date to equivalent 2019 date (same weekday, similar time of year)
- Uses 2019 conditioning but applies to COVID timeline
- Tests sensitivity to market regime vs temporal effects

### 3. LLM-Knob (LLM Models Only)
Systematically shifts specific PCA components to test controllability.

**Purpose**: Fine-grained controllability testing
**Implementation**:
- Shifts first PCA component by {-2σ, -1σ, +1σ, +2σ}
- Uses PCA model statistics for proper scaling
- Tests directional controllability

## Output Structure

```
results/addons/period_slices/B/
├── plan.json                          # Execution plan
├── manifest.json                      # Complete results
├── comparative_analysis.json          # Risk metric analysis
└── <window_id>/
    ├── explicit/
    │   ├── real-conditions/
    │   │   └── samples.npy             # [paths, seq_len]
    │   └── calm-conditions/
    │       └── samples.npy
    └── llm/
        ├── real-conditions/
        │   └── samples.npy
        ├── calm-conditions/
        │   └── samples.npy
        ├── llm-knob-comp0-shift-2.0sigma/
        │   └── samples.npy
        ├── llm-knob-comp0-shift-1.0sigma/
        │   └── samples.npy
        ├── llm-knob-comp0-shift+1.0sigma/
        │   └── samples.npy
        └── llm-knob-comp0-shift+2.0sigma/
            └── samples.npy
```

## Risk Metrics Analyzed

### Core Metrics
- **VaR (1%, 5%, 10%)**: Value at Risk at different confidence levels
- **ES (1%, 5%)**: Expected Shortfall (Conditional VaR)
- **Volatility**: Standard deviation of path returns
- **Tail Mass**: Probability of extreme negative outcomes
- **Skewness/Kurtosis**: Distribution shape characteristics

### Comparative Analysis
- **Real vs Calm**: How much does market regime matter?
- **LLM Knob Effects**: How controllable is the model through PCA manipulation?
- **Controllability Scores**: Quantified measure of model responsiveness

## Example Results

### Controllability Assessment
```json
{
    "controllability_assessment": {
        "scores": {
            "explicit": 4,
            "llm": 7
        },
        "interpretation": "Higher scores indicate greater controllability"
    }
}
```

### Risk Metric Changes
```json
{
    "model_comparisons": {
        "explicit": {
            "real_vs_calm": {
                "var_5": {
                    "real": -0.156,
                    "calm": -0.089,
                    "percent_change": +75.3
                }
            }
        }
    }
}
```

## Usage Examples

### 1. COVID Crash Controllability

```bash
# Test how models respond to COVID crash conditions
python experiment_B_evaluator.py \
    --window covid_crash \
    --num-paths 5000 \
    --csv-file sp500_data.csv
```

Expected outcomes:
- **Real-conditions**: High volatility, extreme VaR
- **Calm-conditions**: Moderate risk metrics
- **LLM-knob**: Gradual changes with PCA shifts

### 2. Recovery Period Testing

```bash
# Test controllability during recovery
python experiment_B_evaluator.py \
    --window covid_recovery \
    --num-paths 2000
```

Expected outcomes:
- Smaller differences between real/calm
- More stable LLM knob responses

### 3. High-Frequency Testing

```bash
# Quick validation run
python experiment_B_evaluator.py \
    --window covid_crash \
    --num-paths 100 \
    --seq-len 30
```

## Analysis Workflow

### 1. Load and Compare Results

```python
import json
import numpy as np

# Load comparative analysis
with open('results/addons/period_slices/B/comparative_analysis.json', 'r') as f:
    analysis = json.load(f)

# Extract controllability scores
scores = analysis['controllability_assessment']['scores']
print(f"Explicit model controllability: {scores['explicit']}")
print(f"LLM model controllability: {scores['llm']}")
```

### 2. Analyze Risk Metric Sensitivity

```python
# Compare risk metrics across modes
for model in ['explicit', 'llm']:
    model_analysis = analysis['model_comparisons'][model]
    
    if 'real_vs_calm' in model_analysis:
        var_change = model_analysis['real_vs_calm']['var_5']['percent_change']
        print(f"{model} VaR sensitivity: {var_change:+.1f}%")
```

### 3. LLM Knob Analysis

```python
# Analyze LLM knob effects
llm_effects = analysis['model_comparisons']['llm']['llm_knob_effects']

for knob_mode, effects in llm_effects.items():
    if 'var_5' in effects:
        var_change = effects['var_5']['percent_change']
        print(f"{knob_mode}: VaR change {var_change:+.1f}%")
```

## Model Support

### Explicit Model
- ✅ **Real-conditions**: Market regime + volatility + trend
- ✅ **Calm-conditions**: Uses 2019 market conditions
- ❌ **LLM-knob**: Not applicable (no PCA components)

### LLM Model  
- ✅ **Real-conditions**: PCA-reduced embeddings
- ✅ **Calm-conditions**: Uses 2019 embeddings
- ✅ **LLM-knob**: Systematic PCA component manipulation

### Zero Model
- ❌ **Not Tested**: No conditioning mechanism to manipulate

## Interpretation Guide

### Controllability Scores
- **0-2**: Low controllability - model outputs barely change with conditioning
- **3-5**: Moderate controllability - some sensitivity to conditioning changes  
- **6-8**: High controllability - strong response to conditioning manipulation
- **9+**: Very high controllability - dramatic changes possible

### Risk Metric Changes
- **<5%**: Minimal impact
- **5-20%**: Moderate impact
- **20-50%**: Strong impact  
- **>50%**: Dramatic impact

### Real vs Calm Comparison
Large differences indicate the model is sensitive to market conditions:
- **High sensitivity**: Model captures market regime effects well
- **Low sensitivity**: Model may be under-responsive to conditions

### LLM Knob Effects
Systematic changes with PCA shifts indicate:
- **Linear response**: Predictable controllability
- **Non-linear response**: Complex but controllable
- **No response**: Poor controllability despite conditioning

## Practical Applications

### 1. Scenario Generation
```python
# Generate "what-if" scenarios
# "What if COVID crash had occurred during calm market conditions?"
calm_covid_samples = np.load('results/.../explicit/calm-conditions/samples.npy')
real_covid_samples = np.load('results/.../explicit/real-conditions/samples.npy')

# Compare risk profiles
calm_var = np.percentile(calm_covid_samples.sum(axis=1), 5)
real_var = np.percentile(real_covid_samples.sum(axis=1), 5)
print(f"VaR reduction with calm conditions: {((real_var - calm_var)/abs(real_var)*100):.1f}%")
```

### 2. Model Selection
```python
# Choose most controllable model for scenario work
with open('comparative_analysis.json', 'r') as f:
    analysis = json.load(f)

scores = analysis['controllability_assessment']['scores']
best_model = max(scores, key=scores.get)
print(f"Most controllable model: {best_model} (score: {scores[best_model]})")
```

### 3. Risk Management
```python
# Estimate conditioning impact on risk metrics
def conditioning_impact(model_name):
    model_analysis = analysis['model_comparisons'][model_name]
    
    if 'real_vs_calm' in model_analysis:
        impacts = {}
        for metric in ['var_5', 'es_5', 'volatility']:
            if metric in model_analysis['real_vs_calm']:
                impacts[metric] = model_analysis['real_vs_calm'][metric]['percent_change']
        return impacts
    return {}

explicit_impact = conditioning_impact('explicit')
llm_impact = conditioning_impact('llm')
```

## Performance Considerations

### Sample Generation Time
- **Base modes**: ~5-10 minutes for 1000 paths per model
- **LLM knob modes**: ~20-30 minutes total (4 shifts × LLM model)
- **Complete experiment**: ~30-40 minutes for full analysis

### Memory Usage
- **1000 paths**: ~2-3MB per mode
- **Complete experiment**: ~20-30MB total
- Scales linearly with `num_paths`

### Computational Cost
- **Explicit model**: Moderate (regime calculation overhead)
- **LLM model**: Higher (PCA transformations + multiple knob modes)
- **Risk analysis**: Minimal (simple statistics)

## Troubleshooting

### No Controllable Models Found
```
Error: No controllable checkpoints found (need explicit or LLM models)
```
**Solution**: Ensure you have trained explicit and/or LLM models in your checkpoints directory.

### Conditioning Generation Failures
```
Warning: Failed to generate calm conditions for model X
```
**Solution**: Check that your returns data includes 2019 data for calm condition mapping.

### LLM Knob Issues
```
Warning: Component 0 not available in conditioning
```
**Solution**: The LLM model has insufficient PCA components. Check the conditioning dimension.

### Memory Issues
```bash
# Reduce sample size for testing
python experiment_B_evaluator.py --num-paths 500 --seq-len 30
```

## Best Practices

1. **Start Small**: Use `--num-paths 100` for initial validation
2. **Check Data**: Ensure returns data covers both 2019 and 2020+ periods
3. **Monitor Logs**: Watch for conditioning generation warnings
4. **Validate Results**: Sanity-check that real vs calm differences make sense
5. **Compare Models**: Use controllability scores for model selection
6. **Document Experiments**: Save comparative_analysis.json for reporting

## Integration with Downstream Analysis

```python
# Load all experiment results for meta-analysis
import glob

# Find all Experiment B results
b_experiments = glob.glob('results/addons/period_slices/B*/comparative_analysis.json')

# Compare controllability across different runs
controllability_trends = {}
for exp_file in b_experiments:
    with open(exp_file, 'r') as f:
        analysis = json.load(f)
    
    scores = analysis['controllability_assessment']['scores']
    controllability_trends[exp_file] = scores

# Analyze trends
for exp, scores in controllability_trends.items():
    print(f"{exp}: Explicit={scores.get('explicit', 'N/A')}, LLM={scores.get('llm', 'N/A')}")
```

Experiment B provides comprehensive insights into model controllability, enabling sophisticated scenario generation and risk analysis capabilities.
