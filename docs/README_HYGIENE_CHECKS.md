# Hygiene & Reproducibility Checks

Comprehensive, automated checks for data integrity, causality, and reproducibility in financial time series experiments. Designed to be lightweight, reusable, and never fail - always logs issues and continues execution.

## Overview

The hygiene checks module provides four core validation categories:
- **🕐 Causality**: Ensures no look-ahead bias in feature construction
- **📋 Spec Fidelity**: Validates proper reconstruction from conditioning_spec.json
- **🚫 Data Leakage**: Detects pre-COVID models using post-COVID training data
- **🔄 Determinism**: Sets up and verifies reproducible execution environment

## Quick Start

### Basic Usage

```python
from hygiene_checks import quick_hygiene_check, setup_reproducible_environment
from pathlib import Path

# Quick comprehensive check
status, results = quick_hygiene_check(
    checkpoint_path=Path('checkpoints/precovid/explicit/20100101-20191231'),
    model_type='explicit',
    seed=42
)

print(f"Status: {status}")  # "clean" or "suspect"
print(f"Issues: {results['summary']['total_issues']}")
```

### Setup Reproducible Environment

```python
# Set up deterministic execution
env_info = setup_reproducible_environment(seed=42)
print(f"Environment setup with seed: {env_info['seed']}")
```

### Command Line Usage

```bash
# Check specific checkpoint
python hygiene_checks.py --checkpoint checkpoints/precovid/explicit/20100101-20191231 --model-type explicit --seed 42

# Setup reproducible environment only
python hygiene_checks.py --seed 123
```

## Check Categories

### 1. 🕐 Causality Checks

**Purpose**: Ensure explicit features for day `t` never use data `> t`

**What it checks**:
- Volatility window doesn't extend into future
- Trend window doesn't extend into future
- Target date has sufficient historical data
- No accidental future data inclusion

**Example**:
```python
from hygiene_checks import check_feature_causality
import pandas as pd

# Check feature causality
returns_data = pd.read_csv('sp500_data.csv', index_col=0, parse_dates=True)
target_dates = [pd.Timestamp('2020-03-15'), pd.Timestamp('2020-03-22')]

is_causal = check_feature_causality(
    returns_data=returns_data,
    target_dates=target_dates,
    model_type='explicit',
    vol_window=20,
    trend_window=60
)
```

**Violations logged as**: `suspect_causality`

### 2. 📋 Spec Fidelity Checks

**Purpose**: Verify conditioning_spec.json contains all required components for proper reconstruction

**What it checks**:
- **Basic fields**: `schema`, `model_type`
- **Explicit model**: `vol_threshold`, `vol_window`, `trend_window`, `vol_scaler`, `trend_scaler`
- **LLM model**: `pca_components`, `explained_variance`, `pca_model_path`
- **Scaler structure**: Proper `mean`/`scale` parameters
- **File existence**: PCA model files exist at specified paths

**Example**:
```python
from hygiene_checks import validate_conditioning_spec
from pathlib import Path

status, issues = validate_conditioning_spec(
    spec_path=Path('checkpoints/precovid/llm/20100101-20191231/conditioning_spec.json'),
    model_type='llm'
)

print(f"Spec status: {status}")  # "clean", "suspect", or "error"
for issue in issues:
    print(f"  - {issue}")
```

**Violations logged as**: `spec_missing_*`

### 3. 🚫 Data Leakage Checks (Experiment A)

**Purpose**: Verify pre-COVID checkpoints use transforms fitted only on ≤ 2019-12-31

**What it checks**:
- Training end date ≤ 2019-12-31
- Scaler fitting dates ≤ 2019-12-31
- PCA fitting date ≤ 2019-12-31
- Checkpoint path indicates pre-COVID nature

**Violations logged as**: `precovid_leakage_*`

### 4. 🔄 Determinism Checks

**Purpose**: Ensure reproducible execution environment

**What it sets up**:
- Python `random.seed()`
- NumPy `np.random.seed()`
- PyTorch `torch.manual_seed()` and `torch.cuda.manual_seed_all()`
- PyTorch deterministic algorithms
- CUDNN deterministic mode

**What it verifies**:
- All random seeds are set
- Deterministic algorithms enabled
- CUDNN settings configured
- Device information logged

**Violations logged as**: `determinism_*`

## API Reference

### Main Classes

#### `HygieneChecker`
Main orchestrator class for all checks.

```python
checker = HygieneChecker()
results = checker.run_all_checks(
    checkpoint_path=Path('checkpoints/precovid/explicit/20100101-20191231'),
    model_type='explicit',
    returns_data=returns_df,  # Optional for causality checks
    target_dates=date_list,   # Optional for causality checks
    check_precovid=True,      # Check for pre-COVID compliance
    setup_determinism=True,   # Setup deterministic environment
    seed=42
)
```

#### `HygieneFlags`
Container for collecting and tracking hygiene issues.

```python
flags = HygieneFlags()
flags.add_causality_issue("Vol window violation: target 2020-03-15 needs 20 days, only 15 available")
flags.add_spec_issue("spec_missing_pca_path: LLM spec missing pca_model_path")

summary = flags.get_summary()
print(f"Overall status: {flags.overall_status}")  # "clean" or "suspect"
```

### Individual Checkers

#### `CausalityChecker`
```python
checker = CausalityChecker()
is_clean = checker.check_explicit_features(returns_data, target_dates, vol_window=20, trend_window=60, flags)
```

#### `SpecFidelityChecker`
```python
checker = SpecFidelityChecker()
result = checker.check_conditioning_spec(spec_path, model_type='explicit', flags)
```

#### `LeakageChecker`
```python
checker = LeakageChecker()
no_leakage = checker.check_precovid_training_dates(spec_path, model_type='llm', flags)
```

#### `DeterminismChecker`
```python
checker = DeterminismChecker()
setup_info = checker.setup_deterministic_execution(flags, seed=42)
verification = checker.verify_deterministic_state(flags)
```

### Convenience Functions

#### `quick_hygiene_check()`
Run all relevant checks in one call:
```python
status, results = quick_hygiene_check(
    checkpoint_path=Path('checkpoints/precovid/explicit/20100101-20191231'),
    model_type='explicit',
    returns_data=returns_df,  # Optional
    target_dates=date_list,   # Optional
    seed=42
)
```

#### `setup_reproducible_environment()`
Setup deterministic execution:
```python
env_info = setup_reproducible_environment(seed=42)
```

#### `validate_conditioning_spec()`
Quick spec validation:
```python
status, issues = validate_conditioning_spec(spec_path, model_type='llm')
```

#### `check_feature_causality()`
Quick causality check:
```python
is_causal = check_feature_causality(returns_data, target_dates, model_type='explicit')
```

## Integration Examples

### With Training Scripts

```python
from hygiene_checks import setup_reproducible_environment, HygieneChecker

def train_model():
    # Setup reproducible environment first
    env_info = setup_reproducible_environment(seed=42)
    logger.info(f"Reproducible environment setup: {env_info['seed']}")
    
    # Your training code here...
    
    # After saving checkpoint, validate it
    checker = HygieneChecker()
    results = checker.run_all_checks(
        checkpoint_path=checkpoint_dir,
        model_type='explicit',
        setup_determinism=False  # Already done
    )
    
    if results['overall_status'] == 'suspect':
        logger.warning(f"Checkpoint validation issues: {results['summary']['total_issues']}")
    
    return results
```

### With Evaluators

```python
from hygiene_checks import quick_hygiene_check

def load_checkpoint_safely(checkpoint_path, model_type):
    # Validate checkpoint before loading
    status, results = quick_hygiene_check(
        checkpoint_path=checkpoint_path,
        model_type=model_type,
        setup_determinism=True
    )
    
    if status == 'suspect':
        logger.warning(f"Loading suspect checkpoint: {results['summary']['total_issues']} issues")
        # Log specific issues but continue
        for category, issues in results['summary']['details'].items():
            for issue in issues:
                logger.warning(f"{category}: {issue}")
    
    # Continue with checkpoint loading...
    return load_checkpoint(checkpoint_path)
```

### With Experiment Pipelines

```python
from hygiene_checks import HygieneChecker

class ExperimentWithHygiene:
    def __init__(self):
        self.hygiene_checker = HygieneChecker()
    
    def run_experiment(self, window_id, checkpoints, returns_data):
        # Check causality for this window
        target_dates = self.get_window_dates(window_id)
        
        for model_name, checkpoint_path in checkpoints.items():
            # Full hygiene check
            results = self.hygiene_checker.run_all_checks(
                checkpoint_path=checkpoint_path,
                model_type=model_name,
                returns_data=returns_data,
                target_dates=target_dates
            )
            
            # Mark window/model as suspect if issues found
            if results['overall_status'] == 'suspect':
                self.mark_window_model_suspect(window_id, model_name, results)
            
            # Continue with sampling regardless
            self.generate_samples(model_name, checkpoint_path, target_dates)
```

## Example Output

### Clean Execution
```
🧹 Starting comprehensive hygiene and reproducibility checks
1️⃣ Setting up deterministic execution
✅ Python random seed set
✅ NumPy random seed set  
✅ PyTorch random seeds set
✅ PyTorch deterministic algorithms enabled
2️⃣ Checking conditioning spec fidelity
✅ Conditioning spec validation passed for explicit
3️⃣ Checking pre-COVID data leakage
✅ No pre-COVID data leakage detected for explicit
4️⃣ Checking feature causality
✅ All explicit features pass causality check
✅ All hygiene checks passed - execution environment is clean
🧹 Hygiene checks completed: 0 issues found

Hygiene Check Results: CLEAN
Total Issues: 0
```

### Suspect Execution
```
🧹 Starting comprehensive hygiene and reproducibility checks
...
SPEC ISSUE: spec_missing_schema: Required field 'schema' missing from spec
SPEC ISSUE: spec_missing_vol_scaler: Explicit model missing 'vol_scaler'
❌ Conditioning spec validation failed for explicit: 2 issues
❌ Hygiene issues detected - marking as suspect (total: 2)
🧹 Hygiene checks completed: 2 issues found

Hygiene Check Results: SUSPECT
Total Issues: 2

Issue Details:
  spec_fidelity: 2 issues
    - spec_missing_schema: Required field 'schema' missing from spec
    - spec_missing_vol_scaler: Explicit model missing 'vol_scaler'
```

## Result Structure

### Full Results Dictionary
```python
{
    "timestamp": "2025-08-30T19:25:42.123456",
    "overall_status": "suspect",  # "clean" or "suspect"
    "checks_performed": ["determinism", "spec_fidelity", "data_leakage", "causality"],
    "causality": {
        "explicit": True  # or False if violations found
    },
    "spec_fidelity": {
        "status": "suspect",
        "issues": ["spec_missing_schema", "spec_missing_vol_scaler"],
        "spec_content": {...}
    },
    "data_leakage": {
        "precovid_compliant": True
    },
    "determinism": {
        "setup": {
            "seed": 42,
            "python_random": True,
            "numpy_random": True,
            "torch_random": True,
            "torch_deterministic": True,
            "device_info": {...}
        },
        "verification": {
            "torch_deterministic": True,
            "warnings": []
        }
    },
    "summary": {
        "overall_status": "suspect",
        "causality_issues": 0,
        "spec_issues": 2,
        "leakage_issues": 0,
        "determinism_issues": 0,
        "total_issues": 2,
        "details": {
            "causality": [],
            "spec_fidelity": ["spec_missing_schema", "spec_missing_vol_scaler"],
            "data_leakage": [],
            "determinism": []
        }
    }
}
```

## Best Practices

### 1. **Always Run Before Training**
```python
# At start of training script
env_info = setup_reproducible_environment(seed=42)
logger.info(f"Training with deterministic seed: {env_info['seed']}")
```

### 2. **Validate Checkpoints After Creation**
```python
# After saving checkpoint
status, results = quick_hygiene_check(checkpoint_path, model_type)
if status == 'suspect':
    logger.warning("Checkpoint has hygiene issues - review before use")
```

### 3. **Check Causality During Feature Engineering**
```python
# During explicit feature computation
is_causal = check_feature_causality(returns_data, target_dates, 'explicit', vol_window, trend_window)
if not is_causal:
    logger.warning("Causality violations detected in features")
```

### 4. **Integration with Experiment Pipelines**
```python
# In experiment evaluators
checker = HygieneChecker()
for checkpoint_path in checkpoints:
    results = checker.run_all_checks(checkpoint_path=checkpoint_path, model_type=model_type)
    if results['overall_status'] == 'suspect':
        # Mark in manifest but continue
        manifest['hygiene_status'] = 'suspect'
        manifest['hygiene_issues'] = results['summary']['total_issues']
```

### 5. **Never Fail on Hygiene Issues**
The module is designed to **never fail** execution:
- Issues are logged as warnings
- Execution continues with "suspect" marking  
- Detailed issue tracking for post-analysis
- Graceful degradation for missing components

## Error Handling

### Missing Files
```python
# Handles missing conditioning specs gracefully
status, results = quick_hygiene_check(nonexistent_path, 'explicit')
# Returns "suspect" status with spec_missing_file issue
```

### Incomplete Specs  
```python
# Handles partial or malformed specs
# Logs specific missing fields: spec_missing_vol_scaler, etc.
# Continues execution with suspect marking
```

### Environment Issues
```python
# Handles missing PyTorch gracefully
# Skips torch-specific checks with warnings
# Continues with available checks
```

## Dependencies

### Required
- `pandas` - For date/time handling
- `numpy` - For random seed management

### Optional
- `torch` - For PyTorch determinism (skipped if not available)

### Standard Library
- `json`, `logging`, `pathlib`, `datetime`, `random`, `os`, `sys`

## Performance

### Execution Time
- **Full check**: ~0.1-0.5 seconds
- **Spec validation**: ~0.01-0.05 seconds  
- **Causality check**: ~0.05-0.2 seconds (depends on data size)
- **Determinism setup**: ~0.01-0.1 seconds

### Memory Usage
- **Minimal overhead**: ~5-10 MB
- **Scales with data**: Causality checks scale with returns data size
- **No persistent state**: Designed for repeated use

The hygiene checks module provides comprehensive validation and reproducibility setup while maintaining lightweight, non-intrusive operation that enhances rather than hinders the experimental workflow.
