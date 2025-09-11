# Integrated Experiment Pipeline v2

Complete end-to-end analysis pipeline that automatically runs sampling, metrics calculation, plotting, and findings extraction for financial time series experiments.

## Overview

The v2 evaluators extend the original experiment evaluators with integrated analysis capabilities, creating a seamless workflow from sample generation to publication-ready results.

### Key Features

- **🔗 Integrated Workflow**: Sampling → Metrics → Plotting → Findings → Summary
- **📊 Automatic Analysis**: No manual intervention between pipeline steps
- **📈 Publication Ready**: Generates plots, tables, and summary statistics
- **🔍 Compact Findings**: ΔVaR/ΔES metrics and DM p-values in JSONL format
- **🛡️ Robust Error Handling**: Continues on partial failures with clear logging
- **📁 Versioning Safety**: Creates A_v2, B_v2, etc. directories automatically

## Available Evaluators

### Experiment A v2: Integrated Stress Testing
**File**: `experiment_A_evaluator_v2.py`

**Purpose**: Out-of-sample stress testing with complete analysis pipeline

**Models Tested**: Zero, Explicit, LLM (all pre-COVID checkpoints)

**Integration Features**:
- Automatic metrics calculation for all 3 models
- Complete plotting suite (ECDF, Q-Q, VaR/ES, volatility)
- Compact findings with model comparisons
- Integrated summary with pipeline success rates

### Experiment B v2: Integrated Controllability Testing
**File**: `experiment_B_evaluator_v2.py`

**Purpose**: Counterfactual controllability testing with complete analysis pipeline

**Models Tested**: Explicit, LLM (controllable models only)

**Modes Tested**: Real-conditions, Calm-conditions, LLM-knob variations

**Integration Features**:
- Metrics for real-conditions mode (baseline comparison)
- Plotting across all generated modes
- Controllability-focused findings extraction
- Specialized controllability summary

## Quick Start

### Experiment A v2 (Stress Testing)

```bash
# Complete integrated stress testing pipeline
python experiment_A_evaluator_v2.py --window covid_crash --num-paths 1000 --csv-file sp500_data.csv

# Quick test
python experiment_A_evaluator_v2.py --window covid_crash --num-paths 100 --csv-file sp500_data.csv
```

### Experiment B v2 (Controllability)

```bash
# Complete integrated controllability pipeline  
python experiment_B_evaluator_v2.py --window covid_crash --num-paths 1000 --csv-file sp500_data.csv

# Quick test
python experiment_B_evaluator_v2.py --window covid_crash --num-paths 100 --csv-file sp500_data.csv
```

## Pipeline Workflow

### Stage 1: Sample Generation
**Original Evaluator Logic**
- Discovers pre-COVID checkpoints
- Generates samples for stress windows
- Saves samples.npy files with metadata

### Stage 2: Real Data Preparation
**New Integration Feature**
- Creates temporary `real_slice.csv` for the specific window
- Filters real market data to match the stress period
- Standardizes format for metrics calculation

### Stage 3: Metrics Calculation
**Automatic Integration**
- Calls `metrics_runner.py` with appropriate parameters
- Generates comprehensive risk and statistical metrics
- Creates `metrics.json` and `tables/*.csv` files

### Stage 4: Plotting Generation
**Automatic Integration**
- Calls `plotting_runner.py` for publication-quality plots
- Generates ECDF overlays, Q-Q plots, VaR/ES analysis, volatility tracking
- Creates both PDF and PNG formats in `figs/` directory

### Stage 5: Findings Extraction
**New Analysis Feature**
- Extracts compact key findings from metrics results
- Focuses on ΔVaR, ΔES, and Diebold-Mariano p-values
- Appends structured findings to `findings.jsonl`

### Stage 6: Summary Generation
**Comprehensive Reporting**
- Creates integrated summary with pipeline success rates
- Includes controllability insights (Experiment B)
- Generates final JSON summary for reporting

## Output Structure

```
results/addons/period_slices/<EXPERIMENT>_v<N>/
├── plan.json                           # Execution plan
├── manifest.json                       # Base experiment results
├── findings.jsonl                      # Compact findings (NEW)
├── integrated_summary.json             # Pipeline summary (NEW)
├── controllability_summary.json        # B-specific insights (NEW)
└── <window_id>/
    ├── <model>/
    │   ├── samples.npy                  # Generated samples
    │   ├── sample_metadata.json        # Sample metadata
    │   └── manifest.json               # Sample manifest
    ├── metrics.json                     # Comprehensive metrics (NEW)
    ├── tables/                         # CSV tables (NEW)
    │   ├── model_comparison.csv
    │   └── pairwise_comparisons.csv
    └── figs/                           # Publication plots (NEW)
        ├── ecdf_overlay.pdf/png
        ├── qq_plots.pdf/png
        ├── var_es_analysis.pdf/png
        └── realized_volatility.pdf/png
```

## Findings Format

### findings.jsonl Structure
Each line contains a JSON object with compact findings:

```json
{
    "window_id": "covid_crash",
    "timestamp": "2025-08-30T19:05:42.067000",
    "experiment": "A",
    "models": {
        "explicit": {
            "var_5pct": -8.703089046477827,
            "es_5pct": -11.521378517150879,
            "volatility": 5.426090717315674
        },
        "zero": {
            "var_5pct": -14.850619411468506,
            "es_5pct": -17.03839111328125,
            "volatility": 9.887288093566895
        }
    },
    "pairwise_comparisons": {
        "explicit_vs_zero": {
            "dm_mse_pvalue": 0.0010936683500817335,
            "dm_mae_pvalue": 2.111033252649719e-06,
            "mse_ratio": 9.240641253285390
        }
    }
}
```

### Key Findings Metrics
- **var_5pct**: 5% Value at Risk
- **es_5pct**: 5% Expected Shortfall  
- **volatility**: Standard deviation of path returns
- **dm_mse_pvalue**: Diebold-Mariano MSE test p-value
- **dm_mae_pvalue**: Diebold-Mariano MAE test p-value
- **mse_ratio**: Ratio of MSE losses (model2/model1)

### Controllability Insights (Experiment B)
```json
{
    "controllability_insights": {
        "var_5pct_diff_pct": 41.4,
        "more_conservative": "explicit"
    }
}
```

## Command Line Arguments

Both evaluators share similar argument structures:

| Argument | Description | Default |
|----------|-------------|---------|
| `--window` | Stress/test window | `covid_crash` |
| `--checkpoints-dir` | Pre-COVID checkpoints directory | `checkpoints/precovid` |
| `--csv-file` | Returns data CSV file | `sp500_data.csv` |
| `--num-paths` | Number of sample paths | `1000` |
| `--seq-len` | Sequence length | `60` |
| `--seeds` | Random seeds | `[42]` |
| `--results-base` | Results base directory | `results/addons/period_slices` |
| `--experiment-name` | Experiment identifier | `A` or `B` |

## Integration Benefits

### 1. Complete Analysis in One Command
**Before**: 
```bash
python experiment_A_evaluator.py --window covid_crash --num-paths 1000
python metrics_runner.py --experiments A --windows covid_crash  
python plotting_runner.py --experiments A --windows covid_crash
# Manual analysis of results
```

**After**:
```bash
python experiment_A_evaluator_v2.py --window covid_crash --num-paths 1000
# Everything generated automatically!
```

### 2. Consistent Data Handling
- Automatic real data slicing for each window
- Consistent CSV format for metrics calculation
- Automatic cleanup of temporary files

### 3. Robust Error Handling
- Continues pipeline even if individual stages fail
- Clear logging of success/failure for each stage
- Partial results still useful for analysis

### 4. Publication-Ready Output
- Professional plots in both PDF and PNG
- Structured metrics tables
- Compact findings for reporting
- Complete audit trail via manifests

## Example Usage

### Complete Stress Testing Analysis

```bash
# Run complete Experiment A pipeline
python experiment_A_evaluator_v2.py \
    --window covid_crash \
    --num-paths 2000 \
    --csv-file sp500_data.csv

# Results automatically available:
# - Samples: results/.../A_v*/covid_crash/*/samples.npy
# - Metrics: results/.../A_v*/covid_crash/metrics.json
# - Plots: results/.../A_v*/covid_crash/figs/*.pdf
# - Findings: results/.../A_v*/findings.jsonl
```

### Controllability Testing with Analysis

```bash
# Run complete Experiment B pipeline
python experiment_B_evaluator_v2.py \
    --window covid_crash \
    --num-paths 1000 \
    --csv-file sp500_data.csv

# Results include all modes:
# - Real-conditions, Calm-conditions, LLM-knob variations
# - Controllability insights and summaries
# - Complete analysis pipeline
```

### Load and Analyze Results

```python
import json
import pandas as pd

# Load findings
findings = []
with open('results/addons/period_slices/A_v8/findings.jsonl', 'r') as f:
    for line in f:
        findings.append(json.loads(line))

# Extract key insights
for finding in findings:
    window = finding['window_id']
    print(f"Window: {window}")
    
    # Model risk metrics
    for model, metrics in finding['models'].items():
        var_5 = metrics['var_5pct']
        es_5 = metrics['es_5pct']
        print(f"  {model}: VaR(5%)={var_5:.3f}, ES(5%)={es_5:.3f}")
    
    # Model comparisons
    for comparison, stats in finding['pairwise_comparisons'].items():
        dm_p = stats['dm_mse_pvalue']
        mse_ratio = stats['mse_ratio']
        print(f"  {comparison}: DM p-value={dm_p:.4f}, MSE ratio={mse_ratio:.2f}")

# Load integrated summary
with open('results/addons/period_slices/A_v8/integrated_summary.json', 'r') as f:
    summary = json.load(f)

print(f"Pipeline success rate: {summary['integrated_pipeline_summary']['metrics_success_rate']:.0%}")
```

## Comparison with Base Evaluators

| Feature | Base Evaluators | v2 Integrated Evaluators |
|---------|-----------------|---------------------------|
| **Sample Generation** | ✅ Full functionality | ✅ Same + integrated pipeline |
| **Metrics Calculation** | ❌ Manual step | ✅ Automatic |
| **Plot Generation** | ❌ Manual step | ✅ Automatic |
| **Findings Extraction** | ❌ Manual analysis | ✅ Automatic compact format |
| **Error Handling** | ❌ Stops on failure | ✅ Continues with partial results |
| **Publication Ready** | ❌ Requires manual assembly | ✅ Complete output |
| **Audit Trail** | ✅ Basic manifests | ✅ Complete pipeline tracking |

## Performance Considerations

### Execution Time
- **Sample Generation**: ~2-5 minutes (same as base)
- **Metrics Calculation**: ~30-60 seconds
- **Plot Generation**: ~30-60 seconds  
- **Findings Extraction**: ~5-10 seconds
- **Total Pipeline**: ~3-7 minutes per window

### Storage Requirements
- **Samples**: ~2-5MB per model (same as base)
- **Metrics**: ~100-500KB per window
- **Plots**: ~1-3MB per window (PDF + PNG)
- **Total Overhead**: ~2-5MB per window

### Memory Usage
- **Peak Usage**: ~200-500MB during sample generation
- **Steady State**: ~50-100MB for metrics/plotting
- **Scales Linearly**: With number of paths and sequence length

## Troubleshooting

### Common Issues

#### Pipeline Stage Failures
```
Integrated Pipeline Results:
  Metrics Success: False
  Plotting Success: True
  Findings Extracted: False
```

**Solution**: Check logs for specific stage errors. Pipeline continues even with failures.

#### Metrics File Not Found
```
WARNING - No metrics file found for covid_crash
```

**Solution**: The metrics are created in base experiment directories (A, B) not versioned ones (A_v8). The v2 evaluators automatically handle this.

#### Missing Real Data
```
WARNING - Returns file not found: sp500_data.csv, using synthetic data
```

**Solution**: Provide the correct path to your returns CSV file with `--csv-file`.

#### Memory Issues
```bash
# Reduce sample size for testing
python experiment_A_evaluator_v2.py --num-paths 500 --seq-len 30
```

### Debugging Pipeline Issues

```bash
# Check what was actually generated
ls -la results/addons/period_slices/A_v*/covid_crash/

# Check metrics file location
find results/addons/period_slices -name "metrics.json" -path "*/covid_crash/*"

# Check plotting results
ls -la results/addons/period_slices/A_v*/covid_crash/figs/

# Examine findings
cat results/addons/period_slices/A_v*/findings.jsonl | jq .
```

## Best Practices

1. **Start Small**: Use `--num-paths 100` for initial testing
2. **Check Dependencies**: Ensure metrics and plotting modules are available
3. **Monitor Logs**: Watch for warnings about missing data or failed stages
4. **Validate Results**: Check that findings make economic sense
5. **Compare Versions**: Use findings.jsonl to compare across runs
6. **Archive Results**: Pipeline generates comprehensive results for reproducibility

## Future Enhancements

### Potential Extensions
- **Multi-Window Batching**: Process multiple windows in one command
- **Custom Metrics**: User-defined metrics in the pipeline
- **Interactive Dashboards**: Web-based result exploration
- **Automated Reporting**: LaTeX/PDF report generation
- **Model Comparison**: Cross-experiment model evaluation
- **Real-Time Updates**: Streaming analysis capabilities

The integrated pipeline v2 provides a comprehensive, production-ready analysis workflow that transforms the manual, multi-step process into a single, robust command that generates publication-quality results with complete audit trails and error handling.
