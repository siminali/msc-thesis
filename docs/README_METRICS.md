# Shared Metrics & Tests Module

Comprehensive evaluation metrics for financial time series models including risk backtests, statistical tests, and distribution analysis.

## Features

- **Risk Backtests**: VaR/ES at 95%/99%, Kupiec POF, Christoffersen independence tests
- **Per-time Quantile Loss**: At α∈{1%, 5%} confidence levels
- **Diebold-Mariano Tests**: With HAC (lag=h-1) and HLN small-sample correction
- **Distribution Analysis**: ECDFs, QQ tails, skewness, kurtosis, realized volatility tracking
- **Comprehensive Output**: JSON metrics and CSV tables
- **Versioning Safety**: Creates _v2.py files to avoid overwrites

## Quick Start

### List Available Experiments

```bash
python metrics_runner.py --list-available
```

Output:
```
Available Experiments and Windows:
========================================
A_v2:
  - covid_crash

A:
  - covid_crash

B:
  - covid_crash
```

### Run All Metrics

```bash
# Process all experiments and windows
python metrics_runner.py --csv-file sp500_data.csv

# Process specific experiments
python metrics_runner.py --experiments B --csv-file sp500_data.csv

# Process specific windows
python metrics_runner.py --windows covid_crash --csv-file sp500_data.csv
```

## Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--csv-file` | CSV file with real market data | `sp500_data.csv` |
| `--results-base` | Base directory with experiment results | `results/addons/period_slices` |
| `--experiments` | Specific experiments to process | All available |
| `--windows` | Specific windows to process | All available |
| `--list-available` | List available experiments and exit | - |

## Metrics Calculated

### Risk Backtesting Metrics

#### Value at Risk (VaR) and Expected Shortfall (ES)
- **VaR 1%/5%**: Percentile-based risk measures
- **ES 1%/5%**: Conditional VaR (mean of losses beyond VaR)
- **Bootstrap CIs**: Confidence intervals for ES estimates

#### Kupiec Proportion of Failures (POF) Test
Tests if VaR violation rate matches theoretical rate:
- **H₀**: Violation rate = α (e.g., 5%)
- **Test Statistic**: Likelihood ratio test
- **Output**: LR statistic, p-value, rejection decision

#### Christoffersen Independence Test
Tests if VaR violations are independent:
- **H₀**: Violations are independent
- **Test Statistic**: Tests clustering of violations
- **Output**: LR statistic, p-value, transition probabilities

### Quantile Loss Evaluation

Per-time quantile loss for forecast accuracy:
```
QL(α) = E[(y - q̂ₐ)(α - 𝟙{y ≤ q̂ₐ})]
```

Where:
- `y`: Actual returns
- `q̂ₐ`: Predicted α-quantile
- `α`: Confidence level (1%, 5%)

### Diebold-Mariano Tests

Pairwise forecast comparison with corrections:

#### Test Statistic
```
DM = d̄ / √(Var̂(d̄)/n)
```

Where `d̄` is mean loss differential.

#### Corrections Applied
- **HAC Variance**: Newey-West with lag h-1
- **HLN Correction**: Harvey-Leybourne-Newbold small-sample adjustment
- **Loss Functions**: MSE and MAE

### Distribution Analysis

#### Empirical CDF Comparison
- **Kolmogorov-Smirnov Test**: Tests if distributions are equal
- **Test Statistic**: Maximum difference between ECDFs
- **Output**: KS statistic, p-value

#### QQ Tail Analysis
Quantile-quantile comparison at tail percentiles:
- **Tail Quantiles**: 1%, 5%, 95%, 99%
- **Metrics**: Absolute and relative differences
- **Purpose**: Assess tail behavior accuracy

#### Moment Analysis
Statistical moment comparison:
- **First 4 Moments**: Mean, standard deviation, skewness, kurtosis
- **Relative Errors**: Percentage differences
- **Normality Tests**: Jarque-Bera statistics

#### Realized Volatility Tracking
Rolling volatility pattern matching:
- **RMSE**: Root mean squared error
- **MAPE**: Mean absolute percentage error  
- **Correlation**: Pearson correlation coefficient
- **Window**: 20-day rolling volatility

## Output Structure

```
results/addons/period_slices/<experiment>/<window>/
├── metrics.json                    # Complete metrics results
└── tables/
    ├── model_comparison.csv         # Summary statistics by model
    └── pairwise_comparisons.csv     # Diebold-Mariano results
```

### metrics.json Structure

```json
{
    "window_id": "covid_crash",
    "experiment": "B",
    "metadata": {
        "calculated_at": "2025-08-30T18:40:00",
        "version": "1.0"
    },
    "real_data_stats": {
        "n_observations": 100,
        "mean": 0.0004,
        "std": 0.015,
        "skewness": -0.5,
        "kurtosis": 3.2
    },
    "models": {
        "explicit": {
            "basic_stats": {...},
            "risk_metrics": {...},
            "quantile_metrics": {...},
            "distribution_metrics": {...},
            "volatility_metrics": {...}
        },
        "llm": {...}
    },
    "pairwise_comparisons": {
        "explicit_vs_llm": {
            "diebold_mariano_tests": {...},
            "individual_losses": {...}
        }
    }
}
```

## Example Usage

### 1. Complete Evaluation

```bash
# Run comprehensive metrics on all experiments
python metrics_runner.py --csv-file sp500_data.csv
```

Expected output:
```
METRICS EVALUATION SUMMARY
============================================================
Status: completed
Results Base: results/addons/period_slices
Total Windows Processed: 3

A_v2:
  covid_crash: success (3 models)

A:
  covid_crash: success (3 models)

B:
  covid_crash: success (2 models)

Errors: 0
Warnings: 2
```

### 2. Focused Analysis

```bash
# Analyze only Experiment B (controllability results)
python metrics_runner.py --experiments B --csv-file sp500_data.csv
```

### 3. Custom Data

```bash
# Use custom data file
python metrics_runner.py --csv-file my_returns.csv --results-base my_results
```

## Interpreting Results

### Risk Metrics Interpretation

#### VaR/ES Results
```json
"risk_metrics": {
    "var_0.050": -0.156,        // 5% VaR
    "es_0.050": -0.243,         // 5% ES
    "es_bootstrap_0.050": {     // Bootstrap CI
        "ci_lower": -0.287,
        "ci_upper": -0.201
    }
}
```

**Interpretation**:
- 5% of returns are worse than -15.6%
- Expected loss given 5% worst outcomes: -24.3%
- 95% confidence: ES between -28.7% and -20.1%

#### Kupiec POF Test
```json
"violation_rate": 0.048,        // Actual violation rate
"expected_rate": 0.050,         // Theoretical rate
"p_value": 0.823,              // Test p-value
"reject_h0": false             // Accept H₀: rate = 5%
```

**Interpretation**: VaR model is well-calibrated (doesn't reject H₀).

### Diebold-Mariano Tests

```json
"dm_mse": {
    "dm_statistic_hln": -2.15,  // HLN-corrected statistic
    "p_value": 0.032,           // Two-tailed p-value
    "reject_h0_5pct": true      // Model 2 significantly better
}
```

**Interpretation**: Model 2 has significantly better MSE than Model 1.

### Distribution Analysis

#### Moment Comparison
```json
"moment_analysis": {
    "skewness": {
        "real": -0.523,         // Real data skewness
        "generated": -0.481,    // Generated skewness
        "relative_error": 8.0   // 8% difference
    }
}
```

**Interpretation**: Generated data captures asymmetry well (8% error).

#### Realized Volatility Tracking
```json
"volatility_metrics": {
    "rmse": 0.0034,            // Root mean squared error
    "mape": 12.5,              // Mean absolute percentage error
    "correlation": 0.847       // Correlation coefficient
}
```

**Interpretation**: Strong volatility pattern matching (84.7% correlation, 12.5% MAPE).

## Advanced Usage

### Load and Analyze Results

```python
import json
import pandas as pd

# Load metrics results
with open('results/addons/period_slices/B/covid_crash/metrics.json', 'r') as f:
    metrics = json.load(f)

# Extract risk metrics
models = metrics['models']
for model_name, model_results in models.items():
    risk_metrics = model_results['risk_metrics']
    var_5 = risk_metrics['var_0.050']
    es_5 = risk_metrics['es_0.050']
    print(f"{model_name}: VaR(5%)={var_5:.3f}, ES(5%)={es_5:.3f}")

# Load comparison tables
df_models = pd.read_csv('results/addons/period_slices/B/covid_crash/tables/model_comparison.csv')
df_dm = pd.read_csv('results/addons/period_slices/B/covid_crash/tables/pairwise_comparisons.csv')

print("Model Comparison:")
print(df_models[['model', 'var_5pct', 'es_5pct', 'volatility']])

print("\nDiebold-Mariano Tests:")
print(df_dm[['comparison', 'dm_mse_pvalue', 'dm_mae_pvalue']])
```

### Custom Analysis

```python
from shared_metrics import MetricsCalculator, RiskMetrics

# Create calculator
calculator = MetricsCalculator()

# Load your data
real_returns = np.load('real_data.npy')
model_samples = {
    'model_a': np.load('samples_a.npy'),
    'model_b': np.load('samples_b.npy')
}

# Calculate metrics
results = calculator.calculate_all_metrics(real_returns, model_samples, 'custom_window')

# Risk backtest example
var_5 = RiskMetrics.value_at_risk(real_returns, 0.05)
es_5 = RiskMetrics.expected_shortfall(real_returns, 0.05)
print(f"Real data: VaR(5%)={var_5:.3f}, ES(5%)={es_5:.3f}")
```

## Model Performance Assessment

### Risk Model Quality
1. **Well-calibrated VaR**: Kupiec POF p-value > 0.05
2. **Independent violations**: Christoffersen p-value > 0.05  
3. **Accurate ES**: Narrow bootstrap confidence intervals
4. **Good quantile forecasts**: Low quantile loss values

### Forecast Accuracy
1. **Superior forecasting**: DM test p-value < 0.05
2. **Loss function matters**: Check both MSE and MAE
3. **Economic significance**: Loss ratios >> 1.0

### Distribution Matching
1. **Overall distribution**: KS test p-value > 0.05
2. **Tail behavior**: Small relative errors in extreme quantiles
3. **Moment matching**: Relative errors < 10% for skewness/kurtosis
4. **Volatility clustering**: High correlation (> 0.8) in realized vol

## Troubleshooting

### Common Issues

#### No Sample Files Found
```
Warning: No sample files found for A/covid_crash
```
**Solution**: Check that experiments have been run and samples.npy files exist.

#### Failed to Load Samples
```
Error: Failed to load samples for model X
```
**Solution**: Verify samples.npy files are valid numpy arrays.

#### Insufficient Data
```
Warning: No real data found for window covid_crash
```
**Solution**: Ensure CSV file covers the window period (2020-02-20 to 2020-04-01).

### Memory Issues

For large sample sets:
```bash
# Process one experiment at a time
python metrics_runner.py --experiments A --csv-file data.csv
python metrics_runner.py --experiments B --csv-file data.csv
```

### Custom Windows

Add new windows to `get_window_real_data()` in `metrics_runner.py`:
```python
window_periods = {
    'covid_crash': ('2020-02-20', '2020-04-01'),
    'my_custom_window': ('2023-01-01', '2023-03-31')  # Add here
}
```

## Integration with Experiments

### Experiment A (Stress Testing)
- **Input**: Real-conditions samples from all models
- **Focus**: Out-of-sample risk assessment
- **Key Metrics**: VaR/ES violations, distribution shifts

### Experiment B (Controllability)
- **Input**: Real-conditions mode samples  
- **Focus**: Baseline model performance
- **Key Metrics**: Model comparisons, forecast accuracy

### Future Extensions
- **Experiment C**: Temporal analysis with different windows
- **Experiment D**: Multi-horizon forecasting evaluation
- **Custom Metrics**: Domain-specific risk measures

## Best Practices

1. **Run After Experiments**: Always run metrics after generating samples
2. **Check Warnings**: Review warnings for data quality issues  
3. **Multiple Windows**: Compare performance across different market conditions
4. **Statistical Significance**: Use p-values < 0.05 for strong evidence
5. **Economic Significance**: Consider practical importance, not just statistical
6. **Bootstrap CIs**: Use for robust uncertainty quantification
7. **Model Selection**: Combine multiple metrics for comprehensive assessment

The shared metrics module provides a robust foundation for comprehensive model evaluation, enabling rigorous assessment of financial time series generation quality across multiple dimensions.
