# Fresh Evaluation Pipeline - Complete Documentation

## Overview

A comprehensive, self-contained evaluation pipeline that regenerates all plots and metrics from scratch for financial diffusion models. Built with strict inverse scaling validation, sanity gates, and publication-quality outputs.

## 🚀 Quick Start

```bash
python tools/evaluate_models_from_scratch.py \
    --real data/sp500_data.csv \
    --models zero explicit llm \
    --checkpoints \
        "results/zero_conditioned/*/checkpoints/best_model.pth" \
        "results/explicit_conditioned/*/checkpoints/best_model.pth" \
        "results/llm_conditioned/*/checkpoints/best_model.pth" \
    --windows \
        "PreCOVID:2017-01-01,2019-12-31" \
        "COVID:2020-02-01,2021-06-30" \
        "PostCOVID:2022-01-01,2023-12-31" \
    --seq-len 60 \
    --outdir results/fresh_evaluation \
    --report-out results/fresh_evaluation/comprehensive_report.pdf \
    --allow-sanity-bypass \
    --pbar --pbar-leave 2
```

## 📋 CLI Arguments

### Required Arguments
- `--real PATH`: CSV with date,close or date,return columns
- `--models {zero,explicit,llm}`: Model types to evaluate
- `--windows "Name:start,end"`: Time windows for evaluation

### Model Configuration
- `--checkpoints PATHS`: Model checkpoint paths (supports globs)
- `--seq-len INT`: Sequence length for models (default: 60)

### Output Configuration
- `--outdir PATH`: Output directory for figures/tables
- `--report-out PATH`: Final PDF report path

### Scaling & Validation
- `--force-inverse-scaling`: Force decimal returns (default: true)
- `--annualise-vol {none,sqrt252}`: Volatility annualization
- `--sanity-std-bounds "min,max"`: Std dev bounds (default: "0.005,0.05")
- `--sanity-absmax FLOAT`: Max absolute return (default: 0.5)
- `--allow-sanity-bypass`: Allow failed sanity checks with warnings

### Progress Tracking
- `--pbar`: Show progress bars
- `--pbar-leave INT`: Progress bar persistence level

## 🏗️ Architecture

### Core Components

1. **`utils/scaling_guard.py`**: Single source of truth for inverse scaling
   - `ReturnsBundle` dataclass with metadata
   - `detect_scaler()` for automatic scale detection
   - `@require_inverse_scaled_data` decorator
   - `inverse_returns()` with validation

2. **`utils/sanity_gate.py`**: Data validation framework
   - `SanityThresholds` configuration
   - `SanityGate.validate()` with bypass options
   - Realistic bounds for daily financial returns

3. **`utils/fresh_plots.py`**: Publication-quality plotting
   - 8 essential figure types with suspect scale tagging
   - Consistent aesthetics and axis scaling
   - NaN/Inf value handling

4. **`utils/fresh_metrics.py`**: Comprehensive financial metrics
   - Basic statistics, tail risk, stylized facts
   - Backtesting (Kupiec, Christoffersen tests)
   - Model comparisons with correlation analysis

5. **`tools/evaluate_models_from_scratch.py`**: Main orchestrator
   - CLI argument parsing and validation
   - Model loading with architecture auto-detection
   - Pipeline coordination with progress tracking

### Data Flow

```
Real CSV → ReturnsBundle → Sanity Gate → Figures
    ↓
Model Checkpoints → Sample Generation → Inverse Scaling → ReturnsBundle → Sanity Gate → Figures
    ↓
All Bundles → Comprehensive Metrics → CSV/LaTeX Tables → PDF Report
```

## 📊 Generated Outputs

### Figures (PDF + PNG)
- **Histograms**: Log-y scale with Gaussian overlays and kurtosis
- **QQ Plots**: Left/right tail analysis with identical axes
- **ACF/PACF**: Returns & squared returns with 95% confidence bands
- **Residuals**: Standardized with N(0,1) overlay and KS test
- **Rolling Volatility**: Overlays and ratios with correlation stats
- **VaR/ES Curves**: Risk metrics across confidence levels [90%-99%]
- **Exceedance Timeline**: VaR breach detection and counting
- **Density/ECDF**: Multi-model overlays with sanity table

### Metrics Tables (CSV + LaTeX)
- **Basic Statistics**: Mean, std, skewness, kurtosis, extremes
- **Tail Risk**: VaR95/99, ES95/99 with confidence intervals
- **Stylized Facts**: Fat tails, leverage effect, volatility clustering
- **Backtesting**: Kupiec/Christoffersen tests with p-values
- **Model Comparisons**: Volatility correlations and ratios

### PDF Report
- **Run Summary**: CLI args, model loading status, sanity decisions
- **Figure References**: Organized by window with file paths
- **Backup Management**: Auto-backup existing reports

## 🔧 Model Architecture Support

The pipeline auto-detects and loads:

### Zero-Conditioned Models
```python
from src.explicit_cond_ddpm import ExplicitConditioningDDPM, ExplicitConditioningTrainer
# Uses zero conditioning vectors of appropriate dimension
```

### Explicit-Conditioned Models
```python
from src.explicit_cond_ddpm import ExplicitConditioningDDPM, ExplicitConditioningTrainer
# Uses market statistics: [mean, std, skew, kurtosis, min]
```

### LLM-Conditioned Models
```python
from src.llm_conditioned_diffusion_refactored import LLMConditionedDiffusion, LLMDiffusionTrainer
# Uses text embedding vectors from news/sentiment data
```

## ⚠️ Sanity Gate System

### Purpose
Ensures all inverse-scaled data represents realistic daily financial returns in decimal units.

### Validation Criteria
- **Standard Deviation**: 0.5% - 5% daily (configurable)
- **Absolute Maximum**: ≤50% single-day move (configurable)
- **Units**: Decimal returns (not percentages)

### Failure Handling
- **Strict Mode**: Fail fast on violations
- **Bypass Mode**: Continue with "SUSPECT SCALE" warnings
- **Visual Tagging**: Failed data marked in plots and tables

### Example Output
```
[WARNING] SanityGate FAIL for explicit/PreCOVID: mean=0.581015, std=2.806718, 
min=-32.882, max=25.331, kurtosis=8.16, scaler=explicit_scaler, kind=returns, 
annualise=none; thresholds std∈[0.005,0.05], absmax≤0.5. Likely causes: 
missing inverse_transform, wrong units (percent), or using prices instead of returns.
```

## 🎯 Best Practices

### 1. Model Checkpoints
- Ensure checkpoints match expected architectures
- Use absolute paths or proper globs
- Verify conditioning dimensions align with training

### 2. Time Windows
- Choose meaningful economic periods
- Ensure sufficient data points (>100 observations)
- Consider market regime changes

### 3. Scaling Validation
- Always review sanity gate outputs
- Investigate failures before bypassing
- Check that returns are in decimal units

### 4. Output Organization
- Use descriptive output directory names
- Include timestamps for reproducibility
- Backup important reports before re-running

## 🐛 Troubleshooting

### Model Loading Failures
```
Error(s) in loading state_dict: size mismatch for conditioning_proj.weight
```
**Solution**: Check conditioning dimensions. All models must use `conditioning_dim=5`.

### NaN/Inf in Plots
```
ValueError: Axis limits cannot be NaN or Inf
```
**Solution**: Enable sanity bypass and check inverse scaling. The pipeline now handles this gracefully.

### Memory Issues
```
CUDA out of memory
```
**Solution**: Reduce `num_samples` in `generate_model_samples()` or use CPU-only mode.

### Dimension Mismatches
```
x and y must have same first dimension
```
**Solution**: Check that model samples and real data are properly aligned. The pipeline trims to match lengths.

## 🔄 Extending the Pipeline

### Adding New Plot Types
1. Create function in `utils/fresh_plots.py`
2. Add `@require_inverse_scaled_data` decorator
3. Handle suspect scale tagging
4. Update `_create_all_figures()` in main CLI

### Adding New Metrics
1. Create computation function in `utils/fresh_metrics.py`
2. Add table creation function
3. Update `save_metrics_tables()` call
4. Ensure proper NaN/Inf handling

### Supporting New Model Types
1. Add loading function to main CLI
2. Handle conditioning vector creation
3. Update dummy model fallbacks
4. Test with representative checkpoints

## 📚 Dependencies

```python
# Core scientific computing
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0

# Deep learning
torch>=1.9.0

# Plotting and visualization  
matplotlib>=3.4.0
seaborn>=0.11.0

# Statistics and time series
statsmodels>=0.12.0

# Progress tracking
tqdm>=4.61.0

# Path handling
pathlib  # Built-in
```

## 🏆 Success Criteria

A successful run should:
- ✅ Load all specified models without dummy fallbacks
- ✅ Generate realistic samples (pass sanity gates)
- ✅ Create all 8 figure types without errors
- ✅ Compute comprehensive metrics for all models
- ✅ Produce publication-ready PDF report
- ✅ Complete within reasonable time (<30 min for 3 models × 3 windows)

## 📄 License & Citation

This pipeline was developed for financial diffusion model evaluation. When using this code, please cite:

```bibtex
@software{fresh_evaluation_pipeline,
  title={Fresh Evaluation Pipeline for Financial Diffusion Models},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo/fresh-evaluation-pipeline}
}
```
