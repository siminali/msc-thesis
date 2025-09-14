# Fresh Plotting Pipeline

A comprehensive plotting pipeline that regenerates all required figures from scratch, without reusing any cached arrays or legacy figures. This tool loads real data and model checkpoints, generates fresh synthetic sequences, applies inverse scaling, recomputes all metrics, renders figures, and assembles a clean PDF report.

## Features

### Core Functionality
- **Fresh Generation**: No cached data or legacy figures - everything computed from scratch
- **Inverse Scaling Pipeline**: Proper scaling detection and inversion with sanity gates
- **Comprehensive Plotting**: 8 different figure types with consistent aesthetics
- **Rich Metrics**: Statistical tests, risk metrics, stylized facts, and model comparisons
- **Progress Tracking**: Rich progress bars with tqdm for all operations
- **PDF Report Generation**: Clean, professional PDF with all figures and summary

### Supported Models
- **Zero-conditioned**: Basic diffusion models without conditioning
- **Explicit-conditioned**: Models with explicit statistical conditioning
- **LLM-conditioned**: Models with large language model embeddings as conditioning

### Figure Types Generated
1. **Histograms**: Per-model histograms with log-y axis, Gaussian overlays, and kurtosis statistics
2. **QQ Plots**: Left and right tail QQ plots with identical axes across models
3. **ACF/PACF**: Autocorrelation functions for returns and squared returns with confidence bands
4. **Residuals**: Standardized residuals with N(0,1) overlay and error metrics
5. **Rolling Volatility**: Overlays and ratios with consistent y-limits starting at 0
6. **VaR/ES Curves**: Value-at-Risk and Expected Shortfall curves across confidence levels
7. **Exceedance Timeline**: VaR breach detection with expected vs observed counts
8. **Density/ECDF**: Optional probability density and empirical CDF overlays

### Metrics Computed
- **Tail Risk**: VaR and ES at multiple confidence levels (95%, 99%)
- **Backtesting**: Kupiec POF and Christoffersen independence tests
- **Residual Statistics**: ME, MAE, MSE, RMSE with standardized residuals
- **Volatility Analysis**: Rolling volatility statistics and correlations
- **Stylized Facts**: Skewness, kurtosis, leverage effects, volatility clustering
- **Model Comparisons**: Diebold-Mariano tests for VaR pinball loss

## Installation & Requirements

### Dependencies
```bash
pip install numpy pandas matplotlib scipy statsmodels scikit-learn tqdm torch
```

### Optional Dependencies (for model loading)
```bash
pip install transformers  # For LLM-conditioned models
```

## Usage

### Basic Usage

```bash
python tools/plots_from_scratch.py \
    --real data/sp500_data.csv \
    --models zero explicit llm \
    --checkpoints "results/*/checkpoints/*.pth" \
    --windows "Calm:2017-01-01,2019-12-31" "COVID:2020-02-01,2020-04-30" "Post:2021-01-01,2022-12-31" \
    --seq-len 60 \
    --outdir results/novelty_comparison/plots_fresh \
    --report-out results/novelty_comparison/latest_final_report.pdf
```

### Advanced Configuration

```bash
python tools/plots_from_scratch.py \
    --real data/sp500_data.csv \
    --models zero explicit llm \
    --checkpoints "results/zero_conditioned/*/checkpoints/best_model.pth" \
    --windows "Calm:2017-01-01,2019-12-31" "COVID:2020-02-01,2020-04-30" \
    --seq-len 60 \
    --outdir results/fresh_plots \
    --report-out results/fresh_report.pdf \
    --annualise-vol sqrt252 \
    --sanity-std-bounds "0.01,0.08" \
    --sanity-absmax 0.3 \
    --allow-sanity-bypass \
    --pbar --pbar-leave
```

## Command Line Arguments

### Required Arguments
- `--real PATH`: Path to CSV file with 'date,close' or 'date,return' columns
- `--models`: List of models to process: `zero`, `explicit`, `llm`
- `--windows`: Window specifications like `"Name:start_date,end_date"`

### Optional Arguments
- `--checkpoints PATHS`: Paths to model checkpoints or glob patterns
- `--seq-len INT`: Sequence length for models (default: 60)
- `--outdir PATH`: Output directory (default: `results/novelty_comparison/plots_fresh`)
- `--report-out PATH`: PDF report path (default: `results/novelty_comparison/latest_final_report.pdf`)

### Processing Options
- `--force-inverse-scaling`: Force inverse scaling (default: True)
- `--annualise-vol {none,sqrt252}`: Volatility annualization (default: none)
- `--invalidate-cache`: Recompute everything from scratch (default: True)

### Sanity Gate Configuration
- `--sanity-std-bounds "MIN,MAX"`: Standard deviation bounds (default: "0.005,0.05")
- `--sanity-absmax FLOAT`: Absolute maximum threshold (default: 0.5)
- `--allow-sanity-bypass`: Allow bypassing sanity gate failures (default: False)

### Progress Bar Options
- `--pbar`: Show progress bars (default: True)
- `--pbar-update-interval INT`: Update interval (default: 1)
- `--pbar-leave`: Leave progress bars visible (default: False)

## Input Data Format

### Real Data CSV
The CSV file should contain either:

**Format 1: Prices**
```csv
date,close
2020-01-01,3200.50
2020-01-02,3250.75
...
```

**Format 2: Returns**
```csv
date,return
2020-01-01,0.0156
2020-01-02,-0.0089
...
```

### Window Specifications
Windows define time periods for analysis:
```bash
--windows "Calm:2017-01-01,2019-12-31" "COVID:2020-02-01,2020-04-30" "Recovery:2021-01-01,2022-12-31"
```

### Model Checkpoints
Checkpoints can be specified as:
- Exact paths: `path/to/model.pth`
- Glob patterns: `results/*/checkpoints/*.pth`
- Multiple paths: `model1.pth model2.pth model3.pth`

## Output Structure

```
outdir/
├── histogram_WindowName.pdf/.png          # Histograms with log-y axis
├── qq_plots_WindowName.pdf/.png           # QQ plots for tails
├── acf_pacf_WindowName.pdf/.png           # ACF/PACF plots
├── standardized_residuals_WindowName.pdf/.png  # Residuals analysis
├── rolling_volatility_WindowName.pdf/.png # Volatility analysis
├── var_es_curves_WindowName.pdf/.png      # Risk curves
├── exceedance_timeline_WindowName.pdf/.png # VaR backtesting
├── density_ecdf_WindowName.pdf/.png       # Density/ECDF plots
├── tail_metrics_WindowName.csv/.tex       # Risk metrics table
├── stylized_facts_WindowName.csv/.tex     # Statistical properties
└── dm_tests_WindowName.csv/.tex           # Model comparisons
```

## Sanity Gate System

The pipeline enforces strict sanity checks on inverse-scaled returns:

### Default Thresholds
- **Standard Deviation**: 0.005 ≤ σ ≤ 0.05 (reasonable for daily returns)
- **Absolute Maximum**: max|r| ≤ 0.5 (prevents extreme values)

### Behavior
- **Strict Mode** (`--allow-sanity-bypass=false`): Fails fast on violations
- **Permissive Mode** (`--allow-sanity-bypass=true`): Continues with warnings and tags suspect figures

### Common Failure Causes
1. Missing inverse transforms (data still scaled)
2. Wrong units (percentages instead of decimals)
3. Using prices instead of returns
4. Incorrect output_kind specification

## Error Handling & Debugging

### Common Issues

**1. Model Loading Failures**
```
Warning: Could not load checkpoint path/to/model.pth: ...
```
- Check checkpoint paths exist
- Verify model architecture compatibility
- Ensure PyTorch version compatibility

**2. Sanity Gate Failures**
```
SanityGateError: scale guard failed (std=0.234 not in [0.005,0.05])
```
- Use `--allow-sanity-bypass` to continue with warnings
- Check data preprocessing pipeline
- Verify inverse scaling implementation

**3. Data Loading Issues**
```
FileNotFoundError: Could not load real data from path/to/data.csv
```
- Verify file exists and is readable
- Check CSV format (date column, close/return column)
- Ensure proper date formatting

### Debugging Tips

1. **Test with Dummy Models**: Use `--allow-sanity-bypass` for initial testing
2. **Check Progress**: Use `--pbar --pbar-leave` for detailed progress tracking
3. **Inspect Outputs**: Check individual CSV/PNG files before PDF generation
4. **Validate Data**: Examine real data statistics in run summary

## Advanced Features

### Custom Scalers
The pipeline automatically detects scalers from model checkpoints. To add custom scalers:

1. Modify `utils/scaling_guard.py::detect_scaler`
2. Implement `inverse_returns` for your scaler type
3. Update `get_inverse_scaled_returns` logic

### Custom Metrics
Add new metrics in `utils/fresh_metrics.py`:

```python
@require_inverse_scaled_data
def compute_custom_metric(bundle: ReturnsBundle) -> Dict[str, float]:
    # Your custom metric computation
    return {'custom_metric': value}
```

### Custom Figure Types
Add new figures in `utils/fresh_plots.py`:

```python
@require_inverse_scaled_data
def create_custom_plot(real_bundle: ReturnsBundle, model_bundles: Dict[str, ReturnsBundle],
                      window_name: str, output_path: Path) -> None:
    # Your custom plotting code
    pass
```

## Performance Considerations

- **Memory Usage**: Large datasets may require chunking for rolling calculations
- **Computation Time**: Progress bars help track long-running computations
- **Disk Space**: Both PDF and PNG versions are saved (can be disabled)
- **Parallel Processing**: Currently single-threaded (future enhancement opportunity)

## Contributing

When contributing to the pipeline:

1. **Follow Decorators**: Use `@require_inverse_scaled_data` for all plotting/metrics functions
2. **Maintain Consistency**: Follow existing matplotlib style and figure layout patterns
3. **Add Progress Tracking**: Use `tqdm.write()` for status updates
4. **Handle Errors**: Graceful degradation with informative error messages
5. **Update Documentation**: Keep this README current with new features

## Validation & Acceptance Criteria

The pipeline passes acceptance when:

✅ **Fresh Generation**: All figures generated from raw model samples and real data  
✅ **Inverse Scaling**: Every plotted series uses decimal returns with proper scaling  
✅ **Realistic Ranges**: No absurd axis ranges (e.g., ±200 for daily returns)  
✅ **Appropriate Magnitudes**: VaR/ES values are small decimals suitable for daily data  
✅ **Consistent Scales**: Rolling volatility overlays share sensible y-limits from 0  
✅ **Proper Ratios**: Volatility ratios hover near 1 unless models genuinely differ  
✅ **Uniform Formatting**: QQ and ACF/PACF plots have consistent scales across models  
✅ **Clean Output**: Final PDF contains only fresh, correctly scaled outputs  
✅ **Progress Visibility**: Progress bars visible during execution  

## License

This pipeline is part of the thesis: "Diffusion Models in Generative AI for Financial Data Synthesis and Risk Management" by Simin Ali.

---

**Author**: Simin Ali  
**Institution**: [University Name]  
**Created**: January 2025  
**Last Updated**: January 2025
