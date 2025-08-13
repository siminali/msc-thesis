# Reproducing the Final Results Thesis PDF

## Prerequisites

- Python 3.8+
- Required packages (see requirements.txt)
- Access to the data files specified below

## Data Requirements

- `data/sp500_data.csv`: S&P 500 daily closing prices (2010-2024)
- `results/garch_returns.npy`: GARCH model synthetic returns
- `results/ddpm_returns.npy`: DDPM model synthetic returns  
- `results/timegrad_returns.npy`: TimeGrad model synthetic returns
- `results/llm_conditioned_returns.npy`: LLM-Conditioned model synthetic returns

## Single Command to Reproduce

```bash
python src/build_final_results.py
```

## Output Structure

```
final_results_thesis/
├── final_results_thesis.pdf          # Main PDF report
├── figures/                          # Vector PDF figures
├── tables/                           # CSV tables
├── metrics_summary.csv               # Consolidated metrics
├── evaluation_results.json           # Full evaluation results
├── build_log.txt                     # Build execution log
└── README_repro.md                   # This file
```

## What Gets Generated

1. **All evaluation metrics** computed from scratch
2. **All figures** as vector PDFs with consistent styling
3. **All tables** as CSV files with full precision
4. **Complete PDF report** with proper formatting and structure

## Performance Optimizations

- BLAS libraries configured for single-threaded operation
- Joblib parallelization for bootstrap computations
- Vectorized MMD computation with cached Gram matrices
- Precomputed statistics reused across comparisons
- Non-interactive matplotlib backend for faster plotting

## Troubleshooting

- Check `build_log.txt` for detailed execution information
- Ensure all input files exist and are accessible
- Verify Python environment has all required packages

## Contact

For questions about reproduction, contact [Your Contact Information]
