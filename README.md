# MSc Thesis: DIFFUSION MODELS FOR SYNTHETIC FINANCIAL RETURNS: CONTROLLABLE SCENARIO GENERATION & STRESS TESTING

**Author**: Simin Ali
**Supervisor**: Dr Mikael Mieskolainen
**Institution**: Imperial College London
**Program**: MSc Artificial Intelligence Applications and Innovation
**Submission Date**: September 2025

## Project Overview
## Project Overview

> For how to run the project, see `USER_GUIDE.md` (quick start and essential workflows).


This repository contains the implementation and evaluation of models for financial data synthesis and risk management, including classical and multiple diffusion-based approaches. The project compares the following:

1. **GARCH(1,1)**: Classical volatility modeling baseline
2. **Zero‑Conditioned DDPM**: Unconditional diffusion baseline
3. **Explicit‑Conditioned DDPM**: Regime + volatility conditioning
4. **LLM‑Conditioned DDPM**: News‑embedding conditioned diffusion
5. **TimeGrad**: Autoregressive diffusion‑based forecasting

## Key Objectives
## Key Objectives

- Implement and evaluate diffusion models for financial time series generation
- Compare performance against classical GARCH models
- Generate comprehensive evaluation metrics and visualizations
- Provide reproducible results for thesis reporting
- Create automated LaTeX table generation for thesis inclusion

## Repository Structure
## Repository Structure

```
Thesis Coding/
├── README.md
├── USER_GUIDE.md
├── requirements/
│   ├── base.txt
│   └── llm_refactored.txt
├── scripts/
│   ├── training/                    # Training pipelines
│   ├── evaluation/                  # Metrics and plotting runners
│   ├── experiments/                 # Experiment A/B evaluators
│   └── exports/                     # LaTeX/report exporters
├── shared/                          # Shared modules
│   ├── metrics.py
│   └── plotting.py
├── src/                             # Model code and utilities
├── notebooks/                       # Jupyter notebooks
├── data/                            # Data files (sp500_data.csv)
├── results/                         # Evaluation results
├── runs/                            # Training/eval run artifacts
├── final_results_benchmarking/      # Final thesis results and figures
├── latex_training_exports/          # LaTeX exports for training
├── comprehensive_latex_exports/     # Comprehensive LaTeX exports
├── docs/                            # Documentation and changelogs
└── configs/                         # Config files
```

## Quick Start
## Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/siminali/msc-thesis.git
cd msc-thesis

# Create virtual environment (recommended)
python -m venv thesis_env
source thesis_env/bin/activate  # On Windows: thesis_env\Scripts\activate

# Install dependencies
pip install -r requirements/base.txt
```

### 2. Data Preparation

The project uses S&P 500 historical data. If you don't have the data file:

```python
import yfinance as yf
import pandas as pd

# Download S&P 500 data
sp500 = yf.download('^GSPC', start='2010-01-01', end='2024-12-31')
sp500.to_csv('data/sp500_data.csv')
```

### 3. Running the Evaluation

## Models at a Glance

- **GARCH(1,1)**: baseline for volatility modeling.
- **Zero‑Conditioned DDPM**: unconditional diffusion baseline.
- **Explicit‑Conditioned DDPM**: concatenates regime one‑hot + volatility scalar as conditioning.
- **LLM‑Conditioned DDPM**: conditions on daily news embeddings (cached under `cache/news_embeddings/`).
- **TimeGrad**: autoregressive diffusion for forecasting.

Final benchmark summaries are in `final_results_benchmarking/metrics_summary.csv` and figures under `final_results_benchmarking/figures/`.


#### Option A: Individual Model Evaluation

1. **Run GARCH notebook**:
   ```bash
   jupyter notebook notebooks/garch.ipynb
   ```

2. **Run DDPM notebook**:
   ```bash
   jupyter notebook notebooks/diffusion.ipynb
   ```

3. **Run TimeGrad notebook**:
   ```bash
   jupyter notebook notebooks/timegrad.ipynb
   ```

#### Option B: Scripted Pipeline

Run end‑to‑end evaluation and plotting:
```bash
python scripts/evaluation/run_comprehensive_evaluation.py
```

Or run components:
```bash
# Metrics
python scripts/evaluation/metrics_runner.py --results-base results/addons/period_slices

# Plotting
python scripts/evaluation/plotting_runner.py --results-base results/addons/period_slices
```


#### Option C: Experiment Evaluators / Notebooks

Scripted evaluators (integrated sampling + metrics + plots):
```bash
python scripts/experiments/experiment_A_evaluator_v2.py
python scripts/experiments/experiment_B_evaluator_v2.py
```

Or use the notebooks for exploratory runs:
```bash
jupyter notebook notebooks/comprehensive_evaluation.ipynb
```

## Evaluation Framework
## Evaluation Framework

The comprehensive evaluation framework provides:

### Metrics Computed

1. **Basic Statistics**
   - Mean, Standard Deviation, Skewness, Kurtosis
   - Min, Max, Quartiles

2. **Tail Risk Metrics**
   - Value at Risk (VaR) at 1%, 5%, 95%, 99% levels
   - Expected Shortfall (ES) at same levels

3. **Volatility Metrics**
   - Volatility clustering (ACF of squared returns)
   - Volatility persistence
   - Rolling volatility statistics

4. **Distribution Tests**
   - Kolmogorov-Smirnov test
   - Anderson-Darling test
   - Maximum Mean Discrepancy (MMD)

5. **VaR Backtesting**
   - Kupiec test for violation rate accuracy
   - Christoffersen test for independence of violations

### Outputs Generated

- **Plots**: High-resolution PDF/PNG files for thesis inclusion
- **Tables**: LaTeX-formatted tables ready for thesis reporting
- **Results**: JSON files for reproducibility
- **Summary**: Markdown reports with key findings

## Using Results in Your Thesis
## Using Results in Your Thesis

### LaTeX Tables

The evaluation framework generates LaTeX tables in `results/*/tables/`:

```latex
% Include in your thesis
\input{results/comprehensive_evaluation/tables/basic_statistics.tex}
\input{results/comprehensive_evaluation/tables/distribution_tests.tex}
\input{results/comprehensive_evaluation/tables/volatility_metrics.tex}
```

### Plots

High-resolution plots are saved in `results/*/plots/`:

```latex
% Include in your thesis
\includegraphics[width=0.8\textwidth]{results/comprehensive_evaluation/plots/distribution_comparison.pdf}
\includegraphics[width=0.8\textwidth]{results/comprehensive_evaluation/plots/volatility_clustering.pdf}
```

### Reproducibility

All results are saved as JSON files for complete reproducibility:

```python
import json
with open('results/comprehensive_evaluation/evaluation_results.json', 'r') as f:
    results = json.load(f)
```

## Customization
## Customization

### Adding New Models

To add a new model to the evaluation framework:

1. Implement your model in a new notebook
2. Save results using the standard format:
   ```python
   np.save('../results/your_model_returns.npy', your_synthetic_data)
   ```
3. Add your model to the evaluation framework:
   ```python
   evaluator = FinancialModelEvaluator(model_names=['GARCH', 'DDPM', 'TimeGrad', 'YourModel'])
   ```

### Custom Metrics

The evaluation framework is modular. Add custom metrics by extending the `FinancialModelEvaluator` class:

```python
def compute_custom_metric(self, data, model_name):
    """Compute your custom metric."""
    # Your implementation here
    return {'Model': model_name, 'CustomMetric': value}
``` 

## Thesis Integration
## Supervisor Feedback Implementation

This evaluation framework addresses the supervisor's specific requirements:

 **Automated evaluation metrics** - Comprehensive statistical tests
 **LaTeX table generation** - Ready-to-use tables for thesis
 **Automated plotting** - Publication-ready figures
 **Reproducible results** - JSON exports and version control
 **Cross-model comparison** - Systematic evaluation across all models

## Thesis Integration

### Results Chapter Structure

1. **Model Implementation** - Technical details of each model
2. **Evaluation Methodology** - Description of metrics and tests
3. **Results and Discussion** - Use generated tables and plots
4. **Comparison Analysis** - Cross-model performance analysis
5. **Conclusions** - Key findings and recommendations

### Key Tables to Include

- Basic statistical measures comparison
- Distribution similarity tests
- Volatility metrics comparison
- VaR backtesting results
- Summary comparison table

### Key Plots to Include

- Distribution comparison histograms
- Time series sample comparisons
- Volatility clustering analysis
- QQ plots for normality assessment
- Autocorrelation function comparisons

## Troubleshooting
## Troubleshooting

### Common Issues

1. **Import errors**: Ensure you're in the correct directory and virtual environment
2. **Data not found**: Check that `data/sp500_data.csv` exists
3. **Memory issues**: Reduce batch sizes in model training
4. **CUDA errors**: Set `device = torch.device("cpu")` for CPU-only execution

### Getting Help

- Check the notebook outputs for error messages
- Verify all dependencies are installed: `pip install -r requirements/base.txt`
- Ensure data files are in the correct locations
- Check that evaluation results are being saved to the `results/` directory

## References
## References

- Ho, J., Jain, A., & Abbeel, P. (2020). Denoising Diffusion Probabilistic Models. NeurIPS.
- Cont, R. (2001). Empirical properties of asset returns: stylized facts and statistical issues. Quantitative Finance.
- Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity. Journal of Econometrics.

## Contact
## Contact

For questions about this implementation, contact:
- **Author**: Simin Ali
- **Supervisor**: Dr Mikael Mieskolainen
- **Institution**: Imperial College London

---

**Note**: This repository is part of an MSc thesis project. All code and results are for academic research purposes.


## License
## License

This project is licensed under the MIT License. See `LICENSE` for details.
