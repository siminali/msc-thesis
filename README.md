# MSc Thesis: Diffusion Models in Generative AI for Financial Data Synthesis and Risk Management

**Author**: Simin Ali  
**Supervisor**: Dr Mikael Mieskolainen  
**Institution**: Imperial College London  
**Program**: MSc Artificial Intelligence Applications and Innovation  
**Submission Date**: September 2025

## 📋 Project Overview

This repository contains the implementation and evaluation of diffusion models for financial data synthesis and risk management. The project compares three approaches:

1. **GARCH(1,1)**: Classical volatility modeling baseline
2. **DDPM**: Denoising Diffusion Probabilistic Model for unconditional synthetic return generation
3. **TimeGrad**: Autoregressive diffusion-based forecasting

## 🎯 Key Objectives

- Implement and evaluate diffusion models for financial time series generation
- Compare performance against classical GARCH models
- Generate comprehensive evaluation metrics and visualizations
- Provide reproducible results for thesis reporting
- Create automated LaTeX table generation for thesis inclusion

## 📁 Repository Structure

```
Thesis Coding/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── .gitignore                         # Git ignore rules
├── src/                               # Source code
│   ├── evaluation_framework.py        # Comprehensive evaluation framework
│   └── add_evaluation_to_notebooks.py # Script to add evaluation to notebooks
├── notebooks/                         # Jupyter notebooks
│   ├── garch.ipynb                    # GARCH(1,1) implementation
│   ├── diffusion.ipynb                # DDPM implementation
│   ├── timegrad.ipynb                 # TimeGrad implementation
│   └── comprehensive_evaluation.ipynb # Cross-model comparison
├── data/                              # Data files
│   └── sp500_data.csv                 # S&P 500 historical data
├── results/                           # Evaluation results
│   ├── garch_evaluation/              # GARCH evaluation outputs
│   ├── ddpm_evaluation/               # DDPM evaluation outputs
│   ├── timegrad_evaluation/           # TimeGrad evaluation outputs
│   └── comprehensive_evaluation/      # Cross-model comparison outputs
└── docs/                              # Documentation
    ├── background_report.pdf          # Background literature review
    └── intro chapter.pdf              # Introduction chapter
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/siminali/mcc-thesis.git
cd mcc-thesis

# Create virtual environment (recommended)
python -m venv thesis_env
source thesis_env/bin/activate  # On Windows: thesis_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
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

#### Option B: Automated Evaluation Setup

```bash
# Add evaluation cells to all notebooks automatically
python src/add_evaluation_to_notebooks.py

# Then run each notebook to execute the evaluation
```

#### Option C: Cross-Model Comparison

```bash
# Run comprehensive evaluation notebook
jupyter notebook notebooks/comprehensive_evaluation.ipynb
```

## 📊 Evaluation Framework

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

## 📈 Using Results in Your Thesis

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

## 🔧 Customization

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

## 📝 Supervisor Feedback Implementation

This evaluation framework addresses the supervisor's specific requirements:

✅ **Automated evaluation metrics** - Comprehensive statistical tests  
✅ **LaTeX table generation** - Ready-to-use tables for thesis  
✅ **Automated plotting** - Publication-ready figures  
✅ **Reproducible results** - JSON exports and version control  
✅ **Cross-model comparison** - Systematic evaluation across all models  

## 🎓 Thesis Integration

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

## 🚨 Troubleshooting

### Common Issues

1. **Import errors**: Ensure you're in the correct directory and virtual environment
2. **Data not found**: Check that `data/sp500_data.csv` exists
3. **Memory issues**: Reduce batch sizes in model training
4. **CUDA errors**: Set `device = torch.device("cpu")` for CPU-only execution

### Getting Help

- Check the notebook outputs for error messages
- Verify all dependencies are installed: `pip install -r requirements.txt`
- Ensure data files are in the correct locations
- Check that evaluation results are being saved to the `results/` directory

## 📚 References

- Ho, J., Jain, A., & Abbeel, P. (2020). Denoising Diffusion Probabilistic Models. NeurIPS.
- Cont, R. (2001). Empirical properties of asset returns: stylized facts and statistical issues. Quantitative Finance.
- Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity. Journal of Econometrics.

## 📞 Contact

For questions about this implementation, contact:
- **Author**: Simin Ali
- **Supervisor**: Dr Mikael Mieskolainen
- **Institution**: Imperial College London

---

**Note**: This repository is part of an MSc thesis project. All code and results are for academic research purposes.
