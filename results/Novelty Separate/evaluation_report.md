# Unified Evaluation Report

## Overview
This report summarizes the evaluation results for all three DDPM models:
- Zero-Conditioned (Unconditional)
- Explicit-Conditioned (Regime + Volatility)
- LLM-Conditioned (News Embeddings)

## Key Metrics Summary


### Zero Conditioned

- **Distributional Fidelity**: KS=0.5360 (p=0.0000)
- **Forecast Accuracy**: MSE=1.410367, MAE=0.869898
- **Risk Metrics**: VaR 95%=-1.8539, ES 95%=-2.6581
- **Stylized Facts**: Skew=0.5868, Kurtosis=8.9018


### Explicit Conditioned

- **Distributional Fidelity**: KS=0.5747 (p=0.0000)
- **Forecast Accuracy**: MSE=8.296746, MAE=2.061857
- **Risk Metrics**: VaR 95%=-3.5435, ES 95%=-5.4925
- **Stylized Facts**: Skew=0.0782, Kurtosis=8.5911


### Llm Conditioned

- **Distributional Fidelity**: KS=0.5211 (p=0.0000)
- **Forecast Accuracy**: MSE=953.869278, MAE=23.512960
- **Risk Metrics**: VaR 95%=-52.7929, ES 95%=-70.3665
- **Stylized Facts**: Skew=-0.1147, Kurtosis=0.8353


## Generated Files

### Figures
All plots are saved in `results/figures/<model_type>/`:

#### Basic Analysis
- `stylized_facts.pdf`: Histogram with Gaussian overlay
- `ecdf_comparison.pdf`: ECDF comparison with real data
- `qq_tails.pdf`: Q-Q plots for both tails
- `acf_pacf.pdf`: ACF and PACF for returns and squared returns
- `rolling_volatility.pdf`: Rolling volatility comparison
- `sample_paths.pdf`: Sample generated paths

#### Enhanced Analysis
- `training_curves.pdf`: Training and validation loss curves
- `var_es_curves.pdf`: VaR and ES curves across confidence levels
- `exceedance_timeline.pdf`: VaR violation timeline plots
- `volatility_clustering.pdf`: Volatility clustering analysis

#### Model-Specific Analysis
- **Explicit Model**: `controllability_analysis.pdf` (volatility scatter, reliability curves, residuals, regime confusion matrix)
- **LLM Model**: `llm_controllability.pdf` (sentiment buckets, ablation studies, volatility ratios, correlation heatmaps)

### Tables
All LaTeX tables are saved in `results/tables/<model_type>/`:
- `metrics.tex`: Comprehensive metrics table with all categories
- `metrics.csv`: Metrics in CSV format
- `metrics.json`: Metrics in JSON format

### Consolidated Data
- `consolidated_metrics.csv`: All metrics in one CSV file
- `consolidated_metrics.json`: All metrics in one JSON file

## Evaluation Parameters
- Seed: 42
- Number of samples: 500
- VaR levels: [0.95, 0.99]
- Reliability bins: 20
- ACF lags: 20
- Rolling window: 20

Generated on: 2025-08-17 15:53:45
