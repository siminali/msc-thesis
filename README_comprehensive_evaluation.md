# Comprehensive Evaluation Pipeline for DDPM Models

## Overview

This comprehensive evaluation pipeline generates **all plots, tables, and metrics needed for your thesis** by analyzing three novelty DDPM models:

1. **Zero-Conditioned**: Unconditional baseline DDPM
2. **Explicit-Conditioned**: Regime + volatility conditioned DDPM  
3. **LLM-Conditioned**: News sentiment conditioned DDPM

## 🎯 What It Generates

### 📊 **Stylized Facts Analysis**
- **Histograms with Gaussian overlays** for all models
- **ECDF comparisons** (Real vs. Generated)
- **Q-Q plots for both tails** (left and right)
- **ACF/PACF analysis** for returns and squared returns

### 🚨 **Risk Management Analysis**
- **VaR/ES curves** across confidence levels (90% to 99.9%)
- **Exceedance timelines** with violations highlighted
- **Kupiec and Christoffersen tests** for backtesting
- **Breach analysis tables** with expected vs. actual rates

### 🎮 **Controllability Analysis**
- **Explicit Model**: Volatility scatter, reliability curves, residuals, regime confusion matrices
- **LLM Model**: Probe vs. realized volatility, sentiment buckets, ablation studies
- **Zero Model**: Unconditional reference analysis

### 🌊 **Diversity & Coverage Analysis**
- **MMD estimates** for distribution comparison
- **Hill tail index comparison** for extreme value analysis
- **Correlation heatmaps** for temporal dependencies
- **Side-by-side path panels** with consistent axes

### 🔍 **Interpretability Analysis**
- **LLM embedding space** visualization (PCA, clustering, similarity)
- **Explicit conditioning vectors** analysis (regime distribution, volatility patterns)

### 💼 **COVID-2020 Business Case Study**
- **Stress scenario simulations** comparing real vs. generated returns
- **Portfolio-level VaR/ES** comparisons
- **Regime response analysis** during crisis periods

## 🚀 Quick Start

### 1. **Run the Pipeline**
```bash
# Basic run with defaults
python run_comprehensive_evaluation.py

# Custom configuration
python src/comprehensive_evaluation_pipeline.py \
    --models_dir results \
    --results_dir results/comparisons \
    --seed 42 \
    --num_samples 1000
```

### 2. **Check Results**
```bash
# View generated results
ls -la results/comparisons/

# Check figures
ls -la results/comparisons/figures/

# Check tables  
ls -la results/comparisons/tables/
```

## 📁 Output Structure

```
results/comparisons/
├── figures/
│   ├── stylized_facts/           # Histograms, ECDF, Q-Q, ACF/PACF
│   ├── risk_management/          # VaR/ES curves, exceedance timelines
│   ├── controllability/          # Model-specific controllability
│   ├── diversity_coverage/       # MMD, Hill index, correlations
│   ├── interpretability/         # Embedding analysis, conditioning
│   └── covid_case_study/         # COVID-2020 stress scenarios
├── tables/
│   ├── risk_management/          # Backtesting results
│   ├── covid_case_study/         # COVID metrics
│   └── [other analysis tables]
├── consolidated_metrics.csv       # All metrics in CSV
├── consolidated_metrics.json      # All metrics in JSON
└── evaluation_report.md           # Comprehensive summary report
```

## ⚙️ Configuration Options

### **Basic Parameters**
- `--seed`: Random seed for reproducibility (default: 42)
- `--num_samples`: Number of samples to generate (default: 1000)
- `--models_dir`: Directory containing trained models (default: results)
- `--results_dir`: Output directory (default: results/comparisons)

### **Advanced Parameters**
- `--var_levels`: VaR confidence levels (default: [0.95, 0.99])
- `--reliability_bins`: Bins for reliability curves (default: 20)
- `--acf_lags`: Lags for ACF/PACF analysis (default: 20)
- `--rolling_window`: Rolling volatility window (default: 20)
- `--hill_threshold`: Threshold for Hill tail index (default: 0.95)

## 🔧 Technical Details

### **Dependencies**
- **Core**: NumPy, Pandas, Matplotlib, Seaborn
- **Statistics**: SciPy, Statsmodels, Scikit-learn
- **Deep Learning**: PyTorch
- **Visualization**: Matplotlib (Agg backend for headless operation)

### **Robustness Features**
- **Error Handling**: Graceful fallback if metrics can't be computed
- **Deterministic Seeds**: Reproducible results across runs
- **Device Safety**: Works on CPU and CUDA
- **Memory Efficient**: Processes data in batches

### **Performance Optimizations**
- **Vectorized Operations**: Fast NumPy-based computations
- **Efficient Plotting**: Minimal memory usage, high DPI output
- **Parallel Processing**: Where applicable for independent analyses

## 📈 Generated Metrics

### **Basic Statistics**
- Mean, Standard Deviation, Skewness, Excess Kurtosis
- Min/Max values, Sample counts

### **Risk Metrics**
- VaR at 95% and 99% confidence levels
- Expected Shortfall (ES) at 95% and 99%
- Violation rates and backtesting statistics

### **Distributional Fidelity**
- **Kolmogorov-Smirnov test** statistic and p-value
- **Wasserstein distance** (if available)
- **MMD (Maximum Mean Discrepancy)** estimates
- **Hill tail index** for extreme value analysis

### **Controllability Metrics**
- **MAE and R²** for volatility targeting
- **Regime accuracy** percentages
- **Reliability curve** statistics
- **Residual analysis** metrics

## 🎨 Plot Features

### **Consistent Styling**
- **Professional appearance** suitable for thesis
- **Consistent color schemes** across all plots
- **High DPI output** (300 DPI) for publication quality
- **Clear labels and legends** with proper formatting

### **Advanced Visualizations**
- **Multi-panel layouts** for comprehensive analysis
- **Overlay comparisons** (real vs. generated)
- **Statistical annotations** (p-values, confidence intervals)
- **Interactive-ready** PDF outputs

## 📊 Table Outputs

### **CSV Format**
- **Machine-readable** for further analysis
- **Consistent column structure** across all tables
- **Model type identification** for easy filtering

### **JSON Format**
- **Structured data** with metadata
- **Human-readable** formatting
- **API-friendly** for integration

### **LaTeX Tables**
- **Publication-ready** formatting
- **Proper mathematical notation**
- **Consistent styling** across all tables

## 🔍 Analysis Categories

### **1. Stylized Facts Replication**
- **Heavy tails**: Histogram + Gaussian overlay
- **Volatility clustering**: ACF/PACF analysis
- **Leverage effects**: Return vs. volatility relationships

### **2. Risk Management**
- **VaR exceedance**: Timeline analysis with breaches
- **ES curves**: Expected shortfall across confidence levels
- **Backtesting**: Kupiec and Christoffersen tests

### **3. Controllability**
- **Explicit**: Volatility targeting, regime classification
- **LLM**: Sentiment conditioning, probe predictions
- **Zero**: Unconditional baseline reference

### **4. Diversity & Coverage**
- **MMD analysis**: Distribution similarity measures
- **Tail behavior**: Hill index estimation
- **Temporal structure**: Correlation analysis

### **5. Interpretability**
- **Embedding spaces**: LLM conditioning visualization
- **Conditioning vectors**: Explicit model analysis
- **Feature importance**: Model behavior insights

### **6. Business Case Study**
- **COVID-2020**: Stress scenario analysis
- **Portfolio risk**: VaR/ES comparisons
- **Regime responses**: Crisis period modeling

## 🚨 Troubleshooting

### **Common Issues**
1. **No checkpoints found**: Ensure models have been trained first
2. **Memory errors**: Reduce `--num_samples` parameter
3. **Plot failures**: Check matplotlib backend and dependencies
4. **Import errors**: Verify all required packages are installed

### **Debug Mode**
```bash
# Run with verbose output
python src/comprehensive_evaluation_pipeline.py --verbose

# Check specific model
python src/comprehensive_evaluation_pipeline.py --models_dir results --results_dir debug_output
```

## 📚 Integration with Thesis

### **Ready-to-Use Outputs**
- **All figures** are publication-quality PDFs
- **All tables** are LaTeX-ready
- **Metrics** are consolidated in CSV/JSON
- **Report** provides executive summary

### **Customization**
- **Modify config** in `DEFAULT_CONFIG` for different parameters
- **Add new analyses** by extending the evaluator class
- **Custom plots** can be added to existing methods

## 🎉 Success Indicators

When the pipeline completes successfully, you should see:
- ✅ **All models loaded** and samples generated
- ✅ **All analysis categories** completed
- ✅ **Figures saved** in organized subdirectories
- ✅ **Tables exported** in multiple formats
- ✅ **Consolidated report** generated
- ✅ **Final summary** printed to console

## 🔗 Related Files

- **`src/comprehensive_evaluation_pipeline.py`**: Main evaluation engine
- **`run_comprehensive_evaluation.py`**: Simple runner script
- **`src/explicit_cond_ddpm.py`**: Explicit conditioning model
- **`src/llm_conditioned_diffusion_refactored.py`**: LLM conditioning model
- **`src/train_all.py`**: Training pipeline for all models

---

**🎯 This pipeline generates everything you need for your thesis analysis!** 

Run it once and get comprehensive, publication-ready results for all three DDPM models. 🚀✨
