# Appendix Materials Usage Guide

## 📋 Generated LaTeX Tables

### Backtesting Analysis Tables
1. **`kupiec_test.tex`** - Kupiec test p-values for VaR models
   - Usage: `\input{appendix_tables/kupiec_test.tex}`
   - Shows unconditional coverage test results

2. **`christoffersen_test.tex`** - Christoffersen test p-values
   - Usage: `\input{appendix_tables/christoffersen_test.tex}`
   - Shows conditional coverage test results

3. **`breach_analysis.tex`** - VaR breach ratios (Actual/Expected)
   - Usage: `\input{appendix_tables/breach_analysis.tex}`
   - Shows violation rates vs expected rates

### Model Performance Tables
4. **`model_summary.tex`** - Comprehensive model performance summary
   - Usage: `\input{appendix_tables/model_summary.tex}`
   - Includes returns, volatility, skewness, kurtosis, KS stats, VaR, ES

5. **`training_config.tex`** - Training configuration summary
   - Usage: `\input{appendix_tables/training_config.tex}`
   - Shows epochs, batch size, parameters, training time, device

### Period Analysis Tables
6. **`var95_periods.tex`** - VaR 95% by model and time period
   - Usage: `\input{appendix_tables/var95_periods.tex}`
   - Compares COVID, Calm, and Post periods

7. **`hit_rates_95.tex`** - VaR 95% hit rates by period
   - Usage: `\input{appendix_tables/hit_rates_95.tex}`
   - Shows violation frequencies across periods

## 📊 Generated Figures

### Training Diagnostics
1. **`appendix_loss_curves.pdf`** - Training and validation loss curves
   - Usage: `\includegraphics[width=\textwidth]{appendix_figures/appendix_loss_curves.pdf}`
   - Shows convergence behavior across models

2. **`appendix_val_metrics.pdf`** - Validation metrics (KS, MMD)
   - Usage: `\includegraphics[width=0.8\textwidth]{appendix_figures/appendix_val_metrics.pdf}`
   - Bar charts of statistical test results

### Crisis Analysis
3. **`appendix_covid_ecdfs.pdf`** - COVID crisis period ECDFs
   - Usage: `\includegraphics[width=0.8\textwidth]{appendix_figures/appendix_covid_ecdfs.pdf}`
   - Empirical cumulative distribution functions

4. **`appendix_covid_vols.pdf`** - Crisis period risk analysis
   - Usage: `\includegraphics[width=\textwidth]{appendix_figures/appendix_covid_vols.pdf}`
   - Expected shortfall by model and period

## 🎯 Overleaf Integration

### Step 1: Upload Files
1. Create folder `appendix_tables/` in your Overleaf project
2. Upload all `.tex` files from `results/appendix_tables/`
3. Create folder `appendix_figures/` in your Overleaf project  
4. Upload all `.pdf` files from `results/appendix_figures/`

### Step 2: LaTeX Integration
```latex
\appendix
\chapter{Additional Results}

\section{Training Diagnostics}
\begin{figure}[H]
    \centering
    \includegraphics[width=\textwidth]{appendix_figures/appendix_loss_curves.pdf}
    \caption{Training and validation loss curves for all diffusion models.}
    \label{fig:appendix_loss_curves}
\end{figure}

\section{Model Configuration}
\input{appendix_tables/training_config.tex}

\section{Backtesting Results}
\input{appendix_tables/kupiec_test.tex}
\input{appendix_tables/christoffersen_test.tex}
\input{appendix_tables/breach_analysis.tex}

\section{Crisis Period Analysis}
\begin{figure}[H]
    \centering
    \includegraphics[width=0.8\textwidth]{appendix_figures/appendix_covid_vols.pdf}
    \caption{Expected shortfall analysis during crisis periods.}
    \label{fig:covid_risk_analysis}
\end{figure}

\input{appendix_tables/var95_periods.tex}
```

### Step 3: Required Packages
Ensure these packages are in your preamble:
```latex
\usepackage{booktabs}  % For table formatting
\usepackage{graphicx}  % For figures
\usepackage{float}     % For [H] positioning
```

## 📝 Table Customization

### Adjusting Table Formatting
You can modify the generated `.tex` files:

1. **Change caption**: Edit the `\caption{...}` line
2. **Adjust column alignment**: Modify `\begin{tabular}{lrr}` format
3. **Add notes**: Insert `\footnotesize` text before `\end{table}`
4. **Resize tables**: Wrap in `\resizebox{\textwidth}{!}{...}`

### Example Customization
```latex
\begin{table}[H]
\centering
\resizebox{0.9\textwidth}{!}{
\input{appendix_tables/model_summary.tex}
}
\footnotesize
Note: All statistics computed on out-of-sample test data (2020-2024).
\end{table}
```

## 🔧 Troubleshooting

### Common Issues
1. **Missing figures**: Ensure PDF files are uploaded to correct folder
2. **Table formatting**: Check booktabs package is loaded
3. **Caption conflicts**: Use unique `\label{}` for each table/figure
4. **Size issues**: Use `\resizebox` or adjust `\textwidth` scaling

### File Dependencies
- All `.tex` files are self-contained
- Figures require corresponding `.pdf` files
- No external data dependencies

## 📈 Quality Assurance

All generated materials follow:
- ✅ Publication-quality formatting (300 DPI)
- ✅ Consistent color scheme across models
- ✅ Proper LaTeX table structure with booktabs
- ✅ Clear axis labels and units
- ✅ Unique labels for cross-referencing
- ✅ Professional typography settings

Ready for direct integration into your MSc thesis appendix!

