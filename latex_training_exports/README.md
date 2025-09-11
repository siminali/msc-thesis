# Training Data Exports for LaTeX

Generated on: 2025-09-10 19:30:47

## Directory Structure

```
latex_training_exports/
├── figures/
│   ├── loss_curves/           # Individual model training progress
│   ├── performance/           # Cross-model performance comparisons
│   └── system_info/          # Hardware utilization analysis
├── tables/                   # LaTeX-ready tables
├── data/                     # Raw data exports
└── README.md                 # This file
```

## LaTeX Import Instructions

### Including Figures

```latex
% Loss curves
\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.8\textwidth]{latex_training_exports/figures/loss_curves/ddpm_evaluation_training_progress.pdf}
    \caption{DDPM Training Progress}
    \label{fig:ddpm_training}
\end{figure}

% Performance comparison
\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.9\textwidth]{latex_training_exports/figures/performance/model_performance_comparison.pdf}
    \caption{Model Performance Comparison}
    \label{fig:performance_comparison}
\end{figure}
```

### Including Tables

```latex
% Training summary table
\input{latex_training_exports/tables/training_summary.tex}

% Model comparison table
\input{latex_training_exports/tables/model_comparison.tex}
```

## Available Files

### Figures
- `loss_curves/[model]_training_progress.pdf` - Individual model training curves
- `performance/model_performance_comparison.pdf` - Cross-model performance analysis
- `system_info/hardware_utilization.pdf` - Hardware usage analysis

### Tables
- `training_summary.tex` - Detailed training run summary
- `model_comparison.tex` - Statistical comparison across models
- `*.csv` - Raw data versions of all tables

### Quality
- All figures: 300 DPI, publication quality
- All tables: LaTeX-formatted with proper captions and labels
- Consistent styling throughout

## Notes
- All paths are relative to your LaTeX document root
- PDF figures recommended for LaTeX (vector graphics)
- PNG versions available for presentations
- Tables include proper LaTeX formatting and escaping
