# LaTeX Training Data Usage Guide

## 🎯 Quick Start for Your Thesis

### 1. **Loss Curves** (Individual Model Training)
```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.8\textwidth]{latex_training_exports/figures/loss_curves/ddpm_evaluation_training_progress.pdf}
    \caption{DDPM model training progress showing training and validation loss curves across multiple runs}
    \label{fig:ddpm_training}
\end{figure}
```

### 2. **Performance Comparison** (Cross-Model Analysis)
```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.9\textwidth]{latex_training_exports/figures/performance/model_performance_comparison.pdf}
    \caption{Comprehensive performance comparison showing training time, model complexity, efficiency, and final loss across all models}
    \label{fig:model_performance}
\end{figure}
```

### 3. **Hardware Utilization**
```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.7\textwidth]{latex_training_exports/figures/system_info/hardware_utilization.pdf}
    \caption{Hardware utilization analysis showing CPU vs GPU usage distribution and training time implications}
    \label{fig:hardware_usage}
\end{figure}
```

### 4. **Training Summary Table**
```latex
\input{latex_training_exports/tables/training_summary.tex}
```

### 5. **Model Comparison Table**
```latex
\input{latex_training_exports/tables/model_comparison.tex}
```

## 📊 What Each Figure Shows

### Loss Curves (`loss_curves/`)
Each model has a 4-panel figure showing:
- **Top Left**: Training loss progression for all runs
- **Top Right**: Validation loss progression (when available)
- **Bottom Left**: Combined train/validation comparison
- **Bottom Right**: Final loss distribution histogram

### Performance Comparison (`performance/`)
Single comprehensive figure with 4 panels:
- **Top Left**: Average training time by model (bar chart)
- **Top Right**: Model complexity (parameter count)
- **Bottom Left**: Training efficiency scatter plot (time vs parameters)
- **Bottom Right**: Final loss comparison

### Hardware Utilization (`system_info/`)
Analysis of computational resources:
- **Left**: Device usage pie chart (CPU vs GPU)
- **Right**: Training time distribution by device type

## 📋 Tables Content

### Training Summary (`training_summary.tex`)
Detailed per-run information:
- Model name and run ID
- Training time (minutes)
- Parameter count
- Final training/validation losses
- Best epoch and total epochs
- Device used (CPU/GPU)

### Model Comparison (`model_comparison.tex`)
Statistical summary across models:
- Number of runs per model
- Average and standard deviation of training times
- Parameter counts
- Loss statistics (mean, std, best)

## 🎨 Figure Quality
- **Resolution**: 300 DPI (publication quality)
- **Format**: PDF (vector graphics, perfect for LaTeX)
- **Style**: Consistent color scheme and professional formatting
- **Size**: Optimized for thesis layout

## 💡 LaTeX Integration Tips

1. **Place figures in your thesis figures directory**:
   ```bash
   cp latex_training_exports/figures/performance/*.pdf thesis/figures/
   ```

2. **Update paths in LaTeX**:
   ```latex
   \includegraphics[width=0.8\textwidth]{figures/model_performance_comparison.pdf}
   ```

3. **Use subcaptions for multi-panel figures**:
   ```latex
   \usepackage{subcaption}
   ```

4. **Reference figures in text**:
   ```latex
   As shown in Figure~\ref{fig:model_performance}, the DDPM model demonstrates...
   ```

## 🔍 Key Insights for Your Thesis

### Training Efficiency
- DDPM models: ~5 minutes average training time
- Parameter efficiency: 61K parameters for DDPM
- CPU-only training (no GPU acceleration used)

### Model Performance
- Consistent loss convergence across runs
- Best final loss: ~0.994 for DDPM
- Training stability demonstrated through multiple runs

### Computational Requirements
- All training performed on CPU
- Moderate computational requirements
- Scalable approach for financial modeling

## 📚 Suggested Thesis Sections

1. **Methodology**: Use training summary table
2. **Results**: Include performance comparison figure
3. **Implementation**: Reference hardware utilization
4. **Appendix**: Include individual loss curves

This comprehensive export provides everything needed for professional thesis documentation!

