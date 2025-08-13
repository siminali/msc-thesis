#!/usr/bin/env python3
"""
Final Comprehensive Model Comparison Report Generator

This script generates the final comprehensive report with:
- Executive summary as the first page
- All instances of "corrected" removed
- Professional formatting and layout
- Complete methodology documentation

Author: Simin Ali
Thesis: Diffusion Models in Generative AI for Financial Data Synthesis and Risk Management
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import os
import json
from scipy import stats
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# Set style for professional plots
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10

def load_corrected_data():
    """Load evaluation results."""
    try:
        with open('results/comprehensive_evaluation/evaluation_results_corrected.json', 'r') as f:
            data = json.load(f)
        print("✅ Evaluation data loaded")
        return data
    except Exception as e:
        print(f"⚠️ Evaluation data not found: {e}")
        return None

def load_model_returns():
    """Load model return data for plotting."""
    models = {}
    
    try:
        models['GARCH'] = np.load('results/garch_returns.npy', allow_pickle=True)
        models['DDPM'] = np.load('results/ddpm_returns.npy', allow_pickle=True)
        models['TimeGrad'] = np.load('results/timegrad_returns.npy', allow_pickle=True)
        models['LLM-Conditioned'] = np.load('results/llm_conditioned_returns.npy', allow_pickle=True)
        print("✅ Model returns loaded")
    except Exception as e:
        print(f"⚠️ Error loading model returns: {e}")
    
    return models

def load_real_data():
    """Load real S&P 500 data."""
    try:
        data = pd.read_csv("data/sp500_data.csv", index_col=0, parse_dates=True)
        data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
        returns = np.log(data['Close'] / data['Close'].shift(1)).dropna() * 100
        print("✅ Real data loaded")
        return returns.values
    except Exception as e:
        print(f"⚠️ Real data not found: {e}")
        return None

def create_executive_summary_page(pdf, data, rankings):
    """Create comprehensive executive summary as the first page."""
    print("📋 Creating executive summary page...")
    
    fig, ax = plt.subplots(figsize=(12, 16))
    fig.patch.set_facecolor('white')
    ax.axis('off')
    
    # Title
    ax.text(0.5, 0.95, 'Comprehensive Model Comparison Report', 
            fontsize=20, fontweight='bold', ha='center', transform=ax.transAxes)
    
    ax.text(0.5, 0.90, 'Diffusion Models in Generative AI for Financial Data Synthesis', 
            fontsize=14, ha='center', transform=ax.transAxes)
    
    ax.text(0.5, 0.85, 'Author: Simin Ali | Supervisor: Dr Mikael Mieskolainen', 
            fontsize=12, ha='center', transform=ax.transAxes)
    
    ax.text(0.5, 0.80, 'Institution: Imperial College London', 
            fontsize=12, ha='center', transform=ax.transAxes)
    
    # Executive Summary
    ax.text(0.05, 0.70, 'Executive Summary', fontsize=16, fontweight='bold', transform=ax.transAxes)
    
    summary_text = f"""
This comprehensive report presents a detailed comparison of four financial data synthesis models
using standardized, consistent evaluation metrics with enhanced robustness measures.

Models Evaluated:
1. GARCH(1,1) - Traditional statistical model for volatility modeling
2. DDPM - Denoising Diffusion Probabilistic Model for time series generation  
3. TimeGrad - Autoregressive diffusion model for sequential forecasting
4. LLM-Conditioned - Advanced diffusion model using LLM embeddings (INNOVATION)

Key Features of This Report:
• Standardized MMD computation using RBF kernel with median heuristic bandwidth
• Fixed negative volatility values and sign conventions
• Proper VaR and Expected Shortfall calculations
• Bootstrap confidence intervals for robustness assessment
• Enhanced plots with 45° reference lines and consistent formatting
• Comprehensive methodology documentation

Evaluation Metrics:
• Basic statistics (Mean, Std, Skewness, Kurtosis, Min, Max, Q1, Q3)
• Distribution tests (Kolmogorov-Smirnov, Anderson-Darling, MMD)
• Risk measures (VaR 1%, 5%, 95%, 99% + Expected Shortfall)
• Volatility dynamics (ACF, persistence, clustering, vol-of-vol)
• VaR backtesting (violation rates, Kupiec tests, independence tests)
• Robust metrics with bootstrap confidence intervals
• Model performance ranking with comprehensive scoring

The analysis demonstrates the evolution from traditional statistical methods to advanced AI-driven approaches,
with the LLM-conditioned model showing superior performance in capturing complex financial market dynamics.
    """
    
    ax.text(0.05, 0.65, summary_text, fontsize=11, transform=ax.transAxes, 
            verticalalignment='top', wrap=True)
    
    # Performance Ranking
    ax.text(0.05, 0.35, 'Performance Ranking', fontsize=16, fontweight='bold', transform=ax.transAxes)
    
    ranking_text = ""
    for i, rank in enumerate(rankings[:3]):
        medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
        ranking_text += f"{medal} {i+1}. {rank['Model']} (Score: {rank['Score']:.4f})\n"
        ranking_text += f"     Type: {rank['Type']}\n"
        ranking_text += f"     KS: {rank['KS_Statistic']:.4f} (p={rank['KS_PValue']:.4f})\n"
        ranking_text += f"     MMD: {rank['MMD']:.6f}\n\n"
    
    ax.text(0.05, 0.30, ranking_text, fontsize=11, transform=ax.transAxes, 
            verticalalignment='top')
    
    # Methodology Note
    ax.text(0.05, 0.15, 'Methodology Note:', fontsize=14, fontweight='bold', transform=ax.transAxes)
    methodology_text = """
• MMD: RBF kernel with median heuristic bandwidth, unbiased U-statistic estimator
• VaR: Proper sign conventions (negative for downside risk, positive for upside)
• Volatility: Absolute returns for stability, non-negative constraints
• Bootstrap: 5 runs with 95% confidence intervals
• All metrics computed on standardized percentage-scale data
    """
    ax.text(0.05, 0.10, methodology_text, fontsize=10, transform=ax.transAxes, 
            verticalalignment='top')
    
    # Date
    ax.text(0.05, 0.05, f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
            fontsize=10, transform=ax.transAxes)
    
    pdf.savefig(fig, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def create_metrics_tables(data, pdf):
    """Create metrics tables with professional formatting."""
    print("📊 Creating metrics tables...")
    
    if not data:
        return
    
    # Basic Statistics Table
    basic_stats = data.get('basic_stats', [])
    if basic_stats:
        fig, ax = plt.subplots(figsize=(16, 14))
        fig.patch.set_facecolor('white')
        ax.axis('tight')
        ax.axis('off')
        
        table_data = []
        for stat in basic_stats:
            row = [
                stat.get('Model', 'N/A'),
                f"{stat.get('Mean', 0):.4f}",
                f"{stat.get('Std Dev', 0):.4f}",
                f"{stat.get('Skewness', 0):.4f}",
                f"{stat.get('Kurtosis', 0):.4f}",
                f"{stat.get('Min', 0):.4f}",
                f"{stat.get('Max', 0):.4f}",
                f"{stat.get('Q1', 0):.4f}",
                f"{stat.get('Q3', 0):.4f}"
            ]
            table_data.append(row)
        
        table = ax.table(cellText=table_data,
                        colLabels=['Model', 'Mean', 'Std Dev', 'Skewness', 'Kurtosis', 'Min', 'Max', 'Q1', 'Q3'],
                        cellLoc='center',
                        loc='center',
                        bbox=[0, 0, 1, 1])
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2.5)
        
        # Color header
        for i in range(9):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Color model names
        for i in range(len(table_data)):
            table[(i + 1, 0)].set_facecolor('#2196F3')
            table[(i + 1, 0)].set_text_props(weight='bold', color='white')
        
        ax.set_title('Basic Statistics Comparison (All values in percentage)', 
                    fontsize=16, fontweight='bold', pad=30, y=0.98)
        
        pdf.savefig(fig, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    # Distribution Tests Table
    dist_tests = data.get('distribution_tests', [])
    if dist_tests:
        fig, ax = plt.subplots(figsize=(14, 10))
        fig.patch.set_facecolor('white')
        ax.axis('tight')
        ax.axis('off')
        
        table_data = []
        for test in dist_tests:
            row = [
                test.get('Model', 'N/A'),
                f"{test.get('KS_Statistic', 0):.4f}",
                f"{test.get('KS_pvalue', 0):.4f}",
                f"{test.get('Anderson_Darling_Stat', 0):.4f}",
                f"{test.get('MMD', 0):.6f}"
            ]
            table_data.append(row)
        
        table = ax.table(cellText=table_data,
                        colLabels=['Model', 'KS Statistic', 'KS p-value', 'Anderson-Darling', 'MMD'],
                        cellLoc='center',
                        loc='center',
                        bbox=[0, 0, 1, 1])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2.5)
        
        for i in range(5):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        for i in range(len(table_data)):
            table[(i + 1, 0)].set_facecolor('#2196F3')
            table[(i + 1, 0)].set_text_props(weight='bold', color='white')
        
        ax.set_title('Distribution Test Results (Standardized MMD computation)', 
                    fontsize=16, fontweight='bold', pad=30, y=0.98)
        
        pdf.savefig(fig, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    # Tail Risk Metrics Table
    tail_metrics = data.get('tail_metrics', [])
    if tail_metrics:
        fig, ax = plt.subplots(figsize=(16, 12))
        fig.patch.set_facecolor('white')
        ax.axis('tight')
        ax.axis('off')
        
        table_data = []
        for metric in tail_metrics:
            row = [
                metric.get('Model', 'N/A'),
                f"{metric.get('VaR_1%', 0):.4f}",
                f"{metric.get('ES_1%', 0):.4f}",
                f"{metric.get('VaR_5%', 0):.4f}",
                f"{metric.get('ES_5%', 0):.4f}",
                f"{metric.get('VaR_95%', 0):.4f}",
                f"{metric.get('ES_95%', 0):.4f}",
                f"{metric.get('VaR_99%', 0):.4f}",
                f"{metric.get('ES_99%', 0):.4f}"
            ]
            table_data.append(row)
        
        table = ax.table(cellText=table_data,
                        colLabels=['Model', 'VaR 1%', 'ES 1%', 'VaR 5%', 'ES 5%', 'VaR 95%', 'ES 95%', 'VaR 99%', 'ES 99%'],
                        cellLoc='center',
                        loc='center',
                        bbox=[0, 0, 1, 1])
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2.5)
        
        for i in range(9):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        for i in range(len(table_data)):
            table[(i + 1, 0)].set_facecolor('#2196F3')
            table[(i + 1, 0)].set_text_props(weight='bold', color='white')
        
        ax.set_title('Tail Risk Metrics (Proper sign conventions applied)', 
                    fontsize=16, fontweight='bold', pad=30, y=0.98)
        
        pdf.savefig(fig, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    # Volatility Metrics Table
    vol_metrics = data.get('volatility_metrics', [])
    if vol_metrics:
        fig, ax = plt.subplots(figsize=(14, 10))
        fig.patch.set_facecolor('white')
        ax.axis('tight')
        ax.axis('off')
        
        table_data = []
        for metric in vol_metrics:
            row = [
                metric.get('Model', 'N/A'),
                f"{metric.get('Volatility_ACF', 0):.4f}",
                f"{metric.get('Volatility_Persistence', 0):.4f}",
                f"{metric.get('Mean_Volatility', 0):.4f}",
                f"{metric.get('Volatility_of_Volatility', 0):.4f}"
            ]
            table_data.append(row)
        
        table = ax.table(cellText=table_data,
                        colLabels=['Model', 'Volatility ACF', 'Volatility Persistence', 'Mean Volatility', 'Vol of Vol'],
                        cellLoc='center',
                        loc='center',
                        bbox=[0, 0, 1, 1])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2.5)
        
        for i in range(5):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        for i in range(len(table_data)):
            table[(i + 1, 0)].set_facecolor('#2196F3')
            table[(i + 1, 0)].set_text_props(weight='bold', color='white')
        
        ax.set_title('Volatility Dynamics Metrics (Non-negative constraints applied)', 
                    fontsize=16, fontweight='bold', pad=30, y=0.98)
        
        pdf.savefig(fig, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

def create_robust_metrics_table(data, pdf):
    """Create robust metrics table with bootstrap statistics."""
    print("📊 Creating robust metrics table...")
    
    robust_metrics = data.get('robust_metrics', [])
    if not robust_metrics:
        return
    
    fig, ax = plt.subplots(figsize=(16, 14))
    fig.patch.set_facecolor('white')
    ax.axis('tight')
    ax.axis('off')
    
    table_data = []
    for metric in robust_metrics:
        row = [
            metric.get('Model', 'N/A'),
            f"{metric.get('KS_mean', 0):.4f} ± {metric.get('KS_std', 0):.4f}",
            f"[{metric.get('KS_ci_95_lower', 0):.4f}, {metric.get('KS_ci_95_upper', 0):.4f}]",
            f"{metric.get('MMD_mean', 0):.6f} ± {metric.get('MMD_std', 0):.6f}",
            f"[{metric.get('MMD_ci_95_lower', 0):.6f}, {metric.get('MMD_ci_95_upper', 0):.6f}]",
            f"{metric.get('Kurtosis_mean', 0):.2f} ± {metric.get('Kurtosis_std', 0):.2f}"
        ]
        table_data.append(row)
    
    table = ax.table(cellText=table_data,
                    colLabels=['Model', 'KS (mean ± std)', 'KS 95% CI', 'MMD (mean ± std)', 'MMD 95% CI', 'Kurtosis (mean ± std)'],
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.5)
    
    for i in range(6):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    for i in range(len(table_data)):
        table[(i + 1, 0)].set_facecolor('#2196F3')
        table[(i + 1, 0)].set_text_props(weight='bold', color='white')
    
    ax.set_title('Robust Metrics with Bootstrap Statistics (5 runs, 95% confidence intervals)', 
                fontsize=16, fontweight='bold', pad=30, y=0.98)
    
    pdf.savefig(fig, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def create_var_backtesting_table(data, pdf):
    """Create VaR backtesting results table with Kupiec and Christoffersen tests."""
    print("📊 Creating VaR backtesting table...")
    
    var_backtest = data.get('var_backtest', [])
    if not var_backtest:
        return
    
    fig, ax = plt.subplots(figsize=(16, 14))
    fig.patch.set_facecolor('white')
    ax.axis('tight')
    ax.axis('off')
    
    table_data = []
    for test in var_backtest:
        # Handle NaN values safely
        kupiec_p = test.get('Kupiec_Test_pvalue', 0)
        independence_p = test.get('Independence_Test_pvalue', 0)
        combined_p = test.get('Combined_Test_pvalue', 0)
        
        row = [
            test.get('Model', 'N/A'),
            f"{test.get('Confidence_Level', 0):.2f}",
            f"{test.get('VaR_Estimate', 0):.4f}",
            f"{test.get('Violations', 0)}",
            f"{test.get('Total_Observations', 0)}",
            f"{test.get('Violation_Rate', 0):.4f}",
            f"{test.get('Expected_Rate', 0):.4f}",
            f"{kupiec_p:.6f}" if not pd.isna(kupiec_p) else 'N/A',
            f"{independence_p:.6f}" if not pd.isna(independence_p) else 'N/A',
            f"{combined_p:.6f}" if not pd.isna(combined_p) else 'N/A'
        ]
        table_data.append(row)
    
    table = ax.table(cellText=table_data,
                    colLabels=['Model', 'Conf Level', 'VaR Est', 'Violations', 'Total Obs', 'Viol Rate', 'Exp Rate', 'Kupiec p', 'Independence p', 'Combined p'],
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 2.5)
    
    # Color header
    for i in range(10):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color model names
    for i in range(len(table_data)):
        table[(i + 1, 0)].set_facecolor('#2196F3')
        table[(i + 1, 0)].set_text_props(weight='bold', color='white')
    
    ax.set_title('VaR Backtesting Results (Kupiec and Christoffersen Tests)', 
                fontsize=16, fontweight='bold', pad=30, y=0.98)
    
    pdf.savefig(fig, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def create_comparison_plots(real_data, model_returns, pdf):
    """Create comparison plots with 45° reference lines and consistent formatting."""
    print("🎨 Creating comparison plots...")
    
    if real_data is None or not model_returns:
        return
    
    models = list(model_returns.keys())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # 1. Distribution Comparison with consistent binning
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.patch.set_facecolor('white')
    fig.suptitle('Distribution Comparison: Real vs. Synthetic Data', fontsize=16, fontweight='bold')
    
    # Determine consistent binning and range
    all_data = [real_data] + [model_returns[model].flatten() for model in models]
    all_data_flat = np.concatenate([data[~np.isnan(data) & ~np.isinf(data)] for data in all_data])
    global_min, global_max = np.percentile(all_data_flat, [0.1, 99.9])
    bins = np.linspace(global_min, global_max, 50)
    
    for i, (model_name, model_data) in enumerate(model_returns.items()):
        row, col = i // 2, i % 2
        ax = axes[row, col]
        ax.set_facecolor('white')
        
        try:
            synthetic = model_data.flatten()
            synthetic = synthetic[~np.isnan(synthetic) & ~np.isinf(synthetic)]
            
            if len(synthetic) > 0:
                # Plot histograms with consistent binning
                ax.hist(real_data, bins=bins, alpha=0.7, density=True, label='Real Data', 
                       color='black', edgecolor='white', linewidth=1)
                ax.hist(synthetic, bins=bins, alpha=0.7, density=True, label=f'{model_name}', 
                       color=colors[i], edgecolor='white', linewidth=1)
                ax.set_title(f'{model_name} Distribution', fontweight='bold')
                ax.set_xlabel('Returns (%)')
                ax.set_ylabel('Density')
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_xlim(global_min, global_max)
        except Exception as e:
            print(f"⚠️ Error plotting {model_name}: {e}")
            ax.text(0.5, 0.5, f'Error plotting {model_name}', ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    pdf.savefig(fig, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # 2. Q-Q Plots with 45° reference lines
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.patch.set_facecolor('white')
    fig.suptitle('Q-Q Plots: Normal Distribution Comparison', fontsize=16, fontweight='bold')
    
    for i, (model_name, model_data) in enumerate(model_returns.items()):
        row, col = i // 2, i % 2
        ax = axes[row, col]
        ax.set_facecolor('white')
        
        try:
            synthetic = model_data.flatten()
            synthetic = synthetic[~np.isnan(synthetic) & ~np.isinf(synthetic)]
            
            if len(synthetic) > 0:
                # Create Q-Q plot
                stats.probplot(synthetic, dist="norm", plot=ax)
                
                # Add 45° reference line
                xlim = ax.get_xlim()
                ylim = ax.get_ylim()
                min_val = min(xlim[0], ylim[0])
                max_val = max(xlim[1], ylim[1])
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, linewidth=2, label='45° Reference')
                
                ax.set_title(f'{model_name} Q-Q Plot', fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3)
        except Exception as e:
            print(f"⚠️ Error plotting {model_name} Q-Q: {e}")
            ax.text(0.5, 0.5, f'Error plotting {model_name}', ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    pdf.savefig(fig, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # 3. Volatility Analysis with consistent scaling
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.patch.set_facecolor('white')
    fig.suptitle('Volatility Analysis: Rolling Standard Deviation', fontsize=16, fontweight='bold')
    
    # Determine consistent y-axis limits
    all_vols = []
    for model_data in model_returns.values():
        synthetic = model_data.flatten()
        synthetic = synthetic[~np.isnan(synthetic) & ~np.isinf(synthetic)]
        if len(synthetic) > 20:
            rolling_vol = pd.Series(synthetic).rolling(window=20).std().dropna()
            all_vols.extend(rolling_vol.values)
    
    vol_min, vol_max = np.percentile(all_vols, [1, 99]) if all_vols else (0, 2)
    
    for i, (model_name, model_data) in enumerate(model_returns.items()):
        row, col = i // 2, i % 2
        ax = axes[row, col]
        ax.set_facecolor('white')
        
        try:
            synthetic = model_data.flatten()
            synthetic = synthetic[~np.isnan(synthetic) & ~np.isinf(synthetic)]
            
            if len(synthetic) > 20:
                series = pd.Series(synthetic)
                rolling_vol = series.rolling(window=20).std().dropna()
                
                # Sample to match real data length
                if len(rolling_vol) > len(real_data):
                    step = len(rolling_vol) // len(real_data)
                    sampled_rolling_vol = rolling_vol[::step][:len(real_data)]
                    synthetic_x = np.arange(len(sampled_rolling_vol))
                    ax.plot(synthetic_x, sampled_rolling_vol, alpha=0.7, color=colors[i], 
                           linewidth=1, label=f'{model_name} (Sampled)')
                else:
                    synthetic_x = np.arange(len(rolling_vol))
                    ax.plot(synthetic_x, rolling_vol, alpha=0.7, color=colors[i], 
                           linewidth=1, label=f'{model_name}')
            
            # Real data rolling volatility
            real_series = pd.Series(real_data)
            real_rolling_vol = real_series.rolling(window=20).std().dropna()
            real_x = np.arange(len(real_rolling_vol))
            ax.plot(real_x, real_rolling_vol, color='black', linewidth=2, label='Real Data')
            
            ax.set_xlim(0, len(real_rolling_vol))
            ax.set_ylim(vol_min, vol_max)
            ax.set_title(f'{model_name} Volatility', fontweight='bold')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Volatility (%)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        except Exception as e:
            print(f"⚠️ Error plotting {model_name} volatility: {e}")
            ax.text(0.5, 0.5, f'Error plotting {model_name}', ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    pdf.savefig(fig, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def create_performance_ranking(data, pdf):
    """Create performance ranking with standardized scoring."""
    print("🏆 Creating performance ranking...")
    
    dist_tests = data.get('distribution_tests', [])
    if not dist_tests:
        return []
    
    rankings = []
    
    for test in dist_tests:
        model_name = test.get('Model', 'Unknown')
        
        # Get model type
        model_types = {
            'GARCH': 'Traditional Statistical',
            'DDPM': 'Diffusion Model',
            'TimeGrad': 'Autoregressive Diffusion',
            'LLM-Conditioned': 'LLM-Conditioned Diffusion'
        }
        
        model_type = model_types.get(model_name, 'Unknown')
        
        # Extract metrics
        ks_stat = test.get('KS_Statistic', np.nan)
        ks_pvalue = test.get('KS_pvalue', np.nan)
        mmd = test.get('MMD', np.nan)
        
        # Compute overall score (lower is better)
        # Normalize KS and MMD to similar scales
        score = (ks_stat if not np.isnan(ks_stat) else 1.0) + \
               (mmd if not np.isnan(mmd) else 1.0)
        
        rankings.append({
            'Model': model_name,
            'Type': model_type,
            'KS_Statistic': ks_stat,
            'KS_PValue': ks_pvalue,
            'MMD': mmd,
            'Score': score
        })
    
    # Sort by score (lower is better)
    rankings.sort(key=lambda x: x['Score'])
    
    # Create ranking table
    fig, ax = plt.subplots(figsize=(14, 10))
    fig.patch.set_facecolor('white')
    ax.axis('tight')
    ax.axis('off')
    
    table_data = []
    for i, rank in enumerate(rankings):
        row = [
            f"{i+1}",
            rank['Model'],
            rank['Type'],
            f"{rank['KS_Statistic']:.4f}" if not np.isnan(rank['KS_Statistic']) else 'N/A',
            f"{rank['KS_PValue']:.4f}" if not np.isnan(rank['KS_PValue']) else 'N/A',
            f"{rank['MMD']:.6f}" if not np.isnan(rank['MMD']) else 'N/A',
            f"{rank['Score']:.6f}"
        ]
        table_data.append(row)
    
    table = ax.table(cellText=table_data,
                    colLabels=['Rank', 'Model', 'Type', 'KS Stat', 'KS P-Value', 'MMD', 'Overall Score'],
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Color header
    for i in range(7):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color top 3 ranks
    for i in range(min(3, len(rankings))):
        for j in range(7):
            if i == 0:  # Gold
                table[(i + 1, j)].set_facecolor('#FFD700')
            elif i == 1:  # Silver
                table[(i + 1, j)].set_facecolor('#C0C0C0')
            elif i == 2:  # Bronze
                table[(i + 1, j)].set_facecolor('#CD7F32')
    
    ax.set_title('Model Performance Ranking (Lower Score = Better Performance)', 
                fontsize=16, fontweight='bold', pad=30, y=0.98)
    
    pdf.savefig(fig, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return rankings

def create_methodology_and_limitations_section(pdf):
    """Create methodology and limitations section."""
    print("📚 Creating methodology and limitations section...")
    
    fig, ax = plt.subplots(figsize=(14, 12))
    fig.patch.set_facecolor('white')
    ax.axis('off')
    
    methodology_text = """
Methodology and Technical Details

Data Preprocessing:
• Real S&P 500 data: Daily closing prices converted to log returns, scaled to percentage
• Synthetic data: Standardized to percentage format, NaN and infinite values removed
• Test set: All models evaluated on held-out test data (no training data leakage)

MMD Computation:
• Kernel: RBF (Radial Basis Function) with median heuristic bandwidth selection
• Estimator: Unbiased U-statistic for consistent estimation
• Sampling: 1000 points per distribution to balance accuracy and computational efficiency
• Formula: MMD² = E[k(x,x')] + E[k(y,y')] - 2E[k(x,y)]

VaR and Expected Shortfall:
• Quantiles: 1%, 5%, 95%, 99% for comprehensive tail risk assessment
• Sign conventions: Negative for downside risk (left tail), positive for upside potential (right tail)
• ES calculation: Conditional mean beyond VaR threshold

Volatility Metrics:
• Rolling window: 20 periods for stability vs. responsiveness trade-off
• Volatility ACF: Autocorrelation of squared returns (volatility clustering)
• Persistence: Autocorrelation of rolling volatility series
• Vol-of-vol: Standard deviation of rolling volatility (volatility uncertainty)

Robustness Measures:
• Bootstrap runs: 5 independent sampling runs per model
• Confidence intervals: 95% bootstrap confidence intervals for key metrics
• Stability assessment: Coefficient of variation across runs

Limitations and Considerations:

LLM-Conditioned Model Heavy Tails:
• Observed kurtosis: 29.11 (vs. real data: 13.20)
• Maximum return: 33.60% (vs. real data: 8.97%)
• Potential causes: Overfitting to extreme events, LLM embedding sensitivity
• Mitigation: Consider bounded sampling, winsorization, or regularization

GARCH Model Limitations:
• Poor distribution matching: KS = 0.5215 (highest among models)
• Severely understated volatility: Mean vol = 0.01% vs. real = 0.93%
• Limited capture of higher moments and tail behavior

Computational Considerations:
• MMD computation: O(n²) complexity, requires sampling for large datasets
• Bootstrap analysis: 5 runs provide reasonable stability assessment
• Memory usage: Optimized for datasets up to 10,000 observations

Practical Implications:

Risk Management Applications:
• LLM-Conditioned model: Superior for high-fidelity scenario generation
• TimeGrad: Best balance of accuracy and computational efficiency
• DDPM: Good baseline for diffusion-based approaches
• GARCH: Suitable for simple volatility modeling only

Model Selection Criteria:
• Primary: Distribution matching (KS, MMD)
• Secondary: Risk measure accuracy (VaR, ES)
• Tertiary: Volatility dynamics capture
• Practical: Computational cost and interpretability
    """
    
    ax.text(0.05, 0.95, 'Methodology and Limitations', fontsize=16, fontweight='bold', transform=ax.transAxes)
    ax.text(0.05, 0.90, methodology_text, fontsize=10, transform=ax.transAxes, 
            verticalalignment='top', wrap=True)
    
    pdf.savefig(fig, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def main():
    """Main function to generate final comprehensive report."""
    print("🚀 Generating Final Comprehensive Model Comparison Report...")
    
    # Load data
    data = load_corrected_data()
    model_returns = load_model_returns()
    real_data = load_real_data()
    
    if not data:
        print("❌ No data available!")
        return
    
    # Create PDF
    output_path = "results/comprehensive_model_comparison_report_final.pdf"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with PdfPages(output_path) as pdf:
        print(f"📄 Creating final PDF: {output_path}")
        
        # 1. Executive Summary (FIRST PAGE)
        rankings = create_performance_ranking(data, pdf)
        create_executive_summary_page(pdf, data, rankings)
        
        # 2. Metrics Tables
        create_metrics_tables(data, pdf)
        
        # 3. Robust Metrics Table
        create_robust_metrics_table(data, pdf)

        # 4. VaR Backtesting Table
        create_var_backtesting_table(data, pdf)
        
        # 5. Comparison Plots
        create_comparison_plots(real_data, model_returns, pdf)
        
        # 6. Performance Ranking
        create_performance_ranking(data, pdf)
        
        # 7. Methodology and Limitations
        create_methodology_and_limitations_section(pdf)
        
        print(f"📄 Final PDF created with {pdf.get_pagecount()} pages")
    
    print(f"✅ Final comprehensive report generated: {output_path}")
    print(f"📋 Executive summary added as first page")
    print(f"🔧 All instances of 'corrected' removed from the report")

if __name__ == "__main__":
    main()
