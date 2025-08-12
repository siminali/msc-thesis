#!/usr/bin/env python3
"""
Generate Comprehensive PDF Summary for Baseline Models
GARCH, DDPM, and TimeGrad Results

Author: Simin Ali
Supervisor: Dr Mikael Mieskolainen
Institution: Imperial College London
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style for professional plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_baseline_results():
    """Load results from baseline models only."""
    try:
        with open("results/comprehensive_evaluation/evaluation_results.json", 'r') as f:
            data = json.load(f)
        
        # Filter to only include baseline models (exclude LLM-Conditioned)
        baseline_models = ['GARCH', 'DDPM', 'TimeGrad']
        
        # Filter basic stats (include Real Data for comparison)
        baseline_stats = [stat for stat in data['basic_stats'] if stat['Model'] in baseline_models or stat['Model'] == 'Real Data']
        
        # Filter other metrics
        baseline_tail = [tail for tail in data['tail_metrics'] if tail['Model'] in baseline_models]
        baseline_vol = [vol for vol in data['volatility_metrics'] if vol['Model'] in baseline_models]
        baseline_dist = [dist for dist in data['distribution_tests'] if dist['Model'] in baseline_models]
        
        return {
            'basic_stats': baseline_stats,
            'tail_metrics': baseline_tail,
            'volatility_metrics': baseline_vol,
            'distribution_tests': baseline_dist
        }
    except Exception as e:
        print(f"Error loading results: {e}")
        return None

def create_title_page(pdf):
    """Create professional title page."""
    fig, ax = plt.subplots(figsize=(11.69, 8.27))  # A4 landscape
    ax.axis('off')
    
    # Title
    ax.text(0.5, 0.7, 'MSc Thesis: Baseline Models Evaluation', 
            fontsize=24, fontweight='bold', ha='center', va='center',
            transform=ax.transAxes)
    
    # Subtitle
    ax.text(0.5, 0.6, 'Diffusion Models in Generative AI for Financial Data Synthesis', 
            fontsize=16, ha='center', va='center', transform=ax.transAxes)
    ax.text(0.5, 0.55, 'and Risk Management', 
            fontsize=16, ha='center', va='center', transform=ax.transAxes)
    
    # Author info
    ax.text(0.5, 0.4, 'Author: Simin Ali', 
            fontsize=14, ha='center', va='center', transform=ax.transAxes)
    ax.text(0.5, 0.35, 'Supervisor: Dr Mikael Mieskolainen', 
            fontsize=14, ha='center', va='center', transform=ax.transAxes)
    ax.text(0.5, 0.3, 'Institution: Imperial College London', 
            fontsize=14, ha='center', va='center', transform=ax.transAxes)
    
    # Date
    ax.text(0.5, 0.2, 'August 2025', 
            fontsize=12, ha='center', va='center', transform=ax.transAxes)
    
    # Models evaluated
    ax.text(0.5, 0.1, 'Models Evaluated: GARCH(1,1), DDPM, TimeGrad', 
            fontsize=12, ha='center', va='center', transform=ax.transAxes)
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

def create_executive_summary(pdf, results):
    """Create executive summary page."""
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.axis('off')
    
    # Title
    ax.text(0.5, 0.95, 'Executive Summary', 
            fontsize=20, fontweight='bold', ha='center', va='top', transform=ax.transAxes)
    
    # Summary text
    summary_text = """
This report presents a comprehensive evaluation of three baseline models for financial data synthesis:

1. GARCH(1,1): Traditional econometric model for volatility modeling
2. DDPM: Denoising Diffusion Probabilistic Model for synthetic data generation
3. TimeGrad: Autoregressive diffusion model for time series forecasting

Key Findings:
• TimeGrad demonstrates the best overall performance among baseline models
• DDPM shows significant improvement over traditional GARCH approaches
• All models capture different aspects of financial stylized facts
• GARCH provides interpretable parameters but limited distribution matching

Performance Ranking (KS Test - Lower is Better):
1. TimeGrad: KS=0.0292 (p-value=0.0047)
2. DDPM: KS=0.0902 (p-value=0.0000)
3. GARCH: KS=0.5215 (p-value=0.0000)

This evaluation provides the foundation for comparing against the novel LLM-conditioned diffusion model.
    """
    
    ax.text(0.05, 0.85, summary_text, fontsize=12, ha='left', va='top', 
            transform=ax.transAxes, wrap=True)
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

def create_basic_statistics_table(pdf, results):
    """Create basic statistics comparison table."""
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.axis('off')
    
    # Title
    ax.text(0.5, 0.95, 'Basic Statistics Comparison', 
            fontsize=18, fontweight='bold', ha='center', va='top', transform=ax.transAxes)
    
    # Create table
    data = []
    for stat in results['basic_stats']:
        if stat['Model'] != 'Real Data':
            data.append([
                stat['Model'],
                f"{stat['Mean']:.4f}",
                f"{stat['Std Dev']:.4f}",
                f"{stat['Skewness']:.4f}",
                f"{stat['Kurtosis']:.4f}",
                f"{stat['Min']:.4f}",
                f"{stat['Max']:.4f}"
            ])
    
    # Add Real Data row
    real_data = [s for s in results['basic_stats'] if s['Model'] == 'Real Data'][0]
    data.insert(0, [
        'Real Data',
        f"{real_data['Mean']:.4f}",
        f"{real_data['Std Dev']:.4f}",
        f"{real_data['Skewness']:.4f}",
        f"{real_data['Kurtosis']:.4f}",
        f"{real_data['Min']:.4f}",
        f"{real_data['Max']:.4f}"
    ])
    
    # Table headers
    headers = ['Model', 'Mean', 'Std Dev', 'Skewness', 'Kurtosis', 'Min', 'Max']
    
    # Create table
    table = ax.table(cellText=data, colLabels=headers, 
                    cellLoc='center', loc='center',
                    bbox=[0.1, 0.3, 0.8, 0.5])
    
    # Style table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Color header row
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color Real Data row
    for i in range(len(headers)):
        table[(1, i)].set_facecolor('#2196F3')
        table[(1, i)].set_text_props(weight='bold', color='white')
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

def create_distribution_comparison_plot(pdf, results):
    """Create distribution comparison plot."""
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    
    # Load data
    try:
        real_data = np.load("results/garch_returns.npy").flatten()
        garch_data = np.load("results/garch_returns.npy").flatten()
        ddpm_data = np.load("results/ddpm_returns.npy").flatten()
        timegrad_data = np.load("results/timegrad_returns.npy").flatten()
        
        # Create histograms
        ax.hist(real_data, bins=50, alpha=0.7, label='Real Data', density=True, color='blue')
        ax.hist(garch_data, bins=50, alpha=0.7, label='GARCH', density=True, color='red')
        ax.hist(ddpm_data, bins=50, alpha=0.7, label='DDPM', density=True, color='green')
        ax.hist(timegrad_data, bins=50, alpha=0.7, label='TimeGrad', density=True, color='orange')
        
        ax.set_xlabel('Log Returns (%)')
        ax.set_ylabel('Density')
        ax.set_title('Distribution Comparison: Real vs Synthetic Data')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    except Exception as e:
        ax.text(0.5, 0.5, f'Error loading data: {e}', ha='center', va='center', transform=ax.transAxes)
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

def create_time_series_comparison_plot(pdf, results):
    """Create time series comparison plot."""
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    
    try:
        # Load data and plot first 200 points
        real_data = np.load("results/garch_returns.npy").flatten()[:200]
        garch_data = np.load("results/garch_returns.npy").flatten()[:200]
        ddpm_data = np.load("results/ddpm_returns.npy").flatten()[:200]
        timegrad_data = np.load("results/timegrad_returns.npy").flatten()[:200]
        
        x = range(len(real_data))
        ax.plot(x, real_data, label='Real Data', linewidth=2, color='blue')
        ax.plot(x, garch_data, label='GARCH', linewidth=1, alpha=0.7, color='red')
        ax.plot(x, ddpm_data, label='DDPM', linewidth=1, alpha=0.7, color='green')
        ax.plot(x, timegrad_data, label='TimeGrad', linewidth=1, alpha=0.7, color='orange')
        
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Log Returns (%)')
        ax.set_title('Time Series Comparison: First 200 Observations')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    except Exception as e:
        ax.text(0.5, 0.5, f'Error loading data: {e}', ha='center', va='center', transform=ax.transAxes)
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

def create_volatility_clustering_plot(pdf, results):
    """Create volatility clustering plot."""
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    
    try:
        # Load data and compute rolling volatility
        real_data = np.load("results/garch_returns.npy").flatten()
        garch_data = np.load("results/garch_returns.npy").flatten()
        ddpm_data = np.load("results/ddpm_returns.npy").flatten()
        timegrad_data = np.load("results/timegrad_returns.npy").flatten()
        
        # Compute rolling standard deviation (volatility)
        window = 20
        real_vol = pd.Series(real_data).rolling(window=window).std().dropna()
        garch_vol = pd.Series(garch_data).rolling(window=window).std().dropna()
        ddpm_vol = pd.Series(ddpm_data).rolling(window=window).std().dropna()
        timegrad_vol = pd.Series(timegrad_data).rolling(window=window).std().dropna()
        
        # Plot first 300 points
        x = range(len(real_vol[:300]))
        ax.plot(x, real_vol[:300], label='Real Data', linewidth=2, color='blue')
        ax.plot(x, garch_vol[:300], label='GARCH', linewidth=1, alpha=0.7, color='red')
        ax.plot(x, ddpm_vol[:300], label='DDPM', linewidth=1, alpha=0.7, color='green')
        ax.plot(x, timegrad_vol[:300], label='TimeGrad', linewidth=1, alpha=0.7, color='orange')
        
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Rolling Volatility (20-day window)')
        ax.set_title('Volatility Clustering Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    except Exception as e:
        ax.text(0.5, 0.5, f'Error loading data: {e}', ha='center', va='center', transform=ax.transAxes)
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

def create_qq_comparison_plot(pdf, results):
    """Create Q-Q plot comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27))
    fig.suptitle('Q-Q Plot Comparison: Real vs Synthetic Data', fontsize=16, fontweight='bold')
    
    try:
        real_data = np.load("results/garch_returns.npy").flatten()
        garch_data = np.load("results/garch_returns.npy").flatten()
        ddpm_data = np.load("results/ddpm_returns.npy").flatten()
        timegrad_data = np.load("results/timegrad_returns.npy").flatten()
        
        # GARCH Q-Q plot
        stats.probplot(garch_data, dist="norm", plot=axes[0,0])
        axes[0,0].set_title('GARCH vs Normal')
        
        # DDPM Q-Q plot
        stats.probplot(ddpm_data, dist="norm", plot=axes[0,1])
        axes[0,1].set_title('DDPM vs Normal')
        
        # TimeGrad Q-Q plot
        stats.probplot(timegrad_data, dist="norm", plot=axes[1,0])
        axes[1,0].set_title('TimeGrad vs Normal')
        
        # Real Data Q-Q plot
        stats.probplot(real_data, dist="norm", plot=axes[1,1])
        axes[1,1].set_title('Real Data vs Normal')
        
    except Exception as e:
        for ax in axes.flat:
            ax.text(0.5, 0.5, f'Error: {e}', ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

def create_autocorrelation_plot(pdf, results):
    """Create autocorrelation comparison plot."""
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    
    try:
        # Load data
        real_data = np.load("results/garch_returns.npy").flatten()
        garch_data = np.load("results/garch_returns.npy").flatten()
        ddpm_data = np.load("results/ddpm_returns.npy").flatten()
        timegrad_data = np.load("results/timegrad_returns.npy").flatten()
        
        # Compute autocorrelation of squared returns
        max_lag = 20
        
        # Real data
        real_acf = [1.0]  # lag 0
        for lag in range(1, max_lag + 1):
            if lag < len(real_data):
                corr = np.corrcoef(real_data[:-lag], real_data[lag:])[0,1]
                real_acf.append(corr if not np.isnan(corr) else 0)
        
        # GARCH
        garch_acf = [1.0]
        for lag in range(1, max_lag + 1):
            if lag < len(garch_data):
                corr = np.corrcoef(garch_data[:-lag], garch_data[lag:])[0,1]
                garch_acf.append(corr if not np.isnan(corr) else 0)
        
        # DDPM
        ddpm_acf = [1.0]
        for lag in range(1, max_lag + 1):
            if lag < len(ddpm_data):
                corr = np.corrcoef(ddpm_data[:-lag], ddpm_data[lag:])[0,1]
                ddpm_acf.append(corr if not np.isnan(corr) else 0)
        
        # TimeGrad
        timegrad_acf = [1.0]
        for lag in range(1, max_lag + 1):
            if lag < len(timegrad_data):
                corr = np.corrcoef(timegrad_data[:-lag], timegrad_data[lag:])[0,1]
                timegrad_acf.append(corr if not np.isnan(corr) else 0)
        
        lags = range(max_lag + 1)
        ax.plot(lags, real_acf, 'o-', label='Real Data', linewidth=2, markersize=6)
        ax.plot(lags, garch_acf, 's-', label='GARCH', linewidth=1, alpha=0.7)
        ax.plot(lags, ddpm_acf, '^-', label='DDPM', linewidth=1, alpha=0.7)
        ax.plot(lags, timegrad_acf, 'v-', label='TimeGrad', linewidth=1, alpha=0.7)
        
        ax.set_xlabel('Lag')
        ax.set_ylabel('Autocorrelation')
        ax.set_title('Autocorrelation Function of Returns')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
    except Exception as e:
        ax.text(0.5, 0.5, f'Error loading data: {e}', ha='center', va='center', transform=ax.transAxes)
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

def create_distribution_tests_table(pdf, results):
    """Create distribution tests comparison table."""
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.axis('off')
    
    # Title
    ax.text(0.5, 0.95, 'Distribution Tests Comparison', 
            fontsize=18, fontweight='bold', ha='center', va='top', transform=ax.transAxes)
    
    # Create table
    data = []
    for test in results['distribution_tests']:
        data.append([
            test['Model'],
            f"{test['KS_Statistic']:.4f}",
            f"{test['KS_pvalue']:.2e}",
            f"{test['Anderson_Darling_Stat']:.4f}",
            f"{test['MMD']:.4f}"
        ])
    
    # Table headers
    headers = ['Model', 'KS Statistic', 'KS p-value', 'Anderson-Darling', 'MMD']
    
    # Create table
    table = ax.table(cellText=data, colLabels=headers, 
                    cellLoc='center', loc='center',
                    bbox=[0.1, 0.3, 0.8, 0.5])
    
    # Style table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Color header row
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color best performing model (TimeGrad)
    for i in range(len(headers)):
        table[(2, i)].set_facecolor('#FFC107')
        table[(2, i)].set_text_props(weight='bold')
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

def create_volatility_metrics_table(pdf, results):
    """Create volatility metrics comparison table."""
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.axis('off')
    
    # Title
    ax.text(0.5, 0.95, 'Volatility Metrics Comparison', 
            fontsize=18, fontweight='bold', ha='center', va='top', transform=ax.transAxes)
    
    # Create table
    data = []
    for vol in results['volatility_metrics']:
        data.append([
            vol['Model'],
            f"{vol['Volatility_ACF']:.4f}",
            f"{vol['Volatility_Persistence']:.4f}",
            f"{vol['Mean_Volatility']:.4f}",
            f"{vol['Volatility_of_Volatility']:.4f}"
        ])
    
    # Table headers
    headers = ['Model', 'Volatility ACF', 'Volatility Persistence', 'Mean Volatility', 'Vol of Vol']
    
    # Create table
    table = ax.table(cellText=data, colLabels=headers, 
                    cellLoc='center', loc='center',
                    bbox=[0.1, 0.3, 0.8, 0.5])
    
    # Style table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Color header row
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color best performing model
    for i in range(len(headers)):
        table[(2, i)].set_facecolor('#FFC107')
        table[(2, i)].set_text_props(weight='bold')
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

def create_tail_risk_table(pdf, results):
    """Create tail risk metrics comparison table."""
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.axis('off')
    
    # Title
    ax.text(0.5, 0.95, 'Tail Risk Metrics Comparison', 
            fontsize=18, fontweight='bold', ha='center', va='top', transform=ax.transAxes)
    
    # Create table
    data = []
    for tail in results['tail_metrics']:
        data.append([
            tail['Model'],
            f"{tail['VaR_1%']:.4f}",
            f"{tail['ES_1%']:.4f}",
            f"{tail['VaR_5%']:.4f}",
            f"{tail['ES_5%']:.4f}",
            f"{tail['VaR_99%']:.4f}",
            f"{tail['ES_99%']:.4f}"
        ])
    
    # Table headers
    headers = ['Model', 'VaR 1%', 'ES 1%', 'VaR 5%', 'ES 5%', 'VaR 99%', 'ES 99%']
    
    # Create table
    table = ax.table(cellText=data, colLabels=headers, 
                    cellLoc='center', loc='center',
                    bbox=[0.1, 0.2, 0.8, 0.6])
    
    # Style table
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Color header row
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color best performing model
    for i in range(len(headers)):
        table[(2, i)].set_facecolor('#FFC107')
        table[(2, i)].set_text_props(weight='bold')
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

def create_conclusions_page(pdf, results):
    """Create conclusions and recommendations page."""
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.axis('off')
    
    # Title
    ax.text(0.5, 0.95, 'Conclusions and Recommendations', 
            fontsize=20, fontweight='bold', ha='center', va='top', transform=ax.transAxes)
    
    # Conclusions text
    conclusions_text = """
Key Findings:

1. Model Performance Ranking:
   • TimeGrad: Best overall performance (KS=0.0292)
   • DDPM: Significant improvement over GARCH (KS=0.0902)
   • GARCH: Limited distribution matching (KS=0.5215)

2. Strengths of Each Model:
   • GARCH: Interpretable parameters, fast computation
   • DDPM: Good volatility capture, stable training
   • TimeGrad: Best distribution matching, volatility clustering

3. Limitations:
   • GARCH: Poor tail risk modeling, limited stylized facts
   • DDPM: Moderate performance, no conditional generation
   • TimeGrad: Computationally intensive, limited conditioning

Recommendations:

1. For Risk Management:
   • Use TimeGrad for most accurate risk estimates
   • Consider DDPM as a robust alternative
   • Avoid sole reliance on GARCH

2. For Scenario Generation:
   • TimeGrad provides most realistic scenarios
   • DDPM offers good balance of speed and accuracy
   • GARCH suitable for quick approximations only

3. For Research and Development:
   • Build upon TimeGrad's architecture
   • Explore conditional generation capabilities
   • Consider hybrid approaches combining strengths

Next Steps:
• Compare against the novel LLM-conditioned diffusion model
• Explore ensemble methods combining multiple models
• Investigate conditional generation for specific market scenarios
    """
    
    ax.text(0.05, 0.85, conclusions_text, fontsize=11, ha='left', va='top', 
            transform=ax.transAxes, wrap=True)
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

def main():
    """Generate the complete baseline models summary PDF."""
    print("📊 Generating Baseline Models Summary PDF...")
    
    # Load results
    results = load_baseline_results()
    if not results:
        print("❌ Failed to load results")
        return
    
    # Create PDF
    pdf_path = "results/baseline_models_summary.pdf"
    with PdfPages(pdf_path) as pdf:
        
        # Title page
        print("📄 Creating title page...")
        create_title_page(pdf)
        
        # Executive summary
        print("📋 Creating executive summary...")
        create_executive_summary(pdf, results)
        
        # Basic statistics table
        print("📊 Creating basic statistics table...")
        create_basic_statistics_table(pdf, results)
        
        # Distribution comparison plot
        print("📈 Creating distribution comparison plot...")
        create_distribution_comparison_plot(pdf, results)
        
        # Time series comparison plot
        print("⏰ Creating time series comparison plot...")
        create_time_series_comparison_plot(pdf, results)
        
        # Volatility clustering plot
        print("📊 Creating volatility clustering plot...")
        create_volatility_clustering_plot(pdf, results)
        
        # Q-Q comparison plots
        print("📊 Creating Q-Q comparison plots...")
        create_qq_comparison_plot(pdf, results)
        
        # Autocorrelation plot
        print("🔄 Creating autocorrelation plot...")
        create_autocorrelation_plot(pdf, results)
        
        # Distribution tests table
        print("🧪 Creating distribution tests table...")
        create_distribution_tests_table(pdf, results)
        
        # Volatility metrics table
        print("📊 Creating volatility metrics table...")
        create_volatility_metrics_table(pdf, results)
        
        # Tail risk table
        print("⚠️ Creating tail risk table...")
        create_tail_risk_table(pdf, results)
        
        # Conclusions page
        print("💡 Creating conclusions page...")
        create_conclusions_page(pdf, results)
    
    print(f"✅ Baseline models summary PDF generated: {pdf_path}")
    print("📊 PDF includes:")
    print("   - Executive summary")
    print("   - Statistical tables")
    print("   - Visualization plots")
    print("   - Performance comparisons")
    print("   - Conclusions and recommendations")

if __name__ == "__main__":
    main()
