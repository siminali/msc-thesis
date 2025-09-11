#!/usr/bin/env python3
"""
Create comprehensive training data visualizations and tables for LaTeX import.
Generates organized folder structure with all graphs and tables needed for thesis.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def setup_output_directory():
    """Create organized output directory structure."""
    base_dir = Path("latex_training_exports")
    subdirs = [
        "figures/loss_curves",
        "figures/performance", 
        "figures/system_info",
        "tables",
        "data"
    ]
    
    base_dir.mkdir(exist_ok=True)
    for subdir in subdirs:
        (base_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    return base_dir

def load_training_data():
    """Load all training data from runs directory."""
    runs_dir = Path("runs")
    training_data = {}
    
    for model_dir in runs_dir.iterdir():
        if not model_dir.is_dir():
            continue
            
        model_name = model_dir.name
        training_data[model_name] = []
        
        for run_dir in model_dir.iterdir():
            if not run_dir.is_dir():
                continue
                
            run_data = {
                'run_id': run_dir.name,
                'model': model_name,
                'path': str(run_dir)
            }
            
            # Load training history
            history_file = run_dir / "training_history.json"
            if history_file.exists():
                with open(history_file) as f:
                    history = json.load(f)
                    run_data['train_losses'] = history.get('train_losses', [])
                    run_data['val_losses'] = history.get('val_losses', [])
                    run_data['best_epoch'] = history.get('best_epoch', None)
            
            # Load metadata
            metadata_file = run_dir / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file) as f:
                    metadata = json.load(f)
                    run_data['training_time'] = metadata.get('training_time_seconds', None)
                    run_data['model_parameters'] = metadata.get('model_parameters', None)
                    run_data['gpu_info'] = metadata.get('gpu_info', {})
                    run_data['dataset_summary'] = metadata.get('dataset_summary', {})
            
            # Load run config
            config_file = run_dir / "run_config.json"
            if config_file.exists():
                with open(config_file) as f:
                    config = json.load(f)
                    run_data['config'] = config
            
            training_data[model_name].append(run_data)
    
    return training_data

def create_loss_curves(training_data, output_dir):
    """Create individual and combined loss curve plots."""
    print("📈 Creating loss curve plots...")
    
    # Individual loss curves for each model
    for model_name, runs in training_data.items():
        if not runs:
            continue
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{model_name.upper()} Training Progress', fontsize=16, fontweight='bold')
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(runs)))
        
        for idx, run_data in enumerate(runs):
            if not run_data.get('train_losses'):
                continue
                
            train_losses = run_data['train_losses']
            val_losses = run_data.get('val_losses', [])
            epochs = range(len(train_losses))
            
            # Plot in first subplot
            ax = axes[0, 0]
            ax.plot(epochs, train_losses, color=colors[idx], alpha=0.7, 
                   label=f"Run {run_data['run_id'][-8:]}", linewidth=2)
            ax.set_title('Training Loss')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Validation losses if available
            if val_losses:
                ax = axes[0, 1]
                ax.plot(epochs[:len(val_losses)], val_losses, color=colors[idx], alpha=0.7,
                       label=f"Run {run_data['run_id'][-8:]}", linewidth=2)
                ax.set_title('Validation Loss')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.grid(True, alpha=0.3)
                ax.legend()
            
            # Combined train/val for this run
            if val_losses:
                ax = axes[1, 0]
                ax.plot(epochs, train_losses, '--', color=colors[idx], alpha=0.7, 
                       label=f"Train {run_data['run_id'][-8:]}")
                ax.plot(epochs[:len(val_losses)], val_losses, '-', color=colors[idx], alpha=0.7,
                       label=f"Val {run_data['run_id'][-8:]}")
                ax.set_title('Train vs Validation Loss')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.grid(True, alpha=0.3)
                ax.legend()
        
        # Final loss distribution
        final_losses = [run['train_losses'][-1] for run in runs if run.get('train_losses')]
        if final_losses:
            ax = axes[1, 1]
            ax.hist(final_losses, bins=max(3, len(final_losses)//2), alpha=0.7, color='skyblue', edgecolor='black')
            ax.set_title('Final Training Loss Distribution')
            ax.set_xlabel('Final Loss')
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plots
        safe_name = model_name.replace('/', '_').replace(' ', '_')
        plt.savefig(output_dir / "figures/loss_curves" / f"{safe_name}_training_progress.pdf", 
                   bbox_inches='tight', dpi=300)
        plt.savefig(output_dir / "figures/loss_curves" / f"{safe_name}_training_progress.png", 
                   bbox_inches='tight', dpi=300)
        plt.close()

def create_performance_comparison(training_data, output_dir):
    """Create performance comparison visualizations."""
    print("⚡ Creating performance comparison plots...")
    
    # Collect performance data
    perf_data = []
    for model_name, runs in training_data.items():
        for run in runs:
            if run.get('training_time') and run.get('model_parameters'):
                perf_data.append({
                    'Model': model_name,
                    'Training_Time_Minutes': run['training_time'] / 60,
                    'Parameters_Thousands': run['model_parameters'] / 1000,
                    'Final_Loss': run['train_losses'][-1] if run.get('train_losses') else None,
                    'Best_Epoch': run.get('best_epoch', None)
                })
    
    if not perf_data:
        print("⚠️  No performance data available")
        return
    
    df = pd.DataFrame(perf_data)
    
    # Create performance comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
    
    # Training time comparison
    ax = axes[0, 0]
    df_grouped = df.groupby('Model')['Training_Time_Minutes'].mean().sort_values()
    bars = ax.bar(range(len(df_grouped)), df_grouped.values, color='lightcoral', alpha=0.8)
    ax.set_xticks(range(len(df_grouped)))
    ax.set_xticklabels(df_grouped.index, rotation=45, ha='right')
    ax.set_ylabel('Training Time (Minutes)')
    ax.set_title('Average Training Time by Model')
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom')
    
    # Model complexity (parameters)
    ax = axes[0, 1]
    df_grouped = df.groupby('Model')['Parameters_Thousands'].mean().sort_values()
    bars = ax.bar(range(len(df_grouped)), df_grouped.values, color='lightblue', alpha=0.8)
    ax.set_xticks(range(len(df_grouped)))
    ax.set_xticklabels(df_grouped.index, rotation=45, ha='right')
    ax.set_ylabel('Parameters (Thousands)')
    ax.set_title('Model Complexity')
    ax.grid(True, alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}K', ha='center', va='bottom')
    
    # Training efficiency (time vs parameters)
    ax = axes[1, 0]
    for model in df['Model'].unique():
        model_data = df[df['Model'] == model]
        ax.scatter(model_data['Parameters_Thousands'], model_data['Training_Time_Minutes'],
                  label=model, alpha=0.7, s=100)
    ax.set_xlabel('Parameters (Thousands)')
    ax.set_ylabel('Training Time (Minutes)')
    ax.set_title('Training Efficiency')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Final loss comparison
    ax = axes[1, 1]
    final_losses = df.dropna(subset=['Final_Loss'])
    if not final_losses.empty:
        df_grouped = final_losses.groupby('Model')['Final_Loss'].mean().sort_values()
        bars = ax.bar(range(len(df_grouped)), df_grouped.values, color='lightgreen', alpha=0.8)
        ax.set_xticks(range(len(df_grouped)))
        ax.set_xticklabels(df_grouped.index, rotation=45, ha='right')
        ax.set_ylabel('Final Training Loss')
        ax.set_title('Final Loss Comparison')
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / "figures/performance" / "model_performance_comparison.pdf", 
               bbox_inches='tight', dpi=300)
    plt.savefig(output_dir / "figures/performance" / "model_performance_comparison.png", 
               bbox_inches='tight', dpi=300)
    plt.close()

def create_system_info_plots(training_data, output_dir):
    """Create system information visualizations."""
    print("💻 Creating system information plots...")
    
    # GPU usage analysis
    gpu_data = []
    for model_name, runs in training_data.items():
        for run in runs:
            gpu_info = run.get('gpu_info', {})
            gpu_data.append({
                'Model': model_name,
                'CUDA_Available': gpu_info.get('cuda_available', False),
                'Device': 'GPU' if gpu_info.get('cuda_available') else 'CPU',
                'Training_Time': run.get('training_time', 0) / 60  # Convert to minutes
            })
    
    if gpu_data:
        df_gpu = pd.DataFrame(gpu_data)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Hardware Utilization Analysis', fontsize=14, fontweight='bold')
        
        # Device usage distribution
        ax = axes[0]
        device_counts = df_gpu['Device'].value_counts()
        ax.pie(device_counts.values, labels=device_counts.index, autopct='%1.1f%%',
               colors=['lightcoral', 'lightblue'])
        ax.set_title('Training Device Distribution')
        
        # Training time by device
        ax = axes[1]
        df_gpu.boxplot(column='Training_Time', by='Device', ax=ax)
        ax.set_title('Training Time by Device Type')
        ax.set_xlabel('Device Type')
        ax.set_ylabel('Training Time (Minutes)')
        plt.suptitle('')  # Remove automatic title
        
        plt.tight_layout()
        plt.savefig(output_dir / "figures/system_info" / "hardware_utilization.pdf", 
                   bbox_inches='tight', dpi=300)
        plt.savefig(output_dir / "figures/system_info" / "hardware_utilization.png", 
                   bbox_inches='tight', dpi=300)
        plt.close()

def create_tables(training_data, output_dir):
    """Create comprehensive tables for LaTeX import."""
    print("📊 Creating data tables...")
    
    # Training summary table
    summary_data = []
    for model_name, runs in training_data.items():
        for run in runs:
            summary_data.append({
                'Model': model_name,
                'Run_ID': run['run_id'],
                'Training_Time_Min': f"{run.get('training_time', 0)/60:.1f}" if run.get('training_time') else "N/A",
                'Parameters': f"{run.get('model_parameters', 0):,}" if run.get('model_parameters') else "N/A",
                'Final_Train_Loss': f"{run['train_losses'][-1]:.4f}" if run.get('train_losses') else "N/A",
                'Final_Val_Loss': f"{run['val_losses'][-1]:.4f}" if run.get('val_losses') and run['val_losses'] else "N/A",
                'Best_Epoch': run.get('best_epoch', "N/A"),
                'Total_Epochs': len(run.get('train_losses', [])),
                'Device': 'GPU' if run.get('gpu_info', {}).get('cuda_available') else 'CPU'
            })
    
    # Save training summary
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(output_dir / "tables" / "training_summary.csv", index=False)
    
    # Create LaTeX table
    latex_table = df_summary.to_latex(
        index=False,
        caption="Training Summary for All Models",
        label="tab:training_summary",
        column_format="l|l|r|r|r|r|r|r|l",
        escape=False
    )
    
    with open(output_dir / "tables" / "training_summary.tex", 'w') as f:
        f.write(latex_table)
    
    # Model comparison table
    model_stats = []
    for model_name, runs in training_data.items():
        if not runs:
            continue
            
        # Calculate statistics across runs
        train_times = [r.get('training_time', 0)/60 for r in runs if r.get('training_time')]
        parameters = [r.get('model_parameters', 0) for r in runs if r.get('model_parameters')]
        final_losses = [r['train_losses'][-1] for r in runs if r.get('train_losses')]
        
        model_stats.append({
            'Model': model_name,
            'Num_Runs': len(runs),
            'Avg_Training_Time_Min': f"{np.mean(train_times):.1f}" if train_times else "N/A",
            'Std_Training_Time_Min': f"{np.std(train_times):.1f}" if len(train_times) > 1 else "N/A",
            'Parameters': f"{parameters[0]:,}" if parameters else "N/A",
            'Avg_Final_Loss': f"{np.mean(final_losses):.4f}" if final_losses else "N/A",
            'Std_Final_Loss': f"{np.std(final_losses):.4f}" if len(final_losses) > 1 else "N/A",
            'Best_Final_Loss': f"{np.min(final_losses):.4f}" if final_losses else "N/A"
        })
    
    # Save model comparison
    df_models = pd.DataFrame(model_stats)
    df_models.to_csv(output_dir / "tables" / "model_comparison.csv", index=False)
    
    # Create LaTeX table
    latex_table = df_models.to_latex(
        index=False,
        caption="Model Comparison Statistics",
        label="tab:model_comparison",
        column_format="l|r|r|r|r|r|r|r",
        escape=False
    )
    
    with open(output_dir / "tables" / "model_comparison.tex", 'w') as f:
        f.write(latex_table)
    
    print(f"✅ Created tables: training_summary.tex, model_comparison.tex")

def create_readme(output_dir):
    """Create README with instructions for LaTeX import."""
    readme_content = f"""# Training Data Exports for LaTeX

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

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
\\begin{{figure}}[htbp]
    \\centering
    \\includegraphics[width=0.8\\textwidth]{{latex_training_exports/figures/loss_curves/ddpm_evaluation_training_progress.pdf}}
    \\caption{{DDPM Training Progress}}
    \\label{{fig:ddpm_training}}
\\end{{figure}}

% Performance comparison
\\begin{{figure}}[htbp]
    \\centering
    \\includegraphics[width=0.9\\textwidth]{{latex_training_exports/figures/performance/model_performance_comparison.pdf}}
    \\caption{{Model Performance Comparison}}
    \\label{{fig:performance_comparison}}
\\end{{figure}}
```

### Including Tables

```latex
% Training summary table
\\input{{latex_training_exports/tables/training_summary.tex}}

% Model comparison table
\\input{{latex_training_exports/tables/model_comparison.tex}}
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
"""
    
    with open(output_dir / "README.md", 'w') as f:
        f.write(readme_content)

def main():
    """Main execution function."""
    print("🚀 Creating LaTeX Training Data Exports")
    print("=" * 50)
    
    # Setup output directory
    output_dir = setup_output_directory()
    print(f"📁 Output directory: {output_dir}")
    
    # Load training data
    print("📖 Loading training data...")
    training_data = load_training_data()
    
    if not training_data:
        print("❌ No training data found in runs/ directory")
        return
    
    total_runs = sum(len(runs) for runs in training_data.values())
    print(f"✅ Loaded {len(training_data)} models with {total_runs} total runs")
    
    # Create visualizations and tables
    create_loss_curves(training_data, output_dir)
    create_performance_comparison(training_data, output_dir)
    create_system_info_plots(training_data, output_dir)
    create_tables(training_data, output_dir)
    create_readme(output_dir)
    
    print("\n🎉 Export Complete!")
    print(f"📂 All files saved to: {output_dir}")
    print("\n📋 Summary:")
    print(f"   • Loss curves: {len(training_data)} models")
    print(f"   • Performance plots: 1 comprehensive comparison")
    print(f"   • System info: Hardware utilization analysis")
    print(f"   • Tables: 2 LaTeX-ready tables")
    print(f"   • Documentation: README with LaTeX instructions")
    
    print("\n💡 Next steps:")
    print("   1. Review generated files in latex_training_exports/")
    print("   2. Copy relevant figures to your LaTeX project")
    print("   3. Use \\input{} commands for tables")
    print("   4. See README.md for detailed LaTeX integration")

if __name__ == "__main__":
    main()

