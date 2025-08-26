#!/usr/bin/env python3
"""
Comprehensive Comparison Framework for DDPM Approaches
Integrates Zero-Conditioned, Explicitly-Conditioned, and LLM-Conditioned Models

Author: Simin Ali
Thesis: Diffusion Models in Generative AI for Financial Data Synthesis and Risk Management
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import the three model implementations
from explicit_cond_ddpm import (
    load_and_prepare_data, 
    create_conditioning_vectors, 
    create_sequences,
    ExplicitConditioningDDPM,
    ExplicitConditioningTrainer,
    EMAModel
)

# Import the refactored LLM model
from llm_conditioned_diffusion_refactored import (
    NewsDataLoader,
    LLMConditionedDiffusion,
    LLMDiffusionTrainer,
    ControllabilityProbe,
    create_time_based_splits
)

class ComprehensiveComparisonFramework:
    """Framework for comparing different DDPM conditioning approaches."""
    
    def __init__(self, results_dir="results/comprehensive_comparison", seed=42):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        
        # Set reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Model results storage
        self.model_results = {}
        self.comparison_metrics = {}
        
        print(f"Comprehensive Comparison Framework initialized")
        print(f"Results directory: {self.results_dir}")
    
    def run_zero_conditioned_evaluation(self, args):
        """Run zero-conditioned DDPM evaluation."""
        print("\n" + "="*60)
        print("ZERO-CONDITIONED DDPM EVALUATION")
        print("="*60)
        
        # Load data
        returns = load_and_prepare_data()
        
        # Create sequences
        X = create_sequences(returns, args.seq_len)
        
        # Split data
        num_sequences = len(X)
        train_split_idx = int(num_sequences * (1 - args.val_split))
        
        X_train = X[:train_split_idx]
        X_val = X[train_split_idx:]
        
        # Create zero conditioning (no explicit conditioning)
        conditioning_dim = 5  # Same as explicit model for consistency
        zero_conditioning_train = np.zeros((len(X_train), conditioning_dim))
        zero_conditioning_val = np.zeros((len(X_val), conditioning_dim))
        
        # Initialize model
        model = ExplicitConditioningDDPM(
            sequence_length=args.seq_len,
            conditioning_dim=conditioning_dim,
            hidden_dim=args.hidden_dim
        )
        
        # Initialize trainer
        trainer = ExplicitConditioningTrainer(
            model, 
            num_timesteps=args.num_timesteps, 
            beta_schedule=args.beta_schedule, 
            device=args.device,
            grad_clip=args.grad_clip,
            cfg_p=0.0  # No conditioning dropout for zero-conditioned
        )
        
        # Train model
        print("Training zero-conditioned model...")
        model, trainer, history = self._train_explicit_model(
            X_train, X_val, zero_conditioning_train, zero_conditioning_val, args
        )
        
        # Evaluate
        print("Evaluating zero-conditioned model...")
        zero_metrics = self._evaluate_model(
            model, trainer, None, zero_conditioning_val, X_val, returns.values, 
            "zero_conditioned", args
        )
        
        self.model_results['zero_conditioned'] = {
            'model': model,
            'trainer': trainer,
            'metrics': zero_metrics,
            'history': history
        }
        
        print("Zero-conditioned evaluation completed")
        return zero_metrics
    
    def run_explicit_conditioned_evaluation(self, args):
        """Run explicitly-conditioned DDPM evaluation."""
        print("\n" + "="*60)
        print("EXPLICITLY-CONDITIONED DDPM EVALUATION")
        print("="*60)
        
        # Load data
        returns = load_and_prepare_data()
        
        # Create conditioning vectors
        conditioning_vectors, regime_labels, metadata = create_conditioning_vectors(
            returns, args.seq_len, args.vol_window, args.val_split
        )
        
        # Create sequences
        X = create_sequences(returns, args.seq_len)
        
        # Split data
        num_sequences = len(X)
        train_split_idx = int(num_sequences * (1 - args.val_split))
        
        X_train = X[:train_split_idx]
        X_val = X[train_split_idx:]
        conditioning_train = conditioning_vectors[:train_split_idx]
        conditioning_val = conditioning_vectors[train_split_idx:]
        
        # Initialize model
        model = ExplicitConditioningDDPM(
            sequence_length=args.seq_len,
            conditioning_dim=conditioning_train.shape[1],
            hidden_dim=args.hidden_dim
        )
        
        # Initialize trainer
        trainer = ExplicitConditioningTrainer(
            model, 
            num_timesteps=args.num_timesteps, 
            beta_schedule=args.beta_schedule, 
            device=args.device,
            grad_clip=args.grad_clip,
            cfg_p=args.cfg_p
        )
        
        # Train model
        print("Training explicitly-conditioned model...")
        model, trainer, history = self._train_explicit_model(
            X_train, X_val, conditioning_train, conditioning_val, args
        )
        
        # Evaluate
        print("Evaluating explicitly-conditioned model...")
        explicit_metrics = self._evaluate_model(
            model, trainer, None, conditioning_val, X_val, returns.values, 
            "explicit_conditioned", args
        )
        
        self.model_results['explicit_conditioned'] = {
            'model': model,
            'trainer': trainer,
            'metrics': explicit_metrics,
            'history': history,
            'metadata': metadata
        }
        
        print("Explicitly-conditioned evaluation completed")
        return explicit_metrics
    
    def run_llm_conditioned_evaluation(self, args):
        """Run LLM-conditioned DDPM evaluation."""
        print("\n" + "="*60)
        print("LLM-CONDITIONED DDPM EVALUATION")
        print("="*60)
        
        # Load data
        returns = load_and_prepare_data()
        
        # Create time-based splits
        X_train, X_val, X_test, train_dates, val_dates, test_dates = create_time_based_splits(
            returns, args.seq_len
        )
        
        # Initialize news data loader
        news_loader = NewsDataLoader()
        
        # Create conditioning vectors for each split
        print("Creating conditioning vectors for training split...")
        conditioning_train = news_loader.create_conditioning_vectors(train_dates, args.seq_len)
        
        print("Creating conditioning vectors for validation split...")
        conditioning_val = news_loader.create_conditioning_vectors(val_dates, args.seq_len)
        
        print("Creating conditioning vectors for test split...")
        conditioning_test = news_loader.create_conditioning_vectors(test_dates, args.seq_len)
        
        # Initialize model
        model = LLMConditionedDiffusion(
            sequence_length=args.seq_len,
            conditioning_dim=conditioning_train.shape[1],
            hidden_dim=args.hidden_dim
        )
        
        # Initialize trainer
        trainer = LLMDiffusionTrainer(
            model, 
            num_timesteps=args.num_timesteps, 
            beta_schedule=args.beta_schedule, 
            device=args.device,
            grad_clip=args.grad_clip,
            cfg_p=args.cfg_p
        )
        
        # Train model
        print("Training LLM-conditioned model...")
        model, trainer, history = self._train_llm_model(
            X_train, X_val, conditioning_train, conditioning_val, args
        )
        
        # Train controllability probe
        print("Training controllability probe...")
        probe = ControllabilityProbe()
        
        # Compute realized volatilities and trends for training data
        train_volatilities, train_trends = self._compute_volatility_trends(X_train, args.vol_window)
        
        # Train probe
        probe.train(conditioning_train, train_volatilities, train_trends)
        
        # Evaluate
        print("Evaluating LLM-conditioned model...")
        llm_metrics = self._evaluate_llm_model(
            model, trainer, probe, conditioning_test, X_test, returns.values, 
            "llm_conditioned", args
        )
        
        self.model_results['llm_conditioned'] = {
            'model': model,
            'trainer': trainer,
            'probe': probe,
            'metrics': llm_metrics,
            'history': history,
            'dates': {'train': train_dates, 'val': val_dates, 'test': test_dates}
        }
        
        print("LLM-conditioned evaluation completed")
        return llm_metrics
    
    def _train_explicit_model(self, X_train, X_val, conditioning_train, conditioning_val, args):
        """Train explicit conditioning model."""
        # Prepare data
        train_dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(conditioning_train, dtype=torch.float32)
        )
        val_dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(conditioning_val, dtype=torch.float32)
        )
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        
        # Training setup
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        
        # Training loop
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(args.epochs):
            # Training
            model.train()
            epoch_train_losses = []
            
            for batch_x, batch_conditioning in train_loader:
                batch_x = batch_x.to(args.device)
                batch_conditioning = batch_conditioning.to(args.device)
                
                loss = trainer.train_step(batch_x, batch_conditioning, optimizer)
                epoch_train_losses.append(loss)
            
            # Validation
            model.eval()
            epoch_val_losses = []
            
            with torch.no_grad():
                for batch_x, batch_conditioning in val_loader:
                    batch_x = batch_x.to(args.device)
                    batch_conditioning = batch_conditioning.to(args.device)
                    
                    # Sample random timesteps for validation
                    batch_size = batch_x.shape[0]
                    t = torch.randint(0, args.num_timesteps, (batch_size,), device=args.device)
                    
                    # Add noise
                    x_noisy, noise = trainer.add_noise(batch_x, t)
                    
                    # Predict noise
                    t_normalized = t.float() / args.num_timesteps
                    predicted_noise = model(x_noisy, t_normalized.unsqueeze(-1), batch_conditioning)
                    
                    # Compute loss
                    loss = torch.nn.functional.mse_loss(predicted_noise, noise)
                    epoch_val_losses.append(loss.item())
            
            # Record losses
            avg_train_loss = np.mean(epoch_train_losses)
            avg_val_loss = np.mean(epoch_val_losses)
            
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            
            # Learning rate scheduling
            scheduler.step()
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Print progress
            if epoch % 10 == 0:
                print(f"Epoch {epoch:3d}: Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
            
            # Early stopping
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        # Save training history
        history = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_epoch': epoch - patience_counter
        }
        
        print(f"Training completed. Best validation loss: {best_val_loss:.6f}")
        return model, trainer, history
    
    def _train_llm_model(self, X_train, X_val, conditioning_train, conditioning_val, args):
        """Train LLM conditioning model."""
        # This would call the LLM training function
        # For now, we'll use a simplified version
        return self._train_explicit_model(X_train, X_val, conditioning_train, conditioning_val, args)
    
    def _compute_volatility_trends(self, X, vol_window):
        """Compute volatility and trend for sequences."""
        volatilities = []
        trends = []
        
        for i in range(len(X)):
            seq_returns = X[i, 0, :]  # Remove channel dimension
            
            # Compute volatility
            rolling_stds = []
            for j in range(len(seq_returns) - vol_window + 1):
                rolling_stds.append(np.std(seq_returns[j:j+vol_window], ddof=1))
            vol = np.mean(rolling_stds[-vol_window:])
            volatilities.append(vol)
            
            # Compute trend
            trend = seq_returns.sum()
            trends.append(trend)
        
        return np.array(volatilities), np.array(trends)
    
    def _evaluate_model(self, model, trainer, ema, conditioning, X, real_returns, model_name, args):
        """Evaluate a model and return metrics."""
        # Generate samples
        num_samples = min(1000, len(conditioning))
        device = next(model.parameters()).device
        conditioning_tensor = torch.tensor(conditioning[:num_samples], dtype=torch.float32, device=device)
        
        samples = trainer.sample(
            conditioning_tensor, 
            num_samples=num_samples, 
            sampler=args.sampler, 
            sample_steps=args.sample_steps,
            cfg_scale=args.cfg_scale
        )
        
        samples = samples.squeeze(1).cpu().numpy()
        synthetic_returns = samples.flatten()
        
        # Basic statistics
        real_stats = {
            'mean': np.mean(real_returns),
            'std': np.std(real_returns),
            'skew': scipy.stats.skew(real_returns),
            'kurtosis': scipy.stats.kurtosis(real_returns)
        }
        
        synthetic_stats = {
            'mean': np.mean(synthetic_returns),
            'std': np.std(synthetic_returns),
            'skew': scipy.stats.skew(synthetic_returns),
            'kurtosis': scipy.stats.kurtosis(synthetic_returns)
        }
        
        # KS test
        ks_stat, ks_pvalue = scipy.stats.ks_2samp(real_returns, synthetic_returns)
        
        # Save samples
        np.save(f"{self.results_dir}/{model_name}_returns.npy", samples)
        np.save(f"{self.results_dir}/{model_name}_returns_flattened.npy", samples.flatten())
        
        metrics = {
            'real_stats': real_stats,
            'synthetic_stats': synthetic_stats,
            'ks_stat': ks_stat,
            'ks_pvalue': ks_pvalue,
            'num_samples': num_samples
        }
        
        # Save metrics
        with open(f"{self.results_dir}/{model_name}_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
        
        return metrics
    
    def _evaluate_llm_model(self, model, trainer, probe, conditioning, X, real_returns, model_name, args):
        """Evaluate LLM model with probe."""
        # This would call the LLM evaluation function
        # For now, we'll use the standard evaluation
        return self._evaluate_model(model, trainer, None, conditioning, X, real_returns, model_name, args)
    
    def run_comprehensive_comparison(self, args):
        """Run comprehensive comparison of all three approaches."""
        print("\n" + "="*80)
        print("COMPREHENSIVE DDPM COMPARISON FRAMEWORK")
        print("="*80)
        
        # Run all three evaluations
        zero_metrics = self.run_zero_conditioned_evaluation(args)
        explicit_metrics = self.run_explicit_conditioned_evaluation(args)
        llm_metrics = self.run_llm_conditioned_evaluation(args)
        
        # Generate comparison plots and tables
        self._generate_comparison_plots()
        self._generate_comparison_tables()
        self._generate_statistical_tests()
        
        # Create comprehensive README
        self._create_comprehensive_readme(args)
        
        print(f"\nComprehensive comparison completed successfully!")
        print(f"Results saved in: {self.results_dir}")
        
        return {
            'zero_conditioned': zero_metrics,
            'explicit_conditioned': explicit_metrics,
            'llm_conditioned': llm_metrics
        }
    
    def _generate_comparison_plots(self):
        """Generate comprehensive comparison plots."""
        print("Generating comparison plots...")
        
        # Create subdirectories
        (self.results_dir / "figures").mkdir(exist_ok=True)
        
        # 1. Training Loss Comparison
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 3, 1)
        for model_name, result in self.model_results.items():
            if 'history' in result:
                plt.plot(result['history']['train_losses'], label=f'{model_name.replace("_", " ").title()}')
        plt.xlabel('Epoch')
        plt.ylabel('Training Loss')
        plt.title('Training Loss Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 3, 2)
        for model_name, result in self.model_results.items():
            if 'history' in result:
                plt.plot(result['history']['val_losses'], label=f'{model_name.replace("_", " ").title()}')
        plt.xlabel('Epoch')
        plt.ylabel('Validation Loss')
        plt.title('Validation Loss Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. Distribution Comparison
        plt.subplot(2, 3, 3)
        for model_name, result in self.model_results.items():
            if 'metrics' in result:
                synthetic_stats = result['metrics']['synthetic_stats']
                plt.bar(f'{model_name.replace("_", " ").title()}', synthetic_stats['kurtosis'], 
                       label=f'Kurtosis: {synthetic_stats["kurtosis"]:.2f}')
        plt.ylabel('Excess Kurtosis')
        plt.title('Distributional Properties Comparison')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 3. KS Test Results
        plt.subplot(2, 3, 4)
        ks_stats = []
        model_names = []
        for model_name, result in self.model_results.items():
            if 'metrics' in result:
                ks_stats.append(result['metrics']['ks_stat'])
                model_names.append(model_name.replace("_", " ").title())
        
        plt.bar(model_names, ks_stats)
        plt.ylabel('KS Statistic')
        plt.title('KS Test Results (Lower is Better)')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 4. Statistical Moments Comparison
        plt.subplot(2, 3, 5)
        moments_data = []
        for model_name, result in self.model_results.items():
            if 'metrics' in result:
                synthetic_stats = result['metrics']['synthetic_stats']
                moments_data.append([
                    synthetic_stats['mean'],
                    synthetic_stats['std'],
                    synthetic_stats['skew'],
                    synthetic_stats['kurtosis']
                ])
        
        moments_data = np.array(moments_data)
        moment_names = ['Mean', 'Std', 'Skew', 'Kurtosis']
        
        for i, moment in enumerate(moment_names):
            plt.bar([f'{name}\n{moment}' for name in model_names], moments_data[:, i], 
                   alpha=0.7, label=moment)
        
        plt.ylabel('Value')
        plt.title('Statistical Moments Comparison')
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 5. Model Complexity Comparison
        plt.subplot(2, 3, 6)
        complexity_metrics = []
        for model_name, result in self.model_results.items():
            if 'model' in result:
                model = result['model']
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                complexity_metrics.append([total_params, trainable_params])
        
        complexity_metrics = np.array(complexity_metrics)
        x = np.arange(len(model_names))
        width = 0.35
        
        plt.bar(x - width/2, complexity_metrics[:, 0], width, label='Total Parameters', alpha=0.7)
        plt.bar(x + width/2, complexity_metrics[:, 1], width, label='Trainable Parameters', alpha=0.7)
        
        plt.xlabel('Model')
        plt.ylabel('Number of Parameters')
        plt.title('Model Complexity Comparison')
        plt.xticks(x, model_names, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/figures/comprehensive_comparison.pdf", dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Comparison plots generated successfully")
    
    def _generate_comparison_tables(self):
        """Generate comprehensive comparison tables."""
        print("Generating comparison tables...")
        
        # Create subdirectories
        (self.results_dir / "tables").mkdir(exist_ok=True)
        
        # 1. Basic Statistics Comparison
        with open(f"{self.results_dir}/tables/comprehensive_statistics.tex", 'w') as f:
            f.write("\\begin{table}[h]\n")
            f.write("\\centering\n")
            f.write("\\begin{tabular}{lcccc}\n")
            f.write("\\hline\n")
            f.write("Model & Mean & Std & Skew & Excess Kurtosis \\\\\n")
            f.write("\\hline\n")
            
            for model_name, result in self.model_results.items():
                if 'metrics' in result:
                    synthetic_stats = result['metrics']['synthetic_stats']
                    model_display = model_name.replace("_", " ").title()
                    f.write(f"{model_display} & {synthetic_stats['mean']:.6f} & {synthetic_stats['std']:.6f} & "
                           f"{synthetic_stats['skew']:.6f} & {synthetic_stats['kurtosis']:.6f} \\\\\n")
            
            f.write("\\hline\n")
            f.write("\\end{tabular}\n")
            f.write("\\caption{Comprehensive Statistics Comparison (Kurtosis values are excess kurtosis)}\n")
            f.write("\\label{tab:comprehensive_statistics}\n")
            f.write("\\end{table}\n")
        
        # 2. KS Test Results
        with open(f"{self.results_dir}/tables/ks_test_comparison.tex", 'w') as f:
            f.write("\\begin{table}[h]\n")
            f.write("\\centering\n")
            f.write("\\begin{tabular}{lcc}\n")
            f.write("\\hline\n")
            f.write("Model & KS Statistic & KS p-value \\\\\n")
            f.write("\\hline\n")
            
            for model_name, result in self.model_results.items():
                if 'metrics' in result:
                    model_display = model_name.replace("_", " ").title()
                    f.write(f"{model_display} & {result['metrics']['ks_stat']:.6f} & "
                           f"{result['metrics']['ks_pvalue']:.6f} \\\\\n")
            
            f.write("\\hline\n")
            f.write("\\end{tabular}\n")
            f.write("\\caption{KS Test Results Comparison}\n")
            f.write("\\label{tab:ks_test_comparison}\n")
            f.write("\\end{table}\n")
        
        print("Comparison tables generated successfully")
    
    def _generate_statistical_tests(self):
        """Generate statistical significance tests between approaches."""
        print("Generating statistical tests...")
        
        # Load all synthetic returns
        all_returns = {}
        for model_name, result in self.model_results.items():
            if 'metrics' in result:
                returns_file = f"{self.results_dir}/{model_name}_returns_flattened.npy"
                if os.path.exists(returns_file):
                    all_returns[model_name] = np.load(returns_file)
        
        # Perform pairwise KS tests
        pairwise_tests = {}
        model_names = list(all_returns.keys())
        
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names):
                if i < j:
                    test_name = f"{model1}_vs_{model2}"
                    ks_stat, ks_pvalue = scipy.stats.ks_2samp(
                        all_returns[model1], all_returns[model2]
                    )
                    pairwise_tests[test_name] = {
                        'ks_stat': ks_stat,
                        'ks_pvalue': ks_pvalue
                    }
        
        # Save statistical tests
        with open(f"{self.results_dir}/pairwise_statistical_tests.json", 'w') as f:
            json.dump(pairwise_tests, f, indent=2)
        
        # Create LaTeX table
        with open(f"{self.results_dir}/tables/pairwise_statistical_tests.tex", 'w') as f:
            f.write("\\begin{table}[h]\n")
            f.write("\\centering\n")
            f.write("\\begin{tabular}{lcc}\n")
            f.write("\\hline\n")
            f.write("Model Comparison & KS Statistic & KS p-value \\\\\n")
            f.write("\\hline\n")
            
            for test_name, test_result in pairwise_tests.items():
                display_name = test_name.replace("_", " ").replace("vs", "vs.").title()
                f.write(f"{display_name} & {test_result['ks_stat']:.6f} & "
                       f"{test_result['ks_pvalue']:.6f} \\\\\n")
            
            f.write("\\hline\n")
            f.write("\\end{tabular}\n")
            f.write("\\caption{Pairwise Statistical Tests Between Models}\n")
            f.write("\\label{tab:pairwise_statistical_tests}\n")
            f.write("\\end{table}\n")
        
        print("Statistical tests generated successfully")
    
    def _create_comprehensive_readme(self, args):
        """Create comprehensive README for the comparison."""
        print("Creating comprehensive README...")
        
        readme_content = f"""# Comprehensive DDPM Comparison Framework

## Run Information
- **Timestamp**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Results Directory**: {self.results_dir}
- **Seed**: {self.seed}

## Overview
This framework provides a comprehensive comparison of three DDPM conditioning approaches:

1. **Zero-Conditioned**: Standard DDPM without any conditioning
2. **Explicitly-Conditioned**: DDPM with explicit regime and volatility conditioning
3. **LLM-Conditioned**: DDPM with news-based LLM embeddings as conditioning

## Model Configurations
- **Sequence Length**: {args.seq_len}
- **Hidden Dimension**: {args.hidden_dim}
- **Number of Timesteps**: {args.num_timesteps}
- **Beta Schedule**: {args.beta_schedule}
- **Sampler**: {args.sampler}
- **Sample Steps**: {args.sample_steps}
- **CFG Scale**: {args.cfg_scale}

## Generated Files

### Figures
- `figures/comprehensive_comparison.pdf` - Comprehensive comparison plots

### Tables
- `tables/comprehensive_statistics.tex` - Basic statistics comparison
- `tables/ks_test_comparison.tex` - KS test results comparison
- `tables/pairwise_statistical_tests.tex` - Pairwise statistical tests

### Data
- `*_returns.npy` - Generated return sequences for each model
- `*_returns_flattened.npy` - Flattened return sequences
- `*_metrics.json` - Evaluation metrics for each model
- `pairwise_statistical_tests.json` - Statistical test results

## Key Findings
The comparison reveals:
1. **Conditioning Impact**: How different conditioning approaches affect generation quality
2. **Statistical Properties**: Distributional characteristics of each approach
3. **Training Dynamics**: Learning curves and convergence patterns
4. **Model Complexity**: Parameter counts and computational requirements

## Usage
Run the framework with:
```python
from comprehensive_comparison_framework import ComprehensiveComparisonFramework

framework = ComprehensiveComparisonFramework()
results = framework.run_comprehensive_comparison(args)
```

## Notes
- All models use identical training configurations for fair comparison
- Statistical tests assess significance of differences between approaches
- Results are saved in structured format for thesis inclusion
"""
        
        with open(f"{self.results_dir}/README_COMPREHENSIVE.md", 'w') as f:
            f.write(readme_content)
        
        print("Comprehensive README created successfully")

def parse_comparison_args():
    """Parse arguments for the comparison framework."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Comprehensive DDPM Comparison Framework')
    
    # Data parameters
    parser.add_argument('--seq-len', type=int, default=60, help='Sequence length for training')
    parser.add_argument('--vol-window', type=int, default=20, help='Volatility rolling window')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Training batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--num-timesteps', type=int, default=1000, help='Number of diffusion timesteps')
    parser.add_argument('--beta-schedule', choices=['cosine', 'linear'], default='cosine', help='Beta schedule')
    parser.add_argument('--sampler', choices=['ddpm', 'ddim'], default='ddim', help='Sampling method')
    parser.add_argument('--sample-steps', type=int, default=50, help='Number of sampling steps')
    parser.add_argument('--val-split', type=float, default=0.2, help='Validation split ratio')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--grad-clip', type=float, default=1.0, help='Gradient clipping value')
    
    # Classifier-free guidance parameters
    parser.add_argument('--cfg-scale', type=float, default=7.5, help='Classifier-free guidance scale')
    
    # Model parameters
    parser.add_argument('--hidden-dim', type=int, default=128, help='Hidden dimension')
    
    # Output parameters
    parser.add_argument('--results-dir', type=str, default="results/comprehensive_comparison", help='Results directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto', help='Device to use')
    
    args = parser.parse_args()
    
    # Safety check
    if args.vol_window > args.seq_len:
        raise ValueError(f"vol_window ({args.vol_window}) cannot be greater than seq_len ({args.seq_len})")
    
    # Set device
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    return args

def main():
    """Main function to run the comprehensive comparison."""
    print("Comprehensive DDPM Comparison Framework")
    print("=" * 80)
    
    # Parse arguments
    args = parse_comparison_args()
    
    # Initialize framework
    framework = ComprehensiveComparisonFramework(
        results_dir=args.results_dir,
        seed=args.seed
    )
    
    # Run comprehensive comparison
    results = framework.run_comprehensive_comparison(args)
    
    print(f"\nComprehensive comparison completed successfully!")
    print(f"Results saved in: {args.results_dir}")
    
    return results

if __name__ == "__main__":
    main()
