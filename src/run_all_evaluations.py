#!/usr/bin/env python3
"""
Master script to run all model evaluations automatically

This script runs all three models (GARCH, DDPM, TimeGrad) and their evaluations,
then performs comprehensive cross-model comparison.

Author: Simin Ali
Thesis: Diffusion Models in Generative AI for Financial Data Synthesis and Risk Management
"""

import os
import sys
import subprocess
import time
import warnings
warnings.filterwarnings('ignore')

def run_script(script_name, description):
    """Run a Python script and handle any errors."""
    print(f"\n{'='*60}")
    print(f"🚀 Running {description}")
    print(f"{'='*60}")
    
    try:
        # Run the script
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, cwd='src')
        
        if result.returncode == 0:
            print(f"✅ {description} completed successfully!")
            if result.stdout:
                print("Output:")
                print(result.stdout)
        else:
            print(f"❌ {description} failed!")
            print("Error:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Error running {description}: {e}")
        return False
    
    return True

def check_data_file():
    """Check if the required data file exists."""
    data_file = "data/sp500_data.csv"
    if not os.path.exists(data_file):
        print(f"⚠️  Data file {data_file} not found!")
        print("Please ensure your data is in the data/ folder")
        return False
    
    print(f"✅ Found data file: {data_file}")
    return True

def create_results_directories():
    """Create necessary results directories."""
    directories = [
        "results",
        "results/garch_evaluation",
        "results/ddpm_evaluation", 
        "results/timegrad_evaluation",
        "results/comprehensive_evaluation",
        "results/garch_evaluation/plots",
        "results/garch_evaluation/tables",
        "results/ddpm_evaluation/plots",
        "results/ddpm_evaluation/tables",
        "results/timegrad_evaluation/plots",
        "results/timegrad_evaluation/tables",
        "results/comprehensive_evaluation/plots",
        "results/comprehensive_evaluation/tables"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("✅ Results directories created")

def main():
    """Main function to run all evaluations."""
    
    print("🎓 MSc Thesis: Diffusion Models Evaluation")
    print("Author: Simin Ali")
    print("Supervisor: Dr Mikael Mieskolainen")
    print("Institution: Imperial College London")
    print("\n" + "="*60)
    
    # Check and setup data
    if not check_data_file():
        return
    
    # Create results directories
    create_results_directories()
    
    # List of scripts to run with descriptions
    scripts = [
        ("garch_simple.py", "GARCH(1,1) Model and Evaluation"),
        ("diffusion_simple.py", "DDPM Model and Evaluation"),
        ("timegrad_simple.py", "TimeGrad Model and Evaluation")
    ]
    
    # Track success
    successful_runs = []
    failed_runs = []
    
    # Run each model
    for script_name, description in scripts:
        success = run_script(script_name, description)
        if success:
            successful_runs.append(description)
        else:
            failed_runs.append(description)
        
        # Small delay between runs
        time.sleep(2)
    
    # Run comprehensive evaluation if at least one model succeeded
    if successful_runs:
        print(f"\n{'='*60}")
        print("🔍 Running Comprehensive Cross-Model Evaluation")
        print(f"{'='*60}")
        
        try:
            # Import and run comprehensive evaluation
            sys.path.append('src')
            from evaluation_framework import FinancialModelEvaluator
            import numpy as np
            import pandas as pd
            
            # Load data from data folder
            data = pd.read_csv("data/sp500_data.csv", index_col=0, parse_dates=True)
            data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
            returns = np.log(data['Close'] / data['Close'].shift(1)).dropna() * 100
            real_data = returns.values
            
            # Initialize evaluator
            evaluator = FinancialModelEvaluator(model_names=['GARCH', 'DDPM', 'TimeGrad'])
            
            # Load synthetic data from each model
            synthetic_data_dict = {}
            
            # Try to load GARCH results
            try:
                garch_returns = np.load('results/garch_returns.npy', allow_pickle=True)
                synthetic_data_dict['GARCH'] = garch_returns
                print("✅ Loaded GARCH results")
            except:
                print("⚠️  GARCH results not found")
            
            # Try to load DDPM results
            try:
                ddpm_returns = np.load('results/ddpm_returns.npy', allow_pickle=True)
                synthetic_data_dict['DDPM'] = ddpm_returns
                print("✅ Loaded DDPM results")
            except:
                print("⚠️  DDPM results not found")
            
            # Try to load TimeGrad results
            try:
                timegrad_returns = np.load('results/timegrad_returns.npy', allow_pickle=True)
                synthetic_data_dict['TimeGrad'] = timegrad_returns
                print("✅ Loaded TimeGrad results")
            except:
                print("⚠️  TimeGrad results not found")
            
            # Run comprehensive evaluation
            if synthetic_data_dict:
                results = evaluator.run_comprehensive_evaluation(
                    real_data=real_data,
                    synthetic_data_dict=synthetic_data_dict,
                    save_path="results/comprehensive_evaluation/"
                )
                print("✅ Comprehensive evaluation completed!")
                successful_runs.append("Comprehensive Cross-Model Evaluation")
            else:
                print("❌ No model results found for comprehensive evaluation")
                failed_runs.append("Comprehensive Cross-Model Evaluation")
                
        except Exception as e:
            print(f"❌ Error in comprehensive evaluation: {e}")
            failed_runs.append("Comprehensive Cross-Model Evaluation")
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 EVALUATION SUMMARY")
    print(f"{'='*60}")
    
    if successful_runs:
        print("✅ Successfully completed:")
        for run in successful_runs:
            print(f"   - {run}")
    
    if failed_runs:
        print("\n❌ Failed to complete:")
        for run in failed_runs:
            print(f"   - {run}")
    
    print(f"\n📁 Results saved in:")
    print("   - results/garch_evaluation/")
    print("   - results/ddpm_evaluation/")
    print("   - results/timegrad_evaluation/")
    print("   - results/comprehensive_evaluation/")
    
    print(f"\n📝 Next steps:")
    print("   1. Check the results/ directory for outputs")
    print("   2. Use LaTeX tables in your thesis")
    print("   3. Include PDF plots in your thesis")
    print("   4. Reference JSON results for reproducibility")
    
    if successful_runs:
        print(f"\n🎉 Evaluation completed successfully!")
    else:
        print(f"\n⚠️  No evaluations completed successfully. Please check the errors above.")

if __name__ == "__main__":
    main()
