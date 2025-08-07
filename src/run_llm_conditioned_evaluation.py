#!/usr/bin/env python3
"""
Run LLM-Conditioned Diffusion Model Evaluation
Integrates the new conditioned model into the existing evaluation framework

Author: Simin Ali
Thesis: Diffusion Models in Generative AI for Financial Data Synthesis and Risk Management
"""

import os
import sys
import subprocess
import time
from llm_conditioned_diffusion import main as run_llm_conditioned

def check_dependencies():
    """Check if required dependencies are installed."""
    print("🔍 Checking dependencies for LLM-conditioned diffusion...")
    
    try:
        import transformers
        print("✅ transformers installed")
    except ImportError:
        print("❌ transformers not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers"])
    
    try:
        import requests
        print("✅ requests installed")
    except ImportError:
        print("❌ requests not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "requests"])

def run_llm_conditioned_evaluation():
    """Run the LLM-conditioned diffusion model evaluation."""
    print("\n" + "="*80)
    print("🚀 Running LLM-Conditioned Diffusion Model Evaluation")
    print("Based on supervisor feedback: Using LLM embeddings as conditioning vectors")
    print("="*80)
    
    try:
        # Run the LLM-conditioned model
        model, trainer, synthetic_data, stats = run_llm_conditioned()
        
        print("\n✅ LLM-conditioned diffusion evaluation completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error in LLM-conditioned evaluation: {str(e)}")
        return False

def integrate_with_comprehensive_evaluation():
    """Integrate LLM-conditioned results with comprehensive evaluation."""
    print("\n🔄 Integrating LLM-conditioned results with comprehensive evaluation...")
    
    try:
        # Import the evaluation framework
        sys.path.append('src')
        from evaluation_framework import FinancialModelEvaluator
        
        # Load existing results
        import json
        with open("results/comprehensive_evaluation/evaluation_results.json", 'r') as f:
            existing_results = json.load(f)
        
        # Load LLM-conditioned results
        with open("results/llm_conditioned_evaluation/llm_conditioned_stats.json", 'r') as f:
            llm_results = json.load(f)
        
        # Add LLM-conditioned results to comprehensive evaluation
        existing_results['basic_stats'].append(llm_results)
        
        # Save updated results
        with open("results/comprehensive_evaluation/evaluation_results.json", 'w') as f:
            json.dump(existing_results, f, indent=2)
        
        print("✅ LLM-conditioned results integrated with comprehensive evaluation")
        return True
        
    except Exception as e:
        print(f"❌ Error integrating results: {str(e)}")
        return False

def main():
    """Main function to run LLM-conditioned evaluation."""
    print("🎓 MSc Thesis: LLM-Conditioned Diffusion Model Evaluation")
    print("Author: Simin Ali")
    print("Supervisor: Dr Mikael Mieskolainen")
    print("Institution: Imperial College London")
    print("\n" + "="*60)
    
    # Check dependencies
    check_dependencies()
    
    # Run LLM-conditioned evaluation
    success = run_llm_conditioned_evaluation()
    
    if success:
        # Integrate with comprehensive evaluation
        integrate_with_comprehensive_evaluation()
        
        print("\n📊 EVALUATION SUMMARY")
        print("="*60)
        print("✅ Successfully completed:")
        print("   - LLM-Conditioned Diffusion Model Training")
        print("   - Synthetic Data Generation with LLM Embeddings")
        print("   - Performance Evaluation")
        print("   - Integration with Comprehensive Evaluation")
        
        print("\n📁 Results saved in:")
        print("   - results/llm_conditioned_evaluation/")
        print("   - results/comprehensive_evaluation/ (updated)")
        
        print("\n🎯 Key Innovation:")
        print("   - LLM embeddings from internet data as conditioning vectors")
        print("   - Conditional generation based on market sentiment")
        print("   - Practical applications for different financial institutions")
        
        print("\n📝 Next steps:")
        print("   1. Review the LLM-conditioned results")
        print("   2. Compare with baseline models (GARCH, DDPM, TimeGrad)")
        print("   3. Analyze the impact of conditioning on generation quality")
        print("   4. Document findings for thesis Results chapter")
        
    else:
        print("\n❌ LLM-conditioned evaluation failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
