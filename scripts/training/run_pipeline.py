#!/usr/bin/env python3
"""
Comprehensive Pipeline for Financial Model Training and Evaluation

This script runs all four training scripts in sequence:
1. GARCH model training/generation
2. DDPM model training/generation  
3. TimeGrad model training/generation
4. LLM-conditioned diffusion model training/generation
5. Baseline evaluation script comparing real data with all baseline models

Each run saves results to a timestamped folder in ./runs/ and ensures all metrics
and plots use only the test set for fair comparison.
"""

import os
import sys
import subprocess
import time
from datetime import datetime
import json
import pandas as pd
import numpy as np

def print_header(title):
    """Print a formatted header."""
    print("\n" + "="*80)
    print(f" {title} ".center(80, "="))
    print("="*80)

def print_step(step_num, total_steps, description):
    """Print a formatted step description."""
    print(f"\n[Step {step_num}/{total_steps}] {description}")
    print("-" * 60)

def run_script(script_path, args, step_name):
    """Run a Python script with given arguments."""
    print(f"Running {step_name}...")
    
    # Construct command
    cmd = [sys.executable, script_path] + args
    print(f"Command: {' '.join(cmd)}")
    
    # Run script
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        elapsed_time = time.time() - start_time
        
        print(f"✅ {step_name} completed successfully in {elapsed_time:.2f} seconds")
        if result.stdout:
            print("Output:")
            print(result.stdout[-1000:])  # Last 1000 characters
        
        return True, elapsed_time
        
    except subprocess.CalledProcessError as e:
        elapsed_time = time.time() - start_time
        print(f"❌ {step_name} failed after {elapsed_time:.2f} seconds")
        print(f"Exit code: {e.returncode}")
        if e.stdout:
            print("Stdout:")
            print(e.stdout[-1000:])
        if e.stderr:
            print("Stderr:")
            print(e.stderr[-1000:])
        return False, elapsed_time

def create_pipeline_summary(run_results, pipeline_dir):
    """Create a comprehensive summary of the pipeline run."""
    print("\nCreating pipeline summary...")
    
    summary = {
        'pipeline_timestamp': datetime.now().isoformat(),
        'total_duration': sum(result['duration'] for result in run_results.values()),
        'steps_completed': len([r for r in run_results.values() if r['success']]),
        'total_steps': len(run_results),
        'run_results': run_results
    }
    
    # Save summary
    summary_path = os.path.join(pipeline_dir, "pipeline_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    # Create markdown report
    report_path = os.path.join(pipeline_dir, "pipeline_report.md")
    with open(report_path, 'w') as f:
        f.write("# Financial Models Pipeline Report\n\n")
        f.write(f"**Pipeline Run**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Summary\n\n")
        f.write(f"- **Total Duration**: {summary['total_duration']:.2f} seconds\n")
        f.write(f"- **Steps Completed**: {summary['steps_completed']}/{summary['total_steps']}\n")
        f.write(f"- **Success Rate**: {summary['steps_completed']/summary['total_steps']*100:.1f}%\n\n")
        
        f.write("## Step Details\n\n")
        for step_name, result in run_results.items():
            status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
            f.write(f"### {step_name}\n")
            f.write(f"- **Status**: {status}\n")
            f.write(f"- **Duration**: {result['duration']:.2f} seconds\n")
            if result['output_dir']:
                f.write(f"- **Output Directory**: {result['output_dir']}\n")
            f.write("\n")
        
        f.write("## Next Steps\n\n")
        f.write("1. Review individual model results in their respective output directories\n")
        f.write("2. Check the baseline evaluation results for model comparisons\n")
        f.write("3. Analyze the comprehensive evaluation metrics and plots\n")
        f.write("4. Review the report notes in each output directory for interpretation guidance\n")
    
    print(f"Pipeline summary saved to: {summary_path}")
    print(f"Pipeline report saved to: {report_path}")

def main():
    """Main pipeline function."""
    print_header("Financial Models Training and Evaluation Pipeline")
    
    # Create pipeline output directory
    pipeline_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pipeline_dir = f"./runs/pipeline_{pipeline_timestamp}"
    os.makedirs(pipeline_dir, exist_ok=True)
    
    print(f"Pipeline output directory: {pipeline_dir}")
    
    # Define pipeline steps
    pipeline_steps = [
        {
            'name': 'GARCH Model',
            'script': 'src/garch_simple.py',
            'args': ['--units', 'percent', '--write-metadata', '--outdir', f'./runs/garch_run_{pipeline_timestamp}'],
            'output_dir': f'./runs/garch_run_{pipeline_timestamp}'
        },
        {
            'name': 'DDPM Model', 
            'script': 'src/diffusion_simple.py',
            'args': ['--units', 'percent', '--write-metadata', '--outdir', f'./runs/ddpm_run_{pipeline_timestamp}'],
            'output_dir': f'./runs/ddpm_run_{pipeline_timestamp}'
        },
        {
            'name': 'TimeGrad Model',
            'script': 'src/timegrad_simple.py', 
            'args': ['--units', 'percent', '--write-metadata', '--outdir', f'./runs/timegrad_run_{pipeline_timestamp}'],
            'output_dir': f'./runs/timegrad_run_{pipeline_timestamp}'
        },
        {
            'name': 'LLM-Conditioned Diffusion Model',
            'script': 'src/llm_conditioned_diffusion.py',
            'args': ['--units', 'percent', '--write-metadata', '--outdir', f'./runs/llm_cond_diffusion_run_{pipeline_timestamp}'],
            'output_dir': f'./runs/llm_cond_diffusion_run_{pipeline_timestamp}'
        },
        {
            'name': 'Baseline Evaluation',
            'script': 'src/complete_llm_evaluation.py',
            'args': ['--outdir', f'./runs/evaluation_baselines_{pipeline_timestamp}'],
            'output_dir': f'./runs/evaluation_baselines_{pipeline_timestamp}'
        }
    ]
    
    # Store results
    run_results = {}
    
    # Run each step
    for i, step in enumerate(pipeline_steps, 1):
        print_step(i, len(pipeline_steps), step['name'])
        
        # Check if script exists
        if not os.path.exists(step['script']):
            print(f"❌ Script not found: {step['script']}")
            run_results[step['name']] = {
                'success': False,
                'duration': 0,
                'output_dir': None,
                'error': 'Script not found'
            }
            continue
        
        # Run script
        success, duration = run_script(step['script'], step['args'], step['name'])
        
        # Store results
        run_results[step['name']] = {
            'success': success,
            'duration': duration,
            'output_dir': step['output_dir'] if success else None,
            'error': None if success else 'Script execution failed'
        }
        
        # Wait a moment between steps
        if i < len(pipeline_steps):
            print("Waiting 5 seconds before next step...")
            time.sleep(5)
    
    # Create pipeline summary
    print_header("Pipeline Summary")
    create_pipeline_summary(run_results, pipeline_dir)
    
    # Print final summary
    successful_steps = [name for name, result in run_results.items() if result['success']]
    failed_steps = [name for name, result in run_results.items() if not result['success']]
    
    print(f"\n🎯 Pipeline completed!")
    print(f"✅ Successful steps ({len(successful_steps)}/{len(pipeline_steps)}):")
    for step in successful_steps:
        print(f"   - {step}")
    
    if failed_steps:
        print(f"❌ Failed steps ({len(failed_steps)}/{len(pipeline_steps)}):")
        for step in failed_steps:
            print(f"   - {step}")
    
    total_duration = sum(result['duration'] for result in run_results.values())
    print(f"\n⏱️  Total pipeline duration: {total_duration:.2f} seconds ({total_duration/60:.1f} minutes)")
    
    if successful_steps:
        print(f"\n📁 Results saved to:")
        for step in successful_steps:
            output_dir = run_results[step]['output_dir']
            if output_dir:
                print(f"   - {step}: {output_dir}")
    
    print(f"\n📊 Pipeline summary: {pipeline_dir}/pipeline_summary.json")
    print(f"📋 Pipeline report: {pipeline_dir}/pipeline_report.md")
    
    if failed_steps:
        print(f"\n⚠️  Some steps failed. Check the error messages above and individual script outputs.")
        return 1
    else:
        print(f"\n🎉 All steps completed successfully!")
        return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

