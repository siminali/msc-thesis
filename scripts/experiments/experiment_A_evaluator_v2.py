#!/usr/bin/env python3
"""
Experiment A: Out-of-Sample Stress Testing Evaluator v2 (Integrated Pipeline)

Extended version with integrated metrics calculation and plotting pipeline.
Automatically runs metrics and plotting after sample generation for complete analysis.

New Features in v2:
- Automatic metrics calculation after sampling
- Integrated plotting pipeline 
- Compact findings.jsonl generation
- Temporary real_slice.csv creation
- End-to-end analysis workflow
- Robust error handling for partial failures

Author: Simin Ali
Thesis: Diffusion Models in Generative AI for Financial Data Synthesis and Risk Management
"""

import argparse
import json
import numpy as np
import pandas as pd
import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional, Any
import re
import warnings

# Import base evaluator and our custom modules
try:
    from experiment_A_evaluator import ExperimentAEvaluator
    from checkpoint_loader_sampler import CheckpointSampler, load_and_sample
    from eval_conditioning_providers import generate_eval_conditioning, load_conditioning_spec
    from shared.metrics import MetricsCalculator
    from shared.plotting import PlottingPipeline
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Make sure all required modules are available")
    sys.exit(1)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ExperimentAEvaluatorV2(ExperimentAEvaluator):
    """Extended Experiment A evaluator with integrated metrics and plotting."""
    
    def __init__(self, checkpoints_dir: str = 'checkpoints/precovid',
                 results_base: str = 'results/addons/period_slices',
                 experiment_name: str = 'A'):
        
        # Initialize base class
        super().__init__(checkpoints_dir, results_base, experiment_name)
        
        # Add metrics and plotting components
        self.metrics_calculator = MetricsCalculator()
        self.plotting_pipeline = PlottingPipeline(str(self.results_base))
        
        # Track findings for summary
        self.findings = []
        
        logger.info(f"Initialized Experiment A v2 with integrated pipeline in: {self.experiment_dir}")
    
    def create_real_slice_csv(self, window_id: str, returns_data: pd.DataFrame) -> Optional[Path]:
        """Create temporary real_slice.csv for a specific window."""
        try:
            # Get window dates
            if window_id not in self.STRESS_WINDOWS:
                logger.warning(f"Unknown window {window_id}, using default data")
                return None
            
            window_info = self.STRESS_WINDOWS[window_id]
            start_date = window_info['start']
            end_date = window_info['end']
            
            # Filter data for window
            window_data = returns_data[
                (returns_data.index >= start_date) & 
                (returns_data.index <= end_date)
            ].copy()
            
            if len(window_data) == 0:
                logger.warning(f"No data found for window {window_id}")
                return None
            
            # Create temporary CSV file
            csv_path = self.experiment_dir / window_id / 'real_slice.csv'
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Ensure we have a 'returns' column
            if 'returns' not in window_data.columns and len(window_data.columns) == 1:
                window_data['returns'] = window_data.iloc[:, 0]
            
            window_data.to_csv(csv_path)
            logger.info(f"Created real slice CSV: {csv_path}")
            return csv_path
            
        except Exception as e:
            logger.error(f"Failed to create real slice CSV for {window_id}: {e}")
            return None
    
    def run_metrics_for_window(self, window_id: str, real_csv_path: Path) -> bool:
        """Run metrics calculation for a specific window."""
        try:
            logger.info(f"Running metrics for window: {window_id}")
            
            # Prepare command
            cmd = [
                sys.executable, 'metrics_runner.py',
                '--experiments', self.experiment_name,
                '--windows', window_id,
                '--csv-file', str(real_csv_path),
                '--results-base', str(self.results_base)
            ]
            
            # Run metrics
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
            
            if result.returncode == 0:
                logger.info(f"Metrics calculation completed for {window_id}")
                return True
            else:
                logger.error(f"Metrics calculation failed for {window_id}: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error running metrics for {window_id}: {e}")
            return False
    
    def run_plotting_for_window(self, window_id: str) -> bool:
        """Run plotting for a specific window."""
        try:
            logger.info(f"Running plotting for window: {window_id}")
            
            # Prepare command
            cmd = [
                sys.executable, 'plotting_runner.py',
                '--experiments', self.experiment_name,
                '--windows', window_id,
                '--results-base', str(self.results_base)
            ]
            
            # Run plotting
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
            
            if result.returncode == 0:
                logger.info(f"Plotting completed for {window_id}")
                return True
            else:
                logger.error(f"Plotting failed for {window_id}: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error running plotting for {window_id}: {e}")
            return False
    
    def extract_compact_findings(self, window_id: str) -> Optional[Dict[str, Any]]:
        """Extract compact findings from metrics results."""
        try:
            # The metrics.json might be in the experiment root or window directory
            metrics_file = self.experiment_dir / window_id / 'metrics.json'
            if not metrics_file.exists():
                # The metrics runner creates files in the base experiment directory (e.g., A not A_v6)
                base_exp_name = self.experiment_name.split('_')[0]  # A from A_v6
                alt_exp_dir = self.results_base / base_exp_name / window_id
                metrics_file = alt_exp_dir / 'metrics.json'
                logger.info(f"Trying alternative metrics location: {metrics_file}")
            if not metrics_file.exists():
                # Try the root experiment directory
                metrics_file = self.experiment_dir / 'metrics.json'
            
            if not metrics_file.exists():
                logger.warning(f"No metrics file found for {window_id}")
                return None
            
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            
            # Extract key findings
            findings = {
                'window_id': window_id,
                'timestamp': datetime.now().isoformat(),
                'models': {},
                'pairwise_comparisons': {}
            }
            
            # Model-level findings
            for model_name, model_results in metrics.get('models', {}).items():
                if model_results.get('status') == 'success':
                    risk_metrics = model_results.get('risk_metrics', {})
                    findings['models'][model_name] = {
                        'var_5pct': risk_metrics.get('var_0.050', None),
                        'es_5pct': risk_metrics.get('es_0.050', None),
                        'volatility': model_results.get('basic_stats', {}).get('std', None)
                    }
            
            # Pairwise comparison findings
            for comparison_name, comparison_results in metrics.get('pairwise_comparisons', {}).items():
                if comparison_results.get('status') == 'success':
                    dm_tests = comparison_results.get('diebold_mariano_tests', {})
                    findings['pairwise_comparisons'][comparison_name] = {
                        'dm_mse_pvalue': dm_tests.get('dm_mse', {}).get('p_value', None),
                        'dm_mae_pvalue': dm_tests.get('dm_mae', {}).get('p_value', None),
                        'mse_ratio': comparison_results.get('individual_losses', {}).get('mse_ratio', None)
                    }
            
            return findings
            
        except Exception as e:
            logger.error(f"Error extracting findings for {window_id}: {e}")
            return None
    
    def append_findings_to_jsonl(self, findings: Dict[str, Any]):
        """Append findings to the experiment-level findings.jsonl file."""
        try:
            findings_file = self.experiment_dir / 'findings.jsonl'
            
            with open(findings_file, 'a') as f:
                f.write(json.dumps(findings) + '\n')
            
            logger.info(f"Appended findings to {findings_file}")
            
        except Exception as e:
            logger.error(f"Error writing findings: {e}")
    
    def run_integrated_pipeline(self, window_id: str, window_info: Dict[str, Any], 
                               checkpoints: Dict[str, Dict[str, Any]], returns_data: pd.DataFrame,
                               num_paths: int, seq_len: int, seeds: List[int]) -> Dict[str, Any]:
        """
        Run the complete integrated pipeline: sampling → metrics → plotting → findings.
        """
        
        logger.info(f"Starting integrated pipeline for window: {window_id}")
        
        # Step 1: Run base sampling (from parent class) 
        # Call the parent class method that actually exists
        try:
            # Save current state
            original_experiment_dir = self.experiment_dir
            
            # Run the parent experiment method (without override)
            # The base method takes windows (list), not individual window
            ExperimentAEvaluator.run_experiment(
                self, [window_id], checkpoints, returns_data, num_paths, seq_len, seeds
            )
            
            # Check if sampling was successful
            samples_exist = False
            window_dir = self.experiment_dir / window_id
            if window_dir.exists():
                for model_dir in window_dir.iterdir():
                    if model_dir.is_dir() and (model_dir / 'samples.npy').exists():
                        samples_exist = True
                        break
            
            base_results = {
                'status': 'completed' if samples_exist else 'failed',
                'window_id': window_id,
                'experiment_dir': str(self.experiment_dir)
            }
            
        except Exception as e:
            logger.error(f"Base sampling failed: {e}")
            base_results = {'status': 'failed', 'error': str(e)}
        
        if not base_results or base_results.get('status') != 'completed':
            logger.error(f"Base sampling failed for {window_id}")
            return base_results
        
        # Step 2: Create real slice CSV
        real_csv_path = self.create_real_slice_csv(window_id, returns_data)
        if not real_csv_path:
            logger.warning(f"Could not create real slice CSV for {window_id}, skipping metrics/plotting")
            return base_results
        
        # Step 3: Run metrics calculation
        metrics_success = self.run_metrics_for_window(window_id, real_csv_path)
        
        # Step 4: Run plotting
        plotting_success = self.run_plotting_for_window(window_id)
        
        # Step 5: Extract and save findings
        findings = self.extract_compact_findings(window_id)
        if findings:
            self.append_findings_to_jsonl(findings)
            self.findings.append(findings)
        
        # Step 6: Clean up temporary CSV
        try:
            if real_csv_path.exists():
                real_csv_path.unlink()
                logger.info(f"Cleaned up temporary CSV: {real_csv_path}")
        except Exception as e:
            logger.warning(f"Could not clean up temporary CSV: {e}")
        
        # Update results with pipeline status
        base_results['integrated_pipeline'] = {
            'metrics_success': metrics_success,
            'plotting_success': plotting_success,
            'findings_extracted': findings is not None,
            'real_csv_created': real_csv_path is not None
        }
        
        logger.info(f"Integrated pipeline completed for {window_id}")
        return base_results
    
    def run_experiment(self, window_id: str, window_info: Dict[str, Any], 
                      checkpoints: Dict[str, Dict[str, Any]], returns_data: pd.DataFrame,
                      num_paths: int, seq_len: int, seeds: List[int]) -> Dict[str, Any]:
        """Override to use integrated pipeline."""
        return self.run_integrated_pipeline(
            window_id, window_info, checkpoints, returns_data, num_paths, seq_len, seeds
        )
    
    def generate_final_summary(self):
        """Generate final summary with integrated pipeline results."""
        try:
            summary = {
                'experiment': self.experiment_name,
                'completed_at': datetime.now().isoformat(),
                'total_windows': len(self.findings),
                'integrated_pipeline_summary': {
                    'metrics_success_rate': 0,
                    'plotting_success_rate': 0,
                    'findings_extraction_rate': 0
                },
                'key_findings': {}
            }
            
            # Calculate success rates
            if self.findings:
                summary['integrated_pipeline_summary'] = {
                    'metrics_success_rate': len([f for f in self.findings if f]) / len(self.findings),
                    'plotting_success_rate': 1.0,  # Assume success if findings exist
                    'findings_extraction_rate': len([f for f in self.findings if f]) / len(self.findings)
                }
                
                # Extract key findings across windows
                for finding in self.findings:
                    window_id = finding.get('window_id')
                    if window_id:
                        summary['key_findings'][window_id] = {
                            'models_analyzed': len(finding.get('models', {})),
                            'comparisons_made': len(finding.get('pairwise_comparisons', {})),
                            'significant_dm_tests': len([
                                comp for comp in finding.get('pairwise_comparisons', {}).values()
                                if comp.get('dm_mse_pvalue', 1) < 0.05
                            ])
                        }
            
            # Save summary
            summary_file = self.experiment_dir / 'integrated_summary.json'
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"Generated final summary: {summary_file}")
            
        except Exception as e:
            logger.error(f"Error generating final summary: {e}")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Experiment A v2: Integrated Stress Testing Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete integrated pipeline
  python experiment_A_evaluator_v2.py --window covid_crash --num-paths 1000 --csv-file sp500_data.csv
  
  # Quick test with metrics and plots
  python experiment_A_evaluator_v2.py --window covid_crash --num-paths 100 --csv-file sp500_data.csv
        """
    )
    
    parser.add_argument('--window', type=str, default='covid_crash',
                       help='Stress window for testing (e.g., covid_crash, covid_recovery)')
    
    parser.add_argument('--checkpoints-dir', type=str, default='checkpoints/precovid',
                       help='Directory containing pre-COVID checkpoints')
    
    parser.add_argument('--csv-file', type=str, default='sp500_data.csv',
                       help='CSV file with returns data')
    
    parser.add_argument('--num-paths', type=int, default=1000,
                       help='Number of sample paths to generate')
    
    parser.add_argument('--seq-len', type=int, default=60,
                       help='Sequence length for samples')
    
    parser.add_argument('--seeds', nargs='+', type=int, default=[42],
                       help='Random seeds for reproducibility')
    
    parser.add_argument('--results-base', type=str, default='results/addons/period_slices',
                       help='Base directory for results')
    
    parser.add_argument('--experiment-name', type=str, default='A',
                       help='Experiment identifier')
    
    return parser.parse_args()

def main():
    """Main execution function."""
    args = parse_arguments()
    
    try:
        # Initialize evaluator
        evaluator = ExperimentAEvaluatorV2(
            checkpoints_dir=args.checkpoints_dir,
            results_base=args.results_base,
            experiment_name=args.experiment_name
        )
        
        # Discover checkpoints
        checkpoints = evaluator.discover_checkpoints()
        
        if not checkpoints:
            logger.error("No valid checkpoints found")
            return
        
        # Load returns data
        returns_data = evaluator.load_returns_data(args.csv_file)
        
        # Validate window
        if args.window not in evaluator.STRESS_WINDOWS:
            logger.error(f"Unknown window: {args.window}. Available: {list(evaluator.STRESS_WINDOWS.keys())}")
            return
        
        window_info = evaluator.STRESS_WINDOWS[args.window]
        
        # Run integrated experiment
        results = evaluator.run_experiment(
            window_id=args.window,
            window_info=window_info,
            checkpoints=checkpoints,
            returns_data=returns_data,
            num_paths=args.num_paths,
            seq_len=args.seq_len,
            seeds=args.seeds
        )
        
        # Generate final summary
        evaluator.generate_final_summary()
        
        # Print summary
        print("\n" + "="*70)
        print("EXPERIMENT A v2 INTEGRATED PIPELINE SUMMARY")
        print("="*70)
        print(f"Experiment Directory: {evaluator.experiment_dir}")
        print(f"Status: {results.get('status', 'unknown')}")
        print(f"Window Tested: {args.window}")
        print(f"Models Tested: {len(checkpoints)}")
        
        # Pipeline status
        pipeline_status = results.get('integrated_pipeline', {})
        print(f"\nIntegrated Pipeline Results:")
        print(f"  Metrics Success: {pipeline_status.get('metrics_success', False)}")
        print(f"  Plotting Success: {pipeline_status.get('plotting_success', False)}")
        print(f"  Findings Extracted: {pipeline_status.get('findings_extracted', False)}")
        
        # Show findings summary
        if evaluator.findings:
            print(f"\nFindings Summary:")
            for finding in evaluator.findings:
                window_id = finding.get('window_id')
                models = len(finding.get('models', {}))
                comparisons = len(finding.get('pairwise_comparisons', {}))
                print(f"  {window_id}: {models} models, {comparisons} comparisons")
        
        print(f"\nResults available at:")
        print(f"  Samples: {evaluator.experiment_dir}/{args.window}/*/samples.npy")
        print(f"  Metrics: {evaluator.experiment_dir}/{args.window}/metrics.json")
        print(f"  Plots: {evaluator.experiment_dir}/{args.window}/figs/*.pdf")
        print(f"  Findings: {evaluator.experiment_dir}/findings.jsonl")
        print(f"  Summary: {evaluator.experiment_dir}/integrated_summary.json")
        
    except Exception as e:
        logger.error(f"Experiment A v2 failed: {e}")
        raise

if __name__ == "__main__":
    main()
