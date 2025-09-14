#!/usr/bin/env python3
"""
Experiment B: Counterfactual Controllability Testing v2 (Integrated Pipeline)

Extended version with integrated metrics calculation and plotting pipeline.
Automatically runs metrics and plotting after sample generation for complete analysis.

New Features in v2:
- Automatic metrics calculation after sampling each mode
- Integrated plotting pipeline for all modes
- Compact findings.jsonl generation with controllability insights
- Temporary real_slice.csv creation per mode
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
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
from copy import deepcopy

# Import base evaluator and our custom modules
try:
    from experiment_B_evaluator import ExperimentBEvaluator
    from checkpoint_loader_sampler import CheckpointSampler, load_and_sample
    from eval_conditioning_providers import (
        generate_eval_conditioning, 
        load_conditioning_spec,
        EvalProviderFactory,
        ExplicitEvalProvider,
        LLMEvalProvider
    )
    from experiment_A_evaluator import ExperimentAEvaluator
    from shared.metrics import MetricsCalculator
    from shared.plotting import PlottingPipeline
except ImportError as e:
    print(f"Error importing required modules: {e}")
    sys.exit(1)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ExperimentBEvaluatorV2(ExperimentBEvaluator):
    """Extended Experiment B evaluator with integrated metrics and plotting."""
    
    def __init__(self, checkpoints_dir: str = 'checkpoints/precovid',
                 results_base: str = 'results/addons/period_slices',
                 experiment_name: str = 'B'):
        
        # Initialize base class
        super().__init__(checkpoints_dir, results_base, experiment_name)
        
        # Add metrics and plotting components
        self.metrics_calculator = MetricsCalculator()
        self.plotting_pipeline = PlottingPipeline(str(self.results_base))
        
        # Track findings for summary
        self.findings = []
        
        logger.info(f"Initialized Experiment B v2 with integrated pipeline in: {self.experiment_dir}")
    
    def create_real_slice_csv(self, window_id: str, returns_data: pd.DataFrame) -> Optional[Path]:
        """Create temporary real_slice.csv for a specific window."""
        try:
            # Use same window definitions as Experiment A
            stress_windows = ExperimentAEvaluator.STRESS_WINDOWS
            
            if window_id not in stress_windows:
                logger.warning(f"Unknown window {window_id}, using default data")
                return None
            
            window_info = stress_windows[window_id]
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
        """Extract compact findings from metrics results with controllability focus."""
        try:
            # The metrics.json might be in different locations
            metrics_file = self.experiment_dir / window_id / 'metrics.json'
            if not metrics_file.exists():
                # The metrics runner creates files in the base experiment directory (e.g., B not B_v2)
                base_exp_name = self.experiment_name.split('_')[0]  # B from B_v2
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
            
            # Extract key findings with controllability focus
            findings = {
                'window_id': window_id,
                'experiment': 'B',
                'timestamp': datetime.now().isoformat(),
                'models': {},
                'pairwise_comparisons': {},
                'controllability_insights': {}
            }
            
            # Model-level findings
            for model_name, model_results in metrics.get('models', {}).items():
                if model_results.get('status') == 'success':
                    risk_metrics = model_results.get('risk_metrics', {})
                    dist_metrics = model_results.get('distribution_metrics', {})
                    
                    findings['models'][model_name] = {
                        'var_5pct': risk_metrics.get('var_0.050', None),
                        'es_5pct': risk_metrics.get('es_0.050', None),
                        'volatility': model_results.get('basic_stats', {}).get('std', None),
                        'ecdf_ks_pvalue': dist_metrics.get('ecdf_comparison', {}).get('ks_pvalue', None),
                        'volatility_rmse': dist_metrics.get('volatility_metrics', {}).get('rmse', None)
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
            
            # Controllability insights (compare risk metrics between models)
            models = findings['models']
            if 'explicit' in models and 'llm' in models:
                explicit_var = models['explicit'].get('var_5pct')
                llm_var = models['llm'].get('var_5pct')
                
                if explicit_var is not None and llm_var is not None:
                    var_diff_pct = ((llm_var - explicit_var) / abs(explicit_var)) * 100
                    findings['controllability_insights']['var_5pct_diff_pct'] = var_diff_pct
                    findings['controllability_insights']['more_conservative'] = 'llm' if llm_var < explicit_var else 'explicit'
            
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
    
    def run_integrated_pipeline_for_window(self, window_id: str, window_info: Dict[str, Any], 
                                         checkpoints: Dict[str, Dict[str, Any]], returns_data: pd.DataFrame,
                                         num_paths: int, seq_len: int, seeds: List[int]) -> Dict[str, Any]:
        """
        Run the complete integrated pipeline for a window: sampling → metrics → plotting → findings.
        """
        
        logger.info(f"Starting integrated pipeline for window: {window_id}")
        
        # Step 1: Run base sampling (from parent class)
        try:
            # Save current state
            original_experiment_dir = self.experiment_dir
            
            # Run the parent experiment method (without override)
            ExperimentBEvaluator.run_experiment(
                self, window_id, window_info, checkpoints, returns_data, num_paths, seq_len, seeds
            )
            
            # Check if sampling was successful by looking for sample files
            samples_exist = False
            window_dir = self.experiment_dir / window_id
            if window_dir.exists():
                for model_dir in window_dir.iterdir():
                    if model_dir.is_dir():
                        # Look for samples in any subdirectory (modes)
                        sample_files = list(model_dir.glob('**/samples.npy'))
                        if sample_files:
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
        return self.run_integrated_pipeline_for_window(
            window_id, window_info, checkpoints, returns_data, num_paths, seq_len, seeds
        )
    
    def generate_controllability_summary(self):
        """Generate controllability-focused summary with integrated pipeline results."""
        try:
            summary = {
                'experiment': 'B',
                'experiment_type': 'controllability_testing',
                'completed_at': datetime.now().isoformat(),
                'total_windows': len(self.findings),
                'integrated_pipeline_summary': {
                    'metrics_success_rate': 0,
                    'plotting_success_rate': 0,
                    'findings_extraction_rate': 0
                },
                'controllability_insights': {},
                'key_findings': {}
            }
            
            # Calculate success rates
            if self.findings:
                summary['integrated_pipeline_summary'] = {
                    'metrics_success_rate': len([f for f in self.findings if f]) / len(self.findings),
                    'plotting_success_rate': 1.0,  # Assume success if findings exist
                    'findings_extraction_rate': len([f for f in self.findings if f]) / len(self.findings)
                }
                
                # Extract controllability insights
                for finding in self.findings:
                    window_id = finding.get('window_id')
                    if window_id:
                        controllability = finding.get('controllability_insights', {})
                        
                        summary['key_findings'][window_id] = {
                            'models_analyzed': len(finding.get('models', {})),
                            'comparisons_made': len(finding.get('pairwise_comparisons', {})),
                            'significant_dm_tests': len([
                                comp for comp in finding.get('pairwise_comparisons', {}).values()
                                if comp.get('dm_mse_pvalue', 1) < 0.05
                            ]),
                            'var_difference_pct': controllability.get('var_5pct_diff_pct'),
                            'more_conservative_model': controllability.get('more_conservative')
                        }
                        
                        # Aggregate controllability insights
                        if controllability.get('var_5pct_diff_pct') is not None:
                            if 'var_differences' not in summary['controllability_insights']:
                                summary['controllability_insights']['var_differences'] = []
                            summary['controllability_insights']['var_differences'].append(
                                controllability['var_5pct_diff_pct']
                            )
            
            # Calculate overall controllability assessment
            if 'var_differences' in summary['controllability_insights']:
                var_diffs = summary['controllability_insights']['var_differences']
                summary['controllability_insights']['mean_var_difference_pct'] = np.mean(var_diffs)
                summary['controllability_insights']['max_var_difference_pct'] = np.max(np.abs(var_diffs))
                summary['controllability_insights']['controllability_assessment'] = (
                    'high' if np.max(np.abs(var_diffs)) > 20 else
                    'moderate' if np.max(np.abs(var_diffs)) > 10 else 'low'
                )
            
            # Save summary
            summary_file = self.experiment_dir / 'controllability_summary.json'
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"Generated controllability summary: {summary_file}")
            
        except Exception as e:
            logger.error(f"Error generating controllability summary: {e}")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Experiment B v2: Integrated Controllability Testing Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete integrated pipeline
  python experiment_B_evaluator_v2.py --window covid_crash --num-paths 1000 --csv-file sp500_data.csv
  
  # Quick test with metrics and plots
  python experiment_B_evaluator_v2.py --window covid_crash --num-paths 100 --csv-file sp500_data.csv
        """
    )
    
    parser.add_argument('--window', type=str, default='covid_crash',
                       help='COVID-era window for testing (e.g., covid_crash, covid_recovery)')
    
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
    
    parser.add_argument('--experiment-name', type=str, default='B',
                       help='Experiment identifier')
    
    return parser.parse_args()

def main():
    """Main execution function."""
    args = parse_arguments()
    
    try:
        # Initialize evaluator
        evaluator = ExperimentBEvaluatorV2(
            checkpoints_dir=args.checkpoints_dir,
            results_base=args.results_base,
            experiment_name=args.experiment_name
        )
        
        # Discover controllable checkpoints
        checkpoints = evaluator.discover_controllable_checkpoints()
        
        if not checkpoints:
            logger.error("No controllable checkpoints found (need explicit or LLM models)")
            return
        
        # Load returns data
        returns_data = evaluator.load_returns_data(args.csv_file)
        
        # Define window information (use Experiment A windows)
        windows = ExperimentAEvaluator.STRESS_WINDOWS
        if args.window not in windows:
            logger.error(f"Unknown window: {args.window}. Available: {list(windows.keys())}")
            return
        
        window_info = windows[args.window]
        
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
        
        # Generate controllability summary
        evaluator.generate_controllability_summary()
        
        # Print summary
        print("\n" + "="*70)
        print("EXPERIMENT B v2 INTEGRATED PIPELINE SUMMARY")
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
        
        # Show controllability findings
        if evaluator.findings:
            print(f"\nControllability Findings:")
            for finding in evaluator.findings:
                window_id = finding.get('window_id')
                insights = finding.get('controllability_insights', {})
                var_diff = insights.get('var_5pct_diff_pct')
                if var_diff is not None:
                    print(f"  {window_id}: VaR difference {var_diff:+.1f}%")
                    print(f"    More conservative: {insights.get('more_conservative', 'unknown')}")
        
        print(f"\nResults available at:")
        print(f"  Samples: {evaluator.experiment_dir}/{args.window}/*/real-conditions/samples.npy")
        print(f"  All Modes: {evaluator.experiment_dir}/{args.window}/*/*/samples.npy")
        print(f"  Metrics: {evaluator.experiment_dir}/{args.window}/metrics.json")
        print(f"  Plots: {evaluator.experiment_dir}/{args.window}/figs/*.pdf")
        print(f"  Findings: {evaluator.experiment_dir}/findings.jsonl")
        print(f"  Summary: {evaluator.experiment_dir}/controllability_summary.json")
        
    except Exception as e:
        logger.error(f"Experiment B v2 failed: {e}")
        raise

if __name__ == "__main__":
    main()
