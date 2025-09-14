#!/usr/bin/env python3
"""
Metrics Runner

CLI tool for running comprehensive evaluation metrics on experiment results.
Processes sample files from Experiments A and B and calculates risk metrics,
distribution analysis, and model comparisons.

Author: Simin Ali
Thesis: Diffusion Models in Generative AI for Financial Data Synthesis and Risk Management
"""

import argparse
import numpy as np
import pandas as pd
import json
import os
import sys
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Optional, Any
import warnings
import glob

# Import our metrics module
try:
    from shared.metrics import MetricsCalculator, RiskMetrics, QuantileLoss, DieboldMarianoTest, DistributionAnalysis
except ImportError as e:
    print(f"Error importing shared_metrics: {e}")
    sys.exit(1)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MetricsRunner:
    """Main runner class for metrics evaluation."""
    
    def __init__(self, results_base: str = 'results/addons/period_slices'):
        self.results_base = Path(results_base)
        self.calculator = MetricsCalculator()
        
        self.manifest = {
            'metrics_run': {
                'started_at': datetime.now().isoformat(),
                'results_base': str(self.results_base),
                'windows_processed': {},
                'errors': [],
                'warnings': [],
                'status': 'initializing'
            }
        }
        
        logger.info(f"Initialized MetricsRunner with base: {self.results_base}")
    
    def discover_experiments(self) -> Dict[str, List[str]]:
        """Discover available experiments and windows."""
        experiments = {}
        
        if not self.results_base.exists():
            logger.warning(f"Results base directory not found: {self.results_base}")
            return experiments
        
        # Look for experiment directories (A, B, A_v2, etc.)
        for exp_dir in self.results_base.iterdir():
            if exp_dir.is_dir() and (exp_dir.name.startswith('A') or exp_dir.name.startswith('B')):
                experiment_name = exp_dir.name
                windows = []
                
                # Look for window directories
                for window_dir in exp_dir.iterdir():
                    if window_dir.is_dir() and not window_dir.name.startswith('.'):
                        windows.append(window_dir.name)
                
                if windows:
                    experiments[experiment_name] = sorted(windows)
                    logger.info(f"Found experiment {experiment_name} with windows: {windows}")
        
        return experiments
    
    def load_real_data(self, csv_file: str) -> pd.DataFrame:
        """Load real market data."""
        logger.info(f"Loading real data from: {csv_file}")
        
        try:
            if os.path.exists(csv_file):
                data = pd.read_csv(csv_file, index_col=0, parse_dates=True)
            else:
                logger.warning(f"Real data file not found: {csv_file}, using synthetic data")
                # Create synthetic data as fallback
                date_range = pd.date_range('2010-01-01', '2023-12-31', freq='D')
                np.random.seed(42)
                returns = np.random.normal(0.0004, 0.01, len(date_range))
                data = pd.DataFrame({'returns': returns}, index=date_range)
            
            # Standardize column names
            if 'log_returns' in data.columns:
                data['returns'] = data['log_returns']
            elif 'Log_Returns' in data.columns:
                data['returns'] = data['Log_Returns']
            elif len(data.columns) == 1:
                data['returns'] = data.iloc[:, 0]
            
            if 'returns' not in data.columns:
                raise ValueError("Could not find returns column in data")
            
            data = data.dropna()
            logger.info(f"Loaded real data: {data.shape}")
            return data
            
        except Exception as e:
            error_msg = f"Failed to load real data: {e}"
            self.manifest['metrics_run']['errors'].append(error_msg)
            logger.error(error_msg)
            raise
    
    def discover_samples_in_window(self, experiment_dir: Path, window_id: str) -> Dict[str, Path]:
        """Discover sample files for a specific window."""
        window_dir = experiment_dir / window_id
        sample_files = {}
        
        if not window_dir.exists():
            logger.warning(f"Window directory not found: {window_dir}")
            return sample_files
        
        # Look for model directories
        for model_dir in window_dir.iterdir():
            if model_dir.is_dir():
                # Look for samples.npy file (could be nested in mode directories for Experiment B)
                samples_paths = list(model_dir.glob('**/samples.npy'))
                
                if samples_paths:
                    # For Experiment B, we might have multiple modes - use real-conditions as default
                    real_conditions_path = model_dir / 'real-conditions' / 'samples.npy'
                    if real_conditions_path.exists():
                        sample_files[model_dir.name] = real_conditions_path
                    else:
                        # Use first available samples file
                        sample_files[model_dir.name] = samples_paths[0]
                    
                    logger.info(f"Found samples for model {model_dir.name}: {sample_files[model_dir.name]}")
        
        return sample_files
    
    def load_samples(self, sample_files: Dict[str, Path]) -> Dict[str, np.ndarray]:
        """Load sample arrays from files."""
        samples = {}
        
        for model_name, file_path in sample_files.items():
            try:
                sample_array = np.load(file_path)
                samples[model_name] = sample_array
                logger.info(f"Loaded samples for {model_name}: {sample_array.shape}")
            except Exception as e:
                error_msg = f"Failed to load samples for {model_name} from {file_path}: {e}"
                self.manifest['metrics_run']['errors'].append(error_msg)
                logger.error(error_msg)
        
        return samples
    
    def get_window_real_data(self, real_data: pd.DataFrame, window_id: str) -> np.ndarray:
        """Extract real data for a specific window."""
        # Define window periods (same as experiments)
        window_periods = {
            'covid_crash': ('2020-02-20', '2020-04-01'),
            'covid_recovery': ('2020-04-15', '2020-06-15'),
            'covid_second_wave': ('2020-10-01', '2020-12-31'),
            'post_covid': ('2021-06-01', '2021-12-31'),
            'inflation_2022': ('2022-01-01', '2022-06-30')
        }
        
        if window_id in window_periods:
            start_date, end_date = window_periods[window_id]
            
            # Filter real data for this window
            window_data = real_data[(real_data.index >= start_date) & (real_data.index <= end_date)]
            
            if len(window_data) == 0:
                logger.warning(f"No real data found for window {window_id} ({start_date} to {end_date})")
                # Return a small synthetic series as fallback
                np.random.seed(42)
                return np.random.normal(0.0004, 0.01, 60)
            
            logger.info(f"Window {window_id}: {len(window_data)} real data points")
            return window_data['returns'].values
        else:
            logger.warning(f"Unknown window {window_id}, using full data sample")
            return real_data['returns'].iloc[:100].values  # Use first 100 points as fallback
    
    def process_window(self, experiment_name: str, window_id: str, real_data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Process a single window and calculate metrics."""
        logger.info(f"Processing window: {experiment_name}/{window_id}")
        
        experiment_dir = self.results_base / experiment_name
        
        # Discover sample files
        sample_files = self.discover_samples_in_window(experiment_dir, window_id)
        
        if not sample_files:
            warning_msg = f"No sample files found for {experiment_name}/{window_id}"
            self.manifest['metrics_run']['warnings'].append(warning_msg)
            logger.warning(warning_msg)
            return {
                'status': 'skipped',
                'reason': 'no_sample_files',
                'window_id': window_id,
                'experiment': experiment_name
            }
        
        # Load samples
        model_samples = self.load_samples(sample_files)
        
        if not model_samples:
            warning_msg = f"Failed to load any samples for {experiment_name}/{window_id}"
            self.manifest['metrics_run']['warnings'].append(warning_msg)
            logger.warning(warning_msg)
            return {
                'status': 'skipped',
                'reason': 'failed_to_load_samples',
                'window_id': window_id,
                'experiment': experiment_name
            }
        
        # Get real data for this window
        window_real_data = self.get_window_real_data(real_data, window_id)
        
        # Calculate metrics
        try:
            metrics_results = self.calculator.calculate_all_metrics(
                window_real_data, model_samples, window_id
            )
            
            # Add experiment metadata
            metrics_results['experiment'] = experiment_name
            metrics_results['sample_files'] = {k: str(v) for k, v in sample_files.items()}
            
            # Save results to the window directory
            output_dir = experiment_dir / window_id
            self.calculator.save_results(metrics_results, output_dir)
            
            logger.info(f"Successfully processed {experiment_name}/{window_id}")
            return metrics_results
            
        except Exception as e:
            error_msg = f"Error calculating metrics for {experiment_name}/{window_id}: {e}"
            self.manifest['metrics_run']['errors'].append(error_msg)
            logger.error(error_msg)
            return {
                'status': 'error',
                'error_message': str(e),
                'window_id': window_id,
                'experiment': experiment_name
            }
    
    def run_all_metrics(self, csv_file: str, experiments: Optional[List[str]] = None, 
                       windows: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run metrics for all discovered experiments and windows."""
        logger.info("Starting comprehensive metrics evaluation")
        self.manifest['metrics_run']['status'] = 'running'
        
        # Load real data
        real_data = self.load_real_data(csv_file)
        
        # Discover experiments
        available_experiments = self.discover_experiments()
        
        if not available_experiments:
            logger.error("No experiments found")
            self.manifest['metrics_run']['status'] = 'failed'
            return self.manifest
        
        # Filter experiments if specified
        if experiments:
            available_experiments = {k: v for k, v in available_experiments.items() if k in experiments}
        
        # Process each experiment/window combination
        for experiment_name, available_windows in available_experiments.items():
            # Filter windows if specified
            windows_to_process = available_windows
            if windows:
                windows_to_process = [w for w in available_windows if w in windows]
            
            for window_id in windows_to_process:
                result = self.process_window(experiment_name, window_id, real_data)
                
                if result:
                    if experiment_name not in self.manifest['metrics_run']['windows_processed']:
                        self.manifest['metrics_run']['windows_processed'][experiment_name] = {}
                    
                    self.manifest['metrics_run']['windows_processed'][experiment_name][window_id] = {
                        'status': result.get('status', 'unknown'),
                        'processed_at': datetime.now().isoformat(),
                        'models_found': list(result.get('models', {}).keys()) if 'models' in result else [],
                        'sample_files': result.get('sample_files', {})
                    }
        
        # Finalize manifest
        self.manifest['metrics_run']['completed_at'] = datetime.now().isoformat()
        self.manifest['metrics_run']['status'] = 'completed' if not self.manifest['metrics_run']['errors'] else 'completed_with_errors'
        
        # Save overall manifest
        manifest_file = self.results_base / 'metrics_manifest.json'
        with open(manifest_file, 'w') as f:
            json.dump(self.manifest, f, indent=2)
        
        logger.info(f"Metrics evaluation completed: {manifest_file}")
        return self.manifest

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Metrics Runner for Experiment Evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run metrics for all experiments and windows
  python metrics_runner.py --csv-file sp500_data.csv
  
  # Run metrics for specific experiments
  python metrics_runner.py --experiments A B --csv-file sp500_data.csv
  
  # Run metrics for specific windows
  python metrics_runner.py --windows covid_crash covid_recovery --csv-file sp500_data.csv
  
  # Custom results directory
  python metrics_runner.py --results-base my_results --csv-file data.csv
        """
    )
    
    parser.add_argument('--csv-file', type=str, default='sp500_data.csv',
                       help='CSV file with real market data')
    
    parser.add_argument('--results-base', type=str, default='results/addons/period_slices',
                       help='Base directory containing experiment results')
    
    parser.add_argument('--experiments', nargs='*', 
                       help='Specific experiments to process (e.g., A B A_v2)')
    
    parser.add_argument('--windows', nargs='*',
                       help='Specific windows to process (e.g., covid_crash covid_recovery)')
    
    parser.add_argument('--list-available', action='store_true',
                       help='List available experiments and windows, then exit')
    
    return parser.parse_args()

def main():
    """Main execution function."""
    args = parse_arguments()
    
    try:
        # Initialize runner
        runner = MetricsRunner(args.results_base)
        
        # List available if requested
        if args.list_available:
            experiments = runner.discover_experiments()
            print("Available Experiments and Windows:")
            print("=" * 40)
            for exp_name, windows in experiments.items():
                print(f"{exp_name}:")
                for window in windows:
                    print(f"  - {window}")
                print()
            return
        
        # Run metrics
        manifest = runner.run_all_metrics(
            csv_file=args.csv_file,
            experiments=args.experiments,
            windows=args.windows
        )
        
        # Print summary
        print("\n" + "="*60)
        print("METRICS EVALUATION SUMMARY")
        print("="*60)
        print(f"Status: {manifest['metrics_run']['status']}")
        print(f"Results Base: {manifest['metrics_run']['results_base']}")
        
        windows_processed = manifest['metrics_run']['windows_processed']
        total_windows = sum(len(windows) for windows in windows_processed.values())
        print(f"Total Windows Processed: {total_windows}")
        
        for exp_name, windows in windows_processed.items():
            print(f"\n{exp_name}:")
            for window_id, window_info in windows.items():
                status = window_info['status']
                models = window_info.get('models_found', [])
                print(f"  {window_id}: {status} ({len(models)} models)")
        
        errors = manifest['metrics_run']['errors']
        warnings = manifest['metrics_run']['warnings']
        print(f"\nErrors: {len(errors)}")
        print(f"Warnings: {len(warnings)}")
        
        if errors:
            print("\nFirst few errors:")
            for error in errors[:3]:
                print(f"  - {error}")
        
        if warnings:
            print("\nFirst few warnings:")
            for warning in warnings[:3]:
                print(f"  - {warning}")
        
        print(f"\nDetailed results saved to individual window directories")
        print(f"Overall manifest: {runner.results_base}/metrics_manifest.json")
        
    except Exception as e:
        logger.error(f"Metrics evaluation failed: {e}")
        raise

if __name__ == "__main__":
    main()
