#!/usr/bin/env python3
"""
Experiment A: Out-of-Sample Stress Testing Evaluator

Orchestrates comprehensive out-of-sample evaluation using pre-COVID checkpoints.
Tests model performance on stress periods (COVID crash, recovery, etc.) using
models trained only on pre-COVID data (2010-2019).

Features:
- Automatic checkpoint discovery and validation
- Pre-COVID checkpoint verification 
- Stress period window definitions
- Causal conditioning generation
- Sample generation orchestration
- Comprehensive metadata tracking
- Versioning safety (A_v2, A_v3, etc.)
- Detailed execution manifests

Author: Simin Ali
Thesis: Diffusion Models in Generative AI for Financial Data Synthesis and Risk Management
"""

import argparse
import json
import numpy as np
import pandas as pd
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional, Any
import re
import warnings

# Import our custom modules
try:
    from checkpoint_loader_sampler import CheckpointSampler, load_and_sample
    from eval_conditioning_providers import generate_eval_conditioning, load_conditioning_spec
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Make sure checkpoint_loader_sampler.py and eval_conditioning_providers.py are available")
    sys.exit(1)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ExperimentAEvaluator:
    """Main evaluator class for Experiment A."""
    
    # Pre-defined stress testing windows
    STRESS_WINDOWS = {
        'covid_crash': {
            'name': 'COVID Market Crash',
            'start': '2020-02-20',
            'end': '2020-04-01',
            'description': 'Initial COVID-19 market crash period'
        },
        'covid_recovery': {
            'name': 'COVID Recovery',
            'start': '2020-04-15',
            'end': '2020-06-15',
            'description': 'Post-crash recovery period'
        },
        'covid_second_wave': {
            'name': 'COVID Second Wave',
            'start': '2020-10-01',
            'end': '2020-12-31',
            'description': 'COVID second wave and vaccine news'
        },
        'post_covid': {
            'name': 'Post-COVID Normal',
            'start': '2021-06-01',
            'end': '2021-12-31',
            'description': 'Post-COVID normalization period'
        },
        'inflation_2022': {
            'name': 'Inflation Concerns',
            'start': '2022-01-01',
            'end': '2022-06-30',
            'description': 'High inflation and rate hike concerns'
        }
    }
    
    def __init__(self, checkpoints_dir: str = 'checkpoints/precovid',
                 results_base: str = 'results/addons/period_slices',
                 experiment_name: str = 'A'):
        
        self.checkpoints_dir = Path(checkpoints_dir)
        self.results_base = Path(results_base)
        self.experiment_name = experiment_name
        
        # Find available experiment directory (versioning)
        self.experiment_dir = self._find_experiment_dir()
        
        # Initialize tracking
        self.manifest = {
            'experiment': experiment_name,
            'experiment_dir': str(self.experiment_dir),
            'started_at': datetime.now().isoformat(),
            'checkpoints_dir': str(self.checkpoints_dir),
            'windows': {},
            'models': {},
            'results': {},
            'errors': [],
            'warnings': [],
            'status': 'initializing'
        }
        
        # Create experiment directory
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized Experiment A in: {self.experiment_dir}")
    
    def _find_experiment_dir(self) -> Path:
        """Find available experiment directory with versioning."""
        base_dir = self.results_base / self.experiment_name
        
        if not base_dir.exists():
            return base_dir
        
        # Find next available version
        version = 2
        while True:
            versioned_dir = self.results_base / f"{self.experiment_name}_v{version}"
            if not versioned_dir.exists():
                logger.info(f"Using versioned experiment directory: {versioned_dir}")
                return versioned_dir
            version += 1
    
    def discover_checkpoints(self) -> Dict[str, Dict[str, Any]]:
        """Discover and validate pre-COVID checkpoints."""
        logger.info(f"Discovering checkpoints in: {self.checkpoints_dir}")
        
        checkpoints = {}
        
        if not self.checkpoints_dir.exists():
            error_msg = f"Checkpoints directory not found: {self.checkpoints_dir}"
            self.manifest['errors'].append(error_msg)
            logger.error(error_msg)
            return checkpoints
        
        # Scan for model directories
        for model_dir in self.checkpoints_dir.iterdir():
            if not model_dir.is_dir():
                continue
            
            model_name = model_dir.name
            logger.info(f"Found model directory: {model_name}")
            
            # Look for training period subdirectories
            checkpoint_info = self._validate_model_checkpoints(model_dir, model_name)
            if checkpoint_info:
                checkpoints[model_name] = checkpoint_info
        
        logger.info(f"Discovered {len(checkpoints)} valid model checkpoints")
        self.manifest['models'] = checkpoints
        
        return checkpoints
    
    def _validate_model_checkpoints(self, model_dir: Path, model_name: str) -> Optional[Dict[str, Any]]:
        """Validate checkpoints for a specific model."""
        valid_checkpoints = []
        
        # Look for period subdirectories (e.g., 20100101-20191231)
        for period_dir in model_dir.iterdir():
            if not period_dir.is_dir():
                continue
            
            period_name = period_dir.name
            
            # Validate this is a pre-COVID checkpoint
            is_precovid, warning = self._validate_precovid_period(period_name)
            
            # Check for required files
            meta_file = period_dir / 'meta.json'
            spec_file = period_dir / 'conditioning_spec.json'
            best_checkpoint = period_dir / 'best.pt'
            
            if not all([meta_file.exists(), spec_file.exists(), best_checkpoint.exists()]):
                warning_msg = f"Incomplete checkpoint: {period_dir} (missing required files)"
                self.manifest['warnings'].append(warning_msg)
                logger.warning(warning_msg)
                continue
            
            # Load metadata
            try:
                with open(meta_file, 'r') as f:
                    meta_data = json.load(f)
                
                with open(spec_file, 'r') as f:
                    conditioning_spec = json.load(f)
                
                checkpoint_info = {
                    'period': period_name,
                    'path': str(period_dir),
                    'is_precovid': is_precovid,
                    'meta_data': meta_data,
                    'conditioning_spec': conditioning_spec,
                    'status': 'valid' if is_precovid else 'suspect_for_A'
                }
                
                if warning:
                    checkpoint_info['warning'] = warning
                    self.manifest['warnings'].append(f"{model_name}/{period_name}: {warning}")
                
                valid_checkpoints.append(checkpoint_info)
                logger.info(f"Validated checkpoint: {model_name}/{period_name} ({'pre-COVID' if is_precovid else 'SUSPECT'})")
                
            except Exception as e:
                error_msg = f"Failed to load metadata for {period_dir}: {e}"
                self.manifest['errors'].append(error_msg)
                logger.error(error_msg)
        
        if not valid_checkpoints:
            return None
        
        return {
            'model_name': model_name,
            'checkpoints': valid_checkpoints,
            'primary_checkpoint': valid_checkpoints[0]  # Use first valid checkpoint
        }
    
    def _validate_precovid_period(self, period_name: str) -> Tuple[bool, Optional[str]]:
        """Validate if a period is clearly pre-COVID."""
        
        # Expected format: YYYYMMDD-YYYYMMDD
        period_pattern = r'(\d{8})-(\d{8})'
        match = re.match(period_pattern, period_name)
        
        if not match:
            return False, f"Invalid period format: {period_name}"
        
        start_str, end_str = match.groups()
        
        try:
            start_date = pd.Timestamp(start_str)
            end_date = pd.Timestamp(end_str)
            
            # Pre-COVID cutoff: 2019-12-31
            precovid_cutoff = pd.Timestamp('2019-12-31')
            
            if end_date <= precovid_cutoff:
                return True, None  # Clearly pre-COVID
            elif start_date <= precovid_cutoff < end_date:
                return False, f"Period spans COVID boundary: {start_date.date()} to {end_date.date()}"
            else:
                return False, f"Period starts after COVID: {start_date.date()}"
                
        except Exception as e:
            return False, f"Failed to parse dates: {e}"
    
    def load_returns_data(self, csv_file: str) -> pd.DataFrame:
        """Load and validate returns data."""
        logger.info(f"Loading returns data from: {csv_file}")
        
        try:
            # Try different common formats
            if os.path.exists(csv_file):
                returns_data = pd.read_csv(csv_file, index_col=0, parse_dates=True)
            else:
                # Create synthetic data as fallback
                logger.warning(f"Returns file not found: {csv_file}, using synthetic data")
                date_range = pd.date_range('2010-01-01', '2023-12-31', freq='D')
                np.random.seed(42)
                returns = np.random.normal(0.0004, 0.01, len(date_range))
                returns_data = pd.DataFrame({'returns': returns}, index=date_range)
            
            # Standardize column names
            if 'log_returns' in returns_data.columns:
                returns_data['returns'] = returns_data['log_returns']
            elif 'Log_Returns' in returns_data.columns:
                returns_data['returns'] = returns_data['Log_Returns']
            elif len(returns_data.columns) == 1:
                returns_data['returns'] = returns_data.iloc[:, 0]
            
            if 'returns' not in returns_data.columns:
                raise ValueError("Could not find returns column in data")
            
            # Remove weekends and invalid data
            returns_data = returns_data.dropna()
            
            logger.info(f"Loaded returns data: {returns_data.shape} from {returns_data.index.min()} to {returns_data.index.max()}")
            return returns_data
            
        except Exception as e:
            error_msg = f"Failed to load returns data: {e}"
            self.manifest['errors'].append(error_msg)
            logger.error(error_msg)
            raise
    
    def generate_samples_for_window_model(self, window_id: str, window_info: Dict[str, Any],
                                        model_name: str, checkpoint_info: Dict[str, Any],
                                        returns_data: pd.DataFrame, num_paths: int, 
                                        seq_len: int, seed: int) -> Dict[str, Any]:
        """Generate samples for a specific window-model combination."""
        
        logger.info(f"Generating samples: {window_id} × {model_name}")
        
        # Set random seed for reproducibility
        np.random.seed(seed)
        
        # Get checkpoint path
        checkpoint_dir = checkpoint_info['path']
        
        # Create output directory
        output_dir = self.experiment_dir / window_id / model_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate target dates for the window
        start_date = pd.Timestamp(window_info['start'])
        end_date = pd.Timestamp(window_info['end'])
        
        # Create weekly sampling dates within the window
        target_dates = pd.date_range(start_date, end_date, freq='W').tolist()
        if end_date not in target_dates:
            target_dates.append(end_date)
        
        try:
            # Generate evaluation conditioning
            logger.info(f"Generating conditioning for {len(target_dates)} dates")
            conditioning, conditioning_warnings = generate_eval_conditioning(
                checkpoint_dir, returns_data, target_dates
            )
            
            # Generate samples using checkpoint sampler
            logger.info(f"Generating {num_paths} sample paths")
            samples = load_and_sample(
                checkpoint_dir=checkpoint_dir,
                dates=target_dates,
                num_paths=num_paths,
                output_dir=str(output_dir),
                seq_len=seq_len
            )
            
            # Create comprehensive result metadata
            result_info = {
                'window_id': window_id,
                'window_info': window_info,
                'model_name': model_name,
                'checkpoint_info': {
                    'period': checkpoint_info['period'],
                    'path': checkpoint_info['path'],
                    'is_precovid': checkpoint_info['is_precovid'],
                    'status': checkpoint_info['status']
                },
                'generation_info': {
                    'target_dates': [d.isoformat() for d in target_dates],
                    'num_paths': num_paths,
                    'seq_len': seq_len,
                    'seed': seed,
                    'samples_shape': list(samples.shape) if samples is not None else None
                },
                'conditioning_info': {
                    'type': checkpoint_info['conditioning_spec']['type'],
                    'conditioning_dim': checkpoint_info['conditioning_spec']['conditioning_dim'],
                    'conditioning_shape': list(conditioning.shape) if conditioning is not None else None,
                    'warnings': conditioning_warnings
                },
                'output_files': {
                    'samples': str(output_dir / 'samples.npy'),
                    'metadata': str(output_dir / 'sample_metadata.json'),
                    'manifest': str(output_dir / 'manifest.json')
                },
                'generated_at': datetime.now().isoformat(),
                'status': 'success'
            }
            
            # Save additional metadata
            metadata_file = output_dir / 'experiment_A_metadata.json'
            with open(metadata_file, 'w') as f:
                json.dump(result_info, f, indent=2)
            
            logger.info(f"Successfully generated samples: {samples.shape if samples is not None else 'None'}")
            return result_info
            
        except Exception as e:
            error_msg = f"Failed to generate samples for {window_id} × {model_name}: {e}"
            logger.error(error_msg)
            
            result_info = {
                'window_id': window_id,
                'model_name': model_name,
                'checkpoint_path': checkpoint_dir,
                'error': str(e),
                'status': 'failed',
                'generated_at': datetime.now().isoformat()
            }
            
            self.manifest['errors'].append(error_msg)
            return result_info
    
    def run_experiment(self, windows: List[str], checkpoints: Dict[str, Dict[str, Any]],
                      returns_data: pd.DataFrame, num_paths: int, seq_len: int, 
                      seeds: List[int]) -> Dict[str, Any]:
        """Run the complete Experiment A."""
        
        logger.info("Starting Experiment A execution")
        self.manifest['status'] = 'running'
        
        # Validate windows
        valid_windows = {}
        for window_id in windows:
            if window_id in self.STRESS_WINDOWS:
                valid_windows[window_id] = self.STRESS_WINDOWS[window_id]
            else:
                warning_msg = f"Unknown window: {window_id}"
                self.manifest['warnings'].append(warning_msg)
                logger.warning(warning_msg)
        
        if not valid_windows:
            error_msg = "No valid windows specified"
            self.manifest['errors'].append(error_msg)
            logger.error(error_msg)
            return self.manifest
        
        self.manifest['windows'] = valid_windows
        
        # Create execution plan
        plan = self._create_execution_plan(valid_windows, checkpoints, num_paths, seq_len, seeds)
        
        # Save plan
        plan_file = self.experiment_dir / 'plan.json'
        with open(plan_file, 'w') as f:
            json.dump(plan, f, indent=2)
        logger.info(f"Saved execution plan: {plan_file}")
        
        # Execute the plan
        results = {}
        total_tasks = len(valid_windows) * len(checkpoints)
        completed_tasks = 0
        
        for window_id, window_info in valid_windows.items():
            results[window_id] = {}
            
            for model_name, model_info in checkpoints.items():
                checkpoint_info = model_info['primary_checkpoint']
                
                # Use first seed for main generation
                main_seed = seeds[0] if seeds else 42
                
                logger.info(f"Progress: {completed_tasks + 1}/{total_tasks} - {window_id} × {model_name}")
                
                result_info = self.generate_samples_for_window_model(
                    window_id, window_info, model_name, checkpoint_info,
                    returns_data, num_paths, seq_len, main_seed
                )
                
                results[window_id][model_name] = result_info
                completed_tasks += 1
                
                # Update manifest progress
                self.manifest['progress'] = f"{completed_tasks}/{total_tasks}"
        
        # Finalize manifest
        self.manifest['results'] = results
        self.manifest['completed_at'] = datetime.now().isoformat()
        self.manifest['status'] = 'completed' if not self.manifest['errors'] else 'completed_with_errors'
        
        # Save final manifest
        manifest_file = self.experiment_dir / 'manifest.json'
        with open(manifest_file, 'w') as f:
            json.dump(self.manifest, f, indent=2)
        
        logger.info(f"Experiment A completed: {manifest_file}")
        logger.info(f"Total tasks: {total_tasks}, Errors: {len(self.manifest['errors'])}, Warnings: {len(self.manifest['warnings'])}")
        
        return self.manifest
    
    def _create_execution_plan(self, windows: Dict[str, Any], checkpoints: Dict[str, Any],
                              num_paths: int, seq_len: int, seeds: List[int]) -> Dict[str, Any]:
        """Create detailed execution plan."""
        
        plan = {
            'experiment': 'A',
            'description': 'Out-of-sample stress testing with pre-COVID checkpoints',
            'created_at': datetime.now().isoformat(),
            'parameters': {
                'num_paths': num_paths,
                'seq_len': seq_len,
                'seeds': seeds
            },
            'windows': windows,
            'models': {name: {
                'checkpoints': len(info['checkpoints']),
                'primary_checkpoint': info['primary_checkpoint']['period'],
                'status': info['primary_checkpoint']['status']
            } for name, info in checkpoints.items()},
            'execution_matrix': {},
            'total_tasks': len(windows) * len(checkpoints),
            'estimated_outputs': []
        }
        
        # Create execution matrix
        for window_id in windows:
            plan['execution_matrix'][window_id] = {}
            for model_name in checkpoints:
                output_path = f"results/addons/period_slices/A/{window_id}/{model_name}/samples.npy"
                plan['execution_matrix'][window_id][model_name] = {
                    'output_path': output_path,
                    'status': 'planned'
                }
                plan['estimated_outputs'].append(output_path)
        
        return plan

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Experiment A: Out-of-Sample Stress Testing Evaluator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default windows
  python experiment_A_evaluator.py --csv-file sp500_data.csv --num-paths 1000
  
  # Specific windows and parameters
  python experiment_A_evaluator.py \\
    --windows covid_crash covid_recovery \\
    --csv-file data/returns.csv \\
    --num-paths 5000 \\
    --seq-len 60 \\
    --seeds 42 123 456
  
  # Custom directories
  python experiment_A_evaluator.py \\
    --checkpoints-dir custom_checkpoints/precovid \\
    --results-base custom_results \\
    --csv-file data.csv
        """
    )
    
    parser.add_argument('--windows', nargs='+', 
                       default=['covid_crash', 'covid_recovery', 'post_covid'],
                       help='Stress testing windows to evaluate')
    
    parser.add_argument('--checkpoints-dir', type=str, default='checkpoints/precovid',
                       help='Directory containing pre-COVID checkpoints')
    
    parser.add_argument('--csv-file', type=str, default='sp500_data.csv',
                       help='CSV file with returns data')
    
    parser.add_argument('--seq-len', type=int, default=60,
                       help='Sequence length for sample generation')
    
    parser.add_argument('--num-paths', type=int, default=1000,
                       help='Number of sample paths to generate')
    
    parser.add_argument('--seeds', nargs='+', type=int, default=[42],
                       help='Random seeds for reproducibility')
    
    parser.add_argument('--results-base', type=str, default='results/addons/period_slices',
                       help='Base directory for results')
    
    parser.add_argument('--experiment-name', type=str, default='A',
                       help='Experiment identifier')
    
    parser.add_argument('--list-windows', action='store_true',
                       help='List available stress testing windows and exit')
    
    return parser.parse_args()

def main():
    """Main execution function."""
    args = parse_arguments()
    
    # List available windows if requested
    if args.list_windows:
        print("Available Stress Testing Windows:")
        print("=" * 40)
        for window_id, info in ExperimentAEvaluator.STRESS_WINDOWS.items():
            print(f"{window_id:20} {info['start']} to {info['end']}")
            print(f"{'':20} {info['description']}")
            print()
        return
    
    try:
        # Initialize evaluator
        evaluator = ExperimentAEvaluator(
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
        
        # Run experiment
        manifest = evaluator.run_experiment(
            windows=args.windows,
            checkpoints=checkpoints,
            returns_data=returns_data,
            num_paths=args.num_paths,
            seq_len=args.seq_len,
            seeds=args.seeds
        )
        
        # Print summary
        print("\n" + "="*60)
        print("EXPERIMENT A SUMMARY")
        print("="*60)
        print(f"Experiment Directory: {evaluator.experiment_dir}")
        print(f"Status: {manifest['status']}")
        print(f"Windows Tested: {len(manifest['windows'])}")
        print(f"Models Tested: {len(manifest['models'])}")
        print(f"Total Tasks: {manifest.get('progress', 'N/A')}")
        print(f"Errors: {len(manifest['errors'])}")
        print(f"Warnings: {len(manifest['warnings'])}")
        
        if manifest['errors']:
            print("\nErrors:")
            for error in manifest['errors'][:5]:  # Show first 5
                print(f"  - {error}")
            if len(manifest['errors']) > 5:
                print(f"  ... and {len(manifest['errors']) - 5} more")
        
        if manifest['warnings']:
            print("\nWarnings:")
            for warning in manifest['warnings'][:3]:  # Show first 3
                print(f"  - {warning}")
            if len(manifest['warnings']) > 3:
                print(f"  ... and {len(manifest['warnings']) - 3} more")
        
        print(f"\nResults saved to: {evaluator.experiment_dir}")
        print("Use the manifest.json and plan.json files for detailed information.")
        
    except Exception as e:
        logger.error(f"Experiment A failed: {e}")
        raise

if __name__ == "__main__":
    main()
