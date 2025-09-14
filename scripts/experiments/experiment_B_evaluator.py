#!/usr/bin/env python3
"""
Experiment B: Counterfactual Controllability Testing

Tests the controllability of pre-COVID models by manipulating conditioning inputs
while keeping model weights fixed. Evaluates how different conditioning scenarios
affect generation patterns and risk metrics.

Features:
- Real-conditions: Use actual COVID-era conditioning
- Calm-conditions: Use pre-COVID calm period conditioning on COVID dates
- LLM-knob: Systematically shift PCA components to test controllability
- Risk metric analysis (VaR, ES, tail mass)
- Comprehensive JSON summaries
- Versioning safety

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
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
from copy import deepcopy

# Import our custom modules
try:
    from checkpoint_loader_sampler import CheckpointSampler, load_and_sample
    from eval_conditioning_providers import (
        generate_eval_conditioning, 
        load_conditioning_spec,
        EvalProviderFactory,
        ExplicitEvalProvider,
        LLMEvalProvider
    )
    from experiment_A_evaluator import ExperimentAEvaluator
except ImportError as e:
    print(f"Error importing required modules: {e}")
    sys.exit(1)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ExperimentBEvaluator:
    """Main evaluator class for Experiment B."""
    
    def __init__(self, checkpoints_dir: str = 'checkpoints/precovid',
                 results_base: str = 'results/addons/period_slices',
                 experiment_name: str = 'B'):
        
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
            'window': {},
            'models': {},
            'modes': ['real-conditions', 'calm-conditions', 'llm-knob'],
            'results': {},
            'risk_analysis': {},
            'errors': [],
            'warnings': [],
            'status': 'initializing'
        }
        
        # Create experiment directory
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized Experiment B in: {self.experiment_dir}")
    
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
    
    def discover_controllable_checkpoints(self) -> Dict[str, Dict[str, Any]]:
        """Discover checkpoints suitable for controllability testing (explicit and LLM only)."""
        logger.info(f"Discovering controllable checkpoints in: {self.checkpoints_dir}")
        
        # Use the existing checkpoint discovery from Experiment A
        exp_a = ExperimentAEvaluator(str(self.checkpoints_dir))
        all_checkpoints = exp_a.discover_checkpoints()
        
        # Filter for controllable models (explicit and LLM)
        controllable_checkpoints = {}
        
        for model_name, model_info in all_checkpoints.items():
            if model_name in ['explicit', 'llm']:
                controllable_checkpoints[model_name] = model_info
                logger.info(f"Found controllable model: {model_name}")
            else:
                logger.info(f"Skipping non-controllable model: {model_name}")
        
        if not controllable_checkpoints:
            logger.warning("No controllable checkpoints found (explicit or LLM models)")
        
        self.manifest['models'] = controllable_checkpoints
        return controllable_checkpoints
    
    def load_returns_data(self, csv_file: str) -> pd.DataFrame:
        """Load and validate returns data."""
        logger.info(f"Loading returns data from: {csv_file}")
        
        try:
            if os.path.exists(csv_file):
                returns_data = pd.read_csv(csv_file, index_col=0, parse_dates=True)
            else:
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
            
            returns_data = returns_data.dropna()
            logger.info(f"Loaded returns data: {returns_data.shape}")
            return returns_data
            
        except Exception as e:
            error_msg = f"Failed to load returns data: {e}"
            self.manifest['errors'].append(error_msg)
            logger.error(error_msg)
            raise
    
    def generate_real_conditions(self, model_name: str, checkpoint_info: Dict[str, Any],
                                returns_data: pd.DataFrame, target_dates: List[pd.Timestamp]) -> np.ndarray:
        """Generate real conditioning for the target period."""
        logger.info(f"Generating real conditions for {model_name}")
        
        checkpoint_dir = checkpoint_info['path']
        conditioning, warnings = generate_eval_conditioning(
            checkpoint_dir, returns_data, target_dates
        )
        
        if warnings:
            self.manifest['warnings'].extend(warnings)
        
        return conditioning
    
    def generate_calm_conditions(self, model_name: str, checkpoint_info: Dict[str, Any],
                                returns_data: pd.DataFrame, target_dates: List[pd.Timestamp]) -> np.ndarray:
        """Generate calm conditions by using 2019 conditioning on 2020 dates."""
        logger.info(f"Generating calm conditions for {model_name}")
        
        # Find equivalent dates from 2019 (same day of week, similar time of year)
        calm_dates = []
        
        for target_date in target_dates:
            # Look for same weekday in 2019, preferring similar month
            year_2019_candidate = target_date.replace(year=2019)
            
            # If the exact date doesn't exist (e.g., Feb 29), find closest
            try:
                if year_2019_candidate in returns_data.index:
                    calm_dates.append(year_2019_candidate)
                else:
                    # Find closest available date in 2019
                    available_2019 = returns_data[returns_data.index.year == 2019].index
                    if len(available_2019) > 0:
                        closest_date = min(available_2019, key=lambda x: abs((x - year_2019_candidate).days))
                        calm_dates.append(closest_date)
                    else:
                        # Fallback to using actual target date
                        calm_dates.append(target_date)
                        logger.warning(f"No 2019 data available, using actual date {target_date}")
            except ValueError:
                # Handle invalid dates (e.g., Feb 29)
                closest_date = min(returns_data[returns_data.index.year == 2019].index,
                                 key=lambda x: abs((x.replace(year=2020) - target_date).days))
                calm_dates.append(closest_date)
        
        logger.info(f"Using calm dates from 2019: {calm_dates[:3]}... (showing first 3)")
        
        # Generate conditioning using calm dates
        checkpoint_dir = checkpoint_info['path']
        conditioning, warnings = generate_eval_conditioning(
            checkpoint_dir, returns_data, calm_dates
        )
        
        if warnings:
            self.manifest['warnings'].extend(warnings)
        
        return conditioning
    
    def generate_llm_knob_conditions(self, model_name: str, checkpoint_info: Dict[str, Any],
                                   returns_data: pd.DataFrame, target_dates: List[pd.Timestamp],
                                   component_idx: int = 0, shift_factor: float = 1.0) -> np.ndarray:
        """Generate LLM conditions with specific PCA component shifted."""
        logger.info(f"Generating LLM knob conditions (component {component_idx}, shift {shift_factor}σ)")
        
        if model_name != 'llm':
            raise ValueError("LLM knob mode only supported for LLM models")
        
        # First generate real conditions
        real_conditioning = self.generate_real_conditions(
            model_name, checkpoint_info, returns_data, target_dates
        )
        
        if real_conditioning is None:
            return None
        
        # Load the PCA model to get component statistics
        checkpoint_dir = checkpoint_info['path']
        pca_path = Path(checkpoint_dir) / 'pca_model.pkl'
        
        if not pca_path.exists():
            logger.warning(f"PCA model not found at {pca_path}, using standard deviation estimate")
            # Estimate component standard deviation from the conditioning
            component_std = real_conditioning[:, component_idx].std() if real_conditioning.shape[1] > component_idx else 1.0
        else:
            # Load PCA model to get proper component scaling
            import pickle
            try:
                with open(pca_path, 'rb') as f:
                    pca_model = pickle.load(f)
                # Use explained variance as a proxy for component importance
                component_std = np.sqrt(pca_model.explained_variance_[component_idx]) if component_idx < len(pca_model.explained_variance_) else 1.0
            except Exception as e:
                logger.warning(f"Failed to load PCA model: {e}, using standard deviation estimate")
                component_std = real_conditioning[:, component_idx].std() if real_conditioning.shape[1] > component_idx else 1.0
        
        # Create modified conditioning
        modified_conditioning = real_conditioning.copy()
        
        # Apply shift to the specified component
        if component_idx < modified_conditioning.shape[1]:
            shift_amount = shift_factor * component_std
            modified_conditioning[:, component_idx] += shift_amount
            logger.info(f"Shifted component {component_idx} by {shift_amount:.4f} ({shift_factor}σ)")
        else:
            logger.warning(f"Component {component_idx} not available in conditioning (dim={modified_conditioning.shape[1]})")
        
        return modified_conditioning
    
    def generate_samples_for_mode(self, model_name: str, checkpoint_info: Dict[str, Any],
                                 mode: str, returns_data: pd.DataFrame, 
                                 target_dates: List[pd.Timestamp], num_paths: int, 
                                 seq_len: int, seed: int, **mode_kwargs) -> Optional[np.ndarray]:
        """Generate samples for a specific mode."""
        logger.info(f"Generating samples for {model_name} × {mode}")
        
        # Set random seed
        np.random.seed(seed)
        
        # Generate conditioning based on mode
        try:
            if mode == 'real-conditions':
                conditioning = self.generate_real_conditions(
                    model_name, checkpoint_info, returns_data, target_dates
                )
            elif mode == 'calm-conditions':
                conditioning = self.generate_calm_conditions(
                    model_name, checkpoint_info, returns_data, target_dates
                )
            elif mode == 'llm-knob':
                if model_name != 'llm':
                    logger.warning(f"LLM knob mode not supported for {model_name} model")
                    return None
                
                component_idx = mode_kwargs.get('component_idx', 0)
                shift_factor = mode_kwargs.get('shift_factor', 1.0)
                conditioning = self.generate_llm_knob_conditions(
                    model_name, checkpoint_info, returns_data, target_dates,
                    component_idx, shift_factor
                )
            else:
                raise ValueError(f"Unknown mode: {mode}")
            
            # Generate samples using the checkpoint sampler
            checkpoint_dir = checkpoint_info['path']
            
            # Create a temporary conditioning provider with the modified conditioning
            # We'll do this by temporarily modifying the checkpoint sampler
            sampler = CheckpointSampler(checkpoint_dir)
            
            # Replace the conditioning generation with our pre-computed conditioning
            if conditioning is not None:
                # Override the conditioning provider's generate_conditioning method
                original_method = sampler.generator.conditioning_provider.generate_conditioning
                
                def custom_conditioning(dates, num_paths):
                    # Return our pre-computed conditioning, replicated for num_paths
                    if len(conditioning.shape) == 2 and conditioning.shape[0] == len(dates):
                        # Replicate conditioning for each path
                        return np.tile(conditioning, (num_paths // len(dates) + 1, 1))[:num_paths]
                    else:
                        return conditioning
                
                sampler.generator.conditioning_provider.generate_conditioning = custom_conditioning
            
            # Generate samples
            samples = sampler.generate_samples(target_dates, num_paths, seq_len=seq_len)
            
            # Restore original method
            if conditioning is not None:
                sampler.generator.conditioning_provider.generate_conditioning = original_method
            
            return samples
            
        except Exception as e:
            error_msg = f"Failed to generate samples for {model_name} × {mode}: {e}"
            logger.error(error_msg)
            self.manifest['errors'].append(error_msg)
            return None
    
    def calculate_risk_metrics(self, samples: np.ndarray) -> Dict[str, float]:
        """Calculate risk metrics from samples."""
        if samples is None or len(samples) == 0:
            return {}
        
        # Calculate path-level returns (sum over sequence)
        path_returns = samples.sum(axis=1)
        
        # Risk metrics
        metrics = {
            'mean_return': float(path_returns.mean()),
            'volatility': float(path_returns.std()),
            'var_1': float(np.percentile(path_returns, 1)),
            'var_5': float(np.percentile(path_returns, 5)),
            'var_10': float(np.percentile(path_returns, 10)),
            'es_1': float(path_returns[path_returns <= np.percentile(path_returns, 1)].mean()),
            'es_5': float(path_returns[path_returns <= np.percentile(path_returns, 5)].mean()),
            'tail_mass_neg2': float((path_returns <= -2 * path_returns.std()).mean()),
            'tail_mass_neg1': float((path_returns <= -1 * path_returns.std()).mean()),
            'skewness': float(((path_returns - path_returns.mean()) ** 3).mean() / (path_returns.std() ** 3)),
            'kurtosis': float(((path_returns - path_returns.mean()) ** 4).mean() / (path_returns.std() ** 4) - 3)
        }
        
        return metrics
    
    def run_experiment(self, window_id: str, window_info: Dict[str, Any], 
                      checkpoints: Dict[str, Dict[str, Any]], returns_data: pd.DataFrame,
                      num_paths: int, seq_len: int, seeds: List[int]) -> Dict[str, Any]:
        """Run the complete Experiment B."""
        
        logger.info("Starting Experiment B execution")
        self.manifest['status'] = 'running'
        self.manifest['window'] = {window_id: window_info}
        
        # Generate target dates for the window
        start_date = pd.Timestamp(window_info['start'])
        end_date = pd.Timestamp(window_info['end'])
        target_dates = pd.date_range(start_date, end_date, freq='W').tolist()
        if end_date not in target_dates:
            target_dates.append(end_date)
        
        # Create execution plan
        plan = self._create_execution_plan(window_id, checkpoints, num_paths, seq_len, seeds)
        
        # Save plan
        plan_file = self.experiment_dir / 'plan.json'
        with open(plan_file, 'w') as f:
            json.dump(plan, f, indent=2)
        logger.info(f"Saved execution plan: {plan_file}")
        
        # Execute experiments
        results = {}
        risk_analysis = {}
        
        for model_name, model_info in checkpoints.items():
            checkpoint_info = model_info['primary_checkpoint']
            results[model_name] = {}
            risk_analysis[model_name] = {}
            
            # Base modes (all models)
            base_modes = ['real-conditions', 'calm-conditions']
            
            for mode in base_modes:
                # Create output directory
                output_dir = self.experiment_dir / window_id / model_name / mode
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Generate samples
                main_seed = seeds[0] if seeds else 42
                samples = self.generate_samples_for_mode(
                    model_name, checkpoint_info, mode, returns_data,
                    target_dates, num_paths, seq_len, main_seed
                )
                
                if samples is not None:
                    # Save samples
                    samples_file = output_dir / 'samples.npy'
                    np.save(samples_file, samples)
                    
                    # Calculate risk metrics
                    risk_metrics = self.calculate_risk_metrics(samples)
                    
                    # Save results
                    results[model_name][mode] = {
                        'status': 'success',
                        'samples_shape': list(samples.shape),
                        'output_path': str(samples_file),
                        'risk_metrics': risk_metrics
                    }
                    risk_analysis[model_name][mode] = risk_metrics
                    
                    logger.info(f"Completed {model_name} × {mode}: {samples.shape}")
                else:
                    results[model_name][mode] = {'status': 'failed'}
            
            # LLM knob mode (LLM models only)
            if model_name == 'llm':
                knob_results = {}
                knob_risk = {}
                
                # Test different shift factors
                shift_factors = [-2.0, -1.0, 1.0, 2.0]
                component_idx = 0  # Test first PCA component
                
                for shift_factor in shift_factors:
                    mode_name = f'llm-knob-comp{component_idx}-shift{shift_factor:+.1f}sigma'
                    
                    # Create output directory
                    output_dir = self.experiment_dir / window_id / model_name / mode_name
                    output_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Generate samples
                    main_seed = seeds[0] if seeds else 42
                    samples = self.generate_samples_for_mode(
                        model_name, checkpoint_info, 'llm-knob', returns_data,
                        target_dates, num_paths, seq_len, main_seed,
                        component_idx=component_idx, shift_factor=shift_factor
                    )
                    
                    if samples is not None:
                        # Save samples
                        samples_file = output_dir / 'samples.npy'
                        np.save(samples_file, samples)
                        
                        # Calculate risk metrics
                        risk_metrics = self.calculate_risk_metrics(samples)
                        
                        knob_results[mode_name] = {
                            'status': 'success',
                            'samples_shape': list(samples.shape),
                            'output_path': str(samples_file),
                            'risk_metrics': risk_metrics,
                            'component_idx': component_idx,
                            'shift_factor': shift_factor
                        }
                        knob_risk[mode_name] = risk_metrics
                        
                        logger.info(f"Completed {model_name} × {mode_name}: {samples.shape}")
                    else:
                        knob_results[mode_name] = {'status': 'failed'}
                
                results[model_name].update(knob_results)
                risk_analysis[model_name].update(knob_risk)
        
        # Generate comparative analysis
        comparative_analysis = self._generate_comparative_analysis(risk_analysis)
        
        # Finalize manifest
        self.manifest['results'] = results
        self.manifest['risk_analysis'] = risk_analysis
        self.manifest['comparative_analysis'] = comparative_analysis
        self.manifest['completed_at'] = datetime.now().isoformat()
        self.manifest['status'] = 'completed' if not self.manifest['errors'] else 'completed_with_errors'
        
        # Save final manifest
        manifest_file = self.experiment_dir / 'manifest.json'
        with open(manifest_file, 'w') as f:
            json.dump(self.manifest, f, indent=2)
        
        # Save comparative analysis separately
        analysis_file = self.experiment_dir / 'comparative_analysis.json'
        with open(analysis_file, 'w') as f:
            json.dump(comparative_analysis, f, indent=2)
        
        logger.info(f"Experiment B completed: {manifest_file}")
        
        return self.manifest
    
    def _create_execution_plan(self, window_id: str, checkpoints: Dict[str, Any],
                              num_paths: int, seq_len: int, seeds: List[int]) -> Dict[str, Any]:
        """Create detailed execution plan."""
        
        plan = {
            'experiment': 'B',
            'description': 'Counterfactual controllability testing',
            'created_at': datetime.now().isoformat(),
            'parameters': {
                'window_id': window_id,
                'num_paths': num_paths,
                'seq_len': seq_len,
                'seeds': seeds
            },
            'models': list(checkpoints.keys()),
            'modes': {
                'base': ['real-conditions', 'calm-conditions'],
                'llm-specific': ['llm-knob (various shifts)']
            },
            'estimated_outputs': []
        }
        
        # Estimate outputs
        for model_name in checkpoints:
            # Base modes
            for mode in ['real-conditions', 'calm-conditions']:
                output_path = f"results/addons/period_slices/B/{window_id}/{model_name}/{mode}/samples.npy"
                plan['estimated_outputs'].append(output_path)
            
            # LLM knob modes
            if model_name == 'llm':
                for shift in [-2.0, -1.0, 1.0, 2.0]:
                    mode_name = f'llm-knob-comp0-shift{shift:+.1f}sigma'
                    output_path = f"results/addons/period_slices/B/{window_id}/{model_name}/{mode_name}/samples.npy"
                    plan['estimated_outputs'].append(output_path)
        
        return plan
    
    def _generate_comparative_analysis(self, risk_analysis: Dict[str, Dict[str, Dict[str, float]]]) -> Dict[str, Any]:
        """Generate comparative analysis across modes and models."""
        
        analysis = {
            'summary': 'Counterfactual controllability analysis',
            'generated_at': datetime.now().isoformat(),
            'model_comparisons': {},
            'mode_effects': {},
            'controllability_assessment': {}
        }
        
        for model_name, model_results in risk_analysis.items():
            if len(model_results) < 2:
                continue
            
            model_analysis = {}
            
            # Compare real vs calm conditions
            if 'real-conditions' in model_results and 'calm-conditions' in model_results:
                real_metrics = model_results['real-conditions']
                calm_metrics = model_results['calm-conditions']
                
                mode_comparison = {}
                for metric in ['var_5', 'es_5', 'volatility', 'tail_mass_neg1']:
                    if metric in real_metrics and metric in calm_metrics:
                        real_val = real_metrics[metric]
                        calm_val = calm_metrics[metric]
                        pct_change = ((real_val - calm_val) / abs(calm_val)) * 100 if calm_val != 0 else 0
                        
                        mode_comparison[metric] = {
                            'real': real_val,
                            'calm': calm_val,
                            'difference': real_val - calm_val,
                            'percent_change': pct_change
                        }
                
                model_analysis['real_vs_calm'] = mode_comparison
            
            # Analyze LLM knob effects
            llm_knob_modes = [k for k in model_results.keys() if k.startswith('llm-knob')]
            if llm_knob_modes and 'real-conditions' in model_results:
                baseline = model_results['real-conditions']
                knob_analysis = {}
                
                for knob_mode in sorted(llm_knob_modes):
                    knob_metrics = model_results[knob_mode]
                    shift_comparison = {}
                    
                    for metric in ['var_5', 'es_5', 'volatility']:
                        if metric in baseline and metric in knob_metrics:
                            baseline_val = baseline[metric]
                            knob_val = knob_metrics[metric]
                            pct_change = ((knob_val - baseline_val) / abs(baseline_val)) * 100 if baseline_val != 0 else 0
                            
                            shift_comparison[metric] = {
                                'baseline': baseline_val,
                                'shifted': knob_val,
                                'difference': knob_val - baseline_val,
                                'percent_change': pct_change
                            }
                    
                    knob_analysis[knob_mode] = shift_comparison
                
                model_analysis['llm_knob_effects'] = knob_analysis
            
            analysis['model_comparisons'][model_name] = model_analysis
        
        # Overall controllability assessment
        controllability_scores = {}
        for model_name, model_analysis in analysis['model_comparisons'].items():
            score = 0
            
            # Score based on real vs calm sensitivity
            if 'real_vs_calm' in model_analysis:
                real_calm = model_analysis['real_vs_calm']
                for metric, comparison in real_calm.items():
                    if abs(comparison['percent_change']) > 5:  # 5% threshold
                        score += 1
            
            # Score based on LLM knob controllability
            if 'llm_knob_effects' in model_analysis:
                knob_effects = model_analysis['llm_knob_effects']
                significant_effects = 0
                for knob_mode, effects in knob_effects.items():
                    for metric, comparison in effects.items():
                        if abs(comparison['percent_change']) > 10:  # 10% threshold
                            significant_effects += 1
                score += min(significant_effects, 5)  # Cap contribution
            
            controllability_scores[model_name] = score
        
        analysis['controllability_assessment'] = {
            'scores': controllability_scores,
            'interpretation': 'Higher scores indicate greater controllability through conditioning manipulation'
        }
        
        return analysis

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Experiment B: Counterfactual Controllability Testing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python experiment_B_evaluator.py --window covid_crash --csv-file sp500_data.csv --num-paths 1000
  
  # Custom settings
  python experiment_B_evaluator.py \\
    --window covid_recovery \\
    --csv-file data/returns.csv \\
    --num-paths 2000 \\
    --seq-len 60 \\
    --seeds 42 123
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
        evaluator = ExperimentBEvaluator(
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
        
        # Define window information
        windows = ExperimentAEvaluator.STRESS_WINDOWS
        if args.window not in windows:
            logger.error(f"Unknown window: {args.window}. Available: {list(windows.keys())}")
            return
        
        window_info = windows[args.window]
        
        # Run experiment
        manifest = evaluator.run_experiment(
            window_id=args.window,
            window_info=window_info,
            checkpoints=checkpoints,
            returns_data=returns_data,
            num_paths=args.num_paths,
            seq_len=args.seq_len,
            seeds=args.seeds
        )
        
        # Print summary
        print("\n" + "="*60)
        print("EXPERIMENT B SUMMARY")
        print("="*60)
        print(f"Experiment Directory: {evaluator.experiment_dir}")
        print(f"Status: {manifest['status']}")
        print(f"Window Tested: {args.window}")
        print(f"Models Tested: {len(manifest['models'])}")
        print(f"Errors: {len(manifest['errors'])}")
        print(f"Warnings: {len(manifest['warnings'])}")
        
        # Print controllability scores
        if 'comparative_analysis' in manifest and 'controllability_assessment' in manifest['comparative_analysis']:
            scores = manifest['comparative_analysis']['controllability_assessment']['scores']
            print(f"\nControllability Scores:")
            for model, score in scores.items():
                print(f"  {model}: {score}")
        
        # Show key risk metric changes
        if 'comparative_analysis' in manifest and 'model_comparisons' in manifest['comparative_analysis']:
            print(f"\nKey Risk Metric Changes (Real vs Calm):")
            for model, analysis in manifest['comparative_analysis']['model_comparisons'].items():
                if 'real_vs_calm' in analysis:
                    real_calm = analysis['real_vs_calm']
                    if 'var_5' in real_calm:
                        var_change = real_calm['var_5']['percent_change']
                        print(f"  {model} VaR(5%) change: {var_change:+.1f}%")
        
        print(f"\nResults saved to: {evaluator.experiment_dir}")
        print("Check comparative_analysis.json for detailed analysis.")
        
    except Exception as e:
        logger.error(f"Experiment B failed: {e}")
        raise

if __name__ == "__main__":
    main()
