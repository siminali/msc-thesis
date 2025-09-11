#!/usr/bin/env python3
"""
Hygiene Integration Examples

Demonstrates how to integrate hygiene checks into existing trainers and evaluators.
Shows practical patterns for adding hygiene validation without disrupting workflows.

Features:
- Training script integration
- Evaluator integration  
- Checkpoint validation
- Causality checking during feature engineering
- Reproducible environment setup

Author: Simin Ali
Thesis: Diffusion Models in Generative AI for Financial Data Synthesis and Risk Management
"""

import logging
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

from hygiene_checks import (
    setup_reproducible_environment, 
    quick_hygiene_check,
    check_feature_causality,
    HygieneChecker
)

logger = logging.getLogger(__name__)

class TrainerWithHygiene:
    """Example trainer class with integrated hygiene checks."""
    
    def __init__(self, model_type: str, checkpoint_dir: Path, seed: int = 42):
        self.model_type = model_type
        self.checkpoint_dir = checkpoint_dir
        self.seed = seed
        self.hygiene_results = None
        
        # Setup reproducible environment immediately
        logger.info("🧹 Setting up reproducible training environment")
        env_info = setup_reproducible_environment(seed=seed)
        
        if env_info.get('torch_deterministic', False):
            logger.info(f"✅ Deterministic training environment ready (seed: {seed})")
        else:
            logger.warning(f"⚠️ Deterministic setup had issues - check logs")
        
        self.env_info = env_info
    
    def validate_checkpoint_after_training(self) -> Dict[str, Any]:
        """Validate checkpoint after training completion."""
        logger.info("🧹 Validating checkpoint after training")
        
        status, results = quick_hygiene_check(
            checkpoint_path=self.checkpoint_dir,
            model_type=self.model_type
        )
        
        self.hygiene_results = results
        
        if status == 'suspect':
            logger.warning(f"Checkpoint validation failed: {results['summary']['total_issues']} issues found")
            
            # Log specific issues for debugging
            for category, issues in results['summary']['details'].items():
                if issues:
                    logger.warning(f"{category} issues:")
                    for issue in issues:
                        logger.warning(f"  - {issue}")
        else:
            logger.info("✅ Checkpoint validation passed - no hygiene issues")
        
        # Save hygiene results with checkpoint
        hygiene_file = self.checkpoint_dir / 'hygiene_report.json'
        with open(hygiene_file, 'w') as f:
            import json
            json.dump(results, f, indent=2)
        
        logger.info(f"Hygiene report saved: {hygiene_file}")
        return results
    
    def train_model(self, returns_data: pd.DataFrame) -> Dict[str, Any]:
        """Example training method with hygiene integration."""
        logger.info(f"🚀 Starting {self.model_type} model training")
        
        # Your actual training code would go here...
        # For demonstration, we'll just create a mock checkpoint
        
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Mock training metadata
        training_metadata = {
            'model_type': self.model_type,
            'training_completed': datetime.now().isoformat(),
            'seed_used': self.seed,
            'data_shape': returns_data.shape,
            'environment_info': self.env_info
        }
        
        # Save mock checkpoint files
        import json
        with open(self.checkpoint_dir / 'meta.json', 'w') as f:
            json.dump(training_metadata, f, indent=2)
        
        # Mock conditioning spec (incomplete to demonstrate hygiene detection)
        conditioning_spec = {
            'vol_window': 20,
            'trend_window': 60,
            'vol_threshold': 0.0076
            # Intentionally missing required fields to trigger hygiene warnings
        }
        
        with open(self.checkpoint_dir / 'conditioning_spec.json', 'w') as f:
            json.dump(conditioning_spec, f, indent=2)
        
        logger.info("Training completed - validating checkpoint")
        
        # Validate checkpoint after training
        hygiene_results = self.validate_checkpoint_after_training()
        
        return {
            'status': 'completed',
            'checkpoint_dir': str(self.checkpoint_dir),
            'hygiene_status': hygiene_results['overall_status'],
            'hygiene_issues': hygiene_results['summary']['total_issues']
        }

class EvaluatorWithHygiene:
    """Example evaluator class with integrated hygiene checks."""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.hygiene_checker = HygieneChecker()
        
        # Setup reproducible evaluation environment
        logger.info("🧹 Setting up reproducible evaluation environment")
        env_info = setup_reproducible_environment(seed=seed)
        self.env_info = env_info
    
    def load_checkpoint_safely(self, checkpoint_path: Path, model_type: str) -> Dict[str, Any]:
        """Load checkpoint with hygiene validation."""
        logger.info(f"🧹 Validating checkpoint before loading: {checkpoint_path}")
        
        status, results = quick_hygiene_check(
            checkpoint_path=checkpoint_path,
            model_type=model_type
        )
        
        if status == 'suspect':
            logger.warning(f"Loading suspect checkpoint: {results['summary']['total_issues']} issues")
            
            # Log specific issues but continue
            for category, issues in results['summary']['details'].items():
                if issues:
                    for issue in issues:
                        logger.warning(f"{category}: {issue}")
        else:
            logger.info("✅ Checkpoint validation passed")
        
        # Your actual checkpoint loading code would go here...
        # For demonstration, return mock loaded model
        return {
            'model': f"MockModel_{model_type}",
            'hygiene_status': status,
            'hygiene_results': results
        }
    
    def validate_feature_causality(self, returns_data: pd.DataFrame, 
                                 target_dates: List[pd.Timestamp], 
                                 model_type: str) -> bool:
        """Validate feature causality during evaluation."""
        
        if model_type != 'explicit':
            logger.debug(f"Causality check not applicable for {model_type}")
            return True
        
        logger.info(f"🧹 Checking feature causality for {len(target_dates)} dates")
        
        is_causal = check_feature_causality(
            returns_data=returns_data,
            target_dates=target_dates,
            model_type=model_type,
            vol_window=20,
            trend_window=60
        )
        
        if not is_causal:
            logger.warning("⚠️ Causality violations detected - features may use future data")
        else:
            logger.info("✅ All features pass causality check")
        
        return is_causal
    
    def run_evaluation_with_hygiene(self, checkpoint_path: Path, model_type: str,
                                   returns_data: pd.DataFrame, 
                                   target_dates: List[pd.Timestamp]) -> Dict[str, Any]:
        """Run evaluation with comprehensive hygiene checking."""
        
        logger.info(f"🚀 Starting evaluation with hygiene checks")
        
        # 1. Load checkpoint with validation
        loaded_model = self.load_checkpoint_safely(checkpoint_path, model_type)
        
        # 2. Check feature causality
        causality_ok = self.validate_feature_causality(returns_data, target_dates, model_type)
        
        # 3. Run comprehensive hygiene check
        comprehensive_results = self.hygiene_checker.run_all_checks(
            checkpoint_path=checkpoint_path,
            returns_data=returns_data,
            target_dates=target_dates,
            model_type=model_type,
            setup_determinism=False  # Already done
        )
        
        # 4. Your actual evaluation code would go here...
        # For demonstration, mock evaluation results
        
        evaluation_results = {
            'status': 'completed',
            'model_type': model_type,
            'checkpoint_path': str(checkpoint_path),
            'target_dates_count': len(target_dates),
            'causality_validated': causality_ok,
            'hygiene_status': comprehensive_results['overall_status'],
            'hygiene_summary': comprehensive_results['summary'],
            'samples_generated': True  # Mock
        }
        
        # Log final status
        if comprehensive_results['overall_status'] == 'suspect':
            logger.warning(f"Evaluation completed with hygiene issues: {comprehensive_results['summary']['total_issues']}")
        else:
            logger.info("✅ Evaluation completed with clean hygiene status")
        
        return evaluation_results

class ExperimentPipelineWithHygiene:
    """Example experiment pipeline with integrated hygiene checks."""
    
    def __init__(self, experiment_name: str, seed: int = 42):
        self.experiment_name = experiment_name
        self.seed = seed
        self.hygiene_checker = HygieneChecker()
        
        # Setup environment
        env_info = setup_reproducible_environment(seed=seed)
        self.env_info = env_info
        
        # Track hygiene results across pipeline
        self.pipeline_hygiene = {
            'checkpoints': {},
            'windows': {},
            'overall_issues': 0
        }
    
    def validate_checkpoint_for_experiment(self, checkpoint_path: Path, 
                                         model_type: str) -> Dict[str, Any]:
        """Validate checkpoint for experimental use."""
        
        logger.info(f"🧹 Validating {model_type} checkpoint for {self.experiment_name}")
        
        # Check if this is supposed to be a pre-COVID checkpoint for Experiment A
        check_precovid = (self.experiment_name.upper() == 'A' and 'precovid' in str(checkpoint_path))
        
        results = self.hygiene_checker.run_all_checks(
            checkpoint_path=checkpoint_path,
            model_type=model_type,
            check_precovid=check_precovid,
            setup_determinism=False
        )
        
        # Store results
        checkpoint_key = f"{model_type}_{checkpoint_path.name}"
        self.pipeline_hygiene['checkpoints'][checkpoint_key] = results
        self.pipeline_hygiene['overall_issues'] += results['summary']['total_issues']
        
        if results['overall_status'] == 'suspect':
            logger.warning(f"Checkpoint {checkpoint_key} has hygiene issues - marking as suspect")
        
        return results
    
    def validate_window_causality(self, window_id: str, returns_data: pd.DataFrame,
                                target_dates: List[pd.Timestamp]) -> bool:
        """Validate causality for a specific window."""
        
        logger.info(f"🧹 Validating causality for window: {window_id}")
        
        # Check explicit features causality
        is_causal = check_feature_causality(
            returns_data=returns_data,
            target_dates=target_dates,
            model_type='explicit'  # Most restrictive check
        )
        
        self.pipeline_hygiene['windows'][window_id] = {
            'causality_validated': is_causal,
            'target_dates_count': len(target_dates)
        }
        
        if not is_causal:
            logger.warning(f"Window {window_id} has causality issues")
            self.pipeline_hygiene['overall_issues'] += 1
        
        return is_causal
    
    def run_experiment_pipeline(self, checkpoints: Dict[str, Path], 
                              windows: Dict[str, Dict], 
                              returns_data: pd.DataFrame) -> Dict[str, Any]:
        """Run complete experiment pipeline with hygiene integration."""
        
        logger.info(f"🚀 Starting {self.experiment_name} pipeline with hygiene checks")
        
        # 1. Validate all checkpoints
        for model_type, checkpoint_path in checkpoints.items():
            self.validate_checkpoint_for_experiment(checkpoint_path, model_type)
        
        # 2. Validate all windows
        for window_id, window_info in windows.items():
            target_dates = self.get_window_target_dates(window_info)
            self.validate_window_causality(window_id, returns_data, target_dates)
        
        # 3. Your actual experiment execution would go here...
        # Mock experiment results
        experiment_results = {
            'experiment': self.experiment_name,
            'status': 'completed',
            'checkpoints_processed': len(checkpoints),
            'windows_processed': len(windows),
            'hygiene_summary': self.get_pipeline_hygiene_summary()
        }
        
        # Log final hygiene status
        total_issues = self.pipeline_hygiene['overall_issues']
        if total_issues > 0:
            logger.warning(f"Pipeline completed with {total_issues} hygiene issues")
        else:
            logger.info("✅ Pipeline completed with clean hygiene status")
        
        return experiment_results
    
    def get_window_target_dates(self, window_info: Dict) -> List[pd.Timestamp]:
        """Mock method to get target dates for a window."""
        # In real implementation, this would extract dates from window_info
        return [
            pd.Timestamp('2020-03-15'),
            pd.Timestamp('2020-03-22'),
            pd.Timestamp('2020-03-29')
        ]
    
    def get_pipeline_hygiene_summary(self) -> Dict[str, Any]:
        """Get summary of hygiene results across the pipeline."""
        return {
            'total_issues': self.pipeline_hygiene['overall_issues'],
            'checkpoint_issues': sum(
                result['summary']['total_issues'] 
                for result in self.pipeline_hygiene['checkpoints'].values()
            ),
            'window_issues': sum(
                0 if result['causality_validated'] else 1
                for result in self.pipeline_hygiene['windows'].values()  
            ),
            'checkpoints_validated': len(self.pipeline_hygiene['checkpoints']),
            'windows_validated': len(self.pipeline_hygiene['windows']),
            'overall_status': 'clean' if self.pipeline_hygiene['overall_issues'] == 0 else 'suspect'
        }

def example_trainer_integration():
    """Example of integrating hygiene checks into a trainer."""
    print("=== Trainer Integration Example ===")
    
    # Mock returns data
    returns_data = pd.DataFrame({
        'returns': [0.01, -0.02, 0.005, -0.01, 0.03]
    }, index=pd.date_range('2020-01-01', periods=5))
    
    # Create trainer with hygiene
    trainer = TrainerWithHygiene(
        model_type='explicit',
        checkpoint_dir=Path('example_checkpoints/explicit_test'),
        seed=42
    )
    
    # Train model
    results = trainer.train_model(returns_data)
    print(f"Training completed: {results['status']}")
    print(f"Hygiene status: {results['hygiene_status']}")
    print(f"Issues found: {results['hygiene_issues']}")

def example_evaluator_integration():
    """Example of integrating hygiene checks into an evaluator."""
    print("\n=== Evaluator Integration Example ===")
    
    # Mock data
    returns_data = pd.DataFrame({
        'returns': [0.01, -0.02, 0.005, -0.01, 0.03]
    }, index=pd.date_range('2020-01-01', periods=5))
    
    target_dates = [pd.Timestamp('2020-01-03'), pd.Timestamp('2020-01-04')]
    
    # Create evaluator with hygiene
    evaluator = EvaluatorWithHygiene(seed=123)
    
    # Run evaluation
    results = evaluator.run_evaluation_with_hygiene(
        checkpoint_path=Path('example_checkpoints/explicit_test'),
        model_type='explicit',
        returns_data=returns_data,
        target_dates=target_dates
    )
    
    print(f"Evaluation completed: {results['status']}")
    print(f"Hygiene status: {results['hygiene_status']}")
    print(f"Causality validated: {results['causality_validated']}")

def example_pipeline_integration():
    """Example of integrating hygiene checks into an experiment pipeline."""
    print("\n=== Pipeline Integration Example ===")
    
    # Mock returns data
    returns_data = pd.DataFrame({
        'returns': [0.01, -0.02, 0.005, -0.01, 0.03]
    }, index=pd.date_range('2020-01-01', periods=5))
    
    # Mock checkpoints and windows
    checkpoints = {
        'explicit': Path('checkpoints/precovid/explicit/20100101-20191231'),
        'llm': Path('checkpoints/precovid/llm/20100101-20191231')
    }
    
    windows = {
        'covid_crash': {'start': '2020-03-15', 'end': '2020-03-29'}
    }
    
    # Create pipeline with hygiene
    pipeline = ExperimentPipelineWithHygiene(experiment_name='A', seed=456)
    
    # Run pipeline
    results = pipeline.run_experiment_pipeline(checkpoints, windows, returns_data)
    
    print(f"Pipeline completed: {results['status']}")
    print(f"Hygiene summary: {results['hygiene_summary']}")

if __name__ == "__main__":
    # Run examples
    example_trainer_integration()
    example_evaluator_integration() 
    example_pipeline_integration()
