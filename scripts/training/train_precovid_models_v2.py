#!/usr/bin/env python3
"""
Pre-COVID Training Runner V2 - Enhanced Version

This is a versioning-safe alternative that extends the original training runner
with additional features and improvements.

Author: Simin Ali
Thesis: Diffusion Models in Generative AI for Financial Data Synthesis and Risk Management
"""

# Import the main training functionality from the original script
import sys
import os

# Add the current directory to path to import the original script
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    # Import all functions from the original script
    from train_precovid_models import *
    
    # Additional imports for enhanced functionality
    import matplotlib.pyplot as plt
    import seaborn as sns
    from pathlib import Path
    
    def create_training_report(args, trained_models, metadata):
        """Create a comprehensive training report."""
        logger.info("Creating training report...")
        
        report_dir = os.path.join(args.checkpoint_dir, 'training_report')
        os.makedirs(report_dir, exist_ok=True)
        
        # Create summary report
        report_content = f"""# Pre-COVID Training Report
        
## Training Summary
- **Training Period**: 2010-01-01 to 2019-12-31
- **Validation Period**: 2019-07-01 to 2019-12-31
- **Models Trained**: {list(trained_models.keys())}
- **Total Parameters**: {sum([sum(p.numel() for p in model[0].parameters()) for model in trained_models.values()])}
- **Training Time**: {datetime.now().isoformat()}

## Data Information
- **Training Sequences**: {metadata['data_info']['train_sequences']}
- **Validation Sequences**: {metadata['data_info']['val_sequences']}
- **Sequence Length**: {metadata['data_info']['sequence_length']}

## System Information
- **Device**: {metadata['system_info']['device']}
- **PyTorch Version**: {metadata['system_info']['torch_version']}
- **CUDA Available**: {metadata['system_info']['cuda_available']}
- **Random Seed**: {metadata['system_info']['seed']}

## Model Details
"""
        
        for model_name, (model, trainer) in trained_models.items():
            param_count = sum(p.numel() for p in model.parameters())
            report_content += f"""
### {model_name.upper()} Model
- **Parameters**: {param_count:,}
- **Architecture**: {type(model).__name__}
- **Checkpoint Directory**: checkpoints/precovid/{model_name}/20100101-20191231/
"""
        
        # Save report
        with open(os.path.join(report_dir, 'training_summary.md'), 'w') as f:
            f.write(report_content)
        
        logger.info(f"Training report saved to: {report_dir}")
    
    def enhanced_main():
        """Enhanced main function with additional features."""
        print("=" * 80)
        print("Pre-COVID Training Runner V2 - Enhanced Version")
        print("=" * 80)
        
        # Parse arguments with V2 specific options
        parser = argparse.ArgumentParser(
            description='Enhanced Pre-COVID Training Runner for Financial Models'
        )
        
        # Import all arguments from original parser
        args = parse_arguments()
        
        # Add V2 specific arguments
        parser.add_argument('--create-report', action='store_true', default=True,
                           help='Create comprehensive training report')
        parser.add_argument('--plot-training', action='store_true', default=True,
                           help='Create training loss plots')
        
        # Re-parse with new arguments
        args = parser.parse_args()
        
        # Run the original main function
        set_deterministic_mode(args.seed)
        
        device = torch.device(args.device)
        logger.info(f"Enhanced V2 training starting on {device}")
        
        try:
            # Load and prepare data (using original functions)
            train_data, val_data = load_and_prepare_data()
            X_train, train_indices = create_sequences(train_data)
            X_val, val_indices = create_sequences(val_data)
            
            # Prepare metadata
            metadata = {
                'system_info': {
                    'device': str(device),
                    'torch_version': torch.__version__,
                    'cuda_available': torch.cuda.is_available(),
                    'python_version': sys.version,
                    'timestamp': datetime.now().isoformat(),
                    'seed': args.seed,
                    'version': 'v2_enhanced'
                },
                'data_info': {
                    'train_period': f"{train_data.index[0]} to {train_data.index[-1]}",
                    'val_period': f"{val_data.index[0]} to {val_data.index[-1]}",
                    'train_sequences': len(X_train),
                    'val_sequences': len(X_val),
                    'sequence_length': SEQ_LEN,
                    'train_stats': {
                        'mean': float(train_data.mean()),
                        'std': float(train_data.std()),
                        'min': float(train_data.min()),
                        'max': float(train_data.max())
                    }
                }
            }
            
            # Create checkpoint directory
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            
            # Train models (reusing original training functions)
            trained_models = {}
            
            for model_type in args.models:
                logger.info(f"\n{'='*60}")
                logger.info(f"Training {model_type.upper()} model (V2 Enhanced)")
                logger.info(f"{'='*60}")
                
                try:
                    if model_type == 'zero':
                        model, trainer = train_zero_model(X_train, X_val, args, metadata)
                        trained_models['zero'] = (model, trainer)
                        
                    elif model_type == 'explicit':
                        conditioning_train, conditioning_spec = create_explicit_conditioning(
                            X_train, train_indices, args.vol_window, args.trend_window
                        )
                        conditioning_val, _ = create_explicit_conditioning(
                            X_val, val_indices, args.vol_window, args.trend_window
                        )
                        
                        model, trainer = train_explicit_model(
                            X_train, X_val, conditioning_train, conditioning_val,
                            conditioning_spec, args, metadata
                        )
                        trained_models['explicit'] = (model, trainer)
                        
                    elif model_type == 'llm':
                        try:
                            all_indices = train_indices + val_indices
                            conditioning_all, conditioning_spec, pca = create_llm_conditioning(
                                all_indices, SEQ_LEN, args.pca_components, args.device, args.llm_fallback
                            )
                            
                            conditioning_train = conditioning_all[:len(X_train)]
                            conditioning_val = conditioning_all[len(X_train):]
                            
                            # Save PCA model
                            pca_path = os.path.join(args.checkpoint_dir, 'llm', '20100101-20191231')
                            os.makedirs(pca_path, exist_ok=True)
                            with open(os.path.join(pca_path, 'pca_model.pkl'), 'wb') as f:
                                pickle.dump(pca, f)
                            
                            model, trainer = train_llm_model(
                                X_train, X_val, conditioning_train, conditioning_val,
                                conditioning_spec, args, metadata
                            )
                            trained_models['llm'] = (model, trainer)
                            
                        except Exception as e:
                            if args.llm_fallback:
                                logger.warning(f"LLM conditioning failed, falling back to zero: {e}")
                                model, trainer = train_zero_model(X_train, X_val, args, metadata)
                                trained_models['llm_fallback'] = (model, trainer)
                            else:
                                raise
                
                except Exception as e:
                    error_msg = f"Failed to train {model_type} model: {e}"
                    logger.error(error_msg)
                    
                    if args.skip_on_error:
                        logger.warning(f"Skipping {model_type} model due to error")
                        continue
                    else:
                        raise RuntimeError(error_msg)
            
            # Create enhanced reports if requested
            if hasattr(args, 'create_report') and args.create_report:
                create_training_report(args, trained_models, metadata)
            
            # Summary
            logger.info(f"\n{'='*60}")
            logger.info("V2 ENHANCED TRAINING SUMMARY")
            logger.info(f"{'='*60}")
            logger.info(f"Successfully trained {len(trained_models)} models:")
            for model_type in trained_models:
                param_count = sum(p.numel() for p in trained_models[model_type][0].parameters())
                logger.info(f"  ✓ {model_type} ({param_count:,} parameters)")
            
            logger.info(f"\nCheckpoints saved to: {args.checkpoint_dir}")
            logger.info("V2 Enhanced Pre-COVID training completed successfully!")
            
            return trained_models, metadata
            
        except Exception as e:
            logger.error(f"V2 Enhanced training failed: {e}")
            raise
    
    if __name__ == "__main__":
        enhanced_main()

except ImportError as e:
    print(f"Error importing from train_precovid_models.py: {e}")
    print("Please ensure the original training script is in the same directory.")
    sys.exit(1)
