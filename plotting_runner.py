#!/usr/bin/env python3
"""
Plotting Runner

CLI tool for generating publication-quality plots from experiment results.
Creates ECDF overlays, Q-Q plots, VaR/ES analysis, and realized volatility tracking.

Author: Simin Ali
Thesis: Diffusion Models in Generative AI for Financial Data Synthesis and Risk Management
"""

import argparse
import sys
from pathlib import Path
import logging
from typing import Optional, List

# Import our plotting module
try:
    from shared_plotting import PlottingPipeline, PlotGenerator, ColorPalette
except ImportError as e:
    print(f"Error importing shared_plotting: {e}")
    sys.exit(1)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Plotting Runner for Experiment Visualization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate plots for all experiments and windows
  python plotting_runner.py
  
  # Generate plots for specific experiments
  python plotting_runner.py --experiments A B
  
  # Generate plots for specific windows
  python plotting_runner.py --windows covid_crash covid_recovery
  
  # Custom results directory
  python plotting_runner.py --results-base my_results
  
  # List available experiments and windows
  python plotting_runner.py --list-available
        """
    )
    
    parser.add_argument('--results-base', type=str, default='results/addons/period_slices',
                       help='Base directory containing experiment results')
    
    parser.add_argument('--experiments', nargs='*',
                       help='Specific experiments to process (e.g., A B A_v2)')
    
    parser.add_argument('--windows', nargs='*',
                       help='Specific windows to process (e.g., covid_crash covid_recovery)')
    
    parser.add_argument('--list-available', action='store_true',
                       help='List available experiments and windows, then exit')
    
    parser.add_argument('--output-formats', nargs='*', default=['pdf', 'png'],
                       choices=['pdf', 'png', 'svg'],
                       help='Output formats for plots')
    
    return parser.parse_args()

def main():
    """Main execution function."""
    args = parse_arguments()
    
    try:
        # Initialize pipeline
        pipeline = PlottingPipeline(args.results_base)
        
        # List available if requested
        if args.list_available:
            experiments = pipeline.discover_experiments()
            print("Available Experiments and Windows with Metrics:")
            print("=" * 50)
            
            if not experiments:
                print("No experiments with metrics found.")
                print("Make sure to run metrics_runner.py first.")
                return
                
            for exp_name, windows in experiments.items():
                print(f"{exp_name}:")
                for window in windows:
                    print(f"  - {window}")
                print()
            return
        
        # Run plotting pipeline
        manifest = pipeline.run_all_plots(
            experiments=args.experiments,
            windows=args.windows
        )
        
        # Print summary
        print("\n" + "="*60)
        print("PLOTTING PIPELINE SUMMARY")
        print("="*60)
        print(f"Status: {manifest['plotting_run']['status']}")
        print(f"Results Base: {manifest['plotting_run']['results_base']}")
        
        summary = manifest['plotting_run'].get('summary', {})
        print(f"Total Attempted: {summary.get('total_attempted', 0)}")
        print(f"Total Success: {summary.get('total_success', 0)}")
        print(f"Success Rate: {summary.get('success_rate', 0):.1%}")
        
        windows_plotted = manifest['plotting_run']['windows_plotted']
        for exp_name, windows in windows_plotted.items():
            print(f"\n{exp_name}:")
            for window_id, window_info in windows.items():
                status = window_info['status']
                plots = window_info.get('plots_created', [])
                print(f"  {window_id}: {status} ({len(plots)} plots)")
                if status == 'success':
                    output_dir = Path(window_info['output_directory'])
                    print(f"    → {output_dir}")
        
        errors = manifest['plotting_run']['errors']
        warnings = manifest['plotting_run']['warnings']
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
        
        # Show example output files
        if summary.get('total_success', 0) > 0:
            print(f"\nExample output files:")
            for exp_name, windows in windows_plotted.items():
                for window_id, window_info in windows.items():
                    if window_info['status'] == 'success':
                        output_dir = Path(window_info['output_directory'])
                        if output_dir.exists():
                            pdf_files = list(output_dir.glob('*.pdf'))
                            if pdf_files:
                                print(f"  {pdf_files[0]}")
                                break
                if pdf_files:
                    break
        
        print(f"\nOverall manifest: {pipeline.results_base}/plotting_manifest.json")
        
    except Exception as e:
        logger.error(f"Plotting pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()
