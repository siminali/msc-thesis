#!/usr/bin/env python3
"""
Build Final Results Script
==========================

Builds the comprehensive Final Results PDF for the thesis.
This script orchestrates the entire evaluation and PDF generation process.

Author: [Your Name]
Date: [Current Date]
"""

import os
import sys
import json
import logging
import warnings
warnings.filterwarnings('ignore')

# Configuration
export_overleaf = True  # Set to True to generate Overleaf-compatible LaTeX tables and figures

# Set environment variables for BLAS libraries before importing scientific libraries
os.environ['OMP_NUM_THREADS'] = '1'  # OpenMP single thread
os.environ['MKL_NUM_THREADS'] = '1'  # Intel MKL single thread
os.environ['OPENBLAS_NUM_THREADS'] = '1'  # OpenBLAS single thread
os.environ['BLIS_NUM_THREADS'] = '1'  # BLIS single thread
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'  # Accelerate single thread

# Import scientific libraries after setting environment variables
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
import multiprocessing

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

# Import custom modules
from final_results_evaluator import FinalResultsEvaluator
from final_results_pdf_builder import FinalResultsPDFBuilder

# Configuration
real_data_path = "data/sp500_data.csv"
synthetic_dir = "results"
outputs_dir = "final_results_thesis"

# Model names and their data files
MODELS = {
    "GARCH": "garch_returns.npy",
    "DDPM": "ddpm_returns.npy", 
    "TimeGrad": "timegrad_returns.npy",
    "LLM-Conditioned": "llm_conditioned_returns.npy"
}

def setup_logging():
    """Setup logging configuration"""
    log_file = os.path.join(outputs_dir, "build_log.txt")
    os.makedirs(outputs_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Log system information
    logger = logging.getLogger(__name__)
    logger.info("=== BUILD LOG START ===")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"OS: {os.uname().sysname} {os.uname().release}")
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    
    # Log BLAS and threading information
    logger.info("=== PERFORMANCE SETTINGS ===")
    logger.info(f"OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS', 'Not set')}")
    logger.info(f"MKL_NUM_THREADS: {os.environ.get('MKL_NUM_THREADS', 'Not set')}")
    logger.info(f"OPENBLAS_NUM_THREADS: {os.environ.get('OPENBLAS_NUM_THREADS', 'Not set')}")
    logger.info(f"CPU cores: {multiprocessing.cpu_count()}")
    
    # Log MMD bootstrap settings
    logger.info("=== MMD BOOTSTRAP SETTINGS ===")
    logger.info("B=300, subsample_size=400, n_jobs=-1, CI=95%")
    logger.info("dtype: float32 for kernel math")
    logger.info("RBF kernel with median heuristic bandwidth")
    logger.info("Unbiased U-statistic estimator")
    logger.info("Cached Gram matrices with joblib parallelization")
    
    # Log library versions
    try:
        import numpy as np
        import pandas as pd
        import matplotlib
        import seaborn as sns
        import scipy
        import statsmodels
        
        logger.info(f"NumPy version: {np.__version__}")
        logger.info(f"Pandas version: {pd.__version__}")
        logger.info(f"Matplotlib version: {matplotlib.__version__}")
        logger.info(f"Seaborn version: {sns.__version__}")
        logger.info(f"SciPy version: {scipy.__version__}")
        logger.info(f"Statsmodels version: {statsmodels.__version__}")
        
        # Detect BLAS vendor
        try:
            blas_info = scipy.linalg.blas_info()
            logger.info(f"BLAS vendor: {blas_info.get('vendor', 'Unknown')}")
        except:
            logger.info("BLAS vendor: Could not detect")
            
    except ImportError as e:
        logger.warning(f"Could not determine version for: {e}")
    
    return logger

def validate_inputs():
    """Validate that all required input files exist"""
    logger = logging.getLogger(__name__)
    
    # Check real data
    if not os.path.exists(real_data_path):
        raise FileNotFoundError(
            f"Real S&P 500 data not found at {real_data_path}. "
            "Please ensure the CSV file exists with Date and Close columns."
        )
    
    # Check synthetic data
    missing_models = []
    for model_name, filename in MODELS.items():
        filepath = os.path.join(synthetic_dir, filename)
        if not os.path.exists(filepath):
            missing_models.append(f"{model_name} ({filename})")
    
    if missing_models:
        raise FileNotFoundError(
            f"Missing synthetic data files for models: {', '.join(missing_models)}. "
            f"Please ensure all .npy files exist in {synthetic_dir}."
        )
    
    logger.info("All input files validated successfully")
    logger.info(f"Real data: {real_data_path}")
    logger.info(f"Synthetic data directory: {synthetic_dir}")
    for model_name, filename in MODELS.items():
        logger.info(f"  {model_name}: {filename}")

def log_performance_summary():
    """Log final performance and threading information"""
    logger = logging.getLogger(__name__)
    
    logger.info("=== PERFORMANCE SUMMARY ===")
    logger.info(f"CPU core count: {multiprocessing.cpu_count()}")
    logger.info(f"Effective thread settings:")
    logger.info(f"  OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS', 'Not set')}")
    logger.info(f"  MKL_NUM_THREADS: {os.environ.get('MKL_NUM_THREADS', 'Not set')}")
    logger.info(f"  OPENBLAS_NUM_THREADS: {os.environ.get('OPENBLAS_NUM_THREADS', 'Not set')}")
    
    # Try to detect actual BLAS threading
    try:
        import scipy
        if hasattr(scipy, 'linalg') and hasattr(scipy.linalg, 'blas_info'):
            blas_info = scipy.linalg.blas_info()
            logger.info(f"Detected BLAS vendor: {blas_info.get('vendor', 'Unknown')}")
    except:
        pass
    
    logger.info("Build completed with performance optimizations")

def main():
    """Main execution function"""
    logger = setup_logging()
    
    try:
        # Validate inputs
        logger.info("Validating input files...")
        validate_inputs()
        
        # Create output directories
        logger.info("Creating output directories...")
        os.makedirs(outputs_dir, exist_ok=True)
        os.makedirs(os.path.join(outputs_dir, "figures"), exist_ok=True)
        os.makedirs(os.path.join(outputs_dir, "tables"), exist_ok=True)
        
        # Initialize evaluator
        logger.info("Initializing evaluation framework...")
        evaluator = FinalResultsEvaluator(
            real_data_path=real_data_path,
            synthetic_dir=synthetic_dir,
            models=MODELS
        )
        
        # Compute all metrics
        logger.info("Computing evaluation metrics...")
        results = evaluator.compute_all_metrics()
        
        # Save results
        logger.info("Saving evaluation results...")
        results_file = os.path.join(outputs_dir, "evaluation_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Generate metrics summary
        logger.info("Generating metrics summary...")
        summary_file = os.path.join(outputs_dir, "metrics_summary.csv")
        evaluator.save_metrics_summary(summary_file)
        
        # Initialize PDF builder
        logger.info("Initializing PDF builder...")
        
        # Log section-specific settings and data availability
        logger.info("=== SECTION 5-7 IMPLEMENTATION SETTINGS ===")
        
        # Section 5: Conditioning and Controllability
        if 'conditioning_analysis' in results and results['conditioning_analysis']['LLM-Conditioned']['condition_response']:
            logger.info("Section 5: Conditioning analysis data available")
            logger.info(f"Target volatility bins: Low/Medium/High (33rd/67th percentiles)")
            logger.info(f"Coverage constraint: ±10% of target")
            logger.info("Regime labels: Not available (explicit metadata required)")
        else:
            logger.info("Section 5: Conditioning analysis data not available")
            logger.info("Required: target volatility specifications and regime labels")
        
        # Section 6: Robustness and Stability
        if 'robustness_analysis' in results:
            logger.info("Section 6: Robustness analysis data available")
            for model, data in results['robustness_analysis'].items():
                logger.info(f"  {model}: {data['n_samples']} samples")
            logger.info("Bootstrap settings: Reusing from earlier computation")
            logger.info("Confidence intervals: 95% level")
        else:
            logger.info("Section 6: Robustness analysis data not available")
        
        # Section 7: Use-Case Panels
        logger.info("Section 7: Use-case panels implementation")
        logger.info("  Hedge Funds: Condition→Response analysis from Section 5")
        logger.info("  Credit/Insurance: EVT analysis requires additional computation")
        logger.info("  Traditional Banks: VaR backtesting from Section 3")
        
        # Log Overleaf export settings
        logger.info("=== OVERLEAF EXPORT SETTINGS ===")
        logger.info(f"export_overleaf: {export_overleaf}")
        if export_overleaf:
            logger.info("Overleaf export enabled - will generate LaTeX tables and figure stubs")
            logger.info("Output directory: final_results_thesis/overleaf/")
            logger.info("Format: booktabs tables, vector PDF figures, LaTeX stubs with captions and labels")
        
        pdf_builder = FinalResultsPDFBuilder(
            results=results,
            outputs_dir=outputs_dir,
            evaluator=evaluator
        )
        
        # Build PDF
        logger.info("Building Final Results PDF...")
        pdf_path = pdf_builder.build_pdf()
        
        logger.info(f"Final Results PDF created successfully: {pdf_path}")
        logger.info("Build completed successfully!")
        
        # Export to Overleaf if enabled
        if export_overleaf:
            logger.info("Exporting to Overleaf format...")
            try:
                from overleaf_exporter import OverleafExporter
                overleaf_exporter = OverleafExporter(results, outputs_dir, evaluator)
                overleaf_exporter.export_all()
                logger.info("Overleaf export completed successfully!")
            except Exception as e:
                logger.error(f"Overleaf export failed: {str(e)}")
                logger.error("Check the build log for details")
        
        # Log performance summary
        log_performance_summary()
        
        # Create reproducibility README
        logger.info("Creating reproducibility documentation...")
        create_reproducibility_readme()
        
    except Exception as e:
        logger.error(f"Build failed with error: {str(e)}")
        logger.error("Check the build log for details")
        raise

def create_reproducibility_readme():
    """Create README_repro.md with reproduction instructions"""
    readme_content = """# Reproducing the Final Results Thesis PDF

## Prerequisites

- Python 3.8+
- Required packages (see requirements.txt)
- Access to the data files specified below

## Data Requirements

- `data/sp500_data.csv`: S&P 500 daily closing prices (2010-2024)
- `results/garch_returns.npy`: GARCH model synthetic returns
- `results/ddpm_returns.npy`: DDPM model synthetic returns  
- `results/timegrad_returns.npy`: TimeGrad model synthetic returns
- `results/llm_conditioned_returns.npy`: LLM-Conditioned model synthetic returns

## Single Command to Reproduce

```bash
python src/build_final_results.py
```

## Output Structure

```
final_results_thesis/
├── final_results_thesis.pdf          # Main PDF report
├── figures/                          # Vector PDF figures
├── tables/                           # CSV tables
├── metrics_summary.csv               # Consolidated metrics
├── evaluation_results.json           # Full evaluation results
├── build_log.txt                     # Build execution log
└── README_repro.md                   # This file
```

## What Gets Generated

1. **All evaluation metrics** computed from scratch
2. **All figures** as vector PDFs with consistent styling
3. **All tables** as CSV files with full precision
4. **Complete PDF report** with proper formatting and structure

## Performance Optimizations

- BLAS libraries configured for single-threaded operation
- Joblib parallelization for bootstrap computations
- Vectorized MMD computation with cached Gram matrices
- Precomputed statistics reused across comparisons
- Non-interactive matplotlib backend for faster plotting

## Troubleshooting

- Check `build_log.txt` for detailed execution information
- Ensure all input files exist and are accessible
- Verify Python environment has all required packages

## Contact

For questions about reproduction, contact [Your Contact Information]
"""
    
    readme_path = os.path.join(outputs_dir, "README_repro.md")
    with open(readme_path, 'w') as f:
        f.write(readme_content)

if __name__ == "__main__":
    main()
