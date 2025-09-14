# Experimental Pipeline Runbook

Complete runbook for financial time series diffusion model experiments with example commands using actual implemented tools.

## Overview

This runbook provides end-to-end workflows for:
- **Pre-COVID Training**: Training models on 2010-2019 data
- **Experiment A**: Out-of-sample stress testing using pre-COVID checkpoints  
- **Experiment B**: Counterfactual controllability testing
- **Report Generation**: Professional PDF compilation
- **Hygiene Validation**: Data integrity and reproducibility checks

## Quick Start Commands

### Complete Pipeline Example
```bash
# 1. Train pre-COVID models
python train_precovid_simplified.py

# 2. Run stress testing experiment
python experiment_A_evaluator_v2.py --window covid_crash --num-paths 1000 --csv-file data/sp500_data.csv

# 3. Run controllability experiment  
python experiment_B_evaluator_v2.py --window covid_crash --num-paths 1000 --csv-file data/sp500_data.csv

# 4. Generate professional reports
python report_compiler.py --expdir results/addons/period_slices/A_v8
python report_compiler.py --expdir results/addons/period_slices/B_v3
```

## 1. Pre-COVID Model Training

### Basic Training
```bash
# Train all models with default settings
python train_precovid_simplified.py

# Train with custom data file
python train_precovid_simplified.py --csv-file data/sp500_data.csv

# Train with specific seed
python train_precovid_simplified.py --seed 123

# Train only specific models
python train_precovid_simplified.py --models zero explicit
```

### Training Configuration
```bash
# Custom training parameters
python train_precovid_simplified.py \
    --csv-file data/sp500_data.csv \
    --seq-len 60 \
    --batch-size 64 \
    --epochs 100 \
    --learning-rate 1e-4 \
    --hidden-dim 128 \
    --models zero explicit llm \
    --seed 42
```

### Expected Outputs
```
checkpoints/precovid/
├── zero/20100101-20191231/
│   ├── best.pt
│   ├── last.pt
│   ├── meta.json
│   └── conditioning_spec.json
├── explicit/20100101-20191231/
│   ├── best.pt
│   ├── last.pt
│   ├── meta.json
│   └── conditioning_spec.json
└── llm/20100101-20191231/
    ├── best.pt
    ├── last.pt
    ├── meta.json
    ├── conditioning_spec.json
    └── pca_model.pkl
```

## 2. Experiment A: Out-of-Sample Stress Testing

### Single Window Testing
```bash
# Test COVID crash period
python experiment_A_evaluator_v2.py \
    --window covid_crash \
    --num-paths 1000 \
    --csv-file data/sp500_data.csv \
    --seed 42

# Test COVID recovery period  
python experiment_A_evaluator_v2.py \
    --window covid_recovery \
    --num-paths 1000 \
    --csv-file data/sp500_data.csv \
    --seed 42
```

### Multiple Checkpoints
```bash
# Use specific checkpoint directory
python experiment_A_evaluator_v2.py \
    --window covid_crash \
    --checkpoints-dir checkpoints/precovid \
    --num-paths 2000 \
    --seq-len 60 \
    --csv-file data/sp500_data.csv
```

### Quick Testing
```bash
# Fast testing with fewer paths
python experiment_A_evaluator_v2.py \
    --window covid_crash \
    --num-paths 100 \
    --csv-file data/sp500_data.csv
```

### Expected Outputs
```
results/addons/period_slices/A_v8/
├── plan.json
├── manifest.json
├── findings.jsonl
├── integrated_summary.json
└── covid_crash/
    ├── zero/samples.npy
    ├── explicit/samples.npy
    ├── llm/samples.npy
    ├── metrics.json
    ├── tables/
    └── figs/
        ├── ecdf_overlay.pdf/png
        ├── qq_plots.pdf/png
        ├── var_es_analysis.pdf/png
        └── realized_volatility.pdf/png
```

## 3. Experiment B: Counterfactual Controllability Testing

### Basic Controllability Testing
```bash
# Test controllability on COVID crash
python experiment_B_evaluator_v2.py \
    --window covid_crash \
    --num-paths 1000 \
    --csv-file data/sp500_data.csv \
    --seed 42

# Test different window
python experiment_B_evaluator_v2.py \
    --window covid_recovery \
    --num-paths 1000 \
    --csv-file data/sp500_data.csv
```

### Custom Configuration
```bash
# Extended controllability analysis
python experiment_B_evaluator_v2.py \
    --window covid_crash \
    --checkpoints-dir checkpoints/precovid \
    --num-paths 2000 \
    --seq-len 60 \
    --seeds 42 123 456 \
    --csv-file data/sp500_data.csv
```

### Expected Outputs
```
results/addons/period_slices/B_v3/
├── plan.json
├── manifest.json
├── findings.jsonl
├── controllability_summary.json
└── covid_crash/
    ├── explicit/
    │   ├── real-conditions/samples.npy
    │   ├── calm-conditions/samples.npy
    │   └── llm-knob-*/samples.npy
    ├── llm/
    │   ├── real-conditions/samples.npy
    │   ├── calm-conditions/samples.npy
    │   └── llm-knob-*/samples.npy
    ├── metrics.json
    ├── tables/
    └── figs/
```

## 4. Standalone Tools

### Checkpoint Loading and Sampling
```bash
# Load checkpoint and generate samples
python checkpoint_loader_sampler.py \
    --checkpoint checkpoints/precovid/explicit/20100101-20191231 \
    --dates 2020-03-15 2020-03-22 2020-03-29 \
    --num-paths 500 \
    --output-dir custom_samples/
```

### Metrics Calculation
```bash
# Calculate metrics for existing experiments
python metrics_runner.py \
    --experiments A B \
    --windows covid_crash \
    --csv-file data/sp500_data.csv \
    --results-base results/addons/period_slices
```

### Plotting Generation
```bash
# Generate plots for experiments
python plotting_runner.py \
    --experiments A B \
    --windows covid_crash \
    --results-base results/addons/period_slices
```

### Hygiene Validation
```bash
# Validate specific checkpoint
python hygiene_checks.py \
    --checkpoint checkpoints/precovid/explicit/20100101-20191231 \
    --model-type explicit \
    --seed 42

# Quick environment setup
python -c "from hygiene_checks import setup_reproducible_environment; setup_reproducible_environment(42)"
```

## 5. Report Generation

### Individual Reports
```bash
# Generate report for Experiment A
python report_compiler.py --expdir results/addons/period_slices/A_v8

# Generate report for Experiment B
python report_compiler.py --expdir results/addons/period_slices/B_v3

# Generate report for base experiments
python report_compiler.py --expdir results/addons/period_slices/A
python report_compiler.py --expdir results/addons/period_slices/B
```

### Batch Report Generation
```bash
# Generate all available reports
for exp in A A_v8 B B_v3; do
    if [ -d "results/addons/period_slices/$exp" ]; then
        echo "Generating report for $exp..."
        python report_compiler.py --expdir "results/addons/period_slices/$exp"
    fi
done
```

### Expected Report Outputs
```
results/addons/period_slices/A_v8/report_A_v8.pdf    # ~878KB
results/addons/period_slices/B_v3/report_B_v3.pdf    # ~889KB
```

## 6. Complete Workflows

### Full Experiment A Workflow
```bash
#!/bin/bash
# Complete Experiment A: Pre-COVID Training → Stress Testing → Report

echo "🚀 Starting complete Experiment A workflow..."

# 1. Train pre-COVID models
echo "📚 Training pre-COVID models..."
python train_precovid_simplified.py --csv-file data/sp500_data.csv --seed 42

# 2. Run stress testing
echo "💥 Running stress testing..."
python experiment_A_evaluator_v2.py \
    --window covid_crash \
    --num-paths 1000 \
    --csv-file data/sp500_data.csv \
    --seed 42

# 3. Generate report
echo "📄 Generating report..."
python report_compiler.py --expdir results/addons/period_slices/A_v8

echo "✅ Experiment A completed!"
echo "📊 Report: results/addons/period_slices/A_v8/report_A_v8.pdf"
```

### Full Experiment B Workflow
```bash
#!/bin/bash
# Complete Experiment B: Pre-COVID Training → Controllability → Report

echo "🚀 Starting complete Experiment B workflow..."

# 1. Ensure pre-COVID models exist
if [ ! -d "checkpoints/precovid" ]; then
    echo "📚 Training pre-COVID models..."
    python train_precovid_simplified.py --csv-file data/sp500_data.csv --seed 42
fi

# 2. Run controllability testing
echo "🎛️ Running controllability testing..."
python experiment_B_evaluator_v2.py \
    --window covid_crash \
    --num-paths 1000 \
    --csv-file data/sp500_data.csv \
    --seed 42

# 3. Generate report
echo "📄 Generating report..."
python report_compiler.py --expdir results/addons/period_slices/B_v3

echo "✅ Experiment B completed!"
echo "📊 Report: results/addons/period_slices/B_v3/report_B_v3.pdf"
```

### Batch Processing Multiple Windows
```bash
#!/bin/bash
# Process multiple stress windows

WINDOWS=("covid_crash" "covid_recovery")
CSV_FILE="data/sp500_data.csv"
NUM_PATHS=1000

for window in "${WINDOWS[@]}"; do
    echo "Processing window: $window"
    
    # Experiment A
    python experiment_A_evaluator_v2.py \
        --window $window \
        --num-paths $NUM_PATHS \
        --csv-file $CSV_FILE
    
    # Experiment B
    python experiment_B_evaluator_v2.py \
        --window $window \
        --num-paths $NUM_PATHS \
        --csv-file $CSV_FILE
done

# Generate reports
python report_compiler.py --expdir results/addons/period_slices/A_v8
python report_compiler.py --expdir results/addons/period_slices/B_v3
```

## 7. Troubleshooting

### Common Issues and Solutions

#### Missing Data File
```bash
# Error: FileNotFoundError: data/sp500_data.csv not found
# Solution: Use synthetic data or provide correct path
python experiment_A_evaluator_v2.py --window covid_crash --num-paths 100
# Note: Evaluators will use synthetic data if CSV not found
```

#### Checkpoint Not Found
```bash
# Error: Checkpoint directory not found
# Solution: Train pre-COVID models first
python train_precovid_simplified.py
```

#### Memory Issues
```bash
# Error: CUDA out of memory
# Solution: Reduce batch size or number of paths
python experiment_A_evaluator_v2.py --window covid_crash --num-paths 500
```

#### Hygiene Issues
```bash
# Warning: Hygiene issues detected
# Solution: Check hygiene report, but execution will continue
python hygiene_checks.py --checkpoint checkpoints/precovid/explicit/20100101-20191231 --model-type explicit
```

### Debug Commands
```bash
# Check experiment structure
find results/addons/period_slices -name "*.json" | head -10

# Verify checkpoints
ls -la checkpoints/precovid/*/20100101-20191231/

# Check report generation
ls -la results/addons/period_slices/*/report_*.pdf

# Validate data files
head -5 data/sp500_data.csv

# Test hygiene validation
python -c "from hygiene_checks import quick_hygiene_check; print(quick_hygiene_check.__doc__)"
```

## 8. Performance Guidelines

### Recommended Parameters

| Task | Paths | Seq Len | Time | Memory |
|------|-------|---------|------|--------|
| Quick Test | 100 | 60 | ~1 min | ~500MB |
| Standard Run | 1000 | 60 | ~5 min | ~2GB |
| Publication Quality | 2000+ | 60 | ~10 min | ~4GB |

### Scaling Considerations
```bash
# For large-scale experiments
python experiment_A_evaluator_v2.py \
    --window covid_crash \
    --num-paths 5000 \
    --seq-len 60 \
    --seeds 42 123 456 789 \
    --csv-file data/sp500_data.csv
```

## 9. File Organization

### Expected Directory Structure
```
/Users/siminali/Desktop/Thesis Coding/
├── data/
│   └── sp500_data.csv
├── checkpoints/
│   └── precovid/
│       ├── zero/20100101-20191231/
│       ├── explicit/20100101-20191231/
│       └── llm/20100101-20191231/
├── results/
│   └── addons/
│       └── period_slices/
│           ├── A/
│           ├── A_v8/
│           ├── B/
│           └── B_v3/
└── tools/ (main directory scripts)
    ├── train_precovid_simplified.py
    ├── experiment_A_evaluator_v2.py
    ├── experiment_B_evaluator_v2.py
    ├── report_compiler.py
    └── hygiene_checks.py
```

## 10. Integration Examples

### With External Workflows
```bash
# Export results for external analysis
python -c "
import json
with open('results/addons/period_slices/A_v8/findings.jsonl', 'r') as f:
    for line in f:
        finding = json.loads(line)
        print(f'Window: {finding[\"window_id\"]}, Models: {len(finding[\"models\"])}')
"
```

### Automated Pipeline
```bash
#!/bin/bash
# Automated daily experiment pipeline

DATE=$(date +%Y%m%d)
LOG_FILE="experiment_log_$DATE.txt"

{
    echo "Starting automated pipeline: $DATE"
    
    # Update data
    echo "Updating market data..."
    # python fetch_latest_data.py  # Your data update script
    
    # Run experiments
    echo "Running Experiment A..."
    python experiment_A_evaluator_v2.py --window covid_crash --num-paths 1000 --csv-file data/sp500_data.csv
    
    echo "Running Experiment B..."
    python experiment_B_evaluator_v2.py --window covid_crash --num-paths 1000 --csv-file data/sp500_data.csv
    
    # Generate reports
    echo "Generating reports..."
    python report_compiler.py --expdir results/addons/period_slices/A_v8
    python report_compiler.py --expdir results/addons/period_slices/B_v3
    
    echo "Pipeline completed: $DATE"
    
} 2>&1 | tee $LOG_FILE
```

## Notes

- **Versioning**: The system automatically creates versioned directories (A_v8, B_v3, etc.) to avoid overwrites
- **Data Handling**: If `data/sp500_data.csv` is not found, evaluators will use synthetic data with warnings
- **Checkpoints**: Pre-COVID checkpoints are required for Experiments A and B
- **Hygiene**: All tools include hygiene validation that logs issues but never fails execution
- **Reports**: Professional PDF reports are generated with publication-quality figures and analysis
- **Integration**: All tools are designed for easy integration into larger workflows

The runbook provides a complete reference for running financial time series diffusion model experiments with professional reporting and validation.
