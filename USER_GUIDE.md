
# User Guide

This guide shows how to set up the environment and run the essential workflows: training, evaluation, plotting, and report exports.

## 1) Setup

### Python
- Recommended: Python 3.10+
- Create and activate a virtual environment:
```
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

### Install requirements
- Base requirements:
```
pip install -r requirements/base.txt
```
- LLM-conditioned extras (if needed):
```
pip install -r requirements/llm_refactored.txt
```

### Data
- Ensure S&P 500 returns CSV is available at `data/sp500_data.csv` (default path). You can change paths via CLI flags where supported.

## 2) Training (pre-COVID models)
Run the integrated training pipeline:
```
python scripts/training/run_pipeline.py
```
Or run specific training scripts:
```
python scripts/training/train_precovid_models.py
python scripts/training/train_precovid_models_v2.py
python scripts/training/train_precovid_simplified.py
```
Outputs: checkpoints under `checkpoints/`, intermediate artifacts under `runs/`.

## 3) Comprehensive Evaluation
Evaluate across Experiment A/B windows and compile metrics/plots.

### One-shot end-to-end
```
python scripts/evaluation/run_comprehensive_evaluation.py
```
This orchestrates sampling (if needed), metrics, and plotting.

### Components individually
- Metrics calculation:
```
python scripts/evaluation/metrics_runner.py   --results-base results/addons/period_slices   --experiments A B   --windows covid_crash covid_recovery
```
- Plotting:
```
python scripts/evaluation/plotting_runner.py   --results-base results/addons/period_slices   --experiments A B   --windows covid_crash covid_recovery
```
- Experiment evaluators (integrated sample+metrics+plots):
```
python scripts/experiments/experiment_A_evaluator_v2.py
python scripts/experiments/experiment_B_evaluator_v2.py
```

## 4) Reporting and Exports
- LaTeX-ready exports for training/evaluation:
```
python scripts/exports/create_latex_training_exports.py
python scripts/exports/create_comprehensive_latex_exports.py
```
- Report compiler:
```
python scripts/exports/report_compiler.py
```
Artifacts are written to `final_results_benchmarking/`, `latex_training_exports/`, and `comprehensive_latex_exports/`.

## 5) Reproducing Final Benchmarks
Final benchmark assets live in `final_results_benchmarking/`.
- Primary summary CSV: `final_results_benchmarking/metrics_summary.csv`
- Full evaluation JSON: `final_results_benchmarking/evaluation_results.json`
- Figures: `final_results_benchmarking/figures/`
- Overleaf-ready: `final_results_benchmarking/overleaf/`

## 6) Quick commands
- Run all: `python scripts/evaluation/run_comprehensive_evaluation.py`
- Only metrics: `python scripts/evaluation/metrics_runner.py`
- Only plots: `python scripts/evaluation/plotting_runner.py`
- Experiment A integrated: `python scripts/experiments/experiment_A_evaluator_v2.py`
- Experiment B integrated: `python scripts/experiments/experiment_B_evaluator_v2.py`

## 7) Configuration
- Training config: `configs/training_config.json`
- Checkpoints: `checkpoints/precovid/*`
- Results: `results/`, `runs/`

## 8) Troubleshooting
- If imports fail for shared modules, ensure package structure is visible and you run from repo root:
```
python -m scripts.evaluation.metrics_runner  # alternative invocation
```
- Missing data: confirm `data/sp500_data.csv` or pass `--csv-file` to metrics/evaluators.
- Figures not appearing: verify write permissions to `results/` and `final_results_benchmarking/`.

