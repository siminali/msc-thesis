#!/usr/bin/env python3
"""
Generate training/config/compute reports from repo logs and final evaluation exports.

Outputs (created deterministically):
- results/compute_reports/
  - summary.csv / .tex (per-run config + metadata)
  - compute_profile.csv / .tex (from final_results_thesis/evaluation_results.json if present)
  - plots/*.pdf (loss curves per run when available; bar charts for training time, VRAM)
- If final_results_thesis/overleaf/ exists, copies the .tex tables into its tables/.

Notes:
- Safe to run multiple times; files are overwritten.
- No randomness beyond deterministic sorting.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, List

import numpy as np


def _read_json_safe(path: Path) -> Dict[str, Any]:
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception:
        return {}


def _collect_runs(root: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if not root.exists():
        return records
    # Walk two levels under runs/*/*
    for model_dir in sorted([d for d in root.iterdir() if d.is_dir()]):
        for run_dir in sorted([d for d in model_dir.iterdir() if d.is_dir()]):
            rec: Dict[str, Any] = {
                'model_key': model_dir.name,
                'run_name': run_dir.name,
                'run_path': str(run_dir),
            }
            # Load files if present
            training_history = _read_json_safe(run_dir / 'training_history.json')
            metadata = _read_json_safe(run_dir / 'metadata.json')
            run_config = _read_json_safe(run_dir / 'run_config.json')
            # Flatten a few interesting fields
            rec['training_time_seconds'] = metadata.get('training_time_seconds', np.nan)
            rec['gpu_model'] = (metadata.get('gpu_info', {}) or {}).get('name', metadata.get('gpu_info', 'Unknown'))
            rec['device'] = run_config.get('device', metadata.get('device', 'Unknown'))
            rec['num_epochs'] = run_config.get('num_epochs', np.nan)
            rec['num_timesteps'] = run_config.get('num_timesteps', run_config.get('timesteps', np.nan))
            rec['batch_size'] = run_config.get('batch_size', np.nan)
            rec['learning_rate'] = run_config.get('learning_rate', np.nan)
            rec['model_parameters'] = metadata.get('model_parameters', np.nan)
            # Loss curves availability
            rec['has_train_losses'] = bool(training_history.get('train_losses'))
            rec['has_val_losses'] = bool(training_history.get('val_losses'))
            # Save file paths to plot later
            if rec['has_train_losses']:
                rec['train_losses'] = training_history.get('train_losses', [])
            if rec['has_val_losses']:
                rec['val_losses'] = training_history.get('val_losses', [])
            records.append(rec)
    return records


def _ensure_dirs(out_root: Path):
    (out_root / 'plots').mkdir(parents=True, exist_ok=True)
    (out_root / 'tables').mkdir(parents=True, exist_ok=True)


def _write_tables(records: List[Dict[str, Any]], out_root: Path):
    try:
        import pandas as pd
    except Exception:
        return
    df = pd.DataFrame(records)
    # Stable sort
    df = df.sort_values(by=['model_key', 'run_name'])
    df.to_csv(out_root / 'summary.csv', index=False)
    with open(out_root / 'summary.tex', 'w') as f:
        f.write(df.to_latex(index=False))

    # Compute profile table from final_results_thesis if available
    fr = Path('final_results_thesis') / 'evaluation_results.json'
    if fr.exists():
        data = _read_json_safe(fr)
        comp = data.get('compute_profile', {})
        if comp:
            rows = []
            for model, prof in comp.items():
                rows.append({
                    'Model': model,
                    'Training_Time_Seconds': prof.get('training_time_seconds', np.nan),
                    'Inference_Time_Seconds': prof.get('inference_time_seconds', np.nan),
                    'Peak_VRAM_MB': prof.get('peak_vram_mb', np.nan),
                    'Total_GPU_VRAM_MB': prof.get('total_gpu_vram_mb', np.nan),
                    'GPU_Model': prof.get('gpu_model', 'Unknown'),
                    'Parameters': prof.get('parameters', np.nan),
                })
            dfp = pd.DataFrame(rows)
            dfp = dfp.sort_values(by=['Model'])
            dfp.to_csv(out_root / 'compute_profile.csv', index=False)
            with open(out_root / 'tables' / 'compute_profile.tex', 'w') as f:
                f.write(dfp.to_latex(index=False))


def _plots(records: List[Dict[str, Any]], out_root: Path):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import pandas as pd
    except Exception:
        return
    # Loss curves per run
    for rec in records:
        run_tag = f"{rec['model_key']}_{rec['run_name']}"
        if rec.get('has_train_losses'):
            plt.figure(figsize=(6, 3.2))
            tl = np.array(rec.get('train_losses', []), dtype=float)
            plt.plot(tl, label='Train')
            if rec.get('has_val_losses'):
                vl = np.array(rec.get('val_losses', []), dtype=float)
                plt.plot(vl, label='Val')
            plt.title(f"Losses: {run_tag}")
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_root / 'plots' / f"loss_{run_tag}.pdf")
            plt.close()

    # Aggregated bar charts: training time by model_key
    import pandas as pd  # noqa
    df = pd.DataFrame(records)
    if 'training_time_seconds' in df.columns and df['training_time_seconds'].notna().any():
        g = df.groupby('model_key', as_index=False)['training_time_seconds'].median()
        plt.figure(figsize=(6, 3.2))
        plt.bar(g['model_key'], g['training_time_seconds'])
        plt.ylabel('Median training time (s)')
        plt.title('Training time by model')
        plt.xticks(rotation=20)
        plt.tight_layout()
        plt.savefig(out_root / 'plots' / 'training_time_by_model.pdf')
        plt.close()

    # If compute_profile available, VRAM and inference-time bars
    fr = Path('final_results_thesis') / 'evaluation_results.json'
    comp = _read_json_safe(fr).get('compute_profile', {}) if fr.exists() else {}
    if comp:
        models = sorted(comp.keys())
        peak = [comp[m].get('peak_vram_mb', np.nan) for m in models]
        plt.figure(figsize=(6, 3.2))
        plt.bar(models, peak)
        plt.ylabel('Peak VRAM (MB)')
        plt.title('Peak VRAM by model')
        plt.xticks(rotation=20)
        plt.tight_layout()
        plt.savefig(out_root / 'plots' / 'peak_vram_by_model.pdf')
        plt.close()

        infer = [comp[m].get('inference_time_seconds', np.nan) for m in models]
        if np.any(np.isfinite(infer)):
            plt.figure(figsize=(6, 3.2))
            plt.bar(models, infer)
            plt.ylabel('Inference time (s)')
            plt.title('Inference time by model')
            plt.xticks(rotation=20)
            plt.tight_layout()
            plt.savefig(out_root / 'plots' / 'inference_time_by_model.pdf')
            plt.close()

    # Parameter-count bars by model_key (median across runs)
    if 'model_parameters' in df.columns and df['model_parameters'].notna().any():
        gp = df.groupby('model_key', as_index=False)['model_parameters'].median()
        plt.figure(figsize=(6, 3.2))
        plt.bar(gp['model_key'], gp['model_parameters'])
        plt.ylabel('Parameters (median)')
        plt.title('Model parameter counts')
        plt.xticks(rotation=20)
        plt.tight_layout()
        plt.savefig(out_root / 'plots' / 'parameters_by_model.pdf')
        plt.close()


def _copy_to_overleaf(out_root: Path):
    # If Overleaf dir exists, copy tables
    ol = Path('final_results_thesis') / 'overleaf' / 'tables'
    if not ol.exists():
        return
    try:
        import shutil
        for name in ['summary.tex', 'compute_profile.csv', 'tables/compute_profile.tex']:
            src = out_root / name
            if src.exists():
                dst = ol / (src.name if src.name != 'summary.tex' else 'training_runs_summary.tex')
                shutil.copy2(src, dst)
    except Exception:
        pass


def main():
    out_root = Path('results') / 'compute_reports'
    _ensure_dirs(out_root)
    records = _collect_runs(Path('runs'))
    _write_tables(records, out_root)
    _plots(records, out_root)
    _copy_to_overleaf(out_root)
    print(f"✅ Training/compute reports saved to: {out_root}")


if __name__ == '__main__':
    main()


