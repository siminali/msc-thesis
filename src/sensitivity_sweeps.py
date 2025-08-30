#!/usr/bin/env python3
"""
Lightweight sensitivity sweep harness for timesteps, beta schedule, and cfg_scale.

Constraints:
- Reuse existing checkpoints for cfg_scale and sampling-only parameters.
- Retrain per-config ONLY when training-time parameters change (num_timesteps, beta_schedule).
- Deterministic and lightweight: tiny defaults, minimal epochs.
- Save per-config metrics and aggregate summary.(csv|tex) under results/sweeps/<model>/.
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import torch


def _slugify(config: Dict[str, Any]) -> str:
    parts = []
    for k in sorted(config.keys()):
        v = str(config[k]).replace(' ', '')
        parts.append(f"{k}-{v}")
    return "_".join(parts)


def run_sampling_only_sweep(model: str, cfg_scale_list: List[float], output_dir: Path) -> List[Dict[str, Any]]:
    """Vary cfg_scale at inference without retraining, reuse latest checkpoint and evaluation pipeline."""
    import sys as _sys
    from pathlib import Path as _Path
    # Ensure local src is importable when running as a script
    _sys.path.append(str(_Path(__file__).parent))
    from evaluate_all import UnifiedEvaluator  # reuse robust evaluator
    # Base config: tiny sample size for speed
    config = {
        'seed': 42,
        'num_samples': 200,
        'var_levels': [0.95, 0.99],
        'reliability_bins': 10,
        'acf_lags': 10,
        'rolling_window': 20,
        'ablation_samples': 200,
        'correlation_lags': 5,
        'outlier_threshold': 3.0,
    }
    # Deterministic seeds
    np.random.seed(config['seed'])
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    torch.manual_seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['seed'])
    results = []
    out_model_dir = output_dir / model
    out_model_dir.mkdir(parents=True, exist_ok=True)

    # Discover checkpoints
    ev = UnifiedEvaluator(config, out_model_dir)
    ckpts = ev.discover_checkpoints(Path('results'))
    if model not in ckpts:
        print(f"No checkpoint found for {model}; skipping.")
        return results

    # Load model once
    mdl, trn = ev.load_model(model, ckpts[model])

    for cfg in cfg_scale_list:
        # Build conditioning and sample with desired cfg_scale (no retraining)
        num = int(config['num_samples'])
        if model == 'explicit_conditioned':
            cond = torch.tensor(ev.conditioning_vectors[:num], dtype=torch.float32, device=trn.device)
        elif model == 'llm_conditioned':
            rand = torch.randn(num, 64, device=trn.device)
            cond = torch.nn.functional.normalize(rand, dim=1)
        else:  # zero_conditioned
            cond = torch.zeros(num, 5, dtype=torch.float32, device=trn.device)
        with torch.no_grad():
            try:
                samples_t = trn.sample(cond, num_samples=num, sampler='ddim', sample_steps=50, cfg_scale=float(cfg))
            except TypeError:
                samples_t = trn.sample(num_samples=num, sampler='ddim', sample_steps=50, cfg_scale=float(cfg))
        samples = samples_t.squeeze(1).cpu().numpy()

        # Evaluate via the same evaluator (deterministic)
        stylized = ev.evaluate_stylized_facts(samples, model)
        dist = ev.evaluate_distributional_fidelity(samples, model)
        risk = ev.evaluate_risk_metrics(samples, model)
        metrics = {**stylized, **dist, **risk}

        cfg_rec = {'cfg_scale': cfg}
        slug = _slugify(cfg_rec)
        run_dir = out_model_dir / 'figures' / slug
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(out_model_dir / (slug.replace(':', '_') + '.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
        results.append({'config': cfg_rec, 'metrics': metrics})

    # Aggregate
    rows = []
    for r in results:
        row = {'cfg_scale': r['config']['cfg_scale']}
        row.update({k: r['metrics'].get(k, np.nan) for k in ['ks_statistic', 'mmd', 'var_95', 'es_95', 'var_99', 'es_99']})
        rows.append(row)
    try:
        import pandas as pd
        df = pd.DataFrame(rows)
        df.to_csv(out_model_dir / 'summary.csv', index=False)
        with open(out_model_dir / 'summary.tex', 'w') as f:
            f.write(df.to_latex(index=False))
    except Exception:
        pass

    return results


def generate_plots(model: str, output_dir: Path) -> None:
    """Generate PDF plots from summary.csv for the given model directory."""
    try:
        import pandas as pd
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"Plotting dependencies missing: {e}")
        return
    model_dir = output_dir / model
    summary_path = model_dir / 'summary.csv'
    if not summary_path.exists():
        print(f"No summary.csv for {model}; skipping plots.")
        return
    df = pd.read_csv(summary_path)
    fig_dir = model_dir / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)
    metrics = ['ks_statistic', 'mmd', 'var_95', 'es_95', 'var_99', 'es_99']
    for m in metrics:
        if m in df.columns:
            plt.figure(figsize=(5, 3.2))
            plt.plot(df['cfg_scale'], df[m], marker='o')
            plt.xlabel('cfg_scale')
            plt.ylabel(m)
            plt.title(f'{model}: {m} vs cfg_scale')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(fig_dir / f'cfg_scale_vs_{m}.pdf')
            plt.close()
    # Combined grid figure
    present = [m for m in metrics if m in df.columns]
    if present:
        n = len(present)
        cols = 3
        rows = int(np.ceil(n / cols))
        plt.figure(figsize=(cols * 4, rows * 3))
        for i, m in enumerate(present, 1):
            ax = plt.subplot(rows, cols, i)
            ax.plot(df['cfg_scale'], df[m], marker='o')
            ax.set_xlabel('cfg_scale')
            ax.set_ylabel(m)
            ax.set_title(m)
            ax.grid(True, alpha=0.3)
        plt.suptitle(f'{model}: Sensitivity Summary')
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        plt.savefig(fig_dir / 'sensitivity_summary.pdf')
        plt.close()


def main():
    p = argparse.ArgumentParser(description='Sensitivity sweep harness (lightweight, deterministic).')
    p.add_argument('--model', choices=['explicit_conditioned', 'llm_conditioned', 'zero_conditioned'], default='llm_conditioned')
    p.add_argument('--num-timesteps', nargs='*', type=int, default=[1000])
    p.add_argument('--beta-schedule', nargs='*', choices=['cosine', 'linear'], default=['cosine'])
    p.add_argument('--cfg-scale', nargs='*', type=float, default=[1.0, 7.5])
    p.add_argument('--repeats', type=int, default=1)
    p.add_argument('--output-dir', type=str, default='results/sweeps')
    p.add_argument('--plot-only', action='store_true', default=False, help='Only generate plots from existing summary.csv')
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        generate_plots(args.model, out_dir)
        print(f"✅ Plots generated. See {out_dir / args.model / 'figures'}")
        return

    # For this lightweight version, implement sampling-only sweeps for cfg_scale
    _ = run_sampling_only_sweep(args.model, args.cfg_scale, out_dir)
    generate_plots(args.model, out_dir)

    print(f"✅ Sensitivity sweep completed. Outputs in {out_dir / args.model}")


if __name__ == '__main__':
    main()


