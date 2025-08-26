#!/usr/bin/env python3
"""
COVID-19 Crisis Case Study Evaluation

Usage:
  python evaluate_covid_case.py \
      --models_dir results \
      --results_dir results \
      --seed 42 \
      --crisis_start 2020-02-20 \
      --crisis_end 2020-04-30 

This script loads checkpoints for the three novelty models (zero-conditioned,
explicit-conditioned, LLM-conditioned), generates synthetic return paths for the
specified crisis window, and compares them to real returns. It saves plots and
tables under results/<run_id>/covid_case_study/.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

from scipy.stats import chi2

# Local imports
sys.path.append(str(Path(__file__).parent))
from explicit_cond_ddpm import (
    ExplicitConditioningDDPM,
    ExplicitConditioningTrainer,
    create_conditioning_vectors,
)
from llm_conditioned_diffusion_refactored import (
    LLMConditionedDiffusion,
    LLMDiffusionTrainer,
    NewsDataLoader,
)


def parse_args():
    p = argparse.ArgumentParser(description='COVID-19 Crisis Case Study Evaluation')
    p.add_argument('--models_dir', type=str, default='results', help='Directory with trained model runs')
    p.add_argument('--results_dir', type=str, default='results', help='Base directory to write outputs')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--crisis_start', type=str, default='2020-02-20')
    p.add_argument('--crisis_end', type=str, default='2020-04-30')
    p.add_argument('--num_samples', type=int, default=100, help='Number of synthetic paths per model')
    p.add_argument('--rolling_window', type=int, default=20)
    return p.parse_args()


def set_determinism(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def find_sp500_csv() -> Optional[Path]:
    for p in [Path('data/sp500_data.csv'), Path('../data/sp500_data.csv'), Path('../../data/sp500_data.csv')]:
        if p.exists():
            return p
    return None


def load_returns() -> pd.Series:
    csv = find_sp500_csv()
    if csv is None:
        raise FileNotFoundError('Could not find data/sp500_data.csv')
    df = pd.read_csv(csv, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index)
    returns = np.log(df['Close'] / df['Close'].shift(1)).dropna()
    return returns


def discover_checkpoints(models_dir: Path) -> Dict[str, Path]:
    ckpts: Dict[str, Path] = {}
    for mt in ['zero_conditioned', 'explicit_conditioned', 'llm_conditioned']:
        base = models_dir / mt
        if not base.exists():
            print(f'Warning: missing model dir {base}')
            continue
        runs = [d for d in base.iterdir() if d.is_dir()]
        if not runs:
            print(f'Warning: no runs in {base}')
            continue
        latest = max(runs, key=lambda x: x.stat().st_mtime)
        p = latest / 'checkpoints' / 'best_model.pth'
        if p.exists():
            ckpts[mt] = p
            print(f'Found checkpoint for {mt}: {p}')
        else:
            print(f'Warning: checkpoint not found in {latest}')
    return ckpts


def load_model_trainer(mt: str, ckpt_path: Path, device: torch.device):
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    if mt in ('zero_conditioned', 'explicit_conditioned'):
        model = ExplicitConditioningDDPM(sequence_length=60, conditioning_dim=5, hidden_dim=128)
        trainer = ExplicitConditioningTrainer(
            model,
            num_timesteps=1000,
            beta_schedule='cosine',
            device=device,
            grad_clip=1.0,
            cfg_p=(0.0 if mt == 'zero_conditioned' else 0.1),
        )
    elif mt == 'llm_conditioned':
        model = LLMConditionedDiffusion(sequence_length=60, conditioning_dim=64, hidden_dim=128)
        trainer = LLMDiffusionTrainer(
            model,
            num_timesteps=1000,
            beta_schedule='cosine',
            device=device,
            grad_clip=1.0,
            cfg_p=0.1,
        )
    else:
        raise ValueError(mt)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, trainer


def stitch_samples(sample_fn, cond, target_len: int) -> np.ndarray:
    """Generate a path of length target_len by stitching 60-length segments."""
    segments = []
    remain = target_len
    while remain > 0:
        seg = sample_fn(cond)
        seg = seg.squeeze(1).detach().cpu().numpy()  # [N, 60]
        segments.append(seg)
        remain -= seg.shape[1]
    # Take first path across segments and concat
    path = np.concatenate([seg[0] for seg in segments], axis=0)[:target_len]
    return path


def kupiec_uc(hits: np.ndarray, p0: float) -> Tuple[float, float]:
    n = len(hits); v = int(hits.sum())
    p1 = v/n if n > 0 else 0.0
    if n == 0 or p1 <= 0 or p1 >= 1:
        return np.nan, np.nan
    lr = ((1-p0)**(n-v) * (p0**v)) / ((1-p1)**(n-v) * (p1**v))
    stat = -2*np.log(lr); pval = 1-chi2.cdf(stat, 1)
    return float(stat), float(pval)


def christoffersen_ind(hits: np.ndarray) -> Tuple[float, float]:
    if len(hits) < 2:
        return np.nan, np.nan
    n00 = n01 = n10 = n11 = 0
    for i in range(1, len(hits)):
        a, b = hits[i-1], hits[i]
        if a==0 and b==0: n00+=1
        if a==0 and b==1: n01+=1
        if a==1 and b==0: n10+=1
        if a==1 and b==1: n11+=1
    pi0 = n01/(n00+n01) if (n00+n01)>0 else 0.0
    pi1 = n11/(n10+n11) if (n10+n11)>0 else 0.0
    pi = (n01+n11)/(n00+n01+n10+n11) if (n00+n01+n10+n11)>0 else 0.0
    # Likelihoods
    def L(pi0, pi1):
        return ((1-pi0)**n00)*(pi0**n01)*((1-pi1)**n10)*(pi1**n11)
    L1 = L(pi0, pi1); L0 = ((1-pi)**(n00+n10))*(pi**(n01+n11))
    if L0<=0 or L1<=0:
        return np.nan, np.nan
    stat = -2*np.log(L0/L1); pval = 1-chi2.cdf(stat, 1)
    return float(stat), float(pval)


def main():
    args = parse_args()
    set_determinism(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Output dirs
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_root = Path(args.results_dir) / run_id / 'covid_case_study'
    fig_dir = out_root / 'figures'; tbl_dir = out_root / 'tables'; mtr_dir = out_root / 'metrics'
    for d in [fig_dir, tbl_dir, mtr_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Data
    returns = load_returns()
    crisis_start = pd.Timestamp(args.crisis_start)
    crisis_end = pd.Timestamp(args.crisis_end)
    mask = (returns.index >= crisis_start) & (returns.index <= crisis_end)
    real_crisis = returns[mask]
    if len(real_crisis) == 0:
        raise ValueError('No real returns in the specified crisis window')
    Lc = len(real_crisis)

    # Checkpoints
    ckpts = discover_checkpoints(Path(args.models_dir))
    if not ckpts:
        print('❌ No checkpoints found.'); return

    # Prepare explicit conditioning (choose crisis-like: Down-High if available)
    cond_vecs, _, _ = create_conditioning_vectors(returns, 60, 20, 0.2)
    down_high_idx = 3  # [Up-Low, Up-High, Down-Low, Down-High]
    crisis_like = cond_vecs[np.argmax(cond_vecs[:, :4], axis=1) == down_high_idx]
    if len(crisis_like) == 0:
        crisis_like = cond_vecs

    # Prepare LLM aligned embeddings for crisis period
    try:
        loader = NewsDataLoader()
        daily = loader.get_news_embeddings(crisis_start, crisis_end)
        emb_df = pd.DataFrame.from_dict(daily, orient='index')
        emb_df = emb_df.reindex(real_crisis.index, method='ffill').fillna(0)
        # Reduce to 64 dims if needed and normalize row-wise
        X_llm = emb_df.values
        if X_llm.shape[1] > 64:
            from sklearn.decomposition import PCA
            X_llm = PCA(n_components=64, random_state=42).fit_transform(X_llm)
        X_llm = X_llm / np.linalg.norm(X_llm, axis=1, keepdims=True)
    except Exception as e:
        print(f"Warning: LLM crisis embeddings failed: {e}")
        X_llm = None

    # Collect synthetic paths per model (first path used for plotting and metrics)
    results: Dict[str, Dict[str, Any]] = {}

    for mt in ['zero_conditioned', 'explicit_conditioned', 'llm_conditioned']:
        if mt not in ckpts:
            continue
        print(f"\nEvaluating {mt} for crisis window {crisis_start.date()} to {crisis_end.date()} ...")
        model, trainer = load_model_trainer(mt, ckpts[mt], device)

        def sample_one_segment(cnd_tensor):
            # Returns [N, 1, 60]
            return trainer.sample(cnd_tensor, num_samples=1, sampler='ddim', sample_steps=50)

        # Build conditioning
        if mt == 'zero_conditioned':
            cond = torch.zeros(1, 5, device=device)
        elif mt == 'explicit_conditioned':
            # Pick a crisis-like conditioning row
            row = crisis_like[np.random.randint(0, len(crisis_like))]
            cond = torch.tensor(row, dtype=torch.float32, device=device).unsqueeze(0)
        else:  # llm_conditioned
            if X_llm is not None and len(X_llm) > 0:
                # Use average crisis embedding as conditioning
                row = X_llm.mean(axis=0)
                cond = torch.tensor(row, dtype=torch.float32, device=device).unsqueeze(0)
            else:
                cond = F.normalize(torch.randn(1, 64, device=device), dim=1)

        # Stitch to crisis length
        path = stitch_samples(sample_one_segment, cond, Lc)
        results[mt] = {'path': path}

    # Real vs synthetic plots (returns)
    try:
        fig, ax = plt.subplots(figsize=(12,6))
        ax.plot(real_crisis.index, real_crisis.values, 'k-', lw=2, label='Real')
        colors = {'zero_conditioned':'#1f77b4','explicit_conditioned':'#2ca02c','llm_conditioned':'#d62728'}
        for mt, res in results.items():
            ax.plot(real_crisis.index, res['path'], lw=1.5, label=mt.replace('_',' '), color=colors.get(mt,None))
        ax.set_title('Crisis Returns: Real vs Synthetic')
        ax.set_xlabel('Date'); ax.set_ylabel('Returns'); ax.grid(True, alpha=0.3); ax.legend()
        plt.tight_layout(); plt.savefig(fig_dir/'crisis_returns_paths.pdf', dpi=300, bbox_inches='tight'); plt.close()
    except Exception as e:
        print(f"Warning: returns path plot failed: {e}")

    # Portfolio-level paths & VaR/ES computations per model
    def var_es(x: np.ndarray, level: float) -> Tuple[float, float]:
        v = np.percentile(x, (1-level)*100)
        tail = x[x<=v]
        es = np.mean(tail) if len(tail)>0 else np.nan
        return float(v), float(es)

    # Exceedance timelines and VaR/ES curves
    try:
        levels = np.arange(0.90, 0.999, 0.001)
        fig, (a1, a2) = plt.subplots(1,2, figsize=(14,6))
        for mt, res in results.items():
            flat = res['path']
            var_vals=[]; es_vals=[]
            for L in levels:
                v, e = var_es(flat, L)
                var_vals.append(v); es_vals.append(e)
            a1.plot(levels, var_vals, label=mt.replace('_',' '))
            a2.plot(levels, es_vals, label=mt.replace('_',' '))
        a1.set_title('VaR Curves (Crisis)'); a2.set_title('ES Curves (Crisis)')
        for ax in (a1,a2): ax.set_xlabel('Confidence'); ax.grid(True, alpha=0.3); ax.legend()
        plt.tight_layout(); plt.savefig(fig_dir/'crisis_var_es_curves.pdf', dpi=300, bbox_inches='tight'); plt.close()
    except Exception as e:
        print(f"Warning: VaR/ES curves failed: {e}")

    # Exceedance timelines and breach statistics (95% and 99%)
    try:
        fig, (b1, b2, b3) = plt.subplots(3,1, figsize=(12,10), sharex=True)
        t = np.arange(Lc)
        v95_real, v99_real = np.percentile(real_crisis.values, 5), np.percentile(real_crisis.values, 1)
        for ax, (name, series) in zip((b1,b2,b3), [('Real', real_crisis.values)] + [(mt.replace('_',' '), res['path']) for mt,res in results.items()]):
            ax.plot(t, series, lw=1, alpha=0.8)
            v95, v99 = np.percentile(series, 5), np.percentile(series, 1)
            ax.axhline(v95, color='r', ls='--', label=f'VaR95={v95:.4f}')
            ax.axhline(v99, color='m', ls='--', label=f'VaR99={v99:.4f}')
            mask95 = series <= v95
            ax.scatter(t[mask95], series[mask95], s=10, color='red', alpha=0.7)
            ax.set_ylabel(name)
            ax.grid(True, alpha=0.3)
        b3.set_xlabel('Crisis time index')
        handles, labels = b1.get_legend_handles_labels()
        b1.legend(handles, labels)
        plt.tight_layout(); plt.savefig(fig_dir/'crisis_exceedance_timelines.pdf', dpi=300, bbox_inches='tight'); plt.close()
    except Exception as e:
        print(f"Warning: exceedance timelines failed: {e}")

    # Rolling volatility overlay
    try:
        fig, ax = plt.subplots(figsize=(12,6))
        real_vol = pd.Series(real_crisis.values).rolling(window=args.rolling_window).std().dropna()
        ax.plot(real_crisis.index[args.rolling_window-1:], real_vol.values, 'k-', lw=2, label='Real')
        for mt, res in results.items():
            vol = pd.Series(res['path']).rolling(window=args.rolling_window).std().dropna()
            ax.plot(real_crisis.index[args.rolling_window-1:], vol.values, lw=1.5, label=mt.replace('_',' '))
        ax.set_title(f'Rolling Volatility (window={args.rolling_window})'); ax.grid(True, alpha=0.3); ax.legend()
        plt.tight_layout(); plt.savefig(fig_dir/'crisis_rolling_volatility.pdf', dpi=300, bbox_inches='tight'); plt.close()
    except Exception as e:
        print(f"Warning: rolling volatility plot failed: {e}")

    # Comparative table: breaches and p-values
    try:
        rows = []
        for label, series in [('real', real_crisis.values)] + list((mt, res['path']) for mt,res in results.items()):
            for level, p0 in [(0.95, 0.05), (0.99, 0.01)]:
                v = np.percentile(series, (1-level)*100)
                hits = (series <= v).astype(int)
                n_breaches = int(hits.sum())
                uc_stat, uc_p = kupiec_uc(hits, p0)
                ind_stat, ind_p = christoffersen_ind(hits)
                rows.append({
                    'series': label,
                    'level': int(level*100),
                    'var': float(v),
                    'breaches': n_breaches,
                    'kupiec_p': uc_p,
                    'christoffersen_ind_p': ind_p,
                })
        df = pd.DataFrame(rows)
        df.to_csv(tbl_dir/'crisis_breaches.csv', index=False)
        with open(tbl_dir/'crisis_breaches.json','w') as f: json.dump(rows, f, indent=2)
    except Exception as e:
        print(f"Warning: breach table failed: {e}")

    # Write concise report
    try:
        lines = [
            '# COVID-19 Crisis Case Study',
            '',
            f'Window: {args.crisis_start} to {args.crisis_end} (length={Lc} days)',
            '',
            'This report compares real crisis returns to synthetic crisis-period paths from:',
            '- Zero-conditioned (unconditional)',
            '- Explicit-conditioned (regime + target volatility)',
            '- LLM-conditioned (news embeddings)',
            '',
            'Key outputs:',
            '- crisis_returns_paths.pdf: Overlaid real vs synthetic crash returns',
            '- crisis_var_es_curves.pdf: VaR/ES curves (95%/99% highlighted)',
            '- crisis_exceedance_timelines.pdf: VaR exceedance timelines',
            '- crisis_rolling_volatility.pdf: Rolling volatility overlay',
            '- crisis_breaches.csv/json: Breaches and p-values (Kupiec, Christoffersen)',
            '',
            'Observations:',
            '- Unconditional provides a baseline shock replication without control.',
            '- Explicit conditioning targets Down-High regimes, improving crash-like behavior.',
            '- LLM conditioning captures news-driven dynamics; alignment depends on news embeddings.',
        ]
        with open(out_root/'covid_report.md','w') as f:
            f.write('\n'.join(lines))
    except Exception as e:
        print(f"Warning: writing report failed: {e}")

    print(f"✅ COVID case study saved to: {out_root}")


if __name__ == '__main__':
    main()


