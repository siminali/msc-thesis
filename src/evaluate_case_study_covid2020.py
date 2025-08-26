#!/usr/bin/env python3
"""
Case Study: COVID-19 Crash (Feb–Apr 2020) for Novelty Models

Generates crisis-focused evaluation for zero-, explicit-, and LLM-conditioned models.
Outputs:
  results/case_study_covid2020/
    - figures/<model_type>/
    - tables/
    - metrics/{consolidated_metrics.csv,json}
    - evaluation_report.md
    - final_case_study.pdf
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
from datetime import datetime

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn.functional as F

from scipy.stats import chi2
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

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
    p = argparse.ArgumentParser(description='COVID-19 Crash Case Study (Feb–Apr 2020)')
    p.add_argument('--models_dir', type=str, default='results')
    p.add_argument('--results_dir', type=str, default='results/case_study_covid2020')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--crisis_start', type=str, default='2020-02-20')
    p.add_argument('--crisis_end', type=str, default='2020-04-30')
    p.add_argument('--num_paths', type=int, default=20, help='Sample paths per model for overlays')
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
            continue
        runs = [d for d in base.iterdir() if d.is_dir()]
        if not runs:
            continue
        latest = max(runs, key=lambda x: x.stat().st_mtime)
        p = latest / 'checkpoints' / 'best_model.pth'
        if p.exists():
            ckpts[mt] = p
    return ckpts


def load_model_trainer(mt: str, ckpt_path: Path, device: torch.device):
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    if mt in ('zero_conditioned', 'explicit_conditioned'):
        model = ExplicitConditioningDDPM(sequence_length=60, conditioning_dim=5, hidden_dim=128)
        trainer = ExplicitConditioningTrainer(model, num_timesteps=1000, beta_schedule='cosine', device=device, grad_clip=1.0, cfg_p=(0.0 if mt=='zero_conditioned' else 0.1))
    else:
        model = LLMConditionedDiffusion(sequence_length=60, conditioning_dim=64, hidden_dim=128)
        trainer = LLMDiffusionTrainer(model, num_timesteps=1000, beta_schedule='cosine', device=device, grad_clip=1.0, cfg_p=0.1)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, trainer


def stitch_path(sample_fn, cond_tensor, target_len: int) -> np.ndarray:
    segments = []
    remain = target_len
    while remain > 0:
        seg = sample_fn(cond_tensor)  # [1,1,60]
        arr = seg.squeeze(1).detach().cpu().numpy()[0]  # [60]
        segments.append(arr)
        remain -= len(arr)
    path = np.concatenate(segments)[:target_len]
    return path


def compute_training_stats(returns_dec: pd.Series, train_end: pd.Timestamp) -> Tuple[float, float]:
    train_mask = returns_dec.index < train_end
    train = returns_dec[train_mask]
    if len(train) < 30:
        # fallback to full history if insufficient
        train = returns_dec
    return float(train.mean()), float(train.std())


def series_to_percent(series_input: np.ndarray, train_mean: float, train_std: float, assume_standardized: bool, label: str) -> np.ndarray:
    # Convert standardized or decimal series into percent units exactly once
    if assume_standardized:
        series_dec = (series_input * train_std) + train_mean
    else:
        series_dec = series_input
    series_pct = series_dec * 100.0
    # Legacy double-scaling detector
    med_abs = float(np.median(np.abs(series_pct))) if series_pct.size else 0.0
    if med_abs >= 100.0:
        print(f"Warning: Detected potential duplicate percent scaling for {label} (median |%|≈{med_abs:.1f}). Auto-correcting by ÷100.")
        series_pct = series_pct / 100.0
    return series_pct


def enforce_crisis_sanity(name: str, series_pct: np.ndarray, window: int = 20, max_abs_ret: float = 25.0, max_daily_vol: float = 15.0):
    if series_pct.size == 0:
        return
    if np.any(np.abs(series_pct) > max_abs_ret):
        raise RuntimeError(f"Sanity check failed for {name}: absolute daily return exceeded {max_abs_ret}%. Check for duplicate ×100 or missing inverse transform.")
    vol = pd.Series(series_pct).rolling(window=window).std().dropna().values
    if vol.size and np.max(vol) > max_daily_vol:
        raise RuntimeError(f"Sanity check failed for {name}: daily rolling volatility exceeded {max_daily_vol}% (window={window}). Check scaling and inverse transform.")


def kupiec_uc(hits: np.ndarray, p0: float) -> Tuple[float, float]:
    n = len(hits); v = int(hits.sum())
    p1 = v/n if n>0 else 0.0
    if n==0 or p1<=0 or p1>=1:
        return np.nan, np.nan
    lr = ((1-p0)**(n-v) * (p0**v)) / ((1-p1)**(n-v) * (p1**v))
    stat = -2*np.log(lr); pval = 1-chi2.cdf(stat, 1)
    return float(stat), float(pval)


def christoffersen_ind(hits: np.ndarray) -> Tuple[float, float]:
    if len(hits)<2: return np.nan, np.nan
    n00=n01=n10=n11=0
    for i in range(1,len(hits)):
        a,b=hits[i-1],hits[i]
        if a==0 and b==0: n00+=1
        if a==0 and b==1: n01+=1
        if a==1 and b==0: n10+=1
        if a==1 and b==1: n11+=1
    pi0 = n01/(n00+n01) if (n00+n01)>0 else 0.0
    pi1 = n11/(n10+n11) if (n10+n11)>0 else 0.0
    pi = (n01+n11)/(n00+n01+n10+n11) if (n00+n01+n10+n11)>0 else 0.0
    L1 = ((1-pi0)**n00)*(pi0**n01)*((1-pi1)**n10)*(pi1**n11)
    L0 = ((1-pi)**(n00+n10))*(pi**(n01+n11))
    if L0<=0 or L1<=0: return np.nan, np.nan
    stat = -2*np.log(L0/L1); p = 1-chi2.cdf(stat, 1)
    return float(stat), float(p)


def main():
    args = parse_args()
    set_determinism(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Output dirs
    out_root = Path(args.results_dir)
    fig_dir = out_root / 'figures'; tbl_dir = out_root / 'tables'; mtr_dir = out_root / 'metrics'
    for d in [fig_dir, tbl_dir, mtr_dir]: d.mkdir(parents=True, exist_ok=True)
    for mt in ['zero_conditioned','explicit_conditioned','llm_conditioned','combined']:
        (fig_dir/mt).mkdir(parents=True, exist_ok=True)
    
    # Data and crisis window
    # Load raw returns in decimals (not percent)
    returns_dec = load_returns()
    # Determine training stats for inverse transform reference (use pre-crisis end)
    train_mean, train_std = compute_training_stats(returns_dec, pd.Timestamp(args.crisis_start))
    # Convert real crisis window to percent with single pipeline
    returns_pct = series_to_percent(returns_dec.values, train_mean, train_std, assume_standardized=False, label='Real')
    returns_pct = pd.Series(returns_pct, index=returns_dec.index)
    crisis_start = pd.Timestamp(args.crisis_start)
    crisis_end = pd.Timestamp(args.crisis_end)
    mask = (returns_dec.index>=crisis_start) & (returns_dec.index<=crisis_end)
    real_crisis = returns_pct[mask]
    if len(real_crisis)==0: raise ValueError('No data in crisis window')
    Lc = len(real_crisis)

    # Checkpoints
    ckpts = discover_checkpoints(Path(args.models_dir))
    if not ckpts:
        print('❌ No checkpoints found.'); return

    # Conditioning utilities
    explicit_cond, _, _ = create_conditioning_vectors(returns_dec, 60, 20, 0.2)
    down_high_idx = 3
    crisis_like = explicit_cond[np.argmax(explicit_cond[:,:4], axis=1)==down_high_idx]
    if len(crisis_like)==0: crisis_like = explicit_cond

    # LLM crisis embeddings and sentiment buckets (PCA-1 sign proxy)
    try:
        loader = NewsDataLoader()
        daily = loader.get_news_embeddings(crisis_start, crisis_end)
        emb_df = pd.DataFrame.from_dict(daily, orient='index')
        emb_df = emb_df.reindex(real_crisis.index, method='ffill').fillna(0)
        X_llm = emb_df.values
        # sentiment proxy by first PC sign
        from sklearn.decomposition import PCA
        pc1 = PCA(n_components=1, random_state=42).fit_transform(X_llm).squeeze()
        sentiment_bucket = np.where(pc1>=0, 'positive', 'negative')
        # reduce to at most 64 dims, respecting sample limit, then zero-pad to 64
        k = min(64, X_llm.shape[1], max(1, X_llm.shape[0]-1))
        if X_llm.shape[1] != 64:
            Xr = PCA(n_components=k, random_state=42).fit_transform(X_llm)
        else:
            Xr = X_llm
        if Xr.shape[1] < 64:
            pad = np.zeros((Xr.shape[0], 64 - Xr.shape[1]))
            Xr = np.concatenate([Xr, pad], axis=1)
        X_llm = Xr / (np.linalg.norm(Xr, axis=1, keepdims=True)+1e-12)
    except Exception as e:
        print(f"Warning: LLM crisis embeddings failed: {e}")
        X_llm = None; sentiment_bucket = None

    # Generate multiple paths per model
    model_paths: Dict[str, List[np.ndarray]] = {k:[] for k in ['zero_conditioned','explicit_conditioned','llm_conditioned']}
    for mt in model_paths.keys():
        if mt not in ckpts: continue
        model, trainer = load_model_trainer(mt, ckpts[mt], device)
        def sample_seg(cond):
            return trainer.sample(cond, num_samples=1, sampler='ddim', sample_steps=50)
        for _ in range(args.num_paths):
            if mt=='zero_conditioned':
                cond = torch.zeros(1,5,device=device)
            elif mt=='explicit_conditioned':
                row = crisis_like[np.random.randint(0,len(crisis_like))]
                cond = torch.tensor(row, dtype=torch.float32, device=device).unsqueeze(0)
            else:
                if X_llm is not None and len(X_llm)>0:
                    # mean crisis embedding (64-d, normalized)
                    row = X_llm.mean(axis=0)
                    cond = torch.tensor(row, dtype=torch.float32, device=device).unsqueeze(0)
                else:
                    # fallback: zero embedding to avoid extreme amplitudes under uncertainty
                    cond = torch.zeros(1,64, device=device)
            path_std = stitch_path(sample_seg, cond, Lc)  # assume standardized like training output
            path_pct = series_to_percent(path_std, train_mean, train_std, assume_standardized=True, label=f'{mt} synthetic')
            model_paths[mt].append(path_pct)

    # Hard sanity checks (real and first N paths per model)
    enforce_crisis_sanity('Real', real_crisis.values, window=args.rolling_window)
    for mt, paths in model_paths.items():
        for i, p in enumerate(paths[:min(5, len(paths))]):
            enforce_crisis_sanity(f'{mt} path {i+1}', p, window=args.rolling_window)

    # Shared y-limits for crisis returns and VaR/ES overlays
    all_series = [real_crisis.values] + [p for paths in model_paths.values() for p in paths]
    if len(all_series):
        extrema = float(np.max(np.abs(np.concatenate(all_series))))
    else:
        extrema = 15.0
    ylim = min(20.0, max(15.0, float(np.ceil(extrema * 1.05))))

    def label_map(name: str) -> str:
        return {
            'zero_conditioned': 'Zero-conditioned',
            'explicit_conditioned': 'Explicit-conditioned',
            'llm_conditioned': 'LLM-conditioned',
        }.get(name, name)

    # Outlier check after percent scaling
    def outlier_warning(series_list: List[np.ndarray], thr: float = 20.0) -> bool:
        for s in series_list:
            if np.any(np.abs(s) > thr):
                return True
        return False

    # Create case-study figures directory with canonical names
    case_fig_dir = fig_dir / 'case_study'
    case_fig_dir.mkdir(parents=True, exist_ok=True)

    # Overlay return paths (multiple per model)
    plt.style.use('seaborn-v0_8-white')
    colors={'zero_conditioned':'#1f77b4','explicit_conditioned':'#2ca02c','llm_conditioned':'#d62728','real':'#000000'}
    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(real_crisis.index, real_crisis.values, color=colors['real'], lw=2, label='Real')
    for mt, paths in model_paths.items():
        for j, p in enumerate(paths[:min(5,len(paths))]):
            ax.plot(real_crisis.index, p, color=colors[mt], lw=1, alpha=0.7, label=label_map(mt) if j==0 else '')
    ax.set_title('COVID-19 Crisis Returns: Real vs Synthetic'); ax.set_xlabel('Date'); ax.set_ylabel('Returns (%)'); ax.grid(True, alpha=0.3)
    ax.set_ylim([-ylim, ylim])
    ax.legend(ncol=2)
    # Caption and outlier annotation
    caption = f"COVID-19 window ({args.crisis_start} to {args.crisis_end}); returns in %."
    fig.text(0.01, 0.01, caption, fontsize=9)
    if outlier_warning([real_crisis.values] + [p for paths in model_paths.values() for p in paths]):
        fig.text(0.01, 0.03, 'WARNING: Outlier >20% detected; verify scaling and data preprocessing.', color='red', fontsize=9)
    plt.tight_layout(); plt.savefig(fig_dir/'combined'/'crisis_paths_overlay.pdf', dpi=300, bbox_inches='tight'); plt.close()
    # Export canonical name
    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(real_crisis.index, real_crisis.values, color=colors['real'], lw=2, label='Real')
    for mt, paths in model_paths.items():
        for j, p in enumerate(paths[:min(5,len(paths))]):
            ax.plot(real_crisis.index, p, color=colors[mt], lw=1, alpha=0.7, label=label_map(mt) if j==0 else '')
    ax.set_title('COVID-19 Crisis Returns: Real vs Synthetic'); ax.set_xlabel('Date'); ax.set_ylabel('Returns (%)'); ax.grid(True, alpha=0.3)
    ax.legend(ncol=2); fig.text(0.01, 0.01, caption, fontsize=9)
    if outlier_warning([real_crisis.values] + [p for paths in model_paths.values() for p in paths]):
        fig.text(0.01, 0.03, 'WARNING: Outlier >20% detected; verify scaling and data preprocessing.', color='red', fontsize=9)
    ax.set_ylim([-ylim, ylim])
    # Explicit statement if Zero-conditioned underestimates crisis volatility
    if len(model_paths['zero_conditioned']):
        z_std = float(np.std(np.concatenate(model_paths['zero_conditioned'])))
        r_std = float(np.std(real_crisis.values))
        if z_std < r_std:
            fig.text(0.01, 0.05, 'Finding: Zero-conditioned underestimates crisis volatility relative to Real.', fontsize=9)
    plt.tight_layout(); plt.savefig(case_fig_dir/'covid_returns_overlay.pdf', dpi=300, bbox_inches='tight'); plt.close()

    # Real-only context figure
    fig, ax = plt.subplots(figsize=(12,4))
    ax.plot(real_crisis.index, real_crisis.values, color=colors['real'], lw=2)
    ax.set_title('Real crisis returns (context)'); ax.set_xlabel('Date'); ax.set_ylabel('Returns (%)'); ax.grid(True, alpha=0.3)
    ax.set_ylim([-ylim, ylim])
    fig.text(0.01,0.01, caption, fontsize=9)
    plt.tight_layout(); plt.savefig(case_fig_dir/'covid_returns_real_context.pdf', dpi=300, bbox_inches='tight'); plt.close()

    # VaR/ES curves (95/99) for crisis window (percent units)
    levels = np.arange(0.90, 0.999, 0.001)
    fig, (a1,a2) = plt.subplots(1,2, figsize=(14,6))
    flat_real = real_crisis.values
    vr=[]; er=[]
    for L in levels:
        v = np.percentile(flat_real, (1-L)*100)
        vr.append(v)
        tail = flat_real[flat_real<=v]
        er.append(np.mean(tail) if len(tail)>0 else np.nan)
    a1.plot(levels, vr, color=colors['real'], linestyle='--', label='Real'); a2.plot(levels, er, color=colors['real'], linestyle='--', label='Real')
    for mt, paths in model_paths.items():
        flat = np.concatenate(paths) if len(paths)>0 else np.array([])
        if len(flat)==0: continue
        vv=[]; ee=[]
        for L in levels:
            v = np.percentile(flat, (1-L)*100)
            vv.append(v); tail = flat[flat<=v]; ee.append(np.mean(tail) if len(tail)>0 else np.nan)
        a1.plot(levels, vv, color=colors[mt], label=mt.replace('_',' '))
        a2.plot(levels, ee, color=colors[mt], label=mt.replace('_',' '))
    a1.set_title('Crisis VaR Curve'); a2.set_title('Crisis ES Curve')
    for ax in (a1,a2): ax.set_xlabel('Confidence'); ax.set_ylabel('Returns (%)'); ax.grid(True, alpha=0.3); ax.legend()
    plt.tight_layout(); plt.savefig(fig_dir/'combined'/'crisis_var_es_curves.pdf', dpi=300, bbox_inches='tight'); plt.close()
    # Canonical export
    fig, (a1,a2) = plt.subplots(1,2, figsize=(14,6))
    a1.plot(levels, vr, color=colors['real'], linestyle='--', label='Real'); a2.plot(levels, er, color=colors['real'], linestyle='--', label='Real')
    for mt, paths in model_paths.items():
        flat = np.concatenate(paths) if len(paths)>0 else np.array([])
        if len(flat)==0: continue
        vv=[]; ee=[]
        for L in levels:
            v = np.percentile(flat, (1-L)*100)
            vv.append(v); tail = flat[flat<=v]; ee.append(np.mean(tail) if len(tail)>0 else np.nan)
        a1.plot(levels, vv, color=colors[mt], label=mt.replace('_',' '))
        a2.plot(levels, ee, color=colors[mt], label=mt.replace('_',' '))
    a1.set_title('Crisis VaR Curve'); a2.set_title('Crisis ES Curve')
    for ax in (a1,a2): ax.set_xlabel('Confidence'); ax.set_ylabel('Returns (%)'); ax.grid(True, alpha=0.3); ax.legend()
    fig.text(0.01, 0.01, f"Returns in %, COVID-19 window {args.crisis_start} to {args.crisis_end}", fontsize=9)
    plt.tight_layout(); plt.savefig(case_fig_dir/'covid_var_es_curves.pdf', dpi=300, bbox_inches='tight'); plt.close()

    # Separate ES-only export with same styling (for requested file name)
    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(levels, er, color=colors['real'], linestyle='--', label='Real')
    for mt, paths in model_paths.items():
        flat = np.concatenate(paths) if len(paths)>0 else np.array([])
        if len(flat)==0: continue
        ee=[]
        for L in levels:
            v = np.percentile(flat, (1-L)*100)
            tail = flat[flat<=v]
            ee.append(np.mean(tail) if len(tail)>0 else np.nan)
        ax.plot(levels, ee, color=colors[mt], label=mt.replace('_',' '))
    ax.set_title('Crisis ES Curve'); ax.set_xlabel('Confidence'); ax.set_ylabel('Returns (%)'); ax.grid(True, alpha=0.3); ax.legend()
    fig.text(0.01, 0.01, f"Returns in %, COVID-19 window {args.crisis_start} to {args.crisis_end}", fontsize=9)
    plt.tight_layout(); plt.savefig(case_fig_dir/'covid_es_95_99.pdf', dpi=300, bbox_inches='tight'); plt.close()

    # Exceedance timelines (95/99) with diagnostics for representative path per model
    fig, axes = plt.subplots(4,1, figsize=(12,12), sharex=True)
    t = np.arange(Lc)
    v95r, v99r = np.percentile(flat_real,5), np.percentile(flat_real,1)
    axes[0].plot(t, flat_real, lw=1); axes[0].axhline(v95r,color='r',ls='--'); axes[0].axhline(v99r,color='m',ls='--'); axes[0].set_ylabel('Real (%)')
    hits95=(flat_real<=v95r).astype(int); hits99=(flat_real<=v99r).astype(int)
    _, kp95 = kupiec_uc(hits95,0.05); _, ip95 = christoffersen_ind(hits95)
    _, kp99 = kupiec_uc(hits99,0.01); _, ip99 = christoffersen_ind(hits99)
    exp95 = 0.05 * Lc; exp99 = 0.01 * Lc
    axes[0].text(0.01, 0.95, f"Real — 95% breaches: obs = {hits95.sum()} / N = {Lc} (exp ≈ p·N = {exp95:.1f} at p = 0.95), Kupiec p={kp95:.3f}, Christoffersen p={ip95:.3f}\n99% breaches: obs = {hits99.sum()} / N = {Lc} (exp ≈ p·N = {exp99:.1f} at p = 0.99), Kupiec p={kp99:.3f}, Christoffersen p={ip99:.3f}", transform=axes[0].transAxes, fontsize=10, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    axes[0].set_ylim([-ylim, ylim])
    for idx, mt in enumerate(['zero_conditioned','explicit_conditioned','llm_conditioned'], start=1):
        if len(model_paths[mt])==0: continue
        x = model_paths[mt][0]
        v95, v99 = np.percentile(x,5), np.percentile(x,1)
        axes[idx].plot(t, x, lw=1); axes[idx].axhline(v95,color='r',ls='--'); axes[idx].axhline(v99,color='m',ls='--'); axes[idx].set_ylabel(f"{label_map(mt)} (%)")
        h95=(x<=v95).astype(int); h99=(x<=v99).astype(int)
        _, kp95 = kupiec_uc(h95,0.05); _, ip95 = christoffersen_ind(h95)
        _, kp99 = kupiec_uc(h99,0.01); _, ip99 = christoffersen_ind(h99)
        axes[idx].text(0.01, 0.95, f"{label_map(mt)} — 95% breaches: obs = {h95.sum()} / N = {Lc} (exp ≈ p·N = {exp95:.1f} at p = 0.95), Kupiec p={kp95:.3f}, Christoffersen p={ip95:.3f}\n99% breaches: obs = {h99.sum()} / N = {Lc} (exp ≈ p·N = {exp99:.1f} at p = 0.99), Kupiec p={kp99:.3f}, Christoffersen p={ip99:.3f}", transform=axes[idx].transAxes, fontsize=10, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        axes[idx].set_ylim([-ylim, ylim])
    axes[-1].set_xlabel('Crisis time index')
    for ax in axes: ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(fig_dir/'combined'/'crisis_exceedance_timelines.pdf', dpi=300, bbox_inches='tight'); plt.close()
    # Canonical export name
    fig, axes = plt.subplots(4,1, figsize=(12,12), sharex=True)
    axes[0].plot(t, flat_real, lw=1); axes[0].axhline(v95r,color='r',ls='--'); axes[0].axhline(v99r,color='m',ls='--'); axes[0].set_ylabel('Real (%)')
    hits95=(flat_real<=v95r).astype(int); hits99=(flat_real<=v99r).astype(int)
    _, kp95 = kupiec_uc(hits95,0.05); _, ip95 = christoffersen_ind(hits95)
    _, kp99 = kupiec_uc(hits99,0.01); _, ip99 = christoffersen_ind(hits99)
    exp95 = 0.05 * Lc; exp99 = 0.01 * Lc
    axes[0].text(0.01, 0.95, f"Real — 95% breaches: obs = {hits95.sum()} / N = {Lc} (exp ≈ p·N = {exp95:.1f} at p = 0.95), Kupiec p={kp95:.3f}, Christoffersen p={ip95:.3f}\n99% breaches: obs = {hits99.sum()} / N = {Lc} (exp ≈ p·N = {exp99:.1f} at p = 0.99), Kupiec p={kp99:.3f}, Christoffersen p={ip99:.3f}", transform=axes[0].transAxes, fontsize=10, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    axes[0].set_ylim([-ylim, ylim])
    for idx, mt in enumerate(['zero_conditioned','explicit_conditioned','llm_conditioned'], start=1):
        if len(model_paths[mt])==0: continue
        x = model_paths[mt][0]
        v95, v99 = np.percentile(x,5), np.percentile(x,1)
        axes[idx].plot(t, x, lw=1); axes[idx].axhline(v95,color='r',ls='--'); axes[idx].axhline(v99,color='m',ls='--'); axes[idx].set_ylabel(f"{label_map(mt)} (%)")
        h95=(x<=v95).astype(int); h99=(x<=v99).astype(int)
        _, kp95 = kupiec_uc(h95,0.05); _, ip95 = christoffersen_ind(h95)
        _, kp99 = kupiec_uc(h99,0.01); _, ip99 = christoffersen_ind(h99)
        axes[idx].text(0.01, 0.95, f"{label_map(mt)} — 95% breaches: obs = {h95.sum()} / N = {Lc} (exp ≈ p·N = {exp95:.1f} at p = 0.95), Kupiec p={kp95:.3f}, Christoffersen p={ip95:.3f}\n99% breaches: obs = {h99.sum()} / N = {Lc} (exp ≈ p·N = {exp99:.1f} at p = 0.99), Kupiec p={kp99:.3f}, Christoffersen p={ip99:.3f}", transform=axes[idx].transAxes, fontsize=10, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        axes[idx].set_ylim([-ylim, ylim])
    axes[-1].set_xlabel('Crisis time index')
    fig.text(0.01,0.01, f"Returns and VaR/ES in %, shared y-limits across panels; COVID-19 window {args.crisis_start} to {args.crisis_end}", fontsize=9)
    for ax in axes: ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(case_fig_dir/'covid_var_95_99.pdf', dpi=300, bbox_inches='tight'); plt.close()

    # Portfolio-level comparison (equal-weight across sample paths)
    rows=[]
    for label, series in [('real', flat_real)]:
        for level,p0 in [(0.95,0.05),(0.99,0.01)]:
            v = np.percentile(series,(1-level)*100)
            hits = (series<=v).astype(int)
            uc_stat, uc_p = kupiec_uc(hits,p0); ind_stat, ind_p = christoffersen_ind(hits)
            rows.append({'model_type':label,'level':int(level*100),'var':float(v),'breaches':int(hits.sum()),'kupiec_p':uc_p,'christoffersen_ind_p':ind_p})
    for mt, paths in model_paths.items():
        if len(paths)==0: continue
        # Equal-weight portfolio across first K paths
        K = min(10, len(paths))
        P = np.mean(np.stack(paths[:K], axis=0), axis=0)
        fig, ax = plt.subplots(figsize=(12,5))
        ax.plot(real_crisis.index, P, lw=1, label=f'{mt} portfolio')
        for level in [0.95,0.99]:
            v = np.percentile(P, (1-level)*100)
            ax.axhline(v, color='r' if level==0.95 else 'm', ls='--', label=f'VaR{int(level*100)}')
        ax.set_title(f'Portfolio Losses with VaR Overlays - {mt}'); ax.set_ylabel('Returns (%)'); ax.grid(True, alpha=0.3); ax.legend()
        plt.tight_layout(); plt.savefig(fig_dir/mt/f'portfolio_var_overlays.pdf', dpi=300, bbox_inches='tight'); plt.close()
        for level,p0 in [(0.95,0.05),(0.99,0.01)]:
            v = np.percentile(P,(1-level)*100); hits=(P<=v).astype(int)
            uc_stat, uc_p = kupiec_uc(hits,p0); ind_stat, ind_p = christoffersen_ind(hits)
            rows.append({'model_type':mt,'level':int(level*100),'var':float(v),'breaches':int(hits.sum()),'kupiec_p':uc_p,'christoffersen_ind_p':ind_p})
    df = pd.DataFrame(rows)
    df.to_csv(tbl_dir/'portfolio_breaches.csv', index=False)
    with open(tbl_dir/'portfolio_breaches.json','w') as f: json.dump(rows,f,indent=2)

    # Explicit model regime grid (crash-conditioned)
    if len(model_paths['explicit_conditioned']):
        fig, axes = plt.subplots(2,2, figsize=(14,8), sharex=True, sharey=True)
        regimes=['Up-Low','Up-High','Down-Low','Down-High']
        for idx,name in enumerate(regimes):
            r = idx//2; c=idx%2
            ax=axes[r][c]
            # pick rows matching regime; fallback to generic
            mask = np.argmax(explicit_cond[:,:4], axis=1)==idx
            base = explicit_cond[mask] if mask.any() else explicit_cond
            # sample a few using those conditions
            model, trainer = load_model_trainer('explicit_conditioned', ckpts['explicit_conditioned'], device)
            def sample_seg(cond): return trainer.sample(cond, num_samples=1, sampler='ddim', sample_steps=50)
            for _ in range(5):
                row = base[np.random.randint(0,len(base))]
                cond = torch.tensor(row, dtype=torch.float32, device=device).unsqueeze(0)
                p = stitch_path(sample_seg, cond, Lc)
                p = series_to_percent(p, train_mean, train_std, assume_standardized=True, label='explicit regime')
                ax.plot(real_crisis.index, p, lw=1, alpha=0.7)
            nobs = 5 * Lc
            ax.set_title(f"{name} (n={Lc} days, nobs={nobs} synthetic observations across paths)")
            ax.set_ylabel('Returns (%)'); ax.grid(True, alpha=0.3); ax.set_ylim([-ylim, ylim])
        plt.tight_layout(); plt.savefig(fig_dir/'explicit_conditioned'/'regime_grid_crash.pdf', dpi=300, bbox_inches='tight'); plt.close()
        # Canonical export
        fig, axes = plt.subplots(2,2, figsize=(14,8), sharex=True, sharey=True)
        for idx,name in enumerate(regimes):
            r = idx//2; c=idx%2
            ax=axes[r][c]
            mask = np.argmax(explicit_cond[:,:4], axis=1)==idx
            base = explicit_cond[mask] if mask.any() else explicit_cond
            model, trainer = load_model_trainer('explicit_conditioned', ckpts['explicit_conditioned'], device)
            def sample_seg(cond): return trainer.sample(cond, num_samples=1, sampler='ddim', sample_steps=50)
            for _ in range(5):
                row = base[np.random.randint(0,len(base))]
                cond = torch.tensor(row, dtype=torch.float32, device=device).unsqueeze(0)
                p = stitch_path(sample_seg, cond, Lc)
                p = series_to_percent(p, train_mean, train_std, assume_standardized=True, label='explicit regime')
                ax.plot(real_crisis.index, p, lw=1, alpha=0.7)
            nobs = 5 * Lc
            ax.set_title(f"{name} (n={Lc} days, nobs={nobs} synthetic observations across paths)")
            ax.set_ylabel('Returns (%)'); ax.grid(True, alpha=0.3); ax.set_ylim([-ylim, ylim])
        plt.tight_layout(); plt.savefig(case_fig_dir/'covid_regimes.pdf', dpi=300, bbox_inches='tight'); plt.close()

    # LLM sentiment-bucket comparison (vol distribution)
    if sentiment_bucket is not None and len(model_paths['llm_conditioned']):
        vols={'positive':[],'negative':[]}
        for mt in ['llm_conditioned']:
            for p in model_paths[mt][:min(10,len(model_paths[mt]))]:
                v = pd.Series(p).rolling(window=args.rolling_window).std().dropna().mean()
                # assign by median pc1 sign proportion; simple proxy
                bucket = 'positive' if (np.mean(pc1)>=0) else 'negative'
                vols[bucket].append(v)
        fig, ax = plt.subplots(figsize=(8,4))
        ax.hist(vols['positive'], bins=20, histtype='step', label='Positive', color='#2ca02c')
        ax.hist(vols['negative'], bins=20, histtype='step', label='Negative', color='#d62728')
        ax.set_title('LLM Sentiment Buckets: Realized Volatility'); ax.grid(True, alpha=0.3); ax.legend()
        plt.tight_layout(); plt.savefig(fig_dir/'llm_conditioned'/'sentiment_bucket_volatility.pdf', dpi=300, bbox_inches='tight'); plt.close()

    # Stylised facts under crisis: histogram (log y, line), ECDF, QQ tails, rolling vol
    # Histogram (percent units, y=log count)
    pooled = [flat_real] + [np.concatenate(paths) for paths in model_paths.values() if len(paths)>0]
    pooled_flat = np.concatenate(pooled) if len(pooled)>0 else flat_real
    bins = np.linspace(np.percentile(pooled_flat, 0.5), np.percentile(pooled_flat, 99.5), 80)
    fig, ax = plt.subplots(figsize=(10,6))
    ax.set_yscale('log')
    hr, br = np.histogram(flat_real, bins=bins, density=False)
    cr = (br[:-1]+br[1:])/2
    ax.plot(cr, hr, color='black', linestyle='--', label='Real')
    for mt,paths in model_paths.items():
        flat = np.concatenate(paths) if len(paths)>0 else np.array([])
        if len(flat)==0: continue
        hh,bb=np.histogram(flat,bins=bins,density=False); cc=(bb[:-1]+bb[1:])/2
        ax.plot(cc,hh,label=mt.replace('_',' '))
    ax.set_title('Crisis Histogram'); ax.set_xlabel('Returns (%)'); ax.set_ylabel('Count (log scale)'); ax.grid(True, alpha=0.3); ax.legend()
    fig.text(0.01,0.01, f"COVID-19 window {args.crisis_start} to {args.crisis_end}; histogram y-axis is log; units are %.", fontsize=9)
    plt.tight_layout(); plt.savefig(fig_dir/'combined'/'crisis_hist_logy.pdf', dpi=300, bbox_inches='tight'); plt.close()
    # Canonical export
    fig, ax = plt.subplots(figsize=(10,6))
    ax.set_yscale('log')
    ax.plot(cr, hr, color='black', linestyle='--', label='Real')
    for mt,paths in model_paths.items():
        flat = np.concatenate(paths) if len(paths)>0 else np.array([])
        if len(flat)==0: continue
        hh,bb=np.histogram(flat,bins=bins,density=False); cc=(bb[:-1]+bb[1:])/2
        ax.plot(cc,hh,label=mt.replace('_',' '))
    ax.set_title('Crisis Histogram'); ax.set_xlabel('Returns (%)'); ax.set_ylabel('Count (log scale)'); ax.grid(True, alpha=0.3); ax.legend()
    fig.text(0.01,0.01, f"COVID-19 window {args.crisis_start} to {args.crisis_end}; histogram y-axis is log; units are %.", fontsize=9)
    plt.tight_layout(); plt.savefig(case_fig_dir/'covid_loghist.pdf', dpi=300, bbox_inches='tight'); plt.close()
    # ECDF
    fig, ax = plt.subplots(figsize=(10,6))
    sr = np.sort(flat_real); yr=np.arange(1,len(sr)+1)/len(sr); ax.plot(sr,yr,label='Real',color='black',linestyle='--')
    for mt,paths in model_paths.items():
        flat = np.concatenate(paths) if len(paths)>0 else np.array([])
        if len(flat)==0: continue
        sg=np.sort(flat); yg=np.arange(1,len(sg)+1)/len(sg); ax.plot(sg,yg,label=mt.replace('_',' '))
    ax.set_title('Crisis ECDF Overlay'); ax.set_xlabel('Returns (%)'); ax.set_ylabel('ECDF'); ax.grid(True, alpha=0.3); ax.legend()
    plt.tight_layout(); plt.savefig(fig_dir/'combined'/'crisis_ecdf_overlay.pdf', dpi=300, bbox_inches='tight'); plt.close()
    # Canonical export
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(sr,yr,label='Real',color='black',linestyle='--')
    for mt,paths in model_paths.items():
        flat = np.concatenate(paths) if len(paths)>0 else np.array([])
        if len(flat)==0: continue
        sg=np.sort(flat); yg=np.arange(1,len(sg)+1)/len(sg); ax.plot(sg,yg,label=mt.replace('_',' '))
    ax.set_title('Crisis ECDF Overlay'); ax.set_xlabel('Returns (%)'); ax.set_ylabel('ECDF'); ax.grid(True, alpha=0.3); ax.legend()
    fig.text(0.01,0.01, f"COVID-19 window {args.crisis_start} to {args.crisis_end}; units are %.", fontsize=9)
    plt.tight_layout(); plt.savefig(case_fig_dir/'covid_ecdf.pdf', dpi=300, bbox_inches='tight'); plt.close()
    # QQ tails
    # Q-Q plots with shared axes across models using pooled limits
    pooled_all = [flat_real] + [np.concatenate(paths) for mt,paths in model_paths.items() if len(paths)>0]
    pooled_flat_all = np.concatenate(pooled_all)
    ql = np.percentile(pooled_flat_all, 0.5); qr = np.percentile(pooled_flat_all, 99.5)
    left_lim = 1.05*abs(min(0.0, ql))
    right_lim = 1.05*max(0.0, qr)
    from scipy import stats as spstats
    # Left tail panel
    fig_left, axes_left = plt.subplots(2,2, figsize=(12,10), sharex=True, sharey=True)
    entries = [('Real', flat_real), ('Zero', np.concatenate(model_paths['zero_conditioned']) if len(model_paths['zero_conditioned']) else np.array([])), ('Explicit', np.concatenate(model_paths['explicit_conditioned']) if len(model_paths['explicit_conditioned']) else np.array([])), ('LLM', np.concatenate(model_paths['llm_conditioned']) if len(model_paths['llm_conditioned']) else np.array([]))]
    for i,(lab,flat) in enumerate(entries):
        if flat.size==0: continue
        ax = axes_left[i//2][i%2]
        left = flat[flat<np.percentile(flat,10)]
        (osm, osr), (slope, intercept, _) = spstats.probplot(left, dist='norm')
        ax.plot(osm, osr, 'o', ms=2); ax.plot(osm, slope*osm+intercept, 'r--', lw=1)
        ax.set_title(lab)
        ax.set_xlim([-left_lim, 0]); ax.set_ylim([-left_lim, 0])
        ax.set_xlabel('Theoretical quantiles (%)'); ax.set_ylabel('Ordered values (%)')
        ax.grid(True, alpha=0.3)
    fig_left.suptitle('Q-Q Left Tail (shared axes, units: %)')
    fig_left.text(0.01,0.01, f"COVID-19 window {args.crisis_start} to {args.crisis_end}", fontsize=9)
    plt.tight_layout(); fig_left.savefig(case_fig_dir/'covid_qq_left.pdf', dpi=300, bbox_inches='tight'); plt.close(fig_left)
    # Right tail panel
    fig_right, axes_right = plt.subplots(2,2, figsize=(12,10), sharex=True, sharey=True)
    for i,(lab,flat) in enumerate(entries):
        if flat.size==0: continue
        ax = axes_right[i//2][i%2]
        right = flat[flat>np.percentile(flat,90)]
        (osm, osr), (slope, intercept, _) = spstats.probplot(right, dist='norm')
        ax.plot(osm, osr, 'o', ms=2); ax.plot(osm, slope*osm+intercept, 'r--', lw=1)
        ax.set_title(lab)
        ax.set_xlim([0, right_lim]); ax.set_ylim([0, right_lim])
        ax.set_xlabel('Theoretical quantiles (%)'); ax.set_ylabel('Ordered values (%)')
        ax.grid(True, alpha=0.3)
    fig_right.suptitle('Q-Q Right Tail (shared axes, units: %)')
    fig_right.text(0.01,0.01, f"COVID-19 window {args.crisis_start} to {args.crisis_end}", fontsize=9)
    plt.tight_layout(); fig_right.savefig(case_fig_dir/'covid_qq_right.pdf', dpi=300, bbox_inches='tight'); plt.close(fig_right)
    # Rolling volatility
    fig, ax = plt.subplots(figsize=(12,6))
    rv = pd.Series(flat_real).rolling(window=args.rolling_window).std().dropna()
    ax.plot(real_crisis.index[args.rolling_window-1:], rv.values, color='black', lw=2, label='Real')
    for mt, paths in model_paths.items():
        if len(paths)==0: continue
        gv = pd.Series(paths[0]).rolling(window=args.rolling_window).std().dropna()
        ax.plot(real_crisis.index[args.rolling_window-1:], gv.values, label=mt.replace('_',' '))
    ax.set_title(f'Crisis Rolling Volatility (daily %, window={args.rolling_window}, not annualized)'); ax.set_xlabel('Date'); ax.set_ylabel('Volatility (daily %, window=20)'); ax.grid(True, alpha=0.3); ax.legend()
    plt.tight_layout(); plt.savefig(fig_dir/'combined'/'crisis_rolling_volatility.pdf', dpi=300, bbox_inches='tight'); plt.close()
    # Canonical export
    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(real_crisis.index[args.rolling_window-1:], rv.values, color='black', lw=2, label='Real')
    for mt, paths in model_paths.items():
        if len(paths)==0: continue
        gv = pd.Series(paths[0]).rolling(window=args.rolling_window).std().dropna()
        ax.plot(real_crisis.index[args.rolling_window-1:], gv.values, label=mt.replace('_',' '))
    ax.set_title(f'Crisis Rolling Volatility (daily %, window={args.rolling_window}, not annualized)'); ax.set_xlabel('Date'); ax.set_ylabel('Volatility (daily %, window=20)'); ax.grid(True, alpha=0.3); ax.legend()
    fig.text(0.01,0.01, f"COVID-19 window {args.crisis_start} to {args.crisis_end}; units are daily %; not annualized.", fontsize=9)
    plt.tight_layout(); plt.savefig(case_fig_dir/'covid_vol_rolling.pdf', dpi=300, bbox_inches='tight'); plt.close()

    # Consolidated metrics (crisis-only): simple VaR/ES, breach p-values on first path per model
    metrics=[]
    def var_es(series, level):
        v=np.percentile(series,(1-level)*100); tail=series[series<=v]; es=float(np.mean(tail)) if len(tail)>0 else np.nan
        return float(v), es
    for mt in ['real','zero_conditioned','explicit_conditioned','llm_conditioned']:
        if mt=='real': x=flat_real
        else:
            if len(model_paths[mt])==0: continue
            x = model_paths[mt][0]
        v95, es95 = var_es(x,0.95); v99, es99 = var_es(x,0.99)
        hits95=(x<=v95).astype(int); hits99=(x<=v99).astype(int)
        _, kp95 = kupiec_uc(hits95,0.05); _, ip95 = christoffersen_ind(hits95)
        _, kp99 = kupiec_uc(hits99,0.01); _, ip99 = christoffersen_ind(hits99)
        metrics.append({'model_type':mt,'var_95':v95,'es_95':es95,'kupiec_p_95':kp95,'christoffersen_ind_p_95':ip95,'var_99':v99,'es_99':es99,'kupiec_p_99':kp99,'christoffersen_ind_p_99':ip99})
    pd.DataFrame(metrics).to_csv(mtr_dir/'consolidated_metrics.csv', index=False)
    with open(mtr_dir/'consolidated_metrics.json','w') as f: json.dump(metrics,f,indent=2)

    # Console summary table (percent units)
    try:
        df_console = pd.DataFrame(metrics)
        df_console['var_95'] = df_console['var_95'].map(lambda x: f"{x:.3f}")
        df_console['es_95'] = df_console['es_95'].map(lambda x: f"{x:.3f}")
        df_console['kupiec_p_95'] = df_console['kupiec_p_95'].map(lambda x: f"{x:.3f}")
        df_console['christoffersen_ind_p_95'] = df_console['christoffersen_ind_p_95'].map(lambda x: f"{x:.3f}")
        df_console['var_99'] = df_console['var_99'].map(lambda x: f"{x:.3f}")
        df_console['es_99'] = df_console['es_99'].map(lambda x: f"{x:.3f}")
        df_console['kupiec_p_99'] = df_console['kupiec_p_99'].map(lambda x: f"{x:.3f}")
        df_console['christoffersen_ind_p_99'] = df_console['christoffersen_ind_p_99'].map(lambda x: f"{x:.3f}")
        print('\nCrisis-window VaR/ES summary (units: %):')
        print(df_console[['model_type','var_95','es_95','kupiec_p_95','christoffersen_ind_p_95','var_99','es_99','kupiec_p_99','christoffersen_ind_p_99']].to_string(index=False))
    except Exception as e:
        print(f"Warning: could not print console summary: {e}")

    # Report
    lines=["# COVID-19 Crisis Case Study (Feb–Apr 2020)", "", f"Window: {args.crisis_start} to {args.crisis_end}", "", "Key crisis metrics (first-path per model):"]
    for m in metrics:
        lines.append(f"- {m['model_type']}: VaR95={m['var_95']:.4f}, ES95={m['es_95']:.4f}, Kupiec p(95)={m['kupiec_p_95']:.3f}, Christoffersen p(95)={m['christoffersen_ind_p_95']:.3f}")
        lines.append(f"  VaR99={m['var_99']:.4f}, ES99={m['es_99']:.4f}, Kupiec p(99)={m['kupiec_p_99']:.3f}, Christoffersen p(99)={m['christoffersen_ind_p_99']:.3f}")
    with open(out_root/'evaluation_report.md','w') as f: f.write("\n".join(lines))

    # Compile final PDF (fixed canonical set)
    try:
        from PyPDF2 import PdfMerger
        merger=PdfMerger()
        pdfs = [
            case_fig_dir/'covid_returns_overlay.pdf',
            case_fig_dir/'covid_returns_real_context.pdf',
            case_fig_dir/'covid_loghist.pdf',
            case_fig_dir/'covid_ecdf.pdf',
            case_fig_dir/'covid_qq_left.pdf',
            case_fig_dir/'covid_qq_right.pdf',
            case_fig_dir/'covid_vol_rolling.pdf',
            case_fig_dir/'covid_var_95_99.pdf',
            case_fig_dir/'covid_regimes.pdf',
        ]
        for f in pdfs:
            if f.exists():
                merger.append(str(f))
        merger.write(str(out_root/'final_case_study_fixed.pdf'))
        merger.close()
    except Exception as e:
        print(f"Warning: could not compile final PDF: {e}")

    print(f"✅ COVID-19 case study saved to: {out_root}")


if __name__=='__main__':
    main()


