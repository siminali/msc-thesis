#!/usr/bin/env python3
"""
Strict Novelty Comparison Evaluation

Generates thesis-ready plots/tables comparing zero-, explicit-, and LLM-conditioned models
with consistent styling and combined overlays, plus uncertainty bands and summary report.

Outputs in: results/novelty_comparison/

CLI:
  python evaluate_novelty_strict.py \
    --models_dir results \
    --results_dir results/novelty_comparison \
    --seed 42 \
    --num_samples 500
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

from scipy import stats
from scipy.stats import ks_2samp, anderson_ksamp, wasserstein_distance
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent))
from explicit_cond_ddpm import (
    ExplicitConditioningDDPM,
    ExplicitConditioningTrainer,
    create_sequences,
    create_conditioning_vectors,
)
from llm_conditioned_diffusion_refactored import (
    LLMConditionedDiffusion,
    LLMDiffusionTrainer,
    NewsDataLoader,
)


def parse_args():
    p = argparse.ArgumentParser(description='Strict Novelty Comparison Evaluation')
    p.add_argument('--models_dir', type=str, default='results')
    p.add_argument('--results_dir', type=str, default='results/novelty_comparison')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--num_samples', type=int, default=500)
    p.add_argument('--acf_lags', type=int, default=20)
    p.add_argument('--rolling_window', type=int, default=20)
    p.add_argument('--var_levels', nargs='+', type=float, default=[0.95, 0.99])
    p.add_argument('--ablation_samples', type=int, default=500)
    p.add_argument('--bootstrap_iters', type=int, default=200)
    p.add_argument('--llm_use_aligned_embeddings', action='store_true')
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


def mmd_rbf(x: np.ndarray, y: np.ndarray, gamma: float = None, perms: int = 200) -> Tuple[float, float]:
    x = x.reshape(-1,1); y = y.reshape(-1,1)
    if gamma is None:
        v = np.var(np.concatenate([x,y]))
        gamma = 1.0 / (2.0 * v + 1e-8)
    def k(a,b):
        d2 = (a - b.T)**2
        return np.exp(-gamma * d2)
    Kxx = k(x,x); Kyy = k(y,y); Kxy = k(x,y)
    m = len(x); n = len(y)
    mmd2 = Kxx.mean() + Kyy.mean() - 2*Kxy.mean()
    z = np.concatenate([x,y], axis=0)
    obs = mmd2
    cnt = 0
    for _ in range(perms):
        idx = np.random.permutation(m+n)
        x_p = z[idx[:m]]; y_p = z[idx[m:]]
        Kxx_p = k(x_p,x_p); Kyy_p = k(y_p,y_p); Kxy_p = k(x_p,y_p)
        mmd2_p = Kxx_p.mean() + Kyy_p.mean() - 2*Kxy_p.mean()
        if mmd2_p >= obs:
            cnt += 1
    pval = (cnt + 1) / (perms + 1)
    return float(mmd2), float(pval)


def bootstrap_ecdf_diff(real: np.ndarray, gen: np.ndarray, iters: int = 200, alpha: float = 0.05) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    xs = np.linspace(min(real.min(), gen.min()), max(real.max(), gen.max()), 200)
    def ecdf(a):
        a = np.sort(a)
        return np.searchsorted(a, xs, side='right') / len(a)
    diffs = []
    for _ in range(iters):
        r_bs = np.random.choice(real, size=len(real), replace=True)
        g_bs = np.random.choice(gen, size=len(gen), replace=True)
        diffs.append(ecdf(r_bs) - ecdf(g_bs))
    diffs = np.array(diffs)
    lo = np.percentile(diffs, 100*alpha/2, axis=0)
    hi = np.percentile(diffs, 100*(1-alpha/2), axis=0)
    return xs, lo, hi


def main():
    args = parse_args()
    set_determinism(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Output dirs (fixed path, no timestamp)
    out_root = Path(args.results_dir)
    fig_dir = out_root / 'figures'; tbl_dir = out_root / 'tables'; mtr_dir = out_root / 'metrics'
    for d in [fig_dir, tbl_dir, mtr_dir]:
        d.mkdir(parents=True, exist_ok=True)
    for mt in ['zero_conditioned','explicit_conditioned','llm_conditioned','combined']:
        (fig_dir/mt).mkdir(parents=True, exist_ok=True)
        if mt!='combined':
            (tbl_dir/mt).mkdir(parents=True, exist_ok=True)

    # Data
    returns = load_returns()
    X = create_sequences(returns, 60)
    explicit_cond, _, _ = create_conditioning_vectors(returns, 60, 20, 0.2)

    # Optional LLM aligned conditioning
    llm_cond = None
    if args.llm_use_aligned_embeddings:
        try:
            loader = NewsDataLoader()
            emb = loader.create_conditioning_vectors(returns.index, seq_len=60, embedding_dim=64)
            llm_cond = emb / np.linalg.norm(emb, axis=1, keepdims=True)
            np.save(mtr_dir/'llm_aligned_conditioning.npy', llm_cond)
        except Exception as e:
            print(f"Warning: LLM aligned embeddings failed: {e}")

    # Checkpoints and sampling
    ckpts = discover_checkpoints(Path(args.models_dir))
    if not ckpts:
        print('❌ No checkpoints found.'); return

    results: Dict[str, Dict[str, Any]] = {}
    model_list = ['zero_conditioned','explicit_conditioned','llm_conditioned']
    with tqdm(total=len(model_list), desc='Models', unit='model') as pbar:
        for mt in model_list:
            if mt not in ckpts:
                pbar.update(1)
                continue
            model, trainer = load_model_trainer(mt, ckpts[mt], device)
            n = args.num_samples
            if mt=='zero_conditioned':
                cond = torch.zeros(n, 5, device=device)
                samples = trainer.sample(cond, num_samples=n, sampler='ddim', sample_steps=50)
            elif mt=='explicit_conditioned':
                cond = torch.tensor(explicit_cond[:n], dtype=torch.float32, device=device)
                samples = trainer.sample(cond, num_samples=n, sampler='ddim', sample_steps=50, cfg_scale=7.5)
            else:
                if llm_cond is not None and len(llm_cond)>=n:
                    cond = torch.tensor(llm_cond[:n], dtype=torch.float32, device=device)
                else:
                    cond = F.normalize(torch.randn(n, 64, device=device), dim=1)
                samples = trainer.sample(cond, num_samples=n, sampler='ddim', sample_steps=50, cfg_scale=7.5)
            samples = samples.squeeze(1).detach().cpu().numpy()
            results[mt] = {'samples': samples}
            pbar.update(1)

    # Styling
    plt.style.use('seaborn-v0_8-white')
    colors = {'zero_conditioned':'#1f77b4','explicit_conditioned':'#2ca02c','llm_conditioned':'#d62728','real':'#000000'}

    # Stylised facts: histogram (log y, line only)
    real = returns.values
    fig, ax = plt.subplots(figsize=(10,6))
    ax.set_yscale('log')
    for mt, res in results.items():
        flat = res['samples'].flatten()
        hist, bins = np.histogram(flat, bins=100, density=True)
        centers = (bins[:-1]+bins[1:])/2
        ax.plot(centers, hist, label=mt.replace('_',' '), color=colors[mt])
    h_r, b_r = np.histogram(real, bins=100, density=True)
    c_r = (b_r[:-1]+b_r[1:])/2
    ax.plot(c_r, h_r, label='Real', color=colors['real'], linestyle='--')
    ax.set_title('Heavy-tailed Histogram (log y)'); ax.grid(True, alpha=0.3); ax.legend()
    plt.tight_layout(); plt.savefig(fig_dir/'combined'/'hist_logy_overlay.pdf', dpi=300, bbox_inches='tight'); plt.close()

    
    # Density (KDE) and ECDF overlay with bootstrap bands for ECDF differences
    fig, ax = plt.subplots(figsize=(10,6))
    try:
        sns.kdeplot(real, ax=ax, label='Real', color=colors['real'], linestyle='--')
    except Exception:
        pass
    for mt, res in results.items():
        try:
            sns.kdeplot(res['samples'].flatten(), ax=ax, label=mt.replace('_',' '), color=colors[mt])
        except Exception:
            flat = res['samples'].flatten()
            hist, bins = np.histogram(flat, bins=100, density=True)
            centers = (bins[:-1]+bins[1:])/2
            ax.plot(centers, hist, label=mt.replace('_',' '), color=colors[mt])
    ax.set_title('Density Overlay'); ax.grid(True, alpha=0.3); ax.legend()
    plt.tight_layout(); plt.savefig(fig_dir/'combined'/'density_overlay.pdf', dpi=300, bbox_inches='tight'); plt.close()

    fig, ax = plt.subplots(figsize=(10,6))
    r_sorted = np.sort(real); yr = np.arange(1,len(r_sorted)+1)/len(r_sorted)
    ax.plot(r_sorted, yr, label='Real', color=colors['real'], linestyle='--')
    for mt, res in results.items():
        g = np.sort(res['samples'].flatten()); yg = np.arange(1,len(g)+1)/len(g)
        ax.plot(g, yg, label=mt.replace('_',' '), color=colors[mt])
    ax.set_title('ECDF Overlay'); ax.grid(True, alpha=0.3); ax.legend()
    plt.tight_layout(); plt.savefig(fig_dir/'combined'/'ecdf_overlay.pdf', dpi=300, bbox_inches='tight'); plt.close()

    # QQ plots (tails)
    for mt, res in results.items():
        flat = res['samples'].flatten()
        fig, (a1,a2) = plt.subplots(1,2, figsize=(14,6))
        left_g = flat[flat < np.percentile(flat, 10)]
        right_g = flat[flat > np.percentile(flat, 90)]
        stats.probplot(left_g, dist='norm', plot=a1); a1.set_title(f'{mt}: Q-Q Left Tail')
        stats.probplot(right_g, dist='norm', plot=a2); a2.set_title(f'{mt}: Q-Q Right Tail')
        plt.tight_layout(); plt.savefig(fig_dir/mt/'qq_tails.pdf', dpi=300, bbox_inches='tight'); plt.close()

    # ACF/PACF for returns and squared returns with 95% CI (statsmodels draws bands)
    for mt, res in results.items():
        flat = res['samples'].flatten()
        fig, ((a1,a2),(a3,a4)) = plt.subplots(2,2, figsize=(14,10))
        plot_acf(flat, lags=args.acf_lags, ax=a1, title=f'ACF - {mt}'); plot_pacf(flat, lags=args.acf_lags, ax=a2, title=f'PACF - {mt}')
        sq = flat**2
        plot_acf(sq, lags=args.acf_lags, ax=a3, title=f'ACF (squared) - {mt}'); plot_pacf(sq, lags=args.acf_lags, ax=a4, title=f'PACF (squared) - {mt}')
        plt.tight_layout(); plt.savefig(fig_dir/mt/'acf_pacf.pdf', dpi=300, bbox_inches='tight'); plt.close()

    # Stylised facts heatmap (pass/fail)
    for mt, res in results.items():
        flat = res['samples'].flatten()
        rules = [
            ('Heavy tails (kurt>3)', float(stats.kurtosis(flat))>3.0),
            ('Leverage (corr<0)', np.corrcoef(flat[1:], (flat**2)[:-1])[0,1]<0 if len(flat)>1 else False),
            ('Vol clustering (ACF>0)', acf((flat**2), nlags=1, fft=False)[1]>0 if len(flat)>2 else False),
        ]
        arr = np.array([[int(v) for _,v in rules]], dtype=float)
        fig, ax = plt.subplots(figsize=(6,2))
        im=ax.imshow(arr, cmap='Greens', vmin=0, vmax=1, aspect='auto'); ax.set_yticks([0]); ax.set_yticklabels(['Pass/Fail'])
        ax.set_xticks(range(len(rules))); ax.set_xticklabels([n for n,_ in rules], rotation=20, ha='right')
        plt.tight_layout(); plt.savefig(fig_dir/mt/'stylised_facts_heatmap.pdf', dpi=300, bbox_inches='tight'); plt.close()

    # Rolling volatility overlay
    fig, ax = plt.subplots(figsize=(12,6))
    real_vol = returns.rolling(window=args.rolling_window).std().dropna()
    ax.plot(real_vol.index, real_vol.values, label='Real', color=colors['real'])
    for mt, res in results.items():
        gen = pd.Series(res['samples'][0]).rolling(window=args.rolling_window).std().dropna()
        dates = pd.date_range(start=pd.Timestamp('2024-01-01'), periods=len(gen), freq='D')
        ax.plot(dates, gen.values, label=mt.replace('_',' '), color=colors[mt])
    ax.set_title(f'Rolling Volatility (window={args.rolling_window})'); ax.grid(True, alpha=0.3); ax.legend()
    plt.tight_layout(); plt.savefig(fig_dir/'combined'/'rolling_volatility_all.pdf', dpi=300, bbox_inches='tight'); plt.close()

    # Prediction error residual histograms and metrics
    val_start = int(len(X)*0.8)
    val = X[val_start:]
    for mt, res in results.items():
        mse=[]; mae=[]; errs=[]
        for i, real_seq in enumerate(val[:min(100, len(res['samples']))]):
            r = real_seq[0,:]; g = res['samples'][i]
            e = r - g; errs.extend(list(e))
            mse.append(((r-g)**2).mean()); mae.append(np.mean(np.abs(r-g)))
        ME = float(np.mean(errs)) if errs else np.nan
        MSE = float(np.mean(mse)) if mse else np.nan
        MAE = float(np.mean(mae)) if mae else np.nan
        RMSE = float(np.sqrt(MSE)) if np.isfinite(MSE) else np.nan
        with open(tbl_dir/mt/'residual_metrics.json','w') as f: json.dump({'ME':ME,'MAE':MAE,'MSE':MSE,'RMSE':RMSE}, f, indent=2)
        fig, ax = plt.subplots(figsize=(10,6))
        hist, bins = np.histogram(errs, bins=80, density=True)
        centers = (bins[:-1]+bins[1:])/2
        ax.plot(centers, hist, label=f"ME={ME:.4f}, MAE={MAE:.4f}, MSE={MSE:.4f}, RMSE={RMSE:.4f}")
        ax.set_title(f'Residual Histogram - {mt}'); ax.grid(True, alpha=0.3); ax.legend()
        plt.tight_layout(); plt.savefig(fig_dir/mt/'residual_histogram.pdf', dpi=300, bbox_inches='tight'); plt.close()

    # Risk: VaR/ES overlay and exceedance timelines
    levels = np.arange(0.90, 0.999, 0.001)
    fig, (a1,a2) = plt.subplots(1,2, figsize=(14,6))
    for mt, res in results.items():
        flat = res['samples'].flatten(); var_vals=[]; es_vals=[]
        for L in levels:
            v = np.percentile(flat, (1-L)*100)
            var_vals.append(v)
            tail = flat[flat<=v]; es_vals.append(np.mean(tail) if len(tail)>0 else np.nan)
        a1.plot(levels, var_vals, label=mt.replace('_',' '), color=colors[mt])
        a2.plot(levels, es_vals, label=mt.replace('_',' '), color=colors[mt])
    a1.set_title('VaR Curve (95–99)'); a2.set_title('ES Curve (95–99)')
    for ax in (a1,a2): ax.set_xlabel('Confidence'); ax.grid(True, alpha=0.3); ax.legend()
    plt.tight_layout(); plt.savefig(fig_dir/'combined'/'var_es_curves_all.pdf', dpi=300, bbox_inches='tight'); plt.close()

    # Exceedance timelines
    fig, axes = plt.subplots(len(results), 1, figsize=(12, 3*len(results)), sharex=True)
    if len(results)==1: axes=[axes]
    for ax, (mt,res) in zip(axes, results.items()):
        flat = res['samples'].flatten()
        v95 = np.percentile(flat, 5); v99 = np.percentile(flat, 1)
        t = np.arange(len(flat))
        ax.plot(t, flat, lw=1)
        ax.axhline(v95, color='r', ls='--'); ax.axhline(v99, color='m', ls='--')
        mask95 = flat<=v95; mask99 = flat<=v99
        ax.scatter(t[mask95], flat[mask95], s=10, color='red', alpha=0.6)
        ax.set_ylabel(mt)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel('Time index')
    plt.tight_layout(); plt.savefig(fig_dir/'combined'/'exceedance_timelines.pdf', dpi=300, bbox_inches='tight'); plt.close()

    # Controllability: explicit (reliability, scatter, residuals), LLM (probe), zero (reference)
    # Explicit
    if 'explicit_conditioned' in results:
        target = explicit_cond[:len(results['explicit_conditioned']['samples']), -1]
        realized = np.array([np.std(s, ddof=1) for s in results['explicit_conditioned']['samples']])
        fig, (b1,b2,b3) = plt.subplots(1,3, figsize=(18,5))
        m,M = target.min(), target.max()
        b1.scatter(target, realized, alpha=0.6); b1.plot([m,M],[m,M],'r--'); b1.set_title('Target vs Realized'); b1.grid(True, alpha=0.3)
        bins = np.linspace(m,M,10); centers=[]; means=[]
        for i in range(len(bins)-1):
            mask=(target>=bins[i])&(target<bins[i+1])
            if mask.any(): centers.append((bins[i]+bins[i+1])/2); means.append(float(realized[mask].mean()))
        if centers:
            b2.plot(centers, means, 'bo-'); b2.plot([m,M],[m,M],'r--'); b2.set_title('Reliability'); b2.grid(True, alpha=0.3)
        resid = realized - target; b3.scatter(target, resid, alpha=0.6); b3.axhline(0, color='r', ls='--'); b3.set_title('Residuals'); b3.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(fig_dir/'explicit_conditioned'/'controllability.pdf', dpi=300, bbox_inches='tight'); plt.close()

    # LLM probe
    if 'llm_conditioned' in results:
        try:
            from sklearn.linear_model import LinearRegression, LogisticRegression
            from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score
            if llm_cond is None or len(llm_cond) != len(explicit_cond):
                raise RuntimeError('No aligned LLM conditioning to train probe')
            sigma_star = explicit_cond[:, -1]; regime_idx = np.argmax(explicit_cond[:, :4], axis=1)
            split = int(0.8*len(llm_cond))
            Xtr, Xval = llm_cond[:split], llm_cond[split:]
            yv_tr, yv_val = sigma_star[:split], sigma_star[split:]
            yr_tr, yr_val = regime_idx[:split], regime_idx[split:]
            vol_model = LinearRegression().fit(Xtr, yv_tr)
            yv_pred = vol_model.predict(Xval)
            mae = mean_absolute_error(yv_val, yv_pred); r2 = r2_score(yv_val, yv_pred)
            clf = LogisticRegression(max_iter=1000, multi_class='multinomial').fit(Xtr, yr_tr)
            yr_pred = clf.predict(Xval); acc = accuracy_score(yr_val, yr_pred)
            with open(tbl_dir/'llm_conditioned'/'probe_scores.json','w') as f: json.dump({'vol_mae':float(mae),'vol_r2':float(r2),'regime_acc':float(acc)}, f, indent=2)
            # Plots
            fig, (c1,c2,c3) = plt.subplots(1,3, figsize=(18,5))
            m,M = yv_val.min(), yv_val.max()
            c1.scatter(yv_val, yv_pred, alpha=0.6); c1.plot([m,M],[m,M],'r--'); c1.set_title(f'LLM Probe Scatter (MAE={mae:.3f}, R2={r2:.3f})'); c1.grid(True, alpha=0.3)
            bins = np.linspace(m,M,10); centers=[]; means=[]
            for i in range(len(bins)-1):
                mask=(yv_val>=bins[i])&(yv_val<bins[i+1])
                if mask.any(): centers.append((bins[i]+bins[i+1])/2); means.append(float(yv_pred[mask].mean()))
            if centers:
                c2.plot(centers, means, 'bo-'); c2.plot([m,M],[m,M],'r--'); c2.set_title('LLM Reliability'); c2.grid(True, alpha=0.3)
            resid = yv_pred - yv_val
            c3.scatter(yv_val, resid, alpha=0.6); c3.axhline(0, color='r', ls='--'); c3.set_title('LLM Residuals'); c3.grid(True, alpha=0.3)
            plt.tight_layout(); plt.savefig(fig_dir/'llm_conditioned'/'llm_probe_controllability.pdf', dpi=300, bbox_inches='tight'); plt.close()
            # Regime confusion
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(yr_val, yr_pred, labels=[0,1,2,3], normalize='true')
            fig, ax = plt.subplots(figsize=(5,4)); im=ax.imshow(cm, cmap='Purples', vmin=0, vmax=1); plt.colorbar(im, ax=ax)
            ax.set_title('LLM Regime Confusion (Probe)')
            plt.tight_layout(); plt.savefig(fig_dir/'llm_conditioned'/'llm_probe_regime_confusion.pdf', dpi=300, bbox_inches='tight'); plt.close()
        except Exception as e:
            print(f"Warning: LLM probe skipped: {e}")

    # Metrics + p-values: KS, AD (k-sample), MMD with permutation
    rows=[]
    for mt, res in results.items():
        flat = res['samples'].flatten()
        ks_stat, ks_p = ks_2samp(returns.values, flat)
        try:
            ad_res = anderson_ksamp([returns.values, flat])
            ad_stat = float(ad_res.statistic); ad_p = float(ad_res.significance_level/100.0)
        except Exception:
            ad_stat = np.nan; ad_p = np.nan
        mmd_stat, mmd_p = mmd_rbf(returns.values, flat, perms=200)
        rows.append({'model_type': mt, 'ks_stat': float(ks_stat), 'ks_pvalue': float(ks_p), 'ad_stat': ad_stat, 'ad_pvalue': ad_p, 'mmd_stat': mmd_stat, 'mmd_pvalue': mmd_p})
    df = pd.DataFrame(rows)
    df.to_csv(mtr_dir/'consolidated_metrics.csv', index=False)
    with open(mtr_dir/'consolidated_metrics.json','w') as f: json.dump(rows, f, indent=2)

    # Report
    lines=["# Novelty Comparison Evaluation (Strict)", "", "Key metrics:", ""]
    for r in rows:
        lines += [f"- {r['model_type']}: KS={r['ks_stat']:.4f} (p={r['ks_pvalue']:.4f}), AD={r['ad_stat']:.4f} (p≈{r['ad_pvalue']:.3f}), MMD={r['mmd_stat']:.4f} (p={r['mmd_pvalue']:.3f})"]
    with open(out_root/'evaluation_report.md','w') as f:
        f.write("\n".join(lines))

    # Compile PDF report (best-effort)
    try:
        from PyPDF2 import PdfMerger
        merger = PdfMerger()
        pdfs = []
        for sub in ['combined','zero_conditioned','explicit_conditioned','llm_conditioned']:
            p = fig_dir/sub
            if p.exists():
                for f in sorted(p.glob('*.pdf')):
                    pdfs.append(f)
        for f in pdfs:
            merger.append(str(f))
        merger.write(str(out_root/'final_report.pdf'))
        merger.close()
    except Exception as e:
        print(f"Warning: could not compile PDF report: {e}")

    print(f"✅ Strict novelty comparison saved to: {out_root}")


if __name__ == '__main__':
    main()


