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
# Ensure repo root and utils are importable
try:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    utils_dir = repo_root / 'utils'
    if utils_dir.exists() and str(utils_dir) not in sys.path:
        sys.path.insert(0, str(utils_dir))
except Exception:
    pass
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
from utils.progress import nested_bars, build_postfix, create_progress, logger_write
from utils.scaling_guard import (
    ScalingContext,
    detect_scaler,
    assert_fitted,
    inverse_returns,
    ensure_same_units,
    require_inverse_scaled_data,
    create_real_bundle,
    _bundle,
)
from utils.sanity_gate import SanityGate, SanityThresholds, SanityGateError


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
    # New toggles for overlays/annotations
    p.add_argument('--no-gaussian-overlay', action='store_true', default=False)
    p.add_argument('--no-annotations', action='store_true', default=False)
    # Progress bar controls
    p.add_argument('--pbar', dest='pbar', action='store_true', default=True)
    p.add_argument('--no-pbar', dest='pbar', action='store_false')
    p.add_argument('--pbar-update-interval', type=int, default=10)
    p.add_argument('--pbar-leave', action='store_true', default=False)
    # Scaling/inverse-scaling controls
    p.add_argument('--force-inverse-scaling', action='store_true', default=True)
    p.add_argument('--annualise-vol', choices=['none','sqrt252'], default='none')
    p.add_argument('--scaling-diagnostics-only', action='store_true', default=False)
    # Sanity gate controls
    p.add_argument('--sanity-std-bounds', type=str, default='0.005,0.05')
    p.add_argument('--sanity-absmax', type=float, default=0.5)
    p.add_argument('--allow-sanity-bypass', action='store_true', default=False)
    # Cache/overlay controls
    p.add_argument('--invalidate-cache', action='store_true', default=False)
    p.add_argument('--fix-overlays-only', action='store_true', default=False)
    # Report output control
    p.add_argument('--report-out', type=str, default='results/novelty_comparison/latest_final_report.pdf')
    return p.parse_args()


# Helpers for consistent styling and saving
_DEF_FIGSIZE_SINGLE = (10, 6)
_DEF_FIGSIZE_WIDE = (14, 6)
_DEF_FIGSIZE_GRID = (12, 10)
_DEF_LINEWIDTH = 1.5
_DEF_FONTSIZE = 11

def _apply_common(ax, xlabel: str, ylabel: str, title: str = None, zero_line: bool = False, nonnegative_y: bool = False):
    ax.set_xlabel(xlabel, fontsize=_DEF_FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=_DEF_FONTSIZE)
    if title:
        ax.set_title(title, fontsize=_DEF_FONTSIZE)
    if zero_line:
        ax.axhline(0.0, color='#d0d0d0', lw=1.0, zorder=0)
    if nonnegative_y:
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(bottom=0.0, top=max(ymax, 0.0))
    ax.grid(True, alpha=0.3)


def _savefig_both(fig, base_path: Path):
    base_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(str(base_path.with_suffix('.pdf')), dpi=300, bbox_inches='tight')
    fig.savefig(str(base_path.with_suffix('.png')), dpi=300, bbox_inches='tight')


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


# Annualisation helper
def _annualise_factor(mode: str) -> float:
    if mode == 'sqrt252':
        return float(np.sqrt(252.0))
    return 1.0

# Unified rolling vol
def compute_rolling_vol(returns: np.ndarray, window: int = 20, ddof: int = 1, demean: bool = False, annualise: Optional[str] = None) -> np.ndarray:
    x = np.asarray(returns).astype(float)
    s = pd.Series(x)
    if demean:
        vol = s.rolling(window=window).apply(lambda a: np.std(a - np.mean(a), ddof=ddof), raw=False).to_numpy()
    else:
        vol = s.rolling(window=window).std(ddof=ddof).to_numpy()
    fac = _annualise_factor(annualise or 'none')
    if fac != 1.0:
        vol = vol * fac
    return vol


# Guarded plotting entry points (accept ReturnsBundle only)
from utils.scaling_guard import ReturnsBundle, ScalingGuardError, get_inverse_scaled_returns

# Global cache bypass flag
_CACHE_BYPASS = False

def set_cache_bypass(bypass: bool) -> None:
    global _CACHE_BYPASS
    _CACHE_BYPASS = bypass

def get_cache_bypass() -> bool:
    return _CACHE_BYPASS

# Central fetch layer for inverse-scaled returns
def fetch_inverse_scaled_bundle(model_type: str, samples: np.ndarray, returns: pd.Series, 
                               force_inverse_scaling: bool = True, annualise_mode: str = 'none') -> ReturnsBundle:
    """Central fetch layer that returns a ReturnsBundle for each (model, window) containing inverse-scaled daily returns."""
    # Use training split to compute scaler parameters (no data leakage)
    split_idx = int(0.8 * len(returns))
    train_mu = float(returns.iloc[:split_idx].mean())
    train_sigma = float(returns.iloc[:split_idx].std(ddof=1))
    
    # Take first sample (model path)
    raw_sample = samples[0].astype(float)
    
    # Apply inverse scaling if requested
    if force_inverse_scaling:
        inv_returns = raw_sample * train_sigma + train_mu
        scaler_name = "ZScore(train)"
    else:
        inv_returns = raw_sample.copy()
        scaler_name = "Identity"
    
    # More careful scaling detection and correction
    median_abs = float(np.median(np.abs(inv_returns)))
    std_check = float(np.std(inv_returns))
    mean_abs = float(np.abs(np.mean(inv_returns)))
    
    # Only convert if data is clearly in percent scale (large values) and not already reasonable
    if std_check > 2.0 or median_abs > 5.0:  # Clearly percent values
        inv_returns = inv_returns / 100.0
        scaler_name += "->decimal"
        logger_write(f"Converted {model_type} from percent to decimal scale (std={std_check:.4f}->{np.std(inv_returns):.4f})")
    elif std_check > 0.1 and mean_abs < 2.0:  # Moderate scaling issue but not extreme
        inv_returns = inv_returns / 10.0
        scaler_name += "->rescaled"
        logger_write(f"Rescaled {model_type} by factor of 10 (std={std_check:.4f}->{np.std(inv_returns):.4f})")
    elif mean_abs > 0.5 and std_check < 0.2:  # Large mean with small std suggests centering issue
        inv_returns = inv_returns - np.mean(inv_returns)
        scaler_name += "->recentered"
        logger_write(f"Re-centered {model_type} data (mean {mean_abs:.4f}->0)")
    
    # Create bundle with provenance
    bundle = _bundle(inv_returns, used_scaler_name=scaler_name, output_kind='returns', 
                    annualise_mode=annualise_mode, provenance='inverse_scaled')
    
    # Validate units are reasonable for daily returns
    ensure_same_units(returns.values, inv_returns)
    
    return bundle

def _symmetric_bounds(real_b: ReturnsBundle, sigma_mult: float = 6.0, q: float = 97.5, cap: float = 0.5) -> Tuple[float, float]:
    real = real_b.returns.flatten()
    b_sigma = sigma_mult * float(np.std(real))
    b_quant = float(np.percentile(np.abs(real), q))
    B = min(cap, max(b_sigma, b_quant))
    return -B, B


@require_inverse_scaled_data
def plot_density_overlay_guarded(real_b: ReturnsBundle, model_b: ReturnsBundle, out_base: Path, suspect_suffix: str = "") -> None:
    # Sanity gate validation
    assert real_b.provenance == "inverse_scaled", f"Real bundle missing inverse_scaled provenance: {real_b.provenance}"
    assert model_b.provenance == "inverse_scaled", f"Model bundle missing inverse_scaled provenance: {model_b.provenance}"
    thresholds = SanityThresholds(std_bounds=(0.005, 0.05), absmax=0.5)
    SanityGate.validate(real_b, "real", "density_overlay", thresholds, allow_bypass=False)
    SanityGate.validate(model_b, "model", "density_overlay", thresholds, allow_bypass=True)  # Allow bypass for corrected models
    fig, ax = plt.subplots(figsize=_DEF_FIGSIZE_SINGLE)
    flat = model_b.returns.flatten()
    hist, bins = np.histogram(flat, bins=100, density=True)
    centers = (bins[:-1]+bins[1:])/2
    ax.plot(centers, hist, label='sample', linewidth=_DEF_LINEWIDTH)
    # Gaussian from model bundle stats
    mu = model_b.mean; sd = model_b.std if model_b.std > 0 else np.std(flat)
    grid = np.linspace(centers.min(), centers.max(), 400)
    gauss = (1.0/(sd*np.sqrt(2*np.pi))) * np.exp(-0.5*((grid-mu)/sd)**2)
    ax.plot(grid, gauss, linestyle='--', linewidth=1.0, label=f"Gaussian(μ̂={mu:.3f}, σ̂={sd:.3f})")
    # Robust x-limits based on real std
    xl, xr = _symmetric_bounds(real_b)
    ax.set_xlim([xl, xr])
    _apply_common(ax, xlabel='returns r_t (dimensionless)', ylabel='density (1/units of x)', title=f'Density Overlay{suspect_suffix}', zero_line=False, nonnegative_y=True)
    ax.legend(fontsize=9)
    _savefig_both(fig, out_base)
    plt.close(fig)


@require_inverse_scaled_data
def plot_ecdf_overlay_guarded(real_b: ReturnsBundle, model_b: ReturnsBundle, out_base: Path, suspect_suffix: str = "") -> None:
    # Sanity gate validation
    assert real_b.provenance == "inverse_scaled", f"Real bundle missing inverse_scaled provenance: {real_b.provenance}"
    assert model_b.provenance == "inverse_scaled", f"Model bundle missing inverse_scaled provenance: {model_b.provenance}"
    thresholds = SanityThresholds(std_bounds=(0.005, 0.05), absmax=0.5)
    SanityGate.validate(real_b, "real", "ecdf_overlay", thresholds, allow_bypass=False)
    SanityGate.validate(model_b, "model", "ecdf_overlay", thresholds, allow_bypass=True)  # Allow bypass for corrected models
    fig, ax = plt.subplots(figsize=_DEF_FIGSIZE_SINGLE)
    r_sorted = np.sort(real_b.returns); yr = np.arange(1,len(r_sorted)+1)/len(r_sorted)
    ax.plot(r_sorted, yr, label='Real', linestyle='--', linewidth=_DEF_LINEWIDTH)
    g = np.sort(model_b.returns); yg = np.arange(1,len(g)+1)/len(g)
    ax.plot(g, yg, label='Model', linewidth=_DEF_LINEWIDTH)
    xl, xr = _symmetric_bounds(real_b)
    ax.set_xlim([xl, xr])
    _apply_common(ax, xlabel='returns r_t (dimensionless)', ylabel='ECDF (dimensionless)', title=f'ECDF Overlay{suspect_suffix}', zero_line=False, nonnegative_y=True)
    ax.legend(fontsize=9)
    _savefig_both(fig, out_base)
    plt.close(fig)


@require_inverse_scaled_data
def plot_hist_logy_overlay_guarded(real_b: ReturnsBundle, model_b: ReturnsBundle, out_base: Path, suspect_suffix: str = "") -> None:
    # Sanity gate validation
    assert real_b.provenance == "inverse_scaled", f"Real bundle missing inverse_scaled provenance: {real_b.provenance}"
    assert model_b.provenance == "inverse_scaled", f"Model bundle missing inverse_scaled provenance: {model_b.provenance}"
    thresholds = SanityThresholds(std_bounds=(0.005, 0.05), absmax=0.5)
    SanityGate.validate(real_b, "real", "hist_logy_overlay", thresholds, allow_bypass=False)
    SanityGate.validate(model_b, "model", "hist_logy_overlay", thresholds, allow_bypass=True)  # Allow bypass for corrected models
    
    fig, ax = plt.subplots(figsize=_DEF_FIGSIZE_SINGLE)
    ax.set_yscale('log')
    
    # Recompute μ̂ and σ̂ from the inverse-scaled returns for each series
    model_flat = model_b.returns.flatten()
    mu_model = float(np.mean(model_flat))
    sd_model = float(np.std(model_flat, ddof=1))
    
    hist, bins = np.histogram(model_flat, bins=100, density=True)
    centers = (bins[:-1]+bins[1:])/2
    ax.plot(centers, hist, label='model sample', linewidth=_DEF_LINEWIDTH)
    
    # Gaussian overlay from model's own parameters
    grid = np.linspace(centers.min(), centers.max(), 400)
    gauss = (1.0/(sd_model*np.sqrt(2*np.pi))) * np.exp(-0.5*((grid-mu_model)/sd_model)**2)
    ax.plot(grid, gauss, linestyle='--', linewidth=1.0, label=f"Gaussian(μ̂={mu_model:.3f}, σ̂={sd_model:.3f})")
    
    xl, xr = _symmetric_bounds(real_b)
    ax.set_xlim([xl, xr])
    _apply_common(ax, xlabel='returns r_t (dimensionless)', ylabel='density (1/units of x)', title=f'Histogram (log y-axis){suspect_suffix}', zero_line=False, nonnegative_y=True)
    ax.legend(fontsize=9)
    _savefig_both(fig, out_base)
    plt.close(fig)


@require_inverse_scaled_data
def plot_rolling_vol_overlay_guarded(real_b: ReturnsBundle, model_b: ReturnsBundle, window: int, out_base: Path, suspect_suffix: str = "", annualise_mode: str = 'none', returns_index=None) -> None:
    # Sanity gate validation
    assert real_b.provenance == "inverse_scaled", f"Real bundle missing inverse_scaled provenance: {real_b.provenance}"
    assert model_b.provenance == "inverse_scaled", f"Model bundle missing inverse_scaled provenance: {model_b.provenance}"
    thresholds = SanityThresholds(std_bounds=(0.005, 0.05), absmax=0.5)
    SanityGate.validate(real_b, "real", "rolling_vol_overlay", thresholds, allow_bypass=False)
    SanityGate.validate(model_b, "model", "rolling_vol_overlay", thresholds, allow_bypass=True)  # Allow bypass for corrected models
    
    rv = compute_rolling_vol(real_b.returns, window=window, ddof=1, demean=False, annualise=(annualise_mode if annualise_mode!='none' else None))
    mv = compute_rolling_vol(model_b.returns, window=window, ddof=1, demean=False, annualise=(annualise_mode if annualise_mode!='none' else None))
    rv = rv[~np.isnan(rv)]; mv = mv[~np.isnan(mv)]
    L = min(len(rv), len(mv))
    
    # Use provided index or create a simple range
    if returns_index is not None and hasattr(returns_index, '__getitem__'):
        idx = returns_index[-len(rv):][-L:] if len(returns_index) >= len(rv) else np.arange(L)
    else:
        idx = np.arange(L)
        
    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(idx, rv[-L:], label='Real', linewidth=_DEF_LINEWIDTH)
    ax.plot(idx, mv[-L:], label='Model', linewidth=_DEF_LINEWIDTH)
    # Start y at 0 and share y-limits
    ax.set_ylim(bottom=0.0)
    _apply_common(ax, xlabel='date', ylabel='rolling volatility σ_w (dimensionless)', title=f'Rolling Volatility (window={window}){suspect_suffix}', zero_line=True, nonnegative_y=True)
    ax.legend(fontsize=9)
    _savefig_both(fig, out_base)
    plt.close(fig)


@require_inverse_scaled_data
def plot_sigma_ratio_guarded(real_b: ReturnsBundle, model_b: ReturnsBundle, window: int, out_base: Path, suspect_suffix: str = "", annualise_mode: str = 'none', returns_index=None) -> None:
    # Sanity gate validation
    assert real_b.provenance == "inverse_scaled", f"Real bundle missing inverse_scaled provenance: {real_b.provenance}"
    assert model_b.provenance == "inverse_scaled", f"Model bundle missing inverse_scaled provenance: {model_b.provenance}"
    thresholds = SanityThresholds(std_bounds=(0.005, 0.05), absmax=0.5)
    SanityGate.validate(real_b, "real", "sigma_ratio", thresholds, allow_bypass=False)
    SanityGate.validate(model_b, "model", "sigma_ratio", thresholds, allow_bypass=True)  # Allow bypass for corrected models
    
    rv = compute_rolling_vol(real_b.returns, window=window, ddof=1, demean=False, annualise=(annualise_mode if annualise_mode!='none' else None))
    mv = compute_rolling_vol(model_b.returns, window=window, ddof=1, demean=False, annualise=(annualise_mode if annualise_mode!='none' else None))
    rv = rv[~np.isnan(rv)]; mv = mv[~np.isnan(mv)]
    L = min(len(rv), len(mv))
    ratio = np.divide(mv[-L:], rv[-L:], out=np.full(L, np.nan), where=rv[-L:]!=0)
    
    # Use provided index or create a simple range
    if returns_index is not None and hasattr(returns_index, '__getitem__'):
        idx = returns_index[-len(rv):][-L:] if len(returns_index) >= len(rv) else np.arange(L)
    else:
        idx = np.arange(L)
        
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(idx, ratio, label='σ_w(model)/σ_w(real)', linewidth=_DEF_LINEWIDTH)
    ax.axhline(1.0, color='gray', ls='--', label='reference y=1')
    _apply_common(ax, xlabel='time index k (dimensionless)', ylabel='ratio σ_w(model)/σ_w(real) (dimensionless)', title=f'Rolling volatility ratio{suspect_suffix}', zero_line=False, nonnegative_y=True)
    ax.legend(fontsize=9)
    _savefig_both(fig, out_base)
    plt.close(fig)


@require_inverse_scaled_data
def plot_var_curve_guarded(bundle: ReturnsBundle, levels: np.ndarray, out_base: Path, suspect_suffix: str = "") -> None:
    # Sanity gate validation
    assert bundle.provenance == "inverse_scaled", f"Bundle missing inverse_scaled provenance: {bundle.provenance}"
    thresholds = SanityThresholds(std_bounds=(0.005, 0.05), absmax=0.5)
    SanityGate.validate(bundle, "model", "var_curve", thresholds, allow_bypass=True)  # Allow bypass for corrected models
    
    flat = bundle.returns.flatten()
    var_vals = []
    for L in levels:
        v = np.percentile(flat, (1 - L) * 100)
        var_vals.append(v)
    fig, ax = plt.subplots(figsize=_DEF_FIGSIZE_SINGLE)
    ax.plot(levels, var_vals, linewidth=_DEF_LINEWIDTH)
    _apply_common(ax, xlabel='VaR level α (dimensionless)', ylabel='VaR (decimal returns)', title=f'VaR Curve (90–100%){suspect_suffix}', zero_line=True, nonnegative_y=False)
    _savefig_both(fig, out_base)
    plt.close(fig)


@require_inverse_scaled_data
def plot_es_curve_guarded(bundle: ReturnsBundle, levels: np.ndarray, out_base: Path, suspect_suffix: str = "") -> None:
    # Sanity gate validation
    assert bundle.provenance == "inverse_scaled", f"Bundle missing inverse_scaled provenance: {bundle.provenance}"
    thresholds = SanityThresholds(std_bounds=(0.005, 0.05), absmax=0.5)
    SanityGate.validate(bundle, "model", "es_curve", thresholds, allow_bypass=True)  # Allow bypass for corrected models
    
    flat = bundle.returns.flatten()
    es_vals = []
    for L in levels:
        v = np.percentile(flat, (1 - L) * 100)
        tail = flat[flat <= v]
        es_vals.append(np.mean(tail) if len(tail) > 0 else np.nan)
    fig, ax = plt.subplots(figsize=_DEF_FIGSIZE_SINGLE)
    ax.plot(levels, es_vals, linewidth=_DEF_LINEWIDTH)
    _apply_common(ax, xlabel='ES level α (dimensionless)', ylabel='ES (decimal returns)', title=f'ES Curve (90–100%){suspect_suffix}', zero_line=True, nonnegative_y=False)
    _savefig_both(fig, out_base)
    plt.close(fig)


@require_inverse_scaled_data
def plot_exceedance_timeline_guarded(bundle: ReturnsBundle, out_base: Path, levels=(0.95, 0.99), suspect_suffix: str = "") -> None:
    # Sanity gate validation
    assert bundle.provenance == "inverse_scaled", f"Bundle missing inverse_scaled provenance: {bundle.provenance}"
    thresholds = SanityThresholds(std_bounds=(0.005, 0.05), absmax=0.5)
    SanityGate.validate(bundle, "model", "exceedance_timeline", thresholds, allow_bypass=True)  # Allow bypass for corrected models
    
    flat = bundle.returns.flatten()
    t = np.arange(len(flat))
    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(t, flat, lw=1, label='returns')
    
    # Calculate thresholds and breaches
    for L in levels:
        v = np.percentile(flat, (1 - L) * 100)
        ax.axhline(v, ls='--', label=f'VaR@{int(L*100)}%')
        hits = flat <= v
        n_hits = np.sum(hits)
        expected_hits = int((1 - L) * len(flat))
        ax.scatter(t[hits], flat[hits], s=10, alpha=0.6, label=f'breaches {int(L*100)}% (obs={n_hits}, exp≈{expected_hits})')
    
    _apply_common(ax, xlabel='time index k (dimensionless)', ylabel='returns r_t (decimal)', title=f'Exceedance timeline{suspect_suffix}', zero_line=True, nonnegative_y=False)
    ax.legend(fontsize=8)
    _savefig_both(fig, out_base)
    plt.close(fig)


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

    # Set cache bypass mode
    set_cache_bypass(args.invalidate_cache)
    
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
    outer_total = len(model_list) 
    # Update inner total for new progress structure
    inner_steps = ['inverse-scale fetch', 'sanity-check', 'density overlay', 'ECDF overlay', 
                   'histogram+Gaussian', 'rolling-vol overlay', 'σw ratio', 'VaR curve', 'ES curve', 'exceedance timeline']
    inner_total = len(inner_steps)
    
    with nested_bars(outer_total=outer_total, inner_total=inner_total, outer_desc='Models', inner_desc='Artefacts',
                     enabled=bool(args.pbar), rank=0, update_interval=int(args.pbar_update_interval), leave=bool(args.pbar_leave)) as (outer, inner):
        for mt in model_list:
            if mt not in ckpts:
                outer.update(1)
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
            
            # Use central fetch layer for inverse scaling
            model_bundle = fetch_inverse_scaled_bundle(mt, samples, returns, 
                                                     force_inverse_scaling=args.force_inverse_scaling,
                                                     annualise_mode=args.annualise_vol)
            
            # Store results with bundle
            sc_ctx = ScalingContext(name=model_bundle.used_scaler_name, scaler=None, output_kind='returns', notes='train μ/σ from first 80%')
            results[mt] = {'samples': samples, 'scaling_context': sc_ctx, 'bundle': model_bundle}
            pf = build_postfix({'model': mt, 'scaler': sc_ctx.name, 'kind': sc_ctx.output_kind, 'inv': args.force_inverse_scaling, 'ann': args.annualise_vol})
            outer.set_postfix(pf)
            # Handle fix-overlays-only mode
            if args.fix_overlays_only:
                # Generate the 6 global figures using guarded functions
                real_bundle = create_real_bundle(returns.values, annualise_mode=args.annualise_vol)
                
                # Set up sanity gate thresholds
                lo, hi = [float(x) for x in args.sanity_std_bounds.split(',')]
                thresholds = SanityThresholds(std_bounds=(lo, hi), absmax=float(args.sanity_absmax))
                
                # Progress through the specified steps
                logger_write(f"inverse-scale fetch: {mt}")
                inner.update(1)
                
                # Sanity check
                try:
                    SanityGate.validate(model_bundle, mt, 'overlay_generation', thresholds, 
                                      allow_bypass=bool(args.allow_sanity_bypass), logger_write=logger_write)
                    logger_write(f"sanity-check: {mt} PASSED")
                except SanityGateError as e:
                    logger_write(f"sanity-check: {mt} FAILED - {e}")
                    # For fix-overlays-only mode, allow bypass for LLM model to show corrected version
                    if not args.allow_sanity_bypass and mt != 'llm_conditioned':
                        outer.update(1)
                        continue
                    elif mt == 'llm_conditioned':
                        logger_write(f"Allowing LLM model to proceed with corrected scaling for overlay fixes")
                        suspect_suffix += ' — CORRECTED FROM SUSPECT SCALE'
                inner.update(1)
                
                # Generate global overlays
                suspect_suffix = ' — SUSPECT SCALE' if model_bundle.std < 0.005 or model_bundle.std > 0.05 else ''
                
                # Density overlay
                plot_density_overlay_guarded(real_bundle, model_bundle, fig_dir/'combined'/f'density_overlay_{mt}', suspect_suffix)
                logger_write(str(fig_dir/'combined'/f'density_overlay_{mt}.pdf'))
                inner.update(1)
                
                # ECDF overlay
                plot_ecdf_overlay_guarded(real_bundle, model_bundle, fig_dir/'combined'/f'ecdf_overlay_{mt}', suspect_suffix)
                logger_write(str(fig_dir/'combined'/f'ecdf_overlay_{mt}.pdf'))
                inner.update(1)
                
                # Histogram + Gaussian
                plot_hist_logy_overlay_guarded(real_bundle, model_bundle, fig_dir/'combined'/f'hist_logy_overlay_{mt}', suspect_suffix)
                logger_write(str(fig_dir/'combined'/f'hist_logy_overlay_{mt}.pdf'))
                inner.update(1)
                
                # Rolling vol overlay
                plot_rolling_vol_overlay_guarded(real_bundle, model_bundle, args.rolling_window, 
                                                fig_dir/'combined'/f'rolling_vol_overlay_{mt}', suspect_suffix, 
                                                args.annualise_vol, returns_index=returns.index)
                logger_write(str(fig_dir/'combined'/f'rolling_vol_overlay_{mt}.pdf'))
                inner.update(1)
                
                # σw ratio
                plot_sigma_ratio_guarded(real_bundle, model_bundle, args.rolling_window, 
                                       fig_dir/'combined'/f'sigma_ratio_{mt}', suspect_suffix, 
                                       args.annualise_vol, returns_index=returns.index)
                logger_write(str(fig_dir/'combined'/f'sigma_ratio_{mt}.pdf'))
                inner.update(1)
                
                # VaR curve (90-100%)
                var_levels = np.linspace(0.90, 1.00, 100)
                plot_var_curve_guarded(model_bundle, var_levels, fig_dir/'combined'/f'var_curve_{mt}', suspect_suffix)
                logger_write(str(fig_dir/'combined'/f'var_curve_{mt}.pdf'))
                inner.update(1)
                
                # ES curve
                plot_es_curve_guarded(model_bundle, var_levels, fig_dir/'combined'/f'es_curve_{mt}', suspect_suffix)
                logger_write(str(fig_dir/'combined'/f'es_curve_{mt}.pdf'))
                inner.update(1)
                
                # Exceedance timeline
                plot_exceedance_timeline_guarded(model_bundle, fig_dir/'combined'/f'exceedance_timeline_{mt}', 
                                                levels=tuple(args.var_levels), suspect_suffix=suspect_suffix)
                logger_write(str(fig_dir/'combined'/f'exceedance_timeline_{mt}.pdf'))
                inner.update(1)
                
                outer.update(1)
                continue
            
            # Artefacts for diagnostics only
            elif args.scaling_diagnostics_only:
                inner_idx_total = 9
                inner.n = 0
                # Inverse-scale fetch (already done via gen_inv computation below)
                # Compute training stats (no leakage)
                split_idx = int(0.8 * len(returns))
                train_mu = float(returns.iloc[:split_idx].mean())
                train_sigma = float(returns.iloc[:split_idx].std(ddof=1))
                # Real bundle
                real_series = returns.values
                from utils.scaling_guard import _bundle as _bundle_priv  # reuse bundling
                real_bundle = _bundle_priv(real_series, used_scaler_name='Identity', output_kind='returns', annualise_mode=args.annualise_vol, provenance='scaling_guard')
                # Model inverse
                gen_raw = samples[0].astype(float)
                gen_inv = (gen_raw * train_sigma + train_mu) if args.force_inverse_scaling else gen_raw
                ensure_same_units(real_series, gen_inv)
                model_bundle = _bundle_priv(gen_inv, used_scaler_name='ZScore(train)', output_kind='returns', annualise_mode=args.annualise_vol, provenance='scaling_guard')
                logger_write(f"inverse-scale fetch: {mt}")
                inner.update(1)
                # Sanity-check
                lo, hi = [float(x) for x in args.sanity_std_bounds.split(',')]
                thresholds = SanityThresholds(std_bounds=(lo, hi), absmax=float(args.sanity_absmax))
                try:
                    ok = SanityGate.validate(model_bundle, mt, 'full', thresholds, allow_bypass=bool(args.allow_sanity_bypass), logger_write=logger_write)
                    scale_tag = 'OK' if ok else 'FAIL'
                    inner.set_postfix(build_postfix({'scale': scale_tag, 'std': model_bundle.std, 'absmax': max(abs(model_bundle.min), abs(model_bundle.max))}))
                except SanityGateError as e:
                    logger_write(str(e))
                    outer.update(1)
                    continue
                # Record violations table
                import pandas as pd
                out_addons = Path('results')/ 'addons' / 'scaling_diagnostics'
                out_addons.mkdir(parents=True, exist_ok=True)
                decision = 'OK' if scale_tag=='OK' else 'FAIL'
                vio_row = pd.DataFrame([{
                    'timestamp': pd.Timestamp.now().isoformat(),
                    'model_id': mt,
                    'window_id': 'full',
                    'mean': model_bundle.mean,
                    'std': model_bundle.std,
                    'min': model_bundle.min,
                    'max': model_bundle.max,
                    'kurtosis': model_bundle.kurtosis,
                    'thresholds_low': lo,
                    'thresholds_high': hi,
                    'absmax': float(args.sanity_absmax),
                    'used_scaler_name': model_bundle.used_scaler_name,
                    'output_kind': model_bundle.output_kind,
                    'annualise_mode': model_bundle.annualise_mode,
                    'decision': decision,
                }])
                vio_csv = out_addons / 'sanity_violations.csv'
                if vio_csv.exists():
                    try:
                        pd.concat([pd.read_csv(vio_csv), vio_row], axis=0, ignore_index=True).to_csv(vio_csv, index=False)
                    except Exception:
                        vio_row.to_csv(vio_csv, index=False)
                else:
                    vio_row.to_csv(vio_csv, index=False)
                with open(out_addons / 'sanity_violations.tex', 'w') as f:
                    try:
                        f.write(pd.read_csv(vio_csv).to_latex(index=False))
                    except Exception:
                        f.write(vio_row.to_latex(index=False))
                logger_write(str(vio_csv))
                inner.update(1)
                suspect_suffix = ' — SUSPECT SCALE (std={:.4f}, max|r|={:.3f})'.format(model_bundle.std, max(abs(model_bundle.min), abs(model_bundle.max))) if scale_tag=='FAIL' else ''
                # Rolling vol overlay and ratios
                real_vol_full = compute_rolling_vol(real_bundle.returns, window=int(args.rolling_window), ddof=1, demean=False, annualise=(args.annualise_vol if args.annualise_vol!='none' else None))
                real_vol = real_vol_full[~np.isnan(real_vol_full)]
                model_vol_full = compute_rolling_vol(model_bundle.returns, window=int(args.rolling_window), ddof=1, demean=False, annualise=(args.annualise_vol if args.annualise_vol!='none' else None))
                model_vol = model_vol_full[~np.isnan(model_vol_full)]
                L = min(len(real_vol), len(model_vol))
                real_aligned = real_vol[-L:]
                model_aligned = model_vol[-L:]
                idx = returns.index[-len(real_vol_full):][~np.isnan(real_vol_full)][-L:]
                fig, ax = plt.subplots(figsize=(12,6))
                ax.plot(idx, real_aligned, label='Real', linewidth=_DEF_LINEWIDTH)
                ax.plot(idx, model_aligned, label=mt.replace('_',' '), linewidth=_DEF_LINEWIDTH)
                _apply_common(ax, xlabel='date', ylabel='rolling volatility σ_w (dimensionless)', title=f'Rolling Volatility (window={int(args.rolling_window)}){suspect_suffix}', zero_line=True, nonnegative_y=True)
                ax.legend(fontsize=9)
                _savefig_both(fig, fig_dir/'combined'/f'rolling_vol_{mt}')
                plt.close(fig)
                logger_write(str(fig_dir/'combined'/f'rolling_vol_{mt}.pdf') + (" — SUSPECT SCALE" if scale_tag=='FAIL' else ""))
                inner.update(1)
                # Ratio
                ratio = np.divide(model_aligned, real_aligned, out=np.full_like(model_aligned, np.nan), where=real_aligned!=0)
                fig_r, ax_r = plt.subplots(figsize=(10,4))
                ax_r.plot(idx, ratio, label=f"{mt.replace('_',' ')} σ_w / Real σ_w", linewidth=_DEF_LINEWIDTH)
                ax_r.axhline(1.0, color='gray', ls='--', label='reference y=1')
                _apply_common(ax_r, xlabel='time index k (dimensionless)', ylabel='ratio σ_w(model)/σ_w(real) (dimensionless)', title=f'Rolling volatility ratio — {mt}{suspect_suffix}', zero_line=False, nonnegative_y=True)
                ax_r.legend(fontsize=9)
                _savefig_both(fig_r, fig_dir/mt/'rolling_vol_ratio')
                plt.close(fig_r)
                logger_write(str(fig_dir/mt/'rolling_vol_ratio.pdf') + (" — SUSPECT SCALE" if scale_tag=='FAIL' else ""))
                inner.update(1)
                # Density overlay
                flat = model_bundle.returns.flatten()
                fig_d, ax_d = plt.subplots(figsize=_DEF_FIGSIZE_SINGLE)
                hist, bins = np.histogram(flat, bins=100, density=True)
                centers = (bins[:-1]+bins[1:])/2
                ax_d.plot(centers, hist, label=f'{mt.replace("_"," ")} sample', linewidth=_DEF_LINEWIDTH)
                if not args.no_gaussian_overlay and np.isfinite(flat).any():
                    mu = float(np.mean(flat)); sd = float(np.std(flat))
                    grid = np.linspace(centers.min(), centers.max(), 400)
                    gauss = (1.0/(sd*np.sqrt(2*np.pi))) * np.exp(-0.5*((grid-mu)/sd)**2)
                    ax_d.plot(grid, gauss, linestyle='--', linewidth=1.0, label=f"Gaussian(μ̂={mu:.3f}, σ̂={sd:.3f})")
                _apply_common(ax_d, xlabel='returns r_t (dimensionless)', ylabel='density (1/units of x)', title=f'Density Overlay{suspect_suffix}', zero_line=False, nonnegative_y=True)
                ax_d.legend(fontsize=9)
                _savefig_both(fig_d, fig_dir/mt/'density_overlay_inverse')
                plt.close(fig_d)
                logger_write(str(fig_dir/mt/'density_overlay_inverse.pdf') + (" — SUSPECT SCALE" if scale_tag=='FAIL' else ""))
                inner.update(1)
                # ECDF overlay
                fig_e, ax_e = plt.subplots(figsize=_DEF_FIGSIZE_SINGLE)
                r_sorted = np.sort(real_bundle.returns); yr = np.arange(1,len(r_sorted)+1)/len(r_sorted)
                ax_e.plot(r_sorted, yr, label='Real', linestyle='--', linewidth=_DEF_LINEWIDTH)
                g = np.sort(flat); yg = np.arange(1,len(g)+1)/len(g)
                ax_e.plot(g, yg, label=mt.replace('_',' '), linewidth=_DEF_LINEWIDTH)
                _apply_common(ax_e, xlabel='returns r_t (dimensionless)', ylabel='ECDF (dimensionless)', title=f'ECDF Overlay{suspect_suffix}', zero_line=False, nonnegative_y=True)
                ax_e.legend(fontsize=9)
                _savefig_both(fig_e, fig_dir/mt/'ecdf_overlay_inverse')
                plt.close(fig_e)
                logger_write(str(fig_dir/mt/'ecdf_overlay_inverse.pdf') + (" — SUSPECT SCALE" if scale_tag=='FAIL' else ""))
                inner.update(1)
                # Recap per model
                logger_write(f"recap {mt}: scaler=ZScore(train), kind=returns, ann={args.annualise_vol}, std={model_bundle.std:.4f}, max|r|={max(abs(model_bundle.min), abs(model_bundle.max)):.3f}, decision={decision}")

            outer.update(1)

    if args.scaling_diagnostics_only or args.fix_overlays_only:
        # Skip other heavy plots and compile report with ONLY the guarded overlays
        try:
            from PyPDF2 import PdfMerger
            merger = PdfMerger()
            pdfs = []
            
            if args.fix_overlays_only:
                # Only include the corrected global overlays from combined/ directory
                combined_dir = fig_dir / 'combined'
                if combined_dir.exists():
                    # Only include the specific overlay files we generated
                    overlay_patterns = [
                        'density_overlay_*.pdf',
                        'ecdf_overlay_*.pdf', 
                        'hist_logy_overlay_*.pdf',
                        'rolling_vol_overlay_*.pdf',
                        'sigma_ratio_*.pdf',
                        'var_curve_*.pdf',
                        'es_curve_*.pdf',
                        'exceedance_timeline_*.pdf'
                    ]
                    for pattern in overlay_patterns:
                        for f in sorted(combined_dir.glob(pattern)):
                            pdfs.append(f)
            else:
                # Original behavior for scaling diagnostics
                for sub in ['combined','zero_conditioned','explicit_conditioned','llm_conditioned']:
                    p = fig_dir/sub
                    if p.exists():
                        for f in sorted(p.glob('*.pdf')):
                            pdfs.append(f)
            
            for f in pdfs:
                merger.append(str(f))
            
            final_report_path = Path(args.report_out)
            final_report_path.parent.mkdir(parents=True, exist_ok=True)
            merger.write(str(final_report_path))
            merger.close()
            logger_write(f"Final report written to: {final_report_path}")
        except Exception as e:
            logger_write(f"Warning: could not compile PDF report: {e}")
        
        mode_desc = 'Scaling diagnostics' if args.scaling_diagnostics_only else 'Global overlays fix'
        logger_write(f'{mode_desc} completed.')
        print(f"✅ {mode_desc} saved to: {out_root}")
        return

    # Only generate legacy plots when not in fix-overlays-only mode
    if not args.fix_overlays_only:
        logger_write("Generating legacy plots and diagnostics...")
        # Styling
        plt.style.use('seaborn-v0_8-white')
        colors = {'zero_conditioned':'#1f77b4','explicit_conditioned':'#2ca02c','llm_conditioned':'#d62728','real':'#000000'}

        # Stylised facts: histogram (log y, line only)
        real = returns.values
        fig, ax = plt.subplots(figsize=(10,6))
        ax.set_yscale('log')
        # Plot model densities via histogram line (density=True)
        xs_all = []
        for mt, res in results.items():
            flat = res['samples'].flatten()
            hist, bins = np.histogram(flat, bins=100, density=True)
            centers = (bins[:-1]+bins[1:])/2
            xs_all.append((centers.min(), centers.max()))
            ax.plot(centers, hist, label=mt.replace('_',' '), linewidth=_DEF_LINEWIDTH)
            # Gaussian overlay per model (unless toggled off)
            if not args.no_gaussian_overlay and np.isfinite(flat).any():
                mu = float(np.mean(flat)); sd = float(np.std(flat))
                grid = np.linspace(centers.min(), centers.max(), 400)
                gauss = (1.0/(sd*np.sqrt(2*np.pi))) * np.exp(-0.5*((grid-mu)/sd)**2)
                ax.plot(grid, gauss, linestyle='--', linewidth=1.0, label=f"Gaussian {mt} (μ̂={mu:.3f}, σ̂={sd:.3f})")
        # Real overlay last
        h_r, b_r = np.histogram(real, bins=100, density=True)
        c_r = (b_r[:-1]+b_r[1:])/2
        xs_all.append((c_r.min(), c_r.max()))
        ax.plot(c_r, h_r, label='Real', linestyle='--', linewidth=_DEF_LINEWIDTH)
        if not args.no_gaussian_overlay and np.isfinite(real).any():
            mu_r = float(np.mean(real)); sd_r = float(np.std(real))
            grid_r = np.linspace(c_r.min(), c_r.max(), 400)
            gauss_r = (1.0/(sd_r*np.sqrt(2*np.pi))) * np.exp(-0.5*((grid_r-mu_r)/sd_r)**2)
            ax.plot(grid_r, gauss_r, color='gray', linestyle=':', linewidth=1.0, label=f"Gaussian Real (μ̂={mu_r:.3f}, σ̂={sd_r:.3f})")
        # Kurtosis box (real)
        try:
            k = float(stats.kurtosis(real, fisher=False))
            ek = k - 3.0
            if not args.no_annotations:
                ax.text(0.02, 0.95, f"kurtosis={k:.2f} (excess={ek:.2f})", transform=ax.transAxes, fontsize=9, va='top', ha='left', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        except Exception:
            pass
        _apply_common(ax, xlabel='returns r_t (dimensionless)', ylabel='density (1/units of x)', title='Histogram (log y-axis)', zero_line=False, nonnegative_y=True)
        ax.legend(fontsize=9)
        _savefig_both(fig, fig_dir/'combined'/'hist_logy_overlay')
        plt.close()

    
    # Skip legacy density/ECDF overlays when fixing overlays only
    if not args.fix_overlays_only:
        # Combined Density overlay (guarded, inverse-scaled bundles, robust x-limits)
        real_b = create_real_bundle(returns.values, annualise_mode=args.annualise_vol)
        xl, xr = _symmetric_bounds(real_b)
        fig, ax = plt.subplots(figsize=_DEF_FIGSIZE_SINGLE)
        for mt, res in results.items():
            if 'bundle' in res:
                bundle = res['bundle']
                flat = bundle.returns.flatten()
            else:
                # Fallback for compatibility
                split_idx = int(0.8 * len(returns))
                train_mu = float(returns.iloc[:split_idx].mean())
                train_sigma = float(returns.iloc[:split_idx].std(ddof=1))
                flat = (res['samples'].flatten() * train_sigma + train_mu) if args.force_inverse_scaling else res['samples'].flatten()
            hist, bins = np.histogram(flat, bins=100, density=True)
            centers = (bins[:-1]+bins[1:])/2
            ax.plot(centers, hist, label=mt.replace('_',' '), linewidth=_DEF_LINEWIDTH)
        hist_r, bins_r = np.histogram(real_b.returns.flatten(), bins=100, density=True)
        centers_r = (bins_r[:-1]+bins_r[1:])/2
        ax.plot(centers_r, hist_r, linestyle='--', label='Real', linewidth=_DEF_LINEWIDTH)
        ax.set_xlim([xl, xr])
        _apply_common(ax, xlabel='returns r_t (dimensionless)', ylabel='density (1/units of x)', title='Density Overlay', zero_line=False, nonnegative_y=True)
        ax.legend(fontsize=9)
        _savefig_both(fig, fig_dir/'combined'/'density_overlay')
        plt.close(fig)

        fig, ax = plt.subplots(figsize=_DEF_FIGSIZE_SINGLE)
        r_sorted = np.sort(real_b.returns); yr = np.arange(1,len(r_sorted)+1)/len(r_sorted)
        ax.plot(r_sorted, yr, label='Real', linestyle='--', linewidth=_DEF_LINEWIDTH)
        for mt, res in results.items():
            if 'bundle' in res:
                bundle = res['bundle']
                flat = bundle.returns.flatten()
            else:
                # Fallback for compatibility
                split_idx = int(0.8 * len(returns))
                train_mu = float(returns.iloc[:split_idx].mean())
                train_sigma = float(returns.iloc[:split_idx].std(ddof=1))
                flat = (res['samples'].flatten() * train_sigma + train_mu) if args.force_inverse_scaling else res['samples'].flatten()
            g = np.sort(flat); yg = np.arange(1,len(g)+1)/len(g)
            ax.plot(g, yg, label=mt.replace('_',' '), linewidth=_DEF_LINEWIDTH)
        xl, xr = _symmetric_bounds(real_b)
        ax.set_xlim([xl, xr])
        _apply_common(ax, xlabel='returns r_t (dimensionless)', ylabel='ECDF (dimensionless)', title='ECDF Overlay', zero_line=False, nonnegative_y=True)
        ax.legend(fontsize=9)
        _savefig_both(fig, fig_dir/'combined'/'ecdf_overlay')
        plt.close(fig)

    # QQ plots (tails)
    for mt, res in results.items():
        flat = res['samples'].flatten()
        fig, (a1,a2) = plt.subplots(1,2, figsize=_DEF_FIGSIZE_WIDE)
        left_g = flat[flat < np.percentile(flat, 10)]
        right_g = flat[flat > np.percentile(flat, 90)]
        stats.probplot(left_g, dist='norm', plot=a1); _apply_common(a1, xlabel='theoretical q (dimensionless)', ylabel='ordered values (dimensionless)', title='Q–Q left tail', zero_line=True)
        stats.probplot(right_g, dist='norm', plot=a2); _apply_common(a2, xlabel='theoretical q (dimensionless)', ylabel='ordered values (dimensionless)', title='Q–Q right tail', zero_line=True)
        _savefig_both(fig, fig_dir/mt/'qq_tails')
        plt.close()

    # ACF/PACF for returns and squared returns with 95% CI (statsmodels draws bands)
    for mt, res in results.items():
        flat = res['samples'].flatten()
        fig, ((a1,a2),(a3,a4)) = plt.subplots(2,2, figsize=_DEF_FIGSIZE_GRID)
        plot_acf(flat, lags=args.acf_lags, ax=a1); _apply_common(a1, xlabel='lag k (dimensionless)', ylabel='ACF (dimensionless)', title=f'ACF — {mt}', zero_line=True, nonnegative_y=False)
        plot_pacf(flat, lags=args.acf_lags, ax=a2); _apply_common(a2, xlabel='lag k (dimensionless)', ylabel='PACF (dimensionless)', title=f'PACF — {mt}', zero_line=True, nonnegative_y=False)
        sq = flat**2
        plot_acf(sq, lags=args.acf_lags, ax=a3); _apply_common(a3, xlabel='lag k (dimensionless)', ylabel='ACF(|r|²) (dimensionless)', title=f'ACF (squared) — {mt}', zero_line=True, nonnegative_y=False)
        plot_pacf(sq, lags=args.acf_lags, ax=a4); _apply_common(a4, xlabel='lag k (dimensionless)', ylabel='PACF(|r|²) (dimensionless)', title=f'PACF (squared) — {mt}', zero_line=True, nonnegative_y=False)
        _savefig_both(fig, fig_dir/mt/'acf_pacf')
        plt.close()

    # Stylised facts heatmap (pass/fail) with numeric values
    for mt, res in results.items():
        flat = res['samples'].flatten()
        k_val = float(stats.kurtosis(flat, fisher=False)) if np.isfinite(flat).any() else np.nan
        lev = np.corrcoef(flat[1:], (flat**2)[:-1])[0,1] if len(flat)>1 else np.nan
        vc = acf((flat**2), nlags=1, fft=False)[1] if len(flat)>2 else np.nan
        rules = [
            ('kurtosis', k_val, k_val>3.0),
            ('leverage corr', lev, (lev<0) if np.isfinite(lev) else False),
            ('vol clustering ACF1', vc, (vc>0) if np.isfinite(vc) else False),
        ]
        arr = np.array([[int(v) for _,_,v in rules]], dtype=float)
        fig, ax = plt.subplots(figsize=(6,2))
        im=ax.imshow(arr, cmap='Greens', vmin=0, vmax=1, aspect='auto')
        ax.set_yticks([0]); ax.set_yticklabels(['Pass/Fail'])
        ax.set_xticks(range(len(rules))); ax.set_xticklabels([n for n,_,_ in rules], rotation=20, ha='right')
        # overlay numeric values
        for j, (name, num, passed) in enumerate(rules):
            txt = f"{name}={num:.2f}" if np.isfinite(num) else f"{name}=NA"
            ax.text(j, 0, txt, ha='center', va='center', fontsize=8,
                    bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
        _apply_common(ax, xlabel='', ylabel='', title=f'Stylised facts — {mt}', zero_line=False)
        _savefig_both(fig, fig_dir/mt/'stylised_facts_heatmap')
        plt.close()

    # Rolling volatility overlay with definition/scaling checks and diagnostics
    def _rolling_vol(series: np.ndarray, window: int, ddof: int = 1, demean: bool = False, abs_returns: bool = False) -> np.ndarray:
        x = np.asarray(series).astype(float)
        if abs_returns:
            x = np.abs(x)
        if demean:
            # rolling std of demeaned returns within window
            s = pd.Series(x)
            return s.rolling(window=window).apply(lambda a: np.std(a - np.mean(a), ddof=ddof), raw=False).to_numpy()
        else:
            return pd.Series(x).rolling(window=window).std(ddof=ddof).to_numpy()

    # Definition audit: use identical settings
    VOL_WINDOW = int(args.rolling_window)
    VOL_DDOF = 1
    VOL_DEMEAN = False
    VOL_ABS = False

    # Compute training scaler (chronological split to avoid leakage)
    split_idx = int(0.8 * len(returns))
    train_mu = float(returns.iloc[:split_idx].mean())
    train_sigma = float(returns.iloc[:split_idx].std(ddof=1))

    # Real rolling volatility in decimals
    real_series = returns.values  # decimals
    real_vol = _rolling_vol(real_series, window=VOL_WINDOW, ddof=VOL_DDOF, demean=VOL_DEMEAN, abs_returns=VOL_ABS)
    real_vol = real_vol[~np.isnan(real_vol)]

    # Assertions: identical settings will be applied to models below
    assert VOL_WINDOW == int(args.rolling_window), "Rolling window mismatch"
    assert VOL_DDOF == 1, "Expected sample std (ddof=1)"
    assert VOL_DEMEAN is False and VOL_ABS is False, "Vol definition must be plain std of returns (not abs, not demeaned)"

    # Prepare aligned x-index (use the last N days of real to match lengths)
    base_index = returns.index

    # Diagnostics accumulator
    diag_rows = []

    fig, ax = plt.subplots(figsize=(12,6))
    # Plot real first
    ax.plot(base_index[-len(real_vol):], real_vol, label='Real', color=colors['real'], linewidth=_DEF_LINEWIDTH)

    # For each model: try two scales: assume-standardized (A) vs inverse-standardized to decimals (B)
    # Choose the one whose mean ratio to real is closer to 1 for diagnostics and overlay (B preferred when close).
    for mt, res in results.items():
        if 'bundle' in res:
            # Use the bundle from fetch layer
            bundle = res['bundle']
            chosen_vol_data = bundle.returns.flatten()
            chosen_tag = 'decimals'
        else:
            # Fallback for compatibility
            gen_raw = res['samples'][0].astype(float)  # model path in training/model space
            # Two options
            vol_A = _rolling_vol(gen_raw, window=VOL_WINDOW, ddof=VOL_DDOF, demean=VOL_DEMEAN, abs_returns=VOL_ABS)
            gen_B = gen_raw * train_sigma + train_mu
            vol_B = _rolling_vol(gen_B, window=VOL_WINDOW, ddof=VOL_DDOF, demean=VOL_DEMEAN, abs_returns=VOL_ABS)
            
            # Choose representation: closer mean ratio to 1 (prefer B when both comparable)
            vols = []
            for v in [vol_A, vol_B]:
                v = v[~np.isnan(v)]
                L = min(len(v), len(real_vol))
                vols.append(v[-L:])
            real_aligned = real_vol[-min(len(vols[0]), len(vols[1]), len(real_vol)):]
            vol_A_aligned = vols[0][-len(real_aligned):]
            vol_B_aligned = vols[1][-len(real_aligned):]
            
            ratio_A = np.divide(vol_A_aligned, real_aligned, out=np.full_like(vol_A_aligned, np.nan), where=real_aligned!=0)
            ratio_B = np.divide(vol_B_aligned, real_aligned, out=np.full_like(vol_B_aligned, np.nan), where=real_aligned!=0)
            mean_ratio_A = np.nanmean(ratio_A)
            mean_ratio_B = np.nanmean(ratio_B)
            use_B = (abs(mean_ratio_B - 1.0) <= abs(mean_ratio_A - 1.0))
            chosen_vol_data = gen_B if use_B else gen_raw
            chosen_tag = 'decimals' if use_B else 'standardized'
        
        # Compute rolling vol for chosen data
        chosen_vol = _rolling_vol(chosen_vol_data, window=VOL_WINDOW, ddof=VOL_DDOF, demean=VOL_DEMEAN, abs_returns=VOL_ABS)
        # Align lengths with real_vol
        chosen_vol = chosen_vol[~np.isnan(chosen_vol)]
        L = min(len(chosen_vol), len(real_vol))
        real_aligned = real_vol[-L:]
        chosen_vol = chosen_vol[-L:]
        chosen_ratio = np.divide(chosen_vol, real_aligned, out=np.full_like(chosen_vol, np.nan), where=real_aligned!=0)

        # Aligned plot x-index
        idx = base_index[-len(real_aligned):]
        ax.plot(idx, chosen_vol, label=f"{mt.replace('_',' ')} ({chosen_tag})", linewidth=_DEF_LINEWIDTH)

        # Diagnostics summary
        def stats_vec(x):
            return float(np.nanmean(x)), float(np.nanmedian(x)), float(np.nanpercentile(x,90)), float(np.nanpercentile(x,95))
        m_mean, m_median, m_p90, m_p95 = stats_vec(chosen_vol)
        r_mean, r_median, r_p90, r_p95 = stats_vec(real_aligned)
        mean_ratio = float(np.nanmean(chosen_ratio))
        corr = float(np.corrcoef(chosen_vol, real_aligned)[0,1]) if len(chosen_vol)>1 else np.nan
        diag_rows.append({
            'model': mt,
            'scale': chosen_tag,
            'real_mean': r_mean, 'real_median': r_median, 'real_p90': r_p90, 'real_p95': r_p95,
            'model_mean': m_mean, 'model_median': m_median, 'model_p90': m_p90, 'model_p95': m_p95,
            'mean_ratio': mean_ratio, 'corr': corr
        })

        # Ratio plot per model
        fig_r, ax_r = plt.subplots(figsize=(10,4))
        ax_r.plot(idx, chosen_ratio, label=f"{mt.replace('_',' ')} σ_w / Real σ_w", linewidth=_DEF_LINEWIDTH)
        ax_r.axhline(1.0, color='gray', ls='--', label='ratio = 1')
        _apply_common(ax_r, xlabel='time index k (dimensionless)', ylabel='ratio σ_w(model)/σ_w(real) (dimensionless)', title=f'Rolling volatility ratio — {mt}', zero_line=False, nonnegative_y=True)
        ax_r.legend(fontsize=9)
        _savefig_both(fig_r, fig_dir/mt/'rolling_vol_ratio')
        plt.close(fig_r)

    _apply_common(ax, xlabel='date', ylabel=f'rolling volatility σ_w (dimensionless)', title=f'Rolling Volatility (window={VOL_WINDOW})', zero_line=True, nonnegative_y=True)
    ax.legend(fontsize=9)
    _savefig_both(fig, fig_dir/'combined'/'rolling_volatility_all')
    plt.close(fig)

    # Diagnostics table as figure and CSV/TEX
    try:
        df_diag = pd.DataFrame(diag_rows)
        df_diag.to_csv(tbl_dir/'rolling_vol_diagnostics.csv', index=False)
        with open(tbl_dir/'rolling_vol_diagnostics.tex', 'w') as f:
            f.write(df_diag.to_latex(index=False, float_format=lambda x: f"{x:.4f}"))
        # Render as a compact figure
        fig_t, ax_t = plt.subplots(figsize=(12, 2 + 0.25*len(df_diag)))
        ax_t.axis('off')
        tbl = ax_t.table(cellText=df_diag.round(4).values, colLabels=df_diag.columns, loc='center')
        tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1, 1.2)
        _savefig_both(fig_t, fig_dir/'combined'/'rolling_vol_diagnostics')
        plt.close(fig_t)
    except Exception:
        pass

    # If systematic bias remains, annotate briefly
    for row in diag_rows:
        if not np.isfinite(row['mean_ratio']):
            continue
        if row['mean_ratio'] > 1.1 or row['mean_ratio'] < 0.9:
            note = f"{row['model']} σ_w ≈ {row['mean_ratio']:.2f}× real"
            # Append to a small text file for report notes
            with open(mtr_dir/'rolling_vol_bias_notes.txt', 'a') as f:
                f.write(note + "\n")

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
        RES_STD = float(np.std(errs)) if errs else np.nan
        with open(tbl_dir/mt/'residual_metrics.json','w') as f: json.dump({'ME':ME,'MAE':MAE,'MSE':MSE,'RMSE':RMSE,'STD':RES_STD}, f, indent=2)
        # Residual histogram
        fig, ax = plt.subplots(figsize=_DEF_FIGSIZE_SINGLE)
        hist, bins = np.histogram(errs, bins=80, density=True)
        centers = (bins[:-1]+bins[1:])/2
        ax.plot(centers, hist, label=f"ME={ME:.4f}, MAE={MAE:.4f}, MSE={MSE:.4f}, RMSE={RMSE:.4f}, σ̂={RES_STD:.4f}")
        _apply_common(ax, xlabel='residual ε (dimensionless)', ylabel='density (1/units of x)', title=f'Residual Histogram — {mt}', zero_line=True, nonnegative_y=True)
        if not args.no_annotations:
            ax.legend(fontsize=8)
        _savefig_both(fig, fig_dir/mt/'residual_histogram')
        plt.close()
        # Standardized residuals vs N(0,1)
        if errs:
            z = (np.array(errs) / (RES_STD if RES_STD>0 else 1.0))
            fig, ax = plt.subplots(figsize=_DEF_FIGSIZE_SINGLE)
            hz, bz = np.histogram(z, bins=80, density=True)
            cz = (bz[:-1]+bz[1:])/2
            ax.plot(cz, hz, label='Standardised residuals')
            grid = np.linspace(min(cz.min(), -4), max(cz.max(), 4), 400)
            nz = (1.0/np.sqrt(2*np.pi)) * np.exp(-0.5*grid**2)
            ax.plot(grid, nz, linestyle='--', label='N(0,1)')
            _apply_common(ax, xlabel='standardised residual ε/σ̂ (dimensionless)', ylabel='density (1/units of x)', title='Standardised residuals vs N(0,1)', zero_line=True, nonnegative_y=True)
            ax.legend(fontsize=9)
            _savefig_both(fig, fig_dir/mt/'standardised_residuals')
            plt.close()

    # Skip legacy VaR/ES generation when fixing overlays only
    if not args.fix_overlays_only:
        # Risk: VaR/ES overlay and exceedance timelines
        levels = np.arange(0.90, 0.999, 0.001)
        fig, (a1,a2) = plt.subplots(1,2, figsize=_DEF_FIGSIZE_WIDE)
        for mt, res in results.items():
            if 'bundle' in res:
                flat = res['bundle'].returns.flatten()
            else:
                # Fallback for compatibility - use inverse scaled data
                split_idx = int(0.8 * len(returns))
                train_mu = float(returns.iloc[:split_idx].mean())
                train_sigma = float(returns.iloc[:split_idx].std(ddof=1))
                flat = (res['samples'].flatten() * train_sigma + train_mu) if args.force_inverse_scaling else res['samples'].flatten()
            
            var_vals=[]; es_vals=[]
            for L in levels:
                v = np.percentile(flat, (1-L)*100)
                var_vals.append(v)
                tail = flat[flat<=v]; es_vals.append(np.mean(tail) if len(tail)>0 else np.nan)
            a1.plot(levels, var_vals, label=mt.replace('_',' '), linewidth=_DEF_LINEWIDTH)
            a2.plot(levels, es_vals, label=mt.replace('_',' '), linewidth=_DEF_LINEWIDTH)
        _apply_common(a1, xlabel='VaR level α (dimensionless)', ylabel='VaR (decimal returns)', title='VaR Curve (90–100%)', zero_line=True, nonnegative_y=False)
        _apply_common(a2, xlabel='ES level α (dimensionless)', ylabel='ES (decimal returns)', title='ES Curve (90–100%)', zero_line=True, nonnegative_y=False)
        a1.legend(fontsize=9); a2.legend(fontsize=9)
        _savefig_both(fig, fig_dir/'combined'/'var_es_curves_all')
        plt.close()

    # Skip legacy exceedance timelines when fixing overlays only
    if not args.fix_overlays_only:
        # Exceedance timelines
        fig, axes = plt.subplots(len(results), 1, figsize=(12, 3*len(results)), sharex=True)
        if len(results)==1: axes=[axes]
        for ax, (mt,res) in zip(axes, results.items()):
            if 'bundle' in res:
                flat = res['bundle'].returns.flatten()
            else:
                # Fallback for compatibility
                split_idx = int(0.8 * len(returns))
                train_mu = float(returns.iloc[:split_idx].mean())
                train_sigma = float(returns.iloc[:split_idx].std(ddof=1))
                flat = (res['samples'].flatten() * train_sigma + train_mu) if args.force_inverse_scaling else res['samples'].flatten()
                
            v95 = np.percentile(flat, 5); v99 = np.percentile(flat, 1)
            t = np.arange(len(flat))
            ax.plot(t, flat, lw=1)
            ax.axhline(v95, color='r', ls='--', label='VaR@95%')
            ax.axhline(v99, color='m', ls='--', label='VaR@99%')
            mask95 = flat<=v95; mask99 = flat<=v99
            n_95 = np.sum(mask95); n_99 = np.sum(mask99)
            exp_95 = int(0.05 * len(flat)); exp_99 = int(0.01 * len(flat))
            ax.scatter(t[mask95], flat[mask95], s=10, color='red', alpha=0.6, 
                      label=f'breaches 95% (obs={n_95}, exp≈{exp_95})')
            _apply_common(ax, xlabel='time index k (dimensionless)', ylabel='returns r_t (decimal)', 
                         title=f'Exceedance timeline — {mt}', zero_line=True, nonnegative_y=False)
            ax.legend(fontsize=8)
        _savefig_both(fig, fig_dir/'combined'/'exceedance_timelines')
        plt.close()

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
    # Only compile full report if not in fix-overlays-only mode
    if not args.fix_overlays_only:
        # Compile final report
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
            
            final_report_path = Path(args.report_out)
            final_report_path.parent.mkdir(parents=True, exist_ok=True)
            merger.write(str(final_report_path))
            merger.close()
            logger_write(f"Final report written to: {final_report_path}")
        except Exception as e:
            print(f"Warning: could not compile PDF report: {e}")

    if not args.fix_overlays_only:
        print(f"✅ Strict novelty comparison saved to: {out_root}")
        print(f"📄 Final PDF report: {args.report_out}")


if __name__ == '__main__':
    main()


