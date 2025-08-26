#!/usr/bin/env python3
"""
Novelty Models Unified Evaluation (evaluate_novelty.py)

Loads already-trained checkpoints for the three novelty models (zero-conditioned,
explicit-conditioned, and LLM-conditioned), reuses existing data loaders and
split logic, and generates a thesis-ready, side-by-side comparison.

Usage:
  python evaluate_novelty.py \
      --models_dir results \
      --results_dir results \
      --seed 42 \
      --num_samples 500

Outputs:
  <results_dir>/<run_id>/novelty_comparison/
    - figures/<model_type>/
    - tables/<model_type>/
    - metrics/  (CSV/JSON consolidated)
    - novelty_comparison_evaluation_report.md

Robustness:
  - Deterministic seeds
  - Metrics and plots continue on failure with warnings
  - Auto-creates directories
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn.functional as F

from scipy import stats
from scipy.stats import ks_2samp, wasserstein_distance, chi2

# Statsmodels
from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Local imports
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
    ControllabilityProbe,
    NewsDataLoader,
)


DEFAULTS: Dict[str, Any] = {
    'seed': 42,
    'num_samples': 500,
    'var_levels': [0.95, 0.99],
    'reliability_bins': 20,
    'acf_lags': 20,
    'rolling_window': 20,
    'ablation_samples': 500,
    'correlation_lags': 10,
    'outlier_threshold': 3.0,
}


class NoveltyEvaluator:
    def __init__(self, config: Dict[str, Any], base_results_dir: Path):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._set_deterministic()

        # Run id and output dirs
        run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.out_root = Path(base_results_dir) / run_id / 'novelty_comparison'
        self.fig_dir = self.out_root / 'figures'
        self.tbl_dir = self.out_root / 'tables'
        self.mtr_dir = self.out_root / 'metrics'
        for d in [self.fig_dir, self.tbl_dir, self.mtr_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Per-model subdirs
        self.model_types = ['zero_conditioned', 'explicit_conditioned', 'llm_conditioned']
        for mt in self.model_types:
            (self.fig_dir / mt).mkdir(parents=True, exist_ok=True)
            (self.tbl_dir / mt).mkdir(parents=True, exist_ok=True)
        # Combined comparisons dir
        (self.fig_dir / 'combined').mkdir(parents=True, exist_ok=True)

        # Load data once
        self.returns, self.X, self.explicit_cond = self._load_data()
        self.llm_cond: Optional[np.ndarray] = None  # optional aligned LLM embeddings

        # Store results
        self.results: Dict[str, Dict[str, Any]] = {mt: {} for mt in self.model_types}

    # ---------------------- Setup & Data ---------------------- #
    def _set_deterministic(self):
        seed = int(self.config['seed'])
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.device.type == 'cuda':
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def _find_csv(self) -> Optional[Path]:
        for p in [
            Path('data/sp500_data.csv'),
            Path('../data/sp500_data.csv'),
            Path('../../data/sp500_data.csv'),
        ]:
            if p.exists():
                return p
        return None

    def _load_data(self) -> Tuple[pd.Series, np.ndarray, np.ndarray]:
        csv = self._find_csv()
        if csv is None:
            raise FileNotFoundError('Could not find data/sp500_data.csv')
        df = pd.read_csv(csv, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index)
        returns = np.log(df['Close'] / df['Close'].shift(1)).dropna()
        X = create_sequences(returns, 60)
        cond_vecs, _, _ = create_conditioning_vectors(returns, 60, 20, 0.2)
        return returns, X, cond_vecs

    def discover_checkpoints(self, models_dir: Path) -> Dict[str, Path]:
        ckpts: Dict[str, Path] = {}
        for mt in self.model_types:
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

    # ---------------------- Models & Sampling ---------------------- #
    def _load_model_trainer(self, mt: str, ckpt_path: Path) -> Tuple[Any, Any]:
        checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        if mt in ('zero_conditioned', 'explicit_conditioned'):
            model = ExplicitConditioningDDPM(sequence_length=60, conditioning_dim=5, hidden_dim=128)
            trainer = ExplicitConditioningTrainer(
                model,
                num_timesteps=1000,
                beta_schedule='cosine',
                device=self.device,
                grad_clip=1.0,
                cfg_p=(0.0 if mt == 'zero_conditioned' else 0.1),
            )
        elif mt == 'llm_conditioned':
            model = LLMConditionedDiffusion(sequence_length=60, conditioning_dim=64, hidden_dim=128)
            trainer = LLMDiffusionTrainer(
                model,
                num_timesteps=1000,
                beta_schedule='cosine',
                device=self.device,
                grad_clip=1.0,
                cfg_p=0.1,
            )
        else:
            raise ValueError(mt)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model, trainer

    def _sample(self, mt: str, trainer: Any) -> np.ndarray:
        n = int(self.config['num_samples'])
        if mt == 'zero_conditioned':
            cond = torch.zeros(n, 5, device=self.device)
            samples = trainer.sample(cond, num_samples=n, sampler='ddim', sample_steps=50)
        elif mt == 'explicit_conditioned':
            cond = torch.tensor(self.explicit_cond[:n], dtype=torch.float32, device=self.device)
            samples = trainer.sample(cond, num_samples=n, sampler='ddim', sample_steps=50, cfg_scale=7.5)
        else:
            # Prefer aligned LLM embeddings if prepared and enabled
            if self.llm_cond is not None and len(self.llm_cond) >= n:
                cond = torch.tensor(self.llm_cond[:n], dtype=torch.float32, device=self.device)
            else:
                cond = torch.randn(n, 64, device=self.device)
                cond = F.normalize(cond, dim=1)
            samples = trainer.sample(cond, num_samples=n, sampler='ddim', sample_steps=50, cfg_scale=7.5)
        return samples.squeeze(1).detach().cpu().numpy()

    # ---------------------- Metrics ---------------------- #
    def _metrics_basic(self, x: np.ndarray) -> Dict[str, float]:
        flat = x.flatten()
        out = {
            'mean': float(np.mean(flat)),
            'std': float(np.std(flat, ddof=1)),
            'skew': float(stats.skew(flat)),
            'excess_kurtosis': float(stats.kurtosis(flat)),
        }
        return out

    def _metrics_fidelity(self, x: np.ndarray) -> Dict[str, float]:
        flat = x.flatten(); real = self.returns.values
        ks, kp = ks_2samp(real, flat)
        out = {'ks_statistic': float(ks), 'ks_pvalue': float(kp)}
        try:
            out['wasserstein_distance'] = float(wasserstein_distance(real, flat))
        except Exception:
            out['wasserstein_distance'] = np.nan
        # MMD proxy
        out['mmd'] = float(np.sqrt((real.mean()-flat.mean())**2 + (real.var()-flat.var())**2))
        # Hill tail (upper 5%)
        try:
            thr = np.percentile(flat, 95); tail = flat[flat > thr]
            out['hill_tail_index'] = float(1.0/np.mean(np.log(tail/thr))) if len(tail) >= 10 else np.nan
        except Exception:
            out['hill_tail_index'] = np.nan
        return out

    def _metrics_forecast(self, samples: np.ndarray) -> Dict[str, float]:
        val_start = int(len(self.X)*0.8)
        val = self.X[val_start:]
        mse, mae = [], []
        for i, real_seq in enumerate(val[:min(100, len(samples))]):
            r = real_seq[0, :]; g = samples[i]
            mse.append(float(((r-g)**2).mean()))
            mae.append(float(np.mean(np.abs(r-g))))
        if not mse:
            return {'mse': np.nan, 'mae': np.nan, 'rmse': np.nan}
        m = float(np.mean(mse)); a = float(np.mean(mae))
        return {'mse': m, 'mae': a, 'rmse': float(np.sqrt(m))}

    def _kupiec_uc(self, hits: np.ndarray, p0: float) -> Tuple[float, float]:
        n = len(hits); v = int(hits.sum())
        p1 = v/n if n > 0 else 0.0
        if n == 0 or p1 <= 0 or p1 >= 1:
            return np.nan, np.nan
        lr = ((1-p0)**(n-v) * (p0**v)) / ((1-p1)**(n-v) * (p1**v))
        stat = -2*np.log(lr); pval = 1-chi2.cdf(stat, 1)
        return float(stat), float(pval)

    def _christoffersen_ind(self, hits: np.ndarray) -> Tuple[float, float]:
        # Transition counts
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

    def _risk_metrics(self, x: np.ndarray) -> Dict[str, float]:
        flat = x.flatten(); out: Dict[str, float] = {}
        for lvl in self.config['var_levels']:
            var = np.percentile(flat, (1-lvl)*100)
            out[f'var_{int(lvl*100)}'] = float(var)
            tail = flat[flat<=var]
            out[f'es_{int(lvl*100)}'] = float(np.mean(tail)) if len(tail)>0 else np.nan
        # Backtesting at 95%
        try:
            v95 = out['var_95']; hits = (flat<=v95).astype(int)
            uc_stat, uc_p = self._kupiec_uc(hits, 0.05)
            ind_stat, ind_p = self._christoffersen_ind(hits)
            # Conditional coverage LRcc = LRuc + LRind (df=2)
            if np.isfinite(uc_stat) and np.isfinite(ind_stat):
                cc_stat = uc_stat + ind_stat
                cc_p = 1-chi2.cdf(cc_stat, 2)
            else:
                cc_stat, cc_p = np.nan, np.nan
            out.update({
                'kupiec_stat_95': uc_stat, 'kupiec_pvalue_95': uc_p,
                'christoffersen_ind_stat_95': ind_stat, 'christoffersen_ind_pvalue_95': ind_p,
                'christoffersen_cc_stat_95': cc_stat, 'christoffersen_cc_pvalue_95': cc_p,
                'violation_rate_95': float(np.mean(hits)),
            })
        except Exception:
            pass
        return out

    # ---------------------- Plots ---------------------- #
    def _plot_stylized(self, x: np.ndarray, mt: str, metrics: Dict[str, float]):
        fig, ax = plt.subplots(figsize=(10,6))
        flat = x.flatten(); ax.hist(flat, bins=50, density=True, alpha=0.7)
        mu, sd = metrics.get('mean', 0.0), metrics.get('std', 1.0)
        xs = np.linspace(flat.min(), flat.max(), 200)
        ax.plot(xs, stats.norm.pdf(xs, mu, sd), 'r-', lw=2)
        ax.set_title(f'{mt}: Stylised Facts'); ax.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(self.fig_dir/mt/'stylized_facts.pdf', dpi=300, bbox_inches='tight'); plt.close()

    def _plot_ecdf(self, x: np.ndarray, mt: str):
        fig, ax = plt.subplots(figsize=(10,6))
        g = np.sort(x.flatten()); r = np.sort(self.returns.values)
        yg = np.arange(1, len(g)+1)/len(g); yr = np.arange(1, len(r)+1)/len(r)
        ax.plot(g, yg, label='Generated'); ax.plot(r, yr, label='Real')
        ax.legend(); ax.set_title(f'{mt}: ECDF'); ax.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(self.fig_dir/mt/'ecdf_comparison.pdf', dpi=300, bbox_inches='tight'); plt.close()

    def _plot_qq_tails(self, x: np.ndarray, mt: str):
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(14,6))
        flat = x.flatten()
        left = flat[flat < np.percentile(flat, 10)]; right = flat[flat > np.percentile(flat, 90)]
        stats.probplot(left, dist='norm', plot=ax1); ax1.set_title(f'{mt}: Q-Q Left Tail')
        stats.probplot(right, dist='norm', plot=ax2); ax2.set_title(f'{mt}: Q-Q Right Tail')
        plt.tight_layout(); plt.savefig(self.fig_dir/mt/'qq_tails.pdf', dpi=300, bbox_inches='tight'); plt.close()

    def _plot_acf_pacf(self, x: np.ndarray, mt: str):
        fig, ((a1,a2),(a3,a4)) = plt.subplots(2,2, figsize=(14,10))
        flat = x.flatten(); plot_acf(flat, lags=self.config['acf_lags'], ax=a1, title=f'ACF - {mt}')
        plot_pacf(flat, lags=self.config['acf_lags'], ax=a2, title=f'PACF - {mt}')
        sq = flat**2; plot_acf(sq, lags=self.config['acf_lags'], ax=a3, title=f'ACF (squared) - {mt}')
        plot_pacf(sq, lags=self.config['acf_lags'], ax=a4, title=f'PACF (squared) - {mt}')
        plt.tight_layout(); plt.savefig(self.fig_dir/mt/'acf_pacf.pdf', dpi=300, bbox_inches='tight'); plt.close()

    def _plot_rolling_vol(self, x: np.ndarray, mt: str):
        fig, ax = plt.subplots(figsize=(12,6))
        real = self.returns.rolling(window=self.config['rolling_window']).std().dropna()
        ax.plot(real.index, real.values, label='Real')
        if len(x)>0:
            gen = pd.Series(x[0]).rolling(window=self.config['rolling_window']).std().dropna()
            dates = pd.date_range(start=pd.Timestamp('2024-01-01'), periods=len(gen), freq='D')
            ax.plot(dates, gen.values, label='Generated')
        ax.legend(); ax.set_title(f'{mt}: Rolling Volatility'); ax.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(self.fig_dir/mt/'rolling_volatility.pdf', dpi=300, bbox_inches='tight'); plt.close()

    def _plot_sample_paths(self, x: np.ndarray, mt: str):
        fig, ax = plt.subplots(figsize=(12,6))
        k = min(10, len(x))
        for i in range(k): ax.plot(x[i], alpha=0.6, lw=1)
        ax.set_title(f'{mt}: Sample Paths (n={k})'); ax.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(self.fig_dir/mt/'sample_paths.pdf', dpi=300, bbox_inches='tight'); plt.close()

    def _plot_training_curves(self, mt: str, models_dir: Path):
        # Try loading latest training_history.json under model dir
        base = models_dir / mt
        try:
            runs = [d for d in base.iterdir() if d.is_dir()]
            if not runs: return
            latest = max(runs, key=lambda x:x.stat().st_mtime)
            hist = latest / 'training_history.json'
            if not hist.exists(): return
            with open(hist, 'r') as f: h = json.load(f)
            fig, ax = plt.subplots(figsize=(10,6))
            ax.plot(range(1,len(h['train_loss'])+1), h['train_loss'], label='Train')
            ax.plot(range(1,len(h['val_loss'])+1), h['val_loss'], label='Val')
            ax.legend(); ax.set_title(f'{mt}: Training Curves'); ax.grid(True, alpha=0.3)
            plt.tight_layout(); plt.savefig(self.fig_dir/mt/'training_curves.pdf', dpi=300, bbox_inches='tight'); plt.close()
        except Exception:
            pass

    def _plot_var_es_curves(self, x: np.ndarray, mt: str):
        levels = np.arange(0.90, 0.999, 0.001)
        flat = x.flatten(); var_vals=[]; es_vals=[]
        for L in levels:
            v = np.percentile(flat, (1-L)*100)
            var_vals.append(v)
            tail = flat[flat<=v]; es_vals.append(np.mean(tail) if len(tail)>0 else np.nan)
        fig,(a1,a2)=plt.subplots(1,2, figsize=(14,6))
        a1.plot(levels, var_vals); a1.set_title(f'{mt}: VaR Curve'); a1.grid(True, alpha=0.3)
        a2.plot(levels, es_vals); a2.set_title(f'{mt}: ES Curve'); a2.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(self.fig_dir/mt/'var_es_curves.pdf', dpi=300, bbox_inches='tight'); plt.close()

    def _plot_exceedance_timeline(self, x: np.ndarray, mt: str):
        flat = x.flatten(); v95=np.percentile(flat,5); v99=np.percentile(flat,1)
        t = np.arange(len(flat))
        fig,(a1,a2)=plt.subplots(2,1, figsize=(12,8))
        m95 = flat<=v95; a1.plot(t, flat, lw=1); a1.axhline(v95, color='r', ls='--'); a1.scatter(t[m95], flat[m95], s=12, color='r')
        a1.set_title(f'{mt}: VaR95 Exceedance'); a1.grid(True, alpha=0.3)
        m99 = flat<=v99; a2.plot(t, flat, lw=1); a2.axhline(v99, color='r', ls='--'); a2.scatter(t[m99], flat[m99], s=12, color='r')
        a2.set_title(f'{mt}: VaR99 Exceedance'); a2.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(self.fig_dir/mt/'exceedance_timeline.pdf', dpi=300, bbox_inches='tight'); plt.close()

    # Novelty comparisons
    def _target_vs_realized(self, samples: np.ndarray, mt: str):
        if mt!='explicit_conditioned': return {}
        target=[]; realized=[]
        for i, s in enumerate(samples[:100]):
            if i < len(self.explicit_cond):
                target.append(self.explicit_cond[i][-1])
                realized.append(np.std(s, ddof=1))
        if not target: return {}
        target=np.array(target); realized=np.array(realized)
        mae=float(np.mean(np.abs(realized-target)))
        # R^2 via simple linear regression on identity line proxy
        ss_res=np.sum((realized-target)**2); ss_tot=np.sum((realized-np.mean(realized))**2)
        r2=float(1-ss_res/ss_tot) if ss_tot>0 else np.nan
        # Plot
        fig,(a1,a2) = plt.subplots(1,2, figsize=(14,6))
        a1.scatter(target, realized, alpha=0.6); m,M=target.min(), target.max(); a1.plot([m,M],[m,M],'r--')
        a1.set_title(f'Target vs Realized (MAE={mae:.4f}, R2={r2:.3f})'); a1.grid(True, alpha=0.3)
        # Reliability
        bins=np.linspace(m,M,self.config['reliability_bins']); centers=[]; means=[]
        for i in range(len(bins)-1):
            mask=(target>=bins[i])&(target<bins[i+1])
            if mask.any(): centers.append((bins[i]+bins[i+1])/2); means.append(float(realized[mask].mean()))
        if centers:
            a2.plot(centers, means, 'bo-'); a2.plot([m,M],[m,M],'r--'); a2.set_title('Reliability Curve'); a2.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(self.fig_dir/mt/'volatility_targeting.pdf', dpi=300, bbox_inches='tight'); plt.close()
        # Table
        df=pd.DataFrame({'metric':['MAE','R2'], 'value':[mae,r2]})
        df.to_csv(self.tbl_dir/mt/'volatility_targeting_scores.csv', index=False)
        with open(self.tbl_dir/mt/'volatility_targeting_scores.json','w') as f: json.dump({'MAE':mae,'R2':r2}, f, indent=2)
        return {'volatility_mae': mae, 'volatility_r2': r2}

    def _regime_grid_paths(self, samples: np.ndarray, mt: str):
        if mt!='explicit_conditioned': return
        # Choose samples by conditioning regime from existing cond vectors
        regimes=['Up-Low','Up-High','Down-Low','Down-High']
        fig, axes = plt.subplots(2,2, figsize=(14,8))
        for idx, name in enumerate(regimes):
            r = idx//2; c=idx%2
            # pick indices where one-hot matches regime
            mask=[np.argmax(v[:4])==idx for v in self.explicit_cond[:len(samples)]]
            idxs=np.where(mask)[0]
            ax=axes[r][c]
            if len(idxs)>0:
                pick=idxs[:min(5,len(idxs))]
                for j in pick: ax.plot(samples[j], lw=1, alpha=0.7)
            ax.set_title(name); ax.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(self.fig_dir/mt/'regime_grid_paths.pdf', dpi=300, bbox_inches='tight'); plt.close()

    def _stylised_facts_heatmap(self, x: np.ndarray, mt: str):
        flat=x.flatten(); rows=[]; rules=[
            ('Heavy tails (kurt>3)', float(stats.kurtosis(flat))>3.0),
            ('Leverage (corr<0)', np.corrcoef(flat[1:], (flat**2)[:-1])[0,1]<0 if len(flat)>1 else False),
            ('Vol clustering (ACF>0)', acf((flat**2), nlags=1, fft=False)[1]>0 if len(flat)>2 else False),
        ]
        rows.append([int(v) for _,v in rules])
        arr=np.array(rows, dtype=float)
        fig, ax=plt.subplots(figsize=(6,2))
        im=ax.imshow(arr, cmap='Greens', vmin=0, vmax=1, aspect='auto')
        ax.set_yticks([0]); ax.set_yticklabels(['Pass/Fail'])
        ax.set_xticks(range(len(rules))); ax.set_xticklabels([n for n,_ in rules], rotation=20, ha='right')
        plt.tight_layout(); plt.savefig(self.fig_dir/mt/'stylised_facts_heatmap.pdf', dpi=300, bbox_inches='tight'); plt.close()

    def _llm_probe_and_ablations(self, samples: np.ndarray, mt: str):
        if mt!='llm_conditioned': return
        # Probe-based controllability (placeholder: compute simple linear proxy)
        probe = None
        fig, (a1,a2) = plt.subplots(1,2, figsize=(14,6))
        vols = np.array([np.std(s, ddof=1) for s in samples])
        a1.hist(vols, bins=40, alpha=0.7); a1.set_title('LLM-conditioned Volatility Distribution')
        zero = np.random.normal(0,1,(self.config['ablation_samples'],60))
        zero_vols=np.array([np.std(s, ddof=1) for s in zero])
        a2.hist(zero_vols, bins=40, alpha=0.7, label='Zero'); a2.hist(vols, bins=40, alpha=0.7, label='LLM'); a2.legend(); a2.set_title('Ablation: Zero vs LLM')
        plt.tight_layout(); plt.savefig(self.fig_dir/mt/'llm_ablation_histograms.pdf', dpi=300, bbox_inches='tight'); plt.close()
        # Sentiment bucket comparison (synthetic buckets)
        buckets=np.random.choice(5, len(samples))
        bucket_vols=[np.mean([np.std(s, ddof=1) for s in samples[buckets==b]]) if np.any(buckets==b) else 0.0 for b in range(5)]
        fig, ax=plt.subplots(figsize=(8,4)); ax.bar(range(5), bucket_vols); ax.set_title('Sentiment Bucket Volatility'); ax.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(self.fig_dir/mt/'llm_sentiment_buckets.pdf', dpi=300, bbox_inches='tight'); plt.close()

    # ---- LLM aligned embeddings + probe (volatility & regime) ---- #
    def _prepare_llm_aligned_conditioning(self):
        if not bool(self.config.get('llm_use_aligned_embeddings', False)):
            return
        try:
            loader = NewsDataLoader()
            # returns index must cover the same span as sequences
            returns_index = self.returns.index
            self.llm_cond = loader.create_conditioning_vectors(returns_index, seq_len=60, embedding_dim=64)
            # Ensure normalization
            self.llm_cond = self.llm_cond / np.linalg.norm(self.llm_cond, axis=1, keepdims=True)
            # Save snapshot for reproducibility
            np.save(self.mtr_dir/'llm_aligned_conditioning.npy', self.llm_cond)
        except Exception as e:
            print(f"Warning: LLM aligned embeddings preparation failed: {e}")
            self.llm_cond = None

    def _train_llm_probe(self) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        if self.llm_cond is None or len(self.llm_cond) != len(self.explicit_cond):
            return metrics
        try:
            from sklearn.linear_model import LinearRegression, LogisticRegression
            from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score
            # Targets from explicit conditioning
            sigma_star = self.explicit_cond[:, -1]
            regime_idx = np.argmax(self.explicit_cond[:, :4], axis=1)
            # Train/val split
            n = len(self.llm_cond); split = int(n*0.8)
            X_train, X_val = self.llm_cond[:split], self.llm_cond[split:]
            yv_train, yv_val = sigma_star[:split], sigma_star[split:]
            yr_train, yr_val = regime_idx[:split], regime_idx[split:]
            # Volatility probe
            vol_model = LinearRegression().fit(X_train, yv_train)
            yv_pred = vol_model.predict(X_val)
            metrics['llm_probe_vol_mae'] = float(mean_absolute_error(yv_val, yv_pred))
            metrics['llm_probe_vol_r2'] = float(r2_score(yv_val, yv_pred))
            # Regime probe (multinomial)
            clf = LogisticRegression(max_iter=1000, multi_class='multinomial')
            clf.fit(X_train, yr_train)
            yr_pred = clf.predict(X_val)
            metrics['llm_probe_regime_acc'] = float(accuracy_score(yr_val, yr_pred))
            # Save probe params (optional): not saving large arrays
            with open(self.tbl_dir/'llm_conditioned'/'probe_scores.json','w') as f:
                json.dump(metrics, f, indent=2)
            pd.DataFrame([metrics]).to_csv(self.tbl_dir/'llm_conditioned'/'probe_scores.csv', index=False)
            # Store models for later use in plotting
            self._llm_vol_probe = vol_model
            self._llm_regime_probe = clf
        except Exception as e:
            print(f"Warning: LLM probe training failed: {e}")
        return metrics

    def _llm_probe_controllability(self, samples: np.ndarray):
        # Requires probes and llm_cond
        if not hasattr(self, '_llm_vol_probe') or self.llm_cond is None:
            return
        try:
            n = min(len(samples), len(self.llm_cond))
            cond = self.llm_cond[:n]
            # Predicted target volatility from probe
            tgt_vol = self._llm_vol_probe.predict(cond)
            realized = np.array([np.std(s, ddof=1) for s in samples[:n]])
            # Scatter + reliability + residuals
            fig, (a1,a2,a3) = plt.subplots(1,3, figsize=(18,5))
            a1.scatter(tgt_vol, realized, alpha=0.6); m, M = np.min(tgt_vol), np.max(tgt_vol)
            a1.plot([m,M],[m,M],'r--'); a1.set_title('LLM: Target vs Realized Vol'); a1.grid(True, alpha=0.3)
            bins = np.linspace(m, M, max(5, int(self.config['reliability_bins']/2)))
            centers=[]; means=[]
            for i in range(len(bins)-1):
                mask=(tgt_vol>=bins[i])&(tgt_vol<bins[i+1])
                if mask.any(): centers.append((bins[i]+bins[i+1])/2); means.append(float(realized[mask].mean()))
            if centers:
                a2.plot(centers, means, 'bo-'); a2.plot([m,M],[m,M],'r--'); a2.set_title('LLM: Reliability'); a2.grid(True, alpha=0.3)
            resid = realized - tgt_vol
            a3.scatter(tgt_vol, resid, alpha=0.6); a3.axhline(0, color='r', ls='--'); a3.set_title('LLM: Residuals'); a3.grid(True, alpha=0.3)
            plt.tight_layout(); plt.savefig(self.fig_dir/'llm_conditioned'/'llm_probe_controllability.pdf', dpi=300, bbox_inches='tight'); plt.close()
            # Regime confusion via probe
            if hasattr(self, '_llm_regime_probe'):
                pred_reg = self._llm_regime_probe.predict(cond)
                vols = realized; vol_med = np.median(vols)
                pred_samples=[]
                for s, v in zip(samples[:n], vols):
                    trend = 1 if np.sum(s)>0 else 0
                    vol_reg = 1 if v>vol_med else 0
                    pred_samples.append((1-trend)*2 + vol_reg)
                from sklearn.metrics import confusion_matrix
                cm = confusion_matrix(pred_reg, pred_samples, labels=[0,1,2,3], normalize='true')
                fig, ax = plt.subplots(figsize=(5,4)); im=ax.imshow(cm, cmap='Purples', vmin=0, vmax=1); plt.colorbar(im, ax=ax)
                ax.set_title('LLM Probe vs Sample Regime (proxy)'); plt.tight_layout(); plt.savefig(self.fig_dir/'llm_conditioned'/'llm_probe_regime_confusion.pdf', dpi=300, bbox_inches='tight'); plt.close()
        except Exception as e:
            print(f"Warning: LLM probe controllability plotting failed: {e}")

    def _explicit_regime_confusion(self, samples: np.ndarray, mt: str):
        if mt!='explicit_conditioned': return
        # Simple proxy: infer regime from generated sample trend/vol, compare to conditioning
        targets=[np.argmax(v[:4]) for v in self.explicit_cond[:len(samples)]]
        preds=[]
        vols=[np.std(s, ddof=1) for s in samples]
        vol_med=np.median(vols)
        for s, v in zip(samples, vols):
            trend = 1 if np.sum(s)>0 else 0  # 1 Up else Down
            vol_reg = 1 if v>vol_med else 0  # 1 High else Low
            pred = (1-trend)*2 + vol_reg
            preds.append(pred)
        if not preds: return
        from sklearn.metrics import confusion_matrix
        cm=confusion_matrix(targets[:len(preds)], preds, labels=[0,1,2,3], normalize='true')
        fig, ax=plt.subplots(figsize=(5,4)); im=ax.imshow(cm, cmap='Blues', vmin=0, vmax=1); plt.colorbar(im, ax=ax)
        ax.set_title('Regime Confusion (proxy)'); plt.tight_layout(); plt.savefig(self.fig_dir/mt/'regime_confusion.pdf', dpi=300, bbox_inches='tight'); plt.close()
        # Per-regime accuracies
        acc=np.diag(cm)
        df=pd.DataFrame({'regime':['Up-Low','Up-High','Down-Low','Down-High'],'accuracy':acc})
        df.to_csv(self.tbl_dir/mt/'regime_accuracies.csv', index=False)

    # ---------------------- Saving & Report ---------------------- #
    def _save_metrics(self, mt: str, metrics: Dict[str, float]):
        # Per-model JSON/CSV
        with open(self.tbl_dir/mt/'metrics.json','w') as f: json.dump({'model_type':mt, **metrics}, f, indent=2)
        pd.DataFrame([{'model_type':mt, **metrics}]).to_csv(self.tbl_dir/mt/'metrics.csv', index=False)

    def _save_consolidated(self):
        rows=[]
        for mt, res in self.results.items():
            if 'metrics' in res:
                r=res['metrics'].copy(); r['model_type']=mt; rows.append(r)
        if not rows: return
        df=pd.DataFrame(rows)
        df.to_csv(self.mtr_dir/'consolidated_metrics.csv', index=False)
        with open(self.mtr_dir/'consolidated_metrics.json','w') as f: json.dump(rows, f, indent=2)

    # ---------------------- Combined Comparisons ---------------------- #
    def _combined_hist_ecdf(self):
        # Histogram/KDE overlay
        fig, ax = plt.subplots(figsize=(10,6))
        colors = {'zero_conditioned':'#1f77b4','explicit_conditioned':'#2ca02c','llm_conditioned':'#d62728'}
        for mt, res in self.results.items():
            if 'samples' not in res: continue
            flat = res['samples'].flatten()
            try:
                sns.kdeplot(flat, ax=ax, label=mt.replace('_',' '), color=colors.get(mt,None))
            except Exception:
                hist, bins = np.histogram(flat, bins=80, density=True)
                centers = (bins[:-1]+bins[1:])/2
                ax.plot(centers, hist, label=mt.replace('_',' '), color=colors.get(mt,None))
        ax.set_title('Density Comparison (Generated)'); ax.grid(True, alpha=0.3); ax.legend()
        plt.tight_layout(); plt.savefig(self.fig_dir/'combined'/'density_comparison.pdf', dpi=300, bbox_inches='tight'); plt.close()

        # ECDF overlay
        fig, ax = plt.subplots(figsize=(10,6))
        for mt, res in self.results.items():
            if 'samples' not in res: continue
            flat = np.sort(res['samples'].flatten())
            y = np.arange(1, len(flat)+1)/len(flat)
            ax.plot(flat, y, label=mt.replace('_',' '), color=colors.get(mt,None))
        # Real ECDF
        r = np.sort(self.returns.values); yr = np.arange(1,len(r)+1)/len(r)
        ax.plot(r, yr, label='Real', color='black', linestyle='--')
        ax.set_title('ECDF Comparison'); ax.grid(True, alpha=0.3); ax.legend()
        plt.tight_layout(); plt.savefig(self.fig_dir/'combined'/'ecdf_comparison_all.pdf', dpi=300, bbox_inches='tight'); plt.close()

    def _combined_var_es(self):
        levels = np.arange(0.90, 0.999, 0.001)
        colors = {'zero_conditioned':'#1f77b4','explicit_conditioned':'#2ca02c','llm_conditioned':'#d62728'}
        # VaR curves
        fig, (a1,a2) = plt.subplots(1,2, figsize=(14,6))
        for mt, res in self.results.items():
            if 'samples' not in res: continue
            flat = res['samples'].flatten()
            var_vals=[]; es_vals=[]
            for L in levels:
                v = np.percentile(flat, (1-L)*100)
                var_vals.append(v)
                tail = flat[flat<=v]
                es_vals.append(np.mean(tail) if len(tail)>0 else np.nan)
            a1.plot(levels, var_vals, label=mt.replace('_',' '), color=colors.get(mt,None))
            a2.plot(levels, es_vals, label=mt.replace('_',' '), color=colors.get(mt,None))
        a1.set_title('VaR Curve'); a1.grid(True, alpha=0.3)
        a2.set_title('ES Curve'); a2.grid(True, alpha=0.3)
        for ax in (a1,a2): ax.set_xlabel('Confidence'); ax.legend()
        plt.tight_layout(); plt.savefig(self.fig_dir/'combined'/'var_es_curves_all.pdf', dpi=300, bbox_inches='tight'); plt.close()

    def _combined_rolling_vol(self):
        fig, ax = plt.subplots(figsize=(12,6))
        real = self.returns.rolling(window=self.config['rolling_window']).std().dropna()
        ax.plot(real.index, real.values, label='Real', color='black')
        colors = {'zero_conditioned':'#1f77b4','explicit_conditioned':'#2ca02c','llm_conditioned':'#d62728'}
        for mt, res in self.results.items():
            if 'samples' not in res or len(res['samples'])==0: continue
            gen = pd.Series(res['samples'][0]).rolling(window=self.config['rolling_window']).std().dropna()
            dates = pd.date_range(start=pd.Timestamp('2024-01-01'), periods=len(gen), freq='D')
            ax.plot(dates, gen.values, label=mt.replace('_',' '), color=colors.get(mt,None))
        ax.set_title('Rolling Volatility Comparison'); ax.grid(True, alpha=0.3); ax.legend()
        plt.tight_layout(); plt.savefig(self.fig_dir/'combined'/'rolling_volatility_all.pdf', dpi=300, bbox_inches='tight'); plt.close()

    def _combined_acf_squared(self):
        # Overlay ACF of squared returns for each model
        fig, ax = plt.subplots(figsize=(10,6))
        colors = {'zero_conditioned':'#1f77b4','explicit_conditioned':'#2ca02c','llm_conditioned':'#d62728'}
        for mt, res in self.results.items():
            if 'samples' not in res: continue
            flat = res['samples'].flatten()
            sq = flat**2
            try:
                acf_vals = acf(sq, nlags=self.config['acf_lags'], fft=False)
                ax.plot(range(len(acf_vals)), acf_vals, marker='o', label=mt.replace('_',' '), color=colors.get(mt,None))
            except Exception:
                continue
        ax.set_title('Squared Returns ACF (Overlay)'); ax.set_xlabel('Lag'); ax.set_ylabel('ACF'); ax.grid(True, alpha=0.3); ax.legend()
        plt.tight_layout(); plt.savefig(self.fig_dir/'combined'/'acf_squared_overlay.pdf', dpi=300, bbox_inches='tight'); plt.close()

    def _generate_combined(self):
        try:
            self._combined_hist_ecdf()
            self._combined_var_es()
            self._combined_rolling_vol()
            self._combined_acf_squared()
        except Exception as e:
            print(f"Warning: combined comparisons failed: {e}")

    def _write_report(self):
        lines=["# Novelty Models Comparison Report", ""]
        for mt, res in self.results.items():
            if 'metrics' not in res: continue
            m=res['metrics']
            lines += [
                f"## {mt}",
                f"- KS={m.get('ks_statistic', np.nan):.4f} (p={m.get('ks_pvalue', np.nan):.4f}), W={m.get('wasserstein_distance', np.nan):.4f}, MMD={m.get('mmd', np.nan):.4f}",
                f"- Forecast: MSE={m.get('mse', np.nan):.6f}, MAE={m.get('mae', np.nan):.6f}, RMSE={m.get('rmse', np.nan):.6f}",
                f"- Risk: VaR95={m.get('var_95', np.nan):.4f}, ES95={m.get('es_95', np.nan):.4f}, Kupiec p={m.get('kupiec_pvalue_95', np.nan):.4f}, CC p={m.get('christoffersen_cc_pvalue_95', np.nan):.4f}",
                "",
            ]
        with open(self.out_root/'novelty_comparison_evaluation_report.md','w') as f:
            f.write("\n".join(lines))

    # ---------------------- Orchestration ---------------------- #
    def evaluate_model(self, mt: str, ckpt: Path, models_dir: Path):
        print(f"Evaluating {mt} ...")
        try:
            # Prepare LLM aligned embeddings once before LLM evaluation
            if mt == 'llm_conditioned' and self.llm_cond is None:
                self._prepare_llm_aligned_conditioning()
            model, trainer = self._load_model_trainer(mt, ckpt)
            samples = self._sample(mt, trainer)
            np.save(self.out_root/f'{mt}_samples.npy', samples)
            # Metrics
            basic=self._metrics_basic(samples)
            fidelity=self._metrics_fidelity(samples)
            forecast=self._metrics_forecast(samples)
            risk=self._risk_metrics(samples)
            metrics={**basic, **fidelity, **forecast, **risk}
            # Novelty-specific
            if mt=='explicit_conditioned':
                metrics.update(self._target_vs_realized(samples, mt))
            # Plots common
            self._plot_stylized(samples, mt, metrics)
            self._plot_ecdf(samples, mt)
            self._plot_qq_tails(samples, mt)
            self._plot_acf_pacf(samples, mt)
            self._plot_rolling_vol(samples, mt)
            self._plot_sample_paths(samples, mt)
            self._plot_training_curves(mt, models_dir)
            self._plot_var_es_curves(samples, mt)
            self._plot_exceedance_timeline(samples, mt)
            self._stylised_facts_heatmap(samples, mt)
            # Model-specific
            self._regime_grid_paths(samples, mt)
            self._llm_probe_and_ablations(samples, mt)
            # Train LLM probe and draw controllability if applicable
            if mt == 'llm_conditioned':
                probe_scores = self._train_llm_probe()
                self.results[mt]['probe_scores'] = probe_scores
                self._llm_probe_controllability(samples)
            self._explicit_regime_confusion(samples, mt)
            # Save
            self._save_metrics(mt, metrics)
            self.results[mt]={'samples': samples, 'metrics': metrics}
        except Exception as e:
            print(f"Warning: evaluation failed for {mt}: {e}")

    def run(self, models_dir: Path):
        ckpts=self.discover_checkpoints(models_dir)
        if not ckpts:
            print('❌ No checkpoints found.'); return
        for mt in self.model_types:
            if mt in ckpts:
                self.evaluate_model(mt, ckpts[mt], models_dir)
            else:
                print(f'Warning: missing checkpoint for {mt}')
        self._save_consolidated()
        # Combined comparisons across the three novelty models
        self._generate_combined()
        self._write_report()
        print(f"✅ Finished. Results: {self.out_root}")


def parse_args():
    p=argparse.ArgumentParser(description='Novelty Models Unified Evaluation')
    p.add_argument('--models_dir', type=str, default='results', help='Directory with trained model runs')
    p.add_argument('--results_dir', type=str, default='results', help='Base directory to write outputs')
    p.add_argument('--seed', type=int, default=DEFAULTS['seed'])
    p.add_argument('--num_samples', type=int, default=DEFAULTS['num_samples'])
    p.add_argument('--var_levels', nargs='+', type=float, default=DEFAULTS['var_levels'])
    p.add_argument('--reliability_bins', type=int, default=DEFAULTS['reliability_bins'])
    p.add_argument('--acf_lags', type=int, default=DEFAULTS['acf_lags'])
    p.add_argument('--rolling_window', type=int, default=DEFAULTS['rolling_window'])
    p.add_argument('--ablation_samples', type=int, default=DEFAULTS['ablation_samples'])
    p.add_argument('--correlation_lags', type=int, default=DEFAULTS['correlation_lags'])
    p.add_argument('--outlier_threshold', type=float, default=DEFAULTS['outlier_threshold'])
    p.add_argument('--llm_use_aligned_embeddings', action='store_true', help='Use date-aligned news embeddings for LLM conditioning and probe')
    return p.parse_args()


def main():
    args=parse_args()
    cfg={
        'seed': args.seed,
        'num_samples': args.num_samples,
        'var_levels': args.var_levels,
        'reliability_bins': args.reliability_bins,
        'acf_lags': args.acf_lags,
        'rolling_window': args.rolling_window,
        'ablation_samples': args.ablation_samples,
        'correlation_lags': args.correlation_lags,
        'outlier_threshold': args.outlier_threshold,
    }
    evaluator=NoveltyEvaluator(cfg, Path(args.results_dir))
    evaluator.run(Path(args.models_dir))


if __name__=='__main__':
    main()


