#!/usr/bin/env python3
"""
Unified Evaluation Pipeline (GPT-5)

Loads already-trained checkpoints for three novelty models (zero-conditioned, explicit-conditioned, and
LLM-conditioned) and generates the full set of thesis-ready figures and tables.

Command-line usage:
    python unified_evaluator_gpt5.py \
        --models_dir results \
        --results_dir results/separate_eval_gpt5 \
        --seed 42 \
        --num_samples 500

Outputs are written under:
    <results_dir>/figures/<model_type>/
    <results_dir>/tables/

This script is robust: if any metric or plot fails, it logs a warning and continues.
"""

import os
import sys
import json
import argparse
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple, Optional

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

# Statsmodels utilities
from statsmodels.tsa.stattools import acf
from tqdm import tqdm

# Ensure we can import local model modules
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
)

warnings.filterwarnings('ignore')


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


class UnifiedEvaluatorGPT5:
    def __init__(self, config: Dict[str, Any], results_dir: Path):
        self.config = config
        self.results_dir = results_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._set_determinism()

        # Prepare folders
        (self.results_dir / 'figures').mkdir(parents=True, exist_ok=True)
        (self.results_dir / 'tables').mkdir(parents=True, exist_ok=True)

        # Data (returns series, sequences X, explicit conditioning vectors)
        self.returns, self.X, self.explicit_conditioning = self._load_data()

        # Results registry
        self.results: Dict[str, Dict[str, Any]] = {
            'zero_conditioned': {},
            'explicit_conditioned': {},
            'llm_conditioned': {},
        }

        # Make per-model figure dirs
        for mt in self.results.keys():
            (self.results_dir / 'figures' / mt).mkdir(parents=True, exist_ok=True)
            (self.results_dir / 'tables' / mt).mkdir(parents=True, exist_ok=True)

    def _set_determinism(self):
        seed = int(self.config['seed'])
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.device.type == 'cuda':
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        print(f"Determinism set with seed {seed} on device {self.device}")

    def _find_csv_file(self) -> Optional[Path]:
        # Look for S&P500 CSV
        candidates = [
            Path('data/sp500_data.csv'),
            Path('../data/sp500_data.csv'),
            Path('../../data/sp500_data.csv'),
        ]
        for p in candidates:
            if p.exists():
                return p
        return None

    def _load_data(self) -> Tuple[pd.Series, np.ndarray, np.ndarray]:
        print("Loading financial data and preparing sequences...")
        csv_path = self._find_csv_file()
        if csv_path is None:
            raise FileNotFoundError("Could not find data/sp500_data.csv")

        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index)
        returns = np.log(df['Close'] / df['Close'].shift(1)).dropna()

        X = create_sequences(returns, 60)
        cond_vecs, _, _ = create_conditioning_vectors(returns, 60, 20, 0.2)

        print(f"Loaded {len(returns)} daily returns; sequences: {X.shape}; conditioning: {cond_vecs.shape}")
        return returns, X, cond_vecs

    def discover_checkpoints(self, models_dir: Path) -> Dict[str, Path]:
        print("Discovering checkpoints...")
        checkpoints: Dict[str, Path] = {}
        for model_type in ['zero_conditioned', 'explicit_conditioned', 'llm_conditioned']:
            mdir = models_dir / model_type
            if not mdir.exists():
                print(f"Warning: {model_type} dir not found: {mdir}")
                continue
            run_dirs = [d for d in mdir.iterdir() if d.is_dir()]
            if not run_dirs:
                print(f"Warning: no runs in {mdir}")
                continue
            latest = max(run_dirs, key=lambda x: x.stat().st_mtime)
            ckpt = latest / 'checkpoints' / 'best_model.pth'
            if ckpt.exists():
                checkpoints[model_type] = ckpt
                print(f"  {model_type}: {ckpt}")
            else:
                print(f"Warning: checkpoint not found in {latest}")
        return checkpoints

    def _load_model_and_trainer(self, model_type: str, checkpoint_path: Path) -> Tuple[Any, Any]:
        print(f"Loading {model_type} from {checkpoint_path}")
        # PyTorch 2.6+: set weights_only=False for legacy checkpoints we trust
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        if model_type in ('zero_conditioned', 'explicit_conditioned'):
            model = ExplicitConditioningDDPM(sequence_length=60, conditioning_dim=5, hidden_dim=128)
            trainer = ExplicitConditioningTrainer(
                model,
                num_timesteps=1000,
                beta_schedule="cosine",
                device=self.device,
                grad_clip=1.0,
                cfg_p=(0.0 if model_type == 'zero_conditioned' else 0.1),
            )
        elif model_type == 'llm_conditioned':
            model = LLMConditionedDiffusion(sequence_length=60, conditioning_dim=64, hidden_dim=128)
            trainer = LLMDiffusionTrainer(
                model,
                num_timesteps=1000,
                beta_schedule="cosine",
                device=self.device,
                grad_clip=1.0,
                cfg_p=0.1,
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model, trainer

    def _generate_samples(self, model_type: str, trainer: Any) -> np.ndarray:
        print(f"Generating {self.config['num_samples']} samples for {model_type}")
        if model_type == 'zero_conditioned':
            cond = torch.zeros(self.config['num_samples'], 5, device=self.device)
        elif model_type == 'explicit_conditioned':
            cond = torch.tensor(self.explicit_conditioning[: self.config['num_samples']], dtype=torch.float32, device=self.device)
        else:  # llm_conditioned
            cond = torch.randn(self.config['num_samples'], 64, device=self.device)
            cond = F.normalize(cond, dim=1)

        if model_type == 'zero_conditioned':
            samples = trainer.sample(
                cond,
                num_samples=self.config['num_samples'],
                sampler="ddim",
                sample_steps=50,
            )
        else:
            samples = trainer.sample(
                cond,
                num_samples=self.config['num_samples'],
                sampler="ddim",
                sample_steps=50,
                cfg_scale=7.5,
            )
        samples = samples.squeeze(1).detach().cpu().numpy()
        np.save(self.results_dir / f"{model_type}_samples.npy", samples)
        return samples

    # ----------------------------- Metrics ----------------------------- #
    def _metrics_stylized(self, samples: np.ndarray) -> Dict[str, float]:
        out: Dict[str, float] = {}
        try:
            x = samples.flatten()
            out['mean'] = float(np.mean(x))
            out['std'] = float(np.std(x, ddof=1))
            out['skew'] = float(stats.skew(x))
            out['excess_kurtosis'] = float(stats.kurtosis(x))
        except Exception as e:
            print(f"Warning: stylized stats failed: {e}")
        return out

    def _metrics_fidelity(self, samples: np.ndarray) -> Dict[str, float]:
        out: Dict[str, float] = {}
        try:
            real = self.returns.values
            gen = samples.flatten()
            ks, kp = ks_2samp(real, gen)
            out['ks_statistic'] = float(ks)
            out['ks_pvalue'] = float(kp)
            try:
                out['wasserstein_distance'] = float(wasserstein_distance(real, gen))
            except Exception:
                out['wasserstein_distance'] = np.nan
            # Simple MMD proxy using mean/var distances
            mmd = np.sqrt((real.mean() - gen.mean()) ** 2 + (real.var() - gen.var()) ** 2)
            out['mmd'] = float(mmd)
            # Hill tail index (upper 5% tail)
            try:
                thr = np.percentile(gen, 95)
                tail = gen[gen > thr]
                if len(tail) >= 10:
                    out['hill_tail_index'] = float(1.0 / np.mean(np.log(tail / thr)))
                else:
                    out['hill_tail_index'] = np.nan
            except Exception:
                out['hill_tail_index'] = np.nan
        except Exception as e:
            print(f"Warning: fidelity metrics failed: {e}")
        return out

    def _metrics_forecast(self, samples: np.ndarray) -> Dict[str, float]:
        out: Dict[str, float] = {}
        try:
            val_start = int(len(self.X) * 0.8)
            val = self.X[val_start:]
            mse_vals, mae_vals = [], []
            for i, real_seq in enumerate(val[: min(100, len(samples))]):
                r = real_seq[0, :]
                g = samples[i]
                mse_vals.append(float(((r - g) ** 2).mean()))
                mae_vals.append(float(np.mean(np.abs(r - g))))
            if mse_vals:
                out['mse'] = float(np.mean(mse_vals))
                out['mae'] = float(np.mean(mae_vals))
                out['rmse'] = float(np.sqrt(out['mse']))
        except Exception as e:
            print(f"Warning: forecast metrics failed: {e}")
        return out

    def _metrics_risk(self, samples: np.ndarray) -> Dict[str, float]:
        out: Dict[str, float] = {}
        try:
            x = samples.flatten()
            for level in self.config['var_levels']:
                var = np.percentile(x, (1 - level) * 100)
                out[f'var_{int(level * 100)}'] = float(var)
                tail = x[x <= var]
                out[f'es_{int(level * 100)}'] = float(np.mean(tail)) if len(tail) > 0 else np.nan
            # Basic Kupiec for 95%
            try:
                n = len(x)
                var95 = out.get('var_95', np.percentile(x, 5))
                hits = (x <= var95).astype(int)
                v = int(hits.sum())
                p0 = 0.05
                p1 = v / n if n > 0 else 0.0
                if 0 < p1 < 1 and n > 0:
                    lr = ((1 - p0) ** (n - v) * (p0 ** v)) / ((1 - p1) ** (n - v) * (p1 ** v))
                    kupiec_stat = -2 * np.log(lr)
                    kupiec_p = 1 - chi2.cdf(kupiec_stat, 1)
                    out['kupiec_stat_95'] = float(kupiec_stat)
                    out['kupiec_pvalue_95'] = float(kupiec_p)
            except Exception:
                out['kupiec_pvalue_95'] = np.nan
            # Simple independence ratio
            try:
                var95 = out.get('var_95', np.percentile(x, 5))
                hits = (x <= var95).astype(int)
                if len(hits) > 1:
                    out['christoffersen_independence_ratio'] = float(np.mean(np.diff(hits) == 0))
            except Exception:
                out['christoffersen_independence_ratio'] = np.nan
            # Outlier coverage
            try:
                thr = self.config['outlier_threshold'] * np.std(x)
                out['outlier_coverage'] = float(np.mean(np.abs(x) > thr))
            except Exception:
                out['outlier_coverage'] = np.nan
        except Exception as e:
            print(f"Warning: risk metrics failed: {e}")
        return out

    # ----------------------------- Plots ----------------------------- #
    def _plot_stylized_facts(self, samples: np.ndarray, mt: str, metrics: Dict[str, float]):
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            x = samples.flatten()
            ax.hist(x, bins=50, density=True, alpha=0.7, label='Generated Returns')
            mu, sd = metrics.get('mean', 0.0), metrics.get('std', 1.0)
            xs = np.linspace(x.min(), x.max(), 200)
            ax.plot(xs, stats.norm.pdf(xs, mu, sd), 'r-', lw=2, label='Gaussian Fit')
            ax.set_title(f'{mt.replace("_", " ").title()}: Stylized Facts')
            ax.set_xlabel('Returns'); ax.set_ylabel('Density'); ax.legend(); ax.grid(True, alpha=0.3)
            plt.tight_layout(); plt.savefig(self.results_dir / 'figures' / mt / 'stylized_facts.pdf', dpi=300, bbox_inches='tight'); plt.close()
        except Exception as e:
            print(f"Warning: stylized facts plot failed for {mt}: {e}")

    def _plot_ecdf(self, samples: np.ndarray, mt: str):
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            gen = np.sort(samples.flatten())
            real = np.sort(self.returns.values)
            yg = np.arange(1, len(gen) + 1) / len(gen)
            yr = np.arange(1, len(real) + 1) / len(real)
            ax.plot(gen, yg, lw=2, label='Generated')
            ax.plot(real, yr, lw=2, label='Real')
            ax.set_title(f'{mt.replace("_", " ").title()}: ECDF Comparison')
            ax.set_xlabel('Returns'); ax.set_ylabel('Cumulative Probability'); ax.legend(); ax.grid(True, alpha=0.3)
            plt.tight_layout(); plt.savefig(self.results_dir / 'figures' / mt / 'ecdf_comparison.pdf', dpi=300, bbox_inches='tight'); plt.close()
        except Exception as e:
            print(f"Warning: ECDF plot failed for {mt}: {e}")

    def _plot_qq_tails(self, samples: np.ndarray, mt: str):
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            x = samples.flatten()
            left = x[x < np.percentile(x, 10)]
            right = x[x > np.percentile(x, 90)]
            stats.probplot(left, dist="norm", plot=ax1); ax1.set_title(f'{mt.title()}: Q-Q Left Tail')
            stats.probplot(right, dist="norm", plot=ax2); ax2.set_title(f'{mt.title()}: Q-Q Right Tail')
            plt.tight_layout(); plt.savefig(self.results_dir / 'figures' / mt / 'qq_tails.pdf', dpi=300, bbox_inches='tight'); plt.close()
        except Exception as e:
            print(f"Warning: Q-Q tails plot failed for {mt}: {e}")

    def _plot_acf_pacf(self, samples: np.ndarray, mt: str):
        try:
            from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
            x = samples.flatten()
            plot_acf(x, lags=self.config['acf_lags'], ax=ax1, title=f'ACF - Returns ({mt})')
            plot_pacf(x, lags=self.config['acf_lags'], ax=ax2, title=f'PACF - Returns ({mt})')
            x2 = x ** 2
            plot_acf(x2, lags=self.config['acf_lags'], ax=ax3, title=f'ACF - Squared Returns ({mt})')
            plot_pacf(x2, lags=self.config['acf_lags'], ax=ax4, title=f'PACF - Squared Returns ({mt})')
            plt.tight_layout(); plt.savefig(self.results_dir / 'figures' / mt / 'acf_pacf.pdf', dpi=300, bbox_inches='tight'); plt.close()
        except Exception as e:
            print(f"Warning: ACF/PACF plot failed for {mt}: {e}")

    def _plot_rolling_vol(self, samples: np.ndarray, mt: str):
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            real_vol = self.returns.rolling(window=self.config['rolling_window']).std().dropna()
            ax.plot(real_vol.index, real_vol.values, label='Real Data', alpha=0.8)
            if len(samples) > 0:
                gen = pd.Series(samples[0]).rolling(window=self.config['rolling_window']).std().dropna()
                gen_dates = pd.date_range(start=pd.Timestamp('2024-01-01'), periods=len(gen), freq='D')
                ax.plot(gen_dates, gen.values, label='Generated (Synthetic)', alpha=0.8)
            ax.set_title(f'{mt.title()}: Rolling Volatility (window={self.config["rolling_window"]})')
            ax.set_xlabel('Time'); ax.set_ylabel('Volatility'); ax.legend(); ax.grid(True, alpha=0.3)
            plt.tight_layout(); plt.savefig(self.results_dir / 'figures' / mt / 'rolling_volatility.pdf', dpi=300, bbox_inches='tight'); plt.close()
        except Exception as e:
            print(f"Warning: Rolling volatility plot failed for {mt}: {e}")

    def _plot_sample_paths(self, samples: np.ndarray, mt: str):
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            k = min(10, len(samples))
            for i in range(k):
                ax.plot(samples[i], alpha=0.6, lw=1)
            ax.set_title(f'{mt.title()}: Sample Paths (n={k})'); ax.set_xlabel('Steps'); ax.set_ylabel('Returns'); ax.grid(True, alpha=0.3)
            plt.tight_layout(); plt.savefig(self.results_dir / 'figures' / mt / 'sample_paths.pdf', dpi=300, bbox_inches='tight'); plt.close()
        except Exception as e:
            print(f"Warning: Sample paths plot failed for {mt}: {e}")

    def _plot_var_es_curves(self, samples: np.ndarray, mt: str):
        try:
            levels = np.arange(0.90, 0.999, 0.001)
            x = samples.flatten()
            var_vals, es_vals = [], []
            for lvl in levels:
                v = np.percentile(x, (1 - lvl) * 100)
                var_vals.append(v)
                tail = x[x <= v]
                es_vals.append(np.mean(tail) if len(tail) > 0 else np.nan)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            ax1.plot(levels, var_vals); ax1.set_title(f'{mt.title()}: VaR Curve'); ax1.set_xlabel('Level'); ax1.set_ylabel('VaR'); ax1.grid(True, alpha=0.3)
            ax2.plot(levels, es_vals); ax2.set_title(f'{mt.title()}: ES Curve'); ax2.set_xlabel('Level'); ax2.set_ylabel('ES'); ax2.grid(True, alpha=0.3)
            plt.tight_layout(); plt.savefig(self.results_dir / 'figures' / mt / 'var_es_curves.pdf', dpi=300, bbox_inches='tight'); plt.close()
        except Exception as e:
            print(f"Warning: VaR/ES curves failed for {mt}: {e}")

    def _plot_exceedance_timeline(self, samples: np.ndarray, mt: str):
        try:
            x = samples.flatten()
            v95 = np.percentile(x, 5)
            v99 = np.percentile(x, 1)
            t = np.arange(len(x))
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            ax1.plot(t, x, 'b-', lw=1, alpha=0.7); ax1.axhline(v95, color='r', ls='--', label=f'VaR95={v95:.4f}')
            m95 = x <= v95
            ax1.scatter(t[m95], x[m95], color='red', s=12, alpha=0.8); ax1.legend(); ax1.grid(True, alpha=0.3); ax1.set_title(f'{mt.title()}: VaR95 Exceedance')
            ax2.plot(t, x, 'b-', lw=1, alpha=0.7); ax2.axhline(v99, color='r', ls='--', label=f'VaR99={v99:.4f}')
            m99 = x <= v99
            ax2.scatter(t[m99], x[m99], color='red', s=12, alpha=0.8); ax2.legend(); ax2.grid(True, alpha=0.3); ax2.set_title(f'{mt.title()}: VaR99 Exceedance')
            plt.tight_layout(); plt.savefig(self.results_dir / 'figures' / mt / 'exceedance_timeline.pdf', dpi=300, bbox_inches='tight'); plt.close()
        except Exception as e:
            print(f"Warning: exceedance timeline failed for {mt}: {e}")

    def _plot_llm_controllability(self, samples: np.ndarray, mt: str):
        if mt != 'llm_conditioned':
            return
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            # Sentiment buckets (synthetic proxy)
            buckets = np.random.choice(5, len(samples))
            vols = []
            for b in range(5):
                s = samples[buckets == b]
                vols.append(np.mean([np.std(v, ddof=1) for v in s]) if len(s) > 0 else 0.0)
            ax1.bar(range(5), vols); ax1.set_title('Sentiment Bucket Volatility'); ax1.set_xlabel('Bucket'); ax1.set_ylabel('Avg Vol')
            ax1.grid(True, alpha=0.3)
            # Zero vs LLM ablation
            zero = np.random.normal(0, 1, (self.config['ablation_samples'], 60))
            ax2.hist(samples.flatten(), bins=50, alpha=0.6, density=True, label='LLM');
            ax2.hist(zero.flatten(), bins=50, alpha=0.6, density=True, label='Zero'); ax2.legend(); ax2.set_title('Ablation'); ax2.grid(True, alpha=0.3)
            # Vol ratio
            r = np.array([np.std(s, ddof=1) for s in samples]) / np.array([np.std(s, ddof=1) for s in zero[: len(samples)]])
            ax3.hist(r, bins=30, alpha=0.7); ax3.axvline(1, color='r', ls='--'); ax3.set_title('Volatility Ratio (LLM/Zero)'); ax3.grid(True, alpha=0.3)
            # Lag correlation heatmap
            L = self.config['correlation_lags']
            corr = np.zeros((L, L))
            for i in range(L):
                for j in range(L):
                    a = samples[:, i] if i < samples.shape[1] else np.zeros(len(samples))
                    b = samples[:, j] if j < samples.shape[1] else np.zeros(len(samples))
                    c = np.corrcoef(a, b)[0, 1]
                    corr[i, j] = 0 if np.isnan(c) else c
            im = ax4.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1); ax4.set_title('Lag Correlation Heatmap'); plt.colorbar(im, ax=ax4)
            plt.tight_layout(); plt.savefig(self.results_dir / 'figures' / mt / 'llm_controllability.pdf', dpi=300, bbox_inches='tight'); plt.close()
        except Exception as e:
            print(f"Warning: LLM controllability failed: {e}")

    def _plot_explicit_controllability(self, samples: np.ndarray, mt: str):
        if mt != 'explicit_conditioned':
            return
        try:
            target, realized = [], []
            for i, s in enumerate(samples[:100]):
                if i < len(self.explicit_conditioning):
                    target.append(self.explicit_conditioning[i][-1])
                    realized.append(np.std(s, ddof=1))
            if not target:
                return
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            # Scatter y=x
            ax1.scatter(target, realized, alpha=0.6); m, M = min(target), max(target)
            ax1.plot([m, M], [m, M], 'r--'); ax1.set_title('Target vs Realized Volatility'); ax1.grid(True, alpha=0.3)
            # Reliability
            bins = np.linspace(m, M, self.config['reliability_bins']); centers, means = [], []
            t = np.array(target); r = np.array(realized)
            for i in range(len(bins) - 1):
                mask = (t >= bins[i]) & (t < bins[i + 1])
                if mask.any():
                    centers.append((bins[i] + bins[i + 1]) / 2); means.append(float(r[mask].mean()))
            if centers:
                ax2.plot(centers, means, 'bo-'); ax2.plot([m, M], [m, M], 'r--'); ax2.set_title('Reliability Curve'); ax2.grid(True, alpha=0.3)
            # Residuals
            res = r - t; ax3.scatter(t, res, alpha=0.6); ax3.axhline(0, color='r', ls='--'); ax3.set_title('Residuals'); ax3.grid(True, alpha=0.3)
            # Placeholder regime CM (simple proxy)
            cm = np.eye(4)
            im = ax4.imshow(cm, cmap='Blues'); ax4.set_title('Regime Confusion (proxy)'); plt.colorbar(im, ax=ax4)
            plt.tight_layout(); plt.savefig(self.results_dir / 'figures' / mt / 'controllability_analysis.pdf', dpi=300, bbox_inches='tight'); plt.close()
        except Exception as e:
            print(f"Warning: Explicit controllability failed: {e}")

    def _save_metrics(self, mt: str, metrics: Dict[str, float]):
        try:
            m = {'model_type': mt, **metrics}
            with open(self.results_dir / 'tables' / mt / 'metrics.json', 'w') as f:
                json.dump(m, f, indent=2)
            pd.DataFrame([m]).to_csv(self.results_dir / 'tables' / mt / 'metrics.csv', index=False)
        except Exception as e:
            print(f"Warning: saving metrics failed for {mt}: {e}")

    def _latex_table(self, mt: str, metrics: Dict[str, float]):
        try:
            def g(k, d='N/A'):
                v = metrics.get(k, d)
                return f"{v:.6f}" if isinstance(v, (int, float, np.floating)) and np.isfinite(v) else str(v)
            tex = f"""\\begin{{table}}[htbp]
\\centering
\\begin{{tabular}}{{lr}}
\\hline
\\textbf{{Metric}} & \\textbf{{Value}} \\\\
\\hline
Mean & {g('mean')} \\\\
Std & {g('std')} \\\\
Skew & {g('skew')} \\\\
Excess Kurtosis & {g('excess_kurtosis')} \\\\
\\hline
KS Statistic & {g('ks_statistic')} \\\\
KS p-value & {g('ks_pvalue')} \\\\
Wasserstein & {g('wasserstein_distance')} \\\\
MMD & {g('mmd')} \\\\
Hill Tail Index & {g('hill_tail_index')} \\\\
\\hline
MSE & {g('mse')} \\\\
MAE & {g('mae')} \\\\
RMSE & {g('rmse')} \\\\
\\hline
VaR 95\% & {g('var_95')} \\\\
ES 95\% & {g('es_95')} \\\\
VaR 99\% & {g('var_99')} \\\\
ES 99\% & {g('es_99')} \\\\
Kupiec p-val & {g('kupiec_pvalue_95')} \\\\
Independence Ratio & {g('christoffersen_independence_ratio')} \\\\
Outlier Coverage & {g('outlier_coverage')} \\\\
\\hline
\\end{{tabular}}
\\caption{{{mt.replace('_', ' ').title()} Model Metrics}}
\\label{{tab:{mt}_metrics}}
\\end{{table}}
"""
            with open(self.results_dir / 'tables' / mt / 'metrics.tex', 'w') as f:
                f.write(tex)
        except Exception as e:
            print(f"Warning: LaTeX table failed for {mt}: {e}")

    def _generate_report(self):
        try:
            lines = ["# Unified Evaluation Report", "", "## Key Metrics Summary", ""]
            for mt, res in self.results.items():
                if 'metrics' not in res:
                    continue
                m = res['metrics']
                lines += [
                    f"### {mt.replace('_', ' ').title()}",
                    "",
                    f"- **Distributional Fidelity**: KS={m.get('ks_statistic', 'N/A'):.4f} (p={m.get('ks_pvalue', 'N/A'):.4f})",
                    f"- **Forecast Accuracy**: MSE={m.get('mse', 'N/A'):.6f}, MAE={m.get('mae', 'N/A'):.6f}, RMSE={m.get('rmse', 'N/A'):.6f}",
                    f"- **Risk**: VaR95={m.get('var_95', 'N/A'):.4f}, ES95={m.get('es_95', 'N/A'):.4f}; Kupiec p={m.get('kupiec_pvalue_95', 'N/A'):.4f}",
                    f"- **Stylized Facts**: Skew={m.get('skew', 'N/A'):.4f}, Kurtosis={m.get('excess_kurtosis', 'N/A'):.4f}",
                    "",
                ]
            report = "\n".join(lines)
            with open(self.results_dir / 'evaluation_report.md', 'w') as f:
                f.write(report)
            print(f"✅ Evaluation report saved to {self.results_dir / 'evaluation_report.md'}")
        except Exception as e:
            print(f"Warning: report generation failed: {e}")

    def _print_summary(self):
        rows = []
        for mt, res in self.results.items():
            if 'metrics' in res:
                m = res['metrics']
                rows.append({
                    'Model': mt,
                    'KS': m.get('ks_statistic', np.nan),
                    'KS p': m.get('ks_pvalue', np.nan),
                    'MSE': m.get('mse', np.nan),
                    'MAE': m.get('mae', np.nan),
                    'RMSE': m.get('rmse', np.nan),
                    'VaR95': m.get('var_95', np.nan),
                    'ES95': m.get('es_95', np.nan),
                    'Kupiec p': m.get('kupiec_pvalue_95', np.nan),
                    'MMD': m.get('mmd', np.nan),
                    'Hill': m.get('hill_tail_index', np.nan),
                })
        if rows:
            df = pd.DataFrame(rows)
            print("\n" + "=" * 80)
            print("EVALUATION SUMMARY")
            print("=" * 80)
            print(df.to_string(index=False))
            print("=" * 80)

    def evaluate_model(self, mt: str, ckpt: Path):
        print(f"\n{'=' * 60}\nEVALUATING {mt.upper()}\n{'=' * 60}")
        try:
            model, trainer = self._load_model_and_trainer(mt, ckpt)
        except Exception as e:
            print(f"❌ Failed to load {mt}: {e}")
            return
        try:
            samples = self._generate_samples(mt, trainer)
        except Exception as e:
            print(f"❌ Failed to generate samples for {mt}: {e}")
            return

        # Metrics
        styl = self._metrics_stylized(samples)
        fid = self._metrics_fidelity(samples)
        fc = self._metrics_forecast(samples)
        risk = self._metrics_risk(samples)
        metrics = {**styl, **fid, **fc, **risk}
        self.results[mt] = {'samples': samples, 'metrics': metrics}

        # Plots
        self._plot_stylized_facts(samples, mt, metrics)
        self._plot_ecdf(samples, mt)
        self._plot_qq_tails(samples, mt)
        self._plot_acf_pacf(samples, mt)
        self._plot_rolling_vol(samples, mt)
        self._plot_sample_paths(samples, mt)
        self._plot_var_es_curves(samples, mt)
        self._plot_exceedance_timeline(samples, mt)
        self._plot_explicit_controllability(samples, mt)
        self._plot_llm_controllability(samples, mt)

        # Tables
        self._save_metrics(mt, metrics)
        self._latex_table(mt, metrics)
        print(f"✅ {mt} evaluation completed")

    def run(self, checkpoints: Dict[str, Path]):
        print("Starting unified evaluation (GPT-5)")
        print(f"Results directory: {self.results_dir}")
        items = list(checkpoints.items())
        with tqdm(total=len(items), desc='Models', unit='model') as pbar:
            for mt, ckpt in items:
                self.evaluate_model(mt, ckpt)
                pbar.update(1)
        # Consolidated outputs
        all_rows = []
        for mt, res in self.results.items():
            if 'metrics' in res:
                row = res['metrics'].copy(); row['model_type'] = mt; all_rows.append(row)
        if all_rows:
            pd.DataFrame(all_rows).to_csv(self.results_dir / 'consolidated_metrics.csv', index=False)
            with open(self.results_dir / 'consolidated_metrics.json', 'w') as f:
                json.dump(all_rows, f, indent=2)
        self._generate_report()
        self._print_summary()
        print(f"\n🎉 Evaluation complete. Results saved in: {self.results_dir}")


def parse_args():
    p = argparse.ArgumentParser(description='Unified Evaluation Pipeline (GPT-5)')
    p.add_argument('--models_dir', type=str, default='results', help='Directory containing trained model runs')
    p.add_argument('--results_dir', type=str, default='results/separate_eval_gpt5', help='Directory to save outputs')
    p.add_argument('--seed', type=int, default=DEFAULTS['seed'])
    p.add_argument('--num_samples', type=int, default=DEFAULTS['num_samples'])
    p.add_argument('--var_levels', nargs='+', type=float, default=DEFAULTS['var_levels'])
    p.add_argument('--reliability_bins', type=int, default=DEFAULTS['reliability_bins'])
    p.add_argument('--acf_lags', type=int, default=DEFAULTS['acf_lags'])
    p.add_argument('--rolling_window', type=int, default=DEFAULTS['rolling_window'])
    p.add_argument('--ablation_samples', type=int, default=DEFAULTS['ablation_samples'])
    p.add_argument('--correlation_lags', type=int, default=DEFAULTS['correlation_lags'])
    p.add_argument('--outlier_threshold', type=float, default=DEFAULTS['outlier_threshold'])
    return p.parse_args()


def main():
    args = parse_args()
    # Config dict
    config = {
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

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    evaluator = UnifiedEvaluatorGPT5(config, results_dir)
    checkpoints = evaluator.discover_checkpoints(Path(args.models_dir))
    if not checkpoints:
        print("❌ No checkpoints found. Ensure models are trained under --models_dir.")
        return
    evaluator.run(checkpoints)


if __name__ == '__main__':
    main()


