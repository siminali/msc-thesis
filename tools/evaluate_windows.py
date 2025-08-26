#!/usr/bin/env python3
"""
Evaluate fixed trained checkpoints across multiple date windows without retraining.

Outputs per-window metrics (CSV/JSON), plots, and LaTeX tables under
results/addons/period_slices/<window_name>/.

This script only creates new artifacts under tools/ and results/addons/period_slices/.
It does not modify any existing repository files.
"""

import argparse
import json
import os
import sys
import glob
import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Local utilities (reuse, do not modify)
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import risk as risk_utils  # type: ignore
from utils import stats as stats_utils  # type: ignore
from utils import plots as plot_utils  # type: ignore


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_windows(window_args: List[str]) -> List[Tuple[str, pd.Timestamp, pd.Timestamp]]:
    windows: List[Tuple[str, pd.Timestamp, pd.Timestamp]] = []
    for w in window_args:
        # Format: "Name:YYYY-MM-DD,YYYY-MM-DD"
        if ':' not in w or ',' not in w:
            raise ValueError(f"Invalid window spec: {w}. Expected 'Name:YYYY-MM-DD,YYYY-MM-DD'")
        name, dates = w.split(':', 1)
        start_str, end_str = dates.split(',', 1)
        start = pd.to_datetime(start_str.strip())
        end = pd.to_datetime(end_str.strip())
        if end < start:
            raise ValueError(f"End before start in window: {w}")
        windows.append((name.strip(), start, end))
    return windows


def parse_models(model_args: List[str]) -> List[Tuple[str, Optional[str]]]:
    models: List[Tuple[str, Optional[str]]] = []
    for m in model_args:
        if ':' in m:
            prefix, path = m.split(':', 1)
            models.append((prefix.strip(), path.strip()))
        else:
            models.append((m.strip(), None))
    return models


def read_real_series(real_csv: str) -> pd.DataFrame:
    df = pd.read_csv(real_csv)
    # Expect columns: Date, Close
    if 'Date' not in df.columns or 'Close' not in df.columns:
        raise ValueError("Expected columns 'Date' and 'Close' in real data CSV")
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    # Compute log returns (%) consistent with existing plots
    df['log_close'] = np.log(df['Close'])
    df['ret'] = 100.0 * df['log_close'].diff()
    df = df.dropna().reset_index(drop=True)
    return df[['Date', 'ret']]


def clopper_pearson_ci(k: int, n: int, conf: float = 0.95) -> Tuple[float, float]:
    # Two-sided Clopper–Pearson (exact) interval
    if n == 0:
        return (math.nan, math.nan)
    from scipy.stats import beta  # lazy import
    alpha = 1 - conf
    lower = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
    upper = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)
    return float(lower), float(upper)


def compute_summary_stats(x: np.ndarray) -> Dict[str, float]:
    from scipy.stats import skew, kurtosis
    return {
        'mean': float(np.mean(x)),
        'std': float(np.std(x, ddof=1)) if len(x) > 1 else float('nan'),
        'skew': float(skew(x)) if len(x) > 2 else float('nan'),
        'kurtosis': float(kurtosis(x)) if len(x) > 3 else float('nan'),
    }


def load_array_from_spec(path_spec: str) -> Optional[np.ndarray]:
    # Support direct file or glob
    paths = []
    if any(ch in path_spec for ch in ['*', '?', '[']):
        paths = sorted(glob.glob(path_spec))
    else:
        paths = [path_spec]
    for p in paths:
        if not os.path.exists(p):
            continue
        if p.endswith('.npy'):
            try:
                arr = np.load(p, allow_pickle=True)
                return arr
            except Exception:
                continue
    return None


def align_model_returns(
    real_df: pd.DataFrame,
    model_arr: np.ndarray,
    controls_csv: Optional[str]
) -> pd.Series:
    # Returns a pandas Series indexed by real_df index with aligned length
    # Strategy:
    # 1) If controls_csv provided with dates matching real_df, align by date order
    # 2) Else assume model_arr aligns 1:1 to real_df length
    if controls_csv and os.path.exists(controls_csv):
        try:
            cdf = pd.read_csv(controls_csv)
            if 'date' in cdf.columns:
                cdf['date'] = pd.to_datetime(cdf['date'])
                # Keep only dates present in real_df
                merged = real_df[['Date']].merge(cdf[['date']], left_on='Date', right_on='date', how='left')
                mask = merged['date'].notna().values
                if np.sum(mask) == len(model_arr):
                    out = pd.Series(index=real_df.index, dtype=float)
                    out.loc[np.where(mask)[0]] = model_arr.reshape(-1)
                    out = out.interpolate(limit_direction='both')  # fill any gaps conservatively
                    return out
        except Exception:
            pass
    # Fallback: assume same length
    if len(model_arr.reshape(-1)) != len(real_df):
        # Last resort: align to overlapping end portion
        min_len = min(len(model_arr.reshape(-1)), len(real_df))
        warnings.warn(
            f"Model returns length {len(model_arr)} differs from real length {len(real_df)}; truncating to {min_len} for alignment."
        )
        s = pd.Series(model_arr.reshape(-1)[-min_len:])
        s.index = real_df.index[-min_len:]
        out = pd.Series(index=real_df.index, dtype=float)
        out.loc[s.index] = s.values
        out = out.interpolate(limit_direction='both')
        return out
    return pd.Series(model_arr.reshape(-1), index=real_df.index)


def compute_regimes_from_explicit(
    real_df: pd.DataFrame,
    explicit_csv: Optional[str],
    vol_window: int,
    trend_window: int,
    trend_deadband: float,
    vol_quantiles: Tuple[float, float]
) -> Optional[pd.DataFrame]:
    if explicit_csv and os.path.exists(explicit_csv):
        try:
            cdf = pd.read_csv(explicit_csv)
            if 'date' not in cdf.columns:
                return None
            cdf['date'] = pd.to_datetime(cdf['date'])
            cdf = cdf.sort_values('date')
            return cdf
        except Exception:
            return None
    # Compute on full real data with warning (TRAINING quantiles not provided)
    warnings.warn("Explicit conditions CSV not provided; computing regime labels from full series.")
    df = real_df.copy()
    # Rolling volatility
    df['sigma'] = df['ret'].rolling(window=vol_window).std()
    q_low, q_high = df['sigma'].quantile(vol_quantiles[0]), df['sigma'].quantile(vol_quantiles[1])
    def vol_bucket(v: float) -> str:
        if np.isnan(v):
            return 'UNK'
        if v <= q_low:
            return 'LOW'
        if v >= q_high:
            return 'HIGH'
        return 'MID'
    df['mu20'] = df['ret'].rolling(window=trend_window).sum()
    def trend_bucket(m: float) -> str:
        if np.isnan(m):
            return 'FLAT'
        if m >= trend_deadband:
            return 'UP'
        if m <= -trend_deadband:
            return 'DOWN'
        return 'FLAT'
    df['vol_bucket'] = df['sigma'].map(vol_bucket)
    df['trend_bucket'] = df['mu20'].map(trend_bucket)
    df['regime_label'] = df['vol_bucket'] + '_' + df['trend_bucket']
    # Normalized sigma
    s = df['sigma']
    df['sigma_star'] = ((s - s.mean()) / (s.std() if s.std() > 0 else 1.0)).clip(-3, 3)
    return pd.DataFrame({'date': df['Date'], 'regime_label': df['regime_label'], 'sigma_star': df['sigma_star']})


def safe_save_csv_json(df: pd.DataFrame, csv_path: str, json_path: str) -> None:
    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient='records', indent=2)


def latex_table_from_df(df: pd.DataFrame, caption: str, label: str) -> str:
    # Minimal LaTeX tabular using pandas to_latex but without index
    return df.to_latex(index=False, caption=caption, label=label, escape=True, float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else str(x))


def evaluate_window(
    window_name: str,
    window_slice: pd.Series,
    real_slice: pd.Series,
    model_slices: Dict[str, pd.Series],
    out_dir: str,
    explicit_df: Optional[pd.DataFrame] = None,
    llm_controls_df: Optional[pd.DataFrame] = None,
    es_bootstrap_B: Optional[int] = None,
    es_ci: Optional[float] = None,
) -> Dict[str, Dict[str, float]]:
    ensure_dir(out_dir)
    ensure_dir(os.path.join(out_dir, 'figures'))
    ensure_dir(os.path.join(out_dir, 'tables'))

    # Metrics per model
    metrics_rows: List[Dict[str, float]] = []
    dist_rows: List[Dict[str, float]] = []
    summary_rows: List[Dict[str, float]] = []

    # Real series for comparison in distribution tests
    real_vals = real_slice.values

    for model_name, series in [('real', real_slice), *list(model_slices.items())]:
        vals = series.dropna().values
        if len(vals) == 0:
            warnings.warn(f"No data for model {model_name} in window {window_name}; skipping.")
            continue

        # Risk metrics
        var95, es95 = risk_utils.var_es(vals, alpha=0.05)
        var99, es99 = risk_utils.var_es(vals, alpha=0.01)

        # Backtests
        n = len(vals)
        hits95 = (vals < var95).astype(int)
        hits99 = (vals < var99).astype(int)
        n95 = int(hits95.sum()); n99 = int(hits99.sum())
        hr95 = n95 / n; hr99 = n99 / n
        ci95 = clopper_pearson_ci(n95, n, conf=0.95)
        ci99 = clopper_pearson_ci(n99, n, conf=0.95)
        kup95 = risk_utils.kupiec_pvalue(n, n95, 0.05)
        kup99 = risk_utils.kupiec_pvalue(n, n99, 0.01)
        chr95 = risk_utils.christoffersen_independence_pvalue(hits95)
        chr99 = risk_utils.christoffersen_independence_pvalue(hits99)

        # Distribution tests vs real
        ks_ad = stats_utils.ks_ad(real_vals, vals)

        # Summary
        summ = compute_summary_stats(vals)

        metrics_rows.append({
            'model': model_name,
            'n': n,
            'var_95': var95, 'es_95': es95,
            'hit_rate_95': hr95, 'hit_ci95_low': ci95[0], 'hit_ci95_high': ci95[1],
            'kupiec_p_95': kup95, 'christoffersen_p_95': chr95,
            'var_99': var99, 'es_99': es99,
            'hit_rate_99': hr99, 'hit_ci99_low': ci99[0], 'hit_ci99_high': ci99[1],
            'kupiec_p_99': kup99, 'christoffersen_p_99': chr99,
        })

        dist_rows.append({
            'model': model_name,
            'ks_stat': ks_ad.get('ks_statistic', np.nan),
            'ks_pvalue': ks_ad.get('ks_pvalue', np.nan),
            'ad_stat': ks_ad.get('ad_statistic', np.nan),
            'ad_pvalue': ks_ad.get('ad_pvalue', np.nan) if ks_ad.get('ad_pvalue', None) is not None else np.nan,
        })

        summary_row = {'model': model_name}
        summary_row.update(summ)
        summary_rows.append(summary_row)

        # Plots: distribution comparison and sequence with VaR overlays
        try:
            plot_utils.hist_line_logy([real_vals, vals], labels=['Real', model_name], title=f"{window_name}: Distribution (log-y)")
            plt.savefig(os.path.join(out_dir, 'figures', f'{model_name}_distribution_comparison.pdf'))
            plt.savefig(os.path.join(out_dir, 'figures', f'{model_name}_distribution_comparison.png'), dpi=200)
            plt.close()
        except Exception:
            warnings.warn(f"Failed to save distribution plot for {model_name} in {window_name}")

        try:
            plt.figure(figsize=(12, 4))
            plt.plot(window_slice.index, series.values, label=model_name, linewidth=1.2)
            plt.axhline(var95, color='orange', linestyle='--', label='VaR 95%')
            plt.axhline(var99, color='red', linestyle='--', label='VaR 99%')
            plt.title(f"{window_name}: Sequence with VaR overlays — {model_name}")
            plt.xlabel('Index')
            plt.ylabel('Return (%)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, 'figures', f'{model_name}_sequence_var_overlays.pdf'))
            plt.savefig(os.path.join(out_dir, 'figures', f'{model_name}_sequence_var_overlays.png'), dpi=200)
            plt.close()
        except Exception:
            warnings.warn(f"Failed to save sequence plot for {model_name} in {window_name}")

    # Regime diagnostics where applicable
    if explicit_df is not None and 'regime_label' in explicit_df.columns:
        try:
            # Align to window
            edf = explicit_df.copy()
            edf['date'] = pd.to_datetime(edf['date'])
            mask = (edf['date'] >= window_slice.index.min()) & (edf['date'] <= window_slice.index.max())
            edf_w = edf.loc[mask].reset_index(drop=True)
            # For models with series, compute conditional coverage for VaR95
            for model_name, series in model_slices.items():
                vals = series.dropna().values
                if len(vals) != len(edf_w):
                    continue
                var95, _ = risk_utils.var_es(vals, alpha=0.05)
                hits = (vals < var95).astype(int)
                df_cond = pd.DataFrame({'regime': edf_w['regime_label'].astype(str), 'hit': hits})
                cond = df_cond.groupby('regime')['hit'].agg(['mean', 'count', 'sum']).reset_index()
                plt.figure(figsize=(10, 4))
                plt.bar(cond['regime'], cond['mean'], color='steelblue')
                plt.axhline(0.05, color='red', linestyle='--', label='Target 5%')
                plt.title(f"{window_name}: Conditional VaR95 coverage — {model_name}")
                plt.xticks(rotation=30, ha='right')
                plt.legend(); plt.tight_layout()
                plt.savefig(os.path.join(out_dir, 'figures', f'{model_name}_conditional_coverage_var95.pdf'))
                plt.savefig(os.path.join(out_dir, 'figures', f'{model_name}_conditional_coverage_var95.png'), dpi=200)
                plt.close()
        except Exception:
            warnings.warn(f"Failed regime-conditional plots for {window_name}")

    # Save tables
    metrics_df = pd.DataFrame(metrics_rows)
    dist_df = pd.DataFrame(dist_rows)
    summary_df = pd.DataFrame(summary_rows)

    safe_save_csv_json(metrics_df, os.path.join(out_dir, 'metrics.csv'), os.path.join(out_dir, 'metrics.json'))
    metrics_tex = latex_table_from_df(metrics_df, caption=f"Risk metrics — {window_name}", label=f"tab:risk_{window_name}")
    with open(os.path.join(out_dir, 'tables', 'risk.tex'), 'w') as f:
        f.write(metrics_tex)

    dist_df.to_csv(os.path.join(out_dir, 'dist.csv'), index=False)
    with open(os.path.join(out_dir, 'tables', 'dist.tex'), 'w') as f:
        f.write(latex_table_from_df(dist_df, caption=f"Distribution tests — {window_name}", label=f"tab:dist_{window_name}"))

    summary_df.to_csv(os.path.join(out_dir, 'summary.csv'), index=False)
    with open(os.path.join(out_dir, 'tables', 'summary.tex'), 'w') as f:
        f.write(latex_table_from_df(summary_df, caption=f"Summary statistics — {window_name}", label=f"tab:summary_{window_name}"))

    return {row['model']: row for row in metrics_rows}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate fixed checkpoints across multiple windows.")
    parser.add_argument('--windows', nargs='+', required=True, help="One or more window specs: 'Name:YYYY-MM-DD,YYYY-MM-DD'")
    parser.add_argument('--real', required=True, help='Path to real data CSV (Date, Close)')
    parser.add_argument('--models', nargs='+', required=True, help="Models to evaluate: 'real' or '<type>:<path_or_glob>'")
    parser.add_argument('--outdir', default='results/addons/period_slices', help='Base output directory')

    # Explicit conditioning
    parser.add_argument('--explicit-cond-csv', default=None, help='Path to explicit conditions CSV (date, regime_label, sigma_star)')
    parser.add_argument('--explicit-vol-window', type=int, default=21)
    parser.add_argument('--explicit-trend-window', type=int, default=20)
    parser.add_argument('--explicit-trend-deadband', type=float, default=0.001)
    parser.add_argument('--explicit-vol-quantiles', default='0.33,0.66')

    # LLM controls
    parser.add_argument('--llm-emb', default=None, help='Path to llm_embeddings.npy (optional)')
    parser.add_argument('--llm-probe', default=None, help='Path to llm_probe.json (optional)')
    parser.add_argument('--llm-controls-csv', default=None, help='Path to llm_controls.csv (optional)')

    # ES CI optional
    parser.add_argument('--es-bootstrap-B', type=int, default=None)
    parser.add_argument('--es-ci', type=float, default=0.95)

    args = parser.parse_args()

    windows = parse_windows(args.windows)
    models = parse_models(args.models)
    real_df = read_real_series(args.real)
    real_df = real_df.set_index('Date')

    # Prepare conditioning sources
    vol_q_parts = [float(x) for x in args.explicit_vol_quantiles.split(',')]
    vol_quantiles = (vol_q_parts[0], vol_q_parts[1]) if len(vol_q_parts) == 2 else (0.33, 0.66)
    explicit_df = compute_regimes_from_explicit(
        real_df.reset_index().rename(columns={'index': 'Date'}),
        args.explicit_cond_csv,
        args.explicit_vol_window,
        args.explicit_trend_window,
        args.explicit_trend_deadband,
        vol_quantiles,
    )

    llm_controls_df = None
    if args.llm_controls_csv and os.path.exists(args.llm_controls_csv):
        try:
            llm_controls_df = pd.read_csv(args.llm_controls_csv)
            if 'date' in llm_controls_df.columns:
                llm_controls_df['date'] = pd.to_datetime(llm_controls_df['date'])
        except Exception:
            warnings.warn('Failed to read LLM controls CSV; skipping controllability diagnostics for LLM.')

    base_outdir = args.outdir
    ensure_dir(base_outdir)

    # Prepare model series aligned to real index (full series)
    model_full_series: Dict[str, pd.Series] = {}
    for mtype, spec in models:
        if mtype == 'real':
            continue
        if spec is None:
            warnings.warn(f"Model '{mtype}' missing path; skipping.")
            continue
        arr = load_array_from_spec(spec)
        if arr is None:
            warnings.warn(f"Could not load model array from '{spec}'; skipping {mtype}.")
            continue
        controls_csv = None
        if mtype == 'explicit_conditioned':
            controls_csv = args.explicit_cond_csv
        elif mtype == 'llm_conditioned':
            controls_csv = args.llm_controls_csv
        ser = align_model_returns(real_df.reset_index()[['Date']], arr.reshape(-1), controls_csv)
        ser.index = real_df.index  # ensure same index type
        model_full_series[mtype] = ser

    # Evaluate per window
    combined_summary_rows: List[Dict[str, object]] = []
    for name, start, end in windows:
        win_dir = os.path.join(base_outdir, name)
        ensure_dir(win_dir)

        mask = (real_df.index >= start) & (real_df.index <= end)
        if mask.sum() == 0:
            warnings.warn(f"Window {name} {start.date()}–{end.date()} has no overlap with data; skipping.")
            continue
        real_slice = real_df.loc[mask, 'ret']
        model_slices = {mn: ms.loc[mask] for mn, ms in model_full_series.items()}

        metrics_by_model = evaluate_window(
            name,
            real_slice,
            real_slice,
            model_slices,
            win_dir,
            explicit_df=explicit_df,
            llm_controls_df=llm_controls_df,
            es_bootstrap_B=args.es_bootstrap_B,
            es_ci=args.es_ci,
        )

        # Collect combined summary across windows
        for model_name, m in metrics_by_model.items():
            row = {'window': name, 'model': model_name,
                   'var_95': m['var_95'], 'es_95': m['es_95'], 'hit_rate_95': m['hit_rate_95'],
                   'var_99': m['var_99'], 'es_99': m['es_99'], 'hit_rate_99': m['hit_rate_99']}
            combined_summary_rows.append(row)

    # Save combined summary
    if combined_summary_rows:
        comb_df = pd.DataFrame(combined_summary_rows)
        comb_df.to_csv(os.path.join(base_outdir, 'summary.csv'), index=False)
        with open(os.path.join(base_outdir, 'summary.tex'), 'w') as f:
            f.write(latex_table_from_df(comb_df, caption='Cross-window summary', label='tab:period_summary'))


if __name__ == '__main__':
    main()


