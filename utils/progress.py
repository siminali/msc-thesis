#!/usr/bin/env python3
"""
Lightweight progress utilities wrapping tqdm.auto with sensible defaults.

Features:
- Safe no-op when tqdm is unavailable or bars are disabled.
- Rank-aware factories (show only on rank 0).
- Nested bars via a context manager.
- Dynamic postfix builder from metrics dict.
- Logger shim using tqdm.write to avoid breaking bars.
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from typing import Any, Dict, Optional


def _isatty() -> bool:
    try:
        return sys.stdout.isatty()
    except Exception:
        return False


class _NullBar:
    def __init__(self, total: Optional[int] = None):
        self.total = total
        self.n = 0

    def update(self, n: int = 1) -> None:
        self.n += n

    def set_postfix(self, ordered_dict: Optional[Dict[str, Any]] = None, refresh: bool = True) -> None:
        return

    def write(self, s: str) -> None:
        print(s)

    def close(self) -> None:
        return

    def __enter__(self) -> "_NullBar":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return


def build_postfix(metrics: Dict[str, Any], precision: int = 3) -> Dict[str, str]:
    pf: Dict[str, str] = {}
    for k, v in metrics.items():
        try:
            if isinstance(v, float):
                pf[k] = f"{v:.{precision}f}"
            else:
                pf[k] = str(v)
        except Exception:
            pf[k] = str(v)
    return pf


def logger_write(msg: str) -> None:
    try:
        from tqdm.auto import tqdm
        tqdm.write(msg)
    except Exception:
        print(msg)


def create_progress(total: Optional[int], desc: str = "", enabled: bool = True, leave: Optional[bool] = None,
                    miniters: Optional[int] = None, position: int = 0, rank: int = 0, update_interval: int = 10,
                    smoothing: float = 0.1, ascii_fallback: bool = False) -> Any:
    """Create a tqdm progress bar or a null bar when disabled/unavailable.

    - enabled: master switch; also auto-disables when not a TTY unless user explicitly enables.
    - leave: if None, defaults to True when TTY, False otherwise.
    - miniters: if None, derived from total and update_interval.
    - rank: bars shown only when rank==0.
    """
    if not enabled or rank != 0:
        return _NullBar(total)
    if not _isatty() and os.getenv("FORCE_TQDM", "0") != "1":
        # Allow forcing in CI via env var
        return _NullBar(total)
    try:
        from tqdm.auto import tqdm
    except Exception:
        return _NullBar(total)
    try:
        dyn_cols = True
        if leave is None:
            leave = _isatty()
        if miniters is None:
            if total and total > 0:
                miniters = max(1, total // max(1, update_interval))
            else:
                miniters = 1
        bar = tqdm(total=total, desc=desc, position=position, leave=leave, dynamic_ncols=dyn_cols,
                   smoothing=smoothing, ascii=ascii_fallback, miniters=miniters)
        return bar
    except Exception:
        return _NullBar(total)


@contextmanager
def nested_bars(outer_total: int, inner_total: int, outer_desc: str = "", inner_desc: str = "",
                enabled: bool = True, rank: int = 0, update_interval: int = 10, leave: Optional[bool] = None):
    outer = create_progress(outer_total, desc=outer_desc, enabled=enabled, position=0, rank=rank,
                            update_interval=update_interval, leave=leave)
    inner = create_progress(inner_total, desc=inner_desc, enabled=enabled, position=1, rank=rank,
                            update_interval=update_interval, leave=leave)
    try:
        yield outer, inner
    finally:
        try:
            inner.close()
        finally:
            outer.close()


