"""Percentile-bootstrap confidence intervals and percentile bands."""
from __future__ import annotations

from typing import Callable

import numpy as np


def bootstrap_ci(values, stat: Callable = np.median, n_boot: int = 4000, ci: float = 95,
                 seed: int = 0) -> tuple[float, float, float]:
    """(stat, lo, hi): percentile bootstrap over the finite values; NaNs if none."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    boot = stat(v[idx], axis=1)
    lo, hi = np.percentile(boot, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return float(stat(v)), float(lo), float(hi)


def median_ci(values, n_boot: int = 4000, ci: float = 95, seed: int = 0) -> tuple[float, float, float]:
    """Median and its percentile-bootstrap CI (the paper's tables use these defaults)."""
    return bootstrap_ci(values, np.median, n_boot, ci, seed)


def pctl_band(values_2d, lo: float = 2.5, hi: float = 97.5):
    """Per-column (median, lo-percentile, hi-percentile) of a (rows, cols) array, ignoring NaNs."""
    M = np.asarray(values_2d, dtype=float)
    return (np.nanmedian(M, axis=0), np.nanpercentile(M, lo, axis=0),
            np.nanpercentile(M, hi, axis=0))
