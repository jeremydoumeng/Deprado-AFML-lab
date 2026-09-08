"""Fractional differentiation  --  AFML Chapter 5.

Integer differencing (returns) makes a price series stationary but wipes out
all memory: the level of the price, which is what mean-reversion and
momentum signals are about, is gone.  Fractional differencing of order
``d`` in (0, 1) is the minimal transformation that buys stationarity while
retaining as much memory as possible.

Two variants, both from the chapter:

* ``frac_diff``      -- expanding window, weights truncated by cumulative
  weight loss.  The effective ``d`` drifts with the window, so different
  observations are not strictly comparable.
* ``frac_diff_ffd``  -- fixed-width window, weights truncated when they fall
  below ``thres``.  This is the one to use: driftless and stationary.

``min_ffd`` sweeps ``d`` and returns the smallest value that passes an
augmented Dickey-Fuller test, which is the chapter's central recipe.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "get_weights",
    "get_weights_ffd",
    "frac_diff",
    "frac_diff_ffd",
    "min_ffd",
]


# ----------------------------------------------------------------------------
# Snippets 5.1 / 5.2 -- binomial weights of the backshift expansion
# ----------------------------------------------------------------------------
def get_weights(d: float, size: int) -> np.ndarray:
    """Weights of ``(1 - B)^d`` truncated at ``size`` terms.

    ``w_k = -w_{k-1} * (d - k + 1) / k``, returned oldest-lag-last so that a
    dot product against a chronologically ordered window is a plain
    convolution.
    """
    w = [1.0]
    for k in range(1, size):
        w.append(-w[-1] / k * (d - k + 1))
    return np.array(w[::-1]).reshape(-1, 1)


def get_weights_ffd(d: float, thres: float = 1e-5, max_size: int = 10_000) -> np.ndarray:
    """Weights truncated once ``|w_k| < thres`` -- the fixed-width window."""
    w, k = [1.0], 1
    while k < max_size:
        w_ = -w[-1] / k * (d - k + 1)
        if abs(w_) < thres:
            break
        w.append(w_)
        k += 1
    return np.array(w[::-1]).reshape(-1, 1)


# ----------------------------------------------------------------------------
# Snippet 5.2 -- expanding-window fractional differentiation
# ----------------------------------------------------------------------------
def frac_diff(series: pd.DataFrame | pd.Series, d: float, thres: float = 0.01) -> pd.DataFrame:
    """Expanding-window frac-diff, dropping the burn-in where weight loss > ``thres``."""
    frame = series.to_frame() if isinstance(series, pd.Series) else series
    w = get_weights(d, frame.shape[0])
    w_cum = np.cumsum(abs(w))
    w_cum /= w_cum[-1]
    skip = int((w_cum > thres).sum())

    out = {}
    for name in frame.columns:
        s = frame[[name]].ffill().dropna()
        vals = s[name].to_numpy()
        idx = s.index
        res = np.full(len(idx), np.nan)
        for i in range(skip, len(idx)):
            res[i] = float(w[-(i + 1):, 0] @ vals[: i + 1])
        out[name] = pd.Series(res[skip:], index=idx[skip:])
    return pd.concat(out, axis=1)


# ----------------------------------------------------------------------------
# Snippet 5.3 -- fixed-width window fractional differentiation
# ----------------------------------------------------------------------------
def frac_diff_ffd(series: pd.DataFrame | pd.Series, d: float, thres: float = 1e-5) -> pd.DataFrame:
    """Fixed-width frac-diff.  Same weights at every date, so ``d`` is constant.

    Implemented as a convolution: with ``w`` ordered oldest-last, the FFD
    value at ``t`` is ``w . x[t-width : t+1]``, which is exactly the 'valid'
    part of ``convolve(x, w[::-1])``.
    """
    frame = series.to_frame() if isinstance(series, pd.Series) else series
    w = get_weights_ffd(d, thres)[:, 0]
    width = len(w) - 1

    out = {}
    for name in frame.columns:
        s = frame[name].ffill().dropna()
        vals = s.to_numpy(dtype=float)
        if len(vals) <= width:
            out[name] = pd.Series(dtype=float)
            continue
        conv = np.convolve(vals, w[::-1], mode="valid")
        out[name] = pd.Series(conv, index=s.index[width:])
    return pd.concat(out, axis=1)


# ----------------------------------------------------------------------------
# Snippet 5.4 -- the minimum d that passes ADF
# ----------------------------------------------------------------------------
def min_ffd(
    series: pd.Series,
    d_grid: np.ndarray | None = None,
    thres: float = 1e-5,
    p_target: float = 0.05,
    min_obs: int = 100,
) -> tuple[float, pd.DataFrame]:
    """Smallest ``d`` on ``d_grid`` whose FFD series rejects a unit root.

    Returns
    -------
    d_star : float
        The selected order (``nan`` if no grid point passes).
    table : DataFrame
        ADF statistic, p-value, 95% critical value, sample size, and the
        correlation between the FFD series and the original -- the memory
        retained.

    Two details the book's Snippet 5.4 glosses over.  The fixed-width window
    grows quickly as ``d`` falls (thousands of lags below ``d = 0.2``), so a
    grid point whose window exceeds the sample is dropped rather than
    silently evaluated on a handful of points.  And because each ``d`` then
    produces a series of a different length, the correlations are computed on
    the index they all share, otherwise the "memory retained" column compares
    different samples and need not even be monotone in ``d``.
    """
    from statsmodels.tsa.stattools import adfuller

    if d_grid is None:
        d_grid = np.linspace(0.0, 1.0, 21)

    base = series.ffill().dropna()
    computed: dict[float, pd.Series] = {}
    for d in d_grid:
        ffd = frac_diff_ffd(base, float(d), thres).iloc[:, 0].dropna()
        if len(ffd) >= min_obs:
            computed[float(d)] = ffd
    if not computed:
        return float("nan"), pd.DataFrame()

    common = None
    for ffd in computed.values():
        common = ffd.index if common is None else common.intersection(ffd.index)
    if common is None or len(common) < min_obs:  # fall back to each series' own index
        common = None

    rows = []
    for d, ffd in computed.items():
        try:  # statsmodels >= 0.15 warns unless the return shape is pinned
            res = adfuller(ffd.to_numpy(), maxlag=1, regression="c",
                           autolag=None, result_object=False)
        except TypeError:
            res = adfuller(ffd.to_numpy(), maxlag=1, regression="c", autolag=None)
        stat, pval, crit = res[0], res[1], res[4]
        idx = ffd.index if common is None else common
        corr = float(np.corrcoef(base.loc[idx].to_numpy(), ffd.loc[idx].to_numpy())[0, 1])
        rows.append(
            {"d": d, "adf_stat": stat, "p_value": pval, "crit_95": crit["5%"],
             "corr_with_original": corr, "n_obs": len(ffd)}
        )

    table = pd.DataFrame(rows).set_index("d").sort_index()
    passing = table.index[table["p_value"] < p_target]
    d_star = float(passing.min()) if len(passing) else float("nan")
    return d_star, table
