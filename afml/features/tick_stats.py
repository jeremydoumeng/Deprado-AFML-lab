"""Tick-level microstructure estimators, aggregated onto a bar clock.

Roll's spread, Kyle's, Amihud's and Hasbrouck's lambdas are defined on
*trades*.  Estimating them on bar closes, as a naive implementation would,
destroys them: the bid-ask bounce they are built to measure is diluted away
by aggregation, and at bar frequency the return autocovariance is dominated
by drift, so Roll's estimator floors at zero almost everywhere.

The right construction, and the one used here, keeps the estimators on the
tape and only the *aggregation* on the bar clock.  Each estimator is a ratio
of sums over ticks, so per-bar partial sums are sufficient statistics: sum
them within each bar once, then roll those bar-level sums over a window of
``w`` bars to obtain the estimator over the last ``w`` bars of tape.  Cost is
one pass over the ticks regardless of how many windows are requested.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..bars import tick_rule

__all__ = ["tick_partial_sums", "rolling_tick_microstructure"]


def _bar_end_positions(tick_index: pd.DatetimeIndex, bar_index: pd.DatetimeIndex) -> np.ndarray:
    """Exclusive end position, in the tick array, of each bar."""
    return np.searchsorted(tick_index.values, bar_index.values, side="right")


def tick_partial_sums(ticks: pd.DataFrame, bar_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Per-bar sums of the tick-level quantities the estimators need."""
    price = ticks["price"].to_numpy(dtype=float)
    vol = ticks["volume"].to_numpy(dtype=float)
    dol = ticks["dollar"].to_numpy(dtype=float)
    b = tick_rule(price).astype(float)

    dp = np.diff(np.log(price), prepend=np.log(price[0]))  # dp[0] := 0
    dp_lag = np.roll(dp, 1)
    dp_lag[0] = 0.0

    signed_vol = b * vol
    signed_sqrt_dollar = b * np.sqrt(dol)

    cols = {
        "n": np.ones_like(dp),
        "s_dp": dp,
        "s_dp_lag": dp_lag,
        "s_dp_dplag": dp * dp_lag,
        "s_dp_sv": dp * signed_vol,          # Kyle numerator
        "s_sv2": signed_vol ** 2,            # Kyle denominator
        "s_absdp_dol": np.abs(dp) * dol,     # Amihud numerator
        "s_dol2": dol ** 2,                  # Amihud denominator
        "s_dp_ssd": dp * signed_sqrt_dollar,  # Hasbrouck numerator
        "s_ssd2": signed_sqrt_dollar ** 2,   # Hasbrouck denominator
        "s_buy_vol": np.where(b > 0, vol, 0.0),
        "s_vol": vol,
    }

    ends = _bar_end_positions(ticks.index, bar_index)
    starts = np.concatenate(([0], ends[:-1]))
    out = {}
    for name, arr in cols.items():
        cum = np.concatenate(([0.0], np.cumsum(arr)))
        out[name] = cum[ends] - cum[starts]
    return pd.DataFrame(out, index=bar_index)


def rolling_tick_microstructure(
    ticks: pd.DataFrame,
    bar_index: pd.DatetimeIndex,
    windows: tuple[int, ...] = (20, 100),
) -> pd.DataFrame:
    """Roll / Kyle / Amihud / Hasbrouck / VPIN over rolling windows of bars.

    Every column is computed from ticks but indexed and updated on the bar
    clock, so it is available at the bar close and nowhere earlier.
    """
    ps = tick_partial_sums(ticks, bar_index)
    feats: dict[str, pd.Series] = {}

    for w in windows:
        r = ps.rolling(w).sum()
        n = r["n"]

        # Roll (1984): effective spread = 2 * sqrt(-cov(dp_t, dp_{t-1}))
        cov = (r["s_dp_dplag"] - r["s_dp"] * r["s_dp_lag"] / n) / (n - 1.0)
        feats[f"roll_spread_{w}"] = 2.0 * np.sqrt(np.maximum(-cov, 0.0))
        feats[f"roll_cov_{w}"] = cov

        # Kyle (1985): dp = lam * (b * V)
        feats[f"kyle_lambda_{w}"] = r["s_dp_sv"] / r["s_sv2"].replace(0.0, np.nan)
        # Amihud (2002): |r| = lam * (p * V)
        feats[f"amihud_lambda_{w}"] = r["s_absdp_dol"] / r["s_dol2"].replace(0.0, np.nan)
        # Hasbrouck (2009): r = lam * b * sqrt(p * V)
        feats[f"hasbrouck_lambda_{w}"] = r["s_dp_ssd"] / r["s_ssd2"].replace(0.0, np.nan)
        # VPIN: share of volume that is one-sided
        feats[f"vpin_{w}"] = (
            (2.0 * r["s_buy_vol"] - r["s_vol"]).abs() / r["s_vol"].replace(0.0, np.nan)
        )

    return pd.DataFrame(feats, index=bar_index)
