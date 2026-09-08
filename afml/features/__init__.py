"""Feature engineering on bars.

Everything here is strictly backward looking.  A bar is time-stamped at the
instant its last tick prints, so any statistic computed from bars up to and
including ``t`` is in the information set at ``t``; no ``shift(-k)``, no
centred window, no full-sample scaling ever appears in this module.  That
discipline is what makes the purged cross-validation of Chapter 7 meaningful
rather than decorative.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..fracdiff import frac_diff_ffd
from .microstructure import (
    amihud_lambda,
    bar_order_flow_imbalance,
    corwin_schultz,
    hasbrouck_lambda,
    kyle_lambda,
    roll_impact,
    roll_measure,
    vpin,
)
from .tick_stats import rolling_tick_microstructure, tick_partial_sums

__all__ = [
    "build_features",
    "rolling_tick_microstructure",
    "tick_partial_sums",
    "amihud_lambda",
    "bar_order_flow_imbalance",
    "corwin_schultz",
    "hasbrouck_lambda",
    "kyle_lambda",
    "roll_impact",
    "roll_measure",
    "vpin",
]

_DEFAULT_WINDOWS = (5, 20, 50)


def build_features(
    bars: pd.DataFrame,
    ticks: pd.DataFrame | None = None,
    d_ffd: float = 0.4,
    windows: tuple[int, ...] = _DEFAULT_WINDOWS,
    micro_windows: tuple[int, ...] = (20, 100),
    ffd_thres: float = 1e-4,
) -> pd.DataFrame:
    """Assemble the model's feature matrix from a bar frame.

    Parameters
    ----------
    bars : DataFrame
        Output of :mod:`afml.bars` -- needs ``open/high/low/close/volume/
        dollar/n_ticks/buy_volume/signed_dollar``.
    ticks : DataFrame, optional
        The raw tape.  When supplied, Roll / Kyle / Amihud / Hasbrouck / VPIN
        are estimated tick by tick and aggregated onto the bar clock (see
        :mod:`afml.features.tick_stats`), which is the only way those
        estimators mean anything.  When omitted, coarse bar-level fallbacks
        are used instead.
    d_ffd : float
        Fractional differencing order for the price level; pick it with
        :func:`afml.fracdiff.min_ffd` rather than by hand.
    windows : tuple of int
        Rolling lookbacks, in bars, used by every window-based feature.

    Returns
    -------
    DataFrame indexed like ``bars`` (rows with any NaN dropped at the end).
    """
    close, high, low = bars["close"], bars["high"], bars["low"]
    volume, dollar = bars["volume"], bars["dollar"]
    log_p = np.log(close)
    ret = log_p.diff()

    feats: dict[str, pd.Series] = {}

    # --- memory-preserving stationary price level (Ch. 5) -------------------
    ffd = frac_diff_ffd(log_p, d_ffd, thres=ffd_thres).iloc[:, 0]
    feats["ffd_price"] = ffd
    feats["ffd_price_z"] = (ffd - ffd.rolling(250).mean()) / ffd.rolling(250).std()

    # --- volatility ---------------------------------------------------------
    vol_ref = ret.rolling(50).std()
    feats["vol_50"] = vol_ref
    feats["vol_ratio"] = ret.rolling(10).std() / vol_ref

    # --- risk-adjusted momentum over several horizons -----------------------
    for w in windows:
        feats[f"mom_{w}"] = (log_p - log_p.shift(w)) / (vol_ref * np.sqrt(w))
    feats["autocorr_50"] = ret.rolling(50).corr(ret.shift(1))

    # --- order flow: the tape's directional signature -----------------------
    ofi = bar_order_flow_imbalance(bars["signed_dollar"], dollar)
    feats["ofi"] = ofi
    for w in windows:
        feats[f"ofi_{w}"] = ofi.rolling(w).mean()
    feats["buy_ratio_20"] = (bars["buy_volume"] / volume).rolling(20).mean()

    # --- activity / bar clock ----------------------------------------------
    elapsed = pd.Series(bars.index, index=bars.index).diff().dt.total_seconds()
    feats["log_duration"] = np.log(elapsed.clip(lower=1e-3))
    feats["duration_z"] = (
        feats["log_duration"] - feats["log_duration"].rolling(250).mean()
    ) / feats["log_duration"].rolling(250).std()
    feats["log_n_ticks"] = np.log(bars["n_ticks"])
    feats["volume_z"] = (volume - volume.rolling(250).mean()) / volume.rolling(250).std()

    # --- microstructure (Ch. 19) -------------------------------------------
    cs = corwin_schultz(high, low, 20)
    feats["cs_spread"] = cs["spread"]
    feats["cs_sigma"] = cs["sigma"]

    out = pd.DataFrame(feats, index=bars.index)

    if ticks is not None:
        micro = rolling_tick_microstructure(ticks, bars.index, micro_windows)
        micro = micro.drop(columns=[c for c in micro.columns if c.startswith("roll_cov_")])
        out = out.join(micro)
    else:  # coarse bar-level fallbacks -- kept for completeness, not preferred
        signs = np.sign(bars["signed_dollar"]).replace(0.0, np.nan).ffill().fillna(1.0)
        for w in micro_windows:
            out[f"roll_spread_{w}"] = roll_measure(close, w)
            out[f"kyle_lambda_{w}"] = kyle_lambda(close, volume, signs, w)
            out[f"amihud_lambda_{w}"] = amihud_lambda(close, dollar, w)
            out[f"hasbrouck_lambda_{w}"] = hasbrouck_lambda(close, dollar, signs, w)
            out[f"vpin_{w}"] = vpin(bars["buy_volume"], volume, w)
        out["roll_impact_20"] = roll_impact(close, dollar, 20)

    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    return out
