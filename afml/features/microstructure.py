"""Microstructural features  --  AFML Chapter 19.

Three families, all computed on a rolling window of *bars* (each bar already
carries the tick-level aggregates it needs, see :mod:`afml.bars`):

First generation -- price sequences
    ``roll_measure`` (Roll 1984) inverts the negative autocovariance that the
    bid-ask bounce injects into observed returns to recover the effective
    spread; ``roll_impact`` scales it by dollar volume.
    ``corwin_schultz`` (2012) estimates the spread from high-low ranges,
    exploiting that the range is variance-driven while the high-low *ratio*
    carries the spread.

Second generation -- strategic trade models
    ``kyle_lambda`` (1985), ``amihud_lambda`` (2002) and
    ``hasbrouck_lambda`` (2009) are three regressions of price change on a
    signed-flow regressor; the slope is the illiquidity / adverse-selection
    cost of executing.

Third generation -- order-flow toxicity
    ``vpin`` (Easley, Lopez de Prado, O'Hara 2011): volume-synchronised
    probability of informed trading, the fraction of volume that is
    one-sided within a window of bars.

Caveats worth knowing, and asserted in the tests: Roll's estimator assumes
serially *uncorrelated* order flow, and under sign autocorrelation ``rho`` it
converges to ``spread * (1 - rho)`` rather than ``spread``; it also returns
zero whenever the sample autocovariance happens to be positive.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "roll_measure",
    "roll_impact",
    "corwin_schultz",
    "kyle_lambda",
    "amihud_lambda",
    "hasbrouck_lambda",
    "vpin",
    "bar_order_flow_imbalance",
]


def _rolling_ols_no_intercept(y: pd.Series, x: pd.Series, window: int) -> pd.Series:
    """Rolling slope of ``y ~ x`` through the origin: ``sum(xy) / sum(x^2)``."""
    num = (x * y).rolling(window).sum()
    den = (x * x).rolling(window).sum()
    return num / den.replace(0.0, np.nan)


# ----------------------------------------------------------------------------
# First generation
# ----------------------------------------------------------------------------
def roll_measure(close: pd.Series, window: int = 20) -> pd.Series:
    """Effective spread, ``2 * sqrt(-cov(dp_t, dp_{t-1}))``  (Snippet 19.1).

    Computed on log prices, so the result is a relative spread. Negative
    autocovariances are floored at zero, as in the original paper.
    """
    dp = np.log(close).diff()
    cov = dp.rolling(window).cov(dp.shift(1))
    return 2.0 * np.sqrt(np.maximum(-cov, 0.0))


def roll_impact(close: pd.Series, dollar_volume: pd.Series, window: int = 20) -> pd.Series:
    """Roll measure per unit of dollar volume  (Snippet 19.2)."""
    return roll_measure(close, window) / dollar_volume.replace(0.0, np.nan)


def corwin_schultz(high: pd.Series, low: pd.Series, window: int = 20) -> pd.DataFrame:
    """Corwin-Schultz high-low spread and its implied volatility (Snippets 19.3-19.4).

    Returns a frame with ``spread`` (relative, floored at zero as the authors
    prescribe) and ``sigma`` (the Becker-Parkinson volatility that falls out
    of the same two moments).
    """
    h, l = high.astype(float), low.astype(float)
    hl = np.log(h / l) ** 2                                   # single-period
    beta = hl.rolling(2).sum().rolling(window).mean()          # E[beta]
    gamma = (
        np.log(h.rolling(2).max() / l.rolling(2).min()) ** 2   # two-period range
    )

    den = 3.0 - 2.0 * np.sqrt(2.0)
    alpha = (np.sqrt(2.0) - 1.0) * np.sqrt(beta) / den - np.sqrt(gamma / den)
    alpha = alpha.clip(lower=0.0)                              # negative -> 0
    spread = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))

    k2 = np.sqrt(8.0 / np.pi)
    sigma = (
        (np.sqrt(2.0) - 1.0) * np.sqrt(beta) / (3.0 - 2.0 * np.sqrt(2.0)) / k2
        + np.sqrt(gamma / (k2 ** 2 * den))
    )
    sigma = sigma.clip(lower=0.0)
    return pd.DataFrame({"spread": spread, "sigma": sigma})


# ----------------------------------------------------------------------------
# Second generation -- price impact
# ----------------------------------------------------------------------------
def kyle_lambda(close: pd.Series, volume: pd.Series, signs: pd.Series, window: int = 20) -> pd.Series:
    """Kyle's lambda: ``dp_t = lam * (b_t * V_t) + eps``  (Snippet 19.5)."""
    dp = close.diff()
    net = signs * volume
    return _rolling_ols_no_intercept(dp, net, window)


def amihud_lambda(close: pd.Series, dollar_volume: pd.Series, window: int = 20) -> pd.Series:
    """Amihud's lambda: ``|r_t| = lam * (p_t V_t) + eps``  (Snippet 19.6)."""
    r = np.log(close).diff().abs()
    return _rolling_ols_no_intercept(r, dollar_volume, window)


def hasbrouck_lambda(
    close: pd.Series, dollar_volume: pd.Series, signs: pd.Series, window: int = 20
) -> pd.Series:
    """Hasbrouck's lambda: ``r_t = lam * b_t * sqrt(p_t V_t) + eps``  (Snippet 19.7)."""
    r = np.log(close).diff()
    x = signs * np.sqrt(dollar_volume)
    return _rolling_ols_no_intercept(r, x, window)


# ----------------------------------------------------------------------------
# Third generation -- order-flow toxicity
# ----------------------------------------------------------------------------
def vpin(buy_volume: pd.Series, volume: pd.Series, window: int = 20) -> pd.Series:
    """Volume-synchronised probability of informed trading  (Snippet 19.8).

    ``VPIN = sum_i |2 V^B_i - V_i| / sum_i V_i`` over a rolling window of
    volume bars.  In [0, 1]: zero when every bar is perfectly balanced, one
    when all volume is on a single side.
    """
    num = (2.0 * buy_volume - volume).abs().rolling(window).sum()
    den = volume.rolling(window).sum()
    return num / den.replace(0.0, np.nan)


def bar_order_flow_imbalance(signed_dollar: pd.Series, dollar: pd.Series) -> pd.Series:
    """Normalised signed dollar flow of a bar, in [-1, 1].

    Not a chapter-19 estimator, but the direct empirical counterpart of the
    ``theta_T`` that drives imbalance bars, and the natural place for the
    tape's directional information to show up.
    """
    return signed_dollar / dollar.replace(0.0, np.nan)
