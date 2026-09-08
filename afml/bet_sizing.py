"""Bet sizing  --  AFML Chapter 10.

Meta-labeling gives a probability that a bet is worth taking; this turns that
probability into a position size.  Three steps, all from the chapter:

1. **Probability to size.**  Under the null "all classes equally likely",
   ``z = (p - 1/k) / sqrt(p (1 - p))`` is a t-value, and ``m = 2 Phi(z) - 1``
   maps it to ``[-1, 1]``.  The point is that size grows with statistical
   confidence, not linearly with the raw probability, so a 0.55 probability
   is a small bet rather than a slightly-smaller-than-full one.
2. **Averaging active bets.**  Bets overlap in time.  Holding the sum of all
   open bets would lever up mechanically whenever signals cluster, so the
   position at any instant is the *average* of the bets still active then.
3. **Discretisation.**  Rounding the size to a grid of ``step_size``
   suppresses the constant small rebalancing that a continuous signal
   produces and whose only certain effect is transaction cost.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from .multiprocess import mp_pandas_obj

__all__ = ["prob_to_size", "avg_active_signals", "discrete_signal", "get_signal"]


# ----------------------------------------------------------------------------
# Snippet 10.1 -- from probability to size
# ----------------------------------------------------------------------------
def prob_to_size(prob: pd.Series, pred: pd.Series | None = None, num_classes: int = 2) -> pd.Series:
    """``m = pred * (2 * Phi(z) - 1)`` with ``z = (p - 1/k) / sqrt(p (1 - p))``."""
    prob = prob.clip(1e-6, 1.0 - 1e-6)
    z = (prob - 1.0 / num_classes) / np.sqrt(prob * (1.0 - prob))
    size = 2.0 * norm.cdf(z) - 1.0
    if pred is not None:
        size = size * pred
    return pd.Series(size, index=prob.index, name="signal")


# ----------------------------------------------------------------------------
# Snippet 10.2 -- average the bets that are simultaneously active
# ----------------------------------------------------------------------------
def _mp_avg_active_signals(signals: pd.DataFrame, molecule) -> pd.Series:
    out = pd.Series(dtype=float)
    for loc in molecule:
        active = (signals.index.values <= loc) & (
            (loc < signals["t1"].values) | pd.isnull(signals["t1"].values)
        )
        act = signals.loc[active, "signal"]
        out[loc] = act.mean() if act.shape[0] > 0 else 0.0
    return out


def avg_active_signals(signals: pd.DataFrame, num_threads: int = 1) -> pd.Series:
    """Average of every bet still open, evaluated on the union of start/end times.

    The output is a step function of the position through time: it changes
    only when a bet opens or closes, which is exactly when the target
    position can change.
    """
    t_pnts = set(signals["t1"].dropna().to_numpy())
    t_pnts = t_pnts.union(signals.index.to_numpy())
    t_pnts = sorted(t_pnts)
    return mp_pandas_obj(
        func=_mp_avg_active_signals,
        pd_obj=("molecule", t_pnts),
        num_threads=num_threads,
        signals=signals,
    ).rename("position")


# ----------------------------------------------------------------------------
# Snippet 10.3 -- discretise
# ----------------------------------------------------------------------------
def discrete_signal(signal: pd.Series, step_size: float) -> pd.Series:
    """Round to a grid of ``step_size`` and cap at unit leverage."""
    if step_size <= 0:
        return signal.clip(-1.0, 1.0)
    out = (signal / step_size).round() * step_size
    return out.clip(-1.0, 1.0)


# ----------------------------------------------------------------------------
# The full Snippet 10.1 pipeline
# ----------------------------------------------------------------------------
def get_signal(
    events: pd.DataFrame,
    prob: pd.Series,
    pred: pd.Series | None = None,
    num_classes: int = 2,
    step_size: float = 0.0,
    num_threads: int = 1,
    average_active: bool = True,
) -> pd.Series:
    """Probability -> size -> side -> averaged over active bets -> discretised.

    When ``events`` carries a ``side`` column (meta-labeling), the size is
    multiplied by it: the secondary model only ever scales a bet the primary
    model chose, and can scale it to zero, but never reverses it.
    """
    if prob.shape[0] == 0:
        return pd.Series(dtype=float)

    signal = prob_to_size(prob, pred, num_classes)
    if "side" in events.columns:
        signal = signal * events.loc[signal.index, "side"]

    if not average_active:
        return discrete_signal(signal, step_size)

    df = signal.to_frame("signal").join(events[["t1"]], how="left")
    position = avg_active_signals(df, num_threads)
    return discrete_signal(position, step_size)
