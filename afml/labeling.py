"""Triple-barrier labeling and meta-labeling  --  AFML Chapter 3.

Fixed-horizon labels ("sign of the return in N bars") ignore the path, so
they label as a win a trade that would have been stopped out first, and they
apply one threshold to every volatility regime.  The triple-barrier method
fixes both: each event gets two horizontal barriers scaled by an estimate of
local volatility (profit taking, stop loss) plus one vertical barrier (a
holding-period limit), and the label is the barrier touched first.

Meta-labeling (Section 3.6) then splits the problem in two.  A primary model
decides the *side*; a secondary model, trained on binary labels
"did the primary model's bet make money", decides the *size* -- including
size zero.  Because the secondary model only has to filter, it can raise
precision without ever damaging the primary model's recall, and its
predicted probability is directly usable as a bet size (Chapter 10).

Pipeline order:  ``get_vol`` -> ``cusum_filter`` -> ``get_events``
-> ``get_bins`` -> ``drop_labels``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .multiprocess import mp_pandas_obj

__all__ = [
    "get_vol",
    "cusum_filter",
    "add_vertical_barrier",
    "apply_pt_sl_on_t1",
    "get_events",
    "get_bins",
    "drop_labels",
]


# ----------------------------------------------------------------------------
# Snippet 3.1 -- volatility target for the horizontal barriers
# ----------------------------------------------------------------------------
def get_vol(
    close: pd.Series,
    span: int = 100,
    horizon: pd.Timedelta = pd.Timedelta(days=1),
) -> pd.Series:
    """EWMA standard deviation of returns measured over ``horizon``.

    The book calls this ``getDailyVol``; the horizon is a parameter here
    because with 50 dollar bars a day a one-day barrier is a very long trade.
    The returned series is the unit in which ``pt_sl`` is expressed: a
    profit-taking multiple of 2 means "two horizon-volatilities away".
    """
    idx = close.index.searchsorted(close.index - horizon)
    idx = idx[idx > 0]
    prev = pd.Series(
        close.index[idx - 1], index=close.index[close.shape[0] - idx.shape[0]:]
    )
    ret = close.loc[prev.index].to_numpy() / close.loc[prev.to_numpy()].to_numpy() - 1.0
    return pd.Series(ret, index=prev.index).ewm(span=span).std().rename("trgt")


# ----------------------------------------------------------------------------
# Snippet 2.4 -- the symmetric CUSUM filter (event sampling)
# ----------------------------------------------------------------------------
def cusum_filter(raw: pd.Series, h: float | pd.Series) -> pd.DatetimeIndex:
    """Sample an event whenever cumulative drift exceeds ``h``.

    Two running sums, one for each direction, accumulate log returns and are
    reset to zero when they breach ``h``.  Unlike a fixed-interval sample this
    does not fire twice on the same move, and unlike a Bollinger-style rule it
    cannot fire repeatedly while the price oscillates around a threshold.

    ``h`` may be a Series (typically the volatility target), giving a
    regime-adaptive sampling rate.
    """
    t_events: list[pd.Timestamp] = []
    s_pos = s_neg = 0.0
    diff = np.log(raw).diff().dropna()

    if isinstance(h, pd.Series):
        h = h.reindex(diff.index).ffill().bfill()
        h_arr = h.to_numpy(dtype=float)
    else:
        h_arr = np.full(len(diff), float(h))

    vals = diff.to_numpy(dtype=float)
    index = diff.index
    for i in range(len(vals)):
        s_pos = max(0.0, s_pos + vals[i])
        s_neg = min(0.0, s_neg + vals[i])
        if s_neg < -h_arr[i]:
            s_neg = 0.0
            t_events.append(index[i])
        elif s_pos > h_arr[i]:
            s_pos = 0.0
            t_events.append(index[i])
    return pd.DatetimeIndex(t_events)


# ----------------------------------------------------------------------------
# Snippet 3.4 -- the vertical barrier
# ----------------------------------------------------------------------------
def add_vertical_barrier(
    t_events: pd.DatetimeIndex,
    close: pd.Series,
    holding: pd.Timedelta,
) -> pd.Series:
    """Timestamp of the first bar at least ``holding`` after each event."""
    idx = close.index.searchsorted(t_events + holding)
    idx = idx[idx < close.shape[0]]
    return pd.Series(close.index[idx], index=t_events[: idx.shape[0]], name="t1")


# ----------------------------------------------------------------------------
# Snippet 3.2 -- which barrier is touched first
# ----------------------------------------------------------------------------
def apply_pt_sl_on_t1(
    close: pd.Series,
    events: pd.DataFrame,
    pt_sl: tuple[float, float],
    molecule,
) -> pd.DataFrame:
    """First touch time of the profit-taking and stop-loss barriers.

    Returns a frame with columns ``t1`` (the vertical barrier), ``pt`` and
    ``sl``; ``NaT`` where a barrier is never touched before ``t1``.  Note the
    multiplication by ``side``: for a short, a *fall* in price is the profit.
    """
    events_ = events.loc[molecule]
    out = events_[["t1"]].copy(deep=True)

    if pt_sl[0] > 0:
        pt = pt_sl[0] * events_["trgt"]
    else:
        pt = pd.Series(index=events_.index, dtype=float)  # all NaN -> never touched
    if pt_sl[1] > 0:
        sl = -pt_sl[1] * events_["trgt"]
    else:
        sl = pd.Series(index=events_.index, dtype=float)

    last = close.index[-1]
    for loc, t1 in events_["t1"].fillna(last).items():
        path = close[loc:t1]
        path = (path / close[loc] - 1.0) * events_.at[loc, "side"]  # path returns
        out.loc[loc, "sl"] = path[path < sl[loc]].index.min()
        out.loc[loc, "pt"] = path[path > pt[loc]].index.min()
    return out


# ----------------------------------------------------------------------------
# Snippet 3.3 / 3.6 -- assembling the events
# ----------------------------------------------------------------------------
def get_events(
    close: pd.Series,
    t_events: pd.DatetimeIndex,
    pt_sl: tuple[float, float],
    trgt: pd.Series,
    min_ret: float = 0.0,
    num_threads: int = 1,
    t1: pd.Series | None = None,
    side: pd.Series | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """Find the first barrier touched for every sampled event.

    Parameters
    ----------
    pt_sl : (float, float)
        Non-negative multiples of ``trgt`` for the profit-taking and
        stop-loss barriers.  ``0`` disables that barrier.
    side : Series, optional
        The primary model's position (+1 / -1).  ``None`` means we are
        *learning* the side, so the barriers are forced symmetric
        (``pt_sl[0]`` on both) and the output has no ``side`` column.
        Supplying ``side`` switches the run to meta-labeling, where the
        barriers may legitimately be asymmetric.

    Returns
    -------
    DataFrame indexed by event start, with ``t1`` (first touch of *any*
    barrier), ``trgt``, and ``side`` when meta-labeling.
    """
    trgt = trgt.reindex(t_events).dropna()
    trgt = trgt[trgt > min_ret]
    if len(trgt) == 0:
        raise ValueError("no event survives the min_ret filter")

    if t1 is None:
        t1 = pd.Series(pd.NaT, index=trgt.index)
    else:
        t1 = t1.reindex(trgt.index)

    if side is None:
        side_ = pd.Series(1.0, index=trgt.index)
        pt_sl_ = (pt_sl[0], pt_sl[0])   # symmetric: we do not know the side yet
    else:
        side_ = side.reindex(trgt.index)
        pt_sl_ = (pt_sl[0], pt_sl[1])

    events = pd.concat({"t1": t1, "trgt": trgt, "side": side_}, axis=1).dropna(subset=["trgt"])

    touches = mp_pandas_obj(
        func=apply_pt_sl_on_t1,
        pd_obj=("molecule", events.index),
        num_threads=num_threads,
        verbose=verbose,
        close=close,
        events=events,
        pt_sl=pt_sl_,
    )
    events["t1"] = touches.dropna(how="all").min(axis=1)
    if side is None:
        events = events.drop(columns=["side"])
    return events


# ----------------------------------------------------------------------------
# Snippet 3.5 / 3.7 -- labels
# ----------------------------------------------------------------------------
def get_bins(events: pd.DataFrame, close: pd.Series, min_ret: float = 0.0) -> pd.DataFrame:
    """Label each event by the barrier it touched.

    Without ``side``: ``bin`` in {-1, +1} is the sign of the realised return,
    i.e. the *direction* a primary model should learn.

    With ``side`` (meta-labeling): ``ret`` is already signed by the position,
    and ``bin`` in {0, 1} answers "was taking this bet profitable" -- the
    secondary model's target.  ``min_ret`` lets a bet that merely broke even
    (after costs, say) be labelled 0.
    """
    events_ = events.dropna(subset=["t1"])
    px = events_.index.union(pd.DatetimeIndex(events_["t1"].to_numpy())).drop_duplicates()
    px = close.reindex(px, method="bfill")

    out = pd.DataFrame(index=events_.index)
    out["ret"] = px.loc[events_["t1"].to_numpy()].to_numpy() / px.loc[events_.index].to_numpy() - 1.0
    out["t1"] = events_["t1"]
    out["trgt"] = events_["trgt"]

    meta = "side" in events_.columns
    if meta:
        out["ret"] = out["ret"] * events_["side"]
        out["side"] = events_["side"]

    out["bin"] = np.sign(out["ret"])
    if meta:
        out.loc[out["ret"] <= min_ret, "bin"] = 0.0
        out["bin"] = (out["bin"] > 0).astype(float)
    else:
        # a return of exactly zero is arbitrary; assign it to the long class
        out.loc[out["bin"] == 0.0, "bin"] = 1.0
    return out


# ----------------------------------------------------------------------------
# Snippet 3.8 -- prune rare labels
# ----------------------------------------------------------------------------
def drop_labels(events: pd.DataFrame, min_pct: float = 0.05) -> pd.DataFrame:
    """Recursively drop classes rarer than ``min_pct``.

    Sklearn classifiers do not cope with a class that appears a handful of
    times; the book's remedy is to remove it and re-check, since removing one
    class changes the frequencies of the others.
    """
    out = events
    while True:
        df = out["bin"].value_counts(normalize=True)
        if df.min() > min_pct or df.shape[0] < 3:
            break
        out = out[out["bin"] != df.idxmin()]
    return out
