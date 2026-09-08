"""Bar construction  --  AFML Chapter 2.

Time bars oversample quiet periods and undersample active ones, and their
returns are further from IID-normal than activity-based alternatives.  This
module implements the book's full ladder:

* **Standard bars** -- tick, volume and dollar bars (Section 2.3.1).
  Dollar bars are the default throughout the pipeline: their size is robust
  to the price drift that biases tick and volume bars over long samples.
* **Information-driven bars** -- imbalance bars (TIB/VIB/DIB, Section 2.3.2.1)
  and run bars (TRB/VRB/DRB, Section 2.3.2.2), which sample whenever order
  flow surprises a running EWMA expectation, so that informed trading gets
  more observations than noise trading.

All of them share the tick rule (Section 2.3.2.1) for classifying trade
aggressor side from the transaction tape alone.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "tick_rule",
    "standard_bars",
    "tick_bars",
    "volume_bars",
    "dollar_bars",
    "imbalance_bars",
    "run_bars",
]

_OHLCV = ["open", "high", "low", "close", "volume", "dollar", "n_ticks", "vwap",
          "buy_volume", "signed_dollar"]


# ----------------------------------------------------------------------------
# The tick rule
# ----------------------------------------------------------------------------
def tick_rule(price: pd.Series | np.ndarray) -> np.ndarray:
    """Classify each trade as buyer (+1) or seller (-1) initiated.

    ``b_t = b_{t-1}`` if ``dp_t == 0`` else ``sign(dp_t)``  (AFML eq. 2.1).
    """
    p = np.asarray(price, dtype=float)
    b = np.sign(np.diff(p, prepend=p[0]))
    b[0] = 1.0
    # forward-fill the zeros without a Python loop
    idx = np.where(b != 0, np.arange(len(b)), 0)
    np.maximum.accumulate(idx, out=idx)
    return b[idx].astype(np.int8)


# ----------------------------------------------------------------------------
# Bar assembly shared by every sampling scheme
# ----------------------------------------------------------------------------
def _assemble(ticks: pd.DataFrame, signs: np.ndarray, edges: np.ndarray) -> pd.DataFrame:
    """Aggregate ticks into OHLCV bars given the *inclusive* end index of each bar."""
    if len(edges) == 0:
        raise ValueError("no bar was closed -- lower the sampling threshold")

    price = ticks["price"].to_numpy()
    vol = ticks["volume"].to_numpy()
    dol = ticks["dollar"].to_numpy()
    stamps = ticks.index.to_numpy()

    starts = np.concatenate(([0], edges[:-1] + 1))
    ends = edges + 1  # exclusive

    cum_v = np.concatenate(([0.0], np.cumsum(vol)))
    cum_d = np.concatenate(([0.0], np.cumsum(dol)))
    cum_bv = np.concatenate(([0.0], np.cumsum(np.where(signs > 0, vol, 0.0))))
    cum_sd = np.concatenate(([0.0], np.cumsum(signs * dol)))

    rows = {
        "open": price[starts],
        "high": np.maximum.reduceat(price, starts),
        "low": np.minimum.reduceat(price, starts),
        "close": price[ends - 1],
        "volume": cum_v[ends] - cum_v[starts],
        "dollar": cum_d[ends] - cum_d[starts],
        "n_ticks": (ends - starts).astype(float),
        "buy_volume": cum_bv[ends] - cum_bv[starts],
        "signed_dollar": cum_sd[ends] - cum_sd[starts],
    }
    rows["vwap"] = rows["dollar"] / rows["volume"]
    out = pd.DataFrame(rows, index=pd.DatetimeIndex(stamps[ends - 1], name="timestamp"))
    return out[_OHLCV]


# ----------------------------------------------------------------------------
# Standard bars  --  Section 2.3.1
# ----------------------------------------------------------------------------
def standard_bars(ticks: pd.DataFrame, threshold: float, kind: str = "dollar") -> pd.DataFrame:
    """Close a bar every time the cumulated ``kind`` crosses ``threshold``.

    Parameters
    ----------
    kind : {'tick', 'volume', 'dollar'}
        ``tick`` counts trades, ``volume`` sums shares, ``dollar`` sums
        traded value.  Dollar bars are preferred (AFML 2.3.1.3) because a
        fixed share or trade count is not comparable across price levels.
    """
    if kind == "tick":
        weights = np.ones(len(ticks))
    elif kind == "volume":
        weights = ticks["volume"].to_numpy(dtype=float)
    elif kind == "dollar":
        weights = ticks["dollar"].to_numpy(dtype=float)
    else:
        raise ValueError("kind must be one of 'tick', 'volume', 'dollar'")

    cum = np.cumsum(weights)
    # index of the last tick belonging to bar k = last i with cum[i] < (k+1)*threshold
    n_bars = int(cum[-1] // threshold)
    if n_bars == 0:
        raise ValueError("threshold exceeds the total sampled quantity")
    levels = threshold * np.arange(1, n_bars + 1)
    edges = np.searchsorted(cum, levels, side="left")
    edges = np.unique(np.minimum(edges, len(ticks) - 1))
    return _assemble(ticks, tick_rule(ticks["price"]), edges)


def tick_bars(ticks: pd.DataFrame, threshold: float) -> pd.DataFrame:
    return standard_bars(ticks, threshold, kind="tick")


def volume_bars(ticks: pd.DataFrame, threshold: float) -> pd.DataFrame:
    return standard_bars(ticks, threshold, kind="volume")


def dollar_bars(ticks: pd.DataFrame, threshold: float) -> pd.DataFrame:
    return standard_bars(ticks, threshold, kind="dollar")


# ----------------------------------------------------------------------------
# Information-driven bars  --  Section 2.3.2
# ----------------------------------------------------------------------------
def _ewma_update(prev: float, x: float, alpha: float) -> float:
    return x if np.isnan(prev) else alpha * x + (1.0 - alpha) * prev


def _run_length_alpha(span: int) -> float:
    return 2.0 / (span + 1.0)


def imbalance_bars(
    ticks: pd.DataFrame,
    kind: str = "dollar",
    expected_ticks_span: int = 100,
    expected_imbalance_span: int = 100,
    warm_up_ticks: float = 500.0,
    min_ticks_per_bar: int = 1,
    max_ticks_per_bar: int | None = None,
    max_bars: int | None = None,
) -> pd.DataFrame:
    """Imbalance bars (TIB / VIB / DIB)  --  AFML 2.3.2.1.

    Let ``theta_T = sum_{t<=T} b_t * v_t`` with ``v_t = 1`` (tick), share
    volume, or dollar volume.  A bar closes at the first ``T`` such that

        |theta_T| >= E_0[T] * |2 * P[b=1] * E[v | b=1] - E[v]|

    where the right-hand side is an EWMA over past bars of the bar length and
    over past ticks of the signed flow.  Because the expectation adapts, the
    sampler produces *more* bars when order flow becomes one-sided -- i.e. it
    allocates observations where information is arriving.

    A note on stability, which the book leaves to the reader.  The threshold
    is *linear* in ``E_0[T]`` while a balanced-flow imbalance only grows like
    ``sqrt(T)``, so the ``E_0[T]`` feedback loop has no interior fixed point:
    a run of two-sided flow makes bars longer, which raises the threshold,
    which makes them longer still.  Clamping ``E_0[T]`` (and hence the bar
    length) to ``[min_ticks_per_bar, max_ticks_per_bar]`` is the minimal
    intervention that keeps the sampler well behaved without altering its
    behaviour in the informative regime the bars exist to capture.
    """
    return _information_bars(
        ticks, kind, expected_ticks_span, expected_imbalance_span,
        warm_up_ticks, min_ticks_per_bar, max_ticks_per_bar, max_bars, mode="imbalance",
    )


def run_bars(
    ticks: pd.DataFrame,
    kind: str = "dollar",
    expected_ticks_span: int = 100,
    expected_imbalance_span: int = 100,
    warm_up_ticks: float = 500.0,
    min_ticks_per_bar: int = 1,
    max_ticks_per_bar: int | None = None,
    max_bars: int | None = None,
) -> pd.DataFrame:
    """Run bars (TRB / VRB / DRB)  --  AFML 2.3.2.2.

    Instead of the net imbalance, track the larger of the buy-side and
    sell-side cumulated flow within the bar,

        theta_T = max( sum_{b_t=1} v_t ,  -sum_{b_t=-1} v_t )

    and close when it exceeds ``E_0[T] * max(P[b=1] E[v|b=1], (1-P[b=1]) E[v|b=-1])``.
    Run bars are less sensitive than imbalance bars to order fragmentation,
    which chops a single parent order into alternating child trades.
    """
    return _information_bars(
        ticks, kind, expected_ticks_span, expected_imbalance_span,
        warm_up_ticks, min_ticks_per_bar, max_ticks_per_bar, max_bars, mode="run",
    )


def _information_bars(
    ticks: pd.DataFrame,
    kind: str,
    expected_ticks_span: int,
    expected_imbalance_span: int,
    warm_up_ticks: float,
    min_ticks_per_bar: int,
    max_ticks_per_bar: int | None,
    max_bars: int | None,
    mode: str,
) -> pd.DataFrame:
    if kind == "tick":
        v = np.ones(len(ticks))
    elif kind == "volume":
        v = ticks["volume"].to_numpy(dtype=float)
    elif kind == "dollar":
        v = ticks["dollar"].to_numpy(dtype=float)
    else:
        raise ValueError("kind must be one of 'tick', 'volume', 'dollar'")

    b = tick_rule(ticks["price"]).astype(float)
    signed = b * v

    a_t = _run_length_alpha(expected_ticks_span)
    a_i = _run_length_alpha(expected_imbalance_span)

    exp_ticks = float(warm_up_ticks)      # E_0[T]
    exp_buy = np.nan                      # EWMA of v_t on buy ticks
    exp_sell = np.nan                     # EWMA of v_t on sell ticks
    p_buy = np.nan                        # EWMA of 1{b_t = 1}

    edges: list[int] = []
    theta_pos = theta_neg = 0.0
    n_in_bar = 0

    for i in range(len(ticks)):
        n_in_bar += 1
        if b[i] > 0:
            theta_pos += v[i]
            exp_buy = _ewma_update(exp_buy, v[i], a_i)
        else:
            theta_neg += v[i]
            exp_sell = _ewma_update(exp_sell, v[i], a_i)
        p_buy = _ewma_update(p_buy, 1.0 if b[i] > 0 else 0.0, a_i)

        if np.isnan(exp_buy) or np.isnan(exp_sell) or n_in_bar < min_ticks_per_bar:
            continue

        if mode == "imbalance":
            theta = abs(theta_pos - theta_neg)
            exp_v = p_buy * exp_buy + (1.0 - p_buy) * exp_sell
            expected = abs(2.0 * p_buy * exp_buy - exp_v)
        else:  # run bars
            theta = max(theta_pos, theta_neg)
            expected = max(p_buy * exp_buy, (1.0 - p_buy) * exp_sell)

        forced = max_ticks_per_bar is not None and n_in_bar >= max_ticks_per_bar
        if forced or theta >= exp_ticks * expected:
            edges.append(i)
            exp_ticks = _ewma_update(exp_ticks, float(n_in_bar), a_t)
            exp_ticks = max(exp_ticks, float(min_ticks_per_bar))
            if max_ticks_per_bar is not None:
                exp_ticks = min(exp_ticks, float(max_ticks_per_bar))
            theta_pos = theta_neg = 0.0
            n_in_bar = 0
            if max_bars is not None and len(edges) >= max_bars:
                break

    return _assemble(ticks, tick_rule(ticks["price"]), np.asarray(edges, dtype=int))
