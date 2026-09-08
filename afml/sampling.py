"""Sample uniqueness and weights  --  AFML Chapter 4.

Labels produced by the triple-barrier method overlap: event ``i`` spans
``[t_i, t1_i]`` and event ``j`` starting inside that span shares part of the
same price path.  Standard bagging and standard cross-validation both assume
IID draws, so overlap silently inflates the effective sample size, makes
bootstrap samples redundant, and lets in-sample information leak.  Chapter 4
supplies three corrections, all implemented here:

1. **Concurrency and average uniqueness** -- how many labels are open at each
   bar, and hence how much of each label is its own.
2. **Sequential bootstrap** -- draw with replacement but with probabilities
   updated after each draw to favour observations that overlap least with
   what is already in the bag, which makes the sample much closer to IID
   than a uniform bootstrap.
3. **Sample weights** -- combine uniqueness with return attribution (a label
   spanning a large absolute return deserves more weight than one spanning
   noise) and, optionally, time decay so that old observations matter less.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .multiprocess import mp_pandas_obj

__all__ = [
    "num_co_events",
    "get_num_co_events",
    "sample_tw",
    "get_avg_uniqueness_series",
    "get_ind_matrix",
    "get_avg_uniqueness",
    "seq_bootstrap",
    "sample_weights",
    "get_time_decay",
]


# ----------------------------------------------------------------------------
# Snippet 4.1 -- concurrency
# ----------------------------------------------------------------------------
def num_co_events(close_idx: pd.DatetimeIndex, t1: pd.Series, molecule) -> pd.Series:
    """Number of labels spanning each bar, restricted to ``molecule``."""
    t1 = t1.fillna(close_idx[-1])
    t1 = t1[t1 >= molecule[0]]
    t1 = t1.loc[: t1.loc[molecule].max()]

    iloc = close_idx.searchsorted(np.array([t1.index[0], t1.max()]))
    count = pd.Series(0, index=close_idx[iloc[0]: iloc[1] + 1])
    for t_in, t_out in t1.items():
        count.loc[t_in:t_out] += 1
    return count.loc[molecule[0]: t1.loc[molecule].max()]


def get_num_co_events(close_idx: pd.DatetimeIndex, t1: pd.Series, num_threads: int = 1) -> pd.Series:
    """Concurrency over the whole sample, reindexed onto ``close_idx``."""
    out = mp_pandas_obj(
        func=num_co_events,
        pd_obj=("molecule", t1.index),
        num_threads=num_threads,
        close_idx=close_idx,
        t1=t1,
    )
    out = out[~out.index.duplicated(keep="last")]
    return out.reindex(close_idx).fillna(0)


# ----------------------------------------------------------------------------
# Snippet 4.2 -- average uniqueness of each label
# ----------------------------------------------------------------------------
def sample_tw(t1: pd.Series, co_events: pd.Series, molecule) -> pd.Series:
    """Average uniqueness ``mean_t 1 / c_t`` over each label's lifespan."""
    w = pd.Series(index=molecule, dtype=float)
    for t_in, t_out in t1.loc[molecule].items():
        w.loc[t_in] = (1.0 / co_events.loc[t_in:t_out]).mean()
    return w


def get_avg_uniqueness_series(
    t1: pd.Series, co_events: pd.Series, num_threads: int = 1
) -> pd.Series:
    return mp_pandas_obj(
        func=sample_tw,
        pd_obj=("molecule", t1.index),
        num_threads=num_threads,
        t1=t1,
        co_events=co_events,
    ).rename("tW")


# ----------------------------------------------------------------------------
# Snippets 4.3 - 4.5 -- the sequential bootstrap (reference implementation)
# ----------------------------------------------------------------------------
def get_ind_matrix(bar_idx: pd.Index, t1: pd.Series) -> pd.DataFrame:
    """Binary indicator matrix: rows are bars, columns are labels (Snippet 4.3)."""
    ind_m = pd.DataFrame(0, index=bar_idx, columns=range(t1.shape[0]))
    for i, (t0, t1_) in enumerate(t1.items()):
        ind_m.loc[t0:t1_, i] = 1
    return ind_m


def get_avg_uniqueness(ind_m: pd.DataFrame) -> pd.Series:
    """Average uniqueness of each label given the indicator matrix (Snippet 4.4)."""
    c = ind_m.sum(axis=1)                       # concurrency
    u = ind_m.div(c, axis=0)                    # uniqueness per bar
    return u[u > 0].mean()


def _seq_bootstrap_reference(ind_m: pd.DataFrame, s_length: int | None, rng) -> list[int]:
    """Snippet 4.5, verbatim in spirit: O(n^2 * bars), for small problems and tests."""
    if s_length is None:
        s_length = ind_m.shape[1]
    phi: list[int] = []
    while len(phi) < s_length:
        avg_u = pd.Series(dtype=float)
        for i in ind_m:
            ind_m_ = ind_m[phi + [i]]
            avg_u.loc[i] = get_avg_uniqueness(ind_m_).iloc[-1]
        prob = (avg_u / avg_u.sum()).values
        phi.append(rng.choice(ind_m.columns, p=prob))
    return phi


def seq_bootstrap(
    t1: pd.Series,
    bar_idx: pd.Index,
    s_length: int | None = None,
    seed: int | None = None,
    method: str = "fast",
) -> np.ndarray:
    """Sequential bootstrap indices (Snippet 4.5).

    ``method='reference'`` is the book's loop, kept for validation.
    ``method='fast'`` exploits a structural property the book does not use:
    every triple-barrier label occupies a *contiguous* run of bars, so its
    average uniqueness is a windowed mean of ``1 / (c_t + 1)`` and can be read
    off a prefix sum.  Maintaining that prefix sum costs O(#bars) per draw
    instead of O(n * #bars), which turns an intractable loop into a second of
    work for a few thousand labels.  The two agree exactly for a given seed.
    """
    n = len(t1)
    if s_length is None:
        s_length = n
    rng = np.random.default_rng(seed)

    if method == "reference":
        ind_m = get_ind_matrix(bar_idx, t1)
        return np.asarray(_seq_bootstrap_reference(ind_m, s_length, rng), dtype=int)

    bar_idx = pd.Index(bar_idx)
    start = bar_idx.searchsorted(t1.index.to_numpy(), side="left")
    end = bar_idx.searchsorted(t1.to_numpy(), side="right") - 1
    end = np.maximum(end, start)
    span = (end - start + 1).astype(float)

    n_bars = len(bar_idx)
    c = np.zeros(n_bars)                       # concurrency of the current bag
    phi = np.empty(s_length, dtype=int)

    for k in range(s_length):
        w = 1.0 / (c + 1.0)                    # uniqueness if label i were added
        prefix = np.concatenate(([0.0], np.cumsum(w)))
        avg_u = (prefix[end + 1] - prefix[start]) / span
        prob = avg_u / avg_u.sum()
        j = rng.choice(n, p=prob)
        phi[k] = j
        c[start[j]: end[j] + 1] += 1.0
    return phi


# ----------------------------------------------------------------------------
# Snippets 4.10 / 4.11 -- return attribution and time decay
# ----------------------------------------------------------------------------
def _sample_w(t1: pd.Series, co_events: pd.Series, close: pd.Series, molecule) -> pd.Series:
    ret = np.log(close).diff()
    w = pd.Series(index=molecule, dtype=float)
    for t_in, t_out in t1.loc[molecule].items():
        w.loc[t_in] = (ret.loc[t_in:t_out] / co_events.loc[t_in:t_out]).sum()
    return w.abs()


def sample_weights(
    t1: pd.Series,
    co_events: pd.Series,
    close: pd.Series,
    num_threads: int = 1,
    normalise: bool = True,
) -> pd.Series:
    """Weight each label by the absolute return it captures, net of overlap.

    ``w_i = | sum_{t in [t_i, t1_i]} r_t / c_t |``  (Snippet 4.10).  Uniqueness
    is already inside the sum through ``c_t``, so this must *not* be
    multiplied by the average uniqueness again.  Normalised to average one so
    that the effective sample size is unchanged.
    """
    w = mp_pandas_obj(
        func=_sample_w,
        pd_obj=("molecule", t1.index),
        num_threads=num_threads,
        t1=t1,
        co_events=co_events,
        close=close,
    )
    if normalise:
        w = w * (w.shape[0] / w.sum())
    return w.rename("w")


def get_time_decay(tw: pd.Series, last_w: float = 1.0) -> pd.Series:
    """Piecewise-linear decay in *cumulative uniqueness*, not in clock time.

    ``last_w`` is the weight of the oldest observation: 1 means no decay,
    0 means the oldest observation gets zero weight, and a negative value
    means the oldest ``-last_w`` fraction of cumulative uniqueness is erased
    entirely (Snippet 4.11).
    """
    clf_w = tw.sort_index().cumsum()
    if last_w >= 0:
        slope = (1.0 - last_w) / clf_w.iloc[-1]
    else:
        slope = 1.0 / ((last_w + 1) * clf_w.iloc[-1])
    const = 1.0 - slope * clf_w.iloc[-1]
    clf_w = const + slope * clf_w
    clf_w[clf_w < 0] = 0.0
    return clf_w.rename("time_decay")
