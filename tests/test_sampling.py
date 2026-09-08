"""Chapter 4 -- concurrency, uniqueness, sequential bootstrap, weights."""

import numpy as np
import pandas as pd

from afml import sampling as S


# ----------------------------------------------------------------------------
# The book's worked example (Snippets 4.3 - 4.5)
# ----------------------------------------------------------------------------
def _toy():
    return pd.Series([2, 3, 5], index=[0, 2, 4]), pd.Index(range(6))


def test_indicator_matrix_matches_the_book():
    t1, bar_idx = _toy()
    m = S.get_ind_matrix(bar_idx, t1)
    expected = np.array([[1, 1, 1, 0, 0, 0],
                         [0, 0, 1, 1, 0, 0],
                         [0, 0, 0, 0, 1, 1]])
    assert np.array_equal(m.to_numpy().T, expected)


def test_average_uniqueness_matches_the_book():
    t1, bar_idx = _toy()
    u = S.get_avg_uniqueness(S.get_ind_matrix(bar_idx, t1))
    # bars 2 is shared by labels 0 and 1; label 2 overlaps nothing
    assert np.allclose(u.to_numpy(), [5 / 6, 3 / 4, 1.0])


def test_fast_sequential_bootstrap_equals_the_reference():
    t1, bar_idx = _toy()
    for seed in (0, 1, 7, 42):
        ref = S.seq_bootstrap(t1, bar_idx, seed=seed, method="reference")
        fast = S.seq_bootstrap(t1, bar_idx, seed=seed, method="fast")
        assert np.array_equal(ref, fast)


# ----------------------------------------------------------------------------
# Concurrency and uniqueness on real events
# ----------------------------------------------------------------------------
def test_concurrency_counts_open_labels(events, dollar_bars):
    ev, _ = events
    t1 = ev["t1"].dropna()
    co = S.get_num_co_events(dollar_bars.index, t1)
    assert (co >= 0).all()
    assert co.max() >= 2                                  # labels do overlap
    # spot-check three bars against a brute-force count
    rng = np.random.default_rng(0)
    for stamp in rng.choice(dollar_bars.index[500:-500], 3, replace=False):
        stamp = pd.Timestamp(stamp)
        brute = ((t1.index <= stamp) & (t1.to_numpy() >= stamp)).sum()
        assert co.loc[stamp] == brute


def test_uniqueness_is_bounded_and_inverse_to_overlap(events, dollar_bars):
    ev, _ = events
    t1 = ev["t1"].dropna()
    co = S.get_num_co_events(dollar_bars.index, t1)
    tw = S.get_avg_uniqueness_series(t1, co)
    assert tw.between(0.0, 1.0).all()
    assert 0.0 < tw.mean() < 1.0
    span = (t1 - t1.index).dt.total_seconds()
    assert np.corrcoef(span, tw)[0, 1] < 0                # longer labels are less unique


def test_sequential_bootstrap_beats_uniform_on_uniqueness(events, dollar_bars):
    """The whole justification of Snippet 4.5, checked over repeated draws."""
    ev, _ = events
    t1 = ev["t1"].dropna().iloc[:400]
    bar_idx = dollar_bars.index

    start = bar_idx.searchsorted(t1.index.to_numpy())
    end = np.maximum(bar_idx.searchsorted(t1.to_numpy(), "right") - 1, start)

    def mean_uniqueness(sel):
        c = np.zeros(len(bar_idx))
        for j in sel:
            c[start[j]: end[j] + 1] += 1
        return np.mean([(1.0 / c[start[j]: end[j] + 1]).mean() for j in sel])

    seq, uni = [], []
    for seed in range(5):
        seq.append(mean_uniqueness(S.seq_bootstrap(t1, bar_idx, seed=seed)))
        uni.append(mean_uniqueness(
            np.random.default_rng(seed).integers(0, len(t1), len(t1))
        ))
    assert np.mean(seq) > np.mean(uni)


# ----------------------------------------------------------------------------
# Weights
# ----------------------------------------------------------------------------
def test_return_attribution_weights(events, dollar_bars):
    ev, _ = events
    t1 = ev["t1"].dropna()
    co = S.get_num_co_events(dollar_bars.index, t1)
    w = S.sample_weights(t1, co, dollar_bars["close"])
    assert (w >= 0).all()
    assert np.isclose(w.mean(), 1.0)                      # normalised
    # labels spanning bigger moves get more weight
    realised = (dollar_bars["close"].reindex(t1).to_numpy()
                / dollar_bars["close"].reindex(t1.index).to_numpy() - 1)
    assert np.corrcoef(np.abs(realised), w)[0, 1] > 0.3


def test_time_decay_endpoints():
    tw = pd.Series(np.full(100, 0.5), index=pd.RangeIndex(100))
    for last_w in (1.0, 0.5, 0.0):
        d = S.get_time_decay(tw, last_w)
        assert np.isclose(d.iloc[-1], 1.0)                # newest always weight 1
        assert np.isclose(d.iloc[0], last_w, atol=0.02)   # oldest set by last_w
        assert d.is_monotonic_increasing
    erased = S.get_time_decay(tw, -0.5)                   # oldest half discarded
    assert (erased == 0).sum() > 0
    assert np.isclose(erased.iloc[-1], 1.0)
