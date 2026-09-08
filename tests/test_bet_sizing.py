"""Chapter 10 -- turning probabilities into positions."""

import numpy as np
import pandas as pd
from scipy.stats import norm

from afml import bet_sizing as BS


def test_size_matches_the_closed_form():
    p = pd.Series([0.5, 0.6, 0.75, 0.9])
    z = (p - 0.5) / np.sqrt(p * (1 - p))
    assert np.allclose(BS.prob_to_size(p).to_numpy(), 2 * norm.cdf(z) - 1)


def test_size_is_monotone_bounded_and_zero_at_indifference():
    p = pd.Series(np.linspace(0.01, 0.99, 99))
    m = BS.prob_to_size(p)
    assert m.is_monotonic_increasing
    assert m.between(-1.0, 1.0).all()
    assert abs(BS.prob_to_size(pd.Series([0.5])).iloc[0]) < 1e-12
    # confidence, not the raw probability: 0.55 is a small bet, not half a bet
    assert BS.prob_to_size(pd.Series([0.55])).iloc[0] < 0.15


def test_side_is_applied_and_never_reversed():
    p = pd.Series([0.8, 0.8])
    pred = pd.Series([1.0, -1.0])
    m = BS.prob_to_size(p, pred)
    assert m.iloc[0] > 0 and m.iloc[1] < 0
    assert np.isclose(m.iloc[0], -m.iloc[1])


def test_active_signals_are_averaged_not_summed():
    idx = pd.to_datetime(["2021-01-01", "2021-01-02", "2021-01-03"])
    signals = pd.DataFrame(
        {"signal": [1.0, 1.0, 1.0],
         "t1": pd.to_datetime(["2021-01-04", "2021-01-03", "2021-01-05"])},
        index=idx,
    )
    pos = BS.avg_active_signals(signals)
    assert pos.loc[pd.Timestamp("2021-01-01")] == 1.0        # one bet open
    assert pos.loc[pd.Timestamp("2021-01-02")] == 1.0        # two bets, same size
    assert pos.max() <= 1.0                                   # never levers up
    assert pos.loc[pd.Timestamp("2021-01-05")] == 0.0        # all closed


def test_discretisation_snaps_to_the_grid_and_caps_leverage():
    s = pd.Series([-1.4, -0.11, 0.0, 0.13, 0.9, 1.7])
    d = BS.discrete_signal(s, 0.1)
    assert np.allclose(d.to_numpy(), [-1.0, -0.1, 0.0, 0.1, 0.9, 1.0])
    assert d.between(-1, 1).all()
    assert np.allclose(BS.discrete_signal(s, 0.0).to_numpy(), s.clip(-1, 1).to_numpy())


def test_get_signal_end_to_end():
    idx = pd.to_datetime(["2021-01-01", "2021-01-02", "2021-01-03"])
    events = pd.DataFrame(
        {"t1": pd.to_datetime(["2021-01-04", "2021-01-03", "2021-01-05"]),
         "side": [1.0, -1.0, 1.0]},
        index=idx,
    )
    prob = pd.Series([0.9, 0.8, 0.6], index=idx)
    pos = BS.get_signal(events, prob, step_size=0.25)
    assert pos.index[0] == idx[0] and pos.index[-1] == pd.Timestamp("2021-01-05")
    assert pos.between(-1, 1).all()
    assert pos.iloc[-1] == 0.0
    assert set(np.round(pos.to_numpy() / 0.25, 9)) <= set(range(-4, 5))


def test_empty_input_returns_empty():
    assert len(BS.get_signal(pd.DataFrame(columns=["t1"]), pd.Series(dtype=float))) == 0
