"""Chapter 2 -- bar construction."""

import numpy as np
import pandas as pd
import pytest

from afml import bars as B


def test_tick_rule_matches_definition():
    px = pd.Series([10.0, 10.0, 10.1, 10.1, 10.0, 10.0])
    b = B.tick_rule(px)
    # first tick seeded +1; zeros inherit the previous sign
    assert list(b) == [1, 1, 1, 1, -1, -1]


def test_tick_rule_recovers_true_aggressor(ticks, truth):
    """The bounce dominates tick-level price changes, so the rule should be
    right well over chance -- real TAQ studies report ~85%."""
    acc = (B.tick_rule(ticks["price"]) == truth["true_sign"].to_numpy()).mean()
    assert 0.70 < acc < 1.0


def test_dollar_bars_conserve_volume_and_are_ordered(ticks, dollar_bars):
    n_used = int(dollar_bars["n_ticks"].sum())
    assert np.isclose(dollar_bars["volume"].sum(), ticks["volume"].iloc[:n_used].sum())
    assert np.isclose(dollar_bars["dollar"].sum(), ticks["dollar"].iloc[:n_used].sum())
    assert dollar_bars.index.is_monotonic_increasing
    assert (dollar_bars["high"] >= dollar_bars["low"]).all()
    assert (dollar_bars["high"] >= dollar_bars[["open", "close"]].max(axis=1)).all()
    assert (dollar_bars["low"] <= dollar_bars[["open", "close"]].min(axis=1)).all()


def test_dollar_bars_hit_their_threshold(ticks):
    thr = ticks["dollar"].sum() / 200.0
    bars = B.dollar_bars(ticks, thr)
    # every closed bar must have reached the threshold, except through its last tick
    assert (bars["dollar"] >= thr * 0.5).all()
    assert 150 <= len(bars) <= 210


def test_vwap_is_consistent(dollar_bars):
    assert np.allclose(
        dollar_bars["vwap"], dollar_bars["dollar"] / dollar_bars["volume"]
    )
    assert (dollar_bars["vwap"].between(dollar_bars["low"], dollar_bars["high"])).all()


def test_dollar_bars_are_closer_to_normal_than_time_bars(ticks, dollar_bars):
    """AFML 2.4: activity-based sampling brings returns closer to IID normal."""
    r_dollar = np.log(dollar_bars["close"]).diff().dropna()
    r_time = np.log(ticks["price"]).resample("1h").last().dropna().diff().dropna()
    assert abs(r_dollar.kurt()) < abs(r_time.kurt())


def test_bar_sign_aggregates_match_tick_rule(ticks, dollar_bars):
    b = B.tick_rule(ticks["price"])
    n = int(dollar_bars["n_ticks"].iloc[0])
    expected_buy = ticks["volume"].iloc[:n][b[:n] > 0].sum()
    assert np.isclose(dollar_bars["buy_volume"].iloc[0], expected_buy)


@pytest.mark.parametrize("fn", [B.imbalance_bars, B.run_bars])
def test_information_bars_respect_their_clamps(ticks, fn):
    bars = fn(ticks, "dollar", warm_up_ticks=200, min_ticks_per_bar=50,
              max_ticks_per_bar=2000, max_bars=400)
    assert len(bars) > 0
    assert bars["n_ticks"].min() >= 50
    assert bars["n_ticks"].max() <= 2000
    assert bars.index.is_monotonic_increasing


def test_threshold_too_large_raises(ticks):
    with pytest.raises(ValueError):
        B.dollar_bars(ticks, ticks["dollar"].sum() * 2)
