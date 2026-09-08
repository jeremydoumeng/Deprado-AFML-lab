"""Chapter 3 -- CUSUM sampling, triple barrier, meta-labeling."""

import numpy as np
import pandas as pd

from afml import labeling as L


# ----------------------------------------------------------------------------
# CUSUM
# ----------------------------------------------------------------------------
def test_cusum_fires_on_cumulative_moves_not_on_oscillation():
    idx = pd.date_range("2021-01-01", periods=9, freq="h")
    # +1% four times in a row, then alternating +-1%
    px = pd.Series([100, 101, 102.01, 103.03, 104.06, 103.02, 104.05, 103.01, 104.04],
                   index=idx)
    fired = L.cusum_filter(px, 0.025)
    assert len(fired) >= 1
    assert fired[0] <= idx[4]                 # the run of gains triggers
    # the oscillation afterwards must not trigger once per swing
    assert (fired > idx[4]).sum() <= 1


def test_cusum_resets_after_firing():
    idx = pd.date_range("2021-01-01", periods=40, freq="h")
    px = pd.Series(100 * np.exp(np.arange(40) * 0.01), index=idx)   # steady 1%/bar
    fired = L.cusum_filter(px, 0.045)
    gaps = np.diff([idx.get_loc(t) for t in fired])
    assert len(fired) >= 3
    assert set(gaps) == {5}                   # exactly one event per 5% of drift


def test_cusum_accepts_a_dynamic_threshold(dollar_bars):
    close = dollar_bars["close"]
    vol = L.get_vol(close, 100, pd.Timedelta(hours=4))
    few = L.cusum_filter(close, vol * 2.0)
    many = L.cusum_filter(close, vol * 0.5)
    assert len(many) > len(few) > 0


# ----------------------------------------------------------------------------
# Barriers
# ----------------------------------------------------------------------------
def test_first_touch_is_the_profit_barrier(step_path):
    """Price reaches +2.5% before it reaches -3%, so profit taking comes first."""
    events = pd.DataFrame(
        {"t1": [step_path.index[10]], "trgt": [0.02], "side": [1.0]},
        index=[step_path.index[0]],
    )
    touch = L.apply_pt_sl_on_t1(step_path, events, (1.0, 1.0), events.index)
    assert touch["pt"].iloc[0] == step_path.index[2]     # 102.5 clears +2% first
    assert touch["sl"].iloc[0] == step_path.index[5]     # 97.0 clears -2% later


def test_short_side_flips_the_barriers(step_path):
    events = pd.DataFrame(
        {"t1": [step_path.index[10]], "trgt": [0.02], "side": [-1.0]},
        index=[step_path.index[0]],
    )
    touch = L.apply_pt_sl_on_t1(step_path, events, (1.0, 1.0), events.index)
    # for a short, the rise to 102.5 is the loss and the fall to 97 the profit
    assert touch["sl"].iloc[0] == step_path.index[2]
    assert touch["pt"].iloc[0] == step_path.index[5]


def test_zero_disables_a_barrier(step_path):
    events = pd.DataFrame(
        {"t1": [step_path.index[10]], "trgt": [0.02], "side": [1.0]},
        index=[step_path.index[0]],
    )
    touch = L.apply_pt_sl_on_t1(step_path, events, (1.0, 0.0), events.index)
    assert pd.isna(touch["sl"].iloc[0])                  # stop loss switched off
    assert touch["pt"].iloc[0] == step_path.index[2]


def test_vertical_barrier_is_the_first_bar_past_the_horizon(dollar_bars):
    close = dollar_bars["close"]
    t_events = close.index[::200]
    t1 = L.add_vertical_barrier(t_events, close, pd.Timedelta(hours=6))
    elapsed = (t1 - t1.index).dt.total_seconds() / 3600
    assert (elapsed >= 6.0).all()
    # and it really is the *first* such bar
    prev = close.index[close.index.searchsorted(t1.to_numpy()) - 1]
    assert ((prev - t1.index).total_seconds() / 3600 < 6.0).all()


def test_t1_is_the_earliest_of_the_three_barriers(events, dollar_bars):
    """Events near the end of the sample legitimately have no barrier at all."""
    ev, bins = events
    touched = ev.dropna(subset=["t1"])
    assert len(touched) / len(ev) > 0.9
    assert (touched["t1"] > touched.index).all()
    assert touched["t1"].max() <= dollar_bars.index[-1]
    assert bins.index.equals(touched.index)   # unlabelled events are dropped, not guessed


# ----------------------------------------------------------------------------
# Labels
# ----------------------------------------------------------------------------
def test_primary_labels_are_the_sign_of_the_realised_return(events):
    ev, bins = events
    assert set(np.unique(bins["bin"])) <= {-1.0, 1.0}
    assert "side" not in ev.columns                     # side is being learned
    assert (np.sign(bins["ret"].replace(0.0, 1.0)) == bins["bin"]).all()


def test_meta_labels_are_binary_and_signed_by_the_side(dollar_bars):
    close = dollar_bars["close"]
    vol = L.get_vol(close, 100, pd.Timedelta(hours=4))
    t_events = L.cusum_filter(close, vol * 1.0)
    t1 = L.add_vertical_barrier(t_events, close, pd.Timedelta(hours=12))
    rng = np.random.default_rng(0)
    side = pd.Series(rng.choice([-1.0, 1.0], len(t_events)), index=t_events)

    ev = L.get_events(close, t_events, (1.0, 1.0), vol, t1=t1, side=side)
    bins = L.get_bins(ev, close)
    assert set(np.unique(bins["bin"])) <= {0.0, 1.0}
    # the return is expressed from the position's point of view
    raw = close.reindex(bins["t1"]).to_numpy() / close.reindex(bins.index).to_numpy() - 1
    assert np.allclose(bins["ret"].to_numpy(), raw * bins["side"].to_numpy())
    assert ((bins["ret"] > 0) == (bins["bin"] == 1)).all()


def test_min_ret_labels_marginal_bets_as_no_bet(dollar_bars):
    close = dollar_bars["close"]
    vol = L.get_vol(close, 100, pd.Timedelta(hours=4))
    t_events = L.cusum_filter(close, vol * 1.0)
    t1 = L.add_vertical_barrier(t_events, close, pd.Timedelta(hours=12))
    side = pd.Series(1.0, index=t_events)
    ev = L.get_events(close, t_events, (1.0, 1.0), vol, t1=t1, side=side)
    free = L.get_bins(ev, close, min_ret=0.0)["bin"]
    costed = L.get_bins(ev, close, min_ret=0.005)["bin"]
    assert costed.sum() < free.sum()          # some winners no longer clear costs
    assert ((costed == 1) <= (free == 1)).all()


def test_vol_target_tracks_realised_volatility(dollar_bars):
    close = dollar_bars["close"]
    v4 = L.get_vol(close, 100, pd.Timedelta(hours=4))
    v16 = L.get_vol(close, 100, pd.Timedelta(hours=16))
    assert (v4.dropna() > 0).all()
    # volatility scales roughly with the square root of the horizon
    ratio = v16.median() / v4.median()
    assert 1.4 < ratio < 3.0


def test_drop_labels_removes_rare_classes():
    """Snippet 3.8 prunes rare classes but never below two -- a one-class
    problem is not a classification problem."""
    bins = pd.DataFrame({"bin": [1.0] * 97 + [-1.0] * 2 + [0.0] * 1})
    out = L.drop_labels(bins, min_pct=0.05)
    assert set(out["bin"].unique()) == {1.0, -1.0}    # the 1% class went first
    assert len(out) == 99

    balanced = pd.DataFrame({"bin": [1.0] * 50 + [-1.0] * 50})
    assert len(L.drop_labels(balanced, min_pct=0.05)) == 100
