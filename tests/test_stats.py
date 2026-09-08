"""Chapters 14 and 15 -- backtest statistics and strategy risk."""

import numpy as np
import pandas as pd
import pytest

from afml import stats as ST


def test_sharpe_annualises_correctly():
    idx = pd.date_range("2021-01-01", periods=2520, freq="D")
    r = pd.Series(np.full(2520, 0.001), index=idx) + pd.Series(
        np.random.default_rng(0).normal(0, 0.01, 2520), index=idx
    )
    sr = ST.sharpe_ratio(r, 252)
    assert np.isclose(sr, r.mean() / r.std(ddof=1) * np.sqrt(252))


def test_psr_rises_with_track_record_length():
    """Holding the observed Sharpe fixed, a longer record is more credible."""
    rng = np.random.default_rng(0)
    out = []
    for n in (60, 250, 2000):
        idx = pd.date_range("2021-01-01", periods=n, freq="D")
        z = rng.normal(0, 1, n)
        z = (z - z.mean()) / z.std(ddof=1)          # pin the sample moments
        r = pd.Series(0.0004 + 0.01 * z, index=idx)
        out.append(ST.probabilistic_sharpe_ratio(r, 0.0, 252))
    assert out[0] < out[1] < out[2]
    assert 0.0 <= min(out) and max(out) <= 1.0


def test_psr_penalises_negative_skew():
    """Two series with the same mean and variance but different skew must not
    get the same significance -- that is the whole point of the correction."""
    rng = np.random.default_rng(1)
    n = 1000
    idx = pd.date_range("2021-01-01", periods=n, freq="D")
    sym = pd.Series(rng.normal(0.0005, 0.01, n), index=idx)
    skewed = -pd.Series(rng.chisquare(1, n), index=idx)
    skewed = (skewed - skewed.mean()) / skewed.std() * sym.std() + sym.mean()
    assert skewed.skew() < -1.0
    assert ST.probabilistic_sharpe_ratio(skewed, 0.0, 252) < \
           ST.probabilistic_sharpe_ratio(sym, 0.0, 252)


def test_expected_max_sharpe_grows_with_the_number_of_trials():
    vals = [ST.expected_max_sharpe(n, 1.0) for n in (2, 10, 100, 1000)]
    assert vals == sorted(vals)
    assert ST.expected_max_sharpe(1, 1.0) == 0.0
    # scale: sd of trial Sharpes multiplies the expectation
    assert np.isclose(ST.expected_max_sharpe(100, 4.0),
                      2 * ST.expected_max_sharpe(100, 1.0))


def test_deflated_sharpe_is_stricter_than_psr():
    rng = np.random.default_rng(3)
    idx = pd.date_range("2021-01-01", periods=1000, freq="D")
    r = pd.Series(rng.normal(0.0006, 0.01, 1000), index=idx)
    psr = ST.probabilistic_sharpe_ratio(r, 0.0, 252)
    dsr = ST.deflated_sharpe_ratio(r, n_trials=200, var_trials_sr=0.5, periods_per_year=252)
    assert dsr < psr


def test_implied_precision_reproduces_the_book():
    """AFML 15.4: symmetric 1% payouts, 250 bets a year, target Sharpe 1."""
    p = ST.implied_precision(-0.01, 0.01, 250, 1.0)
    assert p == pytest.approx(0.5316, abs=1e-3)
    # more bets per year -> a lower precision suffices
    assert ST.implied_precision(-0.01, 0.01, 1000, 1.0) < p
    # asymmetric payouts in your favour -> lower precision suffices
    assert ST.implied_precision(-0.01, 0.02, 250, 1.0) < p


def test_prob_failure_flags_a_strategy_that_cannot_reach_its_target():
    idx = pd.date_range("2021-01-01", periods=400, freq="D")
    rng = np.random.default_rng(0)
    good = pd.Series(np.where(rng.random(400) < 0.70, 0.01, -0.01), index=idx)
    bad = pd.Series(np.where(rng.random(400) < 0.40, 0.01, -0.01), index=idx)
    r_good, thr_g, p_good = ST.prob_failure(good, 250, 1.0)
    r_bad, thr_b, p_bad = ST.prob_failure(bad, 250, 1.0)
    assert p_good > thr_g and r_good < 0.05
    assert p_bad < thr_b and r_bad > 0.95


def test_drawdown_and_time_under_water():
    idx = pd.date_range("2021-01-01", periods=6, freq="D")
    equity = pd.Series([1.0, 1.2, 0.9, 1.0, 1.3, 1.25], index=idx)
    dd, tuw = ST.drawdown_time_under_water(equity)
    assert np.isclose(dd.max(), 1 - 0.9 / 1.2)              # peak 1.2 -> trough 0.9
    assert (tuw >= 0).all()
    assert len(dd) == len(tuw)


def test_hhi_bounds():
    uniform = pd.Series(np.full(50, 0.01))
    assert ST.hhi(uniform) == pytest.approx(0.0, abs=1e-9)
    concentrated = pd.Series([1.0] + [1e-9] * 49)
    assert ST.hhi(concentrated) > 0.95
    assert np.isnan(ST.hhi(pd.Series([0.01, 0.02])))        # too few observations


def test_bet_timing_finds_flats_and_flips():
    idx = pd.date_range("2021-01-01", periods=7, freq="D")
    pos = pd.Series([0.5, 0.5, 0.0, -0.5, -0.5, 0.5, 0.5], index=idx)
    bets = ST.bet_timing(pos)
    assert idx[2] in bets            # flattened
    assert idx[5] in bets            # flipped sign
    assert idx[-1] in bets           # end of sample always closes the last bet


def test_performance_summary_is_complete():
    idx = pd.date_range("2021-01-01", periods=500, freq="D")
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0005, 0.01, 500), index=idx)
    eq = (1 + r).cumprod()
    bets = pd.Series(rng.normal(0.002, 0.02, 120))
    s = ST.performance_summary(r, eq, bets, periods_per_year=252,
                               n_trials=10, var_trials_sr=0.4)
    for k in ("sharpe", "psr_vs_0", "dsr", "max_drawdown", "hit_rate",
              "prob_failure", "hhi_positive", "cagr"):
        assert k in s.index and np.isfinite(s[k])
    assert np.isclose(s["total_return"], eq.iloc[-1] / eq.iloc[0] - 1)
