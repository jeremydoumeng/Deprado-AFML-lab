"""The backtest engine: timing, costs, and the absence of look-ahead."""

import numpy as np
import pandas as pd
import pytest

from afml.backtest import CostModel, run_backtest


@pytest.fixture
def path():
    idx = pd.date_range("2021-01-01", periods=2000, freq="h")
    rng = np.random.default_rng(0)
    r = rng.normal(0, 0.004, 2000)
    return pd.Series(100 * np.exp(np.cumsum(r)), index=idx)


def test_positions_are_lagged_by_one_bar(path):
    """A signal known at the close of t may only earn the return of t+1.
    Dropping this lag is the single most common way to invent a backtest."""
    pos = pd.Series(np.sign(np.arange(len(path)) % 7 - 3.0), index=path.index)
    res = run_backtest(pos, path, CostModel(0.0, 0.0))
    expected = pos.shift(1).fillna(0.0) * path.pct_change().fillna(0.0)
    assert np.allclose(res.gross_returns.to_numpy(), expected.to_numpy())


def test_perfect_foresight_dominates_and_a_lagged_copy_does_not(path):
    """The oracle uses the *next* return, so it must be spectacular; the same
    signal shifted into the past must be worthless.  If both look good, the
    engine is peeking."""
    ret = path.pct_change()
    oracle = pd.Series(np.sign(ret.shift(-1)).fillna(0.0), index=path.index)
    stale = oracle.shift(5).fillna(0.0)
    sr_oracle = run_backtest(oracle, path, CostModel(0.0, 0.0)).summary["sharpe"]
    sr_stale = run_backtest(stale, path, CostModel(0.0, 0.0)).summary["sharpe"]
    assert sr_oracle > 20
    assert abs(sr_stale) < 5


def test_costs_scale_with_turnover_and_reduce_the_sharpe(path):
    churn = pd.Series(np.where(np.arange(len(path)) % 2 == 0, 1.0, -1.0), index=path.index)
    free = run_backtest(churn, path, CostModel(0.0, 0.0))
    charged = run_backtest(churn, path, CostModel(5e-4, 0.0))
    assert charged.summary["sharpe"] < free.summary["sharpe"]
    assert np.isclose(free.summary["total_cost"], 0.0)
    # turnover here is 2 units per bar after the first
    assert charged.summary["total_cost"] == pytest.approx(5e-4 * 2 * (len(path) - 1), rel=0.01)


def test_positions_are_forward_filled_and_clipped(path):
    sparse = pd.Series([2.0, -3.0], index=[path.index[10], path.index[500]])
    res = run_backtest(sparse, path, CostModel(0.0, 0.0))
    assert res.position.iloc[:10].eq(0.0).all()          # flat before the first signal
    assert res.position.iloc[10:500].eq(1.0).all()       # held, and capped at 1
    assert res.position.iloc[500:].eq(-1.0).all()
    assert res.position.between(-1, 1).all()


def test_equity_is_the_compounded_net_return(path):
    pos = pd.Series(np.sign(np.sin(np.arange(len(path)) / 30.0)), index=path.index)
    res = run_backtest(pos, path, CostModel(2e-4, 1e-4), initial_equity=10.0)
    assert np.allclose(res.equity.to_numpy(), 10.0 * (1 + res.returns).cumprod().to_numpy())
    assert np.allclose(res.returns, res.gross_returns - res.costs)


def test_bets_are_delimited_by_flats_and_flips(path):
    pos = pd.Series(0.0, index=path.index)
    pos.iloc[100:200] = 1.0
    pos.iloc[400:500] = -1.0
    res = run_backtest(pos, path, CostModel(0.0, 0.0))
    assert 2 <= len(res.bet_returns) <= 4
    assert res.summary["n_bets"] == len(res.bet_returns)


def test_annualisation_is_inferred_from_the_bar_clock(path):
    pos = pd.Series(1.0, index=path.index)
    res = run_backtest(pos, path, CostModel(0.0, 0.0))
    years = (path.index[-1] - path.index[0]).total_seconds() / (365.25 * 24 * 3600)
    assert res.summary["years"] == pytest.approx(years, rel=1e-6)
    buy_and_hold = path.iloc[-1] / path.iloc[0] - 1
    assert res.summary["total_return"] == pytest.approx(buy_and_hold, rel=1e-6)
