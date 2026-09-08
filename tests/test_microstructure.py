"""Chapter 19 -- microstructural estimators, checked against ground truth.

The simulator's parameters are known, so these are not smoke tests: each
estimator is asked to recover the quantity it claims to measure, and the
known biases are asserted rather than ignored.
"""

import numpy as np
import pytest

from afml.data import TickSimConfig, simulate_ticks
from afml.features import microstructure as M
from afml.features.tick_stats import rolling_tick_microstructure, tick_partial_sums


def test_roll_recovers_the_spread_when_its_assumptions_hold():
    """Roll (1984) assumes serially uncorrelated order flow and no permanent
    impact.  With ``c = 0`` and ``lam = 0`` the simulator satisfies both, and
    the estimator must land on the true spread to within a percent."""
    cfg = TickSimConfig(n_ticks=200_000, c=0.0, a=0.0, lam=0.0, seed=7)
    ticks, truth = simulate_ticks(cfg)
    bar_idx = ticks.index[::200]
    est = rolling_tick_microstructure(ticks, bar_idx, (200,))["roll_spread_200"].median()
    assert est == pytest.approx(truth["spread"], rel=0.02)


def test_roll_is_inflated_by_permanent_impact():
    """Adding Kyle impact makes ``dp_t`` and the bounce at ``t-1`` covary, so
    the estimator converges to ``s * sqrt(1 + 2 lam E[sqrt(V)] / s)`` rather
    than ``s``.  The bias is upward and predictable."""
    cfg = TickSimConfig(n_ticks=200_000, c=0.0, a=0.0, lam=7e-6, seed=7)
    ticks, truth = simulate_ticks(cfg)
    e_sqrt_v = np.sqrt(ticks["volume"]).mean()
    predicted = truth["spread"] * np.sqrt(1 + 2 * cfg.lam * e_sqrt_v / truth["spread"])
    est = rolling_tick_microstructure(ticks, ticks.index[::200], (200,))["roll_spread_200"].median()
    assert est > truth["spread"]
    assert est == pytest.approx(predicted, rel=0.03)


def test_roll_is_attenuated_by_order_flow_autocorrelation():
    """With sign autocorrelation rho the estimator converges to s*(1-rho),
    because cov(dp_t, dp_{t-1}) picks up -(s/2)^2 (1-rho)^2 instead of -(s/2)^2."""
    cfg = TickSimConfig(n_ticks=200_000, c=0.20, a=0.0, seed=7)
    ticks, truth = simulate_ticks(cfg)
    rho = 2 * cfg.c
    bar_idx = ticks.index[::200]
    est = rolling_tick_microstructure(ticks, bar_idx, (200,))["roll_spread_200"].median()
    assert est == pytest.approx(truth["spread"] * (1 - rho), rel=0.12)
    assert est < truth["spread"]                       # always understated


def test_roll_floors_at_zero_on_bar_data(dollar_bars):
    """Aggregation destroys the bounce: on bar closes the autocovariance is
    dominated by drift and Roll's estimator collapses.  This is why the
    package estimates it on ticks and only aggregates the result."""
    est = M.roll_measure(dollar_bars["close"], 50)
    assert (est.dropna() == 0).mean() > 0.4


@pytest.mark.parametrize("col", ["kyle_lambda_100", "hasbrouck_lambda_100"])
def test_impact_lambdas_rise_with_true_price_impact(col):
    est = []
    for lam in (0.0, 3e-5):
        ticks, _ = simulate_ticks(TickSimConfig(n_ticks=150_000, lam=lam, seed=11))
        f = rolling_tick_microstructure(ticks, ticks.index[::200], (100,))
        est.append(f[col].median())
    assert est[1] > 2.0 * est[0]


@pytest.mark.parametrize("col", ["kyle_lambda_100", "hasbrouck_lambda_100"])
def test_impact_lambdas_are_contaminated_by_the_spread(col):
    """With no permanent impact at all the estimates are still strictly
    positive, because the regressor is correlated with the bid-ask bounce in
    the dependent variable.  This is why the literature runs these
    regressions on midquotes, and why the level of an impact lambda estimated
    from transaction prices should never be read as a structural parameter --
    only its variation through time."""
    ticks, _ = simulate_ticks(TickSimConfig(n_ticks=150_000, lam=0.0, seed=11))
    f = rolling_tick_microstructure(ticks, ticks.index[::200], (100,))
    assert f[col].median() > 0


def test_amihud_lambda_is_positive(ticks):
    f = rolling_tick_microstructure(ticks, ticks.index[::200], (100,))
    assert (f["amihud_lambda_100"].dropna() > 0).all()


def test_vpin_is_a_fraction_and_responds_to_one_sided_flow():
    balanced, _ = simulate_ticks(TickSimConfig(n_ticks=120_000, c=0.0, a=0.0, seed=5))
    trending, _ = simulate_ticks(TickSimConfig(n_ticks=120_000, c=0.35, a=0.0, seed=5))
    v_bal = rolling_tick_microstructure(balanced, balanced.index[::200], (50,))["vpin_50"]
    v_trd = rolling_tick_microstructure(trending, trending.index[::200], (50,))["vpin_50"]
    assert v_bal.dropna().between(0, 1).all()
    assert v_trd.median() > v_bal.median()       # persistent flow is more toxic


def test_partial_sums_are_exactly_additive(ticks):
    """Bar-level sums must reproduce the tick-level totals, or every rolling
    estimator built on them is silently wrong."""
    bar_idx = ticks.index[::500]
    ps = tick_partial_sums(ticks, bar_idx)
    n_used = int(ps["n"].sum())
    assert np.isclose(ps["s_vol"].sum(), ticks["volume"].iloc[:n_used].sum())
    assert np.isclose(ps["s_dol2"].sum(), (ticks["dollar"].iloc[:n_used] ** 2).sum())
    assert ps["s_buy_vol"].sum() <= ps["s_vol"].sum()


def test_corwin_schultz_returns_a_non_negative_spread(dollar_bars):
    cs = M.corwin_schultz(dollar_bars["high"], dollar_bars["low"], 20)
    assert (cs["spread"].dropna() >= 0).all()
    assert (cs["sigma"].dropna() >= 0).all()
    assert cs["spread"].dropna().median() < 0.05     # a sane order of magnitude


def test_order_flow_imbalance_is_bounded(dollar_bars):
    ofi = M.bar_order_flow_imbalance(dollar_bars["signed_dollar"], dollar_bars["dollar"])
    assert ofi.dropna().between(-1.0, 1.0).all()
