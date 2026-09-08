"""Chapter 5 -- fractional differentiation."""

import numpy as np

from afml import fracdiff as F


def test_weight_recursion():
    d = 0.4
    w = F.get_weights(d, 5)[:, 0][::-1]  # w_0, w_1, ...
    assert w[0] == 1.0
    for k in range(1, 5):
        assert np.isclose(w[k], -w[k - 1] / k * (d - k + 1))


def test_integer_orders_are_the_familiar_operators(dollar_bars):
    lp = np.log(dollar_bars["close"])
    assert len(F.get_weights_ffd(0.0, 1e-5)) == 1        # d=0 -> identity
    zero = F.frac_diff_ffd(lp, 0.0).iloc[:, 0]
    assert np.allclose(zero.to_numpy(), lp.reindex(zero.index).to_numpy())

    one = F.frac_diff_ffd(lp, 1.0).iloc[:, 0]
    assert np.allclose(one.to_numpy(), lp.diff().dropna().reindex(one.index).to_numpy())


def test_window_width_shrinks_with_d():
    widths = [len(F.get_weights_ffd(d, 1e-4)) for d in (0.2, 0.4, 0.6, 0.8)]
    assert widths == sorted(widths, reverse=True)


def test_memory_decreases_monotonically_with_d(dollar_bars):
    """Correlations are computed on a common index, so they must be ordered."""
    _, table = F.min_ffd(np.log(dollar_bars["close"]), d_grid=np.linspace(0, 1, 11))
    corr = table["corr_with_original"]
    assert corr.iloc[0] > 0.99                       # d=0 is the level itself
    assert corr.is_monotonic_decreasing
    assert corr.iloc[-1] < 0.2                       # d=1 keeps essentially no memory


def test_min_ffd_finds_a_stationary_order_that_keeps_memory(dollar_bars):
    d_star, table = F.min_ffd(np.log(dollar_bars["close"]))
    assert 0.0 < d_star < 1.0
    row = table.loc[d_star]
    assert row["p_value"] < 0.05                     # stationary
    assert row["adf_stat"] < row["crit_95"]
    # the whole point of the chapter: stationarity well before memory is gone
    assert row["corr_with_original"] > 0.5
    below = table.loc[table.index < d_star, "p_value"]
    assert (below >= 0.05).all()                     # d_star really is the minimum


def test_ffd_is_exactly_the_fixed_width_dot_product(dollar_bars):
    """The convolution shortcut must equal the definition, term by term."""
    lp = np.log(dollar_bars["close"]).iloc[:1500]
    d, thres = 0.5, 1e-4
    w = F.get_weights_ffd(d, thres)[:, 0]
    width = len(w) - 1
    ffd = F.frac_diff_ffd(lp, d, thres).iloc[:, 0]
    vals = lp.to_numpy()
    for i in (width, width + 7, len(lp) - 1):
        expected = float(w @ vals[i - width: i + 1])
        assert np.isclose(ffd.loc[lp.index[i]], expected)


def test_expanding_window_uses_a_growing_window(dollar_bars):
    """And the expanding variant must equal *its* definition, which is the
    reason its effective d drifts and the book prefers the fixed width."""
    lp = np.log(dollar_bars["close"]).iloc[:600]
    d, thres = 0.5, 0.01
    exp = F.frac_diff(lp, d, thres).iloc[:, 0]
    w_full = F.get_weights(d, len(lp))[:, 0]
    vals = lp.to_numpy()
    for stamp in (exp.index[0], exp.index[len(exp) // 2], exp.index[-1]):
        i = lp.index.get_loc(stamp)
        expected = float(w_full[-(i + 1):] @ vals[: i + 1])
        assert np.isclose(exp.loc[stamp], expected)
    # the window at the end is strictly longer than at the start
    assert lp.index.get_loc(exp.index[-1]) > lp.index.get_loc(exp.index[0])
