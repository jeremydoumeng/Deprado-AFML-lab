"""The feature matrix: causality above all."""

import numpy as np
import pandas as pd

from afml.features import build_features


def test_no_feature_looks_into_the_future(dollar_bars, ticks):
    """The decisive property of the whole pipeline.

    Features are recomputed on a truncated history; every value at a shared
    timestamp must be bit-identical.  Any centred window, any full-sample
    scaling, any ``shift(-k)`` anywhere in the chain breaks this immediately.
    """
    cut = len(dollar_bars) * 3 // 4
    full = build_features(dollar_bars, ticks, d_ffd=0.3)
    truncated = build_features(
        dollar_bars.iloc[:cut],
        ticks.loc[: dollar_bars.index[cut - 1]],
        d_ffd=0.3,
    )
    common = full.index.intersection(truncated.index)
    assert len(common) > 100
    pd.testing.assert_frame_equal(
        full.loc[common], truncated.loc[common], check_exact=False, rtol=1e-9
    )


def test_feature_matrix_is_finite_and_aligned(dollar_bars, ticks):
    X = build_features(dollar_bars, ticks, d_ffd=0.3)
    assert X.index.is_monotonic_increasing
    assert X.index.isin(dollar_bars.index).all()
    assert np.isfinite(X.to_numpy()).all()
    assert not X.isna().any().any()
    assert len(X) < len(dollar_bars)          # rolling burn-in is dropped, not filled


def test_microstructure_columns_come_from_ticks_when_available(dollar_bars, ticks):
    with_ticks = build_features(dollar_bars, ticks, d_ffd=0.3)
    without = build_features(dollar_bars, None, d_ffd=0.3)
    assert "roll_spread_20" in with_ticks and "roll_spread_20" in without
    # the tick-based Roll estimate is informative; the bar-based one collapses
    assert with_ticks["roll_spread_20"].std() > 0
    assert (without["roll_spread_20"] == 0).mean() > (with_ticks["roll_spread_20"] == 0).mean()
