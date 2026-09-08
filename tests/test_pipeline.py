"""End-to-end: a miniature version of scripts/run_pipeline.py."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_pipeline import build_dataset, make_classifier, run_strategy  # noqa: E402

CFG = {
    "bars_per_day": 50, "vol_horizon_h": 4, "cusum_mult": 0.5, "holding_h": 12,
    "pt": 1.0, "sl": 1.0, "min_ret": 0.0, "half_spread": 2.5e-4, "commission": 0.5e-4,
    "bet_step": 0.05, "wf_splits": 3, "embargo": 0.01, "min_train_frac": 0.4,
    "time_decay_last_w": 0.5, "round_trip_cost": 6e-4,
}


@pytest.fixture(scope="module")
def dataset(ticks, dollar_bars):
    return build_dataset(CFG, ticks, dollar_bars, d_ffd=0.3)


def test_dataset_pieces_are_mutually_aligned(dataset):
    X, y, events, labels, sw, tw, co = dataset
    assert X.index.equals(y.index)
    assert X.index.equals(events.index)
    assert X.index.equals(sw.index)
    assert set(np.unique(y)) <= {0, 1}
    assert "side" in events.columns                   # meta-labeling
    assert events["side"].isin([-1.0, 1.0]).all()
    assert (sw > 0).all() and np.isclose(sw.mean(), 1.0)
    assert (events["t1"] > events.index).all()


def test_labels_answer_the_meta_question(dataset, dollar_bars):
    """bin=1 exactly when the primary model's bet made more than costs."""
    _, y, events, labels, _, _, _ = dataset
    close = dollar_bars["close"]
    raw = (close.reindex(labels["t1"]).to_numpy()
           / close.reindex(labels.index).to_numpy() - 1)
    signed = raw * events["side"].to_numpy()
    assert np.allclose(labels["ret"].to_numpy(), signed)
    assert np.array_equal((signed > CFG["round_trip_cost"]).astype(int), y.to_numpy())


def test_walk_forward_run_produces_a_coherent_backtest(ticks, dollar_bars):
    art = run_strategy(CFG, ticks, dollar_bars, d_ffd=0.3, n_estimators=40)
    res, proba = art["result"], art["proba"]

    # predictions are out-of-sample only, and start after the training window
    assert proba.index.min() > art["X"].index[int(len(art["X"]) * 0.3)]
    assert np.allclose(proba.sum(axis=1), 1.0)

    s = res.summary
    assert res.position.between(-1, 1).all()
    assert np.isclose(res.equity.iloc[-1], (1 + res.returns).prod(), rtol=1e-9)
    assert s["total_cost"] > 0
    assert 0.0 <= s["max_drawdown"] <= 1.0
    assert s["n_bets"] > 0
    assert np.isfinite(s["sharpe"])
    assert np.isfinite(s["psr_vs_0"]) and 0.0 <= s["psr_vs_0"] <= 1.0
    # costs can only reduce performance
    assert s["sharpe"] <= s["gross_sharpe"]


def test_classifier_is_configured_as_the_book_recommends():
    clf = make_classifier()
    assert clf.max_features == 1                      # so MDI sees masked features
    assert clf.min_weight_fraction_leaf > 0           # leaves regularised by weight
    assert clf.criterion == "entropy"
    assert clf.class_weight == "balanced_subsample"
