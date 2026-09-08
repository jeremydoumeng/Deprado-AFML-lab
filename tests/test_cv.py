"""Chapter 7 -- purging, embargo, walk-forward."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold

from afml import cv as CV


@pytest.fixture
def overlapping():
    """3000 labels, each spanning 100 bars: heavy, deliberate overlap."""
    n, h = 3000, 100
    idx = pd.date_range("2021-01-01", periods=n + h, freq="h")
    t1 = pd.Series(idx[h: n + h], index=idx[:n])
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(n, 4)), index=idx[:n],
                     columns=[f"x{i}" for i in range(4)])
    y = pd.Series(rng.integers(0, 2, n), index=idx[:n])
    return X, y, t1


# ----------------------------------------------------------------------------
# Purging
# ----------------------------------------------------------------------------
def test_get_train_times_drops_all_three_overlap_shapes():
    idx = pd.to_datetime(["2021-01-01", "2021-01-03", "2021-01-05", "2021-01-09"])
    t1 = pd.Series(pd.to_datetime(
        ["2021-01-04",   # ends inside the test window   -> dropped
         "2021-01-07",   # starts inside the test window  -> dropped
         "2021-01-06",   # starts and ends inside         -> dropped
         "2021-01-10"]   # entirely after                 -> kept
    ), index=idx)
    envelop = pd.Series(pd.to_datetime(["2021-01-08"]), index=[pd.Timestamp("2021-01-02")])
    test_times = pd.Series([pd.Timestamp("2021-01-06")], index=[pd.Timestamp("2021-01-04")])

    kept = CV.get_train_times(t1, test_times)
    assert list(kept.index) == [pd.Timestamp("2021-01-09")]
    assert len(CV.get_train_times(envelop, test_times)) == 0   # envelops the test set


def test_purged_kfold_leaves_zero_contamination(overlapping):
    _, _, t1 = overlapping
    audit = CV.label_overlap_audit(t1, n_splits=5, pct_embargo=0.01)
    assert audit.loc["kfold_no_purge", "n_contaminated"] > 0
    assert audit.loc["kfold_no_purge", "pct_contaminated"] > 0.05
    assert audit.loc["purged_embargo_0.01", "n_contaminated"] == 0


def test_purged_kfold_partitions_the_test_folds(overlapping):
    X, _, t1 = overlapping
    seen = []
    for train, test in CV.PurgedKFold(5, t1, 0.01).split(X):
        seen.append(test)
        assert len(np.intersect1d(train, test)) == 0
    all_test = np.concatenate(seen)
    assert np.array_equal(np.sort(all_test), np.arange(len(X)))   # exactly once each


def test_embargo_only_removes_training_data(overlapping):
    X, _, t1 = overlapping
    sizes = {}
    for emb in (0.0, 0.01, 0.05):
        n_train = [len(tr) for tr, _ in CV.PurgedKFold(5, t1, emb).split(X)]
        sizes[emb] = sum(n_train)
    assert sizes[0.0] > sizes[0.01] > sizes[0.05]


def test_get_embargo_times_shifts_forward():
    idx = pd.date_range("2021-01-01", periods=100, freq="D")
    mbrg = CV.get_embargo_times(idx, 0.05)
    assert (mbrg.loc[idx[:-5]].to_numpy() > idx[:-5].to_numpy()).all()
    assert mbrg.iloc[-1] == idx[-1]


def test_purged_kfold_requires_aligned_t1(overlapping):
    X, y, t1 = overlapping
    with pytest.raises(ValueError):
        list(CV.PurgedKFold(3, t1.iloc[::-1], 0.0).split(X))
    with pytest.raises(ValueError):
        CV.PurgedKFold(3, t1.to_frame(), 0.0)


# ----------------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------------
def test_cv_score_honours_sample_weights(overlapping):
    X, y, t1 = overlapping
    clf = RandomForestClassifier(n_estimators=20, min_weight_fraction_leaf=0.1,
                                 random_state=0, n_jobs=1)
    flat = CV.cv_score(clf, X, y, None, "accuracy", t1=t1, cv=3)
    skewed = pd.Series(np.linspace(0.1, 2.0, len(X)), index=X.index)
    weighted = CV.cv_score(clf, X, y, skewed, "accuracy", t1=t1, cv=3)
    assert flat.shape == weighted.shape == (3,)
    assert not np.allclose(flat, weighted)


def test_cv_score_rejects_unknown_scoring(overlapping):
    X, y, t1 = overlapping
    with pytest.raises(ValueError):
        CV.cv_score(RandomForestClassifier(n_estimators=5), X, y, scoring="f1", t1=t1)


# ----------------------------------------------------------------------------
# Walk-forward
# ----------------------------------------------------------------------------
def test_walk_forward_never_trains_on_the_future(overlapping):
    X, _, t1 = overlapping
    wf = CV.PurgedWalkForward(n_splits=5, t1=t1, pct_embargo=0.01, min_train_frac=0.4)
    folds = list(wf.split(X))
    assert len(folds) == 5
    prev_end = -1
    for train, test in folds:
        assert train.max() < test.min()                       # strictly in the past
        # and purged: no training label reaches into the test block
        assert (t1.iloc[train].to_numpy() < X.index[test[0]]).all()
        assert test.min() > prev_end
        prev_end = test.max()
    assert folds[0][0].size < folds[-1][0].size               # expanding window


def test_oof_predictions_cover_only_tested_rows(overlapping):
    X, y, t1 = overlapping
    clf = RandomForestClassifier(n_estimators=20, min_weight_fraction_leaf=0.1,
                                 random_state=0, n_jobs=1)
    wf = CV.PurgedWalkForward(n_splits=4, t1=t1, pct_embargo=0.01, min_train_frac=0.5)
    proba = CV.oof_predict_proba(clf, X, y, wf)
    tested = np.concatenate([te for _, te in wf.split(X)])
    assert proba.index.equals(X.index[np.sort(tested)])
    assert len(proba) < len(X)                                # the burn-in is excluded
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_compare_cv_reports_all_three_protocols(overlapping):
    X, y, t1 = overlapping
    clf = RandomForestClassifier(n_estimators=20, min_weight_fraction_leaf=0.1,
                                 random_state=0, n_jobs=1)
    tbl = CV.compare_cv(clf, X, y, t1, cv=3, scoring="accuracy", pct_embargo=0.02)
    assert list(tbl.index) == ["kfold_no_purge", "purged", "purged_embargo_0.02"]
    assert tbl["mean"].between(0.0, 1.0).all()
