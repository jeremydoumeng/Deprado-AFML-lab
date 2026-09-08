"""Chapter 8 -- feature importance, audited against planted ground truth."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from afml import importance as I


@pytest.fixture(scope="module")
def planted():
    X, y = I.make_test_data(n_features=20, n_informative=4, n_redundant=4,
                            n_samples=2500, seed=1)
    t1 = pd.Series(X.index, index=X.index)          # non-overlapping labels
    return X, y, t1


@pytest.fixture(scope="module")
def forest():
    return RandomForestClassifier(
        n_estimators=60, max_features=1, criterion="entropy",
        min_weight_fraction_leaf=0.05, class_weight="balanced_subsample",
        random_state=0, n_jobs=2,
    )


def _by_group(series: pd.Series) -> pd.Series:
    return series.groupby([c.split("_")[0] for c in series.index]).mean()


def test_synthetic_data_has_the_promised_structure(planted):
    X, y, _ = planted
    assert sum(c.startswith("I_") for c in X.columns) == 4
    assert sum(c.startswith("R_") for c in X.columns) == 4
    assert sum(c.startswith("N_") for c in X.columns) == 12
    assert y.nunique() == 2
    # rows must be shuffled, otherwise contiguous test folds see one class only
    assert 0.4 < y.iloc[: len(y) // 2].mean() < 0.6
    # redundant columns really are substitutes for informative ones
    best = max(abs(X[f"I_{j}"].corr(X["R_0"])) for j in range(4))
    assert best > 0.95


def test_mdi_separates_signal_from_noise(planted, forest):
    X, y, _ = planted
    fit = forest.fit(X, y)
    mdi = I.feat_imp_mdi(fit, list(X.columns))
    assert np.isclose(mdi["mean"].sum(), 1.0)
    g = _by_group(mdi["mean"])
    assert g["I"] > 3 * g["N"]
    assert g["R"] > 3 * g["N"]                      # substitution shares the credit


def test_mda_separates_signal_from_noise(planted, forest):
    X, y, t1 = planted
    mda, base = I.feat_imp_mda(forest, X, y, cv=4, t1=t1, scoring="neg_log_loss")
    assert base < 0                                  # it is a negative log loss
    g = _by_group(mda["mean"])
    assert g["I"] > 0 and g["R"] > 0
    assert abs(g["N"]) < 0.2 * min(g["I"], g["R"])   # noise permutes for free
    assert set(mda["mean"].nlargest(6).index) & {f"I_{i}" for i in range(4)}


def test_sfi_is_immune_to_substitution(planted, forest):
    """SFI fits one feature at a time, so a redundant copy scores like the
    original instead of being masked by it."""
    X, y, t1 = planted
    sfi = I.feat_imp_sfi(forest, X, y, cv=4, t1=t1, scoring="neg_log_loss")
    g = _by_group(sfi["mean"])
    assert g["I"] > g["N"] and g["R"] > g["N"]
    assert abs(g["I"] - g["R"]) < 0.5 * abs(g["I"] - g["N"])


def test_orthogonal_features_are_uncorrelated_and_ordered(planted):
    X, _, _ = planted
    P = I.ortho_features(X, var_thres=0.95)
    assert P.shape[0] == X.shape[0] and P.shape[1] <= X.shape[1]
    corr = np.corrcoef(P.to_numpy(), rowvar=False)
    off = corr[~np.eye(len(corr), dtype=bool)]
    assert np.abs(off).max() < 1e-8                  # components are orthogonal
    var = P.var(axis=0).to_numpy()
    assert np.all(np.diff(var) <= 1e-8)              # ordered by explained variance


def test_report_combines_the_three_methods(planted, forest):
    X, y, t1 = planted
    rep = I.feature_importance_report(forest, X, y, t1=t1, cv=3)
    assert {"mdi", "mda", "sfi"} <= set(rep.columns)
    assert rep.index.sort_values().equals(X.columns.sort_values())
    assert rep["mda"].is_monotonic_decreasing        # sorted by MDA
    top = set(rep.head(8).index)
    planted_cols = {f"I_{i}" for i in range(4)} | {f"R_{i}" for i in range(4)}
    assert len(top & planted_cols) >= 5              # signal dominates the top
