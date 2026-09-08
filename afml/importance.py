"""Feature importance  --  AFML Chapter 8.

Three complementary methods, because each fails differently:

* **MDI** (mean decrease impurity) -- in-sample, tree-specific, and *cheap*
  because it is a by-product of fitting.  Its blind spot is substitution: two
  correlated features share the credit, so both look half as important as
  either would alone, and a feature can look important only because it was
  useful early in a few trees.
* **MDA** (mean decrease accuracy) -- out-of-sample and model-agnostic:
  permute one column in the test fold and measure the score deterioration.
  Must run on *purged* folds or it measures leakage; it also suffers from
  substitution, and can rank an important feature at zero if a duplicate
  covers for it.
* **SFI** (single feature importance) -- fit a model on one feature at a
  time.  Immune to substitution, but blind to interactions: a feature that
  only matters jointly with another scores zero.

The chapter's advice is to read all three together, and to check them
against ``make_test_data``, which plants a known number of informative,
redundant and noise features so the methods can be audited rather than
trusted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, accuracy_score

from .cv import PurgedKFold, cv_score

__all__ = [
    "feat_imp_mdi",
    "feat_imp_mda",
    "feat_imp_sfi",
    "get_eigen_vectors",
    "ortho_features",
    "make_test_data",
    "feature_importance_report",
]


# ----------------------------------------------------------------------------
# Snippet 8.2 -- MDI
# ----------------------------------------------------------------------------
def feat_imp_mdi(fit, feature_names: list[str]) -> pd.DataFrame:
    """Mean decrease impurity, averaged across the trees of a fitted ensemble.

    Zeros are replaced by NaN before averaging: a feature that a given tree
    never selected contributes no information about its importance, and
    counting those zeros biases every importance towards the mean.  Requires
    ``max_features=1`` on the ensemble to keep substitution effects from
    masking features entirely (Section 8.3.1).
    """
    imp0 = {i: tree.feature_importances_ for i, tree in enumerate(fit.estimators_)}
    imp0 = pd.DataFrame.from_dict(imp0, orient="index")
    imp0.columns = feature_names
    imp0 = imp0.replace(0.0, np.nan)  # max_features=1 => many structural zeros
    imp = pd.concat(
        {"mean": imp0.mean(), "std": imp0.std() * imp0.shape[0] ** -0.5}, axis=1
    )
    imp /= imp["mean"].sum()
    return imp.sort_values("mean", ascending=False)


# ----------------------------------------------------------------------------
# Snippet 8.3 -- MDA
# ----------------------------------------------------------------------------
def feat_imp_mda(
    clf,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
    sample_weight: pd.Series | None = None,
    t1: pd.Series | None = None,
    pct_embargo: float = 0.0,
    scoring: str = "neg_log_loss",
    seed: int = 0,
) -> tuple[pd.DataFrame, float]:
    """Mean decrease accuracy on purged out-of-sample folds.

    Returns ``(importance_table, baseline_score)``.  The importance is the
    *relative* deterioration ``(s0 - s1) / -s1`` for log loss and
    ``(s0 - s1) / (1 - s1)`` for accuracy, as in the book, so that the number
    is comparable across folds with different baselines.
    """
    if sample_weight is None:
        sample_weight = pd.Series(1.0, index=X.index)
    cv_gen = PurgedKFold(n_splits=cv, t1=t1, pct_embargo=pct_embargo)
    rng = np.random.default_rng(seed)

    scr0 = pd.Series(dtype=float)
    scr1 = pd.DataFrame(columns=X.columns, dtype=float)

    for i, (train, test) in enumerate(cv_gen.split(X=X)):
        X0, y0, w0 = X.iloc[train], y.iloc[train], sample_weight.iloc[train]
        X1, y1, w1 = X.iloc[test], y.iloc[test], sample_weight.iloc[test]
        fit = clf.fit(X=X0, y=y0, sample_weight=w0.to_numpy())

        if scoring == "neg_log_loss":
            prob = fit.predict_proba(X1)
            scr0.loc[i] = -log_loss(y1, prob, sample_weight=w1.to_numpy(), labels=clf.classes_)
        else:
            pred = fit.predict(X1)
            scr0.loc[i] = accuracy_score(y1, pred, sample_weight=w1.to_numpy())

        for col in X.columns:
            X1_ = X1.copy(deep=True)
            X1_[col] = rng.permutation(X1_[col].to_numpy())  # break only this column
            if scoring == "neg_log_loss":
                prob = fit.predict_proba(X1_)
                scr1.loc[i, col] = -log_loss(
                    y1, prob, sample_weight=w1.to_numpy(), labels=clf.classes_
                )
            else:
                pred = fit.predict(X1_)
                scr1.loc[i, col] = accuracy_score(y1, pred, sample_weight=w1.to_numpy())

    imp = (-scr1).add(scr0, axis=0)
    if scoring == "neg_log_loss":
        imp = imp / -scr1
    else:
        imp = imp / (1.0 - scr1)

    out = pd.concat(
        {"mean": imp.mean(), "std": imp.std() * imp.shape[0] ** -0.5}, axis=1
    ).sort_values("mean", ascending=False)
    return out, float(scr0.mean())


# ----------------------------------------------------------------------------
# Snippet 8.4 -- SFI
# ----------------------------------------------------------------------------
def feat_imp_sfi(
    clf,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
    sample_weight: pd.Series | None = None,
    t1: pd.Series | None = None,
    pct_embargo: float = 0.0,
    scoring: str = "neg_log_loss",
) -> pd.DataFrame:
    """Cross-validated score of a model fitted on each feature alone."""
    rows = {}
    for col in X.columns:
        scores = cv_score(
            clf, X=X[[col]], y=y, sample_weight=sample_weight,
            scoring=scoring, t1=t1, cv=cv, pct_embargo=pct_embargo,
        )
        rows[col] = {"mean": scores.mean(), "std": scores.std() * scores.shape[0] ** -0.5}
    return pd.DataFrame.from_dict(rows, orient="index").sort_values("mean", ascending=False)


# ----------------------------------------------------------------------------
# Snippets 8.5 / 8.6 -- orthogonal features
# ----------------------------------------------------------------------------
def get_eigen_vectors(dot: np.ndarray, var_thres: float = 0.95) -> pd.DataFrame:
    """Eigen-decomposition of a correlation matrix, truncated by explained variance."""
    e_val, e_vec = np.linalg.eigh(dot)
    idx = e_val.argsort()[::-1]
    e_val, e_vec = e_val[idx], e_vec[:, idx]
    e_val = pd.Series(e_val, index=[f"PC_{i + 1}" for i in range(e_val.shape[0])])
    e_vec = pd.DataFrame(e_vec, index=range(e_vec.shape[0]), columns=e_val.index)
    e_vec = e_vec.loc[:, e_val > 0]
    e_val = e_val[e_val > 0]
    cum = e_val.cumsum() / e_val.sum()
    dim = int(cum.values.searchsorted(var_thres)) + 1
    return e_val.iloc[:dim], e_vec.iloc[:, :dim]


def ortho_features(X: pd.DataFrame, var_thres: float = 0.95) -> pd.DataFrame:
    """Standardise and rotate features onto their principal components.

    PCA is unsupervised, so it cannot overfit the labels.  If a supervised
    importance method ranks the *same* components highly that PCA finds
    dominant, that agreement is evidence the signal is structural rather than
    an artefact of the fitting procedure (Section 8.4.2).
    """
    Z = (X - X.mean()) / X.std()
    dot = (Z.T @ Z).to_numpy() / Z.shape[0]
    e_val, e_vec = get_eigen_vectors(dot, var_thres)
    P = Z.to_numpy() @ e_vec.to_numpy()
    return pd.DataFrame(P, index=X.index, columns=e_val.index)


# ----------------------------------------------------------------------------
# Snippet 8.7 -- a synthetic dataset with known ground truth
# ----------------------------------------------------------------------------
def make_test_data(
    n_features: int = 40,
    n_informative: int = 10,
    n_redundant: int = 10,
    n_samples: int = 10_000,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.Series]:
    """Classification data with informative, redundant and pure-noise columns.

    Columns are named ``I_*`` (informative), ``R_*`` (linear combinations of
    informative ones, i.e. substitutes) and ``N_*`` (noise).  Used to audit
    the importance methods: MDI and MDA should split credit across I and R
    columns and give N columns nothing; SFI should rank I and R alike.

    One deviation from Snippet 8.7, and it matters: the book passes
    ``shuffle=False`` to ``make_classification`` to keep the informative
    columns first, but that flag also leaves the *rows* sorted by class.
    Combined with the contiguous test folds of :class:`PurgedKFold`, every
    fold then contains a single class, MDA collapses to zero and SFI ranks
    noise above signal.  Here the columns are ordered by construction and the
    rows are shuffled explicitly.
    """
    from sklearn.datasets import make_classification

    rng = np.random.default_rng(seed)
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features - n_redundant,
        n_informative=n_informative,
        n_redundant=0,
        shuffle=False,
        random_state=seed,
    )
    cols = [f"I_{i}" for i in range(n_informative)]
    cols += [f"N_{i}" for i in range(n_features - n_redundant - n_informative)]
    X = pd.DataFrame(X, columns=cols)
    y = pd.Series(y)

    for k in range(n_redundant):
        j = rng.integers(0, n_informative)
        X[f"R_{k}"] = X[f"I_{j}"] + rng.normal(0.0, 0.05, X.shape[0])

    order = rng.permutation(n_samples)
    X = X.iloc[order].reset_index(drop=True)
    y = y.iloc[order].reset_index(drop=True)
    return X, y


# ----------------------------------------------------------------------------
# Convenience wrapper
# ----------------------------------------------------------------------------
def feature_importance_report(
    clf,
    X: pd.DataFrame,
    y: pd.Series,
    sample_weight: pd.Series | None = None,
    t1: pd.Series | None = None,
    cv: int = 5,
    pct_embargo: float = 0.01,
    scoring: str = "neg_log_loss",
    fit_full: bool = True,
) -> pd.DataFrame:
    """MDI, MDA and SFI side by side, ranked by MDA."""
    out = {}
    if fit_full:
        w = None if sample_weight is None else sample_weight.to_numpy()
        fit = clf.fit(X=X, y=y, sample_weight=w)
        if hasattr(fit, "estimators_"):
            out["mdi"] = feat_imp_mdi(fit, list(X.columns))["mean"]

    mda, base = feat_imp_mda(
        clf, X, y, cv=cv, sample_weight=sample_weight, t1=t1,
        pct_embargo=pct_embargo, scoring=scoring,
    )
    out["mda"] = mda["mean"]
    out["mda_std"] = mda["std"]
    sfi = feat_imp_sfi(
        clf, X, y, cv=cv, sample_weight=sample_weight, t1=t1,
        pct_embargo=pct_embargo, scoring=scoring,
    )
    out["sfi"] = sfi["mean"]

    report = pd.DataFrame(out)
    report.attrs["mda_baseline"] = base
    return report.sort_values("mda", ascending=False)
