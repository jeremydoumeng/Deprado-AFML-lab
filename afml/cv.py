"""Purged cross-validation  --  AFML Chapter 7.

Plain k-fold cross-validation is invalid on financial labels for two
reasons, and both are fatal rather than cosmetic:

* **Overlap.**  A label at ``t`` spans ``[t, t1]``.  If ``t`` is in the train
  fold and any part of ``[t, t1]`` is in the test fold, the two folds share
  the same outcome.  The classifier can memorise it.  *Purging* removes from
  the training set every observation whose span intersects the test set.
* **Serial correlation.**  Even non-overlapping observations immediately
  after the test set are informative about it, because features are built on
  overlapping rolling windows.  An *embargo* drops a further ``pct_embargo``
  of the sample after each test fold.

The practical symptom of skipping this is a cross-validated score that looks
excellent and a live performance that does not exist.  ``compare_cv`` below
is the diagnostic: it reports the naive score and the purged score side by
side, and the gap is the leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import KFold

__all__ = [
    "get_train_times",
    "get_embargo_times",
    "PurgedKFold",
    "PurgedWalkForward",
    "cv_score",
    "compare_cv",
    "label_overlap_audit",
    "oof_predict_proba",
]


# ----------------------------------------------------------------------------
# Snippet 7.1 -- purging
# ----------------------------------------------------------------------------
def get_train_times(t1: pd.Series, test_times: pd.Series) -> pd.Series:
    """Drop from ``t1`` every observation overlapping any interval in ``test_times``.

    Three ways a training label can overlap a test interval ``[i, j]``:
    it starts inside it, it ends inside it, or it envelops it.  All three are
    removed.
    """
    trn = t1.copy(deep=True)
    for i, j in test_times.items():
        df0 = trn[(i <= trn.index) & (trn.index <= j)].index   # starts within test
        df1 = trn[(i <= trn) & (trn <= j)].index               # ends within test
        df2 = trn[(trn.index <= i) & (j <= trn)].index         # envelops test
        trn = trn.drop(df0.union(df1).union(df2))
    return trn


# ----------------------------------------------------------------------------
# Snippet 7.2 -- embargo
# ----------------------------------------------------------------------------
def get_embargo_times(times: pd.Index, pct_embargo: float) -> pd.Series:
    """Map each timestamp to the end of its embargo period."""
    step = int(times.shape[0] * pct_embargo)
    if step == 0:
        return pd.Series(times, index=times)
    mbrg = pd.Series(times[step:], index=times[:-step])
    mbrg = pd.concat([mbrg, pd.Series(times[-1], index=times[-step:])])
    return mbrg


# ----------------------------------------------------------------------------
# Snippet 7.3 -- the cross-validator
# ----------------------------------------------------------------------------
class PurgedKFold(KFold):
    """K-fold with contiguous test folds, purging and an embargo.

    Test folds are contiguous *in time* (never shuffled -- shuffling would
    scatter overlapping labels across folds and defeat the purge).  Training
    observations whose ``[t, t1]`` span touches a test fold are purged, and
    ``pct_embargo`` of the sample immediately following a test fold is
    dropped as well.

    Parameters
    ----------
    t1 : Series
        Index = label start, value = label end.  Must be aligned with ``X``.
    pct_embargo : float
        Fraction of the *whole* sample embargoed after each test fold.  The
        book's rule of thumb is 0.01 or a little more.
    """

    def __init__(self, n_splits: int = 3, t1: pd.Series | None = None, pct_embargo: float = 0.0):
        if not isinstance(t1, pd.Series):
            raise ValueError("t1 must be a pandas Series indexed by label start")
        super().__init__(n_splits=n_splits, shuffle=False, random_state=None)
        self.t1 = t1
        self.pct_embargo = pct_embargo

    def split(self, X, y=None, groups=None):
        if (X.index != self.t1.index).any():
            raise ValueError("X and t1 must share the same index, in the same order")

        indices = np.arange(X.shape[0])
        mbrg = int(X.shape[0] * self.pct_embargo)
        test_ranges = [(i[0], i[-1] + 1) for i in np.array_split(indices, self.n_splits)]

        for start, end in test_ranges:
            t0 = self.t1.index[start]                       # start of the test fold
            test_indices = indices[start:end]
            max_t1_idx = self.t1.index.searchsorted(self.t1.iloc[test_indices].max())

            # left side: labels that ended before the test fold began
            train_indices = self.t1.index.searchsorted(self.t1[self.t1 <= t0].index)
            # right side: everything after the last label of the test fold, plus embargo
            if max_t1_idx < X.shape[0]:
                train_indices = np.concatenate((train_indices, indices[max_t1_idx + mbrg:]))
            yield train_indices, test_indices


# ----------------------------------------------------------------------------
# Snippet 7.4 -- scoring with sample weights
# ----------------------------------------------------------------------------
def cv_score(
    clf,
    X: pd.DataFrame,
    y: pd.Series,
    sample_weight: pd.Series | np.ndarray | None = None,
    scoring: str = "neg_log_loss",
    t1: pd.Series | None = None,
    cv: int = 5,
    cv_gen=None,
    pct_embargo: float = 0.0,
    return_folds: bool = False,
):
    """Cross-validated score that respects sample weights and purging.

    ``sklearn.cross_val_score`` cannot be used here: it neither passes sample
    weights to ``fit`` and to the scorer, nor understands overlapping labels.

    ``neg_log_loss`` is preferred to accuracy (Section 7.4): investment
    decisions size positions by predicted probability, so a model that is
    right with low confidence and wrong with high confidence is worse than
    its hit rate suggests, and only a probability-aware loss sees that.
    """
    if scoring not in ("neg_log_loss", "accuracy"):
        raise ValueError("scoring must be 'neg_log_loss' or 'accuracy'")
    if cv_gen is None:
        cv_gen = PurgedKFold(n_splits=cv, t1=t1, pct_embargo=pct_embargo)
    if sample_weight is None:
        sample_weight = pd.Series(1.0, index=X.index)
    sample_weight = pd.Series(np.asarray(sample_weight, dtype=float), index=X.index)

    scores = []
    for train, test in cv_gen.split(X=X):
        fit = clf.fit(
            X=X.iloc[train], y=y.iloc[train], sample_weight=sample_weight.iloc[train].to_numpy()
        )
        if scoring == "neg_log_loss":
            prob = fit.predict_proba(X.iloc[test])
            score = -log_loss(
                y.iloc[test], prob,
                sample_weight=sample_weight.iloc[test].to_numpy(),
                labels=fit.classes_,
            )
        else:
            pred = fit.predict(X.iloc[test])
            score = accuracy_score(
                y.iloc[test], pred, sample_weight=sample_weight.iloc[test].to_numpy()
            )
        scores.append(score)

    scores = np.array(scores)
    return (scores, cv_gen) if return_folds else scores


def label_overlap_audit(
    t1: pd.Series, n_splits: int = 5, pct_embargo: float = 0.01
) -> pd.DataFrame:
    """Count, exactly, how many training labels a naive k-fold contaminates.

    For each fold, a training observation is contaminated when its span
    ``[t, t1]`` *strictly* intersects the span covered by the test labels
    (touching endpoints share no return and are not leakage).  This is a
    deterministic property of the label geometry -- no model, no randomness,
    no score to interpret -- so it isolates *how much* leakage the protocol
    admits from the separate question of how well a given model exploits it.

    Under :class:`PurgedKFold` the same count is zero by construction, which
    the test suite asserts.  The contamination rate scales with the ratio of
    label span to fold span, so it is small for short holding periods and
    severe for the multi-day barriers that are common in practice.
    """
    X = pd.DataFrame(index=t1.index)
    n = len(t1)
    starts = t1.index.to_numpy()
    ends = t1.to_numpy()

    rows = []
    naive = KFold(n_splits=n_splits, shuffle=False)
    purged = PurgedKFold(n_splits=n_splits, t1=t1, pct_embargo=pct_embargo)
    for name, gen in (("kfold_no_purge", naive), (f"purged_embargo_{pct_embargo:g}", purged)):
        for k, (train, test) in enumerate(gen.split(X)):
            lo, hi = starts[test].min(), ends[test].max()
            overlap = (starts[train] < hi) & (ends[train] > lo)
            rows.append({
                "protocol": name, "fold": k,
                "n_train": len(train), "n_contaminated": int(overlap.sum()),
                "pct_contaminated": float(overlap.mean()),
            })
    out = pd.DataFrame(rows)
    return out.groupby("protocol").agg(
        n_train=("n_train", "mean"),
        n_contaminated=("n_contaminated", "sum"),
        pct_contaminated=("pct_contaminated", "mean"),
    )


def compare_cv(
    clf,
    X: pd.DataFrame,
    y: pd.Series,
    t1: pd.Series,
    sample_weight: pd.Series | None = None,
    scoring: str = "neg_log_loss",
    cv: int = 5,
    pct_embargo: float = 0.01,
) -> pd.DataFrame:
    """Score the same model under naive k-fold and under purged k-fold.

    The difference is a direct, quantitative measure of how much the naive
    protocol was leaking.  If it is large, every number produced under the
    naive protocol -- feature importances included -- is unreliable.
    """
    rows = {}
    naive = cv_score(clf, X, y, sample_weight, scoring, cv_gen=KFold(n_splits=cv, shuffle=False))
    rows["kfold_no_purge"] = naive
    purged_only = cv_score(clf, X, y, sample_weight, scoring, t1=t1, cv=cv, pct_embargo=0.0)
    rows["purged"] = purged_only
    purged_emb = cv_score(clf, X, y, sample_weight, scoring, t1=t1, cv=cv, pct_embargo=pct_embargo)
    rows[f"purged_embargo_{pct_embargo:g}"] = purged_emb

    return pd.DataFrame(
        {k: {"mean": v.mean(), "std": v.std(ddof=1), "min": v.min(), "max": v.max()}
         for k, v in rows.items()}
    ).T


# ----------------------------------------------------------------------------
# Walk-forward, for the numbers that are actually reported
# ----------------------------------------------------------------------------
class PurgedWalkForward:
    """Expanding-window walk-forward with purging and an embargo.

    Purged k-fold is the right tool for *model selection*: every observation
    gets to be out-of-sample, which is what makes feature importances and
    hyper-parameter comparisons stable.  It is the wrong tool for the final
    performance number, because folds after the first are trained partly on
    the future.  A walk-forward never is: the training set for a test block
    ends before that block begins, minus the labels whose barrier reaches
    into it, minus the embargo.

    Parameters
    ----------
    n_splits : int
        Number of sequential test blocks.
    t1 : Series
        Label end times, indexed by label start.
    pct_embargo : float
        Fraction of the whole sample dropped from the end of each training
        set, on top of the purge.
    min_train_frac : float
        Fraction of the sample reserved as the initial training window; the
        test blocks partition what is left.
    """

    def __init__(
        self,
        n_splits: int = 6,
        t1: pd.Series | None = None,
        pct_embargo: float = 0.01,
        min_train_frac: float = 0.4,
    ):
        if not isinstance(t1, pd.Series):
            raise ValueError("t1 must be a pandas Series indexed by label start")
        self.n_splits = n_splits
        self.t1 = t1
        self.pct_embargo = pct_embargo
        self.min_train_frac = min_train_frac

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def split(self, X, y=None, groups=None):
        if (X.index != self.t1.index).any():
            raise ValueError("X and t1 must share the same index, in the same order")
        n = X.shape[0]
        start0 = int(n * self.min_train_frac)
        mbrg = int(n * self.pct_embargo)
        blocks = np.array_split(np.arange(start0, n), self.n_splits)

        t1_values = self.t1.to_numpy()
        for block in blocks:
            if len(block) == 0:
                continue
            test_start = int(block[0])
            test_start_time = X.index[test_start]
            candidate = np.arange(0, max(test_start - mbrg, 0))
            # purge: a training label whose barrier reaches into the test block
            keep = t1_values[candidate] < test_start_time
            train = candidate[keep]
            if len(train) == 0:
                continue
            yield train, block


def oof_predict_proba(
    clf,
    X: pd.DataFrame,
    y: pd.Series,
    cv_gen,
    sample_weight: pd.Series | None = None,
) -> pd.DataFrame:
    """Out-of-fold predicted probabilities under any of the splitters above.

    Rows never assigned to a test fold (the initial training window of a
    walk-forward) are absent from the result rather than filled, so that a
    downstream backtest cannot silently trade on in-sample predictions.
    """
    if sample_weight is None:
        sample_weight = pd.Series(1.0, index=X.index)
    frames = []
    for train, test in cv_gen.split(X=X):
        fit = clf.fit(
            X=X.iloc[train], y=y.iloc[train],
            sample_weight=sample_weight.iloc[train].to_numpy(),
        )
        proba = pd.DataFrame(
            fit.predict_proba(X.iloc[test]), index=X.index[test], columns=fit.classes_
        )
        frames.append(proba)
    if not frames:
        return pd.DataFrame(index=X.index[:0])
    out = pd.concat(frames).sort_index()
    return out[~out.index.duplicated(keep="last")]
