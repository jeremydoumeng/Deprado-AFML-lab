# Deviations from the book, and why

The book's snippets are pseudocode written for exposition. Six places needed a decision:

**Imbalance bars have no interior fixed point.** The threshold is *linear* in `E₀[T]`
while a balanced-flow imbalance grows like `√T`, so a stretch of two-sided flow makes
bars longer, which raises the threshold, which makes them longer still — until one bar
swallows the sample. `min_ticks_per_bar` / `max_ticks_per_bar` clamp `E₀[T]`; the
sampler is untouched in the informative regime the bars exist to capture.

**Snippet 8.7 leaves the rows sorted by class.** `make_classification(shuffle=False)`
keeps the informative columns first — which is what the book wants — but it also leaves
the samples ordered by label. Combined with the contiguous test folds of `PurgedKFold`,
every fold then contains a single class: MDA collapses to zero and SFI ranks noise above
signal. The rows are shuffled explicitly here.

**Microstructure estimators belong on ticks, not on bars.** Roll's estimator inverts the
autocovariance the bid-ask bounce injects; aggregate to bar closes and the bounce is gone
while drift dominates, so the estimate floors at zero more than 40% of the time (asserted
in the tests). `features/tick_stats.py` keeps the estimators on the tape and puts only
the *aggregation* on the bar clock: each estimator is a ratio of sums over ticks, so
per-bar partial sums are sufficient statistics, and rolling those gives every window in
one pass over the ticks.

**The sequential bootstrap as written is O(n²·bars).** Snippet 4.5 recomputes every
candidate's average uniqueness after every draw, which is intractable past a few hundred
labels. But a triple-barrier label occupies a *contiguous* run of bars, so its average
uniqueness is a windowed mean of `1/(c+1)` and can be read off a prefix sum: O(bars) per
draw instead of O(n·bars). The book's loop is kept as `method="reference"` and the tests
assert the two agree exactly for a given seed.

**`min_ffd` compares different samples.** The fixed-width window grows quickly as `d`
falls — thousands of lags below `d=0.2` — so each grid point yields a series of a
different length. Comparing "memory retained" across them is comparing different samples,
and the column need not even be monotone in `d`. Here ADF uses each series in full (for
power) while the correlations are computed on the index they all share.

**Purged k-fold is for model selection, not for the headline.** Every fold after the
first is trained partly on the future. `PurgedWalkForward` — expanding window, purged at
the boundary, embargoed — produces the reported performance; `PurgedKFold` produces the
feature importances and the CV comparison, where using all the data out-of-sample is
what makes them stable.
