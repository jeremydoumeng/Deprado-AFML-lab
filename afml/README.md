# Package layout

```
afml/
  data/synthetic.py    Ch. 2   tick simulator with known microstructure truth
  bars.py              Ch. 2   tick/volume/dollar bars, imbalance bars, run bars
  fracdiff.py          Ch. 5   FFD weights, expanding and fixed-width, min-d search
  features/            Ch. 19  feature matrix; Roll, Corwin-Schultz, Kyle,
                                Amihud, Hasbrouck, VPIN, order-flow imbalance
  labeling.py          Ch. 3   CUSUM filter, triple barrier, meta-labelling
  sampling.py          Ch. 4   concurrency, uniqueness, sequential bootstrap, weights
  cv.py                Ch. 7   purged k-fold, embargo, walk-forward, overlap audit
  importance.py        Ch. 8   MDI, MDA, SFI, orthogonal features, planted-truth data
  bet_sizing.py        Ch. 10  probability -> size -> averaged over active bets
  backtest.py          Ch. 14  event-driven backtest with a cost model
  stats.py         Ch. 14, 15  PSR, DSR, drawdown, HHI, implied precision, prob. failure
  multiprocess.py      Ch. 20  the job engine every heavy loop runs on
```

See `../scripts/run_pipeline.py` for the ten stages wired end to end, and
`../tests/README.md` for how each module is graded.

## The ten stages

1. **Ticks.** Simulated tape; ground-truth spread, impact and order-flow persistence retained for the tests.
2. **Bars.** Dollar bars at ~50/day. Their returns have lower excess kurtosis (+0.46) than hourly time bars (+0.77), which is the property §2.4 claims for activity-based sampling.
3. **Stationarity.** `min_ffd` sweeps `d`, runs ADF on each fixed-width series and returns the smallest order that rejects a unit root.
4. **Features.** 29 columns: fractionally differenced price level, risk-adjusted momentum, order-flow imbalance at three horizons, bar-clock activity, and the Chapter 19 estimators.
5. **Events.** CUSUM sampling on a volatility-scaled threshold, then the triple barrier: 87% of events end on a horizontal barrier, 13% on the vertical one, median holding 4.4 hours.
6. **Weights.** Concurrency (mean 3.3 labels open at once), average uniqueness (0.31), return attribution, and time decay, multiplied into one weight per observation and normalised to mean 1.
7. **Leakage.** The overlap audit and the naive-vs-purged score comparison (see [../docs/results.md](../docs/results.md)).
8. **Importance.** MDI, MDA and SFI on purged folds, ranked by MDA.
9. **Backtest.** Six expanding walk-forward folds, positions from the meta-model's probability on a 0.05 grid, costs charged on turnover.
10. **Risk.** PSR, a 12-configuration sweep, the deflated Sharpe against `E[max SR]` from that sweep, and the Chapter 15 implied-precision check.

## The primary model is a rule, on purpose

Meta-labelling needs something that already picks a side. Here that is an explicit rule —
`side = sign(5-bar order-flow imbalance)` — not a second classifier. Two reasons: it keeps
the division of labour legible (the rule owns recall, the classifier owns precision, and
the classifier can size a bet to zero but never reverse it), and it avoids the trap of a
primary model whose in-sample predictions become the secondary model's features.

## Deviations from the book

Six places where the book's pseudocode needed a real decision — imbalance-bar bounds,
row shuffling before purged CV, keeping microstructure estimators on ticks rather than
bars, an O(bars) sequential bootstrap, comparing `min_ffd` series of different lengths,
and what purged k-fold is (and isn't) used for — are written up in
[../docs/deviations.md](../docs/deviations.md).
