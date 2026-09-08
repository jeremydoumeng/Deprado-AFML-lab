# Tests

```
104 passed in 59s
```

They are written to fail if the implementation is wrong, not to confirm that it runs:

- `test_features.py` recomputes the whole feature matrix on a truncated history and
  requires bit-identical values at shared timestamps. Any centred window, full-sample
  scaling, or stray `shift(-k)` anywhere in the chain breaks it. This is also the
  research/production contract described in
  [../docs/production.md](../docs/production.md).
- `test_backtest.py` checks that a perfect-foresight signal scores above Sharpe 20 while
  the same signal shifted five bars into the past scores under 5 — if both look good, the
  engine is peeking.
- `test_sampling.py` reproduces the indicator matrix and the average uniqueness
  `{5/6, 3/4, 1}` of the worked example in Snippets 4.3–4.4.
- `test_stats.py` reproduces the book's implied precision of 0.5316 for symmetric 1%
  payouts at 250 bets a year targeting Sharpe 1.
- `test_microstructure.py` grades each estimator against the simulator's parameters,
  including the biases: Roll under order-flow autocorrelation, and Kyle's and Hasbrouck's
  lambdas contaminated by the spread when run on transaction rather than midquote prices.
- `test_cv.py` asserts that purging leaves *exactly zero* contaminated training labels
  where naive k-fold leaves more than 5%.

Run them with:

```bash
python -m pytest -q
```
