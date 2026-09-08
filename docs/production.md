# Research vs. production

The starting rule: the research pipeline and the production pipeline are two different
systems that must share the feature code. Nearly every production bug in financial ML
is train/serve skew.

## Data (~40% of the real work)

- Clean consolidated ticks: cancelled trades, off-exchange prints, opening/closing
  auctions, halts. This is unglamorous and it's most of the time spent.
- **Point-in-time storage.** Adjusted prices are a trap: a split announced today
  rewrites the entire history retroactively. Store raw prices plus adjustment factors,
  each tagged with its announcement timestamp.
- The bar constructor must be **incremental and streaming**, not batch — and it must be
  the *same code* offline and online. Two implementations guarantee divergence.

## Features

- `tests/test_features.py` (recompute on a truncated history → identical values)
  becomes the research/production contract. In production, add its mirror: log the
  value computed live at `t`, recompute it offline at `t`, diff the two. A divergence is
  a bug, not a discretionary call.
- **State at restart.** A 250-bar rolling z-score means the system is wrong for 250 bars
  after every restart. Either persist state across restarts or refuse to trade until
  it's warm. This never shows up in a backtest.

## Model

- **Retraining cadence is set by the label horizon, not the calendar.** With a 12h
  vertical barrier, the freshest usable label is 12h old — you cannot retrain on bets
  whose barriers haven't closed yet.
- **Walk-forward in production is literally what `PurgedWalkForward` simulates** — the
  same purge, the same embargo, applied at the boundary of the last retrain.
- **Registry.** Every deployed model pinned to (code hash, data snapshot, feature set,
  training window). Without this, degradation is undiagnosable.

## Execution (~30%, and where the backtest lies the most)

The cost model here — half-spread plus commission on turnover — is a convenient
fiction. What it leaves out:

- **Queue position.** Sending a limit order doesn't guarantee a fill; *not* being filled
  when price moves against you is pure adverse selection.
- **Latency.** The backtest assumes execution at the dollar bar's close. In reality
  there's a delay between the bar closing and the order arriving.
- **Capacity.** The real question isn't "what Sharpe" but "at what size does the
  strategy's own impact eat the edge." The Kyle's lambda the pipeline already estimates
  gives the first-order answer: impact ≈ λ × participation. At 1,064 bets/year on a
  single asset, capacity is probably negligible.

The decision test that comes before everything else: gross 6.51 → net 2.73 at 3bp of
cost (see [results](results.md)). At 6bp, it's ~0. This strategy lives or dies on
execution quality, not on the model — and that's knowable in an hour, which is what
saves six months of engineering.

## Monitoring (~20%)

- **Pre-trade:** position limits, a drawdown kill switch, data-freshness checks.
- **Chapter 15 as a live monitor.** Track realised precision against the precision
  implied by the target Sharpe (here, 0.366 vs. 0.328). When realised falls durably
  below the threshold, the strategy is dead — a statistical stopping rule, not a
  discretionary call. This is the most underused idea in the book.
- **Drift.** Feature drift (PSI/KS) and label-distribution drift.
