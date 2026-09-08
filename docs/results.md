# Results

Walk-forward, out-of-sample, net of a 3bp cost per unit of turnover, on ~1.5 years of
held-out tape (`python scripts/run_pipeline.py`). Background on why these numbers are a
property of the simulator, not a market claim, is in [simulator.md](simulator.md).

| | |
|---|---|
| Net Sharpe (gross) | **2.73** (6.51) |
| CAGR / annualised vol | 9.2% / 3.3% |
| Max drawdown / longest underwater | 2.5% / 89 days |
| Bets per year / hit rate | 1,064 / 36.6% |
| PSR, `P[SR > 0]` | 0.9997 |
| **DSR** after 12 configurations tried | **0.9963** |
| Realised precision vs. precision implied by SR=1 | 0.366 vs 0.328 |

A 36.6% hit rate with a positive Sharpe is the triple barrier working as designed:
profit-taking barriers are hit less often than stops but the vertical barrier converts
many would-be losses into small ones, so the payoff is asymmetric. The deflated Sharpe
is the honest headline — the plain Sharpe is the best of twelve configurations, and
`E[max SR]` under the null for twelve trials with `sd(SR)=0.35` is 0.58.

![report](../results/report.png)

## Three findings worth reading off the run

**Fractional differencing gets stationarity almost for free.** The minimum `d` passing
an ADF test at 5% is **0.10**, and that series still has a **0.99** correlation with the
log price. Plain returns (`d=1`) retain 0.006. The memory that momentum and
mean-reversion signals live on is thrown away by first differencing for no statistical
gain (Ch. 5).

**Contamination scales with the label span, and purging removes all of it.** A
deterministic audit — no model, no randomness — counts how many training labels reach
into the test fold under naive 5-fold:

| label span | naive 5-fold | purged + 1% embargo |
|---:|---:|---:|
| 12h | 61 (0.12%) | 0 |
| 48h | 219 (0.43%) | 0 |
| 120h | 522 (1.02%) | 0 |
| 480h | 2,104 (4.19%) | 0 |
| 1440h | 6,312 (13.13%) | 0 |

The rate is roughly `1.6 × (label span / fold span)` — 1.6 fold boundaries per fold on
average across five folds. For the 12-hour barriers traded here it is negligible, and
the measured score gap between naive and purged k-fold is correspondingly ~0.0000
accuracy. For the multi-week barriers a daily strategy uses it is 4–13% of the training
set. Reporting the audit rather than assuming a leak is the point: the contamination
rate is a property of the label geometry alone, while whether a *model* converts it into
optimism also depends on that model's capacity to memorise.

**The three importance measures disagree in the way the book predicts.** On planted
ground truth (`make_test_data`: 4 informative, 4 redundant copies, 12 noise), MDI and MDA
split credit between the informative columns and their substitutes, while SFI — which
fits one feature at a time — ranks a redundant copy as highly as the original. All three
give the noise columns nothing. These are assertions in `tests/test_importance.py`, not
illustrations.
