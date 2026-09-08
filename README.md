# afml-lab — a systematic trading pipeline after López de Prado

An implementation, from scratch, of the machine-learning workflow in Marcos López de
Prado's *Advances in Financial Machine Learning* (Wiley, 2018): information-driven
bars, triple-barrier labelling, meta-labelling, sample uniqueness, purged
cross-validation with an embargo, MDI/MDA/SFI feature importance, bet sizing, and a
walk-forward backtest closed out by a deflated Sharpe ratio.

Every stage is written against the book's own snippets, and every stage is tested
against something that can actually be checked — a hand-computed example, a closed-form
identity, or a simulator whose parameters are known.

The input is simulated tick data from a structural microstructure model whose
parameters are known, which is what makes the estimators gradeable rather than just
plausible-looking — see [docs/simulator.md](docs/simulator.md) for why and how.

---

## Results

Walk-forward, out-of-sample, net of a 3bp cost per unit of turnover:

| | |
|---|---|
| Net Sharpe (gross) | **2.73** (6.51) |
| CAGR / annualised vol | 9.2% / 3.3% |
| Max drawdown / longest underwater | 2.5% / 89 days |
| Bets per year / hit rate | 1,064 / 36.6% |
| **DSR** after 12 configurations tried | **0.9963** |

![report](results/report.png)

Full table, the deflated-Sharpe reasoning, and three findings worth reading off the run
(fractional differencing, purged-CV leakage audit, MDI/MDA/SFI disagreement) are in
[docs/results.md](docs/results.md).

---

## Layout

```
afml/       library code, one file/folder per chapter — see afml/README.md
scripts/    scripts/run_pipeline.py runs the ten stages end to end
tests/      104 tests — see tests/README.md
results/    artefacts from the last run (report.png, csvs, run.json)
docs/       simulator.md, results.md, deviations.md, production.md
```

## Quickstart

```bash
pip install -r requirements.txt

python scripts/run_pipeline.py --outdir results   # full run, ~12 min on 2 cores
python scripts/run_pipeline.py --quick            # 1M ticks, no sweep, ~2 min
python -m pytest -q                               # 104 tests, ~1 min
```

Artefacts land in `results/`: `report.png`, `performance.csv`, `feature_importance.csv`,
`overlap_audit.csv`, `cv_comparison.csv`, `fracdiff_adf.csv`, `equity_curve.csv`,
`bet_returns.csv`, and `run.json` (full configuration, trial Sharpes, runtime).

Using it as a library:

```python
from afml import bars, labeling as L, sampling as W
from afml.cv import PurgedKFold, cv_score
from afml.features import build_features

bars_df = bars.dollar_bars(ticks, threshold=1e6)
X       = build_features(bars_df, ticks, d_ffd=0.10)

vol      = L.get_vol(bars_df["close"], span=100, horizon=pd.Timedelta(hours=4))
t_events = L.cusum_filter(bars_df["close"], vol * 0.5)
t1       = L.add_vertical_barrier(t_events, bars_df["close"], pd.Timedelta(hours=12))
events   = L.get_events(bars_df["close"], t_events, (1., 1.), vol, t1=t1, side=side)
labels   = L.get_bins(events, bars_df["close"], min_ret=6e-4)

co = W.get_num_co_events(bars_df.index, events["t1"])
w  = W.sample_weights(events["t1"], co, bars_df["close"])

scores = cv_score(clf, X, labels["bin"], w, "neg_log_loss",
                  t1=events["t1"], cv=5, pct_embargo=0.01)
```

Module-by-module detail — the ten pipeline stages, why the primary model is a fixed rule
rather than a classifier, and six places the implementation deviates from the book's
pseudocode — is in [afml/README.md](afml/README.md).

---

## What this is not

- **Not a strategy.** The alpha is planted in the simulator. On real data the
  microstructure signal it imitates decays in minutes and competes with faster capital.
- **No CPCV or PBO.** Combinatorially purged CV (Ch. 12) and the probability of
  backtest overfitting are the natural next step; the deflated Sharpe here uses the
  variance of the twelve trial Sharpes actually run, which is the honest but weaker
  version.
- **A simple cost model.** Half-spread plus commission on turnover, with an optional
  square-root impact term that defaults off. No queue position, no partial fills, no
  borrow.
- **Single asset.** Chapter 16's allocation machinery (HRP) is not implemented.

What it would take to close each of these gaps for production — data, features, model,
execution, monitoring — is in [docs/production.md](docs/production.md).

## Tests

```
104 passed in 59s
```

Written to fail if the implementation is wrong, not to confirm that it runs — details
and what each test actually asserts are in [tests/README.md](tests/README.md).

## Reference

Marcos López de Prado, *Advances in Financial Machine Learning*, Wiley, 2018.
Chapters 2–5, 7, 8, 10, 14, 15, 19, 20.
