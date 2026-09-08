#!/usr/bin/env python3
"""End-to-end run of the AFML pipeline, from raw ticks to a deflated Sharpe.

    python scripts/run_pipeline.py --outdir results
    python scripts/run_pipeline.py --quick          # 250k ticks, no sweep

Stages, in the order the book prescribes:

  1  Ticks           simulated tape with known microstructure (Ch. 2)
  2  Bars            dollar bars (Ch. 2)
  3  Stationarity    minimum FFD order passing ADF (Ch. 5)
  4  Features        momentum, order flow, tick-level microstructure (Ch. 19)
  5  Events          CUSUM sampling + triple barrier + meta-labels (Ch. 3)
  6  Weights         concurrency, uniqueness, return attribution (Ch. 4)
  7  Leakage         naive k-fold vs purged vs purged+embargo (Ch. 7)
  8  Importance      MDI / MDA / SFI on purged folds (Ch. 8)
  9  Backtest        walk-forward, bet sizing, costs (Ch. 10, 14)
 10  Risk            PSR, deflated Sharpe, implied precision (Ch. 14, 15)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier

from afml import bars as B
from afml import labeling as L
from afml import sampling as W
from afml import stats as ST
from afml.backtest import CostModel, run_backtest
from afml.bet_sizing import get_signal
from afml.cv import (
    PurgedWalkForward,
    compare_cv,
    label_overlap_audit,
    oof_predict_proba,
)
from afml.data import TickSimConfig, simulate_ticks
from afml.features import build_features
from afml.fracdiff import min_ffd
from afml.importance import feature_importance_report

SEP = "=" * 78


def section(title: str) -> None:
    print(f"\n{SEP}\n{title}\n{SEP}")


def make_classifier(seed: int = 0, n_estimators: int = 300, n_jobs: int = 2):
    """The book's recommended forest (Sections 6.3 and 8.3.1).

    ``max_features=1`` so that substitution effects do not mask features in
    MDI; ``min_weight_fraction_leaf`` regularises leaves, which matters here
    because sample weights are very uneven; ``class_weight`` rebalances the
    meta-label classes; bootstrap sampling is left on but kept small so that
    overlapping labels are not repeatedly redrawn into the same tree.
    """
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_features=1,
        criterion="entropy",
        min_weight_fraction_leaf=0.05,
        class_weight="balanced_subsample",
        max_samples=0.6,
        random_state=seed,
        n_jobs=n_jobs,
    )


# ----------------------------------------------------------------------------
def build_dataset(cfg: dict, ticks: pd.DataFrame, bars_df: pd.DataFrame, d_ffd: float):
    """Features, triple-barrier meta-labels and sample weights for one config."""
    close = bars_df["close"]

    X = build_features(bars_df, ticks, d_ffd=d_ffd)

    vol = L.get_vol(close, span=100, horizon=pd.Timedelta(hours=cfg["vol_horizon_h"]))
    t_events = L.cusum_filter(close, vol * cfg["cusum_mult"])
    t_events = t_events.intersection(X.index)
    t1 = L.add_vertical_barrier(t_events, close, pd.Timedelta(hours=cfg["holding_h"]))

    # --- primary model: an explicit, transparent rule -----------------------
    # Meta-labeling only makes sense on top of a model that already picks a
    # side.  A rule keeps the two stages cleanly separable: the rule owns
    # recall, the classifier owns precision.
    side = np.sign(X["ofi_5"].reindex(t_events)).replace(0.0, 1.0)

    events = L.get_events(
        close, t_events, (cfg["pt"], cfg["sl"]), vol,
        min_ret=cfg["min_ret"], t1=t1, side=side,
    )
    labels = L.get_bins(events, close, min_ret=cfg["round_trip_cost"])
    labels = L.drop_labels(labels, min_pct=0.02)

    idx = labels.index.intersection(X.index)
    labels = labels.loc[idx]
    events = events.loc[idx]
    X = X.loc[idx]
    y = labels["bin"].astype(int)

    # --- Chapter 4 weights --------------------------------------------------
    co = W.get_num_co_events(close.index, events["t1"])
    tw = W.get_avg_uniqueness_series(events["t1"], co)
    w_ret = W.sample_weights(events["t1"], co, close)
    decay = W.get_time_decay(tw, last_w=cfg["time_decay_last_w"])
    sample_weight = (w_ret * decay).reindex(idx)
    sample_weight = sample_weight * (len(sample_weight) / sample_weight.sum())

    return X, y, events, labels, sample_weight, tw, co


def run_strategy(
    cfg: dict, ticks, bars_df, d_ffd, seed: int = 0, n_estimators: int = 300, verbose: bool = True
):
    """One full configuration: fit walk-forward, size the bets, backtest."""
    close = bars_df["close"]
    X, y, events, labels, sw, tw, co = build_dataset(cfg, ticks, bars_df, d_ffd)

    clf = make_classifier(seed, n_estimators)
    wf = PurgedWalkForward(
        n_splits=cfg["wf_splits"], t1=events["t1"],
        pct_embargo=cfg["embargo"], min_train_frac=cfg["min_train_frac"],
    )
    proba = oof_predict_proba(clf, X, y, wf, sw)
    if proba.empty:
        raise RuntimeError("walk-forward produced no out-of-sample predictions")

    p_take = proba[1] if 1 in proba.columns else proba.iloc[:, -1]
    position = get_signal(
        events.loc[p_take.index], p_take, num_classes=2, step_size=cfg["bet_step"]
    )
    result = run_backtest(
        position, close.loc[close.index >= p_take.index[0]],
        costs=CostModel(cfg["half_spread"], cfg["commission"]),
        target_sr=1.0,
    )
    return {
        "X": X, "y": y, "events": events, "labels": labels, "sample_weight": sw,
        "uniqueness": tw, "co_events": co, "proba": proba, "position": position,
        "result": result, "clf": clf, "wf": wf,
    }


# ----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--quick", action="store_true", help="smaller tape, no configuration sweep")
    ap.add_argument("--n-ticks", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-sweep", action="store_true")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    cfg = {
        "bars_per_day": 50,
        "vol_horizon_h": 4,
        "cusum_mult": 0.5,
        "holding_h": 12,
        "pt": 1.0,
        "sl": 1.0,
        "min_ret": 0.0,
        "half_spread": 2.5e-4,
        "commission": 0.5e-4,
        "bet_step": 0.05,
        "wf_splits": 6,
        "embargo": 0.01,
        "min_train_frac": 0.40,
        "time_decay_last_w": 0.5,
    }
    cfg["round_trip_cost"] = 2 * (cfg["half_spread"] + cfg["commission"])

    # -- 1. ticks ------------------------------------------------------------
    section("1. TICKS  (Ch. 2)")
    sim_cfg = TickSimConfig(seed=20211231)
    if args.quick:
        sim_cfg.n_ticks = 1_000_000
    if args.n_ticks:
        sim_cfg.n_ticks = args.n_ticks
    ticks, truth = simulate_ticks(sim_cfg)
    days = (ticks.index[-1] - ticks.index[0]).total_seconds() / 86400
    print(f"  {len(ticks):,} ticks over {days:.0f} days "
          f"({ticks.index[0]:%Y-%m-%d} -> {ticks.index[-1]:%Y-%m-%d})")
    print(f"  ground truth: spread={truth['spread']:.2e}  kyle_lambda={truth['lam']:.2e}  "
          f"sign autocorr={2 * truth['c']:.2f}")

    # -- 2. bars -------------------------------------------------------------
    section("2. BARS  (Ch. 2)")
    thr = ticks["dollar"].sum() / (days * cfg["bars_per_day"])
    bars_df = B.dollar_bars(ticks, thr)
    log_ret_dollar = np.log(bars_df["close"]).diff().dropna()
    time_bar = np.log(ticks["price"]).resample("1h").last().dropna().diff().dropna()
    print(f"  dollar bars: {len(bars_df):,}  (threshold ${thr:,.0f}, "
          f"{len(bars_df) / days:.1f}/day, median {bars_df['n_ticks'].median():.0f} ticks)")
    print(f"  excess kurtosis of returns -- dollar bars {log_ret_dollar.kurt():+.2f} "
          f"vs hourly time bars {time_bar.kurt():+.2f}   (lower is closer to IID normal)")

    # -- 3. stationarity -----------------------------------------------------
    section("3. FRACTIONAL DIFFERENTIATION  (Ch. 5)")
    d_star, ffd_table = min_ffd(np.log(bars_df["close"]))
    row = ffd_table.loc[d_star]
    print(f"  minimum d passing ADF at 5%: d* = {d_star:.2f}")
    print(f"  ADF {row['adf_stat']:.2f} (crit {row['crit_95']:.2f}, p={row['p_value']:.4f}), "
          f"correlation with the original log price = {row['corr_with_original']:.3f}")
    print(f"  for reference, d=1 (plain returns) keeps "
          f"{ffd_table.loc[1.0, 'corr_with_original']:.3f} of that correlation"
          if 1.0 in ffd_table.index else "")
    ffd_table.to_csv(outdir / "fracdiff_adf.csv")

    # -- 4-6. dataset --------------------------------------------------------
    section("4-6. FEATURES, TRIPLE BARRIER, SAMPLE WEIGHTS  (Ch. 19, 3, 4)")
    art = run_strategy(cfg, ticks, bars_df, d_star, seed=args.seed,
                       n_estimators=120 if args.quick else 250)
    X, y, events, labels, sw = art["X"], art["y"], art["events"], art["labels"], art["sample_weight"]
    print(f"  features: {X.shape[1]} columns x {X.shape[0]:,} labelled events "
          f"({X.shape[0] / days:.1f}/day)")
    hold_h = (labels["t1"] - labels.index).dt.total_seconds() / 3600
    vert = (hold_h > cfg["holding_h"] * 0.98).mean()
    print(f"  barriers touched: {1 - vert:.0%} horizontal, {vert:.0%} vertical; "
          f"median holding {hold_h.median():.2f}h")
    print(f"  meta-labels: {y.mean():.1%} positive "
          f"(a bet is positive only if it beats {cfg['round_trip_cost'] * 1e4:.0f}bp round-trip cost)")
    print(f"  mean concurrency {art['co_events'][art['co_events'] > 0].mean():.2f} labels open, "
          f"mean uniqueness {art['uniqueness'].mean():.3f}")
    print(f"  sample weights: mean {sw.mean():.2f}, "
          f"p5 {sw.quantile(0.05):.3f}, p95 {sw.quantile(0.95):.2f}")

    micro = pd.Series({
        "roll_spread_100 (median)": X["roll_spread_100"].median(),
        "  theory  s*(1-rho)": truth["spread"] * (1 - 2 * truth["c"]),
        "  true spread s": truth["spread"],
        "kyle_lambda_100 (median)": X["kyle_lambda_100"].median(),
        "vpin_100 (median)": X["vpin_100"].median(),
    })
    print("\n  microstructure estimates vs the simulator's ground truth:")
    print(micro.to_string(float_format=lambda v: f"{v:.3e}").replace("\n", "\n    "))

    # -- 7. leakage ----------------------------------------------------------
    section("7. CROSS-VALIDATION: WHAT DOES NAIVE K-FOLD LET THROUGH?  (Ch. 7)")
    clf = make_classifier(args.seed, 100 if args.quick else 200)
    purged_row = f"purged_embargo_{cfg['embargo']:g}"

    print("  (a) deterministic audit -- training labels whose span reaches into the test fold\n")
    print(f"      {'label span':>12}  {'naive 5-fold':>22}  {'purged+embargo':>16}")
    curve = []
    for hours in (cfg["holding_h"], 48, 120, 480, 1440):
        t1_h = L.add_vertical_barrier(
            X.index, bars_df["close"], pd.Timedelta(hours=hours)
        )
        a_tbl = label_overlap_audit(t1_h, n_splits=5, pct_embargo=cfg["embargo"])
        naive_n = a_tbl.loc["kfold_no_purge", "n_contaminated"]
        naive_p = a_tbl.loc["kfold_no_purge", "pct_contaminated"]
        purged_n = a_tbl.loc[purged_row, "n_contaminated"]
        curve.append({"span_hours": hours, "n_contaminated": naive_n, "pct": naive_p})
        print(f"      {hours:>9}h    {naive_n:>10.0f} labels ({naive_p:>6.2%})  "
              f"{purged_n:>10.0f} labels")
    curve = pd.DataFrame(curve).set_index("span_hours")
    curve.to_csv(outdir / "overlap_audit.csv")
    print("\n      Contamination is ~ 1.6 x (label span / fold span) -- 1.6 boundaries per fold on")
    print("      average over five folds.  It is negligible for the intraday barriers traded")
    print("      here and severe for the multi-week barriers a daily strategy uses; purging")
    print("      drives it to exactly zero either way, which is the only reason a purged score")
    print("      can be compared across label geometries at all.")

    print("\n  (b) what the contamination costs in score, on the labels actually traded\n")
    nll = compare_cv(clf, X, y, events["t1"], sw, scoring="neg_log_loss",
                     cv=5, pct_embargo=cfg["embargo"])
    acc = compare_cv(clf, X, y, events["t1"], sw, scoring="accuracy",
                     cv=5, pct_embargo=cfg["embargo"])
    tbl = nll[["mean", "std"]].join(acc[["mean", "std"]], rsuffix="_acc")
    tbl.columns = ["neg_log_loss", "nll_std", "accuracy", "acc_std"]
    print("      " + tbl.round(4).to_string().replace("\n", "\n      "))
    leak_acc = tbl.loc["kfold_no_purge", "accuracy"] - tbl.loc[purged_row, "accuracy"]
    leak_nll = tbl.loc["kfold_no_purge", "neg_log_loss"] - tbl.loc[purged_row, "neg_log_loss"]
    print(f"\n      optimism of the naive protocol: {leak_acc:+.4f} accuracy, "
          f"{leak_nll:+.4f} log loss")
    print("      The score gap is the product of the contamination rate above and the model's")
    print("      capacity to memorise shared outcomes.  A weight-regularised forest on 12-hour")
    print("      labels leaks almost nothing -- which is a fact to be measured, as here, and")
    print("      never one to be assumed.")
    tbl.to_csv(outdir / "cv_comparison.csv")

    # -- 8. importance -------------------------------------------------------
    section("8. FEATURE IMPORTANCE  (Ch. 8)")
    imp_clf = make_classifier(args.seed, 80 if args.quick else 150)
    report = feature_importance_report(
        imp_clf, X, y, sample_weight=sw, t1=events["t1"],
        cv=4, pct_embargo=cfg["embargo"], scoring="neg_log_loss",
    )
    print(f"  purged-CV baseline neg-log-loss: {report.attrs['mda_baseline']:.4f}")
    print(report.head(12).round(4).to_string())
    print("\n  bottom 5 by MDA:")
    print(report.tail(5).round(4).to_string())
    report.to_csv(outdir / "feature_importance.csv")

    # -- 9. backtest ---------------------------------------------------------
    section("9. WALK-FORWARD BACKTEST  (Ch. 10, 14)")
    res = art["result"]
    print(f"  {cfg['wf_splits']} expanding-window folds, {cfg['min_train_frac']:.0%} initial train, "
          f"embargo {cfg['embargo']:.0%}; positions on a {cfg['bet_step']:g} grid")
    print(f"  costs: {cfg['half_spread'] * 1e4:.1f}bp half-spread + "
          f"{cfg['commission'] * 1e4:.1f}bp commission per unit of turnover")
    keys = ["years", "total_return", "cagr", "ann_vol", "gross_sharpe", "sharpe",
            "max_drawdown", "max_time_under_water_days", "turnover_per_year",
            "avg_abs_position", "n_bets", "bets_per_year", "hit_rate"]
    print(res.summary[keys].to_string(float_format=lambda v: f"{v:,.4f}"))

    # -- 10. risk ------------------------------------------------------------
    section("10. IS IT REAL?  (Ch. 14, 15)")
    trial_sr: list[float] = []
    if not (args.quick or args.no_sweep):
        grid = [
            {"holding_h": h, "pt": p, "sl": sl, "bet_step": b}
            for h in (8, 12, 24) for (p, sl) in ((1.0, 1.0), (1.5, 1.0)) for b in (0.05, 0.20)
        ]
        print(f"  sweeping {len(grid)} configurations to measure the variance of trial Sharpes")
        for i, g in enumerate(grid, 1):
            c = dict(cfg, **g)
            try:
                r = run_strategy(c, ticks, bars_df, d_star, seed=args.seed, n_estimators=100)
                sr = float(r["result"].summary["sharpe"])
            except Exception as exc:  # a configuration can label everything one class
                print(f"    [{i:2d}/{len(grid)}] {g} -> skipped ({type(exc).__name__})")
                continue
            trial_sr.append(sr)
            print(f"    [{i:2d}/{len(grid)}] holding={g['holding_h']:>2}h pt/sl={g['pt']}/{g['sl']} "
                  f"step={g['bet_step']:.2f} -> Sharpe {sr:+.3f}")
    n_trials = max(len(trial_sr), 1)
    var_sr = float(np.var(trial_sr, ddof=1)) if len(trial_sr) > 1 else 0.0

    net = res.returns
    ppy = len(res.returns) / res.summary["years"]
    psr = ST.probabilistic_sharpe_ratio(net, 0.0, ppy)
    emax = ST.expected_max_sharpe(n_trials, var_sr) if n_trials > 1 else 0.0
    dsr = ST.deflated_sharpe_ratio(net, n_trials, var_sr, ppy) if n_trials > 1 else psr
    risk, thres, prec = ST.prob_failure(res.bet_returns, res.summary["bets_per_year"], 1.0)

    print(f"\n  observed Sharpe                     {res.summary['sharpe']:+.3f}")
    print(f"  PSR    P[true SR > 0]               {psr:.4f}")
    if n_trials > 1:
        print(f"  trials run                          {n_trials}  (sd of trial Sharpes {np.sqrt(var_sr):.3f})")
        print(f"  expected max Sharpe of that many    {emax:+.3f}")
        print(f"  DSR    P[true SR > that maximum]    {dsr:.4f}   "
              f"{'-> survives the multiple-testing correction' if dsr > 0.95 else '-> NOT distinguishable from selection bias'}")
    print(f"\n  realised precision                  {prec:.4f}")
    print(f"  precision implied by SR=1            {thres:.4f}")
    print(f"  P[precision falls below it]          {risk:.4f}")
    print(f"  return concentration  HHI(+) {res.summary['hhi_positive']:.3f}  "
          f"HHI(-) {res.summary['hhi_negative']:.3f}  HHI(monthly) {res.summary['hhi_monthly']:.3f}")

    # -- outputs -------------------------------------------------------------
    res.summary.to_csv(outdir / "performance.csv")
    res.equity.rename("equity").to_frame().join(res.position.rename("position")).to_csv(
        outdir / "equity_curve.csv"
    )
    meta = {
        "config": cfg, "sim": {k: v for k, v in truth.items() if not isinstance(v, pd.Series)},
        "d_star": d_star, "n_bars": len(bars_df), "n_events": int(len(X)),
        "trial_sharpes": trial_sr, "n_trials": n_trials, "var_trials_sr": var_sr,
        "psr": psr, "dsr": dsr, "sharpe": float(res.summary["sharpe"]),
        "runtime_s": round(time.time() - t_start, 1),
    }
    (outdir / "run.json").write_text(json.dumps(meta, indent=2, default=str))

    res.bet_returns.to_csv(outdir / "bet_returns.csv")
    if not args.no_plot:
        _plot(outdir, res, report, ffd_table, trial_sr, d_star, emax)

    section(f"DONE in {time.time() - t_start:.0f}s -- artefacts in {outdir}/")
    for f in sorted(outdir.iterdir()):
        print(f"  {f.name}")


def _plot(outdir: Path, res, report, ffd_table, trial_sr, d_star, emax) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    ink, warm, teal, sand = "#1b4965", "#bc4749", "#2a9d8f", "#e9c46a"
    fig, ax = plt.subplots(2, 3, figsize=(16.5, 9))
    fig.suptitle(
        "AFML pipeline on a simulated tape -- walk-forward, out-of-sample, net of costs",
        fontsize=13.5,
    )

    def _dates(a):
        a.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=5))
        a.xaxis.set_major_formatter(mdates.ConciseDateFormatter(a.xaxis.get_major_locator()))
        a.grid(alpha=0.3)

    # 1. equity, gross vs net -- the cost drag made visible
    gross_eq = (1.0 + res.gross_returns).cumprod()
    ax[0, 0].plot(gross_eq.index, gross_eq.values, lw=1.0, color=sand, label="gross")
    ax[0, 0].plot(res.equity.index, res.equity.values, lw=1.4, color=ink, label="net")
    ax[0, 0].set_title(
        f"Equity   net Sharpe {res.summary['sharpe']:.2f}   "
        f"(gross {res.summary['gross_sharpe']:.2f})"
    )
    ax[0, 0].set_ylabel("growth of 1")
    ax[0, 0].legend(frameon=False, fontsize=9)
    _dates(ax[0, 0])

    # 2. drawdown
    dd = res.equity / res.equity.cummax() - 1.0
    ax[0, 1].fill_between(dd.index, dd.values * 100, 0, color=warm, alpha=0.75, lw=0)
    ax[0, 1].set_title(
        f"Drawdown   max {res.summary['max_drawdown']:.2%}   "
        f"longest {res.summary['max_time_under_water_days']:.0f}d"
    )
    ax[0, 1].set_ylabel("%")
    _dates(ax[0, 1])

    # 3. the multiple-testing correction, drawn
    if trial_sr:
        order = np.argsort(trial_sr)[::-1]
        vals = np.asarray(trial_sr)[order]
        ax[0, 2].bar(range(len(vals)), vals, color=teal, alpha=0.85)
        ax[0, 2].axhline(emax, color=warm, ls="--", lw=1.4,
                         label=f"E[max SR] of {len(vals)} trials = {emax:.2f}")
        ax[0, 2].axhline(res.summary["sharpe"], color=ink, lw=1.4,
                         label=f"reported SR = {res.summary['sharpe']:.2f}")
        ax[0, 2].set_title("Ch. 14: is the Sharpe just the best of N tries?")
        ax[0, 2].set_xlabel("configuration, ranked")
        ax[0, 2].set_ylabel("Sharpe")
        ax[0, 2].legend(frameon=False, fontsize=8, loc="lower left")
    else:
        ax[0, 2].axis("off")
    ax[0, 2].grid(alpha=0.3, axis="y")

    # 4. MDA importance
    top = report["mda"].head(12).iloc[::-1] * 1e3
    err = report["mda_std"].reindex(top.index) * 1e3
    ax[1, 0].barh(top.index, top.values, xerr=err.values, color=teal,
                  error_kw={"lw": 0.8, "ecolor": "#33333366"})
    ax[1, 0].set_title("Ch. 8: MDA importance on purged folds")
    ax[1, 0].set_xlabel("mean decrease in log-loss score (x1000)")
    ax[1, 0].tick_params(labelsize=8)
    ax[1, 0].grid(alpha=0.3, axis="x")

    # 5. memory vs stationarity
    a5 = ax[1, 1]
    stationary = ffd_table["p_value"] < 0.05
    a5.plot(ffd_table.index, ffd_table["corr_with_original"], color=ink, lw=1.6)
    a5.fill_between(ffd_table.index, 0, 1, where=stationary, color=teal, alpha=0.15,
                    transform=a5.get_xaxis_transform(), label="ADF rejects a unit root")
    if np.isfinite(d_star):
        a5.axvline(d_star, color=warm, ls="--", lw=1.4)
        keep = float(ffd_table.loc[d_star, "corr_with_original"])
        a5.annotate(f"d* = {d_star:.2f}\n{keep:.0%} of the memory kept",
                    xy=(d_star, keep), xytext=(d_star + 0.12, min(keep + 0.05, 0.95)),
                    fontsize=9, color=warm,
                    arrowprops={"arrowstyle": "->", "color": warm, "lw": 1.0})
    a5.set_xlabel("fractional differencing order d")
    a5.set_ylabel("correlation with the log price")
    a5.set_ylim(0, 1.05)
    a5.set_title("Ch. 5: stationarity bought with as little memory as possible")
    a5.legend(frameon=False, fontsize=8, loc="lower left")
    a5.grid(alpha=0.3)

    # 6. bet returns
    br = res.bet_returns.to_numpy() * 100
    ax[1, 2].hist(br, bins=60, color="#457b9d")
    ax[1, 2].axvline(0, color="k", lw=0.8)
    ax[1, 2].axvline(br.mean(), color=warm, lw=1.4, ls="--",
                     label=f"mean {br.mean():+.3f}%")
    ax[1, 2].set_title(
        f"Bet returns   hit rate {res.summary['hit_rate']:.1%}   "
        f"{res.summary['bets_per_year']:.0f}/yr"
    )
    ax[1, 2].set_xlabel("% per bet, net of costs")
    ax[1, 2].legend(frameon=False, fontsize=9)
    ax[1, 2].grid(alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(outdir / "report.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
