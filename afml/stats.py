"""Backtest statistics and strategy risk  --  AFML Chapters 14 and 15.

A Sharpe ratio computed from a backtest is not an estimate of future
performance; it is the maximum of however many configurations were tried,
and the maximum of ``N`` noisy draws grows like ``sqrt(2 log N)`` even when
every draw has zero expected return.  The statistics here are the ones that
make that inflation visible:

* ``probabilistic_sharpe_ratio`` -- the probability that the true Sharpe
  exceeds a benchmark, correcting for track-record length, skewness and
  kurtosis.  Negative skew and fat tails, which is what most strategies have,
  make a given Sharpe much less significant than the Gaussian formula says.
* ``deflated_sharpe_ratio`` -- the same, with the benchmark raised to the
  Sharpe one would expect from the *best* of ``n_trials`` independent
  attempts.  This is the number to quote after a hyper-parameter search.
* ``drawdown_time_under_water``, ``bet_holding_period``, ``hhi`` -- run-level
  diagnostics: a strategy whose profit is concentrated in a handful of bets,
  or in one month, is not the strategy the average return suggests.

Chapter 15 adds the inverse question.  ``implied_precision`` returns the hit
rate a strategy *must* achieve to reach a target Sharpe given its payouts,
and ``prob_failure`` the probability that the realised precision falls short
of it -- a far better sanity check than any in-sample metric.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

__all__ = [
    "sharpe_ratio",
    "probabilistic_sharpe_ratio",
    "expected_max_sharpe",
    "deflated_sharpe_ratio",
    "drawdown_time_under_water",
    "bet_timing",
    "bet_holding_period",
    "hhi",
    "implied_precision",
    "prob_failure",
    "performance_summary",
]


# ----------------------------------------------------------------------------
# Sharpe and its significance
# ----------------------------------------------------------------------------
def sharpe_ratio(returns: pd.Series, periods_per_year: float, rf: float = 0.0) -> float:
    """Annualised Sharpe of a series of per-period returns."""
    r = returns.dropna()
    if r.std(ddof=1) == 0 or len(r) < 2:
        return float("nan")
    return float((r.mean() - rf / periods_per_year) / r.std(ddof=1) * np.sqrt(periods_per_year))


def probabilistic_sharpe_ratio(
    returns: pd.Series, benchmark_sr: float = 0.0, periods_per_year: float = 252.0
) -> float:
    """PSR: ``P[true SR > benchmark]``, adjusted for skew and kurtosis.

    Bailey & Lopez de Prado (2012).  Both the observed Sharpe and the
    benchmark are handled in *per-period* units internally, so the annualised
    inputs are deannualised first.
    """
    r = returns.dropna()
    n = len(r)
    if n < 3 or r.std(ddof=1) == 0:
        return float("nan")

    sr = r.mean() / r.std(ddof=1)                       # per period
    sr_star = benchmark_sr / np.sqrt(periods_per_year)  # per period
    gamma3 = float(r.skew())
    gamma4 = float(r.kurt()) + 3.0                      # pandas gives excess kurtosis

    denom = np.sqrt(1.0 - gamma3 * sr + (gamma4 - 1.0) / 4.0 * sr ** 2)
    if not np.isfinite(denom) or denom <= 0:
        return float("nan")
    z = (sr - sr_star) * np.sqrt(n - 1) / denom
    return float(norm.cdf(z))


def expected_max_sharpe(n_trials: int, var_trials_sr: float) -> float:
    """Expected maximum Sharpe across ``n_trials`` independent zero-skill trials.

    ``E[max] = sqrt(V) * ((1 - g) Phi^-1(1 - 1/N) + g Phi^-1(1 - 1/(N e)))``
    with ``g`` the Euler-Mascheroni constant.  This is the benchmark a
    backtested Sharpe has to beat before it means anything.
    """
    if n_trials < 2:
        return 0.0
    g = 0.5772156649015329
    e = np.e
    z = (1.0 - g) * norm.ppf(1.0 - 1.0 / n_trials) + g * norm.ppf(1.0 - 1.0 / (n_trials * e))
    return float(np.sqrt(var_trials_sr) * z)


def deflated_sharpe_ratio(
    returns: pd.Series,
    n_trials: int,
    var_trials_sr: float,
    periods_per_year: float = 252.0,
) -> float:
    """PSR against the expected maximum Sharpe of ``n_trials`` (Bailey & LdP 2014).

    ``var_trials_sr`` is the variance of the *annualised* Sharpe ratios
    obtained across the trials actually run -- measure it, do not guess it.
    A DSR below 0.95 means the result is indistinguishable from the best of
    that many lucky draws.
    """
    sr0 = expected_max_sharpe(n_trials, var_trials_sr)
    return probabilistic_sharpe_ratio(returns, benchmark_sr=sr0, periods_per_year=periods_per_year)


# ----------------------------------------------------------------------------
# Snippet 14.4 -- drawdown and time under water
# ----------------------------------------------------------------------------
def drawdown_time_under_water(
    equity: pd.Series, dollars: bool = False
) -> tuple[pd.Series, pd.Series]:
    """Series of drawdowns and of the time (in years) spent recovering each."""
    df = equity.to_frame("pnl")
    df["hwm"] = equity.cummax()
    groups = df.groupby("hwm").agg(min=("pnl", "min"), time=("pnl", lambda x: x.index[0]))
    groups = groups[groups["min"] < groups.index]  # high-water marks followed by a loss

    if dollars:
        dd = groups.index - groups["min"]
    else:
        dd = 1.0 - groups["min"] / groups.index
    dd.index = groups["time"]
    dd.index.name = "start"

    ends = list(groups["time"][1:]) + [equity.index[-1]]
    tuw = pd.Series(
        [(e - s).total_seconds() / (365.25 * 24 * 3600) for s, e in zip(groups["time"], ends)],
        index=dd.index,
    )
    return dd.rename("drawdown"), tuw.rename("time_under_water")


# ----------------------------------------------------------------------------
# Snippets 14.1 - 14.3 -- bet timing, holding period, concentration
# ----------------------------------------------------------------------------
def bet_timing(target_position: pd.Series) -> pd.DatetimeIndex:
    """Timestamps at which a bet closes: the book flattens or flips (Snippet 14.1)."""
    pos = target_position
    flat = pos[pos == 0].index
    prev_nonzero = pos.shift(1)
    prev_nonzero = prev_nonzero[prev_nonzero != 0].index
    bets = flat.intersection(prev_nonzero)                 # position went to zero

    prod = pos.iloc[1:].to_numpy() * pos.iloc[:-1].to_numpy()
    flips = pos.index[1:][prod < 0]                        # position changed sign
    bets = bets.union(flips).sort_values()
    if len(pos) and pos.index[-1] not in bets:
        bets = bets.append(pos.index[-1:])
    return bets


def bet_holding_period(target_position: pd.Series) -> tuple[pd.Series, float]:
    """Average holding period, in days, weighted by position size (Snippet 14.2)."""
    hp, t_entry = pd.DataFrame(columns=["dT", "w"]), 0.0
    p_diff = target_position.diff()
    t_diff = (target_position.index - target_position.index[0]).total_seconds() / (24 * 3600)

    for i in range(1, target_position.shape[0]):
        if float(p_diff.iloc[i]) * float(target_position.iloc[i - 1]) >= 0:  # increase or hold
            if float(target_position.iloc[i]) != 0:
                t_entry = (
                    t_entry * float(target_position.iloc[i - 1]) + t_diff[i] * float(p_diff.iloc[i])
                ) / float(target_position.iloc[i])
        else:  # decrease
            if float(target_position.iloc[i]) * float(target_position.iloc[i - 1]) < 0:  # flip
                hp.loc[target_position.index[i], ["dT", "w"]] = (
                    t_diff[i] - t_entry, abs(float(target_position.iloc[i - 1]))
                )
                t_entry = t_diff[i]
            else:
                hp.loc[target_position.index[i], ["dT", "w"]] = (
                    t_diff[i] - t_entry, abs(float(p_diff.iloc[i]))
                )

    if hp["w"].sum() > 0:
        mean_hp = float((hp["dT"] * hp["w"]).sum() / hp["w"].sum())
    else:
        mean_hp = float("nan")
    return hp, mean_hp


def hhi(bet_returns: pd.Series) -> float:
    """Herfindahl-Hirschman concentration of returns, rescaled to [0, 1] (Snippet 14.3).

    0 means returns are perfectly uniform, 1 that a single bet produced
    everything.  Computed separately on positive and negative returns, and on
    monthly aggregates, it exposes the strategies whose backtest is one lucky
    trade wearing a track record.
    """
    r = bet_returns.dropna()
    if r.shape[0] <= 2 or r.sum() == 0:
        return float("nan")
    w = r / r.sum()
    h = (w ** 2).sum()
    return float((h - 1.0 / r.shape[0]) / (1.0 - 1.0 / r.shape[0]))


# ----------------------------------------------------------------------------
# Snippets 15.1 / 15.4 -- strategy risk
# ----------------------------------------------------------------------------
def implied_precision(
    stop_loss: float, profit_taking: float, freq: float, target_sr: float
) -> float:
    """Hit rate required to reach ``target_sr`` given the payouts (Snippet 15.3).

    ``stop_loss`` is negative, ``profit_taking`` positive, ``freq`` the number
    of bets per year.  The answer is often uncomfortable: a symmetric-payout
    strategy betting 250 times a year needs a precision near 0.54 for a
    Sharpe of 1, and precision is exactly what is hardest to estimate.
    """
    a = (freq + target_sr ** 2) * (profit_taking - stop_loss) ** 2
    b = (2 * freq * stop_loss - target_sr ** 2 * (profit_taking - stop_loss)) * (
        profit_taking - stop_loss
    )
    c = freq * stop_loss ** 2
    disc = b ** 2 - 4 * a * c
    if disc < 0:
        return float("nan")
    return float((-b + disc ** 0.5) / (2.0 * a))


def prob_failure(
    bet_returns: pd.Series, freq: float, target_sr: float
) -> tuple[float, float, float]:
    """Probability that realised precision falls below the implied threshold.

    Estimates the two payouts from the realised positive and negative bet
    returns, computes the precision required for ``target_sr``, and returns
    ``(p_fail, p_implied, p_realised)`` using the normal approximation to the
    binomial precision estimate (Snippet 15.4).
    """
    r = bet_returns.dropna()
    pos, neg = r[r > 0], r[r <= 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan"), float("nan"), float("nan")
    pt, sl = float(pos.mean()), float(neg.mean())
    p = float(len(pos) / len(r))
    thres = implied_precision(sl, pt, freq, target_sr)
    risk = float(norm.cdf(thres, p, np.sqrt(max(p * (1 - p), 1e-12) / len(r))))
    return risk, thres, p


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------
def performance_summary(
    returns: pd.Series,
    equity: pd.Series,
    bet_returns: pd.Series | None = None,
    periods_per_year: float = 252.0,
    n_trials: int = 1,
    var_trials_sr: float = 0.0,
    target_sr: float = 1.0,
) -> pd.Series:
    """One flat table with everything the two chapters recommend reporting."""
    years = (equity.index[-1] - equity.index[0]).total_seconds() / (365.25 * 24 * 3600)
    total = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    out = {
        "n_periods": float(len(returns)),
        "years": years,
        "total_return": total,
        "cagr": (1.0 + total) ** (1.0 / years) - 1.0 if years > 0 else float("nan"),
        "ann_vol": float(returns.std(ddof=1) * np.sqrt(periods_per_year)),
        "sharpe": sharpe_ratio(returns, periods_per_year),
        "skew": float(returns.skew()),
        "excess_kurtosis": float(returns.kurt()),
        "psr_vs_0": probabilistic_sharpe_ratio(returns, 0.0, periods_per_year),
    }
    if n_trials > 1:
        out["expected_max_sr"] = expected_max_sharpe(n_trials, var_trials_sr)
        out["dsr"] = deflated_sharpe_ratio(returns, n_trials, var_trials_sr, periods_per_year)

    dd, tuw = drawdown_time_under_water(equity)
    out["max_drawdown"] = float(dd.max()) if len(dd) else 0.0
    out["max_time_under_water_days"] = float(tuw.max() * 365.25) if len(tuw) else 0.0

    if bet_returns is not None and len(bet_returns) > 2:
        pos, neg = bet_returns[bet_returns > 0], bet_returns[bet_returns <= 0]
        out["n_bets"] = float(len(bet_returns))
        out["bets_per_year"] = float(len(bet_returns) / years) if years > 0 else float("nan")
        out["hit_rate"] = float(len(pos) / len(bet_returns))
        out["avg_win"] = float(pos.mean()) if len(pos) else float("nan")
        out["avg_loss"] = float(neg.mean()) if len(neg) else float("nan")
        out["hhi_positive"] = hhi(pos)
        out["hhi_negative"] = hhi(-neg) if len(neg) else float("nan")
        risk, thres, p = prob_failure(bet_returns, out["bets_per_year"], target_sr)
        out[f"implied_precision_sr{target_sr:g}"] = thres
        out["realised_precision"] = p
        out["prob_failure"] = risk

    monthly = returns.resample("ME").sum()
    out["hhi_monthly"] = hhi(monthly[monthly > 0])
    return pd.Series(out, name="performance")
