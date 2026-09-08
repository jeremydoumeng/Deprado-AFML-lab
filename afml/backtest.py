"""Event-driven backtest on the bar clock.

The position is the step function produced by :mod:`afml.bet_sizing`,
forward-filled onto the bar index.  A position established at the close of
bar ``t`` earns bar ``t+1``'s return -- the shift is the whole discipline of
a backtest, and dropping it is the most common way to manufacture a Sharpe
ratio out of nothing.

Costs are charged on turnover: crossing the spread costs half the spread per
unit of position changed, plus commission, plus a slippage term proportional
to the square root of participation (a linear-impact model would understate
the cost of the very trades a signal wants to make hardest).  The default
cost model is deliberately conservative; a strategy whose edge disappears
under it did not have one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .stats import bet_timing, performance_summary

__all__ = ["CostModel", "BacktestResult", "run_backtest"]


@dataclass
class CostModel:
    """Per-unit-of-turnover trading costs, in fractions of notional.

    half_spread : float
        Cost of crossing to the other side of the book.  For the simulated
        tape the true value is ``spread / 2``; on real data use the Roll or
        Corwin-Schultz estimate the pipeline already computes.
    commission : float
        Broker fee per unit traded.
    slippage_coef : float
        Coefficient of a square-root impact term applied to the turnover
        itself; ``0`` disables it.
    """

    half_spread: float = 2.5e-4
    commission: float = 0.5e-4
    slippage_coef: float = 0.0

    def cost(self, turnover: pd.Series) -> pd.Series:
        base = (self.half_spread + self.commission) * turnover
        if self.slippage_coef:
            base = base + self.slippage_coef * np.sqrt(turnover)
        return base


@dataclass
class BacktestResult:
    returns: pd.Series          # per-bar net returns
    gross_returns: pd.Series
    costs: pd.Series
    position: pd.Series
    equity: pd.Series
    bet_returns: pd.Series      # net return of each closed bet
    summary: pd.Series

    def __repr__(self) -> str:  # pragma: no cover -- convenience only
        return f"<BacktestResult sharpe={self.summary.get('sharpe', float('nan')):.2f} " \
               f"n_bets={int(self.summary.get('n_bets', 0))}>"


def _bet_returns(equity: pd.Series, position: pd.Series) -> pd.Series:
    """Return of each bet, delimited by the flattening / flipping times."""
    stops = bet_timing(position)
    if len(stops) == 0:
        return pd.Series(dtype=float)
    marks = equity.reindex(stops).dropna()
    return marks.pct_change().dropna().rename("bet_return")


def run_backtest(
    position: pd.Series,
    close: pd.Series,
    costs: CostModel | None = None,
    periods_per_year: float | None = None,
    initial_equity: float = 1.0,
    n_trials: int = 1,
    var_trials_sr: float = 0.0,
    target_sr: float = 1.0,
) -> BacktestResult:
    """Run the backtest and compute the Chapter 14/15 report.

    Parameters
    ----------
    position : Series
        Target position in [-1, 1], indexed by the times at which it changes
        (typically the output of :func:`afml.bet_sizing.get_signal`).
    close : Series
        Bar closes; the return series and the reporting clock.
    periods_per_year : float, optional
        Annualisation factor.  Inferred from the average bar spacing when
        omitted, which is the right thing to do for an irregular bar clock.
    """
    costs = costs or CostModel()

    pos = position.reindex(close.index.union(position.index)).ffill().reindex(close.index)
    pos = pos.fillna(0.0).clip(-1.0, 1.0)

    ret = close.pct_change().fillna(0.0)
    held = pos.shift(1).fillna(0.0)             # decided at t-1, earns r_t
    gross = held * ret

    turnover = held.diff().abs().fillna(held.abs())
    cost = costs.cost(turnover)
    net = gross - cost

    equity = initial_equity * (1.0 + net).cumprod()

    if periods_per_year is None:
        seconds = (close.index[-1] - close.index[0]).total_seconds()
        periods_per_year = len(close) / (seconds / (365.25 * 24 * 3600))

    bets = _bet_returns(equity, pos)
    summary = performance_summary(
        net, equity, bets, periods_per_year=periods_per_year,
        n_trials=n_trials, var_trials_sr=var_trials_sr, target_sr=target_sr,
    )
    summary["gross_sharpe"] = (
        gross.mean() / gross.std(ddof=1) * np.sqrt(periods_per_year) if gross.std(ddof=1) else np.nan
    )
    summary["total_cost"] = float(cost.sum())
    summary["avg_abs_position"] = float(pos.abs().mean())
    summary["turnover_per_year"] = float(turnover.sum() / (len(close) / periods_per_year))

    return BacktestResult(net, gross, cost, pos, equity, bets, summary)
