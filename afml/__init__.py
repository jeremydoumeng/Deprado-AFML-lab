"""A systematic trading pipeline built from Marcos Lopez de Prado's
*Advances in Financial Machine Learning* (Wiley, 2018).

Module            Chapter   What it is for
----------------  --------  ---------------------------------------------
``data``          2         Tick simulator with known microstructure truth
``bars``          2         Tick/volume/dollar, imbalance and run bars
``fracdiff``      5         Stationarity with maximum memory retained
``features``      5, 19     Feature matrix incl. microstructural estimators
``labeling``      3         Triple barrier, CUSUM sampling, meta-labeling
``sampling``      4         Concurrency, uniqueness, sequential bootstrap
``cv``            7         Purged k-fold, embargo, walk-forward
``importance``    8         MDI / MDA / SFI, orthogonalisation
``bet_sizing``    10        Probability to position size
``backtest``      14, 15    Event-driven backtest with costs
``stats``         14, 15    PSR, DSR, drawdown, concentration, risk
``multiprocess``  20        The job engine every heavy loop runs on
"""

__version__ = "0.1.0"

from . import (  # noqa: F401
    backtest,
    bars,
    bet_sizing,
    cv,
    data,
    features,
    fracdiff,
    importance,
    labeling,
    multiprocess,
    sampling,
    stats,
)

__all__ = [
    "backtest", "bars", "bet_sizing", "cv", "data", "features", "fracdiff",
    "importance", "labeling", "multiprocess", "sampling", "stats",
]
