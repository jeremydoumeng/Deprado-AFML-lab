"""Shared fixtures.  Kept small so the whole suite runs in well under a minute."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from afml import bars as B
from afml import labeling as L
from afml.data import TickSimConfig, simulate_ticks


@pytest.fixture(scope="session")
def sim():
    """A short tape plus the parameters that generated it."""
    cfg = TickSimConfig(n_ticks=300_000, seed=12345)
    ticks, truth = simulate_ticks(cfg)
    return ticks, truth


@pytest.fixture(scope="session")
def ticks(sim):
    return sim[0]


@pytest.fixture(scope="session")
def truth(sim):
    return sim[1]


@pytest.fixture(scope="session")
def dollar_bars(ticks):
    days = (ticks.index[-1] - ticks.index[0]).total_seconds() / 86400
    return B.dollar_bars(ticks, ticks["dollar"].sum() / (days * 50))


@pytest.fixture(scope="session")
def events(dollar_bars):
    """Triple-barrier events with the vertical barrier binding often enough to matter."""
    close = dollar_bars["close"]
    vol = L.get_vol(close, span=100, horizon=pd.Timedelta(hours=4))
    t_events = L.cusum_filter(close, vol * 0.5)
    t1 = L.add_vertical_barrier(t_events, close, pd.Timedelta(hours=12))
    ev = L.get_events(close, t_events, (1.0, 1.0), vol, t1=t1)
    return ev, L.get_bins(ev, close)


@pytest.fixture
def step_path():
    """A deterministic price path with hand-checkable barrier touches."""
    idx = pd.date_range("2021-01-01", periods=11, freq="h")
    # thresholds of +-2% sit at 102 and 98; every crossing below is unambiguous
    #              0     1      2      3     4     5     6     7     8     9    10
    px = np.array([100, 100.5, 102.5, 103, 99.0, 97.0, 100, 100, 100, 100, 100], dtype=float)
    return pd.Series(px, index=idx)
