"""A microstructure-faithful tick generator with known ground truth.

De Prado's methods are designed for *tick* data: the tick rule, imbalance
bars, Roll's spread estimator and Kyle's lambda are all defined on individual
trades.  Rather than degrade to daily OHLC, this module simulates a limit
order book's trade tape from a structural model whose parameters we know,
which lets the test suite assert that the estimators actually recover them.

Structural model
----------------
Latent information flow (AR(1), the source of all predictability)::

    I_t = phi * I_{t-1} + eta_t,        eta_t ~ N(0, sigma_I^2)

Trade sign (autocorrelated order flow, tilted by information)::

    P[b_t = +1] = clip(0.5 + a * I_t + c * b_{t-1}, eps, 1 - eps)

Trade size (activity clusters with information intensity)::

    v_t = round(exp(mu_v + s_v * z_t + b_v * |I_t|))

Efficient (unobservable) log price, Kyle (1985) linear price impact::

    m_t = m_{t-1} + lam * b_t * sqrt(v_t) + sigma_t * eps_t

Observed transaction log price adds a bid-ask bounce of half-spread::

    p_t = m_t + (spread / 2) * b_t

Consequences that the pipeline is meant to exploit or measure:

* ``I_t`` is persistent and drives both current order flow and future
  efficient-price drift, so aggregated order-flow imbalance carries genuine,
  decaying predictive power -- the alpha the model has to find.
* The bounce makes observed returns negatively autocorrelated with
  covariance ``-(spread/2)^2``, which is exactly what Roll's estimator
  inverts (Chapter 19).
* ``lam`` is recoverable by regressing price changes on signed root-volume
  (Kyle / Hasbrouck lambdas, Chapter 19).
* Stochastic volatility makes a *fixed* profit-taking threshold wrong and a
  volatility-scaled triple barrier right (Chapter 3).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.signal import lfilter

__all__ = ["TickSimConfig", "simulate_ticks"]


@dataclass
class TickSimConfig:
    """Ground-truth parameters of the tick simulator."""

    n_ticks: int = 4_000_000
    start: str = "2021-01-04 00:00:00"
    p0: float = 100.0

    # latent information process
    phi: float = 0.995            # AR(1) persistence of information
    sigma_i: float = 1.0          # innovation scale of information

    # order flow
    a: float = 0.030              # information -> P[buy] tilt
    c: float = 0.22               # order-flow autocorrelation
    eps: float = 0.02             # clip on the buy probability

    # trade size (log-normal)
    mu_v: float = 4.0
    s_v: float = 0.85
    b_v: float = 0.12             # information -> volume amplification

    # price
    lam: float = 7.0e-6           # Kyle's lambda, per sqrt(share)
    spread: float = 5.0e-4        # full bid-ask spread, in log price
    sigma_bar: float = 1.1e-4     # average per-tick noise volatility
    vol_phi: float = 0.9995       # persistence of log-volatility
    vol_sigma: float = 0.012      # innovation of log-volatility

    # trade arrivals (seconds), scaled down when activity is high
    mean_interarrival: float = 26.0   # ~1000 days of continuous tape

    seed: int = 20211231

    def as_dict(self) -> dict:
        return asdict(self)


def simulate_ticks(cfg: TickSimConfig | None = None) -> tuple[pd.DataFrame, dict]:
    """Simulate a trade tape.

    Returns
    -------
    ticks : DataFrame
        Indexed by timestamp with columns ``price``, ``volume``, ``dollar``.
        Only these three are observable downstream -- the tick rule has to
        re-infer trade signs, as it would on real TAQ data.
    truth : dict
        The generating parameters plus the hidden series (efficient price,
        true trade signs, information), used only by the tests.
    """
    cfg = cfg or TickSimConfig()
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_ticks

    # --- latent information: AR(1), simulated iteratively ------------------
    eta = rng.normal(0.0, cfg.sigma_i, n)
    eta[0] /= np.sqrt(1.0 - cfg.phi ** 2)  # start from the stationary law
    info = lfilter([1.0], [1.0, -cfg.phi], eta)
    info /= info.std()  # normalise so that `a` is interpretable in prob. units

    # --- trade signs: persistent and information-tilted ---------------------
    u = rng.random(n)
    signs = np.empty(n, dtype=np.int8)
    signs[0] = 1 if u[0] < 0.5 else -1
    a, c, e = cfg.a, cfg.c, cfg.eps
    prev = signs[0]
    for t in range(1, n):
        p_buy = 0.5 + a * info[t] + c * prev
        if p_buy < e:
            p_buy = e
        elif p_buy > 1.0 - e:
            p_buy = 1.0 - e
        prev = 1 if u[t] < p_buy else -1
        signs[t] = prev

    # --- trade sizes --------------------------------------------------------
    z = rng.normal(0.0, 1.0, n)
    volume = np.exp(cfg.mu_v + cfg.s_v * z + cfg.b_v * np.abs(info))
    volume = np.maximum(np.round(volume), 1.0)

    # --- stochastic volatility (AR(1) in logs, unit mean) -------------------
    w = rng.normal(0.0, cfg.vol_sigma, n)
    w[0] /= np.sqrt(1.0 - cfg.vol_phi ** 2)
    log_vol = lfilter([1.0], [1.0, -cfg.vol_phi], w)
    log_vol -= log_vol.mean() + 0.5 * log_vol.var()
    sigma = cfg.sigma_bar * np.exp(log_vol)

    # --- efficient log price: permanent impact + noise ----------------------
    impact = cfg.lam * signs * np.sqrt(volume)
    noise = sigma * rng.normal(0.0, 1.0, n)
    m = np.log(cfg.p0) + np.cumsum(impact + noise)

    # --- observed price: efficient price + bid-ask bounce --------------------
    log_price = m + 0.5 * cfg.spread * signs
    price = np.exp(log_price)

    # --- irregular arrival times, faster when information is intense --------
    intensity = np.exp(-0.25 * np.abs(info))          # mean-1-ish multiplier
    gaps = rng.exponential(cfg.mean_interarrival, n) * intensity
    gaps = np.maximum(gaps, 1e-3)
    stamps = pd.Timestamp(cfg.start) + pd.to_timedelta(np.cumsum(gaps), unit="s")

    ticks = pd.DataFrame(
        {"price": price, "volume": volume, "dollar": price * volume},
        index=pd.DatetimeIndex(stamps, name="timestamp"),
    )

    truth = cfg.as_dict()
    truth.update(
        {
            "efficient_log_price": pd.Series(m, index=ticks.index),
            "true_sign": pd.Series(signs, index=ticks.index),
            "information": pd.Series(info, index=ticks.index),
            "sigma": pd.Series(sigma, index=ticks.index),
        }
    )
    return ticks, truth
