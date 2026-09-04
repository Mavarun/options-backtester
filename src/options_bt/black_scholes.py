"""Black-Scholes-Merton European option pricing and Greeks.

Closed-form under constant rate, constant volatility, no dividends unless
``q`` is set. Citations: Black & Scholes (1973); Merton (1973) for continuous
dividend yield. Not a live pricing engine -- no smile, no early exercise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


def _d1_d2(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
) -> tuple[float, float]:
    S, K, T, r, sigma, q = float(S), float(K), float(T), float(r), float(sigma), float(q)
    if S <= 0 or K <= 0:
        raise ValueError("S and K must be positive")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if T < 0:
        raise ValueError("T must be non-negative")
    if T == 0.0:
        # At expiry d1/d2 are undefined; callers use intrinsic path.
        return float("nan"), float("nan")
    vol_sqrt_t = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    return float(d1), float(d2)


def intrinsic(S: float, K: float, option_type: str = "call") -> float:
    """European intrinsic value at expiry (or T=0)."""
    S, K = float(S), float(K)
    ot = option_type.lower()
    if ot == "call":
        return max(S - K, 0.0)
    if ot == "put":
        return max(K - S, 0.0)
    raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")


def price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    q: float = 0.0,
) -> float:
    """Black-Scholes-Merton European price.

    Parameters
    ----------
    S, K : spot and strike
    T : time to expiry in years
    r : continuous risk-free rate
    sigma : volatility (annualized)
    option_type : 'call' or 'put'
    q : continuous dividend yield
    """
    if float(T) == 0.0:
        return intrinsic(S, K, option_type)
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    ot = option_type.lower()
    disc_q = np.exp(-q * T)
    disc_r = np.exp(-r * T)
    if ot == "call":
        return float(S * disc_q * norm.cdf(d1) - K * disc_r * norm.cdf(d2))
    if ot == "put":
        return float(K * disc_r * norm.cdf(-d2) - S * disc_q * norm.cdf(-d1))
    raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")


def delta(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    q: float = 0.0,
) -> float:
    """dV/dS under BSM."""
    if float(T) == 0.0:
        S, K = float(S), float(K)
        ot = option_type.lower()
        if ot == "call":
            if S > K:
                return 1.0
            if S < K:
                return 0.0
            return 0.5
        if ot == "put":
            if S < K:
                return -1.0
            if S > K:
                return 0.0
            return -0.5
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    disc_q = np.exp(-q * T)
    ot = option_type.lower()
    if ot == "call":
        return float(disc_q * norm.cdf(d1))
    if ot == "put":
        return float(disc_q * (norm.cdf(d1) - 1.0))
    raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")


def gamma(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
) -> float:
    """d2V/dS2 (same for call and put under BSM)."""
    if float(T) == 0.0:
        return 0.0
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    disc_q = np.exp(-q * T)
    return float(disc_q * norm.pdf(d1) / (float(S) * float(sigma) * np.sqrt(T)))


def vega(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
) -> float:
    """dV/dsigma (raw, per 1.0 vol point; divide by 100 for per-percent)."""
    if float(T) == 0.0:
        return 0.0
    d1, _ = _d1_d2(S, K, T, r, sigma, q)
    disc_q = np.exp(-q * T)
    return float(float(S) * disc_q * norm.pdf(d1) * np.sqrt(T))


def theta(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    q: float = 0.0,
) -> float:
    """dV/dT calendar form is usually /365; this returns dV/dT in years (raw BSM)."""
    if float(T) == 0.0:
        return 0.0
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    S, K, T, r, sigma, q = float(S), float(K), float(T), float(r), float(sigma), float(q)
    disc_q = np.exp(-q * T)
    disc_r = np.exp(-r * T)
    first = -S * disc_q * norm.pdf(d1) * sigma / (2.0 * np.sqrt(T))
    ot = option_type.lower()
    if ot == "call":
        return float(first + q * S * disc_q * norm.cdf(d1) - r * K * disc_r * norm.cdf(d2))
    if ot == "put":
        return float(first - q * S * disc_q * norm.cdf(-d1) + r * K * disc_r * norm.cdf(-d2))
    raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")


@dataclass(frozen=True)
class Greeks:
    price: float
    delta: float
    gamma: float
    vega: float
    theta: float


def greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    q: float = 0.0,
) -> Greeks:
    """Bundle price + first-order Greeks."""
    return Greeks(
        price=price(S, K, T, r, sigma, option_type, q),
        delta=delta(S, K, T, r, sigma, option_type, q),
        gamma=gamma(S, K, T, r, sigma, q),
        vega=vega(S, K, T, r, sigma, q),
        theta=theta(S, K, T, r, sigma, option_type, q),
    )


def put_call_parity_lhs(
    call: float,
    put: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float = 0.0,
) -> float:
    """Return C - P - (S e^{-qT} - K e^{-rT}); should be ~0 under BSM."""
    return float(call - put - (S * np.exp(-q * T) - K * np.exp(-r * T)))
