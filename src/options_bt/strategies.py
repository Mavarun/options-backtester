"""Simple multi-leg option strategy mark-to-model PnL.

Legs are marked with Black-Scholes at each path step. This is a research
toy: no exchange tape, no American exercise, no discrete dividends.
Position PnL is change in portfolio mark; entry/exit costs live in backtest.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from .black_scholes import price as bs_price

StrategyName = Literal["long_straddle", "iron_condor", "covered_call"]


@dataclass(frozen=True)
class Leg:
    """One European option or underlying share position."""

    kind: Literal["call", "put", "stock"]
    strike: float | None  # None for stock
    qty: float  # +1 long, -1 short; stock qty in shares
    expiry_years: float  # remaining T at trade open for options


def _mark_leg(
    leg: Leg,
    S: float,
    T_left: float,
    r: float,
    sigma: float,
    q: float = 0.0,
) -> float:
    """Mark one leg; options expire to intrinsic when T_left <= 0."""
    if leg.kind == "stock":
        return float(leg.qty) * float(S)
    T = max(float(T_left), 0.0)
    K = float(leg.strike)
    prem = bs_price(S, K, T, r, sigma, option_type=leg.kind, q=q)
    return float(leg.qty) * prem


def long_straddle_legs(
    S: float,
    expiry_years: float,
    strike: float | None = None,
) -> list[Leg]:
    """ATM (or given strike) long call + long put, 1 each."""
    K = float(S if strike is None else strike)
    return [
        Leg(kind="call", strike=K, qty=1.0, expiry_years=expiry_years),
        Leg(kind="put", strike=K, qty=1.0, expiry_years=expiry_years),
    ]


def iron_condor_legs(
    S: float,
    expiry_years: float,
    wing_pct: float = 0.10,
    body_pct: float = 0.05,
) -> list[Leg]:
    """Short OTM put/call body, long farther wings (credit structure).

    Puts: long K_put_wing, short K_put_short
    Calls: short K_call_short, long K_call_wing
    """
    S = float(S)
    kp_w = S * (1.0 - wing_pct)
    kp_s = S * (1.0 - body_pct)
    kc_s = S * (1.0 + body_pct)
    kc_w = S * (1.0 + wing_pct)
    return [
        Leg(kind="put", strike=kp_w, qty=1.0, expiry_years=expiry_years),
        Leg(kind="put", strike=kp_s, qty=-1.0, expiry_years=expiry_years),
        Leg(kind="call", strike=kc_s, qty=-1.0, expiry_years=expiry_years),
        Leg(kind="call", strike=kc_w, qty=1.0, expiry_years=expiry_years),
    ]


def covered_call_legs(
    S: float,
    expiry_years: float,
    call_otm_pct: float = 0.05,
) -> list[Leg]:
    """Long 100 shares notionally as 1 share unit + short OTM call."""
    K = float(S) * (1.0 + call_otm_pct)
    return [
        Leg(kind="stock", strike=None, qty=1.0, expiry_years=expiry_years),
        Leg(kind="call", strike=K, qty=-1.0, expiry_years=expiry_years),
    ]


def build_legs(
    strategy: StrategyName,
    S: float,
    expiry_years: float,
    **kwargs,
) -> list[Leg]:
    if strategy == "long_straddle":
        return long_straddle_legs(S, expiry_years, strike=kwargs.get("strike"))
    if strategy == "iron_condor":
        return iron_condor_legs(
            S,
            expiry_years,
            wing_pct=float(kwargs.get("wing_pct", 0.10)),
            body_pct=float(kwargs.get("body_pct", 0.05)),
        )
    if strategy == "covered_call":
        return covered_call_legs(
            S,
            expiry_years,
            call_otm_pct=float(kwargs.get("call_otm_pct", 0.05)),
        )
    raise ValueError(f"unknown strategy: {strategy!r}")


def portfolio_mark(
    legs: list[Leg],
    S: float,
    T_left: float,
    r: float,
    sigma: float,
    q: float = 0.0,
) -> float:
    return float(sum(_mark_leg(leg, S, T_left, r, sigma, q) for leg in legs))


def n_option_contracts(legs: list[Leg]) -> int:
    """Count option legs (for cost scaling); stock is not a contract fill here."""
    return sum(1 for leg in legs if leg.kind in ("call", "put"))


def path_pnl(
    spots: pd.Series,
    vols: pd.Series,
    strategy: StrategyName,
    expiry_years: float,
    r: float = 0.02,
    q: float = 0.0,
    entry_idx: int = 0,
    hold_steps: int | None = None,
    **strategy_kwargs,
) -> pd.DataFrame:
    """Mark-to-model daily PnL for one held structure opened at ``entry_idx``.

    Columns: mark, pnl (uncosted delta mark), S, sigma, T_left.
    """
    spots = pd.Series(spots, dtype=float)
    vols = pd.Series(vols, dtype=float).reindex(spots.index).ffill().bfill()
    if entry_idx < 0 or entry_idx >= len(spots):
        raise IndexError("entry_idx out of range")
    n = len(spots) - entry_idx
    if hold_steps is not None:
        n = min(n, int(hold_steps) + 1)
    idx = spots.index[entry_idx : entry_idx + n]
    S0 = float(spots.iloc[entry_idx])
    legs = build_legs(strategy, S0, expiry_years, **strategy_kwargs)
    # Map calendar steps to year fraction of original expiry
    dt = expiry_years / max(len(idx) - 1, 1)

    marks = []
    T_lefts = []
    for i, t in enumerate(idx):
        T_left = max(expiry_years - i * dt, 0.0)
        m = portfolio_mark(legs, float(spots.loc[t]), T_left, r, float(vols.loc[t]), q)
        marks.append(m)
        T_lefts.append(T_left)

    out = pd.DataFrame(
        {
            "S": spots.loc[idx].astype(float).values,
            "sigma": vols.loc[idx].astype(float).values,
            "T_left": T_lefts,
            "mark": marks,
        },
        index=idx,
    )
    out["pnl"] = out["mark"].diff().fillna(0.0)
    out.attrs["n_contracts"] = n_option_contracts(legs)
    out.attrs["strategy"] = strategy
    out.attrs["entry_mark"] = float(out["mark"].iloc[0])
    return out
