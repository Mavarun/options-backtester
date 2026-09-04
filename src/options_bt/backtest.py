"""Costed option-strategy path backtest and date walk-forward.

Costs: bid-ask half-spread + slippage in bps of underlying, charged per
option contract on open and close (and once on entry mark for stock legs
via a share spread). Ignoring costs inflates PnL -- that is the point of
the costed vs uncosted comparison.

This is not live PnL and not a claim of alpha.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .strategies import StrategyName, n_option_contracts, path_pnl, build_legs

# Defaults (not optimized)
SPREAD_BPS = 10.0  # half-spread per option contract, vs spot
SLIPPAGE_BPS = 5.0
SHARE_SPREAD_BPS = 1.0  # for covered-call stock leg
HOLD_FRACTION = 1.0  # hold until expiry by default
TRAIN_FRAC = 0.60
ANNUALIZATION = 252


def _sharpe(returns: pd.Series, periods: int = ANNUALIZATION) -> float:
    r = returns.dropna().astype(float)
    if len(r) < 2:
        return float("nan")
    sd = float(r.std(ddof=1))
    if sd == 0.0:
        return 0.0
    return float(r.mean() / sd * np.sqrt(periods))


def _max_drawdown(returns: pd.Series) -> float:
    r = returns.fillna(0.0).astype(float)
    equity = (1.0 + r).cumprod()
    if equity.empty:
        return float("nan")
    dd = equity / equity.cummax() - 1.0
    return float(dd.min())


def entry_exit_cost(
    S: float,
    n_contracts: int,
    has_stock: bool,
    spread_bps: float = SPREAD_BPS,
    slippage_bps: float = SLIPPAGE_BPS,
    share_spread_bps: float = SHARE_SPREAD_BPS,
) -> float:
    """Dollar cost for one open (or one close) of the structure."""
    opt = abs(int(n_contracts)) * float(S) * (float(spread_bps) + float(slippage_bps)) / 1e4
    stock = float(S) * float(share_spread_bps) / 1e4 if has_stock else 0.0
    return float(opt + stock)


@dataclass
class TradeResult:
    strategy: str
    regime: str
    entry_date: object
    exit_date: object
    uncosted_pnl: float
    costed_pnl: float
    total_cost: float
    entry_mark: float
    n_contracts: int


@dataclass
class BacktestResult:
    strategy: str
    uncosted_total_pnl: float
    costed_total_pnl: float
    uncosted_sharpe: float
    costed_sharpe: float
    max_drawdown_costed: float
    n_trades: int
    trades: list[TradeResult] = field(repr=False)
    daily_uncosted: pd.Series = field(repr=False)
    daily_costed: pd.Series = field(repr=False)


def _regime_at(df: pd.DataFrame, i: int) -> str:
    if "regime" in df.columns:
        return str(df["regime"].iloc[i])
    return "unknown"


def run_strategy_path(
    df: pd.DataFrame,
    strategy: StrategyName,
    expiry_years: float = 30 / 252,
    r: float | None = None,
    q: float = 0.0,
    spread_bps: float = SPREAD_BPS,
    slippage_bps: float = SLIPPAGE_BPS,
    reentry_gap: int = 5,
    regimes: list[str] | None = None,
    **strategy_kwargs,
) -> BacktestResult:
    """Roll a held structure to expiry, reopen after ``reentry_gap`` days.

    Daily returns for Sharpe are PnL / entry |mark| (or spot if mark~0).
    """
    spots = df["S"].astype(float)
    vols = df["sigma"].astype(float)
    rate = float(df["r"].iloc[0] if r is None and "r" in df.columns else (r if r is not None else 0.02))

    hold_steps = max(int(round(expiry_years * 252)), 1)
    trades: list[TradeResult] = []
    daily_u = pd.Series(0.0, index=spots.index, dtype=float)
    daily_c = pd.Series(0.0, index=spots.index, dtype=float)

    i = 0
    n = len(spots)
    while i < n - 2:
        reg = _regime_at(df, i)
        if regimes is not None and reg not in regimes:
            i += 1
            continue
        path = path_pnl(
            spots,
            vols,
            strategy,
            expiry_years=expiry_years,
            r=rate,
            q=q,
            entry_idx=i,
            hold_steps=hold_steps,
            **strategy_kwargs,
        )
        if len(path) < 2:
            break
        legs = build_legs(strategy, float(spots.iloc[i]), expiry_years, **strategy_kwargs)
        n_c = n_option_contracts(legs)
        has_stock = any(leg.kind == "stock" for leg in legs)
        S_entry = float(spots.iloc[i])
        S_exit = float(path["S"].iloc[-1])
        open_c = entry_exit_cost(S_entry, n_c, has_stock, spread_bps, slippage_bps)
        close_c = entry_exit_cost(S_exit, n_c, has_stock, spread_bps, slippage_bps)
        total_cost = open_c + close_c

        uncosted = float(path["mark"].iloc[-1] - path["mark"].iloc[0])
        costed = uncosted - total_cost

        # Book daily uncosted pnl; dump costs on entry and exit bars
        daily_u.loc[path.index] = daily_u.loc[path.index] + path["pnl"]
        daily_c.loc[path.index] = daily_c.loc[path.index] + path["pnl"]
        daily_c.iloc[daily_c.index.get_loc(path.index[0])] -= open_c
        daily_c.iloc[daily_c.index.get_loc(path.index[-1])] -= close_c

        trades.append(
            TradeResult(
                strategy=strategy,
                regime=reg,
                entry_date=path.index[0],
                exit_date=path.index[-1],
                uncosted_pnl=uncosted,
                costed_pnl=costed,
                total_cost=total_cost,
                entry_mark=float(path["mark"].iloc[0]),
                n_contracts=n_c,
            )
        )
        # Advance past this hold + gap
        i = spots.index.get_loc(path.index[-1]) + 1 + int(reentry_gap)

    # Scale to return-like series for Sharpe: / typical spot
    scale = float(spots.mean())
    if scale <= 0:
        scale = 1.0
    ru = daily_u / scale
    rc = daily_c / scale

    return BacktestResult(
        strategy=strategy,
        uncosted_total_pnl=float(sum(t.uncosted_pnl for t in trades)),
        costed_total_pnl=float(sum(t.costed_pnl for t in trades)),
        uncosted_sharpe=_sharpe(ru),
        costed_sharpe=_sharpe(rc),
        max_drawdown_costed=_max_drawdown(rc),
        n_trades=len(trades),
        trades=trades,
        daily_uncosted=daily_u,
        daily_costed=daily_c,
    )


@dataclass
class WalkForwardResult:
    strategy: str
    train_end: object
    is_result: BacktestResult
    oos_result: BacktestResult


def walk_forward(
    df: pd.DataFrame,
    strategy: StrategyName,
    train_frac: float = TRAIN_FRAC,
    train_end: object | None = None,
    **kwargs,
) -> WalkForwardResult:
    """Date split: fit/select nothing heavy -- just report IS vs OOS costed PnL.

    For this slice the 'model' is the fixed strategy rule; walk-forward checks
    whether costed edge in the early window survives later dates / vol regimes.
    """
    df = df.sort_index()
    if train_end is None:
        cut = int(len(df) * float(train_frac))
        cut = max(cut, 20)
        cut = min(cut, len(df) - 10)
        train_end = df.index[cut - 1]
    is_df = df.loc[:train_end]
    oos_df = df.loc[df.index > train_end]
    is_res = run_strategy_path(is_df, strategy, **kwargs)
    oos_res = run_strategy_path(oos_df, strategy, **kwargs)
    return WalkForwardResult(
        strategy=strategy,
        train_end=train_end,
        is_result=is_res,
        oos_result=oos_res,
    )


def compare_cost_impact(
    df: pd.DataFrame,
    strategies: list[StrategyName] | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Table: uncosted vs costed total PnL and Sharpe per strategy."""
    strategies = strategies or ["long_straddle", "iron_condor", "covered_call"]
    rows = []
    for s in strategies:
        res = run_strategy_path(df, s, **kwargs)
        rows.append(
            {
                "strategy": s,
                "n_trades": res.n_trades,
                "uncosted_pnl": res.uncosted_total_pnl,
                "costed_pnl": res.costed_total_pnl,
                "pnl_inflation": res.uncosted_total_pnl - res.costed_total_pnl,
                "uncosted_sharpe": res.uncosted_sharpe,
                "costed_sharpe": res.costed_sharpe,
                "max_dd_costed": res.max_drawdown_costed,
            }
        )
    return pd.DataFrame(rows)
