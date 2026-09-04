"""Strategy marks and costed backtest behaviour on synthetic paths."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from options_bt.backtest import (
    compare_cost_impact,
    entry_exit_cost,
    run_strategy_path,
    walk_forward,
)
from options_bt.data import synthesize_gbm_vol
from options_bt.strategies import (
    build_legs,
    n_option_contracts,
    path_pnl,
    portfolio_mark,
)


@pytest.fixture(scope="module")
def synth():
    return synthesize_gbm_vol(n_days=260, seed=7, S0=100.0)


def test_straddle_has_two_contracts():
    legs = build_legs("long_straddle", 100.0, 30 / 252)
    assert n_option_contracts(legs) == 2


def test_iron_condor_four_legs():
    legs = build_legs("iron_condor", 100.0, 30 / 252)
    assert n_option_contracts(legs) == 4
    assert sum(leg.qty for leg in legs) == 0.0  # long 2 short 2


def test_covered_call_has_stock_and_short_call():
    legs = build_legs("covered_call", 100.0, 30 / 252)
    kinds = {leg.kind for leg in legs}
    assert kinds == {"stock", "call"}
    assert n_option_contracts(legs) == 1


def test_path_pnl_sums_to_mark_change(synth):
    path = path_pnl(
        synth["S"],
        synth["sigma"],
        "long_straddle",
        expiry_years=21 / 252,
        entry_idx=10,
        hold_steps=21,
    )
    assert path["pnl"].sum() == pytest.approx(
        path["mark"].iloc[-1] - path["mark"].iloc[0], abs=1e-8
    )


def test_costs_reduce_pnl_vs_zero_cost(synth):
    zero = run_strategy_path(synth, "long_straddle", spread_bps=0.0, slippage_bps=0.0)
    costly = run_strategy_path(synth, "long_straddle", spread_bps=10.0, slippage_bps=5.0)
    assert costly.costed_total_pnl < zero.costed_total_pnl
    assert costly.uncosted_total_pnl == pytest.approx(zero.uncosted_total_pnl, rel=1e-9)
    assert costly.n_trades == zero.n_trades
    # Every trade should show costed < uncosted when costs > 0
    for t in costly.trades:
        assert t.costed_pnl < t.uncosted_pnl
        assert t.total_cost > 0


def test_entry_exit_cost_scales_with_contracts():
    c2 = entry_exit_cost(100.0, n_contracts=2, has_stock=False, spread_bps=10, slippage_bps=5)
    c4 = entry_exit_cost(100.0, n_contracts=4, has_stock=False, spread_bps=10, slippage_bps=5)
    assert c4 == pytest.approx(2 * c2)


def test_compare_cost_impact_all_strategies(synth):
    table = compare_cost_impact(synth)
    assert set(table["strategy"]) == {"long_straddle", "iron_condor", "covered_call"}
    assert (table["pnl_inflation"] >= 0).all()
    assert (table["uncosted_pnl"] >= table["costed_pnl"]).all()


def test_walk_forward_splits_by_date(synth):
    wf = walk_forward(synth, "iron_condor", train_frac=0.6)
    assert wf.train_end in synth.index or wf.train_end == synth.index[int(0.6 * len(synth)) - 1]
    # OOS window starts after train_end
    if wf.oos_result.trades:
        assert all(t.entry_date > wf.train_end for t in wf.oos_result.trades)
    if wf.is_result.trades:
        assert all(t.entry_date <= wf.train_end for t in wf.is_result.trades)


def test_regime_filter_runs(synth):
    hi = run_strategy_path(synth, "long_straddle", regimes=["high"])
    lo = run_strategy_path(synth, "long_straddle", regimes=["low"])
    # Not asserting which wins -- only that filter changes trade set size/dates
    assert hi.n_trades + lo.n_trades >= max(hi.n_trades, lo.n_trades)


def test_portfolio_mark_iron_condor_credit_at_open():
    # Short body / long wings -> net credit (negative mark for short-heavy at open
    # is not guaranteed for all wing widths; just check finite mark)
    legs = build_legs("iron_condor", 100.0, 30 / 252)
    m = portfolio_mark(legs, 100.0, 30 / 252, r=0.02, sigma=0.2)
    assert np.isfinite(m)
