#!/usr/bin/env python3
"""Run the first options research slice: BS checks, costed vs uncosted, walk-forward.

Usage (from repo root, with package on PYTHONPATH or installed editable)::

    python scripts/run_options_slice.py

Does not claim live PnL or alpha. Defaults use synthetic GBM + regime vol.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from options_bt.backtest import compare_cost_impact, walk_forward
from options_bt.black_scholes import greeks, price, put_call_parity_lhs
from options_bt.data import load_path, save_sample_csv


def main() -> int:
    sample = ROOT / "data" / "sample_gbm_vol.csv"
    if not sample.exists():
        save_sample_csv(sample, n_days=504, seed=42, S0=100.0)
        print(f"Wrote synthetic sample: {sample}")

    df = load_path(csv_path=sample, prefer_yahoo=False)
    print(f"Path: {len(df)} business days | S0={df['S'].iloc[0]:.2f} | "
          f"sigma low/high = {df.loc[df.regime=='low','sigma'].iloc[0]:.2f}/"
          f"{df.loc[df.regime=='high','sigma'].iloc[0]:.2f}")

    # Closed-form sanity
    ref = dict(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.20, q=0.0)
    c = price(**ref, option_type="call")
    p = price(**ref, option_type="put")
    g = greeks(**ref, option_type="call")
    parity = put_call_parity_lhs(c, p, **{k: ref[k] for k in ("S", "K", "T", "r", "q")})
    print("\n=== Black-Scholes reference (S=K=100, T=1, r=5%, sigma=20%) ===")
    print(f"call={c:.4f} put={p:.4f} parity_err={parity:.2e}")
    print(f"delta={g.delta:.4f} gamma={g.gamma:.6f} vega={g.vega:.4f} theta={g.theta:.4f}")

    print("\n=== Costed vs uncosted (full sample, synthetic) ===")
    table = compare_cost_impact(df, expiry_years=21 / 252)
    print(table.to_string(index=False, float_format=lambda x: f"{x:,.4f}"))

    print("\n=== Walk-forward OOS (train_frac=0.60) ===")
    rows = []
    for strat in ("long_straddle", "iron_condor", "covered_call"):
        wf = walk_forward(df, strat, train_frac=0.60, expiry_years=21 / 252)
        rows.append(
            {
                "strategy": strat,
                "train_end": str(wf.train_end.date() if hasattr(wf.train_end, "date") else wf.train_end),
                "IS_costed_pnl": wf.is_result.costed_total_pnl,
                "OOS_costed_pnl": wf.oos_result.costed_total_pnl,
                "IS_costed_sharpe": wf.is_result.costed_sharpe,
                "OOS_costed_sharpe": wf.oos_result.costed_sharpe,
                "IS_trades": wf.is_result.n_trades,
                "OOS_trades": wf.oos_result.n_trades,
            }
        )
    import pandas as pd

    wf_table = pd.DataFrame(rows)
    print(wf_table.to_string(index=False, float_format=lambda x: f"{x:,.4f}"))

    # Honest summary for README paste
    summary = {
        "bs_call": c,
        "bs_put": p,
        "parity_err": parity,
        "cost_table": table.to_dict(orient="records"),
        "walk_forward": rows,
        "note": "Synthetic path only. Not live PnL. Not alpha.",
    }
    out = ROOT / "data" / "last_run_metrics.json"
    out.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote {out}")
    print(
        "\nHonest read: if OOS costed PnL and Sharpe are weak or negative, "
        "there is no surviving edge on this toy path after costs -- report that."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
