# options-backtester

Research slice for **Black-Scholes pricing + Greeks** and **costed** toy backtests of long straddle, iron condor, and covered call on a synthetic GBM path with a two-regime vol switch. Marks are model prices, not exchange tapes.

This is a method check. **Do not read any number here as live PnL or alpha.**

## Today's hypothesis

1. Black-Scholes + Greeks (delta, gamma, vega, theta) correctly price European options vs known closed-form checks.
2. Simple strategy backtests (long straddle, iron condor, covered call) on a vol-regime split show that ignoring bid-ask/slippage inflates PnL.
3. Walk-forward / OOS by date will show whether any edge survives costs; report honestly if it does not.

## What is implemented

| Piece | What it does | What it does not do |
| --- | --- | --- |
| `black_scholes` | BSM European price + delta, gamma, vega, theta; put-call parity helper | No smile, no American, no discrete dividends |
| `strategies` | Build legs + mark-to-model path PnL for straddle / iron condor / covered call | No listed strikes grid, no early exit rule |
| `backtest` | Roll structures to expiry, charge open+close costs, walk-forward by date | No portfolio optimizer, no IV surface, no fills |
| `data` | Seeded GBM + regime sigma, optional yfinance **underlying** only | Yahoo is not an options tape; realized vol != IV |

## Default costs (not optimized)

| Name | Default | Role |
| --- | --- | --- |
| `SPREAD_BPS` | 10 | Half-spread per option contract, vs spot |
| `SLIPPAGE_BPS` | 5 | Extra slippage per option contract |
| `SHARE_SPREAD_BPS` | 1 | Stock leg (covered call) |
| `expiry` | 21/252 year | Hold to synthetic expiry |
| `TRAIN_FRAC` | 0.60 | Walk-forward date split |

Round-trip cost ~= 2 x n_contracts x S x 15 bps (plus share spread for covered call). That is a guess, not an exchange tape.

## Local metrics (synthetic seed=42, 504 days)

Black-Scholes reference `S=K=100, T=1, r=5%, sigma=20%`: call **10.4506**, put **5.5735**, parity error **0**, delta~=0.6368, gamma~=0.0188, vega~=37.52, theta~=-6.41 (year units).

### Costed vs uncosted (full sample)

| strategy | n_trades | uncosted_pnl | costed_pnl | inflation | uncosted_sharpe | costed_sharpe |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| long_straddle | 19 | -5.51 | -16.70 | 11.19 | -0.20 | -0.63 |
| iron_condor | 19 | -2.91 | -25.30 | 22.39 | -0.25 | -1.89 |
| covered_call | 19 | -8.02 | -13.99 | 5.97 | -0.22 | -0.39 |

**Hypothesis 2 holds on this path:** uncosted PnL is always higher (less negative) than costed; iron condor inflation is largest (4 contracts).

### Walk-forward OOS (`train_end=2021-02-26`)

| strategy | IS costed PnL | OOS costed PnL | IS Sharpe | OOS Sharpe |
| --- | ---: | ---: | ---: | ---: |
| long_straddle | -22.73 | +2.50 | -2.93 | +0.18 |
| iron_condor | -8.60 | -14.19 | -2.02 | -2.36 |
| covered_call | +1.54 | +9.21 | +0.12 | +0.56 |

**Honest OOS read:** iron condor has **no** surviving edge after costs. Long straddle OOS is barely positive with tiny Sharpe on one synthetic split -- not evidence of alpha. Covered call looks better on this seeded GBM path because the short call harvests vol in a model world with no crash/jump and no realistic borrow; that is a **toy artifact**, not a live claim.

## Assumptions

- European BSM with constant `r`, constant path sigma within a day, continuous dividend yield `q` (default 0).
- Strategies are marked with the same BSM engine used to "price" them (no bid/ask quote simulation beyond bps costs).
- Synthetic spot is GBM (`dS = mu S dt + sigma_t S dW`); sigma switches low->high by construction. Citation: standard GBM / Black-Scholes world.
- Optional `yfinance` path supplies Adj Close + realized-vol proxy only -- **not** implied vol, **not** option chains.
- Sharpe uses daily PnL / mean spot, annualized with sqrt252. Small samples -> noisy.

## Why this can fail

BSM is wrong when vol is stochastic, jumps exist, or early exercise matters. Marking both entry and exit with mid model prices understates real bid-ask on wings. A single seeded path and one date split can flip sign by luck. Covered-call "edge" here is not portable to live markets (gap risk, assignment, borrow, dividends). Costs are linear bps guesses. **None of this is live PnL.**

## Install / run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pytest -q
python scripts/run_options_slice.py
```

`requirements.txt` mirrors `pyproject.toml`: pandas, numpy, scipy, statsmodels, scikit-learn, yfinance, pytest. No PyTorch.

## Layout

```
src/options_bt/   black_scholes.py, strategies.py, backtest.py, data.py
tests/            parity, known BS values, costs reduce PnL, walk-forward split
scripts/          run_options_slice.py
data/             sample_gbm_vol.csv (synthetic), last_run_metrics.json
```

## What is still weak

- No implied-vol surface or listed chain; sigma is either regime constant or realized-vol proxy.
- No jump/stochastic-vol stress; covered-call OOS can look flattering.
- Costs are flat bps, not size- or moneyness-dependent.
- Walk-forward does not re-tune strikes; it only date-splits a fixed rule.
- statsmodels / sklearn are declared for the research stack but unused in this first slice (reserved for later vol / regime work).
