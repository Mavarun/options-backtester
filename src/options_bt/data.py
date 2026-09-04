"""Underlying and implied-vol path loaders.

Prefer synthetic GBM + regime vol for offline reproducibility. Optional
yfinance Adj Close for the underlying only -- Yahoo is not a research-grade
options tape; we never claim live option quotes from it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def synthesize_gbm_vol(
    n_days: int = 504,
    S0: float = 100.0,
    mu: float = 0.05,
    sigma_low: float = 0.15,
    sigma_high: float = 0.35,
    high_vol_frac: float = 0.35,
    r: float = 0.02,
    seed: int = 42,
    start: str = "2020-01-02",
) -> pd.DataFrame:
    """Synthetic daily spot via GBM with a two-regime vol path.

    Citation for GBM: standard geometric Brownian motion
    ``dS = mu S dt + sigma_t S dW`` (Black-Scholes world). The vol switch is
    a simple deterministic regime split for research, not a calibrated SV model.

    Returns columns: S, sigma, regime ('low'|'high'), r
    """
    rng = np.random.default_rng(int(seed))
    dates = pd.bdate_range(start=start, periods=int(n_days))
    n = len(dates)
    n_high = int(round(n * float(high_vol_frac)))
    # Second half-ish is high vol so walk-forward train often sits in low vol
    regime = np.array(["low"] * n, dtype=object)
    if n_high > 0:
        start_h = max(n - n_high, n // 2)
        regime[start_h:] = "high"
    sigma = np.where(regime == "high", float(sigma_high), float(sigma_low)).astype(float)

    dt = 1.0 / 252.0
    z = rng.standard_normal(n)
    log_ret = (float(mu) - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    log_S = np.log(float(S0)) + np.cumsum(log_ret)
    S = np.exp(log_S)

    return pd.DataFrame(
        {
            "S": S,
            "sigma": sigma,
            "regime": regime,
            "r": float(r),
        },
        index=dates,
    )


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load a synthetic or cached path CSV (comment lines with # allowed)."""
    path = Path(path)
    df = pd.read_csv(path, comment="#", parse_dates=["date"])
    df = df.set_index("date").sort_index()
    need = {"S", "sigma"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing {missing}; have {list(df.columns)}")
    if "r" not in df.columns:
        df["r"] = 0.02
    if "regime" not in df.columns:
        med = float(df["sigma"].median())
        df["regime"] = np.where(df["sigma"] >= med, "high", "low")
    return df[["S", "sigma", "regime", "r"]].astype(
        {"S": float, "sigma": float, "r": float},
        errors="ignore",
    )


def save_sample_csv(
    path: str | Path,
    **kwargs,
) -> Path:
    """Write a synthetic sample under ``data/`` for offline runs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = synthesize_gbm_vol(**kwargs)
    out = df.reset_index().rename(columns={"index": "date"})
    header = (
        "# Synthetic GBM + two-regime vol path for options-backtester.\n"
        "# NOT live market data. Seeded; see synthesize_gbm_vol docstring.\n"
    )
    with path.open("w", encoding="utf-8") as f:
        f.write(header)
        out.to_csv(f, index=False)
    return path


def download_underlying(
    ticker: str = "SPY",
    start: str = "2020-01-01",
    end: str | None = None,
    vol_window: int = 21,
    r: float = 0.02,
) -> pd.DataFrame:
    """Yahoo Adj Close + realized vol proxy (not implied vol).

    Raises on empty download so callers can fall back to synthetic CSV.
    Realized vol is a research proxy only -- not option IV.
    """
    import yfinance as yf

    raw = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if raw is None or raw.empty:
        raise RuntimeError(f"yfinance returned no rows for {ticker}")
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"][ticker] if ticker in raw["Close"].columns else raw["Close"].iloc[:, 0]
    else:
        close = raw["Close"] if "Close" in raw.columns else raw["Adj Close"]
    S = close.astype(float).dropna()
    ret = np.log(S).diff()
    sigma = ret.rolling(int(vol_window)).std() * np.sqrt(252.0)
    sigma = sigma.bfill().ffill()
    med = float(sigma.median())
    regime = np.where(sigma >= med, "high", "low")
    return pd.DataFrame(
        {"S": S.values, "sigma": sigma.values, "regime": regime, "r": float(r)},
        index=S.index,
    )


def load_path(
    csv_path: str | Path | None = None,
    prefer_yahoo: bool = False,
    ticker: str = "SPY",
    **synth_kwargs,
) -> pd.DataFrame:
    """Unified loader: optional Yahoo, else CSV, else fresh synthetic."""
    if prefer_yahoo:
        try:
            return download_underlying(ticker=ticker)
        except Exception:
            pass
    if csv_path is not None and Path(csv_path).exists():
        return load_csv(csv_path)
    return synthesize_gbm_vol(**synth_kwargs)
