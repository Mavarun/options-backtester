"""Data helpers: synthetic GBM path and CSV round-trip."""

from __future__ import annotations

from pathlib import Path

import pytest

from options_bt.data import load_csv, save_sample_csv, synthesize_gbm_vol


def test_synthesize_has_regimes():
    df = synthesize_gbm_vol(n_days=100, seed=1)
    assert {"S", "sigma", "regime", "r"} <= set(df.columns)
    assert set(df["regime"].unique()) <= {"low", "high"}
    assert (df["S"] > 0).all()
    assert (df["sigma"] > 0).all()


def test_csv_roundtrip(tmp_path: Path):
    p = tmp_path / "sample.csv"
    save_sample_csv(p, n_days=50, seed=99)
    df = load_csv(p)
    assert len(df) == 50
    assert df["S"].iloc[0] > 0
