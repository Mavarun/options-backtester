"""Synthetic closed-form checks for Black-Scholes price and Greeks."""

from __future__ import annotations

import math

import numpy as np
import pytest

from options_bt.black_scholes import (
    delta,
    gamma,
    greeks,
    price,
    put_call_parity_lhs,
    theta,
    vega,
)


# Haug / textbook-style reference: S=100, K=100, T=1, r=0.05, sigma=0.2, q=0
# Call ~= 10.4506, Put ~= 5.5735 (common published check within a few cents)
REF = dict(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.20, q=0.0)


def test_atm_call_price_near_known_value():
    c = price(**REF, option_type="call")
    assert c == pytest.approx(10.45058357, rel=1e-4, abs=1e-3)


def test_atm_put_price_near_known_value():
    p = price(**REF, option_type="put")
    assert p == pytest.approx(5.57352602, rel=1e-4, abs=1e-3)


def test_put_call_parity():
    c = price(**REF, option_type="call")
    p = price(**REF, option_type="put")
    lhs = put_call_parity_lhs(c, p, REF["S"], REF["K"], REF["T"], REF["r"], REF["q"])
    assert abs(lhs) < 1e-8


def test_delta_call_between_0_and_1():
    d = delta(**REF, option_type="call")
    assert 0.0 < d < 1.0
    # ATM with r>0 -> call delta slightly > 0.5
    assert d == pytest.approx(0.6368, abs=5e-3)


def test_delta_put_call_relationship():
    dc = delta(**REF, option_type="call")
    dp = delta(**REF, option_type="put")
    # With q=0: delta _call - delta _put = e^{-qT} = 1
    assert dc - dp == pytest.approx(1.0, abs=1e-8)


def test_gamma_positive_and_shared():
    g = gamma(REF["S"], REF["K"], REF["T"], REF["r"], REF["sigma"], REF["q"])
    assert g > 0
    # Finite-difference check
    eps = 1e-2
    c_up = price(REF["S"] + eps, REF["K"], REF["T"], REF["r"], REF["sigma"], "call", REF["q"])
    c_dn = price(REF["S"] - eps, REF["K"], REF["T"], REF["r"], REF["sigma"], "call", REF["q"])
    c0 = price(**REF, option_type="call")
    g_fd = (c_up - 2 * c0 + c_dn) / (eps**2)
    assert g == pytest.approx(g_fd, rel=1e-2, abs=1e-4)


def test_vega_positive_fd():
    v = vega(REF["S"], REF["K"], REF["T"], REF["r"], REF["sigma"], REF["q"])
    assert v > 0
    eps = 1e-4
    c_up = price(REF["S"], REF["K"], REF["T"], REF["r"], REF["sigma"] + eps, "call", REF["q"])
    c_dn = price(REF["S"], REF["K"], REF["T"], REF["r"], REF["sigma"] - eps, "call", REF["q"])
    v_fd = (c_up - c_dn) / (2 * eps)
    assert v == pytest.approx(v_fd, rel=1e-3, abs=1e-3)


def test_theta_fd_sign_for_long_call():
    th = theta(**REF, option_type="call")
    # Long ATM call usually has negative calendar theta in year units
    assert th < 0
    eps = 1e-5
    c_now = price(**REF, option_type="call")
    c_later = price(
        REF["S"], REF["K"], REF["T"] - eps, REF["r"], REF["sigma"], "call", REF["q"]
    )
    # dV/dT ~= (V(T) - V(T-eps)) / eps ; theta is dV/dt with t^ reducing T
    # BSM theta = dV/dt where T shrinks as calendar advances: roughly (V(T-eps)-V(T))/eps
    th_fd = (c_later - c_now) / eps
    assert th == pytest.approx(th_fd, rel=5e-2, abs=0.05)


def test_greeks_bundle():
    g = greeks(**REF, option_type="call")
    assert math.isfinite(g.price)
    assert g.gamma > 0
    assert g.vega > 0


def test_expiry_intrinsic():
    assert price(110, 100, 0.0, 0.05, 0.2, "call") == pytest.approx(10.0)
    assert price(90, 100, 0.0, 0.05, 0.2, "put") == pytest.approx(10.0)
    assert price(90, 100, 0.0, 0.05, 0.2, "call") == pytest.approx(0.0)
