"""Tests for ``src.risk.sizer`` — Kelly, ATR stops, position sizing, trailing."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from src.risk.sizer import (
    SizingResult,
    atr_stop_price,
    estimate_win_stats,
    half_kelly_fraction,
    size_position,
    update_trailing_stop,
)
from src.strategies.base import RiskParams, Side, Signal


def test_half_kelly_win_rate_below_zero_raises() -> None:
    with pytest.raises(ValueError, match="win_rate"):
        half_kelly_fraction(-0.1, 1.5)


def test_half_kelly_win_rate_above_one_raises() -> None:
    with pytest.raises(ValueError, match="win_rate"):
        half_kelly_fraction(1.1, 1.5)


def test_half_kelly_zero_win_loss_ratio_returns_zero() -> None:
    assert half_kelly_fraction(0.6, 0.0) == 0.0


def test_half_kelly_negative_edge_clamped_to_zero() -> None:
    # win_rate low, ratio high enough that full Kelly would be negative
    f = half_kelly_fraction(0.3, 1.0, cap=0.5)
    assert f == 0.0


def test_half_kelly_positive_edge_halved() -> None:
    # b=2, p=0.6 -> full Kelly = (0.6*2 - 0.4)/2 = 0.4, half = 0.2
    assert half_kelly_fraction(0.6, 2.0, cap=1.0) == pytest.approx(0.2)


def test_half_kelly_respects_cap() -> None:
    assert half_kelly_fraction(0.9, 10.0, cap=0.05) == pytest.approx(0.05)


def test_atr_stop_long_below_entry() -> None:
    sp = atr_stop_price(entry=100.0, atr_value=2.0, side=Side.LONG, atr_multiple=2.0)
    assert sp == 96.0


def test_atr_stop_short_above_entry() -> None:
    sp = atr_stop_price(entry=100.0, atr_value=2.0, side=Side.SHORT, atr_multiple=2.0)
    assert sp == 104.0


def test_atr_stop_nonpositive_atr_raises() -> None:
    with pytest.raises(ValueError, match="ATR"):
        atr_stop_price(100.0, 0.0, Side.LONG, 2.0)


def test_atr_stop_flat_raises() -> None:
    with pytest.raises(ValueError, match="FLAT"):
        atr_stop_price(100.0, 1.0, Side.FLAT, 2.0)


def test_size_position_flat_signal_returns_none() -> None:
    risk = RiskParams(
        equity=100_000.0,
        per_trade_pct=0.01,
        half_kelly=True,
        kelly_cap=0.2,
        max_single_position_pct=0.2,
    )
    sig = Signal("X.NS", Side.FLAT, 1.0, datetime(2024, 1, 1))
    assert size_position(sig, 100.0, 2.0, 2.0, risk) is None


def test_size_position_nonpositive_entry_returns_none() -> None:
    risk = RiskParams(100_000.0, 0.01, True, 0.2, 0.2)
    sig = Signal("X.NS", Side.LONG, 1.0, datetime(2024, 1, 1))
    assert size_position(sig, 0.0, 2.0, 2.0, risk) is None


def test_size_position_zero_risk_per_share_returns_none() -> None:
    risk = RiskParams(100_000.0, 0.01, True, 0.2, 0.2)
    sig = Signal("X.NS", Side.LONG, 1.0, datetime(2024, 1, 1))
    # atr_multiple=0 -> stop at entry -> no risk per share
    assert size_position(sig, 100.0, 2.0, 0.0, risk) is None


def test_size_position_lot_rounding() -> None:
    risk = RiskParams(1_000_000.0, 0.05, True, 0.25, 0.5)
    sig = Signal("X.NS", Side.LONG, 1.0, datetime(2024, 1, 1))
    out = size_position(sig, 100.0, 1.0, 2.0, risk, lot_size=10)
    assert out is not None
    assert out.shares % 10 == 0
    assert isinstance(out, SizingResult)


def test_size_position_fraction_of_equity() -> None:
    risk = RiskParams(200_000.0, 0.02, True, 0.25, 0.5)
    sig = Signal("X.NS", Side.LONG, 1.0, datetime(2024, 1, 1))
    out = size_position(sig, 50.0, 1.0, 2.0, risk)
    assert out is not None
    assert out.fraction_of_equity == pytest.approx(out.notional / 200_000.0)


def test_update_trailing_long_unchanged_before_activation_threshold() -> None:
    entry = 100.0
    atr_v = 1.0
    initial_stop = 98.0
    # last_price barely moved — below 1.5*ATR profit
    new_stop = update_trailing_stop(
        initial_stop,
        last_price=101.0,
        atr_value=atr_v,
        atr_multiple=2.0,
        side=Side.LONG,
        activate_after_atr=1.5,
        entry_price=entry,
    )
    assert new_stop == initial_stop


def test_update_trailing_long_ratchets_when_profitable() -> None:
    entry = 100.0
    atr_v = 1.0
    initial_stop = 98.0
    last_price = 110.0  # well past activation
    new_stop = update_trailing_stop(
        initial_stop,
        last_price=last_price,
        atr_value=atr_v,
        atr_multiple=2.0,
        side=Side.LONG,
        activate_after_atr=1.5,
        entry_price=entry,
    )
    assert new_stop >= initial_stop
    assert new_stop == pytest.approx(last_price - 2.0 * atr_v)


def test_update_trailing_short_ratchets_when_profitable() -> None:
    entry = 100.0
    initial_stop = 102.0
    last_price = 90.0
    new_stop = update_trailing_stop(
        initial_stop,
        last_price=last_price,
        atr_value=1.0,
        atr_multiple=2.0,
        side=Side.SHORT,
        activate_after_atr=1.5,
        entry_price=entry,
    )
    assert new_stop <= initial_stop
    assert new_stop == pytest.approx(last_price + 2.0)


def test_update_trailing_flat_returns_unchanged() -> None:
    assert update_trailing_stop(50.0, 100.0, 1.0, 2.0, Side.FLAT, entry_price=100.0) == 50.0


def test_estimate_win_stats_empty() -> None:
    wr, r = estimate_win_stats(pd.Series(dtype=float))
    assert wr == 0.5 and r == 1.0


def test_estimate_win_stats_mixed_trades() -> None:
    rets = pd.Series([0.02, -0.01, 0.03, -0.02, 0.01])
    wr, wl = estimate_win_stats(rets)
    assert wr == pytest.approx(3 / 5)
    assert wl > 0


def test_estimate_win_stats_all_wins_returns_default_ratio_branch() -> None:
    rets = pd.Series([0.01, 0.02])
    wr, wl = estimate_win_stats(rets)
    assert wr == 0.5 and wl == 1.0


def test_estimate_win_stats_all_losses_returns_default() -> None:
    rets = pd.Series([-0.01, -0.02])
    wr, wl = estimate_win_stats(rets)
    assert wr == 0.5 and wl == 1.0
