"""Indian equity cost-model tests — verify against hand-calculated values.

Reference (Zerodha equity-delivery, 2024):

    For a buy of ₹1,00,000 of any equity:
        Brokerage  : ₹20    (capped at ₹20, would otherwise be ₹30 = 0.03%)
        STT        : ₹100   (0.10% of buy notional)
        Exchange   : ₹3.25  (0.00325%)
        GST        : (₹20 + ₹3.25) × 18% = ₹4.185
        SEBI       : ₹0.10  (₹10 / Cr)
        Stamp duty : ₹15    (0.015%)
        Slippage   : ₹50    (5 bps default)
        ─────────────────────────────────
        TOTAL      : ₹192.535

    For a sell of ₹1,00,000:
        Brokerage  : ₹20
        STT        : ₹100
        Exchange   : ₹3.25
        GST        : ₹4.185
        SEBI       : ₹0.10
        Stamp duty : ₹0      (sell side)
        Slippage   : ₹50
        ─────────────────────────────────
        TOTAL      : ₹177.535

    Round-trip on ₹1,00,000 ≈ ₹370.07
"""

from __future__ import annotations

import pytest

from src.backtest.costs import (
    ILLIQUID_COST_MODEL,
    LIQUID_COST_MODEL,
    IndianEquityCostModel,
)


@pytest.fixture
def model() -> IndianEquityCostModel:
    return IndianEquityCostModel()  # all defaults


# ---------------------------------------------------------------------------
# Hand-calculated round-trip on ₹1 Lakh
# ---------------------------------------------------------------------------


def test_buy_leg_matches_hand_calculation(model: IndianEquityCostModel) -> None:
    breakdown = model.apply(notional=100_000, side="buy")
    # Each component:
    assert breakdown.brokerage == pytest.approx(20.0)  # capped
    assert breakdown.stt == pytest.approx(100.0)  # 0.10%
    assert breakdown.exchange == pytest.approx(3.25)  # 0.00325%
    assert breakdown.gst == pytest.approx((20 + 3.25) * 0.18)
    assert breakdown.sebi == pytest.approx(0.10)  # ₹10 / Cr
    assert breakdown.stamp_duty == pytest.approx(15.0)  # 0.015%
    assert breakdown.slippage == pytest.approx(50.0)  # 5 bps
    # Total
    assert breakdown.total == pytest.approx(192.535, abs=0.005)


def test_sell_leg_matches_hand_calculation(model: IndianEquityCostModel) -> None:
    breakdown = model.apply(notional=100_000, side="sell")
    # Sell side has no stamp duty
    assert breakdown.stamp_duty == 0.0
    # Other components same as buy
    assert breakdown.brokerage == pytest.approx(20.0)
    assert breakdown.stt == pytest.approx(100.0)
    assert breakdown.total == pytest.approx(177.535, abs=0.005)


def test_round_trip_costs_about_0_37_pct(model: IndianEquityCostModel) -> None:
    rt = model.round_trip(100_000)
    # ~₹370 on ₹1L round-trip → 0.37% of one-leg
    assert 360 < rt < 400
    # As a percentage of one-leg notional:
    assert 0.0036 < rt / 100_000 < 0.004


# ---------------------------------------------------------------------------
# Brokerage cap behaviour
# ---------------------------------------------------------------------------


def test_brokerage_capped_at_20_for_large_notional(model: IndianEquityCostModel) -> None:
    # 0.03% of ₹10L = ₹300 — must cap at ₹20
    breakdown = model.apply(10_00_000, "buy")
    assert breakdown.brokerage == 20.0


def test_brokerage_uncapped_for_small_notional(model: IndianEquityCostModel) -> None:
    # 0.03% of ₹10k = ₹3 — well below cap
    breakdown = model.apply(10_000, "buy")
    assert breakdown.brokerage == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Slippage variants
# ---------------------------------------------------------------------------


def test_liquid_model_has_lower_slippage() -> None:
    a = LIQUID_COST_MODEL.apply(100_000, "buy")
    b = ILLIQUID_COST_MODEL.apply(100_000, "buy")
    assert b.slippage > a.slippage
    assert b.total > a.total


def test_slippage_scales_linearly_with_notional(model: IndianEquityCostModel) -> None:
    a = model.apply(100_000, "buy").slippage
    b = model.apply(200_000, "buy").slippage
    assert b == pytest.approx(2 * a)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_negative_notional_raises(model: IndianEquityCostModel) -> None:
    with pytest.raises(ValueError):
        model.apply(-1, "buy")


def test_invalid_side_raises(model: IndianEquityCostModel) -> None:
    with pytest.raises(ValueError):
        model.apply(100_000, "hedge")


def test_total_helper_matches_breakdown_total(model: IndianEquityCostModel) -> None:
    n = 75_000
    assert model.total(n, "buy") == pytest.approx(model.apply(n, "buy").total)
    assert model.total(n, "sell") == pytest.approx(model.apply(n, "sell").total)
