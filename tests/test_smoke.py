"""Smoke tests — verify the project skeleton imports cleanly and core types behave.

These are intentionally tiny. They exist so CI has something green to run on
every commit; richer tests come per-component as we build.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from src.strategies.base import (
    ExitDecision,
    ExitReason,
    MarketState,
    Position,
    RiskParams,
    Side,
    Signal,
    Strategy,
)
from src.utils.settings import settings


def test_settings_loads_with_defaults() -> None:
    assert settings.mode in {"paper", "demo", "live"}
    assert settings.get("market", "exchange") == "NSE"
    assert isinstance(settings.get("universe", "size"), int)
    assert settings.get("universe", "size") == 500


def test_settings_dotted_get_returns_default_for_missing_key() -> None:
    assert settings.get("not", "a", "real", "key", default="fallback") == "fallback"


def test_signal_validates_conviction_range() -> None:
    sig = Signal(
        ticker="RELIANCE.NS",
        side=Side.LONG,
        conviction=0.7,
        timestamp=datetime(2024, 1, 1),
    )
    assert sig.ticker == "RELIANCE.NS"
    assert sig.side is Side.LONG

    with pytest.raises(ValueError):
        Signal(
            ticker="X.NS",
            side=Side.LONG,
            conviction=1.5,
            timestamp=datetime(2024, 1, 1),
        )

    with pytest.raises(ValueError):
        Signal(
            ticker="X.NS",
            side=Side.SHORT,
            conviction=-0.1,
            timestamp=datetime(2024, 1, 1),
        )


def test_strategy_is_abstract_and_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        Strategy()  # type: ignore[abstract]


def test_half_kelly_helper_caps_and_clamps() -> None:
    # win_rate 60%, win/loss ratio 2 → full Kelly 0.4 → half 0.2
    assert Strategy._half_kelly_fraction(0.6, 2.0, cap=0.5) == pytest.approx(0.2)
    # negative edge → zero
    assert Strategy._half_kelly_fraction(0.3, 1.0, cap=0.5) == 0.0
    # cap respected
    assert Strategy._half_kelly_fraction(0.9, 5.0, cap=0.1) == 0.1


def test_exit_decision_and_position_construct() -> None:
    pos = Position(
        ticker="TCS.NS",
        side=Side.LONG,
        quantity=10,
        entry_price=3500.0,
        entry_time=datetime(2024, 1, 1, 9, 30),
        initial_stop=3400.0,
        current_stop=3450.0,
        strategy_name="momentum",
    )
    assert pos.ticker == "TCS.NS"
    assert pos.current_stop > pos.initial_stop  # trailing stop tightened

    market = MarketState(
        timestamp=datetime(2024, 1, 5, 10, 0),
        last_price=3600.0,
        atr=50.0,
        realised_vol=0.25,
        regime="trending_low_vol",
    )
    decision = ExitDecision(should_exit=False)
    assert decision.should_exit is False
    assert market.last_price > pos.current_stop


def test_risk_params_holds_expected_fields() -> None:
    rp = RiskParams(
        equity=1_000_000.0,
        per_trade_pct=0.01,
        half_kelly=True,
        kelly_cap=0.20,
        max_single_position_pct=0.20,
    )
    assert rp.equity == 1_000_000.0
    assert rp.half_kelly is True


def test_exit_reason_has_expected_members() -> None:
    expected = {
        "SIGNAL",
        "STOP_LOSS",
        "TRAIL_STOP",
        "TAKE_PROFIT",
        "TIME_STOP",
        "REGIME",
        "CIRCUIT",
        "MANUAL",
    }
    assert {member.name for member in ExitReason} == expected
