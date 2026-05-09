"""Circuit-breaker tests."""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest

from src.orders.base import Fill
from src.risk.circuit_breaker import BreakerState, CircuitBreaker
from src.risk.portfolio import Portfolio


@pytest.fixture
def portfolio_with_curve() -> Portfolio:
    """A portfolio with an equity curve we control via marks."""
    p = Portfolio(cash_inr=1_000_000.0, initial_equity_inr=1_000_000.0)
    p.equity_curve[date(2024, 1, 1)] = 1_000_000.0
    p.high_watermark = 1_000_000.0
    p.high_watermark_date = date(2024, 1, 1)
    return p


def _push_equity(portfolio: Portfolio, equity: float, on: date) -> None:
    """Cheaply move equity to ``equity`` on date ``on`` for testing."""
    portfolio.cash_inr = equity
    portfolio.equity_curve[on] = equity
    if equity > portfolio.high_watermark:
        portfolio.high_watermark = equity
        portfolio.high_watermark_date = on


# ---------------------------------------------------------------------------
# Drawdown halt
# ---------------------------------------------------------------------------


def test_breaker_halts_at_12pct_drawdown(portfolio_with_curve: Portfolio) -> None:
    breaker = CircuitBreaker(max_drawdown_halt=0.12)
    # Move equity to 87% of peak → -13% DD → halt.
    _push_equity(portfolio_with_curve, 870_000.0, date(2024, 6, 1))
    decision = breaker.check(portfolio_with_curve, date(2024, 6, 1))
    assert decision.state is BreakerState.HALTED
    assert not decision.allow_new_entries
    assert decision.allow_exits


def test_breaker_does_not_halt_at_5pct_drawdown(portfolio_with_curve: Portfolio) -> None:
    """5% drawdown should NOT trigger the catastrophic halt (12% threshold).

    Use a multi-day equity decay so we don't accidentally trip the daily
    spike pause too — we want to isolate the drawdown rule.
    """
    breaker = CircuitBreaker(max_drawdown_halt=0.12, max_daily_loss=0.03, max_5day_loss=0.50)
    base = date(2024, 5, 1)
    portfolio_with_curve.equity_curve.clear()
    portfolio_with_curve.equity_curve[base] = 1_000_000.0
    portfolio_with_curve.high_watermark = 1_000_000.0
    portfolio_with_curve.high_watermark_date = base
    # Slow decline 1% per day to reach 95% over 5 days — no daily spike.
    for i, eq in enumerate([990_000.0, 980_000.0, 970_000.0, 960_000.0, 950_000.0], start=1):
        portfolio_with_curve.equity_curve[base + timedelta(days=i)] = eq
    portfolio_with_curve.cash_inr = 950_000.0
    decision = breaker.check(portfolio_with_curve, base + timedelta(days=5))
    assert decision.state is not BreakerState.HALTED
    assert decision.allow_new_entries


# ---------------------------------------------------------------------------
# Daily-loss pause
# ---------------------------------------------------------------------------


def test_breaker_pauses_day_on_3pct_daily_loss(portfolio_with_curve: Portfolio) -> None:
    breaker = CircuitBreaker(max_daily_loss=0.03)
    portfolio_with_curve.equity_curve[date(2024, 6, 1)] = 1_000_000.0
    portfolio_with_curve.equity_curve[date(2024, 6, 2)] = 960_000.0  # -4%
    portfolio_with_curve.cash_inr = 960_000.0
    decision = breaker.check(portfolio_with_curve, date(2024, 6, 2))
    assert decision.state is BreakerState.PAUSE_DAY
    assert not decision.allow_new_entries


def test_pause_day_clears_next_session(portfolio_with_curve: Portfolio) -> None:
    breaker = CircuitBreaker(max_daily_loss=0.03)
    breaker.state = BreakerState.PAUSE_DAY
    breaker.last_pause_date = date(2024, 6, 1)
    portfolio_with_curve.equity_curve[date(2024, 6, 2)] = 1_000_000.0
    decision = breaker.check(portfolio_with_curve, date(2024, 6, 2))
    assert decision.state is BreakerState.HEALTHY


# ---------------------------------------------------------------------------
# Manual kill
# ---------------------------------------------------------------------------


def test_manual_kill_is_sticky(portfolio_with_curve: Portfolio) -> None:
    breaker = CircuitBreaker()
    breaker.manual_kill_switch(date(2024, 6, 1))
    decision = breaker.check(portfolio_with_curve, date(2024, 6, 1))
    assert decision.state is BreakerState.HALTED
    assert "manual" in decision.reason


def test_reset_clears_halt(portfolio_with_curve: Portfolio) -> None:
    breaker = CircuitBreaker()
    breaker.manual_kill_switch(date(2024, 6, 1))
    breaker.reset()
    decision = breaker.check(portfolio_with_curve, date(2024, 6, 2))
    assert decision.state is BreakerState.HEALTHY


# ---------------------------------------------------------------------------
# Strategy disabling on consecutive stops
# ---------------------------------------------------------------------------


def test_three_stops_disables_strategy(portfolio_with_curve: Portfolio) -> None:
    breaker = CircuitBreaker(consecutive_stops_to_disable_strategy=3, strategy_disable_days=5)
    breaker.record_stop_hit("momentum", date(2024, 6, 1))
    breaker.record_stop_hit("momentum", date(2024, 6, 2))
    breaker.record_stop_hit("momentum", date(2024, 6, 3))
    decision = breaker.check(portfolio_with_curve, date(2024, 6, 3))
    assert "momentum" in decision.disabled_strategies


def test_winning_close_resets_streak(portfolio_with_curve: Portfolio) -> None:
    breaker = CircuitBreaker(consecutive_stops_to_disable_strategy=3)
    breaker.record_stop_hit("momentum", date(2024, 6, 1))
    breaker.record_stop_hit("momentum", date(2024, 6, 2))
    breaker.record_winning_close("momentum")
    breaker.record_stop_hit("momentum", date(2024, 6, 3))
    decision = breaker.check(portfolio_with_curve, date(2024, 6, 3))
    # Only 1 in current streak → no disable
    assert "momentum" not in decision.disabled_strategies


def test_disabled_strategy_expires_after_window(portfolio_with_curve: Portfolio) -> None:
    breaker = CircuitBreaker(consecutive_stops_to_disable_strategy=2, strategy_disable_days=3)
    breaker.record_stop_hit("momentum", date(2024, 6, 1))
    breaker.record_stop_hit("momentum", date(2024, 6, 2))
    decision = breaker.check(portfolio_with_curve, date(2024, 6, 2))
    assert "momentum" in decision.disabled_strategies
    decision_later = breaker.check(portfolio_with_curve, date(2024, 6, 10))
    assert "momentum" not in decision_later.disabled_strategies


# ---------------------------------------------------------------------------
# Rolling 5-day loss → size reduction
# ---------------------------------------------------------------------------


def test_5day_loss_halves_sizes(portfolio_with_curve: Portfolio) -> None:
    breaker = CircuitBreaker(max_drawdown_halt=0.12, max_daily_loss=0.99, max_5day_loss=0.07)
    # Build 6 dates of equity declining a lot in the last 5 days.
    base = date(2024, 6, 1)
    portfolio_with_curve.equity_curve.clear()
    portfolio_with_curve.equity_curve[base] = 1_000_000.0
    portfolio_with_curve.high_watermark = 1_000_000.0
    portfolio_with_curve.high_watermark_date = base
    for i, eq in enumerate([990_000.0, 970_000.0, 950_000.0, 930_000.0, 910_000.0], start=1):
        portfolio_with_curve.equity_curve[base + timedelta(days=i)] = eq
    portfolio_with_curve.cash_inr = 910_000.0
    decision = breaker.check(portfolio_with_curve, base + timedelta(days=5))
    assert decision.allow_new_entries
    assert decision.new_entry_size_multiplier == pytest.approx(0.5, abs=1e-9)


# ---------------------------------------------------------------------------
# Defensive
# ---------------------------------------------------------------------------


def test_breaker_handles_empty_curve() -> None:
    p = Portfolio(cash_inr=1_000_000.0, initial_equity_inr=1_000_000.0)
    breaker = CircuitBreaker()
    decision = breaker.check(p, date(2024, 1, 1))
    # No data → healthy
    assert decision.state is BreakerState.HEALTHY


# Keep imports referenced even when their helpers aren't directly used in tests.
_ = (Fill, datetime)
