"""Risk manager + order router integration tests."""

from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest

from src.backtest.costs import IndianEquityCostModel
from src.orders.paper import PaperBroker
from src.orders.router import OrderRouter
from src.risk.circuit_breaker import CircuitBreaker, circuit_breaker_from_settings
from src.risk.manager import RiskLimits, RiskManager, TradeRequest, risk_limits_from_settings
from src.risk.portfolio import Portfolio
from src.strategies.base import Side, Signal

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fresh_setup() -> tuple[Portfolio, RiskManager, OrderRouter, PaperBroker]:
    portfolio = Portfolio(
        cash_inr=1_000_000.0,
        initial_equity_inr=1_000_000.0,
        sector_lookup={"RELIANCE.NS": "Energy", "TCS.NS": "IT", "INFY.NS": "IT"},
    )
    portfolio.equity_curve[date(2024, 1, 1)] = 1_000_000.0
    portfolio.high_watermark = 1_000_000.0
    portfolio.high_watermark_date = date(2024, 1, 1)

    risk = RiskManager(
        limits=RiskLimits(
            per_trade_pct=0.02,  # generous for testing
            max_single_position_pct=0.20,
            max_sector_pct=0.35,
            cash_buffer_pct=0.05,
        ),
        breaker=CircuitBreaker(),
    )
    broker = PaperBroker(
        cost_model=IndianEquityCostModel(slippage_bps=5.0),
        initial_cash=1_000_000.0,
    )
    broker.set_mark("RELIANCE.NS", 1500.0)
    broker.set_mark("TCS.NS", 3500.0)
    broker.set_mark("INFY.NS", 1500.0)
    router = OrderRouter(broker=broker, risk_manager=risk, portfolio=portfolio)
    return portfolio, risk, router, broker


def _ohlc_history(close_level: float, n: int = 60) -> pd.DataFrame:
    """Synthetic OHLC for ATR computation."""
    rng = np.random.default_rng(0)
    rets = rng.normal(0.0, 0.012, n)
    close = close_level * np.exp(np.cumsum(rets))
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.005,
            "low": close * 0.995,
            "close": close,
        },
        index=idx,
    )


# ---------------------------------------------------------------------------
# Risk Manager — direct approve()
# ---------------------------------------------------------------------------


def test_request_approved_within_caps(
    fresh_setup: tuple[Portfolio, RiskManager, OrderRouter, PaperBroker],
) -> None:
    portfolio, risk, _router, _broker = fresh_setup
    request = TradeRequest(
        signal=Signal(
            ticker="RELIANCE.NS",
            side=Side.LONG,
            conviction=1.0,
            timestamp=datetime(2024, 1, 5),
        ),
        quantity=100,
        entry_price=1500.0,
        stop_price=1450.0,  # ₹50 risk per share
        strategy_name="momentum",
    )
    decision = risk.approve(request, portfolio, today=date(2024, 1, 5))
    assert decision.approved
    assert decision.approved_quantity > 0


def test_oversize_request_capped_to_position_limit(
    fresh_setup: tuple[Portfolio, RiskManager, OrderRouter, PaperBroker],
) -> None:
    """Asking for 1000 shares of ₹1500 stock = ₹15L = 150% of equity. Must cap."""
    portfolio, risk, _router, _broker = fresh_setup
    request = TradeRequest(
        signal=Signal(
            ticker="RELIANCE.NS",
            side=Side.LONG,
            conviction=1.0,
            timestamp=datetime(2024, 1, 5),
        ),
        quantity=1000,
        entry_price=1500.0,
        stop_price=1450.0,
        strategy_name="momentum",
    )
    decision = risk.approve(request, portfolio, today=date(2024, 1, 5))
    if decision.approved:
        # If approved, must be capped well below 1000.
        assert decision.approved_quantity < 1000


def test_invalid_request_rejected(
    fresh_setup: tuple[Portfolio, RiskManager, OrderRouter, PaperBroker],
) -> None:
    portfolio, risk, _router, _broker = fresh_setup
    request = TradeRequest(
        signal=Signal(
            ticker="RELIANCE.NS",
            side=Side.LONG,
            conviction=1.0,
            timestamp=datetime(2024, 1, 5),
        ),
        quantity=0,
        entry_price=1500.0,
        stop_price=1450.0,
        strategy_name="momentum",
    )
    decision = risk.approve(request, portfolio, today=date(2024, 1, 5))
    assert not decision.approved
    assert decision.reason == "invalid_request"


def test_zero_stop_distance_rejected(
    fresh_setup: tuple[Portfolio, RiskManager, OrderRouter, PaperBroker],
) -> None:
    portfolio, risk, _router, _broker = fresh_setup
    request = TradeRequest(
        signal=Signal(
            ticker="RELIANCE.NS",
            side=Side.LONG,
            conviction=1.0,
            timestamp=datetime(2024, 1, 5),
        ),
        quantity=100,
        entry_price=1500.0,
        stop_price=1500.0,  # SAME as entry — zero risk per share
        strategy_name="momentum",
    )
    decision = risk.approve(request, portfolio, today=date(2024, 1, 5))
    assert not decision.approved
    assert decision.reason == "bad_stop"


def test_drawdown_halts_new_entries(
    fresh_setup: tuple[Portfolio, RiskManager, OrderRouter, PaperBroker],
) -> None:
    portfolio, risk, _router, _broker = fresh_setup
    # Force -15% drawdown
    portfolio.cash_inr = 850_000.0
    portfolio.equity_curve[date(2024, 6, 1)] = 850_000.0

    request = TradeRequest(
        signal=Signal(
            ticker="RELIANCE.NS",
            side=Side.LONG,
            conviction=1.0,
            timestamp=datetime(2024, 6, 1),
        ),
        quantity=10,
        entry_price=1500.0,
        stop_price=1450.0,
        strategy_name="momentum",
    )
    decision = risk.approve(request, portfolio, today=date(2024, 6, 1))
    assert not decision.approved
    assert decision.reason == "breaker"


# ---------------------------------------------------------------------------
# Order Router — full pipeline
# ---------------------------------------------------------------------------


def test_router_routes_long_signal_through_to_fill(
    fresh_setup: tuple[Portfolio, RiskManager, OrderRouter, PaperBroker],
) -> None:
    portfolio, _risk, router, _broker = fresh_setup
    history = _ohlc_history(close_level=1500.0)
    sig = Signal(
        ticker="RELIANCE.NS",
        side=Side.LONG,
        conviction=0.7,
        timestamp=datetime(2024, 1, 5),
    )
    result = router.route_signal(
        signal=sig,
        prices_history=history,
        today=date(2024, 1, 5),
        strategy_name="momentum",
    )
    assert result.approval is not None and result.approval.approved
    assert result.record is not None
    # Position should now exist in the portfolio book.
    assert "RELIANCE.NS" in portfolio.positions
    assert portfolio.positions["RELIANCE.NS"].quantity > 0


def test_router_retry_same_signal_idempotent(
    fresh_setup: tuple[Portfolio, RiskManager, OrderRouter, PaperBroker],
) -> None:
    portfolio, _risk, router, _broker = fresh_setup
    history = _ohlc_history(close_level=1500.0)
    sig = Signal(
        ticker="RELIANCE.NS",
        side=Side.LONG,
        conviction=0.7,
        timestamp=datetime(2024, 1, 5),
    )
    r1 = router.route_signal(
        signal=sig,
        prices_history=history,
        today=date(2024, 1, 5),
        strategy_name="momentum",
    )
    r2 = router.route_signal(
        signal=sig,
        prices_history=history,
        today=date(2024, 1, 5),
        strategy_name="momentum",
    )
    assert r1.record is not None and r2.record is not None
    assert r1.record is r2.record
    q1 = r1.record.fills[0].quantity
    assert portfolio.positions["RELIANCE.NS"].quantity == q1


def test_router_rejects_short_in_long_only(
    fresh_setup: tuple[Portfolio, RiskManager, OrderRouter, PaperBroker],
) -> None:
    _portfolio, _risk, router, _broker = fresh_setup
    history = _ohlc_history(close_level=1500.0)
    sig = Signal(
        ticker="RELIANCE.NS",
        side=Side.SHORT,
        conviction=0.7,
        timestamp=datetime(2024, 1, 5),
    )
    result = router.route_signal(
        signal=sig,
        prices_history=history,
        today=date(2024, 1, 5),
        strategy_name="momentum",
    )
    assert result.rejected_reason == "short_disabled_long_only"


def test_kill_switch_exits_all(
    fresh_setup: tuple[Portfolio, RiskManager, OrderRouter, PaperBroker],
) -> None:
    portfolio, _risk, router, _broker = fresh_setup
    # First open a position
    history = _ohlc_history(close_level=1500.0)
    router.route_signal(
        signal=Signal(
            ticker="RELIANCE.NS",
            side=Side.LONG,
            conviction=0.7,
            timestamp=datetime(2024, 1, 5),
        ),
        prices_history=history,
        today=date(2024, 1, 5),
        strategy_name="momentum",
    )
    assert "RELIANCE.NS" in portfolio.positions
    # Hit the kill switch
    router.kill_switch(last_marks={"RELIANCE.NS": 1500.0})
    # Position should be closed; breaker halted
    assert "RELIANCE.NS" not in portfolio.positions
    assert router.risk_manager.breaker.manual_kill is True


def test_router_aborts_on_short_history(
    fresh_setup: tuple[Portfolio, RiskManager, OrderRouter, PaperBroker],
) -> None:
    _portfolio, _risk, router, _broker = fresh_setup
    short_history = _ohlc_history(close_level=1500.0, n=10)
    sig = Signal(
        ticker="RELIANCE.NS",
        side=Side.LONG,
        conviction=0.7,
        timestamp=datetime(2024, 1, 5),
    )
    result = router.route_signal(
        signal=sig,
        prices_history=short_history,
        today=date(2024, 1, 5),
        strategy_name="momentum",
    )
    assert result.rejected_reason == "insufficient_history_for_atr"


def test_risk_limits_from_settings_is_valid() -> None:
    lim = risk_limits_from_settings()
    assert isinstance(lim, RiskLimits)
    assert 0 < lim.per_trade_pct < 1
    assert 0 < lim.max_single_position_pct <= 1


def test_circuit_breaker_from_settings_reads_risk_yaml(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.utils import settings as settings_mod

    real_settings = settings_mod.settings

    class _FakeSettings:
        def get(self, key: str, default=None):
            if key == "risk":
                return {
                    "portfolio_max_drawdown": 0.15,
                    "circuit_daily_loss_pause": 0.04,
                    "circuit_5day_loss_half_size": 0.09,
                    "circuit_consecutive_stops_to_disable": 4,
                    "circuit_strategy_disable_days": 7,
                }
            return real_settings.get(key, default=default)

    monkeypatch.setattr(settings_mod, "settings", _FakeSettings())
    cb = circuit_breaker_from_settings()
    assert cb.max_drawdown_halt == 0.15
    assert cb.max_daily_loss == 0.04
    assert cb.max_5day_loss == 0.09
    assert cb.consecutive_stops_to_disable_strategy == 4
    assert cb.strategy_disable_days == 7
