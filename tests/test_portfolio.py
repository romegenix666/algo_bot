"""Portfolio book tests."""

from __future__ import annotations

from datetime import date, datetime

import pytest

from src.orders.base import Fill
from src.risk.portfolio import Portfolio


@pytest.fixture
def portfolio() -> Portfolio:
    return Portfolio(
        cash_inr=1_000_000.0,
        initial_equity_inr=1_000_000.0,
        sector_lookup={"RELIANCE.NS": "Energy", "TCS.NS": "IT", "INFY.NS": "IT"},
    )


def _fill(
    ticker: str,
    side: str,
    qty: int,
    price: float,
    cost: float = 100.0,
    when: datetime | None = None,
    stop_price: float | None = None,
    strategy_name: str = "test",
) -> Fill:
    return Fill(
        timestamp=when or datetime(2024, 1, 1, 9, 30),
        client_order_id="cli-1",
        broker_order_id="brk-1",
        ticker=ticker,
        side=side,
        quantity=qty,
        price=price,
        cost_inr=cost,
        strategy_name=strategy_name,
        stop_price=stop_price,
    )


# ---------------------------------------------------------------------------
# Buy / sell mechanics
# ---------------------------------------------------------------------------


def test_buy_creates_long_position(portfolio: Portfolio) -> None:
    f = _fill("RELIANCE.NS", "buy", 100, 1500.0, cost=200.0)
    portfolio.apply_fill(f)

    assert "RELIANCE.NS" in portfolio.positions
    pos = portfolio.positions["RELIANCE.NS"]
    assert pos.quantity == 100
    assert pos.avg_entry_price == 1500.0
    assert portfolio.cash_inr == 1_000_000.0 - 100 * 1500 - 200


def test_sell_to_close_long_realises_pnl(portfolio: Portfolio) -> None:
    portfolio.apply_fill(_fill("RELIANCE.NS", "buy", 100, 1500.0))
    portfolio.apply_fill(_fill("RELIANCE.NS", "sell", 100, 1600.0))

    # 100 * (1600 - 1500) = 10000 realised P&L
    assert portfolio.realised_pnl_total == pytest.approx(10000.0, abs=1e-6)
    # Position is closed → removed from book
    assert "RELIANCE.NS" not in portfolio.positions


def test_partial_sell_realises_partial_pnl(portfolio: Portfolio) -> None:
    portfolio.apply_fill(_fill("RELIANCE.NS", "buy", 100, 1500.0))
    portfolio.apply_fill(_fill("RELIANCE.NS", "sell", 60, 1600.0))

    # 60 * 100 = 6000 realised on the 60 sold; 40 remaining
    assert portfolio.realised_pnl_total == pytest.approx(6000.0, abs=1e-6)
    pos = portfolio.positions["RELIANCE.NS"]
    assert pos.quantity == 40


def test_fifo_close_with_multiple_lots(portfolio: Portfolio) -> None:
    """Two buys at different prices → first lot closes first."""
    portfolio.apply_fill(_fill("RELIANCE.NS", "buy", 50, 1500.0))
    portfolio.apply_fill(_fill("RELIANCE.NS", "buy", 50, 1600.0))

    # Close 60 — 50 from first lot, 10 from second.
    portfolio.apply_fill(_fill("RELIANCE.NS", "sell", 60, 1700.0))

    # Realised = 50*(1700-1500) + 10*(1700-1600) = 10000 + 1000 = 11000
    assert portfolio.realised_pnl_total == pytest.approx(11000.0, abs=1e-6)
    pos = portfolio.positions["RELIANCE.NS"]
    assert pos.quantity == 40
    assert pos.lots[0].entry_price == 1600.0  # only the second lot is left


# ---------------------------------------------------------------------------
# Mark-to-market + drawdown
# ---------------------------------------------------------------------------


def test_mark_to_market_updates_equity_curve(portfolio: Portfolio) -> None:
    portfolio.apply_fill(_fill("RELIANCE.NS", "buy", 100, 1500.0, cost=0.0))
    portfolio.mark_to_market({"RELIANCE.NS": 1550.0}, as_of=date(2024, 1, 2))
    portfolio.mark_to_market({"RELIANCE.NS": 1600.0}, as_of=date(2024, 1, 3))
    assert portfolio.equity_curve[date(2024, 1, 2)] > 0
    assert portfolio.equity_curve[date(2024, 1, 3)] > portfolio.equity_curve[date(2024, 1, 2)]


def test_high_watermark_tracks_peak(portfolio: Portfolio) -> None:
    portfolio.apply_fill(_fill("RELIANCE.NS", "buy", 100, 1500.0, cost=0.0))
    portfolio.mark_to_market({"RELIANCE.NS": 1700.0}, as_of=date(2024, 1, 2))
    portfolio.mark_to_market({"RELIANCE.NS": 1500.0}, as_of=date(2024, 1, 3))

    assert portfolio.high_watermark > 1_000_000.0
    assert portfolio.high_watermark_date == date(2024, 1, 2)
    assert portfolio.drawdown() < 0


# ---------------------------------------------------------------------------
# Sector / position weights
# ---------------------------------------------------------------------------


def test_sector_weights_aggregate_correctly(portfolio: Portfolio) -> None:
    portfolio.apply_fill(_fill("TCS.NS", "buy", 100, 3500.0, cost=0.0))
    portfolio.apply_fill(_fill("INFY.NS", "buy", 100, 1500.0, cost=0.0))
    portfolio.apply_fill(_fill("RELIANCE.NS", "buy", 100, 1500.0, cost=0.0))
    portfolio.mark_to_market(
        {"TCS.NS": 3500.0, "INFY.NS": 1500.0, "RELIANCE.NS": 1500.0},
        as_of=date(2024, 1, 2),
    )

    weights = portfolio.sector_weights()
    # IT bucket should be (TCS + INFY) market values / equity.
    assert "IT" in weights
    assert "Energy" in weights
    assert weights["IT"] > weights["Energy"]


def test_position_weight_is_signed(portfolio: Portfolio) -> None:
    portfolio.apply_fill(_fill("RELIANCE.NS", "buy", 100, 1500.0, cost=0.0))
    portfolio.mark_to_market({"RELIANCE.NS": 1500.0}, as_of=date(2024, 1, 2))
    w = portfolio.position_weight("RELIANCE.NS")
    assert w > 0
    assert w == pytest.approx(0.15, abs=0.01)  # 100*1500 / equity ≈ 15%


# ---------------------------------------------------------------------------
# Defensive
# ---------------------------------------------------------------------------


def test_unknown_ticker_weight_zero(portfolio: Portfolio) -> None:
    assert portfolio.position_weight("UNKNOWN") == 0.0


def test_to_dataframe_columns(portfolio: Portfolio) -> None:
    portfolio.apply_fill(_fill("RELIANCE.NS", "buy", 100, 1500.0, cost=0.0))
    portfolio.mark_to_market({"RELIANCE.NS": 1550.0}, as_of=date(2024, 1, 2))
    df = portfolio.to_dataframe()
    assert {"ticker", "side", "quantity", "avg_entry", "mark", "market_value"} <= set(df.columns)
