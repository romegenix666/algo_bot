"""Paper-broker tests."""

from __future__ import annotations

from datetime import datetime

import pytest

from src.backtest.costs import IndianEquityCostModel
from src.orders.base import Order, OrderStatus, OrderType
from src.orders.paper import PaperBroker


@pytest.fixture
def broker() -> PaperBroker:
    b = PaperBroker(
        cost_model=IndianEquityCostModel(slippage_bps=5.0),
        initial_cash=1_000_000.0,
    )
    b.set_mark("RELIANCE.NS", 1500.0)
    return b


# ---------------------------------------------------------------------------
# Market order fills immediately
# ---------------------------------------------------------------------------


def test_market_buy_fills_with_slippage(broker: PaperBroker) -> None:
    order = Order(
        client_order_id="cli-1",
        ticker="RELIANCE.NS",
        side="buy",
        quantity=100,
        order_type=OrderType.MARKET,
    )
    record = broker.submit(order)
    assert record.status is OrderStatus.FILLED
    assert len(record.fills) == 1
    fill = record.fills[0]
    # Slippage = +5 bps for a buy
    assert fill.price > 1500.0
    assert fill.price == pytest.approx(1500.0 * 1.0005, rel=1e-9)


def test_market_sell_fills_with_slippage(broker: PaperBroker) -> None:
    order = Order(
        client_order_id="cli-2",
        ticker="RELIANCE.NS",
        side="sell",
        quantity=100,
    )
    record = broker.submit(order)
    fill = record.fills[0]
    # Slippage = -5 bps for a sell
    assert fill.price < 1500.0
    assert fill.price == pytest.approx(1500.0 * 0.9995, rel=1e-9)


def test_market_buy_reduces_cash(broker: PaperBroker) -> None:
    starting_cash = broker.cash
    order = Order(
        client_order_id="cli-3",
        ticker="RELIANCE.NS",
        side="buy",
        quantity=100,
    )
    broker.submit(order)
    assert broker.cash < starting_cash


def test_market_sell_increases_cash(broker: PaperBroker) -> None:
    starting_cash = broker.cash
    order = Order(
        client_order_id="cli-4",
        ticker="RELIANCE.NS",
        side="sell",
        quantity=100,
    )
    broker.submit(order)
    assert broker.cash > starting_cash


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


def test_resubmit_same_id_returns_existing_record(broker: PaperBroker) -> None:
    order = Order(client_order_id="cli-idem", ticker="RELIANCE.NS", side="buy", quantity=10)
    a = broker.submit(order)
    b = broker.submit(order)
    assert a is b
    # Only one fill
    assert len(b.fills) == 1


def test_unique_client_order_ids_are_unique() -> None:
    a = Order.new_id()
    b = Order.new_id()
    assert a != b
    assert a.startswith("clord-")


# ---------------------------------------------------------------------------
# Rejections
# ---------------------------------------------------------------------------


def test_rejection_when_no_mark_price() -> None:
    broker = PaperBroker()
    order = Order(client_order_id="cli-x", ticker="UNKNOWN.NS", side="buy", quantity=10)
    record = broker.submit(order)
    assert record.status is OrderStatus.REJECTED
    assert record.rejection_reason is not None


def test_rejection_when_circuit_band_simulated() -> None:
    broker = PaperBroker(circuit_limit_pct=0.05)
    broker.set_circuit_reference("RELIANCE.NS", 100.0)
    broker.set_mark("RELIANCE.NS", 106.0)
    order = Order(client_order_id="cli-circ", ticker="RELIANCE.NS", side="buy", quantity=10)
    record = broker.submit(order)
    assert record.status is OrderStatus.REJECTED
    assert record.rejection_reason is not None
    assert "circuit_band_sim" in record.rejection_reason


# ---------------------------------------------------------------------------
# Limit + Stop orders
# ---------------------------------------------------------------------------


def test_limit_buy_does_not_fill_above_limit(broker: PaperBroker) -> None:
    order = Order(
        client_order_id="cli-lim",
        ticker="RELIANCE.NS",
        side="buy",
        quantity=10,
        order_type=OrderType.LIMIT,
        limit_price=1400.0,  # mark is 1500 — limit too low
    )
    record = broker.submit(order)
    assert record.status is OrderStatus.SUBMITTED  # still resting
    assert len(record.fills) == 0


def test_limit_buy_fills_when_price_drops_to_limit(broker: PaperBroker) -> None:
    order = Order(
        client_order_id="cli-lim2",
        ticker="RELIANCE.NS",
        side="buy",
        quantity=10,
        order_type=OrderType.LIMIT,
        limit_price=1450.0,
    )
    broker.submit(order)
    fills = broker.tick("RELIANCE.NS", 1440.0, datetime(2024, 1, 1, 9, 30))
    assert len(fills) == 1
    record = broker.status("cli-lim2")
    assert record is not None
    assert record.status is OrderStatus.FILLED


def test_stop_sell_triggers_below_stop(broker: PaperBroker) -> None:
    """Long stop-loss: triggers when mark falls AT OR BELOW stop_price."""
    # First go long
    broker.submit(Order(client_order_id="cli-buy", ticker="RELIANCE.NS", side="buy", quantity=10))
    # Place a sell-stop at 1450
    order = Order(
        client_order_id="cli-stop",
        ticker="RELIANCE.NS",
        side="sell",
        quantity=10,
        order_type=OrderType.STOP,
        stop_price=1450.0,
    )
    broker.submit(order)
    # Price drops → stop should trigger
    fills = broker.tick("RELIANCE.NS", 1440.0, datetime(2024, 1, 1, 15, 0))
    record = broker.status("cli-stop")
    assert record is not None
    assert record.status is OrderStatus.FILLED
    assert len(fills) == 1


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------


def test_cancel_open_order(broker: PaperBroker) -> None:
    order = Order(
        client_order_id="cli-cancel",
        ticker="RELIANCE.NS",
        side="buy",
        quantity=10,
        order_type=OrderType.LIMIT,
        limit_price=1300.0,
    )
    broker.submit(order)
    rec = broker.cancel("cli-cancel")
    assert rec.status is OrderStatus.CANCELLED


def test_cancel_filled_order_is_noop(broker: PaperBroker) -> None:
    order = Order(client_order_id="cli-filled", ticker="RELIANCE.NS", side="buy", quantity=10)
    broker.submit(order)
    rec = broker.cancel("cli-filled")
    assert rec.status is OrderStatus.FILLED  # was already filled


def test_cancel_unknown_raises(broker: PaperBroker) -> None:
    with pytest.raises(KeyError):
        broker.cancel("not-real")


# ---------------------------------------------------------------------------
# Open orders + positions
# ---------------------------------------------------------------------------


def test_open_orders_excludes_filled(broker: PaperBroker) -> None:
    broker.submit(Order(client_order_id="cli-a", ticker="RELIANCE.NS", side="buy", quantity=10))
    broker.submit(
        Order(
            client_order_id="cli-b",
            ticker="RELIANCE.NS",
            side="buy",
            quantity=10,
            order_type=OrderType.LIMIT,
            limit_price=1.0,
        )
    )
    open_orders = broker.open_orders()
    assert len(open_orders) == 1
    assert open_orders[0].order.client_order_id == "cli-b"


def test_positions_aggregate_correctly(broker: PaperBroker) -> None:
    broker.submit(Order(client_order_id="cli-1", ticker="RELIANCE.NS", side="buy", quantity=10))
    broker.submit(Order(client_order_id="cli-2", ticker="RELIANCE.NS", side="buy", quantity=15))
    broker.submit(Order(client_order_id="cli-3", ticker="RELIANCE.NS", side="sell", quantity=5))
    positions = broker.positions()
    assert positions["RELIANCE.NS"] == 20
