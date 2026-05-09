"""Broker abstraction — the same API for paper, demo, and live trading.

Why this lives in a separate module:
    The bot must be broker-agnostic. Today we paper-trade via an in-memory
    broker; in Phase 7 we plug in Zerodha Kite Connect; later we may also
    plug in Upstox or Angel One. Strategies, risk manager, and dashboards
    don't care which — they only see this interface.

Order lifecycle (simplified)::

    Order(NEW) ──submit()──► (queued) ──fill──► Fill(s) → COMPLETE
                              ⤷ cancel() ──► CANCELLED
                              ⤷ reject ────► REJECTED

We keep idempotency keys (``client_order_id``) so retries are safe — if
the bot crashes mid-submit and re-submits, the broker recognises the same
``client_order_id`` and returns the existing order rather than duplicating it.

References:
    - Chan §5 — "Use idempotent order submission. Retries should never
      double-place an order."
    - Zerodha Kite Connect REST API documentation (for parity).
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum


class OrderStatus(StrEnum):
    NEW = "new"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class OrderType(StrEnum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


@dataclass(frozen=True)
class Order:
    """Immutable order intent. The broker may amend status fields elsewhere."""

    client_order_id: str  # idempotency key
    ticker: str
    side: str  # "buy" / "sell"
    quantity: int
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    stop_price: float | None = None
    strategy_name: str | None = None
    submitted_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict = field(default_factory=dict)

    @staticmethod
    def new_id() -> str:
        return f"clord-{uuid.uuid4().hex[:16]}"


@dataclass(frozen=True)
class Fill:
    """One execution event reported by the broker."""

    timestamp: datetime
    client_order_id: str
    broker_order_id: str
    ticker: str
    side: str
    quantity: int
    price: float
    cost_inr: float  # broker fees + slippage
    strategy_name: str | None = None
    stop_price: float | None = None  # so the portfolio can record the lot's stop


@dataclass
class OrderRecord:
    """The broker's mutable view of an order — status transitions live here."""

    order: Order
    broker_order_id: str
    status: OrderStatus
    filled_quantity: int = 0
    avg_fill_price: float = 0.0
    fills: list[Fill] = field(default_factory=list)
    rejection_reason: str | None = None
    last_update: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def is_terminal(self) -> bool:
        return self.status in {
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
        }


class Broker(ABC):
    """Anything that can route orders implements this."""

    def set_mark(self, ticker: str, price: float) -> None:
        """Push a mark price for paper-style simulation. Default: no-op (live brokers)."""
        return None

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def submit(self, order: Order) -> OrderRecord:
        """Idempotent: if ``order.client_order_id`` was already submitted,
        return the existing record without re-submitting."""

    @abstractmethod
    def cancel(self, client_order_id: str) -> OrderRecord:
        """Cancel an open order. Returns the (possibly already-terminal) record."""

    @abstractmethod
    def status(self, client_order_id: str) -> OrderRecord | None:
        """Look up an order by client_order_id."""

    @abstractmethod
    def open_orders(self) -> list[OrderRecord]:
        """All non-terminal orders currently with the broker."""

    @abstractmethod
    def positions(self) -> dict[str, int]:
        """Net positions {ticker: signed quantity}."""


__all__ = [
    "Broker",
    "Fill",
    "Order",
    "OrderRecord",
    "OrderStatus",
    "OrderType",
]
