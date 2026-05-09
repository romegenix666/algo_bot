"""Dry-run broker wrapper — logs orders without sending to the exchange."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.orders.base import (
    Broker,
    Order,
    OrderRecord,
    OrderStatus,
    OrderType,
)
from src.utils.logging import logger


@dataclass
class DryRunBroker(Broker):
    """Delegates marks and position queries to ``inner``; blocks real ``submit``."""

    inner: Broker
    _seen: set[str] = field(default_factory=set)

    @property
    def name(self) -> str:
        return f"dry_run({self.inner.name})"

    def set_mark(self, ticker: str, price: float) -> None:
        self.inner.set_mark(ticker, price)

    def submit(self, order: Order) -> OrderRecord:
        key = order.client_order_id
        if key in self._seen:
            logger.info("DRY-RUN idempotent skip {}", key)
            return OrderRecord(
                order=order,
                broker_order_id=f"dry-{key[:12]}",
                status=OrderStatus.REJECTED,
                rejection_reason="dry_run_duplicate",
            )
        self._seen.add(key)
        logger.warning(
            "DRY-RUN — would {} {} {} qty={} (strategy={})",
            order.side.upper(),
            order.ticker,
            order.order_type.value,
            order.quantity,
            order.strategy_name,
        )
        return OrderRecord(
            order=order,
            broker_order_id=f"dry-{key[:12]}",
            status=OrderStatus.REJECTED,
            rejection_reason="dry_run",
        )

    def cancel(self, client_order_id: str) -> OrderRecord:
        logger.warning("DRY-RUN cancel {}", client_order_id)
        placeholder = Order(
            client_order_id=client_order_id,
            ticker="DRYRUN",
            side="buy",
            quantity=0,
            order_type=OrderType.MARKET,
        )
        return OrderRecord(
            order=placeholder,
            broker_order_id="dry",
            status=OrderStatus.CANCELLED,
        )

    def status(self, client_order_id: str) -> OrderRecord | None:
        return self.inner.status(client_order_id)

    def open_orders(self) -> list[OrderRecord]:
        return self.inner.open_orders()

    def positions(self) -> dict[str, int]:
        return self.inner.positions()


__all__ = ["DryRunBroker"]
