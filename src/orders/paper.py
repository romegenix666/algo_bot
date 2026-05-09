"""In-memory paper broker — drop-in replacement for a real broker.

Used for:
    - Phase 5–6 paper trading (the 30-day mandatory gauntlet).
    - Backtesting in event-driven mode.
    - Local testing without hitting a real exchange.

Fill model:
    - **Market orders** fill immediately at ``mark_price`` ± slippage.
      Slippage is configurable in basis points; default 5 bps each way.
    - **Limit orders** fill only if mark_price is at or beyond limit_price.
      No partial fills — Indian exchanges fill in one go for retail-size
      delivery orders almost always.
    - **Stop orders** convert to market when ``mark_price`` crosses
      ``stop_price``.

Cost model:
    - Reuses ``IndianEquityCostModel`` for fees + STT + GST + stamp duty.
    - Slippage is added on top so the simulated fill price is realistic.

Idempotency:
    - Re-submitting the same ``client_order_id`` returns the existing
      record (Chan §5 rule).

What we *don't* simulate:
    - Order book queue position / partial fills (immaterial for our daily
      bar cadence on liquid Nifty 500 stocks).
    - Pre-market / post-market behaviour.

Optional **circuit band** (coarse NSE-style proxy): if ``circuit_limit_pct > 0``
and a prior close was registered via ``set_circuit_reference``, market orders
reject when ``|mark - prior_close| / prior_close`` is already at or beyond the
band (single threshold; real NSE uses tiered 5/10/20% rules per symbol).

Tests cover: fill correctness, idempotency, cancel semantics, stop trigger.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from src.backtest.costs import IndianEquityCostModel
from src.orders.base import (
    Broker,
    Fill,
    Order,
    OrderRecord,
    OrderStatus,
    OrderType,
)
from src.utils.logging import logger


@dataclass
class PaperBroker(Broker):
    """A toy broker for paper trading. Marks must be pushed in via ``set_mark``."""

    cost_model: IndianEquityCostModel = field(default_factory=IndianEquityCostModel)
    initial_cash: float = 1_000_000.0
    circuit_limit_pct: float = 0.0  # 0 = off; e.g. 0.2 = reject if |mark-ref|/ref >= 20%

    # State
    cash: float = field(init=False)
    last_marks: dict[str, float] = field(default_factory=dict)
    circuit_reference_close: dict[str, float] = field(default_factory=dict)
    records: dict[str, OrderRecord] = field(default_factory=dict)
    _broker_seq: int = 0

    def __post_init__(self) -> None:
        self.cash = self.initial_cash

    # ----------------------------------------------------------------
    @property
    def name(self) -> str:
        return "paper"

    # ----------------------------------------------------------------
    def set_mark(self, ticker: str, price: float) -> None:
        """Push the latest mark price for ``ticker`` (typically last close)."""
        if price <= 0:
            return
        self.last_marks[ticker.upper()] = float(price)

    def set_circuit_reference(self, ticker: str, prev_close: float) -> None:
        """Register prior session close for optional circuit-band simulation."""
        if prev_close <= 0:
            return
        self.circuit_reference_close[ticker.upper()] = float(prev_close)

    # ----------------------------------------------------------------
    def submit(self, order: Order) -> OrderRecord:
        """Idempotent submit: returns existing record if already accepted."""
        existing = self.records.get(order.client_order_id)
        if existing is not None:
            logger.info(
                "Paper broker: idempotent re-submit of {} (status={})",
                order.client_order_id,
                existing.status,
            )
            return existing

        self._broker_seq += 1
        record = OrderRecord(
            order=order,
            broker_order_id=f"paper-{self._broker_seq:08d}",
            status=OrderStatus.SUBMITTED,
        )
        self.records[order.client_order_id] = record

        # Try to fill immediately for MARKET orders.
        if order.order_type is OrderType.MARKET:
            self._try_fill(record)
        # LIMIT / STOP orders sit in the book until ``tick``.
        return record

    # ----------------------------------------------------------------
    def cancel(self, client_order_id: str) -> OrderRecord:
        record = self.records.get(client_order_id)
        if record is None:
            raise KeyError(f"Unknown client_order_id: {client_order_id}")
        if record.is_terminal:
            return record
        record.status = OrderStatus.CANCELLED
        record.last_update = datetime.now(UTC)
        return record

    # ----------------------------------------------------------------
    def status(self, client_order_id: str) -> OrderRecord | None:
        return self.records.get(client_order_id)

    # ----------------------------------------------------------------
    def open_orders(self) -> list[OrderRecord]:
        return [r for r in self.records.values() if not r.is_terminal]

    # ----------------------------------------------------------------
    def positions(self) -> dict[str, int]:
        """Aggregate net positions from all filled orders.

        Long-only convention: we don't simulate margin / shorts here for
        the equity-delivery use case. SHORT signals from strategies are
        coerced to "exit existing long" elsewhere.
        """
        net: dict[str, int] = {}
        for record in self.records.values():
            for fill in record.fills:
                sign = 1 if fill.side == "buy" else -1
                net[fill.ticker] = net.get(fill.ticker, 0) + sign * fill.quantity
        return {k: v for k, v in net.items() if v != 0}

    # ----------------------------------------------------------------
    def tick(self, ticker: str, mark_price: float, when: datetime) -> list[Fill]:
        """Simulate one bar. Marks the price and tries to fill any
        resting orders (limits / stops) that have been triggered.

        Returns the list of fills that occurred this tick (so the caller
        can route them to the portfolio book).
        """
        ticker = ticker.upper()
        self.set_mark(ticker, mark_price)
        fills: list[Fill] = []

        for record in list(self.records.values()):
            if record.is_terminal:
                continue
            if record.order.ticker.upper() != ticker:
                continue

            order = record.order
            if (
                order.order_type is OrderType.LIMIT and self._limit_triggered(order, mark_price)
            ) or (order.order_type is OrderType.STOP and self._stop_triggered(order, mark_price)):
                new_fills = self._fill_at(record, mark_price, when, idempotent=False)
                fills.extend(new_fills)

        return fills

    # ----------------------------------------------------------------
    # Internals
    # ----------------------------------------------------------------
    def _try_fill(self, record: OrderRecord) -> None:
        """For market orders: try to fill at the current mark."""
        order = record.order
        mark = self.last_marks.get(order.ticker.upper())
        if mark is None or mark <= 0:
            record.status = OrderStatus.REJECTED
            record.rejection_reason = f"no mark price for {order.ticker}"
            logger.warning("Paper broker: rejecting {} — no mark", order.client_order_id)
            return
        ref = self.circuit_reference_close.get(order.ticker.upper())
        if (
            self.circuit_limit_pct > 0
            and ref is not None
            and ref > 0
            and abs(mark - ref) / ref >= self.circuit_limit_pct
        ):
            record.status = OrderStatus.REJECTED
            record.rejection_reason = (
                f"circuit_band_sim |mark-ref|/ref>={self.circuit_limit_pct:.4f} "
                f"(ref={ref:.4f} mark={mark:.4f})"
            )
            logger.warning("Paper broker: rejecting {} — {}", order.client_order_id, record.rejection_reason)
            return
        self._fill_at(record, mark, datetime.now(UTC), idempotent=True)

    def _fill_at(
        self,
        record: OrderRecord,
        mark_price: float,
        when: datetime,
        idempotent: bool,
    ) -> list[Fill]:
        """Apply slippage + costs and produce one Fill."""
        if record.is_terminal and idempotent:
            return record.fills.copy()

        order = record.order
        side = order.side.lower()
        # Slippage works against us: buy at higher, sell at lower.
        slippage_pct = self.cost_model.slippage_bps / 1e4
        if side == "buy":
            fill_price = mark_price * (1 + slippage_pct)
        else:
            fill_price = mark_price * (1 - slippage_pct)

        notional = fill_price * order.quantity
        fee_breakdown = self.cost_model.apply(notional, side)
        # Subtract slippage from fee, since we already baked it into the price:
        all_in_fees = fee_breakdown.total - fee_breakdown.slippage
        cost_inr = all_in_fees + fee_breakdown.slippage  # back to total — keeps API symmetric

        fill = Fill(
            timestamp=when,
            client_order_id=order.client_order_id,
            broker_order_id=record.broker_order_id,
            ticker=order.ticker.upper(),
            side=side,
            quantity=order.quantity,
            price=fill_price,
            cost_inr=cost_inr,
            strategy_name=order.strategy_name,
            stop_price=order.stop_price,
        )
        record.fills.append(fill)
        record.filled_quantity += order.quantity
        record.avg_fill_price = (
            sum(f.price * f.quantity for f in record.fills) / record.filled_quantity
        )
        record.status = OrderStatus.FILLED
        record.last_update = when

        if side == "buy":
            self.cash -= notional + cost_inr
        else:
            self.cash += notional - cost_inr

        logger.info(
            "Paper fill: {} {} {}@₹{:.2f} (cost ₹{:.2f}) cash=₹{:,.0f}",
            order.ticker,
            side.upper(),
            order.quantity,
            fill_price,
            cost_inr,
            self.cash,
        )
        return [fill]

    @staticmethod
    def _limit_triggered(order: Order, mark: float) -> bool:
        if order.limit_price is None:
            return False
        if order.side == "buy":
            return mark <= order.limit_price
        return mark >= order.limit_price

    @staticmethod
    def _stop_triggered(order: Order, mark: float) -> bool:
        if order.stop_price is None:
            return False
        if order.side == "buy":  # buy-stop (cover short) — triggers when price RISES
            return mark >= order.stop_price
        return mark <= order.stop_price  # sell-stop (long stop-loss) — triggers on FALL


__all__ = ["PaperBroker"]
