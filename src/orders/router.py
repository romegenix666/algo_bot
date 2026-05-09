"""Order Router — turns Signals into approved, sized, broker-bound orders.

Pipeline::

    Signal (from strategy)
       │
       ▼
    sized TradeRequest (Half-Kelly + ATR stop, via RiskManager.make_request)
       │
       ▼
    RiskManager.approve() — circuit breaker, position cap, sector cap, cash buffer
       │           │
       ▼           ▼
    Broker.submit  REJECTED (logged, alerted)
       │
       ▼
    Fill → Portfolio.apply_fill

The router is the only thing in the system that:
    - Mints client_order_ids (idempotency keys)
    - Knows about both the broker and the portfolio
    - Logs every approval / rejection / fill for the audit trail

The risk manager is the *gate*; the router is the *plumbing*.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import date, datetime, timezone

import pandas as pd

from src.features.indicators import atr as compute_atr
from src.orders.base import (
    Broker,
    Fill,
    Order,
    OrderRecord,
    OrderStatus,
    OrderType,
)
from src.risk.manager import (
    ApprovalDecision,
    RiskLimits,
    RiskManager,
    TradeRequest,
)
from src.risk.portfolio import Portfolio
from src.strategies.base import Side, Signal
from src.utils.logging import logger


def _blake_client_id(prefix: str, payload: str) -> str:
    """Short deterministic idempotency key (Kite-safe length)."""
    return prefix + hashlib.blake2b(payload.encode("utf-8"), digest_size=12).hexdigest()


@dataclass(frozen=True)
class RoutingResult:
    """One Signal → outcome bundle for the audit log."""

    signal: Signal
    request: TradeRequest | None
    approval: ApprovalDecision | None
    record: OrderRecord | None
    rejected_reason: str | None = None


@dataclass
class OrderRouter:
    """Routes ranked signals through risk → broker → portfolio."""

    broker: Broker
    risk_manager: RiskManager
    portfolio: Portfolio
    atr_window: int = 14
    atr_multiple_per_strategy: dict[str, float] = field(
        default_factory=lambda: {
            "momentum": 2.0,
            "mean_reversion": 1.5,
            "pairs": 1.0,
            "multi_factor": 2.5,
            "breakout": 2.0,
            "dual_momentum": 2.0,
            "sector_rotation": 2.0,
            "sentiment_momentum": 2.0,
        }
    )
    default_atr_multiple: float = 2.0

    # ----------------------------------------------------------------
    def route_signal(
        self,
        signal: Signal,
        prices_history: pd.DataFrame,
        today: date,
        strategy_name: str,
    ) -> RoutingResult:
        """Process one signal through the full risk → broker pipeline.

        ``prices_history`` is the per-ticker OHLC frame (datetime-indexed)
        used to compute the ATR for stop placement. Routes a SHORT signal
        as "exit existing long" if we're long-only.
        """
        # 1. Compute ATR + last price for sizing.
        ticker = signal.ticker.upper()
        if prices_history.empty or len(prices_history) < self.atr_window + 5:
            return RoutingResult(
                signal=signal,
                request=None,
                approval=None,
                record=None,
                rejected_reason="insufficient_history_for_atr",
            )
        last_price = float(prices_history["close"].iloc[-1])
        atr_series = compute_atr(
            prices_history["high"],
            prices_history["low"],
            prices_history["close"],
            window=self.atr_window,
        )
        atr_value = float(atr_series.iloc[-1]) if not atr_series.empty else 0.0
        if atr_value <= 0:
            return RoutingResult(
                signal=signal,
                request=None,
                approval=None,
                record=None,
                rejected_reason="non_positive_atr",
            )

        # 2. SHORT signal handling: in long-only paper, treat as exit.
        if signal.side is Side.SHORT and self.portfolio.position_weight(ticker) <= 0:
            return RoutingResult(
                signal=signal,
                request=None,
                approval=None,
                record=None,
                rejected_reason="short_disabled_long_only",
            )

        # 3. Build the TradeRequest (sized + stop placed).
        atr_mult = self.atr_multiple_per_strategy.get(strategy_name, self.default_atr_multiple)
        request = self.risk_manager.make_request(
            signal=signal,
            atr_value=atr_value,
            atr_multiple=atr_mult,
            equity=self.portfolio.equity,
            limits=self.risk_manager.limits,
            last_price=last_price,
            strategy_name=strategy_name,
        )
        if request is None:
            return RoutingResult(
                signal=signal,
                request=None,
                approval=None,
                record=None,
                rejected_reason="zero_size",
            )

        # 4. Risk manager gate.
        approval = self.risk_manager.approve(request, self.portfolio, today)
        if not approval.approved:
            logger.warning(
                "Order REJECTED for {}: {} ({})",
                ticker,
                approval.reason,
                approval.detail,
            )
            return RoutingResult(
                signal=signal,
                request=request,
                approval=approval,
                record=None,
                rejected_reason=approval.reason,
            )

        # 5. Submit to broker.
        broker_side = "buy" if signal.side is Side.LONG else "sell"
        idem = _blake_client_id(
            "e",
            f"{today.isoformat()}|{signal.timestamp.isoformat()}|{strategy_name}|"
            f"{ticker}|{broker_side}|{approval.approved_quantity}",
        )
        order = Order(
            client_order_id=idem,
            ticker=ticker,
            side=broker_side,
            quantity=approval.approved_quantity,
            order_type=OrderType.MARKET,
            stop_price=request.stop_price,
            strategy_name=strategy_name,
            metadata={
                "signal_conviction": signal.conviction,
                "atr": atr_value,
                "atr_multiple": atr_mult,
            },
        )
        record = self.broker.submit(order)

        # 6. Apply any immediate fills to the portfolio.
        for fill in record.fills:
            self.portfolio.apply_fill(fill)

        return RoutingResult(
            signal=signal,
            request=request,
            approval=approval,
            record=record,
            rejected_reason=None,
        )

    # ----------------------------------------------------------------
    def exit_position(
        self,
        ticker: str,
        reason: str,
        last_price: float,
        strategy_name: str | None = None,
        as_of: date | None = None,
    ) -> OrderRecord | None:
        """Close an existing position at market. Used by stop-loss /
        circuit-breaker /manual kill paths."""
        ticker = ticker.upper()
        bar_date = as_of or date.today()
        position = self.portfolio.positions.get(ticker)
        if position is None or position.quantity == 0:
            return None
        broker_side = "sell" if position.side is Side.LONG else "buy"

        # Push the broker its current mark before submitting market exit.
        if isinstance(self.broker, type(self.broker)) and hasattr(self.broker, "set_mark"):
            self.broker.set_mark(ticker, last_price)

        xid = _blake_client_id(
            "x",
            f"{bar_date.isoformat()}|{ticker}|{position.quantity}|{reason}",
        )
        order = Order(
            client_order_id=xid,
            ticker=ticker,
            side=broker_side,
            quantity=position.quantity,
            order_type=OrderType.MARKET,
            strategy_name=strategy_name or position.strategy_name,
            metadata={"exit_reason": reason},
        )
        record = self.broker.submit(order)
        for fill in record.fills:
            self.portfolio.apply_fill(fill)
        if record.status is OrderStatus.FILLED:
            logger.info("Closed {} ({}). Reason: {}", ticker, broker_side.upper(), reason)
        return record

    # ----------------------------------------------------------------
    def kill_switch(
        self,
        last_marks: dict[str, float],
        reason: str = "manual_kill_switch",
        as_of: date | None = None,
    ) -> list[OrderRecord]:
        """Hard-exit every open position. Use sparingly — and unrecoverable
        until the breaker is manually reset."""
        bar_date = as_of or date.today()
        self.risk_manager.breaker.manual_kill_switch(when=bar_date)
        records: list[OrderRecord] = []
        for ticker, position in list(self.portfolio.positions.items()):
            if position.quantity == 0:
                continue
            mark = last_marks.get(ticker, position.avg_entry_price)
            rec = self.exit_position(
                ticker=ticker,
                reason=reason,
                last_price=mark,
                as_of=bar_date,
            )
            if rec is not None:
                records.append(rec)
        return records


# ---------------------------------------------------------------------------
# Helpers (referenced for cleanliness)
# ---------------------------------------------------------------------------


_ = (Fill, RiskLimits, datetime, timezone)


__all__ = ["OrderRouter", "RoutingResult"]
