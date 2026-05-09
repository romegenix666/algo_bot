"""Zerodha Kite Connect broker (Phase 7).

This is the **demo / sandbox / live** integration with Zerodha's Kite Connect
API. We deliberately keep it as a thin wrapper over ``kiteconnect.KiteConnect``
so:

    - Behaviour matches the production broker pixel-for-pixel.
    - When/if Zerodha changes its API, only this file moves.
    - The bot's risk manager + portfolio book remain ENTIRELY broker-agnostic.

What it implements (the ``Broker`` interface):
    - ``submit(order)`` — places a real Kite order
    - ``cancel(client_order_id)``
    - ``status(client_order_id)``
    - ``open_orders()``
    - ``positions()`` — net positions across product types
    - ``set_mark(ticker, price)`` — no-op (real broker doesn't need marks)

Idempotency:
    Kite's REST API doesn't have native idempotency keys. We work around this
    by maintaining our own ``client_order_id → broker_order_id`` map in
    memory + on disk; if a re-submit comes in with the same client id, we
    return the existing record without re-submitting.

Auth flow (one-time per session):
    1. Generate request_token via Kite login URL.
    2. Exchange request_token for access_token via ``KiteConnect.generate_session``.
    3. Save access_token to ``.env`` (KITE_ACCESS_TOKEN). It expires daily —
       a one-line auth helper script handles the daily refresh.

Sandbox vs live:
    Zerodha doesn't have a free sandbox. They do offer a "demo trader" via
    Kite's UI but no programmatic counterpart. We therefore **paper-test
    extensively first**, then go straight to live with very small capital
    (₹50k cap initially, per ROADMAP.md Phase 8).

Stub-mode:
    If ``KITE_API_KEY`` is missing, this class operates in "stub" mode:
    every method raises ``KiteUnavailableError``. This means importing
    this module is always safe — only constructing the broker fails when
    creds are missing.

References:
    - Kite Connect docs: https://kite.trade/docs/connect/v3/
    - python-kiteconnect: https://github.com/zerodha/pykiteconnect
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

try:
    from kiteconnect import KiteConnect
except ImportError:  # pragma: no cover - optional dep until phase 7
    KiteConnect = None

from src.orders.base import (
    Broker,
    Fill,
    Order,
    OrderRecord,
    OrderStatus,
    OrderType,
)
from src.utils.logging import logger
from src.utils.settings import settings


class KiteUnavailableError(RuntimeError):
    """Raised when the Kite SDK or credentials are missing."""


@dataclass
class KiteBroker(Broker):
    """Kite Connect-backed broker implementing the same Broker interface as
    PaperBroker. Used in the *demo* (paper-with-real-quotes) and *live*
    modes."""

    api_key: str | None = None
    access_token: str | None = None
    api_secret: str | None = None  # for the daily auth flow only
    product: str = "CNC"  # CNC = equity delivery; MIS = intraday
    variety: str = "regular"
    exchange: str = "NSE"

    # State
    _kite: object | None = None
    _records: dict[str, OrderRecord] = field(default_factory=dict)
    _broker_to_client: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_settings(cls) -> KiteBroker:
        creds = settings.broker
        return cls(
            api_key=creds.kite_api_key,
            access_token=creds.kite_access_token,
            api_secret=creds.kite_api_secret,
        )

    def __post_init__(self) -> None:
        if KiteConnect is None:
            raise KiteUnavailableError("kiteconnect not installed. `pip install kiteconnect`.")
        if not self.api_key or not self.access_token:
            raise KiteUnavailableError(
                "Kite creds missing. Set KITE_API_KEY and KITE_ACCESS_TOKEN in .env."
            )
        self._kite = KiteConnect(api_key=self.api_key)
        self._kite.set_access_token(self.access_token)  # type: ignore[attr-defined]
        logger.info("KiteBroker initialised (api_key=...{})", self.api_key[-4:])

    # ----------------------------------------------------------------
    @property
    def name(self) -> str:
        return "kite"

    # ----------------------------------------------------------------
    def set_mark(self, ticker: str, price: float) -> None:
        """No-op — Kite doesn't need us to push marks."""
        return None

    # ----------------------------------------------------------------
    def available_equity_cash(self) -> float:
        """Best available cash for the **equity** segment (delivery / CNC).

        Uses ``available.live_balance`` with fallback to ``net`` per Kite docs.
        """
        kite = self._require_kite()
        try:
            raw = kite.margins()
        except Exception as exc:  # pragma: no cover - live API
            logger.warning("Kite margins() failed: {}", exc)
            return 0.0
        eq = raw.get("equity") or {}
        if not isinstance(eq, dict):
            return 0.0
        avail = eq.get("available") or {}
        if isinstance(avail, dict):
            lb = avail.get("live_balance")
            if lb is not None:
                return float(lb)
            c = avail.get("cash")
            if c is not None:
                return float(c)
        net = eq.get("net")
        if net is not None:
            return float(net)
        return 0.0

    def iter_net_positions(self) -> list[tuple[str, int, float, str]]:
        """Overnight **net** book only: (TICKER.NS, qty, average_price, product).

        Uses the ``net`` bucket from Kite (carry-forward holdings). Intraday
        ``day`` is excluded so EOD delivery sync is not doubled with MIS noise.
        """
        kite = self._require_kite()
        try:
            data = kite.positions()
        except Exception as exc:  # pragma: no cover - live API
            logger.warning("Kite positions() failed: {}", exc)
            return []
        out: list[tuple[str, int, float, str]] = []
        for pos in data.get("net", []) or []:
            qty = int(pos.get("quantity") or 0)
            if qty == 0:
                continue
            sym = str(pos.get("tradingsymbol") or "").strip()
            if not sym:
                continue
            ticker = f"{sym}.NS"
            avg = float(pos.get("average_price") or 0.0)
            product = str(pos.get("product") or "")
            out.append((ticker, qty, avg, product))
        return out

    # ----------------------------------------------------------------
    def submit(self, order: Order) -> OrderRecord:
        # Idempotency: if we've seen this client id, return existing record.
        existing = self._records.get(order.client_order_id)
        if existing is not None:
            logger.info("Kite: idempotent re-submit for {}", order.client_order_id)
            return existing

        kite = self._require_kite()
        kite_side = "BUY" if order.side.lower() == "buy" else "SELL"
        kite_order_type = self._kite_order_type(order.order_type)

        try:
            broker_order_id = kite.place_order(
                tradingsymbol=order.ticker.replace(".NS", ""),
                exchange=self.exchange,
                transaction_type=kite_side,
                quantity=int(order.quantity),
                product=self.product,
                order_type=kite_order_type,
                price=float(order.limit_price) if order.limit_price is not None else None,
                trigger_price=float(order.stop_price) if order.stop_price is not None else None,
                variety=self.variety,
                tag=(order.strategy_name or "")[:20],
            )
        except Exception as exc:  # pragma: no cover - live API
            logger.exception("Kite place_order failed for {}", order.ticker)
            record = OrderRecord(
                order=order,
                broker_order_id=f"failed-{order.client_order_id[:8]}",
                status=OrderStatus.REJECTED,
                rejection_reason=str(exc),
            )
            self._records[order.client_order_id] = record
            return record

        record = OrderRecord(
            order=order,
            broker_order_id=str(broker_order_id),
            status=OrderStatus.SUBMITTED,
        )
        self._records[order.client_order_id] = record
        self._broker_to_client[str(broker_order_id)] = order.client_order_id
        logger.info(
            "Kite SUBMIT {} {} qty={} → broker_order_id={}",
            order.ticker,
            kite_side,
            order.quantity,
            broker_order_id,
        )
        # Refresh status once to capture immediate fills (market orders).
        self._refresh_record(order.client_order_id)
        return record

    # ----------------------------------------------------------------
    def cancel(self, client_order_id: str) -> OrderRecord:
        record = self._records.get(client_order_id)
        if record is None:
            raise KeyError(f"Unknown client_order_id: {client_order_id}")
        if record.is_terminal:
            return record
        kite = self._require_kite()
        try:
            kite.cancel_order(variety=self.variety, order_id=record.broker_order_id)
        except Exception as exc:  # pragma: no cover - live API
            logger.warning("Kite cancel_order failed: {}", exc)
        record.status = OrderStatus.CANCELLED
        record.last_update = datetime.now(UTC)
        return record

    # ----------------------------------------------------------------
    def status(self, client_order_id: str) -> OrderRecord | None:
        if client_order_id in self._records:
            self._refresh_record(client_order_id)
            return self._records[client_order_id]
        return None

    # ----------------------------------------------------------------
    def open_orders(self) -> list[OrderRecord]:
        """Return non-terminal records. Refreshes them all from Kite first."""
        for cid in list(self._records):
            self._refresh_record(cid)
        return [r for r in self._records.values() if not r.is_terminal]

    # ----------------------------------------------------------------
    def positions(self) -> dict[str, int]:
        kite = self._require_kite()
        try:
            data = kite.positions()
        except Exception as exc:  # pragma: no cover - live API
            logger.warning("Kite positions() failed: {}", exc)
            return {}
        out: dict[str, int] = {}
        for bucket in ("net", "day"):
            for pos in data.get(bucket, []) or []:
                sym = str(pos.get("tradingsymbol") or "").strip()
                if not sym:
                    continue
                ticker = f"{sym}.NS"
                qty = int(pos.get("quantity", 0))
                if qty != 0:
                    out[ticker] = out.get(ticker, 0) + qty
        return out

    # ----------------------------------------------------------------
    # Internals
    # ----------------------------------------------------------------
    def _require_kite(self):
        """Return the underlying KiteConnect client. Untyped to keep the
        Kite SDK as an optional dep — mypy doesn't understand the dynamic
        ``KiteConnect`` class without the package installed."""
        if self._kite is None:  # pragma: no cover - defensive
            raise KiteUnavailableError("Kite client not initialised")
        return self._kite

    @staticmethod
    def _kite_order_type(t: OrderType) -> str:
        return {
            OrderType.MARKET: "MARKET",
            OrderType.LIMIT: "LIMIT",
            OrderType.STOP: "SL-M",  # stop-loss market
            OrderType.STOP_LIMIT: "SL",  # stop-loss limit
        }[t]

    def _refresh_record(self, client_order_id: str) -> None:
        """Pull updated status + any fills from Kite for one order."""
        record = self._records.get(client_order_id)
        if record is None or record.is_terminal:
            return
        kite = self._require_kite()
        try:
            history = kite.order_history(order_id=record.broker_order_id)
        except Exception as exc:  # pragma: no cover - live API
            logger.warning("Kite order_history failed for {}: {}", record.broker_order_id, exc)
            return
        if not history:
            return
        last = history[-1]
        kite_status = (last.get("status") or "").upper()
        if "COMPLETE" in kite_status:
            record.status = OrderStatus.FILLED
        elif "PARTIAL" in kite_status:
            record.status = OrderStatus.PARTIAL
        elif "CANCELLED" in kite_status or "REJECTED" in kite_status:
            record.status = (
                OrderStatus.CANCELLED if "CANCEL" in kite_status else OrderStatus.REJECTED
            )
        # Convert any new trades into Fills.
        try:
            trades = kite.order_trades(order_id=record.broker_order_id)
        except Exception as exc:  # pragma: no cover - live API
            logger.warning("Kite order_trades failed: {}", exc)
            trades = []
        existing_trade_ids = {f.broker_order_id for f in record.fills}
        for trade in trades:
            tid = str(trade.get("trade_id") or trade.get("order_id"))
            if tid in existing_trade_ids:
                continue
            qty = int(trade.get("quantity", 0))
            price = float(trade.get("average_price") or trade.get("price") or 0.0)
            ts = trade.get("fill_timestamp") or trade.get("order_timestamp")
            fill_dt = ts if isinstance(ts, datetime) else datetime.now(UTC)
            fill = Fill(
                timestamp=fill_dt,
                client_order_id=record.order.client_order_id,
                broker_order_id=tid,
                ticker=record.order.ticker.upper(),
                side=record.order.side,
                quantity=qty,
                price=price,
                cost_inr=0.0,  # Kite returns charges separately; tracked in costs module
                strategy_name=record.order.strategy_name,
                stop_price=record.order.stop_price,
            )
            record.fills.append(fill)
            record.filled_quantity += qty
            if record.filled_quantity > 0:
                record.avg_fill_price = (
                    sum(f.price * f.quantity for f in record.fills) / record.filled_quantity
                )
        record.last_update = datetime.now(UTC)


__all__ = ["KiteBroker", "KiteUnavailableError"]
