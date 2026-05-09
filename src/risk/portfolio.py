"""In-memory portfolio book.

Tracks open positions, daily P&L, and equity curve **as of the latest mark**.
The risk manager and order router both read from this; the order router
also writes to it on fills.

Why a separate book (not just a DataFrame):
    - Atomic updates (apply_fill / mark_to_market) are conceptually
      tied to a position and need to be testable in isolation.
    - Sector + concentration aggregations are derived properties so
      we can't accidentally show stale numbers when a fill comes in.
    - Future: this is where corporate-action handlers will live.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import TYPE_CHECKING

import pandas as pd

from src.strategies.base import Side

if TYPE_CHECKING:
    from src.orders.base import Fill


@dataclass
class Lot:
    """One open lot of a ticker. We use FIFO on exits (Indian tax rules)."""

    quantity: int  # always positive; the side is on the parent Position
    entry_price: float  # average cost
    entry_time: datetime
    initial_stop: float
    current_stop: float

    @property
    def cost_basis(self) -> float:
        return self.quantity * self.entry_price


@dataclass
class Position:
    """Net position in one ticker. May contain multiple FIFO lots."""

    ticker: str
    side: Side
    sector: str | None = None
    strategy_name: str | None = None
    lots: list[Lot] = field(default_factory=list)

    # ----------------------------------------------------------------
    @property
    def quantity(self) -> int:
        return sum(lot.quantity for lot in self.lots)

    @property
    def avg_entry_price(self) -> float:
        n = self.quantity
        if n == 0:
            return 0.0
        return sum(lot.cost_basis for lot in self.lots) / n

    @property
    def cost_basis(self) -> float:
        return sum(lot.cost_basis for lot in self.lots)

    def market_value(self, mark_price: float) -> float:
        return self.quantity * mark_price * (-1.0 if self.side is Side.SHORT else 1.0)

    def unrealised_pnl(self, mark_price: float) -> float:
        if self.side is Side.LONG:
            return self.quantity * (mark_price - self.avg_entry_price)
        if self.side is Side.SHORT:
            return self.quantity * (self.avg_entry_price - mark_price)
        return 0.0


@dataclass
class FillRecord:
    """Recorded fill — used for tax / P&L attribution audits later."""

    timestamp: datetime
    ticker: str
    side: str  # "buy" / "sell"
    quantity: int
    price: float
    cost_inr: float  # broker fees + slippage rolled in
    realised_pnl: float = 0.0
    client_order_id: str | None = None  # idempotent re-apply guard (matches broker Fill)


@dataclass
class Portfolio:
    """The single source of truth for what we hold and what we've earned.

    Use ``apply_fill`` to ingest broker fills (or paper-broker fills).
    Use ``mark_to_market`` daily after market close.
    The risk manager reads ``equity``, ``daily_drawdown``, ``sector_weights``
    etc. before approving new trades.
    """

    cash_inr: float = 1_000_000.0
    initial_equity_inr: float = 1_000_000.0
    sector_lookup: dict[str, str | None] = field(default_factory=dict)  # ticker → sector
    positions: dict[str, Position] = field(default_factory=dict)
    fills: list[FillRecord] = field(default_factory=list)
    equity_curve: dict[date, float] = field(default_factory=dict)  # date → equity at close
    realised_pnl_total: float = 0.0
    last_mark_price: dict[str, float] = field(default_factory=dict)
    high_watermark: float = 0.0  # peak equity ever seen
    high_watermark_date: date | None = None

    def __post_init__(self) -> None:
        if self.high_watermark == 0.0:
            self.high_watermark = self.initial_equity_inr

    # ----------------------------------------------------------------
    @property
    def equity(self) -> float:
        """Cash + market value of all positions (using last mark)."""
        positions_value = sum(
            pos.market_value(self.last_mark_price.get(pos.ticker, pos.avg_entry_price))
            for pos in self.positions.values()
        )
        return self.cash_inr + positions_value

    @property
    def gross_exposure(self) -> float:
        """|Long| + |Short| / equity. >1 means leveraged."""
        eq = self.equity
        if eq <= 0:
            return 0.0
        gross = sum(
            abs(pos.market_value(self.last_mark_price.get(pos.ticker, pos.avg_entry_price)))
            for pos in self.positions.values()
        )
        return gross / eq

    @property
    def net_exposure(self) -> float:
        eq = self.equity
        if eq <= 0:
            return 0.0
        net = sum(
            pos.market_value(self.last_mark_price.get(pos.ticker, pos.avg_entry_price))
            for pos in self.positions.values()
        )
        return net / eq

    # ----------------------------------------------------------------
    def position_weight(self, ticker: str) -> float:
        """Position weight as fraction of equity (signed)."""
        eq = self.equity
        if eq <= 0:
            return 0.0
        pos = self.positions.get(ticker.upper())
        if pos is None or pos.quantity == 0:
            return 0.0
        mark = self.last_mark_price.get(ticker.upper(), pos.avg_entry_price)
        return pos.market_value(mark) / eq

    def sector_weights(self) -> dict[str, float]:
        """{sector: net weight as fraction of equity}."""
        eq = self.equity
        if eq <= 0:
            return {}
        out: dict[str, float] = defaultdict(float)
        for pos in self.positions.values():
            sector = self.sector_lookup.get(pos.ticker) or pos.sector or "_unknown"
            mark = self.last_mark_price.get(pos.ticker, pos.avg_entry_price)
            out[sector] += pos.market_value(mark) / eq
        return dict(out)

    # ----------------------------------------------------------------
    def drawdown(self) -> float:
        """Current drawdown from all-time high (negative or zero)."""
        if self.high_watermark <= 0:
            return 0.0
        return self.equity / self.high_watermark - 1.0

    def days_in_drawdown(self, as_of: date | None = None) -> int:
        as_of = as_of or (max(self.equity_curve) if self.equity_curve else date.today())
        if not self.equity_curve or self.high_watermark_date is None:
            return 0
        relevant = [d for d in self.equity_curve if self.high_watermark_date <= d <= as_of]
        return len(relevant) - 1 if relevant else 0

    def daily_pnl(self, as_of: date) -> float:
        """Today's P&L as a fraction of yesterday's equity."""
        if as_of not in self.equity_curve:
            return 0.0
        sorted_dates = sorted(self.equity_curve)
        idx = sorted_dates.index(as_of)
        if idx == 0:
            return 0.0
        prev_eq = self.equity_curve[sorted_dates[idx - 1]]
        if prev_eq <= 0:
            return 0.0
        return self.equity_curve[as_of] / prev_eq - 1.0

    # ----------------------------------------------------------------
    def apply_fill(self, fill: Fill) -> FillRecord:
        """Ingest one broker fill, mutating cash + positions atomically."""
        ticker = fill.ticker.upper()
        cid = fill.client_order_id
        if cid:
            for fr in self.fills:
                if (
                    fr.client_order_id == cid
                    and fr.ticker == ticker
                    and fr.side == fill.side
                    and fr.quantity == fill.quantity
                    and fr.price == fill.price
                ):
                    return fr

        record = FillRecord(
            timestamp=fill.timestamp,
            ticker=ticker,
            side=fill.side,
            quantity=fill.quantity,
            price=fill.price,
            cost_inr=fill.cost_inr,
            client_order_id=cid or None,
        )

        notional = fill.quantity * fill.price
        if fill.side == "buy":
            # Open or increase a long; or close a short.
            self.cash_inr -= notional + fill.cost_inr
            existing = self.positions.get(ticker)
            if existing is None or existing.side is Side.FLAT:
                self.positions[ticker] = Position(
                    ticker=ticker,
                    side=Side.LONG,
                    sector=self.sector_lookup.get(ticker),
                    strategy_name=fill.strategy_name,
                    lots=[
                        Lot(
                            quantity=fill.quantity,
                            entry_price=fill.price,
                            entry_time=fill.timestamp,
                            initial_stop=fill.stop_price or fill.price,
                            current_stop=fill.stop_price or fill.price,
                        )
                    ],
                )
            elif existing.side is Side.LONG:
                # Add a lot
                existing.lots.append(
                    Lot(
                        quantity=fill.quantity,
                        entry_price=fill.price,
                        entry_time=fill.timestamp,
                        initial_stop=fill.stop_price or fill.price,
                        current_stop=fill.stop_price or fill.price,
                    )
                )
            else:  # SHORT — buying back covers
                record.realised_pnl = self._close_lots(existing, fill.quantity, fill.price)
                self.realised_pnl_total += record.realised_pnl
                if existing.quantity == 0:
                    del self.positions[ticker]
        elif fill.side == "sell":
            self.cash_inr += notional - fill.cost_inr
            existing = self.positions.get(ticker)
            if existing is None or existing.side is Side.FLAT:
                # Open a fresh short
                self.positions[ticker] = Position(
                    ticker=ticker,
                    side=Side.SHORT,
                    sector=self.sector_lookup.get(ticker),
                    strategy_name=fill.strategy_name,
                    lots=[
                        Lot(
                            quantity=fill.quantity,
                            entry_price=fill.price,
                            entry_time=fill.timestamp,
                            initial_stop=fill.stop_price or fill.price,
                            current_stop=fill.stop_price or fill.price,
                        )
                    ],
                )
            elif existing.side is Side.LONG:
                record.realised_pnl = self._close_lots(existing, fill.quantity, fill.price)
                self.realised_pnl_total += record.realised_pnl
                if existing.quantity == 0:
                    del self.positions[ticker]
            else:  # SHORT — selling more = adding
                existing.lots.append(
                    Lot(
                        quantity=fill.quantity,
                        entry_price=fill.price,
                        entry_time=fill.timestamp,
                        initial_stop=fill.stop_price or fill.price,
                        current_stop=fill.stop_price or fill.price,
                    )
                )
        else:  # pragma: no cover - defensive
            raise ValueError(f"unknown side {fill.side}")

        self.fills.append(record)
        self.last_mark_price[ticker] = fill.price
        return record

    # ----------------------------------------------------------------
    def _close_lots(self, position: Position, quantity: int, price: float) -> float:
        """FIFO close for tax compliance. Returns realised P&L."""
        remaining = quantity
        realised = 0.0
        i = 0
        while remaining > 0 and i < len(position.lots):
            lot = position.lots[i]
            close_qty = min(lot.quantity, remaining)
            if position.side is Side.LONG:
                realised += close_qty * (price - lot.entry_price)
            else:  # SHORT
                realised += close_qty * (lot.entry_price - price)
            lot.quantity -= close_qty
            remaining -= close_qty
            if lot.quantity == 0:
                position.lots.pop(i)
            else:
                i += 1
        return realised

    # ----------------------------------------------------------------
    def mark_to_market(self, marks: dict[str, float], as_of: date) -> float:
        """Update last-known prices and persist equity at this date."""
        for ticker, price in marks.items():
            self.last_mark_price[ticker.upper()] = float(price)
        equity = self.equity
        self.equity_curve[as_of] = equity
        if equity > self.high_watermark:
            self.high_watermark = equity
            self.high_watermark_date = as_of
        return equity

    # ----------------------------------------------------------------
    def to_dataframe(self) -> pd.DataFrame:
        """Snapshot positions for inspection."""
        rows = []
        for pos in self.positions.values():
            mark = self.last_mark_price.get(pos.ticker, pos.avg_entry_price)
            rows.append(
                {
                    "ticker": pos.ticker,
                    "side": pos.side.value,
                    "quantity": pos.quantity,
                    "avg_entry": pos.avg_entry_price,
                    "mark": mark,
                    "market_value": pos.market_value(mark),
                    "unrealised_pnl": pos.unrealised_pnl(mark),
                    "sector": pos.sector or self.sector_lookup.get(pos.ticker),
                    "strategy": pos.strategy_name,
                }
            )
        return pd.DataFrame(rows)


__all__ = ["FillRecord", "Lot", "Portfolio", "Position"]
