"""Synchronise the in-memory ``Portfolio`` with Zerodha Kite (live mode).

After a restart, local ``state.json`` may diverge from the broker (manual
trades, partial fills, token refresh). On each **live** run we optionally
re-base cash and holdings from Kite before applying today's logic.

References:
    - Kite margins: https://kite.trade/docs/connect/v3/user/#margins
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from src.risk.portfolio import Lot, Portfolio, Position
from src.strategies.base import Side
from src.utils.logging import logger

if TYPE_CHECKING:
    from src.orders.kite import KiteBroker


def stop_price_from_state_fills(fills: list[dict], ticker: str) -> float | None:
    """Best-effort: last recorded stop for this ticker from persisted fills."""
    t = ticker.upper()
    for f in reversed(fills):
        if str(f.get("ticker", "")).upper() != t:
            continue
        sp = f.get("stop_price")
        if sp is not None:
            try:
                return float(sp)
            except (TypeError, ValueError):
                continue
    return None


def sync_portfolio_from_kite(
    portfolio: Portfolio,
    broker: KiteBroker,
    *,
    state_fills: list[dict],
    default_stop_frac_below_avg: float = 0.12,
) -> None:
    """Replace ``portfolio`` cash and positions from Kite's net book.

    - Cash: ``equity.available.live_balance`` (fallback: ``equity.net``).
    - Positions: ``positions()['net']`` with non-zero quantity, product match.

    Stop prices: use last ``stop_price`` from ``state_fills`` if any; else
    ``avg * (1 - default_stop_frac_below_avg)`` as a **placeholder** until the
    bot manages the position with fresh ATR (documented trade-off).

    Long-only retail: negative quantities (shorts) are skipped with a log line.
    """
    cash = broker.available_equity_cash()
    portfolio.cash_inr = max(0.0, cash)
    portfolio.positions.clear()

    for ticker, qty, avg_price, product in broker.iter_net_positions():
        if qty == 0:
            continue
        if qty < 0:
            logger.warning(
                "live_sync: skipping short position {} qty={} (long-only bot)",
                ticker,
                qty,
            )
            continue
        if product and product != broker.product:
            logger.info(
                "live_sync: skipping {} product={} (broker product filter={})",
                ticker,
                product,
                broker.product,
            )
            continue

        stop = stop_price_from_state_fills(state_fills, ticker)
        if stop is None or stop <= 0 or stop >= avg_price:
            stop = float(avg_price) * (1.0 - default_stop_frac_below_avg)

        sector = portfolio.sector_lookup.get(ticker.upper())
        now = datetime.now(UTC)
        portfolio.positions[ticker.upper()] = Position(
            ticker=ticker.upper(),
            side=Side.LONG,
            sector=sector,
            strategy_name="kite_sync",
            lots=[
                Lot(
                    quantity=int(qty),
                    entry_price=float(avg_price),
                    entry_time=now,
                    initial_stop=float(stop),
                    current_stop=float(stop),
                )
            ],
        )

    logger.info(
        "live_sync: cash=₹{:,.0f}, {} open positions from Kite",
        portfolio.cash_inr,
        len(portfolio.positions),
    )


__all__ = ["stop_price_from_state_fills", "sync_portfolio_from_kite"]
