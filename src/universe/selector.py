"""Universe selector — pick the top-N tradable NSE stocks for a given month.

Filters applied (in order):

1. **Status active** — drop delisted / suspended.
2. **Price band** — drop sub-₹50 (penny risk) and >₹10000 (illiquid for
   small accounts because lot size is forced).
3. **Liquidity** — average daily turnover (close × volume) over the last
   ``turnover_window_days`` must exceed ``min_avg_turnover_cr``.
4. **Listing age** — must have at least ``min_history_days`` of price
   history (we don't trade IPOs in their first 6 months).
5. **Market cap** (when available) — top ``N`` by market cap from what's
   left.

Output: a list of ``UniverseEntry`` rows, ranked, with the underlying
metric values populated for traceability.

The result is also stored in the ``universe_snapshots`` table so backtests
can replay the universe as it was on a historical date.

Why these filters?
    Chan §2: low capacity / high liquidity strategies fly under institutional
    radar (good for us), but you still need to *be able* to fill orders. The
    ₹5 Cr / day floor is approximately what an entry-level retail bot
    (₹50k–₹5L) can move without becoming the market.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd

from src.data.storage import DataStore
from src.utils.logging import logger


@dataclass(frozen=True)
class UniverseEntry:
    """One row of a universe ranking, ready to write to ``universe_snapshots``."""

    symbol: str
    rank: int
    last_price: float
    avg_turnover_cr: float
    market_cap_cr: float | None
    history_days: int


@dataclass
class UniverseSelector:
    """Stateless selector — read-only views of the data store."""

    store: DataStore

    # Filter parameters (the 5 we tune)
    target_size: int = 500
    price_min: float = 50.0
    price_max: float = 10_000.0
    min_avg_turnover_cr: float = 5.0
    turnover_window_days: int = 30
    min_history_days: int = 180  # ~6 months

    # ------------------------------------------------------------------
    def select(
        self,
        as_of: date | None = None,
        market_caps: dict[str, float] | None = None,
    ) -> list[UniverseEntry]:
        """Compute the universe as of ``as_of`` (default: today).

        Args:
            as_of: Date for the selection. We use prices on or before this
                date and require all metrics to be computable from history
                ending here.
            market_caps: Optional ``{symbol: market_cap_in_crore}`` mapping.
                Caller is responsible for sourcing this (yfinance ``info``
                isn't reliable in bulk; for now we rank by liquidity if
                missing).
        """
        as_of = as_of or date.today()
        window_start = as_of - timedelta(days=self.turnover_window_days * 2)

        active = self.store.list_tickers(status="active")
        if not active:
            logger.warning("No active tickers found in store")
            return []

        rows: list[UniverseEntry] = []
        skipped_count = 0

        for ticker in active:
            symbol = ticker.symbol
            bars = self.store.fetch_prices(symbol, start=window_start, end=as_of)
            if bars.empty:
                skipped_count += 1
                continue

            history_days = len(bars)
            if history_days < self.min_history_days:
                # need full history check too — ask separately
                full = self.store.fetch_prices(symbol, end=as_of)
                if len(full) < self.min_history_days:
                    skipped_count += 1
                    continue

            recent = bars.tail(self.turnover_window_days)
            if recent.empty:
                skipped_count += 1
                continue

            last_price = float(recent["close"].iloc[-1])
            if not (self.price_min <= last_price <= self.price_max):
                continue

            # Daily turnover in INR Crore (1 Cr = 1e7 rupees)
            turnover = (recent["close"] * recent["volume"]) / 1e7
            avg_turnover_cr = float(turnover.mean())
            if avg_turnover_cr < self.min_avg_turnover_cr:
                continue

            mcap = market_caps.get(symbol) if market_caps else None
            rows.append(
                UniverseEntry(
                    symbol=symbol,
                    rank=0,  # filled in below after sorting
                    last_price=last_price,
                    avg_turnover_cr=avg_turnover_cr,
                    market_cap_cr=mcap,
                    history_days=int(self.store.fetch_prices(symbol, end=as_of).shape[0]),
                )
            )

        # Rank: by market cap if available, else by avg turnover
        if rows and any(r.market_cap_cr is not None for r in rows):
            rows.sort(
                key=lambda r: (
                    -(r.market_cap_cr or 0.0),
                    -r.avg_turnover_cr,
                )
            )
        else:
            rows.sort(key=lambda r: -r.avg_turnover_cr)

        # Apply target_size cut and assign ranks
        final = rows[: self.target_size]
        ranked = [
            UniverseEntry(
                symbol=r.symbol,
                rank=i + 1,
                last_price=r.last_price,
                avg_turnover_cr=r.avg_turnover_cr,
                market_cap_cr=r.market_cap_cr,
                history_days=r.history_days,
            )
            for i, r in enumerate(final)
        ]

        logger.info(
            "Universe @ {}: {} tickers selected (target {}, "
            "active checked {}, skipped for missing/short history {})",
            as_of,
            len(ranked),
            self.target_size,
            len(active),
            skipped_count,
        )
        return ranked

    # ------------------------------------------------------------------
    def save_snapshot(
        self,
        as_of: date,
        entries: list[UniverseEntry],
        notes: str | None = None,
    ) -> int:
        rows = [
            {
                "symbol": e.symbol,
                "rank": e.rank,
                "market_cap_cr": e.market_cap_cr,
                "avg_turnover_cr": e.avg_turnover_cr,
                "notes": notes,
            }
            for e in entries
        ]
        return self.store.save_universe_snapshot(as_of, rows, replace=True)

    # ------------------------------------------------------------------
    def to_dataframe(self, entries: list[UniverseEntry]) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "rank": e.rank,
                    "symbol": e.symbol,
                    "last_price": e.last_price,
                    "avg_turnover_cr": e.avg_turnover_cr,
                    "market_cap_cr": e.market_cap_cr,
                    "history_days": e.history_days,
                }
                for e in entries
            ]
        )


__all__ = ["UniverseEntry", "UniverseSelector"]
