"""Point-in-time universe lookup for survivorship-bias-free backtests.

Why this matters:
    If you backtest a strategy on "today's Nifty 500", you've already
    excluded every stock that *was* in the index 5 years ago but got
    delisted / merged / went bust since. That biases your backtest upward
    by 2–8% CAGR (Chan §3.3). The fix is point-in-time: at every date in
    the simulation, you must use the universe as it was on *that* date.

How we use it:
    Each month (Phase 1 cron), we save the current ``UniverseSnapshot``
    to the database. Six months later, when running a backtest from
    January, the backtester calls ``replay.universe_as_of(jan_15)`` and
    gets the snapshot taken closest *before* Jan 15 — what the bot would
    have seen in real life.

    This module is read-only over the snapshots table.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date

import pandas as pd

from src.data.storage import DataStore


@dataclass
class UniverseReplay:
    store: DataStore

    def universe_as_of(self, as_of: date) -> list[str]:
        """Symbols of the most recent snapshot ON OR BEFORE ``as_of``."""
        return self.store.fetch_universe_as_of(as_of)

    def universe_per_rebalance(self, dates: Iterable[date]) -> dict[date, list[str]]:
        """Convenience: bulk lookup for many rebalance dates."""
        return {d: self.universe_as_of(d) for d in dates}

    def to_dataframe(self, as_of: date) -> pd.DataFrame:
        """Return the snapshot as a DataFrame ready to inspect / export."""
        symbols = self.universe_as_of(as_of)
        return pd.DataFrame({"symbol": symbols, "rank": range(1, len(symbols) + 1)})


__all__ = ["UniverseReplay"]
