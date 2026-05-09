"""Daily data-refresh orchestrator.

Pipeline (run after market close, ~16:30 IST):

    1. Pull the ticker list (NSE / seed CSV).
    2. Upsert tickers into ``tickers`` table.
    3. For each ticker:
        a. Determine the latest stored bar.
        b. Fetch incremental OHLCV (just from latest+1 to today).
        c. Run anomaly detector on the new bars.
        d. Insert clean bars; record anomalies.
    4. Refresh corporate actions monthly (rare events; daily is wasteful).

The orchestrator is **idempotent**: running it twice produces the same
state. ``insert_prices(replace_overlapping=True)`` ensures re-running on
the same window doesn't double-insert.

Designed to be called from:
    - the CLI (``scripts/fetch_history.py``)
    - APScheduler (Phase 5)
    - GitHub Actions (Phase 7+ for cloud runs)
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

from src.data.anomaly import Anomaly, detect_anomalies, summarise_anomalies
from src.data.nse_client import NSEClient
from src.data.storage import DataStore
from src.data.yfinance_client import YFinanceClient, lookback_window
from src.utils.logging import logger


@dataclass
class RefreshReport:
    started_at: datetime
    finished_at: datetime | None = None
    tickers_upserted: int = 0
    tickers_with_data: int = 0
    bars_inserted: int = 0
    fetch_failures: list[str] = field(default_factory=list)
    anomalies: list[Anomaly] = field(default_factory=list)

    def summary(self) -> str:
        elapsed = (
            (self.finished_at or datetime.utcnow()) - self.started_at
        ).total_seconds()
        anomaly_summary = summarise_anomalies(self.anomalies)
        return (
            f"Refresh done in {elapsed:.1f}s | "
            f"tickers upserted={self.tickers_upserted}, "
            f"with data={self.tickers_with_data}, "
            f"bars inserted={self.bars_inserted}, "
            f"failures={len(self.fetch_failures)}, "
            f"anomalies={anomaly_summary or 'none'}"
        )


@dataclass
class DataRefresher:
    store: DataStore
    yf_client: YFinanceClient = field(default_factory=YFinanceClient)
    nse_client: NSEClient = field(default_factory=NSEClient)

    # ------------------------------------------------------------------
    def initial_load(
        self,
        years: int = 5,
        symbols: Iterable[str] | None = None,
    ) -> RefreshReport:
        """One-shot: upsert universe metadata + fetch ``years`` of history.

        Use this once after ``init_db``. Subsequent days call ``daily``.
        """
        report = RefreshReport(started_at=datetime.utcnow())

        # 1. Upsert tickers
        meta = self.nse_client.list_nifty_500()
        rows = [
            {
                "symbol": m.symbol,
                "name": m.name,
                "sector": m.sector,
                "industry": m.industry,
            }
            for m in meta
        ]
        report.tickers_upserted = self.store.upsert_tickers(rows)

        # 2. Bulk-fetch history
        target_symbols = list(symbols) if symbols else [m.symbol for m in meta]
        start, end = lookback_window(years=years)

        logger.info(
            "Initial load: fetching {} symbols from {} to {}",
            len(target_symbols),
            start,
            end,
        )
        results = self.yf_client.fetch_history(target_symbols, start=start, end=end)

        for r in results:
            if not r.ok:
                report.fetch_failures.append(f"{r.symbol}: {r.error}")
                continue
            try:
                inserted = self.store.insert_prices(r.symbol, r.bars, replace_overlapping=True)
                report.bars_inserted += inserted
                report.tickers_with_data += 1
                anomalies = detect_anomalies(r.symbol, r.bars)
                if anomalies:
                    report.anomalies.extend(anomalies)
            except ValueError as exc:
                # ticker not in DB (race during upsert) — log + skip
                report.fetch_failures.append(f"{r.symbol}: {exc}")

        report.finished_at = datetime.utcnow()
        logger.info(report.summary())
        return report

    # ------------------------------------------------------------------
    def daily(self, target_date: date | None = None) -> RefreshReport:
        """Incremental refresh: pull only the bars we don't already have.

        Args:
            target_date: latest bar date to pull (default: yesterday IST).
        """
        report = RefreshReport(started_at=datetime.utcnow())
        if target_date is None:
            target_date = (datetime.utcnow() - timedelta(days=1)).date()

        active = self.store.list_tickers(status="active")
        if not active:
            logger.warning("daily(): no active tickers, did you run initial_load?")
            report.finished_at = datetime.utcnow()
            return report

        for ticker in active:
            sym = ticker.symbol
            existing = self.store.fetch_prices(sym, end=target_date)
            if existing.empty:
                start = target_date - timedelta(days=365 * 5)  # cold start
            else:
                last_local = existing.index.max().date()
                if last_local >= target_date:
                    continue  # already have it
                start = last_local + timedelta(days=1)

            result = self.yf_client.fetch_one(sym, start=start, end=target_date)
            if not result.ok:
                report.fetch_failures.append(f"{sym}: {result.error}")
                continue
            try:
                inserted = self.store.insert_prices(sym, result.bars, replace_overlapping=True)
                report.bars_inserted += inserted
                if inserted > 0:
                    report.tickers_with_data += 1
                    anomalies = detect_anomalies(sym, result.bars)
                    if anomalies:
                        report.anomalies.extend(anomalies)
            except ValueError as exc:
                report.fetch_failures.append(f"{sym}: {exc}")

        report.tickers_upserted = len(active)
        report.finished_at = datetime.utcnow()
        logger.info(report.summary())
        return report

    # ------------------------------------------------------------------
    def refresh_actions(self, symbols: Iterable[str] | None = None) -> int:
        """Pull dividend + split history. Run monthly, not daily."""
        targets = (
            [s for s in symbols] if symbols
            else [t.symbol for t in self.store.list_tickers(status="active")]
        )
        total = 0
        for sym in targets:
            actions = self.yf_client.fetch_actions(sym)
            if actions.empty:
                continue
            try:
                payload = actions.to_dict(orient="records")
                total += self.store.insert_actions(sym, payload, replace=True)
            except ValueError as exc:
                logger.warning("Action insert failed for {}: {}", sym, exc)
        logger.info("Refreshed actions for {} tickers, total rows {}", len(targets), total)
        return total


__all__ = ["DataRefresher", "RefreshReport"]
