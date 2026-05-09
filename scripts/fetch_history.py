"""Bulk-fetch historical OHLCV from Yahoo Finance into the local database.

Usage::

    # Default: 5 years for every ticker in data/seed/nifty_seed.csv
    python -m scripts.fetch_history

    # Subset for quick testing:
    python -m scripts.fetch_history --symbols RELIANCE TCS HDFCBANK --years 2

    # Daily-style incremental refresh:
    python -m scripts.fetch_history --mode daily
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date

from src.data.refresh import DataRefresher
from src.data.storage import DataStore
from src.utils.logging import logger


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("initial", "daily", "actions"),
        default="initial",
        help="initial = pull --years of history; daily = incremental top-up; actions = refresh splits/divs",
    )
    parser.add_argument("--years", type=int, default=5)
    parser.add_argument(
        "--symbols",
        nargs="*",
        help="Optional subset of symbols (otherwise everything in seed CSV)",
    )
    parser.add_argument("--target-date", type=str, help="(daily mode) Latest bar date YYYY-MM-DD")
    args = parser.parse_args()

    store = DataStore.from_settings()
    store.create_all()  # idempotent — auto-creates if missing

    refresher = DataRefresher(store=store)

    if args.mode == "initial":
        report = refresher.initial_load(years=args.years, symbols=args.symbols)
    elif args.mode == "daily":
        target = date.fromisoformat(args.target_date) if args.target_date else None
        report = refresher.daily(target_date=target)
    elif args.mode == "actions":
        n = refresher.refresh_actions(symbols=args.symbols)
        logger.info("Action rows refreshed: {}", n)
        return 0
    else:  # pragma: no cover - argparse guards this
        raise ValueError(args.mode)

    summary = {
        "mode": args.mode,
        "tickers_upserted": report.tickers_upserted,
        "tickers_with_data": report.tickers_with_data,
        "bars_inserted": report.bars_inserted,
        "fetch_failures": len(report.fetch_failures),
        "anomaly_count": len(report.anomalies),
    }
    print(json.dumps(summary, indent=2), file=sys.stderr)
    return 0 if report.tickers_with_data > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
