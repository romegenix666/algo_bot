"""Compute and persist a universe snapshot.

Usage::

    # Snapshot today using current store data
    python -m scripts.build_universe

    # Snapshot a specific date (must have history up to that date)
    python -m scripts.build_universe --as-of 2025-04-01

    # Custom size / liquidity threshold
    python -m scripts.build_universe --target-size 200 --min-turnover-cr 10

The snapshot is written to:
    - the ``universe_snapshots`` table in SQLite (point-in-time replay)
    - a CSV at ``data/universe/{YYYY-MM}.csv`` (human-readable + git-friendly)
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from src.data.storage import DataStore
from src.universe.selector import UniverseSelector
from src.utils.logging import logger
from src.utils.settings import PROJECT_ROOT


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", type=str, help="Snapshot date YYYY-MM-DD (default today)")
    parser.add_argument("--target-size", type=int, default=500)
    parser.add_argument("--price-min", type=float, default=50.0)
    parser.add_argument("--price-max", type=float, default=10_000.0)
    parser.add_argument("--min-turnover-cr", type=float, default=5.0)
    parser.add_argument("--turnover-window-days", type=int, default=30)
    parser.add_argument("--min-history-days", type=int, default=180)
    args = parser.parse_args()

    as_of = date.fromisoformat(args.as_of) if args.as_of else date.today()

    store = DataStore.from_settings()
    selector = UniverseSelector(
        store=store,
        target_size=args.target_size,
        price_min=args.price_min,
        price_max=args.price_max,
        min_avg_turnover_cr=args.min_turnover_cr,
        turnover_window_days=args.turnover_window_days,
        min_history_days=args.min_history_days,
    )

    entries = selector.select(as_of=as_of)
    if not entries:
        logger.error("No universe entries — make sure you've run init_db + fetch_history")
        return 2

    selector.save_snapshot(as_of, entries, notes=f"target={args.target_size}")

    out_dir: Path = PROJECT_ROOT / "data" / "universe"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{as_of.strftime('%Y-%m')}.csv"
    selector.to_dataframe(entries).to_csv(csv_path, index=False)

    logger.info("Saved {} entries to {} and to universe_snapshots table", len(entries), csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
