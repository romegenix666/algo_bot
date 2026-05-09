"""Initialise the local SQLite database (idempotent).

Usage::

    python -m scripts.init_db
    python -m scripts.init_db --drop      # destructive; recreate fresh

Run this once after cloning the repo. Subsequent runs are no-ops.
"""

from __future__ import annotations

import argparse
import sys

from src.data.storage import DataStore
from src.utils.logging import logger


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--drop",
        action="store_true",
        help="Drop and recreate all tables (DESTRUCTIVE).",
    )
    args = parser.parse_args()

    store = DataStore.from_settings()
    logger.info("Database URL: {}", store.database_url)

    if args.drop:
        logger.warning("Dropping all tables — this destroys local data!")
        confirm = input("Type 'YES' to confirm: ")
        if confirm.strip() != "YES":
            logger.info("Aborted.")
            return 1
        store.drop_all()

    store.create_all()
    logger.info("Schema ready. Tables: tickers, prices, corporate_actions, universe_snapshots")
    print("OK", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
