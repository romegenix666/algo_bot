"""Refresh news + per-ticker sentiment scores.

Pipeline (run hourly or once a day):

    1. Pull RSS feeds → upsert into ``articles`` (idempotent by URL hash).
    2. Score every article (VADER) → group by (date, ticker).
    3. Smooth with a 7-day EMA → upsert ``sentiment_scores``.

Usage::

    python -m scripts.refresh_sentiment                # full refresh
    python -m scripts.refresh_sentiment --since 7      # rescore last 7 days
    python -m scripts.refresh_sentiment --no-fetch     # rescore stored articles only
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from src.data.storage import DataStore
from src.sentiment.aggregator import build_default_aggregator
from src.sentiment.dedupe import dedupe_scraped_rows, recent_article_content_hashes
from src.sentiment.scrapers import RSSScraper, to_storage_rows
from src.sentiment.storage import (
    fetch_articles,
    upsert_articles,
)
from src.utils.logging import logger


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--since", type=int, default=14, help="Look-back window in days for sentiment computation."
    )
    parser.add_argument(
        "--no-fetch", action="store_true", help="Skip RSS fetch — only re-score existing articles."
    )
    parser.add_argument(
        "--dedupe-near",
        type=float,
        default=0.92,
        help="Fuzzy dedupe threshold in [0,1]; 0 disables fuzzy (exact hash only). Default 0.92.",
    )
    parser.add_argument(
        "--parquet-snapshot",
        type=str,
        default=None,
        help="If set, write raw scraped rows to this parquet file (parent dirs created).",
    )
    args = parser.parse_args()

    store = DataStore.from_settings()
    store.create_all()

    scraped = []
    if not args.no_fetch:
        scraper = RSSScraper()
        scraped = scraper.fetch_all()
        rows = to_storage_rows(scraped)
        prior_hashes = recent_article_content_hashes(store, limit=10_000)
        fuzzy = args.dedupe_near if args.dedupe_near > 0 else None
        rows = dedupe_scraped_rows(
            rows, existing_hashes=prior_hashes, near_duplicate_min_ratio=fuzzy
        )
        n_inserted = upsert_articles(store, rows)
        logger.info(
            "RSS fetch: scraped {} articles, {} after dedupe, {} new in DB",
            len(scraped),
            len(rows),
            n_inserted,
        )
        if args.parquet_snapshot:
            snap = Path(args.parquet_snapshot)
            snap.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(scraped).to_parquet(snap, index=False)
            logger.info("Parquet snapshot → {}", snap)
    else:
        n_inserted = 0

    since = datetime.now(UTC) - timedelta(days=args.since)
    articles = fetch_articles(store, since=since)
    logger.info("Scoring {} articles (since {})", len(articles), since.date())

    aggregator = build_default_aggregator(store)
    n_scored = aggregator.aggregate_and_store(store, articles)

    summary = {
        "fetched": len(scraped),
        "new_articles": n_inserted,
        "scored_rows": n_scored,
    }
    json.dump(summary, sys.stderr, default=str)
    sys.stderr.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
