"""Aggregator — turns scored articles into per-day-ticker sentiment scores.

Pipeline:

    1. For each article, find which tickers it mentions (TickerMatcher).
    2. Score the article (VADER / FinBERT) → compound score in [-1, +1].
    3. Group by (publish_date, ticker) and AVERAGE article scores.
    4. Apply a 7-day exponentially-weighted average per ticker for smoothing.
    5. Persist to ``sentiment_scores`` table.

Only smoothed scores are exposed to the strategies — single-day spikes
are too noisy.

Why the 7-day EMA (not a simple mean)?
    News flows in bursts. EMA recovers from a single negative day faster
    than a simple mean would, which is the *right* behaviour: a one-day
    fraud rumour that gets debunked the next day shouldn't permanently
    lock our bot out of that name.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime

import pandas as pd

from src.data.storage import DataStore
from src.sentiment.scorer import (
    ScoringResult,
    TickerMatcher,
    VaderScorer,
)
from src.sentiment.storage import (
    Article,
    fetch_articles,
    fetch_sentiment_panel,
    upsert_sentiment_scores,
)
from src.utils.logging import logger


@dataclass
class SentimentAggregator:
    """Score + dedupe + smooth pipeline. Stateless across days."""

    matcher: TickerMatcher
    scorer: VaderScorer  # any object with .score() returning ScoringResult
    smoothing_window: int = 7  # days
    min_articles_for_smoothing: int = 1

    # ----------------------------------------------------------------
    def aggregate(self, articles: Iterable[Article]) -> pd.DataFrame:
        """Return a frame with ``[score_date, ticker, score, n_articles, sources]``.

        ``score`` here is the *raw* daily mean — no smoothing yet.
        """
        # bucket: (date, ticker) → list of (score, source)
        buckets: dict[tuple[date, str], list[tuple[float, str]]] = defaultdict(list)

        for article in articles:
            text = " ".join(filter(None, [article.title, article.summary or ""]))
            tickers = self.matcher.find(text)
            if not tickers:
                continue
            try:
                result: ScoringResult = self.scorer.score(text)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Sentiment score failed: {}", exc)
                continue
            d = (article.published_at or datetime.now(UTC)).date()
            for ticker in tickers:
                buckets[(d, ticker)].append((result.score, article.source))

        rows = []
        for (d, ticker), items in buckets.items():
            scores = [s for s, _src in items]
            sources = sorted({src for _s, src in items})
            rows.append(
                {
                    "score_date": d,
                    "ticker": ticker,
                    "score": float(sum(scores) / len(scores)),
                    "n_articles": len(scores),
                    "sources": ",".join(sources)[:255],
                }
            )
        return pd.DataFrame(rows).sort_values(["score_date", "ticker"]) if rows else pd.DataFrame()

    # ----------------------------------------------------------------
    def smooth(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Add a 7-day EMA-smoothed column ``score_smoothed``."""
        if raw.empty:
            raw["score_smoothed"] = raw["score"] if "score" in raw.columns else []
            return raw
        out = raw.copy()
        out["score_smoothed"] = out["score"]
        # Per-ticker EMA over score_date (sorted ascending)
        out = out.sort_values(["ticker", "score_date"])
        out["score_smoothed"] = out.groupby("ticker")["score"].transform(
            lambda s: s.ewm(span=self.smoothing_window, adjust=False).mean()
        )
        return out

    # ----------------------------------------------------------------
    def aggregate_and_store(self, store: DataStore, articles: Iterable[Article]) -> int:
        """Score, smooth, and persist. Returns rows touched.

        After this call, ``fetch_sentiment_panel`` returns up-to-date data.
        """
        # Pull existing scores so the EMA is continuous across runs.
        existing = fetch_sentiment_panel(store)
        new_raw = self.aggregate(articles)

        if existing.empty and new_raw.empty:
            return 0

        # Stitch new + existing (raw scores), then re-smooth.
        if not existing.empty:
            existing_flat = existing.reset_index()[["score_date", "ticker", "score", "n_articles"]]
        else:
            existing_flat = pd.DataFrame(columns=["score_date", "ticker", "score", "n_articles"])

        if new_raw.empty:
            combined = existing_flat.copy()
            combined["sources"] = None
        else:
            new_raw = new_raw.copy()
            existing_keys = set(
                zip(
                    existing_flat["score_date"],
                    existing_flat["ticker"],
                    strict=False,
                )
            )
            new_keys_mask = ~new_raw.apply(
                lambda r: (r["score_date"], r["ticker"]) in existing_keys, axis=1
            )
            new_only = new_raw[new_keys_mask]

            # For overlapping (date, ticker), prefer the new score (recomputed).
            overlapping = new_raw[~new_keys_mask].set_index(["score_date", "ticker"])
            existing_indexed = existing_flat.set_index(["score_date", "ticker"])
            for key, row in overlapping.iterrows():
                if key in existing_indexed.index:
                    existing_indexed.loc[key, "score"] = row["score"]
                    existing_indexed.loc[key, "n_articles"] = row["n_articles"]

            combined = pd.concat(
                [existing_indexed.reset_index(), new_only],
                ignore_index=True,
            )

        smoothed = self.smooth(combined)
        rows = [
            {
                "score_date": r["score_date"],
                "ticker": r["ticker"],
                "score": float(r["score"]),
                "score_smoothed": float(r["score_smoothed"]),
                "n_articles": int(r.get("n_articles", 0) or 0),
                "sources": r.get("sources"),
            }
            for _, r in smoothed.iterrows()
        ]
        n = upsert_sentiment_scores(store, rows)
        logger.info(
            "Sentiment aggregator: stored {} rows ({} from new articles)",
            n,
            len(new_raw) if isinstance(new_raw, pd.DataFrame) else 0,
        )
        return n


def build_default_aggregator(store: DataStore) -> SentimentAggregator:
    """Create an aggregator wired with VADER + ticker list from the DB."""
    tickers = store.list_tickers(status="active")
    df = pd.DataFrame([{"symbol": t.symbol, "name": t.name or ""} for t in tickers])
    matcher = TickerMatcher.from_dataframe(df)
    return SentimentAggregator(matcher=matcher, scorer=VaderScorer())


__all__ = ["SentimentAggregator", "build_default_aggregator", "fetch_articles"]
