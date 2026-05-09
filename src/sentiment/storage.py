"""SQLAlchemy storage for raw articles + per-day-ticker sentiment scores.

Two tables:

    articles            — raw scraped articles (deduplicated by URL hash).
                          Columns: hash, url, source, title, summary,
                                   published_at, fetched_at, body_excerpt.
    sentiment_scores    — daily aggregated per-ticker sentiment in [-1, +1].
                          Columns: score_date, ticker, score, n_articles,
                                   sources, computed_at.

We keep raw articles around so we can rescore retroactively (e.g. when we
swap VADER for FinBERT or change the smoothing window).

All timestamps stored UTC.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, date, datetime
from hashlib import sha256
from typing import Any

import pandas as pd
from sqlalchemy import (
    Date,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    UniqueConstraint,
    select,
)
from sqlalchemy.orm import Mapped, mapped_column

from src.data.storage import Base, DataStore

# ---------------------------------------------------------------------------
# ORM models — registered against the same Base as the rest of the schema.
# ---------------------------------------------------------------------------


class Article(Base):
    __tablename__ = "articles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    url_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    url: Mapped[str] = mapped_column(String(512), nullable=False)
    source: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    summary: Mapped[str | None] = mapped_column(String(2048))
    body_excerpt: Mapped[str | None] = mapped_column(String(4096))
    published_at: Mapped[datetime | None] = mapped_column(DateTime, index=True)
    fetched_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))

    __table_args__ = (Index("ix_articles_published", "published_at"),)


class SentimentScore(Base):
    __tablename__ = "sentiment_scores"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    score_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    ticker: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    score: Mapped[float] = mapped_column(Float, nullable=False)  # raw daily score
    score_smoothed: Mapped[float] = mapped_column(Float, nullable=False)  # 7-day EMA
    n_articles: Mapped[int] = mapped_column(Integer, default=0)
    sources: Mapped[str | None] = mapped_column(String(255))
    computed_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))

    __table_args__ = (UniqueConstraint("score_date", "ticker", name="uq_sentiment_date_ticker"),)


# ---------------------------------------------------------------------------
# DAO additions — extension on DataStore via free functions.
# We keep these free functions (rather than methods on DataStore) so the
# DataStore class doesn't grow into a mega-class.
# ---------------------------------------------------------------------------


def url_hash(url: str) -> str:
    """Stable dedupe key — SHA-256 hex of normalised URL."""
    return sha256(url.strip().lower().encode("utf-8")).hexdigest()


def upsert_articles(store: DataStore, rows: Iterable[dict[str, Any]]) -> int:
    """Insert articles whose hash isn't already stored. Idempotent.

    Each row needs ``url`` + ``title`` + ``source``; everything else optional.
    Returns the number of NEW rows inserted (not total).
    """
    inserted = 0
    with store.session() as sess:
        existing_hashes = {h for (h,) in sess.execute(select(Article.url_hash)).all()}
        for row in rows:
            url = row.get("url", "").strip()
            if not url:
                continue
            h = url_hash(url)
            if h in existing_hashes:
                continue
            sess.add(
                Article(
                    url_hash=h,
                    url=url,
                    source=row.get("source", "unknown"),
                    title=(row.get("title") or "").strip()[:500],
                    summary=(row.get("summary") or None),
                    body_excerpt=(row.get("body_excerpt") or None),
                    published_at=row.get("published_at"),
                )
            )
            existing_hashes.add(h)
            inserted += 1
    return inserted


def fetch_articles(
    store: DataStore,
    since: datetime | None = None,
    until: datetime | None = None,
    sources: list[str] | None = None,
    limit: int | None = None,
) -> list[Article]:
    with store.session() as sess:
        stmt = select(Article).order_by(Article.published_at.desc())
        if since is not None:
            stmt = stmt.where(Article.published_at >= since)
        if until is not None:
            stmt = stmt.where(Article.published_at <= until)
        if sources:
            stmt = stmt.where(Article.source.in_(sources))
        if limit is not None:
            stmt = stmt.limit(limit)
        return list(sess.scalars(stmt).all())


def upsert_sentiment_scores(store: DataStore, rows: Iterable[dict[str, Any]]) -> int:
    """Insert/update one row per (score_date, ticker)."""
    n = 0
    with store.session() as sess:
        for row in rows:
            existing = sess.scalar(
                select(SentimentScore).where(
                    SentimentScore.score_date == row["score_date"],
                    SentimentScore.ticker == row["ticker"].upper(),
                )
            )
            if existing is None:
                sess.add(
                    SentimentScore(
                        score_date=row["score_date"],
                        ticker=row["ticker"].upper(),
                        score=float(row["score"]),
                        score_smoothed=float(row.get("score_smoothed", row["score"])),
                        n_articles=int(row.get("n_articles", 0)),
                        sources=row.get("sources"),
                    )
                )
            else:
                existing.score = float(row["score"])
                existing.score_smoothed = float(row.get("score_smoothed", row["score"]))
                existing.n_articles = int(row.get("n_articles", 0))
                existing.sources = row.get("sources")
                existing.computed_at = datetime.now(UTC)
            n += 1
    return n


def fetch_sentiment_panel(
    store: DataStore,
    since: date | None = None,
    until: date | None = None,
    tickers: list[str] | None = None,
) -> pd.DataFrame:
    """Return ``(score_date, ticker)`` indexed sentiment frame.

    Columns: ``score, score_smoothed, n_articles``.
    """
    with store.session() as sess:
        stmt = select(SentimentScore)
        if since is not None:
            stmt = stmt.where(SentimentScore.score_date >= since)
        if until is not None:
            stmt = stmt.where(SentimentScore.score_date <= until)
        if tickers:
            stmt = stmt.where(SentimentScore.ticker.in_([t.upper() for t in tickers]))
        rows = sess.scalars(stmt.order_by(SentimentScore.score_date)).all()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(
        [
            {
                "score_date": r.score_date,
                "ticker": r.ticker,
                "score": r.score,
                "score_smoothed": r.score_smoothed,
                "n_articles": r.n_articles,
            }
            for r in rows
        ]
    )
    return df.set_index(["score_date", "ticker"]).sort_index()


def latest_per_ticker_sentiment(
    store: DataStore, as_of: date, lookback_days: int = 7
) -> pd.DataFrame:
    """Return ``[ticker, score]`` rows — the latest *smoothed* score per ticker
    on or before ``as_of`` (only honouring a ``lookback_days`` window)."""
    from datetime import timedelta

    since = as_of - timedelta(days=lookback_days)
    with store.session() as sess:
        rows = sess.scalars(
            select(SentimentScore)
            .where(
                SentimentScore.score_date <= as_of,
                SentimentScore.score_date >= since,
            )
            .order_by(SentimentScore.score_date)
        ).all()
    if not rows:
        return pd.DataFrame(columns=["ticker", "score"])
    latest: dict[str, SentimentScore] = {}
    for r in rows:
        latest[r.ticker] = r  # last write wins per ticker (sorted asc by date)
    return pd.DataFrame([{"ticker": t, "score": r.score_smoothed} for t, r in latest.items()])


__all__ = [
    "Article",
    "SentimentScore",
    "fetch_articles",
    "fetch_sentiment_panel",
    "latest_per_ticker_sentiment",
    "upsert_articles",
    "upsert_sentiment_scores",
    "url_hash",
]
