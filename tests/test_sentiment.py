"""Sentiment pipeline tests — storage, scoring, ticker matching, aggregation."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pandas as pd
import pytest

from src.data.storage import DataStore
from src.sentiment.aggregator import SentimentAggregator
from src.sentiment.scorer import (
    ScoringResult,
    SentimentLabel,
    TickerMatcher,
    VaderScorer,
)
from src.sentiment.storage import (
    Article,
    fetch_articles,
    fetch_sentiment_panel,
    latest_per_ticker_sentiment,
    upsert_articles,
    upsert_sentiment_scores,
    url_hash,
)

# ---------------------------------------------------------------------------
# url_hash + storage
# ---------------------------------------------------------------------------


def test_url_hash_is_stable() -> None:
    assert url_hash("  https://example.com/x ") == url_hash("https://example.com/X")


def test_url_hash_different_urls() -> None:
    assert url_hash("https://a.com") != url_hash("https://b.com")


@pytest.fixture
def store_with_tickers() -> DataStore:
    s = DataStore.in_memory()
    s.create_all()
    s.upsert_tickers(
        [
            {"symbol": "RELIANCE", "name": "Reliance Industries"},
            {"symbol": "TCS", "name": "Tata Consultancy Services"},
            {"symbol": "INFY", "name": "Infosys"},
            {"symbol": "HDFCBANK", "name": "HDFC Bank"},
        ]
    )
    return s


def test_upsert_articles_dedupes_by_url(store_with_tickers: DataStore) -> None:
    rows = [
        {"url": "https://x.com/a", "title": "T1", "source": "test"},
        {"url": "https://x.com/a", "title": "T1 dupe", "source": "test"},
        {"url": "https://x.com/b", "title": "T2", "source": "test"},
    ]
    n = upsert_articles(store_with_tickers, rows)
    assert n == 2  # only 2 unique URLs


def test_fetch_articles_filters_by_date(store_with_tickers: DataStore) -> None:
    upsert_articles(
        store_with_tickers,
        [
            {
                "url": f"https://x.com/{i}",
                "title": f"T{i}",
                "source": "test",
                "published_at": datetime(2024, 1, i, tzinfo=UTC),
            }
            for i in range(1, 6)
        ],
    )
    out = fetch_articles(
        store_with_tickers,
        since=datetime(2024, 1, 3, tzinfo=UTC),
    )
    # SQLite strips tzinfo on read-back; compare naively (date only).
    assert all(
        a.published_at is not None and a.published_at.date() >= date(2024, 1, 3) for a in out
    )


def test_sentiment_score_upsert_round_trip(store_with_tickers: DataStore) -> None:
    rows = [
        {
            "score_date": date(2024, 1, 5),
            "ticker": "RELIANCE",
            "score": 0.4,
            "score_smoothed": 0.4,
            "n_articles": 3,
        },
        {
            "score_date": date(2024, 1, 5),
            "ticker": "TCS",
            "score": -0.2,
            "score_smoothed": -0.2,
            "n_articles": 2,
        },
    ]
    n = upsert_sentiment_scores(store_with_tickers, rows)
    assert n == 2
    panel = fetch_sentiment_panel(store_with_tickers)
    assert not panel.empty
    assert (date(2024, 1, 5), "RELIANCE") in panel.index


def test_latest_sentiment_picks_most_recent(store_with_tickers: DataStore) -> None:
    upsert_sentiment_scores(
        store_with_tickers,
        [
            {
                "score_date": date(2024, 1, 1),
                "ticker": "RELIANCE",
                "score": 0.0,
                "score_smoothed": 0.0,
            },
            {
                "score_date": date(2024, 1, 5),
                "ticker": "RELIANCE",
                "score": 0.5,
                "score_smoothed": 0.5,
            },
        ],
    )
    out = latest_per_ticker_sentiment(store_with_tickers, as_of=date(2024, 1, 6), lookback_days=10)
    assert len(out) == 1
    assert out["score"].iloc[0] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# VaderScorer
# ---------------------------------------------------------------------------


def test_vader_positive_text() -> None:
    s = VaderScorer().score("Reliance reports record profits, surges to all-time high")
    assert s.score > 0.2
    assert s.label is SentimentLabel.POSITIVE


def test_vader_negative_text() -> None:
    s = VaderScorer().score("Tata Steel posts massive losses; CEO resigns amid fraud allegations")
    assert s.score < -0.2
    assert s.label is SentimentLabel.NEGATIVE


def test_vader_neutral_text() -> None:
    s = VaderScorer().score("HDFC Bank announces a board meeting on Tuesday")
    assert -0.4 < s.score < 0.4


def test_vader_handles_empty() -> None:
    s = VaderScorer().score("")
    assert s.label is SentimentLabel.NEUTRAL
    assert s.score == 0.0


# ---------------------------------------------------------------------------
# TickerMatcher
# ---------------------------------------------------------------------------


def test_ticker_matcher_finds_symbol_and_name() -> None:
    df = pd.DataFrame(
        [
            {"symbol": "RELIANCE", "name": "Reliance Industries"},
            {"symbol": "TCS", "name": "Tata Consultancy Services"},
        ]
    )
    matcher = TickerMatcher.from_dataframe(df)
    found = matcher.find("Reliance Industries reports record results.")
    assert found == {"RELIANCE"}


def test_ticker_matcher_finds_multiple() -> None:
    df = pd.DataFrame(
        [
            {"symbol": "RELIANCE", "name": "Reliance Industries"},
            {"symbol": "TCS", "name": "Tata Consultancy Services"},
        ]
    )
    matcher = TickerMatcher.from_dataframe(df)
    found = matcher.find("Reliance and TCS both posted strong Q2 numbers.")
    assert {"RELIANCE", "TCS"} <= found


def test_ticker_matcher_handles_no_aliases() -> None:
    matcher = TickerMatcher(aliases={}).compile()
    assert matcher.find("anything") == set()


def test_ticker_matcher_skips_short_aliases() -> None:
    """2-letter aliases would generate noise — we filter them out."""
    matcher = TickerMatcher.from_dataframe(
        pd.DataFrame([{"symbol": "GE", "name": "General Electric"}])
    )
    # The 2-letter "GE" symbol gets skipped, but "General Electric" works.
    assert matcher.find("ge ge ge ge") == set()
    assert matcher.find("General Electric had a great Q1") == {"GE"}


# ---------------------------------------------------------------------------
# SentimentAggregator
# ---------------------------------------------------------------------------


def _article(
    url: str,
    title: str,
    summary: str = "",
    when: datetime | None = None,
    source: str = "test",
) -> Article:
    """Build an Article ORM object directly (no DB needed for aggregator tests)."""
    return Article(
        url_hash=url_hash(url),
        url=url,
        title=title,
        summary=summary,
        source=source,
        published_at=when or datetime(2024, 1, 5, tzinfo=UTC),
    )


def test_aggregator_groups_by_date_and_ticker() -> None:
    df = pd.DataFrame(
        [
            {"symbol": "RELIANCE", "name": "Reliance Industries"},
            {"symbol": "TCS", "name": "Tata Consultancy Services"},
        ]
    )
    matcher = TickerMatcher.from_dataframe(df)
    agg = SentimentAggregator(matcher=matcher, scorer=VaderScorer())

    articles = [
        _article(
            "https://x.com/a",
            "Reliance Industries reports record profits",
            when=datetime(2024, 1, 5, tzinfo=UTC),
        ),
        _article(
            "https://x.com/b",
            "Reliance Industries faces regulatory probe",
            when=datetime(2024, 1, 5, tzinfo=UTC),
        ),
        _article(
            "https://x.com/c",
            "Tata Consultancy Services wins major contract, surges",
            when=datetime(2024, 1, 5, tzinfo=UTC),
        ),
    ]
    raw = agg.aggregate(articles)
    assert not raw.empty
    by_ticker = raw.set_index(["score_date", "ticker"])
    # RELIANCE has 2 articles
    assert by_ticker.loc[(date(2024, 1, 5), "RELIANCE"), "n_articles"] == 2
    assert by_ticker.loc[(date(2024, 1, 5), "TCS"), "n_articles"] == 1


def test_aggregator_returns_empty_for_articles_without_tickers() -> None:
    matcher = TickerMatcher.from_dataframe(
        pd.DataFrame([{"symbol": "RELIANCE", "name": "Reliance Industries"}])
    )
    agg = SentimentAggregator(matcher=matcher, scorer=VaderScorer())
    articles = [_article("https://x.com/a", "Random news about something else")]
    raw = agg.aggregate(articles)
    assert raw.empty


def test_aggregator_smooths_with_ema() -> None:
    raw = pd.DataFrame(
        [
            {"score_date": date(2024, 1, d), "ticker": "RELIANCE", "score": v, "n_articles": 1}
            for d, v in zip([1, 2, 3, 4, 5], [-0.5, 0.0, 0.5, 0.0, 0.5], strict=True)
        ]
    )
    matcher = TickerMatcher.from_dataframe(
        pd.DataFrame([{"symbol": "RELIANCE", "name": "Reliance Industries"}])
    )
    agg = SentimentAggregator(matcher=matcher, scorer=VaderScorer(), smoothing_window=3)
    smoothed = agg.smooth(raw)
    # First raw value bootstraps EMA; the rest should be a smoothed value
    last = smoothed.iloc[-1]
    assert -0.5 < last["score_smoothed"] < 0.6


def test_aggregate_and_store_persists(store_with_tickers: DataStore) -> None:
    matcher = TickerMatcher.from_dataframe(
        pd.DataFrame([{"symbol": "RELIANCE", "name": "Reliance Industries"}])
    )
    agg = SentimentAggregator(matcher=matcher, scorer=VaderScorer(), smoothing_window=3)

    upsert_articles(
        store_with_tickers,
        [
            {
                "url": f"https://x.com/{i}",
                "title": "Reliance Industries reports record profits",
                "source": "test",
                "published_at": datetime(2024, 1, 5 + i, tzinfo=UTC),
            }
            for i in range(3)
        ],
    )
    articles = fetch_articles(store_with_tickers)
    assert articles
    n = agg.aggregate_and_store(store_with_tickers, articles)
    assert n > 0
    panel = fetch_sentiment_panel(store_with_tickers)
    assert "score_smoothed" in panel.columns
    assert not panel.empty


# Stub so static analysers see we kept the imports referenced
_ = (timedelta, ScoringResult)
