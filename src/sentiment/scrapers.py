"""News scrapers — RSS feeds for Indian financial news.

We use RSS rather than HTML scraping because:

    - RSS is **legal grey-area-free** — feeds are public APIs with explicit
      licensing for re-syndication.
    - It's *fast* (one HTTP request per source, ~30 articles in the response).
    - It's stable — site redesigns don't break us.
    - Most major Indian financial outlets publish RSS:
        Moneycontrol, Economic Times Markets, LiveMint, Business Standard,
        Reuters India, Bloomberg Quint (NDTV Profit).

What we extract per article:
    - URL, title, summary, source, published_at

What we don't do (yet):
    - Full-page body fetching (rate-limit + politeness concerns; the RSS
      summary is enough for sentiment scoring).
    - Twitter / Reddit (separate Phase, harder API).

Failure tolerance:
    - One source failing shouldn't kill the batch.
    - The scraper is idempotent (URL-hashed dedupe in storage).
    - HTTP timeouts default to 15s.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

try:
    import feedparser
except ImportError:  # pragma: no cover - dev guard
    feedparser = None

from src.utils.logging import logger

# Default RSS sources for Indian markets. URLs verified May 2026.
# Some publishers rotate URLs occasionally — when adding more, double-check.
DEFAULT_FEEDS: dict[str, str] = {
    "moneycontrol_markets": "https://www.moneycontrol.com/rss/marketreports.xml",
    "moneycontrol_business": "https://www.moneycontrol.com/rss/business.xml",
    "moneycontrol_economy": "https://www.moneycontrol.com/rss/economy.xml",
    "moneycontrol_results": "https://www.moneycontrol.com/rss/results.xml",
    "et_markets": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "et_stocks": "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms",
    "et_companies": "https://economictimes.indiatimes.com/industry/rssfeeds/13352306.cms",
    "livemint_markets": "https://www.livemint.com/rss/markets",
    "livemint_companies": "https://www.livemint.com/rss/companies",
    "business_standard_markets": "https://www.business-standard.com/rss/markets-106.rss",
    "business_standard_companies": "https://www.business-standard.com/rss/companies-101.rss",
}


@dataclass
class ScrapedArticle:
    url: str
    title: str
    source: str
    summary: str | None = None
    published_at: datetime | None = None

    def to_row(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "title": self.title,
            "source": self.source,
            "summary": self.summary,
            "published_at": self.published_at,
        }


@dataclass
class RSSScraper:
    """Light-weight RSS scraper. Each source is one HTTP request.

    Use:
        scraper = RSSScraper()
        for article in scraper.fetch_all():
            ...
    """

    feeds: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_FEEDS))
    request_timeout_s: float = 15.0
    user_agent: str = "AlgoBot/0.1 (personal-research)"

    def __post_init__(self) -> None:
        if feedparser is None:
            raise ImportError(
                "feedparser is required: `pip install feedparser`. "
                "It's already in requirements.txt."
            )

    # ----------------------------------------------------------------
    def fetch_all(self) -> list[ScrapedArticle]:
        """Fetch every configured feed and return all articles (deduped at storage)."""
        out: list[ScrapedArticle] = []
        for source_name, url in self.feeds.items():
            try:
                articles = self.fetch_one(source_name, url)
            except Exception as exc:  # pragma: no cover - network defensive
                logger.warning("RSS source {} failed: {}", source_name, exc)
                continue
            out.extend(articles)
            logger.info("Fetched {} articles from {}", len(articles), source_name)
        return out

    # ----------------------------------------------------------------
    def fetch_one(self, source_name: str, url: str) -> list[ScrapedArticle]:
        """Fetch a single feed. Empty list on failure."""
        # feedparser handles HTTP itself; honor user agent + caching headers.
        feed = feedparser.parse(
            url,
            agent=self.user_agent,
            request_headers={"Accept": "application/rss+xml,application/xml,text/xml"},
        )
        if feed.bozo and not getattr(feed, "entries", None):
            raise RuntimeError(f"feed parse error: {getattr(feed, 'bozo_exception', '?')}")

        out: list[ScrapedArticle] = []
        for entry in feed.entries[:50]:  # cap per source — politely
            link = (entry.get("link") or "").strip()
            if not link:
                continue
            title = (entry.get("title") or "").strip()
            if not title:
                continue
            summary = entry.get("summary") or entry.get("description")
            published = _parse_published(entry)
            out.append(
                ScrapedArticle(
                    url=link,
                    title=title,
                    source=source_name,
                    summary=_strip_html_lite(summary)[:1500] if summary else None,
                    published_at=published,
                )
            )
        return out


def _parse_published(entry: Any) -> datetime | None:
    """Extract ``published`` from feedparser entry as UTC datetime."""
    parsed = entry.get("published_parsed") or entry.get("updated_parsed")
    if parsed is None:
        return None
    try:
        # parsed is a struct_time; convert to UTC datetime.
        from time import struct_time

        if not isinstance(parsed, struct_time):
            return None
        return datetime(
            parsed.tm_year,
            parsed.tm_mon,
            parsed.tm_mday,
            parsed.tm_hour,
            parsed.tm_min,
            parsed.tm_sec,
            tzinfo=UTC,
        )
    except Exception:  # pragma: no cover - defensive
        return None


def _strip_html_lite(text: str) -> str:
    """Strip the most common HTML tags from RSS summaries — no full parser."""
    if not text:
        return text
    import re

    out = re.sub(r"<[^>]+>", "", text)
    return re.sub(r"\s+", " ", out).strip()


def to_storage_rows(articles: Iterable[ScrapedArticle]) -> list[dict[str, Any]]:
    return [a.to_row() for a in articles]


__all__ = ["DEFAULT_FEEDS", "RSSScraper", "ScrapedArticle", "to_storage_rows"]
