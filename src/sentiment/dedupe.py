"""Content-level deduplication for scraped articles (beyond URL hashing).

``storage.upsert_articles`` already skips identical URLs. This module catches:

    - Same story under different URLs.
    - Near-duplicate headlines + summaries inside one scrape batch.

Uses SHA-256 fingerprints on normalised text plus optional ``difflib`` fuzzy
matching (stdlib only).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from difflib import SequenceMatcher
from hashlib import sha256
from typing import Any, TypeVar

from sqlalchemy import select

from src.data.storage import DataStore
from src.sentiment.storage import Article

T = TypeVar("T", bound=Mapping[str, Any])


def normalise_text(s: str | None) -> str:
    return " ".join((s or "").split()).strip().lower()


def article_content_hash(title: str | None, summary: str | None) -> str:
    """Stable hash of semantic content (not URL)."""
    blob = normalise_text(title) + "|" + normalise_text(summary)
    return sha256(blob.encode("utf-8")).hexdigest()


def combined_text(row: Mapping[str, Any]) -> str:
    return normalise_text(row.get("title")) + " " + normalise_text(row.get("summary"))


def content_similarity_row(x: Mapping[str, Any], y: Mapping[str, Any]) -> float:
    return SequenceMatcher(None, combined_text(x), combined_text(y)).ratio()


def recent_article_content_hashes(store: DataStore, *, limit: int = 10_000) -> set[str]:
    """Fingerprints for the most recently inserted rows (by primary key)."""
    out: set[str] = set()
    with store.session() as sess:
        q = select(Article.title, Article.summary).order_by(Article.id.desc()).limit(max(1, limit))
        for title, summary in sess.execute(q).all():
            if not (title or summary):
                continue
            out.add(article_content_hash(title, summary))
    return out


def dedupe_scraped_rows(
    rows: Iterable[T],
    *,
    existing_hashes: set[str] | None = None,
    near_duplicate_min_ratio: float | None = 0.92,
) -> list[T]:
    """Drop exact duplicates, optionally drop near-duplicates within the batch.

    Order is preserved for the first occurrence. When ``existing_hashes`` is
    provided, rows whose fingerprint already exists there are skipped and new
    fingerprints are added so later rows in this batch don't duplicate earlier
    accepted ones.
    """
    rows_list = list(rows)
    seen: set[str] = set(existing_hashes) if existing_hashes is not None else set()
    out: list[T] = []

    for row in rows_list:
        url = (row.get("url") or "").strip()
        title = (row.get("title") or "").strip()
        if not url and not title:
            continue
        fp = article_content_hash(row.get("title"), row.get("summary"))
        if fp in seen:
            continue
        if near_duplicate_min_ratio is not None and near_duplicate_min_ratio > 0:
            dup = False
            for prior in out:
                if content_similarity_row(row, prior) >= near_duplicate_min_ratio:
                    dup = True
                    break
            if dup:
                continue
        seen.add(fp)
        out.append(row)

    return out


__all__ = [
    "article_content_hash",
    "content_similarity_row",
    "dedupe_scraped_rows",
    "normalise_text",
    "recent_article_content_hashes",
]
