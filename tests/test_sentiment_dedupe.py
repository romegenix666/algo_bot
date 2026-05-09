"""Tests for sentiment content deduplication."""

from __future__ import annotations

from src.sentiment.dedupe import (
    article_content_hash,
    dedupe_scraped_rows,
)


def test_article_content_hash_stable() -> None:
    h1 = article_content_hash("Reliance Gains", "Stock up 2%")
    h2 = article_content_hash("  reliance gains  ", "Stock up 2%")
    assert h1 == h2


def test_dedupe_exact_content() -> None:
    rows = [
        {"url": "http://a/1", "title": "Same story", "summary": "Body"},
        {"url": "http://b/2", "title": "Same story", "summary": "Body"},
    ]
    out = dedupe_scraped_rows(rows, existing_hashes=None, near_duplicate_min_ratio=None)
    assert len(out) == 1
    assert out[0]["url"] == "http://a/1"


def test_dedupe_fuzzy_near_duplicate() -> None:
    rows = [
        {
            "url": "http://a/1",
            "title": "Stock market today: Sensex gains 500 points on broad rally",
            "summary": "Mumbai",
        },
        {
            "url": "http://b/2",
            "title": "Stock market today Sensex gains 500 points on broad rally",
            "summary": "Mumbai",
        },
    ]
    out = dedupe_scraped_rows(rows, existing_hashes=set(), near_duplicate_min_ratio=0.92)
    assert len(out) == 1


def test_dedupe_respects_existing_hashes() -> None:
    rows = [{"url": "http://x", "title": "Only title", "summary": None}]
    h = article_content_hash("Only title", None)
    out = dedupe_scraped_rows(rows, existing_hashes={h}, near_duplicate_min_ratio=None)
    assert out == []
