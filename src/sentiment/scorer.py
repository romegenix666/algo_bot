"""Article sentiment scoring + ticker matching.

Two scorers, with the heavier one optional:

    1. **VADER** (default) — rule-based, instant, no model download.
       Works "well enough" for headlines + summaries which is what we have.
    2. **FinBERT** (optional) — finance-tuned BERT, requires `transformers`
       + `torch`. Gives more nuanced scores but takes ~150ms per article on
       CPU. Loaded lazily — first use triggers the model download (~440 MB).

The scorer always returns ``score ∈ [-1, +1]`` and a discrete label
(``negative`` / ``neutral`` / ``positive``) so downstream code is
scorer-agnostic.

Ticker matching:
    A simple but effective approach — match against company name AND symbol
    AND a small alias list. Loaded from the ``tickers`` table at runtime,
    cached.

References:
    - Hutto & Gilbert (2014) — *VADER: A Parsimonious Rule-based Model*.
    - Araci (2019) — *FinBERT: Financial Sentiment Analysis with Pre-trained
      Language Models*. (`ProsusAI/finbert` on Hugging Face.)
    - Loughran & McDonald (2011) — finance-specific dictionaries.
    - Tetlock (2007) — sentiment as a *filter* not a primary signal.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from functools import lru_cache

import pandas as pd

from src.utils.logging import logger


class SentimentLabel(StrEnum):
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"


@dataclass(frozen=True)
class ScoringResult:
    score: float  # in [-1, +1]
    label: SentimentLabel
    scorer: str  # "vader" / "finbert"


# ---------------------------------------------------------------------------
# Scorers
# ---------------------------------------------------------------------------


class VaderScorer:
    """Rule-based sentiment via VADER. Bundled with `vaderSentiment`."""

    name = "vader"

    def __init__(self) -> None:
        try:
            from vaderSentiment.vaderSentiment import (
                SentimentIntensityAnalyzer,
            )
        except ImportError as exc:  # pragma: no cover - dev guard
            raise ImportError("Install vaderSentiment: `pip install vaderSentiment`") from exc
        self._analyzer = SentimentIntensityAnalyzer()

    def score(self, text: str) -> ScoringResult:
        if not text or not text.strip():
            return ScoringResult(0.0, SentimentLabel.NEUTRAL, self.name)
        scores = self._analyzer.polarity_scores(text)
        compound = float(scores["compound"])  # [-1, +1]
        if compound >= 0.05:
            label = SentimentLabel.POSITIVE
        elif compound <= -0.05:
            label = SentimentLabel.NEGATIVE
        else:
            label = SentimentLabel.NEUTRAL
        return ScoringResult(compound, label, self.name)


class FinBertScorer:
    """FinBERT — finance-tuned BERT. Loaded lazily."""

    name = "finbert"

    def __init__(self, model_id: str = "ProsusAI/finbert") -> None:
        self.model_id = model_id
        self._pipeline = None

    def _load(self):
        if self._pipeline is not None:
            return self._pipeline
        try:
            from transformers import pipeline
        except ImportError as exc:  # pragma: no cover - heavy optional dep
            raise ImportError(
                "FinBERT needs transformers + torch — `pip install transformers torch`."
            ) from exc
        logger.info("Loading FinBERT ({}) — first call may download ~440 MB.", self.model_id)
        self._pipeline = pipeline(
            "sentiment-analysis",
            model=self.model_id,
            tokenizer=self.model_id,
            top_k=None,
        )
        return self._pipeline

    def score(self, text: str) -> ScoringResult:
        if not text or not text.strip():
            return ScoringResult(0.0, SentimentLabel.NEUTRAL, self.name)
        text = text[:512]  # FinBERT max input
        clf = self._load()
        out = clf(text)
        # `out` is a list of dicts with keys label + score
        flat = out[0] if (out and isinstance(out[0], list)) else out
        prob_pos = next((d["score"] for d in flat if d["label"].lower() == "positive"), 0.0)
        prob_neg = next((d["score"] for d in flat if d["label"].lower() == "negative"), 0.0)
        compound = float(prob_pos - prob_neg)  # [-1, +1]
        if compound >= 0.15:
            label = SentimentLabel.POSITIVE
        elif compound <= -0.15:
            label = SentimentLabel.NEGATIVE
        else:
            label = SentimentLabel.NEUTRAL
        return ScoringResult(compound, label, self.name)


# ---------------------------------------------------------------------------
# Ticker matcher
# ---------------------------------------------------------------------------


@dataclass
class TickerMatcher:
    """Find which tickers an article is about.

    Strategy:
        - Build a regex of all symbols + names + aliases at construction.
        - Search title + summary; return the set of unique matches.

    We deliberately keep it simple — matching against well-known company
    names + NSE symbols. Custom aliases (e.g. "Adani Green" → ADANIGREEN)
    can be added per-ticker.
    """

    aliases: dict[str, list[str]] = field(default_factory=dict)
    _compiled: re.Pattern[str] | None = field(default=None, init=False)
    _alias_to_symbol: dict[str, str] = field(default_factory=dict, init=False)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> TickerMatcher:
        """Build from a frame with columns ``symbol`` and ``name``."""
        aliases: dict[str, list[str]] = {}
        for _, row in df.iterrows():
            sym = str(row["symbol"]).upper().strip()
            if not sym or sym.startswith("^"):
                continue
            names = [sym]
            n = row.get("name")
            if isinstance(n, str) and n.strip():
                names.append(n.strip())
                # Also add a "first two words" version (e.g. "Reliance Industries"
                # → "Reliance"). Helps articles that don't use the full name.
                tokens = n.strip().split()
                if len(tokens) >= 1:
                    names.append(tokens[0])
            aliases[sym] = list(dict.fromkeys(names))  # preserve order, dedupe
        return cls(aliases=aliases).compile()

    def compile(self) -> TickerMatcher:
        if not self.aliases:
            self._compiled = None
            return self
        # Build alias-to-symbol lookup (case-insensitive).
        lookup: dict[str, str] = {}
        for sym, names in self.aliases.items():
            for name in names:
                # Skip extremely short ambiguous names (e.g. "ITC" 3-letter is fine,
                # but a 2-letter alias would generate noise).
                if len(name) < 3:
                    continue
                lookup[name.lower()] = sym
        self._alias_to_symbol = lookup
        if not lookup:
            self._compiled = None
            return self
        # Sort longer aliases first so "Adani Green" matches before "Adani".
        keys = sorted(lookup.keys(), key=lambda k: -len(k))
        pattern = r"\b(" + "|".join(re.escape(k) for k in keys) + r")\b"
        self._compiled = re.compile(pattern, re.IGNORECASE)
        return self

    def find(self, text: str) -> set[str]:
        if not text or self._compiled is None:
            return set()
        out: set[str] = set()
        for match in self._compiled.finditer(text):
            sym = self._alias_to_symbol.get(match.group(0).lower())
            if sym:
                out.add(sym)
        return out


# ---------------------------------------------------------------------------
# Default factory — VADER, lazy FinBERT loaded only when requested.
# ---------------------------------------------------------------------------


@lru_cache(maxsize=2)
def get_scorer(name: str = "vader") -> VaderScorer | FinBertScorer:
    name = name.lower()
    if name == "vader":
        return VaderScorer()
    if name == "finbert":
        return FinBertScorer()
    raise ValueError(f"Unknown scorer '{name}' — choose 'vader' or 'finbert'")


__all__ = [
    "FinBertScorer",
    "ScoringResult",
    "SentimentLabel",
    "TickerMatcher",
    "VaderScorer",
    "get_scorer",
]
