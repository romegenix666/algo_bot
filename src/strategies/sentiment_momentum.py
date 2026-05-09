"""Sentiment-Augmented Momentum.

Pattern: **Decorator** over an existing momentum strategy.

Logic:
    1. Run an underlying ``MomentumStrategy`` (or any strategy that returns
       cross-sectional signals) to get a candidate list.
    2. Apply a sentiment filter:
         - Long signals are KEPT only if the rolling N-day sentiment EMA
           for that ticker is ≥ ``sentiment_long_min`` (default 0.0,
           i.e. "not negative").
         - Short signals are KEPT only if sentiment ≤ ``sentiment_short_max``.
    3. The conviction of the surviving signals is *boosted* by a small
       amount when sentiment is strongly aligned (e.g. very positive
       sentiment on a long), bounded to ``[0, 1]``.

Why use sentiment as a filter, not a primary signal:
    - Tetlock (2007) showed news sentiment predicts short-term returns,
      but the effect is small and noisy. Trading on sentiment alone has
      high turnover and underwhelming Sharpe.
    - As a *filter* on an existing edge, sentiment removes catastrophic
      shocks (fraud allegations, regulatory action, M&A failures) that
      pure momentum walks straight into. This is documented in Loughran
      & McDonald (2011) and Heston & Sinha (2017).

References:
    - Tetlock (2007). *Giving Content to Investor Sentiment*.
    - Loughran & McDonald (2011). *When is a Liability not a Liability?*
    - Heston & Sinha (2017). *News vs. Sentiment*.
    - Saha (2017) — Indian-market specific, Twitter-based.
    - Kakushadze & Serur (2018) §3.20 (Alpha combos / signal combinations).

Sentiment data shape:
    Caller passes a DataFrame ``sentiment`` with at minimum columns
    ``[ticker, score]`` (and optionally ``date`` if multi-day). Score in
    ``[-1, +1]`` with ``-1 = very negative``, ``+1 = very positive``.
"""

from __future__ import annotations

from typing import ClassVar

import pandas as pd

from src.strategies.base import (
    ExitDecision,
    MarketState,
    Position,
    RiskParams,
    Side,
    Signal,
    Strategy,
)
from src.strategies.momentum import MomentumStrategy


class SentimentMomentumStrategy(Strategy):
    """Wraps a momentum strategy with a news-sentiment filter."""

    name = "sentiment_momentum"
    timeframe = "1d"
    is_dollar_neutral: ClassVar[bool] = False

    def __init__(
        self,
        base: MomentumStrategy | None = None,
        sentiment_long_min: float = 0.0,
        sentiment_short_max: float = 0.0,
        sentiment_boost_threshold: float = 0.4,
        sentiment_boost: float = 0.15,
    ) -> None:
        self.base = base or MomentumStrategy()
        self.sentiment_long_min = sentiment_long_min
        self.sentiment_short_max = sentiment_short_max
        self.sentiment_boost_threshold = sentiment_boost_threshold
        self.sentiment_boost = sentiment_boost

    # ----------------------------------------------------------------------
    def required_features(self) -> list[str]:
        # In addition to base features, we need sentiment scores per ticker.
        return [*self.base.required_features(), "sentiment_score"]

    # ----------------------------------------------------------------------
    def generate_signals(
        self,
        prices: pd.DataFrame,
        features: pd.DataFrame,
        sentiment: pd.DataFrame | None = None,
    ) -> list[Signal]:
        # 1. Base signals.
        base_signals = self.base.generate_signals(prices, features, sentiment=None)
        if not base_signals:
            return []

        # 2. Build a ticker → score mapping. Defaults to 0 if missing.
        score_map: dict[str, float] = {}
        if sentiment is not None and not sentiment.empty:
            score_map = self._latest_scores(sentiment)

        filtered: list[Signal] = []
        for sig in base_signals:
            score = float(score_map.get(sig.ticker, 0.0))
            keep = True

            if (sig.side is Side.LONG and score < self.sentiment_long_min) or (
                sig.side is Side.SHORT and score > self.sentiment_short_max
            ):
                keep = False

            if not keep:
                continue

            # Conviction adjustment based on sentiment strength.
            new_conv = sig.conviction
            if (sig.side is Side.LONG and score >= self.sentiment_boost_threshold) or (
                sig.side is Side.SHORT and score <= -self.sentiment_boost_threshold
            ):
                new_conv = min(1.0, sig.conviction + self.sentiment_boost)

            filtered.append(
                Signal(
                    ticker=sig.ticker,
                    side=sig.side,
                    conviction=new_conv,
                    timestamp=sig.timestamp,
                    metadata={
                        **sig.metadata,
                        "base_conviction": sig.conviction,
                        "sentiment_score": score,
                    },
                )
            )
        return filtered

    # ----------------------------------------------------------------------
    @staticmethod
    def _latest_scores(sentiment: pd.DataFrame) -> dict[str, float]:
        """Pick the most recent sentiment score per ticker from the frame.

        Accepts either:
            - flat frame with columns ``[ticker, score]`` (already aggregated), or
            - long frame with ``[date, ticker, score]`` — we take the last
              date's score per ticker.
        """
        if "ticker" not in sentiment.columns or "score" not in sentiment.columns:
            return {}
        if "date" in sentiment.columns:
            df = sentiment.sort_values("date")
            return df.groupby("ticker", sort=False)["score"].last().to_dict()
        return sentiment.set_index("ticker")["score"].to_dict()

    # ----------------------------------------------------------------------
    def position_size(
        self,
        signal: Signal,
        risk: RiskParams,
        win_rate_estimate: float,
        win_loss_ratio_estimate: float,
    ) -> float:
        return self.base.position_size(signal, risk, win_rate_estimate, win_loss_ratio_estimate)

    # ----------------------------------------------------------------------
    def exit_rules(self, position: Position, market: MarketState) -> ExitDecision:
        return self.base.exit_rules(position, market)


__all__ = ["SentimentMomentumStrategy"]
