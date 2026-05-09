"""Multi-Factor (Value + Momentum + Quality + Low-Vol) Composite.

Logic:
    1. For each ticker, compute four factor scores from rolling fundamental
       and price data:
         - **Value**:      lower P/E or P/B  → higher score
         - **Momentum**:   12-1 return        → higher score
         - **Quality**:    higher ROE, lower D/E → higher score
         - **Low-Vol**:    lower 90-day realised vol → higher score
    2. Cross-sectionally z-score each factor (so each factor contributes on
       a comparable scale).
    3. Composite = weighted average of factor z-scores (default equal weight).
    4. Long the top quintile (top 20%) by composite score; optionally short
       the bottom quintile.
    5. Rebalance quarterly. Low turnover → tax-efficient.

References:
    - Fama & French (1993, 2015). *Common Risk Factors / Five-Factor Model*.
    - Asness, Frazzini, Pedersen (2019). *Quality Minus Junk*.
    - Frazzini & Pedersen (2014). *Betting Against Beta*.
    - Carhart (1997). *On Persistence in Mutual Fund Performance* (4-factor).
    - Sehgal & Balakrishnan (2015) — confirms factor anomalies in NSE-500.
    - Kakushadze & Serur (2018) §3.6.

Why a composite, not factor-by-factor:
    Each individual factor has long, painful drawdowns. The combined score
    averages them out — when momentum has a bad year (2009, 2016, 2022 in
    India), value or quality usually picks up the slack. This is documented
    repeatedly in factor literature.

How to obtain Indian fundamentals (free):
    - ``yfinance`` returns ``Ticker.info`` with trailingPE, priceToBook,
      returnOnEquity, debtToEquity. Best-effort, sometimes stale.
    - ``screener.in`` scrape (legal grey area; rate-limit politely).
    - For backtests: download once a quarter and snapshot.
    - Caller passes the precomputed fundamentals DataFrame in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import pandas as pd

from src.features.indicators import momentum_12_1, rolling_volatility
from src.risk.sizer import half_kelly_fraction
from src.strategies.base import (
    ExitDecision,
    ExitReason,
    MarketState,
    Position,
    RiskParams,
    Side,
    Signal,
    Strategy,
)
from src.strategies.momentum import _to_datetime


@dataclass(frozen=True)
class FactorWeights:
    value: float = 0.25
    momentum: float = 0.25
    quality: float = 0.25
    low_vol: float = 0.25

    def normalised(self) -> FactorWeights:
        total = self.value + self.momentum + self.quality + self.low_vol
        if total <= 0:
            raise ValueError("factor weights must sum to a positive number")
        return FactorWeights(
            value=self.value / total,
            momentum=self.momentum / total,
            quality=self.quality / total,
            low_vol=self.low_vol / total,
        )


class MultiFactorStrategy(Strategy):
    """Composite of Value + Momentum + Quality + Low-Vol."""

    name = "multi_factor"
    timeframe = "1d"
    is_dollar_neutral: ClassVar[bool] = False

    def __init__(
        self,
        weights: FactorWeights | None = None,
        top_pct: float = 0.20,
        bottom_pct: float = 0.0,  # set > 0 for a long-short version
        vol_window: int = 90,
        momentum_lookback: int = 252,
        momentum_skip: int = 21,
        max_hold_days: int = 365,
    ) -> None:
        if not 0 < top_pct < 1:
            raise ValueError("top_pct must be in (0,1)")
        if not 0 <= bottom_pct < 1:
            raise ValueError("bottom_pct must be in [0,1)")
        self.weights = (weights or FactorWeights()).normalised()
        self.top_pct = top_pct
        self.bottom_pct = bottom_pct
        self.vol_window = vol_window
        self.momentum_lookback = momentum_lookback
        self.momentum_skip = momentum_skip
        self.max_hold_days = max_hold_days

    # ----------------------------------------------------------------------
    def required_features(self) -> list[str]:
        # Fundamentals are passed in via the ``features`` argument.
        return ["close", "atr_14", "pe_ratio", "pb_ratio", "roe", "debt_to_equity"]

    # ----------------------------------------------------------------------
    def generate_signals(
        self,
        prices: pd.DataFrame,
        features: pd.DataFrame,
        sentiment: pd.DataFrame | None = None,
    ) -> list[Signal]:
        del sentiment
        if prices.empty or features.empty:
            return []

        if isinstance(prices.index, pd.MultiIndex):
            close_wide = prices["close"].unstack("ticker").sort_index()
        else:
            close_wide = prices.sort_index()

        if len(close_wide) < self.momentum_lookback + 5:
            return []

        as_of = close_wide.index[-1]
        tickers = list(close_wide.columns)

        # ---- Compute each factor on the latest bar ----
        scores = pd.DataFrame(index=tickers, dtype=float)

        # Momentum
        mom = close_wide.apply(
            lambda c: momentum_12_1(c, self.momentum_lookback, self.momentum_skip).iloc[-1]
        )
        scores["momentum"] = mom

        # Low-Vol — invert so lower vol → higher score.
        vol = close_wide.apply(lambda c: rolling_volatility(c, self.vol_window).iloc[-1])
        scores["low_vol"] = -vol

        # Value & Quality come from the fundamentals frame supplied by caller.
        latest_fundamentals = (
            features.tail(1).reset_index(drop=True).set_index("ticker")
            if "ticker" in features.columns
            else features.iloc[[-1]]
        )

        scores["value"] = (
            -self._safe_pull(latest_fundamentals, "pe_ratio", tickers)  # lower P/E better
            - self._safe_pull(latest_fundamentals, "pb_ratio", tickers)  # lower P/B better
        ) / 2.0
        scores["quality"] = (
            self._safe_pull(latest_fundamentals, "roe", tickers)  # higher ROE better
            - self._safe_pull(latest_fundamentals, "debt_to_equity", tickers)  # lower D/E better
        ) / 2.0

        # ---- Cross-sectional z-score each factor, then combine ----
        z = (scores - scores.mean()) / scores.std(ddof=1).replace(0.0, np.nan)
        composite = (
            self.weights.value * z["value"].fillna(0.0)
            + self.weights.momentum * z["momentum"].fillna(0.0)
            + self.weights.quality * z["quality"].fillna(0.0)
            + self.weights.low_vol * z["low_vol"].fillna(0.0)
        )
        composite = composite.dropna()
        if composite.empty:
            return []

        # ---- Pick top quintile (and optionally bottom) ----
        n = len(composite)
        n_top = max(1, int(n * self.top_pct))
        n_bot = int(n * self.bottom_pct) if self.bottom_pct > 0 else 0

        signals: list[Signal] = []
        timestamp = _to_datetime(as_of)

        longs = composite.nlargest(n_top)
        if not longs.empty:
            best, worst = longs.max(), longs.min()
            spread = max(best - worst, 1e-9)
            for ticker, score in longs.items():
                conviction = 0.5 + 0.5 * (score - worst) / spread
                signals.append(
                    Signal(
                        ticker=str(ticker),
                        side=Side.LONG,
                        conviction=float(min(1.0, max(0.0, conviction))),
                        timestamp=timestamp,
                        metadata={
                            "composite": float(score),
                            "value_z": float(z.loc[ticker, "value"] or 0.0),
                            "momentum_z": float(z.loc[ticker, "momentum"] or 0.0),
                            "quality_z": float(z.loc[ticker, "quality"] or 0.0),
                            "low_vol_z": float(z.loc[ticker, "low_vol"] or 0.0),
                        },
                    )
                )

        if n_bot > 0:
            shorts = composite.nsmallest(n_bot)
            if not shorts.empty:
                best, worst = shorts.min(), shorts.max()
                spread = max(worst - best, 1e-9)
                for ticker, score in shorts.items():
                    conviction = 0.5 + 0.5 * (worst - score) / spread
                    signals.append(
                        Signal(
                            ticker=str(ticker),
                            side=Side.SHORT,
                            conviction=float(min(1.0, max(0.0, conviction))),
                            timestamp=timestamp,
                            metadata={"composite": float(score)},
                        )
                    )

        return signals

    # ----------------------------------------------------------------------
    @staticmethod
    def _safe_pull(frame: pd.DataFrame, col: str, tickers: list[str]) -> pd.Series:
        if col not in frame.columns:
            return pd.Series(0.0, index=tickers)
        return frame.reindex(tickers)[col].astype(float).fillna(frame[col].median())

    # ----------------------------------------------------------------------
    def position_size(
        self,
        signal: Signal,
        risk: RiskParams,
        win_rate_estimate: float,
        win_loss_ratio_estimate: float,
    ) -> float:
        kelly = half_kelly_fraction(win_rate_estimate, win_loss_ratio_estimate, cap=risk.kelly_cap)
        kelly *= signal.conviction
        return min(
            risk.equity * kelly,
            risk.equity * risk.max_single_position_pct,
        )

    # ----------------------------------------------------------------------
    def exit_rules(self, position: Position, market: MarketState) -> ExitDecision:
        if position.side is Side.LONG and market.last_price <= position.current_stop:
            return ExitDecision(should_exit=True, reason=ExitReason.STOP_LOSS, note="ATR stop")
        if position.side is Side.SHORT and market.last_price >= position.current_stop:
            return ExitDecision(should_exit=True, reason=ExitReason.STOP_LOSS, note="ATR stop")

        held = (market.timestamp - position.entry_time).days
        if held >= self.max_hold_days:
            return ExitDecision(
                should_exit=True, reason=ExitReason.TIME_STOP, note=f"held {held} days"
            )

        return ExitDecision(should_exit=False)


__all__ = ["FactorWeights", "MultiFactorStrategy"]
