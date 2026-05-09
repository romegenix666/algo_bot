"""Cross-Sectional Momentum (12-1) — Jegadeesh & Titman (1993).

Logic:
    1. For each ticker on each rebalance date, compute the 12-month return
       *excluding the most recent month* (the "skip month" defeats the
       1-month reversal effect — Lehmann 1990, Jegadeesh 1990).
    2. Rank the universe cross-sectionally on this score.
    3. Long the top ``top_n`` (typically the top decile).
    4. (Optional) Short the bottom ``top_n`` for a dollar-neutral version.
    5. Rebalance monthly. Hold all positions until next rebalance.
    6. Conviction = scaled rank (best name = 1.0, last in long basket = 0.5).

References:
    - Jegadeesh, Titman (1993), *J. Finance*.
    - Asness, Moskowitz, Pedersen (2013), *Value and Momentum Everywhere*.
    - Sehgal, Jain (2011) — confirms momentum in Indian markets (NSE).
    - Kakushadze & Serur (2018) §3.1.

Why for India:
    The momentum anomaly is the most-replicated long-horizon effect globally
    AND is documented to work in NSE-500 (Sehgal et al.). Mid-cap momentum
    in India is particularly strong — a useful tilt for our universe.
"""

from __future__ import annotations

from datetime import datetime
from typing import ClassVar

import pandas as pd

from src.features.indicators import momentum_12_1
from src.risk.sizer import half_kelly_fraction, size_position
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


class MomentumStrategy(Strategy):
    """Cross-sectional 12-1 momentum, monthly rebalance, top-N long basket."""

    name = "momentum"
    timeframe = "1d"
    is_dollar_neutral: ClassVar[bool] = False

    def __init__(
        self,
        lookback_days: int = 252,
        skip_days: int = 21,
        top_n: int = 5,
        bottom_n: int = 0,
        min_history_days: int | None = None,
    ) -> None:
        if lookback_days <= skip_days:
            raise ValueError("lookback must be larger than skip")
        if top_n <= 0:
            raise ValueError("top_n must be positive")
        if bottom_n < 0:
            raise ValueError("bottom_n must be non-negative")
        self.lookback_days = lookback_days
        self.skip_days = skip_days
        self.top_n = top_n
        self.bottom_n = bottom_n
        self.min_history_days = min_history_days or (lookback_days + 5)

    # ----------------------------------------------------------------------
    def required_features(self) -> list[str]:
        return ["close", "atr_14"]

    # ----------------------------------------------------------------------
    def generate_signals(
        self,
        prices: pd.DataFrame,
        features: pd.DataFrame,
        sentiment: pd.DataFrame | None = None,
    ) -> list[Signal]:
        """Score each ticker by 12-1 return and pick the top/bottom N."""
        del sentiment  # not used by plain momentum
        del features  # we read close from prices

        if prices.empty:
            return []

        # ``prices`` has MultiIndex (date, ticker). Pivot to wide for vectorised math.
        if isinstance(prices.index, pd.MultiIndex):
            close_wide = prices["close"].unstack("ticker").sort_index()
        else:
            # Already wide: rows = dates, columns = tickers.
            close_wide = prices.sort_index()

        if len(close_wide) < self.min_history_days:
            return []

        as_of = close_wide.index[-1]
        # Compute 12-1 momentum for every ticker as of the last date.
        scores: pd.Series = close_wide.apply(
            lambda col: momentum_12_1(col, self.lookback_days, self.skip_days).iloc[-1]
        )
        scores = scores.dropna()
        if scores.empty:
            return []

        # Long basket: top_n highest scores. Ranked 1..top_n where 1 is best.
        longs = scores.nlargest(self.top_n)
        signals: list[Signal] = []

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
                        timestamp=_to_datetime(as_of),
                        metadata={"score": float(score), "rank": "top"},
                    )
                )

        if self.bottom_n > 0:
            shorts = scores.nsmallest(self.bottom_n)
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
                            timestamp=_to_datetime(as_of),
                            metadata={"score": float(score), "rank": "bottom"},
                        )
                    )

        return signals

    # ----------------------------------------------------------------------
    def position_size(
        self,
        signal: Signal,
        risk: RiskParams,
        win_rate_estimate: float,
        win_loss_ratio_estimate: float,
    ) -> float:
        """Returns rupee allocation; the order manager turns this into shares."""
        kelly = half_kelly_fraction(win_rate_estimate, win_loss_ratio_estimate, cap=risk.kelly_cap)
        kelly *= signal.conviction
        return min(
            risk.equity * kelly,
            risk.equity * risk.max_single_position_pct,
        )

    # ----------------------------------------------------------------------
    def exit_rules(self, position: Position, market: MarketState) -> ExitDecision:
        """Momentum exits are mostly handled by the monthly rebalance loop and
        the ATR trailing stop in the risk manager. The strategy itself only
        flags an exit if the position has been held longer than 90 days
        without rebalance (defensive) or the current bar's price has crossed
        the trailing stop."""
        if position.side is Side.LONG and market.last_price <= position.current_stop:
            return ExitDecision(should_exit=True, reason=ExitReason.STOP_LOSS, note="ATR stop hit")
        if position.side is Side.SHORT and market.last_price >= position.current_stop:
            return ExitDecision(should_exit=True, reason=ExitReason.STOP_LOSS, note="ATR stop hit")

        held_days = (market.timestamp - position.entry_time).days
        if held_days > 90:
            return ExitDecision(
                should_exit=True,
                reason=ExitReason.TIME_STOP,
                note="held > 90 days without rebalance",
            )

        return ExitDecision(should_exit=False)


def _to_datetime(idx: object) -> datetime:
    """Coerce a pandas index value into a python ``datetime``."""
    if isinstance(idx, datetime):
        return idx
    if isinstance(idx, pd.Timestamp):
        return idx.to_pydatetime()
    return pd.Timestamp(idx).to_pydatetime()


# Re-export the helper so other strategies can use the same coercion.
__all__ = ["MomentumStrategy"]


# Keep a reference so size_position remains importable for the tests.
_ = size_position
