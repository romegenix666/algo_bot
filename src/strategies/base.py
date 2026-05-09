"""Abstract Strategy base class.

Every concrete strategy (momentum, mean-reversion, pairs, multi-factor,
sentiment-momentum) implements this interface. The backtester and the live
order manager only know about ``Strategy``; they do not know how a particular
strategy works internally.

This is the **Strategy design pattern** — interchangeable algorithms behind a
stable interface.

References:
    - Chan (2009), "Quantitative Trading", esp. Ch. 7 on strategy categories.
    - Kakushadze & Serur (2018), "151 Trading Strategies", §3 Stocks.
    - Gang of Four, *Design Patterns* (1994), Strategy chapter.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any, ClassVar

import pandas as pd


class Side(StrEnum):
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


class ExitReason(StrEnum):
    SIGNAL = "signal"  # Strategy says exit (e.g., z-score crossed)
    STOP_LOSS = "stop_loss"  # ATR / fixed stop hit
    TRAIL_STOP = "trail_stop"  # Trailing stop hit
    TAKE_PROFIT = "take_profit"  # Profit target hit
    TIME_STOP = "time_stop"  # Max-hold-days reached
    REGIME = "regime"  # Regime detector deactivated this strategy
    CIRCUIT = "circuit"  # Portfolio circuit-breaker tripped
    MANUAL = "manual"  # Human kill-switch


@dataclass(frozen=True)
class Signal:
    """A directional signal for one ticker on one bar.

    Conviction is a float in ``[0, 1]`` — the strategy's confidence. Position
    sizer multiplies this into the Half-Kelly fraction.
    """

    ticker: str
    side: Side
    conviction: float
    timestamp: datetime
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.conviction <= 1.0:
            raise ValueError(f"conviction must be in [0,1], got {self.conviction}")


@dataclass
class Position:
    ticker: str
    side: Side
    quantity: int
    entry_price: float
    entry_time: datetime
    initial_stop: float
    current_stop: float
    strategy_name: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExitDecision:
    should_exit: bool
    reason: ExitReason | None = None
    note: str | None = None


@dataclass(frozen=True)
class MarketState:
    """Slice of market context handed to ``Strategy.exit_rules`` per bar."""

    timestamp: datetime
    last_price: float
    atr: float
    realised_vol: float
    regime: str | None = None


@dataclass(frozen=True)
class RiskParams:
    equity: float
    per_trade_pct: float
    half_kelly: bool
    kelly_cap: float
    max_single_position_pct: float


class Strategy(ABC):
    """Base class — concrete strategies subclass this.

    Subclasses MUST set ``name`` and implement the four abstract methods. They
    SHOULD keep parameter count ≤ 5 (Chan's overfitting rule of thumb) and
    must declare every parameter via the constructor so backtests can sweep
    them without monkey-patching.
    """

    name: ClassVar[str] = "abstract"
    timeframe: ClassVar[str] = "1d"
    is_dollar_neutral: ClassVar[bool] = False  # subclass may override

    @abstractmethod
    def required_features(self) -> list[str]:
        """Names of feature columns this strategy needs from the feature store.

        e.g. ``["close", "volume", "atr_14", "rsi_14"]``.
        """

    @abstractmethod
    def generate_signals(
        self,
        prices: pd.DataFrame,
        features: pd.DataFrame,
        sentiment: pd.DataFrame | None = None,
    ) -> list[Signal]:
        """Produce signals for the *current* bar.

        Args:
            prices: OHLCV per ticker, MultiIndex (date, ticker), columns
                ``open, high, low, close, volume``. Sorted ascending by date.
                **Only data on or before the current bar is allowed** — strategies
                that look at the future are bugs (look-ahead bias).
            features: Feature dataframe matching ``required_features()``.
            sentiment: Optional daily sentiment per ticker, columns ``["score"]``
                in ``[-1, +1]``.

        Returns:
            List of ``Signal`` objects — possibly empty.
        """

    @abstractmethod
    def position_size(
        self,
        signal: Signal,
        risk: RiskParams,
        win_rate_estimate: float,
        win_loss_ratio_estimate: float,
    ) -> float:
        """Return the rupee allocation for this signal (positive number).

        Default behaviour for most subclasses: Half-Kelly capped by
        ``risk.kelly_cap`` and ``risk.max_single_position_pct``. Subclasses can
        override (e.g. pairs trading sizes both legs).
        """

    @abstractmethod
    def exit_rules(
        self,
        position: Position,
        market: MarketState,
    ) -> ExitDecision:
        """Decide whether to exit ``position`` given current ``market`` state.

        Strategy-specific exits go here (e.g. RSI > 50 for mean-reversion).
        ATR stop-loss and circuit-breaker logic live in the risk manager,
        not in the strategy.
        """

    # ------------------------------------------------------------------
    # Default helpers — strategies can override but usually don't need to.
    # ------------------------------------------------------------------

    @staticmethod
    def _half_kelly_fraction(
        win_rate: float,
        win_loss_ratio: float,
        cap: float,
    ) -> float:
        if win_loss_ratio <= 0:
            return 0.0
        full = (win_rate * win_loss_ratio - (1.0 - win_rate)) / win_loss_ratio
        half = max(0.0, full / 2.0)
        return min(half, cap)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Strategy name={self.name!r} timeframe={self.timeframe!r}>"


__all__ = [
    "ExitDecision",
    "ExitReason",
    "MarketState",
    "Position",
    "RiskParams",
    "Side",
    "Signal",
    "Strategy",
]
