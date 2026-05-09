"""Donchian Channel Breakout (Turtle-style trend following).

Logic:
    1. Compute the 20-day Donchian channel (rolling max-high / min-low,
       *excluding* the current bar — see ``donchian_channel`` for why).
    2. Optional regime/trend filter: only take longs when ADX > ``adx_min``
       (trend is strong) and price > 100-SMA (medium-term uptrend). Without
       this filter, breakout systems get whipsawed in choppy markets.
    3. Long entry: today's close pierces the upper Donchian band.
    4. Short entry (if enabled): close pierces the lower band.
    5. Exit: 10-day Donchian channel in the *opposite* direction (the
       classic Turtle "S2 system" exit). The 10-day channel exits faster
       than the 20-day entry channel so we cut losers quickly and let
       winners run until momentum fades.

References:
    - Faith, C. (2003), *The Way of the Turtle*. The Original Turtle System.
    - Donchian, R. (1960). 4-week rule trading.
    - Sullivan, Timmermann & White (1999) — caveats on data-snooping with
      moving-average / breakout systems.
    - Kakushadze & Serur (2018) §3.15 (Channel).
    - Wilder (1978) — ADX as a regime filter.

Trade-offs:
    Breakout systems have low win-rates (35–45%) but high win/loss ratios
    (2.5–4.0). Drawdowns of 25%+ are normal even for profitable systems.
    Use tight risk-budgeting and accept the discomfort.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd

from src.features.indicators import adx, donchian_channel, rolling_mean
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


class DonchianBreakoutStrategy(Strategy):
    """Donchian channel breakout with ADX trend filter."""

    name = "breakout"
    timeframe = "1d"
    is_dollar_neutral: ClassVar[bool] = False

    def __init__(
        self,
        entry_window: int = 20,
        exit_window: int = 10,
        adx_min: float = 25.0,
        adx_window: int = 14,
        sma_filter_period: int = 100,
        allow_short: bool = False,
        max_hold_days: int = 120,
    ) -> None:
        if exit_window >= entry_window:
            raise ValueError("exit window must be < entry window for asymmetric speed")
        self.entry_window = entry_window
        self.exit_window = exit_window
        self.adx_min = adx_min
        self.adx_window = adx_window
        self.sma_filter_period = sma_filter_period
        self.allow_short = allow_short
        self.max_hold_days = max_hold_days

    # ----------------------------------------------------------------------
    def required_features(self) -> list[str]:
        return ["open", "high", "low", "close", "volume", "atr_14"]

    # ----------------------------------------------------------------------
    def generate_signals(
        self,
        prices: pd.DataFrame,
        features: pd.DataFrame,
        sentiment: pd.DataFrame | None = None,
    ) -> list[Signal]:
        del sentiment, features

        if prices.empty:
            return []
        if not isinstance(prices.index, pd.MultiIndex):
            raise ValueError(
                "DonchianBreakoutStrategy needs OHLC; pass MultiIndex prices "
                "with columns open/high/low/close."
            )

        # Pivot OHLC into wide frames per channel.
        high = prices["high"].unstack("ticker").sort_index()
        low = prices["low"].unstack("ticker").sort_index()
        close = prices["close"].unstack("ticker").sort_index()

        min_required = max(self.entry_window, self.sma_filter_period, self.adx_window) + 5
        if len(close) < min_required:
            return []

        as_of = close.index[-1]
        signals: list[Signal] = []
        timestamp = _to_datetime(as_of)

        for ticker in close.columns:
            h = high[ticker].dropna()
            l = low[ticker].dropna()
            c = close[ticker].dropna()
            if min(len(h), len(l), len(c)) < min_required:
                continue
            # Align
            common = h.index.intersection(l.index).intersection(c.index)
            h, l, c = h.loc[common], l.loc[common], c.loc[common]

            channel = donchian_channel(h, l, self.entry_window)
            adx_df = adx(h, l, c, self.adx_window)
            sma = rolling_mean(c, self.sma_filter_period)

            last_close = float(c.iloc[-1])
            upper = float(channel["upper"].iloc[-1])
            lower = float(channel["lower"].iloc[-1])
            adx_val = float(adx_df["adx"].iloc[-1])
            sma_val = float(sma.iloc[-1])

            if any(np.isnan([upper, lower, adx_val, sma_val])):
                continue

            # Long breakout: close > upper, ADX strong, price above 100-SMA
            if last_close > upper and adx_val > self.adx_min and last_close > sma_val:
                # Conviction scales with how strong the ADX is and how far
                # past the breakout we've gone (capped).
                strength = min(1.0, (adx_val - self.adx_min) / 25.0)
                pierce = min(1.0, (last_close - upper) / max(upper - lower, 1e-9))
                conviction = float(min(1.0, 0.4 + 0.3 * strength + 0.3 * pierce))
                signals.append(
                    Signal(
                        ticker=str(ticker),
                        side=Side.LONG,
                        conviction=conviction,
                        timestamp=timestamp,
                        metadata={
                            "upper": upper,
                            "lower": lower,
                            "close": last_close,
                            "adx": adx_val,
                            "above_100sma": True,
                        },
                    )
                )
            elif (
                self.allow_short
                and last_close < lower
                and adx_val > self.adx_min
                and last_close < sma_val
            ):
                strength = min(1.0, (adx_val - self.adx_min) / 25.0)
                pierce = min(1.0, (lower - last_close) / max(upper - lower, 1e-9))
                conviction = float(min(1.0, 0.4 + 0.3 * strength + 0.3 * pierce))
                signals.append(
                    Signal(
                        ticker=str(ticker),
                        side=Side.SHORT,
                        conviction=conviction,
                        timestamp=timestamp,
                        metadata={
                            "upper": upper,
                            "lower": lower,
                            "close": last_close,
                            "adx": adx_val,
                            "below_100sma": True,
                        },
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

        # The classic Turtle exit (10-day channel against position) is checked
        # at the portfolio level by the order manager since it needs the live
        # high/low series. The strategy only enforces stops & time here.
        return ExitDecision(should_exit=False)


__all__ = ["DonchianBreakoutStrategy"]
