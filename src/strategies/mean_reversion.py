"""Bollinger + RSI Mean-Reversion (single-stock).

Logic (long only by default):
    1. Filter the universe to stocks whose price *changes* are stationary
       (Augmented Dickey-Fuller p-value < 0.05 over the lookback). Most
       stocks are NOT mean-reverting in price; only changes/spreads are.
       This is critical — Chan §7 warns explicitly that picking momentum
       stocks vs. mean-revert stocks is a regime-based decision.
    2. Long entry when ALL of:
        a. Close < lower Bollinger band (20, 2σ)  — price is "stretched"
        b. RSI(14) < 30                            — oversold on momentum
        c. Above the 200-SMA                       — only buy dips in uptrends
    3. Exit when ANY of:
        a. RSI(14) > 50                            — momentum recovered
        b. Close > 20-SMA (Bollinger mid)          — reverted to mean
        c. ATR stop tripped                        — handled by risk module
        d. Held > ``max_hold_days``                — time stop, no infinite holds

References:
    - Chan (2009), *Quantitative Trading*, Chapter 7.
    - Bollinger (2002), *Bollinger on Bollinger Bands*.
    - Connors & Alvarez (2009), *Short Term Trading Strategies That Work*
      (RSI(2) variant).
    - Kakushadze & Serur (2018) §3.9, §4.4.

A note on the long bias:
    Indian retail traders cannot easily short stocks (intraday only via SEBI
    rules; no naked overnight short). We default to long-only. The class
    exposes a ``allow_short`` flag for paper / backtest use.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from src.features.indicators import bollinger_bands, rolling_mean, rsi
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


class MeanReversionStrategy(Strategy):
    """Bollinger + RSI mean reversion with stationarity gate."""

    name = "mean_reversion"
    timeframe = "1d"
    is_dollar_neutral: ClassVar[bool] = False

    def __init__(
        self,
        bb_period: int = 20,
        bb_std: float = 2.0,
        rsi_period: int = 14,
        rsi_entry: float = 30.0,
        rsi_exit: float = 50.0,
        trend_filter_period: int = 200,
        adf_pvalue_max: float = 0.05,
        adf_lookback_days: int = 252,
        max_hold_days: int = 10,
        allow_short: bool = False,
    ) -> None:
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.rsi_period = rsi_period
        self.rsi_entry = rsi_entry
        self.rsi_exit = rsi_exit
        self.trend_filter_period = trend_filter_period
        self.adf_pvalue_max = adf_pvalue_max
        self.adf_lookback_days = adf_lookback_days
        self.max_hold_days = max_hold_days
        self.allow_short = allow_short

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
        del sentiment, features

        if prices.empty:
            return []

        if isinstance(prices.index, pd.MultiIndex):
            close_wide = prices["close"].unstack("ticker").sort_index()
        else:
            close_wide = prices.sort_index()

        # Need enough history for the trend filter and the ADF lookback.
        min_required = max(
            self.trend_filter_period + 5,
            self.adf_lookback_days + 5,
        )
        if len(close_wide) < min_required:
            return []

        as_of = close_wide.index[-1]
        signals: list[Signal] = []

        for ticker in close_wide.columns:
            close = close_wide[ticker].dropna()
            if len(close) < min_required:
                continue

            # ---- Stationarity gate (ADF on returns) ----
            if not self._is_stationary(close.tail(self.adf_lookback_days)):
                continue

            bands = bollinger_bands(close, self.bb_period, self.bb_std)
            rsi_series = rsi(close, self.rsi_period)
            sma = rolling_mean(close, self.trend_filter_period)

            last_close = float(close.iloc[-1])
            lower = float(bands["lower"].iloc[-1])
            upper = float(bands["upper"].iloc[-1])
            mid = float(bands["mid"].iloc[-1])
            last_rsi = float(rsi_series.iloc[-1])
            last_sma = float(sma.iloc[-1])

            if any(np.isnan([lower, upper, last_rsi, last_sma, mid])):
                continue

            # Long entry: oversold + below band + above 200-SMA (uptrend dip).
            if last_close < lower and last_rsi < self.rsi_entry and last_close > last_sma:
                # Conviction: how far below the band, scaled.
                stretch = (lower - last_close) / max(lower - mid, 1e-9)
                conviction = min(1.0, 0.4 + 0.6 * float(stretch))
                signals.append(
                    Signal(
                        ticker=str(ticker),
                        side=Side.LONG,
                        conviction=conviction,
                        timestamp=_to_datetime(as_of),
                        metadata={
                            "rsi": last_rsi,
                            "stretch": float(stretch),
                            "close": last_close,
                            "lower_band": lower,
                            "above_200sma": True,
                        },
                    )
                )

            # Short entry (only if allowed): overbought + above band + below 200-SMA.
            elif self.allow_short and (
                last_close > upper and last_rsi > (100.0 - self.rsi_entry) and last_close < last_sma
            ):
                stretch = (last_close - upper) / max(mid - upper, 1e-9) * -1.0
                conviction = min(1.0, 0.4 + 0.6 * float(stretch))
                signals.append(
                    Signal(
                        ticker=str(ticker),
                        side=Side.SHORT,
                        conviction=conviction,
                        timestamp=_to_datetime(as_of),
                        metadata={
                            "rsi": last_rsi,
                            "stretch": float(stretch),
                            "close": last_close,
                            "upper_band": upper,
                            "below_200sma": True,
                        },
                    )
                )

        return signals

    # ----------------------------------------------------------------------
    def _is_stationary(self, series: pd.Series) -> bool:
        """Run an Augmented Dickey-Fuller test on *returns* of the series.

        Stocks rarely have stationary *prices* (random walk), but their
        *returns* often are. The mean-reversion logic on Bollinger bands is
        equivalent to assuming price changes (around a moving mean) revert.
        """
        returns = series.pct_change().dropna()
        if len(returns) < 30:
            return False
        try:
            result = adfuller(returns, autolag="AIC", regression="c")
        except Exception:  # pragma: no cover - defensive
            return False
        pvalue = float(result[1])
        return pvalue < self.adf_pvalue_max

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
        # ATR stop (defensive double-check; risk manager is the primary owner)
        if position.side is Side.LONG and market.last_price <= position.current_stop:
            return ExitDecision(should_exit=True, reason=ExitReason.STOP_LOSS, note="ATR stop")
        if position.side is Side.SHORT and market.last_price >= position.current_stop:
            return ExitDecision(should_exit=True, reason=ExitReason.STOP_LOSS, note="ATR stop")

        # Time stop — mean-reversion that hasn't reverted in 10 days is broken.
        held = (market.timestamp - position.entry_time).days
        if held >= self.max_hold_days:
            return ExitDecision(
                should_exit=True,
                reason=ExitReason.TIME_STOP,
                note=f"held {held} days, max {self.max_hold_days}",
            )

        # Strategy exits: RSI recovered, or price reverted past mid.
        meta = position.metadata or {}
        midpoint = meta.get("midpoint")
        last_rsi = meta.get("last_rsi")  # written by feature pipeline at update time

        if position.side is Side.LONG:
            if last_rsi is not None and last_rsi > self.rsi_exit:
                return ExitDecision(
                    should_exit=True, reason=ExitReason.SIGNAL, note="RSI recovered"
                )
            if midpoint is not None and market.last_price > midpoint:
                return ExitDecision(
                    should_exit=True, reason=ExitReason.SIGNAL, note="reverted to mean"
                )

        if position.side is Side.SHORT:
            if last_rsi is not None and last_rsi < (100.0 - self.rsi_exit):
                return ExitDecision(
                    should_exit=True, reason=ExitReason.SIGNAL, note="RSI recovered"
                )
            if midpoint is not None and market.last_price < midpoint:
                return ExitDecision(
                    should_exit=True, reason=ExitReason.SIGNAL, note="reverted to mean"
                )

        return ExitDecision(should_exit=False)


__all__ = ["MeanReversionStrategy"]
