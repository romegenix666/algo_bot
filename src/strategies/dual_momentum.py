"""Dual Momentum (Antonacci 2014).

Logic — at each rebalance (monthly):
    1. **Absolute momentum filter** (the trend gate): is the broad market
       index (Nifty 50 / Nifty 500) up over the last ``abs_lookback`` months
       relative to a risk-free benchmark?
       - YES → equity allocation is enabled.
       - NO  → switch to a risk-free / cash / gold-ETF defensive bucket.
    2. **Relative momentum**: among the equity universe, rank by 12-month
       return. Long the top ``top_n`` (typically just one or a handful).
    3. Hold for ``hold_days`` (default ~21 = 1 month). Re-rank every cycle.

Why dual momentum is special:
    - Solves the central weakness of pure relative momentum: when the entire
      market is crashing, even the "best" stock is still going down. By
      adding the absolute-momentum filter, dual momentum sidesteps bear
      markets entirely (reduced 2008 drawdown by ~half in Antonacci's
      backtests).
    - Has only 2 parameters (lookback, top_n) → very low overfit risk.
    - Antonacci's backtest: GEM (Global Equities Momentum) returned ~16%
      CAGR with ~18% max DD over 1971–2014 — dramatically better than
      buy-and-hold.

References:
    - Antonacci, G. (2014). *Dual Momentum Investing*. McGraw-Hill.
    - Antonacci, G. (2017). *Risk Premia Harvesting Through Dual Momentum*.
    - Moskowitz, Ooi & Pedersen (2012). *Time Series Momentum*.
    - Faber, M. (2007). *A Quantitative Approach to Tactical Asset Allocation*
      (the simpler 10-month MA filter cousin of dual momentum).

For India:
    - Use Nifty 50 (or Nifty 500) as the broad market index proxy.
    - Risk-free fallback: short-duration G-Sec ETF (e.g. ``LIQUIDBEES.NS``)
      or simply 100% cash.
    - Equity universe: top 100 Nifty stocks by liquidity.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd

from src.features.indicators import rolling_return
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


class DualMomentumStrategy(Strategy):
    """Antonacci-style dual momentum (absolute + relative)."""

    name = "dual_momentum"
    timeframe = "1d"
    is_dollar_neutral: ClassVar[bool] = False

    def __init__(
        self,
        market_index_ticker: str = "^NSEI",  # Nifty 50
        defensive_ticker: str = "LIQUIDBEES.NS",  # cash-equivalent ETF
        abs_lookback_days: int = 252,
        rel_lookback_days: int = 252,
        top_n: int = 5,
        hold_days: int = 21,
    ) -> None:
        if top_n <= 0:
            raise ValueError("top_n must be positive")
        self.market_index_ticker = market_index_ticker
        self.defensive_ticker = defensive_ticker
        self.abs_lookback_days = abs_lookback_days
        self.rel_lookback_days = rel_lookback_days
        self.top_n = top_n
        self.hold_days = hold_days

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

        if len(close_wide) < max(self.abs_lookback_days, self.rel_lookback_days) + 5:
            return []

        as_of = close_wide.index[-1]
        timestamp = _to_datetime(as_of)

        # ---- Absolute momentum gate ----
        if self.market_index_ticker not in close_wide.columns:
            # Without a market index series we cannot apply the abs filter;
            # fall back to defensive.
            return [self._defensive_signal(timestamp)]

        market_close = close_wide[self.market_index_ticker].dropna()
        if len(market_close) < self.abs_lookback_days + 1:
            return [self._defensive_signal(timestamp)]
        market_return = float(rolling_return(market_close, self.abs_lookback_days).iloc[-1])
        if np.isnan(market_return) or market_return <= 0.0:
            # Market is down over the lookback → go defensive.
            return [self._defensive_signal(timestamp)]

        # ---- Relative momentum (only if abs filter passes) ----
        equity_close = close_wide.drop(
            columns=[
                c
                for c in (self.market_index_ticker, self.defensive_ticker)
                if c in close_wide.columns
            ],
            errors="ignore",
        )
        scores = equity_close.apply(
            lambda c: rolling_return(c, self.rel_lookback_days).iloc[-1]
        ).dropna()
        scores = scores[scores > 0.0]  # Antonacci further requires positive abs momentum per asset
        if scores.empty:
            return [self._defensive_signal(timestamp)]

        longs = scores.nlargest(self.top_n)
        signals: list[Signal] = []
        if not longs.empty:
            best, worst = float(longs.max()), float(longs.min())
            spread = max(best - worst, 1e-9)
            for ticker, score in longs.items():
                conviction = 0.5 + 0.5 * (float(score) - worst) / spread
                signals.append(
                    Signal(
                        ticker=str(ticker),
                        side=Side.LONG,
                        conviction=float(min(1.0, max(0.0, conviction))),
                        timestamp=timestamp,
                        metadata={
                            "score": float(score),
                            "market_return": market_return,
                            "regime": "risk_on",
                        },
                    )
                )
        return signals

    # ----------------------------------------------------------------------
    def _defensive_signal(self, timestamp) -> Signal:
        return Signal(
            ticker=self.defensive_ticker,
            side=Side.LONG,
            conviction=1.0,
            timestamp=timestamp,
            metadata={"regime": "risk_off", "reason": "abs_momentum_negative"},
        )

    # ----------------------------------------------------------------------
    def position_size(
        self,
        signal: Signal,
        risk: RiskParams,
        win_rate_estimate: float,
        win_loss_ratio_estimate: float,
    ) -> float:
        if signal.metadata.get("regime") == "risk_off":
            # Allocate ~100% to the defensive bucket — capital preservation.
            return risk.equity * risk.max_single_position_pct
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

        held = (market.timestamp - position.entry_time).days
        if held >= self.hold_days:
            # Hold-period elapsed → next rebalance picks fresh names.
            return ExitDecision(
                should_exit=True,
                reason=ExitReason.SIGNAL,
                note=f"hold period {self.hold_days} elapsed → rebalance",
            )

        return ExitDecision(should_exit=False)


__all__ = ["DualMomentumStrategy"]
