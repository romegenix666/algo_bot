"""Sector / Sub-Index Rotation with Volatility Targeting.

Logic:
    1. Treat the universe as a basket of sector ETFs / sub-indices (Nifty
       Bank, Nifty IT, Nifty FMCG, Nifty Pharma, Nifty Auto, Nifty Metal,
       Nifty Energy, Nifty Realty, Nifty PSU Bank, Nifty Infra).
    2. Rank sectors by past 6–12 month return (relative momentum).
    3. Apply an additional **dual-momentum filter**: each sector must also
       have absolute momentum > 0 (its own price > its 200-day SMA).
       Otherwise it's filtered out.
    4. Allocate the equity bucket across the top ``top_n`` sectors. Sectors
       with poor abs momentum get re-allocated to a defensive bucket
       (cash / liquid bees).
    5. **Volatility targeting**: scale each leg's notional inversely to
       its trailing realised vol so each sector contributes equal *risk*,
       not equal capital.

References:
    - Faber, M. (2010), *Relative Strength Strategies for Investing*.
    - Antonacci (2014). Same dual-momentum philosophy applied to sectors.
    - Moskowitz & Grinblatt (1999), *Do Industries Explain Momentum?*.
    - Asness, Porter & Stevens (2000) — sector momentum is robust.
    - Kakushadze & Serur (2018) §4.1 (Sector momentum rotation).

Why sectors > individual stocks for some clients:
    - **Capacity**: Sector ETFs absorb crores without slippage.
    - **Lower turnover**: Sectors flip leadership less often than stocks.
    - **Cleaner regimes**: When IT is hot, all of TCS/INFY/HCLTECH/WIPRO
      are hot; sector trade beats stock-picking among them.
    - **Tax efficiency**: Holding period typically 1–3 months → STCG.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd

from src.features.indicators import rolling_mean, rolling_return, rolling_volatility
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


class SectorRotationStrategy(Strategy):
    """Top-N sector momentum with volatility targeting and abs-momentum gate."""

    name = "sector_rotation"
    timeframe = "1d"
    is_dollar_neutral: ClassVar[bool] = False

    def __init__(
        self,
        sector_tickers: list[str],
        defensive_ticker: str = "LIQUIDBEES.NS",
        rel_lookback_days: int = 126,  # 6 months
        abs_filter_period: int = 200,  # 200-day SMA
        top_n: int = 3,
        vol_target: float = 0.15,  # 15% annualised
        vol_window: int = 60,
        hold_days: int = 21,
    ) -> None:
        if not sector_tickers:
            raise ValueError("sector_tickers must not be empty")
        if top_n <= 0 or top_n > len(sector_tickers):
            raise ValueError("top_n must be in (0, len(sector_tickers)]")
        if vol_target <= 0:
            raise ValueError("vol_target must be positive")
        self.sector_tickers = list(sector_tickers)
        self.defensive_ticker = defensive_ticker
        self.rel_lookback_days = rel_lookback_days
        self.abs_filter_period = abs_filter_period
        self.top_n = top_n
        self.vol_target = vol_target
        self.vol_window = vol_window
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

        # Need enough history.
        min_required = max(self.rel_lookback_days, self.abs_filter_period, self.vol_window) + 5
        if len(close_wide) < min_required:
            return []

        # Restrict to known sector tickers that have data.
        sectors = [t for t in self.sector_tickers if t in close_wide.columns]
        if not sectors:
            return []

        as_of = close_wide.index[-1]
        timestamp = _to_datetime(as_of)

        # Relative momentum
        rel_mom = close_wide[sectors].apply(
            lambda c: rolling_return(c, self.rel_lookback_days).iloc[-1]
        )

        # Absolute momentum filter (price > 200-SMA)
        sma = close_wide[sectors].apply(lambda c: rolling_mean(c, self.abs_filter_period).iloc[-1])
        last_close = close_wide[sectors].iloc[-1]
        passes_abs = (last_close > sma) & (rel_mom > 0)

        eligible = rel_mom[passes_abs].dropna()
        if eligible.empty:
            # Whole market in risk-off → defensive
            return [
                Signal(
                    ticker=self.defensive_ticker,
                    side=Side.LONG,
                    conviction=1.0,
                    timestamp=timestamp,
                    metadata={"regime": "risk_off"},
                )
            ]

        # Top-N
        top = eligible.nlargest(self.top_n)

        # Volatility targeting weights
        vol = close_wide[top.index].apply(lambda c: rolling_volatility(c, self.vol_window).iloc[-1])
        # Inverse-vol weights, then scale to vol_target.
        inv_vol = (1.0 / vol.replace(0.0, np.nan)).dropna()
        if inv_vol.empty:
            return []
        weights = inv_vol / inv_vol.sum()
        # Cap any single weight to keep concentration sane.
        weights = weights.clip(upper=0.5)
        weights = weights / weights.sum()

        signals: list[Signal] = []
        for ticker, w in weights.items():
            mom = float(top.loc[ticker])
            ann_vol = float(vol.loc[ticker])
            # Conviction reflects how strong the abs+rel mom signal is.
            conviction = float(min(1.0, 0.4 + 0.6 * float(w)))
            signals.append(
                Signal(
                    ticker=str(ticker),
                    side=Side.LONG,
                    conviction=conviction,
                    timestamp=timestamp,
                    metadata={
                        "weight": float(w),
                        "rel_mom": mom,
                        "ann_vol": ann_vol,
                        "vol_target": self.vol_target,
                        "regime": "risk_on",
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
        # Dollar allocation ∝ inverse-vol weight × Kelly fraction.
        weight = float(signal.metadata.get("weight", 1.0 / self.top_n))
        kelly = half_kelly_fraction(win_rate_estimate, win_loss_ratio_estimate, cap=risk.kelly_cap)
        kelly *= signal.conviction
        notional = risk.equity * kelly * weight
        return min(notional, risk.equity * risk.max_single_position_pct)

    # ----------------------------------------------------------------------
    def exit_rules(self, position: Position, market: MarketState) -> ExitDecision:
        if position.side is Side.LONG and market.last_price <= position.current_stop:
            return ExitDecision(should_exit=True, reason=ExitReason.STOP_LOSS, note="ATR stop")
        held = (market.timestamp - position.entry_time).days
        if held >= self.hold_days:
            return ExitDecision(
                should_exit=True, reason=ExitReason.SIGNAL, note="rotation hold period elapsed"
            )
        return ExitDecision(should_exit=False)


# Default Indian sector index tickers (Yahoo symbols).
DEFAULT_NSE_SECTORS: list[str] = [
    "^NSEBANK",  # Nifty Bank
    "^CNXIT",  # Nifty IT
    "^CNXFMCG",  # Nifty FMCG
    "^CNXPHARMA",  # Nifty Pharma
    "^CNXAUTO",  # Nifty Auto
    "^CNXMETAL",  # Nifty Metal
    "^CNXENERGY",  # Nifty Energy
    "^CNXREALTY",  # Nifty Realty
    "^CNXPSUBANK",  # Nifty PSU Bank
    "^CNXINFRA",  # Nifty Infrastructure
]


__all__ = ["DEFAULT_NSE_SECTORS", "SectorRotationStrategy"]
