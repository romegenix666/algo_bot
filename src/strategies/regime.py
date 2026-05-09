"""Market Regime Detector + Strategy Allocation Map.

Why we need this:
    No single strategy works in every market condition.

    - Momentum thrives in trending markets, dies in choppy ones.
    - Mean-reversion thrives in choppy / range-bound markets, gets crushed
      by big trends.
    - Pairs trading is regime-agnostic but is hurt when correlations break
      (typically high-vol crisis periods).
    - Multi-factor is the long-horizon "always on" baseline.

    We classify the *current* market regime daily using simple, robust
    metrics (annualised vol + trend score) and produce an allocation
    weight per strategy. This is the only place strategy weights are
    decided — every other module just reads the output.

Regimes:
    - **Trending Low-Vol**   (best of all worlds): big momentum allocation.
    - **Trending High-Vol**  (manic markets):       smaller momentum + breakout, hold cash.
    - **Range-Bound Low-Vol**(boring):              mean-reversion + pairs heavy.
    - **Choppy High-Vol**    (crisis / panic):      pairs only, lots of cash.

Inputs:
    A ``pandas.Series`` of the broad market index closes (Nifty 50 daily by
    default), ascending date index. Method ``classify`` returns a regime
    label and per-strategy weights.

References:
    - Hamilton (1989). *Markov-Switching Regimes* (canonical regime model).
    - Gatev, Goetzmann & Rouwenhorst (2006) on pairs vs. regimes.
    - Asness, Israelov & Liew (2011). *International Diversification Works*.
    - Chan §7 — *"Mean-reverting regimes are more prevalent than trending."*
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pandas as pd

from src.features.indicators import atr, rolling_mean, rolling_volatility


class Regime(StrEnum):
    TRENDING_UP_LOW_VOL = "trending_up_low_vol"  # strong uptrend, calm
    TRENDING_UP_HIGH_VOL = "trending_up_high_vol"  # strong uptrend, manic
    TRENDING_DOWN_LOW_VOL = "trending_down_low_vol"  # bear market, orderly
    TRENDING_DOWN_HIGH_VOL = "trending_down_high_vol"  # crash / panic
    RANGE_LOW_VOL = "range_low_vol"  # boring, sideways
    CHOPPY_HIGH_VOL = "choppy_high_vol"  # crisis-y, no clear trend
    UNKNOWN = "unknown"

    # Backwards-compatible aliases to the previous (direction-less) regime
    # names — old configs / saved snapshots can still reference them.
    TRENDING_LOW_VOL = "trending_up_low_vol"
    TRENDING_HIGH_VOL = "trending_up_high_vol"


@dataclass(frozen=True)
class RegimeDiagnostics:
    """Numbers the detector used so we can audit decisions."""

    realised_vol_annual: float
    trend_score: float
    sma_long: float
    last_price: float
    regime: Regime


@dataclass(frozen=True)
class RegimeAllocation:
    """Per-strategy weights summing to <= 1.0. Remainder = cash."""

    weights: dict[str, float]
    regime: Regime
    diagnostics: RegimeDiagnostics

    @property
    def cash_weight(self) -> float:
        return max(0.0, 1.0 - sum(self.weights.values()))


# Default allocation map — read once, can be overridden by config/yaml.
# Direction matters: long-only strategies must NOT be heavy in downtrends.
DEFAULT_ALLOCATION_MAP: dict[Regime, dict[str, float]] = {
    # Strong uptrend, calm: best-case for long momentum.
    Regime.TRENDING_UP_LOW_VOL: {
        "momentum": 0.35,
        "multi_factor": 0.20,
        "breakout": 0.20,
        "dual_momentum": 0.15,
        "sector_rotation": 0.10,
        # cash 0%
    },
    # Strong uptrend, manic vol: same direction but smaller, more cash.
    Regime.TRENDING_UP_HIGH_VOL: {
        "momentum": 0.20,
        "breakout": 0.15,
        "multi_factor": 0.15,
        "pairs": 0.10,
        # 40% cash buffer
    },
    # Bear market, orderly: long-only is dangerous. Lean on dual_momentum
    # (which has an absolute-momentum filter and goes to cash automatically),
    # pairs (market-neutral), multi-factor (low-vol/quality bias buys
    # defensive names that hold up).
    Regime.TRENDING_DOWN_LOW_VOL: {
        "dual_momentum": 0.20,  # auto-defensive
        "pairs": 0.20,  # market-neutral
        "multi_factor": 0.15,  # quality + low-vol tilt
        "mean_reversion": 0.10,  # buy oversold dips, fast time-stop
        # 35% cash
    },
    # Crash / panic: extreme defensive. Capital preservation > returns.
    Regime.TRENDING_DOWN_HIGH_VOL: {
        "dual_momentum": 0.15,  # 100% defensive bucket from this strategy
        "pairs": 0.10,  # tight neutral exposure
        # 75% cash
    },
    # Sideways, calm: mean-reversion's heyday.
    Regime.RANGE_LOW_VOL: {
        "mean_reversion": 0.35,
        "pairs": 0.25,
        "multi_factor": 0.20,
        "sector_rotation": 0.10,
        # 10% cash
    },
    # No clear direction, high vol: pairs only.
    Regime.CHOPPY_HIGH_VOL: {
        "pairs": 0.25,
        "multi_factor": 0.10,
        "dual_momentum": 0.10,
        # 55% cash — defensive
    },
    Regime.UNKNOWN: {
        "multi_factor": 0.30,
        # mostly cash until we have enough history
    },
}


class RegimeDetector:
    """Classify the current market into one of four regimes."""

    def __init__(
        self,
        vol_window: int = 60,
        trend_window: int = 200,
        high_vol_threshold: float = 0.22,  # ~22% annualised
        trending_score_threshold: float = 1.0,
        allocation_map: dict[Regime, dict[str, float]] | None = None,
    ) -> None:
        self.vol_window = vol_window
        self.trend_window = trend_window
        self.high_vol_threshold = high_vol_threshold
        self.trending_score_threshold = trending_score_threshold
        self.allocation_map = allocation_map or DEFAULT_ALLOCATION_MAP

    # ----------------------------------------------------------------------
    def classify(self, index_ohlc: pd.DataFrame) -> RegimeAllocation:
        """Classify current regime from the broad market index OHLC frame.

        Args:
            index_ohlc: DataFrame indexed by date with columns
                ``high, low, close`` for the market index (e.g. Nifty 50).
        """
        if {"high", "low", "close"} - set(index_ohlc.columns):
            raise ValueError("index_ohlc must have columns: high, low, close")

        close = index_ohlc["close"].dropna()
        high = index_ohlc["high"].dropna()
        low = index_ohlc["low"].dropna()

        if len(close) < max(self.vol_window, self.trend_window) + 5:
            diag = RegimeDiagnostics(
                realised_vol_annual=float("nan"),
                trend_score=float("nan"),
                sma_long=float("nan"),
                last_price=float(close.iloc[-1]) if len(close) else float("nan"),
                regime=Regime.UNKNOWN,
            )
            return RegimeAllocation(
                weights=self.allocation_map.get(Regime.UNKNOWN, {}),
                regime=Regime.UNKNOWN,
                diagnostics=diag,
            )

        last_price = float(close.iloc[-1])
        rv = rolling_volatility(close, self.vol_window).iloc[-1]
        sma_long = rolling_mean(close, self.trend_window).iloc[-1]
        atr_long = atr(high, low, close, self.trend_window).iloc[-1]

        if any(pd.isna(v) for v in (rv, sma_long, atr_long)):
            return self._unknown(last_price)

        # Trend score: how many ATRs is the price above (or below) the long SMA?
        trend_score = float((last_price - sma_long) / atr_long) if atr_long > 0 else 0.0
        rv_annual = float(rv)

        is_high_vol = rv_annual > self.high_vol_threshold
        is_trending = abs(trend_score) > self.trending_score_threshold
        is_uptrend = trend_score > 0

        if is_trending and is_uptrend and not is_high_vol:
            regime = Regime.TRENDING_UP_LOW_VOL
        elif is_trending and is_uptrend and is_high_vol:
            regime = Regime.TRENDING_UP_HIGH_VOL
        elif is_trending and not is_uptrend and not is_high_vol:
            regime = Regime.TRENDING_DOWN_LOW_VOL
        elif is_trending and not is_uptrend and is_high_vol:
            regime = Regime.TRENDING_DOWN_HIGH_VOL
        elif not is_trending and not is_high_vol:
            regime = Regime.RANGE_LOW_VOL
        else:
            regime = Regime.CHOPPY_HIGH_VOL

        diag = RegimeDiagnostics(
            realised_vol_annual=rv_annual,
            trend_score=trend_score,
            sma_long=float(sma_long),
            last_price=last_price,
            regime=regime,
        )
        return RegimeAllocation(
            weights=dict(self.allocation_map.get(regime, {})),
            regime=regime,
            diagnostics=diag,
        )

    # ----------------------------------------------------------------------
    def _unknown(self, last_price: float) -> RegimeAllocation:
        diag = RegimeDiagnostics(
            realised_vol_annual=float("nan"),
            trend_score=float("nan"),
            sma_long=float("nan"),
            last_price=last_price,
            regime=Regime.UNKNOWN,
        )
        return RegimeAllocation(
            weights=self.allocation_map.get(Regime.UNKNOWN, {}),
            regime=Regime.UNKNOWN,
            diagnostics=diag,
        )


# Convenience: detect regime and return only the allocation weights dict.
def detect_allocation(
    index_ohlc: pd.DataFrame,
    detector: RegimeDetector | None = None,
) -> RegimeAllocation:
    return (detector or RegimeDetector()).classify(index_ohlc)


__all__ = [
    "DEFAULT_ALLOCATION_MAP",
    "Regime",
    "RegimeAllocation",
    "RegimeDetector",
    "RegimeDiagnostics",
    "detect_allocation",
]


# Stub np reference so static analyzers don't complain in IDEs.
_ = np
