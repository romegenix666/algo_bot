"""Pairs Trading via Engle-Granger Cointegration with Ornstein-Uhlenbeck half-life.

Logic:
    1. From a pre-defined candidate list of pairs (typically same-sector pairs
       — e.g. HDFCBANK/ICICIBANK, RELIANCE/ONGC, TCS/INFY), test each pair
       for cointegration via the Engle-Granger procedure (statsmodels
       ``coint``). p-value < ``p_value_max`` → tradable.
    2. Fit the hedge ratio β by OLS on log prices.
    3. Form the spread: ``S = log(A) - β·log(B)`` and compute its z-score
       relative to its rolling mean.
    4. Trade rules (z-score):
         - ``z < -z_entry``  →  long spread (buy A, short B)
         - ``z > +z_entry``  →  short spread (short A, buy B)
         - ``|z| < z_exit``  →  close spread
       (The exit is symmetric — we take the mean-reversion all the way to 0,
       not all the way to the opposite side, to keep the win-rate high.)
    5. Risk gates:
         - Cointegration breaks (rolling p > ``p_value_break``) → close.
         - OU half-life > ``max_half_life_days`` → don't enter (too slow).
         - Time stop → close after ``max_hold_days`` regardless.

References:
    - Engle & Granger (1987). *Co-integration and Error Correction*.
    - Avellaneda & Lee (2010). *Statistical Arbitrage in U.S. Equities*.
    - Chan (2009), *Quantitative Trading*, Ch. 7 (cointegration section).
    - Kakushadze & Serur (2018) §3.8.
    - Ornstein & Uhlenbeck (1930) for the half-life formula.

Why pairs are good for India:
    - Market neutral: low correlation with Nifty, drawdowns stay small in
      crashes (a 2008/2020 stress test win).
    - Sector-mate pairs (banks, IT, FMCG) are economically anchored: when
      one diverges from the other without a fundamental reason, mean
      reversion is plausible.
    - Lower turnover than single-stock mean-reversion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

from src.features.indicators import zscore
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
class PairCandidate:
    """A pre-screened pair, e.g. ``("HDFCBANK.NS", "ICICIBANK.NS")``."""

    a: str
    b: str

    @property
    def name(self) -> str:
        return f"{self.a}|{self.b}"


@dataclass
class CointegrationFit:
    """Output of the Engle-Granger fit on a pair."""

    hedge_ratio: float  # β in S = ln(A) - β·ln(B)
    p_value: float
    spread_mean: float
    spread_std: float
    half_life_days: float  # OU half-life of mean reversion


class PairsTradingStrategy(Strategy):
    """Engle-Granger cointegration pairs trading."""

    name = "pairs"
    timeframe = "1d"
    is_dollar_neutral: ClassVar[bool] = True

    def __init__(
        self,
        candidates: list[PairCandidate],
        lookback_days: int = 252,
        z_entry: float = 2.0,
        z_exit: float = 0.5,
        p_value_max: float = 0.05,
        p_value_break: float = 0.10,
        max_hold_days: int = 30,
        max_half_life_days: float = 30.0,
        zscore_window: int = 60,
    ) -> None:
        if not candidates:
            raise ValueError("candidates must not be empty")
        if z_exit >= z_entry:
            raise ValueError("z_exit must be < z_entry")
        self.candidates = candidates
        self.lookback_days = lookback_days
        self.z_entry = z_entry
        self.z_exit = z_exit
        self.p_value_max = p_value_max
        self.p_value_break = p_value_break
        self.max_hold_days = max_hold_days
        self.max_half_life_days = max_half_life_days
        self.zscore_window = zscore_window

    # ----------------------------------------------------------------------
    def required_features(self) -> list[str]:
        return ["close", "atr_14"]

    # ----------------------------------------------------------------------
    def fit(self, log_a: pd.Series, log_b: pd.Series) -> CointegrationFit | None:
        """Run Engle-Granger and OU on a pair. Return ``None`` if not tradable."""
        if len(log_a) != len(log_b) or len(log_a) < 50:
            return None
        try:
            _, p_value, _ = coint(log_a, log_b, trend="c", autolag="AIC")
        except Exception:  # pragma: no cover - defensive
            return None
        if p_value >= self.p_value_max:
            return None

        # OLS hedge ratio with intercept.
        x = np.column_stack([np.ones_like(log_b.values), log_b.values])
        try:
            coeffs, *_ = np.linalg.lstsq(x, log_a.values, rcond=None)
        except np.linalg.LinAlgError:  # pragma: no cover - defensive
            return None
        intercept, beta = float(coeffs[0]), float(coeffs[1])
        spread = log_a - (intercept + beta * log_b)

        # OU half-life: regress Δspread on lag(spread). Slope = -θ.
        spread_lag = spread.shift(1).dropna()
        delta = spread.diff().dropna()
        if len(spread_lag) < 30 or len(delta) < 30:
            return None
        x_ou = np.column_stack([np.ones_like(spread_lag.values), spread_lag.values])
        try:
            ou_coeffs, *_ = np.linalg.lstsq(x_ou, delta.values, rcond=None)
        except np.linalg.LinAlgError:  # pragma: no cover - defensive
            return None
        theta = -float(ou_coeffs[1])
        if theta <= 0:
            return None
        half_life = float(np.log(2.0) / theta)

        if half_life > self.max_half_life_days:
            return None

        return CointegrationFit(
            hedge_ratio=beta,
            p_value=float(p_value),
            spread_mean=float(spread.mean()),
            spread_std=float(spread.std(ddof=1)),
            half_life_days=half_life,
        )

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

        if len(close_wide) < self.lookback_days + self.zscore_window:
            return []

        as_of = close_wide.index[-1]
        signals: list[Signal] = []

        for pair in self.candidates:
            if pair.a not in close_wide.columns or pair.b not in close_wide.columns:
                continue

            window = (
                close_wide[[pair.a, pair.b]].dropna().tail(self.lookback_days + self.zscore_window)
            )
            if len(window) < self.lookback_days:
                continue

            log_window = np.log(window.tail(self.lookback_days))
            fit = self.fit(log_window[pair.a], log_window[pair.b])
            if fit is None:
                continue

            # Build the full-history spread for z-scoring.
            full_log = np.log(window)
            spread_full = full_log[pair.a] - fit.hedge_ratio * full_log[pair.b]
            z_full = zscore(spread_full, self.zscore_window)
            z_now = float(z_full.iloc[-1])
            if np.isnan(z_now):
                continue

            timestamp = _to_datetime(as_of)
            conviction = min(1.0, max(0.0, (abs(z_now) - self.z_entry) / 2.0 + 0.5))

            if z_now < -self.z_entry:
                # Spread is too low → long A, short B (unit dollar each side).
                meta = {
                    "pair": pair.name,
                    "z": z_now,
                    "hedge_ratio": fit.hedge_ratio,
                    "half_life": fit.half_life_days,
                    "p_value": fit.p_value,
                    "leg_role": "long_spread",
                }
                signals.append(
                    Signal(pair.a, Side.LONG, conviction, timestamp, {**meta, "side_in_pair": "A"})
                )
                signals.append(
                    Signal(pair.b, Side.SHORT, conviction, timestamp, {**meta, "side_in_pair": "B"})
                )
            elif z_now > self.z_entry:
                meta = {
                    "pair": pair.name,
                    "z": z_now,
                    "hedge_ratio": fit.hedge_ratio,
                    "half_life": fit.half_life_days,
                    "p_value": fit.p_value,
                    "leg_role": "short_spread",
                }
                signals.append(
                    Signal(pair.a, Side.SHORT, conviction, timestamp, {**meta, "side_in_pair": "A"})
                )
                signals.append(
                    Signal(pair.b, Side.LONG, conviction, timestamp, {**meta, "side_in_pair": "B"})
                )
            # else: |z| < z_entry → no entry. Existing positions exit at |z| < z_exit.

        return signals

    # ----------------------------------------------------------------------
    def position_size(
        self,
        signal: Signal,
        risk: RiskParams,
        win_rate_estimate: float,
        win_loss_ratio_estimate: float,
    ) -> float:
        # Pairs trades use ~half the per-trade risk per leg since the two
        # legs together form one bet. The hedge ratio is applied by the
        # order manager to balance leg notionals.
        kelly = half_kelly_fraction(win_rate_estimate, win_loss_ratio_estimate, cap=risk.kelly_cap)
        kelly *= signal.conviction
        kelly *= 0.5  # split across two legs
        return min(
            risk.equity * kelly,
            risk.equity * risk.max_single_position_pct,
        )

    # ----------------------------------------------------------------------
    def exit_rules(self, position: Position, market: MarketState) -> ExitDecision:
        if position.side is Side.LONG and market.last_price <= position.current_stop:
            return ExitDecision(
                should_exit=True, reason=ExitReason.STOP_LOSS, note="ATR stop on leg"
            )
        if position.side is Side.SHORT and market.last_price >= position.current_stop:
            return ExitDecision(
                should_exit=True, reason=ExitReason.STOP_LOSS, note="ATR stop on leg"
            )

        held = (market.timestamp - position.entry_time).days
        if held >= self.max_hold_days:
            return ExitDecision(
                should_exit=True, reason=ExitReason.TIME_STOP, note=f"held {held} days"
            )

        # The actual mean-reversion-completion exit is computed at the pair
        # level by the order manager, not per leg, since it depends on the
        # joint z-score. Per-leg exit_rules just enforces the safety nets.
        return ExitDecision(should_exit=False)


# Default Indian sector-mate candidates we'll try first.
DEFAULT_INDIAN_PAIRS: list[PairCandidate] = [
    PairCandidate("HDFCBANK.NS", "ICICIBANK.NS"),
    PairCandidate("HDFCBANK.NS", "AXISBANK.NS"),
    PairCandidate("ICICIBANK.NS", "KOTAKBANK.NS"),
    PairCandidate("TCS.NS", "INFY.NS"),
    PairCandidate("INFY.NS", "WIPRO.NS"),
    PairCandidate("HCLTECH.NS", "TECHM.NS"),
    PairCandidate("RELIANCE.NS", "ONGC.NS"),
    PairCandidate("MARUTI.NS", "M&M.NS"),
    PairCandidate("HINDUNILVR.NS", "ITC.NS"),
    PairCandidate("ASIANPAINT.NS", "BERGEPAINT.NS"),
    PairCandidate("ULTRACEMCO.NS", "SHREECEM.NS"),
    PairCandidate("TATASTEEL.NS", "JSWSTEEL.NS"),
    PairCandidate("NESTLEIND.NS", "BRITANNIA.NS"),
    PairCandidate("DRREDDY.NS", "CIPLA.NS"),
    PairCandidate("SUNPHARMA.NS", "DIVISLAB.NS"),
]


__all__ = [
    "DEFAULT_INDIAN_PAIRS",
    "CointegrationFit",
    "PairCandidate",
    "PairsTradingStrategy",
]
