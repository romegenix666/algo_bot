"""Strategy Selector — orchestrates regime detection + multi-strategy run.

The selector is the glue between:
    - the registry (which builds strategy instances)
    - the regime detector (which assigns weights per regime)
    - the per-strategy ``generate_signals`` calls

It produces a single, unified list of *weighted* signals that the order
manager can size and route. This is how "switch strategies according to
the situation" is realised.

Pipeline::

    market_state_df          ← Nifty 50 OHLC (last N days)
       │
       ▼
    RegimeDetector.classify  → RegimeAllocation (weights per strategy)
       │
       ▼
    For each strategy with weight > 0:
        signals = strategy.generate_signals(prices, features, sentiment)
        each signal's conviction *= regime_weight
       │
       ▼
    Concatenate, deduplicate (per ticker, sum convictions), normalise.
       │
       ▼
    Top-N picker → final signals
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import pandas as pd

from src.strategies.base import Side, Signal, Strategy
from src.strategies.regime import RegimeAllocation, RegimeDetector


@dataclass
class WeightedSignal:
    """Signal from one strategy, with the regime weight already applied."""

    signal: Signal
    strategy_name: str
    regime_weight: float

    @property
    def weighted_conviction(self) -> float:
        return float(self.signal.conviction * self.regime_weight)


@dataclass
class SelectionResult:
    """The output of ``StrategySelector.select``."""

    regime_allocation: RegimeAllocation
    raw_signals: list[WeightedSignal]
    final_signals: list[Signal]
    debug: dict[str, list[str]] = field(default_factory=dict)


class StrategySelector:
    """Run all active strategies and merge their signals by regime weights."""

    def __init__(
        self,
        strategies: list[Strategy],
        regime_detector: RegimeDetector | None = None,
        top_n_final: int = 5,
        min_weighted_conviction: float = 0.05,
    ) -> None:
        if not strategies:
            raise ValueError("At least one strategy is required")
        self.strategies = strategies
        self.regime_detector = regime_detector or RegimeDetector()
        self.top_n_final = top_n_final
        self.min_weighted_conviction = min_weighted_conviction

    # ------------------------------------------------------------------
    def select(
        self,
        index_ohlc: pd.DataFrame,
        prices: pd.DataFrame,
        features: pd.DataFrame,
        sentiment: pd.DataFrame | None = None,
    ) -> SelectionResult:
        """Run the full pipeline and return the final picks.

        Args:
            index_ohlc: Broad-market index OHLC (Nifty 50). Used for regime
                detection ONLY.
            prices: MultiIndex (date, ticker) OHLCV for all candidate
                tickers. Strategies pivot internally as they wish.
            features: Pre-computed features per ticker (ATR, fundamentals).
            sentiment: Optional ``[ticker, score]`` per-ticker sentiment.
        """
        regime_allocation = self.regime_detector.classify(index_ohlc)
        weights = regime_allocation.weights

        raw: list[WeightedSignal] = []
        debug: dict[str, list[str]] = {}

        for strat in self.strategies:
            w = float(weights.get(strat.name, 0.0))
            if w <= 0:
                debug.setdefault("skipped_zero_weight", []).append(strat.name)
                continue
            try:
                signals = strat.generate_signals(prices, features, sentiment)
            except Exception as exc:  # pragma: no cover - defensive
                debug.setdefault("errored", []).append(f"{strat.name}: {exc}")
                continue
            for sig in signals:
                raw.append(WeightedSignal(signal=sig, strategy_name=strat.name, regime_weight=w))

        # Aggregate per (ticker, side) — sum weighted convictions.
        aggregated: dict[tuple[str, Side], list[WeightedSignal]] = {}
        for ws in raw:
            key = (ws.signal.ticker, ws.signal.side)
            aggregated.setdefault(key, []).append(ws)

        final_signals: list[Signal] = []
        for (ticker, side), group in aggregated.items():
            total_conv = sum(ws.weighted_conviction for ws in group)
            if total_conv < self.min_weighted_conviction:
                continue
            # Use the latest timestamp; merge metadata across contributors.
            latest = max(group, key=lambda ws: ws.signal.timestamp)
            metadata = {
                "contributors": {
                    ws.strategy_name: round(ws.weighted_conviction, 4) for ws in group
                },
                "regime": str(regime_allocation.regime),
            }
            final_signals.append(
                Signal(
                    ticker=ticker,
                    side=side,
                    conviction=float(min(1.0, total_conv)),
                    timestamp=latest.signal.timestamp,
                    metadata=metadata,
                )
            )

        # Top-N by conviction (the "5 stocks that will make money" picker).
        final_signals.sort(key=lambda s: s.conviction, reverse=True)
        final_signals = final_signals[: self.top_n_final]

        return SelectionResult(
            regime_allocation=regime_allocation,
            raw_signals=raw,
            final_signals=final_signals,
            debug=debug,
        )

    # ------------------------------------------------------------------
    def names(self) -> list[str]:
        return [s.name for s in self.strategies]


def merge_signals(weighted: Iterable[WeightedSignal], top_n: int) -> list[Signal]:
    """Standalone helper used by tests + scripts."""
    aggregated: dict[tuple[str, Side], float] = {}
    for ws in weighted:
        key = (ws.signal.ticker, ws.signal.side)
        aggregated[key] = aggregated.get(key, 0.0) + ws.weighted_conviction
    out = [
        Signal(
            ticker=t,
            side=s,
            conviction=min(1.0, v),
            timestamp=max(ws.signal.timestamp for ws in weighted if ws.signal.ticker == t),
        )
        for (t, s), v in aggregated.items()
    ]
    out.sort(key=lambda x: x.conviction, reverse=True)
    return out[:top_n]


__all__ = ["SelectionResult", "StrategySelector", "WeightedSignal", "merge_signals"]
