"""Strategy Registry & Selector — the Factory Pattern.

This is the *only* place the rest of the codebase asks for strategies.
Anything else just says "give me strategies named X, Y, Z" and gets back
fully-configured ``Strategy`` instances.

Usage::

    from src.strategies.registry import build_strategies, ACTIVE_STRATEGIES

    strategies = build_strategies(["momentum", "mean_reversion", "pairs"])

    # Or load straight from settings.yaml:
    strategies = build_active_strategies()

When you want a new strategy:
    1. Implement it as a subclass of ``Strategy`` in its own file.
    2. Add a builder entry in ``_BUILDERS`` below.
    3. Reference its name in ``config/default.yaml`` under
       ``strategies.active``.

That's it — the backtester, regime detector, and order manager all keep
working unchanged.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from src.strategies.base import Strategy
from src.strategies.breakout import DonchianBreakoutStrategy
from src.strategies.dual_momentum import DualMomentumStrategy
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.momentum import MomentumStrategy
from src.strategies.multi_factor import FactorWeights, MultiFactorStrategy
from src.strategies.pairs import DEFAULT_INDIAN_PAIRS, PairsTradingStrategy
from src.strategies.sector_rotation import DEFAULT_NSE_SECTORS, SectorRotationStrategy
from src.strategies.sentiment_momentum import SentimentMomentumStrategy
from src.utils.settings import settings

StrategyBuilder = Callable[[dict[str, Any]], Strategy]


def _build_momentum(cfg: dict[str, Any]) -> Strategy:
    return MomentumStrategy(
        lookback_days=int(cfg.get("lookback_days", 252)),
        skip_days=int(cfg.get("skip_days", 21)),
        top_n=int(cfg.get("top_n", 5)),
        bottom_n=int(cfg.get("bottom_n", 0)),
    )


def _build_mean_reversion(cfg: dict[str, Any]) -> Strategy:
    return MeanReversionStrategy(
        bb_period=int(cfg.get("bb_period", 20)),
        bb_std=float(cfg.get("bb_std", 2.0)),
        rsi_period=int(cfg.get("rsi_period", 14)),
        rsi_entry=float(cfg.get("rsi_entry", 30)),
        rsi_exit=float(cfg.get("rsi_exit", 50)),
        max_hold_days=int(cfg.get("max_hold_days", 10)),
        adf_pvalue_max=float(cfg.get("adf_pvalue_max", 0.05)),
        allow_short=bool(cfg.get("allow_short", False)),
    )


def _build_pairs(cfg: dict[str, Any]) -> Strategy:
    candidates = cfg.get("candidates")
    if candidates:
        from src.strategies.pairs import PairCandidate

        candidates = [PairCandidate(a=p["a"], b=p["b"]) for p in candidates]
    else:
        candidates = DEFAULT_INDIAN_PAIRS
    return PairsTradingStrategy(
        candidates=candidates,
        lookback_days=int(cfg.get("cointegration_lookback_days", 252)),
        z_entry=float(cfg.get("z_entry", 2.0)),
        z_exit=float(cfg.get("z_exit", 0.5)),
        p_value_max=float(cfg.get("p_value_max", 0.05)),
        p_value_break=float(cfg.get("p_value_break", 0.10)),
        max_hold_days=int(cfg.get("max_hold_days", 30)),
        zscore_window=int(cfg.get("zscore_window", 60)),
    )


def _build_multi_factor(cfg: dict[str, Any]) -> Strategy:
    weights_cfg = cfg.get("factor_weights", [0.25, 0.25, 0.25, 0.25])
    if isinstance(weights_cfg, list) and len(weights_cfg) == 4:
        weights = FactorWeights(
            value=float(weights_cfg[0]),
            momentum=float(weights_cfg[1]),
            quality=float(weights_cfg[2]),
            low_vol=float(weights_cfg[3]),
        )
    elif isinstance(weights_cfg, dict):
        weights = FactorWeights(
            value=float(weights_cfg.get("value", 0.25)),
            momentum=float(weights_cfg.get("momentum", 0.25)),
            quality=float(weights_cfg.get("quality", 0.25)),
            low_vol=float(weights_cfg.get("low_vol", 0.25)),
        )
    else:
        weights = FactorWeights()
    return MultiFactorStrategy(
        weights=weights,
        top_pct=float(cfg.get("top_quintile", 0.20)),
        bottom_pct=float(cfg.get("bottom_quintile", 0.0)),
    )


def _build_breakout(cfg: dict[str, Any]) -> Strategy:
    return DonchianBreakoutStrategy(
        entry_window=int(cfg.get("entry_window", 20)),
        exit_window=int(cfg.get("exit_window", 10)),
        adx_min=float(cfg.get("adx_min", 25.0)),
        adx_window=int(cfg.get("adx_window", 14)),
        sma_filter_period=int(cfg.get("sma_filter_period", 100)),
        allow_short=bool(cfg.get("allow_short", False)),
        max_hold_days=int(cfg.get("max_hold_days", 120)),
    )


def _build_dual_momentum(cfg: dict[str, Any]) -> Strategy:
    return DualMomentumStrategy(
        market_index_ticker=str(cfg.get("market_index_ticker", "^NSEI")),
        defensive_ticker=str(cfg.get("defensive_ticker", "LIQUIDBEES.NS")),
        abs_lookback_days=int(cfg.get("abs_lookback_days", 252)),
        rel_lookback_days=int(cfg.get("rel_lookback_days", 252)),
        top_n=int(cfg.get("top_n", 5)),
        hold_days=int(cfg.get("hold_days", 21)),
    )


def _build_sector_rotation(cfg: dict[str, Any]) -> Strategy:
    return SectorRotationStrategy(
        sector_tickers=list(cfg.get("sector_tickers", DEFAULT_NSE_SECTORS)),
        defensive_ticker=str(cfg.get("defensive_ticker", "LIQUIDBEES.NS")),
        rel_lookback_days=int(cfg.get("rel_lookback_days", 126)),
        abs_filter_period=int(cfg.get("abs_filter_period", 200)),
        top_n=int(cfg.get("top_n", 3)),
        vol_target=float(cfg.get("vol_target", 0.15)),
        vol_window=int(cfg.get("vol_window", 60)),
        hold_days=int(cfg.get("hold_days", 21)),
    )


def _build_sentiment_momentum(cfg: dict[str, Any]) -> Strategy:
    base_cfg = cfg.get("base", {})
    base = MomentumStrategy(
        lookback_days=int(base_cfg.get("lookback_days", 252)),
        skip_days=int(base_cfg.get("skip_days", 21)),
        top_n=int(base_cfg.get("top_n", 5)),
    )
    return SentimentMomentumStrategy(
        base=base,
        sentiment_long_min=float(cfg.get("sentiment_filter_long", 0.0)),
        sentiment_short_max=float(cfg.get("sentiment_filter_short", 0.0)),
        sentiment_boost_threshold=float(cfg.get("sentiment_boost_threshold", 0.4)),
        sentiment_boost=float(cfg.get("sentiment_boost", 0.15)),
    )


_BUILDERS: dict[str, StrategyBuilder] = {
    "momentum": _build_momentum,
    "mean_reversion": _build_mean_reversion,
    "pairs": _build_pairs,
    "multi_factor": _build_multi_factor,
    "breakout": _build_breakout,
    "dual_momentum": _build_dual_momentum,
    "sector_rotation": _build_sector_rotation,
    "sentiment_momentum": _build_sentiment_momentum,
}


def available_strategies() -> list[str]:
    """All strategy names known to the registry."""
    return sorted(_BUILDERS)


def build_strategy(name: str, config: dict[str, Any] | None = None) -> Strategy:
    """Build one strategy by name. ``config`` overrides defaults from yaml."""
    if name not in _BUILDERS:
        raise KeyError(f"Unknown strategy '{name}'. Available: {available_strategies()}")
    yaml_cfg = settings.get("strategies", name, default={}) or {}
    merged: dict[str, Any] = {**yaml_cfg, **(config or {})}
    return _BUILDERS[name](merged)


def build_strategies(
    names: Iterable[str], configs: dict[str, dict[str, Any]] | None = None
) -> list[Strategy]:
    configs = configs or {}
    return [build_strategy(n, configs.get(n)) for n in names]


def build_active_strategies() -> list[Strategy]:
    """Build whatever ``strategies.active`` lists in config/default.yaml."""
    active = settings.get("strategies", "active", default=[])
    if not active:
        return []
    return build_strategies(active)


__all__ = [
    "available_strategies",
    "build_active_strategies",
    "build_strategies",
    "build_strategy",
]
