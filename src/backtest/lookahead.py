"""Look-ahead-bias auditor — Chan §3 truncate-and-rerun.

How it works (literally Chan's recipe):

    1. Run the full backtest on data through ``end``. Save the trade log.
    2. Run the same backtest again on data through ``end - n_days``.
    3. Compare the two trade logs over the OVERLAPPING period.

If the strategy is look-ahead-clean, the trades on the overlap should be
*identical* between the two runs — because by definition the strategy
shouldn't have used data from after ``end - n_days`` for trades before that
date. If trades differ, there's a leak.

Why this is the right test:
    Static code review can miss subtle look-ahead — e.g. fitting a
    regression on the *whole* training set and using its coefficients
    on every bar of the backtest. The truncate-and-rerun test catches
    these because the regression coefficients change when the data
    window shrinks, and the difference flushes through to the trade log.

Why ``strategy_factory`` takes the prices:
    A strategy that fits a model in ``__init__`` (or anywhere outside
    ``generate_signals``) using the data it received at construction-time
    is the *exact* class of bug this auditor catches. Passing ``prices``
    into the factory lets such a strategy build its illegal model from
    different data on the full vs. truncated runs, and so trades diverge.
    A strategy that respects the engine's per-bar slicing produces
    identical trades regardless of how the prices argument was sliced
    on the way in.

Returns a ``LookaheadReport`` with:
    - matched, mismatched and missing trades
    - a verdict ("clean" / "leaks")
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from src.backtest.engine import Backtester, BacktestResult
from src.strategies.base import Strategy
from src.utils.logging import logger

StrategyFactory = Callable[[pd.DataFrame], Strategy]


@dataclass(frozen=True)
class LookaheadReport:
    full_trades: int
    truncated_trades: int
    overlapping_window_start: pd.Timestamp
    overlapping_window_end: pd.Timestamp
    matched: int
    mismatched: int
    only_in_full: int
    only_in_truncated: int
    verdict: str  # "clean" | "leaks"

    def pretty(self) -> str:
        lines = [
            f"Verdict           : {self.verdict.upper()}",
            f"Window            : {self.overlapping_window_start.date()} → {self.overlapping_window_end.date()}",
            f"Full run trades   : {self.full_trades}",
            f"Truncated trades  : {self.truncated_trades}",
            f"Matched (overlap) : {self.matched}",
            f"Mismatched        : {self.mismatched}",
            f"Only in full      : {self.only_in_full}",
            f"Only in truncated : {self.only_in_truncated}",
        ]
        return "\n".join(lines)


def audit_strategy(
    strategy_factory: StrategyFactory,
    backtester: Backtester,
    prices: pd.DataFrame,
    truncate_bars: int = 60,
    index_ohlc: pd.DataFrame | None = None,
    features: pd.DataFrame | None = None,
    sentiment: pd.DataFrame | None = None,
    weight_atol: float = 1e-6,
) -> LookaheadReport:
    """Run the truncate-and-rerun test on a strategy.

    Args:
        strategy_factory: ``(prices) -> Strategy``. The factory receives
            the prices panel that the engine will run on, so any
            data-dependent setup the strategy does (which is itself a
            potential leak) sees DIFFERENT data in the full vs. truncated
            runs.
        backtester: pre-built ``Backtester`` (cost model + risk + freq).
        prices: full-history MultiIndex (date, ticker) panel.
        truncate_bars: how many bars to chop off the end. 60 is standard.
        weight_atol: per-trade weight-delta tolerance below which we
            consider trades "identical" (avoids float-precision noise).
    """
    if not isinstance(prices.index, pd.MultiIndex):
        raise ValueError("prices must have MultiIndex (date, ticker)")
    all_dates = prices.index.get_level_values("date").unique().sort_values()
    if len(all_dates) < truncate_bars + backtester.minimum_history_bars + 5:
        raise ValueError(
            f"Need at least {truncate_bars + backtester.minimum_history_bars + 5} bars; "
            f"got {len(all_dates)}"
        )

    truncate_end = all_dates[-truncate_bars - 1]

    # ---- Full run ----
    full = backtester.run(
        strategy=strategy_factory(prices),
        prices=prices,
        index_ohlc=index_ohlc,
        features=features,
        sentiment=sentiment,
    )

    # ---- Truncated run ----
    truncated_prices = prices.loc[prices.index.get_level_values("date") <= truncate_end]
    truncated_index = (
        index_ohlc.loc[index_ohlc.index <= truncate_end] if index_ohlc is not None else None
    )
    truncated = backtester.run(
        strategy=strategy_factory(truncated_prices),
        prices=truncated_prices,
        index_ohlc=truncated_index,
        features=features,
        sentiment=sentiment,
    )

    overlap_start = truncated.equity.index.min()
    overlap_end = truncated.equity.index.max()
    return _compare_runs(full, truncated, overlap_start, overlap_end, weight_atol)


def _compare_runs(
    full: BacktestResult,
    truncated: BacktestResult,
    overlap_start: pd.Timestamp,
    overlap_end: pd.Timestamp,
    weight_atol: float,
) -> LookaheadReport:
    """Compare trades over the overlapping window. Equal trades = clean."""
    full_t = [t for t in full.trades if overlap_start <= t.rebalance_date <= overlap_end]
    trunc_t = [t for t in truncated.trades if overlap_start <= t.rebalance_date <= overlap_end]

    full_keyed = {(t.rebalance_date, t.ticker): t for t in full_t}
    trunc_keyed = {(t.rebalance_date, t.ticker): t for t in trunc_t}

    matched = mismatched = 0
    only_in_full = 0
    only_in_truncated = 0

    for key, t_full in full_keyed.items():
        t_trunc = trunc_keyed.get(key)
        if t_trunc is None:
            only_in_full += 1
            continue
        if abs(t_full.weight_delta - t_trunc.weight_delta) <= weight_atol:
            matched += 1
        else:
            mismatched += 1
            logger.debug(
                "Trade mismatch @ {} {}: full Δw={:.6f}, trunc Δw={:.6f}",
                t_full.rebalance_date,
                t_full.ticker,
                t_full.weight_delta,
                t_trunc.weight_delta,
            )

    for key in trunc_keyed:
        if key not in full_keyed:
            only_in_truncated += 1

    verdict = (
        "clean" if (mismatched == 0 and only_in_full == 0 and only_in_truncated == 0) else "leaks"
    )
    return LookaheadReport(
        full_trades=len(full.trades),
        truncated_trades=len(truncated.trades),
        overlapping_window_start=overlap_start,
        overlapping_window_end=overlap_end,
        matched=matched,
        mismatched=mismatched,
        only_in_full=only_in_full,
        only_in_truncated=only_in_truncated,
        verdict=verdict,
    )


__all__ = ["LookaheadReport", "audit_strategy"]


# Keep referenced import alive so importers don't get lint hits.
_ = Strategy
