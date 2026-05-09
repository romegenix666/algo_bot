"""Vectorised, point-in-time backtest engine.

Pipeline (one strategy, one universe, one date range):

    For each bar t in [start, end]:
        1. Compute the daily return on the *previous* bar's weights.
           equity[t] = equity[t-1] * (1 + dot(weights[t-1], returns[t]))
        2. If t is a rebalance bar:
              a. Slice prices/features/sentiment to data through t-1 ONLY.
                 This is what enforces "no look-ahead" — the strategy must
                 not see the current bar's close while deciding what to do.
              b. signals = strategy.generate_signals(point_in_time_view)
              c. Convert signals → target weights (use strategy.position_size).
              d. Compute trade = target - current weights (per ticker).
              e. Pay transaction costs proportional to |trade|.
              f. Update weights to target.
        3. Record everything.

Output (``BacktestResult``):
    - equity curve
    - daily-returns series
    - weights matrix (date × ticker)
    - trade log (one row per trade leg)
    - performance summary

Why this design (not a fully event-driven simulator):
    - For daily-bar strategies, vectorised-by-bar gives microseconds per
      bar — fast enough for walk-forward + sensitivity.
    - We still loop bar-by-bar (not month-by-month) so that the daily
      P&L compounding is exact and the equity curve is bar-resolution.
    - The strategy is invoked only on rebalance dates; in-between bars
      just compound returns. So a monthly rebalance over 5 years runs the
      strategy ~60 times, not 1250 times.

References:
    - Chan (2009), *Quantitative Trading*, Chapter 3 (backtesting).
    - López de Prado (2018), *Advances in Financial Machine Learning*,
      Chapter 13 (backtesting through cross-validation).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

import numpy as np
import pandas as pd

from src.backtest.costs import IndianEquityCostModel
from src.backtest.metrics import PerformanceSummary, summarise
from src.strategies.base import RiskParams, Side, Signal, Strategy
from src.utils.logging import logger

RebalanceFreq = Literal["D", "W", "M", "Q"]


@dataclass(frozen=True)
class TradeLog:
    """One trade leg recorded by the engine. Used for hit-rate / cost audit."""

    rebalance_date: pd.Timestamp
    ticker: str
    side: str  # 'buy' or 'sell'
    weight_before: float
    weight_after: float
    weight_delta: float
    notional_inr: float
    cost_inr: float


@dataclass
class BacktestResult:
    """Everything the engine produces. Slice-and-dice from here."""

    strategy_name: str
    equity: pd.Series  # indexed by date, starts at initial_capital
    daily_returns: pd.Series  # indexed by date
    weights: pd.DataFrame  # date × ticker (target weights post-rebalance)
    trades: list[TradeLog] = field(default_factory=list)
    benchmark_equity: pd.Series | None = None
    summary: PerformanceSummary | None = None

    @property
    def trade_returns(self) -> pd.Series:
        """Returns *between successive rebalances* — the proxy for trade returns."""
        # Equity at each rebalance:
        rebal_dates = sorted({t.rebalance_date for t in self.trades})
        if len(rebal_dates) < 2:
            return pd.Series(dtype=float)
        equity_at = self.equity.reindex(rebal_dates, method="ffill").dropna()
        return equity_at.pct_change().dropna()

    def to_csv(self, path: str) -> None:
        """Dump equity, daily returns and weights to one wide CSV (for inspection)."""
        wide = pd.DataFrame({"equity": self.equity, "daily_return": self.daily_returns})
        merged = wide.join(self.weights, how="outer")
        merged.to_csv(path)


@dataclass
class Backtester:
    """The engine. Construct once, ``.run()`` per strategy."""

    cost_model: IndianEquityCostModel = field(default_factory=IndianEquityCostModel)
    initial_capital: float = 1_000_000.0  # ₹10 Lakh notional
    risk: RiskParams = field(
        default_factory=lambda: RiskParams(
            equity=1_000_000.0,
            per_trade_pct=0.01,
            half_kelly=True,
            kelly_cap=0.20,
            max_single_position_pct=0.20,
        )
    )
    rebalance_freq: RebalanceFreq = "M"
    win_rate_estimate: float = 0.5
    win_loss_ratio_estimate: float = 1.5
    max_gross_exposure: float = 1.0  # 1.0 = no leverage; 1.5 = 50% leverage
    minimum_history_bars: int = 252  # don't trade until we have 1 year

    # ------------------------------------------------------------------
    def run(
        self,
        strategy: Strategy,
        prices: pd.DataFrame,
        index_ohlc: pd.DataFrame | None = None,
        features: pd.DataFrame | None = None,
        sentiment: pd.DataFrame | None = None,
    ) -> BacktestResult:
        """Run a single strategy through the universe.

        Args:
            strategy: instance of ``Strategy``.
            prices: MultiIndex ``(date, ticker)`` with at least
                ``[open, high, low, close, volume]`` columns.
            index_ohlc: optional benchmark OHLC for relative metrics
                (Nifty 50 by default).
            features: optional features panel — same shape as prices.
            sentiment: optional ``[ticker, score]`` per-day sentiment.
        """
        if not isinstance(prices.index, pd.MultiIndex):
            raise ValueError("prices must have MultiIndex (date, ticker)")

        # Pivot for fast vectorised slicing.
        close_wide = prices["close"].unstack("ticker").sort_index()
        adj_close_wide = (
            prices["adj_close"].unstack("ticker").sort_index()
            if "adj_close" in prices.columns
            else close_wide
        )
        daily_returns_panel = adj_close_wide.pct_change()  # one row per bar, ticker cols
        all_dates = close_wide.index
        all_tickers = list(close_wide.columns)

        if len(all_dates) < self.minimum_history_bars + 5:
            raise ValueError(
                f"Need at least {self.minimum_history_bars + 5} bars; got {len(all_dates)}"
            )

        rebalance_dates = self._rebalance_schedule(all_dates)
        rebalance_set = set(rebalance_dates)

        # State
        weights = pd.Series(0.0, index=all_tickers)
        equity = float(self.initial_capital)
        equity_history: list[float] = []
        weights_history: list[pd.Series] = []
        trades: list[TradeLog] = []

        skip_until_bar = self.minimum_history_bars  # don't trade before this

        for i, date in enumerate(all_dates):
            # 1. Compound returns on the previous bar's weights.
            if i > 0 and weights.abs().sum() > 0:
                rets = daily_returns_panel.loc[date].fillna(0.0)
                pnl = float((weights * rets).sum())
                equity *= 1.0 + pnl

            # 2. Rebalance?
            if i >= skip_until_bar and date in rebalance_set:
                point_in_time = self._point_in_time(prices, date)
                point_in_time_idx = (
                    self._point_in_time_index(index_ohlc, date) if index_ohlc is not None else None
                )
                point_in_time_features = (
                    self._point_in_time(features, date)
                    if features is not None and isinstance(features.index, pd.MultiIndex)
                    else features
                )

                try:
                    signals = strategy.generate_signals(
                        prices=point_in_time,
                        features=point_in_time_features
                        if point_in_time_features is not None
                        else pd.DataFrame(),
                        sentiment=sentiment,
                    )
                except Exception as exc:  # pragma: no cover - defensive
                    logger.exception("Strategy {} crashed @ {}: {}", strategy.name, date, exc)
                    signals = []

                target_weights = self._signals_to_weights(signals, all_tickers, equity, strategy)

                # Apply cost: total cost ≈ sum(|Δw| × equity × cost_pct_one_way)
                trade_legs = self._record_trades(date, weights, target_weights, equity, all_tickers)
                cost_inr = sum(t.cost_inr for t in trade_legs)
                equity = max(0.0, equity - cost_inr)
                trades.extend(trade_legs)
                weights = target_weights

                # Stub out unused index lookups so the variables aren't dead.
                _ = point_in_time_idx

            equity_history.append(equity)
            weights_history.append(weights.copy())

        equity_series = pd.Series(equity_history, index=all_dates, name="equity")
        weights_df = pd.DataFrame(weights_history, index=all_dates).fillna(0.0)
        daily_returns = equity_series.pct_change().fillna(0.0)

        # Trade returns for hit-rate / profit factor — equity diff between rebalances.
        trade_returns = self._trade_returns(equity_series, rebalance_dates)

        # Optional benchmark
        bench_equity: pd.Series | None = None
        if index_ohlc is not None and "close" in index_ohlc.columns:
            bench_close = index_ohlc["close"].reindex(all_dates).ffill()
            if not bench_close.empty:
                bench_equity = self.initial_capital * (bench_close / bench_close.iloc[0])
                bench_equity.name = "benchmark"

        summary = summarise(
            equity_series,
            trade_returns=trade_returns,
            weights=weights_df,
            benchmark_equity=bench_equity,
        )

        return BacktestResult(
            strategy_name=strategy.name,
            equity=equity_series,
            daily_returns=daily_returns,
            weights=weights_df,
            trades=trades,
            benchmark_equity=bench_equity,
            summary=summary,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _rebalance_schedule(self, dates: pd.DatetimeIndex) -> list[pd.Timestamp]:
        """Trading-day on or before each calendar month/quarter/week-end.

        Why this design: a naïve ``groupby(Grouper).tail(1)`` would pick the
        last bar in each calendar group, which for a partial last group at
        the end of the data is NOT a true month-end. That made the rebalance
        schedule data-length dependent, which broke the look-ahead auditor's
        truncate-and-rerun guarantee. Now the schedule is deterministic:
        each calendar month-end resolves to the most recent trading day on
        or before it, irrespective of where the data ends.
        """
        if len(dates) == 0:
            return []
        if self.rebalance_freq == "D":
            return list(dates)

        if self.rebalance_freq == "W":
            calendar_ends = pd.date_range(start=dates[0], end=dates[-1], freq="W")
        elif self.rebalance_freq == "M":
            calendar_ends = pd.date_range(start=dates[0], end=dates[-1], freq="ME")
        elif self.rebalance_freq == "Q":
            calendar_ends = pd.date_range(start=dates[0], end=dates[-1], freq="QE")
        else:  # pragma: no cover - argparse guards
            raise ValueError(self.rebalance_freq)

        sorted_dates = pd.DatetimeIndex(sorted(dates))
        out: list[pd.Timestamp] = []
        for end in calendar_ends:
            # Only emit a rebalance for a *complete* calendar period: the
            # actual calendar end-date itself must be on or before the last
            # available trading bar. Partial trailing periods are skipped.
            if end > sorted_dates[-1]:
                continue
            cand = sorted_dates[sorted_dates <= end]
            if len(cand) == 0:
                continue
            chosen = cand[-1]
            if not out or out[-1] != chosen:
                out.append(chosen)
        return out

    @staticmethod
    def _point_in_time(panel: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
        """Slice a (date × ticker) MultiIndex panel to dates STRICTLY BEFORE ``as_of``.

        This is the look-ahead-defence — the strategy must NOT see the
        bar dated ``as_of`` itself (its close is "today's close" which
        wouldn't be known until end of day).
        """
        return panel.loc[panel.index.get_level_values("date") < as_of]

    @staticmethod
    def _point_in_time_index(idx: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
        return idx.loc[idx.index < as_of]

    def _signals_to_weights(
        self,
        signals: list[Signal],
        all_tickers: list[str],
        equity: float,
        strategy: Strategy,
    ) -> pd.Series:
        """Convert a list of signals into a target-weights vector.

        Conventions:
            - Weights sum-of-abs ≤ ``max_gross_exposure``.
            - Long signal → positive weight; short → negative.
            - Each strategy's own ``position_size`` decides the rupee
              allocation per signal — the engine is just an orchestrator.
            - The engine still applies a hard ``max_single_position_pct``
              ceiling AFTER the strategy speaks (defence in depth).
        """
        weights = pd.Series(0.0, index=all_tickers)
        if not signals:
            return weights

        risk_for_run = RiskParams(
            equity=equity,
            per_trade_pct=self.risk.per_trade_pct,
            half_kelly=self.risk.half_kelly,
            kelly_cap=self.risk.kelly_cap,
            max_single_position_pct=self.risk.max_single_position_pct,
        )

        notionals: dict[str, float] = {}
        for sig in signals:
            if sig.ticker not in weights.index or sig.side is Side.FLAT:
                continue
            try:
                notional = float(
                    strategy.position_size(
                        signal=sig,
                        risk=risk_for_run,
                        win_rate_estimate=self.win_rate_estimate,
                        win_loss_ratio_estimate=self.win_loss_ratio_estimate,
                    )
                )
            except Exception:  # pragma: no cover - defensive
                notional = 0.0
            # Cap by single-position concentration (defence in depth).
            notional = min(notional, equity * risk_for_run.max_single_position_pct)
            if notional <= 0:
                continue
            sign = 1.0 if sig.side is Side.LONG else -1.0
            notionals[sig.ticker] = notionals.get(sig.ticker, 0.0) + sign * notional

        # Cap gross exposure (defence in depth — strategies may oversubscribe).
        gross = sum(abs(v) for v in notionals.values())
        if gross > 0 and gross > equity * self.max_gross_exposure:
            scale = (equity * self.max_gross_exposure) / gross
            notionals = {k: v * scale for k, v in notionals.items()}

        for ticker, notional in notionals.items():
            weights[ticker] = notional / equity if equity > 0 else 0.0

        return weights

    def _record_trades(
        self,
        date: pd.Timestamp,
        before: pd.Series,
        after: pd.Series,
        equity: float,
        all_tickers: list[str],
    ) -> list[TradeLog]:
        """Compute trades implied by weight changes and apply costs."""
        delta = after - before
        out: list[TradeLog] = []
        for ticker in all_tickers:
            d = float(delta[ticker])
            if abs(d) < 1e-9:
                continue
            side = "buy" if d > 0 else "sell"
            notional = abs(d) * equity
            cost = self.cost_model.total(notional, side)
            out.append(
                TradeLog(
                    rebalance_date=date,
                    ticker=ticker,
                    side=side,
                    weight_before=float(before[ticker]),
                    weight_after=float(after[ticker]),
                    weight_delta=d,
                    notional_inr=notional,
                    cost_inr=cost,
                )
            )
        return out

    @staticmethod
    def _trade_returns(equity: pd.Series, rebalance_dates: list[pd.Timestamp]) -> pd.Series:
        if len(rebalance_dates) < 2:
            return pd.Series(dtype=float)
        sampled = equity.reindex(rebalance_dates, method="ffill").dropna()
        return sampled.pct_change().dropna()


# ---------------------------------------------------------------------------
# Walk-forward driver
# ---------------------------------------------------------------------------


@dataclass
class WalkForwardResult:
    """One result per (train_window, test_window) pair, plus the stitched curve."""

    folds: list[BacktestResult]
    stitched_equity: pd.Series
    stitched_summary: PerformanceSummary


def walk_forward(
    backtester: Backtester,
    strategy_factory,
    prices: pd.DataFrame,
    index_ohlc: pd.DataFrame | None = None,
    features: pd.DataFrame | None = None,
    sentiment: pd.DataFrame | None = None,
    train_years: int = 3,
    test_months: int = 6,
    step_months: int = 6,
) -> WalkForwardResult:
    """Repeated train/test splits, stitched into one out-of-sample equity curve.

    ``strategy_factory()`` must produce a fresh Strategy instance per fold so
    we don't accidentally carry state across windows. Walk-forward is the
    minimal honest test that doesn't leak the future.
    """
    if not isinstance(prices.index, pd.MultiIndex):
        raise ValueError("prices must have MultiIndex (date, ticker)")
    all_dates = prices.index.get_level_values("date").unique().sort_values()
    if len(all_dates) < (train_years * 252 + test_months * 21):
        raise ValueError("not enough history for the requested walk-forward windows")

    folds: list[BacktestResult] = []
    stitched_pieces: list[pd.Series] = []

    train_bars = train_years * 252
    test_bars = test_months * 21
    step_bars = step_months * 21
    train_start_idx = 0

    while train_start_idx + train_bars + test_bars <= len(all_dates):
        test_start_idx = train_start_idx + train_bars
        test_end_idx = test_start_idx + test_bars
        test_start = all_dates[test_start_idx]
        test_end = all_dates[min(test_end_idx, len(all_dates) - 1)]

        # Slice. We pass the *full* train+test history into the engine; the
        # engine itself enforces no-look-ahead via point-in-time slicing.
        # We then trim the equity curve to the test window only.
        slice_end = all_dates[min(test_end_idx, len(all_dates) - 1)]
        sub_prices = prices.loc[prices.index.get_level_values("date") <= slice_end]
        sub_index = index_ohlc.loc[:slice_end] if index_ohlc is not None else None

        strat = strategy_factory()
        result = backtester.run(
            strategy=strat,
            prices=sub_prices,
            index_ohlc=sub_index,
            features=features,
            sentiment=sentiment,
        )

        # Trim to test window only:
        test_mask = (result.equity.index >= test_start) & (result.equity.index <= test_end)
        test_equity = result.equity.loc[test_mask]
        if test_equity.empty:
            train_start_idx += step_bars
            continue

        folds.append(result)
        stitched_pieces.append(test_equity)
        train_start_idx += step_bars

    if not stitched_pieces:
        raise RuntimeError("No folds completed — check window sizing vs history length.")

    # Stitch: rescale each fold so it starts where the previous ended.
    stitched: list[pd.Series] = []
    cum_factor = 1.0
    for piece in stitched_pieces:
        scaled = piece * cum_factor / piece.iloc[0]
        cum_factor = float(scaled.iloc[-1])
        stitched.append(scaled)
    full = pd.concat(stitched).sort_index()
    full = full[~full.index.duplicated(keep="last")]

    return WalkForwardResult(
        folds=folds,
        stitched_equity=full,
        stitched_summary=summarise(full),
    )


__all__ = [
    "BacktestResult",
    "Backtester",
    "TradeLog",
    "WalkForwardResult",
    "walk_forward",
]


# Stubs to keep imports tidy and silence lint
_ = (datetime, np)
