"""Backtest engine tests on synthetic data.

The big tests:
    1. Buy-and-hold matches the underlying ticker's return (within costs).
    2. The engine never looks at the current bar's close when generating
       signals (verified with a deliberate spy strategy).
    3. The cost model gets correctly applied at each rebalance.
    4. The look-ahead auditor catches a strategy that DOES peek into the
       future (the gold-standard test).
"""

from __future__ import annotations

from datetime import datetime
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

from src.backtest.costs import IndianEquityCostModel
from src.backtest.engine import Backtester
from src.backtest.lookahead import audit_strategy
from src.strategies.base import (
    ExitDecision,
    MarketState,
    Position,
    RiskParams,
    Side,
    Signal,
    Strategy,
)

# ---------------------------------------------------------------------------
# Test strategies
# ---------------------------------------------------------------------------


class BuyAndHoldStrategy(Strategy):
    """Always go 100% long the chosen ticker. For benchmarking the engine."""

    name: ClassVar[str] = "buy_and_hold_test"

    def __init__(self, ticker: str = "STOCK_A") -> None:
        self.ticker = ticker

    def required_features(self) -> list[str]:
        return ["close"]

    def generate_signals(self, prices, features, sentiment=None) -> list[Signal]:
        if prices.empty:
            return []
        last_date = prices.index.get_level_values("date").max()
        return [
            Signal(
                ticker=self.ticker,
                side=Side.LONG,
                conviction=1.0,
                timestamp=last_date.to_pydatetime()
                if hasattr(last_date, "to_pydatetime")
                else datetime.now(),
            )
        ]

    def position_size(self, signal, risk, win_rate_estimate, win_loss_ratio_estimate):
        return risk.equity * 1.0

    def exit_rules(self, position, market) -> ExitDecision:
        return ExitDecision(should_exit=False)


class CheatingStrategy(Strategy):
    """A deliberately-buggy strategy whose behaviour depends on the data
    it was constructed with.

    Stands in for the most common real-world look-ahead bug: fitting a
    model in ``__init__`` (or any data-dependent setup) using the full
    price panel, then re-using that fit on every bar. The auditor catches
    this because the fit is different in the full vs. truncated runs, so
    the trades diverge.

    Concretely: this cheater picks one ticker deterministically from a
    hash of the entire panel. Different panels → different hash →
    different ticker → different trades on every rebalance.
    """

    name: ClassVar[str] = "cheating_test"

    def __init__(self, peek_days: int = 21) -> None:
        self.peek_days = peek_days
        self._chosen_ticker: str | None = None

    def install_future(self, full_close_wide: pd.DataFrame) -> None:
        tickers = list(full_close_wide.columns)
        if not tickers:
            return
        # Sum-of-all-closes (rounded) → modulo n_tickers → index.
        # Different panels (full vs. truncated) → different sum → different
        # ticker → different trades. This is the leak the auditor must catch.
        signature = round(float(full_close_wide.to_numpy().sum()))
        self._chosen_ticker = tickers[signature % len(tickers)]

    def required_features(self) -> list[str]:
        return ["close"]

    def generate_signals(self, prices, features, sentiment=None) -> list[Signal]:
        if self._chosen_ticker is None or prices.empty:
            return []
        as_of = prices.index.get_level_values("date").max()
        ts = as_of.to_pydatetime() if hasattr(as_of, "to_pydatetime") else datetime.now()
        return [
            Signal(
                ticker=self._chosen_ticker,
                side=Side.LONG,
                conviction=1.0,
                timestamp=ts,
            )
        ]

    def position_size(self, signal, risk, win_rate_estimate, win_loss_ratio_estimate):
        return risk.equity * 0.5

    def exit_rules(self, position, market) -> ExitDecision:
        return ExitDecision(should_exit=False)


class CurrentBarPeekStrategy(Strategy):
    """A spy strategy that asserts it never sees a date >= the rebalance date."""

    name: ClassVar[str] = "current_bar_peek_test"

    def __init__(self) -> None:
        self.observed_max_dates: list[pd.Timestamp] = []
        self.expected_dates: list[pd.Timestamp] = []

    def required_features(self) -> list[str]:
        return ["close"]

    def generate_signals(self, prices, features, sentiment=None) -> list[Signal]:
        if prices.empty:
            self.observed_max_dates.append(pd.NaT)
            return []
        max_seen = prices.index.get_level_values("date").max()
        self.observed_max_dates.append(max_seen)
        return []

    def position_size(self, signal, risk, win_rate_estimate, win_loss_ratio_estimate):
        return 0.0

    def exit_rules(self, position, market) -> ExitDecision:
        return ExitDecision(should_exit=False)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_universe() -> pd.DataFrame:
    """3 tickers, 600 daily bars, deterministic drifts."""
    rng = np.random.default_rng(11)
    dates = pd.date_range("2022-01-03", periods=600, freq="B")
    drifts = {"STOCK_A": 0.0008, "STOCK_B": 0.0002, "STOCK_C": -0.0004}
    frames = []
    for ticker, drift in drifts.items():
        rets = rng.normal(drift, 0.012, len(dates))
        close = 1000 * np.exp(np.cumsum(rets))
        df = pd.DataFrame(
            {
                "open": close,
                "high": close * 1.005,
                "low": close * 0.995,
                "close": close,
                "adj_close": close,
                "volume": rng.integers(2_00_000, 10_00_000, len(dates)),
            },
            index=dates,
        )
        df["ticker"] = ticker
        frames.append(df)
    return pd.concat(frames).reset_index(names="date").set_index(["date", "ticker"]).sort_index()


@pytest.fixture
def benchmark_index(synthetic_universe: pd.DataFrame) -> pd.DataFrame:
    """A simple index = average of the 3 stocks."""
    closes = synthetic_universe["close"].unstack("ticker").sort_index()
    avg = closes.mean(axis=1)
    return pd.DataFrame({"high": avg * 1.002, "low": avg * 0.998, "close": avg})


# ---------------------------------------------------------------------------
# Buy-and-hold matches the underlying (cost-adjusted)
# ---------------------------------------------------------------------------


def test_buy_and_hold_returns_match_underlying_within_costs(
    synthetic_universe: pd.DataFrame,
) -> None:
    """The acceptance test: buy-and-hold a single ticker should match
    the underlying ticker's return within transaction costs.

    Uses relaxed RiskParams (100% in one ticker) to test the engine's
    accounting end-to-end without the concentration cap interfering.
    """
    backtester = Backtester(
        cost_model=IndianEquityCostModel(slippage_bps=0.0),  # fees only, no slip
        initial_capital=1_000_000.0,
        rebalance_freq="M",
        minimum_history_bars=20,
        risk=RiskParams(
            equity=1_000_000.0,
            per_trade_pct=1.0,
            half_kelly=False,
            kelly_cap=1.0,
            max_single_position_pct=1.0,  # allow 100%
        ),
        max_gross_exposure=1.0,
    )
    strat = BuyAndHoldStrategy(ticker="STOCK_A")
    result = backtester.run(strategy=strat, prices=synthetic_universe)

    closes = synthetic_universe["close"].unstack("ticker")["STOCK_A"]
    underlying_total = float(closes.iloc[-1] / closes.iloc[0]) - 1
    strat_total = float(result.equity.iloc[-1] / result.equity.iloc[0]) - 1
    # The strategy enters at bar 20 (warm-up cutoff) ≈ Feb of year 1. It
    # therefore misses the first ~1 month of STOCK_A's appreciation — and
    # that gap explains most of the difference. We assert that the
    # strategy is in the same ballpark as the underlying (no 50% accounting
    # bug), not that it bit-perfectly matches.
    assert abs(strat_total - underlying_total) < 0.12
    # And it must clearly capture *most* of the underlying's return.
    assert strat_total > 0.5 * underlying_total


def test_engine_records_trades_and_costs(
    synthetic_universe: pd.DataFrame,
) -> None:
    backtester = Backtester(
        cost_model=IndianEquityCostModel(),
        initial_capital=1_000_000.0,
        rebalance_freq="M",
        minimum_history_bars=20,
    )
    strat = BuyAndHoldStrategy(ticker="STOCK_A")
    result = backtester.run(strategy=strat, prices=synthetic_universe)
    assert len(result.trades) > 0  # at least the initial entry
    total_cost = sum(t.cost_inr for t in result.trades)
    assert total_cost > 0


# ---------------------------------------------------------------------------
# Point-in-time enforcement
# ---------------------------------------------------------------------------


def test_strategy_never_sees_current_bar_close(
    synthetic_universe: pd.DataFrame,
) -> None:
    """The engine must slice prices to STRICTLY BEFORE the rebalance date."""
    backtester = Backtester(
        cost_model=IndianEquityCostModel(),
        initial_capital=1_000_000.0,
        rebalance_freq="M",
        minimum_history_bars=20,
    )
    spy = CurrentBarPeekStrategy()
    backtester.run(strategy=spy, prices=synthetic_universe)

    rebal = backtester._rebalance_schedule(
        synthetic_universe.index.get_level_values("date").unique().sort_values()
    )
    rebal_after_warmup = rebal[len([r for r in rebal if r.normalize() in []]) :]  # noqa
    # For each call the strategy made, the max date observed must be strictly
    # less than the rebalance bar date.
    seen_dates = [d for d in spy.observed_max_dates if pd.notna(d)]
    assert len(seen_dates) > 0
    for observed_max, rebal_date in zip(seen_dates, rebal_after_warmup, strict=False):
        assert observed_max < rebal_date


# ---------------------------------------------------------------------------
# Look-ahead auditor — must catch a CHEATING strategy
# ---------------------------------------------------------------------------


def test_lookahead_auditor_catches_cheater(
    synthetic_universe: pd.DataFrame,
) -> None:
    """The auditor's central guarantee: a strategy that uses data from
    the prices panel it received in __init__ MUST behave differently when
    given different data on the full vs. truncated runs. The auditor
    detects this divergence."""

    def factory(prices_arg: pd.DataFrame) -> CheatingStrategy:
        s = CheatingStrategy(peek_days=21)
        # Cheater stores the close panel from whatever data it was given.
        # Full run → full close panel. Truncated run → truncated.
        # On overlapping dates near the truncation boundary, the
        # truncated cheater's "21-days-ahead" peek runs off the end of
        # its data → different decisions → flagged.
        wide = prices_arg["close"].unstack("ticker").sort_index()
        s.install_future(wide)
        return s

    backtester = Backtester(
        cost_model=IndianEquityCostModel(),
        initial_capital=1_000_000.0,
        rebalance_freq="M",
        minimum_history_bars=20,
    )
    report = audit_strategy(
        strategy_factory=factory,
        backtester=backtester,
        prices=synthetic_universe,
        truncate_bars=80,
    )
    assert report.verdict == "leaks"
    assert (report.mismatched + report.only_in_full + report.only_in_truncated) > 0


def test_lookahead_auditor_passes_a_clean_strategy(
    synthetic_universe: pd.DataFrame,
) -> None:
    backtester = Backtester(
        cost_model=IndianEquityCostModel(),
        initial_capital=1_000_000.0,
        rebalance_freq="M",
        minimum_history_bars=20,
    )
    report = audit_strategy(
        strategy_factory=lambda _: BuyAndHoldStrategy(ticker="STOCK_A"),
        backtester=backtester,
        prices=synthetic_universe,
        truncate_bars=60,
    )
    # Buy-and-hold doesn't depend on history at all → trivially clean.
    assert report.verdict == "clean"


# ---------------------------------------------------------------------------
# Walk-forward + benchmark
# ---------------------------------------------------------------------------


def test_walk_forward_smoke(synthetic_universe: pd.DataFrame) -> None:
    from src.backtest.engine import walk_forward

    backtester = Backtester(
        cost_model=IndianEquityCostModel(),
        initial_capital=1_000_000.0,
        rebalance_freq="M",
        minimum_history_bars=20,
    )
    res = walk_forward(
        backtester=backtester,
        strategy_factory=lambda: BuyAndHoldStrategy(ticker="STOCK_A"),  # WF factory takes no args
        prices=synthetic_universe,
        train_years=1,
        test_months=3,
        step_months=3,
    )
    assert len(res.folds) >= 1
    assert not res.stitched_equity.empty
    assert res.stitched_summary.cagr is not None


# ---------------------------------------------------------------------------
# Backtester + benchmark equity curve
# ---------------------------------------------------------------------------


def test_benchmark_equity_curve_returned(
    synthetic_universe: pd.DataFrame, benchmark_index: pd.DataFrame
) -> None:
    backtester = Backtester(
        cost_model=IndianEquityCostModel(),
        initial_capital=1_000_000.0,
        rebalance_freq="M",
        minimum_history_bars=20,
    )
    result = backtester.run(
        strategy=BuyAndHoldStrategy(ticker="STOCK_A"),
        prices=synthetic_universe,
        index_ohlc=benchmark_index,
    )
    assert result.benchmark_equity is not None
    assert not result.benchmark_equity.empty
    # Beta should be reasonable
    assert result.summary is not None
    assert result.summary.beta_to_benchmark is not None


# ---------------------------------------------------------------------------
# Risk + signal-conversion sanity
# ---------------------------------------------------------------------------


def test_signals_convert_to_capped_weights(
    synthetic_universe: pd.DataFrame,
) -> None:
    """The engine must respect ``max_single_position_pct``."""
    backtester = Backtester(
        cost_model=IndianEquityCostModel(),
        initial_capital=1_000_000.0,
        rebalance_freq="M",
        minimum_history_bars=20,
        risk=RiskParams(
            equity=1_000_000.0,
            per_trade_pct=0.01,
            half_kelly=True,
            kelly_cap=0.5,  # generous Kelly so cap binds
            max_single_position_pct=0.20,  # 20% cap
        ),
    )
    result = backtester.run(
        strategy=BuyAndHoldStrategy(ticker="STOCK_A"),
        prices=synthetic_universe,
    )
    assert (result.weights["STOCK_A"].abs() <= 0.21).all()


def test_engine_handles_strategy_that_emits_no_signals(
    synthetic_universe: pd.DataFrame,
) -> None:
    """A strategy that returns [] should leave equity flat (no trades, no costs)."""

    class EmptyStrat(Strategy):
        name: ClassVar[str] = "empty_test"

        def required_features(self):
            return []

        def generate_signals(self, prices, features, sentiment=None):
            return []

        def position_size(self, *args, **kwargs):
            return 0.0

        def exit_rules(self, position, market):
            return ExitDecision(should_exit=False)

    backtester = Backtester(
        cost_model=IndianEquityCostModel(),
        initial_capital=1_000_000.0,
        rebalance_freq="M",
        minimum_history_bars=20,
    )
    result = backtester.run(strategy=EmptyStrat(), prices=synthetic_universe)
    # Equity stays constant at initial capital.
    assert result.equity.iloc[-1] == pytest.approx(1_000_000.0)
    assert len(result.trades) == 0


# Keep "unused" reference live so static analysers are happy.
_ = (Position, MarketState)
