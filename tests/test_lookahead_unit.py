"""Look-ahead auditor surface tests (no full backtest)."""

from __future__ import annotations

import pandas as pd
import pytest

from src.backtest.engine import Backtester
from src.backtest.lookahead import LookaheadReport, audit_strategy


def test_lookahead_report_pretty_multiline() -> None:
    rep = LookaheadReport(
        full_trades=10,
        truncated_trades=8,
        overlapping_window_start=pd.Timestamp("2024-01-02"),
        overlapping_window_end=pd.Timestamp("2024-06-30"),
        matched=7,
        mismatched=0,
        only_in_full=0,
        only_in_truncated=0,
        verdict="clean",
    )
    text = rep.pretty()
    assert "CLEAN" in text
    assert "Matched" in text
    assert "2024-01-02" in text


def test_audit_strategy_rejects_non_multiindex() -> None:
    bt = Backtester(initial_capital=100_000.0, rebalance_freq="M")
    bad_prices = pd.DataFrame({"x": [1, 2]})
    with pytest.raises(ValueError, match="MultiIndex"):
        audit_strategy(lambda _p: None, bt, bad_prices)  # type: ignore[arg-type, return-value]


def test_audit_strategy_rejects_short_history() -> None:
    from src.backtest.costs import IndianEquityCostModel

    bt = Backtester(
        initial_capital=100_000.0,
        rebalance_freq="M",
        cost_model=IndianEquityCostModel(),
    )
    idx = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-01", periods=20, freq="B"), ["A.NS"]],
        names=["date", "ticker"],
    )
    prices = pd.DataFrame(
        {"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0},
        index=idx,
    )
    with pytest.raises(ValueError, match="Need at least"):
        audit_strategy(lambda _p: None, bt, prices, truncate_bars=60)  # type: ignore[arg-type, return-value]
