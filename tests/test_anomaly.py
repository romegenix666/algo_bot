"""Anomaly detector tests — synthetic bad bars, no network."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.data.anomaly import Anomaly, Severity, detect_anomalies, summarise_anomalies


@pytest.fixture
def good_bars() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=20, freq="B")
    rng = np.random.default_rng(0)
    close = 1000 + np.cumsum(rng.normal(0, 5, 20))
    return pd.DataFrame(
        {
            "open": close - 1,
            "high": close + 5,
            "low": close - 5,
            "close": close,
            "volume": rng.integers(1_00_000, 10_00_000, 20),
        },
        index=idx,
    )


def test_clean_bars_have_no_errors(good_bars: pd.DataFrame) -> None:
    out = detect_anomalies("TEST", good_bars)
    # Should be empty (or only INFO at worst)
    errors = [a for a in out if a.severity is Severity.ERROR]
    assert errors == []


def test_negative_price_is_error(good_bars: pd.DataFrame) -> None:
    bad = good_bars.copy()
    bad.iloc[3, bad.columns.get_loc("low")] = -1.0
    out = detect_anomalies("TEST", bad)
    codes = {a.code for a in out}
    assert "bad_price" in codes


def test_nan_close_is_error(good_bars: pd.DataFrame) -> None:
    bad = good_bars.copy()
    bad.iloc[5, bad.columns.get_loc("close")] = float("nan")
    out = detect_anomalies("TEST", bad)
    assert any(a.code == "bad_price" for a in out)


def test_high_below_low_is_inconsistent(good_bars: pd.DataFrame) -> None:
    bad = good_bars.copy()
    bad.iloc[7, bad.columns.get_loc("high")] = bad.iloc[7]["low"] - 1
    out = detect_anomalies("TEST", bad)
    assert any(a.code == "ohlc_inconsistent" for a in out)


def test_close_outside_high_low_inconsistent(good_bars: pd.DataFrame) -> None:
    bad = good_bars.copy()
    bad.iloc[2, bad.columns.get_loc("close")] = bad.iloc[2]["high"] + 10
    out = detect_anomalies("TEST", bad)
    assert any(a.code == "ohlc_inconsistent" for a in out)


def test_zero_volume_warns(good_bars: pd.DataFrame) -> None:
    bad = good_bars.copy()
    bad.iloc[10, bad.columns.get_loc("volume")] = 0
    out = detect_anomalies("TEST", bad)
    assert any(a.code == "zero_volume" for a in out)


def test_stale_feed_detection() -> None:
    """3+ consecutive bars with same OHLC and zero volume → stale_feed."""
    idx = pd.date_range("2024-01-01", periods=10, freq="B")
    bars = pd.DataFrame(
        {
            "open": [100.0] * 3 + list(range(101, 108)),
            "high": [100.0] * 3 + list(range(102, 109)),
            "low": [100.0] * 3 + list(range(99, 106)),
            "close": [100.0] * 3 + list(range(101, 108)),
            "volume": [0, 0, 0] + [1_00_000] * 7,
        },
        index=idx,
    )
    out = detect_anomalies("TEST", bars)
    assert any(a.code == "stale_feed" for a in out)


def test_unexplained_gap_flags_when_no_action(good_bars: pd.DataFrame) -> None:
    bad = good_bars.copy()
    # 50% jump on day 5
    bad.iloc[5, bad.columns.get_loc("close")] *= 1.5
    bad.iloc[5, bad.columns.get_loc("high")] = bad.iloc[5]["close"] + 1
    out = detect_anomalies("TEST", bad)
    assert any(a.code == "unexplained_gap" for a in out)


def test_unexplained_gap_suppressed_by_action(good_bars: pd.DataFrame) -> None:
    bad = good_bars.copy()
    # Simulate a real stock split: a 1.5x bump applies to ALL bars from
    # the ex-date onwards (not just the one bar). With a split action
    # recorded on that day, the gap should NOT flag.
    gap_date = bad.index[5].date()
    for col in ("open", "high", "low", "close"):
        ci = bad.columns.get_loc(col)
        bad.iloc[5:, ci] = bad.iloc[5:, ci].to_numpy() * 1.5
    actions = pd.DataFrame([{"ex_date": gap_date, "action_type": "split", "ratio": 1.5}])
    out = detect_anomalies("TEST", bad, actions=actions)
    assert not any(a.code == "unexplained_gap" for a in out)


def test_summarise_anomalies_groups_by_code() -> None:
    anomalies = [
        Anomaly("X", date(2024, 1, 1), "zero_volume", Severity.WARN, ""),
        Anomaly("X", date(2024, 1, 2), "zero_volume", Severity.WARN, ""),
        Anomaly("X", date(2024, 1, 5), "bad_price", Severity.ERROR, ""),
    ]
    summary = summarise_anomalies(anomalies)
    assert summary == {"zero_volume": 2, "bad_price": 1}


def test_empty_bars_returns_empty() -> None:
    out = detect_anomalies("X", pd.DataFrame())
    assert out == []


def test_missing_columns_raises() -> None:
    bad = pd.DataFrame({"close": [1, 2, 3]})
    with pytest.raises(ValueError, match="missing required columns"):
        detect_anomalies("X", bad)
