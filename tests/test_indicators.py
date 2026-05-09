"""Unit tests for technical indicators.

Strategy: build small synthetic series with *known* expected values and
verify the indicator output. Where closed-form expectations are tedious,
verify monotonic / structural properties instead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.indicators import (
    adx,
    atr,
    bollinger_bands,
    cross_sectional_zscore,
    donchian_channel,
    internal_bar_strength,
    log_returns,
    macd,
    momentum_12_1,
    rolling_return,
    rolling_volatility,
    rsi,
    simple_returns,
    true_range,
    zscore,
)


@pytest.fixture
def ramp() -> pd.Series:
    """A monotonically rising price series — forces RSI = 100 etc."""
    idx = pd.date_range("2024-01-01", periods=60, freq="B")
    return pd.Series(np.arange(100, 160), index=idx, dtype=float)


@pytest.fixture
def random_walk() -> pd.DataFrame:
    """Reproducible random OHLC for indicator sanity checks."""
    rng = np.random.default_rng(42)
    n = 250
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    rets = rng.normal(0.0005, 0.012, n)
    close = 1000.0 * np.exp(np.cumsum(rets))
    high = close * (1 + np.abs(rng.normal(0.0, 0.005, n)))
    low = close * (1 - np.abs(rng.normal(0.0, 0.005, n)))
    open_ = close * (1 + rng.normal(0.0, 0.003, n))
    volume = rng.integers(1_00_000, 10_00_000, n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


# ---------------------------------------------------------------------------
# Returns & momentum
# ---------------------------------------------------------------------------


def test_simple_returns_first_value_is_nan(ramp: pd.Series) -> None:
    r = simple_returns(ramp)
    assert pd.isna(r.iloc[0])
    assert r.iloc[1] == pytest.approx((101 - 100) / 100)


def test_log_returns_first_value_is_nan(ramp: pd.Series) -> None:
    lr = log_returns(ramp)
    assert pd.isna(lr.iloc[0])
    assert lr.iloc[1] == pytest.approx(np.log(101 / 100))


def test_rolling_return_window(ramp: pd.Series) -> None:
    r10 = rolling_return(ramp, 10)
    assert pd.isna(r10.iloc[9])
    assert r10.iloc[10] == pytest.approx(110 / 100 - 1)


def test_momentum_12_1_skips_recent_window(ramp: pd.Series) -> None:
    # On a ramp, 12-1 momentum should still be positive (ramp grows).
    m = momentum_12_1(ramp, lookback=30, skip=5)
    last_valid = m.dropna().iloc[-1]
    assert last_valid > 0


def test_momentum_12_1_rejects_bad_args(ramp: pd.Series) -> None:
    with pytest.raises(ValueError):
        momentum_12_1(ramp, lookback=10, skip=10)


# ---------------------------------------------------------------------------
# RSI
# ---------------------------------------------------------------------------


def test_rsi_pegs_at_100_for_monotonic_rise(ramp: pd.Series) -> None:
    r = rsi(ramp, window=14).dropna()
    assert (r.iloc[-5:] > 99).all()  # all wins → RSI ~ 100


def test_rsi_pegs_at_0_for_monotonic_fall(ramp: pd.Series) -> None:
    falling = pd.Series(ramp.values[::-1], index=ramp.index)
    r = rsi(falling, window=14).dropna()
    assert (r.iloc[-5:] < 1).all()


def test_rsi_within_bounds(random_walk: pd.DataFrame) -> None:
    r = rsi(random_walk["close"], window=14).dropna()
    assert (r >= 0).all()
    assert (r <= 100).all()


def test_rsi_rejects_bad_window(random_walk: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        rsi(random_walk["close"], window=1)


# ---------------------------------------------------------------------------
# ATR / True Range
# ---------------------------------------------------------------------------


def test_true_range_non_negative(random_walk: pd.DataFrame) -> None:
    tr = true_range(random_walk["high"], random_walk["low"], random_walk["close"])
    assert (tr.dropna() >= 0).all()


def test_atr_positive(random_walk: pd.DataFrame) -> None:
    a = atr(random_walk["high"], random_walk["low"], random_walk["close"], window=14)
    assert (a.dropna() > 0).all()


# ---------------------------------------------------------------------------
# Bollinger
# ---------------------------------------------------------------------------


def test_bollinger_bands_ordering(random_walk: pd.DataFrame) -> None:
    bb = bollinger_bands(random_walk["close"], window=20, n_std=2.0).dropna()
    assert (bb["upper"] > bb["mid"]).all()
    assert (bb["mid"] > bb["lower"]).all()


# ---------------------------------------------------------------------------
# MACD
# ---------------------------------------------------------------------------


def test_macd_columns(random_walk: pd.DataFrame) -> None:
    m = macd(random_walk["close"]).dropna()
    assert set(m.columns) == {"macd", "signal", "hist"}
    assert ((m["hist"] - (m["macd"] - m["signal"])).abs() < 1e-10).all()


def test_macd_rejects_bad_periods(random_walk: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        macd(random_walk["close"], fast=26, slow=12)


# ---------------------------------------------------------------------------
# Donchian
# ---------------------------------------------------------------------------


def test_donchian_excludes_current_bar(ramp: pd.Series) -> None:
    """Critical test: a new high today should NOT be matched by today's
    own value — we only look at the trailing window."""
    high = ramp
    low = ramp - 1
    dc = donchian_channel(high, low, window=10).dropna()
    # On a ramp, today's close is always > yesterday's max-of-10.
    assert (high.loc[dc.index] > dc["upper"]).all()


# ---------------------------------------------------------------------------
# ADX
# ---------------------------------------------------------------------------


def test_adx_high_in_strong_trend() -> None:
    idx = pd.date_range("2024-01-01", periods=120, freq="B")
    close = pd.Series(np.linspace(100, 200, 120), index=idx, dtype=float)
    high = close + 1
    low = close - 1
    a = adx(high, low, close, window=14).dropna()
    assert a["adx"].iloc[-10:].mean() > 25


def test_adx_low_in_choppy() -> None:
    idx = pd.date_range("2024-01-01", periods=120, freq="B")
    rng = np.random.default_rng(0)
    close = pd.Series(100 + rng.normal(0, 0.5, 120).cumsum(), index=idx, dtype=float)
    high = close + 1
    low = close - 1
    a = adx(high, low, close, window=14).dropna()
    # Random walk → ADX stays low on average.
    assert a["adx"].mean() < 35  # generous, ADX can spike briefly


# ---------------------------------------------------------------------------
# Volatility & z-score
# ---------------------------------------------------------------------------


def test_rolling_volatility_annualised(random_walk: pd.DataFrame) -> None:
    v = rolling_volatility(random_walk["close"], window=60).dropna()
    # Daily ~1.2% vol scales to ~19% annualised.
    assert 0.05 < v.mean() < 0.50


def test_zscore_zero_when_constant() -> None:
    s = pd.Series([5.0] * 30)
    z = zscore(s, window=10).dropna()
    # std=0 → emits NaN, not inf
    assert z.empty or z.isna().all()


def test_cross_sectional_zscore() -> None:
    panel = pd.DataFrame(
        {"A": [1.0, 2.0, 3.0], "B": [2.0, 3.0, 4.0], "C": [4.0, 5.0, 6.0]},
        index=pd.date_range("2024-01-01", periods=3, freq="B"),
    )
    z = cross_sectional_zscore(panel)
    # Each row should sum to ~0 since mean is removed.
    assert z.sum(axis=1).abs().max() < 1e-9


# ---------------------------------------------------------------------------
# IBS
# ---------------------------------------------------------------------------


def test_internal_bar_strength_bounds(random_walk: pd.DataFrame) -> None:
    ibs = internal_bar_strength(
        random_walk["high"], random_walk["low"], random_walk["close"]
    ).dropna()
    assert (ibs >= 0).all()
    assert (ibs <= 1).all()
