"""Tests for market regime detection and allocation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategies.regime import (
    DEFAULT_ALLOCATION_MAP,
    Regime,
    RegimeAllocation,
    RegimeDetector,
    RegimeDiagnostics,
    detect_allocation,
)


def _ohlc_from_close(close: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "high": close * 1.002,
            "low": close * 0.998,
            "close": close,
        }
    )


def test_classify_requires_high_low_close() -> None:
    det = RegimeDetector(vol_window=5, trend_window=10)
    bad = pd.DataFrame({"close": [1, 2, 3]})
    with pytest.raises(ValueError, match="high, low, close"):
        det.classify(bad)


def test_short_history_returns_unknown() -> None:
    det = RegimeDetector(vol_window=60, trend_window=200)
    idx = pd.date_range("2024-01-01", periods=50, freq="B")
    close = pd.Series(100.0, index=idx)
    out = det.classify(_ohlc_from_close(close))
    assert out.regime is Regime.UNKNOWN
    assert np.isnan(out.diagnostics.realised_vol_annual)


def test_uptrend_low_vol_regime() -> None:
    det = RegimeDetector(
        vol_window=10,
        trend_window=20,
        high_vol_threshold=0.50,
        trending_score_threshold=0.05,
    )
    n = 120
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    # Smooth ramp up — low vol, strong uptrend vs SMA
    close = pd.Series(np.linspace(100.0, 130.0, n), index=idx)
    out = det.classify(_ohlc_from_close(close))
    assert out.regime is Regime.TRENDING_UP_LOW_VOL
    assert out.diagnostics.trend_score > det.trending_score_threshold
    assert "momentum" in out.weights


def test_downtrend_negative_trend_score() -> None:
    det = RegimeDetector(
        vol_window=10,
        trend_window=25,
        high_vol_threshold=0.99,
        trending_score_threshold=0.02,
    )
    n = 100
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    close = pd.Series(np.linspace(180.0, 110.0, n), index=idx)
    out = det.classify(_ohlc_from_close(close))
    assert out.diagnostics.trend_score < 0
    assert out.regime is Regime.TRENDING_DOWN_LOW_VOL


def test_range_low_vol_when_not_trending() -> None:
    det = RegimeDetector(
        vol_window=10,
        trend_window=30,
        high_vol_threshold=0.80,
        trending_score_threshold=2.0,
    )
    n = 100
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    close = pd.Series(100 + 0.1 * np.sin(np.linspace(0, 4 * np.pi, n)), index=idx)
    out = det.classify(_ohlc_from_close(close))
    assert out.regime is Regime.RANGE_LOW_VOL


def test_regime_allocation_cash_weight() -> None:
    weights = {"a": 0.3, "b": 0.2}
    diag = RegimeDiagnostics(0.1, 0.5, 100.0, 105.0, Regime.TRENDING_UP_LOW_VOL)
    ra = RegimeAllocation(weights=weights, regime=Regime.TRENDING_UP_LOW_VOL, diagnostics=diag)
    assert ra.cash_weight == pytest.approx(0.5)


def test_trending_low_vol_alias_points_to_uptrend() -> None:
    assert Regime.TRENDING_LOW_VOL == Regime.TRENDING_UP_LOW_VOL


def test_default_map_covers_all_regime_enum_values() -> None:
    for r in Regime:
        if r in (Regime.TRENDING_LOW_VOL, Regime.TRENDING_HIGH_VOL):
            continue
        assert r in DEFAULT_ALLOCATION_MAP


def test_detect_allocation_uses_default_detector() -> None:
    det = RegimeDetector(vol_window=10, trend_window=20, high_vol_threshold=0.9, trending_score_threshold=0.01)
    n = 80
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    close = pd.Series(np.linspace(50.0, 80.0, n), index=idx)
    out = detect_allocation(_ohlc_from_close(close), detector=det)
    assert isinstance(out, RegimeAllocation)


def test_custom_allocation_map() -> None:
    custom = {Regime.UNKNOWN: {"pairs": 1.0}}
    det = RegimeDetector(vol_window=60, trend_window=200, allocation_map=custom)
    idx = pd.date_range("2024-01-01", periods=10, freq="B")
    close = pd.Series(100.0, index=idx)
    out = det.classify(_ohlc_from_close(close))
    assert out.regime is Regime.UNKNOWN
    assert out.weights == {"pairs": 1.0}


def test_mean_reverting_noise_classifies_without_crash() -> None:
    """High daily noise around a flat level — regime logic should return a label."""
    det = RegimeDetector(
        vol_window=10,
        trend_window=25,
        high_vol_threshold=0.50,
        trending_score_threshold=2.0,
    )
    rng = np.random.default_rng(1)
    n = 90
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    close = pd.Series(100.0 + rng.normal(0, 0.8, n), index=idx)
    out = det.classify(_ohlc_from_close(close))
    assert out.regime in Regime
    assert np.isfinite(out.diagnostics.last_price)
