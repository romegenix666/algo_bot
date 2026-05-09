"""Additional indicator edge-case coverage."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.indicators import (
    momentum_12_1,
    rolling_mean,
    rolling_return,
    rolling_std,
    rolling_volatility,
)


def test_rolling_mean_matches_manual_window() -> None:
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    m = rolling_mean(s, 3)
    assert pd.isna(m.iloc[0]) and pd.isna(m.iloc[1])
    assert m.iloc[2] == pytest.approx(2.0)


def test_rolling_std_sample_ddof() -> None:
    s = pd.Series([1.0, 2.0, 3.0, 5.0])
    st = rolling_std(s, 3)
    assert st.iloc[-1] > 0


def test_rolling_return_rejects_nonpositive_window() -> None:
    s = pd.Series([1.0, 2.0])
    with pytest.raises(ValueError, match="positive"):
        rolling_return(s, 0)


def test_momentum_12_1_rejects_lookback_le_skip() -> None:
    s = pd.Series(np.linspace(1, 2, 30))
    with pytest.raises(ValueError, match="lookback"):
        momentum_12_1(s, lookback=5, skip=10)


def test_rolling_volatility_no_annualize() -> None:
    idx = pd.date_range("2024-01-01", periods=80, freq="B")
    close = pd.Series(100 * np.exp(np.cumsum(np.random.default_rng(3).normal(0, 0.01, 80))), index=idx)
    vol_d = rolling_volatility(close, window=20, annualize=False).dropna()
    vol_a = rolling_volatility(close, window=20, annualize=True).dropna()
    assert (vol_a / vol_d).mean() == pytest.approx(np.sqrt(252), rel=0.01)


def test_rolling_return_positive_window_one() -> None:
    s = pd.Series([10.0, 11.0, 12.0])
    r = rolling_return(s, 1)
    assert r.iloc[1] == pytest.approx(0.1)


def test_rolling_mean_preserves_index() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="B")
    s = pd.Series([1, 2, 3, 4, 5], index=idx)
    m = rolling_mean(s, 2)
    assert m.index.equals(s.index)
