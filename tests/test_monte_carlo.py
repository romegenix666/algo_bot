"""Tests for block-bootstrap Monte Carlo report."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.monte_carlo import MonteCarloReport, block_bootstrap


def _daily_returns(n: int, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.Series(rng.normal(0.0003, 0.01, n), index=idx)


def test_block_bootstrap_raises_when_too_few_bars() -> None:
    short = _daily_returns(15)
    with pytest.raises(ValueError, match="Not enough returns"):
        block_bootstrap(short, n_simulations=10, block_size=5)


def test_block_bootstrap_deterministic_with_seed() -> None:
    r = _daily_returns(80)
    a = block_bootstrap(r, n_simulations=30, block_size=5, seed=123)
    b = block_bootstrap(r, n_simulations=30, block_size=5, seed=123)
    assert a.sharpe_mean == b.sharpe_mean
    assert a.max_dd_p50 == b.max_dd_p50


def test_block_bootstrap_different_seed_usually_changes_draw() -> None:
    r = _daily_returns(200)
    a = block_bootstrap(r, n_simulations=80, block_size=5, seed=11)
    b = block_bootstrap(r, n_simulations=80, block_size=5, seed=999)
    # Extremely unlikely all reported moments match exactly across independent paths.
    assert (
        abs(a.sharpe_mean - b.sharpe_mean) > 1e-9
        or abs(a.max_dd_mean - b.max_dd_mean) > 1e-9
        or abs(a.cagr_mean - b.cagr_mean) > 1e-9
    )


def test_monte_carlo_percentile_ordering() -> None:
    r = _daily_returns(100)
    rep = block_bootstrap(r, n_simulations=200, block_size=5, seed=7)
    assert rep.sharpe_p05 <= rep.sharpe_p50 <= rep.sharpe_p95
    assert rep.max_dd_p05 <= rep.max_dd_p50  # all negative or zero; p05 is more extreme


def test_monte_carlo_report_pretty_contains_key_lines() -> None:
    rep = MonteCarloReport(
        n_simulations=10,
        block_size=5,
        sharpe_mean=0.5,
        sharpe_p05=0.1,
        sharpe_p50=0.5,
        sharpe_p95=0.9,
        cagr_mean=0.1,
        cagr_p05=0.05,
        cagr_p95=0.15,
        max_dd_mean=-0.1,
        max_dd_p05=-0.2,
        max_dd_p50=-0.1,
        max_dd_p95=-0.05,
    )
    text = rep.pretty()
    assert "simulations" in text
    assert "Sharpe" in text


def test_block_bootstrap_accepts_na_dropped() -> None:
    r = _daily_returns(50)
    r.iloc[3] = np.nan
    rep = block_bootstrap(r, n_simulations=20, block_size=5, seed=0)
    assert rep.n_simulations == 20
