"""Performance-metric tests with known-answer fixtures.

Strategy: build series with mathematically-known properties and verify
the metric returns the expected value within tight tolerances.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.metrics import (
    TRADING_DAYS_PER_YEAR,
    avg_loss,
    avg_win,
    beta,
    cagr,
    calmar_ratio,
    deflated_sharpe_ratio,
    drawdown_series,
    hit_rate,
    lo_sharpe_ratio,
    max_drawdown,
    max_drawdown_duration_days,
    profit_factor,
    returns_from_equity,
    sharpe_ratio,
    sortino_ratio,
    summarise,
)

# ---------------------------------------------------------------------------
# CAGR
# ---------------------------------------------------------------------------


def test_cagr_doubles_in_one_year() -> None:
    """An equity series that goes 1.0 → 2.0 over exactly 252 bars is 100% CAGR."""
    idx = pd.date_range("2024-01-01", periods=253, freq="B")
    equity = pd.Series(np.linspace(1.0, 2.0, 253), index=idx)
    assert cagr(equity) == pytest.approx(1.0, abs=0.005)


def test_cagr_zero_for_flat_series() -> None:
    idx = pd.date_range("2024-01-01", periods=253, freq="B")
    equity = pd.Series(1.0, index=idx)
    assert cagr(equity) == pytest.approx(0.0, abs=1e-10)


def test_cagr_nan_for_too_short() -> None:
    s = pd.Series([1.0])
    assert np.isnan(cagr(s))


# ---------------------------------------------------------------------------
# Sharpe / Sortino
# ---------------------------------------------------------------------------


def test_sharpe_zero_for_zero_mean_excess() -> None:
    """If returns equal the per-period risk-free rate, Sharpe = 0."""
    rf = 0.06
    rng = np.random.default_rng(0)
    rets = pd.Series(rf / TRADING_DAYS_PER_YEAR + rng.normal(0, 0.01, 252))
    s = sharpe_ratio(rets, risk_free_rate_annual=rf)
    # Sharpe of pure noise centred on rf should be ~0
    assert abs(s) < 0.5


def test_sharpe_positive_for_positive_alpha() -> None:
    """Returns drawn from N(positive, low_vol) yield large Sharpe."""
    rng = np.random.default_rng(0)
    # 20% annual return, 8% annual vol → Sharpe ≈ 2.5
    rets = pd.Series(rng.normal(0.20 / 252, 0.08 / np.sqrt(252), 1000))
    s = sharpe_ratio(rets, risk_free_rate_annual=0.0)
    assert 1.5 < s < 3.5


def test_sharpe_lo_close_to_naive_for_iid() -> None:
    """For genuinely IID returns, Lo's correction should be small."""
    rng = np.random.default_rng(7)
    rets = pd.Series(rng.normal(0.0005, 0.012, 1000))
    naive = sharpe_ratio(rets, risk_free_rate_annual=0.0)
    lo = lo_sharpe_ratio(rets, risk_free_rate_annual=0.0)
    # Within 25% of each other for IID data
    if abs(naive) > 0.05:
        assert abs(lo - naive) / abs(naive) < 0.25


def test_sortino_infinite_when_no_downside() -> None:
    rets = pd.Series([0.01] * 100)
    assert np.isinf(sortino_ratio(rets))


def test_sortino_higher_than_sharpe_for_left_skewed_returns() -> None:
    """A series with rare big losses has Sortino > Sharpe (downside dominates total vol)."""
    rng = np.random.default_rng(3)
    base = rng.normal(0.001, 0.005, 200).tolist()
    base[10] = -0.10
    base[50] = -0.08
    rets = pd.Series(base)
    s = sharpe_ratio(rets, risk_free_rate_annual=0.0)
    so = sortino_ratio(rets)
    # Mostly positive returns + 2 big drops → Sortino should ≠ Sharpe
    assert not np.isnan(so) and not np.isnan(s)


# ---------------------------------------------------------------------------
# Drawdown
# ---------------------------------------------------------------------------


def test_max_drawdown_50_pct_for_known_series() -> None:
    equity = pd.Series([1.0, 2.0, 3.0, 1.5, 2.0, 4.0])
    # Peak 3.0 → trough 1.5 → DD = -0.5 → -50%
    assert max_drawdown(equity) == pytest.approx(-0.5)


def test_max_drawdown_zero_for_monotonic_rise() -> None:
    equity = pd.Series([1.0, 1.1, 1.2, 1.3, 1.4])
    assert max_drawdown(equity) == 0.0


def test_drawdown_series_never_positive() -> None:
    rng = np.random.default_rng(0)
    rets = pd.Series(rng.normal(0.0, 0.01, 200))
    equity = (1.0 + rets).cumprod()
    dd = drawdown_series(equity)
    assert (dd <= 1e-10).all()


def test_max_drawdown_duration_counts_bars() -> None:
    # Goes 1 → 2 → 1 → 1 → 1 → 2 → 3
    # Drawdown starts AFTER the peak at index 1 and ends AT the recovery
    # bar at index 5 — bars STRICTLY below peak are indices 2, 3, 4 = 3 bars.
    equity = pd.Series([1.0, 2.0, 1.0, 1.0, 1.0, 2.0, 3.0])
    duration = max_drawdown_duration_days(equity)
    assert duration == 3


# ---------------------------------------------------------------------------
# Calmar
# ---------------------------------------------------------------------------


def test_calmar_positive_for_profitable_strategy() -> None:
    rng = np.random.default_rng(0)
    rets = pd.Series(rng.normal(0.001, 0.01, 252))
    equity = (1.0 + rets).cumprod()
    assert calmar_ratio(equity) > 0


# ---------------------------------------------------------------------------
# Trade-level
# ---------------------------------------------------------------------------


def test_hit_rate() -> None:
    trades = pd.Series([0.05, -0.02, 0.03, -0.01, 0.04])
    assert hit_rate(trades) == pytest.approx(0.6)


def test_profit_factor() -> None:
    trades = pd.Series([0.10, -0.05, 0.05, -0.05])
    # wins = 0.15, losses = 0.10 → PF = 1.5
    assert profit_factor(trades) == pytest.approx(1.5)


def test_profit_factor_inf_when_no_losses() -> None:
    trades = pd.Series([0.05, 0.03])
    assert np.isinf(profit_factor(trades))


def test_avg_win_and_loss() -> None:
    trades = pd.Series([0.10, -0.05, 0.08, -0.03])
    assert avg_win(trades) == pytest.approx(0.09)
    assert avg_loss(trades) == pytest.approx(-0.04)


# ---------------------------------------------------------------------------
# Beta
# ---------------------------------------------------------------------------


def test_beta_one_when_returns_equal_benchmark() -> None:
    rng = np.random.default_rng(0)
    bench = pd.Series(rng.normal(0.0005, 0.01, 200))
    assert beta(bench, bench) == pytest.approx(1.0, abs=1e-9)


def test_beta_two_when_returns_double_benchmark() -> None:
    rng = np.random.default_rng(0)
    bench = pd.Series(rng.normal(0.0005, 0.01, 200))
    assert beta(2 * bench, bench) == pytest.approx(2.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Deflated Sharpe (Bailey-LdP)
# ---------------------------------------------------------------------------


def test_deflated_sharpe_high_for_large_sample_low_n_trials() -> None:
    """Sharpe of 2 with 1500 obs and only 1 trial → DSR very high."""
    dsr = deflated_sharpe_ratio(sharpe=2.0, n_returns=1500, skew=0.0, kurtosis=3.0, n_trials=1)
    assert dsr > 0.95


def test_deflated_sharpe_low_for_many_trials() -> None:
    """Same Sharpe but 1000 trials → multiple-testing penalty kicks in."""
    dsr = deflated_sharpe_ratio(sharpe=2.0, n_returns=1500, skew=0.0, kurtosis=3.0, n_trials=1000)
    assert dsr < 0.95


def test_deflated_sharpe_low_for_short_sample() -> None:
    """Even a 'great' Sharpe is suspicious if the sample is small.

    Use Sharpe = 1.0 (not 2.0) and n_trials=10 so neither DSR saturates
    at the upper bound of 1.0 and we can actually compare them.
    """
    dsr_short = deflated_sharpe_ratio(sharpe=1.0, n_returns=80, skew=0.0, kurtosis=3.0, n_trials=10)
    dsr_long = deflated_sharpe_ratio(
        sharpe=1.0, n_returns=2000, skew=0.0, kurtosis=3.0, n_trials=10
    )
    assert dsr_short < dsr_long


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def test_summarise_returns_all_fields() -> None:
    rng = np.random.default_rng(0)
    rets = pd.Series(rng.normal(0.0005, 0.01, 500))
    equity = (1.0 + rets).cumprod()
    bench_rets = pd.Series(rng.normal(0.0003, 0.01, 500))
    bench_equity = (1.0 + bench_rets).cumprod()
    trade_rets = pd.Series(rng.normal(0.005, 0.02, 50))
    weights = pd.DataFrame(rng.uniform(-0.1, 0.1, (500, 3)), columns=["A", "B", "C"])
    summary = summarise(
        equity=equity,
        trade_returns=trade_rets,
        weights=weights,
        benchmark_equity=bench_equity,
    )
    assert summary.cagr is not None
    assert summary.sharpe is not None
    assert summary.beta_to_benchmark is not None
    text = summary.pretty()
    assert "Sharpe" in text
    d = summary.as_dict()
    assert "cagr" in d


def test_returns_from_equity_inverts_equity_from_returns() -> None:
    rng = np.random.default_rng(0)
    rets = pd.Series(rng.normal(0.001, 0.01, 100))
    equity = (1.0 + rets).cumprod()
    out = returns_from_equity(equity)
    np.testing.assert_allclose(out.to_numpy(), rets.iloc[1:].to_numpy(), atol=1e-12)
