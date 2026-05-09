"""End-to-end tests: every strategy generates valid signals on synthetic data."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from src.strategies.base import RiskParams, Side, Signal, Strategy
from src.strategies.breakout import DonchianBreakoutStrategy
from src.strategies.dual_momentum import DualMomentumStrategy
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.momentum import MomentumStrategy
from src.strategies.multi_factor import FactorWeights, MultiFactorStrategy
from src.strategies.pairs import PairCandidate, PairsTradingStrategy
from src.strategies.regime import Regime, RegimeDetector
from src.strategies.registry import (
    available_strategies,
    build_strategies,
    build_strategy,
)
from src.strategies.sector_rotation import SectorRotationStrategy
from src.strategies.selector import StrategySelector
from src.strategies.sentiment_momentum import SentimentMomentumStrategy

# ---------------------------------------------------------------------------
# Fixtures: synthetic universe
# ---------------------------------------------------------------------------


@pytest.fixture
def universe_prices() -> pd.DataFrame:
    """A 5-stock universe with diverse trends so different strategies fire.

    Returns a MultiIndex (date, ticker) frame with OHLCV columns.
    """
    rng = np.random.default_rng(7)
    dates = pd.date_range("2022-01-03", periods=400, freq="B")

    # Drifts chosen so that the cumulative drift over the 252-day momentum
    # window (~drift × 252) clearly exceeds the noise std-error
    # (~vol × √252). For STRONG_UP: drift=0.0025 → +63% expected; vol=0.012
    # → ±19% std-err. So STRONG_UP wins reliably across seeds.
    drifts = {
        "STRONG_UP.NS": 0.0025,
        "MILD_UP.NS": 0.0008,
        "FLAT.NS": 0.0,
        "MILD_DOWN.NS": -0.0006,
        "STRONG_DOWN.NS": -0.0020,
    }
    vols = {
        "STRONG_UP.NS": 0.012,
        "MILD_UP.NS": 0.010,
        "FLAT.NS": 0.010,
        "MILD_DOWN.NS": 0.011,
        "STRONG_DOWN.NS": 0.014,
    }

    frames = []
    for ticker, drift in drifts.items():
        rets = rng.normal(drift, vols[ticker], len(dates))
        close = 1000 * np.exp(np.cumsum(rets))
        high = close * (1 + np.abs(rng.normal(0, 0.005, len(dates))))
        low = close * (1 - np.abs(rng.normal(0, 0.005, len(dates))))
        open_ = close * (1 + rng.normal(0, 0.003, len(dates)))
        vol = rng.integers(2_00_000, 10_00_000, len(dates))
        df = pd.DataFrame(
            {
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": vol,
            },
            index=dates,
        )
        df["ticker"] = ticker
        frames.append(df)

    full = pd.concat(frames).reset_index(names="date")
    full = full.set_index(["date", "ticker"]).sort_index()
    return full


@pytest.fixture
def index_ohlc() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    dates = pd.date_range("2022-01-03", periods=400, freq="B")
    rets = rng.normal(0.0006, 0.011, len(dates))
    close = 18000 * np.exp(np.cumsum(rets))
    high = close * (1 + np.abs(rng.normal(0, 0.004, len(dates))))
    low = close * (1 - np.abs(rng.normal(0, 0.004, len(dates))))
    return pd.DataFrame({"high": high, "low": low, "close": close}, index=dates)


@pytest.fixture
def fundamentals(universe_prices: pd.DataFrame) -> pd.DataFrame:
    """Static fundamentals snapshot, indexed by (date, ticker)."""
    tickers = universe_prices.index.get_level_values("ticker").unique()
    rng = np.random.default_rng(3)
    dates = universe_prices.index.get_level_values("date").unique()
    rows = []
    for d in dates:
        for t in tickers:
            rows.append(
                {
                    "date": d,
                    "ticker": t,
                    "pe_ratio": float(rng.uniform(10, 40)),
                    "pb_ratio": float(rng.uniform(1, 8)),
                    "roe": float(rng.uniform(5, 30)),
                    "debt_to_equity": float(rng.uniform(0.1, 1.5)),
                }
            )
    return pd.DataFrame(rows).set_index(["date", "ticker"]).sort_index()


@pytest.fixture
def sentiment_scores() -> pd.DataFrame:
    """Explicit per-ticker sentiment so tests don't depend on ticker ordering."""
    return pd.DataFrame(
        [
            {"ticker": "STRONG_UP.NS", "score": 0.6},
            {"ticker": "MILD_UP.NS", "score": 0.3},
            {"ticker": "FLAT.NS", "score": 0.0},
            {"ticker": "MILD_DOWN.NS", "score": -0.4},
            {"ticker": "STRONG_DOWN.NS", "score": -0.7},
        ]
    )


@pytest.fixture
def standard_risk_params() -> RiskParams:
    return RiskParams(
        equity=1_000_000.0,
        per_trade_pct=0.01,
        half_kelly=True,
        kelly_cap=0.20,
        max_single_position_pct=0.20,
    )


# ---------------------------------------------------------------------------
# Common assertions for any Strategy implementation
# ---------------------------------------------------------------------------


def _assert_valid_signals(signals: list[Signal]) -> None:
    for sig in signals:
        assert isinstance(sig, Signal)
        assert sig.ticker
        assert sig.side in (Side.LONG, Side.SHORT)
        assert 0.0 <= sig.conviction <= 1.0


# ---------------------------------------------------------------------------
# Per-strategy tests
# ---------------------------------------------------------------------------


def test_momentum_signals_top_are_uptrending(
    universe_prices: pd.DataFrame,
) -> None:
    strat = MomentumStrategy(top_n=2)
    sigs = strat.generate_signals(universe_prices, pd.DataFrame())
    _assert_valid_signals(sigs)
    # The top-2 should include STRONG_UP (highest drift).
    longs = [s.ticker for s in sigs if s.side is Side.LONG]
    assert "STRONG_UP.NS" in longs


def test_mean_reversion_runs_without_error(
    universe_prices: pd.DataFrame,
) -> None:
    strat = MeanReversionStrategy(adf_pvalue_max=0.5)  # relaxed for synthetic data
    sigs = strat.generate_signals(universe_prices, pd.DataFrame())
    _assert_valid_signals(sigs)


def test_pairs_strategy_runs_without_error(
    universe_prices: pd.DataFrame,
) -> None:
    strat = PairsTradingStrategy(
        candidates=[
            PairCandidate("STRONG_UP.NS", "MILD_UP.NS"),
            PairCandidate("STRONG_DOWN.NS", "MILD_DOWN.NS"),
        ],
        lookback_days=200,
        zscore_window=40,
        p_value_max=1.0,  # relaxed: synthetic series rarely cointegrate
        max_half_life_days=10_000,
    )
    sigs = strat.generate_signals(universe_prices, pd.DataFrame())
    _assert_valid_signals(sigs)
    # Pairs ALWAYS produce two legs (or none) per pair.
    pair_names = {s.metadata.get("pair") for s in sigs if s.metadata.get("pair")}
    for name in pair_names:
        legs = [s for s in sigs if s.metadata.get("pair") == name]
        sides = {s.side for s in legs}
        if len(legs) > 0:
            assert sides == {Side.LONG, Side.SHORT}


def test_multi_factor_runs(
    universe_prices: pd.DataFrame,
    fundamentals: pd.DataFrame,
) -> None:
    strat = MultiFactorStrategy(weights=FactorWeights(), top_pct=0.40)
    # Flatten fundamentals to wide-with-ticker-col like the strategy expects.
    latest = fundamentals.tail(5).reset_index()
    sigs = strat.generate_signals(universe_prices, latest)
    _assert_valid_signals(sigs)
    assert any(s.side is Side.LONG for s in sigs)


def test_breakout_finds_uptrending_stocks(
    universe_prices: pd.DataFrame,
) -> None:
    strat = DonchianBreakoutStrategy(adx_min=10.0, sma_filter_period=50)
    sigs = strat.generate_signals(universe_prices, pd.DataFrame())
    _assert_valid_signals(sigs)
    # Don't enforce STRONG_UP must trigger every day — depends on noise.


def test_dual_momentum_risk_off(universe_prices: pd.DataFrame) -> None:
    """When the broad-market series isn't present, dual momentum should
    fall back to the defensive ticker."""
    strat = DualMomentumStrategy(market_index_ticker="MISSING.NS")
    sigs = strat.generate_signals(universe_prices, pd.DataFrame())
    assert len(sigs) == 1
    assert sigs[0].metadata["regime"] == "risk_off"
    assert sigs[0].ticker == "LIQUIDBEES.NS"


def test_sector_rotation_runs(universe_prices: pd.DataFrame) -> None:
    strat = SectorRotationStrategy(
        sector_tickers=list(universe_prices.index.get_level_values("ticker").unique()),
        rel_lookback_days=100,
        abs_filter_period=100,
        top_n=2,
    )
    sigs = strat.generate_signals(universe_prices, pd.DataFrame())
    _assert_valid_signals(sigs)


def test_sentiment_filters_out_negative(
    universe_prices: pd.DataFrame, sentiment_scores: pd.DataFrame
) -> None:
    base = MomentumStrategy(top_n=5)
    strat = SentimentMomentumStrategy(base=base, sentiment_long_min=0.0)
    sigs = strat.generate_signals(universe_prices, pd.DataFrame(), sentiment_scores)
    _assert_valid_signals(sigs)
    # Tickers with negative sentiment should NOT appear as longs.
    longs = [s.ticker for s in sigs if s.side is Side.LONG]
    assert "STRONG_DOWN.NS" not in longs
    assert "MILD_DOWN.NS" not in longs


# ---------------------------------------------------------------------------
# Risk-sizer tests
# ---------------------------------------------------------------------------


def test_position_sizing_returns_positive(
    universe_prices: pd.DataFrame, standard_risk_params: RiskParams
) -> None:
    strat = MomentumStrategy(top_n=1)
    sigs = strat.generate_signals(universe_prices, pd.DataFrame())
    assert sigs, "expected at least one signal on synthetic universe"
    rupees = strat.position_size(
        sigs[0],
        risk=standard_risk_params,
        win_rate_estimate=0.55,
        win_loss_ratio_estimate=1.6,
    )
    assert rupees > 0
    assert rupees <= standard_risk_params.equity * standard_risk_params.max_single_position_pct


# ---------------------------------------------------------------------------
# Exit rules
# ---------------------------------------------------------------------------


def test_atr_stop_triggers_exit() -> None:
    from src.strategies.base import MarketState, Position

    strat = MomentumStrategy()
    pos = Position(
        ticker="X.NS",
        side=Side.LONG,
        quantity=100,
        entry_price=1000.0,
        entry_time=datetime(2024, 1, 1),
        initial_stop=950.0,
        current_stop=960.0,
        strategy_name="momentum",
    )
    market = MarketState(
        timestamp=datetime(2024, 1, 5),
        last_price=950.0,  # below current stop
        atr=10.0,
        realised_vol=0.2,
    )
    decision = strat.exit_rules(pos, market)
    assert decision.should_exit is True
    assert decision.reason is not None and decision.reason.value == "stop_loss"


# ---------------------------------------------------------------------------
# Regime detector
# ---------------------------------------------------------------------------


def test_regime_detector_unknown_for_short_history() -> None:
    short = pd.DataFrame(
        {
            "high": np.arange(10, dtype=float),
            "low": np.arange(10, dtype=float),
            "close": np.arange(10, dtype=float),
        },
        index=pd.date_range("2024-01-01", periods=10, freq="B"),
    )
    det = RegimeDetector()
    out = det.classify(short)
    assert out.regime is Regime.UNKNOWN


def test_regime_detector_classifies_strong_uptrend(
    index_ohlc: pd.DataFrame,
) -> None:
    n = len(index_ohlc)
    close = pd.Series(np.linspace(15000, 25000, n), index=index_ohlc.index)
    df = pd.DataFrame({"high": close * 1.005, "low": close * 0.995, "close": close})
    out = RegimeDetector().classify(df)
    assert out.regime in {Regime.TRENDING_UP_LOW_VOL, Regime.TRENDING_UP_HIGH_VOL}
    assert out.diagnostics.trend_score > 0


def test_regime_detector_classifies_strong_downtrend(
    index_ohlc: pd.DataFrame,
) -> None:
    n = len(index_ohlc)
    close = pd.Series(np.linspace(25000, 15000, n), index=index_ohlc.index)
    df = pd.DataFrame({"high": close * 1.005, "low": close * 0.995, "close": close})
    out = RegimeDetector().classify(df)
    assert out.regime in {
        Regime.TRENDING_DOWN_LOW_VOL,
        Regime.TRENDING_DOWN_HIGH_VOL,
    }
    assert out.diagnostics.trend_score < 0
    # A downtrend regime should NOT allocate large weight to long-only momentum.
    assert out.weights.get("momentum", 0.0) <= 0.10


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_lists_all_strategies() -> None:
    names = available_strategies()
    expected = {
        "momentum",
        "mean_reversion",
        "pairs",
        "multi_factor",
        "breakout",
        "dual_momentum",
        "sector_rotation",
        "sentiment_momentum",
    }
    assert expected.issubset(set(names))


def test_registry_builds_each_strategy() -> None:
    for name in available_strategies():
        strat = build_strategy(name)
        assert isinstance(strat, Strategy)
        assert strat.name == name


def test_registry_unknown_strategy() -> None:
    with pytest.raises(KeyError):
        build_strategy("definitely_not_a_strategy")


# ---------------------------------------------------------------------------
# Selector — the orchestration test
# ---------------------------------------------------------------------------


def test_selector_runs_all_active_strategies(
    universe_prices: pd.DataFrame,
    index_ohlc: pd.DataFrame,
    fundamentals: pd.DataFrame,
    sentiment_scores: pd.DataFrame,
) -> None:
    # Force a deterministic uptrend index to land in TRENDING_LOW_VOL,
    # which gives momentum + multi_factor + breakout positive weight.
    n = len(index_ohlc)
    close = pd.Series(np.linspace(15000, 22000, n), index=index_ohlc.index)
    forced_index = pd.DataFrame({"high": close * 1.005, "low": close * 0.995, "close": close})

    strategies = build_strategies(["momentum", "multi_factor", "breakout", "mean_reversion"])
    selector = StrategySelector(strategies=strategies, top_n_final=3)

    # Multi-factor needs the latest fundamentals snapshot.
    latest = fundamentals.tail(5).reset_index()

    result = selector.select(
        index_ohlc=forced_index,
        prices=universe_prices,
        features=latest,
        sentiment=sentiment_scores,
    )

    assert result.regime_allocation.regime in {
        Regime.TRENDING_UP_LOW_VOL,
        Regime.TRENDING_UP_HIGH_VOL,
    }
    assert len(result.final_signals) <= 3
    _assert_valid_signals(result.final_signals)
    # Each final signal should record contributors (regime-weighted).
    for sig in result.final_signals:
        assert "contributors" in sig.metadata
