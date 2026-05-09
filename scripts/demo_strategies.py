"""Run every strategy on a synthetic universe and print the merged top-5.

Usage
-----
    python -m scripts.demo_strategies

This is the smallest possible end-to-end exercise: synthesise prices for a
handful of fake tickers + a fake market index, build all 8 strategies, run
the regime detector, run the selector, and dump the results.

It exists so that BEFORE we hook up real data (yfinance, Kite), we can
prove the strategy library compiles, types align, and the regime → weights
→ merge pipeline works.
"""

from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd

from src.strategies.regime import RegimeDetector
from src.strategies.registry import available_strategies, build_strategies
from src.strategies.selector import StrategySelector


def make_universe_prices(n_days: int = 400, seed: int = 7) -> pd.DataFrame:
    """Synthetic 5-stock universe with diverse drifts."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-03", periods=n_days, freq="B")
    drifts = {
        "STRONG_UP.NS": 0.0010,
        "MILD_UP.NS": 0.0004,
        "FLAT.NS": 0.0,
        "MILD_DOWN.NS": -0.0003,
        "STRONG_DOWN.NS": -0.0008,
    }
    vols = {
        "STRONG_UP.NS": 0.015,
        "MILD_UP.NS": 0.012,
        "FLAT.NS": 0.010,
        "MILD_DOWN.NS": 0.013,
        "STRONG_DOWN.NS": 0.018,
    }
    frames = []
    for ticker, drift in drifts.items():
        rets = rng.normal(drift, vols[ticker], n_days)
        close = 1000 * np.exp(np.cumsum(rets))
        high = close * (1 + np.abs(rng.normal(0, 0.005, n_days)))
        low = close * (1 - np.abs(rng.normal(0, 0.005, n_days)))
        open_ = close * (1 + rng.normal(0, 0.003, n_days))
        volume = rng.integers(2_00_000, 10_00_000, n_days)
        df = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
            index=dates,
        )
        df["ticker"] = ticker
        frames.append(df)
    return pd.concat(frames).reset_index(names="date").set_index(["date", "ticker"]).sort_index()


def make_market_index(n_days: int = 400, seed: int = 11) -> pd.DataFrame:
    """Deterministic uptrend so we land in TRENDING_LOW_VOL."""
    dates = pd.date_range("2022-01-03", periods=n_days, freq="B")
    rng = np.random.default_rng(seed)
    base = np.linspace(15000, 22000, n_days)
    noise = rng.normal(0, 80, n_days)
    close = pd.Series(base + noise, index=dates)
    return pd.DataFrame({"high": close * 1.005, "low": close * 0.995, "close": close})


def make_fundamentals(prices: pd.DataFrame, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tickers = prices.index.get_level_values("ticker").unique()
    last_date = prices.index.get_level_values("date").max()
    rows = []
    for t in tickers:
        rows.append(
            {
                "date": last_date,
                "ticker": t,
                "pe_ratio": float(rng.uniform(12, 32)),
                "pb_ratio": float(rng.uniform(1, 6)),
                "roe": float(rng.uniform(8, 25)),
                "debt_to_equity": float(rng.uniform(0.1, 1.2)),
            }
        )
    return pd.DataFrame(rows)


def make_sentiment(prices: pd.DataFrame) -> pd.DataFrame:
    tickers = list(prices.index.get_level_values("ticker").unique())
    # Crude: positive sentiment for uptrending, negative for down.
    score_map = {
        "STRONG_UP.NS": 0.5,
        "MILD_UP.NS": 0.2,
        "FLAT.NS": 0.0,
        "MILD_DOWN.NS": -0.3,
        "STRONG_DOWN.NS": -0.6,
    }
    return pd.DataFrame({"ticker": tickers, "score": [score_map.get(t, 0.0) for t in tickers]})


def main() -> None:
    print("=" * 78)
    print(f"  Algo Bot strategy demo  ·  {datetime.now().isoformat(timespec='seconds')}")
    print("=" * 78)

    prices = make_universe_prices()
    index_ohlc = make_market_index()
    fundamentals = make_fundamentals(prices)
    sentiment = make_sentiment(prices)

    available = available_strategies()
    print(f"\nRegistered strategies: {available}")

    strategies = build_strategies(
        [
            "momentum",
            "mean_reversion",
            "multi_factor",
            "breakout",
            "sector_rotation",
            "sentiment_momentum",
        ]
    )

    selector = StrategySelector(
        strategies=strategies,
        regime_detector=RegimeDetector(),
        top_n_final=5,
    )

    result = selector.select(
        index_ohlc=index_ohlc,
        prices=prices,
        features=fundamentals,
        sentiment=sentiment,
    )

    diag = result.regime_allocation.diagnostics
    print("\n--- Regime Diagnostics ---")
    print(f"  Regime:              {result.regime_allocation.regime}")
    print(f"  Annualised vol:      {diag.realised_vol_annual:.3f}")
    print(f"  Trend score (ATRs):  {diag.trend_score:.2f}")
    print(f"  Last index price:    {diag.last_price:.2f}")
    print(f"  200-SMA:             {diag.sma_long:.2f}")

    print("\n--- Active strategy weights ---")
    for name, w in result.regime_allocation.weights.items():
        print(f"  {name:22s} {w:>5.2%}")
    print(f"  {'cash':22s} {result.regime_allocation.cash_weight:>5.2%}")

    print(f"\n--- Raw signals ({len(result.raw_signals)}) ---")
    for ws in result.raw_signals:
        sig = ws.signal
        print(
            f"  [{ws.strategy_name:22s} w={ws.regime_weight:>4.2f}] "
            f"{sig.ticker:18s}  {sig.side.value:>5s}  conv={sig.conviction:.2f}"
        )

    print(f"\n--- Final top-{selector.top_n_final} signals (regime-weighted, deduped) ---")
    if not result.final_signals:
        print("  (no surviving signals)")
    for sig in result.final_signals:
        print(f"  {sig.ticker:18s}  {sig.side.value:>5s}  conv={sig.conviction:.3f}")
        print(f"    contributors: {json.dumps(sig.metadata.get('contributors', {}))}")

    if result.debug:
        print("\n--- Debug ---")
        for k, v in result.debug.items():
            print(f"  {k}: {v}")

    print("\nOK — strategy library is wired end-to-end.\n")


if __name__ == "__main__":
    main()
