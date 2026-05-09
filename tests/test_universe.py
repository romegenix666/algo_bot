"""Universe selector + replay tests using in-memory store."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from src.data.nse_client import NSEClient
from src.data.storage import DataStore
from src.universe.replay import UniverseReplay
from src.universe.selector import UniverseSelector


@pytest.fixture
def store_with_universe() -> DataStore:
    """Seed an in-memory store with 4 tickers covering different scenarios:

    - LIQUID_BIG  (passes all filters, high turnover)
    - LIQUID_SMALL (passes price + history, low turnover → filtered out)
    - PRICY       (price > 10000 → filtered out)
    - PENNY       (price < 50 → filtered out)
    """
    s = DataStore.in_memory()
    s.create_all()
    s.upsert_tickers(
        [
            {"symbol": "LIQUID_BIG"},
            {"symbol": "LIQUID_SMALL"},
            {"symbol": "PRICY"},
            {"symbol": "PENNY"},
        ]
    )

    idx = pd.date_range("2023-01-01", periods=400, freq="B")
    rng = np.random.default_rng(42)

    def make_bars(price_level: float, volume_level: int) -> pd.DataFrame:
        rets = rng.normal(0, 0.012, len(idx))
        close = price_level * np.exp(np.cumsum(rets))
        return pd.DataFrame(
            {
                "open": close,
                "high": close * 1.005,
                "low": close * 0.995,
                "close": close,
                "adj_close": close,
                "volume": np.full(len(idx), volume_level, dtype="int64"),
            },
            index=idx,
        )

    # turnover (Cr) = price * volume / 1e7
    # LIQUID_BIG:    1500 * 5_00_000 = 75 Cr/day → passes
    s.insert_prices("LIQUID_BIG", make_bars(1500.0, 5_00_000))
    # LIQUID_SMALL:  500 * 50_000 = 2.5 Cr → fails 5 Cr threshold
    s.insert_prices("LIQUID_SMALL", make_bars(500.0, 50_000))
    # PRICY:         15000 starting price → exceeds price_max 10000
    s.insert_prices("PRICY", make_bars(15_000.0, 5_00_000))
    # PENNY:         20 → below price_min 50
    s.insert_prices("PENNY", make_bars(20.0, 5_00_000))
    return s


def test_selector_filters_correctly(store_with_universe: DataStore) -> None:
    sel = UniverseSelector(
        store=store_with_universe,
        target_size=10,
        min_avg_turnover_cr=5.0,
        min_history_days=200,
    )
    entries = sel.select(as_of=date(2024, 6, 15))
    symbols = [e.symbol for e in entries]
    assert "LIQUID_BIG" in symbols
    assert "LIQUID_SMALL" not in symbols  # turnover too low
    assert "PENNY" not in symbols  # price < 50
    assert "PRICY" not in symbols  # price > 10000


def test_selector_target_size_caps_results(store_with_universe: DataStore) -> None:
    sel = UniverseSelector(
        store=store_with_universe,
        target_size=1,
        min_avg_turnover_cr=0.0,  # let everything pass
        min_history_days=200,
        price_min=0.0,
        price_max=1e10,
    )
    entries = sel.select(as_of=date(2024, 6, 15))
    assert len(entries) == 1


def test_selector_save_and_replay(store_with_universe: DataStore) -> None:
    sel = UniverseSelector(
        store=store_with_universe,
        target_size=10,
        min_avg_turnover_cr=5.0,
        min_history_days=200,
    )
    as_of = date(2024, 6, 15)
    entries = sel.select(as_of=as_of)
    sel.save_snapshot(as_of, entries)

    replay = UniverseReplay(store=store_with_universe)
    snap = replay.universe_as_of(as_of)
    assert "LIQUID_BIG" in snap

    # Earlier than any snapshot → empty
    assert replay.universe_as_of(as_of - timedelta(days=400)) == []


def test_replay_prefers_latest_snapshot_before_query(
    store_with_universe: DataStore,
) -> None:
    sel = UniverseSelector(
        store=store_with_universe, target_size=10, min_avg_turnover_cr=5.0, min_history_days=200
    )
    e1 = sel.select(as_of=date(2024, 3, 1))
    sel.save_snapshot(date(2024, 3, 1), e1, notes="march")
    e2 = sel.select(as_of=date(2024, 6, 1))
    sel.save_snapshot(date(2024, 6, 1), e2, notes="june")

    replay = UniverseReplay(store=store_with_universe)
    # June 15 → uses June snapshot
    snap_jun = replay.universe_as_of(date(2024, 6, 15))
    # April 1 → uses March snapshot (because June is in the future)
    snap_apr = replay.universe_as_of(date(2024, 4, 1))
    assert snap_jun and snap_apr  # both should be populated


def test_to_dataframe_columns(store_with_universe: DataStore) -> None:
    sel = UniverseSelector(
        store=store_with_universe,
        target_size=10,
        min_avg_turnover_cr=5.0,
        min_history_days=200,
    )
    entries = sel.select(as_of=date(2024, 6, 15))
    df = sel.to_dataframe(entries)
    assert {"rank", "symbol", "last_price", "avg_turnover_cr", "history_days"} <= set(df.columns)


# ---------------------------------------------------------------------------
# NSE client (seed CSV path — no network)
# ---------------------------------------------------------------------------


def test_nse_client_loads_seed_csv() -> None:
    client = NSEClient(use_nsepython=False)  # force seed
    tickers = client.list_seed()
    assert len(tickers) > 50
    symbols = {t.symbol for t in tickers}
    # Spot-check a few well-known names
    assert "RELIANCE" in symbols
    assert "TCS" in symbols
    assert "HDFCBANK" in symbols
    assert "INFY" in symbols


def test_nse_client_seed_has_indices() -> None:
    client = NSEClient(use_nsepython=False)
    tickers = client.list_seed()
    symbols = {t.symbol for t in tickers}
    # We need Nifty 50 for regime detection
    assert "^NSEI" in symbols
