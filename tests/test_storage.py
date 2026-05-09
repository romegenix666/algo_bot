"""Storage layer tests — in-memory SQLite, no network."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from src.data.storage import DataStore


@pytest.fixture
def store() -> DataStore:
    s = DataStore.in_memory()
    s.create_all()
    return s


@pytest.fixture
def synthetic_bars() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=20, freq="B")
    rng = np.random.default_rng(0)
    close = 1000 + np.cumsum(rng.normal(0, 5, 20))
    return pd.DataFrame(
        {
            "open": close + rng.normal(0, 1, 20),
            "high": close + np.abs(rng.normal(0, 2, 20)),
            "low": close - np.abs(rng.normal(0, 2, 20)),
            "close": close,
            "adj_close": close,
            "volume": rng.integers(1_00_000, 10_00_000, 20),
        },
        index=idx,
    )


# ---------------------------------------------------------------------------
# Schema + ticker upsert
# ---------------------------------------------------------------------------


def test_create_all_creates_tables(store: DataStore) -> None:
    # Listing tickers on a fresh DB returns []
    assert store.list_tickers() == []


def test_upsert_tickers_inserts_and_updates(store: DataStore) -> None:
    rows = [
        {"symbol": "RELIANCE", "name": "Reliance Industries", "sector": "Energy"},
        {"symbol": "TCS", "name": "Tata Consultancy Services", "sector": "IT"},
    ]
    n = store.upsert_tickers(rows)
    assert n == 2
    assert {t.symbol for t in store.list_tickers()} == {"RELIANCE", "TCS"}

    # Update an existing row
    store.upsert_tickers([{"symbol": "RELIANCE", "name": "Reliance NEW"}])
    rel = store.get_ticker("RELIANCE")
    assert rel is not None
    assert rel.name == "Reliance NEW"


def test_yf_symbol_auto_derived(store: DataStore) -> None:
    store.upsert_tickers([{"symbol": "RELIANCE"}, {"symbol": "^NSEI"}])
    rel = store.get_ticker("RELIANCE")
    nifty = store.get_ticker("^NSEI")
    assert rel is not None and rel.yf_symbol == "RELIANCE.NS"
    assert nifty is not None and nifty.yf_symbol == "^NSEI"


# ---------------------------------------------------------------------------
# Prices
# ---------------------------------------------------------------------------


def test_insert_and_fetch_prices_roundtrip(store: DataStore, synthetic_bars: pd.DataFrame) -> None:
    store.upsert_tickers([{"symbol": "TEST"}])
    inserted = store.insert_prices("TEST", synthetic_bars)
    assert inserted == 20

    out = store.fetch_prices("TEST")
    assert len(out) == 20
    assert list(out.columns) == ["open", "high", "low", "close", "adj_close", "volume"]
    np.testing.assert_allclose(
        out["close"].to_numpy(), synthetic_bars["close"].to_numpy(), atol=1e-9
    )


def test_insert_prices_replace_overlapping(store: DataStore, synthetic_bars: pd.DataFrame) -> None:
    store.upsert_tickers([{"symbol": "TEST"}])
    store.insert_prices("TEST", synthetic_bars)

    # Modify the same dates and re-insert.
    bumped = synthetic_bars.copy()
    bumped["close"] = bumped["close"] + 100
    n = store.insert_prices("TEST", bumped, replace_overlapping=True)
    assert n == 20

    out = store.fetch_prices("TEST")
    np.testing.assert_allclose(out["close"].to_numpy(), bumped["close"].to_numpy(), atol=1e-9)


def test_insert_prices_unknown_symbol_raises(
    store: DataStore, synthetic_bars: pd.DataFrame
) -> None:
    with pytest.raises(ValueError, match="Unknown ticker"):
        store.insert_prices("MISSING", synthetic_bars)


def test_fetch_prices_panel_returns_multiindex(
    store: DataStore, synthetic_bars: pd.DataFrame
) -> None:
    store.upsert_tickers([{"symbol": "A"}, {"symbol": "B"}])
    store.insert_prices("A", synthetic_bars)
    bumped = synthetic_bars.copy()
    bumped[["close", "open", "high", "low"]] *= 1.05
    store.insert_prices("B", bumped)

    panel = store.fetch_prices_panel(["A", "B"])
    assert isinstance(panel.index, pd.MultiIndex)
    assert set(panel.index.get_level_values("ticker").unique()) == {"A", "B"}
    assert {"open", "high", "low", "close", "adj_close", "volume"} <= set(panel.columns)


def test_fetch_prices_date_range(store: DataStore, synthetic_bars: pd.DataFrame) -> None:
    store.upsert_tickers([{"symbol": "TEST"}])
    store.insert_prices("TEST", synthetic_bars)

    start = synthetic_bars.index[5].date()
    end = synthetic_bars.index[14].date()
    out = store.fetch_prices("TEST", start=start, end=end)
    assert len(out) == 10
    assert out.index.min().date() == start
    assert out.index.max().date() == end


# ---------------------------------------------------------------------------
# Corporate actions
# ---------------------------------------------------------------------------


def test_corporate_actions_roundtrip(store: DataStore) -> None:
    store.upsert_tickers([{"symbol": "TEST"}])
    actions = [
        {"ex_date": date(2024, 1, 15), "action_type": "split", "ratio": 2.0},
        {"ex_date": date(2024, 6, 10), "action_type": "dividend", "dividend_amount": 5.5},
    ]
    n = store.insert_actions("TEST", actions)
    assert n == 2

    out = store.fetch_actions("TEST")
    assert len(out) == 2
    types = {a.action_type for a in out}
    assert types == {"split", "dividend"}


# ---------------------------------------------------------------------------
# Universe snapshots — point-in-time
# ---------------------------------------------------------------------------


def test_universe_snapshot_save_and_replay(store: DataStore) -> None:
    store.upsert_tickers([{"symbol": "A"}, {"symbol": "B"}, {"symbol": "C"}])

    # Snapshot 1
    s1 = date(2024, 1, 1)
    store.save_universe_snapshot(
        s1,
        [
            {"symbol": "A", "rank": 1, "avg_turnover_cr": 100.0},
            {"symbol": "B", "rank": 2, "avg_turnover_cr": 80.0},
        ],
    )
    # Snapshot 2 (later) — drops B, adds C
    s2 = date(2024, 4, 1)
    store.save_universe_snapshot(
        s2,
        [
            {"symbol": "A", "rank": 1, "avg_turnover_cr": 110.0},
            {"symbol": "C", "rank": 2, "avg_turnover_cr": 90.0},
        ],
    )

    # Look up "as of" various dates
    assert store.fetch_universe_as_of(date(2023, 1, 1)) == []  # before any snapshot
    assert store.fetch_universe_as_of(date(2024, 2, 1)) == ["A", "B"]  # uses s1
    assert store.fetch_universe_as_of(date(2024, 4, 1)) == ["A", "C"]  # uses s2
    assert store.fetch_universe_as_of(date(2024, 6, 1)) == ["A", "C"]  # latest


def test_universe_snapshot_replace_replaces(store: DataStore) -> None:
    store.upsert_tickers([{"symbol": "A"}, {"symbol": "B"}])
    s = date(2024, 5, 1)
    store.save_universe_snapshot(s, [{"symbol": "A", "rank": 1}])
    store.save_universe_snapshot(s, [{"symbol": "B", "rank": 1}])
    assert store.fetch_universe_as_of(s) == ["B"]


# ---------------------------------------------------------------------------
# Bars normalisation
# ---------------------------------------------------------------------------


def test_normalise_bars_handles_alt_column_names(store: DataStore) -> None:
    """yfinance returns 'Adj Close' (capital + space). We normalise on insert."""
    store.upsert_tickers([{"symbol": "TEST"}])
    idx = pd.date_range("2024-01-01", periods=3, freq="B")
    raw = pd.DataFrame(
        {
            "Open": [100, 101, 102],
            "High": [103, 104, 105],
            "Low": [99, 100, 101],
            "Close": [102, 103, 104],
            "Adj Close": [102, 103, 104],
            "Volume": [1_00_000, 1_50_000, 2_00_000],
        },
        index=idx,
    )
    n = store.insert_prices("TEST", raw)
    assert n == 3
    out = store.fetch_prices("TEST")
    np.testing.assert_allclose(out["close"].to_numpy(), [102, 103, 104], atol=1e-9)


def test_normalise_bars_fills_adj_close_when_missing(store: DataStore) -> None:
    store.upsert_tickers([{"symbol": "TEST"}])
    idx = pd.date_range("2024-01-01", periods=2, freq="B")
    raw = pd.DataFrame(
        {
            "open": [100, 101],
            "high": [103, 104],
            "low": [99, 100],
            "close": [102, 103],
            "volume": [1_00_000, 2_00_000],
        },
        index=idx,
    )
    store.insert_prices("TEST", raw)
    out = store.fetch_prices("TEST")
    # adj_close defaults to close when not provided
    np.testing.assert_allclose(out["adj_close"].to_numpy(), out["close"].to_numpy())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_session_rollback_on_error(store: DataStore) -> None:
    """A raise inside a session() block should not partially commit."""
    with pytest.raises(RuntimeError), store.session() as sess:
        from src.data.storage import Ticker

        sess.add(Ticker(symbol="X", yf_symbol="X.NS"))
        raise RuntimeError("boom")
    # X should not be in the DB.
    assert store.get_ticker("X") is None


def test_in_memory_isolation() -> None:
    """Two in-memory stores must not share data."""
    a = DataStore.in_memory()
    a.create_all()
    b = DataStore.in_memory()
    b.create_all()
    a.upsert_tickers([{"symbol": "X"}])
    assert a.get_ticker("X") is not None
    assert b.get_ticker("X") is None


def test_history_window_overlap_with_bigger_window(
    store: DataStore, synthetic_bars: pd.DataFrame
) -> None:
    """Inserting partial overlap then full overlap is idempotent."""
    store.upsert_tickers([{"symbol": "TEST"}])
    store.insert_prices("TEST", synthetic_bars.head(10))
    store.insert_prices("TEST", synthetic_bars, replace_overlapping=True)

    out = store.fetch_prices("TEST")
    assert len(out) == 20
    np.testing.assert_allclose(
        out["close"].to_numpy(), synthetic_bars["close"].to_numpy(), atol=1e-9
    )


def test_session_context_returns_session(store: DataStore) -> None:
    with store.session() as sess:
        # Just sanity check it's a usable session
        assert sess.is_active


def test_to_yf_symbol_handles_dotted_already() -> None:
    from src.data.storage import _to_yf_symbol

    assert _to_yf_symbol("RELIANCE") == "RELIANCE.NS"
    assert _to_yf_symbol("RELIANCE.NS") == "RELIANCE.NS"
    assert _to_yf_symbol("^NSEI") == "^NSEI"
    assert _to_yf_symbol("  reliance  ") == "RELIANCE.NS"


def test_to_date_helper() -> None:
    from src.data.storage import _to_date

    assert _to_date(date(2024, 1, 1)) == date(2024, 1, 1)
    assert _to_date("2024-03-15") == date(2024, 3, 15)
    assert _to_date(pd.Timestamp("2024-05-01")) == date(2024, 5, 1)
    with pytest.raises(TypeError):
        _to_date(12345)


def test_history_with_date_filter_excludes_outside(
    store: DataStore, synthetic_bars: pd.DataFrame
) -> None:
    store.upsert_tickers([{"symbol": "TEST"}])
    store.insert_prices("TEST", synthetic_bars)
    early_only = store.fetch_prices("TEST", end=synthetic_bars.index[0].date() - timedelta(days=1))
    assert early_only.empty
