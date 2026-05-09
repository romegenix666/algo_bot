"""UniverseReplay delegates to DataStore."""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

from src.universe.replay import UniverseReplay


def test_universe_as_of_delegates_to_store() -> None:
    store = MagicMock()
    store.fetch_universe_as_of.return_value = ["AAA.NS", "BBB.NS"]
    replay = UniverseReplay(store=store)
    out = replay.universe_as_of(date(2025, 6, 1))
    assert out == ["AAA.NS", "BBB.NS"]
    store.fetch_universe_as_of.assert_called_once_with(date(2025, 6, 1))


def test_universe_per_rebalance_bulk() -> None:
    store = MagicMock()
    store.fetch_universe_as_of.side_effect = lambda d: [f"SYM-{d.isoformat()}"]
    replay = UniverseReplay(store=store)
    dates = [date(2025, 1, 2), date(2025, 2, 1)]
    m = replay.universe_per_rebalance(dates)
    assert len(m) == 2
    assert m[dates[0]] == ["SYM-2025-01-02"]


def test_to_dataframe_includes_rank() -> None:
    store = MagicMock()
    store.fetch_universe_as_of.return_value = ["Z.NS", "Y.NS"]
    replay = UniverseReplay(store=store)
    df = replay.to_dataframe(date(2025, 1, 1))
    assert list(df.columns) == ["symbol", "rank"]
    assert df["rank"].tolist() == [1, 2]
    assert df["symbol"].tolist() == ["Z.NS", "Y.NS"]
