"""Strategy registry factory tests."""

from __future__ import annotations

import pytest

from src.strategies.base import Strategy
from src.strategies.registry import (
    available_strategies,
    build_active_strategies,
    build_strategies,
    build_strategy,
)


@pytest.mark.parametrize(
    "name",
    [
        "momentum",
        "mean_reversion",
        "pairs",
        "multi_factor",
        "breakout",
        "dual_momentum",
        "sector_rotation",
        "sentiment_momentum",
    ],
)
def test_build_strategy_each_registered_name(name: str) -> None:
    s = build_strategy(name, config={})
    assert isinstance(s, Strategy)
    assert s.name == name


def test_build_strategy_unknown_raises() -> None:
    with pytest.raises(KeyError, match="Unknown strategy"):
        build_strategy("not_a_real_strategy_xyz")


def test_available_strategies_sorted_unique() -> None:
    names = available_strategies()
    assert names == sorted(names)
    assert len(names) == len(set(names))


def test_build_strategies_respects_order() -> None:
    out = build_strategies(["momentum", "pairs"])
    assert [x.name for x in out] == ["momentum", "pairs"]


def test_build_strategy_config_override_momentum_top_n() -> None:
    s = build_strategy("momentum", config={"top_n": 3, "lookback_days": 60, "skip_days": 5})
    assert s.name == "momentum"
    assert s.top_n == 3  # type: ignore[attr-defined]


def test_build_strategy_pairs_custom_candidates() -> None:
    cfg = {
        "candidates": [{"a": "A.NS", "b": "B.NS"}],
        "cointegration_lookback_days": 30,
    }
    s = build_strategy("pairs", config=cfg)
    assert s.name == "pairs"


def test_build_active_strategies_returns_list() -> None:
    active = build_active_strategies()
    assert isinstance(active, list)
    for strat in active:
        assert isinstance(strat, Strategy)
