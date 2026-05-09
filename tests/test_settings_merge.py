"""Settings helpers — deep merge and nested get."""

from __future__ import annotations

import pytest

from src.utils.settings import (
    AppEnv,
    BrokerCreds,
    NotificationCreds,
    Settings,
    _deep_merge,
    _load_yaml,
)


def test_deep_merge_nested_dicts() -> None:
    base = {"a": 1, "b": {"x": 1, "y": 2}, "c": {"z": 0}}
    override = {"b": {"y": 99}, "c": 3}
    out = _deep_merge(base, override)
    assert out["a"] == 1
    assert out["b"] == {"x": 1, "y": 99}
    assert out["c"] == 3


def test_deep_merge_empty_override() -> None:
    base = {"k": {"inner": 1}}
    assert _deep_merge(base, {}) == base


def test_deep_merge_empty_base() -> None:
    assert _deep_merge({}, {"x": 1}) == {"x": 1}


def test_settings_get_nested_path() -> None:
    s = Settings(
        config={"risk": {"per_trade_pct": 0.012, "nested": {"leaf": True}}},
        env=AppEnv(),
        broker=BrokerCreds(),
        notifications=NotificationCreds(),
    )
    assert s.get("risk", "per_trade_pct") == pytest.approx(0.012)
    assert s.get("risk", "nested", "leaf") is True


def test_settings_get_missing_returns_default() -> None:
    s = Settings(config={}, env=AppEnv(), broker=BrokerCreds(), notifications=NotificationCreds())
    assert s.get("nope", default=123) == 123
    assert s.get("a", "b", "c", default=None) is None


def test_settings_mode_property() -> None:
    s = Settings(config={}, env=AppEnv(), broker=BrokerCreds(), notifications=NotificationCreds())
    assert s.mode == s.env.mode


def test_load_yaml_missing_file_returns_empty() -> None:
    assert _load_yaml("__file_that_does_not_exist_12345__") == {}
