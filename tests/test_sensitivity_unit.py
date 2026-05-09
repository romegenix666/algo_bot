"""Sensitivity report helpers (no full sweep)."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from src.backtest.sensitivity import SensitivityReport, SensitivityRow


def test_sensitivity_report_to_dataframe() -> None:
    rows = [
        SensitivityRow("top_n", 4.0, 0.8, -0.1, 0.12),
        SensitivityRow("top_n", 6.0, 0.7, -0.12, 0.10),
    ]
    rep = SensitivityReport(base_sharpe=0.75, base_max_dd=-0.11, base_cagr=0.11, rows=rows, robust_pct=0.5)
    df = rep.to_dataframe()
    assert len(df) == 2
    assert list(df.columns) == ["parameter", "value", "sharpe", "max_drawdown", "cagr"]


def test_sensitivity_report_pretty_non_empty() -> None:
    rows = [SensitivityRow("x", 1.0, 0.5, -0.05, 0.08)]
    rep = SensitivityReport(0.5, -0.05, 0.08, rows, 1.0)
    assert "x" in rep.pretty()


def test_sensitivity_report_pretty_empty_rows() -> None:
    rep = SensitivityReport(0.0, 0.0, 0.0, [], 0.0)
    assert "empty" in rep.pretty().lower()


def test_sensitivity_row_is_frozen_dataclass() -> None:
    row = SensitivityRow("p", 2.0, 1.0, -0.2, 0.05)
    with pytest.raises(FrozenInstanceError):
        row.parameter = "q"  # type: ignore[misc]
