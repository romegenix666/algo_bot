"""Tests for multiple-testing guardrail helpers."""

from __future__ import annotations

import pytest

from src.backtest.multiple_testing import (
    bonferroni_two_sided_normal_critical,
    multiple_testing_footer,
)


def test_bonferroni_increases_with_more_tests() -> None:
    z1 = bonferroni_two_sided_normal_critical(1)
    z8 = bonferroni_two_sided_normal_critical(8)
    assert z8 > z1


def test_bonferroni_m_at_least_one() -> None:
    assert bonferroni_two_sided_normal_critical(0) == bonferroni_two_sided_normal_critical(1)


def test_bonferroni_alpha_invalid() -> None:
    with pytest.raises(ValueError):
        bonferroni_two_sided_normal_critical(5, alpha=0.0)


def test_footer_mentions_n_trials() -> None:
    text = multiple_testing_footer(8)
    assert "8" in text
    assert "DSR" in text
