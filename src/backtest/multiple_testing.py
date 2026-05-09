"""Multiple-testing reminders for strategy comparison runs.

Harvey, Liu & Zhu (2016) show that conventional significance cutoffs are far
too lenient when many factors (or strategies) are tried on the same data. We
already correct ranking with the **deflated Sharpe ratio** (Bailey & López de
Prado, 2014) using ``n_trials =`` number of strategies in a benchmark batch.

This module adds a **simple Bonferroni-style** normal critical value as an extra
qualitative hurdle: if you pretend each strategy is one independent two-sided
test at family-wise error ``alpha``, the marginal critical z is
``Φ^{-1}(1 - α/(2m))``. That is illustrative only — real returns are not IID
and tests are not independent — but it reinforces "do not trust the lucky
winner of eight backtests."

References:
    - Harvey, Liu & Zhu (2016), *... and the Cross-Section of Expected Returns*.
    - Bailey & López de Prado (2014), *The Deflated Sharpe Ratio*.
"""

from __future__ import annotations

from scipy import stats

__all__ = [
    "bonferroni_two_sided_normal_critical",
    "multiple_testing_footer",
]


def bonferroni_two_sided_normal_critical(m: int, alpha: float = 0.05) -> float:
    """Two-sided normal critical value under Bonferroni across ``m`` tests.

    Args:
        m: Number of parallel comparisons (use strategies benchmarked together).
        alpha: Family-wise two-sided error rate (default 5%).
    """
    m = max(1, int(m))
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0,1)")
    return float(stats.norm.ppf(1 - alpha / (2 * m)))


def multiple_testing_footer(n_strategies_tried: int) -> str:
    """Human-readable footer for ``BenchmarkReport.pretty()``."""
    m = max(1, int(n_strategies_tried))
    z = bonferroni_two_sided_normal_critical(m)
    return (
        f"Multiple-testing: {m} strategies in this run; DSR uses n_trials={m}. "
        f"Illustrative Bonferroni 5% two-sided normal hurdle ≈ {z:.2f}σ "
        "(see Harvey et al. 2016; use walk-forward + paper for real decisions)."
    )
