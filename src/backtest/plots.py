"""Matplotlib helpers for equity curves and drawdown (underwater) charts.

Requires ``matplotlib`` (listed in ``requirements.txt``). Use from notebooks
or one-off analysis scripts — the core backtester stays headless.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def equity_and_drawdown_figure(
    equity: pd.Series,
    *,
    title: str = "Equity & drawdown",
    figsize: tuple[float, float] = (10, 6),
) -> Any:
    """Plot equity level (top) and drawdown from running peak (bottom).

    ``equity`` must be sorted ascending by date (index or column used as x).
    """
    import matplotlib.pyplot as plt

    s = equity.sort_index().astype(float)
    peak = s.cummax()
    dd = s / peak - 1.0

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    ax0, ax1 = axes
    s.plot(ax=ax0, color="steelblue", linewidth=1.2)
    ax0.set_ylabel("Equity")
    ax0.set_title(title)
    ax0.grid(True, alpha=0.3)

    dd.plot(ax=ax1, color="firebrick", linewidth=1.0)
    ax1.set_ylabel("Drawdown")
    ax1.axhline(0.0, color="gray", linewidth=0.5)
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


__all__ = ["equity_and_drawdown_figure"]
