from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import pandas as pd

from src.backtest.plots import equity_and_drawdown_figure


def test_equity_and_drawdown_figure_builds() -> None:
    import matplotlib.pyplot as plt

    idx = pd.date_range("2024-01-01", periods=30, freq="D")
    eq = pd.Series([1e6 * (1 + 0.001 * i) for i in range(30)], index=idx)
    fig = equity_and_drawdown_figure(eq, title="test")
    assert fig is not None
    plt.close(fig)
