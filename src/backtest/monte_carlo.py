"""Monte Carlo bootstrap of backtest results.

Why we need this:
    A single 5-year backtest gives us *one* number for Sharpe and max DD.
    But the future will not replay history — it will be a different
    realisation of similar dynamics. Resampling the trade returns gives
    us a distribution of plausible future paths, and so a confidence
    interval on Sharpe and a worst-case max DD.

How (the simple block-bootstrap):
    1. Take the per-bar realised returns from the backtest.
    2. Resample with replacement, in BLOCKS of size ``block_size`` (to
       preserve any auto-correlation that exists).
    3. Compound into an equity curve. Compute Sharpe / max DD.
    4. Repeat ``n_simulations`` times.
    5. Report mean / median / 5th / 95th percentiles.

Why blocks (not iid)?
    Daily equity returns have mild positive autocorrelation. A pure-iid
    bootstrap underestimates drawdowns. A 5–10 day block bootstrap is
    the standard fix (Politis & Romano 1994).

References:
    - Efron & Tibshirani (1993). *An Introduction to the Bootstrap*.
    - Politis & Romano (1994). *The Stationary Bootstrap*.
    - Bailey, Borwein, López de Prado, Zhu (2014).
      *Pseudo-Mathematics and Financial Charlatanism* — Monte Carlo as
      a sanity check on overfit backtests.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.backtest.metrics import (
    cagr,
    drawdown_series,
    max_drawdown,
    sharpe_ratio,
)


@dataclass(frozen=True)
class MonteCarloReport:
    n_simulations: int
    block_size: int
    sharpe_mean: float
    sharpe_p05: float
    sharpe_p50: float
    sharpe_p95: float
    cagr_mean: float
    cagr_p05: float
    cagr_p95: float
    max_dd_mean: float
    max_dd_p05: float  # most-negative end of the distribution
    max_dd_p50: float
    max_dd_p95: float

    def pretty(self) -> str:
        lines = [
            f"# simulations    : {self.n_simulations}",
            f"Block size       : {self.block_size} days",
            "",
            f"Sharpe   mean    : {self.sharpe_mean:+.3f}",
            f"         p05    : {self.sharpe_p05:+.3f}",
            f"         p50    : {self.sharpe_p50:+.3f}",
            f"         p95    : {self.sharpe_p95:+.3f}",
            f"CAGR     mean    : {self.cagr_mean:+.2%}",
            f"         p05    : {self.cagr_p05:+.2%}",
            f"         p95    : {self.cagr_p95:+.2%}",
            f"Max DD   mean    : {self.max_dd_mean:+.2%}",
            f"         p05 (worst)    : {self.max_dd_p05:+.2%}",
            f"         p50    : {self.max_dd_p50:+.2%}",
            f"         p95    : {self.max_dd_p95:+.2%}",
        ]
        return "\n".join(lines)


def block_bootstrap(
    daily_returns: pd.Series,
    n_simulations: int = 1000,
    block_size: int = 5,
    seed: int | None = 42,
) -> MonteCarloReport:
    """Resample daily returns in blocks; compute distribution over Sharpe + max DD."""
    if len(daily_returns) < block_size * 4:
        raise ValueError("Not enough returns to bootstrap meaningfully")

    rng = np.random.default_rng(seed)
    rets = daily_returns.dropna().to_numpy()
    n = len(rets)
    target_len = n  # match the original length

    n_blocks = int(np.ceil(target_len / block_size))

    sharpes: list[float] = []
    cagrs: list[float] = []
    max_dds: list[float] = []

    for _ in range(n_simulations):
        starts = rng.integers(0, n - block_size + 1, size=n_blocks)
        sample = np.concatenate([rets[s : s + block_size] for s in starts])[:target_len]

        sample_series = pd.Series(sample)
        equity = (1.0 + sample_series).cumprod()

        sharpes.append(sharpe_ratio(sample_series))
        cagrs.append(cagr(equity))
        max_dds.append(max_drawdown(equity))

    s_arr = np.array(sharpes)
    c_arr = np.array(cagrs)
    d_arr = np.array(max_dds)

    return MonteCarloReport(
        n_simulations=n_simulations,
        block_size=block_size,
        sharpe_mean=float(np.nanmean(s_arr)),
        sharpe_p05=float(np.nanpercentile(s_arr, 5)),
        sharpe_p50=float(np.nanpercentile(s_arr, 50)),
        sharpe_p95=float(np.nanpercentile(s_arr, 95)),
        cagr_mean=float(np.nanmean(c_arr)),
        cagr_p05=float(np.nanpercentile(c_arr, 5)),
        cagr_p95=float(np.nanpercentile(c_arr, 95)),
        max_dd_mean=float(np.nanmean(d_arr)),
        max_dd_p05=float(np.nanpercentile(d_arr, 5)),
        max_dd_p50=float(np.nanpercentile(d_arr, 50)),
        max_dd_p95=float(np.nanpercentile(d_arr, 95)),
    )


# Keep this referenced so static checkers don't strip the import.
_ = drawdown_series

__all__ = ["MonteCarloReport", "block_bootstrap"]
