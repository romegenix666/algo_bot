"""Performance metrics — Sharpe, drawdown, deflated Sharpe, hit rate, etc.

Why these specific metrics?

- **CAGR** — easy to compare to "what would Nifty have done?"
- **Sharpe** — risk-adjusted return; what every quant cares about. We
  use the autocorrelation-corrected formula from Lo (2002), not the
  naïve ``√252`` annualisation that overstates Sharpe by ~50% for
  realistic (auto-correlated) return series.
- **Deflated Sharpe** — Bailey & López de Prado (2014). Corrects for
  the fact that we tried N strategies and picked the best — naïve Sharpe
  ratios overstate skill in proportion to N. *Critical* before we
  declare "this strategy is great."
- **Sortino** — Sharpe variant that penalises only downside vol. Often
  prettier for skewed strategies.
- **Calmar** — CAGR / |max DD|. Easy "is the pain worth the gain?" view.
- **Max DD + Duration** — the one number that determines whether you
  abandon the strategy at 2 AM.
- **Hit rate / Profit factor / Avg win / Avg loss** — strategy-DNA tells.
- **Turnover** — proxy for cost-sensitivity.
- **Beta** — exposure to the benchmark.

References:
    - Sharpe (1994). *The Sharpe Ratio*.
    - Lo (2002). *The Statistics of Sharpe Ratios*. Financial Analysts J.
    - Bailey & López de Prado (2014). *The Deflated Sharpe Ratio*.
      J. Portfolio Management.
    - Sortino & van der Meer (1991). *Downside Risk*.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# Returns / equity curve helpers
# ---------------------------------------------------------------------------


def returns_from_equity(equity: pd.Series) -> pd.Series:
    """Daily simple returns from an equity / NAV curve. First row → NaN dropped."""
    return equity.pct_change().dropna()


def equity_from_returns(returns: pd.Series, initial: float = 1.0) -> pd.Series:
    """Cumulative equity from a returns series."""
    return initial * (1.0 + returns).cumprod()


# ---------------------------------------------------------------------------
# Headline metrics
# ---------------------------------------------------------------------------


def cagr(equity: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """Compound Annual Growth Rate."""
    if len(equity) < 2:
        return float("nan")
    n_years = (len(equity) - 1) / periods_per_year
    if n_years <= 0:
        return float("nan")
    final = float(equity.iloc[-1])
    initial = float(equity.iloc[0])
    if initial <= 0 or final <= 0:
        return float("nan")
    return (final / initial) ** (1.0 / n_years) - 1.0


_SD_EPSILON = 1e-12


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate_annual: float = 0.06,  # India 10-yr Gsec proxy
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Annualised Sharpe ratio.

    Note: this is the *naïve* annualisation. For autocorrelated returns
    (most of ours), use ``lo_sharpe_ratio`` for a corrected estimate.
    """
    if len(returns) < 2:
        return float("nan")
    rf_per_period = risk_free_rate_annual / periods_per_year
    excess = returns - rf_per_period
    sd = excess.std(ddof=1)
    # Guard against machine-epsilon noise (e.g. a flat strategy whose
    # equity didn't move): treat tiny sd as zero → NaN, not 1e16.
    if sd <= _SD_EPSILON or np.isnan(sd):
        return float("nan")
    return float((excess.mean() / sd) * np.sqrt(periods_per_year))


def lo_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate_annual: float = 0.06,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    autocorr_lag: int = 1,
) -> float:
    """Autocorrelation-corrected Sharpe ratio (Lo 2002).

    The naïve ``√q`` annualisation assumes IID returns. Real returns
    have autocorrelation; the correct factor is::

        η(q) = q / sqrt(q + 2 * sum_{k=1..q-1} (q - k) * ρ_k)

    For typical equity strategies with mild positive autocorrelation,
    this formula REDUCES the reported Sharpe by 5–30%. That is honest;
    the naïve number is over-optimistic.
    """
    if len(returns) < 2:
        return float("nan")
    rf_per_period = risk_free_rate_annual / periods_per_year
    excess = returns - rf_per_period
    sd = excess.std(ddof=1)
    if sd <= _SD_EPSILON or np.isnan(sd):
        return float("nan")

    sr_period = float(excess.mean() / sd)
    q = periods_per_year

    # Estimate first-`autocorr_lag` autocorrelations from the excess returns.
    rhos: list[float] = []
    for k in range(1, max(1, autocorr_lag) + 1):
        if len(excess) <= k:
            break
        a = excess.iloc[k:].to_numpy()
        b = excess.iloc[:-k].to_numpy()
        if a.std() == 0 or b.std() == 0:
            rhos.append(0.0)
        else:
            rhos.append(float(np.corrcoef(a, b)[0, 1]))

    autocorr_term = sum((q - k) * r for k, r in enumerate(rhos, start=1))
    eta = q / np.sqrt(max(q + 2 * autocorr_term, 1e-9))
    return sr_period * eta


def sortino_ratio(
    returns: pd.Series,
    target: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Sortino: like Sharpe but only penalises downside volatility."""
    if len(returns) < 2:
        return float("nan")
    excess = returns - target / periods_per_year
    downside = excess[excess < 0]
    if downside.empty:
        return float("inf")
    downside_dev = np.sqrt((downside**2).mean())
    if downside_dev <= _SD_EPSILON or np.isnan(downside_dev):
        return float("nan")
    return float((excess.mean() / downside_dev) * np.sqrt(periods_per_year))


def calmar_ratio(equity: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """Calmar = CAGR / |max drawdown|. The "is the pain worth it" ratio."""
    g = cagr(equity, periods_per_year)
    md = abs(max_drawdown(equity))
    if md == 0 or np.isnan(g) or np.isnan(md):
        return float("nan")
    return g / md


# ---------------------------------------------------------------------------
# Drawdown
# ---------------------------------------------------------------------------


def drawdown_series(equity: pd.Series) -> pd.Series:
    """Per-bar drawdown (negative or zero)."""
    if equity.empty:
        return equity
    peak = equity.cummax()
    return equity / peak - 1.0


def max_drawdown(equity: pd.Series) -> float:
    """Maximum (most-negative) drawdown."""
    if equity.empty:
        return float("nan")
    return float(drawdown_series(equity).min())


def max_drawdown_duration_days(equity: pd.Series) -> int:
    """Longest peak-to-recovery span (in *bars*; multiply by 1.4 for ≈ calendar days)."""
    if equity.empty:
        return 0
    dd = drawdown_series(equity)
    longest = 0
    current = 0
    for v in dd.to_numpy():
        if v < 0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


# ---------------------------------------------------------------------------
# Trade-level metrics
# ---------------------------------------------------------------------------


def hit_rate(trade_returns: pd.Series) -> float:
    if trade_returns.empty:
        return float("nan")
    return float((trade_returns > 0).mean())


def profit_factor(trade_returns: pd.Series) -> float:
    if trade_returns.empty:
        return float("nan")
    wins = trade_returns[trade_returns > 0].sum()
    losses = -trade_returns[trade_returns < 0].sum()
    if losses == 0:
        return float("inf") if wins > 0 else float("nan")
    return float(wins / losses)


def avg_win(trade_returns: pd.Series) -> float:
    wins = trade_returns[trade_returns > 0]
    return float(wins.mean()) if not wins.empty else float("nan")


def avg_loss(trade_returns: pd.Series) -> float:
    losses = trade_returns[trade_returns < 0]
    return float(losses.mean()) if not losses.empty else float("nan")


def turnover_annualised(
    weights: pd.DataFrame, periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> float:
    """Average per-period |Δ weight| × periods/yr — proxy for cost sensitivity.

    Inputs is a (date × ticker) weight matrix; absolute weight changes
    summed across tickers per period, then averaged.
    """
    if weights.empty or len(weights) < 2:
        return float("nan")
    delta = weights.diff().abs().sum(axis=1)
    return float(delta.mean() * periods_per_year)


# ---------------------------------------------------------------------------
# Benchmark-relative
# ---------------------------------------------------------------------------


def beta(returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Plain CAPM beta. Aligns on common dates."""
    aligned = pd.concat([returns, benchmark_returns], axis=1, join="inner").dropna()
    if len(aligned) < 5:
        return float("nan")
    a, b = aligned.iloc[:, 0], aligned.iloc[:, 1]
    var_b = b.var(ddof=1)
    if var_b == 0:
        return float("nan")
    return float(a.cov(b) / var_b)


def alpha_annualised(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate_annual: float = 0.06,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Jensen's alpha, annualised."""
    aligned = pd.concat([returns, benchmark_returns], axis=1, join="inner").dropna()
    if len(aligned) < 5:
        return float("nan")
    a, b = aligned.iloc[:, 0], aligned.iloc[:, 1]
    rf = risk_free_rate_annual / periods_per_year
    bt = beta(returns, benchmark_returns)
    if np.isnan(bt):
        return float("nan")
    alpha_per_period = (a - rf).mean() - bt * (b - rf).mean()
    return float(alpha_per_period * periods_per_year)


# ---------------------------------------------------------------------------
# Deflated Sharpe (Bailey & López de Prado 2014)
# ---------------------------------------------------------------------------


def deflated_sharpe_ratio(
    sharpe: float,
    n_returns: int,
    skew: float,
    kurtosis: float,
    n_trials: int,
    sharpe_threshold: float = 0.0,
) -> float:
    """Probability the strategy's Sharpe is genuine, not a false discovery.

    Returns a value in [0, 1] — a *p-value-like* probability.
    DSR > 0.95 means we are 95% confident the strategy isn't a fluke
    of multiple-testing.

    Implementation follows the BLdP paper's equations 7–10:

        SR* = E[max{SR_k}] over n_trials i.i.d. Sharpe ratios with
              variance Var[SR] = (1 + 0.5*SR^2 - skew*SR + (kurt - 3)/4 * SR^2) / (T-1)
        DSR = Φ( (SR_observed - SR*) / sqrt(Var[SR_observed]) )

    Where ``Φ`` is the standard normal CDF.

    Args:
        sharpe: observed Sharpe (annualised).
        n_returns: number of return observations used to compute it (T).
        skew: skewness of the return series.
        kurtosis: kurtosis of the return series (NOT excess; pass 3.0 for normal).
        n_trials: number of strategy variants you tried before picking this one.
        sharpe_threshold: null-hypothesis Sharpe (default 0).
    """
    if n_returns < 2 or n_trials < 1:
        return float("nan")

    # Expected max of N i.i.d. standard normals (BLdP eq. 7).
    # γ ≈ 0.5772 (Euler-Mascheroni)
    gamma = 0.5772156649015329
    e_max = (1 - gamma) * stats.norm.ppf(1 - 1.0 / n_trials) + gamma * stats.norm.ppf(
        1 - 1.0 / (n_trials * np.e)
    )

    # Variance of estimated Sharpe (Mertens 2002 / Lo 2002 generalisation).
    # SR here is the per-period Sharpe → convert from annualised.
    sr_per_period = sharpe / np.sqrt(TRADING_DAYS_PER_YEAR)
    var_sr = (
        1 + 0.5 * sr_per_period**2 - skew * sr_per_period + (kurtosis - 3) / 4.0 * sr_per_period**2
    ) / (n_returns - 1)

    if var_sr <= 0:
        return float("nan")

    # Standardise: how many std-devs above the multi-test threshold are we?
    sr_threshold_per_period = sharpe_threshold / np.sqrt(TRADING_DAYS_PER_YEAR)
    z = (sr_per_period - sr_threshold_per_period - e_max * np.sqrt(var_sr)) / np.sqrt(var_sr)
    return float(stats.norm.cdf(z))


# ---------------------------------------------------------------------------
# Headline summary
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PerformanceSummary:
    """All metrics in one bundle. Pretty-printable; JSON-serialisable."""

    cagr: float
    sharpe: float
    sharpe_lo: float
    sortino: float
    calmar: float
    max_drawdown: float
    max_drawdown_days: int
    hit_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    turnover_ann: float
    beta_to_benchmark: float | None
    alpha_to_benchmark: float | None
    n_obs: int

    def as_dict(self) -> dict[str, float | int | None]:
        return {
            "cagr": self.cagr,
            "sharpe": self.sharpe,
            "sharpe_lo": self.sharpe_lo,
            "sortino": self.sortino,
            "calmar": self.calmar,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_days": self.max_drawdown_days,
            "hit_rate": self.hit_rate,
            "profit_factor": self.profit_factor,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "turnover_ann": self.turnover_ann,
            "beta_to_benchmark": self.beta_to_benchmark,
            "alpha_to_benchmark": self.alpha_to_benchmark,
            "n_obs": self.n_obs,
        }

    def pretty(self) -> str:
        rows = [
            ("CAGR", f"{self.cagr:>+8.2%}"),
            ("Sharpe (naïve)", f"{self.sharpe:>+8.3f}"),
            ("Sharpe (Lo)", f"{self.sharpe_lo:>+8.3f}"),
            ("Sortino", f"{self.sortino:>+8.3f}"),
            ("Calmar", f"{self.calmar:>+8.3f}"),
            ("Max drawdown", f"{self.max_drawdown:>+8.2%}"),
            ("Max DD duration (bars)", f"{self.max_drawdown_days:>8d}"),
            ("Hit rate", f"{self.hit_rate:>+8.2%}"),
            ("Profit factor", f"{self.profit_factor:>+8.3f}"),
            ("Avg win", f"{self.avg_win:>+8.3%}"),
            ("Avg loss", f"{self.avg_loss:>+8.3%}"),
            ("Turnover (ann)", f"{self.turnover_ann:>8.2f}"),
            (
                "Beta",
                f"{self.beta_to_benchmark:>+8.3f}"
                if self.beta_to_benchmark is not None
                else "      n/a",
            ),
            (
                "Alpha (ann)",
                f"{self.alpha_to_benchmark:>+8.2%}"
                if self.alpha_to_benchmark is not None
                else "      n/a",
            ),
            ("# observations", f"{self.n_obs:>8d}"),
        ]
        width = max(len(k) for k, _ in rows)
        return "\n".join(f"  {k:<{width}}  {v}" for k, v in rows)


def summarise(
    equity: pd.Series,
    trade_returns: pd.Series | None = None,
    weights: pd.DataFrame | None = None,
    benchmark_equity: pd.Series | None = None,
    risk_free_rate_annual: float = 0.06,
) -> PerformanceSummary:
    """Compute every metric and bundle it up."""
    rets = returns_from_equity(equity)
    bench_rets = (
        returns_from_equity(benchmark_equity)
        if benchmark_equity is not None and len(benchmark_equity) > 1
        else None
    )

    return PerformanceSummary(
        cagr=cagr(equity),
        sharpe=sharpe_ratio(rets, risk_free_rate_annual),
        sharpe_lo=lo_sharpe_ratio(rets, risk_free_rate_annual),
        sortino=sortino_ratio(rets),
        calmar=calmar_ratio(equity),
        max_drawdown=max_drawdown(equity),
        max_drawdown_days=max_drawdown_duration_days(equity),
        hit_rate=hit_rate(trade_returns) if trade_returns is not None else float("nan"),
        profit_factor=profit_factor(trade_returns) if trade_returns is not None else float("nan"),
        avg_win=avg_win(trade_returns) if trade_returns is not None else float("nan"),
        avg_loss=avg_loss(trade_returns) if trade_returns is not None else float("nan"),
        turnover_ann=turnover_annualised(weights) if weights is not None else float("nan"),
        beta_to_benchmark=beta(rets, bench_rets) if bench_rets is not None else None,
        alpha_to_benchmark=(
            alpha_annualised(rets, bench_rets, risk_free_rate_annual)
            if bench_rets is not None
            else None
        ),
        n_obs=len(equity),
    )


__all__ = [
    "TRADING_DAYS_PER_YEAR",
    "PerformanceSummary",
    "alpha_annualised",
    "avg_loss",
    "avg_win",
    "beta",
    "cagr",
    "calmar_ratio",
    "deflated_sharpe_ratio",
    "drawdown_series",
    "equity_from_returns",
    "hit_rate",
    "lo_sharpe_ratio",
    "max_drawdown",
    "max_drawdown_duration_days",
    "profit_factor",
    "returns_from_equity",
    "sharpe_ratio",
    "sortino_ratio",
    "summarise",
    "turnover_annualised",
]
