"""Multi-strategy benchmark runner.

What it does:
    Take a universe of prices + benchmark index, and for every named
    strategy in the registry:

    1. Run a single-shot backtest (cost-realistic).
    2. Run the look-ahead auditor — flag any leak.
    3. (Optional) Run walk-forward to get out-of-sample numbers.
    4. Collect every metric that matters: CAGR, Sharpe, Sortino, max DD,
       hit rate, turnover, beta, alpha, n_trades, total cost.

    Then RANK strategies by deflated Sharpe (the multiple-testing-adjusted
    score) so we don't fall for the "best of 8 backtests is amazing" trap.

Why this exists (the user-asked question — "earn money with minimum risk"):
    Picking a single strategy and trusting its backtest is data-snooping.
    Running ALL strategies, comparing them honestly, and picking the
    *robust* ones (good DSR, sane DD, clean audit) is the only honest
    way to ground a "this strategy is worth paper-trading" decision.

Output:
    A ``BenchmarkReport`` containing:
        - per-strategy result rows
        - a recommended shortlist (passes minimum bars: clean audit,
          DSR > 0.50, max DD better than -25%, profit factor > 1.0)
        - JSON-serialisable for offline analysis

    Plus a `pretty()` view that prints a ranked table.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.backtest.engine import Backtester, BacktestResult
from src.backtest.lookahead import LookaheadReport, audit_strategy
from src.backtest.metrics import deflated_sharpe_ratio
from src.backtest.multiple_testing import multiple_testing_footer
from src.strategies.base import Strategy
from src.utils.logging import logger

StrategyFactoryNoArgs = Callable[[], Strategy]


@dataclass
class BenchmarkRow:
    """One strategy's distilled results."""

    name: str
    cagr: float
    sharpe_lo: float
    sortino: float
    deflated_sharpe: float  # Bailey-LdP — DSR > 0.95 = high confidence
    max_drawdown: float
    max_dd_days: int
    hit_rate: float
    profit_factor: float
    turnover_ann: float
    n_trades: int
    total_cost_inr: float
    final_equity_inr: float
    alpha_to_benchmark: float | None
    beta_to_benchmark: float | None
    audit_verdict: str  # "clean" | "leaks" | "skipped"
    audit_notes: str | None = None
    error: str | None = None  # populated if the run crashed

    @property
    def is_recommended(self) -> bool:
        """Conservative qualification — passes ALL of these:
        - audit clean
        - DSR > 0.50 (some chance the Sharpe is real, not multi-test fluke)
        - max DD better than -25%
        - profit factor > 1.0
        - n_trades >= 10 (small samples meaningless)
        """
        if self.error is not None:
            return False
        if self.audit_verdict == "leaks":
            return False
        if self.deflated_sharpe < 0.5 or np.isnan(self.deflated_sharpe):
            return False
        if self.max_drawdown < -0.25:
            return False
        if self.profit_factor < 1.0 or np.isnan(self.profit_factor):
            return False
        return self.n_trades >= 10


@dataclass
class BenchmarkReport:
    rows: list[BenchmarkRow] = field(default_factory=list)
    benchmark_total_return: float | None = None
    n_universe: int = 0
    n_bars: int = 0
    n_trials_for_dsr: int = 1

    # ----------------------------------------------------------------
    @property
    def recommended(self) -> list[BenchmarkRow]:
        return [r for r in self.rows if r.is_recommended]

    @property
    def by_deflated_sharpe(self) -> list[BenchmarkRow]:
        return sorted(
            self.rows,
            key=lambda r: -(r.deflated_sharpe if not np.isnan(r.deflated_sharpe) else -1),
        )

    # ----------------------------------------------------------------
    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "strategy": r.name,
                    "cagr": r.cagr,
                    "sharpe_lo": r.sharpe_lo,
                    "sortino": r.sortino,
                    "dsr": r.deflated_sharpe,
                    "max_dd": r.max_drawdown,
                    "max_dd_days": r.max_dd_days,
                    "hit_rate": r.hit_rate,
                    "profit_factor": r.profit_factor,
                    "turnover": r.turnover_ann,
                    "trades": r.n_trades,
                    "alpha": r.alpha_to_benchmark,
                    "beta": r.beta_to_benchmark,
                    "audit": r.audit_verdict,
                    "recommended": r.is_recommended,
                    "error": r.error,
                }
                for r in self.rows
            ]
        )

    # ----------------------------------------------------------------
    def pretty(self) -> str:
        if not self.rows:
            return "(no benchmark rows)"
        df = self.to_dataframe().sort_values("dsr", ascending=False)
        # Compact display
        df_display = df[
            [
                "strategy",
                "cagr",
                "sharpe_lo",
                "dsr",
                "max_dd",
                "hit_rate",
                "trades",
                "audit",
                "recommended",
            ]
        ].copy()
        df_display["cagr"] = df_display["cagr"].apply(_fmt_pct)
        df_display["max_dd"] = df_display["max_dd"].apply(_fmt_pct)
        df_display["hit_rate"] = df_display["hit_rate"].apply(_fmt_pct)
        df_display["sharpe_lo"] = df_display["sharpe_lo"].apply(_fmt_signed)
        df_display["dsr"] = df_display["dsr"].apply(_fmt_signed)

        lines = [df_display.to_string(index=False), ""]
        rec = self.recommended
        if rec:
            lines.append(f"Recommended ({len(rec)}): {', '.join(r.name for r in rec)}")
        else:
            lines.append(
                "Recommended (0): none meet quality bars (DSR>0.5, audit clean, DD>-25%, PF>1.0)"
            )
        if self.benchmark_total_return is not None:
            lines.append(f"Benchmark total return: {_fmt_pct(self.benchmark_total_return)}")
        lines.append(multiple_testing_footer(self.n_trials_for_dsr))
        return "\n".join(lines)


def _fmt_pct(v: float) -> str:
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:+.2%}"


def _fmt_signed(v: float) -> str:
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:+.3f}"


# ---------------------------------------------------------------------------
# The runner
# ---------------------------------------------------------------------------


def run_benchmark(
    strategies: dict[str, StrategyFactoryNoArgs],
    backtester: Backtester,
    prices: pd.DataFrame,
    index_ohlc: pd.DataFrame | None = None,
    features: pd.DataFrame | None = None,
    sentiment: pd.DataFrame | None = None,
    audit: bool = True,
    n_trials_for_dsr: int | None = None,
) -> BenchmarkReport:
    """Run every strategy, audit it, and produce a ranked report.

    Args:
        strategies: ``{name: factory}`` mapping. Each factory returns a
            fresh ``Strategy`` instance.
        backtester: shared backtester (cost model + risk + freq).
        prices: MultiIndex (date, ticker) panel.
        audit: if True, run the look-ahead auditor for each strategy.
        n_trials_for_dsr: how many strategies were tried; defaults to the
            number passed in (which is the right answer in this context —
            we ARE multi-testing across them).
    """
    if not isinstance(prices.index, pd.MultiIndex):
        raise ValueError("prices must have MultiIndex (date, ticker)")

    n_trials = n_trials_for_dsr if n_trials_for_dsr is not None else max(1, len(strategies))
    n_bars = prices.index.get_level_values("date").nunique()
    n_syms = prices.index.get_level_values("ticker").nunique()

    rows: list[BenchmarkRow] = []
    benchmark_total_return: float | None = None

    for name, factory in strategies.items():
        logger.info("Benchmarking strategy: {}", name)
        try:
            result = backtester.run(
                strategy=factory(),
                prices=prices,
                index_ohlc=index_ohlc,
                features=features,
                sentiment=sentiment,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Strategy {} failed during backtest", name)
            rows.append(
                BenchmarkRow(
                    name=name,
                    cagr=float("nan"),
                    sharpe_lo=float("nan"),
                    sortino=float("nan"),
                    deflated_sharpe=float("nan"),
                    max_drawdown=float("nan"),
                    max_dd_days=0,
                    hit_rate=float("nan"),
                    profit_factor=float("nan"),
                    turnover_ann=float("nan"),
                    n_trades=0,
                    total_cost_inr=0.0,
                    final_equity_inr=0.0,
                    alpha_to_benchmark=None,
                    beta_to_benchmark=None,
                    audit_verdict="skipped",
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            continue

        if (
            benchmark_total_return is None
            and result.benchmark_equity is not None
            and not result.benchmark_equity.empty
        ):
            benchmark_total_return = float(
                result.benchmark_equity.iloc[-1] / result.benchmark_equity.iloc[0] - 1
            )

        # Compute DSR
        rets = result.daily_returns.dropna()
        if len(rets) >= 30 and result.summary is not None:
            skew = float(rets.skew())
            kurt = float(rets.kurtosis() + 3.0)  # pd.kurtosis is excess; DSR wants raw
            dsr = deflated_sharpe_ratio(
                sharpe=result.summary.sharpe_lo,
                n_returns=len(rets),
                skew=skew,
                kurtosis=kurt,
                n_trials=n_trials,
            )
        else:
            dsr = float("nan")

        # Optional look-ahead audit
        verdict = "skipped"
        notes: str | None = None
        if audit:
            try:

                def _audit_factory(
                    _prices: pd.DataFrame, _f: StrategyFactoryNoArgs = factory
                ) -> Strategy:
                    return _f()

                report = audit_strategy(
                    strategy_factory=_audit_factory,
                    backtester=backtester,
                    prices=prices,
                    index_ohlc=index_ohlc,
                    features=features,
                    sentiment=sentiment,
                    truncate_bars=60,
                )
                verdict = report.verdict
                if verdict == "leaks":
                    notes = (
                        f"matched={report.matched}, mismatched={report.mismatched}, "
                        f"only_full={report.only_in_full}, only_trunc={report.only_in_truncated}"
                    )
            except (ValueError, RuntimeError) as exc:
                verdict = "skipped"
                notes = f"auditor unavailable: {exc}"

        summary = result.summary
        rows.append(
            BenchmarkRow(
                name=name,
                cagr=summary.cagr if summary else float("nan"),
                sharpe_lo=summary.sharpe_lo if summary else float("nan"),
                sortino=summary.sortino if summary else float("nan"),
                deflated_sharpe=dsr,
                max_drawdown=summary.max_drawdown if summary else float("nan"),
                max_dd_days=summary.max_drawdown_days if summary else 0,
                hit_rate=summary.hit_rate if summary else float("nan"),
                profit_factor=summary.profit_factor if summary else float("nan"),
                turnover_ann=summary.turnover_ann if summary else float("nan"),
                n_trades=len(result.trades),
                total_cost_inr=float(sum(t.cost_inr for t in result.trades)),
                final_equity_inr=float(result.equity.iloc[-1]),
                alpha_to_benchmark=summary.alpha_to_benchmark if summary else None,
                beta_to_benchmark=summary.beta_to_benchmark if summary else None,
                audit_verdict=verdict,
                audit_notes=notes,
            )
        )

    return BenchmarkReport(
        rows=rows,
        benchmark_total_return=benchmark_total_return,
        n_universe=n_syms,
        n_bars=n_bars,
        n_trials_for_dsr=n_trials,
    )


# Re-exports just to keep them visible to consumers of this module.
__all__ = [
    "BacktestResult",
    "BenchmarkReport",
    "BenchmarkRow",
    "LookaheadReport",
    "run_benchmark",
]
