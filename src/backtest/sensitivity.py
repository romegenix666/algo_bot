"""Parameter-sensitivity sweep.

What it does:
    For each numeric parameter of a strategy, perturb it ±20% (or
    user-defined) and re-run the backtest. Report how Sharpe and max DD
    change.

Why:
    A profitable backtest that *only* works at exactly the chosen
    parameter values is data-snooped — change a number by 20% and it
    falls apart. A *robust* strategy stays roughly profitable across
    a band of parameters.

    Chan §3 rule of thumb: ≥ 80% of the perturbed variants should remain
    "broadly profitable" (Sharpe in the same ballpark, max DD not blowing
    up).

References:
    - Chan (2009), *Quantitative Trading*, Ch. 3 (sensitivity analysis).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from src.backtest.engine import Backtester
from src.strategies.base import Strategy
from src.utils.logging import logger

StrategyFactory = Callable[[dict[str, float]], Strategy]


@dataclass(frozen=True)
class SensitivityRow:
    parameter: str
    value: float
    sharpe: float
    max_drawdown: float
    cagr: float


@dataclass(frozen=True)
class SensitivityReport:
    base_sharpe: float
    base_max_dd: float
    base_cagr: float
    rows: list[SensitivityRow]
    robust_pct: float  # share of variants whose Sharpe is in the same ballpark

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "parameter": r.parameter,
                    "value": r.value,
                    "sharpe": r.sharpe,
                    "max_drawdown": r.max_drawdown,
                    "cagr": r.cagr,
                }
                for r in self.rows
            ]
        )

    def pretty(self) -> str:
        df = self.to_dataframe()
        if df.empty:
            return "(empty sensitivity report)"
        return df.to_string(index=False, float_format=lambda x: f"{x:+.3f}")


def run_sensitivity(
    strategy_factory: StrategyFactory,
    base_params: dict[str, float],
    backtester: Backtester,
    prices: pd.DataFrame,
    index_ohlc: pd.DataFrame | None = None,
    features: pd.DataFrame | None = None,
    sentiment: pd.DataFrame | None = None,
    perturb_pct: float = 0.20,
    relative_sharpe_tol: float = 0.30,  # within 30% of base Sharpe = "robust"
) -> SensitivityReport:
    """Sweep each parameter ±perturb_pct, run a backtest, collect metrics.

    ``strategy_factory`` is called like ``factory(params_dict) -> Strategy``.
    """
    base_strat = strategy_factory(base_params)
    base_result = backtester.run(
        strategy=base_strat,
        prices=prices,
        index_ohlc=index_ohlc,
        features=features,
        sentiment=sentiment,
    )
    if base_result.summary is None:
        raise RuntimeError("base backtest produced no summary")
    base_sharpe = base_result.summary.sharpe_lo
    base_max_dd = base_result.summary.max_drawdown
    base_cagr = base_result.summary.cagr

    rows: list[SensitivityRow] = []
    robust = 0
    n_total = 0

    for name, value in base_params.items():
        # Skip booleans (bools are subclasses of int but flipping them ±20%
        # is meaningless).
        if isinstance(value, bool):
            continue
        for mult in (1.0 - perturb_pct, 1.0 + perturb_pct):
            new_params = dict(base_params)
            new_value = value * mult
            # Round integer-typed params (lookbacks etc.) back to ints.
            if isinstance(value, int):
                new_value = round(new_value)
                if new_value <= 0:
                    continue
            new_params[name] = new_value
            try:
                strat = strategy_factory(new_params)
                result = backtester.run(
                    strategy=strat,
                    prices=prices,
                    index_ohlc=index_ohlc,
                    features=features,
                    sentiment=sentiment,
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Sensitivity {}={} crashed: {}", name, new_value, exc)
                continue

            if result.summary is None:
                continue
            sharpe = result.summary.sharpe_lo
            row = SensitivityRow(
                parameter=name,
                value=float(new_value),
                sharpe=sharpe,
                max_drawdown=result.summary.max_drawdown,
                cagr=result.summary.cagr,
            )
            rows.append(row)
            n_total += 1
            if base_sharpe and abs(sharpe - base_sharpe) <= abs(base_sharpe) * relative_sharpe_tol:
                robust += 1

    robust_pct = robust / n_total if n_total > 0 else 0.0

    return SensitivityReport(
        base_sharpe=base_sharpe,
        base_max_dd=base_max_dd,
        base_cagr=base_cagr,
        rows=rows,
        robust_pct=robust_pct,
    )


__all__ = ["SensitivityReport", "SensitivityRow", "run_sensitivity"]
