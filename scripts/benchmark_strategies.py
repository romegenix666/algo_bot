"""Run every registered strategy on the local data and rank them.

Usage::

    # Default: every strategy in registry, monthly rebalance, full DB universe
    python -m scripts.benchmark_strategies

    # Quarterly rebalance, only the 4 cross-sectional strategies:
    python -m scripts.benchmark_strategies \\
        --strategies momentum mean_reversion multi_factor breakout \\
        --rebalance Q

    # Skip the lookahead audit (faster) — only do this for known-clean strategies:
    python -m scripts.benchmark_strategies --no-audit

    # Cap the universe to the top-N by liquidity (faster):
    python -m scripts.benchmark_strategies --limit-universe 50

    # JSON output for downstream tooling:
    python -m scripts.benchmark_strategies --json-out benchmark.json

The output is a ranked table by **deflated Sharpe** — Bailey & López de
Prado's correction for the fact that we tried multiple strategies. The
"recommended" column is True only for strategies that pass ALL bars:
clean lookahead audit, DSR > 0.5, max DD better than -25%, profit
factor > 1.0, and at least 10 trades.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from src.backtest.benchmark import StrategyFactoryNoArgs, run_benchmark
from src.backtest.costs import IndianEquityCostModel
from src.backtest.engine import Backtester
from src.data.storage import DataStore
from src.strategies.base import Strategy
from src.strategies.registry import available_strategies, build_strategy
from src.utils.logging import logger
from src.utils.settings import PROJECT_ROOT


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strategies",
        nargs="*",
        default=None,
        help="Strategy names to benchmark (default = all in registry).",
    )
    parser.add_argument("--rebalance", default="M", choices=("D", "W", "M", "Q"))
    parser.add_argument("--initial-capital", type=float, default=10_00_000)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--index-symbol", default="^NSEI")
    parser.add_argument(
        "--limit-universe",
        type=int,
        default=None,
        help="Cap universe to top-N tickers by sample size (None = all).",
    )
    parser.add_argument(
        "--no-audit",
        action="store_true",
        help="Skip the lookahead audit (faster but less safe).",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Write the full report as JSON to this path.",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=None,
        help="Write the result table as CSV to this path.",
    )
    args = parser.parse_args()

    store = DataStore.from_settings()
    tickers = store.list_tickers(status="active")
    if not tickers:
        print("ERROR: no tickers in DB. Run `python -m scripts.fetch_history` first.")
        return 2

    strategy_names = args.strategies if args.strategies else available_strategies()
    # Some strategies (sentiment, multi_factor, dual_momentum) need extras
    # we don't have yet — flag them rather than silently failing.
    needs_extras = {
        "sentiment_momentum": "sentiment scores (Phase 4)",
        "multi_factor": "fundamentals (P/E, P/B, ROE) — pass via features",
        "dual_momentum": "needs market index + defensive-ETF in DB",
    }

    equity_syms = [t.symbol for t in tickers if not t.symbol.startswith("^")]

    if args.limit_universe is not None:
        # Cheapest filter: keep tickers with the most bars.
        with_bars = []
        for sym in equity_syms:
            df = store.fetch_prices(sym)
            if not df.empty:
                with_bars.append((sym, len(df)))
        with_bars.sort(key=lambda x: -x[1])
        equity_syms = [sym for sym, _n in with_bars[: args.limit_universe]]

    panel = store.fetch_prices_panel(equity_syms)
    if panel.empty:
        print("ERROR: no price history found for the equity universe.")
        return 2

    index_df = store.fetch_prices(args.index_symbol)
    index_ohlc = index_df[["high", "low", "close"]] if not index_df.empty else None

    backtester = Backtester(
        cost_model=IndianEquityCostModel(slippage_bps=args.slippage_bps),
        initial_capital=args.initial_capital,
        rebalance_freq=args.rebalance,
    )

    print("=" * 88)
    print(f"  Strategy Benchmark  ·  {args.rebalance}-rebalance  ·  ₹{args.initial_capital:,.0f}")
    print("=" * 88)
    n_bars = panel.index.get_level_values("date").nunique()
    n_syms = panel.index.get_level_values("ticker").nunique()
    print(f"\nUniverse: {n_syms} tickers · {n_bars} bars")
    if index_ohlc is not None:
        print(f"Benchmark : {args.index_symbol} ({len(index_ohlc)} bars)")
    print(
        f"Cost model: brokerage 0.03% (₹20 cap), STT 0.10%, slippage {args.slippage_bps} bps each way"
    )

    def _strategy_factory(nm: str) -> StrategyFactoryNoArgs:
        def _make() -> Strategy:
            return build_strategy(nm)

        return _make

    factories: dict[str, StrategyFactoryNoArgs] = {}
    for name in strategy_names:
        if name in needs_extras:
            logger.warning(
                "Strategy '{}' typically needs {}; running with stubs.",
                name,
                needs_extras[name],
            )
        factories[name] = _strategy_factory(name)

    print(
        f"\nBenchmarking {len(factories)} strategies (audit={'on' if not args.no_audit else 'off'})…\n"
    )
    report = run_benchmark(
        strategies=factories,
        backtester=backtester,
        prices=panel,
        index_ohlc=index_ohlc,
        audit=not args.no_audit,
    )

    print(report.pretty())

    if args.csv_out:
        path = args.csv_out if args.csv_out.is_absolute() else PROJECT_ROOT / args.csv_out
        path.parent.mkdir(parents=True, exist_ok=True)
        report.to_dataframe().to_csv(path, index=False)
        print(f"\nCSV   → {path}")

    if args.json_out:
        path = args.json_out if args.json_out.is_absolute() else PROJECT_ROOT / args.json_out
        path.parent.mkdir(parents=True, exist_ok=True)
        df = report.to_dataframe()
        # JSON-friendly: replace NaNs with None
        records = json.loads(df.to_json(orient="records"))
        payload = {
            "n_universe": report.n_universe,
            "n_bars": report.n_bars,
            "n_trials_for_dsr": report.n_trials_for_dsr,
            "benchmark_total_return": report.benchmark_total_return,
            "rows": records,
            "recommended": [r.name for r in report.recommended],
        }
        with path.open("w") as fh:
            json.dump(
                payload, fh, indent=2, default=lambda o: float(o) if hasattr(o, "item") else str(o)
            )
        print(f"JSON  → {path}")

    # Quick line for piping into another tool.
    summary_line = {
        "n_strategies": len(report.rows),
        "n_recommended": len(report.recommended),
        "recommended": [r.name for r in report.recommended],
    }
    json.dump(summary_line, sys.stderr, default=str)
    sys.stderr.write("\n")
    return 0


_ = pd  # keep for IDE awareness if we extend later

if __name__ == "__main__":
    raise SystemExit(main())
