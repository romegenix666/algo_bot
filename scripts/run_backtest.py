"""Run a backtest from the CLI.

Examples::

    # Backtest momentum on whatever's in the local DB:
    python -m scripts.run_backtest --strategy momentum

    # Quarterly rebalance, 2 years of capital required:
    python -m scripts.run_backtest --strategy multi_factor --rebalance Q

    # Walk-forward (rolling 3-yr train / 6-mo test):
    python -m scripts.run_backtest --strategy momentum --walk-forward

    # Look-ahead audit:
    python -m scripts.run_backtest --strategy momentum --audit-lookahead

    # Monte Carlo bootstrap of trade returns:
    python -m scripts.run_backtest --strategy momentum --monte-carlo 500

    # Subset of universe (faster):
    python -m scripts.run_backtest --strategy momentum \\
        --universe RELIANCE TCS HDFCBANK INFY ITC

The script reads from the local SQLite DB; ensure ``fetch_history`` has
been run first.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.backtest.costs import IndianEquityCostModel
from src.backtest.engine import Backtester, walk_forward
from src.backtest.lookahead import audit_strategy
from src.backtest.monte_carlo import block_bootstrap
from src.data.storage import DataStore
from src.strategies.registry import available_strategies, build_strategy
from src.utils.logging import logger
from src.utils.settings import PROJECT_ROOT


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strategy",
        required=True,
        choices=available_strategies(),
        help="Which strategy to backtest.",
    )
    parser.add_argument(
        "--universe",
        nargs="*",
        help="Subset of tickers to use; default = every ticker in DB with data.",
    )
    parser.add_argument(
        "--index-symbol",
        default="^NSEI",
        help="Benchmark index symbol stored in the DB (default Nifty 50).",
    )
    parser.add_argument(
        "--rebalance",
        default="M",
        choices=("D", "W", "M", "Q"),
        help="Rebalance cadence (default monthly).",
    )
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=10_00_000,
        help="₹ notional starting capital (default 10 Lakh).",
    )
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--walk-forward", action="store_true")
    parser.add_argument("--audit-lookahead", action="store_true")
    parser.add_argument(
        "--monte-carlo", type=int, default=0, help="Run Monte Carlo with N simulations (e.g. 500)."
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="If set, dump the equity curve and weights to a CSV.",
    )
    args = parser.parse_args()

    store = DataStore.from_settings()
    tickers = store.list_tickers(status="active")
    if not tickers:
        print("ERROR: no tickers in DB. Run `python -m scripts.fetch_history` first.")
        return 2

    # Universe
    if args.universe:
        sym_list = [s.upper() for s in args.universe]
    else:
        sym_list = [t.symbol for t in tickers if not t.symbol.startswith("^")]

    panel = store.fetch_prices_panel(sym_list)
    if panel.empty:
        print(f"ERROR: no price history for any of {sym_list}")
        return 2

    # Index
    index_df = store.fetch_prices(args.index_symbol)
    if index_df.empty:
        print(f"WARN: no data for benchmark {args.index_symbol}; running without benchmark.")
        index_ohlc = None
    else:
        index_ohlc = index_df[["high", "low", "close"]]

    cost_model = IndianEquityCostModel(slippage_bps=args.slippage_bps)
    backtester = Backtester(
        cost_model=cost_model,
        initial_capital=args.initial_capital,
        rebalance_freq=args.rebalance,
    )

    print("=" * 78)
    print(f"  Backtest  ·  {args.strategy}  ·  {args.rebalance}  ·  ₹{args.initial_capital:,.0f}")
    print("=" * 78)
    n_dates = panel.index.get_level_values("date").nunique()
    n_syms = panel.index.get_level_values("ticker").nunique()
    print(f"\nUniverse: {n_syms} tickers · {n_dates} bars")
    print(
        f"Cost model: brokerage 0.03% (capped ₹20), STT 0.10%, slippage {args.slippage_bps} bps each way"
    )

    # ---- Look-ahead audit (optional) ----
    if args.audit_lookahead:
        print("\n--- Look-ahead audit (truncate-and-rerun) ---")
        report = audit_strategy(
            strategy_factory=lambda _prices: build_strategy(args.strategy),
            backtester=backtester,
            prices=panel,
            index_ohlc=index_ohlc,
        )
        print(report.pretty())
        if report.verdict != "clean":
            logger.warning("Look-ahead leaks detected — DO NOT trust this strategy's backtest.")

    # ---- Walk-forward (optional) ----
    if args.walk_forward:
        print("\n--- Walk-forward (3yr train / 6mo test, step 6mo) ---")
        try:
            wf = walk_forward(
                backtester=backtester,
                strategy_factory=lambda: build_strategy(args.strategy),
                prices=panel,
                index_ohlc=index_ohlc,
                train_years=3,
                test_months=6,
                step_months=6,
            )
            print(f"Folds completed: {len(wf.folds)}")
            print("Stitched out-of-sample summary:")
            print(wf.stitched_summary.pretty())
        except (ValueError, RuntimeError) as exc:
            print(f"Walk-forward unavailable: {exc}")
        # Walk-forward replaces the single-shot backtest output.
        return 0

    # ---- Single-shot backtest ----
    print("\n--- Single-shot backtest ---")
    strat = build_strategy(args.strategy)
    result = backtester.run(
        strategy=strat,
        prices=panel,
        index_ohlc=index_ohlc,
    )

    summary = result.summary
    if summary is not None:
        print(summary.pretty())

    print(f"\nTrades: {len(result.trades)}")
    if result.trades:
        total_cost = sum(t.cost_inr for t in result.trades)
        print(f"Total transaction cost: ₹{total_cost:,.2f}")

    if result.benchmark_equity is not None and not result.benchmark_equity.empty:
        bench_total = float(result.benchmark_equity.iloc[-1] / result.benchmark_equity.iloc[0] - 1)
        strat_total = float(result.equity.iloc[-1] / result.equity.iloc[0] - 1)
        print(f"\nStrategy total return : {strat_total:>+.2%}")
        print(f"Benchmark total return: {bench_total:>+.2%}")
        print(f"Strategy alpha (cumul): {strat_total - bench_total:>+.2%}")

    # ---- Monte Carlo (optional) ----
    if args.monte_carlo > 0:
        print(f"\n--- Monte Carlo (block bootstrap, {args.monte_carlo} sims) ---")
        try:
            mc = block_bootstrap(
                result.daily_returns,
                n_simulations=args.monte_carlo,
                block_size=5,
            )
            print(mc.pretty())
        except ValueError as exc:
            print(f"Monte Carlo skipped: {exc}")

    # ---- Optional CSV dump ----
    if args.out_csv:
        path = args.out_csv if args.out_csv.is_absolute() else PROJECT_ROOT / args.out_csv
        path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(str(path))
        print(f"\nEquity + weights written to {path}")

    # JSON line for piping into another tool.
    if summary is not None:
        json.dump(summary.as_dict(), sys.stderr, default=lambda o: o)
        sys.stderr.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
