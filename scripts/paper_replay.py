"""Replay N trading days end-to-end through paper_trade.

This is the **30-day gauntlet** — the roadmap's Phase 6 acceptance test.
We sequentially process each of the last N trading days with the paper-trade
driver, then summarise:

    - Sharpe (annualised, Lo-corrected)
    - CAGR equivalent
    - Max drawdown
    - Hit rate (per-trade)
    - Profit factor
    - vs. Nifty alpha

Why a separate replay harness?
    The live ``paper_trade`` script is *one bar*. Running it 30 times by
    hand is doable but tedious; this script does it deterministically and
    gives us back the metrics that decide whether we proceed to Phase 8
    (real-money small).

Usage::

    # Replay the last 30 trading days (default)
    python -m scripts.paper_replay

    # Replay a specific window
    python -m scripts.paper_replay --start 2025-12-01 --end 2026-01-15

    # Reset state before replay (clean slate)
    python -m scripts.paper_replay --reset --days 30

The replay is **destructive of paper-trade state by default**: it wipes
``data/paper/state.json`` so each replay starts fresh. Pass ``--no-reset``
to chain replays.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from src.backtest.metrics import (
    cagr,
    drawdown_series,
    hit_rate,
    lo_sharpe_ratio,
    max_drawdown,
    profit_factor,
)
from src.data.storage import DataStore
from src.utils.logging import logger
from src.utils.settings import PROJECT_ROOT

PAPER_STATE_DIR = PROJECT_ROOT / "data" / "paper"
EQUITY_FILE = PAPER_STATE_DIR / "equity_curve.csv"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=30, help="Number of trading days to replay.")
    parser.add_argument("--start", type=str, default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--reset",
        dest="reset",
        action="store_true",
        default=True,
        help="Wipe paper state before replay (default).",
    )
    parser.add_argument(
        "--no-reset",
        dest="reset",
        action="store_false",
        help="Keep existing paper state and append.",
    )
    parser.add_argument("--strategies", nargs="*", default=None)
    args = parser.parse_args()

    # ---- Reset paper state ----
    if args.reset:
        if (PAPER_STATE_DIR / "state.json").exists():
            (PAPER_STATE_DIR / "state.json").unlink()
        if EQUITY_FILE.exists():
            EQUITY_FILE.unlink()
        logger.info("Paper state wiped (reset=true).")

    # ---- Determine the date window ----
    store = DataStore.from_settings()
    panel = store.fetch_prices_panel(
        [t.symbol for t in store.list_tickers(status="active") if not t.symbol.startswith("^")]
    )
    if panel.empty:
        print("ERROR: no price data — run scripts.fetch_history first.")
        return 2
    all_dates = sorted(
        {pd.Timestamp(d).date() for d in panel.index.get_level_values("date").unique()}
    )

    if args.start and args.end:
        start = date.fromisoformat(args.start)
        end = date.fromisoformat(args.end)
        replay_dates = [d for d in all_dates if start <= d <= end]
    else:
        end = date.fromisoformat(args.end) if args.end else all_dates[-1]
        candidates = [d for d in all_dates if d <= end]
        replay_dates = candidates[-args.days :]

    if not replay_dates:
        print("ERROR: no trading days in requested window.")
        return 2

    print("=" * 78)
    print(
        f"  Paper-trading replay  ·  {len(replay_dates)} bars from {replay_dates[0]} to {replay_dates[-1]}"
    )
    print("=" * 78)
    if args.strategies:
        print(f"Active strategies: {' '.join(args.strategies)}")

    # ---- Replay each bar via the paper_trade subprocess ----
    for i, d in enumerate(replay_dates, start=1):
        cmd = [
            sys.executable,
            "-m",
            "scripts.paper_trade",
            "--as-of",
            d.isoformat(),
        ]
        if args.strategies:
            cmd.extend(["--strategies", *args.strategies])
        # First bar: pass --reset-state if user asked, but we already wiped.
        # ``--no-reset`` here because we want continuity inside the replay.
        logger.info("Replay {}/{}: {}", i, len(replay_dates), d)
        subprocess.run(cmd, check=False, cwd=str(PROJECT_ROOT))

    # ---- Compute final metrics ----
    if not EQUITY_FILE.exists():
        print("ERROR: no equity curve written; replay did not produce data.")
        return 2

    eq = pd.read_csv(EQUITY_FILE, parse_dates=["date"]).sort_values("date").set_index("date")
    equity = eq["equity"]
    rets = equity.pct_change().dropna()

    # Build benchmark curve from Nifty over the same window.
    index_df = store.fetch_prices("^NSEI")
    bench_total = None
    if not index_df.empty:
        bench = index_df["close"].reindex(equity.index, method="ffill")
        if not bench.empty and bench.iloc[0] > 0:
            bench_total = float(bench.iloc[-1] / bench.iloc[0] - 1.0)

    # Trade returns proxy: from the fills log, compute per-position close P&L.
    state_path = PAPER_STATE_DIR / "state.json"
    trade_returns = pd.Series(dtype=float)
    if state_path.exists():
        import json

        with state_path.open() as fh:
            state = json.load(fh)
        # Match buy→sell pairs per ticker FIFO to compute per-trade returns
        opens: dict[str, list[tuple[float, int]]] = {}
        rs = []
        for f in state.get("fills") or []:
            t, sd, q, p = f["ticker"], f["side"], int(f["quantity"]), float(f["price"])
            if sd == "buy":
                opens.setdefault(t, []).append((p, q))
            elif sd == "sell" and opens.get(t):
                price_in, _qty_in = opens[t].pop(0)
                rs.append((p - price_in) / price_in)
        if rs:
            trade_returns = pd.Series(rs)

    print("\n--- Replay Performance ---")
    print(f"  Bars             : {len(equity)}")
    print(f"  Final equity     : ₹{float(equity.iloc[-1]):,.0f}")
    print(f"  Total return     : {float(equity.iloc[-1] / equity.iloc[0] - 1.0):+.2%}")
    print(f"  CAGR (annualised): {cagr(equity):+.2%}")
    print(f"  Sharpe (Lo)      : {lo_sharpe_ratio(rets):+.3f}")
    print(f"  Max drawdown     : {max_drawdown(equity):+.2%}")
    print(f"  Worst-bar DD     : {drawdown_series(equity).min():+.2%}")
    if not trade_returns.empty:
        print(f"  # closed trades  : {len(trade_returns)}")
        print(f"  Hit rate         : {hit_rate(trade_returns):+.2%}")
        pf = profit_factor(trade_returns)
        print(f"  Profit factor    : {pf:+.3f}" if not pd.isna(pf) else "  Profit factor    : n/a")
    if bench_total is not None:
        strat_total = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
        print(f"\n  Strategy total   : {strat_total:+.2%}")
        print(f"  Benchmark total  : {bench_total:+.2%}")
        print(f"  Cumul. alpha     : {strat_total - bench_total:+.2%}")

    # ---- Acceptance criteria check (per ROADMAP.md Phase 6) ----
    sharpe = lo_sharpe_ratio(rets)
    dd = max_drawdown(equity)
    sharpe_ok = sharpe >= 1.0 if not pd.isna(sharpe) else False
    dd_ok = dd >= -0.18  # paper-trade ceiling
    print("\n--- Acceptance Criteria (Phase 6 ROADMAP) ---")
    print(f"  Sharpe ≥ 1.0        : {sharpe:+.3f}    {'✓' if sharpe_ok else '✗'}")
    print(f"  Max DD ≥ -18%       : {dd:+.2%}    {'✓' if dd_ok else '✗'}")
    print(
        f"  Verdict             : {'PASS — ready for demo (Phase 7)' if sharpe_ok and dd_ok else 'NOT YET — review and iterate'}"
    )

    return 0


# Stubs for static checkers
_ = (datetime, Path)


if __name__ == "__main__":
    raise SystemExit(main())
