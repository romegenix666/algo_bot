"""End-to-end smoke: real NSE data → strategies → top-N picks.

Runs the full pipeline on whatever's in the local SQLite DB:

    1. Read all stored prices.
    2. Build features (currently just close + a placeholder ATR).
    3. Run the regime detector on Nifty 50.
    4. Run all 8 strategies through the selector.
    5. Print the regime-weighted top-N picks.

Prerequisite: ``python -m scripts.fetch_history`` has been run.

This is the ultimate sanity check before Phase 2 (the backtester):
if this works on a tiny universe, the same machinery scales to 500
tickers without code changes.
"""

from __future__ import annotations

import json
from datetime import datetime

import pandas as pd

from src.data.storage import DataStore
from src.strategies.regime import RegimeDetector
from src.strategies.registry import build_strategies
from src.strategies.selector import StrategySelector


def main() -> int:
    print("=" * 78)
    print(f"  Live data pipeline demo  ·  {datetime.now().isoformat(timespec='seconds')}")
    print("=" * 78)

    store = DataStore.from_settings()
    tickers = store.list_tickers(status="active")
    if not tickers:
        print("ERROR: no tickers in DB. Run `python -m scripts.fetch_history` first.")
        return 2

    # We work with whatever has data, plus the index for regime detection.
    index_sym = "^NSEI"
    nifty = store.fetch_prices(index_sym)
    if nifty.empty:
        print(f"ERROR: no data for {index_sym}. Re-run fetch_history including this symbol.")
        return 2
    index_ohlc = nifty[["high", "low", "close"]]

    # Equity universe: every ticker that has any price history (excluding indices).
    equity_syms = [t.symbol for t in tickers if not t.symbol.startswith("^")]
    panel = store.fetch_prices_panel(equity_syms)
    if panel.empty:
        print("ERROR: no equity bars in DB.")
        return 2

    print(
        f"\n Universe: {len(equity_syms)} tickers · "
        f"history span {panel.index.get_level_values('date').min().date()}"
        f" → {panel.index.get_level_values('date').max().date()}"
    )
    print(f"Index proxy ({index_sym}): {len(index_ohlc)} bars")

    strategies = build_strategies(
        ["momentum", "mean_reversion", "multi_factor", "breakout", "sector_rotation"]
    )
    print(f"\nLoaded strategies: {[s.name for s in strategies]}")

    # Multi-factor needs fundamentals; we don't have them yet, so we'll
    # fake a flat snapshot (median values) so it doesn't crash.
    last_date = panel.index.get_level_values("date").max()
    fundamentals_stub = pd.DataFrame(
        [
            {
                "date": last_date,
                "ticker": s,
                "pe_ratio": 20.0,
                "pb_ratio": 3.0,
                "roe": 15.0,
                "debt_to_equity": 0.5,
            }
            for s in equity_syms
        ]
    )

    selector = StrategySelector(
        strategies=strategies,
        regime_detector=RegimeDetector(),
        top_n_final=5,
    )
    result = selector.select(
        index_ohlc=index_ohlc,
        prices=panel,
        features=fundamentals_stub,
        sentiment=None,
    )

    diag = result.regime_allocation.diagnostics
    print("\n--- Regime ---")
    print(f"  Regime               : {result.regime_allocation.regime}")
    print(f"  Annualised vol       : {diag.realised_vol_annual:.3f}")
    print(f"  Trend score (ATRs)   : {diag.trend_score:.2f}")
    print(f"  Last index price     : {diag.last_price:,.2f}")
    print(f"  200-SMA              : {diag.sma_long:,.2f}")

    print("\n--- Active strategy weights ---")
    for name, w in sorted(result.regime_allocation.weights.items(), key=lambda kv: -kv[1]):
        print(f"  {name:<20s} {w:>5.1%}")
    print(f"  {'cash':<20s} {result.regime_allocation.cash_weight:>5.1%}")

    print(f"\n--- Raw signals ({len(result.raw_signals)}) ---")
    for ws in result.raw_signals:
        sig = ws.signal
        print(
            f"  [{ws.strategy_name:<18s} w={ws.regime_weight:>4.2f}] "
            f"{sig.ticker:<14s} {sig.side.value:<5s} conv={sig.conviction:.2f}"
        )

    print(f"\n--- Final top-{selector.top_n_final} (regime-weighted, deduped) ---")
    if not result.final_signals:
        print("  (no signals — universe too small, or every strategy needs more history)")
    for sig in result.final_signals:
        print(f"  {sig.ticker:<14s} {sig.side.value:<5s} conv={sig.conviction:.3f}")
        print(f"    contributors: {json.dumps(sig.metadata.get('contributors', {}))}")

    if result.debug:
        print("\n--- Debug ---")
        for k, v in result.debug.items():
            print(f"  {k}: {v}")

    print("\nOK — live pipeline executed.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
