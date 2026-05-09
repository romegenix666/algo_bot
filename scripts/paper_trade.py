"""Paper-trade driver — run all strategies for one bar through the full pipeline.

What it does (one invocation = one bar):

    1. Loads the universe from the local DB.
    2. Builds the regime detector + active strategies + selector.
    3. Runs every strategy → ranked top-N signals.
    4. Routes each signal through:
        Risk Manager (gate) → Paper Broker → Portfolio book.
    5. Marks the portfolio to market with today's closes.
    6. Checks ATR-based stop-loss on every open position; exits if hit.
    7. Persists the equity curve to disk.
    8. Prints a daily summary (positions + P&L + breaker state).

Why a CLI not a daemon (yet):
    Cron-driven daily invocation is dead simple, robust, and easy to debug.
    The daemon-mode upgrade is Phase 5 polish — for now this script is the
    intended way to run the bot every evening after market close.

Usage::

    # Default — uses today's date and full universe in DB.
    python -m scripts.paper_trade

    # Backfill: replay one specific date (useful after fetching new data).
    python -m scripts.paper_trade --as-of 2026-04-30

    # Manual kill switch (no new entries; existing positions exited at next mark):
    python -m scripts.paper_trade --kill-switch

    # Reset breaker after a manual halt (only operator-initiated):
    python -m scripts.paper_trade --reset-breaker

State persistence:
    Paper mode: ``data/paper/state.json`` (+ ``equity_curve.csv``).
    Live Kite mode: ``data/live/state.json`` (same shape; portfolio book is
    re-synced from Kite each run to avoid drift).

    Set ``ALGO_MODE=live`` and Zerodha keys in ``.env``, or pass ``--broker kite``.
    Use ``--dry-run`` with Kite to log orders without submitting.

"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field, fields
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd

from src.backtest.costs import IndianEquityCostModel
from src.data.storage import DataStore
from src.monitor.telegram import TelegramNotifier
from src.orders.base import Broker, Fill
from src.orders.dry_run import DryRunBroker
from src.orders.kite import KiteBroker, KiteUnavailableError
from src.orders.live_sync import sync_portfolio_from_kite
from src.orders.paper import PaperBroker
from src.orders.router import OrderRouter
from src.risk.circuit_breaker import BreakerState, circuit_breaker_from_settings
from src.risk.manager import RiskManager, risk_limits_from_settings
from src.risk.portfolio import Portfolio
from src.sentiment.storage import latest_per_ticker_sentiment
from src.strategies.base import Side
from src.strategies.regime import RegimeDetector
from src.strategies.registry import build_active_strategies, build_strategies
from src.strategies.selector import StrategySelector
from src.utils.logging import logger
from src.utils.settings import PROJECT_ROOT, settings


def _resolve_paths(*, live: bool) -> tuple[Path, Path]:
    base = PROJECT_ROOT / "data" / ("live" if live else "paper")
    return base / "state.json", base / "equity_curve.csv"


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


@dataclass
class PaperState:
    """Persistent state round-tripped across runs.

    We re-build the in-memory ``Portfolio`` and ``PaperBroker`` from this
    on every run (each script invocation = one bar). What we save:
        - cash (after costs)
        - the *full* fills history (used to reconstruct open positions)
        - daily equity curve
        - circuit-breaker latch + halt reason
        - high watermark for drawdown computation
    """

    initial_capital: float = 10_00_000.0
    cash: float = 10_00_000.0
    equity_curve: dict[str, float] = field(default_factory=dict)  # ISO date → equity
    fills: list[dict] = field(default_factory=list)
    breaker_halted: bool = False
    breaker_halt_reason: str | None = None
    consecutive_stops: dict[str, int] = field(default_factory=dict)
    strategy_disabled_until: dict[str, str] = field(default_factory=dict)  # strategy → ISO end date
    breaker_pause_date: str | None = None  # same-day PAUSE_DAY restore (ISO date)
    high_watermark: float = 0.0
    high_watermark_date: str | None = None
    last_run: str | None = None  # ISO datetime

    @classmethod
    def load(cls, state_file: Path) -> PaperState:
        if not state_file.exists():
            return cls()
        with state_file.open() as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            return cls()
        allowed = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in allowed})

    def save(self, state_file: Path) -> None:
        state_file.parent.mkdir(parents=True, exist_ok=True)
        with state_file.open("w") as fh:
            json.dump(asdict(self), fh, indent=2, default=str)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--as-of",
        type=str,
        default=None,
        help="Bar date to process (YYYY-MM-DD). Defaults to the latest date in DB.",
    )
    parser.add_argument("--initial-capital", type=float, default=10_00_000)
    parser.add_argument(
        "--strategies",
        nargs="*",
        default=None,
        help="Override active strategies; default = config/default.yaml.",
    )
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument(
        "--kill-switch",
        action="store_true",
        help="Manual hard stop: close all positions, halt new entries.",
    )
    parser.add_argument(
        "--reset-breaker",
        action="store_true",
        help="Operator-only: clear a HALTED state after manual review.",
    )
    parser.add_argument(
        "--reset-state",
        action="store_true",
        help="Wipe persisted state for the selected mode (paper vs live). DESTRUCTIVE.",
    )
    parser.add_argument(
        "--broker",
        choices=("auto", "paper", "kite"),
        default="auto",
        help="Execution backend: auto uses Kite when ALGO_MODE=live, else paper.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="With Kite: full pipeline + sync, but orders are logged not sent.",
    )
    args = parser.parse_args()

    use_kite = args.broker == "kite" or (
        args.broker == "auto" and str(settings.mode).lower() == "live"
    )
    state_file, equity_file = _resolve_paths(live=use_kite)

    if args.reset_state:
        if state_file.exists():
            state_file.unlink()
        if equity_file.exists():
            equity_file.unlink()
        print(f"State wiped ({state_file.parent.name}).")
        return 0

    state = PaperState.load(state_file)
    state.initial_capital = state.initial_capital or args.initial_capital

    # ---- Wire up the bot ----
    store = DataStore.from_settings()
    tickers = store.list_tickers(status="active")
    if not tickers:
        print("ERROR: no tickers in DB. Run scripts.fetch_history first.")
        return 2

    sector_lookup = {t.symbol.upper(): t.sector for t in tickers}

    # ---- Price panel + bar date (needed before live Kite sync / marks) ----
    panel = store.fetch_prices_panel([t.symbol for t in tickers if not t.symbol.startswith("^")])
    index_df = store.fetch_prices("^NSEI")
    if panel.empty:
        print("ERROR: no equity price history in DB.")
        return 2

    all_dates = sorted(set(panel.index.get_level_values("date")))
    as_of = pd.Timestamp(args.as_of) if args.as_of else pd.Timestamp(all_dates[-1])
    if as_of not in all_dates:
        print(f"ERROR: no data for {as_of.date()} in DB.")
        return 2

    today: date = as_of.date()
    try:
        _di = all_dates.index(as_of)
    except ValueError:
        _di = -1
    prev_asof: pd.Timestamp | None = all_dates[_di - 1] if _di > 0 else None

    last_marks: dict[str, float] = {}
    today_panel = panel.xs(as_of, level="date")
    for ticker in today_panel.index:
        last_marks[str(ticker)] = float(today_panel.loc[ticker, "close"])
    if not index_df.empty and as_of in index_df.index:
        last_marks["^NSEI"] = float(index_df.loc[as_of, "close"])

    # ---- Portfolio + risk ----
    cash_start = float(state.cash or state.initial_capital) if not use_kite else 0.0
    portfolio = Portfolio(
        cash_inr=cash_start,
        initial_equity_inr=float(state.initial_capital),
        sector_lookup=sector_lookup,
    )
    if state.high_watermark > 0:
        portfolio.high_watermark = state.high_watermark
        if state.high_watermark_date:
            portfolio.high_watermark_date = date.fromisoformat(state.high_watermark_date)

    risk = RiskManager(limits=risk_limits_from_settings(), breaker=circuit_breaker_from_settings())
    risk.breaker.consecutive_stops.clear()
    for k, v in state.consecutive_stops.items():
        risk.breaker.consecutive_stops[str(k)] = int(v)
    risk.breaker.strategy_disabled_until.clear()
    for name, iso in state.strategy_disabled_until.items():
        try:
            risk.breaker.strategy_disabled_until[str(name)] = date.fromisoformat(str(iso))
        except ValueError:
            continue
    if (
        not state.breaker_halted
        and state.breaker_pause_date
        and state.breaker_pause_date == today.isoformat()
    ):
        try:
            risk.breaker.state = BreakerState.PAUSE_DAY
            risk.breaker.last_pause_date = date.fromisoformat(state.breaker_pause_date)
        except ValueError:
            pass
    if state.breaker_halted:
        risk.breaker.manual_kill = True
        risk.breaker.halt_reason = state.breaker_halt_reason
    if args.reset_breaker:
        risk.breaker.reset()
        state.breaker_halted = False
        state.breaker_halt_reason = None

    kite_for_sync: KiteBroker | None = None
    broker: Broker
    if use_kite:
        try:
            kite_for_sync = KiteBroker.from_settings()
        except KiteUnavailableError as exc:
            print(f"ERROR: Kite unavailable: {exc}")
            return 2
        if args.dry_run:
            logger.warning("Kite DRY-RUN — orders will not be sent.")
            broker = DryRunBroker(inner=kite_for_sync)
        else:
            logger.warning("Kite LIVE — real orders will be placed when signals pass risk.")
            broker = kite_for_sync
    else:
        orders_cfg = settings.get("orders", default={}) or {}
        circuit_pct = float(orders_cfg.get("paper_circuit_limit_pct", 0.0) or 0.0)
        broker = PaperBroker(
            cost_model=IndianEquityCostModel(slippage_bps=5.0),
            initial_cash=cash_start,
            circuit_limit_pct=circuit_pct,
        )
        if circuit_pct > 0 and prev_asof is not None:
            prev_panel = panel.xs(prev_asof, level="date")
            for tkr in prev_panel.index:
                broker.set_circuit_reference(str(tkr), float(prev_panel.loc[tkr, "close"]))

    if not use_kite:
        for fdict in state.fills:
            try:
                ts = datetime.fromisoformat(fdict["timestamp"])
            except Exception:
                ts = datetime.now(UTC)
            fill = Fill(
                timestamp=ts,
                client_order_id=fdict.get("client_order_id", "replay"),
                broker_order_id=fdict.get("broker_order_id", "replay"),
                ticker=fdict["ticker"],
                side=fdict["side"],
                quantity=int(fdict["quantity"]),
                price=float(fdict["price"]),
                cost_inr=float(fdict.get("cost_inr", 0.0)),
                strategy_name=fdict.get("strategy_name"),
                stop_price=fdict.get("stop_price"),
            )
            portfolio.apply_fill(fill)
    elif kite_for_sync is not None:
        sync_portfolio_from_kite(
            portfolio,
            kite_for_sync,
            state_fills=state.fills,
        )
        portfolio.mark_to_market(last_marks, as_of=today)

    for t, px in last_marks.items():
        broker.set_mark(t, px)

    # Restore the equity curve so the breaker has history.
    for d_str, eq in state.equity_curve.items():
        try:
            portfolio.equity_curve[date.fromisoformat(d_str)] = float(eq)
        except Exception:
            continue

    router = OrderRouter(broker=broker, risk_manager=risk, portfolio=portfolio)

    mode_lbl = "KITE" if use_kite else "PAPER"
    if args.dry_run and use_kite:
        mode_lbl += "+DRYRUN"
    logger.info("{} trade for {} (equity ₹{:,.0f})", mode_lbl, today, portfolio.equity)

    notifier = TelegramNotifier.from_settings()

    # ---- Kill switch path ----
    if args.kill_switch:
        records = router.kill_switch(
            last_marks=last_marks, reason="manual_kill_switch", as_of=today
        )
        state.breaker_halted = True
        state.breaker_halt_reason = "manual_kill_switch"
        for r in records:
            for fill in r.fills:
                state.fills.append(_fill_to_dict(fill))
        portfolio.mark_to_market(last_marks, as_of=today)
        state.cash = portfolio.cash_inr
        state.equity_curve[today.isoformat()] = portfolio.equity
        state.last_run = datetime.now(UTC).isoformat()
        state.save(state_file)
        notifier.kill_switch(n_closed=len(records))
        print("KILL SWITCH executed. All positions closed.")
        return 0

    # ---- Run any stop-loss exits FIRST (defensive priority) ----
    stops_hit: list[str] = []
    for ticker, position in list(portfolio.positions.items()):
        last_close = last_marks.get(ticker, position.avg_entry_price)
        triggered = (
            position.side is Side.LONG
            and position.lots
            and last_close <= position.lots[0].current_stop
        )
        if triggered:
            router.exit_position(
                ticker=ticker,
                reason="atr_stop",
                last_price=last_close,
                strategy_name=position.strategy_name,
                as_of=today,
            )
            risk.breaker.record_stop_hit(position.strategy_name or "unknown", today)
            notifier.stop_hit(
                ticker=ticker,
                price=last_close,
                strategy=position.strategy_name or "unknown",
            )
            stops_hit.append(ticker)

    # ---- Run strategies through the selector ----
    history_panel = panel.loc[panel.index.get_level_values("date") <= as_of]
    index_ohlc = (
        index_df.loc[index_df.index <= as_of][["high", "low", "close"]]
        if not index_df.empty
        else None
    )

    if args.strategies:
        strategies = build_strategies(args.strategies)
    else:
        strategies = build_active_strategies()
        if not strategies:
            strategies = build_strategies(
                ["momentum", "mean_reversion", "breakout", "multi_factor", "sector_rotation"]
            )
    selector = StrategySelector(
        strategies=strategies,
        regime_detector=RegimeDetector(),
        top_n_final=args.top_n,
    )
    if index_ohlc is None or index_ohlc.empty:
        print("WARN: no Nifty index in DB, regime detector will fall back to UNKNOWN.")
        # Build a dummy ohlc from average close instead
        avg_close = panel["close"].groupby(level="date").mean()
        index_ohlc = pd.DataFrame(
            {"high": avg_close * 1.005, "low": avg_close * 0.995, "close": avg_close}
        )

    # Pull latest per-ticker sentiment (used by sentiment_momentum + as a filter)
    sentiment_df = latest_per_ticker_sentiment(store, as_of=today, lookback_days=7)

    result = selector.select(
        index_ohlc=index_ohlc.loc[index_ohlc.index <= as_of],
        prices=history_panel,
        features=pd.DataFrame(),  # multi_factor will return empty without fundamentals
        sentiment=sentiment_df if not sentiment_df.empty else None,
    )

    print("\n--- Regime ---")
    diag = result.regime_allocation.diagnostics
    print(f"  Regime          : {result.regime_allocation.regime}")
    print(
        f"  Vol (ann.)      : {diag.realised_vol_annual:.2%}"
        if diag.realised_vol_annual is not None
        else "  vol             : nan"
    )
    print(f"  Trend (ATRs)    : {diag.trend_score:.2f}")

    print(f"\n--- Top-{args.top_n} signals ---")
    if not result.final_signals:
        print("  (no signals — strategies idle today)")

    # ---- Route each top signal through the full pipeline ----
    routed = []
    for sig in result.final_signals:
        ticker_history = (
            panel.xs(sig.ticker, level="ticker")[["open", "high", "low", "close"]]
            if sig.ticker in panel.index.get_level_values("ticker")
            else pd.DataFrame()
        )
        ticker_history = ticker_history.loc[ticker_history.index <= as_of]
        # Pick the strategy that contributed the most weight (for ATR-mult lookup).
        contributors = sig.metadata.get("contributors") or {}
        strategy_name = (
            max(contributors.items(), key=lambda kv: kv[1])[0] if contributors else "momentum"
        )
        routing = router.route_signal(
            signal=sig,
            prices_history=ticker_history,
            today=today,
            strategy_name=strategy_name,
        )
        routed.append((sig, routing))
        if routing.approval is not None and routing.approval.approved:
            rec = routing.record
            if rec is not None and rec.rejection_reason == "dry_run":
                print(
                    f"  {sig.ticker:<14s} {sig.side.value:<5s} qty={routing.approval.approved_quantity:>4d}  "
                    f"conv={sig.conviction:.2f}  → DRY-RUN (order not sent)"
                )
            else:
                print(
                    f"  {sig.ticker:<14s} {sig.side.value:<5s} qty={routing.approval.approved_quantity:>4d}  "
                    f"conv={sig.conviction:.2f}  → ROUTED via {strategy_name}"
                )
                for fill in rec.fills if rec else []:
                    state.fills.append(_fill_to_dict(fill))
                    notifier.fill(
                        ticker=fill.ticker,
                        side=fill.side,
                        quantity=fill.quantity,
                        price=fill.price,
                        reason=strategy_name,
                    )
        else:
            reason = routing.rejected_reason or (
                routing.approval.reason if routing.approval else "unknown"
            )
            print(
                f"  {sig.ticker:<14s} {sig.side.value:<5s} {sig.conviction:.2f}  → REJECTED: {reason}"
            )

    # ---- Mark to market + persist ----
    portfolio.mark_to_market(last_marks, as_of=today)
    state.cash = portfolio.cash_inr
    state.equity_curve[today.isoformat()] = portfolio.equity
    state.high_watermark = portfolio.high_watermark
    state.high_watermark_date = (
        portfolio.high_watermark_date.isoformat() if portfolio.high_watermark_date else None
    )
    state.last_run = datetime.now(UTC).isoformat()
    state.consecutive_stops = {k: int(v) for k, v in risk.breaker.consecutive_stops.items()}
    state.strategy_disabled_until = {
        k: d.isoformat() for k, d in risk.breaker.strategy_disabled_until.items()
    }
    if risk.breaker.state == BreakerState.PAUSE_DAY and risk.breaker.last_pause_date:
        state.breaker_pause_date = risk.breaker.last_pause_date.isoformat()
    else:
        state.breaker_pause_date = None
    breaker_state_str = risk.breaker.state.value
    if risk.breaker.state.value == "halted":
        state.breaker_halted = True
        state.breaker_halt_reason = risk.breaker.halt_reason
        notifier.circuit_breaker_tripped(
            state="halted", reason=risk.breaker.halt_reason or "unknown"
        )
    state.save(state_file)

    # Append the equity curve as CSV for easy plotting.
    rows = sorted(state.equity_curve.items())
    pd.DataFrame(rows, columns=["date", "equity"]).to_csv(equity_file, index=False)

    # ---- Daily summary ----
    print("\n--- Portfolio ---")
    pf_df = portfolio.to_dataframe()
    if pf_df.empty:
        print("  (flat — no open positions)")
    else:
        with pd.option_context("display.float_format", lambda x: f"₹{x:,.2f}"):
            print(pf_df.to_string(index=False))

    print(
        f"\nEquity   : ₹{portfolio.equity:>12,.0f}  "
        f"(cash ₹{portfolio.cash_inr:,.0f}, +{(portfolio.equity / state.initial_capital - 1):+.2%} since inception)"
    )
    print(f"Drawdown : {portfolio.drawdown():+.2%}  (peak ₹{portfolio.high_watermark:,.0f})")
    if stops_hit:
        print(f"Stops hit: {', '.join(stops_hit)}")
    print(f"Breaker  : {risk.breaker.state.value.upper()}")
    print(f"\nState saved → {state_file}")

    # End-of-day Telegram summary
    daily_pct = portfolio.daily_pnl(today)
    n_filled_today = sum(
        1 for f in state.fills if f.get("timestamp", "").startswith(today.isoformat())
    )
    notifier.daily_summary(
        as_of=today.isoformat(),
        equity=portfolio.equity,
        daily_pct=daily_pct,
        drawdown_pct=portfolio.drawdown(),
        regime=str(result.regime_allocation.regime),
        n_open=len(portfolio.positions),
        n_filled_today=n_filled_today,
        breaker_state=breaker_state_str,
    )
    return 0


def _fill_to_dict(fill) -> dict:
    return {
        "timestamp": fill.timestamp.isoformat()
        if hasattr(fill.timestamp, "isoformat")
        else str(fill.timestamp),
        "ticker": fill.ticker,
        "side": fill.side,
        "quantity": fill.quantity,
        "price": fill.price,
        "cost_inr": fill.cost_inr,
        "strategy_name": fill.strategy_name,
    }


if __name__ == "__main__":
    raise SystemExit(main())
