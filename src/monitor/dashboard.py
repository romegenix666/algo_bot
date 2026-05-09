"""Streamlit live dashboard.

Run::

    streamlit run src/monitor/dashboard.py

What it shows:
    - Equity curve since inception (line chart, log + linear)
    - Daily P&L bar chart
    - Drawdown chart
    - Open positions (with unrealised P&L)
    - Recent fills
    - Recent sentiment scores
    - Circuit-breaker state + halt reason
    - Regime + active strategy weights

Read-only: never mutates state. Safe to run alongside the trading driver.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

# Streamlit imports are heavy; only import if actually running the UI.
try:
    import plotly.express as px
    import plotly.graph_objects as go
    import streamlit as st
except ImportError:  # pragma: no cover - dashboard-only deps
    st = None
    px = None
    go = None

from src.data.storage import DataStore
from src.sentiment.storage import latest_per_ticker_sentiment
from src.utils.settings import PROJECT_ROOT, settings


def _state_paths() -> tuple[Path, Path]:
    """Paper vs live state matches ``scripts/paper_trade`` layout."""
    sub = "live" if str(settings.mode).lower() == "live" else "paper"
    base = PROJECT_ROOT / "data" / sub
    return base / "state.json", base / "equity_curve.csv"


def load_state() -> dict:
    state_file, _ = _state_paths()
    if not state_file.exists():
        return {}
    with state_file.open() as fh:
        return json.load(fh)


def load_equity_curve() -> pd.DataFrame:
    _, equity_file = _state_paths()
    if not equity_file.exists():
        return pd.DataFrame(columns=["date", "equity"])
    df = pd.read_csv(equity_file, parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


def daily_returns_df(equity_df: pd.DataFrame) -> pd.DataFrame:
    if equity_df.empty:
        return pd.DataFrame(columns=["date", "daily_return", "drawdown"])
    df = equity_df.copy().sort_values("date")
    df["daily_return"] = df["equity"].pct_change().fillna(0.0)
    df["peak"] = df["equity"].cummax()
    df["drawdown"] = df["equity"] / df["peak"] - 1.0
    return df


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------


def render() -> None:  # pragma: no cover - UI  # noqa: C901  (UI render is naturally branchy)
    if st is None:
        raise ImportError("Install streamlit + plotly to use the dashboard.")
    st.set_page_config(page_title="Algo Bot · Live", layout="wide", page_icon="📈")

    st.title("📈 Algo Bot — Paper Trading Dashboard")
    state = load_state()
    equity_df = load_equity_curve()

    if not state and equity_df.empty:
        st.info(
            "No trading state yet for this mode (check ALGO_MODE). "
            "Run `python -m scripts.paper_trade` after the first close."
        )
        return

    # ----- Headline KPIs -----
    initial_capital = state.get("initial_capital", 1_000_000.0)
    last_run = state.get("last_run", "—")
    cash = state.get("cash", initial_capital)
    equity = cash if equity_df.empty else float(equity_df["equity"].iloc[-1])
    pnl_total_pct = (equity / initial_capital - 1.0) if initial_capital else 0.0

    rets = daily_returns_df(equity_df)
    last_dd = float(rets["drawdown"].iloc[-1]) if not rets.empty else 0.0
    daily_pct = float(rets["daily_return"].iloc[-1]) if not rets.empty else 0.0

    cols = st.columns(5)
    cols[0].metric("Equity", f"₹{equity:,.0f}", f"{pnl_total_pct:+.2%}")
    cols[1].metric("Day P&L", f"{daily_pct:+.2%}")
    cols[2].metric("Drawdown", f"{last_dd:+.2%}")
    cols[3].metric("Cash", f"₹{cash:,.0f}")
    breaker_state = "HALTED" if state.get("breaker_halted") else "HEALTHY"
    cols[4].metric("Breaker", breaker_state)

    st.caption(f"Last run: {last_run}")

    # ----- Equity curve -----
    if not rets.empty:
        st.subheader("Equity Curve")
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=rets["date"],
                y=rets["equity"],
                mode="lines",
                name="Equity",
                line={"color": "#1f77b4", "width": 2},
            )
        )
        fig.add_trace(
            go.Scatter(
                x=rets["date"],
                y=rets["peak"],
                mode="lines",
                name="Peak",
                line={"color": "#aaaaaa", "width": 1, "dash": "dash"},
            )
        )
        fig.update_layout(
            height=380,
            margin={"t": 20, "b": 20, "l": 0, "r": 0},
            xaxis={"title": ""},
            yaxis={"title": "₹"},
        )
        st.plotly_chart(fig, use_container_width=True)

        # ---- Drawdown chart ----
        st.subheader("Drawdown")
        dd_fig = px.area(
            rets,
            x="date",
            y="drawdown",
            color_discrete_sequence=["#d62728"],
            labels={"drawdown": "Drawdown"},
        )
        dd_fig.update_yaxes(tickformat=".0%")
        dd_fig.update_layout(height=220, margin={"t": 20, "b": 20, "l": 0, "r": 0})
        st.plotly_chart(dd_fig, use_container_width=True)

    # ----- Open positions -----
    st.subheader("Open Positions")
    fills = state.get("fills") or []
    if not fills:
        st.write("(no fills yet)")
    else:
        # Compute open positions from fills
        net: dict[str, dict] = {}
        for f in fills:
            t = f["ticker"]
            sign = 1 if f["side"] == "buy" else -1
            row = net.setdefault(t, {"ticker": t, "qty": 0, "cost_total": 0.0, "strategies": set()})
            row["qty"] += sign * int(f["quantity"])
            row["cost_total"] += sign * f["price"] * int(f["quantity"])
            if f.get("strategy_name"):
                row["strategies"].add(f["strategy_name"])
        rows = []
        for r in net.values():
            if r["qty"] == 0:
                continue
            avg = abs(r["cost_total"] / r["qty"]) if r["qty"] else 0.0
            rows.append(
                {
                    "ticker": r["ticker"],
                    "qty": r["qty"],
                    "avg_entry": avg,
                    "strategies": ", ".join(sorted(r["strategies"])),
                }
            )
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        else:
            st.write("(flat — no open positions)")

    # ----- Recent fills -----
    st.subheader("Recent Fills (last 20)")
    if fills:
        recent = pd.DataFrame(fills[-20:])
        recent["timestamp"] = pd.to_datetime(recent["timestamp"], errors="coerce")
        st.dataframe(recent.sort_values("timestamp", ascending=False), use_container_width=True)
    else:
        st.write("(no fills)")

    # ----- Sentiment snapshot -----
    st.subheader("Latest Sentiment Scores")
    try:
        store = DataStore.from_settings()
        sent_df = latest_per_ticker_sentiment(store, as_of=pd.Timestamp.now().date())
    except Exception as exc:  # pragma: no cover - UI defensive
        st.write(f"(sentiment unavailable: {exc})")
        sent_df = pd.DataFrame()
    if not sent_df.empty:
        sent_df = sent_df.sort_values("score", ascending=False)
        st.dataframe(sent_df, use_container_width=True)
    else:
        st.write("(no sentiment data — run `python -m scripts.refresh_sentiment`)")


def _path_for_streamlit() -> Path:
    """Return the project root so this can be `streamlit run`-ed from anywhere."""
    return PROJECT_ROOT


if __name__ == "__main__":  # pragma: no cover - UI
    render()
