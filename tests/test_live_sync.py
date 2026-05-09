"""Live portfolio sync helpers."""

from __future__ import annotations

from dataclasses import dataclass

from src.orders.live_sync import stop_price_from_state_fills, sync_portfolio_from_kite
from src.risk.portfolio import Portfolio
from src.strategies.base import Side


@dataclass
class _FakeKite:
    """Minimal stand-in for margin + net position APIs."""

    cash: float
    rows: list[tuple[str, int, float, str]]
    product: str = "CNC"

    def available_equity_cash(self) -> float:
        return self.cash

    def iter_net_positions(self) -> list[tuple[str, int, float, str]]:
        return list(self.rows)


def test_stop_from_state_fills_picks_last() -> None:
    fills = [
        {"ticker": "A.NS", "stop_price": 90.0},
        {"ticker": "B.NS", "stop_price": 1.0},
        {"ticker": "A.NS", "stop_price": 95.0},
    ]
    assert stop_price_from_state_fills(fills, "A.NS") == 95.0


def test_sync_portfolio_long_only() -> None:
    port = Portfolio(
        cash_inr=0.0,
        initial_equity_inr=1_000_000.0,
        sector_lookup={"RELIANCE.NS": "Energy"},
    )
    fk = _FakeKite(
        cash=50_000.0,
        rows=[("RELIANCE.NS", 10, 1000.0, "CNC")],
    )
    sync_portfolio_from_kite(port, fk, state_fills=[])  # type: ignore[arg-type]
    assert port.cash_inr == 50_000.0
    assert "RELIANCE.NS" in port.positions
    assert port.positions["RELIANCE.NS"].side is Side.LONG
    assert port.positions["RELIANCE.NS"].quantity == 10


def test_sync_skips_product_mismatch() -> None:
    port = Portfolio(cash_inr=0.0, initial_equity_inr=1.0, sector_lookup={})
    fk = _FakeKite(cash=10.0, rows=[("X.NS", 1, 50.0, "MIS")], product="CNC")
    sync_portfolio_from_kite(port, fk, state_fills=[])  # type: ignore[arg-type]
    assert port.positions == {}
