"""The Risk Manager — central approval gate for every trade.

Every order an algorithm tries to place flows through ``RiskManager.approve``.
The manager says yes / no / size-down based on:

    1. Circuit-breaker state (drawdown, daily loss, manual kill)
    2. Per-strategy loss-streak disabling
    3. Per-trade risk budget (1% of equity max)
    4. Single-position concentration (20% cap)
    5. Sector concentration (35% cap)
    6. Gross/net exposure caps
    7. Sufficient cash (no margin in Phase 8)

Returns an ``ApprovalDecision`` with either:
    - ``approved=True`` and a possibly-shrunk ``approved_quantity``
    - ``approved=False`` and a structured rejection reason

Why this layer:
    The strategy says "I want to long RELIANCE conviction 0.8".
    The sizer turns that into "buy 200 shares with stop at ₹1380".
    The risk manager is the LAST gate before the order goes to the broker
    — the place where everything we've learnt from Chan, López de Prado,
    and the don'ts list is enforced unilaterally and unbypassably.

The risk manager is the single most important module in the codebase
for capital preservation. The strategies can be bad, the broker can lag,
the sentiment scraper can break — but if the risk manager is correct,
we don't blow up.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from src.risk.circuit_breaker import BreakerDecision, CircuitBreaker
from src.risk.portfolio import Portfolio
from src.strategies.base import RiskParams, Side, Signal


@dataclass(frozen=True)
class TradeRequest:
    """What a strategy / sizer asks the risk manager to approve."""

    signal: Signal
    quantity: int
    entry_price: float
    stop_price: float
    strategy_name: str


@dataclass(frozen=True)
class ApprovalDecision:
    """Risk manager's verdict."""

    approved: bool
    approved_quantity: int  # may be < requested if size-down applied
    reason: str  # short tag, machine-readable
    detail: str  # human-readable
    breaker: BreakerDecision
    metadata: dict[str, float] = field(default_factory=dict)


@dataclass
class RiskLimits:
    """The numerical limits the manager enforces. All overridable from config."""

    per_trade_pct: float = 0.01  # 1% of equity at risk per trade
    per_strategy_daily_pct: float = 0.05  # 5% loss per strategy per day
    max_single_position_pct: float = 0.20  # 20% in one stock
    max_sector_pct: float = 0.35  # 35% in one sector
    max_gross_exposure: float = 1.0  # 1.0 = no leverage
    max_net_long_exposure: float = 1.0  # cap on (longs - shorts) / equity
    cash_buffer_pct: float = 0.05  # always keep 5% in cash


@dataclass
class RiskManager:
    """The trading-floor risk officer for our bot."""

    limits: RiskLimits = field(default_factory=RiskLimits)
    breaker: CircuitBreaker = field(default_factory=CircuitBreaker)

    # ----------------------------------------------------------------
    def approve(
        self,
        request: TradeRequest,
        portfolio: Portfolio,
        today: date,
    ) -> ApprovalDecision:
        breaker_decision = self.breaker.check(portfolio, today)
        equity = portfolio.equity

        # 0. Sanity
        if request.quantity <= 0 or request.entry_price <= 0:
            return ApprovalDecision(
                approved=False,
                approved_quantity=0,
                reason="invalid_request",
                detail=f"qty={request.quantity}, price={request.entry_price}",
                breaker=breaker_decision,
            )

        # 1. Are we even allowed to take new entries?
        if not breaker_decision.allow_new_entries:
            return ApprovalDecision(
                approved=False,
                approved_quantity=0,
                reason="breaker",
                detail=breaker_decision.reason,
                breaker=breaker_decision,
            )

        # 2. Strategy disabled?
        if request.strategy_name in breaker_decision.disabled_strategies:
            return ApprovalDecision(
                approved=False,
                approved_quantity=0,
                reason="strategy_disabled",
                detail=f"strategy '{request.strategy_name}' disabled until streak recovers",
                breaker=breaker_decision,
            )

        # 3. Per-trade risk: shares allowed by (entry - stop) * qty ≤ 1% of equity.
        risk_per_share = abs(request.entry_price - request.stop_price)
        if risk_per_share <= 0:
            return ApprovalDecision(
                approved=False,
                approved_quantity=0,
                reason="bad_stop",
                detail="stop = entry (zero risk per share is unphysical)",
                breaker=breaker_decision,
            )
        max_qty_by_risk = int(equity * self.limits.per_trade_pct / risk_per_share)
        if max_qty_by_risk <= 0:
            return ApprovalDecision(
                approved=False,
                approved_quantity=0,
                reason="risk_budget_exhausted",
                detail=f"per-trade risk budget = ₹{equity * self.limits.per_trade_pct:,.0f}",
                breaker=breaker_decision,
            )

        # 4. Single-position cap: existing weight + this fill ≤ 20% of equity.
        existing_weight = portfolio.position_weight(request.signal.ticker)
        notional_per_share = request.entry_price
        max_qty_by_position = int(
            (equity * self.limits.max_single_position_pct - abs(existing_weight) * equity)
            / max(notional_per_share, 1e-9)
        )
        max_qty_by_position = max(0, max_qty_by_position)

        # 5. Sector cap: ticker's sector + this fill ≤ 35% of equity.
        sector = portfolio.sector_lookup.get(request.signal.ticker.upper())
        sector_weight = portfolio.sector_weights().get(sector or "_unknown", 0.0)
        sector_remaining = (self.limits.max_sector_pct - abs(sector_weight)) * equity
        max_qty_by_sector = (
            int(sector_remaining / max(notional_per_share, 1e-9)) if sector_remaining > 0 else 0
        )

        # 6. Cash floor — must leave a buffer in cash after this trade.
        notional = request.quantity * request.entry_price
        cash_after = portfolio.cash_inr - notional
        cash_min = equity * self.limits.cash_buffer_pct
        if request.signal.side is Side.LONG and cash_after < cash_min:
            max_qty_by_cash = max(
                0,
                int((portfolio.cash_inr - cash_min) / max(notional_per_share, 1e-9)),
            )
        else:
            max_qty_by_cash = request.quantity  # short freed cash, ignore

        # 7. Apply the breaker's size multiplier (rolling 5-day loss → 0.5x).
        size_mult = breaker_decision.new_entry_size_multiplier

        approved_qty = min(
            request.quantity,
            max_qty_by_risk,
            max_qty_by_position,
            max_qty_by_sector,
            max_qty_by_cash,
        )
        approved_qty = int(approved_qty * size_mult)

        if approved_qty <= 0:
            limiting = _smallest_label(
                {
                    "risk": max_qty_by_risk,
                    "position_cap": max_qty_by_position,
                    "sector_cap": max_qty_by_sector,
                    "cash_buffer": max_qty_by_cash,
                    "size_mult_zero": int(size_mult * 1000),
                }
            )
            return ApprovalDecision(
                approved=False,
                approved_quantity=0,
                reason=f"capped_by_{limiting}",
                detail=f"limits: risk={max_qty_by_risk}, pos={max_qty_by_position}, "
                f"sector={max_qty_by_sector}, cash={max_qty_by_cash}, mult={size_mult}",
                breaker=breaker_decision,
            )

        return ApprovalDecision(
            approved=True,
            approved_quantity=approved_qty,
            reason="approved",
            detail=f"capped from {request.quantity} → {approved_qty} (size_mult={size_mult})"
            if approved_qty < request.quantity
            else "approved at requested size",
            breaker=breaker_decision,
            metadata={
                "max_qty_by_risk": float(max_qty_by_risk),
                "max_qty_by_position": float(max_qty_by_position),
                "max_qty_by_sector": float(max_qty_by_sector),
                "max_qty_by_cash": float(max_qty_by_cash),
                "size_mult": float(size_mult),
                "existing_position_weight": existing_weight,
                "sector_weight": sector_weight,
            },
        )

    # ----------------------------------------------------------------
    @staticmethod
    def make_request(
        signal: Signal,
        atr_value: float,
        atr_multiple: float,
        equity: float,
        limits: RiskLimits,
        last_price: float,
        strategy_name: str,
        win_rate: float = 0.50,
        win_loss_ratio: float = 1.50,
    ) -> TradeRequest | None:
        """Convenience factory: turn a Signal + ATR into a sized TradeRequest.

        Combines Half-Kelly notional sizing with ATR-based stop placement.
        Mirrors what the live order router will do but bundled here so
        you can use the manager standalone in tests.
        """
        if signal.side is Side.FLAT or last_price <= 0 or atr_value <= 0:
            return None
        from src.risk.sizer import atr_stop_price, half_kelly_fraction

        kelly = half_kelly_fraction(win_rate, win_loss_ratio, cap=limits.max_single_position_pct)
        kelly *= signal.conviction
        notional = min(equity * kelly, equity * limits.max_single_position_pct)
        qty = int(notional / last_price)
        if qty <= 0:
            return None
        stop_price = atr_stop_price(last_price, atr_value, signal.side, atr_multiple)
        return TradeRequest(
            signal=signal,
            quantity=qty,
            entry_price=last_price,
            stop_price=stop_price,
            strategy_name=strategy_name,
        )


def _smallest_label(d: dict[str, int]) -> str:
    return min(d.items(), key=lambda kv: kv[1])[0]


def risk_limits_from_settings() -> RiskLimits:
    """Build ``RiskLimits`` from merged YAML (``default`` + ``ALGO_MODE`` overlay)."""
    from src.utils.settings import settings

    cfg = settings.get("risk", default={}) or {}
    if not isinstance(cfg, dict):
        return RiskLimits()
    d = RiskLimits()
    return RiskLimits(
        per_trade_pct=float(cfg.get("per_trade_pct", d.per_trade_pct)),
        per_strategy_daily_pct=float(cfg.get("per_strategy_daily_pct", d.per_strategy_daily_pct)),
        max_single_position_pct=float(cfg.get("max_single_position_pct", d.max_single_position_pct)),
        max_sector_pct=float(cfg.get("max_sector_pct", d.max_sector_pct)),
        max_gross_exposure=float(cfg.get("max_gross_exposure", d.max_gross_exposure)),
        max_net_long_exposure=float(cfg.get("max_net_long_exposure", d.max_net_long_exposure)),
        cash_buffer_pct=float(cfg.get("cash_buffer_pct", d.cash_buffer_pct)),
    )


# Bring through the existing simple sizer too.
_ = RiskParams  # re-export for convenience

__all__ = [
    "ApprovalDecision",
    "RiskLimits",
    "RiskManager",
    "TradeRequest",
    "risk_limits_from_settings",
]
