"""Portfolio-level circuit breakers.

A trading bot must have automatic kill switches. *Especially* one running
unsupervised. The risks we guard against:

    1. **Catastrophic drawdown** — total equity drops > X% from peak.
       Halt all new trading; manual review required.
    2. **Daily-loss spike** — losses on a single day exceed Y% of equity.
       Pause new entries for the rest of the day; existing positions kept.
    3. **Rolling drawdown speed** — losses over the past 5 days > Z%.
       Cut all NEW entry sizes by 50% until the streak ends.
    4. **Strategy losing-streak** — N consecutive stop-loss hits in one
       strategy. Disable that strategy for cooling-off.
    5. **Manual kill** — operator pressed the button. Hard stop.

Circuit-breaker state machine::

    HEALTHY ─[daily loss > 3%]──────► PAUSE_DAY (auto-clear next session)
            ─[drawdown > 12%]───────► HALTED (manual reset only)
            ─[3 stops in 1 strategy]► STRATEGY_DISABLED(<name>) (5 sessions)
            ─[manual kill]──────────► HALTED

Order routing consults the breaker state BEFORE submitting any new entry.
Existing positions can still be exited (defensive priority).

References:
    - Chan §6 "Risk Management" — drawdown thresholds
    - Bailey & López de Prado (2014) — backtest overfitting tolerance
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import StrEnum

from src.risk.portfolio import Portfolio


class BreakerState(StrEnum):
    HEALTHY = "healthy"
    PAUSE_DAY = "pause_day"
    HALTED = "halted"


@dataclass(frozen=True)
class BreakerDecision:
    """Output of a single ``check`` call."""

    allow_new_entries: bool
    allow_exits: bool  # almost always True
    new_entry_size_multiplier: float  # 1.0 = full size; 0.5 = half-size
    state: BreakerState
    reason: str
    disabled_strategies: tuple[str, ...] = ()


@dataclass
class CircuitBreaker:
    """Stateful guardrail. Construct once per bot run; ``check`` per bar."""

    # Thresholds (override via config in production)
    max_drawdown_halt: float = 0.12  # halt at -12% from peak
    max_daily_loss: float = 0.03  # pause day at -3% intraday
    max_5day_loss: float = 0.07  # cut sizes by 50%
    consecutive_stops_to_disable_strategy: int = 3
    strategy_disable_days: int = 5

    # State
    state: BreakerState = BreakerState.HEALTHY
    last_pause_date: date | None = None
    halted_at: date | None = None
    halt_reason: str | None = None
    consecutive_stops: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    strategy_disabled_until: dict[str, date] = field(default_factory=dict)
    manual_kill: bool = False

    # ----------------------------------------------------------------
    def manual_kill_switch(self, when: date) -> None:
        """Operator's HARD STOP. Bot will exit-only until reset."""
        self.manual_kill = True
        self.state = BreakerState.HALTED
        self.halted_at = when
        self.halt_reason = "manual_kill_switch"

    def reset(self) -> None:
        """Operator-only — clear HALTED after manual review."""
        self.state = BreakerState.HEALTHY
        self.halted_at = None
        self.halt_reason = None
        self.manual_kill = False
        self.consecutive_stops.clear()
        self.strategy_disabled_until.clear()

    # ----------------------------------------------------------------
    def record_stop_hit(self, strategy_name: str, when: date) -> None:
        """Tell the breaker that a stop was just triggered for a strategy.

        Three in a row → that strategy gets benched for ``strategy_disable_days``.
        """
        self.consecutive_stops[strategy_name] += 1
        if self.consecutive_stops[strategy_name] >= self.consecutive_stops_to_disable_strategy:
            self.strategy_disabled_until[strategy_name] = when + timedelta(
                days=self.strategy_disable_days
            )
            self.consecutive_stops[strategy_name] = 0  # reset the counter once disabled

    def record_winning_close(self, strategy_name: str) -> None:
        """A profitable exit resets the strategy's loss-streak counter."""
        if strategy_name in self.consecutive_stops:
            self.consecutive_stops[strategy_name] = 0

    # ----------------------------------------------------------------
    def check(self, portfolio: Portfolio, today: date) -> BreakerDecision:
        """Evaluate state for the given bar; return per-bar decision."""

        # 1. Manual kill is sticky — overrides everything.
        if self.manual_kill or self.state is BreakerState.HALTED:
            return BreakerDecision(
                allow_new_entries=False,
                allow_exits=True,
                new_entry_size_multiplier=0.0,
                state=BreakerState.HALTED,
                reason=self.halt_reason or "halted",
                disabled_strategies=tuple(self._currently_disabled(today)),
            )

        # 2. Catastrophic drawdown — auto-halt.
        dd = portfolio.drawdown()
        if dd <= -abs(self.max_drawdown_halt):
            self.state = BreakerState.HALTED
            self.halted_at = today
            self.halt_reason = f"drawdown {dd:.2%} ≤ -{self.max_drawdown_halt:.0%}"
            return BreakerDecision(
                allow_new_entries=False,
                allow_exits=True,
                new_entry_size_multiplier=0.0,
                state=BreakerState.HALTED,
                reason=self.halt_reason,
                disabled_strategies=tuple(self._currently_disabled(today)),
            )

        # 3. Single-day loss spike — pause new entries until next session.
        daily_loss = portfolio.daily_pnl(today)
        if daily_loss <= -abs(self.max_daily_loss):
            self.state = BreakerState.PAUSE_DAY
            self.last_pause_date = today
            return BreakerDecision(
                allow_new_entries=False,
                allow_exits=True,
                new_entry_size_multiplier=0.0,
                state=BreakerState.PAUSE_DAY,
                reason=f"daily P&L {daily_loss:.2%} ≤ -{self.max_daily_loss:.0%}",
                disabled_strategies=tuple(self._currently_disabled(today)),
            )

        # If we paused yesterday, auto-clear today (fresh session).
        if self.state is BreakerState.PAUSE_DAY and self.last_pause_date != today:
            self.state = BreakerState.HEALTHY

        # 4. Rolling 5-day loss — cut sizes in half but keep trading.
        size_mult = 1.0
        rolling_loss = self._rolling_pnl(portfolio, today, lookback_days=5)
        if rolling_loss <= -abs(self.max_5day_loss):
            size_mult = 0.5
            reason = (
                f"5-day rolling loss {rolling_loss:.2%} ≤ -{self.max_5day_loss:.0%}; sizes halved"
            )
            return BreakerDecision(
                allow_new_entries=True,
                allow_exits=True,
                new_entry_size_multiplier=size_mult,
                state=BreakerState.HEALTHY,
                reason=reason,
                disabled_strategies=tuple(self._currently_disabled(today)),
            )

        return BreakerDecision(
            allow_new_entries=True,
            allow_exits=True,
            new_entry_size_multiplier=size_mult,
            state=BreakerState.HEALTHY,
            reason="",
            disabled_strategies=tuple(self._currently_disabled(today)),
        )

    # ----------------------------------------------------------------
    def _currently_disabled(self, today: date) -> list[str]:
        """Return names of strategies disabled by losing-streak as of today."""
        out: list[str] = []
        expired = []
        for name, until in self.strategy_disabled_until.items():
            if until <= today:
                expired.append(name)
            else:
                out.append(name)
        for name in expired:
            del self.strategy_disabled_until[name]
        return out

    # ----------------------------------------------------------------
    @staticmethod
    def _rolling_pnl(portfolio: Portfolio, today: date, lookback_days: int) -> float:
        if not portfolio.equity_curve:
            return 0.0
        curve = sorted(portfolio.equity_curve.items())
        # Find today's position in curve
        cutoff_idx = None
        for i, (d, _eq) in enumerate(curve):
            if d > today:
                break
            cutoff_idx = i
        if cutoff_idx is None or cutoff_idx == 0:
            return 0.0
        start_idx = max(0, cutoff_idx - lookback_days)
        start_eq = curve[start_idx][1]
        end_eq = curve[cutoff_idx][1]
        if start_eq <= 0:
            return 0.0
        return end_eq / start_eq - 1.0


def circuit_breaker_from_settings() -> CircuitBreaker:
    """Build thresholds from merged YAML ``risk`` block (see ``config/default.yaml``)."""
    from src.utils.settings import settings

    cfg = settings.get("risk", default={}) or {}
    if not isinstance(cfg, dict):
        return CircuitBreaker()
    base = CircuitBreaker()

    def _f(key: str, default: float) -> float:
        v = cfg.get(key, default)
        if v is None:
            return default
        return float(v)

    def _i(key: str, default: int) -> int:
        v = cfg.get(key, default)
        if v is None:
            return default
        return int(v)

    daily = cfg.get("circuit_daily_loss_pause", cfg.get("max_daily_loss", base.max_daily_loss))
    five_d = cfg.get(
        "circuit_5day_loss_half_size",
        cfg.get("max_5day_loss", base.max_5day_loss),
    )
    return CircuitBreaker(
        max_drawdown_halt=_f("portfolio_max_drawdown", base.max_drawdown_halt),
        max_daily_loss=float(daily) if daily is not None else base.max_daily_loss,
        max_5day_loss=float(five_d) if five_d is not None else base.max_5day_loss,
        consecutive_stops_to_disable_strategy=_i(
            "circuit_consecutive_stops_to_disable",
            base.consecutive_stops_to_disable_strategy,
        ),
        strategy_disable_days=_i("circuit_strategy_disable_days", base.strategy_disable_days),
    )


__all__ = [
    "BreakerDecision",
    "BreakerState",
    "CircuitBreaker",
    "circuit_breaker_from_settings",
]
