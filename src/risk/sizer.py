"""Position sizing — Half-Kelly and ATR-based volatility targeting.

Why this module exists:

- Equal-rupee sizing is wrong: a small-cap with 60% annual vol and a large-cap
  with 18% annual vol carry wildly different risk per share.
- Naive percent-of-equity sizing ignores the strategy's edge.
- Full Kelly is theoretically optimal but practically lethal (Chan §6).

We therefore size positions by:

1. **Risk-budget rule** — never risk more than ``per_trade_pct`` of equity on
   any one trade. Risk = (entry - stop) × shares. Solve for shares.
2. **Half-Kelly cap** — derive a Kelly fraction from the strategy's historical
   win-rate and win/loss ratio; halve it; cap it.
3. **Single-position cap** — hard ceiling (e.g. 20% of equity) regardless of
   the above.
4. **Sector / total-exposure caps** — applied later by the portfolio manager;
   sizer is per-trade only.

References:
    - Kelly (1956). *A New Interpretation of Information Rate*.
    - Vince (1990). *Portfolio Management Formulas* — practical Kelly.
    - Chan (2009), *Quantitative Trading*, Chapter 6 (Money & Risk Management).
    - Antonacci (2014), *Dual Momentum Investing*, Chapter 9 (drawdowns).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.strategies.base import RiskParams, Side, Signal


@dataclass(frozen=True)
class SizingResult:
    """What the sizer hands back to the order manager."""

    ticker: str
    side: Side
    shares: int  # signed positive; side carries direction
    entry_price: float
    stop_price: float  # initial ATR-based stop
    risk_rupees: float  # actual money at risk (entry - stop) * shares
    notional: float  # absolute exposure
    fraction_of_equity: float
    method: str  # "kelly" | "risk_budget" | "capped"


def half_kelly_fraction(win_rate: float, win_loss_ratio: float, cap: float = 0.20) -> float:
    """Half-Kelly fraction with a hard cap.

    Args:
        win_rate: estimated probability of a winning trade, in [0,1].
        win_loss_ratio: average win / average loss (b in Kelly's formula).
        cap: never return a fraction larger than this.
    """
    if not 0.0 <= win_rate <= 1.0:
        raise ValueError("win_rate must be in [0,1]")
    if win_loss_ratio <= 0.0:
        return 0.0
    full = (win_rate * win_loss_ratio - (1.0 - win_rate)) / win_loss_ratio
    return float(min(cap, max(0.0, full / 2.0)))


def atr_stop_price(
    entry: float,
    atr_value: float,
    side: Side,
    atr_multiple: float,
) -> float:
    """Compute the initial ATR-based stop-loss price for a given side.

    Long: stop is *below* entry by ``atr_multiple * ATR``.
    Short: stop is *above* entry by ``atr_multiple * ATR``.
    """
    if atr_value <= 0:
        raise ValueError("ATR must be positive to size a stop")
    if side is Side.LONG:
        return entry - atr_multiple * atr_value
    if side is Side.SHORT:
        return entry + atr_multiple * atr_value
    raise ValueError(f"Cannot size a stop for FLAT side: {side}")


def size_position(
    signal: Signal,
    entry_price: float,
    atr_value: float,
    atr_multiple: float,
    risk: RiskParams,
    win_rate: float = 0.5,
    win_loss_ratio: float = 1.5,
    lot_size: int = 1,
) -> SizingResult | None:
    """Compute the share quantity and stop-loss for one signal.

    Returns ``None`` if the trade is rejected (e.g. zero risk budget or stop
    impossible).

    Args:
        signal: Strategy's signal.
        entry_price: Price we expect to enter at (typically last close + 1 tick).
        atr_value: ATR(14) for this ticker on the entry bar.
        atr_multiple: 1.5 / 2.0 / 2.5 depending on strategy (see config).
        risk: Per-trade risk parameters.
        win_rate: Strategy's historical hit rate (default conservative 0.5).
        win_loss_ratio: Average-win / average-loss ratio (default 1.5).
        lot_size: Round shares down to this multiple (1 for NSE equity, NSE
            F&O has lot sizes — used later in the F&O phase).
    """
    if signal.side is Side.FLAT:
        return None
    if entry_price <= 0:
        return None

    # ---- 1. Risk-budget: how much can we lose on this trade? ----
    risk_budget = risk.equity * risk.per_trade_pct
    stop_price = atr_stop_price(entry_price, atr_value, signal.side, atr_multiple)
    risk_per_share = abs(entry_price - stop_price)
    if risk_per_share <= 0:
        return None
    risk_budget_shares = risk_budget / risk_per_share

    # ---- 2. Half-Kelly cap on notional fraction ----
    kelly_frac = half_kelly_fraction(win_rate, win_loss_ratio, cap=risk.kelly_cap)
    kelly_frac *= signal.conviction  # weighted by strategy's confidence
    kelly_shares = (risk.equity * kelly_frac) / entry_price if kelly_frac > 0 else 0.0

    # ---- 3. Single-position concentration cap ----
    cap_notional = risk.equity * risk.max_single_position_pct
    cap_shares = cap_notional / entry_price

    # ---- 4. Take the *minimum* of all three (most conservative) ----
    candidates = {
        "risk_budget": risk_budget_shares,
        "kelly": kelly_shares,
        "capped": cap_shares,
    }
    method, raw_shares = min(candidates.items(), key=lambda kv: kv[1])
    shares = int((raw_shares // lot_size) * lot_size)
    if shares <= 0:
        return None

    notional = shares * entry_price
    risk_rupees = shares * risk_per_share
    return SizingResult(
        ticker=signal.ticker,
        side=signal.side,
        shares=shares,
        entry_price=entry_price,
        stop_price=stop_price,
        risk_rupees=risk_rupees,
        notional=notional,
        fraction_of_equity=notional / risk.equity if risk.equity > 0 else 0.0,
        method=method,
    )


def update_trailing_stop(
    current_stop: float,
    last_price: float,
    atr_value: float,
    atr_multiple: float,
    side: Side,
    activate_after_atr: float = 1.5,
    entry_price: float | None = None,
) -> float:
    """Ratchet the stop towards the price once it has moved in our favour.

    The trailing stop only moves in the direction that *reduces* risk:
    upward for long positions, downward for short positions. It activates
    only once the position has moved at least ``activate_after_atr * ATR``
    in the favourable direction from entry — this avoids whipsawing out of
    new positions on noise.
    """
    if side is Side.LONG:
        # We need an entry price reference to know "in our favour"
        if entry_price is not None and last_price - entry_price < activate_after_atr * atr_value:
            return current_stop
        candidate = last_price - atr_multiple * atr_value
        return max(current_stop, candidate)

    if side is Side.SHORT:
        if entry_price is not None and entry_price - last_price < activate_after_atr * atr_value:
            return current_stop
        candidate = last_price + atr_multiple * atr_value
        return min(current_stop, candidate)

    return current_stop


def estimate_win_stats(returns: pd.Series) -> tuple[float, float]:
    """Estimate win-rate and win/loss ratio from a series of trade returns.

    Used to feed Half-Kelly during backtesting / live updates. Defaults to
    ``(0.5, 1.0)`` if no closed trades yet.
    """
    if returns.empty:
        return 0.5, 1.0
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    if len(wins) == 0 or len(losses) == 0:
        return 0.5, 1.0
    win_rate = len(wins) / len(returns)
    win_loss_ratio = float(wins.mean() / abs(losses.mean()))
    return win_rate, win_loss_ratio


__all__ = [
    "SizingResult",
    "atr_stop_price",
    "estimate_win_stats",
    "half_kelly_fraction",
    "size_position",
    "update_trailing_stop",
]
