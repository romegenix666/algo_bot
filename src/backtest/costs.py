"""Indian transaction cost model.

Charges that hit a retail equity-delivery trade at Zerodha (or any
SEBI-registered broker, broadly identical):

    1. **Brokerage**       0.03% of turnover (Zerodha equity delivery, ₹20 cap per leg)
                            Some brokers charge 0% on delivery — check yours.
    2. **STT/CTT**          0.10% on buy + sell side (delivery)
                            0.025% sell-side only (intraday)
    3. **Exchange charges** ~0.00325% (NSE) / 0.00375% (BSE) of turnover
    4. **GST**              18% on (brokerage + exchange charges)
    5. **SEBI charges**     ₹10 per Crore (0.0001%) of turnover
    6. **Stamp duty**       0.015% on buy side only (since July 2020)
    7. **Slippage**         5–15 bps depending on liquidity and order size

We model these explicitly, per leg (each buy + each sell is a leg). The
``apply`` method takes a notional value and returns the all-in cost, so
the backtester can subtract it from P&L without having to know the formulae.

References:
    - Zerodha brokerage calculator (for current values)
    - NSE / BSE circulars on STT and exchange charges
    - SEBI circular CIR/MRD/DSA/24/2009 (turnover charges)
    - Chan §3 — "Without incorporating transaction costs, the simplest
      strategies may seem to work at high frequencies."

A note on slippage:
    "Slippage" here is a flat-rate proxy for the *implicit* cost of
    trading: bid-ask spread + market impact + execution delay. For
    daily-bar strategies on Nifty 500 stocks, 5 bps each way is a
    reasonable starting point. Increase to 10–15 bps if you trade
    illiquid mid-caps, decrease to 3 bps if you mostly hit Nifty 50.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostBreakdown:
    """Itemised cost for one trade leg (so we can audit per-leg)."""

    brokerage: float
    stt: float
    exchange: float
    gst: float
    sebi: float
    stamp_duty: float
    slippage: float

    @property
    def total(self) -> float:
        return (
            self.brokerage
            + self.stt
            + self.exchange
            + self.gst
            + self.sebi
            + self.stamp_duty
            + self.slippage
        )


@dataclass(frozen=True)
class IndianEquityCostModel:
    """Cost model tuned for NSE equity-delivery trades.

    All percentages are expressed as decimals (0.0003 = 0.03%).
    Default values reflect Zerodha as of 2024 — adjust if your broker
    differs or rules change.
    """

    brokerage_pct: float = 0.0003  # 0.03%
    brokerage_cap_inr: float = 20.0  # ₹20 cap per leg (Zerodha)
    stt_buy_pct: float = 0.0010  # 0.10% (delivery, buy side)
    stt_sell_pct: float = 0.0010  # 0.10% (delivery, sell side)
    exchange_pct: float = 0.0000325  # 0.00325% NSE
    gst_pct: float = 0.18  # 18% on brokerage + exchange
    sebi_pct: float = 0.0000010  # ₹10 per Crore = 0.0001%
    stamp_duty_buy_pct: float = 0.00015  # 0.015% buy-side only
    slippage_bps: float = 5.0  # 5 basis points (one-way)

    # ------------------------------------------------------------------
    def apply(
        self,
        notional: float,
        side: str,  # "buy" or "sell"
    ) -> CostBreakdown:
        """Return the cost breakdown for one trade leg.

        ``notional`` is the absolute rupee value of the leg
        (price × quantity, no sign).
        """
        if notional < 0:
            raise ValueError("notional must be non-negative")
        side = side.lower()
        if side not in {"buy", "sell"}:
            raise ValueError("side must be 'buy' or 'sell'")

        brokerage = min(notional * self.brokerage_pct, self.brokerage_cap_inr)
        if side == "buy":
            stt = notional * self.stt_buy_pct
            stamp_duty = notional * self.stamp_duty_buy_pct
        else:
            stt = notional * self.stt_sell_pct
            stamp_duty = 0.0

        exchange = notional * self.exchange_pct
        gst = (brokerage + exchange) * self.gst_pct
        sebi = notional * self.sebi_pct
        slippage = notional * (self.slippage_bps / 1e4)

        return CostBreakdown(
            brokerage=brokerage,
            stt=stt,
            exchange=exchange,
            gst=gst,
            sebi=sebi,
            stamp_duty=stamp_duty,
            slippage=slippage,
        )

    # ------------------------------------------------------------------
    def total(self, notional: float, side: str) -> float:
        """Convenience: total all-in cost in rupees for one leg."""
        return self.apply(notional, side).total

    # ------------------------------------------------------------------
    def round_trip(self, notional: float) -> float:
        """All-in cost for a buy + sell at the same notional (rough estimate).

        Use this for quick "what does a round trip cost on a ₹50k position?"
        intuition.
        """
        return self.total(notional, "buy") + self.total(notional, "sell")


# A more conservative model for illiquid mid-caps where slippage hurts.
ILLIQUID_COST_MODEL = IndianEquityCostModel(slippage_bps=15.0)

# An aggressive (optimistic) model for very liquid Nifty 50 trades only.
LIQUID_COST_MODEL = IndianEquityCostModel(slippage_bps=3.0)


__all__ = [
    "ILLIQUID_COST_MODEL",
    "LIQUID_COST_MODEL",
    "CostBreakdown",
    "IndianEquityCostModel",
]
